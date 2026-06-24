//! look-ahead 1 手詰判定 (mate-in-1) サブシステムと健全性診断計装．
//!
//! OR ノード first-visit の終端 seed に用いる．探索状態には依存せず，
//! `Board::mate_move_in_1ply` + `check_cache` のみを参照する．

use crate::board::Board;
use crate::moves::Move;
use crate::types::Color;

use crate::dfpn::solver::DfPnSolver;

impl DfPnSolver {

    /// 王手リストを `out` へ直接追記する zero-copy 経路．
    ///
    /// `generate_check_moves_cached` の ArrayVec<_, 593> 値返し (2.4KB 級
    /// stack copy) を避け，cache hit 時は cache 内 slice から，miss 時は
    /// 生成結果から直接 extend する．生成内容・順序は従来と同一．
    pub(crate) fn check_moves_into(&self, board: &mut Board, out: &mut Vec<Move>) {
        let hash = board.hash;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            mate_cand_bump_cache(false, true);
            out.extend_from_slice(cached);
            return;
        }
        mate_cand_bump_cache(false, false);
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        out.extend_from_slice(moves.as_slice());
    }

    /// per-child 1 手詰 lookahead 専用の zero-copy 経路．
    ///
    /// `generate_check_moves_cached` は cache hit でも ArrayVec<_, 593> の値返し
    /// (全コピー × 2) になるため，hit 時は cache 内 slice を直接
    /// `mate_move_in_1ply` へ渡す．生成内容・順序は従来と同一 (semantics 不変)．
    /// 返り値: (1 手詰の手, 王手手段の有無)．
    pub(crate) fn mate1ply_with_cached_checks(&self, board: &mut Board) -> (Option<Move>, bool) {
        let hash = board.hash;
        let turn = board.turn;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            // mate_move_in_1ply は board のみ触る (check_cache へ再挿入しない) ため
            // 借用中の slice は有効なまま．
            mate_cand_bump_cache(true, true);
            let mm = board.mate_move_in_1ply(cached, turn);
            if mate_cand_enabled() {
                record_mate_cand(board, cached, mm, turn);
            }
            return (mm, !cached.is_empty());
        }
        mate_cand_bump_cache(true, false);
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        let mm = board.mate_move_in_1ply(moves.as_slice(), turn);
        if mate_cand_enabled() {
            record_mate_cand(board, moves.as_slice(), mm, turn);
        }
        (mm, !moves.is_empty())
    }

    /// look-ahead 専用の 1 手詰判定 (玉から Chebyshev 距離 ≤2 の候補のみ検査)．
    ///
    /// full check list は従来通り生成し check_cache に格納する (expansion が再利用 = cache 内容不変)
    /// が，詰み判定の scan は距離 ≤2 候補に限定する (`mate_move_in_1ply_maxdist(_, _, 2)`)．距離 ≤2
    /// は full scan と同一の Some/None・同一の詰み手を返す (= node 不変)，かつ遠方候補の検証・
    /// do_move fallback を省ける．look-ahead は詰み手 (Option) のみ使う
    /// (`does_have_mate_possibility` が不詰判定を担うため has_checks 不要)．
    pub(crate) fn mate1ply_cached_near2(&self, board: &mut Board) -> Option<Move> {
        let hash = board.hash;
        let turn = board.turn;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            return board.mate_move_in_1ply_maxdist(cached, turn, 2);
        }
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        board.mate_move_in_1ply_maxdist(moves.as_slice(), turn, 2)
    }

    /// 敵玉隣接 geometry から候補を構成する look-ahead 1 手詰判定．
    ///
    /// [`DfPnSolver::generate_mate_candidates`] が敵玉隣接 geometry から駒種順で
    /// 1 手詰候補を構成し，検証済 [`Board::mate_move_in_1ply_maxdist`] でスキャンする．
    /// 従来 `mate1ply_cached_near2` との違いは候補列挙のみ (full movegen を回避)．
    /// `check_cache` は更新しない (node が展開される場合は expansion 側が必要時に full 生成する)．
    ///
    /// 逆王手局面 (攻め方自玉が王手 = king-geometry 列挙の前提外) は従来 scan へ
    /// fallback する．`MATE1PLY_VERIFY` 時は健全性 (偽 1 手詰の無さ) と full scan との一致を照合．
    pub(crate) fn mate1ply(&self, board: &mut Board) -> Option<Move> {
        let turn = board.turn;
        if board.king_square(turn).is_some() && board.is_in_check(turn) {
            // 逆王手: king-geometry 列挙の前提外 → 従来の near2 scan で判定する．
            return self.mate1ply_cached_near2(board);
        }
        let candidates = self.generate_mate_candidates(board);
        let kh = board.mate_move_in_1ply_maxdist(candidates.as_slice(), turn, u8::MAX);
        if mate1ply_verify() {
            self.verify_mate1ply(board, kh);
        }
        kh
    }

    /// `MATE1PLY_VERIFY` 用: 候補列挙の健全性と従来 full scan との一致を照合し統計へ加算する．
    /// (a) kh が返す手は必ず真の 1 手詰 (do_move 後に詰み) でなければならない (偽 1 手詰 = bug)．
    /// (b) full scan (`mate1ply_with_cached_checks`, 全王手) との Some/None・手の一致を記録する
    ///     (手の相違・kh miss は構成差ゆえ許容; FALSE_MATE=0 のみ必須)．
    fn verify_mate1ply(&self, board: &mut Board, kh: Option<Move>) {
        let turn = board.turn;
        let defender = turn.opponent();
        let full = self.mate1ply_with_cached_checks(board).0;
        let mut false_pos = false;
        if let Some(m) = kh {
            let cap = board.do_move(m);
            let real = board.is_in_check(defender) && !crate::movegen::has_any_legal_move(board);
            board.undo_move(m, cap);
            false_pos = !real;
        }
        record_mate1ply(kh, full, false_pos);
        if false_pos {
            eprintln!(
                "[mate1ply] FALSE MATE kh={:?} sfen={}",
                kh.map(|m| m.to_usi()),
                board.sfen()
            );
        }
        // mate_miss (full=Some, kh=None) を分類 dump する (最初の 40 件)．near2==full ゆえ
        // miss は dist≤2 の near-king 詰み = (a) 列挙ギャップ か (b) 候補列挙が落とす
        // pattern (玉移動開き王手 等) か を判別する手掛かり．
        if kh.is_none() {
            if let Some(fm) = full {
                let n = MATE1PLY_MISS_DUMP.with(|c| {
                    let v = c.get();
                    c.set(v + 1);
                    v
                });
                if n < 40 {
                    let kingd = board.king_square(defender).map(|k| {
                        let to = fm.to_sq();
                        (to.col() as i32 - k.col() as i32)
                            .abs()
                            .max((to.row() as i32 - k.row() as i32).abs())
                    });
                    let from_is_king =
                        (!fm.is_drop()) && board.king_square(turn) == Some(fm.from_sq());
                    eprintln!(
                        "[mate1ply] MISS full={} drop={} promo={} king_move={} to_king_dist={:?} sfen={}",
                        fm.to_usi(),
                        fm.is_drop(),
                        fm.is_promotion(),
                        from_is_king,
                        kingd,
                        board.sfen(),
                    );
                }
            }
        }
    }
}

/// look-ahead 1 手詰判定で king-geometry 候補列挙を使うか．[`DfPnSolver::mate1ply`] を参照．
pub(crate) fn mate1ply_enabled() -> bool {
    // 常時 ON．look-ahead 1 手詰判定を king-geometry 候補列挙にする．
    true
}

/// `MATE1PLY_VERIFY`: 候補列挙の健全性 + full scan との一致を毎 look-ahead 照合する
/// (default OFF; 重い)．
pub(crate) fn mate1ply_verify() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MATE1PLY_VERIFY").is_ok())
}

thread_local! {
    /// [calls, mate_found, diffmove, mate_miss, false_pos]．`MATE1PLY_VERIFY` 時のみ加算．
    static MATE1PLY_STATS: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
    /// mate_miss dump の出力件数 (最初の数件のみ詳細を出す)．
    static MATE1PLY_MISS_DUMP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// kh verify 1 件分を加算する．
fn record_mate1ply(kh: Option<Move>, full: Option<Move>, false_pos: bool) {
    MATE1PLY_STATS.with(|s| {
        let mut v = s.get();
        v[0] += 1;
        if kh.is_some() {
            v[1] += 1;
        }
        match (kh, full) {
            (Some(a), Some(b)) if a != b => v[2] += 1,
            (None, Some(_)) => v[3] += 1,
            _ => {}
        }
        if false_pos {
            v[4] += 1;
        }
        s.set(v);
    });
}

/// kh verify 統計を reset する (solve 開始時)．
pub(crate) fn reset_mate1ply_stats() {
    MATE1PLY_STATS.with(|s| s.set([0; 5]));
}

/// kh verify 統計を report する (solve 終了時; gate off なら no-op)．
pub(crate) fn report_mate1ply_stats() {
    if !mate1ply_verify() {
        return;
    }
    MATE1PLY_STATS.with(|s| {
        let v = s.get();
        if v[0] == 0 {
            return;
        }
        eprintln!(
            "[mate1ply] calls={} mate_found={} | vs full: diffmove={} mate_miss(full=Some,kh=None)={} | FALSE_MATE(bug)={}",
            v[0], v[1], v[2], v[3], v[4],
        );
    });
}

// ===== MATE1PLY_CAND 診断 =====================================================
// look-ahead の 1 手詰検出について「候補手 (王手) の分布」と「実際に詰みになった手の幾何」を
// 計測する (env `MATE1PLY_CAND` で gate; production path は不変)．目的: king-centric
// constructive mate1ply が **full scan と同じ Some/None を返せる (= node 不変)** か，候補をどこまで
// 絞れるか (= 高速化余地) を経験的に解析する．非詰みは to_sq の玉チェビシェフ距離で分類する
// (非遠方駒の王手は必ず玉隣接，遠方/開き王手のみ遠い)．

#[derive(Default)]
struct MateCandStats {
    calls: u64,
    n_mate: u64,
    sum_checks: u64,
    max_checks: u64,
    sum_checks_king_adj: u64, // to_sq が玉チェビシェフ距離 ≤1 の王手数 (= 非遠方候補)
    sum_checks_slider_far: u64, // 遠方 (距離>1) の飛び駒王手数
    mate_drop: u64,
    mate_move: u64,
    mate_capture: u64,
    mate_slider: u64,
    mate_far: u64, // 詰み手の to_sq が玉から距離>1
    mate_far_slider: u64,
    mate_far_nonslider: u64, // 距離>1 かつ非飛び駒 = 開き王手の疑い (要注目)
    mate_dist: [u64; 9],
    // 差分解析: full scan (ground truth) vs 距離フィルタ subset の verdict 比較 (full が Some の時のみ)．
    // subset ⊆ full なので「subset Some だが full None」は原理上起きない (= 起きたら bug)．
    // MISS = full Some だが subset None (= subset が見落とす遠方詰み)．DIFFMOVE = 両方 Some だが手が
    // 異なる (= どちらも検証済の合法 1 手詰だが選択が違う; proof-hand 変化源)．BUG = subset Some/full None．
    near1_miss: u64, // dist≤1 subset が full の詰みを見落とす
    near1_diffmove: u64,
    near1_bug: u64,
    near2_miss: u64, // dist≤2 subset (= 玉隣接+桂) が見落とす
    near2_diffmove: u64,
    near2_bug: u64,
    // check_cache prepay 値の計測 (constructive で look-ahead generate を除いた際の generate 削減量推定)．
    la_hit: u64,   // look-ahead の cache hit
    la_miss: u64,  // look-ahead の cache miss (= generate 発生)
    cmi_hit: u64,  // expansion (check_moves_into) の cache hit (= 事前生成の再利用)
    cmi_miss: u64, // expansion の cache miss (= generate 発生)
}

/// check_cache hit/miss を計測する (MATE1PLY_CAND 時のみ; la=look-ahead か expansion か)．
pub(crate) fn mate_cand_bump_cache(la: bool, hit: bool) {
    if !mate_cand_enabled() {
        return;
    }
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        match (la, hit) {
            (true, true) => s.la_hit += 1,
            (true, false) => s.la_miss += 1,
            (false, true) => s.cmi_hit += 1,
            (false, false) => s.cmi_miss += 1,
        }
    });
}

thread_local! {
    static MATE_CAND_STATS: std::cell::RefCell<MateCandStats> =
        std::cell::RefCell::new(MateCandStats::default());
}

fn mate_cand_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MATE1PLY_CAND").is_ok())
}

/// MATE1PLY_CAND 統計を reset する (solve 開始時)．
pub(crate) fn reset_mate_cand_stats() {
    if mate_cand_enabled() {
        MATE_CAND_STATS.with(|s| *s.borrow_mut() = MateCandStats::default());
    }
}

/// MATE1PLY_CAND 統計を report する (solve 終了時; gate off なら no-op)．
pub(crate) fn report_mate_cand_stats() {
    if !mate_cand_enabled() {
        return;
    }
    MATE_CAND_STATS.with(|s| {
        let s = s.borrow();
        if s.calls == 0 {
            return;
        }
        let f = s.calls as f64;
        eprintln!(
            "[mate_cand] calls={} mate={} ({:.2}%) | avg_checks={:.2} max_checks={} | per-call avg: king_adj(≤1)={:.2} slider_far(>1)={:.3}",
            s.calls,
            s.n_mate,
            100.0 * s.n_mate as f64 / f,
            s.sum_checks as f64 / f,
            s.max_checks,
            s.sum_checks_king_adj as f64 / f,
            s.sum_checks_slider_far as f64 / f,
        );
        eprintln!(
            "[mate_cand] 候補総数={} うち玉隣接(≤1)={} ({:.1}%) 遠方飛び駒(>1)={} ({:.2}%)",
            s.sum_checks,
            s.sum_checks_king_adj,
            100.0 * s.sum_checks_king_adj as f64 / s.sum_checks.max(1) as f64,
            s.sum_checks_slider_far,
            100.0 * s.sum_checks_slider_far as f64 / s.sum_checks.max(1) as f64,
        );
        eprintln!(
            "[mate_cand] mate move: drop={} move={} capture={} slider={} | far(dist>1)={} far_slider={} far_nonslider={}",
            s.mate_drop, s.mate_move, s.mate_capture, s.mate_slider,
            s.mate_far, s.mate_far_slider, s.mate_far_nonslider,
        );
        eprintln!("[mate_cand] mate_dist(チェビシェフ 0..8)={:?}", s.mate_dist);
        eprintln!(
            "[mate_cand] DIFF vs full(ground truth): near1(≤1) miss={} diffmove={} bug={} | near2(≤2) miss={} diffmove={} bug={}",
            s.near1_miss, s.near1_diffmove, s.near1_bug,
            s.near2_miss, s.near2_diffmove, s.near2_bug,
        );
        eprintln!(
            "[mate_cand]   → near1 が node を変える look-ahead = {} ({:.3}% of {} mates), near2 = {} ({:.3}%)",
            s.near1_miss + s.near1_diffmove,
            100.0 * (s.near1_miss + s.near1_diffmove) as f64 / s.n_mate.max(1) as f64,
            s.n_mate,
            s.near2_miss + s.near2_diffmove,
            100.0 * (s.near2_miss + s.near2_diffmove) as f64 / s.n_mate.max(1) as f64,
        );
        let gen_now = s.la_miss + s.cmi_miss;
        // constructive (look-ahead generate 除去) 後の generate ≈ expansion の miss のみ．
        // look-ahead が prepay した分 (la_miss で生成 → 後で cmi_hit 再利用) のうち expansion
        // されない分が純減．下限推定: la が prepay した cmi_hit は最大 la_miss．
        eprintln!(
            "[mate_cand] cache: LA(hit={} miss={}) EXP(hit={} miss={}) | generate計={} (LA駆動={} EXP駆動={})",
            s.la_hit, s.la_miss, s.cmi_hit, s.cmi_miss, gen_now, s.la_miss, s.cmi_miss,
        );
    });
}

/// 1 look-ahead の候補 (王手) 分布と詰み手の幾何を統計へ加算する (MATE1PLY_CAND 時のみ呼ぶ)．
/// full scan の結果 `mm` を ground truth とし，距離フィルタ subset の verdict と比較する
/// (board は mate_move_in_1ply 呼び出しで一時的に do_move/undo するが復元される)．
fn record_mate_cand(board: &mut Board, checks: &[Move], mm: Option<Move>, turn: Color) {
    use crate::types::{PieceType, Square};
    let king = match board.king_square(turn.opponent()) {
        Some(k) => k,
        None => return,
    };
    let kc = king.col() as i32;
    let kr = king.row() as i32;
    let cheby = |sq: Square| -> i32 {
        (sq.col() as i32 - kc)
            .abs()
            .max((sq.row() as i32 - kr).abs())
    };
    let checker_pt = |m: Move| -> PieceType {
        if m.is_drop() {
            m.drop_piece_type().unwrap()
        } else {
            let raw = PieceType::from_u8(m.moving_piece_type_raw()).unwrap();
            if m.is_promotion() {
                raw.promoted().unwrap()
            } else {
                raw
            }
        }
    };
    let is_slider = |pt: PieceType| {
        matches!(
            pt,
            PieceType::Lance
                | PieceType::Bishop
                | PieceType::Rook
                | PieceType::Horse
                | PieceType::Dragon
        )
    };
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.calls += 1;
        s.sum_checks += checks.len() as u64;
        if checks.len() as u64 > s.max_checks {
            s.max_checks = checks.len() as u64;
        }
        for &c in checks {
            let d = cheby(c.to_sq());
            if d <= 1 {
                s.sum_checks_king_adj += 1;
            }
            if d > 1 && is_slider(checker_pt(c)) {
                s.sum_checks_slider_far += 1;
            }
        }
        if let Some(m) = mm {
            s.n_mate += 1;
            let d = cheby(m.to_sq());
            s.mate_dist[d.clamp(0, 8) as usize] += 1;
            if m.is_drop() {
                s.mate_drop += 1;
            } else {
                s.mate_move += 1;
                if m.captured_piece_raw() != 0 {
                    s.mate_capture += 1;
                }
            }
            let sl = is_slider(checker_pt(m));
            if sl {
                s.mate_slider += 1;
            }
            if d > 1 {
                s.mate_far += 1;
                if sl {
                    s.mate_far_slider += 1;
                } else {
                    s.mate_far_nonslider += 1;
                }
            }
        }
    });

    // === 差分解析: full scan (ground truth) vs 距離フィルタ subset ===
    // full が詰みを返した look-ahead でのみ実施 (no-mate は subset も必ず None = 一致, 計測不要)．
    let full_mm = match mm {
        Some(m) => m,
        None => return,
    };
    // 距離フィルタした subset で同じ verified predicate (mate_move_in_1ply) を実行する．
    // subset ⊆ full ゆえ subset が Some なら full も Some (full_mm)．subset Some/full None は
    // 起き得ない (起きたら mate_move_in_1ply の順序依存 bug)．board は復元される．
    let near1: Vec<Move> = checks
        .iter()
        .copied()
        .filter(|c| cheby(c.to_sq()) <= 1)
        .collect();
    let near2: Vec<Move> = checks
        .iter()
        .copied()
        .filter(|c| cheby(c.to_sq()) <= 2)
        .collect();
    let mm1 = board.mate_move_in_1ply(&near1, turn);
    let mm2 = board.mate_move_in_1ply(&near2, turn);
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        match mm1 {
            None => s.near1_miss += 1,
            Some(m) if m != full_mm => s.near1_diffmove += 1,
            Some(_) => {}
        }
        match mm2 {
            None => s.near2_miss += 1,
            Some(m) if m != full_mm => s.near2_diffmove += 1,
            Some(_) => {}
        }
    });
    // bug 検出: full が None なのに subset が Some は原理上不可能 (subset⊆full)．
    // full_mm が Some の本経路では起き得ないが，対称性のため near*_bug は別途 (no-mate 経路) で
    // 計測する余地を残す (現状は未使用 = 常に 0)．
}
