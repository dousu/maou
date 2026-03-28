//! Df-Pn (Depth-First Proof-Number Search) による詰将棋ソルバー．
//!
//! cshogi と同じ Df-Pn アルゴリズムを採用し，攻め方が玉方を
//! 詰ませる最善手順を求める．先手・後手どちらが攻め方でも動作する．
//!
//! # 引数
//!
//! - `depth`: 最大探索手数(デフォルト 31)．無限ループ防止用．
//! - `nodes`: 最大ノード数(デフォルト 1,048,576 = 2^20)．計算時間・メモリ制限用．
//! - `draw_ply`: 引き分け手数(デフォルト 32767)．

use arrayvec::ArrayVec;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::time::{Duration, Instant};

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square, HAND_KINDS};

/// 王手手/応手の最大数．
/// 将棋の合法手上限は593であり，長手数の詰将棋では
/// 持ち駒が多い局面で320を超えるケースが存在するため，
/// 合法手上限に合わせる．
const MAX_MOVES: usize = 593;

/// `ArrayVec::try_push` のラッパー．
/// デバッグビルドでは容量超過時にパニックし，リリースビルドでは無音で破棄する．
#[inline(always)]
fn push_move<T, const N: usize>(buf: &mut ArrayVec<T, N>, val: T) {
    let result = buf.try_push(val);
    debug_assert!(result.is_ok(), "move buffer overflow: capacity {N} exceeded");
}

/// 証明数・反証数の無限大を表す定数．
const INF: u32 = u32::MAX;

/// AND ノードで合駒(drop)を後回しにするための dn バイアス．
///
/// AND ノード(玉方手番)で王の移動・駒取りなどの非合駒応手を先に
/// 展開すると，それらの証明エントリが転置表に蓄積される．
/// その後に合駒の分岐を探索する際，攻め方が合駒を取った後の
/// 局面が既に証明済みになっていることが多く，高速に証明できる．
///
/// 旧値: INF/2(≈2B)は deferred_children 方式と併用する前提の値であり，
/// drops を children にそのまま含める現方式では事実上の無限バイアスとなり
/// 非 drop 子が全て証明されるまで drop が選択されない問題があった．
///
/// 新値: 256 は king move の初期 dn(1)より十分大きく，
/// king move が探索されて dn が上昇した後に drop の探索が始まる程度のバイアス．
/// これにより df-pn の自然な閾値制御で king move → drop の順序が実現される．
const INTERPOSE_DN_BIAS: u32 = 8;


/// TCA (Threshold Controlling Algorithm, Kishimoto & Müller 2008; Kishimoto 2010)
/// 過小評価対策．
///
/// 巡回グラフ(DCG)上の df-pn では，ループ検出により子ノードが
/// (INF, 0) を返し，兄弟ノードの pn/dn が過小評価されうる．
/// TCA は OR ノードでループ子が存在する場合に MID ループの閾値を
/// 加算的に拡張し，兄弟のより深い探索を促す．
///
/// 拡張量 = `threshold / TCA_EXTEND_DENOM + 1` (約 25% の加算)．
/// 乗算的拡張(2×)は再帰で指数的に増大するが，加算的拡張は
/// 各レベルで独立に適用されるため膨張を抑える．
const TCA_EXTEND_DENOM: u32 = 4;

/// DFPN-E (Kishimoto et al., NeurIPS 2019) エッジコスト型ヒューリスティック．
///
/// 標準 df-pn+ はノード(局面)の特徴で初期 pn/dn を設定するが，
/// DFPN-E は**エッジ(手)**の特徴に基づくコストを加算する．
/// 展開済みノードではエッジコストをゼロに戻すため，
/// 実質的には初期 pn への加算として機能する．
///
/// 詰将棋での手の質:
/// - OR ノードの王手: 成+取 > 取/成 > 近い静か手 > 遠い静か手
/// - AND ノードの応手: 合駒(攻め方有利) < 駒取り < 玉の逃げ

/// OR ノード(攻め方の王手)のエッジコストを計算する(DFPN-E)．
///
/// 有力な王手ほどコストが低く(pn に加算される量が少ない)，
/// ソルバーがその手を優先的に深く探索する．
///
/// - 成+取王手: 0 (最有力)
/// - 成王手/取王手: 0
/// - 近い静か王手(距離≤2): 1
/// - 遠い静か王手(距離≥3): 2
#[inline]
fn edge_cost_or(m: Move, king_sq: Square) -> u32 {
    let promo = m.is_promotion();
    let capture = m.captured_piece_raw() > 0;
    if promo || capture {
        return 0;
    }
    // 静かな王手: 玉との距離でコスト決定
    let to = m.to_sq();
    let dc = (to.col() as i8 - king_sq.col() as i8).unsigned_abs();
    let dr = (to.row() as i8 - king_sq.row() as i8).unsigned_abs();
    let dist = dc.max(dr);
    if dist <= 2 { 1 } else { 2 }
}

/// AND ノード(守備側の応手)のエッジコストを計算する(DFPN-E)．
///
/// 攻め方にとって有利な応手(=詰ませやすい応手)ほどコストが低く，
/// ソルバーがその応手の子ツリーを優先的に探索する．
///
/// - 合駒(drop): 0 (攻め方が取って攻撃続行可能)
/// - 駒取り(応手で攻め駒を取る): 2 (攻め方の戦力が減り危険)
/// - 玉の逃げ(非合駒・非取り): 1
#[inline]
fn edge_cost_and(m: Move) -> u32 {
    if m.is_drop() {
        // 合駒: 攻め方が取って持ち駒に加えられるため有利
        return 0;
    }
    let capture = m.captured_piece_raw() > 0;
    if capture {
        // 駒取り: 攻め駒を除去するため攻め方にとって不利
        return 2;
    }
    // 玉の逃げ・駒移動合い
    1
}

/// 捨て駒のみ王手ブースト(人間的枝刈り)．
///
/// OR ノード(攻め方手番)で利用可能な全王手が「支えなし」の捨て駒である場合，
/// 人間が直感的に「不詰」と見切るのと同様に pn を加算して探索優先度を下げる．
///
/// 「支え」の判定: 王手後のマス(`to_sq`)に攻め方の他の駒が利いているか．
/// 駒移動の場合は移動元を除外して判定する．
///
/// 全王手が捨て駒の場合，王手数に比例するブースト(最低2)を返す．
/// 捨て駒でない王手が1つでもあれば 0 を返す．
fn sacrifice_check_boost(board: &Board, checks: &[Move]) -> u32 {
    if checks.is_empty() {
        return 0;
    }
    let attacker = board.turn;
    for m in checks {
        let to = m.to_sq();
        let excluded = if m.is_drop() { None } else { Some(m.from_sq()) };
        // 攻め方の他の駒が to に利いていれば支えあり → 捨て駒ではない
        if board.is_attacked_by_excluding(to, attacker, false, excluded) {
            return 0;
        }
    }
    // 全王手が捨て駒 → 詰ませにくい(上限2: 大きくしすぎると不詰証明が遅延)
    2
}

/// Deep df-pn の深さ係数 R (Song Zhang et al. 2017)．
///
/// 深い位置ほど初期 dn を高く設定し，探索のスラッシングを抑制する．
/// 高い dn は「この部分木の反証にはコストがかかる」ことを意味し，
/// ソルバーがブランチを早期に見捨てずに深く探索するよう促す．
/// 論文推奨値は R=0.4 (Othello/Hex)．
/// SNDA (Kishimoto 2010) の積極的ソースグループ集約．
///
/// `(source, value)` ペアのリストと通常の sum を受け取り，
/// 同一 source グループの重複分を控除する．
///
/// 積極的 max 集約方式: 同一 source グループ内で最大値のみを残し，
/// 残りを全て控除する(`deduction = sum(group) - max(group)`)．
/// DAG 合流で共有されるリーフの重複カウントを排除し，
/// 過大評価をより正確に補正する．
///
/// TCA(過小評価対策)が実装済みのため，積極的方式による
/// 過小評価リスクは TCA の閾値拡張で緩和される．
///
/// `source == 0` のペアは独立ノード(TT ミス)としてスキップする．
#[inline]
fn snda_dedup(pairs: &mut [(u64, u32)], raw_sum: u32) -> u32 {
    pairs.sort_unstable_by_key(|&(s, _)| s);
    let mut deduction: u64 = 0;
    let mut i = 0;
    while i < pairs.len() {
        let source = pairs[i].0;
        if source == 0 {
            i += 1;
            continue;
        }
        let start = i;
        let mut group_max = pairs[i].1;
        let mut group_sum: u64 = pairs[i].1 as u64;
        i += 1;
        while i < pairs.len() && pairs[i].0 == source {
            group_max = group_max.max(pairs[i].1);
            group_sum += pairs[i].1 as u64;
            i += 1;
        }
        let group_size = i - start;
        if group_size > 1 {
            // 積極的 max 集約: グループ合計から最大値を引いた分を控除
            deduction =
                deduction.saturating_add(group_sum - group_max as u64);
        }
    }
    (raw_sum as u64).saturating_sub(deduction).max(1) as u32
}

/// プロファイリング計測マクロ．
///
/// `profile` feature が有効な場合のみ計測し，結果をフィールドに加算する．
#[cfg(feature = "profile")]
macro_rules! profile_timed {
    ($self:expr, $ns_field:ident, $count_field:ident, $body:expr) => {{
        let _t = Instant::now();
        let _r = $body;
        let _elapsed = _t.elapsed().as_nanos() as u64;
        $self.profile_stats.$ns_field += _elapsed;
        $self.profile_stats.$count_field += 1;
        _r
    }};
}

/// プロファイリング無効時は素通し．
#[cfg(not(feature = "profile"))]
macro_rules! profile_timed {
    ($self:expr, $ns_field:ident, $count_field:ident, $body:expr) => {
        $body
    };
}

/// プロファイリング統計情報．
///
/// `profile` feature が有効な場合に収集される．
/// 各フィールドは `mid()` 内の主要操作の累積時間(ナノ秒)と呼び出し回数を保持する．
#[cfg(feature = "profile")]
#[derive(Debug, Clone, Default)]
pub struct ProfileStats {
    /// position_key() の累積時間(ナノ秒)．
    pub position_key_ns: u64,
    /// position_key() の呼び出し回数．
    pub position_key_count: u64,
    /// ループ検出(path.contains)の累積時間(ナノ秒)．
    pub loop_detect_ns: u64,
    /// ループ検出の呼び出し回数．
    pub loop_detect_count: u64,
    /// TT 参照(look_up_pn_dn)の累積時間(ナノ秒)．
    pub tt_lookup_ns: u64,
    /// TT 参照の呼び出し回数．
    pub tt_lookup_count: u64,
    /// TT 格納(store)の累積時間(ナノ秒)．
    pub tt_store_ns: u64,
    /// TT 格納の呼び出し回数．
    pub tt_store_count: u64,
    /// 王手生成(generate_check_moves)の累積時間(ナノ秒)．
    pub movegen_check_ns: u64,
    /// 王手生成の呼び出し回数．
    pub movegen_check_count: u64,
    /// 応手生成(generate_defense_moves)の累積時間(ナノ秒)．
    pub movegen_defense_ns: u64,
    /// 応手生成の呼び出し回数．
    pub movegen_defense_count: u64,
    /// do_move の累積時間(ナノ秒)．
    pub do_move_ns: u64,
    /// do_move の呼び出し回数．
    pub do_move_count: u64,
    /// undo_move の累積時間(ナノ秒)．
    pub undo_move_ns: u64,
    /// undo_move の呼び出し回数．
    pub undo_move_count: u64,
    /// 子ノード初期化フェーズの累積時間(ナノ秒)．
    pub child_init_ns: u64,
    /// 子ノード初期化の呼び出し回数．
    pub child_init_count: u64,
    /// 子ノード初期化内の静的詰め探索(予算制)の累積時間(ナノ秒)．
    pub static_mate_ns: u64,
    /// 静的詰め探索の呼び出し回数．
    pub static_mate_count: u64,
    /// 静的詰め探索で詰みを検出した回数．
    pub static_mate_hits: u64,
    /// 子ノード初期化内の do_move/undo_move の累積時間(ナノ秒)．
    pub child_init_domove_ns: u64,
    /// 子ノード初期化内の do_move/undo_move の呼び出し回数．
    pub child_init_domove_count: u64,
    /// メインループの pn/dn 収集の累積時間(ナノ秒)．
    pub main_loop_collect_ns: u64,
    /// メインループの pn/dn 収集回数．
    pub main_loop_collect_count: u64,
    /// 深さ制限時の終端処理(`depth_limit_all_checks_refutable` を含む)の累積時間(ナノ秒)．
    pub depth_limit_terminal_ns: u64,
    /// 深さ制限時の終端処理の呼び出し回数．
    pub depth_limit_terminal_count: u64,
    /// NM 昇格のための `depth_limit_all_checks_refutable` の累積時間(ナノ秒)．
    pub nm_promotion_refutable_ns: u64,
    /// NM 昇格のための `depth_limit_all_checks_refutable` の呼び出し回数．
    pub nm_promotion_refutable_count: u64,
    /// 合駒 TT 先読み(`generate_check_moves` + `try_capture_tt_proof`)の累積時間(ナノ秒)．
    pub capture_tt_lookahead_ns: u64,
    /// 合駒 TT 先読みの呼び出し回数．
    pub capture_tt_lookahead_count: u64,
    /// `cross_deduce_children` の累積時間(ナノ秒)．
    pub cross_deduce_ns: u64,
    /// `cross_deduce_children` の呼び出し回数．
    pub cross_deduce_count: u64,
    /// `try_prefilter_block` の累積時間(ナノ秒)．
    pub prefilter_ns: u64,
    /// `try_prefilter_block` の呼び出し回数．
    pub prefilter_count: u64,
    /// MID 全体のウォール時間(ナノ秒)．mid_fallback 内のみ計測．
    pub mid_total_ns: u64,
    /// MID トップレベル呼び出し回数．
    pub mid_total_count: u64,
    /// PNS フェーズのウォール時間(ナノ秒)．
    pub pns_total_ns: u64,
    /// TT エントリ溢れ(置換)の発生回数．
    pub tt_overflow_count: u64,
    /// TT エントリ溢れで置換対象が見つからなかった回数．
    pub tt_overflow_no_victim_count: u64,
    /// TT エントリ数の最大値(1局面あたり)．
    pub tt_max_entries_per_position: usize,
}

#[cfg(feature = "profile")]
impl std::fmt::Display for ProfileStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // child_init_domove は child_init の内訳だが，early return で child_init が
        // 記録されないケースがあるため，未計上分を補正する．
        let child_init_uncaptured = self.child_init_domove_ns
            .saturating_sub(self.child_init_ns);
        let total_ns = self.position_key_ns
            + self.loop_detect_ns
            + self.tt_lookup_ns
            + self.tt_store_ns
            + self.movegen_check_ns
            + self.movegen_defense_ns
            + self.do_move_ns
            + self.undo_move_ns
            + self.child_init_ns
            + child_init_uncaptured
            + self.main_loop_collect_ns
            + self.depth_limit_terminal_ns
            + self.nm_promotion_refutable_ns
            + self.capture_tt_lookahead_ns
            + self.cross_deduce_ns
            + self.prefilter_ns;
        let total_us = total_ns as f64 / 1000.0;

        writeln!(f, "=== DFPN Profile Stats ===")?;
        writeln!(f, "{:<25} {:>12} {:>10} {:>10} {:>6}",
            "Operation", "Total(µs)", "Count", "Avg(ns)", "%")?;
        writeln!(f, "{}", "-".repeat(65))?;

        let items: Vec<(&str, u64, u64)> = vec![
            ("position_key", self.position_key_ns, self.position_key_count),
            ("loop_detect", self.loop_detect_ns, self.loop_detect_count),
            ("tt_lookup", self.tt_lookup_ns, self.tt_lookup_count),
            ("tt_store", self.tt_store_ns, self.tt_store_count),
            ("movegen_check", self.movegen_check_ns, self.movegen_check_count),
            ("movegen_defense", self.movegen_defense_ns, self.movegen_defense_count),
            ("do_move", self.do_move_ns, self.do_move_count),
            ("undo_move", self.undo_move_ns, self.undo_move_count),
            ("child_init", self.child_init_ns, self.child_init_count),
            ("  static_mate", self.static_mate_ns, self.static_mate_count),
            ("  child_do/undo_move", self.child_init_domove_ns, self.child_init_domove_count),
            ("  init_early_domove", child_init_uncaptured, 0),
            ("main_loop_collect", self.main_loop_collect_ns, self.main_loop_collect_count),
            ("depth_limit_terminal", self.depth_limit_terminal_ns, self.depth_limit_terminal_count),
            ("nm_promotion_refut", self.nm_promotion_refutable_ns, self.nm_promotion_refutable_count),
            ("capture_tt_lookahead", self.capture_tt_lookahead_ns, self.capture_tt_lookahead_count),
            ("cross_deduce", self.cross_deduce_ns, self.cross_deduce_count),
            ("prefilter", self.prefilter_ns, self.prefilter_count),
        ];

        for (name, ns, count) in &items {
            let us = *ns as f64 / 1000.0;
            let avg_ns = if *count > 0 { *ns / *count } else { 0 };
            let pct = if total_ns > 0 {
                *ns as f64 / total_ns as f64 * 100.0
            } else {
                0.0
            };
            writeln!(f, "{:<25} {:>12.1} {:>10} {:>10} {:>5.1}%",
                name, us, count, avg_ns, pct)?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f, "{:<25} {:>12.1}", "Total measured", total_us)?;
        if self.static_mate_count > 0 {
            writeln!(f, "  static_mate hits: {} / {} ({:.1}%)",
                self.static_mate_hits,
                self.static_mate_count,
                self.static_mate_hits as f64 / self.static_mate_count as f64 * 100.0)?;
        }
        if self.tt_overflow_count > 0 || self.tt_max_entries_per_position > 0 {
            writeln!(f, "  tt_overflow: {} (no_victim: {}), max_entries/pos: {}",
                self.tt_overflow_count,
                self.tt_overflow_no_victim_count,
                self.tt_max_entries_per_position)?;
        }
        let solve_wall_ns = self.mid_total_ns + self.pns_total_ns;
        if solve_wall_ns > 0 {
            let mid_us = self.mid_total_ns as f64 / 1000.0;
            let pns_us = self.pns_total_ns as f64 / 1000.0;
            let coverage_pct = if self.mid_total_ns > 0 {
                total_ns as f64 / self.mid_total_ns as f64 * 100.0
            } else {
                0.0
            };
            writeln!(f, "  MID wall: {:.1}µs ({} calls), PNS wall: {:.1}µs",
                mid_us, self.mid_total_count, pns_us)?;
            writeln!(f, "  MID profiled coverage: {:.1}%", coverage_pct)?;
        }
        Ok(())
    }
}

/// 同一盤面ハッシュあたりの TT エントリ上限．
///
/// 支配関係(パレートフロンティア)による圧縮により，証明・反証エントリは
/// 互いに比較不能な持ち駒構成のみ保持される．そのため上限に達することは
/// 稀であり，主に中間エントリの蓄積を制限する安全弁として機能する．
const MAX_TT_ENTRIES_PER_POSITION: usize = 16;

/// 持ち駒の要素ごと比較: a の全要素が b 以上なら true．
///
/// 証明駒の優越判定に使用: 持ち駒が多い方が有利(攻め方)．
///
/// SWAR (SIMD Within A Register): 7 バイトを u64 にパックし分岐なしで一括比較する．
/// 各バイトに MSB(0x80) をセットして引き算し，MSB が全て残れば全要素 a[i] >= b[i]．
/// 持ち駒値は 0-18 の範囲なので各バイトの MSB は常に 0 であり，
/// (a[i] + 128) - b[i] >= 110 となるためバイト間の桁借りは発生しない．
#[inline(always)]
fn hand_gte(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
    let a_packed = u64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], 0]);
    let b_packed = u64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0]);
    const H: u64 = 0x8080_8080_8080_8080;
    ((a_packed | H) - b_packed) & H == H
}

/// 前方利き駒チェーン(歩 ≤ 香 ≤ 飛)を考慮した持ち駒の半順序比較．
///
/// 標準の `hand_gte`(駒種ごと独立比較)に加えて，
/// 前方利き駒の上位互換関係を利用した代替判定を行う:
///
/// - 飛車は香車の上位互換(飛車は前方を含む4方向に利く)
/// - 香車は歩の上位互換(香車は前方に複数マス利く)
///
/// これにより「歩で詰む → 香でも詰む → 飛でも詰む」が成立し，
/// TT の証明/反証エントリの再利用範囲が拡大する．
///
/// # アルゴリズム
///
/// 1. 高速パス: 標準の `hand_gte` で判定(大半はここで確定)
/// 2. 低速パス: 桂・銀・金・角は標準比較，歩・香・飛は
///    弱い駒種から順に不足分を上位駒種で補填するカスケード判定
///
/// # 正当性の根拠
///
/// 詰み証明で歩を使う手(打ち・移動)は，香で代替可能:
/// - 歩打ち → 香打ち: 前方の利きは香 ⊇ 歩
/// - 相手に歩を渡す → 香を渡す: 守備側の持ち駒が強くなるが，
///   これは攻め方にとって不利方向のため証明は依然有効
/// - 成り後: と金 = 成香(ともに金の動き)
///
/// 同様に香 → 飛も成立(飛の利き ⊇ 香の利き，竜 ⊇ 成香)．
#[inline(always)]
fn hand_gte_forward_chain(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
    // 高速パス: 全駒種で a[i] >= b[i] なら即 true
    if hand_gte(a, b) {
        return true;
    }
    // 桂(2)・銀(3)・金(4)・角(5): 標準比較(代替関係なし)
    if a[2] < b[2] || a[3] < b[3] || a[4] < b[4] || a[5] < b[5] {
        return false;
    }
    // 歩(0) ≤ 香(1) ≤ 飛(6) チェーン: 不足分をカスケード補填
    // deficit: 弱い駒種の不足を上位駒種で吸収していく
    let deficit = (b[0] as i16 - a[0] as i16).max(0);
    let deficit = (b[1] as i16 + deficit - a[1] as i16).max(0);
    b[6] as i16 + deficit <= a[6] as i16
}

/// 盤面のみのハッシュ(持ち駒を除外)を返す．
///
/// Board が `board_hash` をインクリメンタルに維持しているため O(1)．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    board.board_hash
}

/// 詰将棋の探索結果．
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TsumeResult {
    /// 詰みが見つかった場合．手順を含む．
    Checkmate {
        moves: Vec<Move>,
        nodes_searched: u64,
    },
    /// 詰みは証明済みだが PV (手順)の復元に失敗した場合．
    ///
    /// TT エントリ上限 (`MAX_TT_ENTRIES_PER_POSITION`) 等により，
    /// 詰み証明後に手順を復元できないケースで返される．
    CheckmateNoPv { nodes_searched: u64 },
    /// 不詰の場合．
    NoCheckmate { nodes_searched: u64 },
    /// 探索制限に達した場合(nodes上限 or depth上限)．
    Unknown { nodes_searched: u64 },
}

/// 深さ制限なし(真の証明/反証)を示す定数．
const REMAINING_INFINITE: u16 = u16::MAX;

/// NM remaining 伝播: 子ノードの NM remaining から親ノードの remaining を計算する．
///
/// - 子が `REMAINING_INFINITE` → 親も `REMAINING_INFINITE`(真の不詰)
/// - 子が有限値 → `min(子の remaining + 1, 現在の remaining)` で伝播
///
/// OR ノードでは `child_min_remaining` は全子の NM remaining の最小値．
/// AND ノードでは単一子の NM remaining．
#[inline]
fn propagate_nm_remaining(child_min_remaining: u16, current_remaining: u16) -> u16 {
    if child_min_remaining == REMAINING_INFINITE {
        REMAINING_INFINITE
    } else {
        // 子の remaining + 1 と現在の remaining の小さい方
        let propagated = child_min_remaining.saturating_add(1);
        propagated.min(current_remaining)
    }
}

/// 転置表のエントリ(証明駒/反証駒対応)．
///
/// - hand: 登録時の攻め方の持ち駒(証明駒/反証駒として使用)
/// - pn, dn: 証明数・反証数
/// - remaining: 登録時の残り探索深さ(`depth - ply`)．
///   `REMAINING_INFINITE` は深さ制限なし(真の証明/反証)を示す．
///   深さ制限による不詰(`dn=0`)は `remaining` が有限値となり，
///   より深い探索で再評価可能になる．
#[derive(Debug, Clone, Copy)]
struct DfPnEntry {
    hand: [u8; HAND_KINDS],
    pn: u32,
    dn: u32,
    remaining: u16,
    /// TT Best Move: この局面で最も有望だった手の Move16 エンコーディング．
    /// 0 は「ベストムーブなし」を示す．
    /// 動的手順改善(Dynamic Move Ordering)で TT ヒット時に
    /// この手を先頭に移動させて探索効率を向上させる．
    best_move: u16,
    /// GHI (Graph History Interaction) 対策フラグ．
    /// ループ検出に由来する反証は経路依存(path-dependent)であり，
    /// 異なる探索経路では無効になる可能性がある．
    /// `true` の場合，`remaining` が有限値に制限され，
    /// より深い探索で自動的に再評価される．
    path_dependent: bool,
    /// SNDA (Kishimoto 2010) のソースノードハッシュ．
    /// この pn/dn の値を決定したリーフノードの局面ハッシュ．
    /// DAG 合流による pn/dn の過大評価を検出するために使用する．
    source: u64,
}

/// Best-First Proof Number Search (PNS) のノード．
///
/// PNS では探索木を明示的にメモリ上に保持し，
/// 常に最も有望なリーフ(most-proving node)を展開する．
/// df-pn の閾値ベースの深さ優先探索と異なり，
/// グローバルに最適なノード選択を行うため thrashing が発生しない．
///
/// 参考: Allis (1994), Seo, Iida & Uiterwijk (2001, PN*)
struct PnsNode {
    /// 盤面ハッシュ(持ち駒除外)．TT キー．
    pos_key: u64,
    /// 完全ハッシュ(持ち駒込み)．ループ検出用．
    full_hash: u64,
    /// 攻め方の持ち駒．
    hand: [u8; HAND_KINDS],
    /// 証明数．
    pn: u32,
    /// 反証数．
    dn: u32,
    /// 親ノードのインデックス(`u32::MAX` = ルート)．
    parent: u32,
    /// 親から到達する手．
    move_from_parent: Move,
    /// OR ノード(攻め方手番)か AND ノード(玉方手番)か．
    or_node: bool,
    /// 展開済みフラグ．
    expanded: bool,
    /// 子ノードのインデックス(アリーナ内)．
    children: Vec<u32>,
    /// 残り探索深さ．
    remaining: u16,
    /// AND ノード用: 逐次活性化待ちの合駒(drop)手．
    /// 弱い駒から順に1つずつ子ノードとして展開する．
    deferred_drops: Vec<Move>,
}

/// PNS アリーナの最大ノード数(メモリ上限)．
///
/// 1ノード ≈ 80〜120 bytes(children Vec 含む)．
/// 2M ノードで約 200〜300 MB を使用する．
const PNS_MAX_ARENA_NODES: usize = 5_000_000;

/// HashMap ベースの転置表(証明駒/反証駒対応)．
///
/// キーは盤面のみのハッシュ(持ち駒除外)を使用し，
/// 同一盤面・異なる持ち駒のエントリを同一クラスタに格納する．
/// cshogi と同様のアプローチで，証明駒/反証駒を正確に保持する．
///
/// 参照時に持ち駒の優越関係を利用して TT ヒット率を向上させる:
/// - 証明済み(pn=0): 現在の持ち駒 >= 登録時の持ち駒 → 再利用
/// - 反証済み(dn=0): 登録時の持ち駒 >= 現在の持ち駒 → 再利用
struct TranspositionTable {
    tt: FxHashMap<u64, Vec<DfPnEntry>>,
    /// TT エントリ溢れ(置換)の発生回数．
    #[cfg(feature = "profile")]
    overflow_count: u64,
    /// TT エントリ溢れで置換対象が見つからなかった回数(全て証明/反証済み)．
    #[cfg(feature = "profile")]
    overflow_no_victim_count: u64,
    /// 1局面あたりのエントリ数の最大値．
    #[cfg(feature = "profile")]
    max_entries_per_position: usize,
    // --- TT 増加診断カウンタ ---
    /// 証明エントリ(pn=0)の挿入回数．
    diag_proof_inserts: u64,
    /// 反証エントリ(dn=0)の挿入回数．
    diag_disproof_inserts: u64,
    /// 中間エントリの新規追加(同一 hand なし)回数．
    diag_intermediate_new: u64,
    /// 中間エントリの既存更新(同一 hand あり)回数．
    diag_intermediate_update: u64,
    /// 支配チェックによるスキップ回数．
    diag_dominated_skip: u64,
    /// remaining 値別の挿入回数(0..=31 + 32=INFINITE)．
    diag_remaining_dist: [u64; 33],
}

impl TranspositionTable {
    /// 転置表を生成する．
    fn new() -> Self {
        TranspositionTable {
            tt: FxHashMap::with_capacity_and_hasher(
                65536,
                Default::default(),
            ),
            #[cfg(feature = "profile")]
            overflow_count: 0,
            #[cfg(feature = "profile")]
            overflow_no_victim_count: 0,
            #[cfg(feature = "profile")]
            max_entries_per_position: 0,
            diag_proof_inserts: 0,
            diag_disproof_inserts: 0,
            diag_intermediate_new: 0,
            diag_intermediate_update: 0,
            diag_dominated_skip: 0,
            diag_remaining_dist: [0; 33],
        }
    }

    /// 転置表を参照する(証明駒/反証駒の優越関係を利用)．
    ///
    /// 1. 証明済みエントリ: 現在の持ち駒 >= 登録時 → (0, dn) を返す
    /// 2. 反証済みエントリ: 登録時の持ち駒 >= 現在 かつ 十分な探索深さ → (pn, 0) を返す
    /// 3. 持ち駒完全一致: そのまま返す
    /// 4. 該当なし: (1, 1) を返す
    ///
    /// # 引数
    ///
    /// - `remaining`: 呼び出し元の残り探索深さ．反証済みエントリの有効性判定に使用．
    ///   `0` を指定すると全ての反証済みエントリを受け入れる(事後クエリ用)．
    /// 返り値: `(pn, dn, source)`.
    /// `source` は SNDA 用のソースノードハッシュ．
    /// TT ミス時は `source = 0`(独立ノード: SNDA グルーピング対象外)．
    #[inline(always)]
    fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u64) {
        let entries = match self.tt.get(&pos_key) {
            Some(e) => e,
            None => return (1, 1, 0),
        };

        let mut exact_match: Option<(u32, u32, u64)> = None;

        // 証明(pn=0)を反証(dn=0)より常に優先する．
        // IDS の浅い反復で仮反証が保存された後，深い反復で同一局面が
        // 証明されると証明と反証が共存しうる(retain_proofs で除去されるが
        // 同一反復内では共存する可能性がある)．
        // 単一パスでの early return では entries の格納順に依存して
        // 反証が先に返される場合があるため，証明を先にスキャンする．
        for e in entries {
            if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                return (0, e.dn, e.source);
            }
        }
        for e in entries {
            // 反証済み: 持ち駒が少ない(以下)かつ十分な深さなら再利用．
            // 経路依存反証(path_dependent)は remaining チェックを免除する．
            // GHI ループ検出に由来する反証は propagate_nm_remaining で
            // remaining が極端に小さくなりがちで，同一 IDS 深さの再訪時に
            // マッチせず無限再入(スラッシング)を引き起こすため．
            // IDS 反復間では retain_proofs_only で経路依存反証は除去されるため，
            // 深い反復で古い反証が不正に使われる心配はない．
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && (e.remaining >= remaining || e.path_dependent)
            {
                return (e.pn, 0, e.source);
            }
            // 完全一致(pn=0/dn=0 は上で個別に処理済みなのでスキップ)
            if exact_match.is_none()
                && e.hand == *hand
                && e.pn != 0
                && e.dn != 0
            {
                exact_match = Some((e.pn, e.dn, e.source));
            }
        }

        exact_match.unwrap_or((1, 1, 0))
    }

    /// TT Best Move を参照する．
    ///
    /// 指定局面の完全一致エントリからベストムーブ(Move16)を返す．
    /// 該当なし，または best_move 未記録の場合は `0` を返す．
    #[inline(always)]
    fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let entries = match self.tt.get(&pos_key) {
            Some(e) => e,
            None => return 0,
        };
        for e in entries {
            if e.hand == *hand && e.best_move != 0 {
                return e.best_move;
            }
        }
        0
    }

    /// 転置表を更新する(支配関係によるパレートフロンティア維持)．
    ///
    /// 証明済み・反証済みエントリは持ち駒の半順序(`hand_gte`)に基づく
    /// 支配関係を利用し，冗長なエントリを自動的に除去する:
    ///
    /// - **証明(pn=0)**: 持ち駒が少ないほど強い証明(パレート最小集合を保持)．
    ///   `hand_new ≤ hand_existing` なら既存は不要．
    /// - **反証(dn=0)**: 持ち駒が多いほど強い反証(パレート最大集合を保持)．
    ///   `hand_new ≥ hand_existing` かつ `remaining_new ≥ remaining_existing` なら既存は不要．
    ///
    /// # 引数
    ///
    /// - `remaining`: 登録時の残り探索深さ．
    ///   `REMAINING_INFINITE` は深さ制限なし(真の証明/反証)を示す．
    #[inline(always)]
    fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, 0);
    }

    /// ベストムーブ付きで転置表を更新する．
    ///
    /// MID ループの中間結果保存時に，最善子ノードの手を記録する．
    /// 次回同一局面に到達した際の手順改善(Dynamic Move Ordering)に使用する．
    #[inline(always)]
    fn store_with_best_move(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        best_move: u16,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, best_move);
    }

    /// 経路依存フラグ付きで転置表を更新する．
    ///
    /// `path_dependent = true` の反証エントリは，ループ検出に由来し
    /// 異なる探索経路では無効になる可能性がある．
    /// `remaining` を有限値に制限して保存することで，
    /// より深い探索で自動的に再評価される．
    #[inline(always)]
    fn store_path_dep(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        path_dependent: bool,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, path_dependent, 0);
    }

    #[inline(always)]
    fn store_impl(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        path_dependent: bool,
        best_move: u16,
    ) {
        // remaining 分布カウンタ用インデックス
        let rem_idx = if remaining == REMAINING_INFINITE { 32 } else { (remaining as usize).min(31) };

        let entries =
            self.tt.entry(pos_key).or_default();

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        for e in entries.iter() {
            // 証明済みエントリが支配: hand ≥ e.hand → 新エントリの持ち駒で詰み確定
            if e.pn == 0 && hand_gte_forward_chain(&hand, &e.hand) {
                self.diag_dominated_skip += 1;
                return;
            }
            // 反証済みエントリが支配: e.hand ≥ hand かつ十分な深さ → 不詰確定
            // GHI: 経路依存の反証は経路非依存の反証に支配されない
            // 経路依存の新エントリ(remaining 免除)は経路非依存の既存エントリに支配されない
            if e.dn == 0
                && !e.path_dependent
                && !path_dependent
                && hand_gte_forward_chain(&e.hand, &hand)
                && e.remaining >= remaining
            {
                self.diag_dominated_skip += 1;
                return;
            }
        }

        if pn == 0 {
            // === 証明済みエントリの挿入 ===
            // パレートフロンティア(最小持ち駒集合)を維持:
            // 新証明に支配される既存エントリを除去する．
            // - 証明済み: e.hand ≥ hand → より多い持ち駒での証明は冗長
            // - 中間: e.hand ≥ hand → lookup 時に新証明がヒットするため不要
            entries.retain(|e| {
                // 反証済みは保護(証明と反証は異なる持ち駒領域で共存しうる)
                if e.dn == 0 {
                    return true;
                }
                !hand_gte_forward_chain(&e.hand, &hand)
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, source });
            self.diag_proof_inserts += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            return;
        }

        if dn == 0 {
            // === 反証済みエントリの挿入 ===
            // パレートフロンティア(最大持ち駒 × 最大remaining 集合)を維持:
            // 新反証に支配される既存エントリを除去する．
            // GHI: 経路依存の反証は remaining 免除で lookup に使えるため保護する
            //
            entries.retain(|e| {
                // 証明済みは保護
                if e.pn == 0 {
                    return true;
                }
                if e.dn == 0 {
                    // 経路依存の反証は remaining チェック免除で lookup に使えるため，
                    // 経路非依存の反証では置換しない(remaining 不足で使えなくなる)
                    if e.path_dependent && !path_dependent {
                        return true;
                    }
                    // 反証: e.hand ≤ hand かつ e.remaining ≤ remaining → 冗長
                    return !(hand_gte_forward_chain(&hand, &e.hand)
                        && remaining >= e.remaining);
                }
                // 中間エントリは保護(remaining の不一致で必要になりうる)
                true
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, source });
            self.diag_disproof_inserts += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            return;
        }

        // === 中間エントリ(pn > 0, dn > 0)の挿入 ===

        // 同一持ち駒の既存エントリを更新
        for e in entries.iter_mut() {
            if e.hand == hand {
                // 証明済み(pn=0)は上の共通チェックで return 済み
                if e.dn == 0 {
                    // 反証済みエントリの remaining が新エントリの remaining を
                    // カバーしている場合のみ中間値の上書きをブロックする．
                    // カバーしていない場合(e.remaining < remaining)は，
                    // look_up でも使えない「死んだ」反証であるため，
                    // 中間値で上書きして探索の進行を保証する．
                    if e.remaining >= remaining || e.path_dependent {
                        return;
                    }
                    // remaining 不足の反証: 中間値で上書き
                }
                e.pn = pn;
                e.dn = dn;
                e.remaining = remaining;
                e.source = source;
                e.path_dependent = false;
                if best_move != 0 {
                    e.best_move = best_move;
                }
                self.diag_intermediate_update += 1;
                return;
            }
        }

        // 新規エントリを追加
        if entries.len() < MAX_TT_ENTRIES_PER_POSITION {
            entries.push(DfPnEntry {
                hand,
                pn,
                dn,
                remaining,
                best_move,
                path_dependent: false,
                source,
            });
            self.diag_intermediate_new += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            #[cfg(feature = "profile")]
            {
                if entries.len() > self.max_entries_per_position {
                    self.max_entries_per_position = entries.len();
                }
            }
        } else {
            #[cfg(feature = "profile")]
            { self.overflow_count += 1; }
            // 上限到達時: 証明/反証済みエントリを保護しつつ，
            // 「最も未解決な」(|pn - dn| が最小の)エントリを置換する．
            let mut worst_idx: Option<usize> = None;
            let mut worst_score: u64 = u64::MAX;
            for (i, e) in entries.iter().enumerate() {
                if e.pn == 0 || e.dn == 0 {
                    continue;
                }
                let score = if e.pn > e.dn {
                    (e.pn - e.dn) as u64
                } else {
                    (e.dn - e.pn) as u64
                };
                if score < worst_score {
                    worst_score = score;
                    worst_idx = Some(i);
                }
            }
            if let Some(idx) = worst_idx {
                entries[idx] = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, source };
            } else {
                #[cfg(feature = "profile")]
                { self.overflow_no_victim_count += 1; }
            }
        }
    }

    /// 証明済みエントリの証明駒(登録時の持ち駒)を返す．
    ///
    /// 持ち駒優越で一致する証明済みエントリの hand を返す．
    /// 見つからない場合は渡された hand をそのまま返す．
    #[inline(always)]
    fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                    return e.hand;
                }
            }
        }
        *hand
    }

    /// 反証済みエントリの反証駒(登録時の持ち駒)を返す．
    ///
    /// 持ち駒劣越で一致する反証済みエントリの hand を返す．
    /// 見つからない場合は渡された hand をそのまま返す．
    /// 注: 反証を att_hand で保存する最適化により現在未使用だが，
    /// デバッグ・分析用に保持．
    #[allow(dead_code)]
    #[inline(always)]
    fn get_disproof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte_forward_chain(&e.hand, hand) {
                    return e.hand;
                }
            }
        }
        *hand
    }

    /// 反証エントリが経路依存(path_dependent)かどうかを返す．
    ///
    /// OR ノードで子の反証を集約する際，経路依存の子反証が含まれるなら
    /// 親の反証も経路依存として保存する必要がある．
    /// GHI 由来のループ反証は IDS 間で経路が変わると無効になりうるため．
    fn has_path_dependent_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte_forward_chain(&e.hand, hand) {
                    return e.path_dependent;
                }
            }
        }
        false
    }

    /// 反証エントリの remaining を返す．
    ///
    /// NM の remaining 伝播に使用: 子の NM の remaining が
    /// REMAINING_INFINITE なら親も REMAINING_INFINITE にできる．
    fn get_disproof_remaining(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte_forward_chain(&e.hand, hand) {
                    return e.remaining;
                }
            }
        }
        0
    }

    /// lookup が実際に使用する反証エントリの (remaining, path_dependent) を返す．
    ///
    /// `has_path_dependent_disproof` / `get_disproof_remaining` は
    /// 最初にマッチした反証エントリの値を返すが，`look_up` は
    /// `e.remaining >= remaining || e.path_dependent` を追加でチェックする．
    /// このため，lookup が使うエントリと情報取得関数が返すエントリが
    /// 食い違う場合がある．この関数は lookup と同じ条件でマッチした
    /// エントリの情報を返す．
    fn get_effective_disproof_info(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> Option<(u16, bool)> {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0
                    && hand_gte_forward_chain(&e.hand, hand)
                    && (e.remaining >= remaining || e.path_dependent)
                {
                    return Some((e.remaining, e.path_dependent));
                }
            }
        }
        None
    }

    /// 全エントリをクリアする．
    fn clear(&mut self) {
        self.tt.clear();
    }

    /// 確定済みエントリ(証明・確定反証)を保持し，中間エントリを除去する．
    ///
    /// 反復深化 df-pn で使用: 浅い深さの中間エントリや
    /// 深さ制限による仮反証エントリを除去しつつ，
    /// 確定した証明・反証エントリは再利用する．
    fn retain_proofs(&mut self) {
        self.tt.retain(|_key, entries| {
            entries.retain(|e| {
                // 証明(pn=0): 常に保持
                // 確定反証(dn=0 かつ path_dependent=false): 経路非依存の真の不詰
                //   → どのパスからアクセスしても同じ結果になるため IDS 間で再利用安全
                // 経路依存反証(dn=0 かつ path_dependent=true): ループ由来
                //   → 異なる IDS 反復では経路が変わり無効になりうるため破棄
                e.pn == 0 || (e.dn == 0 && !e.path_dependent)
            });
            !entries.is_empty()
        });
    }

    /// 経路依存の反証エントリを除去する．
    ///
    /// IDS 反復間で使用: ループ由来の経路依存反証は異なる反復では
    /// 無効になりうるため，次の反復の前に除去する．
    /// 証明・非経路依存反証・中間エントリは保持する．
    fn remove_path_dependent_disproofs(&mut self) {
        for entries in self.tt.values_mut() {
            entries.retain(|e| !(e.dn == 0 && e.path_dependent));
        }
    }

    /// 証明エントリ(pn=0)のみを保持し，NM を含む他の全エントリを除去する．
    ///
    /// PNS → IDS 切替時に使用: PNS で蓄積された深さ制限由来の仮 NM エントリを
    /// 除去し，IDS が独立して NM を検出できるようにする．
    /// 証明(pn=0)は常に正しいため保持する．
    fn retain_proofs_only(&mut self) {
        self.tt.retain(|_key, entries| {
            entries.retain(|e| e.pn == 0);
            !entries.is_empty()
        });
    }

    /// 確定エントリ(証明 pn=0 と反証 dn=0)のみ保持し，中間エントリを除去する．
    fn retain_terminal(&mut self) {
        self.tt.retain(|_key, entries| {
            entries.retain(|e| e.pn == 0 || e.dn == 0);
            !entries.is_empty()
        });
    }

    /// 浅い反復で remaining が不足する中間・反証エントリを除去する．
    ///
    /// IDS 反復間で使用: スラッシング防止用の中間エントリ
    /// (pn >= INF-1, dn > 0)を除去し，深い反復で再評価させる．
    ///
    /// 反証エントリは除去しない: PNS で蓄積された深い ply の反証は
    /// remaining が小さい(saved_depth - ply)が，full depth で有効であり，
    /// 除去すると root_pn が大幅に増加する．
    fn remove_stale_for_ids(&mut self) {
        self.tt.retain(|_, entries| {
            entries.retain(|e| {
                // 証明は常に保持
                if e.pn == 0 { return true; }
                // スラッシング防止エントリ(pn >= INF-1, dn > 0)は除去
                if e.pn >= INF - 1 && e.dn > 0 { return false; }
                // remaining=0 の反証は除去．
                // 同一 depth 内でしか再利用できず，IDS depth が進むと
                // remaining > 0 の検索にヒットしないため不要．
                // IDS 反復間でメモリを解放する．
                if e.dn == 0 && e.remaining == 0 { return false; }
                // 反証・その他は保持
                true
            });
            !entries.is_empty()
        });
    }


    /// 指定局面の証明エントリ(pn=0)を除去する．
    ///
    /// IDS の浅い深さで PNS 由来の証明が使われた場合に，
    /// 根の証明を除去して full depth で再証明させるために使用する．
    fn remove_proof(&mut self, pos_key: u64, hand: &[u8; HAND_KINDS]) {
        if let Some(entries) = self.tt.get_mut(&pos_key) {
            entries.retain(|e| !(e.pn == 0 && e.hand == *hand));
        }
    }

    /// TT のポジション数を返す．
    fn len(&self) -> usize {
        self.tt.len()
    }

    /// 反復内 periodic GC: 低価値エントリを除去してメモリを解放する．
    ///
    /// IDS 反復間 GC(`remove_stale_for_ids`)と異なり，反復内で TT が肥大化した際に
    /// 呼び出される．`remaining_threshold` 以下のエントリを除去対象とする．
    ///
    /// 除去対象:
    /// - remaining ≤ threshold の反証(dn=0): 浅い深さ制限到達時の反証
    /// - remaining ≤ threshold の中間(pn>0, dn>0): 浅い探索の中間結果
    /// - 証明(pn=0)は常に保持
    ///
    /// 返り値: 除去されたエントリ数．
    fn gc_shallow_entries(&mut self, remaining_threshold: u16) -> usize {
        let mut removed = 0usize;
        self.tt.retain(|_, entries| {
            let before = entries.len();
            entries.retain(|e| {
                // 証明は常に保持
                if e.pn == 0 { return true; }
                // REMAINING_INFINITE の反証は常に保持(真の不詰)
                if e.dn == 0 && e.remaining == REMAINING_INFINITE { return true; }
                // remaining ≤ threshold のエントリを除去
                e.remaining > remaining_threshold
            });
            removed += before - entries.len();
            !entries.is_empty()
        });
        removed
    }

    /// TT の全エントリ数(全ポジションの Vec 長の合計)を返す．
    ///
    /// 同一盤面・異なる持ち駒のエントリを含む総数．
    /// `len()` がポジション数(HashMap キー数)を返すのに対し，
    /// `total_entries()` は実際のメモリ消費に比例する値を返す．
    #[cfg(feature = "tt_diag")]
    fn total_entries(&self) -> usize {
        self.tt.values().map(|v| v.len()).sum()
    }

    /// 指定ポジションのエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    fn entries_for_position(&self, pos_key: u64) -> usize {
        self.tt.get(&pos_key).map_or(0, |v| v.len())
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    fn dump_entries(&self, pos_key: u64) {
        if let Some(entries) = self.tt.get(&pos_key) {
            for (i, e) in entries.iter().enumerate() {
                eprintln!(
                    "[tt_dump] entry[{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn, e.dn, e.remaining, e.path_dependent, &e.hand
                );
            }
        }
    }

    /// 証明済み(pn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    fn count_proven(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter()).filter(|e| e.pn == 0).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    fn count_disproven(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter()).filter(|e| e.dn == 0).count()
    }

    /// 中間(pn>0 かつ dn>0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    fn count_intermediate(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter())
            .filter(|e| e.pn > 0 && e.dn > 0).count()
    }

    /// TT コンテンツの詳細分析(診断用)．
    fn dump_content_analysis(&self) {
        let mut proof_count: u64 = 0;
        let mut disproof_count: u64 = 0;
        let mut intermediate_count: u64 = 0;
        // 反証の remaining 分布
        let mut disproof_rem: [u64; 33] = [0; 33];
        // 中間エントリの pn 分布
        let mut inter_pn_buckets: [u64; 8] = [0; 8]; // [1], [2-5], [6-20], [21-100], [101-1K], [1K-10K], [10K-100K], [100K+]
        // 中間エントリの remaining 分布
        let mut inter_rem: [u64; 33] = [0; 33];
        // 中間エントリの dn 分布
        let mut inter_dn_buckets: [u64; 5] = [0; 5]; // [1], [2-5], [6-20], [21-100], [100+]
        // 局面あたりのエントリ構成
        let mut pos_proof_only: u64 = 0;
        let mut pos_disproof_only: u64 = 0;
        let mut pos_inter_only: u64 = 0;
        let mut pos_mixed: u64 = 0;

        for entries in self.tt.values() {
            let mut has_proof = false;
            let mut has_disproof = false;
            let mut has_inter = false;
            for e in entries {
                if e.pn == 0 {
                    proof_count += 1;
                    has_proof = true;
                } else if e.dn == 0 {
                    disproof_count += 1;
                    has_disproof = true;
                    let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                    disproof_rem[ri] += 1;
                } else {
                    intermediate_count += 1;
                    has_inter = true;
                    let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                    inter_rem[ri] += 1;
                    // pn バケット
                    let pb = match e.pn {
                        1 => 0,
                        2..=5 => 1,
                        6..=20 => 2,
                        21..=100 => 3,
                        101..=1000 => 4,
                        1001..=10000 => 5,
                        10001..=100000 => 6,
                        _ => 7,
                    };
                    inter_pn_buckets[pb] += 1;
                    // dn バケット
                    let db = match e.dn {
                        1 => 0,
                        2..=5 => 1,
                        6..=20 => 2,
                        21..=100 => 3,
                        _ => 4,
                    };
                    inter_dn_buckets[db] += 1;
                }
            }
            match (has_proof, has_disproof, has_inter) {
                (true, false, false) => pos_proof_only += 1,
                (false, true, false) => pos_disproof_only += 1,
                (false, false, true) => pos_inter_only += 1,
                _ => pos_mixed += 1,
            }
        }

        eprintln!("\n=== TT Content Analysis ===");
        eprintln!("positions: {}  entries: proof={} disproof={} intermediate={}",
            self.tt.len(), proof_count, disproof_count, intermediate_count);
        eprintln!("pos composition: proof_only={} disproof_only={} inter_only={} mixed={}",
            pos_proof_only, pos_disproof_only, pos_inter_only, pos_mixed);

        // 反証 remaining 分布
        let dr: Vec<String> = disproof_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        eprintln!("disproof remaining: [{}]", dr.join(", "));

        // 中間 remaining 分布
        let ir: Vec<String> = inter_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        eprintln!("intermediate remaining: [{}]", ir.join(", "));

        // 中間 pn 分布
        let pn_labels = ["pn=1", "pn=2-5", "pn=6-20", "pn=21-100", "pn=101-1K", "pn=1K-10K", "pn=10K-100K", "pn=100K+"];
        let pb: Vec<String> = inter_pn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", pn_labels[i], c))
            .collect();
        eprintln!("intermediate pn dist: [{}]", pb.join(", "));

        // 中間 dn 分布
        let dn_labels = ["dn=1", "dn=2-5", "dn=6-20", "dn=21-100", "dn=100+"];
        let db: Vec<String> = inter_dn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", dn_labels[i], c))
            .collect();
        eprintln!("intermediate dn dist: [{}]", db.join(", "));
    }

    /// TT ガベージコレクション: メモリ使用量を抑制する．
    ///
    /// 2段階の GC を実行する:
    /// 1. 中間エントリ(pn>0 かつ dn>0)のうち，remaining が閾値以下のものを除去
    /// 2. それでも閾値を超える場合，全中間エントリを除去(retain_proofs 相当)
    ///
    /// 証明済み(pn=0)と確定反証(dn=0, remaining=∞)は常に保持する．
    fn gc(&mut self, target_size: usize) {
        if self.tt.len() <= target_size {
            return;
        }

        // Phase 1: remaining が小さい中間エントリを除去
        // (浅い探索の仮結果は再計算可能)
        let median_remaining = 8u16;
        self.tt.retain(|_key, entries| {
            entries.retain(|e| {
                e.pn == 0
                    || e.dn == 0
                    || e.remaining > median_remaining
            });
            !entries.is_empty()
        });

        if self.tt.len() <= target_size {
            return;
        }

        // Phase 2: 全中間エントリを除去(確定結果のみ保持)
        self.retain_proofs();
    }

    /// 他の TT から確定エントリ(証明・反証)をマージする．
    ///
    /// `other` の全エントリを走査し，証明済み(pn=0)および
    /// 確定反証(dn=0)のエントリのみを `self` に `store()` する．
    /// 中間エントリは破棄される．
    /// プロファイル統計をリセットする．
    #[cfg(feature = "profile")]
    fn reset_profile(&mut self) {
        self.overflow_count = 0;
        self.overflow_no_victim_count = 0;
        self.max_entries_per_position = 0;
    }
}

/// 指し手に応じて子ノードの持ち駒情報を親ノード視点に変換する．
///
/// 指し手による持ち駒の入出を補正する汎用関数．
/// 証明駒・反証駒の両方に同じロジックが適用できる:
/// 駒打ちは持ち駒を消費し，駒取りは持ち駒を増やすため，
/// 親ノード視点では打った駒を加算，取った駒を減算する．
///
/// - 駒打ち: 子の駒 + 打った駒種1枚(持ち駒を消費するため)
/// - 駒取り: 子の駒 - 取った駒種1枚(持ち駒が増えるため)
/// - それ以外: 子の駒をそのまま使用
#[inline]
fn adjust_hand_for_move(
    m: Move,
    child_proof: &[u8; HAND_KINDS],
) -> [u8; HAND_KINDS] {
    let mut ph = *child_proof;
    if m.is_drop() {
        if let Some(pt) = m.drop_piece_type() {
            if let Some(hi) = pt.hand_index() {
                ph[hi] = ph[hi].saturating_add(1);
            }
        }
    } else {
        let cap = m.captured_piece_raw();
        if cap > 0 {
            // captured_piece_raw() は Piece 値(色付き)を返す．
            // 白駒はオフセット 16 が加算されている．

            let piece = Piece::from_raw_u8(cap);
            if let Some(pt) = piece.piece_type() {
                let base_pt =
                    pt.unpromoted().unwrap_or(pt);
                if let Some(hi) = base_pt.hand_index() {
                    ph[hi] = ph[hi].saturating_sub(1);
                }
            }
        }
    }
    ph
}

/// Df-Pn ソルバー．
pub struct DfPnSolver {
    /// 最大探索手数．
    depth: u32,
    /// 最大ノード数．
    max_nodes: u64,
    /// 引き分け手数．
    draw_ply: u32,
    /// 実行時間制限．
    timeout: Duration,
    /// 転置表(固定サイズ，証明駒/反証駒対応)．
    table: TranspositionTable,
    /// 探索ノード数．
    nodes_searched: u64,
    /// 探索中の最大ply(デバッグ用)．
    max_ply: u32,
    /// ply別ノード数(デバッグ用)．
    ply_nodes: [u64; 64],
    /// ply別MIDループイテレーション数(デバッグ用)．
    ply_iters: [u64; 64],
    /// ply別停滞ペナルティ回数(デバッグ用)．
    ply_stag_penalties: [u64; 64],
    /// ルート局面情報(進捗追跡用)．
    diag_root_pk: u64,
    diag_root_hand: [u8; HAND_KINDS],
    /// 探索中のパス(ループ検出用，フルハッシュ)．
    path: FxHashSet<u64>,
    /// 探索開始時刻．
    start_time: Instant,
    /// タイムアウトしたかどうか．
    timed_out: bool,
    /// 攻め方の手番色(solve 時に設定)．
    attacker: Color,
    /// 最短手数探索を行うかどうか(デフォルト: true)．
    ///
    /// true の場合，`solve()` は `complete_or_proofs()` を呼び出して
    /// 全 OR ノードの未証明子を追加証明し，最短手順を保証する．
    /// false の場合，最初に見つかった詰み手順をそのまま返す．
    find_shortest: bool,
    /// PV 復元フェーズで未証明子1つあたりに割り当てるノード予算(デフォルト: 1024)．
    ///
    /// 長手数の詰将棋で [`TsumeResult::CheckmateNoPv`] が返る場合，
    /// この値を増やすことで PV 復元の成功率が向上する．
    pv_nodes_per_child: u64,
    /// 静的詰め探索の1子あたりのノード予算．
    ///
    /// 子ノード初期化時に，Df-Pn のオーバーヘッドなしで
    /// 再帰的に N 手詰めを検出する．予算を使い切ると探索を打ち切る．
    /// 0 にすると静的詰め探索を無効化し，インライン1手・3手詰め判定のみ行う．
    /// デフォルトは 0(無効)．
    mate_budget: u32,
    /// TT GC 閾値: TT のポジション数がこの値を超えると GC を実行する．
    ///
    /// 0 にすると GC を無効化する．
    /// デフォルトは 0(無効)．超長手数問題で OOM を防ぐ場合に設定する．
    /// 推奨値: 探索ノード数の 1/5〜1/2 程度(例: 100M ノードなら 20M〜50M)．
    tt_gc_threshold: usize,
    /// 直前の `generate_defense_moves` で計算されたチェーンマスのビットボード．
    ///
    /// `mid()` 内で合駒がチェーンマスへのドロップかどうかを判定するために使用．
    /// 各 `mid()` 呼び出しで更新され，飛び駒の王手がない場合は空．
    chain_bb_cache: Bitboard,
    /// TT ベース合駒プレフィルタの発火回数(診断用)．
    prefilter_hits: u64,
    /// NM 昇格の反証判定キャッシュ: 判定が false だった局面キーの集合．
    ///
    /// `depth_limit_all_checks_refutable` は局面のみに依存し探索深さに依存しないため，
    /// 一度 false と判定された局面を再判定する必要はない．
    /// MID 内部で同一局面の重複判定を回避し，パフォーマンスを改善する．
    refutable_check_failed: FxHashSet<u64>,
    /// OR ノードの子ポジション別 stale effort 追跡．
    ///
    /// キー: 子ポジションの pos_key．
    /// 値: (stale_effort, best_pn)．
    /// - stale_effort: pn が改善しなかった探索のノード累積．
    ///   pn が best_pn を下回ると 0 にリセットされる．
    /// - best_pn: これまでに観測された最低 pn．
    ///
    /// stale_effort に基づく pn ペナルティを選択時に加算し，
    /// 不正解手(pn 停滞)から正解手(pn 減少)への切替を促進する．
    /// OR ノードの子ポジション別 effort 追跡(現在未使用)．
    or_effort: FxHashMap<u64, (u64, u32, u64)>,
    /// 次に TT GC チェックを行うノード数．
    next_gc_check: u64,
    /// Killer Move テーブル(OR ノード用)．
    ///
    /// ply ごとに最大 2 つの killer move(Move16)を保持する．
    /// 閾値超過でカットオフを引き起こした手を記録し，
    /// 同じ ply の他の局面でも優先的に探索する．
    /// TT Best Move とは異なり局面に依存しない手順ヒントを提供する．
    killer_table: Vec<[u16; 2]>,
    /// プロファイリング統計情報(`profile` feature 有効時のみ)．
    #[cfg(feature = "profile")]
    pub profile_stats: ProfileStats,
    /// TT 診断: 監視対象の ply(0 = 無効)．
    ///
    /// 指定 ply で MID ループの再帰前後に TT サイズを出力し，
    /// エントリ爆増の原因を特定する．
    #[cfg(feature = "tt_diag")]
    diag_ply: u32,
    /// TT 診断: 監視対象の手(USI 形式，例: "P*7g")．
    ///
    /// 空文字列の場合は ply のみでフィルタする．
    #[cfg(feature = "tt_diag")]
    diag_move_usi: String,
    /// TT 診断: MID ループの反復回数上限(0 = 無制限)．
    ///
    /// 爆増が起きる手を特定した後，少数回の反復に絞って詳細を確認する．
    #[cfg(feature = "tt_diag")]
    diag_max_iterations: u32,
    /// TT 診断: MID での deferred → children 逐次活性化回数．
    #[cfg(feature = "tt_diag")]
    pub diag_mid_deferred_activations: u64,
    /// TT 診断: PNS での deferred_drops 活性化回数．
    #[cfg(feature = "tt_diag")]
    pub diag_pns_deferred_activations: u64,
    /// TT 診断: PNS で活性化時に既に TT 証明済みだった回数．
    #[cfg(feature = "tt_diag")]
    pub diag_pns_deferred_already_proven: u64,
    /// TT 診断: cross_deduce_deferred で証明除去された合駒数．
    #[cfg(feature = "tt_diag")]
    pub diag_cross_deduce_hits: u64,
    /// TT 診断: AND ノード MID ループで deferred_children あり & all_proved=false の回数．
    #[cfg(feature = "tt_diag")]
    pub diag_deferred_not_ready: u64,
    /// TT 診断: AND ノード MID ループで deferred_children あり & all_proved=true の回数．
    #[cfg(feature = "tt_diag")]
    pub diag_deferred_ready: u64,
    /// TT 診断: AND ノードで prefilter 後 deferred_children に入った合駒数(累計)．
    #[cfg(feature = "tt_diag")]
    pub diag_deferred_enqueued: u64,
    /// TT 診断: MID ループの総反復数(nodes_searched 以上になりうる)．
    #[cfg(feature = "tt_diag")]
    pub diag_mid_loop_iters: u64,
    /// TT 診断: prefilter が remaining < 3 のためスキップされた回数．
    #[cfg(feature = "tt_diag")]
    pub diag_prefilter_skip_remaining: u64,
    /// TT 診断: prefilter が試行されたが TT ヒットしなかった回数．
    #[cfg(feature = "tt_diag")]
    pub diag_prefilter_miss: u64,
    /// TT 診断: ply ごとの MID 訪問回数(最大64手)．
    #[cfg(feature = "tt_diag")]
    pub diag_ply_visits: [u64; 64],
    /// TT 診断: ply ごとの pn=0 証明ストア回数．
    #[cfg(feature = "tt_diag")]
    pub diag_ply_proofs: [u64; 64],
    /// TT 診断: try_capture_tt_proof の呼び出し / ヒット回数．
    #[cfg(feature = "tt_diag")]
    pub diag_capture_tt_calls: u64,
    #[cfg(feature = "tt_diag")]
    pub diag_capture_tt_hits: u64,
    /// TT 診断: MID 早期リターン(閾値チェック)回数．
    #[cfg(feature = "tt_diag")]
    pub diag_threshold_exits: u64,
    /// TT 診断: MID 早期リターン(tt_pn==0 || tt_dn==0)回数．
    #[cfg(feature = "tt_diag")]
    pub diag_terminal_exits: u64,
    /// TT 診断: MID ループ内 break 原因別カウンタ．
    #[cfg(feature = "tt_diag")]
    pub diag_loop_break_proved: u64,
    #[cfg(feature = "tt_diag")]
    pub diag_loop_break_threshold: u64,
    #[cfg(feature = "tt_diag")]
    pub diag_loop_break_nodes: u64,
    #[cfg(feature = "tt_diag")]
    pub diag_in_path_exits: u64,
    /// TT 診断: init フェーズでの AND 反証リターン回数(ply 別)．
    #[cfg(feature = "tt_diag")]
    pub diag_init_and_disproof_exits: u64,
    /// TT 診断: 単一子最適化パス回数(ply 別)．
    #[cfg(feature = "tt_diag")]
    pub diag_single_child_exits: u64,
    /// TT 診断: ノード制限によるリターン回数．
    #[cfg(feature = "tt_diag")]
    pub diag_node_limit_exits: u64,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64, draw_ply: u32) -> Self {
        Self::with_timeout(depth, max_nodes, draw_ply, 300)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    pub fn with_timeout(depth: u32, max_nodes: u64, draw_ply: u32, timeout_secs: u64) -> Self {
        DfPnSolver {
            depth,
            max_nodes,
            draw_ply,
            timeout: Duration::from_secs(timeout_secs),
            find_shortest: true,
            pv_nodes_per_child: 1024,
            mate_budget: 0,
            chain_bb_cache: Bitboard::EMPTY,
            prefilter_hits: 0,
            refutable_check_failed: FxHashSet::default(),
            or_effort: FxHashMap::default(), // (stale_effort, best_pn, total_effort)
            tt_gc_threshold: 0,
            next_gc_check: 0,
            killer_table: Vec::new(),
            table: TranspositionTable::new(),
            nodes_searched: 0,
            max_ply: 0,
            ply_nodes: [0; 64],
            ply_iters: [0; 64],
            ply_stag_penalties: [0; 64],
            diag_root_pk: 0,
            diag_root_hand: [0; HAND_KINDS],
            path: FxHashSet::default(),
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            #[cfg(feature = "profile")]
            profile_stats: ProfileStats::default(),
            #[cfg(feature = "tt_diag")]
            diag_ply: 0,
            #[cfg(feature = "tt_diag")]
            diag_move_usi: String::new(),
            #[cfg(feature = "tt_diag")]
            diag_max_iterations: 0,
            #[cfg(feature = "tt_diag")]
            diag_mid_deferred_activations: 0,
            #[cfg(feature = "tt_diag")]
            diag_pns_deferred_activations: 0,
            #[cfg(feature = "tt_diag")]
            diag_pns_deferred_already_proven: 0,
            #[cfg(feature = "tt_diag")]
            diag_cross_deduce_hits: 0,
            #[cfg(feature = "tt_diag")]
            diag_deferred_not_ready: 0,
            #[cfg(feature = "tt_diag")]
            diag_deferred_ready: 0,
            #[cfg(feature = "tt_diag")]
            diag_deferred_enqueued: 0,
            #[cfg(feature = "tt_diag")]
            diag_mid_loop_iters: 0,
            #[cfg(feature = "tt_diag")]
            diag_prefilter_skip_remaining: 0,
            #[cfg(feature = "tt_diag")]
            diag_prefilter_miss: 0,
            #[cfg(feature = "tt_diag")]
            diag_ply_visits: [0u64; 64],
            #[cfg(feature = "tt_diag")]
            diag_ply_proofs: [0u64; 64],
            #[cfg(feature = "tt_diag")]
            diag_capture_tt_calls: 0,
            #[cfg(feature = "tt_diag")]
            diag_capture_tt_hits: 0,
            #[cfg(feature = "tt_diag")]
            diag_threshold_exits: 0,
            #[cfg(feature = "tt_diag")]
            diag_terminal_exits: 0,
            #[cfg(feature = "tt_diag")]
            diag_loop_break_proved: 0,
            #[cfg(feature = "tt_diag")]
            diag_loop_break_threshold: 0,
            #[cfg(feature = "tt_diag")]
            diag_loop_break_nodes: 0,
            #[cfg(feature = "tt_diag")]
            diag_in_path_exits: 0,
            #[cfg(feature = "tt_diag")]
            diag_init_and_disproof_exits: 0,
            #[cfg(feature = "tt_diag")]
            diag_single_child_exits: 0,
            #[cfg(feature = "tt_diag")]
            diag_node_limit_exits: 0,
        }
    }

    /// デフォルトパラメータでソルバーを生成する．
    pub fn default_solver() -> Self {
        Self::new(31, 1_048_576, 32767)
    }

    /// 最短手数探索の有無を設定する．
    ///
    /// `false` にすると最初に見つかった詰み手順をそのまま返す(高速化)．
    pub fn set_find_shortest(&mut self, v: bool) -> &mut Self {
        self.find_shortest = v;
        self
    }

    /// PV 復元フェーズの1子あたりノード予算を設定する．
    ///
    /// デフォルトは 1024．長手数(17手以上)の詰将棋で
    /// `CheckmateNoPv` が返る場合に増やすと効果的．
    pub fn set_pv_nodes_per_child(&mut self, v: u64) -> &mut Self {
        self.pv_nodes_per_child = v;
        self
    }

    /// 静的詰め探索のノード予算を設定する．
    ///
    /// 0 にすると静的詰め探索を無効化し，インライン1手・3手詰め判定のみ行う．
    /// デフォルトは 0(無効)．
    pub fn set_mate_budget(&mut self, v: u32) -> &mut Self {
        self.mate_budget = v;
        self
    }

    /// TT GC 閾値を設定する．
    ///
    /// TT のポジション数がこの値を超えると GC を実行する．
    /// 0 にすると GC を無効化する．デフォルトは 2,000,000．
    pub fn set_tt_gc_threshold(&mut self, v: usize) -> &mut Self {
        self.tt_gc_threshold = v;
        self
    }

    /// TT 診断の監視対象を設定する．
    ///
    /// 指定 ply で指定手(USI 形式)が選択された MID ループ反復ごとに，
    /// TT サイズの変化を stderr に出力する．
    ///
    /// # 引数
    ///
    /// - `ply`: 監視対象の ply(0 で無効化)
    /// - `move_usi`: 監視対象の手(空文字列で ply のみフィルタ)
    /// - `max_iterations`: MID ループの反復回数上限(0 で無制限)
    #[cfg(feature = "tt_diag")]
    pub fn set_tt_diag(
        &mut self,
        ply: u32,
        move_usi: &str,
        max_iterations: u32,
    ) -> &mut Self {
        self.diag_ply = ply;
        self.diag_move_usi = move_usi.to_string();
        self.diag_max_iterations = max_iterations;
        self
    }

    /// TT のプロファイル統計を `profile_stats` に転記する．
    ///
    /// `solve()` 完了後に呼ぶことで，TT エントリ溢れ統計を確認できる．
    #[cfg(feature = "profile")]
    pub fn sync_tt_profile(&mut self) {
        self.profile_stats.tt_overflow_count = self.table.overflow_count;
        self.profile_stats.tt_overflow_no_victim_count =
            self.table.overflow_no_victim_count;
        self.profile_stats.tt_max_entries_per_position =
            self.table.max_entries_per_position;
    }

    /// タイムアウトしたかどうかを返す．
    #[inline]
    fn is_timed_out(&self) -> bool {
        self.timed_out || self.start_time.elapsed() >= self.timeout
    }

    /// Deep df-pn: 未探索ノードの pn 初期値に深さバイアスを適用する．
    ///
    /// 標準 df-pn は TT ミス時に `(pn=1, dn=1)` を返すが，これだと
    /// OR ノードで未探索の子が常に最小 pn を持ち，探索済みの子から
    /// 未探索の子へ頻繁にフォーカスが切り替わる(seesaw effect)．
    ///
    /// Deep df-pn では深い ply(depth の後半)にのみバイアスを適用:
    /// `pn = 1 + (ply - depth/2) / R` (ply > depth/2 の場合)．
    /// 浅い ply は標準 df-pn と同じ `pn=1` を維持し，
    /// 不詰検出など浅い探索の効率を損なわない．
    #[inline]
    fn look_up_pn_dn(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u64) {
        let result = self.table.look_up(pos_key, hand, remaining);
        if result.0 == 1 && result.1 == 1 && result.2 == 0 {
            // TT ミス: Deep df-pn バイアスを適用(深い ply のみ)
            let ply = (self.depth as u32).saturating_sub(remaining as u32);
            let half_depth = self.depth / 2;
            if ply > half_depth {
                const DEEP_DFPN_R: u32 = 4;
                let biased_pn = 1 + (ply - half_depth) / DEEP_DFPN_R;
                (biased_pn, 1, 0)
            } else {
                (1, 1, 0)
            }
        } else {
            result
        }
    }

    /// 転置表を更新する(位置キー＋持ち駒指定)．
    #[inline]
    fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
    ) {
        self.table.store(pos_key, hand, pn, dn, remaining, source);
    }

    /// ベストムーブ付きで転置表を更新する．
    #[inline]
    fn store_with_best_move(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        best_move: u16,
    ) {
        self.table.store_with_best_move(pos_key, hand, pn, dn, remaining, source, best_move);
    }

    /// TT Best Move を参照する(位置キー＋持ち駒指定)．
    #[inline]
    fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        self.table.look_up_best_move(pos_key, hand)
    }

    /// Killer Move を記録する．
    ///
    /// 同じ手が既にスロット 0 にあれば何もしない．
    /// そうでなければスロット 1 ← スロット 0，スロット 0 ← 新手の順でシフトする．
    #[inline]
    fn record_killer(&mut self, ply: u32, move16: u16) {
        if move16 == 0 {
            return;
        }
        let p = ply as usize;
        if p >= self.killer_table.len() {
            self.killer_table.resize(p + 1, [0u16; 2]);
        }
        if self.killer_table[p][0] == move16 {
            return;
        }
        self.killer_table[p][1] = self.killer_table[p][0];
        self.killer_table[p][0] = move16;
    }

    /// 指定 ply の Killer Move を取得する．
    #[inline]
    fn get_killers(&self, ply: u32) -> [u16; 2] {
        let p = ply as usize;
        if p < self.killer_table.len() {
            self.killer_table[p]
        } else {
            [0u16; 2]
        }
    }

    /// 経路依存フラグ付きで転置表を更新する(GHI 対策)．
    #[inline]
    fn store_path_dep(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        path_dependent: bool,
    ) {
        self.table.store_path_dep(pos_key, hand, pn, dn, remaining, source, path_dependent);
    }

    /// 転置表を参照する(盤面から自動計算，事後クエリ用)．
    ///
    /// `remaining = 0` で全エントリを受け入れる．
    /// PV 抽出や結果確認など，探索外での参照に使用する．
    #[inline]
    fn look_up_board(&self, board: &Board) -> (u32, u32) {
        let pk = position_key(board);
        let hand = &board.hand[self.attacker.index()];
        // self.depth を remaining として使用し，浅い IDS 反復の
        // 仮反証(NM)を最終結果に採用しないようにする．
        // remaining=0 だと全ての NM エントリを受け入れてしまい，
        // PNS の深さ制限内での仮反証が最終判定を汚染する．
        let (pn, dn, _source) = self.table.look_up(pk, hand, self.depth as u16);
        (pn, dn)
    }

    /// 転置表を更新する(盤面から自動計算)．
    #[inline]
    fn store_board(
        &mut self,
        board: &Board,
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
    ) {
        let pk = position_key(board);
        let hand = board.hand[self.attacker.index()];
        self.table.store(pk, hand, pn, dn, remaining, source);
    }

    /// 証明駒/反証駒を指定して TT に格納する．
    fn store_board_with_hand(
        &mut self,
        board: &Board,
        hand: &[u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
    ) {
        let pk = position_key(board);
        self.table.store(pk, *hand, pn, dn, remaining, source);
    }

    /// 詰将棋を解く(Best-First PNS + MID フォールバック)．
    ///
    /// `board` は攻め方の手番から開始する局面．
    /// 片玉局面(攻め方に玉がない)を想定するが，両玉でも動作する．
    ///
    /// Phase 1: Best-First PNS で探索木をメモリ上に構築し，
    ///          グローバルに最適なノード選択を行う．
    /// Phase 2: PNS がアリーナ上限に達した場合，残りの予算で
    ///          IDS-dfpn (MID) にフォールバックする．
    pub fn solve(&mut self, board: &mut Board) -> TsumeResult {
        self.table.clear();
        self.nodes_searched = 0;
        self.max_ply = 0;
        self.ply_nodes = [0; 64];
        self.path.clear();
        self.killer_table.clear();
        self.refutable_check_failed.clear();
        self.start_time = Instant::now();
        self.timed_out = false;
        self.next_gc_check = 100_000;
        self.attacker = board.turn;
        #[cfg(feature = "profile")]
        {
            self.profile_stats = ProfileStats::default();
            self.table.reset_profile();
        }

        // Phase 1: Best-First PNS
        // PNS は浅い詰将棋を解くのが主目的．全体の 1/4 を割り当てるが，
        // 150K ノードを上限とする．PNS はアリーナ飽和後の反復で効率が
        // 急落するため予算を抑え，残りを MID に回す．
        let saved_max_nodes = self.max_nodes;
        const PNS_BUDGET_CAP: u64 = 150_000;
        self.max_nodes = (saved_max_nodes / 4).min(PNS_BUDGET_CAP);
        #[cfg(feature = "profile")]
        let _pns_start = Instant::now();
        let pns_pv = self.pns_main(board);
        #[cfg(feature = "profile")]
        {
            self.profile_stats.pns_total_ns += _pns_start.elapsed().as_nanos() as u64;
        }
        self.max_nodes = saved_max_nodes;

        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        let (root_pn_after_pns, root_dn_after_pns, _) =
            self.look_up_pn_dn(pk, &att_hand, self.depth as u16);

        // PNS で未解決 + 残り予算あり → MID フォールバック
        if root_pn_after_pns != 0 && root_dn_after_pns != 0
            && self.nodes_searched < self.max_nodes
            && !self.timed_out
        {
            // PNS で蓄積した TT エントリを活用して IDS-dfpn を実行
            eprintln!("[solve] MID fallback start: nodes={}", self.nodes_searched);
            self.mid_fallback(board);
            eprintln!("[solve] MID fallback end: nodes={} time={:.1}s",
                self.nodes_searched, self.start_time.elapsed().as_secs_f64());
        }

        let (root_pn, root_dn) = self.look_up_board(board);
        eprintln!("[solve] root_pn={} root_dn={} nodes={}", root_pn, root_dn, self.nodes_searched);

        if root_pn == 0 {
            // PNS アリーナから PV を抽出できた場合はそちらを優先
            // (TT ベースの extract_pv は PNS 証明パスが不完全になりうる)
            if let Some(pv) = pns_pv {
                if self.find_shortest {
                    // 最短手数探索: PV 長を depth 上限にして追加証明
                    let saved_depth = self.depth;
                    self.depth = pv.len() as u32;
                    self.complete_or_proofs(board);
                    self.depth = saved_depth;
                    let final_moves = self.extract_pv_limited(board, 100_000);
                    let moves = if !final_moves.is_empty()
                        && final_moves.len() <= pv.len()
                    {
                        final_moves
                    } else {
                        pv
                    };
                    return TsumeResult::Checkmate {
                        moves,
                        nodes_searched: self.nodes_searched,
                    };
                }
                return TsumeResult::Checkmate {
                    moves: pv,
                    nodes_searched: self.nodes_searched,
                };
            }

            // アリーナ PV が取れなかった場合は TT ベースにフォールバック
            self.complete_or_proofs(board);

            let moves = self.extract_pv_limited(board, 100_000);
            if moves.is_empty() {
                return TsumeResult::CheckmateNoPv {
                    nodes_searched: self.nodes_searched,
                };
            }
            if self.find_shortest {
                let saved_depth = self.depth;
                self.depth = moves.len() as u32;
                self.complete_or_proofs(board);
                self.depth = saved_depth;
                let final_moves = self.extract_pv_limited(board, 100_000);
                let moves = if !final_moves.is_empty()
                    && final_moves.len() <= moves.len()
                {
                    final_moves
                } else {
                    moves
                };
                TsumeResult::Checkmate {
                    moves,
                    nodes_searched: self.nodes_searched,
                }
            } else {
                TsumeResult::Checkmate {
                    moves,
                    nodes_searched: self.nodes_searched,
                }
            }
        } else if root_dn == 0 {
            TsumeResult::NoCheckmate {
                nodes_searched: self.nodes_searched,
            }
        } else {
            TsumeResult::Unknown {
                nodes_searched: self.nodes_searched,
            }
        }
    }

    /// 深さ制限 OR ノードの再帰的 NM 判定．
    ///
    /// 全王手に対して玉方に不詰を示す応手が存在するかを再帰的に確認する．
    /// 各王手に対し，玉方に「応手後に王手なし」または「応手後の王手が
    /// さらに反証可能」となる逃げ手が存在すれば真の不詰(REMAINING_INFINITE)
    /// として扱える．`max_depth` で再帰の深さを制限し，分岐爆発を防止する．
    /// IDS の NM 判定で使用する構造的不詰検証．
    ///
    /// 全王手に対して「合法な応手を経由して王手が尽きる」ことを再帰的に
    /// 確認する．呼び出し回数上限(`REFUTABLE_CALL_LIMIT`)を超えた場合は
    /// 安全側に倒して false(未証明)を返す．
    fn depth_limit_all_checks_refutable(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        let mut calls: u32 = 0;
        self.all_checks_refutable_recursive(
            board, checks, 5, &mut calls, Self::REFUTABLE_CALL_LIMIT,
        )
    }

    /// MID 内部用の版．完全版と同等の深さ 5・10K 回を使用．
    ///
    /// キャッシュ(`refutable_check_failed`)と組み合わせて使用するため，
    /// 各ユニーク局面は1回のみ呼び出される．キャッシュなし時は
    /// 1回 ~9ms × 数千回で全体の 97% を占めていたが，
    /// キャッシュにより全体のオーバーヘッドはユニーク局面数 × 9ms に削減される．
    fn depth_limit_all_checks_refutable_fast(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        let mut calls: u32 = 0;
        self.all_checks_refutable_recursive(
            board, checks, 5, &mut calls, Self::REFUTABLE_CALL_LIMIT,
        )
    }

    /// TT ベースの NM 昇格判定(MID 内部用)．
    ///
    /// 各王手後の AND ノードが TT 上で REMAINING_INFINITE の
    /// 不詰として記録されているかを確認する．
    /// do_move + TT ルックアップのみで判定するため極めて高速
    /// (~2μs/王手)．TT にエントリがない場合は保守的に false を返す．
    fn all_checks_refutable_by_tt(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        for check in checks {
            let captured = board.do_move(*check);
            let pk = position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            // AND ノードが TT で REMAINING_INFINITE の不詰か確認
            let (_, dn, _) = self.table.look_up(pk, &att_hand, REMAINING_INFINITE);
            board.undo_move(*check, captured);
            if dn != 0 {
                // この王手後の局面が TT で不詰確定していない → 昇格不可
                return false;
            }
        }
        true
    }

    /// 呼び出し回数上限．組合せ爆発を防止する．
    /// 各呼び出しで generate_defense_moves + generate_check_moves を実行するため，
    /// デバッグビルドでの実行時間を考慮して小さめに設定する．
    const REFUTABLE_CALL_LIMIT: u32 = 10_000;

    /// MID 内部用の軽量版の呼び出し回数上限．
    /// キャッシュにより各局面は1回のみ呼び出されるため，
    /// 200 回で分岐爆発を防止しつつ十分な深さを確保する．
    const REFUTABLE_CALL_LIMIT_FAST: u32 = 200;

    /// `depth_limit_all_checks_refutable` の再帰本体．
    ///
    /// `depth` は残りの再帰深さ(0 で打ち切り)．各再帰レベルで
    /// 王手→応手→次の王手 を確認し，最大 `depth` 段階まで追跡する．
    /// `calls` は呼び出し回数カウンタで，`limit` 超過時は false を返す．
    fn all_checks_refutable_recursive(
        &mut self,
        board: &mut Board,
        checks: &[Move],
        depth: u32,
        calls: &mut u32,
        limit: u32,
    ) -> bool {
        for check in checks {
            *calls += 1;
            if *calls > limit {
                return false;
            }
            let captured = board.do_move(*check);
            let defenses = self.generate_defense_moves(board);
            if defenses.is_empty() {
                board.undo_move(*check, captured);
                return false;
            }
            let mut has_refuting_defense = false;
            for defense in &defenses {
                let cap_d = board.do_move(*defense);
                let next_checks = self.generate_check_moves(board);
                if next_checks.is_empty() {
                    board.undo_move(*defense, cap_d);
                    has_refuting_defense = true;
                    break;
                }
                // 再帰: 次の王手もすべて反証可能か確認
                if depth > 0
                    && self.all_checks_refutable_recursive(
                        board, &next_checks, depth - 1, calls, limit,
                    )
                {
                    board.undo_move(*defense, cap_d);
                    has_refuting_defense = true;
                    break;
                }
                board.undo_move(*defense, cap_d);
            }
            board.undo_move(*check, captured);
            if !has_refuting_defense {
                return false;
            }
        }
        true
    }

    /// Df-Pn 探索の中核関数(文献での MID: Multiple-Iterative-Deepening に相当)．
    ///
    /// 証明数(pn)・反証数(dn)の閾値を受け取り，いずれかが閾値に達するまで
    /// 最善子ノードを再帰的に展開する．OR ノード(攻め方)では王手を生成し，
    /// AND ノード(玉方)では王手回避手を生成する．
    ///
    /// TT のキーには盤面のみのハッシュ(持ち駒除外)を使用し，
    /// 持ち駒の優越関係により TT ヒット率を向上させる．
    fn mid(
        &mut self,
        board: &mut Board,
        pn_threshold: u32,
        dn_threshold: u32,
        ply: u32,
        or_node: bool,
    ) {
        // ノード制限・タイムアウトチェック
        if self.nodes_searched >= self.max_nodes {
            #[cfg(feature = "tt_diag")]
            { self.diag_node_limit_exits += 1; }
            return;
        }
        // 1024 ノードごとにタイマーをチェック
        if self.nodes_searched & 0x3FF == 0 && self.is_timed_out() {
            // elapsed >= timeout で初検出時にフラグをキャッシュし，
            // 以降の is_timed_out() でシステムコールを省略する．
            self.timed_out = true;
            return;
        }
        self.nodes_searched += 1;
        if (ply as usize) < 64 {
            self.ply_nodes[ply as usize] += 1;
        }
        // Periodic GC: TT サイズに応じて浅い remaining のエントリを除去
        // 1M ノード毎にチェック(GC はフルスキャンなので頻繁には実行しない)
        if self.nodes_searched % 1_000_000 == 0 {
            let tt_size = self.table.len();
            let gc_threshold: Option<u16> = if tt_size > 60_000_000 {
                Some(1) // remaining ≤ 1 を除去
            } else if tt_size > 50_000_000 {
                Some(0) // remaining = 0 のみ除去
            } else {
                None
            };
            if let Some(threshold) = gc_threshold {
                let removed = self.table.gc_shallow_entries(threshold);
                if removed > 0 {
                    eprintln!("[periodic_gc] threshold={} removed={} tt_positions={}",
                        threshold, removed, self.table.len());
                }
            }
        }
        // Periodic progress: every 1M nodes
        if self.nodes_searched % 1_000_000 == 0 && self.nodes_searched > 0 {
            // Ply distribution: show top consumers
            let mut ply_dist: Vec<(usize, u64)> = self.ply_nodes.iter().enumerate()
                .filter(|(_, &n)| n > 0).map(|(p, &n)| (p, n)).collect();
            ply_dist.sort_by(|a, b| b.1.cmp(&a.1));
            let top5: Vec<String> = ply_dist.iter().take(8)
                .map(|(p, n)| format!("p{}={}K", p, n / 1000)).collect();
            let (r_pn, r_dn, _) = self.look_up_pn_dn(
                self.diag_root_pk, &self.diag_root_hand, self.depth as u16);
            eprintln!("[progress] nodes={}M ply={} or={} time={:.1}s max_ply={} depth={} rpn={} rdn={} tt={} dist=[{}]",
                self.nodes_searched / 1_000_000, ply, or_node,
                self.start_time.elapsed().as_secs_f64(), self.max_ply, self.depth,
                r_pn, r_dn, self.table.len(), top5.join(", "));
            // TT エントリ増加診断(5M ノードごと)
            if self.nodes_searched % 5_000_000 == 0 {
                let t = &self.table;
                let total_ent: usize = t.tt.values().map(|v| v.len()).sum();
                eprintln!("[tt_diag] positions={} entries={} proof={} disproof={} inter_new={} inter_upd={} dominated={}",
                    t.tt.len(), total_ent,
                    t.diag_proof_inserts, t.diag_disproof_inserts,
                    t.diag_intermediate_new, t.diag_intermediate_update,
                    t.diag_dominated_skip);
                // remaining 値分布
                let rem: Vec<String> = t.diag_remaining_dist.iter().enumerate()
                    .filter(|(_, &c)| c > 0)
                    .map(|(r, &c)| {
                        if r == 32 { format!("INF:{}", c) }
                        else { format!("{}:{}", r, c) }
                    }).collect();
                eprintln!("[tt_diag] remaining_dist=[{}]", rem.join(", "));
                // エントリ数別局面分布
                let mut size_dist = [0u64; 17]; // size_dist[n] = # positions with n entries
                for v in t.tt.values() {
                    let n = v.len().min(16);
                    size_dist[n] += 1;
                }
                let sd: Vec<String> = size_dist.iter().enumerate()
                    .filter(|(_, &c)| c > 0)
                    .map(|(n, &c)| format!("{}ent:{}pos", n, c))
                    .collect();
                eprintln!("[tt_diag] entries_per_pos=[{}]", sd.join(", "));
                // 10M ノードごとにコンテンツ分析
                if self.nodes_searched % 10_000_000 == 0 {
                    self.table.dump_content_analysis();
                }
            }
        }
        if ply > self.max_ply {
            self.max_ply = ply;
        }
        #[cfg(feature = "tt_diag")]
        if (ply as usize) < 64 {
            self.diag_ply_visits[ply as usize] += 1;
        }

        let full_hash = board.hash;
        let pos_key = profile_timed!(self, position_key_ns, position_key_count,
            position_key(board));
        let att_hand = board.hand[self.attacker.index()];

        // ループ検出: フルハッシュで判定(持ち駒込みの完全一致)
        let in_path = profile_timed!(self, loop_detect_ns, loop_detect_count,
            self.path.contains(&full_hash));
        if in_path {
            #[cfg(feature = "tt_diag")]
            { self.diag_in_path_exits += 1; }
            if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
                eprintln!("[exit_diag] ply={} in_path_exit: hash={:#x} or={}", ply, full_hash, or_node);
            }
            return;
        }

        // 残り探索深さ
        let remaining = self.depth.saturating_sub(ply) as u16;

        // TT 参照: 既に閾値を超えている/証明済み/反証済みなら
        // 手生成をスキップして早期 return
        let (tt_pn, tt_dn, _) = profile_timed!(self, tt_lookup_ns, tt_lookup_count,
            self.look_up_pn_dn(pos_key, &att_hand, remaining));
        if tt_pn == 0 || tt_dn == 0 {
            #[cfg(feature = "tt_diag")]
            {
                self.diag_terminal_exits += 1;
                if ply == self.diag_ply && self.diag_terminal_exits <= 3 {
                    eprintln!("[tt_diag] ply={} terminal exit: tt_pn={} tt_dn={} remaining={}",
                        ply, tt_pn, tt_dn, remaining);
                }
            }
            // Diagnostic: catch early exits at children of stuck node
            if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
                eprintln!("[exit_diag] ply={} terminal_exit: tt_pn={} tt_dn={} rem={} or={} pk={:#x}",
                    ply, tt_pn, tt_dn, remaining, or_node, pos_key);
            }
            return;
        }
        if tt_pn >= pn_threshold || tt_dn >= dn_threshold {
            #[cfg(feature = "tt_diag")]
            {
                self.diag_threshold_exits += 1;
                if ply == self.diag_ply && self.diag_threshold_exits <= 3 {
                    eprintln!("[tt_diag] ply={} threshold exit: tt_pn={} tt_dn={} pn_th={} dn_th={}",
                        ply, tt_pn, tt_dn, pn_threshold, dn_threshold);
                }
            }
            // Diagnostic: catch threshold exits at children of stuck node
            if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
                eprintln!("[exit_diag] ply={} threshold_exit: tt_pn={} tt_dn={} pn_th={} dn_th={} rem={} or={} pk={:#x}",
                    ply, tt_pn, tt_dn, pn_threshold, dn_threshold, remaining, or_node, pos_key);
            }
            return;
        }
        // TT 診断: ply 35 で terminal exit しなかった場合の TT 状態を出力
        #[cfg(feature = "tt_diag")]
        {
            let visit_count = self.diag_ply_visits[ply as usize];
            if ply == self.diag_ply && (visit_count <= 5 || (visit_count % 1000000 == 0)) {
                let entry_count = self.table.entries_for_position(pos_key);
                eprintln!(
                    "[tt_diag] ply={} non-terminal entry #{}: pos_key={:#x} tt_pn={} tt_dn={} \
                     remaining={} hand={:?} tt_entries_at_key={}",
                    ply, visit_count, pos_key, tt_pn, tt_dn,
                    remaining, &att_hand, entry_count);
                if visit_count <= 3 || visit_count == 1000000 {
                    self.table.dump_entries(pos_key);
                }
            }
        }

        // 終端条件: 深さ制限・手数制限
        if ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            #[cfg(feature = "profile")]
            let _depth_limit_start = Instant::now();
            if or_node {
                // OR ノードの深さ制限: 王手が0手なら真の不詰(REMAINING_INFINITE)．
                // 王手が0手の不詰は深さに依存しないため，IDS 間で再利用可能にする．
                // TT ベースの高速判定のみ使用．
                let checks = self.generate_check_moves(board);
                if checks.is_empty() {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else if self.all_checks_refutable_by_tt(board, &checks) {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else {
                    self.store(pos_key, att_hand, INF, 0, 0, pos_key);
                }
            } else {
                // AND ノードの深さ制限: 深さ制限付き NM(remaining=0)として記録．
                self.store(pos_key, att_hand, INF, 0, 0, pos_key);
            }
            #[cfg(feature = "profile")]
            {
                self.profile_stats.depth_limit_terminal_ns += _depth_limit_start.elapsed().as_nanos() as u64;
                self.profile_stats.depth_limit_terminal_count += 1;
            }
            return;
        }

        // 合法手生成
        let mut moves = if or_node {
            profile_timed!(self, movegen_check_ns, movegen_check_count,
                self.generate_check_moves(board))
        } else {
            profile_timed!(self, movegen_defense_ns, movegen_defense_count,
                self.generate_defense_moves(board))
        };

        // Dynamic Move Ordering: TT Best Move + Killer Moves
        // 前回の探索で最善だった手を優先的に展開し，カットオフを早める．
        // NOTE: OR ノードでのみ適用．AND ノードでは全子の証明が必要なため
        // 手順序の影響は OR より小さく，ソートの安定性を優先する．
        if or_node {
            let mut next_slot = 0usize; // 次に挿入する位置

            // 1. TT Best Move を先頭に移動
            let tt_best = self.look_up_best_move(pos_key, &att_hand);
            if tt_best != 0 {
                if let Some(idx) = moves.iter().position(|m| m.to_move16() == tt_best) {
                    if idx > next_slot {
                        moves.swap(next_slot, idx);
                    }
                    next_slot += 1;
                }
            }

            // 2. Killer Moves を TT Best Move の直後に配置
            let killers = self.get_killers(ply);
            for &km16 in &killers {
                if km16 != 0 && km16 != tt_best {
                    if let Some(idx) = moves[next_slot..].iter()
                        .position(|m| m.to_move16() == km16)
                    {
                        let actual_idx = next_slot + idx;
                        if actual_idx > next_slot {
                            moves.swap(next_slot, actual_idx);
                        }
                        next_slot += 1;
                    }
                }
            }
        }

        // 終端条件チェック
        if moves.is_empty() {
            if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
                eprintln!("[exit_diag] ply={} empty_moves: or={}", ply, or_node);
            }
            if or_node {
                // 王手手段なし → 不詰(反証駒 = 現在の持ち駒)
                // 持ち駒が増えれば打ち駒による新たな王手が生じうるため，
                // PieceType::MAX_HAND_COUNT ではなく実際の持ち駒を使用する．
                // 真の終端条件なので REMAINING_INFINITE を使用する．
                self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key);
            } else {
                // 応手なし → 詰み(証明駒 = 空)
                self.store(
                    pos_key,
                    [0; HAND_KINDS],
                    0,
                    INF,
                    REMAINING_INFINITE,
                    pos_key,
                );
            }
            return;
        }

        // 子ノード情報を事前計算:
        // (Move, full_hash, pos_key, attacker_hand)
        let mut children: ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        > = ArrayVec::new();
        // (合駒は children にそのまま追加し，DN バイアスで後回し探索)
        // (OR ノードの反証は att_hand で保存するため反証駒蓄積は不要)
        // GHI 伝播: init ループ中に反証済み子の path_dependent を蓄積
        let mut init_or_path_dep = false;
        // init フェーズでの OR 子 NM remaining の最小値
        let mut init_or_nm_min_remaining: u16 = REMAINING_INFINITE;
        // AND ノードの init フェーズ用: TT プレフィルタで証明済み合駒の証明駒蓄積
        let mut init_and_proof = [0u8; HAND_KINDS];
        // チェーン合駒コンテキストでの遅延 AND 反証
        let mut init_and_disproof_found = false;
        let mut init_and_disproof_remaining: u16 = 0;
        let mut init_and_disproof_path_dep = false;
        let mut init_prefiltered_count: u32 = 0;
        // DFPN-E: OR ノードのエッジコスト計算用に守備側玉の位置を取得
        let defender_king_sq = if or_node {
            board.king_square(board.turn.opponent())
        } else {
            None
        };
        // chain_bb_cache を退避: 子ノードの初期化(generate_defense_moves 等)が
        // chain_bb_cache を上書きするため，この AND ノードの値を保存する．
        let saved_chain_bb = self.chain_bb_cache;
        #[cfg(feature = "profile")]
        let _child_init_start = Instant::now();
        let _init_start = Instant::now();
        for m in &moves {
            #[cfg(feature = "profile")]
            let _domove_start = Instant::now();
            let captured = board.do_move(*m);
            let child_full_hash = board.hash;
            let child_pk = position_key(board);
            let child_hand = board.hand[self.attacker.index()];
            #[cfg(feature = "profile")]
            {
                self.profile_stats.child_init_domove_ns += _domove_start.elapsed().as_nanos() as u64;
                self.profile_stats.child_init_domove_count += 1;
            }

            let child_remaining = remaining.saturating_sub(1);

            // 深さ制限ファストパス: 子ノードが ply+1 >= depth で即座に
            // 深さ制限に到達する場合，mid() 呼び出しを省略して直接反証を記録する．
            // これにより深さ制限 ply での nodes_searched++ が不要になり，
            // 深い問題でノード削減効果がある．
            // TT に証明が既にある場合はスキップ(PNS 等で蓄積済み)．
            if ply + 1 >= self.depth {
                let (dl_cpn, _, _) =
                    self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
                if dl_cpn != 0 {
                    // 未証明: 深さ制限反証を直接記録
                    if !or_node {
                        // AND 親の子 = OR 局面: 王手の有無で REMAINING_INFINITE 判定
                        let checks = self.generate_check_moves(board);
                        let dl_rem = if checks.is_empty() {
                            REMAINING_INFINITE
                        } else if self.all_checks_refutable_by_tt(board, &checks) {
                            REMAINING_INFINITE
                        } else {
                            0
                        };
                        self.store(child_pk, child_hand, INF, 0, dl_rem, child_pk);
                    } else {
                        // OR 親の子 = AND 局面: 深さ制限反証(remaining=0)
                        self.store(child_pk, child_hand, INF, 0, 0, child_pk);
                    }
                }
                // dl_cpn == 0: 証明済み → 通常フローへ fall through
            }

            let (cpn, cdn, _csrc) =
                self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
            if cpn == 1 && cdn == 1 {
                #[cfg(feature = "profile")]
                let _static_mate_start = Instant::now();
                #[cfg(feature = "profile")]
                let mut _sm_hit = false;

                if or_node {
                    // OR ノードの子(AND 局面): 静的詰め判定
                    // remaining が budget に対して大きすぎる場合は呼び出しを
                    // スキップする(Exhausted になるだけで NPS を浪費するため)．
                    //
                    // 閾値 budget * 2 + 1 の根拠: static_mate は TT ヒット時に
                    // budget を消費しないため，DFPN 本体が蓄積した TT エントリを
                    // 活用すれば budget の約2倍の深さまで到達しうる．
                    // +1 は奇数手詰め(攻方始動)との端数調整．
                    if self.mate_budget > 0 && remaining >= 3
                        && u32::from(child_remaining) <= self.mate_budget.saturating_mul(2).saturating_add(1)
                    {
                        // 予算制静的詰め探索(1手〜N手を統一的に扱う)
                        let mut budget = self.mate_budget;
                        match self.static_mate_and(
                            board, child_remaining as u32, &mut budget,
                        ) {
                            StaticMateResult::Checkmate(_) => {
                                #[cfg(feature = "profile")]
                                { _sm_hit = true; }
                                // TT は static_mate_and 内で記録済み
                            }
                            StaticMateResult::NoCheckmate => {
                                // 確定的に不詰: 応手数で初期 pn/dn を設定
                                // (static_mate_and 内で TT に不詰記録済みだが，
                                //  応手数に基づく初期値も記録する)
                                // 注: static_mate_and 内でも generate_defense_moves
                                // を実行済みだが，初期 pn/dn 設定のため再計算が必要．
                                let defenses = self.generate_defense_moves(board);
                                let n = defenses.len() as u32;
                                if n == 0 {
                                    self.store(child_pk, [0; HAND_KINDS], 0, INF,
                                        REMAINING_INFINITE, child_pk);
                                    #[cfg(feature = "profile")]
                                    { _sm_hit = true; }
                                } else {
                                    let n = defenses.len() as u32;
                                    let mut pn = self.heuristic_and_pn(board, n);
                                    // DFPN-E: エッジコスト加算
                                    if let Some(ksq) = defender_king_sq {
                                        pn = pn.saturating_add(edge_cost_or(*m, ksq));
                                    }
                                    let dn = 1u32;
                                    self.store(child_pk, child_hand, pn, dn,
                                        child_remaining, child_pk);
                                }
                            }
                            StaticMateResult::Exhausted => {
                                // 予算切れ: 応手数で初期 pn/dn を設定．
                                // n=1 だと (1,1) になり再度 static_mate が
                                // トリガーされるため dn を最低2にする．
                                // 注: static_mate_and 内でも generate_defense_moves
                                // を実行済みだが，初期 pn/dn 設定のため再計算が必要．
                                let defenses = self.generate_defense_moves(board);
                                let n = defenses.len() as u32;
                                if n == 0 {
                                    self.store(child_pk, [0; HAND_KINDS], 0, INF,
                                        REMAINING_INFINITE, child_pk);
                                    #[cfg(feature = "profile")]
                                    { _sm_hit = true; }
                                } else {
                                    let n = defenses.len() as u32;
                                    let mut pn = self.heuristic_and_pn(board, n);
                                    // DFPN-E: エッジコスト加算
                                    if let Some(ksq) = defender_king_sq {
                                        pn = pn.saturating_add(edge_cost_or(*m, ksq));
                                    }
                                    let dn = 2u32;
                                    self.store(child_pk, child_hand, pn, dn,
                                        child_remaining, child_pk);
                                }
                            }
                        }
                    } else {
                        // budget=0: インライン1手・3手詰め判定
                        let defenses = self.generate_defense_moves(board);
                        if defenses.is_empty() {
                            // 応手なし → 即詰み確定(budget=0 パスでの検出)
                            self.store(child_pk, [0; HAND_KINDS], 0, INF,
                                REMAINING_INFINITE, child_pk);
                            #[cfg(feature = "profile")]
                            { _sm_hit = true; }
                        } else if ply + 2 < self.depth {
                            // 3手詰め: 全応手に1手詰め判定
                            let mut all_mated = true;
                            for d in &defenses {
                                let cap_d = board.do_move(*d);
                                let mate = if board.is_in_check(
                                    board.turn.opponent(),
                                ) {
                                    false
                                } else {
                                    let checks = self.generate_check_moves(board);
                                    if !checks.is_empty() {
                                        self.has_mate_in_1_with(board, &checks)
                                    } else {
                                        false
                                    }
                                };
                                if mate {
                                    self.store_board(board, 0, INF,
                                        REMAINING_INFINITE, child_pk);
                                }
                                board.undo_move(*d, cap_d);
                                if !mate {
                                    all_mated = false;
                                    break;
                                }
                            }
                            if all_mated {
                                self.store(child_pk, child_hand, 0, INF,
                                    REMAINING_INFINITE, child_pk);
                                #[cfg(feature = "profile")]
                                { _sm_hit = true; }
                            } else {
                                let n = defenses.len() as u32;
                                let mut pn = self.heuristic_and_pn(board, n);
                                // DFPN-E: エッジコスト加算
                                if let Some(ksq) = defender_king_sq {
                                    pn = pn.saturating_add(edge_cost_or(*m, ksq));
                                }
                                let dn = 1u32;
                                self.store(child_pk, child_hand, pn, dn,
                                    child_remaining, child_pk);
                            }
                        } else {
                            // depth 制限超過: 応手生成なし → deep df-pn のみ適用
                            let mut pn = 1u32;
                            // DFPN-E: エッジコスト加算
                            if let Some(ksq) = defender_king_sq {
                                pn = pn.saturating_add(edge_cost_or(*m, ksq));
                            }
                            let dn = 1u32;
                            self.store(child_pk, child_hand, pn, dn,
                                child_remaining, child_pk);
                        }
                    }
                } else {
                    // AND ノードの子(OR 局面): 静的詰め判定
                    // 閾値の根拠は OR 側と同一(budget * 2 + 1)．
                    if self.mate_budget > 0 && remaining >= 3
                        && u32::from(child_remaining) <= self.mate_budget.saturating_mul(2).saturating_add(1)
                    {
                        let mut budget = self.mate_budget;
                        match self.static_mate_or(
                            board, child_remaining as u32, &mut budget,
                        ) {
                            StaticMateResult::Checkmate(_) => {
                                #[cfg(feature = "profile")]
                                { _sm_hit = true; }
                                // TT は static_mate_or 内で記録済み
                            }
                            StaticMateResult::NoCheckmate => {
                                // 確定的に不詰: 王手数で初期化 + 取り後TT参照
                                let checks = self.generate_check_moves(board);
                                if checks.is_empty() {
                                    self.store(child_pk, child_hand, INF, 0,
                                        REMAINING_INFINITE, child_pk);
                                } else if self.try_capture_tt_proof(
                                    board, &checks, child_remaining)
                                {
                                    #[cfg(feature = "profile")]
                                    { _sm_hit = true; }
                                } else {
                                    let nc = checks.len() as u32;
                                    let pn = self.heuristic_or_pn(board, nc)
                                        .saturating_add(edge_cost_and(*m));
                                    let dn = 1u32;
                                    self.store(child_pk, child_hand, pn,
                                        dn, child_remaining, child_pk);
                                }
                            }
                            StaticMateResult::Exhausted => {
                                // 予算切れ: 王手数で初期化 + 取り後TT参照
                                let checks = self.generate_check_moves(board);
                                if checks.is_empty() {
                                    self.store(child_pk, child_hand, INF, 0,
                                        REMAINING_INFINITE, child_pk);
                                } else if self.try_capture_tt_proof(
                                    board, &checks, child_remaining)
                                {
                                    #[cfg(feature = "profile")]
                                    { _sm_hit = true; }
                                } else {
                                    let nc = checks.len() as u32;
                                    let pn = self.heuristic_or_pn(board, nc)
                                        .saturating_add(edge_cost_and(*m));
                                    let dn = 2u32;
                                    self.store(child_pk, child_hand, pn,
                                        dn, child_remaining, child_pk);
                                }
                            }
                        }
                    } else {
                        // budget=0: インライン王手なし/1手詰め判定 + 取り後TT参照
                        let checks = self.generate_check_moves(board);
                        if checks.is_empty() {
                            self.store(child_pk, child_hand, INF, 0,
                                REMAINING_INFINITE, child_pk);
                        } else if ply + 2 < self.depth
                            && self.has_mate_in_1_with(board, &checks)
                        {
                            self.store(child_pk, child_hand, 0, INF,
                                REMAINING_INFINITE, child_pk);
                            #[cfg(feature = "profile")]
                            { _sm_hit = true; }
                        } else if ply + 2 < self.depth
                            && self.try_capture_tt_proof(
                                board, &checks, child_remaining)
                        {
                            // 取りの王手で既証明局面に到達 → 即座に証明
                            #[cfg(feature = "profile")]
                            { _sm_hit = true; }
                        } else {
                            let nc = checks.len() as u32;
                            let pn = self.heuristic_or_pn(board, nc)
                                .saturating_add(edge_cost_and(*m));
                            let dn = 1u32;
                            self.store(child_pk, child_hand, pn,
                                dn, child_remaining, child_pk);
                        }
                    }
                }

                #[cfg(feature = "profile")]
                {
                    let elapsed = _static_mate_start.elapsed().as_nanos() as u64;
                    self.profile_stats.static_mate_ns += elapsed;
                    self.profile_stats.static_mate_count += 1;
                    if _sm_hit {
                        self.profile_stats.static_mate_hits += 1;
                    }
                }
            }

            #[cfg(feature = "profile")]
            let _undomove_start = Instant::now();
            board.undo_move(*m, captured);
            #[cfg(feature = "profile")]
            {
                self.profile_stats.child_init_domove_ns += _undomove_start.elapsed().as_nanos() as u64;
                self.profile_stats.child_init_domove_count += 1;
            }

            // 即座に解決チェック(子ノード初期化時に証明/反証を検出)
            let (cpn_now, cdn_now, _) =
                self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
            if or_node && cpn_now == 0 {
                // OR 証明: 子の証明駒から親の証明駒を計算
                let child_ph = self
                    .table
                    .get_proof_hand(child_pk, &child_hand);
                let mut proof =
                    adjust_hand_for_move(*m, &child_ph);
                // 証明駒を現在の持ち駒で上限クリップ
                for k in 0..HAND_KINDS {
                    proof[k] = proof[k].min(att_hand[k]);
                }
                self.store(pos_key, proof, 0, INF,
                    REMAINING_INFINITE, pos_key);
                #[cfg(feature = "profile")]
                {
                    self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
                    self.profile_stats.child_init_count += 1;
                }
                return;
            }
            if !or_node && cdn_now == 0 {
                // AND 反証: 子の反証を検出．
                //
                // チェーン合駒のコンテキスト(chain_bb_cache が非空)では，
                // 即座に return せず init ループを継続する．これにより
                // 後続のチェーン合駒(ドロップ)に対して TT プレフィルタが
                // 実行され，証明エントリが TT に蓄積される．
                // 即座に return すると王逃げ/駒取りが先に処理されて
                // ドロップが到達不能になり，プレフィルタが一切発火しない．
                //
                // 反証情報は init_and_disproof_* に保存し，ループ終了後に
                // まとめて store + return する．
                #[cfg(feature = "tt_diag")]
                { self.diag_init_and_disproof_exits += 1; }
                if saved_chain_bb.is_not_empty() {
                    // チェーン合駒コンテキスト: 反証情報を記録して継続
                    if !init_and_disproof_found {
                        init_and_disproof_found = true;
                        // lookup と同じ条件でマッチする反証の情報を取得する
                        let (child_nm_rem, child_pd) = self.table
                            .get_effective_disproof_info(
                                child_pk, &child_hand, child_remaining,
                            )
                            .unwrap_or((0, false));
                        init_and_disproof_remaining = if child_pd {
                            remaining
                        } else {
                            propagate_nm_remaining(child_nm_rem, remaining)
                        };
                        init_and_disproof_path_dep = child_pd;
                        #[cfg(feature = "tt_diag")]
                        if ply == self.diag_ply && self.diag_in_path_exits < 10 {
                            eprintln!("[tt_diag] ply={} init AND disproof (deferred): move={} child_rem={} parent_rem={} remaining={} path_dep={} pos_key={:#x}",
                                ply, m.to_usi(), child_nm_rem, init_and_disproof_remaining, remaining,
                                init_and_disproof_path_dep, pos_key);
                            self.diag_in_path_exits += 1;
                        }
                    }
                    continue;
                }
                // 非チェーン: 従来の即座 return
                // lookup と同じ条件でマッチする反証の情報を取得する
                let (child_nm_rem, is_path_dep) = self.table
                    .get_effective_disproof_info(
                        child_pk, &child_hand, child_remaining,
                    )
                    .unwrap_or((0, false));
                if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
                    let parent_nm_rem_preview = if is_path_dep { remaining } else { propagate_nm_remaining(child_nm_rem, remaining) };
                    eprintln!("[exit_diag] ply={} init_and_disproof: child_move={} child_nm_rem={} parent_nm_rem={} remaining={} path_dep={} pk={:#x}",
                        ply, m.to_usi(), child_nm_rem, parent_nm_rem_preview, remaining, is_path_dep, pos_key);
                }
                // 経路依存の反証は同一 IDS 反復内では有効とみなし，
                // remaining を現在の深さに設定して lookup の remaining チェックを通過させる．
                // 非経路依存の反証は通常の NM 伝播で remaining を制限する．
                let parent_nm_remaining = if is_path_dep {
                    remaining
                } else {
                    propagate_nm_remaining(child_nm_rem, remaining)
                };
                #[cfg(feature = "tt_diag")]
                if ply == self.diag_ply && self.diag_in_path_exits < 10 {
                    eprintln!("[tt_diag] ply={} init AND disproof: move={} child_rem={} parent_rem={} remaining={} path_dep={} pos_key={:#x}",
                        ply, m.to_usi(), child_nm_rem, parent_nm_remaining, remaining, is_path_dep, pos_key);
                    self.diag_in_path_exits += 1;
                }
                if is_path_dep {
                    self.store_path_dep(
                        pos_key, att_hand, INF, 0,
                        parent_nm_remaining, pos_key, true,
                    );
                } else {
                    self.store(pos_key, att_hand, INF, 0,
                        parent_nm_remaining, pos_key);
                }
                #[cfg(feature = "tt_diag")]
                {
                    self.diag_init_and_disproof_exits += 1;
                    let visit_count = if (ply as usize) < 64 {
                        self.diag_ply_visits[ply as usize]
                    } else { 0 };
                    let (v_pn, v_dn, _) = self.look_up_pn_dn(pos_key, &att_hand, remaining);
                    if v_dn != 0 {
                        if self.diag_init_and_disproof_exits <= 5
                            || (ply == self.diag_ply && (visit_count % 1000000 == 0))
                        {
                            eprintln!(
                                "[tt_diag] WARNING: non-chain init AND disproof verification FAILED: \
                                 ply={} visit={} pos_key={:#x} hand={:?} stored dn=0 path_dep={} rem={} \
                                 but lookup(rem={}) returns pn={} dn={}",
                                ply, visit_count, pos_key, &att_hand, is_path_dep,
                                parent_nm_remaining, remaining, v_pn, v_dn
                            );
                            self.table.dump_entries(pos_key);
                        }
                    }
                }
                #[cfg(feature = "profile")]
                {
                    self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
                    self.profile_stats.child_init_count += 1;
                }
                return;
            }
            // cdn_now == 0 ブロックに入るのは or_node == true のみ．
            // AND ノードは cdn_now == 0 のとき上で return 済み，
            // AND かつ cdn_now != 0 のときはここを通過して children に追加される．
            if cdn_now == 0 {
                // OR: この子は反証済み(反証は att_hand で保存するため蓄積不要)
                // NM remaining 伝播: 子の remaining の最小値を追跡
                // get_effective_disproof_info を使用: look_up と同じ remaining
                // チェックを行い，正しいエントリの remaining を返す．
                // get_disproof_remaining は remaining を検査しないため，
                // 古いエントリ(低 remaining)を返して NM 伝播を汚染する．
                let child_nm_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                init_or_nm_min_remaining = init_or_nm_min_remaining.min(child_nm_rem);
                // GHI 伝播: 子の反証が経路依存なら蓄積
                init_or_path_dep |= self.table.has_path_dependent_disproof(
                    child_pk, &child_hand,
                );
                continue;
            }

            // AND ノードの合駒(drop)は TT プレフィルタで証明可能かチェック．
            // 証明済みなら children に追加せずスキップする．
            // 未証明の合駒は children にそのまま追加し，DN バイアスで
            // 後回しに探索される(旧 deferred_children 方式を廃止)．
            //
            // 重要: プレフィルタは初回訪問(cpn == 1 && cdn == 1)の子のみ実行．
            // 再訪問時に毎回実行すると generate_legal_moves が呼ばれるため，
            // 42M+ 回の無駄な movegen が発生して NPS が壊滅的に低下する．
            // IDS の浅い反復で TT に蓄積された証明を深い反復で活用するのは
            // 初回訪問時のプレフィルタで十分(§3.5)．
            if !or_node && m.is_drop() && cpn == 1 && cdn == 1 {
                #[cfg(feature = "profile")]
                let _pf_start = Instant::now();
                let _pf_hit = self.try_prefilter_block(
                    board, *m, &child_hand, remaining,
                    &mut init_and_proof,
                );
                #[cfg(feature = "profile")]
                {
                    self.profile_stats.prefilter_ns += _pf_start.elapsed().as_nanos() as u64;
                    self.profile_stats.prefilter_count += 1;
                }
                if _pf_hit {
                    init_prefiltered_count += 1;
                    self.prefilter_hits += 1;
                    continue;
                }
                #[cfg(feature = "tt_diag")]
                { self.diag_deferred_enqueued += 1; }
            }
            push_move(&mut children, (
                *m,
                child_full_hash,
                child_pk,
                child_hand,
            ));
        }

        // OR ノードで全子が反証済み(children が空)
        if or_node && children.is_empty() {
            // Diagnostic: detect OR all-children-disproved at stuck position
            if pos_key == 0xf8322787b7535d9c || (ply == 26 && self.nodes_searched % 1_000_000 < 5 && self.nodes_searched > 200_000) {
                eprintln!("[or_all_disproved] ply={} pk={:#x} moves={} init_or_nm_min_rem={} remaining={} path_dep={}",
                    ply, pos_key, moves.len(), init_or_nm_min_remaining, remaining, init_or_path_dep);
            }
            // NM remaining 伝播: 子の NM remaining の最小値 + 1 を使用．
            // 全子が REMAINING_INFINITE なら親も REMAINING_INFINITE(真の不詰)．
            let mut parent_nm_remaining = propagate_nm_remaining(
                init_or_nm_min_remaining, remaining);
            // MID 内部での REMAINING_INFINITE 昇格(init フェーズ):
            // TT ベースの高速判定のみ使用．depth_limit_all_checks_refutable は
            // 再帰的な movegen を伴い 1 回 ~6ms かかるため，MID の
            // ホットパスでは使用しない(NPS が 1.5K まで低下する)．
            // 完全な昇格判定は IDS 外部ループでのみ実行する．
            if parent_nm_remaining != REMAINING_INFINITE {
                let checks = self.generate_check_moves(board);
                if checks.is_empty() {
                    parent_nm_remaining = REMAINING_INFINITE;
                } else if self.all_checks_refutable_by_tt(board, &checks) {
                    parent_nm_remaining = REMAINING_INFINITE;
                }
            }
            //
            // GHI 伝播: いずれかの子の反証が経路依存なら親も経路依存
            // OR ノード反証: att_hand で保存(TT ヒット率最大化)
            // 実際の持ち駒で不詰が確定しているため，att_hand で登録すれば
            // hand dominance によるカバー範囲が最大になる．
            if init_or_path_dep {
                self.store_path_dep(
                    pos_key, att_hand, INF, 0,
                    parent_nm_remaining, pos_key, true,
                );
            } else {
                self.store(
                    pos_key, att_hand, INF, 0,
                    parent_nm_remaining, pos_key,
                );
            }
            // スラッシング防止: 反証の remaining が呼び出し元の remaining より低い場合，
            // look_up は remaining チェックで反証をスキップし，古い中間値(低 pn)を返す．
            // 親が低 pn の子を繰り返し選択し 1 ノードで帰還する無限ループの原因となる．
            // 呼び出し元の remaining で高 pn の中間エントリを追加保存することで，
            // look_up がこの高 pn 値を返し，親の閾値チェックが発火して他の子に切り替わる．
            // 将来この局面で真の進捗があれば中間値は自然に上書きされる．
            // (PNS フェーズでは mid() は呼ばれないため PNS の証明数に影響しない)
            if parent_nm_remaining < remaining {
                self.store(
                    pos_key, att_hand, INF - 1, 1,
                    remaining, pos_key,
                );
            }
            #[cfg(feature = "profile")]
            {
                self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
                self.profile_stats.child_init_count += 1;
            }
            return;
        }

        // チェーン合駒の遅延 AND 反証: init ループで合駒プレフィルタを
        // 先に実行した後，反証を確定して return する．
        // プレフィルタで蓄積された TT 証明エントリは次回以降の訪問で
        // 高速にチェーン合駒をスキップさせる(§3.5)．
        if init_and_disproof_found {
            if init_and_disproof_path_dep {
                self.store_path_dep(
                    pos_key, att_hand, INF, 0,
                    init_and_disproof_remaining, pos_key, true,
                );
            } else {
                self.store(pos_key, att_hand, INF, 0,
                    init_and_disproof_remaining, pos_key);
            }
            #[cfg(feature = "tt_diag")]
            {
                self.diag_init_and_disproof_exits += 1;
                let visit_count = if (ply as usize) < 64 {
                    self.diag_ply_visits[ply as usize]
                } else { 0 };
                let (v_pn, v_dn, _) = self.look_up_pn_dn(pos_key, &att_hand, remaining);
                if v_dn != 0 {
                    if self.diag_init_and_disproof_exits <= 5
                        || (ply == self.diag_ply && (visit_count % 1000000 == 0))
                    {
                        eprintln!(
                            "[tt_diag] WARNING: deferred init AND disproof verification FAILED: \
                             ply={} visit={} pos_key={:#x} hand={:?} stored dn=0 path_dep={} rem={} \
                             but lookup(rem={}) returns pn={} dn={}",
                            ply, visit_count, pos_key, &att_hand, init_and_disproof_path_dep,
                            init_and_disproof_remaining, remaining, v_pn, v_dn
                        );
                        self.table.dump_entries(pos_key);
                    }
                }
            }
            #[cfg(feature = "profile")]
            {
                self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
                self.profile_stats.child_init_count += 1;
            }
            return;
        }

        #[cfg(feature = "profile")]
        {
            self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
            self.profile_stats.child_init_count += 1;
        }

        // パスに追加(フルハッシュ)
        self.path.insert(full_hash);

        // --- チェーン合駒の DN バイアス用玉位置 ---
        // children 内のドロップ子がチェーンマスへのドロップなら，
        // DN バイアスに Chebyshev 距離を使い内側(玉に近い)から探索する．
        let chain_king_sq =
            if !or_node && saved_chain_bb.is_not_empty() {
                let chain_bb = saved_chain_bb;
                let has_drop = children.iter().any(|(m, _, _, _)| m.is_drop());
                if has_drop {
                    let all_chain = children.iter()
                        .filter(|(m, _, _, _)| m.is_drop())
                        .all(|(m, _, _, _)| chain_bb.contains(m.to_sq()));
                    if all_chain {
                        board.king_square(board.turn)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

        // プレフィルタで全合駒が証明済み(children が空になった場合)
        if !or_node && init_prefiltered_count > 0 && children.is_empty() {
            let mut p = init_and_proof;
            for k in 0..HAND_KINDS {
                p[k] = p[k].min(att_hand[k]);
            }
            self.store(pos_key, p, 0, INF, REMAINING_INFINITE, pos_key);
            self.path.remove(&full_hash);
            return;
        }

        // Init phase duration diagnostic
        {
            let init_elapsed = _init_start.elapsed().as_secs_f64();
            if init_elapsed > 1.0 {
                eprintln!("[init_slow] ply={} or={} moves={} children={} init_time={:.2}s",
                    ply, or_node, moves.len(), children.len(), init_elapsed);
            }
        }

        // --- 単一子最適化 ---
        // 子が1つしかない場合，MID ループ(閾値計算・全子走査)をバイパスし，
        // 親の閾値をそのまま渡して直接再帰する．
        if (ply == 26 || ply == 27) && self.nodes_searched > 200_000 && self.nodes_searched % 500_000 < 5 {
            eprintln!("[exit_diag] ply={} REACHED_MAIN_LOOP: children={} remaining={} or={}",
                ply, children.len(), remaining, or_node);
        }
        // OR ノードでは王手が1手のみ，AND ノードでは合法応手が1手のみの
        // ケースが詰将棋で頻出する．
        if children.len() == 1 {
            #[cfg(feature = "tt_diag")]
            if (ply as usize) < 64 {
                self.diag_ply_proofs[ply as usize] += 1; // reuse as single-child counter
            }
            let (m, child_fh, child_pk, ref child_hand) = children[0];
            let mut _sc_iter: u64 = 0;
            loop {
                _sc_iter += 1;
                if _sc_iter % 100_000 == 0 && self.start_time.elapsed().as_secs_f64() > 3.0 {
                    eprintln!("[sc_loop_hang] ply={} or={} iter={} nodes={} time={:.1}s move={}",
                        ply, or_node, _sc_iter, self.nodes_searched,
                        self.start_time.elapsed().as_secs_f64(), m.to_usi());
                }
                // ノード制限・タイムアウトチェック
                if self.nodes_searched >= self.max_nodes || self.timed_out {
                    break;
                }

                // ループ検出: 子がパス上にある場合は (INF, 0) として扱う
                let is_loop_child = self.path.contains(&child_fh);
                let (cpn, cdn, _csrc) = if is_loop_child {
                    (INF, 0, 0)
                } else {
                    self.look_up_pn_dn(
                        child_pk, child_hand,
                        remaining.saturating_sub(1),
                    )
                };
                if cpn >= pn_threshold || cdn >= dn_threshold {
                    self.store(pos_key, att_hand, cpn, cdn, remaining, pos_key);
                    break;
                }
                if cpn == 0 || cdn == 0 {
                    // 子の証明/反証 → 親に伝播
                    if or_node {
                        if cpn == 0 {
                            let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                            let mut proof = adjust_hand_for_move(m, &child_ph);
                            for k in 0..HAND_KINDS {
                                proof[k] = proof[k].min(att_hand[k]);
                            }
                            self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                        } else {
                            // cdn == 0: 唯一の子が反証 → OR 反証
                            // att_hand で保存(TT ヒット率最大化)
                            let child_path_dep = is_loop_child
                                || self.table.has_path_dependent_disproof(
                                    child_pk, child_hand,
                                );
                            if child_path_dep {
                                self.store_path_dep(
                                    pos_key, att_hand, INF, 0,
                                    remaining, pos_key, true,
                                );
                            } else {
                                self.store(pos_key, att_hand, INF, 0, remaining, pos_key);
                            }
                        }
                    } else {
                        if cdn == 0 {
                            // AND 反証: att_hand で保存(TT ヒット率最大化)
                            let child_path_dep = is_loop_child
                                || self.table.has_path_dependent_disproof(
                                    child_pk, child_hand,
                                );
                            if child_path_dep {
                                self.store_path_dep(
                                    pos_key, att_hand, INF, 0,
                                    remaining, if is_loop_child { 0 } else { pos_key }, true,
                                );
                            } else {
                                self.store(pos_key, att_hand, INF, 0, remaining, pos_key);
                            }
                        } else {
                            // cpn == 0: 唯一の子が証明 → AND 証明
                            let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                            let mut proof = [0u8; HAND_KINDS];
                            for k in 0..HAND_KINDS {
                                proof[k] = child_ph[k].min(att_hand[k]);
                            }
                            self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                        }
                    }
                    break;
                }

                let captured = profile_timed!(self, do_move_ns, do_move_count,
                    board.do_move(m));
                self.mid(board, pn_threshold, dn_threshold, ply + 1, !or_node);
                profile_timed!(self, undo_move_ns, undo_move_count,
                    board.undo_move(m, captured));
            }
            self.path.remove(&full_hash);
            #[cfg(feature = "tt_diag")]
            { self.diag_single_child_exits += 1; }
            return;
        }

        // SNDA 用の (source, value) ペアバッファ(ループ外で確保し再利用)
        let mut snda_pairs: Vec<(u64, u32)> = Vec::new();

        // TT 診断: このノードが監視対象 ply かどうか + 反復カウンタ
        #[cfg(feature = "tt_diag")]
        let _diag_this_node = self.diag_ply > 0 && ply == self.diag_ply;
        #[cfg(feature = "tt_diag")]
        let mut _diag_iteration: u32 = 0;
        #[cfg(feature = "tt_diag")]
        if _diag_this_node && self.diag_ply_visits[ply as usize] <= 2 {
            eprintln!(
                "[tt_diag] === ply={} {} node entered (visit #{}) === \
                 pos_key={:#x} children={} pn_th={} dn_th={} \
                 tt_pn={} tt_dn={} nodes={}",
                ply, if or_node { "OR" } else { "AND" },
                self.diag_ply_visits[ply as usize],
                pos_key, children.len(),
                pn_threshold, dn_threshold,
                tt_pn, tt_dn,
                self.nodes_searched,
            );
            // 各子ノードの初期 pn/dn を出力
            for (i, &(ref cm, _, cpk, ref ch)) in children.iter().enumerate() {
                let (cpn, cdn, _) = self.look_up_pn_dn(
                    cpk, ch, remaining.saturating_sub(1));
                eprintln!(
                    "[tt_diag]   child[{}] move={} drop={} pn={} dn={} pos_key={:#x}",
                    i, cm.to_usi(), cm.is_drop(), cpn, cdn, cpk,
                );
            }
        }

        // MID ループ(証明駒/反証駒の伝播を含む)
        let mut _loop_iter: u64 = 0;
        let _loop_start_nodes = self.nodes_searched;
        let mut _next_diag_nodes = self.nodes_searched.saturating_add(1_000_000);
        // 停滞検出: 子 mid() が消費するノード数が0(閾値で即座に返る)の
        // 連続回数を追跡．一定回数以上ゼロ進捗が続けば MID ループを脱出し，
        // 上位ノードに制御を戻す(dn_floor 由来の空転防止)．
        let mut zero_progress_count: u32 = 0;
        const ZERO_PROGRESS_LIMIT: u32 = 16;
        // 停滞検出: best child の pn/dn と閾値が変化しなければ，
        // 同じ子に同じ予算で mid() を呼んでも結果は変わらない．
        // 連続 STAGNATION_LIMIT 回の無変化で MID ループを脱出する．
        let mut prev_best_idx: usize = usize::MAX;
        let mut prev_best_pn: u32 = 0;
        let mut prev_best_dn: u32 = 0;
        let mut prev_child_pn_th: u32 = 0;
        let mut prev_child_dn_th: u32 = 0;
        // 前回の子 mid() が消費したノード数(ペナルティ保護・停滞検出用)．
        let mut _prev_nodes_used: u64 = 0;
        let mut stagnation_count: u32 = 0;
        const STAGNATION_LIMIT: u32 = 4;
        loop {
            _loop_iter += 1;
            if (ply as usize) < 64 {
                self.ply_iters[ply as usize] += 1;
            }
            // ply=0 は 100K ごと，それ以外は 1M ごとに詳細診断
            if self.nodes_searched >= _next_diag_nodes {
                let consumed = self.nodes_searched - _loop_start_nodes;
                eprintln!("[mid_diag] ply={} or={} consumed={}K iter={} children={} time={:.1}s pn_th={} dn_th={}",
                    ply, or_node, consumed / 1000, _loop_iter, children.len(),
                    self.start_time.elapsed().as_secs_f64(), pn_threshold, dn_threshold);
                for (i, &(ref cm, _, cpk, ref ch)) in children.iter().enumerate() {
                    let child_rem = remaining.saturating_sub(1);
                    let (cpn, cdn, _) = self.look_up_pn_dn(cpk, ch, child_rem);
                    eprintln!("[mid_diag]   child[{}] move={} drop={} pn={} dn={} pk={:#x} {}",
                        i, cm.to_usi(), cm.is_drop(), cpn, cdn, cpk,
                        if cpn == 0 { "PROVED" } else if cdn == 0 { "DISPROVED" } else { "" });
                    // Dump TT entries for stuck children (first diagnostic only)
                    if consumed < 1_100_000 && cpn != 0 && cdn != 0 {
                        if let Some(entries) = self.table.tt.get(&cpk) {
                            for (ei, e) in entries.iter().enumerate() {
                                eprintln!("[tt_dump]     entry[{}]: pn={} dn={} rem={} path_dep={} hand={:?}",
                                    ei, e.pn, e.dn, e.remaining, e.path_dependent, &e.hand);
                            }
                        }
                    }
                }
                // current_pn/dn/best_idx は後で計算されるのでここでは出力しない
                _next_diag_nodes = self.nodes_searched.saturating_add(
                    if ply == 0 { 100_000 } else { 1_000_000 }
                );
            }
            #[cfg(feature = "tt_diag")]
            { self.diag_mid_loop_iters += 1; }
            #[cfg(feature = "profile")]
            let _collect_start = Instant::now();

            // 各子ノードの pn/dn を収集し，証明/反証を検出
            let mut current_pn: u32;
            let mut current_dn: u32;
            let mut best_idx: usize = 0;
            let mut second_best: u32;
            let mut best_pn_dn: (u32, u32) = (INF, 0);
            let mut proved_or_disproved = false;

            // SNDA 用: best child の source を追跡
            let mut best_source: u64 = 0;

            // TCA: OR ノードでのループ子ノード数
            let mut loop_child_count: u32 = 0;
            // OR NM remaining 伝播: 全子 NM の remaining の最小値を追跡
            let mut or_nm_min_remaining: u16 = REMAINING_INFINITE;

            if or_node {
                // OR ノード: min(pn), sum(dn)
                current_pn = INF;
                current_dn = 0;
                second_best = INF; // 2番目に小さい pn(選択用，予算枯渇除外込み)
                let mut select_best_pn: u32 = INF; // 選択用 best pn
                // NM remaining 伝播: init フェーズの値を引き継ぐ
                or_nm_min_remaining = init_or_nm_min_remaining;
                // SNDA: (source, dn) ペアを収集し，同一 source の子は
                // sum の代わりに max で集約して過大評価を補正する
                snda_pairs.clear();

                for (i, &(ref _m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    let (cpn, cdn, csrc) =
                        if self.path.contains(&child_fh) {
                            loop_child_count += 1;
                            (INF, 0, 0)
                        } else {
                            self.look_up_pn_dn(
                                child_pk, child_hand,
                                remaining.saturating_sub(1),
                            )
                        };

                    if cpn == 0 {
                        // 子が証明済み → OR ノード証明
                        // Killer Move 記録: 証明を達成した王手は強力なヒント
                        self.record_killer(ply, children[i].0.to_move16());
                        let child_ph = self
                            .table
                            .get_proof_hand(
                                child_pk, child_hand,
                            );
                        let mut proof = adjust_hand_for_move(
                            children[i].0,
                            &child_ph,
                        );
                        // 証明駒を現在の持ち駒で上限クリップ
                        for k in 0..HAND_KINDS {
                            proof[k] =
                                proof[k].min(att_hand[k]);
                        }
                        self.store(
                            pos_key, proof, 0, INF,
                            REMAINING_INFINITE, csrc,
                        );
                        proved_or_disproved = true;
                        break;
                    }

                    // 反証済みの子: 反証は att_hand で保存するため蓄積不要
                    if cdn == 0 {
                        // NM remaining 伝播: 子の remaining の最小値を追跡
                        let child_nm_rem = self.table.get_effective_disproof_info(
                            child_pk, child_hand,
                            remaining.saturating_sub(1),
                        ).map(|(r, _)| r).unwrap_or(0);
                        or_nm_min_remaining = or_nm_min_remaining.min(child_nm_rem);
                        // GHI 伝播: 子の反証が経路依存なら親も経路依存
                        if !self.path.contains(&child_fh)
                            && self.table.has_path_dependent_disproof(
                                child_pk, child_hand,
                            )
                        {
                            loop_child_count += 1; // path_dependent として扱う
                        }
                    }

                    // True min cpn tracking (for node's proof number).
                    if cpn < current_pn {
                        current_pn = cpn;
                    }
                    if cpn < select_best_pn
                        || (cpn == select_best_pn
                            && cdn < best_pn_dn.1)
                    {
                        second_best = select_best_pn;
                        select_best_pn = cpn;
                        best_idx = i;
                        best_pn_dn = (cpn, cdn);
                        best_source = csrc;
                    } else if cpn < second_best {
                        second_best = cpn;
                    }
                    // SNDA: sum 計算は後段で行う
                    current_dn = (current_dn as u64)
                        .saturating_add(cdn as u64)
                        .min(INF as u64)
                        as u32;
                    // SNDA ペア収集(source=0 は独立ノード → グルーピング対象外)
                    if csrc != 0 && cdn > 0 {
                        snda_pairs.push((csrc, cdn));
                    }
                }

                if proved_or_disproved {
                    self.path.remove(&full_hash);
                    return;
                }

                // 全子が反証済み(dn=0) → OR ノード反証
                if current_dn == 0 {
                    // NM remaining 伝播: 子の NM remaining の最小値 + 1 を使用．
                    let mut parent_nm_remaining = propagate_nm_remaining(
                        or_nm_min_remaining, remaining);
                    // MID 内部での REMAINING_INFINITE 昇格(main loop):
                    // TT ベースの高速判定のみ(init フェーズと同様)
                    if parent_nm_remaining != REMAINING_INFINITE {
                        let checks = self.generate_check_moves(board);
                        if checks.is_empty() {
                            parent_nm_remaining = REMAINING_INFINITE;
                        } else if self.all_checks_refutable_by_tt(board, &checks)
                        {
                            parent_nm_remaining = REMAINING_INFINITE;
                        }
                    }
                    //
                    // GHI 対策: ループ子または経路依存な子の反証が寄与した場合は
                    // 親の反証も経路依存．init フェーズで蓄積した init_or_path_dep
                    // も考慮する(init で反証済みの子が MID ループには残らないため)．
                    // OR ノード反証: att_hand で保存(TT ヒット率最大化)
                    if loop_child_count > 0 || init_or_path_dep {
                        self.store_path_dep(
                            pos_key, att_hand,
                            INF, 0,
                            parent_nm_remaining, pos_key, true,
                        );
                    } else {
                        self.store(
                            pos_key, att_hand,
                            INF, 0,
                            parent_nm_remaining, pos_key,
                        );
                    }
                    // スラッシング防止(main loop): init フェーズと同じ処理
                    if parent_nm_remaining < remaining {
                        self.store(
                            pos_key, att_hand, INF - 1, 1,
                            remaining, pos_key,
                        );
                    }
                    self.path.remove(&full_hash);
                    return;
                }

                // SNDA 補正: 同一 source の子は DAG 合流の可能性
                // 重複グループの最小値分を控除して過大評価を補正
                if snda_pairs.len() >= 2 {
                    current_dn = snda_dedup(&mut snda_pairs, current_dn);
                }

            } else {
                // AND ノード: WPN (Weak Proof Number), min(dn)
                // WPN (Ueda et al. 2008): sum(pn) の代わりに
                // max(pn) + (unproven_count - 1) を使用．
                // DAG 構造での二重計数問題を緩和する．
                current_pn = 0;
                current_dn = INF;
                second_best = INF; // 2番目に小さい dn(選択用，バイアス込み)
                let mut all_proved = true;
                let mut and_proof =
                    [0u8; HAND_KINDS]; // 証明駒の和集合(max)
                // 合駒後回し最適化: 王移動・駒取りなどの非合駒応手を
                // 先に展開し，証明エントリを転置表に蓄積させる．
                // 合駒分岐は攻め方が取った後の局面が既に証明済みに
                // なっていることが多く，高速に証明できる．
                let mut best_effective_dn: u32 = INF;
                // SNDA: (source, pn) ペアを収集
                snda_pairs.clear();
                // WPN: max(cpn) と未証明子の数を追跡
                let mut max_cpn: u32 = 0;
                let mut unproven_count: u32 = 0;
                // CD-WPN: 同一マスのドロップを1グループとして数える
                let mut cd_grouped_count: u32 = 0;
                let mut drop_squares_seen: u128 = 0;

                for (i, &(ref m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    let is_loop_child = self.path.contains(&child_fh);
                    let (cpn, cdn, csrc) =
                        if is_loop_child {
                            (INF, 0, 0)
                        } else {
                            self.look_up_pn_dn(
                                child_pk, child_hand,
                                remaining.saturating_sub(1),
                            )
                        };

                    if cdn == 0 {
                        // 子が反証済み → AND ノード反証
                        // att_hand で保存(TT ヒット率最大化)
                        // AND ノードでは守備側が着手するため att_hand は不変．
                        //
                        let child_nm_rem = self.table.get_effective_disproof_info(
                            child_pk, child_hand,
                            remaining.saturating_sub(1),
                        ).map(|(r, _)| r).unwrap_or(0);
                        let parent_nm_remaining = propagate_nm_remaining(
                            child_nm_rem, remaining);
                        let child_path_dep = is_loop_child
                            || self.table.has_path_dependent_disproof(
                                child_pk, child_hand,
                            );
                        if child_path_dep {
                            self.store_path_dep(
                                pos_key, att_hand, INF, 0,
                                parent_nm_remaining, csrc, true,
                            );
                        } else {
                            self.store(
                                pos_key, att_hand, INF, 0,
                                parent_nm_remaining, csrc,
                            );
                        }
                        proved_or_disproved = true;
                        break;
                    }

                    if cpn == 0 {
                        // VPN (Virtual Proof Number, Saito et al. 2006):
                        // 証明済み子(cpn=0)は AND ノードの pn 合計・子選択から除外する．
                        // 証明駒のみ蓄積し，残りの未証明子に探索リソースを集中させる．
                        // AND ノードでは全子が詰む必要があるため，
                        // 証明駒は各子の証明駒の要素ごと最大値となる．
                        let child_ph = self
                            .table
                            .get_proof_hand(
                                child_pk, child_hand,
                            );
                        for k in 0..HAND_KINDS {
                            if child_ph[k] > and_proof[k]
                            {
                                and_proof[k] = child_ph[k];
                            }
                        }
                        // cross-deduction は all_proved パスで実行される．
                        // VPN: 証明済み子は pn=0 で sum に影響しないため，
                        // child 選択ループもスキップして効率化する．
                        continue;
                    }

                    all_proved = false;

                    // WPN: max(cpn) を追跡し，未証明子をカウント
                    if cpn > max_cpn {
                        max_cpn = cpn;
                    }
                    unproven_count += 1;
                    // CD-WPN: 同一マスのドロップは1グループとして数える
                    if m.is_drop() {
                        let sq_bit = 1u128 << (m.to_sq().index() as u32);
                        if drop_squares_seen & sq_bit == 0 {
                            drop_squares_seen |= sq_bit;
                            cd_grouped_count += 1;
                        }
                    } else {
                        cd_grouped_count += 1;
                    }
                    // TT 保存用: 真の min(dn)
                    if cdn < current_dn {
                        current_dn = cdn;
                    }
                    // SNDA ペア収集(source=0 は独立ノード)
                    if csrc != 0 {
                        snda_pairs.push((csrc, cpn));
                    }
                    // 子ノード選択用: AND ノードの合駒/非合駒バイアス．
                    //
                    // チェーン AND (chain_king_sq あり):
                    //   ドロップ(合駒)を優先し，内側(玉に近い)から探索する．
                    //   cross-deduce が同一マスの兄弟ドロップを一括証明するため，
                    //   1つのドロップを先に証明することがチェーン全体の鍵となる．
                    //   非合駒(玉逃げ・駒取り)には大きなバイアスを加算して後回しにする．
                    //
                    // 非チェーン AND:
                    //   ドロップ(合駒)にバイアスを加算し，非合駒を優先する．
                    //   通常の AND ノードでは玉逃げの反証が速いため．
                    let effective_cdn = if let Some(ksq) = chain_king_sq {
                        // チェーン AND: ドロップ優先，外側ほど後回し
                        if m.is_drop() {
                            let to = m.to_sq();
                            let dr = (to.row() as i8 - ksq.row() as i8)
                                .unsigned_abs() as u32;
                            let dc = (to.col() as i8 - ksq.col() as i8)
                                .unsigned_abs() as u32;
                            // 内側(d=1)はバイアス0，外側は距離に比例
                            cdn.saturating_add(dr.max(dc).saturating_sub(1))
                        } else {
                            // 非合駒: 大きなバイアスで後回し
                            cdn.saturating_add(INTERPOSE_DN_BIAS)
                        }
                    } else if m.is_drop() {
                        cdn.saturating_add(INTERPOSE_DN_BIAS)
                    } else {
                        cdn
                    };
                    if effective_cdn < best_effective_dn
                        || (effective_cdn == best_effective_dn
                            && cpn < best_pn_dn.0)
                    {
                        second_best = best_effective_dn;
                        best_effective_dn = effective_cdn;
                        best_idx = i;
                        best_pn_dn = (cpn, cdn);
                        best_source = csrc;
                    } else if effective_cdn < second_best {
                        second_best = effective_cdn;
                    }
                }

                if proved_or_disproved {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_loop_break_proved += 1; }
                    self.path.remove(&full_hash);
                    return;
                }

                // WPN (Weak Proof Number) / CD-WPN 計算:
                //
                // 通常 WPN: current_pn = max(cpn) + (unproven_count - 1)
                // CD-WPN:   current_pn = max(cpn) + (grouped_count - 1)
                //   where grouped_count = non_drops + unique_drop_squares
                //
                // チェーン AND: CD-WPN を使用．同一マスの未証明ドロップは
                // cross-deduce で一括証明できるため1グループとして数える．
                // 非チェーン AND: 通常 WPN を使用．
                if chain_king_sq.is_some() && cd_grouped_count > 0 {
                    current_pn = (max_cpn as u64)
                        .saturating_add(cd_grouped_count as u64 - 1)
                        .min(INF as u64) as u32;
                } else if unproven_count > 0 {
                    current_pn = (max_cpn as u64)
                        .saturating_add(unproven_count as u64 - 1)
                        .min(INF as u64) as u32;
                }

                // SNDA 補正: 同一 source の子は DAG 合流の可能性
                // 重複グループの最小値分を控除して過大評価を補正
                if snda_pairs.len() >= 2 {
                    current_pn = snda_dedup(&mut snda_pairs, current_pn);
                }

                // AND ノード証明(全子が証明済み)
                if all_proved && current_pn == 0 {
                    // 証明駒を現在の持ち駒で制限
                    for k in 0..HAND_KINDS {
                        and_proof[k] =
                            and_proof[k].min(att_hand[k]);
                    }
                    self.store(
                        pos_key, and_proof, 0, INF,
                        REMAINING_INFINITE, pos_key,
                    );
                    self.path.remove(&full_hash);
                    return;
                }
            }

            #[cfg(feature = "profile")]
            {
                self.profile_stats.main_loop_collect_ns += _collect_start.elapsed().as_nanos() as u64;
                self.profile_stats.main_loop_collect_count += 1;
            }

            // 転置表を更新(TT Best Move: 最善子の手を記録)
            //
            // 停滞ペナルティの保護: MID ループ初回の collect→store で，
            // 前回の stag_break が保存したペナルティ(TT の pn/dn > collect 値)
            // を max で保護する．これにより +1 ペナルティが蓄積可能になる．
            //
            // 2回目以降のイテレーションでは，子の mid() 実行後に pn/dn が
            // 変化する可能性があるため，collect 値をそのまま保存する．
            let best_move16 = children[best_idx].0.to_move16();
            // 停滞ペナルティの方向別保護:
            // OR ノード(攻方)は pn のみ max 保護(証明方向のペナルティ蓄積)，
            // AND ノード(受方)は dn のみ max 保護(反証方向のペナルティ蓄積)．
            // 非証明方向は collect 値をそのまま保存し，反証/証明の伝播を妨げない．
            // 保護は全イテレーションで適用し，ペナルティの上書き消失を防ぐ．
            let (store_pn, store_dn) = if _prev_nodes_used <= 1 {
                let (tt_pn, tt_dn, _) = self.look_up_pn_dn(pos_key, &att_hand, remaining);
                if or_node {
                    (current_pn.max(tt_pn), current_dn)
                } else {
                    (current_pn, current_dn.max(tt_dn))
                }
            } else {
                (current_pn, current_dn)
            };
            profile_timed!(self, tt_store_ns, tt_store_count,
                self.store_with_best_move(pos_key, att_hand, store_pn, store_dn, remaining, best_source, best_move16));

            // TCA (Kishimoto & Müller 2008; Kishimoto 2010): 過小評価対策
            //
            // OR ノードでループ子(path 上の子)が存在する場合，
            // 兄弟の pn/dn が過小評価されている可能性がある．
            // 閾値を加算的に拡張し，兄弟をより深く探索させる．
            // 拡張は MID ループ出口と子閾値の両方に適用する:
            // - MID 出口のみ拡張すると，子閾値が元の値に束縛され
            //   ループが空転する(attempt 2 の教訓)．
            // - 子閾値も含め加算的に拡張することで進捗を保証する．
            let (eff_pn_th, eff_dn_th) = if loop_child_count > 0 {
                (
                    pn_threshold
                        .saturating_add(pn_threshold / TCA_EXTEND_DENOM)
                        .saturating_add(1)
                        .min(INF - 1),
                    dn_threshold
                        .saturating_add(dn_threshold / TCA_EXTEND_DENOM)
                        .saturating_add(1)
                        .min(INF - 1),
                )
            } else {
                (pn_threshold, dn_threshold)
            };

            // 閾値チェック(TCA 拡張済み閾値を使用)
            if current_pn >= eff_pn_th
                || current_dn >= eff_dn_th
            {
                #[cfg(feature = "tt_diag")]
                { self.diag_loop_break_threshold += 1; }
                #[cfg(feature = "tt_diag")]
                if _diag_this_node && _diag_iteration <= 2 {
                    eprintln!(
                        "[tt_diag] ply={} loop break: iter={} pn={}/{} dn={}/{} children={} best={}",
                        ply, _diag_iteration,
                        current_pn, eff_pn_th,
                        current_dn, eff_dn_th,
                        children.len(),
                        children[best_idx].0.to_usi(),
                    );
                }
                // Killer Move 記録: OR ノードで pn 閾値超過時，
                // 最善子(最も有望な王手)を killer として保存する．
                // 同じ ply の別の局面でも同じ手が有効な可能性が高い．
                if or_node {
                    self.record_killer(ply, best_move16);
                }
                break;
            }

            // ノード制限・タイムアウトチェック
            if self.nodes_searched >= self.max_nodes
                || self.timed_out
            {
                #[cfg(feature = "tt_diag")]
                { self.diag_loop_break_nodes += 1; }
                break;
            }


            // TT GC: 定期的にサイズチェックし，閾値超過時に GC 実行
            if self.tt_gc_threshold > 0
                && self.nodes_searched >= self.next_gc_check
            {
                self.next_gc_check =
                    self.nodes_searched + 100_000;
                if self.table.len() > self.tt_gc_threshold {
                    self.table.gc(self.tt_gc_threshold * 3 / 4);
                }
            }

            // 閾値計算(1+ε トリック, Pawlewicz & Lew 2007)
            //
            // 標準 df-pn の second_best + 1 では，best child の pn/dn が
            // 僅かに増加しただけで親に戻りスラッシングが発生する．
            // 乗算型 ε を使用し，pn/dn に比例した余裕を与える:
            //   threshold = second_best + second_best/4 + 1
            //             ≈ ceil(second_best * 5/4)
            //
            // TCA 拡張: eff_*_th を使用し，ループ子存在時は
            // 子ノードにも拡張済み閾値を伝播する．
            let (child_pn_th, child_dn_th) = if or_node {
                // OR ノード dn 閾値の最低保証(dn_floor_or)．
                //
                // OR ノードの dn = sum(child_dn) であり，子が増えると
                // 各子の dn 予算 (dn_th − Σ他兄弟 dn + best_dn) が急速に縮小する．
                // 合駒チェーンの深部では予算が dn_floor(100) 未満に縮退し，
                // 子 AND ノードの TT dn が予算を上回って即座に TT exit する
                // 1-node スラッシングが発生する(ply-35 で 9.8M 回の空振り)．
                // dn_floor_or=100 を保証し，子 AND に探索進捗させる余裕を与える．
                let dn_floor_or: u32 = 100;
                let child_dn_th = eff_dn_th
                    .saturating_sub(current_dn)
                    .saturating_add(best_pn_dn.1)
                    .max(dn_floor_or)
                    .min(INF - 1);
                // OR ノード pn 閾値: 1+ε trick (sibling_based)．
                //
                // 子の pn 予算を sibling_based(second_best + ε)に制限し，
                // 不正解手から正解手への切替を全 OR ノードで強制する．
                let epsilon_or = second_best / 4 + 1;
                let sibling_based_or = second_best.saturating_add(epsilon_or);
                let child_pn_th = sibling_based_or.max(2).min(INF - 1);
                (child_pn_th, child_dn_th)
            } else {
                // AND ノード pn 閾値の最低保証(親予算の 1/2)．
                //
                // 標準の pn 閾値計算 (pn_th - current_pn + best_cpn) では，
                // AND ノードの未証明子が多い場合に current_pn が大きくなり，
                // 子の pn 閾値が急速にゼロに近づく．WPN では
                // current_pn = max(cpn) + (unproven_count - 1) であり，
                // 未証明子が10個なら current_pn >= 10，子の pn 閾値は
                // pn_th - 10 + 1 = pn_th - 9 と大幅に縮小する．
                //
                // 合駒チェーンでは AND ノードが連続し，各レベルで pn_th が
                // (unproven_count-1) ずつ減少するため，2〜3レベルで pn_th が
                // 1以下に縮退し MID が深部に到達できない(pn カスケード縮退)．
                //
                // pn_floor = eff_pn_th / 2 を最低保証することで，
                // 各 AND レベルで pn が最大2倍に縮退する速度に抑える．
                // 12レベルの AND でも pn ≈ INF/2^12 ≈ 1M が確保され，
                // 深い合駒チェーンの探索が可能になる．
                let pn_floor = eff_pn_th / 2;
                // 最低進捗保証: child_pn_th は最低でも best_child.pn + 1 を
                // 保証する．これにより eff_pn_th ≈ current_pn のとき
                // child_pn_th = best_child.pn となり mid() が即座に返る
                // ゼロ進捗パターンを防止する．
                let progress_floor = best_pn_dn.0.saturating_add(1);
                let child_pn_th = eff_pn_th
                    .saturating_sub(current_pn)
                    .saturating_add(best_pn_dn.0)
                    .max(pn_floor)
                    .max(progress_floor)
                    .min(INF - 1);
                let epsilon = second_best / 4 + 1;
                let sibling_based = second_best.saturating_add(epsilon);
                // AND ノード dn 閾値の最低保証．
                //
                // 初期 dn=1(depth_biased_dn 廃止後)では sibling_based ≈ 2 と
                // 極端に小さくなるため，dn_floor なしでは全く深部に到達できない．
                // dn_floor=100 で深部までの到達を確保する．
                //
                // チェーン合駒の AND ノードでは，親 OR ノードからの dn 閾値
                // (eff_dn_th) がチェーンの深さ分だけ縮退し dn_floor を下回る．
                // キャップ(eff_dn_th.min(...))を外して dn_floor を保証し，
                // チェーン末端の証明に十分な探索予算を確保する(§3 最適化)．
                let dn_floor: u32 = 100;
                let child_dn_th = if chain_king_sq.is_some() {
                    sibling_based.max(dn_floor).min(INF - 1)
                } else {
                    eff_dn_th
                        .min(sibling_based.max(dn_floor))
                        .min(INF - 1)
                };
                (child_pn_th, child_dn_th)
            };

            // TT 診断: 反復カウンタ + 上限チェック
            #[cfg(feature = "tt_diag")]
            {
                if _diag_this_node {
                    _diag_iteration += 1;
                    if self.diag_max_iterations > 0
                        && _diag_iteration > self.diag_max_iterations
                    {
                        eprintln!(
                            "[tt_diag] ply={} iteration limit reached ({}), \
                             tt_pos={} tt_ent={} current_pn={} current_dn={}",
                            ply, self.diag_max_iterations,
                            self.table.len(), self.table.total_entries(),
                            current_pn, current_dn,
                        );
                        break;
                    }
                }
            }

            // 子ノードを探索
            let (m, _, _, _) = children[best_idx];
            let captured = profile_timed!(self, do_move_ns, do_move_count,
                board.do_move(m));

            // AND ノードの合駒子ノード選択時: 取り後 TT 先読み
            // MID 再帰前に「取りの王手 → 既証明局面」を1回だけチェックし，
            // 成功すれば MID 再帰を完全にスキップする．
            // 収集ループ内で全子に対して行うと NPS が激減するため，
            // 選択された子に対してのみ実行する．
            if !or_node && m.is_drop() && remaining >= 3 {
                #[cfg(feature = "profile")]
                let _cap_tt_start = Instant::now();
                #[cfg(feature = "tt_diag")]
                { self.diag_capture_tt_calls += 1; }
                let checks = self.generate_check_moves(board);
                if self.try_capture_tt_proof(
                    board, &checks,
                    remaining.saturating_sub(1),
                ) {
                    // 証明を store したが，hand dominance の不一致で
                    // look_up_pn_dn が証明を検出できない場合がある．
                    // 検出できなければ continue せず通常の mid() に fallback する．
                    let child_pk = children[best_idx].2;
                    let child_hand = &children[best_idx].3;
                    let (verified_pn, _, _) = self.look_up_pn_dn(
                        child_pk, child_hand, remaining.saturating_sub(1));
                    if verified_pn == 0 {
                        #[cfg(feature = "tt_diag")]
                        { self.diag_capture_tt_hits += 1; }
                        #[cfg(feature = "profile")]
                        {
                            self.profile_stats.capture_tt_lookahead_ns += _cap_tt_start.elapsed().as_nanos() as u64;
                            self.profile_stats.capture_tt_lookahead_count += 1;
                        }
                        // 証明済みを確認 → MID 再帰をスキップ
                        profile_timed!(self, undo_move_ns, undo_move_count,
                            board.undo_move(m, captured));
                        continue;
                    }
                    // 証明が look_up で検出できなかった → mid() にフォールスルー
                }
                #[cfg(feature = "profile")]
                {
                    self.profile_stats.capture_tt_lookahead_ns += _cap_tt_start.elapsed().as_nanos() as u64;
                    self.profile_stats.capture_tt_lookahead_count += 1;
                }
            }

            #[cfg(feature = "tt_diag")]
            let _diag_match = self.diag_ply > 0 && ply == self.diag_ply && {
                let usi = m.to_usi();
                self.diag_move_usi.is_empty() || usi == self.diag_move_usi
            };
            #[cfg(feature = "tt_diag")]
            let (_diag_tt_pos_before, _diag_tt_ent_before, _diag_nodes_before) = if _diag_match {
                (self.table.len(), self.table.total_entries(), self.nodes_searched)
            } else {
                (0, 0, 0)
            };

            let _pre_mid_nodes = self.nodes_searched;
            self.mid(
                board,
                child_pn_th,
                child_dn_th,
                ply + 1,
                !or_node,
            );
            _prev_nodes_used = self.nodes_searched - _pre_mid_nodes;
            // 零進捗検出: 子 mid() が 0 ノードしか消費しなかった場合，
            // 子は閾値チェックで即座に返っている．これが連続すると
            // dn_floor 由来の空転が発生するため，ZERO_PROGRESS_LIMIT 回
            // 連続で発生したらループを脱出する．
            // 深さ制限近傍(remaining <= 4)では子が深さ制限で返るのは正常動作
            // であるため，この検出を無効化する．
            if remaining > 4 {
                let nodes_used = self.nodes_searched - _pre_mid_nodes;
                if nodes_used <= 1 {
                    zero_progress_count += 1;
                    if zero_progress_count >= ZERO_PROGRESS_LIMIT {
                        board.undo_move(m, captured);
                        break;
                    }
                } else {
                    zero_progress_count = 0;
                }
            }

            // 停滞検出: 同じ子に同じ閾値で mid() を呼んで pn/dn が変化しなければ，
            // 再度呼んでも結果は同じ(TT にキャッシュ済み)．
            // 閾値が増えた場合のみ新たな探索が可能なため，そのケースはリセットする．
            //
            // 最適化: 子 mid() が 2+ ノード消費した場合は探索が進展した可能性が
            // 高いため，stagnation check をスキップして look_up コストを回避する．
            {
                let nodes_used = self.nodes_searched - _pre_mid_nodes;
                if nodes_used > 1 {
                    stagnation_count = 0;
                    prev_best_idx = best_idx;
                    prev_best_pn = 0;
                    prev_best_dn = 0;
                    prev_child_pn_th = child_pn_th;
                    prev_child_dn_th = child_dn_th;
                } else {
                let (cpn_after, cdn_after, _) = self.look_up_pn_dn(
                    children[best_idx].2,
                    &children[best_idx].3,
                    remaining.saturating_sub(1),
                );
                if best_idx == prev_best_idx
                    && cpn_after == prev_best_pn
                    && cdn_after == prev_best_dn
                    && child_pn_th <= prev_child_pn_th
                    && child_dn_th <= prev_child_dn_th
                {
                    stagnation_count += 1;
                    if stagnation_count >= STAGNATION_LIMIT {
                        if (ply as usize) < 64 {
                            self.ply_stag_penalties[ply as usize] += 1;
                        }
                        board.undo_move(m, captured);
                        // 停滞ペナルティ(指数増加): 証明方向のみ TT 値を倍増．
                        //
                        // 線形 +1 では閾値が INF 付近のとき収束に数十億回必要．
                        // 既存の蓄積量(TT - collect)を新たなペナルティとし，
                        // 各 stag_break で gap が倍増するようにする:
                        //   gap: 1 → 2 → 4 → 8 → ... → ~32 回で INF 到達．
                        let (tt_pn, tt_dn, _) = self.look_up_pn_dn(pos_key, &att_hand, remaining);
                        let (stag_pn, stag_dn) = if or_node {
                            let base = current_pn.max(tt_pn);
                            let penalty = tt_pn.saturating_sub(current_pn).max(1);
                            (base.saturating_add(penalty).min(INF - 1), current_dn)
                        } else {
                            let base = current_dn.max(tt_dn);
                            let penalty = tt_dn.saturating_sub(current_dn).max(1);
                            (current_pn, base.saturating_add(penalty).min(INF - 1))
                        };
                        self.store_with_best_move(
                            pos_key, att_hand, stag_pn, stag_dn,
                            remaining, best_source, best_move16);
                        break;
                    }
                } else {
                    stagnation_count = 0;
                }
                prev_best_idx = best_idx;
                prev_best_pn = cpn_after;
                prev_best_dn = cdn_after;
                prev_child_pn_th = child_pn_th;
                prev_child_dn_th = child_dn_th;
            }}

            #[cfg(feature = "tt_diag")]
            if _diag_match {
                let tt_pos_after = self.table.len();
                let tt_ent_after = self.table.total_entries();
                let nodes_after = self.nodes_searched;
                let (cpn_after, cdn_after, _) = self.look_up_pn_dn(
                    children[best_idx].2,
                    &children[best_idx].3,
                    remaining.saturating_sub(1),
                );
                eprintln!(
                    "[tt_diag] ply={} move={} node={} \
                     pn_th={} dn_th={} \
                     child_pn={} child_dn={} \
                     tt_pos: {}→{} (+{}) \
                     tt_ent: {}→{} (+{}) \
                     nodes_used={}",
                    ply, m.to_usi(), self.nodes_searched,
                    child_pn_th, child_dn_th,
                    cpn_after, cdn_after,
                    _diag_tt_pos_before, tt_pos_after,
                    tt_pos_after.saturating_sub(_diag_tt_pos_before),
                    _diag_tt_ent_before, tt_ent_after,
                    tt_ent_after.saturating_sub(_diag_tt_ent_before),
                    nodes_after.saturating_sub(_diag_nodes_before),
                );
            }

            profile_timed!(self, undo_move_ns, undo_move_count,
                board.undo_move(m, captured));

            // ply=0 の mid() 呼び出しごとの消費ノード追跡
            if ply == 0 {
                let child_nodes = self.nodes_searched - _pre_mid_nodes;
                if child_nodes >= 1_000 {
                    let (cpn_now, cdn_now, _) = self.look_up_pn_dn(
                        children[best_idx].2, &children[best_idx].3,
                        remaining.saturating_sub(1));
                    eprintln!("[root_mid] move={} nodes={}K pn_th={} dn_th={} → pn={} dn={} total={}K time={:.1}s",
                        m.to_usi(), child_nodes / 1000,
                        child_pn_th, child_dn_th,
                        cpn_now, cdn_now,
                        self.nodes_searched / 1000,
                        self.start_time.elapsed().as_secs_f64());
                }
            }

            // インライン cross-deduce: AND ノードでドロップ子が証明された直後に，
            // 同一マスの兄弟ドロップを TT 参照で証明する．
            // 旧 deferred_children 方式の cross_deduce_deferred と同等の効果を
            // MID ループ内で実現する．
            if !or_node && m.is_drop() {
                let (cpn_after, _, _) = self.look_up_pn_dn(
                    children[best_idx].2,
                    &children[best_idx].3,
                    remaining.saturating_sub(1),
                );
                if cpn_after == 0 {
                    #[cfg(feature = "profile")]
                    let _cd_start = Instant::now();
                    self.cross_deduce_children(
                        board, m, &children, remaining,
                    );
                    #[cfg(feature = "profile")]
                    {
                        self.profile_stats.cross_deduce_ns += _cd_start.elapsed().as_nanos() as u64;
                        self.profile_stats.cross_deduce_count += 1;
                    }
                }
            }

            // (effort tracking は self.or_effort に蓄積済み．
            // 次のループ反復で effort ペナルティが選択に反映される．)
        }

        // パスから除去
        self.path.remove(&full_hash);
    }

    /// 既に生成済みの王手リストを使って1手詰め判定する．
    ///
    /// AND 子ノード(守備側局面)のヒューリスティック初期 pn を計算する．
    ///
    /// 玉の逃げ場(安全なマスの数)に基づいて pn を調整する:
    /// - 逃げ場が少ない → 詰みやすい → pn を小さく
    /// - 逃げ場が多い → 詰みにくい → pn を大きく
    ///
    /// KomoringHeights v0.4.0 のヒューリスティック初期化を参考にした手法．
    fn heuristic_and_pn(&self, board: &Board, num_defenses: u32) -> u32 {
        let defender = board.turn;
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return num_defenses,
        };

        // 玉の安全な逃げ場をカウント(ビットボード一括判定)
        // compute_king_danger は X-ray(玉を除いた占有)を使うため，
        // 玉が移動した先で飛び駒に当たるケースも正しく検出する．
        let king_moves = attack::step_attacks(defender, PieceType::King, king_sq);
        let our_occ = board.occupied[defender.index()];
        let danger = board.compute_king_danger(defender, king_sq);
        let safe_escapes = (king_moves & !our_occ & !danger).count();

        // 逃げ場に基づく pn 調整
        if safe_escapes == 0 {
            // 逃げ場なし: 合駒・駒取りのみ → 詰みやすい
            (num_defenses * 2 / 3).max(1)
        } else if safe_escapes >= 3 {
            // 逃げ場が多い: 詰みにくい
            num_defenses + safe_escapes / 2
        } else {
            num_defenses
        }
    }

    /// OR 子ノード(攻め方局面)のヒューリスティック初期 pn を計算する(df-pn+)．
    ///
    /// 標準 df-pn では OR ノードの初期 pn=1 だが，これでは全ての
    /// OR ノードが等しく「1手で詰む可能性がある」と見積もられる．
    /// 実際は玉の逃げ場が多い局面ほど詰みにくく，追い詰めに多くの手を要する．
    ///
    /// AND 親ノードの sum(pn) に直接影響し，閾値配分の精度を向上させる．
    fn heuristic_or_pn(&self, board: &Board, num_checks: u32) -> u32 {
        if num_checks == 0 {
            return INF; // 王手なし → 不詰(呼び出し側で処理済みのはず)
        }

        let defender = board.turn.opponent();
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return 1,
        };

        // 玉の安全な逃げ場をカウント(ビットボード一括判定)
        let king_moves = attack::step_attacks(defender, PieceType::King, king_sq);
        let def_occ = board.occupied[defender.index()];
        let danger = board.compute_king_danger(defender, king_sq);
        let safe_escapes = (king_moves & !def_occ & !danger).count();

        // --- 開放空間逃走検出(人間的枝刈り) ---
        // 玉周辺(隣接8マス)への攻め駒の利き数が少なく，かつ逃げ場が多い場合，
        // 人間が「玉が広い方に逃げて捕まらない」と直感するのと同様に
        // pn を引き上げて探索優先度を下げる．
        let king_adjacent = king_moves & !def_occ; // 玉が移動可能なマス(自駒除外)
        let pressured = (king_adjacent & danger).count(); // 攻め方に利かれているマス数
        let adjacent_total = king_adjacent.count(); // 移動可能マス総数

        if adjacent_total >= 5 && pressured == 0 && safe_escapes >= 4 {
            // 玉周辺に攻め駒の利きが皆無の開放空間 → 非常に詰みにくい
            return 3;
        }

        // 王手数が少なく逃げ場が多い → 追い詰めが困難
        // 王手数が多く逃げ場がない → 包囲完成に近い
        if num_checks <= 2 && safe_escapes >= 3 {
            // 王手が少なく逃げ場が多い → 詰みにくい(上限3: 不詰証明遅延を抑制)
            return (2 + safe_escapes / 2).min(3);
        }
        if safe_escapes == 0 {
            // 逃げ場なし → 詰みやすい(合駒のみで防御)
            return 1;
        }
        if safe_escapes >= 4 {
            // 逃げ場が非常に多い → 追い詰めに手数を要する
            return 1 + safe_escapes / 3;
        }
        // 標準的な局面
        1
    }

    /// OR 子ノード(攻め方局面)で，取りの王手が既証明局面に到達するか TT を先読みする．
    ///
    /// 合駒対策の核心的最適化: 異なる駒種の合駒後の局面は盤面が異なるが，
    /// 攻め方がその合駒を取った後の局面は「盤面同一・持ち駒のみ異なる」ため，
    /// TT の持ち駒優越(hand dominance)でマッチする可能性が高い．
    ///
    /// 合駒 A の取り後局面が証明済みなら，合駒 B の取り後局面は
    /// 攻め方の持ち駒が合駒 A の取り後より多い限り TT ヒットする．
    /// これにより，2手先読みのコストで大量の合駒分岐を即座に証明できる．
    fn try_capture_tt_proof(
        &mut self,
        board: &mut Board,
        checks: &ArrayVec<Move, MAX_MOVES>,
        child_remaining: u16,
    ) -> bool {
        if child_remaining < 1 {
            return false;
        }
        let capture_remaining = child_remaining.saturating_sub(1);
        for check in checks {
            if check.is_drop() || check.captured_piece_raw() == 0 {
                continue;
            }
            let captured = board.do_move(*check);
            let cap_pk = position_key(board);
            let cap_hand = board.hand[self.attacker.index()];
            let (cap_pn, _, _) = self.look_up_pn_dn(
                cap_pk, &cap_hand, capture_remaining,
            );
            if cap_pn == 0 {
                // 取り後の局面が証明済み → この OR ノード(子)は証明済み
                // 証明駒は取り後の証明駒を調整して使用
                let cap_proof = self.table.get_proof_hand(cap_pk, &cap_hand);
                let proof = adjust_hand_for_move(*check, &cap_proof);
                // undo_move で子(OR ノード)の局面に戻してから store する．
                // board は子の局面を指すため，store_board_with_hand は
                // 子の position_key で TT に保存する．
                board.undo_move(*check, captured);
                self.store_board_with_hand(board, &proof, 0, INF, REMAINING_INFINITE, cap_pk);
                return true;
            }
            board.undo_move(*check, captured);
        }
        false
    }

    /// ビットボード演算のみで詰み判定を行い，do_move/undo_move の
    /// オーバーヘッドを回避する(cshogi の mateMoveIn1Ply 相当)．
    fn has_mate_in_1_with(
        &mut self,
        board: &mut Board,
        checks: &ArrayVec<Move, MAX_MOVES>,
    ) -> bool {
        let us = board.turn;
        if let Some(mate_move) = board.mate_move_in_1ply(checks.as_slice(), us) {
            // 詰み局面を TT に記録するために do_move が必要
            let captured = board.do_move(mate_move);
            let pk = position_key(board);
            self.store(pk, [0; HAND_KINDS], 0, INF,
                REMAINING_INFINITE, pk);
            board.undo_move(mate_move, captured);
            return true;
        }
        false
    }

    /// TT ベース合駒プレフィルタ: 合駒の捕獲後局面がメイン TT で
    /// 証明済みなら，合駒の OR ノードを展開せずに証明確定する．
    ///
    /// IDS のボトムアップ特性を活用する:
    /// 1. 浅い IDS 反復で深いレベルの合駒チェーン末端が証明される
    /// 2. 証明は `retain_proofs` でメイン TT に保持される
    /// 3. 深い IDS 反復で，浅いレベルの合駒処理時にこの証明を参照し，
    ///    合駒チェーンの展開をスキップする
    ///
    /// 返り値: true なら証明済み(and_proof に蓄積済み)，false なら未証明．
    #[inline(never)]
    fn try_prefilter_block(
        &mut self,
        board: &mut Board,
        block_move: Move,
        child_hand: &[u8; HAND_KINDS],
        remaining: u16,
        and_proof: &mut [u8; HAND_KINDS],
    ) -> bool {
        // 合駒の捕獲後に使える remaining
        let pc_remaining = remaining.saturating_sub(2);
        if pc_remaining == 0 {
            #[cfg(feature = "tt_diag")]
            { self.diag_prefilter_skip_remaining += 1; }
            return false;
        }

        let target_sq = block_move.to_sq();

        // 合駒を盤上で実行
        let captured_by_block = board.do_move(block_move);
        let child_pk = position_key(board); // 合駒後(OR ノード)の position_key

        // 攻方の合法手から，合駒マスへの捕獲かつ王手になる手を探す
        let legal = movegen::generate_legal_moves(board);
        let mut proved = false;

        for cap_mv in legal.iter().filter(|mv| {
            mv.to_sq() == target_sq && mv.captured_piece_raw() > 0
        }) {
            let cap_piece = board.do_move(*cap_mv);

            // 捕獲が王手でなければ詰将棋の合法手ではない
            if !board.is_in_check(board.turn) {
                board.undo_move(*cap_mv, cap_piece);
                continue;
            }

            let pc_pk = position_key(board);
            let pc_hand = board.hand[self.attacker.index()];

            // メイン TT で捕獲後局面の証明を参照
            let (ppn, _, _) = self.table.look_up(pc_pk, &pc_hand, pc_remaining);
            if ppn == 0 {
                // 捕獲後局面が証明済み → 合駒の OR ノードも証明
                let pc_ph = self.table.get_proof_hand(pc_pk, &pc_hand);

                // OR ノードの証明駒: 捕獲で得る駒分を差し引く
                let cap_raw = cap_mv.captured_piece_raw();
                let mut or_ph = pc_ph;
                if cap_raw > 0 {
                    let piece = Piece::from_raw_u8(cap_raw);
                    if let Some(pt) = piece.piece_type() {
                        let base_pt = pt.unpromoted().unwrap_or(pt);
                        if let Some(hi) = base_pt.hand_index() {
                            or_ph[hi] = or_ph[hi].saturating_sub(1);
                        }
                    }
                }
                // 子ノードの持ち駒で上限クリップ
                for k in 0..HAND_KINDS {
                    or_ph[k] = or_ph[k].min(child_hand[k]);
                }

                // 子 TT に証明エントリを格納(後続の look_up で再利用)
                self.table.store(
                    child_pk, or_ph, 0, INF,
                    remaining.saturating_sub(1), child_pk,
                );

                // AND 証明駒の更新
                let adj = adjust_hand_for_move(block_move, &or_ph);
                for k in 0..HAND_KINDS {
                    and_proof[k] = and_proof[k].max(adj[k]);
                }
                proved = true;
            }

            board.undo_move(*cap_mv, cap_piece);
            if proved {
                break;
            }
        }

        board.undo_move(block_move, captured_by_block);
        #[cfg(feature = "tt_diag")]
        if !proved {
            self.diag_prefilter_miss += 1;
        }
        proved
    }

    /// 同一マス合駒の捕獲後 TT 転用(証明のみ)．
    ///
    /// 合駒 `solved_idx` が証明済みになった後，同一マスの他の合駒について
    /// 攻方の捕獲後の共通局面を TT で参照し，証明を転用する．
    ///
    /// ## 原理
    ///
    /// 同一マス S への合駒 P1, P2, ..., Pn は，攻方が捕獲した後の
    /// 盤面(position_key)が全て同一になる(捕獲駒が S に移動し，合駒が
    /// 盤上から消える)．異なるのは攻方の持ち駒のみ(+P_i 分)．
    ///
    /// 合駒 P_i の捕獲後局面が TT で証明済み(pn=0)ならば，攻方は
    /// 「合駒 P_i を取って王手」→「証明済み手順で詰み」と進めるため，
    /// 合駒 P_i の子ノード(OR ノード)も pn=0 と確定できる．
    /// メイン TT 上での同一マス合駒証明転用．
    ///
    /// `children` 内の証明済みドロップ手 `solved_move` に対し，
    /// `deferred_children` の同一マスの合駒を TT から証明転用する．
    ///
    /// 証明された合駒は `deferred_children` から除去し，
    /// `and_proof` に証明駒を蓄積する．
    #[inline(never)]
    fn cross_deduce_deferred(
        &mut self,
        board: &mut Board,
        solved_move: Move,
        deferred_children: &mut ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        remaining: u16,
        and_proof: &mut [u8; HAND_KINDS],
    ) {
        let target_sq = solved_move.to_sq();

        // 同一マスに未解決の合駒がなければスキップ
        let has_siblings = deferred_children.iter().any(|(mj, _, _, _)| {
            mj.to_sq() == target_sq
        });
        if !has_siblings {
            return;
        }

        let solved_pt = match solved_move.drop_piece_type() {
            Some(pt) => pt,
            None => return,
        };
        let solved_hi = match solved_pt.hand_index() {
            Some(hi) => hi,
            None => return,
        };

        // 合駒を実行し，攻方の捕獲手を探索
        let captured_by_block = board.do_move(solved_move);
        let legal = movegen::generate_legal_moves(board);

        // 捕獲手(ターゲットマスへの駒取り)を全て試行
        let mut proven_indices: ArrayVec<usize, MAX_MOVES> = ArrayVec::new();

        for cap_mv in legal.iter().filter(|mv| {
            mv.to_sq() == target_sq && mv.captured_piece_raw() > 0
        }) {
            let cap_piece = board.do_move(*cap_mv);

            // 捕獲が王手でなければ詰将棋の合法手ではない → スキップ
            if !board.is_in_check(board.turn) {
                board.undo_move(*cap_mv, cap_piece);
                continue;
            }

            let pc_pk = position_key(board);
            let base_hand = board.hand[self.attacker.index()];
            board.undo_move(*cap_mv, cap_piece);

            // 各未解決の同一マス合駒について TT 参照
            for (j, &(ref mj, _, child_pk_j, ref child_hand_j))
                in deferred_children.iter().enumerate()
            {
                if mj.to_sq() != target_sq {
                    continue;
                }
                if proven_indices.contains(&j) {
                    continue;
                }

                let pt_j = match mj.drop_piece_type() {
                    Some(pt) => pt,
                    None => continue,
                };
                let hi_j = match pt_j.hand_index() {
                    Some(hi) => hi,
                    None => continue,
                };

                // 合駒 j を捕獲した場合の攻方持ち駒を計算:
                // base_hand は solved_move の駒を捕獲した状態なので，
                // solved の駒分を引いて j の駒分を足す
                let mut hand_j = base_hand;
                hand_j[solved_hi] = hand_j[solved_hi].saturating_sub(1);
                hand_j[hi_j] = hand_j[hi_j].saturating_add(1);

                let pc_remaining = remaining.saturating_sub(2);
                let (ppn, _, _) = self.table.look_up(pc_pk, &hand_j, pc_remaining);

                if ppn == 0 {
                    // 捕獲後局面が証明済み → 合駒 j の OR ノードも証明
                    let pc_ph = self.table.get_proof_hand(pc_pk, &hand_j);

                    // OR ノードの証明駒: 捕獲で得る駒 j の分を差し引く
                    let mut or_ph = pc_ph;
                    or_ph[hi_j] = or_ph[hi_j].saturating_sub(1);
                    for k in 0..HAND_KINDS {
                        or_ph[k] = or_ph[k].min(child_hand_j[k]);
                    }

                    // メイン TT に証明エントリを格納
                    self.table.store(
                        child_pk_j, or_ph, 0, INF,
                        remaining.saturating_sub(1), child_pk_j,
                    );

                    // AND 証明駒の更新
                    let adj = adjust_hand_for_move(*mj, &or_ph);
                    for k in 0..HAND_KINDS {
                        and_proof[k] = and_proof[k].max(adj[k]);
                    }
                    let _ = proven_indices.try_push(j);
                }
            }
        }

        board.undo_move(solved_move, captured_by_block);

        // 証明済みの合駒を deferred_children から除去(降順で安全に削除)
        proven_indices.sort_unstable();
        #[cfg(feature = "tt_diag")]
        { self.diag_cross_deduce_hits += proven_indices.len() as u64; }
        for &i in proven_indices.iter().rev() {
            deferred_children.remove(i);
        }
    }

    /// children 内の証明済みドロップから兄弟ドロップを TT で証明する．
    ///
    /// `cross_deduce_deferred` と同等のロジックだが，`children` を読み取り専用で
    /// 参照し，TT にエントリを格納するのみ(children からの除去は行わない)．
    /// MID ループの次の collect フェーズで cpn=0 として検出される．
    #[inline(never)]
    fn cross_deduce_children(
        &mut self,
        board: &mut Board,
        solved_move: Move,
        children: &ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        remaining: u16,
    ) {
        let target_sq = solved_move.to_sq();

        // 同一マスに未解決のドロップ兄弟がなければスキップ
        let has_siblings = children.iter().any(|(mj, _, _, _)| {
            mj.is_drop() && mj.to_sq() == target_sq
                && !std::ptr::eq(&solved_move as *const _, mj as *const _)
        });
        if !has_siblings {
            return;
        }

        let solved_pt = match solved_move.drop_piece_type() {
            Some(pt) => pt,
            None => return,
        };
        let solved_hi = match solved_pt.hand_index() {
            Some(hi) => hi,
            None => return,
        };

        // 合駒を実行し，攻方の捕獲手を探索
        let captured_by_block = board.do_move(solved_move);
        let legal = movegen::generate_legal_moves(board);
        let mut cross_count: u64 = 0;

        for cap_mv in legal.iter().filter(|mv| {
            mv.to_sq() == target_sq && mv.captured_piece_raw() > 0
        }) {
            let cap_piece = board.do_move(*cap_mv);

            if !board.is_in_check(board.turn) {
                board.undo_move(*cap_mv, cap_piece);
                continue;
            }

            let pc_pk = position_key(board);
            let base_hand = board.hand[self.attacker.index()];
            board.undo_move(*cap_mv, cap_piece);

            // 各兄弟ドロップについて TT 参照
            for (mj, _, child_pk_j, child_hand_j) in children.iter() {
                if !mj.is_drop() || mj.to_sq() != target_sq {
                    continue;
                }
                // 自分自身はスキップ
                if mj.to_move16() == solved_move.to_move16() {
                    continue;
                }
                // 既に証明済みならスキップ
                let (cpn_j, _, _) = self.look_up_pn_dn(
                    *child_pk_j, child_hand_j,
                    remaining.saturating_sub(1),
                );
                if cpn_j == 0 {
                    continue;
                }

                let pt_j = match mj.drop_piece_type() {
                    Some(pt) => pt,
                    None => continue,
                };
                let hi_j = match pt_j.hand_index() {
                    Some(hi) => hi,
                    None => continue,
                };

                // 合駒 j を捕獲した場合の攻方持ち駒を計算
                let mut hand_j = base_hand;
                hand_j[solved_hi] = hand_j[solved_hi].saturating_sub(1);
                hand_j[hi_j] = hand_j[hi_j].saturating_add(1);

                let pc_remaining = remaining.saturating_sub(2);
                let (ppn, _, _) = self.table.look_up(pc_pk, &hand_j, pc_remaining);

                if ppn == 0 {
                    let pc_ph = self.table.get_proof_hand(pc_pk, &hand_j);
                    let mut or_ph = pc_ph;
                    or_ph[hi_j] = or_ph[hi_j].saturating_sub(1);
                    for k in 0..HAND_KINDS {
                        or_ph[k] = or_ph[k].min(child_hand_j[k]);
                    }
                    // メイン TT に証明エントリを格納
                    self.table.store(
                        *child_pk_j, or_ph, 0, INF,
                        remaining.saturating_sub(1), *child_pk_j,
                    );
                    cross_count += 1;
                }
            }
        }

        board.undo_move(solved_move, captured_by_block);

        #[cfg(feature = "tt_diag")]
        { self.diag_cross_deduce_hits += cross_count; }
    }
}

/// 予算制の静的詰め探索の結果．
///
/// 「確定的な不詰」と「予算切れ」を区別することで，
/// AND ノード側で安全に不詰を TT にキャッシュできる．
enum StaticMateResult {
    /// 詰み検出．最小証明駒を返す．
    Checkmate([u8; HAND_KINDS]),
    /// 探索範囲内で確定的に不詰．
    NoCheckmate,
    /// 予算切れにより結論が出ていない．
    Exhausted,
}

impl DfPnSolver {
    /// 予算制の静的詰め探索(OR ノード側)．
    ///
    /// Df-Pn の閾値・パス管理なしで，純粋な再帰探索により
    /// N 手詰めを検出する．`budget` を消費しながら可能な限り深く探索し，
    /// 予算を使い切ると `Exhausted` を返す．
    ///
    /// # 戻り値
    /// - `Checkmate(proof_hand)`: 詰み検出．最小証明駒を返す
    /// - `NoCheckmate`: 探索範囲内で確定的に不詰
    /// - `Exhausted`: 予算切れにより結論が出ていない
    fn static_mate_or(
        &mut self,
        board: &mut Board,
        remaining: u32,
        budget: &mut u32,
    ) -> StaticMateResult {
        if remaining == 0 {
            return StaticMateResult::Exhausted;
        }

        let pos_key = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        let rem16 = remaining as u16;
        let (tt_pn, tt_dn, _) = self.table.look_up(pos_key, &att_hand, rem16);
        // TT に証明/反証エントリがある場合は budget を消費せず即返却．
        // budget は「新規に探索するノード数」の制限として機能させる．
        if tt_pn == 0 {
            return StaticMateResult::Checkmate(
                self.table.get_proof_hand(pos_key, &att_hand),
            );
        }
        if tt_dn == 0 {
            return StaticMateResult::NoCheckmate;
        }

        // TT miss: ここで初めて budget を消費
        if *budget == 0 {
            return StaticMateResult::Exhausted;
        }
        *budget = budget.saturating_sub(1);

        let checks = self.generate_check_moves(board);
        if checks.is_empty() {
            self.table.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key);
            return StaticMateResult::NoCheckmate;
        }

        // 1手詰め判定
        let us = board.turn;
        if let Some(mate_move) = board.mate_move_in_1ply(checks.as_slice(), us) {
            let captured = board.do_move(mate_move);
            let child_pk = position_key(board);
            self.store(child_pk, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, child_pk);
            board.undo_move(mate_move, captured);
            let mut proof = adjust_hand_for_move(mate_move, &[0; HAND_KINDS]);
            for k in 0..HAND_KINDS {
                proof[k] = proof[k].min(att_hand[k]);
            }
            self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
            return StaticMateResult::Checkmate(proof);
        }
        if remaining <= 1 {
            return StaticMateResult::NoCheckmate;
        }

        // 3手以上: 各王手について全応手を再帰的にチェック
        //
        // NOTE: OR ノードでは loop-end での不詰 TT 記録を行わない．
        // OR は全子(全王手)を確認する必要があるため，各 child_init で
        // 蓄積された AND 不詰 TT エントリが連鎖的に再利用され，
        // remaining が不正にエスカレートする問題が発生する．
        // AND 側は逃れ手1つで確定するためこの問題は生じない．
        let mut has_exhausted = false;
        for m in &checks {
            let captured = board.do_move(*m);
            let child_result = self.static_mate_and(board, remaining - 1, budget);
            board.undo_move(*m, captured);
            match child_result {
                StaticMateResult::Checkmate(child_proof) => {
                    let mut proof = adjust_hand_for_move(*m, &child_proof);
                    for k in 0..HAND_KINDS {
                        proof[k] = proof[k].min(att_hand[k]);
                    }
                    self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                    return StaticMateResult::Checkmate(proof);
                }
                StaticMateResult::NoCheckmate => {
                    // この王手は確定的に不成立 → 次の王手を試す
                }
                StaticMateResult::Exhausted => {
                    has_exhausted = true;
                    if *budget == 0 {
                        return StaticMateResult::Exhausted;
                    }
                }
            }
        }
        // 全王手を試行済み:
        // - Exhausted な子が1つでもあれば結論未確定
        // - 全子が NoCheckmate なら確定的に不詰
        if has_exhausted {
            StaticMateResult::Exhausted
        } else {
            StaticMateResult::NoCheckmate
        }
    }

    /// 予算制の静的詰め探索(AND ノード側)．
    ///
    /// # 戻り値
    /// - `Checkmate(proof_hand)`: 全応手に対して詰み検出．最小証明駒を返す
    /// - `NoCheckmate`: いずれかの応手で確定的に不詰
    /// - `Exhausted`: 予算切れにより結論が出ていない
    fn static_mate_and(
        &mut self,
        board: &mut Board,
        remaining: u32,
        budget: &mut u32,
    ) -> StaticMateResult {
        if remaining == 0 {
            return StaticMateResult::Exhausted;
        }

        let pos_key = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        let rem16 = remaining as u16;
        let (tt_pn, tt_dn, _) = self.table.look_up(pos_key, &att_hand, rem16);
        // TT に証明/反証エントリがある場合は budget を消費せず即返却．
        if tt_pn == 0 {
            return StaticMateResult::Checkmate(
                self.table.get_proof_hand(pos_key, &att_hand),
            );
        }
        if tt_dn == 0 {
            return StaticMateResult::NoCheckmate;
        }

        // TT miss: ここで初めて budget を消費
        if *budget == 0 {
            return StaticMateResult::Exhausted;
        }
        *budget = budget.saturating_sub(1);

        let defenses = self.generate_defense_moves(board);
        if defenses.is_empty() {
            self.table.store(pos_key, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, pos_key);
            return StaticMateResult::Checkmate([0; HAND_KINDS]);
        }
        if remaining < 2 {
            // 応手はあるが残り深さ不足で再帰不可 → 予算切れ扱い
            return StaticMateResult::Exhausted;
        }

        // 全応手に詰みがあるか確認(証明駒は要素ごと max)
        let mut and_proof = [0u8; HAND_KINDS];
        let mut any_legal = false;
        for d in &defenses {
            let cap_d = board.do_move(*d);
            if board.is_in_check(board.turn.opponent()) {
                // 応手後に玉方が王手されている(非合法手) → スキップ
                // 全応手が非合法ならループ末尾の Checkmate パスに到達する
                board.undo_move(*d, cap_d);
                continue;
            }
            any_legal = true;
            let child_result = self.static_mate_or(board, remaining - 1, budget);
            board.undo_move(*d, cap_d);
            match child_result {
                StaticMateResult::Checkmate(child_proof) => {
                    // 防御手は玉方の手なので攻方持ち駒は不変 → 調整不要
                    for k in 0..HAND_KINDS {
                        and_proof[k] = and_proof[k].max(child_proof[k]);
                    }
                }
                StaticMateResult::NoCheckmate => {
                    // 子が確定的に不詰 → AND ノードの不詰を TT に記録
                    // NoCheckmate は「予算切れ」と区別された確定結果のため，
                    // OR 側と異なり budget_before ガードは不要:
                    //   AND は逃れ手を1つ見つけた時点で不詰が確定し，
                    //   OR のように全子を巡回して結果を合成する必要がない
                    self.table.store(pos_key, att_hand, INF, 0, rem16, pos_key);
                    return StaticMateResult::NoCheckmate;
                }
                StaticMateResult::Exhausted => {
                    // 予算切れ → 結論なしで即座にリターン
                    return StaticMateResult::Exhausted;
                }
            }
        }
        // 全応手に対して詰み → 証明駒を現在の持ち駒でクリップ
        //
        // defenses が非空にもかかわらず全手が is_in_check でスキップされた場合も
        // ここに到達する．その場合 and_proof = [0; HAND_KINDS] のまま Checkmate
        // を返す(合法応手なし = 詰み)．generate_defense_moves は合法手のみを
        // 返すべきなので，これは movegen のバグを示唆する．
        debug_assert!(
            defenses.is_empty() || any_legal,
            "static_mate_and: 全防御手が非合法 — generate_defense_moves のバグの可能性 (pos_key={:#x})",
            pos_key,
        );
        for k in 0..HAND_KINDS {
            and_proof[k] = and_proof[k].min(att_hand[k]);
        }
        self.table.store(pos_key, and_proof, 0, INF, REMAINING_INFINITE, pos_key);
        StaticMateResult::Checkmate(and_proof)
    }

    /// 玉方の王手回避手を生成する(合い効かずを除外)．
    ///
    /// 全合法手生成の代わりに回避手のみを直接生成する:
    /// 1. 玉の移動(攻め方に利かれていないマスへ)
    /// 2. 王手駒の捕獲(ピンされていない駒による)
    /// 3. 合い駒(飛び駒の王手の場合，間のマスへ移動または打つ)
    ///
    /// 合い効かず(futile interposition)もフィルタする．
    fn generate_defense_moves(
        &mut self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        self.generate_defense_moves_inner(board, false)
    }

    /// 王手回避手の内部実装．
    ///
    /// `early_exit == true` の場合，最初の合法手が見つかった時点で即座にリターンする．
    /// `has_any_defense` と `generate_defense_moves` の両方がこの関数を共有し，
    /// ロジックの重複を排除する．
    fn generate_defense_moves_inner(
        &mut self,
        board: &mut Board,
        early_exit: bool,
    ) -> ArrayVec<Move, MAX_MOVES> {
        self.chain_bb_cache = Bitboard::EMPTY;
        let defender = board.turn;
        let attacker = defender.opponent();

        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => {
                let legal = movegen::generate_legal_moves(board);
                let mut out = ArrayVec::new();
                for m in legal {
                    push_move(&mut out, m);
                    if early_exit {
                        return out;
                    }
                }
                return out;
            }
        };

        // 王手している駒を特定
        let checkers = board.compute_checkers_at(king_sq, attacker);
        if checkers.is_empty() {
            // 王手されていない(通常ありえないが安全策)
            let legal = movegen::generate_legal_moves(board);
            let mut out = ArrayVec::new();
            for m in legal {
                push_move(&mut out, m);
                if early_exit {
                    return out;
                }
            }
            return out;
        }

        let all_occ = board.all_occupied();
        let our_occ = board.occupied[defender.index()];
        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

        // --- 1. 玉の移動 ---
        let king_attacks = attack::step_attacks(defender, PieceType::King, king_sq);
        let king_targets = king_attacks & !our_occ;
        for to in king_targets {
            let captured_piece = board.squares[to.index()];
            let captured_raw = captured_piece.0;
            let m = Move::new_move(king_sq, to, false, captured_raw, PieceType::King as u8);
            // 移動先が安全か(攻め方に利かれていないか)チェック
            let captured = board.do_move(m);
            let safe = !board.is_in_check(defender);
            board.undo_move(m, captured);
            if safe {
                push_move(&mut moves, m);
                if early_exit {
                    return moves;
                }
            }
        }

        // 両王手の場合，玉移動のみ可能
        if checkers.count() > 1 {
            return moves;
        }

        // 単一の王手駒
        let checker_sq = checkers.lsb().unwrap();

        // --- 2. 王手駒の捕獲(玉以外の駒で) ---
        self.generate_capture_checker(
            board, &mut moves, checker_sq, king_sq, defender, all_occ, our_occ,
        );
        if early_exit && !moves.is_empty() {
            return moves;
        }

        // --- 3. 合い駒(飛び駒の王手の場合) ---
        let sliding_checker = self.find_sliding_checker(board, king_sq, attacker);
        if sliding_checker.is_some() {
            let between = attack::between_bb(checker_sq, king_sq);
            if between.is_not_empty() {
                // 合い効かず・チェーンマスを計算
                let (futile, chain) = self.compute_futile_and_chain_squares(
                    board, &between, king_sq, checker_sq, defender, attacker,
                );
                self.chain_bb_cache = chain;
                // 間のマスへの合い駒
                self.generate_interpositions(
                    board, &mut moves, &between, &futile, &chain, king_sq, defender, all_occ, our_occ,
                );
            }
        }

        moves
    }

    /// 王手駒を玉以外の駒で捕獲する手を生成する．
    fn generate_capture_checker(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        checker_sq: Square,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
    ) {
        // 逆引き利き計算: checker_sq を攻撃できる自駒のビットボードを直接求める．
        // is_attacked_by と同じ逆射パターンで，全自駒イテレーション(~16駒)を
        // 駒種ごとのビットボード AND(~10回)に置き換え，該当駒(2-4駒)のみ処理する．
        let attacker = defender.opponent();
        let d = defender.index();
        let mut can_capture = Bitboard::EMPTY;

        // 歩: checker_sq から「相手視点の歩の利き方向」に自歩があるか
        can_capture |= attack::step_attacks(attacker, PieceType::Pawn, checker_sq)
            & board.piece_bb[d][PieceType::Pawn as usize];
        // 桂
        can_capture |= attack::step_attacks(attacker, PieceType::Knight, checker_sq)
            & board.piece_bb[d][PieceType::Knight as usize];
        // 銀
        can_capture |= attack::step_attacks(attacker, PieceType::Silver, checker_sq)
            & board.piece_bb[d][PieceType::Silver as usize];
        // 金 + 成駒
        let gold_reach = attack::step_attacks(attacker, PieceType::Gold, checker_sq);
        can_capture |= gold_reach
            & (board.piece_bb[d][PieceType::Gold as usize]
                | board.piece_bb[d][PieceType::ProPawn as usize]
                | board.piece_bb[d][PieceType::ProLance as usize]
                | board.piece_bb[d][PieceType::ProKnight as usize]
                | board.piece_bb[d][PieceType::ProSilver as usize]);
        // 王・馬・龍(ステップ部分): 玉は後で除外
        let king_reach = attack::step_attacks(attacker, PieceType::King, checker_sq);
        can_capture |= king_reach
            & (board.piece_bb[d][PieceType::Horse as usize]
                | board.piece_bb[d][PieceType::Dragon as usize]);
        // 香
        can_capture |= attack::lance_attacks(attacker, checker_sq, all_occ)
            & board.piece_bb[d][PieceType::Lance as usize];
        // 角・馬
        can_capture |= attack::bishop_attacks(checker_sq, all_occ)
            & (board.piece_bb[d][PieceType::Bishop as usize]
                | board.piece_bb[d][PieceType::Horse as usize]);
        // 飛・龍
        can_capture |= attack::rook_attacks(checker_sq, all_occ)
            & (board.piece_bb[d][PieceType::Rook as usize]
                | board.piece_bb[d][PieceType::Dragon as usize]);

        // 玉を除外(玉による取りは呼び出し元で処理済み)
        can_capture.clear(king_sq);

        let captured_raw = board.squares[checker_sq.index()].0;

        while can_capture.is_not_empty() {
            let from = can_capture.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let in_promo_zone =
                checker_sq.is_promotion_zone(defender) || from.is_promotion_zone(defender);

            if pt.can_promote() && in_promo_zone {
                let m = Move::new_move(from, checker_sq, true, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    push_move(moves, m);
                }
                if !movegen::must_promote(defender, pt, checker_sq) {
                    let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                }
            } else if !movegen::must_promote(defender, pt, checker_sq) {
                let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    push_move(moves, m);
                }
            }
        }
    }

    /// 合い効かずマスとチェーンマス(駒打ち用)を計算する．
    ///
    /// 戻り値 `(futile, chain)`:
    /// - `futile`: 完全に無駄なマス(駒打ちスキップ)
    /// - `chain`: 中合いチェーン防御マス(代表駒のみドロップ生成)
    ///
    /// 各マスの判定基準:
    /// - 守備側(玉を除く)がそのマスに利いていれば，ひもがついているため無駄合いではない
    /// - 玉の隣接マスで，攻め方がチェッカー以外から利かせていなければ，
    ///   玉が取り返せるため無駄合いではない
    /// - それ以外は個別には無駄合い(チェッカーに取られて再び同筋の王手になる)
    ///
    /// ただし中合いチェーン防御を考慮する:
    /// マス X が上記基準で無駄合いでも，X と玉の間に非無駄合いマスが存在すれば，
    /// X への駒打ちは「捨て合い→チェッカー前進→…→非無駄合いマスでブロック」
    /// のチェーン防御の起点となり得る．このようなマスは `chain` に分類し，
    /// 最弱の駒(歩)のみをドロップ候補とする．
    ///
    /// 歩を代表とする根拠: 中合いで打った駒はチェッカーに取られるため，
    /// 攻め方に渡す駒が弱いほど守備側に有利(包含関係)．
    /// 歩の中合いで不詰みなら他の駒種でも不詰み，歩で詰みなら他も詰み．
    fn compute_futile_and_chain_squares(
        &self,
        board: &Board,
        between: &Bitboard,
        king_sq: Square,
        checker_sq: Square,
        defender: Color,
        attacker: Color,
    ) -> (Bitboard, Bitboard) {
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let mut futile = Bitboard::EMPTY;

        for sq in *between {
            // 守備側(玉除く)がひもをつけている → 取り返せるので無駄合いではない
            if board.is_attacked_by_excluding(sq, defender, true, None) {
                continue;
            }
            if king_step.contains(sq) {
                // 攻め方の他駒が利いていない → 玉が取り返せる → 無駄合いではない
                if !board.is_attacked_by_excluding(sq, attacker, false, Some(checker_sq)) {
                    continue;
                }
                // 飛び駒が取り進んだ後に玉の逃げ道があれば無駄合いではない
                if self.king_can_escape_after_slider_capture(
                    board, sq, checker_sq, king_sq, &king_step, defender, attacker,
                ) {
                    continue;
                }
            }
            futile.set(sq);
        }

        // 中合いチェーン伝搬: 玉側からチェッカー方向へ走査し，
        // 非無駄合いマスより遠い(チェッカー側の)無駄合いマスを chain に移す．
        let mut chain = Bitboard::EMPTY;
        if futile.is_not_empty() && futile != *between {
            let dc = (king_sq.col() as i32 - checker_sq.col() as i32).signum();
            let dr = (king_sq.row() as i32 - checker_sq.row() as i32).signum();
            let step_c = -dc;
            let step_r = -dr;
            let mut c = king_sq.col() as i32 + step_c;
            let mut r = king_sq.row() as i32 + step_r;
            let mut has_non_futile = false;
            while c >= 0 && c < 9 && r >= 0 && r < 9 {
                let sq = Square::new(c as u8, r as u8);
                if !between.contains(sq) {
                    break;
                }
                if !futile.contains(sq) {
                    has_non_futile = true;
                } else if has_non_futile {
                    futile.clear(sq);
                    chain.set(sq);
                }
                c += step_c;
                r += step_r;
            }
        }

        (futile, chain)
    }

    /// 飛び駒が玉隣接マスへ取り進んだ後に玉の逃げ道があるか判定する．
    ///
    /// 飛び駒が `capture_sq` へ移動した場合を想定し，
    /// 玉の全ステップ先が安全かどうかを検査する．
    /// 1つでも安全なマスがあれば `true`(= 無駄合いではない)を返す．
    ///
    /// NOTE: 玉が `king_sq` から離れることで新たに発生する素抜き攻撃
    /// (discovered attack) は考慮していない．このため判定は保守的
    /// (futile 判定を甘くする方向)に寄る．
    fn king_can_escape_after_slider_capture(
        &self,
        board: &Board,
        capture_sq: Square,
        checker_sq: Square,
        _king_sq: Square,
        king_step: &Bitboard,
        defender: Color,
        attacker: Color,
    ) -> bool {
        let our_occ = board.occupied[defender.index()];

        // 飛び駒移動後の占有: checker_sq が空き，capture_sq に飛び駒が入る
        let mut occ = board.all_occupied();
        occ.clear(checker_sq);
        occ.set(capture_sq);

        // 飛び駒の capture_sq からの利きを計算
        let checker_piece = board.squares[checker_sq.index()];
        let checker_pt = match checker_piece.piece_type() {
            Some(pt) => pt,
            None => return false,
        };
        let slider_attacks = match checker_pt {
            PieceType::Rook => attack::rook_attacks(capture_sq, occ),
            PieceType::Dragon => {
                attack::rook_attacks(capture_sq, occ)
                    | attack::step_attacks(attacker, PieceType::King, capture_sq)
            }
            PieceType::Bishop => attack::bishop_attacks(capture_sq, occ),
            PieceType::Horse => {
                attack::bishop_attacks(capture_sq, occ)
                    | attack::step_attacks(attacker, PieceType::King, capture_sq)
            }
            PieceType::Lance => attack::lance_attacks(attacker, capture_sq, occ),
            _ => return false,
        };

        // 玉の逃げ先候補: 自駒のないマス(capture_sq の飛び駒は敵駒だが防御済み)
        let escape_candidates = *king_step & !our_occ;

        for esc in escape_candidates {
            // capture_sq は飛び駒が守られているため玉で取れない(既に確認済み)
            if esc == capture_sq {
                continue;
            }
            // 飛び駒の新位置から利かれているか
            if slider_attacks.contains(esc) {
                continue;
            }
            // 他の攻め駒(飛び駒の旧位置を除外)から利かれているか
            if board.is_attacked_by_excluding(esc, attacker, false, Some(checker_sq)) {
                continue;
            }
            // 安全な逃げ先がある → 無駄合いではない
            return true;
        }

        // 逃げ道なし → 詰み → 無駄合い
        false
    }

    /// 間のマスへの合い駒手を生成する(移動・打ち)．
    ///
    /// `futile`: 完全に無駄なマス(駒打ちスキップ)．
    /// `chain`: チェーン防御マス(代表駒のみドロップ生成)．
    ///
    /// chain マスでは包含関係を利用し，3カテゴリの代表駒のみ生成する:
    /// - 前方利き系: 歩(代表) ⊇ {歩,香,銀,金,飛}
    /// - 斜め利き系: 角(代表) — 前方に利かないため歩とは異なる包含
    /// - 跳躍系: 桂(代表) — 打てない段があり独立
    ///
    /// 中合いで打った駒はチェッカーに取られるため，攻め方に渡す駒が弱いほど
    /// 守備側に有利(包含関係)．各カテゴリ内では最弱の駒が代表となる．
    fn generate_interpositions(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        between: &Bitboard,
        futile: &Bitboard,
        chain: &Bitboard,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
    ) {
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let attacker = defender.opponent();
        // futile | chain = 駒移動の無駄合いフィルタ対象
        let futile_or_chain = *futile | *chain;

        let d = defender.index();

        for to in *between {
            // --- 駒移動による合い駒 ---
            // 逆引き利き計算: to マスに到達できる自駒のビットボードを直接求める．
            let mut can_interpose = Bitboard::EMPTY;
            can_interpose |= attack::step_attacks(attacker, PieceType::Pawn, to)
                & board.piece_bb[d][PieceType::Pawn as usize];
            can_interpose |= attack::step_attacks(attacker, PieceType::Knight, to)
                & board.piece_bb[d][PieceType::Knight as usize];
            can_interpose |= attack::step_attacks(attacker, PieceType::Silver, to)
                & board.piece_bb[d][PieceType::Silver as usize];
            let gold_reach = attack::step_attacks(attacker, PieceType::Gold, to);
            can_interpose |= gold_reach
                & (board.piece_bb[d][PieceType::Gold as usize]
                    | board.piece_bb[d][PieceType::ProPawn as usize]
                    | board.piece_bb[d][PieceType::ProLance as usize]
                    | board.piece_bb[d][PieceType::ProKnight as usize]
                    | board.piece_bb[d][PieceType::ProSilver as usize]);
            let king_reach = attack::step_attacks(attacker, PieceType::King, to);
            can_interpose |= king_reach
                & (board.piece_bb[d][PieceType::Horse as usize]
                    | board.piece_bb[d][PieceType::Dragon as usize]);
            can_interpose |= attack::lance_attacks(attacker, to, all_occ)
                & board.piece_bb[d][PieceType::Lance as usize];
            can_interpose |= attack::bishop_attacks(to, all_occ)
                & (board.piece_bb[d][PieceType::Bishop as usize]
                    | board.piece_bb[d][PieceType::Horse as usize]);
            can_interpose |= attack::rook_attacks(to, all_occ)
                & (board.piece_bb[d][PieceType::Rook as usize]
                    | board.piece_bb[d][PieceType::Dragon as usize]);
            // 玉は合駒に使えない
            can_interpose.clear(king_sq);

            while can_interpose.is_not_empty() {
                let from = can_interpose.pop_lsb();
                let piece = board.squares[from.index()];
                let pt = piece.piece_type().unwrap();

                // 駒移動による合い駒の無駄合いフィルタ:
                // futile/chain マスへの移動でも，以下の場合は無駄合いではない:
                // (a) 移動後の駒にひもがついている(from を除いた守備側の利き)
                // (b) 移動元が玉の隣接マスで，空いた後に攻め方から利かれず
                //     玉の逃げ道が新たに生まれる
                if futile_or_chain.contains(to) {
                    let has_support = board.is_attacked_by_excluding(
                        to, defender, true, Some(from),
                    );
                    let opens_escape = king_step.contains(from)
                        && !board.is_attacked_by_excluding(from, attacker, false, None);
                    if !has_support && !opens_escape {
                        continue;
                    }
                }

                let captured_raw = board.squares[to.index()].0;
                let in_promo_zone =
                    to.is_promotion_zone(defender) || from.is_promotion_zone(defender);

                if pt.can_promote() && in_promo_zone {
                    let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                    if !movegen::must_promote(defender, pt, to) {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_evasion_legal(board, m, defender) {
                            push_move(moves, m);
                        }
                    }
                } else if !movegen::must_promote(defender, pt, to) {
                    let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                }
            }

            // --- 駒打ちによる合い駒 ---
            if futile.contains(to) {
                // 完全無駄合い: スキップ
                continue;
            }
            let is_chain = chain.contains(to);
            if is_chain {
                // chain マス: 3カテゴリの代表駒のみ生成
                self.generate_chain_drops(board, moves, to, defender);
            } else {
                // 通常マス: 全駒種を弱い駒から生成(歩→香→桂→銀→金→角→飛)．
                // 弱い合駒を先に証明すると，攻め方が合駒を取った後の
                // 証明パスが TT に蓄積され，強い合駒探索時に同じ詰み筋を
                // TT ヒットで援用できる(攻め方の持ち駒が増えるため)．
                for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
                    if board.hand[defender.index()][hand_idx] == 0 {
                        continue;
                    }
                    if movegen::must_promote(defender, pt, to) {
                        continue;
                    }
                    if pt == PieceType::Pawn {
                        let our_pawns =
                            board.piece_bb[defender.index()][PieceType::Pawn as usize];
                        let col = to.col();
                        if (our_pawns & Bitboard::file_mask(col)).is_not_empty() {
                            continue;
                        }
                    }
                    let m = Move::new_drop(to, pt);
                    if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                        continue;
                    }
                    push_move(moves, m);
                }
            }
        }
    }

    /// チェーンマスへの代表駒ドロップを生成する．
    ///
    /// 3カテゴリの代表駒を試す:
    /// 1. 前方利き系: 歩→香→銀→金→飛(最弱の合法駒1つ)
    /// 2. 斜め利き系: 角
    /// 3. 跳躍系: 桂
    fn generate_chain_drops(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        to: Square,
        defender: Color,
    ) {
        let di = defender.index();

        // カテゴリ1: 前方利き系(歩,香,銀,金,飛) — 最弱の合法駒1つ
        const FORWARD_PIECES: [(usize, PieceType); 5] = [
            (0, PieceType::Pawn),   // hand_idx=0
            (1, PieceType::Lance),  // hand_idx=1
            (3, PieceType::Silver), // hand_idx=3
            (4, PieceType::Gold),   // hand_idx=4
            (6, PieceType::Rook),   // hand_idx=6
        ];
        for &(hand_idx, pt) in &FORWARD_PIECES {
            if board.hand[di][hand_idx] == 0 {
                continue;
            }
            if movegen::must_promote(defender, pt, to) {
                continue;
            }
            if pt == PieceType::Pawn {
                let our_pawns = board.piece_bb[di][PieceType::Pawn as usize];
                let col = to.col();
                if (our_pawns & Bitboard::file_mask(col)).is_not_empty() {
                    continue; // 二歩
                }
                let m = Move::new_drop(to, pt);
                if movegen::is_pawn_drop_mate(board, m) {
                    continue; // 打ち歩詰め
                }
                push_move(moves, m);
            } else {
                push_move(moves, Move::new_drop(to, pt));
            }
            break; // カテゴリ内最弱で代表
        }

        // カテゴリ2: 角
        let bishop_idx = 5; // HAND_PIECES[5] = Bishop
        if board.hand[di][bishop_idx] > 0
            && !movegen::must_promote(defender, PieceType::Bishop, to)
        {
            push_move(moves, Move::new_drop(to, PieceType::Bishop));
        }

        // カテゴリ3: 桂
        let knight_idx = 2; // HAND_PIECES[2] = Knight
        if board.hand[di][knight_idx] > 0
            && !movegen::must_promote(defender, PieceType::Knight, to)
        {
            push_move(moves, Move::new_drop(to, PieceType::Knight));
        }
    }

    /// 回避手の合法性チェック(ピンの確認)．
    #[inline]
    fn is_evasion_legal(&self, board: &mut Board, m: Move, defender: Color) -> bool {
        let captured = board.do_move(m);
        let in_check = board.is_in_check(defender);
        board.undo_move(m, captured);
        !in_check
    }

    /// 飛び駒で王手している駒のマスを返す(単一の場合のみ)．
    ///
    /// 飛び駒が複数(両王手)の場合は None を返す(合い駒不可のため)．
    /// 飛び駒の王手がない場合も None を返す．
    fn find_sliding_checker(
        &self,
        board: &Board,
        king_sq: Square,
        attacker: Color,
    ) -> Option<Square> {
        let occ = board.all_occupied();
        let att = attacker.index();

        let mut checkers = attack::rook_attacks(king_sq, occ)
            & (board.piece_bb[att][PieceType::Rook as usize]
                | board.piece_bb[att][PieceType::Dragon as usize]);
        checkers = checkers
            | (attack::bishop_attacks(king_sq, occ)
                & (board.piece_bb[att][PieceType::Bishop as usize]
                    | board.piece_bb[att][PieceType::Horse as usize]));
        // 香は防御側(玉方)の前方に利く:
        // lance_attacks(defender, king_sq, occ) で玉の前方レイを取得
        let defender = attacker.opponent();
        checkers = checkers
            | (attack::lance_attacks(defender, king_sq, occ)
                & board.piece_bb[att][PieceType::Lance as usize]);

        // 単一の飛び駒のみ対象
        if checkers.count() == 1 {
            checkers.lsb()
        } else {
            None
        }
    }

    /// 攻め方の王手になる手を生成する．
    ///
    /// 最適化: 玉方の玉に王手がかかる手のみを直接生成する．
    /// 全合法手を生成してからフィルタする方式と比べ，生成候補を大幅に削減する．
    fn generate_check_moves(
        &self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        let us = board.turn;
        let them = us.opponent();
        let has_own_king = board.king_square(us).is_some();

        let king_sq = match board.king_square(them) {
            Some(sq) => sq,
            None => return ArrayVec::new(),
        };

        let our_occ = board.occupied[us.index()];
        let all_occ = board.all_occupied();
        let empty = !all_occ;

        // 各駒種について「このマスに置くと玉に王手がかかる」ターゲットを事前計算
        // step_attacks(them, pt, king_sq) は「玉から見た逆利き」= 王手元になれるマス

        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

        // --- 1. 駒打ち: ターゲットマスのみに打つ ---
        for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            if board.hand[us.index()][hand_idx] == 0 {
                continue;
            }

            // この駒種で王手になるマスを計算
            let check_targets = match pt {
                PieceType::Lance => attack::lance_attacks(them, king_sq, all_occ),
                PieceType::Bishop => attack::bishop_attacks(king_sq, all_occ),
                PieceType::Rook => attack::rook_attacks(king_sq, all_occ),
                _ => attack::step_attacks(them, pt, king_sq),
            };
            let mut drop_targets = check_targets & empty;

            // 二歩チェック: 歩が存在する筋を一括マスクして除外
            if pt == PieceType::Pawn {
                let our_pawns = board.piece_bb[us.index()][PieceType::Pawn as usize];
                drop_targets &= !our_pawns.occupied_files();
            }

            // 行き所のない駒の制限
            match pt {
                PieceType::Pawn | PieceType::Lance => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0),
                        Color::White => Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                PieceType::Knight => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0) | Bitboard::rank_mask(1),
                        Color::White => Bitboard::rank_mask(7) | Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                _ => {}
            }

            for to in drop_targets {
                let m = Move::new_drop(to, pt);
                // 打ち歩詰めチェック
                if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                    continue;
                }
                // 駒打ちは自玉への王手放置にならない(片玉でも両玉でも)
                push_move(&mut moves, m);
            }
        }

        // --- 2. 盤上の駒の移動 ---
        // 直接王手: 移動先から玉に利きがある手
        // 開き王手: 駒が移動することで背後のスライド駒から玉に利きが通る手

        // 開き王手の候補を事前計算:
        // 玉からのレイ上にいる自駒で，その間に他の駒がない場合，
        // そこから移動すると開き王手になりうる
        let discoverers = board.compute_discoverers(us, king_sq);

        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks = attack::piece_attacks(us, pt, from, all_occ);
            let targets = attacks & !our_occ;

            let is_discoverer = discoverers.contains(from);

            for to in targets {
                let captured_piece = board.squares[to.index()];
                let captured_raw = captured_piece.0;
                let in_promo_zone = to.is_promotion_zone(us) || from.is_promotion_zone(us);

                // 開き王手の判定: to が from→king_sq のライン上にある場合，
                // 移動後も飛び駒の利きを遮断するため開き王手にならない
                let gives_discovered = is_discoverer
                    && !attack::line_through(from, king_sq).contains(to);

                // 成り先の駒種での王手チェック
                if pt.can_promote() && in_promo_zone {
                    let promoted_pt = pt.promoted().unwrap();
                    let gives_direct = self.attacks_square(us, promoted_pt, to, all_occ, king_sq);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }

                    // 不成
                    if !movegen::must_promote(us, pt, to) {
                        let gives_direct =
                            self.attacks_square(us, pt, to, all_occ, king_sq);
                        if gives_direct || gives_discovered {
                            let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                            if self.is_legal_quick(board, m, has_own_king) {
                                push_move(&mut moves, m);
                            }
                        }
                    }
                } else if !movegen::must_promote(us, pt, to) {
                    let gives_direct = self.attacks_square(us, pt, to, all_occ, king_sq);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }
                }
            }
        }

        // 手順序: 成+取 > 成 > 取 > その他，同カテゴリ内は玉との距離でタイブレーク
        // 近接王手を優先し，詰みに至る手を早期発見する．
        // 距離はチェビシェフ距離(0-8)を使用し，カテゴリ(0-3) * 16 + 距離 で
        // 単一キーにエンコードする．
        let king_col = king_sq.col() as i8;
        let king_row = king_sq.row() as i8;
        moves.sort_unstable_by_key(|m| {
            let promo = m.is_promotion();
            let capture = m.captured_piece_raw() > 0;
            let category: u8 = match (promo, capture) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            let to = m.to_sq();
            let dc = (to.col() as i8 - king_col).unsigned_abs();
            let dr = (to.row() as i8 - king_row).unsigned_abs();
            let dist = dc.max(dr); // チェビシェフ距離(0-8)
            (category as u16) * 16 + dist as u16
        });

        moves
    }

    /// 指定マスに置いた駒が玉のマスに利いているか判定する．
    #[inline]
    fn attacks_square(
        &self,
        color: Color,
        pt: PieceType,
        from: Square,
        occ: Bitboard,
        target: Square,
    ) -> bool {
        attack::piece_attacks(color, pt, from, occ).contains(target)
    }

    /// 開き王手の元になりうる自駒を計算する．
    /// 合法性の簡易チェック(自玉の王手放置のみ)．
    ///
    /// 片玉の場合(自玉なし)は常に合法．
    #[inline]
    fn is_legal_quick(&self, board: &mut Board, m: Move, has_own_king: bool) -> bool {
        if !has_own_king {
            return true;
        }
        let us = board.turn;
        let captured = board.do_move(m);
        let in_check = board.is_in_check(us);
        board.undo_move(m, captured);
        !in_check
    }

    /// PV パス上の OR ノードで未証明の子ノードを追加証明する．
    ///
    /// Df-Pn は OR ノードで1つの子ノードが証明されると他を未探索のまま残す．
    /// PV 抽出で正確な最短詰み手数を計算するため，PV 上の OR ノードで
    /// 未証明の王手を追加証明する．反復的に PV を更新し収束させる．
    ///
    /// # 制限事項
    ///
    /// - 反復回数は2回固定．長手数(29手詰め等)で PV 上の未証明子が多い場合，
    ///   2回の反復では PV が収束せず `CheckmateNoPv` になることがある．
    ///   この場合 `pv_nodes_per_child` を増やしても改善されない．
    /// - `extract_pv_recursive` の深度制限(`depth * 2`)により PV 復元が
    ///   打ち切られるケースでも `CheckmateNoPv` になる(後述)．
    fn complete_or_proofs(&mut self, board: &mut Board) {
        let saved_max = self.max_nodes;
        // 証明完了フェーズのノード予算:
        //   主探索ノード数と pv_nodes_per_child*8 の小さい方を追加予算とする．
        //   ただし短手数の詰将棋 (少ノードで解けた場合) でも PV 復元に
        //   十分なノードを確保するため，最低 pv_nodes_per_child ノードを保証する．
        let mid_nodes = self.nodes_searched;
        let budget_cap = self.pv_nodes_per_child.saturating_mul(8);

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // find_shortest のときは最大 4 回反復:
        //   追加証明で PV が変化 → 新 PV 上の未証明 OR 子がさらにある →
        //   もう 1 回証明 → PV が安定，という収束サイクルを回す．
        // changed == false で早期終了するため，収束済みの場合は追加コストなし．
        self.max_nodes =
            self.nodes_searched.saturating_add(
                mid_nodes.min(budget_cap).max(self.pv_nodes_per_child),
            );

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // 2回固定: 1回目で新たに証明された子が PV を短縮する可能性があるため
        // 2回目を実行する．changed == false で早期終了するため，
        // 収束済みの場合は追加コストなし．
        for _ in 0..2 {
            if self.is_timed_out()
                || self.nodes_searched >= self.max_nodes
            {
                break;
            }
            let pv = self.extract_pv_limited(board, 100_000);
            if pv.is_empty() {
                break;
            }
            let changed =
                self.complete_pv_or_nodes(board, &pv);
            if !changed {
                break;
            }
        }

        self.max_nodes = saved_max;
    }

    /// PV 上の各 OR ノードで未証明子ノードの追加証明を試みる．
    ///
    /// クローンした盤面で PV を辿り，各 OR ノードで未証明の王手を追加証明する．
    /// 返り値: 新たに証明された子ノードがあれば true．
    fn complete_pv_or_nodes(&mut self, board: &mut Board, pv: &[Move]) -> bool {
        let mut board_clone = board.clone();
        let mut any_changed = false;

        for (i, pv_move) in pv.iter().enumerate() {
            let or_node = i % 2 == 0;
            let ply = i as u32;

            if or_node {
                let moves = self.generate_check_moves(&mut board_clone);
                for m in &moves {
                    if self.nodes_searched >= self.max_nodes {
                        break;
                    }
                    let captured = board_clone.do_move(*m);
                    let (cpn, cdn) = self.look_up_board(&board_clone);

                    if cpn != 0 && cdn != 0 {
                        let saved = self.max_nodes;
                        // 1子あたり pv_nodes_per_child ノード上限．
                        // PV 沿いの各未証明子に対する追加証明の予算．
                        self.max_nodes = self
                            .nodes_searched
                            .saturating_add(self.pv_nodes_per_child)
                            .min(saved);
                        self.mid(&mut board_clone, INF - 1, INF - 1, ply + 1, false);
                        self.max_nodes = saved;

                        if self.look_up_board(&board_clone).0 == 0 {
                            any_changed = true;
                        }
                    }

                    board_clone.undo_move(*m, captured);
                }
            }

            // PV に沿って盤面を進める(前進専用: undo_move は不要)
            let _captured = board_clone.do_move(*pv_move);
        }

        any_changed
    }

    /// 訪問数制限付き PV 復元．
    fn extract_pv_limited(&mut self, board: &mut Board, max_visits: u64) -> Vec<Move> {
        let mut board_clone = board.clone();
        let mut visits = 0u64;
        self.extract_pv_recursive_inner(
            &mut board_clone,
            true,
            &mut FxHashSet::default(),
            0,
            false,
            &mut visits,
            max_visits,
        )
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    /// ループ検出にはフルハッシュ，TT 参照には位置キー＋持ち駒を使用する．
    ///
    /// # 深度制限
    ///
    /// 再帰深度は `self.depth * 2` で打ち切る．OR/AND が交互に呼ばれるため，
    /// depth=31 なら最大62手の PV を復元できる．理論上，AND ノードで
    /// 複数の応手候補を評価する際に ply が詰み手数の2倍を超えうるが，
    /// TT に証明済みエントリがあれば即座に返るため，実用上は十分な深さである．
    ///
    /// ただし `complete_pv_or_nodes` で新たに証明された子が PV を変化させた場合，
    /// AND ノードの応手評価で ply が急増し，深度制限に達して空リストが返ることがある．
    /// この場合 `pv_nodes_per_child` を増やしても改善されず，`depth` の増加が必要．
    fn extract_pv_recursive(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
    ) -> Vec<Move> {
        let mut visits = 0u64;
        self.extract_pv_recursive_inner(board, or_node, visited, ply, false, &mut visits, u64::MAX)
    }

    /// `extract_pv_recursive` の内部実装．
    ///
    /// `diag` が true の場合，各plyでの候補手・sub_pv長・選択理由を
    /// 標準エラーに出力する．
    /// `visits` は TT 参照回数を計測し，`max_visits` を超えると空リストを返す．
    fn extract_pv_recursive_inner(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
        diag: bool,
        visits: &mut u64,
        max_visits: u64,
    ) -> Vec<Move> {
        // 訪問数制限チェック
        *visits += 1;
        if *visits > max_visits {
            return Vec::new();
        }

        // スタックオーバーフロー防止: 探索手数の2倍を再帰深度の上限とする
        if ply >= self.depth.saturating_mul(2) {
            if diag {
                eprintln!("[PV diag] ply={} depth_limit reached (max={})", ply, self.depth.saturating_mul(2));
            }
            return Vec::new();
        }
        let full_hash = board.hash;

        // ループ検出(フルハッシュ)
        if visited.contains(&full_hash) {
            if diag {
                eprintln!("[PV diag] ply={} loop detected hash={:#x}", ply, full_hash);
            }
            return Vec::new();
        }

        let (node_pn, _node_dn) = self.look_up_board(board);

        if or_node {
            if node_pn != 0 {
                if diag {
                    eprintln!("[PV diag] ply={} OR node unproven pn={}", ply, node_pn);
                }
                return Vec::new();
            }

            let moves = self.generate_check_moves(board);
            if moves.is_empty() {
                if diag {
                    eprintln!("[PV diag] ply={} OR node no check moves", ply);
                }
                return Vec::new();
            }

            if diag {
                eprintln!("[PV diag] ply={} OR node, {} check moves", ply, moves.len());
            }

            let mut best_pv: Option<Vec<Move>> = None;

            for m in &moves {
                if *visits > max_visits {
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, false, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // OR ノードの PV は奇数長でなければならない
                    // (攻め方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 0 && !sub_pv.is_empty() {
                        if diag {
                            eprintln!(
                                "[PV diag] ply={} OR skip {} (even len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => total_len < prev.len(),
                    };

                    if diag {
                        eprintln!(
                            "[PV diag] ply={} OR candidate {} len={} better={}{}",
                            ply, m.to_usi(), total_len, is_better,
                            if let Some(prev) = &best_pv {
                                format!(" (prev_best={})", prev.len())
                            } else {
                                String::new()
                            }
                        );
                    }

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                    }
                } else if diag {
                    eprintln!(
                        "[PV diag] ply={} OR child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} OR CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} OR no valid PV found", ply);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 呼び出し側の OR ノードで child_pn == 0 を
            // 確認した後のみ再帰するため，この局面は証明済みのはず．
            let moves = self.generate_defense_moves(board);
            if moves.is_empty() {
                if diag {
                    eprintln!("[PV diag] ply={} AND node no defense (checkmate)", ply);
                }
                return Vec::new();
            }

            if diag {
                eprintln!("[PV diag] ply={} AND node, {} defense moves", ply, moves.len());
            }

            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;
            let mut best_is_drop = false;

            for m in &moves {
                if *visits > max_visits {
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, true, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // AND ノードの PV は偶数長でなければならない
                    // (玉方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 1 {
                        if diag {
                            eprintln!(
                                "[PV diag] ply={} AND skip {} (odd len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    let is_capture = m.captured_piece_raw() > 0;
                    let is_drop = m.is_drop();
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => {
                            if total_len > prev.len() {
                                true
                            } else if total_len == prev.len()
                                && is_capture
                                && !best_is_capture
                            {
                                true
                            } else if total_len == prev.len()
                                && is_drop
                                && !best_is_drop
                                && is_capture == best_is_capture
                            {
                                // 同率 & 駒取り状況も同じ場合，
                                // 合駒(打ち駒)を優先する．
                                // 打ち駒のサブツリーは探索で重点的に証明
                                // されやすく，sub_pv が正確な傾向がある．
                                true
                            } else {
                                false
                            }
                        }
                    };

                    if diag {
                        eprintln!(
                            "[PV diag] ply={} AND candidate {} len={} capture={} drop={} better={}{}",
                            ply, m.to_usi(), total_len, is_capture, is_drop, is_better,
                            if let Some(prev) = &best_pv {
                                format!(" (prev_best={} prev_cap={} prev_drop={})", prev.len(), best_is_capture, best_is_drop)
                            } else {
                                String::new()
                            }
                        );
                    }

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                        best_is_capture = is_capture;
                        best_is_drop = is_drop;
                    }
                } else if diag {
                    eprintln!(
                        "[PV diag] ply={} AND child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} AND CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} AND no valid PV found", ply);
                }
            }

            best_pv.unwrap_or_default()
        }
    }

    /// 探索ノード数を返す．
    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    /// IDS-dfpn (MID) フォールバック．
    ///
    /// PNS がアリーナ上限に達した場合に呼び出される．
    /// PNS で蓄積された TT エントリ(証明・中間値)を引き継ぎ，
    /// 残りのノード予算で IDS-dfpn を実行する．
    fn mid_fallback(&mut self, board: &mut Board) {
        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        self.diag_root_pk = pk;
        self.diag_root_hand = att_hand;
        let saved_depth = self.depth;
        let mut ids_depth: u32 = 2;
        let total_max_nodes = self.max_nodes;
        // PNS で蓄積された中間エントリ(pn>0, dn>0)を除去し，
        // 証明(pn=0)のみ保持する．
        //
        // 中間エントリを保持すると以下の問題が発生する:
        // 1. HashMap サイズ増大により CPU キャッシュ効率が低下し NPS が半減する
        //    (338K entries → ~126 NPS vs 12K entries → ~194 NPS)．
        // 2. MID の child init で cpn>1/cdn>1 として扱われ，
        //    底辺の簡単な詰みを再発見する機会が失われる．
        //
        // 証明エントリ(pn=0)は prefilter/cross_deduce に直接活用される．
        self.table.retain_proofs_only();

        #[cfg(feature = "tt_diag")]
        eprintln!("[mid_fallback] after retain_proofs_only: TT_pos={} nodes_so_far={} total_budget={}",
            self.table.len(), self.nodes_searched, total_max_nodes);

        // 停滞検出用: 前回の IDS 反復終了時の root pn/dn を保持する．
        // IDS 反復後に root_pn/dn が変化しなかった場合，MID が
        // dn 閾値カスケード縮退により進捗不能と判断する．
        let mut prev_root_pn: u32 = 0;
        let mut prev_root_dn: u32 = 0;

        // 合駒チェーン最適化の IDS 反復間デルタ追跡
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_hits = self.prefilter_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_cross_deduce = self.diag_cross_deduce_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_deferred_act = self.diag_mid_deferred_activations;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_skip = self.diag_prefilter_skip_remaining;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_miss = self.diag_prefilter_miss;

        loop {
            if ids_depth > saved_depth {
                ids_depth = saved_depth;
            }
            self.depth = ids_depth;
            self.path.clear();
            let remaining = ids_depth as u16;
            let (root_pn, _, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            eprintln!("[ids] depth={}/{} root_pn={} nodes={} time={:.1}s",
                ids_depth, saved_depth, root_pn, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn == 0 {
                eprintln!("[ids] root proved, break");
                break;
            }
            let _budget = if ids_depth < saved_depth {
                // 残り IDS ステップ数に応じた予算配分．
                // 線形進行(+2)で残り何ステップあるかを見積もり，
                // 残りノード予算を均等に分配する．
                // 最終深さ(full depth)に最大の予算を残すため，
                // 浅い反復には 1/(remaining_steps+1) を割り当てる．
                let remaining_budget =
                    total_max_nodes.saturating_sub(self.nodes_searched);
                let remaining_steps =
                    ((saved_depth.saturating_sub(ids_depth)) / 2)
                        .max(1) as u64 + 1;
                let b = (remaining_budget / (remaining_steps + 1))
                    .max(1024);
                self.max_nodes = self.nodes_searched.saturating_add(b);
                b
            } else {
                self.max_nodes = total_max_nodes;
                total_max_nodes.saturating_sub(self.nodes_searched)
            };

            #[cfg(feature = "tt_diag")]
            let pre_nodes = self.nodes_searched;
            #[cfg(feature = "tt_diag")]
            let pre_max_ply = self.max_ply;
            // IDS 反復ごとに max_ply をリセットし，各反復の到達深さを追跡する
            self.max_ply = 0;

            {
                let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if root_pn != 0 && root_dn != 0
                    && self.nodes_searched < self.max_nodes
                    && !self.timed_out
                {
                    // IDS iteration ごとに or_effort をリセット．
                    // 前の depth の TT pn が stale になるため，
                    // depth ジャンプ後は全ての子がフレッシュに再評価される必要がある．
                    self.or_effort.clear();
                    eprintln!("[ids] calling MID: depth={} root_pn={} root_dn={} nodes={}",
                        ids_depth, root_pn, root_dn, self.nodes_searched);
                    #[cfg(feature = "profile")]
                    let _mid_wall_start = Instant::now();
                    self.mid(board, INF - 1, INF - 1, 0, true);
                    eprintln!("[ids] MID returned: depth={} nodes={} time={:.1}s",
                        ids_depth, self.nodes_searched, self.start_time.elapsed().as_secs_f64());
                    #[cfg(feature = "profile")]
                    {
                        self.profile_stats.mid_total_ns += _mid_wall_start.elapsed().as_nanos() as u64;
                        self.profile_stats.mid_total_count += 1;
                    }
                }
            }

            #[cfg(feature = "tt_diag")]
            {
                let used = self.nodes_searched - pre_nodes;
                let (pn, dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                let d_prefilter = self.prefilter_hits - prev_prefilter_hits;
                let d_cross = self.diag_cross_deduce_hits - prev_cross_deduce;
                let d_deferred = self.diag_mid_deferred_activations - prev_deferred_act;
                let d_pf_skip = self.diag_prefilter_skip_remaining - prev_prefilter_skip;
                let d_pf_miss = self.diag_prefilter_miss - prev_prefilter_miss;
                let ids_elapsed = self.start_time.elapsed().as_secs_f64();
                eprintln!("[ids_diag] depth={}/{} budget={} used={} TT_pos={} root_pn={} root_dn={} max_ply={} \
                    prefilter_hit={} prefilter_skip_rem={} prefilter_miss={} cross={} act={} \
                    cap_tt={}/{} thr_exits={} term_exits={} in_path={} lb_prov={} lb_thr={} lb_nodes={} \
                    init_and_dis={} single_ch={} node_lim={} time={:.2}s",
                    ids_depth, saved_depth, _budget, used,
                    self.table.len(), pn, dn,
                    self.max_ply,
                    d_prefilter, d_pf_skip, d_pf_miss,
                    d_cross, d_deferred,
                    self.diag_capture_tt_hits, self.diag_capture_tt_calls,
                    self.diag_threshold_exits,
                    self.diag_terminal_exits,
                    self.diag_in_path_exits,
                    self.diag_loop_break_proved,
                    self.diag_loop_break_threshold,
                    self.diag_loop_break_nodes,
                    self.diag_init_and_disproof_exits,
                    self.diag_single_child_exits,
                    self.diag_node_limit_exits,
                    ids_elapsed);
                // ply ヒストグラム出力(非ゼロ ply のみ)
                let mut ply_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_visits[p] > 0 {
                        if !ply_str.is_empty() { ply_str.push_str(", "); }
                        ply_str.push_str(&format!("{}:{}", p, self.diag_ply_visits[p]));
                    }
                }
                if !ply_str.is_empty() {
                    eprintln!("[ids_diag] ply_visits: {}", ply_str);
                }
                // single-child counter (repurposed diag_ply_proofs)
                let mut sc_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_proofs[p] > 0 {
                        if !sc_str.is_empty() { sc_str.push_str(", "); }
                        sc_str.push_str(&format!("{}:{}", p, self.diag_ply_proofs[p]));
                    }
                }
                if !sc_str.is_empty() {
                    eprintln!("[ids_diag] single_child: {}", sc_str);
                }
                // リセット
                self.diag_ply_visits = [0u64; 64];
                self.diag_ply_proofs = [0u64; 64];
                self.diag_capture_tt_calls = 0;
                self.diag_capture_tt_hits = 0;
                self.diag_threshold_exits = 0;
                self.diag_terminal_exits = 0;
                self.diag_loop_break_proved = 0;
                self.diag_loop_break_threshold = 0;
                self.diag_loop_break_nodes = 0;
                self.diag_in_path_exits = 0;
                self.diag_init_and_disproof_exits = 0;
                self.diag_single_child_exits = 0;
                self.diag_node_limit_exits = 0;
                prev_prefilter_hits = self.prefilter_hits;
                prev_cross_deduce = self.diag_cross_deduce_hits;
                prev_deferred_act = self.diag_mid_deferred_activations;
                prev_prefilter_skip = self.diag_prefilter_skip_remaining;
                prev_prefilter_miss = self.diag_prefilter_miss;
            }
            let (root_pn2, root_dn2, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            eprintln!("[ids] after MID: depth={} root_pn={} root_dn={} nodes={} time={:.1}s",
                ids_depth, root_pn2, root_dn2, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn2 == 0 {
                eprintln!("[ids] proved at depth={}, break", ids_depth);
                break;
            }
            // IDS NM 判定: 構造的判定のみ信頼する．
            //
            // 1. NM remaining が REMAINING_INFINITE なら真の不詰として打ち切る．
            //    MID が深さ制限なしに全パスを網羅して NM を確定した場合にのみ
            //    REMAINING_INFINITE が伝搬される．
            // 2. 全王手が再帰的に反証可能なら REMAINING_INFINITE に昇格して打ち切る．
            //
            // depth 制限由来の仮 NM (remaining < REMAINING_INFINITE) は昇格しない．
            // 深い詰みが存在する局面でも浅い IDS 深さでは NM になるのが当然であり，
            // これを真の不詰と判定すると偽陽性が発生する(例: 39手詰め)．
            if root_dn2 == 0 {
                let root_nm_rem = self.table.get_disproof_remaining(pk, &att_hand);
                eprintln!("[ids] NM: depth={} nm_rem={} REMAINING_INFINITE={}",
                    ids_depth, root_nm_rem, REMAINING_INFINITE);
                if root_nm_rem == REMAINING_INFINITE {
                    eprintln!("[ids] NM INFINITE, break");
                    break;
                }
                let checks = self.generate_check_moves(board);
                let refutable = if checks.is_empty() {
                    true
                } else {
                    self.depth_limit_all_checks_refutable(board, &checks)
                };
                eprintln!("[ids] refutable={} checks={}", refutable, checks.len());
                if checks.is_empty() || refutable {
                    eprintln!("[ids] NM promoted to INFINITE, break");
                    // att_hand で保存(TT ヒット率最大化)
                    self.table.store(
                        pk, att_hand, INF, 0, REMAINING_INFINITE, pk,
                    );
                    break;
                }
                // depth 制限由来の仮 NM → IDS を続行してより深い探索で再検証．
            }
            if self.nodes_searched >= total_max_nodes || self.timed_out {
                break;
            }
            if ids_depth >= saved_depth {
                break;
            }

            // 停滞検出: 予算を使い切ったのに root pn/dn が変化していない場合，
            // dn 閾値カスケード縮退により MID が深部に到達できていない．
            let stagnated = root_pn2 > 0
                && root_dn2 > 0
                && root_pn2 == prev_root_pn
                && root_dn2 == prev_root_dn;

            // 時間ベースの反復カットオフ:
            // 浅い IDS 反復が残り時間の 1/4 以上消費した場合，
            // この深さでの NPS が低く後続の深さに時間を回せないため
            // 最大深さへ即座にジャンプする．
            let total_elapsed = self.start_time.elapsed().as_secs_f64();
            let remaining_time = self.timeout.as_secs_f64() - total_elapsed;
            let time_exceeded = ids_depth < saved_depth
                && remaining_time < self.timeout.as_secs_f64() * 0.25;

            // IDS 反復間でのクリーンアップ:
            // 1. 経路依存の反証を除去(ループ由来，異なる深さでは無効)
            // 2. 浅い反復のスラッシング防止エントリと浅い反証を除去
            self.table.remove_path_dependent_disproofs();
            self.table.remove_stale_for_ids();

            if stagnated || time_exceeded {
                ids_depth = saved_depth;
            } else {
                // 深さ進行:
                // saved_depth > 31 の場合: 段階的 IDS (2→4→8→16→32→41)
                //   深い問題では段階的に TT を構築して探索効率を上げる
                // saved_depth <= 31 の場合: 直接ジャンプ (2→4→31)
                //   浅い問題では中間深さの探索が無駄になるため
                let next = ids_depth.saturating_mul(2).max(ids_depth + 2);
                if saved_depth <= 31 && next > 4 && next < saved_depth {
                    ids_depth = saved_depth;
                } else {
                    ids_depth = next.min(saved_depth);
                }
            }

            prev_root_pn = root_pn2;
            prev_root_dn = root_dn2;
        }
        self.depth = saved_depth;
        self.max_nodes = total_max_nodes;
    }

    // ================================================================
    // Best-First Proof Number Search (PNS)
    // ================================================================

    /// Best-First PNS メインループ．
    ///
    /// 明示的な探索木(アリーナ)上で most-proving node を選択・展開し，
    /// pn/dn をルートまでバックアップする．df-pn の閾値制御を必要とせず，
    /// グローバルに最適なノード選択を行う．
    ///
    /// アリーナが `PNS_MAX_ARENA_NODES` に達した場合は探索を打ち切り，
    /// 呼び出し元で MID ベースの探索にフォールバックする．
    fn pns_main(&mut self, board: &mut Board) -> Option<Vec<Move>> {
        let mut arena: Vec<PnsNode> = Vec::with_capacity(
            PNS_MAX_ARENA_NODES.min(1024 * 1024),
        );

        // ルートノード生成
        let pk = position_key(board);
        let fh = board.hash;
        let hand = board.hand[self.attacker.index()];
        arena.push(PnsNode {
            pos_key: pk,
            full_hash: fh,
            hand,
            pn: 1,
            dn: 1,
            parent: u32::MAX,
            move_from_parent: Move(0),
            or_node: true,
            expanded: false,
            children: Vec::new(),
            remaining: self.depth as u16,
            deferred_drops: Vec::new(),
        });

        // 再利用バッファ(ループ内のアロケーション回避)
        let max_path = self.depth as usize + 2;
        let mut path: Vec<u32> = Vec::with_capacity(max_path);
        let mut captures: Vec<Piece> = Vec::with_capacity(max_path);
        let mut ancestors: FxHashSet<u64> =
            FxHashSet::with_capacity_and_hasher(max_path, Default::default());

        // PNS メインループ
        let mut pns_iters: u64 = 0;
        // PNS 収束検出: root_pn が一定反復数改善しなければ打ち切る．
        // PNS はアリーナ飽和後に選択ウォークだけを繰り返し
        // 予算を消費するため，早期打ち切りで MID に予算を回す．
        let mut best_root_pn: u32 = u32::MAX;
        let mut iters_since_improvement: u64 = 0;
        const PNS_STAGNATION_LIMIT: u64 = 500_000;
        loop {
            pns_iters += 1;
            // 終了条件: ルート証明/反証
            if arena[0].pn == 0 || arena[0].dn == 0 {
                break;
            }
            // 終了条件: ノード制限・タイムアウト
            if self.nodes_searched >= self.max_nodes || self.timed_out {
                break;
            }
            // 終了条件: アリーナ満杯
            if arena.len() >= PNS_MAX_ARENA_NODES {
                break;
            }
            // 終了条件: PNS 収束停滞
            if arena[0].pn < best_root_pn {
                best_root_pn = arena[0].pn;
                iters_since_improvement = 0;
            } else {
                iters_since_improvement += 1;
                if iters_since_improvement >= PNS_STAGNATION_LIMIT {
                    #[cfg(feature = "tt_diag")]
                    eprintln!("[pns_diag] stagnation: root_pn={} no improvement for {} iters, stopping",
                        arena[0].pn, PNS_STAGNATION_LIMIT);
                    break;
                }
            }
            // 定期タイムアウトチェック
            if pns_iters & 0xFF == 0 && self.is_timed_out() {
                self.timed_out = true;
                break;
            }

            // Most-proving node 選択 + 盤面復元
            path.clear();
            captures.clear();
            ancestors.clear();
            path.push(0);
            ancestors.insert(arena[0].full_hash);
            let mut current = 0u32;

            let mut skip_expand = false;

            while arena[current as usize].expanded {
                let ci = current as usize;

                // AND ノード: 全子証明済み + deferred_drops → 逐次活性化
                if !arena[ci].or_node && !arena[ci].deferred_drops.is_empty() {
                    let all_proven = arena[ci].children.iter()
                        .all(|&c| arena[c as usize].pn == 0);
                    if all_proven {
                        // 証明済み子の直接 OR ノードのみ TT に flush．
                        // 中間ノード(子ありの AND/OR)は actual hand で格納するため
                        // 既存の中間エントリを不正に evict し MID 収束を妨げる恐れがある．
                        // 直接 OR 子ノードのみ flush することで，try_prefilter_block が
                        // 「合駒後の OR ノードが証明済みか」を判定できるようにする．
                        for &child_idx in &arena[ci].children {
                            let child = &arena[child_idx as usize];
                            if child.pn == 0 && child.or_node {
                                self.store(
                                    child.pos_key, child.hand, 0, INF,
                                    REMAINING_INFINITE, child.pos_key,
                                );
                            }
                        }

                        let and_remaining = arena[ci].remaining;
                        let att_hand = arena[ci].hand;
                        let mut and_proof = [0u8; HAND_KINDS];

                        let mut activated_unproven = false;
                        while !arena[ci].deferred_drops.is_empty() {
                            let next_drop = arena[ci].deferred_drops.remove(0);
                            #[cfg(feature = "tt_diag")]
                            { self.diag_pns_deferred_activations += 1; }
                            #[cfg(feature = "tt_diag")]
                            eprintln!(
                                "[pns_seq] AND node idx={}: activate drop {} (deferred_remaining={}), tt_entries={}",
                                ci, next_drop.to_usi(), arena[ci].deferred_drops.len(),
                                self.table.total_entries(),
                            );

                            // TT ベースプレフィルタ: 合駒の捕獲後局面が
                            // 既に TT で証明済みなら展開不要．
                            // PNS flush で先行子の証明が TT に反映された後に
                            // チェックすることで，同一マスの兄弟合駒を
                            // TT ヒットで即座に証明できる．
                            if and_remaining >= 3 {
                                let cap_pre = board.do_move(next_drop);
                                let child_hand_pre = board.hand[self.attacker.index()];
                                board.undo_move(next_drop, cap_pre);
                                if self.try_prefilter_block(
                                    board, next_drop, &child_hand_pre,
                                    and_remaining, &mut and_proof,
                                ) {
                                    self.prefilter_hits += 1;
                                    #[cfg(feature = "tt_diag")]
                                    {
                                        self.diag_pns_deferred_already_proven += 1;
                                        eprintln!(
                                            "[pns_seq]   prefilter hit for {} → proven",
                                            next_drop.to_usi(),
                                        );
                                    }
                                    // アリーナにも証明済みとして追加
                                    let cap_pf = board.do_move(next_drop);
                                    let child_pk_pf = position_key(board);
                                    let child_fh_pf = board.hash;
                                    let child_hand_pf =
                                        board.hand[self.attacker.index()];
                                    let child_remaining_pf =
                                        and_remaining.saturating_sub(1);
                                    board.undo_move(next_drop, cap_pf);
                                    let pf_idx = arena.len() as u32;
                                    arena.push(PnsNode {
                                        pos_key: child_pk_pf,
                                        full_hash: child_fh_pf,
                                        hand: child_hand_pf,
                                        pn: 0,
                                        dn: INF,
                                        parent: current,
                                        move_from_parent: next_drop,
                                        or_node: true,
                                        expanded: true,
                                        children: Vec::new(),
                                        remaining: child_remaining_pf,
                                        deferred_drops: Vec::new(),
                                    });
                                    arena[ci].children.push(pf_idx);
                                    continue;
                                }
                            }

                            let cap = board.do_move(next_drop);
                            let child_fh = board.hash;
                            let child_pk = position_key(board);
                            let child_hand = board.hand[self.attacker.index()];
                            let child_remaining =
                                arena[ci].remaining.saturating_sub(1);

                            let is_loop = ancestors.contains(&child_fh);
                            let (cpn, cdn) = if is_loop {
                                (INF, 0u32)
                            } else {
                                let (p, d, _) = self.look_up_pn_dn(
                                    child_pk, &child_hand, child_remaining,
                                );
                                (p, d)
                            };

                            board.undo_move(next_drop, cap);

                            // 反証(防御成功) → AND ノード反証
                            if cdn == 0 {
                                arena[ci].pn = INF;
                                arena[ci].dn = 0;
                                skip_expand = true;
                                activated_unproven = true;
                                break;
                            }

                            // 子ノード生成
                            let child_idx = arena.len() as u32;
                            arena.push(PnsNode {
                                pos_key: child_pk,
                                full_hash: child_fh,
                                hand: child_hand,
                                pn: cpn,
                                dn: cdn,
                                parent: current,
                                move_from_parent: next_drop,
                                or_node: true,
                                expanded: cpn == 0 || cdn == 0,
                                children: Vec::new(),
                                remaining: child_remaining,
                                deferred_drops: Vec::new(),
                            });
                            arena[ci].children.push(child_idx);

                            // 証明済み → 次の deferred drop へ
                            if cpn == 0 {
                                #[cfg(feature = "tt_diag")]
                                { self.diag_pns_deferred_already_proven += 1; }
                                continue;
                            }

                            // 未証明 → この子を MPN として選択
                            let cap = board.do_move(next_drop);
                            captures.push(cap);
                            path.push(child_idx);
                            ancestors.insert(child_fh);
                            current = child_idx;
                            activated_unproven = true;
                            break;
                        }

                        if activated_unproven {
                            break; // MPN 選択ループ終了
                        }

                        // 全 deferred が証明済み → AND ノード証明
                        arena[ci].pn = 0;
                        arena[ci].dn = INF;
                        skip_expand = true;
                        break;
                    }
                }

                // 通常の子ノード選択
                if arena[ci].children.is_empty() {
                    // 子ノードなし(展開済みだが全消去等) → リーフとして再展開
                    break;
                }
                let best_child = if arena[ci].or_node {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].pn, arena[c as usize].dn))
                        .unwrap()
                } else {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].dn, arena[c as usize].pn))
                        .unwrap()
                };
                let child_move = arena[best_child as usize].move_from_parent;
                let captured = board.do_move(child_move);
                captures.push(captured);
                path.push(best_child);
                ancestors.insert(arena[best_child as usize].full_hash);
                current = best_child;
            }

            // リーフ展開(逐次活性化で解決済みの場合はスキップ)
            if !skip_expand {
                let ply = (path.len() - 1) as u32;
                self.pns_expand(board, &mut arena, current, ply, &ancestors);
            }

            // 盤面をルートに戻す
            for i in (1..path.len()).rev() {
                let child_move = arena[path[i] as usize].move_from_parent;
                board.undo_move(child_move, captures[i - 1]);
            }

            // バックアップ: 展開ノードからルートまで pn/dn を更新
            Self::pns_backup(&mut arena, current);
        }

        // 診断: PNS 終了時の状態
        #[cfg(feature = "tt_diag")]
        {
            let pns_nodes_used = self.nodes_searched;
            let root_pn = arena[0].pn;
            let root_dn = arena[0].dn;
            let pns_elapsed = self.start_time.elapsed().as_secs_f64();
            eprintln!("[pns_diag] arena={}/{} iters={} nodes_used={} root_pn={} root_dn={} TT_pos={} time={:.2}s",
                arena.len(), PNS_MAX_ARENA_NODES, pns_iters, pns_nodes_used, root_pn, root_dn,
                self.table.len(), pns_elapsed);
        }

        // 証明/反証結果を TT に格納(PV 抽出用)
        self.pns_store_to_tt(&arena);

        // デバッグ: 証明ツリーの整合性チェック
        #[cfg(debug_assertions)]
        if arena[0].pn == 0 {
            Self::validate_pns_proof(&arena, 0);
        }

        // ルートが証明済みならアリーナから直接 PV を抽出
        if arena[0].pn == 0 {
            let mut visited: FxHashSet<u64> = FxHashSet::default();
            let pv = self.pns_extract_pv(board, &arena, 0, &mut visited);
            if !pv.is_empty() && pv.len() % 2 == 1 {
                return Some(pv);
            }
        }
        None
    }

    /// PNS ノード展開: リーフノードの子を生成し初期 pn/dn を設定する．
    ///
    /// 子ノードの初期化では既存 TT エントリおよびヒューリスティック
    /// (DFPN-E エッジコスト，Deep df-pn 深さバイアス，静的詰め判定)を利用する．
    fn pns_expand(
        &mut self,
        board: &mut Board,
        arena: &mut Vec<PnsNode>,
        node_idx: u32,
        ply: u32,
        ancestors: &FxHashSet<u64>,
    ) {
        self.nodes_searched += 1;
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let or_node = arena[node_idx as usize].or_node;
        let remaining = arena[node_idx as usize].remaining;
        let pos_key = arena[node_idx as usize].pos_key;
        let att_hand = arena[node_idx as usize].hand;

        // 終端: 深さ制限 / 手数制限
        if remaining == 0 || ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            if or_node {
                // OR ノードの深さ制限: 王手が0手なら真の不詰(REMAINING_INFINITE)．
                // 王手がある場合: 2手延長で全王手が即座に反証可能かを確認．
                let checks = self.generate_check_moves(board);
                if checks.is_empty() {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else if self.depth_limit_all_checks_refutable(board, &checks) {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else {
                    self.store(pos_key, att_hand, INF, 0, 0, pos_key);
                }
            } else {
                self.store(pos_key, att_hand, INF, 0, 0, pos_key);
            }
            return;
        }

        // 合法手生成
        let moves = if or_node {
            self.generate_check_moves(board)
        } else {
            self.generate_defense_moves(board)
        };

        if moves.is_empty() {
            if or_node {
                // 王手手段なし → 不詰
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key);
            } else {
                // 応手なし → 詰み
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                self.store(pos_key, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, pos_key);
            }
            arena[node_idx as usize].expanded = true;
            return;
        }

        // DFPN-E: 守備側玉の位置(OR ノードのエッジコスト計算用)
        let defender_king_sq = if or_node {
            board.king_square(board.turn.opponent())
        } else {
            None
        };

        let child_remaining = remaining.saturating_sub(1);
        let child_or_node = !or_node;
        let mut or_nm_min_remaining: u16 = REMAINING_INFINITE;
        // AND ノードの合駒逐次活性化: 最初の unproven drop のみ子ノード生成
        let mut first_unproven_drop_added = or_node; // OR ノードでは無効

        for m in &moves {
            let captured = board.do_move(*m);
            let child_fh = board.hash;
            let child_pk = position_key(board);
            let child_hand = board.hand[self.attacker.index()];

            // ループ検出: 祖先と同一局面なら即座に不詰/無限ループ扱い
            let is_loop = ancestors.contains(&child_fh);

            let (mut cpn, mut cdn) = if is_loop {
                (INF, 0u32)
            } else {
                let (p, d, _) = self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
                (p, d)
            };

            // TT に初期値(1,1)しかない場合: ヒューリスティック初期化
            if cpn == 1 && cdn == 1 && !is_loop {
                if child_or_node {
                    // 子は OR ノード(攻め方手番): 王手数ベース
                    let checks = self.generate_check_moves(board);
                    if checks.is_empty() {
                        cpn = INF;
                        cdn = 0;
                        self.store(child_pk, child_hand, INF, 0, REMAINING_INFINITE, child_pk);
                    } else if self.has_mate_in_1_with(board, &checks) {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, child_hand, 0, INF, REMAINING_INFINITE, child_pk);
                    } else {
                        let nc = checks.len() as u32;
                        cpn = self.heuristic_or_pn(board, nc)
                            .saturating_add(edge_cost_and(*m))
                            .saturating_add(sacrifice_check_boost(board, &checks));
                        cdn = 1;
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk);
                    }
                } else {
                    // 子は AND ノード(玉方手番): 応手数ベース
                    let defenses = self.generate_defense_moves(board);
                    if defenses.is_empty() {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, child_pk);
                    } else {
                        let n = defenses.len() as u32;
                        cpn = self.heuristic_and_pn(board, n);
                        if let Some(ksq) = defender_king_sq {
                            cpn = cpn.saturating_add(edge_cost_or(*m, ksq));
                        }
                        cdn = 1;
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk);
                    }
                }
            }

            board.undo_move(*m, captured);

            // OR ノードで子が即座に証明 → 親を証明して終了
            if or_node && cpn == 0 {
                let child_ph = self.table.get_proof_hand(child_pk, &child_hand);
                let mut proof = adjust_hand_for_move(*m, &child_ph);
                for k in 0..HAND_KINDS {
                    proof[k] = proof[k].min(att_hand[k]);
                }
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                arena[node_idx as usize].expanded = true;
                self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                return;
            }
            // AND ノードで子が即座に反証 → 親を反証して終了
            // att_hand で保存(TT ヒット率最大化)
            if !or_node && cdn == 0 {
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                arena[node_idx as usize].expanded = true;
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                let prop_rem = propagate_nm_remaining(child_rem, remaining);
                self.store(pos_key, att_hand, INF, 0, prop_rem, pos_key);
                return;
            }
            // OR ノードで子が反証済み → 子を追加せずスキップ
            if or_node && cdn == 0 {
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                or_nm_min_remaining = or_nm_min_remaining.min(child_rem);
                continue;
            }

            // AND ノードの合駒逐次活性化:
            // 未証明の drop を1つだけ子ノードとして展開し，残りは deferred_drops に格納．
            // 弱い駒から順に証明 → TT 蓄積 → 強い駒の探索で援用する．
            // 既に証明済み(cpn=0)の drop はそのまま子ノードとして追加する．
            if !or_node && m.is_drop() && cpn > 0 {
                if first_unproven_drop_added {
                    #[cfg(feature = "tt_diag")]
                    eprintln!(
                        "[pns_seq] pns_expand: defer drop {} at AND node idx={}",
                        m.to_usi(), node_idx,
                    );
                    arena[node_idx as usize].deferred_drops.push(*m);
                    continue;
                }
                first_unproven_drop_added = true;
            }

            let child_idx = arena.len() as u32;
            arena.push(PnsNode {
                pos_key: child_pk,
                full_hash: child_fh,
                hand: child_hand,
                pn: cpn,
                dn: cdn,
                parent: node_idx,
                move_from_parent: *m,
                or_node: child_or_node,
                expanded: cpn == 0 || cdn == 0,
                children: Vec::new(),
                remaining: child_remaining,
                deferred_drops: Vec::new(),
            });
            arena[node_idx as usize].children.push(child_idx);
        }

        // OR ノードで全子が反証済み(children 空)
        if or_node && arena[node_idx as usize].children.is_empty() {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            let mut prop_rem = propagate_nm_remaining(or_nm_min_remaining, remaining);
            // 2手延長による REMAINING_INFINITE 昇格
            if prop_rem != REMAINING_INFINITE {
                let checks = self.generate_check_moves(board);
                if checks.is_empty()
                    || self.depth_limit_all_checks_refutable(board, &checks)
                {
                    prop_rem = REMAINING_INFINITE;
                }
            }
            self.store(pos_key, att_hand, INF, 0, prop_rem, pos_key);
            return;
        }

        arena[node_idx as usize].expanded = true;
    }

    /// PNS バックアップ: 展開ノードからルートまで pn/dn を再計算する．
    ///
    /// OR ノード: pn = min(child_pn), dn = sum(child_dn)
    /// AND ノード: WPN = max(child_pn) + (unproven_count - 1), dn = min(child_dn)
    /// pn/dn が変化しなくなった時点で伝播を打ち切る．
    fn pns_backup(arena: &mut [PnsNode], start_idx: u32) {
        let mut current = start_idx;
        loop {
            let ni = current as usize;

            if !arena[ni].expanded || arena[ni].children.is_empty() {
                // 終端ノード(子なし): pn/dn は展開時に設定済み
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            // AND ノードが展開時の早期脱出で反証済み(dn=0)の場合，
            // 部分的な子しかアリーナにないため再計算をスキップする．
            // (再計算すると部分的な子だけから pn=0 と誤判定される)
            if !arena[ni].or_node && arena[ni].dn == 0 {
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            let old_pn = arena[ni].pn;
            let old_dn = arena[ni].dn;

            let (new_pn, new_dn) = if arena[ni].or_node {
                // OR ノード: pn = min(child_pn), dn = sum(child_dn)
                let mut min_pn = INF;
                let mut sum_dn: u64 = 0;
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let ci = arena[ni].children[i] as usize;
                    if arena[ci].pn < min_pn {
                        min_pn = arena[ci].pn;
                    }
                    sum_dn = sum_dn.saturating_add(arena[ci].dn as u64);
                }
                (min_pn, sum_dn.min(INF as u64) as u32)
            } else {
                // AND ノード: WPN, dn = min(child_dn)
                let mut max_pn: u32 = 0;
                let mut min_dn = INF;
                let mut unproven: u32 = 0;
                let mut disproved = false;
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let ci = arena[ni].children[i] as usize;
                    if arena[ci].dn == 0 {
                        disproved = true;
                        break;
                    }
                    if arena[ci].pn == 0 {
                        // VPN: 証明済み子を pn 合計から除外
                        continue;
                    }
                    if arena[ci].pn > max_pn {
                        max_pn = arena[ci].pn;
                    }
                    if arena[ci].dn < min_dn {
                        min_dn = arena[ci].dn;
                    }
                    unproven += 1;
                }
                if disproved {
                    (INF, 0u32)
                } else if unproven == 0 && arena[ni].deferred_drops.is_empty() {
                    // 全子証明済み + deferred なし → AND ノード証明
                    (0u32, INF)
                } else if unproven == 0 {
                    // 全子証明済みだが deferred_drops 残り → 未完了
                    // MPN 選択時に次の合駒を活性化するため pn=1, dn=1 で保持
                    (1u32, 1u32)
                } else {
                    let pn = (max_pn as u64)
                        .saturating_add(unproven as u64 - 1)
                        .min(INF as u64) as u32;
                    (pn, min_dn)
                }
            };

            arena[ni].pn = new_pn;
            arena[ni].dn = new_dn;

            // pn/dn が変化しなければ伝播打ち切り
            if new_pn == old_pn && new_dn == old_dn {
                return;
            }
            if arena[ni].parent == u32::MAX {
                return;
            }
            current = arena[ni].parent;
        }
    }

    /// PNS 証明ツリーの整合性を検証する(デバッグ用)．
    ///
    /// OR ノード(pn=0): 少なくとも1つの子が pn=0
    /// AND ノード(pn=0): 全ての子が pn=0(展開済みの場合)
    #[cfg(debug_assertions)]
    fn validate_pns_proof(arena: &[PnsNode], idx: u32) {
        let node = &arena[idx as usize];
        if node.pn != 0 {
            return;
        }
        if !node.expanded || node.children.is_empty() {
            // リーフ: TT/ヒューリスティックから取得した pn=0
            eprintln!("  PNS leaf proven: idx={}, or={}, pk={:#x}, move={}",
                idx, node.or_node, node.pos_key,
                if idx == 0 { "root".to_string() } else { node.move_from_parent.to_usi() });
            return;
        }
        if node.or_node {
            // OR: 少なくとも1つの子が pn=0
            let has_proven = node.children.iter().any(|&c| arena[c as usize].pn == 0);
            assert!(has_proven,
                "PNS BUG: OR node {} (pk={:#x}) is proven but no child has pn=0. children: {:?}",
                idx, node.pos_key,
                node.children.iter().map(|&c| {
                    let ch = &arena[c as usize];
                    format!("{}(pn={},dn={},or={},exp={})",
                        ch.move_from_parent.to_usi(), ch.pn, ch.dn, ch.or_node, ch.expanded)
                }).collect::<Vec<_>>());
            // 証明された子を表示
            let proven_child = node.children.iter()
                .find(|&&c| arena[c as usize].pn == 0).unwrap();
            eprintln!("  PNS OR proven: idx={}, pk={:#x}, best_child={} ({})",
                idx, node.pos_key, proven_child,
                arena[*proven_child as usize].move_from_parent.to_usi());
            // 再帰: 証明された子のみ
            for &c in &node.children {
                if arena[c as usize].pn == 0 {
                    Self::validate_pns_proof(arena, c);
                }
            }
        } else {
            eprintln!("  PNS AND proven: idx={}, pk={:#x}, move={}, {} children",
                idx, node.pos_key, node.move_from_parent.to_usi(), node.children.len());
            // AND: 全子が pn=0
            for &c in &node.children {
                assert!(arena[c as usize].pn == 0,
                    "PNS BUG: AND node {} (pk={:#x}) is proven but child {} ({}) has pn={}",
                    idx, node.pos_key, c,
                    arena[c as usize].move_from_parent.to_usi(),
                    arena[c as usize].pn);
                Self::validate_pns_proof(arena, c);
            }
        }
    }

    /// PNS アリーナから直接 PV を抽出する．
    ///
    /// TT ベースの `extract_pv` と異なり，PNS が構築した明示的な探索木を
    /// 辿るため，PNS が証明した経路を正確に復元できる．
    /// 展開されていないリーフ(TT から pn=0 を取得した子)では
    /// TT ベースの `extract_pv_recursive` にフォールバックする．
    ///
    /// - OR ノード: 証明済み子のうち最短 PV を選択
    /// - AND ノード: 証明済み子のうち最長 PV を選択(最長抵抗)
    fn pns_extract_pv(
        &mut self,
        board: &mut Board,
        arena: &[PnsNode],
        node_idx: u32,
        visited: &mut FxHashSet<u64>,
    ) -> Vec<Move> {
        let node = &arena[node_idx as usize];

        // 未証明ノード → PV なし
        if node.pn != 0 {
            return Vec::new();
        }

        // ループ検出
        if visited.contains(&node.full_hash) {
            return Vec::new();
        }

        // 未展開リーフまたは子なし(終端) → TT フォールバック
        if !node.expanded || node.children.is_empty() {
            return self.extract_pv_recursive(board, node.or_node, visited, 0);
        }

        if node.or_node {
            // OR ノード: 証明済み子から最短 PV を選択
            let mut best_pv: Option<Vec<Move>> = None;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                // sub_pv が空でないか，AND 終端(応手なし=詰み)なら有効
                let total_len = 1 + sub_pv.len();
                // 奇数長(攻め方の手で終わる)のみ有効な PV
                if total_len % 2 == 0 && !sub_pv.is_empty() {
                    continue;
                }
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => total_len < prev.len(),
                };
                if is_better {
                    let mut pv = vec![child.move_from_parent];
                    pv.extend(sub_pv);
                    best_pv = Some(pv);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 全子が証明済み，最長 PV を選択(最長抵抗)
            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                let total_len = 1 + sub_pv.len();
                // AND ノードの PV は偶数長でなければならない
                if total_len % 2 == 1 {
                    continue;
                }
                let is_capture = child.move_from_parent.captured_piece_raw() > 0;
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => {
                        if total_len > prev.len() {
                            true
                        } else if total_len == prev.len()
                            && is_capture
                            && !best_is_capture
                        {
                            true
                        } else {
                            false
                        }
                    }
                };
                if is_better {
                    let mut pv = vec![child.move_from_parent];
                    pv.extend(sub_pv);
                    best_pv = Some(pv);
                    best_is_capture = is_capture;
                }
            }

            best_pv.unwrap_or_default()
        }
    }

    fn pns_store_to_tt(&mut self, arena: &[PnsNode]) {
        for node in arena {
            if node.pn == 0 && node.expanded && !node.children.is_empty() {
                // 証明済み中間ノード
                if node.or_node {
                    // OR 証明: 証明子の手を TT Best Move に記録
                    let best_child = node.children.iter()
                        .find(|&&c| arena[c as usize].pn == 0);
                    if let Some(&ci) = best_child {
                        let best_move16 = arena[ci as usize].move_from_parent.to_move16();
                        self.store_with_best_move(
                            node.pos_key, node.hand, 0, INF,
                            REMAINING_INFINITE, node.pos_key, best_move16,
                        );
                    }
                } else {
                    // AND 証明: 全子が証明済み
                    self.store(
                        node.pos_key, node.hand, 0, INF,
                        REMAINING_INFINITE, node.pos_key,
                    );
                }
            } else if node.dn == 0 && node.expanded {
                // PNS の反証(NM)は TT にバックプロパゲーションしない．
                // 理由: PNS はアリーナサイズに制限された最良優先探索であり，
                // 探索木を完全には展開しない．そのため PNS の NM は
                // 「アリーナ内で反証された」という意味でしかなく，
                // MID(DFS) の NM のように「深さ R 以内で完全に反証された」
                // とは保証できない．PNS NM を remaining 付きで TT に格納すると，
                // 後続の mid_fallback が NM エントリをヒットして探索をスキップし，
                // 偽 NoCheckmate を引き起こす．
                // 展開フェーズ(expand_pns_node)で各ノードの NM は既に
                // TT に個別に記録済みなので，backprop での追加格納は不要．
            }
        }
    }
}

/// 詰将棋を解く便利関数．
///
/// タイムアウトを指定する場合は [`solve_tsume_with_timeout`] を使用する．
///
/// # 引数
///
/// - `sfen`: 局面のSFEN文字列．
/// - `depth`: 最大探索手数(None でデフォルト 31)．
/// - `nodes`: 最大ノード数(None でデフォルト 1,048,576)．
/// - `draw_ply`: 引き分け手数(None でデフォルト 32767)．
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, draw_ply, None, None, None, None, None)
}

/// タイムアウト指定付きで詰将棋を解く便利関数．
///
/// # 戻り値
///
/// 詰みが証明された場合でも，PV 復元フェーズ(`complete_pv_or_nodes`)の
/// ノード予算が不足すると [`TsumeResult::CheckmateNoPv`] が返ることがある．
/// 特に長手数(17手以上)の詰将棋では，PV 沿いの各未証明子に対する
/// 追加証明の1子あたり予算(デフォルト 1024 ノード)が不足しやすい．
/// `pv_nodes_per_child` を増やすことで改善できる．
///
/// # 引数
///
/// - `find_shortest`: 最短手数探索を行うか(None でデフォルト true)．
///   false にすると `complete_or_proofs()` による追加探索をスキップし，
///   最初に見つかった詰み手順をそのまま返す．ノード数は削減されるが，
///   返される手順が最短とは限らない．
/// - `pv_nodes_per_child`: PV 復元時の1子あたりノード予算(None でデフォルト 1024)．
///   長手数の詰将棋で `CheckmateNoPv` が返る場合に増やすと効果的．
/// - `mate_budget`: 静的詰め探索(`static_mate`)の1子あたりノード予算(None でデフォルト 0)．
///   0 の場合は静的詰め探索を無効化する．値を増やすとノード削減効果が高まるが，
///   1ノードあたりの計算コストが増加する．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    mate_budget: Option<u32>,
    tt_gc_threshold: Option<usize>,
) -> Result<TsumeResult, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(mb) = mate_budget {
        solver.set_mate_budget(mb);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    Ok(solver.solve(&mut board))
}

#[cfg(test)]
mod tests {
    use super::*;

    // === hand_gte / hand_gte_forward_chain のユニットテスト ===

    /// hand_gte: 全駒種で a >= b なら true．
    #[test]
    fn test_hand_gte_basic() {
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte(&a, &b));

        let a = [2, 0, 0, 0, 0, 0, 0]; // 歩2
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte(&a, &b));

        let a = [0, 0, 0, 0, 0, 0, 0]; // 空
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(!hand_gte(&a, &b));
    }

    /// hand_gte: 異種駒では代替不可(従来の挙動)．
    #[test]
    fn test_hand_gte_different_pieces() {
        // 香1 vs 歩1: 香で歩を代替できない(hand_gteでは)
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(!hand_gte(&a, &b));

        // 飛1 vs 香1
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(!hand_gte(&a, &b));
    }

    /// forward_chain: 歩 ≤ 香 ≤ 飛 のチェーン代替．
    #[test]
    fn test_forward_chain_pawn_to_lance() {
        // 香1 >= 歩1 (香は歩の上位互換)
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    #[test]
    fn test_forward_chain_pawn_to_rook() {
        // 飛1 >= 歩1 (飛は歩の上位互換)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    #[test]
    fn test_forward_chain_lance_to_rook() {
        // 飛1 >= 香1 (飛は香の上位互換)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 逆方向は不成立(歩は香の代替にならない)．
    #[test]
    fn test_forward_chain_no_reverse() {
        // 歩1 < 香1 (歩は香の代替にならない)
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 香1 < 飛1
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 歩1 < 飛1
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: カスケード(歩不足 → 香で補填 → 香不足 → 飛で補填)．
    #[test]
    fn test_forward_chain_cascade() {
        // 歩1+香1 を 飛2 で代替
        let a = [0, 0, 0, 0, 0, 0, 2]; // 飛2
        let b = [1, 1, 0, 0, 0, 0, 0]; // 歩1+香1
        assert!(hand_gte_forward_chain(&a, &b));

        // 歩2 を 香1+飛1 で代替(香が歩1つ分を吸収，飛が残り1つを吸収)
        let a = [0, 1, 0, 0, 0, 0, 1]; // 香1+飛1
        let b = [2, 0, 0, 0, 0, 0, 0]; // 歩2
        assert!(hand_gte_forward_chain(&a, &b));

        // 歩2+香1 を 飛1 では不足(飛1で代替できるのは1つだけ)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [2, 1, 0, 0, 0, 0, 0]; // 歩2+香1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 非チェーン駒(桂・銀・金・角)は代替不可．
    #[test]
    fn test_forward_chain_non_chain_pieces() {
        // 桂が足りなければ false
        let a = [1, 1, 0, 0, 0, 0, 1]; // 歩1+香1+飛1
        let b = [0, 0, 1, 0, 0, 0, 0]; // 桂1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 角が足りなければ false
        let a = [0, 0, 0, 0, 0, 0, 2]; // 飛2
        let b = [0, 0, 0, 0, 0, 1, 0]; // 角1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 複合ケース(チェーン + 非チェーン駒)．
    #[test]
    fn test_forward_chain_mixed() {
        // 香1+金1 >= 歩1+金1
        let a = [0, 1, 0, 0, 1, 0, 0]; // 香1+金1
        let b = [1, 0, 0, 0, 1, 0, 0]; // 歩1+金1
        assert!(hand_gte_forward_chain(&a, &b));

        // 飛1+桂1 >= 歩1+桂1
        let a = [0, 0, 1, 0, 0, 0, 1]; // 桂1+飛1
        let b = [1, 0, 1, 0, 0, 0, 0]; // 歩1+桂1
        assert!(hand_gte_forward_chain(&a, &b));

        // 飛1+金0 < 歩1+金1 (金が不足)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [1, 0, 0, 0, 1, 0, 0]; // 歩1+金1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 同一 hand なら常に true (hand_gte の fast path)．
    #[test]
    fn test_forward_chain_identity() {
        let h = [3, 2, 1, 1, 2, 1, 1];
        assert!(hand_gte_forward_chain(&h, &h));
    }

    /// forward_chain: 空の hand(何も必要としない)なら常に true．
    #[test]
    fn test_forward_chain_empty_requirement() {
        let a = [0, 0, 0, 0, 0, 0, 0];
        let b = [0, 0, 0, 0, 0, 0, 0];
        assert!(hand_gte_forward_chain(&a, &b));

        let a = [1, 1, 1, 1, 1, 1, 1];
        assert!(hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 実際のチェーン合駒シナリオ(L*6g → N*6g)．
    ///
    /// L*6g の proof で attacker hand に lance が必要な局面の TT エントリを，
    /// N*6g の探索(attacker hand に knight がある)で再利用できるか．
    #[test]
    fn test_forward_chain_real_scenario() {
        // L*6g 後の proof entry: 攻め方が香を獲得
        let stored = [0, 1, 0, 0, 0, 0, 0]; // 香1(Rx6g で獲得)
        // N*6g 後の current hand: 攻め方が桂を獲得
        let current = [0, 0, 1, 0, 0, 0, 0]; // 桂1(Rx6g で獲得)
        // 桂は香の上位互換ではない → 再利用不可
        assert!(!hand_gte_forward_chain(&current, &stored));

        // 逆に: stored が歩1, current が香1 → 再利用可能
        let stored_pawn = [1, 0, 0, 0, 0, 0, 0];
        let current_lance = [0, 1, 0, 0, 0, 0, 0];
        assert!(hand_gte_forward_chain(&current_lance, &stored_pawn));
    }

    /// 詰将棋画像のテストケース: 小阪昇作，9手詰
    ///
    /// 局面: 後手玉1四，先手角2四・3四，後手銀3一・後手香3二
    /// 先手持ち駒: 飛，歩
    /// 後手持ち駒: 飛，金4，銀3，桂4，香3，歩17
    ///
    /// 正解手順: 1三角成，同玉，2三飛打，1二玉，1三歩打，1一玉，2一飛成，同玉，1二歩成
    #[test]
    fn test_tsume_9te() {
        let sfen = "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(15, 1_048_576, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(),
                    9,
                    "should be 9-move checkmate, got: {:?}",
                    usi_moves
                );

                // 正解手順を検証(USI形式)
                assert_eq!(usi_moves[0], "2d1c+", "move 1: 1三角成");
                assert_eq!(usi_moves[1], "1d1c", "move 2: 同玉");
                assert_eq!(usi_moves[2], "R*2c", "move 3: 2三飛打");
                assert_eq!(usi_moves[3], "1c1b", "move 4: 1二玉");
                assert_eq!(usi_moves[4], "P*1c", "move 5: 1三歩打");
                assert_eq!(usi_moves[5], "1b1a", "move 6: 1一玉");
                assert_eq!(usi_moves[6], "2c2a+", "move 7: 2一飛成");
                assert_eq!(usi_moves[7], "1a2a", "move 8: 同玉");
                assert_eq!(usi_moves[8], "1c1b+", "move 9: 1二歩成");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 簡単な1手詰め．
    #[test]
    fn test_tsume_1te() {
        // 後手玉1一，先手金2三，先手持ち駒: 金
        // G*1b(1二金打)で詰み
        let sfen = "8k/9/7G1/9/9/9/9/9/9 b G 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(3, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1);
                assert_eq!(moves[0].to_usi(), "G*1b", "1手詰め: G*1b(1二金打)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 3手詰め: 後手玉1一，先手飛3三，先手持ち駒: 金
    ///
    /// 正解: 1三飛成，2一玉，2二金打 まで3手詰
    #[test]
    fn test_tsume_3te() {
        let sfen = "8k/9/6R2/9/9/9/9/9/9 b G 1";
        let result = solve_tsume(sfen, Some(7), Some(1_048_576), None).unwrap();

        let expected = ["3c1c+", "1a2a", "G*2b"];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 3,
                    "expected 3 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 不詰のケース．
    #[test]
    fn test_no_checkmate() {
        // 後手玉5一，先手持ち駒: 歩 → 歩では詰まない
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(5, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!("expected NoCheckmate, got {:?}", other),
        }
    }

    /// solve_tsume 便利関数のテスト．
    #[test]
    fn test_solve_tsume_convenience() {
        let result = solve_tsume(
            "8k/9/7G1/9/9/9/9/9/9 b G 1",
            Some(3),
            Some(100_000),
            None,
        )
        .unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(moves.len(), 1);
                // G*1b(12金打) または G*2b(22金打) が正解
                assert!(
                    pv[0] == "G*1b" || pv[0] == "G*2b",
                    "expected G*1b or G*2b, got {}",
                    pv[0],
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    #[test]
    fn test_tsume_2() {
        // 盤面: 5一と，2一王，1一香，2二銀，4三飛，2四角，先手持駒: 金桂
        // 11手詰め: 32金打，同玉，42角成，21玉，31馬，同銀，23飛成，22銀，33桂打，31玉，41と
        let sfen = "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(1_048_576), None).unwrap();

        let expected = [
            "G*3b", "2a3b", "2d4b+", "3b2a", "4b3a", "2b3a",
            "4c2c+", "3a2b", "N*3c", "2a3a", "5a4a",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(moves.len(), 11, "expected 11 moves, got {}: {:?}", moves.len(), usi_moves);
                assert_eq!(
                    usi_moves,
                    expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// between_bb ヘルパーのテスト．
    #[test]
    fn test_between_bb() {
        // 飛5一(col=4,row=0) と 1一(col=0,row=0) の間: 2一,3一,4一
        let between = attack::between_bb(Square::new(4, 0), Square::new(0, 0));
        assert_eq!(between.count(), 3);
        assert!(between.contains(Square::new(1, 0))); // 2一
        assert!(between.contains(Square::new(2, 0))); // 3一
        assert!(between.contains(Square::new(3, 0))); // 4一

        // 隣接マス(間なし)
        let between2 = attack::between_bb(Square::new(0, 0), Square::new(1, 0));
        assert!(between2.is_empty());

        // 斜め
        let between3 = attack::between_bb(Square::new(0, 0), Square::new(3, 3));
        assert_eq!(between3.count(), 2);
        assert!(between3.contains(Square::new(1, 1)));
        assert!(between3.contains(Square::new(2, 2)));
    }

    /// タイムアウト機能のテスト．
    #[test]
    fn test_timeout() {
        // 不詰の局面を極短タイムアウトで探索 → Unknown が返る
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, u64::MAX, 32767, 0);
        // timeout=0 なので即タイムアウト(ただし最初の1024ノードは走る)
        let result = solver.solve(&mut board);

        // NoCheckmate か Unknown のどちらか(歩1枚では詰まない)
        match &result {
            TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {}
            other => panic!("expected NoCheckmate or Unknown, got {:?}", other),
        }
    }

    /// 詰将棋画像3のテストケース．
    ///
    /// 局面: 後手玉2三，後手桂2一，後手香1一，後手飛5四，後手歩1三，後手と1六
    ///       先手歩1五，先手桂3四，先手金2六，先手香3六
    /// 先手持駒: 飛
    /// 後手持駒: 歩15，香2，桂2，銀4，金3，角2
    #[test]
    fn test_tsume_3() {
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        let expected = [
            "3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d",
            "2f2e", "2d2e", "3b3e",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "expected 9 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 2一龍後に2三歩打で詰まないことを検証する．
    ///
    /// 無駄合いフィルタが誤って2三歩打を除外していた問題の回帰テスト．
    /// 2二桂成，同玉，4二飛打，2三玉，3二飛成，2四玉，2一龍 の後，
    /// 後手は2三歩打で合い駒でき，これは無駄合いではない．
    #[test]
    fn test_tsume_3_ryu_2a_not_checkmate() {
        // 2一龍後の局面を作成
        // 初期局面から 3d2b+, 2c2b, R*4b, 2b2c, 4b3b+, 2c2d, 3b2a を実行
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let moves_usi = ["3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d", "3b2a"];
        for usi in &moves_usi {
            let m = board.move_from_usi(usi).expect(&format!("invalid USI: {}", usi));
            board.do_move(m);
        }

        // ここは AND ノード(後手番)．2三歩打(P*2c)が合法手に含まれることを検証
        let defenses = movegen::generate_legal_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();
        // debug: defenses

        // P*2c (2三歩打) が合法手に含まれること
        assert!(
            usi_defenses.contains(&"P*2c".to_string()),
            "P*2c should be a legal defense, got: {:?}", usi_defenses
        );

        // 2三歩打後の局面は詰みではないことを確認
        let p2c = board.move_from_usi("P*2c").unwrap();
        let cap = board.do_move(p2c);

        // 先手番(攻め方)から探索して詰みがないことを検証
        let mut solver = DfPnSolver::new(15, 100_000, 32767);
        let result = solver.solve(&mut board);
        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "P*2c 後の局面は詰みではないはず: {:?}",
            result
        );

        board.undo_move(p2c, cap);
    }

    /// 詰将棋テストケース4．
    ///
    /// 局面: 7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p
    /// 11手詰め(合い効かずにより3二歩打を除外)．
    /// 最終2手は玉の逃げ方により3パターンの正解が存在:
    /// - 1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍
    #[test]
    fn test_tsume_4() {
        let sfen = "7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                // PNS (Best-First) は最短 9 手詰めを発見する．
                // MID フォールバック時は 11 手の解を返す場合がある．
                assert!(
                    usi_moves.len() == 9 || usi_moves.len() == 11,
                    "expected 9 or 11 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert!(
                    usi_moves.len() % 2 == 1,
                    "tsume must have odd number of moves, got {}", usi_moves.len(),
                );
                // 最終2手は玉の逃げ方により3パターン:
                //   1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍
                let last2 = &usi_moves[usi_moves.len() - 2..];
                let valid_endings = [
                    ["1b1a", "3d2b+"],  // 1一玉，2二桂成
                    ["1b1a", "4b2b"],   // 1一玉，2二龍
                    ["1b1c", "4b2b"],   // 1三玉，2二龍
                ];
                assert!(
                    valid_endings.iter().any(|e| last2 == e),
                    "last 2 moves must match a valid ending pattern, got: {:?}\n  valid: {:?}",
                    last2, valid_endings,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// S*1c → △同桂の後は不詰であることを確認する回帰テスト．
    ///
    /// PNS の AND ノード早期脱出時に部分的な子からの誤計算で
    /// 偽の証明(false proof)が発生するバグの回帰テスト．
    /// MID は正しくこの局面を不詰と判定する．
    #[test]
    fn test_pns_no_false_proof_after_dogyoku() {
        // S*1c, 同桂後の局面: 不詰であるべき
        let sfen = "9/7k1/5R2n/7Np/6P2/9/9/9/9 b r2b4g4s2n4l16p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(31, 2_000_000, 512);
        solver.attacker = board.turn;
        solver.start_time = Instant::now();

        // MID は不詰を正しく判定する
        solver.mid_fallback(&mut board);

        let (root_pn, _root_dn) = solver.look_up_board(&board);
        assert_ne!(root_pn, 0,
            "post-dogyoku position must NOT be checkmate (false proof regression)");
    }

    /// generate_check_moves の結果を brute-force と比較する．
    #[test]
    fn test_check_moves_completeness() {
        use std::collections::BTreeSet;
        let test_positions = [
            // 17手詰めの初期局面(OR node)
            "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1",
            // tsume2
            "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1",
            // tsume3
            "l1k6/9/1pB6/9/9/9/9/9/9 b RGrb4g4s4n3l16p 1",
            // 9te (tsume1)
            "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1",
        ];

        let mut solver = DfPnSolver::default_solver();
        for sfen in &test_positions {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            // Brute-force: 全合法手 → フィルタ
            let brute_checks: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .into_iter()
                .filter(|m| {
                    let c = board.do_move(*m);
                    let gives_check = board.is_in_check(board.turn);
                    board.undo_move(*m, c);
                    gives_check
                })
                .map(|m| m.to_usi())
                .collect();

            let optimized_checks: BTreeSet<String> = solver
                .generate_check_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let missing: BTreeSet<_> = brute_checks.difference(&optimized_checks).collect();
            let extra: BTreeSet<_> = optimized_checks.difference(&brute_checks).collect();

            assert!(
                missing.is_empty() && extra.is_empty(),
                "check moves mismatch for sfen: {}\n  missing: {:?}\n  extra: {:?}\n  brute: {:?}\n  opt: {:?}",
                sfen, missing, extra, brute_checks, optimized_checks
            );
        }
    }

    /// 39手詰め PV 途中の全局面で generate_check_moves が brute-force と一致することを検証．
    ///
    /// 特に ply 22 の P*1g が生成されないバグの回帰テスト．
    #[test]
    fn test_check_moves_completeness_39te_pv() {
        use std::collections::BTreeSet;

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        let mut solver = DfPnSolver::default_solver();
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV の各 OR node (攻め方手番) で check_moves の完全性を検証
        for (ply, &usi) in pv.iter().enumerate() {
            if ply % 2 == 0 {
                // OR node: 攻め方手番 → check_moves を検証
                let brute_checks: BTreeSet<String> =
                    movegen::generate_legal_moves(&mut board)
                        .into_iter()
                        .filter(|m| {
                            let c = board.do_move(*m);
                            let gives_check = board.is_in_check(board.turn);
                            board.undo_move(*m, c);
                            gives_check
                        })
                        .map(|m| m.to_usi())
                        .collect();

                let optimized_checks: BTreeSet<String> = solver
                    .generate_check_moves(&mut board)
                    .iter()
                    .map(|m| m.to_usi())
                    .collect();

                let missing: BTreeSet<_> =
                    brute_checks.difference(&optimized_checks).collect();
                let extra: BTreeSet<_> =
                    optimized_checks.difference(&brute_checks).collect();

                assert!(
                    missing.is_empty() && extra.is_empty(),
                    "check moves mismatch at ply {} (next PV move: {})\n  \
                     missing: {:?}\n  extra: {:?}",
                    ply, usi, missing, extra
                );
            }

            let m = board
                .move_from_usi(usi)
                .unwrap_or_else(|| panic!("invalid USI at ply {}: {}", ply, usi));
            board.do_move(m);
        }
    }

    /// 39手詰め PV の各 AND ノードで defense_moves ⊆ legal_moves を検証する．
    ///
    /// chain 最適化により defense_moves は legal_moves のサブセットになるため，
    /// extra(legal にない手)が空であることのみ検証する．
    /// また，PV 上の応手が defense_moves に含まれることも確認する．
    #[test]
    fn test_defense_moves_subset_39te_pv() {
        use std::collections::BTreeSet;

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        let mut solver = DfPnSolver::default_solver();
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        for (ply, &usi) in pv.iter().enumerate() {
            let m = board
                .move_from_usi(usi)
                .unwrap_or_else(|| panic!("invalid USI at ply {}: {}", ply, usi));
            board.do_move(m);

            if ply % 2 == 0 {
                // 攻め手の後 → AND node (玉方手番)
                if !board.is_in_check(board.turn) {
                    continue;
                }

                let legal_moves: BTreeSet<String> =
                    movegen::generate_legal_moves(&mut board)
                        .iter()
                        .map(|m| m.to_usi())
                        .collect();

                let defense_moves: BTreeSet<String> = solver
                    .generate_defense_moves(&mut board)
                    .iter()
                    .map(|m| m.to_usi())
                    .collect();

                // defense_moves ⊆ legal_moves (不正な手がないこと)
                let extra: BTreeSet<_> =
                    defense_moves.difference(&legal_moves).collect();
                assert!(
                    extra.is_empty(),
                    "defense has illegal moves at ply {} (after {})\n  \
                     extra: {:?}",
                    ply + 1, usi, extra
                );

                // PV の次の応手が defense_moves に含まれること
                if ply + 1 < pv.len() {
                    let next_defense = pv[ply + 1];
                    assert!(
                        defense_moves.contains(next_defense),
                        "PV defense move {} missing from defense_moves at ply {}\n  \
                         defense({}): {:?}",
                        next_defense, ply + 1,
                        defense_moves.len(), defense_moves
                    );
                }
            }
        }
    }

    /// generate_defense_moves と generate_legal_moves の結果を比較する．
    ///
    /// 王手がかかっている局面で，回避手生成(evasion)が
    /// 全合法手のサブセットであり，かつ全合法手を漏れなく含むことを検証する．
    #[test]
    fn test_defense_moves_completeness() {
        use std::collections::BTreeSet;
        // テスト局面: 攻め方が王手した直後の局面をいくつか用意
        let test_positions = [
            // S*4a で王手 → 玉方の応手
            "9/3S1Pk2/9/8R/8B/9/9/9/9 w rb4g2s4n4l17p 2",
            // 飛車で王手(スライディング)
            "9/5Pk2/9/5R3/8B/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // 角で王手
            "9/5Pk2/9/8R/5B3/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // test_tsume_3 の中間局面(R*2a 後)
            "l1k6/R8/1pB6/9/9/9/9/9/9 w rb4g4s4n3l16p 2",
        ];

        let mut solver = DfPnSolver::default_solver();

        for sfen in &test_positions {
            let mut board = Board::new();
            if board.set_sfen(sfen).is_err() {
                continue;
            }

            // まず王手されているか確認
            if !board.is_in_check(board.turn) {
                continue;
            }

            let defense_moves: BTreeSet<String> = solver
                .generate_defense_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let legal_moves: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            // defense_moves は legal_moves のサブセットであること
            let extra: BTreeSet<_> = defense_moves.difference(&legal_moves).collect();
            assert!(
                extra.is_empty(),
                "defense has extra moves not in legal: {:?}\nsfen: {}",
                extra, sfen
            );

            // legal_moves は defense_moves のサブセットであること(漏れがない)
            let missing: BTreeSet<_> = legal_moves.difference(&defense_moves).collect();
            assert!(
                missing.is_empty(),
                "defense is missing legal moves: {:?}\nsfen: {}\ndefense: {:?}\nlegal: {:?}",
                missing, sfen, defense_moves, legal_moves
            );
        }
    }

    /// 17手詰め局面の solve() テスト．
    #[test]
    fn test_tsume_5() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched: _,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    pv.len(),
                    17,
                    "expected 17-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );

                // 攻め方の手順を検証(奇数手)
                // 41銀打，32銀打，11飛成，33角成，同馬，23金打，21銀成，32銀成
                assert_eq!(pv[0], "S*4a", "move 1: 41銀打");
                assert_eq!(pv[2], "S*3b", "move 3: 32銀打");
                assert_eq!(pv[4], "1d1a+", "move 5: 11飛成");
                assert_eq!(pv[6], "1e3c+", "move 7: 33角成");
                assert_eq!(pv[8], "3c2b", "move 9: 同馬");
                assert_eq!(pv[10], "G*2c", "move 11: 23金打");
                // move 12: 何を合駒してもよい
                assert_eq!(pv[12], "3b2a+", "move 13: 21銀成");
                assert_eq!(pv[14], "4a3b+", "move 15: 32銀成");
                // move 17: 22成銀(3b2b) or 22金(2c2b) など複数正解
                assert!(
                    pv[16] == "3b2b" || pv[16] == "2c2b",
                    "move 17: expected 3b2b or 2c2b, got {}",
                    pv[16],
                );
            }
            other => panic!("expected Checkmate for tsume5, got {:?}", other),
        }
    }

    /// `find_shortest = false` で最短手数探索をスキップできることを確認．
    #[test]
    fn test_tsume_5_no_shortest() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched: _,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                // find_shortest = false でも17手詰みが見つかる
                assert_eq!(
                    pv.len(),
                    17,
                    "expected 17-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
            }
            other => panic!("expected Checkmate for tsume5, got {:?}", other),
        }

        // find_shortest = true との比較
        let mut board2 = Board::new();
        board2.set_sfen(sfen).unwrap();
        let mut solver2 = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        solver2.set_find_shortest(true);
        let result2 = solver2.solve(&mut board2);

        if let TsumeResult::Checkmate { nodes_searched: n2, .. } = &result2 {
            if let TsumeResult::Checkmate { nodes_searched: n1, .. } = &result {
                assert!(n1 <= n2, "find_shortest=false should use <= nodes");
            }
        }
    }

    /// 29手詰め(tsume6)．
    ///
    /// 深さ制限時の TT 保存バグの回帰テスト．
    /// PieceType::MAX_HAND_COUNT で保存すると不詰として誤判定されていた．
    #[test]
    fn test_tsume_6_29te() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 50_000_000, 32767, 300);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!("=== tsume6 result: {} moves, {} nodes, prefilter_hits={} ===",
                    pv.len(), nodes_searched, solver.prefilter_hits);
                eprintln!("PV: {}", pv.join(" "));

                // 診断PV抽出は完了後のみ実行(Phase 2 後は TT が巨大化するため省略)

                // 8i7g が含まれているか確認 — 27手詰めになるバグの診断
                if let Some(pos) = pv.iter().position(|m| m == "8i7g") {
                    eprintln!("WARNING: 8i7g found at ply {} — this leads to 27-move mate, not 29", pos);
                }

                assert_eq!(
                    pv.len(),
                    29,
                    "expected 29-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
                // ぴよ将棋で検証済みの正解手順 (後手攻め)
                // 手順8(L*7g)と手順14(G*7h)は合駒 — ソルバーが別の駒を選ぶ可能性あり
                // 手順26(8f9g)は玉の逃げ方で分岐 — 8f8g でも同手数の29手詰め
                // 初手3手は 8f8g+/7h8g/S*7i と S*7i/8h9g/8f8g+ の2解あり
                let prefix1 = [
                    "8f8g+", "7h8g", "S*7i", "8h9g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                let prefix2 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                // 8e8g(不成)も同等に正しい — 弱い駒優先順序変更で出現
                let prefix3 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                // Deep df-pn バイアスにより合駒選択が変化した PV
                let prefix4 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "N*9e", "9f9e", "6i7h", "6g7h",
                    "P*8f", "8g9g", "G*9f", "9g9f", "9d9e", "9f8f",
                    "P*8e",
                ];
                assert!(
                    pv[..25] == prefix1 || pv[..25] == prefix2 || pv[..25] == prefix3 || pv[..25] == prefix4,
                    "PV prefix mismatch (first 25 moves):\n  got:      {}\n  pv1: {}\n  pv2: {}\n  pv3: {}\n  pv4: {}",
                    pv[..25].join(" "),
                    prefix1.join(" "),
                    prefix2.join(" "),
                    prefix3.join(" "),
                    prefix4.join(" "),
                );
                // 8i7g は不正解(27手詰めへの分岐)
                assert!(
                    !pv.contains(&"8i7g".to_string()),
                    "PV must not contain 8i7g (leads to 27-move mate): {}",
                    pv.join(" "),
                );
            }
            other => panic!("expected Checkmate for tsume6, got {:?}", other),
        }
    }

    /// 29手詰め: PNS なし(IDS-MID のみ)のロバストネステスト．
    ///
    /// PNS は浅い詰みの発見に使われ，IDS-MID は深い詰みに使われる．
    /// IDS-MID のみで29手詰めを発見できるか確認し，MID 単体のロバストネスを評価する．
    #[test]
    #[ignore]
    fn test_tsume_6_29te_no_pns() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 120_000_000, 32767, 1200);
        solver.max_nodes = 4;
        solver.attacker = board.turn;
        solver.start_time = std::time::Instant::now();
        let _ = solver.pns_main(&mut board);
        solver.max_nodes = 120_000_000;

        // mid_fallback 内の IDS ステップと root ply の診断を取得するため，
        // diag_ply=0 で root ノードの mid_diag をトリガーする．
        // mid_diag の間隔を 500K に短縮して root での応手別消費を追跡する．
        solver.mid_fallback(&mut board);

        let pk = position_key(&board);
        let att_hand = board.hand[solver.attacker.index()];
        let (root_pn, _, _) = solver.look_up_pn_dn(pk, &att_hand, 31);
        eprintln!("[no_pns] root_pn={} nodes={} time={:.1}s",
            root_pn, solver.nodes_searched, solver.start_time.elapsed().as_secs_f64());

        if root_pn == 0 {
            let pv = solver.extract_pv_limited(&mut board, 10_000);
            let pv_usi: Vec<String> = pv.iter().map(|m| m.to_usi()).collect();
            eprintln!("[no_pns] PV ({} moves): {}", pv_usi.len(), pv_usi.join(" "));
            assert_eq!(pv_usi.len(), 29, "expected 29-move PV, got {} moves", pv_usi.len());
        } else {
            eprintln!("[no_pns] NOT PROVED: rpn={} nodes={}", root_pn, solver.nodes_searched);
        }

        // ply 別効率レポート(解決・未解決共通)
        eprintln!("\n[efficiency] {:>3} {:>10} {:>12} {:>8} {:>8}",
            "ply", "nodes", "iters", "n/iter", "stag");
        for p in 0..64 {
            let n = solver.ply_nodes[p];
            let it = solver.ply_iters[p];
            let stag = solver.ply_stag_penalties[p];
            if n > 0 || it > 0 {
                let ratio = if it > 0 { n as f64 / it as f64 } else { 0.0 };
                eprintln!("[efficiency] {:>3} {:>10} {:>12} {:>8.1} {:>8}",
                    p, n, it, ratio, stag);
            }
        }
        eprintln!();

        // TT コンテンツ分析
        solver.table.dump_content_analysis();

        if root_pn != 0 {
            panic!("IDS-MID only should prove 29te checkmate, got pn={}", root_pn);
        }
    }

    /// 29手詰め PV 逆順解析: PV の手順を進め，各中間局面から解けるか検証．
    /// どの深さで解けなくなるかを特定する．
    #[test]
    fn test_tsume_6_29te_pv_analysis() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        // PV の最初 25 手(テストで検証済みの手順)
        let pv_moves = [
            "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
            "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
            "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
            "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
            "P*8e",
        ];

        // 偶数 ply ごとに(攻め方の手番で)テスト
        // 常に depth=31, 5M nodes, 30s で解けるか確認
        for start_ply in (0..pv_moves.len()).step_by(2) {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            // Play first start_ply moves
            for i in 0..start_ply {
                let m = board.move_from_usi(pv_moves[i])
                    .unwrap_or_else(|| panic!("failed to parse move {} at index {}", pv_moves[i], i));
                board.do_move(m);
            }

            let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 30);
            let result = solver.solve(&mut board);

            match &result {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} SOLVED {}te, {} nodes",
                        start_ply, pv_moves[start_ply], moves.len(), nodes_searched
                    );
                }
                TsumeResult::Unknown { nodes_searched } => {
                    eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} FAILED ({} nodes)",
                        start_ply, pv_moves[start_ply], nodes_searched
                    );
                }
                other => {
                    eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} {:?}",
                        start_ply, pv_moves[start_ply], other
                    );
                }
            }
        }
    }

    /// 29手詰め ply1 応手解析: S*7i 後の各応手から解けるか検証．
    #[test]
    fn test_tsume_6_29te_ply1_analysis() {
        use crate::movegen;
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // Play S*7i (correct first move)
        let first_move = board.move_from_usi("S*7i").unwrap();
        board.do_move(first_move);

        // Generate all defense moves at ply 1
        let defenses = movegen::generate_legal_moves(&mut board);
        eprintln!("[ply1_analysis] {} defense moves after S*7i", defenses.len());

        for def in &defenses {
            let cap = board.do_move(*def);
            // From ply 2 position, try to solve
            let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 30);
            let result = solver.solve(&mut board);
            match &result {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    eprintln!(
                        "[ply1_analysis] defense={:<8} CHECKMATE {}te, {} nodes",
                        def.to_usi(), moves.len(), nodes_searched
                    );
                }
                TsumeResult::NoCheckmate { nodes_searched } => {
                    eprintln!(
                        "[ply1_analysis] defense={:<8} NO_CHECKMATE (refuted), {} nodes",
                        def.to_usi(), nodes_searched
                    );
                }
                TsumeResult::Unknown { nodes_searched } => {
                    eprintln!(
                        "[ply1_analysis] defense={:<8} UNKNOWN (stuck), {} nodes",
                        def.to_usi(), nodes_searched
                    );
                }
                other => {
                    eprintln!(
                        "[ply1_analysis] defense={:<8} {:?}",
                        def.to_usi(), other
                    );
                }
            }
            board.undo_move(*def, cap);
        }
    }

    /// 後手番1手詰め．
    ///
    /// 先手攻め test_tsume_1te の盤面を180度回転+色反転した局面．
    /// 先手玉9九(K)，後手金8七(g)，後手持ち駒:金．
    /// 正解: G*8h(8八金打)で詰み(G*9hも正解)．
    #[test]
    fn test_tsume_1te_gote() {
        // 先手玉 9i, 後手金 8g, 後手持ち駒: g
        let sfen = "9/9/9/9/9/9/1g7/9/K8 w g 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(3, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1, "should be 1-move checkmate, got: {:?}",
                    moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>());
                let usi = moves[0].to_usi();
                assert!(
                    usi == "G*8h" || usi == "G*9h",
                    "1手詰め: G*8h(8八金打) or G*9h(9八金打), got: {}", usi
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 後手番3手詰め．
    ///
    /// 先手攻め test_tsume_3te の盤面を180度回転+色反転した局面．
    /// 先手玉9九(K)，後手飛7七(r)，後手持ち駒:金．
    /// 正解: 7g9g+(9七飛成)，9i8i(8九玉)，G*8h(8八金打) まで3手詰．
    #[test]
    fn test_tsume_3te_gote() {
        // 先手玉 9i, 後手飛 7g, 後手持ち駒: g
        // (test_tsume_3te: 8k/9/6R2/9/.../9 b G 1 の反転)
        let sfen = "9/9/9/9/9/9/2r6/9/K8 w g 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let result = solve_tsume_with_timeout(sfen, Some(7), Some(1_048_576), None, None, None, None, None, None).unwrap();

        let expected = ["7g9g+", "9i8i", "G*8h"];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 3,
                    "expected 3 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 打歩詰め回避のため飛不成が正解のケース(7手詰め)．
    ///
    /// 局面: 先手飛4一，先手桂3三・2六，先手歩3四，
    ///       後手玉2二，後手歩3二・2五
    /// 先手持駒: 桂，歩二
    /// 後手持駒: 飛，角二，金四，銀四，桂，香四，歩十三
    ///
    /// 2一飛成(4a2a+)は龍の利きにより打歩詰めの反則が生じるため不詰．
    /// 2一飛不成(4a2a)なら飛車のまま利きが制限され，7手で詰みが成立する．
    #[test]
    fn test_tsume_uchifuzume_rook_no_promote() {
        let sfen = "5R3/6pk1/6N2/6P2/7p1/7N1/9/9/9 b N2Pr2b4g4sn4l13p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 7,
                    "expected 7-move checkmate, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // 初手は飛車不成(4a2a)でなければならない
                assert_eq!(
                    usi_moves[0], "4a2a",
                    "move 1 must be 4a2a (rook WITHOUT promotion), got: {}",
                    usi_moves[0]
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 2一飛成(龍)だと打歩詰めにより詰みがないことの検証．
    ///
    /// 上記 test_tsume_uchifuzume_rook_no_promote の局面で
    /// 4a2a+(飛成)を指した後の局面は，龍の利きにより
    /// 打歩詰めの反則が避けられず詰まない．
    #[test]
    fn test_uchifuzume_promoted_rook_fails() {
        let sfen = "5R3/6pk1/6N2/6P2/7p1/7N1/9/9/9 b N2Pr2b4g4sn4l13p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // 4a2a+(飛成=龍)を指す
        let promote_move = board.move_from_usi("4a2a+").unwrap();
        board.do_move(promote_move);

        // 龍の局面からは詰まないことを検証
        let mut solver = DfPnSolver::new(31, 2_000_000, 32767);
        let result = solver.solve(&mut board);
        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "4a2a+ (promoted rook) should NOT lead to checkmate due to uchifuzume, got: {:?}",
            result
        );
    }

    /// 馬の往復で千日手となり不詰のケース．
    ///
    /// 局面: 後手玉1二，先手馬3二，先手金3一，先手歩3五・2五，後手歩1三
    /// 先手持駒: なし
    /// 後手持駒: 飛二，角，金三，銀四，桂四，香四，歩十五
    ///
    /// 2一馬，2三玉，3二馬，1二玉 の繰り返しで千日手(連続王手の千日手)．
    /// 攻め方に持ち駒がなく打開手段がないため不詰．
    #[test]
    fn test_no_checkmate_perpetual_check() {
        let sfen = "6G2/6+B1k/8p/9/6PP1/9/9/9/9 b 2rb3g4s4n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (perpetual check by horse), got {:?}",
                other
            ),
        }
    }

    /// 逆王手で不詰のケース．
    ///
    /// 局面: 先手玉2四，先手飛4三，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手桂4二，後手銀3四
    /// 先手持駒: なし
    /// 後手持駒: 飛，角二，金四，銀三，桂三，香二，歩十七
    ///
    /// 4三飛→2三飛成は王手だが，後手3三銀の逆王手(2四の先手玉に対する王手)
    /// により攻め方は王手回避を強いられ，詰みにならない．
    /// 先手に持ち駒がなく他の有効な攻めがないため不詰．
    #[test]
    fn test_no_checkmate_counter_check() {
        let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (counter-check by silver), got {:?}",
                other
            ),
        }
    }

    /// 逆王手がある局面で3四玉と上がって詰むケース．
    ///
    /// 局面: 先手玉2四，先手飛4三，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手銀3四
    /// 先手持駒: なし
    /// 後手持駒: 飛，角二，金四，銀三，桂四，香二，歩十七
    ///
    /// 上記の不詰局面から4二の後手桂を除いた形．
    /// 先手3四玉(銀を取って王手回避しつつ接近)から詰みがある．
    #[test]
    fn test_checkmate_with_counter_check_avoidance() {
        let sfen = "7l1/7k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s4n2l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> =
                    moves.iter().map(|m| m.to_usi()).collect();
                // 5手詰め(複数解あり): 探索順序によりいずれかの PV が返る
                // 合駒の駒種は探索順序依存(弱い駒優先 → 歩合が先に証明される)
                let pv1 = vec!["2d3d", "2b3a", "S*4b", "3a3b", "4c3c+"];
                let pv2 = vec!["2d3d", "R*2c", "2e2c+", "2b1a", "1c1b+"];
                let pv3 = vec!["2d3d", "P*2c", "2e2c+", "2b1a", "1c1b+"];
                let pv_str: Vec<&str> = usi_moves.iter().map(|s| s.as_str()).collect();
                assert!(
                    pv_str == pv1 || pv_str == pv2 || pv_str == pv3,
                    "PV must be one of the known solutions:\n  got:  {}\n  pv1: {}\n  pv2: {}\n  pv3: {}",
                    usi_moves.join(" "),
                    pv1.join(" "),
                    pv2.join(" "),
                    pv3.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate (king captures silver), got {:?}",
                other
            ),
        }
    }

    /// 打歩詰めしかなく不詰のケース．
    ///
    /// 局面: 後手玉1一，後手桂2一，先手金1三
    /// 先手持駒: 歩
    /// 後手持駒: 飛二，角二，金三，銀四，桂三，香四，歩十七
    ///
    /// 1二歩打(P*1b)は王手だが打歩詰めの反則(玉が逃げられない)．
    /// 他に有効な王手がないため不詰．
    #[test]
    fn test_no_checkmate_uchifuzume_only() {
        let sfen = "7nk/9/8G/9/9/9/9/9/9 b P2r2b3g4s3n4l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(100_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (only move is uchifuzume), got {:?}",
                other
            ),
        }
    }

    /// 金の移動合いで不詰になるケース．
    ///
    /// 局面: 後手玉1一，後手金2一，後手歩1二，後手銀1三，
    ///       先手歩4三・2四
    /// 先手持駒: 角，桂
    /// 後手持駒: 飛二，角，金三，銀三，桂三，香四，歩十五
    ///
    /// 角打ちに対して金の移動合い(2一金→1二等)が有効で詰まない．
    #[test]
    fn test_no_checkmate_gold_interposition() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BN2rb3g3s3n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(1_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (gold interposition defense), got {:?}",
                other
            ),
        }
    }

    /// 先手持駒に銀を追加した局面で9手詰めになることのテスト．
    ///
    /// 上記 test_no_checkmate_gold_interposition と同じ盤面だが，
    /// 先手持駒に銀が追加(角，銀，桂)され後手の銀が1枚減っている．
    /// 銀の追加により9手詰めが生じる(金の移動合いが最長抵抗)．
    #[test]
    fn test_tsume_9te_with_silver() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "expected 9-move checkmate, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // 初手: 3三角打
                assert_eq!(usi_moves[0], "B*3c", "move 1: B*3c(3三角打)");
                // 2手目: 金の移動合い(最長抵抗)
                assert_eq!(usi_moves[1], "2a2b", "move 2: g(2a→2b)(金の移動合い)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 金の移動合い(2一金→2二)が回避手に含まれることを検証する．
    ///
    /// B*3c(3三角打)に対して，2一の金を2二に移動して合い駒する手が
    /// 回避手として生成されていることを確認する．
    /// この手が漏れていると不正に短手数で詰みと判定される．
    #[test]
    fn test_gold_interposition_is_legal_defense() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // B*3c(3三角打) で王手
        let b3c = board.move_from_usi("B*3c").unwrap();
        board.do_move(b3c);

        // 後手番: 回避手を生成
        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();

        // 金の移動合い 2a2b(2一金→2二) が含まれること
        assert!(
            usi_defenses.contains(&"2a2b".to_string()),
            "g(2a→2b) gold interposition should be a legal defense, got: {:?}",
            usi_defenses
        );

        // 銀の移動合い 1c2b(1三銀→2二) も含まれること
        assert!(
            usi_defenses.contains(&"1c2b".to_string()),
            "s(1c→2b) silver interposition should be a legal defense, got: {:?}",
            usi_defenses
        );
    }

    /// 金の移動合い後もソルバーが正しく詰みを検出することを検証する．
    ///
    /// B*3c(3三角打)後の金移動合い(2a→2b)に対して，
    /// 先手が銀を持っている場合に7手で詰ませられることを確認する．
    /// 全体としては B*3c, g(2a→2b) + 7手 = 9手詰め．
    #[test]
    fn test_after_gold_interposition_with_silver() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // B*3c → g(2a→2b) を実行
        let b3c = board.move_from_usi("B*3c").unwrap();
        board.do_move(b3c);
        let g2b = board.move_from_usi("2a2b").unwrap();
        board.do_move(g2b);

        // この局面から先手が詰ませられるか
        let mut solver = DfPnSolver::new(31, 2_000_000, 32767);
        let result = solver.solve(&mut board);
        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 7,
                    "after gold interposition, expected 7 more moves, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // B*3c → 2a2b の後の正解手順 (ぴよ将棋で検証済み)
                let expected = [
                    "N*2c", "1a2a", "S*3b", "2a3b", "3c4b+", "3b2a", "4b3a",
                ];
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {}\n  expected: {}",
                    usi_moves.join(" "),
                    expected.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate after gold interposition, got {:?}",
                other
            ),
        }
    }

    /// 39手詰めの高難度テスト(6九への合駒が必要)．
    ///
    /// 後手の合駒選択が鍵となる局面．6九に合駒を打つ必要があるが，
    /// 歩・桂・香は打てない(二歩・行き所のない駒)ため，
    /// 金・銀・飛・角のみが候補となる．
    /// 後手の最善応手(最長抵抗)でのみ39手詰めとなる．
    #[test]
    #[ignore] // 約42万ノード / 5秒で解ける重量テスト．明示的に `cargo test -- --ignored` で実行．
    fn test_tsume_39te_aigoma() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver =
            DfPnSolver::with_timeout(41, 10_000_000, 32767, 60);
        solver.set_find_shortest(false);
        #[cfg(feature = "tt_diag")]
        {
            solver.diag_ply = 35;
            solver.diag_max_iterations = 0; // don't break loop
        }
        let start = Instant::now();
        let result = solver.solve(&mut board);
        let elapsed = start.elapsed();
        eprintln!("39te: {} nodes, {:.1}s, max_ply={}, prefilter_hits={}",
            solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply,
            solver.prefilter_hits);
        #[cfg(feature = "profile")]
        {
            solver.sync_tt_profile();
            eprintln!("{}", solver.profile_stats);
        }

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched: _,
            } => {
                let pv: Vec<String> =
                    moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    pv.len(),
                    39,
                    "expected 39-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
                // ぴよ将棋で検証済みの正解手順
                let expected = [
                    "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
                    "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
                    "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
                    "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
                    "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
                    "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
                    "2g2h", "3i4i", "2h4h",
                ];
                assert_eq!(
                    pv, expected,
                    "PV mismatch:\n  got:      {}\n  expected: {}",
                    pv.join(" "),
                    expected.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate for 39te aigoma, got {:?}",
                other
            ),
        }
    }

    /// 39手詰め直接MID テスト(PNS/IDS なし)．
    ///
    /// main ブランチと同様に単一 MID 呼び出しで解けるか確認する．
    #[test]
    #[ignore]
    fn test_tsume_39te_direct_mid() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver =
            DfPnSolver::with_timeout(63, 10_000_000, 32767, 60);
        solver.set_find_shortest(false);

        // 直接 MID を呼び出す(PNS/IDS をバイパス)
        solver.table.clear();
        solver.nodes_searched = 0;
        solver.max_ply = 0;
        solver.path.clear();
        solver.killer_table.clear();
        solver.start_time = Instant::now();
        solver.timed_out = false;
        solver.next_gc_check = 100_000;
        solver.attacker = board.turn;
        solver.mid(&mut board, INF - 1, INF - 1, 0, true);

        let (root_pn, _root_dn) = solver.look_up_board(&board);
        let start = Instant::now();
        let elapsed = start.elapsed();
        eprintln!("39te_direct_mid: {} nodes, {:.1}s, max_ply={}, prefilter={}  pn={}",
            solver.nodes_searched, solver.start_time.elapsed().as_secs_f64(),
            solver.max_ply, solver.prefilter_hits, root_pn);
        assert_eq!(root_pn, 0, "expected pn=0 (proved) for 39te direct MID");
    }

    /// 39手詰めサブ問題実験: PV 終盤側から逆順に詰み探索ノード数を計測し，
    /// 全体を解くのに必要な予算を推定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_budget_estimation() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV: 攻め手(奇数)と玉方(偶数)
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" 39手詰めサブ問題予算推定実験(終盤→序盤)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{:<6} {:<14} {:<10} {:<10} {:<12} {}",
            "Ply", "Nodes", "Time(s)", "MaxPly", "Result", "Remaining");

        // PV を偶数手ずつ進めた局面(攻め番=ORノード)を全て事前構築
        let mut positions: Vec<(usize, Board)> = Vec::new();
        let mut sub_board = board.clone();
        // ply 0
        positions.push((0, sub_board.clone()));
        for ply_start in (0..38).step_by(2) {
            let m1 = sub_board.move_from_usi(pv_usi[ply_start]).unwrap();
            sub_board.do_move(m1);
            let m2 = sub_board.move_from_usi(pv_usi[ply_start + 1]).unwrap();
            sub_board.do_move(m2);
            positions.push((ply_start + 2, sub_board.clone()));
        }

        // 終盤側(簡単)から序盤側(困難)へ逆順で解く
        // 解けなくなったら停止
        positions.reverse();

        let node_limit: u64 = 50_000_000; // 5000万ノード上限
        let timeout = 60; // 60秒

        for (ply, pos) in &positions {
            let remaining_moves = 39 - ply;
            let depth = (remaining_moves + 2).min(41) as u32;

            let mut test_board = pos.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, node_limit, 32767, timeout,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            eprintln!("{:<6} {:<14} {:<10.2} {:<10} {:<12} {}手",
                ply, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, result_str, remaining_moves);

            // 解けなくなったら局面のSFENを出力して停止
            if !solved {
                eprintln!("--- ply {} で未解決 ---", ply);
                eprintln!("  SFEN: {}", pos.sfen());
                eprintln!("  PV残り: {:?}", &pv_usi[*ply..]);

                // 深さを大きくして再試行
                for &d in &[25u32, 31, 41, 51] {
                    let mut test_board2 = pos.clone();
                    let mut solver2 = DfPnSolver::with_timeout(
                        d, 50_000_000, 32767, 60,
                    );
                    solver2.set_find_shortest(false);

                    let start2 = Instant::now();
                    let result2 = solver2.solve(&mut test_board2);
                    let elapsed2 = start2.elapsed();

                    let result_str2 = match &result2 {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } =>
                            "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } =>
                            "NoMate".to_string(),
                        TsumeResult::Unknown { .. } =>
                            "Unknown".to_string(),
                    };

                    eprintln!("  depth={:<4} {:<14} {:<10.2} {:<10} {}",
                        d, solver2.nodes_searched, elapsed2.as_secs_f64(),
                        solver2.max_ply, result_str2);
                }

                // 1手進めた局面(ply+1, 玉方手番=ANDノード)も試す
                eprintln!("\n  --- ply {} (攻め手1手目 {} 適用後、玉方手番) ---", ply, pv_usi[*ply]);
                let mut after1 = pos.clone();
                let m1 = after1.move_from_usi(pv_usi[*ply]).unwrap();
                after1.do_move(m1);
                eprintln!("  SFEN after {}: {}", pv_usi[*ply], after1.sfen());

                // PV の最後の手から逆順に、1手ずつ戻って解けるポイントを探す
                eprintln!("\n  --- 1手ずつ PV を遡り解ける境界を特定 ---");
                // ply+1 (玉方手番後) から ply+16 まで奇数手のみ(OR局面)
                let mut walk_board = pos.clone();
                for step in 0..remaining_moves {
                    let mv_str = pv_usi[*ply + step];
                    let mv = walk_board.move_from_usi(mv_str).unwrap();
                    walk_board.do_move(mv);

                    // OR局面(攻め方手番)のみ詰み探索
                    if step % 2 == 0 {
                        continue; // step=0 で玉方手番、step=1 で攻め方手番
                    }

                    let sub_remaining = remaining_moves - step - 1;
                    if sub_remaining == 0 { break; }
                    let sub_depth = (sub_remaining + 2).min(41) as u32;

                    let mut sub_board = walk_board.clone();
                    let mut sub_solver = DfPnSolver::with_timeout(
                        sub_depth, 50_000_000, 32767, 60,
                    );
                    sub_solver.set_find_shortest(false);
                    let sub_result = sub_solver.solve(&mut sub_board);

                    let sub_result_str = match &sub_result {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } =>
                            "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } =>
                            "NoMate".to_string(),
                        TsumeResult::Unknown { .. } =>
                            "Unknown".to_string(),
                    };

                    eprintln!("  ply{}+{} {:<14} {:<12} rem={}手 SFEN: {}",
                        ply, step + 1, sub_solver.nodes_searched, sub_result_str,
                        sub_remaining, walk_board.sfen());
                }

                // P*1g 後の局面(玉方手番)の合法手を全列挙
                eprintln!("\n  --- P*1g 後の玉方応手分析 ---");
                let mut after_drop = pos.clone();
                let mv_drop = after_drop.move_from_usi("P*1g").unwrap();
                after_drop.do_move(mv_drop);

                let legal_moves = movegen::generate_legal_moves(&mut after_drop);
                eprintln!("  合法手数: {}", legal_moves.len());
                for lm in &legal_moves {
                    let mut after_resp = after_drop.clone();
                    after_resp.do_move(*lm);

                    let mut sub_board = after_resp.clone();
                    let mut sub_solver = DfPnSolver::with_timeout(
                        19, 50_000_000, 32767, 60,
                    );
                    sub_solver.set_find_shortest(false);
                    let sub_result = sub_solver.solve(&mut sub_board);

                    let sub_result_str = match &sub_result {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                        TsumeResult::Unknown { .. } => "Unknown".to_string(),
                    };
                    eprintln!("  {} {:<14} {}", lm.to_usi(), sub_solver.nodes_searched, sub_result_str);
                }

                // 直接診断: depth=19 で solve 後、TT 内のルートエントリをダンプ
                let att_hand22 = pos.hand[Color::Black.index()];
                {
                    let mut test_board = pos.clone();
                    let mut solver = DfPnSolver::with_timeout(19, 50_000_000, 32767, 60);
                    solver.set_find_shortest(false);
                    let result = solver.solve(&mut test_board);

                    let result_str = match &result {
                        TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                        TsumeResult::Unknown { .. } => "Unknown".to_string(),
                    };
                    eprintln!("  depth=19 result: {} nodes={}", result_str, solver.nodes_searched);

                    // TT ルートの全エントリをダンプ
                    let pk = position_key(pos);
                    if let Some(entries) = solver.table.tt.get(&pk) {
                        eprintln!("  TT entries for root (count={})", entries.len());
                        for (i, e) in entries.iter().enumerate() {
                            eprintln!("    [{}] pn={} dn={} remaining={} path_dep={} hand={:?} src={}",
                                i, e.pn, e.dn, e.remaining, e.path_dependent,
                                e.hand, e.source);
                        }
                    } else {
                        eprintln!("  TT: no entries for root");
                    }

                    // remaining=0 vs remaining=19 の look_up 結果
                    let (p0, d0, _) = solver.table.look_up(pk, &att_hand22, 0);
                    let (p19, d19, _) = solver.table.look_up(pk, &att_hand22, 19);
                    eprintln!("  look_up(remaining=0):  pn={} dn={}", p0, d0);
                    eprintln!("  look_up(remaining=19): pn={} dn={}", p19, d19);
                }

                break;
            }
        }

        eprintln!("{}", "=".repeat(80));
    }

    /// 39手詰め逆順サブ問題: 1M ノード / 180 秒で各 OR ノードから解き，
    /// 解けなくなった境界を特定する．解けない局面ではANDノードの各応手の
    /// 探索コスト内訳を報告する．
    #[test]
    fn test_tsume_39te_backward_1m() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_backward_1m.log";
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        let node_limit: u64 = 1_000_000;
        let timeout: u64 = 180;

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め逆順サブ問題 (1M nodes / 180s)").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, "{:<6} {:<10} {:<14} {:<10} {:<10} {:<10} {}",
            "Ply", "Remain", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(90)).unwrap();

        // PV を偶数手ずつ進めた局面(攻め番=ORノード)を全て事前構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        let mut positions: Vec<(usize, Board)> = Vec::new();
        positions.push((0, board.clone()));
        for ply_start in (0..38).step_by(2) {
            let m1 = board.move_from_usi(pv[ply_start]).unwrap();
            board.do_move(m1);
            let m2 = board.move_from_usi(pv[ply_start + 1]).unwrap();
            board.do_move(m2);
            positions.push((ply_start + 2, board.clone()));
        }

        // 終盤(簡単)→序盤(困難)の逆順
        positions.reverse();

        let mut first_unsolved_ply: Option<usize> = None;

        for (ply, pos) in &positions {
            let remaining = 39 - ply;
            let depth = (remaining + 2).min(41) as u32;

            let mut test_board = pos.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, node_limit, 32767, timeout,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<6} {:<10} {:<14} {:<10.2} {:<10} {:<10} {}",
                ply, remaining, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, solver.table.len(), result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "       TT entries: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            if !solved {
                first_unsolved_ply = Some(*ply);

                // --- ANDノードの各応手コスト分析 ---
                // PV の攻め手を1手進めた後(ANDノード)の応手を調べる
                if *ply < pv.len() {
                    let attack_move = pv[*ply];
                    let sub_remaining = remaining - 1; // 攻め手1手分消費
                    if sub_remaining == 0 { continue; }

                    let mut after_attack = pos.clone();
                    let m = after_attack.move_from_usi(attack_move).unwrap();
                    after_attack.do_move(m);

                    writeln!(out, "\n  --- AND node analysis: after {} (ply {}) ---",
                        attack_move, ply + 1).unwrap();
                    writeln!(out, "  SFEN: {}", after_attack.sfen()).unwrap();

                    // 応手一覧
                    let mut defense_solver = DfPnSolver::default_solver();
                    let defenses = defense_solver.generate_defense_moves(&mut after_attack);
                    writeln!(out, "  Defense moves: {} (PV: {})", defenses.len(),
                        if *ply + 1 < pv.len() { pv[*ply + 1] } else { "N/A" }).unwrap();

                    writeln!(out, "  {:<12} {:<14} {:<10} {:<10} {:<10} {}",
                        "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

                    for def_mv in &defenses {
                        let mut after_def = after_attack.clone();
                        after_def.do_move(*def_mv);

                        // 守り手を指した後 → OR ノード(攻め番)
                        let def_remaining = sub_remaining - 1;
                        if def_remaining == 0 {
                            writeln!(out, "  {:<12} --- (remaining=0)", def_mv.to_usi()).unwrap();
                            continue;
                        }
                        let sub_depth = (def_remaining + 2).min(41) as u32;
                        // 応手あたりのノード予算: 全体の1/4か100Kの大きい方
                        let per_move_budget = (node_limit / 4).max(100_000);

                        let mut sub_board = after_def.clone();
                        let mut sub_solver = DfPnSolver::with_timeout(
                            sub_depth, per_move_budget, 32767, 30,
                        );
                        sub_solver.set_find_shortest(false);

                        let sub_start = Instant::now();
                        let sub_result = sub_solver.solve(&mut sub_board);
                        let sub_elapsed = sub_start.elapsed();

                        let sub_result_str = match &sub_result {
                            TsumeResult::Checkmate { moves, .. } =>
                                format!("Mate({})", moves.len()),
                            TsumeResult::CheckmateNoPv { .. } =>
                                "MateNoPV".to_string(),
                            TsumeResult::NoCheckmate { .. } =>
                                "NoMate".to_string(),
                            TsumeResult::Unknown { .. } =>
                                "Unknown".to_string(),
                        };
                        let marker = if *ply + 1 < pv.len() && def_mv.to_usi() == pv[*ply + 1] {
                            " ← PV"
                        } else { "" };
                        writeln!(out, "  {:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                            def_mv.to_usi(), sub_solver.nodes_searched,
                            sub_elapsed.as_secs_f64(), sub_solver.max_ply,
                            sub_solver.table.len(), sub_result_str, marker).unwrap();

                        #[cfg(feature = "tt_diag")]
                        {
                            let proven = sub_solver.table.count_proven();
                            let disproven = sub_solver.table.count_disproven();
                            let intermediate = sub_solver.table.count_intermediate();
                            let total = sub_solver.table.total_entries();
                            writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                                total, proven, disproven, intermediate).unwrap();
                        }
                    }
                }
                break; // 最初の未解決局面の分析後に停止
            }
        }

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        if let Some(ply) = first_unsolved_ply {
            writeln!(out, "境界: ply {} (残り{}手) で 1M ノードでは解けない",
                ply, 39 - ply).unwrap();
        } else {
            writeln!(out, "全局面 1M ノード以内で解決").unwrap();
        }
            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: /tmp/tsume_39te_backward_1m.log");
    }

    /// 39手詰めの必要ノード数を推定する．
    ///
    /// 方針: ply 24 境界の 4 Unknown 応手(1g1f, N*6g, P*7g, N*7g)を
    /// 個別に 1M ノードで解き，応手別コストの合計から推定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_budget_estimation() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_budget_est.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め予算推定: 境界ply の応手別 1M 個別分析").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // Phase 1 結果(backward_1m より)を記載
        writeln!(out, "\n--- Phase 1 結果サマリー (backward_1m) ---").unwrap();
        writeln!(out, "ply 26 (残り13): 104 nodes → Mate(13)").unwrap();
        writeln!(out, "ply 24 (残り15): 473K nodes → Unknown (境界)").unwrap();
        writeln!(out, "ply 22-4: 全て Unknown at 1M").unwrap();

        // Phase 2: ply 24 の AND ノード(5g6f後)の各応手を 1M で個別分析
        writeln!(out, "\n--- Phase 2: ply 24 境界の応手別 1M 分析 ---").unwrap();

        // ply 24 の局面を構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }
        // 攻め手 5g6f を指して AND ノードへ
        let attack_m = board.move_from_usi(pv[24]).unwrap(); // 5g6f
        board.do_move(attack_m);
        writeln!(out, "AND node after 5g6f (ply 25)").unwrap();
        writeln!(out, "SFEN: {}", board.sfen()).unwrap();

        let mut defense_solver = DfPnSolver::default_solver();
        let defenses = defense_solver.generate_defense_moves(&mut board);
        writeln!(out, "Defense moves: {} (PV: 1g1h)\n", defenses.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        let mut total_nodes: u64 = 0;
        let mut solved_count = 0;
        let mut unknown_moves: Vec<(String, u64)> = Vec::new();

        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            // 残り: 39 - 24 - 1(攻め) - 1(受け) = 13 手
            let def_remaining = 13;
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let (sub_result_str, sub_solved) = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };
            let marker = if def_mv.to_usi() == "1g1h" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                sub_result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            total_nodes += nodes;
            if sub_solved {
                solved_count += 1;
            } else {
                unknown_moves.push((def_mv.to_usi(), nodes));
            }
        }

        writeln!(out, "{}", "-".repeat(80)).unwrap();
        writeln!(out, "合計 nodes: {}, 解決: {}/{}, Unknown: {}",
            total_nodes, solved_count, defenses.len(), unknown_moves.len()).unwrap();

        if unknown_moves.is_empty() {
            writeln!(out, "\n→ ply 24 推定必要ノード ≈ {} ({:.1}M)",
                total_nodes, total_nodes as f64 / 1_000_000.0).unwrap();
        } else {
            writeln!(out, "\n→ 1M/応手でも未解決: {:?}", unknown_moves).unwrap();
        }

        // Phase 3: ply 22 の AND ノード(P*1g後)の各応手分析
        writeln!(out, "\n--- Phase 3: ply 22 の応手別 1M 分析 ---").unwrap();

        let mut board22 = Board::new();
        board22.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board22.move_from_usi(pv[i]).unwrap();
            board22.do_move(m);
        }
        // 攻め手 P*1g を指して AND ノードへ
        let attack_m22 = board22.move_from_usi(pv[22]).unwrap(); // P*1g
        board22.do_move(attack_m22);
        writeln!(out, "AND node after P*1g (ply 23)").unwrap();
        writeln!(out, "SFEN: {}", board22.sfen()).unwrap();

        let defenses22 = defense_solver.generate_defense_moves(&mut board22);
        writeln!(out, "Defense moves: {} (PV: 1f1g)\n", defenses22.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        let mut total22: u64 = 0;
        let mut solved22 = 0;
        let mut unknown22: Vec<(String, u64)> = Vec::new();

        for def_mv in &defenses22 {
            let mut after_def = board22.clone();
            after_def.do_move(*def_mv);

            // 残り: 39 - 22 - 1(攻め) - 1(受け) = 15 手
            let def_remaining = 15;
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let (sub_result_str, sub_solved) = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };
            let marker = if def_mv.to_usi() == "1f1g" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                sub_result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            total22 += nodes;
            if sub_solved {
                solved22 += 1;
            } else {
                unknown22.push((def_mv.to_usi(), nodes));
            }
        }

        writeln!(out, "{}", "-".repeat(70)).unwrap();
        writeln!(out, "合計 nodes: {}, 解決: {}/{}, Unknown: {}",
            total22, solved22, defenses22.len(), unknown22.len()).unwrap();

        if unknown22.is_empty() {
            writeln!(out, "→ ply 22 推定必要ノード ≈ {} ({:.1}M)",
                total22, total22 as f64 / 1_000_000.0).unwrap();
        } else {
            writeln!(out, "→ 1M/応手でも未解決: {:?}", unknown22).unwrap();
        }

        // Phase 4: 4 Unknown 応手の再帰分解(1レベル深い)
        // 各 Unknown 応手の後の OR ノード → 攻め手 → AND ノードの応手数を調査
        writeln!(out, "\n--- Phase 4: Unknown 応手の再帰分解 ---").unwrap();

        let unknown_defenses = ["1g1f", "N*6g", "P*7g", "N*7g"];

        for &def_usi in &unknown_defenses {
            let mut after_def = board.clone();
            let def_m = after_def.move_from_usi(def_usi).unwrap();
            after_def.do_move(def_m);

            writeln!(out, "\n--- {} 後 (OR node, 攻め番) ---", def_usi).unwrap();
            writeln!(out, "SFEN: {}", after_def.sfen()).unwrap();

            let attacks = defense_solver.generate_check_moves(&mut after_def);
            writeln!(out, "Attack moves: {}", attacks.len()).unwrap();

            writeln!(out, "{:<12} {:<8} {:<14} {:<10} {}",
                "Attack", "#Def", "Nodes", "Time(s)", "Result").unwrap();

            let mut def_total: u64 = 0;
            let mut def_unknown = 0;

            for atk in &attacks {
                // AND ノードの応手数を数える
                let mut count_board = after_def.clone();
                count_board.do_move(*atk);
                let and_defenses = defense_solver.generate_defense_moves(&mut count_board);
                let num_def = and_defenses.len();

                // 各攻め手の AND サブ問題を 100K で試行
                let sub_remaining = 11; // 13 - 2 (def + atk)
                let sub_depth = (sub_remaining + 2).min(41) as u32;

                let mut sub_solver = DfPnSolver::with_timeout(
                    sub_depth, 100_000, 32767, 30,
                );
                sub_solver.set_find_shortest(false);

                let sub_start = Instant::now();
                let sub_result = sub_solver.solve(&mut count_board);
                let sub_elapsed = sub_start.elapsed();
                let nodes = sub_solver.nodes_searched;

                let (result_str, solved) = match &sub_result {
                    TsumeResult::Checkmate { moves, .. } =>
                        (format!("Mate({})", moves.len()), true),
                    TsumeResult::CheckmateNoPv { .. } =>
                        ("MateNoPV".to_string(), true),
                    TsumeResult::NoCheckmate { .. } =>
                        ("NoMate".to_string(), false),
                    TsumeResult::Unknown { .. } =>
                        ("Unknown".to_string(), false),
                };
                writeln!(out, "{:<12} {:<8} {:<14} {:<10.2} {}",
                    atk.to_usi(), num_def, nodes, sub_elapsed.as_secs_f64(),
                    result_str).unwrap();

                def_total += nodes;
                if !solved { def_unknown += 1; }
            }
            writeln!(out, "合計: {} nodes, Unknown 攻め手: {}/{}",
                def_total, def_unknown, attacks.len()).unwrap();
        }

        // Phase 5: ply 20 の分析(P*1f 後)
        writeln!(out, "\n--- Phase 5: ply 20 の応手別 1M 分析 ---").unwrap();
        let mut board20 = Board::new();
        board20.set_sfen(sfen).unwrap();
        for i in 0..20 {
            let m = board20.move_from_usi(pv[i]).unwrap();
            board20.do_move(m);
        }
        let attack_m20 = board20.move_from_usi(pv[20]).unwrap(); // P*1f
        board20.do_move(attack_m20);
        writeln!(out, "AND node after P*1f (ply 21)").unwrap();
        writeln!(out, "SFEN: {}", board20.sfen()).unwrap();

        let defenses20 = defense_solver.generate_defense_moves(&mut board20);
        writeln!(out, "Defense moves: {} (PV: 1e1f)\n", defenses20.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        for def_mv in &defenses20 {
            let mut after_def = board20.clone();
            after_def.do_move(*def_mv);
            let def_remaining = 17; // 39 - 20 - 1 - 1 = 17
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let result_str = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } =>
                    "NoMate".to_string(),
                TsumeResult::Unknown { .. } =>
                    "Unknown".to_string(),
            };
            let marker = if def_mv.to_usi() == "1e1f" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // 推定サマリー
        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        writeln!(out, " 推定サマリー").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: {}", out_path);
    }

    /// 39手詰め問題の必要予算を段階的に推定する．
    ///
    /// backward_1m の結果を踏まえ，ply 24 の未解決応手を
    /// 段階的に予算増加して解き，ply 間のコスト成長率から
    /// 全体の必要予算を外挿する．
    #[test]
    #[ignore]
    fn test_tsume_39te_budget_scaling() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_budget_scaling.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め予算スケーリング推定").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // ===== Phase A: ply 24 未解決応手の予算スケーリング =====
        writeln!(out, "\n### Phase A: ply 24 未解決応手の段階的予算増加").unwrap();
        writeln!(out, "ply 25 AND node (after 5g6f) → 4 Unknown defense moves\n").unwrap();

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }
        let attack_m = board.move_from_usi(pv[24]).unwrap(); // 5g6f
        board.do_move(attack_m);

        let unknown_defenses = ["1g1f", "N*6g", "P*7g", "N*7g"];
        let budgets: &[u64] = &[1_000_000, 5_000_000, 10_000_000];

        for &def_usi in &unknown_defenses {
            writeln!(out, "--- {} ---", def_usi).unwrap();
            writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
                "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

            let def_m = board.move_from_usi(def_usi).unwrap();

            for &budget in budgets {
                let mut after_def = board.clone();
                after_def.do_move(def_m);

                // depth=41(最大)を使用: 非PV変化では PV より長い詰みが存在しうる
                let sub_depth = 41_u32;

                let mut sub_solver = DfPnSolver::with_timeout(
                    sub_depth, budget, 32767, 600,
                );
                sub_solver.set_find_shortest(false);

                let sub_start = Instant::now();
                let sub_result = sub_solver.solve(&mut after_def);
                let sub_elapsed = sub_start.elapsed();

                let (result_str, solved) = match &sub_result {
                    TsumeResult::Checkmate { moves, .. } =>
                        (format!("Mate({})", moves.len()), true),
                    TsumeResult::CheckmateNoPv { .. } =>
                        ("MateNoPV".to_string(), true),
                    TsumeResult::NoCheckmate { .. } =>
                        ("NoMate".to_string(), false),
                    TsumeResult::Unknown { .. } =>
                        ("Unknown".to_string(), false),
                };

                writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                    format!("{}M", budget / 1_000_000),
                    sub_solver.nodes_searched,
                    sub_elapsed.as_secs_f64(),
                    sub_solver.max_ply,
                    sub_solver.table.len(),
                    result_str).unwrap();

                #[cfg(feature = "tt_diag")]
                {
                    let proven = sub_solver.table.count_proven();
                    let disproven = sub_solver.table.count_disproven();
                    let intermediate = sub_solver.table.count_intermediate();
                    let total = sub_solver.table.total_entries();
                    writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                        total, proven, disproven, intermediate).unwrap();
                }

                if solved { break; } // 解けたらこの応手は終了
            }
            writeln!(out).unwrap();
        }

        // ===== Phase B: ply 24 全体を解くのに必要な予算 =====
        writeln!(out, "### Phase B: ply 24 全体の予算スケーリング").unwrap();
        writeln!(out, "ply 24 OR node を段階的予算で直接解く\n").unwrap();

        let ply24_budgets: &[u64] = &[5_000_000, 10_000_000, 50_000_000];

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

        let mut board24 = Board::new();
        board24.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board24.move_from_usi(pv[i]).unwrap();
            board24.do_move(m);
        }

        for &budget in ply24_budgets {
            let mut test_board = board24.clone();
            let depth = 41_u32; // 最大深さ: 非PV変化で長い手順が存在
            let mut solver = DfPnSolver::with_timeout(
                depth, budget, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, _solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                format!("{}M", budget / 1_000_000),
                solver.nodes_searched,
                elapsed.as_secs_f64(),
                solver.max_ply,
                solver.table.len(),
                result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // ===== Phase C: ply 22 全体の予算スケーリング =====
        writeln!(out, "\n### Phase C: ply 22 全体の予算スケーリング").unwrap();
        writeln!(out, "ply 22 OR node を段階的予算で直接解く\n").unwrap();

        let ply22_budgets: &[u64] = &[5_000_000, 10_000_000, 50_000_000];

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

        let mut board22 = Board::new();
        board22.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board22.move_from_usi(pv[i]).unwrap();
            board22.do_move(m);
        }

        for &budget in ply22_budgets {
            let mut test_board = board22.clone();
            let depth = 41_u32; // 最大深さ
            let mut solver = DfPnSolver::with_timeout(
                depth, budget, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, _solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                format!("{}M", budget / 1_000_000),
                solver.nodes_searched,
                elapsed.as_secs_f64(),
                solver.max_ply,
                solver.table.len(),
                result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // ===== Phase D: ply 20 以前の推定 =====
        writeln!(out, "\n### Phase D: ply 20, 18, 16 の予算スケーリング").unwrap();
        writeln!(out, "各 OR node を 50M で解く\n").unwrap();

        for target_ply in [20_usize, 18, 16] {
            let mut board_t = Board::new();
            board_t.set_sfen(sfen).unwrap();
            for i in 0..target_ply {
                let m = board_t.move_from_usi(pv[i]).unwrap();
                board_t.do_move(m);
            }

            let remaining = 39 - target_ply;
            let depth = 41_u32; // 最大深さ

            let mut test_board = board_t.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, 50_000_000, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } =>
                    "NoMate".to_string(),
                TsumeResult::Unknown { .. } =>
                    "Unknown".to_string(),
            };

            writeln!(out, "ply {:<4} remaining={:<4} nodes={:<14} time={:<10.2}s maxply={:<4} TT_pos={:<10} {}",
                target_ply, remaining, solver.nodes_searched,
                elapsed.as_secs_f64(), solver.max_ply,
                solver.table.len(), result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "         TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();

            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: {}", out_path);
    }

    /// ply 24 のノード数急増を診断する．
    ///
    /// ply 25 ANDノード(5g6f 後)の合駒フィルタ(futile/chain)分類と，
    /// Unknown となった4応手(1g1f, N*6g, P*7g, N*7g)の探索構造を調査する．
    /// 各Unknownの応手について，攻め手が取り進んだ後の再帰的なチェーン構造を
    /// 深さ2まで展開して報告する．
    #[test]
    fn test_ply24_diagnostic() {
        use std::io::Write;
        let out_path = "/tmp/ply24_diagnostic.log";
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        // ply 25 ANDノード: 5g6f 後の局面
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";
        let mut board = Board::new();
        board.set_sfen(and_sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent(); // Black

        let king_sq = board.king_square(defender).unwrap();
        let mut solver = DfPnSolver::default_solver();
        let checker_sq = board.compute_checkers_at(king_sq, attacker).lsb().unwrap();

        writeln!(out, "=== Ply 24 Diagnostic ===").unwrap();
        writeln!(out, "King: {}{}  Checker: {}{}",
            9 - king_sq.col(), (b'a' + king_sq.row()) as char,
            9 - checker_sq.col(), (b'a' + checker_sq.row()) as char).unwrap();

        // between_bb
        let between = attack::between_bb(checker_sq, king_sq);
        write!(out, "Between squares:").unwrap();
        for sq in between {
            write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap();
        }
        writeln!(out).unwrap();

        // futile/chain 分類
        let (futile, chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );
        write!(out, "Futile squares:").unwrap();
        for sq in futile { write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap(); }
        writeln!(out).unwrap();
        write!(out, "Chain squares:").unwrap();
        for sq in chain { write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap(); }
        writeln!(out).unwrap();
        write!(out, "Normal squares:").unwrap();
        for sq in between {
            if !futile.contains(sq) && !chain.contains(sq) {
                write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap();
            }
        }
        writeln!(out).unwrap();

        // 防御手一覧
        let defenses = solver.generate_defense_moves(&mut board);
        writeln!(out, "\nDefense moves ({}):", defenses.len()).unwrap();
        for m in &defenses {
            writeln!(out, "  {}", m.to_usi()).unwrap();
        }

        // 問題の4応手 + 比較用に解ける応手を分析
        let targets = ["1g1f", "N*6g", "P*7g", "N*7g", "L*6g", "B*7g"];

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        writeln!(out, "=== Unknown応手の探索構造分析 ===").unwrap();

        for &target_usi in &targets {
            writeln!(out, "\n--- {} ---", target_usi).unwrap();

            // 応手を適用
            let mut after_def = board.clone();
            let def_move = after_def.move_from_usi(target_usi).unwrap();
            after_def.do_move(def_move);
            writeln!(out, "SFEN after: {}", after_def.sfen()).unwrap();

            // この局面(ORノード)で攻め手を生成
            let attack_solver = DfPnSolver::default_solver();
            let attacks = attack_solver.generate_check_moves(&mut after_def);
            writeln!(out, "Attack moves ({}):", attacks.len()).unwrap();

            // 各攻め手について，1手進めた後のANDノードを簡易分析
            for atk in &attacks {
                let mut after_atk = after_def.clone();
                after_atk.do_move(*atk);

                // この後の防御手数をカウント
                let mut sub_solver = DfPnSolver::default_solver();
                let sub_defenses = sub_solver.generate_defense_moves(&mut after_atk);

                // チェーン構造の確認: 飛び駒の王手ならbetween/futile/chainを表示
                let sub_king_sq = match after_atk.king_square(after_atk.turn()) {
                    Some(sq) => sq,
                    None => {
                        writeln!(out, "  {} → no king (capture?), defenses={}", atk.to_usi(), sub_defenses.len()).unwrap();
                        continue;
                    }
                };
                let sub_attacker = after_atk.turn().opponent();
                let sub_checkers = after_atk.compute_checkers_at(sub_king_sq, sub_attacker);
                if sub_checkers.is_empty() {
                    writeln!(out, "  {} → not check, defenses={}", atk.to_usi(), sub_defenses.len()).unwrap();
                    continue;
                }
                let sub_checker_sq = sub_checkers.lsb().unwrap();
                let sub_sliding = sub_solver.find_sliding_checker(&after_atk, sub_king_sq, sub_attacker);
                let chain_info = if sub_sliding.is_some() {
                    let sub_between = attack::between_bb(sub_checker_sq, sub_king_sq);
                    let (sf, sc) = sub_solver.compute_futile_and_chain_squares(
                        &after_atk, &sub_between, sub_king_sq, sub_checker_sq,
                        after_atk.turn(), sub_attacker,
                    );
                    format!("between={} futile={} chain={} normal={}",
                        sub_between.count(), sf.count(), sc.count(),
                        sub_between.count() - sf.count() - sc.count())
                } else {
                    "non-sliding".to_string()
                };

                writeln!(out, "  {} → defenses={} [{}]",
                    atk.to_usi(), sub_defenses.len(), chain_info).unwrap();

                // 攻め手が飛び駒の取り進みの場合(チェーン再帰)，2段目も展開
                if sub_sliding.is_some() && sub_defenses.len() > 5 {
                    // 防御手のうちドロップ(合駒)の数
                    let drop_count = sub_defenses.iter().filter(|m| m.is_drop()).count();
                    let non_drop_count = sub_defenses.len() - drop_count;
                    writeln!(out, "    (drops={}, non-drops={})", drop_count, non_drop_count).unwrap();
                }
            }

            // 250Kノードで解いて各深さのノード使用量を確認
            let mut solve_board = after_def.clone();
            let remaining = 15 - 1; // 攻め手1手消費
            let depth = (remaining + 2).min(41) as u32;
            let mut solve_solver = DfPnSolver::with_timeout(
                depth, 250_000, 32767, 30,
            );
            solve_solver.set_find_shortest(false);

            let start = std::time::Instant::now();
            let result = solve_solver.solve(&mut solve_board);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            writeln!(out, "Solve: {} nodes={} time={:.2}s max_ply={}",
                result_str, solve_solver.nodes_searched,
                elapsed.as_secs_f64(), solve_solver.max_ply).unwrap();
            writeln!(out, "TT entries: {}", solve_solver.table.len()).unwrap();
        }

            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: /tmp/ply24_diagnostic.log");
    }

    /// ply 24 TT共有効果の測定．
    ///
    /// 同じANDノードの兄弟応手間で TT を共有した場合としない場合の
    /// ノード数差を計測し，hand dominance による TT ヒットの実効性を検証する．
    ///
    /// 検証方法:
    /// 1. L*6g (解ける) を解いた後の TT を保持したまま N*6g を解く (共有あり)
    /// 2. N*6g を新規 TT で解く (共有なし)
    /// 3. ノード数とTTエントリ数を比較
    #[test]
    fn test_ply24_tt_sharing_effectiveness() {
        use std::io::Write;
        let out_path = "/tmp/ply24_tt_sharing.log";
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        // ply 25 ANDノード: 5g6f 後
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";

        // チェーン合駒ペア: (先に解く応手, 後に解く応手)
        let pairs = [
            ("L*6g", "N*6g", "同一マス(6g)への異種駒ドロップ"),
            ("B*7g", "N*7g", "同一マス(7g)への異種駒ドロップ"),
            ("B*7g", "P*7g", "同一マス(7g)への異種駒ドロップ(歩)"),
            ("L*6g", "P*7g", "異なるマスへの異種駒ドロップ"),
        ];

        writeln!(out, "=== TT共有効果の測定 ===\n").unwrap();

        for (first_usi, second_usi, desc) in &pairs {
            writeln!(out, "--- {} ---", desc).unwrap();
            writeln!(out, "先行: {}, 後行: {}\n", first_usi, second_usi).unwrap();

            // --- (A) TT共有あり: first → second (TT保持) ---
            let mut board_first = Board::new();
            board_first.set_sfen(and_sfen).unwrap();
            let m_first = board_first.move_from_usi(first_usi).unwrap();
            let mut after_first = board_first.clone();
            after_first.do_move(m_first);

            let depth = 16u32;
            let budget = 500_000u64;
            let mut solver = DfPnSolver::with_timeout(depth, budget, 32767, 60);
            solver.set_find_shortest(false);

            // first を解く
            let start = Instant::now();
            let r1 = solver.solve(&mut after_first);
            let first_nodes = solver.nodes_searched;
            let first_tt = solver.table.len();
            let first_time = start.elapsed();
            let r1_str = match &r1 {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                _ => "Other".to_string(),
            };
            writeln!(out, "(A) {} 単独: nodes={}, TT={}, {:.2}s → {}",
                first_usi, first_nodes, first_tt, first_time.as_secs_f64(), r1_str).unwrap();

            // TT を保持したまま second を解く
            // solve() は table.clear() するので、手動で状態をリセット
            let mut board_second = Board::new();
            board_second.set_sfen(and_sfen).unwrap();
            let m_second = board_second.move_from_usi(second_usi).unwrap();
            let mut after_second = board_second.clone();
            after_second.do_move(m_second);

            // solve() の内部を手動で再現(table.clear() をスキップ)
            solver.nodes_searched = 0;
            solver.max_ply = 0;
            solver.path.clear();
            solver.killer_table.clear();
            solver.start_time = Instant::now();
            solver.timed_out = false;
            solver.next_gc_check = 100_000;
            solver.attacker = after_second.turn; // Black (attacker)
            // table.clear() を意図的にスキップ
            let tt_before = solver.table.len();

            let saved_max_nodes = solver.max_nodes;
            solver.max_nodes = saved_max_nodes / 2;
            let _pns_pv = solver.pns_main(&mut after_second);
            solver.max_nodes = saved_max_nodes;

            let pk = position_key(&after_second);
            let att_hand = after_second.hand[solver.attacker.index()];
            let (root_pn, root_dn, _) = solver.look_up_pn_dn(pk, &att_hand, solver.depth as u16);
            if root_pn != 0 && root_dn != 0 && !solver.timed_out && solver.nodes_searched < solver.max_nodes {
                solver.mid(&mut after_second, INF - 1, INF - 1, 0, true);
            }

            let shared_nodes = solver.nodes_searched;
            let shared_tt = solver.table.len();
            let shared_time = solver.start_time.elapsed();
            let (final_pn, final_dn, _) = solver.look_up_pn_dn(pk, &att_hand, solver.depth as u16);
            let r2_shared = if final_pn == 0 { "Proved" } else if final_dn == 0 { "Disproved" } else { "Unknown" };
            writeln!(out, "(A) {} (TT共有): nodes={}, TT={}→{} (+{}), {:.2}s → {}",
                second_usi, shared_nodes, tt_before, shared_tt, shared_tt - tt_before,
                shared_time.as_secs_f64(), r2_shared).unwrap();

            // --- (B) TT共有なし: second を新規ソルバで解く ---
            let mut fresh_board = Board::new();
            fresh_board.set_sfen(and_sfen).unwrap();
            let m_fresh = fresh_board.move_from_usi(second_usi).unwrap();
            let mut after_fresh = fresh_board.clone();
            after_fresh.do_move(m_fresh);

            let mut fresh_solver = DfPnSolver::with_timeout(depth, budget, 32767, 60);
            fresh_solver.set_find_shortest(false);

            let start = Instant::now();
            let r2 = fresh_solver.solve(&mut after_fresh);
            let fresh_nodes = fresh_solver.nodes_searched;
            let fresh_tt = fresh_solver.table.len();
            let fresh_time = start.elapsed();
            let r2_str = match &r2 {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            writeln!(out, "(B) {} (新規TT): nodes={}, TT={}, {:.2}s → {}",
                second_usi, fresh_nodes, fresh_tt, fresh_time.as_secs_f64(), r2_str).unwrap();

            // 効果
            if shared_nodes < fresh_nodes && shared_nodes > 0 {
                writeln!(out, "→ TT共有効果: {:.1}x 削減 ({} → {} nodes)\n",
                    fresh_nodes as f64 / shared_nodes as f64, fresh_nodes, shared_nodes).unwrap();
            } else if shared_nodes > 0 {
                writeln!(out, "→ TT共有効果: なし (shared={}, fresh={})\n",
                    shared_nodes, fresh_nodes).unwrap();
            } else {
                writeln!(out, "→ TT共有効果: 計測不能\n").unwrap();
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: /tmp/ply24_tt_sharing.log");
    }

    /// ply 22 偽証明: 最終局面の合駒生成と between_bb を診断．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_ids_depth_diagnosis() {
        // 最終局面: 飛車 8e から玉 1e への王手
        let final_sfen = "9/3+N1P3/7+R1/9/1R6k/9/8P/3S5/9 w 2b4g3s3n4l16p 30";

        let mut final_board = Board::new();
        final_board.set_sfen(final_sfen).unwrap();

        let defender = final_board.turn(); // White
        let attacker = defender.opponent(); // Black
        let king_sq = final_board.king_square(defender).unwrap();
        eprintln!("King square: {:?} (col={}, row={})", king_sq, king_sq.col(), king_sq.row());

        // compute_checkers_at
        let checkers = final_board.compute_checkers_at(king_sq, attacker);
        eprintln!("Checkers: count={}", checkers.count());
        for sq in checkers {
            eprintln!("  checker at {:?} (col={}, row={})", sq, sq.col(), sq.row());
        }

        // find_sliding_checker
        let mut solver = DfPnSolver::default_solver();
        let sliding = solver.find_sliding_checker(&final_board, king_sq, attacker);
        eprintln!("find_sliding_checker: {:?}", sliding.map(|s| format!("col={}, row={}", s.col(), s.row())));

        // checker_sq
        let checker_sq = checkers.lsb().unwrap();

        // between_bb
        let between = attack::between_bb(checker_sq, king_sq);
        eprintln!("between_bb({:?}, {:?}): count={}", checker_sq, king_sq, between.count());
        for sq in between {
            eprintln!("  between: col={}, row={}", sq.col(), sq.row());
        }

        // compute_futile_and_chain_squares
        let (futile, chain) = solver.compute_futile_and_chain_squares(
            &final_board, &between, king_sq, checker_sq, defender, attacker,
        );
        eprintln!("futile: count={}", futile.count());
        for sq in futile {
            eprintln!("  futile: col={}, row={}", sq.col(), sq.row());
        }
        eprintln!("chain: count={}", chain.count());
        for sq in chain {
            eprintln!("  chain: col={}, row={}", sq.col(), sq.row());
        }

        // generate_defense_moves
        let defenses = solver.generate_defense_moves(&mut final_board);
        eprintln!("generate_defense_moves: {} moves", defenses.len());
        for d in &defenses {
            eprintln!("  {}", d.to_usi());
        }

        // 比較: generate_legal_moves
        let legal = movegen::generate_legal_moves(&mut final_board);
        eprintln!("generate_legal_moves: {} moves", legal.len());

        // between_bb が空なら合駒生成がスキップされる → バグの原因
        assert!(
            between.count() > 0,
            "between_bb is empty for checker={:?} king={:?}, blocking moves will be skipped!",
            checker_sq, king_sq
        );
    }

    /// 無駄合い判定テスト: 飛車の王手に対して合駒で詰みが回避できる局面．
    ///
    /// 飛車が横(rank)方向に王手しているが，玉の逃げ道が飛び駒の取り進みで
    /// 塞がれない場合，合駒は無駄合いではない．
    /// `compute_futile_and_chain_squares` が全マスを futile にせず，
    /// `generate_defense_moves` が合駒(駒打ち)を生成することを検証する．
    #[test]
    fn test_futile_check_rook_rank_not_futile_when_king_can_escape() {
        // 飛車(8e)が横王手，金(2d)が 2e を支えている
        // 飛車が 2e に取り進んだ後，玉は 1f に逃げられるので無駄合いではない
        //
        // 盤面:
        //   9  8  7  6  5  4  3  2  1
        //                            |  rank 1
        //                            |  rank 2
        //                            |  rank 3
        //                      G     |  rank 4  (金 at 2d)
        //      R                 k   |  rank 5  (飛車 at 8e, 玉 at 1e)
        //                            |  rank 6  (1f = 玉の逃げ先)
        let sfen = "9/9/9/7G1/1R6k/9/9/9/9 w r2b3g4s4n4l18p 2";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent(); // Black
        let king_sq = board.king_square(defender).unwrap();
        let checkers = board.compute_checkers_at(king_sq, attacker);
        assert_eq!(checkers.count(), 1, "Expected exactly one checker");
        let checker_sq = checkers.lsb().unwrap();

        let mut solver = DfPnSolver::default_solver();

        // between_bb が正しく計算されること
        let between = attack::between_bb(checker_sq, king_sq);
        assert!(between.count() > 0, "between_bb must not be empty");

        // compute_futile_and_chain_squares: king-adjacent マスが futile にならないこと
        let (futile, _chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );
        // 飛車が 2e(king-adjacent)に取り進んだ後，玉は 1f に逃げられる
        // → king-adjacent マスは futile ではない → futile != between
        assert!(
            futile != between,
            "All between squares are futile — king escape after slider capture not checked"
        );

        // generate_defense_moves: 合駒(駒打ち)が含まれること
        let defenses = solver.generate_defense_moves(&mut board);
        let has_drop = defenses.iter().any(|m| m.is_drop());
        assert!(
            has_drop,
            "generate_defense_moves must include drop interpositions, got {} moves: {:?}",
            defenses.len(),
            defenses.iter().map(|m| m.to_usi()).collect::<Vec<_>>()
        );
    }

    /// 無駄合い判定テスト: 飛車王手で玉が完全に囲まれており合駒が無駄な局面．
    ///
    /// 飛び駒が取り進んだ後も玉の逃げ道がない場合，合駒は無駄合いとなる．
    /// `compute_futile_and_chain_squares` が king-adjacent マスを futile にし，
    /// 合駒(駒打ち)がスキップされることを検証する．
    #[test]
    fn test_futile_check_rook_rank_futile_when_king_trapped() {
        // 後手玉 1a, 先手飛 9a から横王手
        // 先手: 飛9a, 金2a, 金1b → 玉の全逃げ道が塞がれている
        // 飛車が 2a に取り進むと金が利いており，玉は逃げられない
        let sfen = "R1G5k/1G7/9/9/9/9/9/9/9 w 2b2r3s4n4l18p 2";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent();
        let king_sq = board.king_square(defender).unwrap();
        let checkers = board.compute_checkers_at(king_sq, attacker);

        if checkers.is_empty() {
            // チェッカーが検出されない場合はテストスキップ
            return;
        }
        let checker_sq = checkers.lsb().unwrap();

        let mut solver = DfPnSolver::default_solver();
        let between = attack::between_bb(checker_sq, king_sq);

        if between.is_empty() {
            // between が空(隣接王手)の場合はテストスキップ
            return;
        }

        let (futile, _chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );

        // 玉が完全に囲まれているので，king-adjacent マスは futile であるべき
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let king_adj_between = between & king_step;
        if king_adj_between.is_not_empty() {
            for sq in king_adj_between {
                assert!(
                    futile.contains(sq),
                    "King-adjacent between square (col={}, row={}) should be futile when king is trapped",
                    sq.col(), sq.row()
                );
            }
        }
    }

    /// 39手詰め ply 22 局面で偽の短手数詰みを返すバグのリグレッションテスト．
    ///
    /// ソルバーが Mate(7) を返すが，最終局面 8g8e の後に合法手が36手あり
    /// 詰みではない(is_checkmate=false)．証明ツリーが不正．
    /// `find_shortest=true` でも同じ Mate(7) を返すため PV 抽出ではなく
    /// 証明自体のバグ．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_pv_must_end_in_checkmate() {
        // 39手詰めの ply 22 局面(攻め番)
        let sfen = "9/3+N1P3/7+R1/9/9/8k/1R2S4/3p5/9 b P2b4g3s3n4l15p 23";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(19, 1_000_000, 32767, 180);
        solver.set_find_shortest(false);

        let mut test_board = board.clone();
        let result = solver.solve(&mut test_board);

        if let TsumeResult::Checkmate { moves, .. } = &result {
            // PV の全手が合法手であること
            let mut vb = board.clone();
            for (i, m) in moves.iter().enumerate() {
                let legal = movegen::generate_legal_moves(&mut vb);
                assert!(
                    legal.iter().any(|lm| lm.to_usi() == m.to_usi()),
                    "PV move {} ({}) is illegal at SFEN: {}",
                    i + 1, m.to_usi(), vb.sfen()
                );
                // 攻め手(偶数 index)は王手であること
                vb.do_move(*m);
                if i % 2 == 0 {
                    assert!(
                        vb.is_in_check(vb.turn()),
                        "ATK move {} ({}) does not give check",
                        i + 1, m.to_usi()
                    );
                }
            }

            // 最終局面が詰み(合法手0 かつ王手)であること
            let final_legal = movegen::generate_legal_moves(&mut vb);
            assert!(
                final_legal.is_empty() && vb.is_in_check(vb.turn()),
                "PV of length {} does not end in checkmate: \
                 legal_moves={}, in_check={}, SFEN={}",
                moves.len(),
                final_legal.len(),
                vb.is_in_check(vb.turn()),
                vb.sfen()
            );
        }
        // Checkmate 以外(Unknown 等)は許容: 解けなかっただけ
    }

    /// ply 22 OR ノードの王手ごとのノード消費を調査する．
    ///
    /// 151K ノードで NoMate (1M 予算を使い切らない) の原因を特定:
    /// - depth 制限による打ち切り
    /// - NM (不詰) の誤判定
    /// - IDS の早期終了
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_or_node_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        // ply 22 の局面を構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }

        let ply22_sfen = board.sfen();
        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" Ply 22 OR node breakdown (残り17手, PV: P*1g)");
        eprintln!(" SFEN: {}", ply22_sfen);
        eprintln!("{}", "=".repeat(80));

        // 1. まず全体を depth=19 で解いてみる
        eprintln!("\n--- 全体 solve (depth=19, 1M nodes) ---");
        {
            let mut b = board.clone();
            let mut solver = DfPnSolver::with_timeout(19, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut b);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            eprintln!("  depth=19: {} nodes={} time={:.2}s max_ply={}",
                result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 2. depth を変えて解いてみる
        eprintln!("\n--- depth 別 solve (1M nodes) ---");
        for depth in [17u32, 19, 21, 23, 25, 31, 41] {
            let mut b = board.clone();
            let mut solver = DfPnSolver::with_timeout(depth, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut b);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            eprintln!("  depth={:<4} {} nodes={:<10} time={:.2}s max_ply={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 3. check_moves の一覧と個別探索
        eprintln!("\n--- 王手一覧と個別探索 (depth=17, 250K nodes each) ---");
        let check_solver = DfPnSolver::default_solver();
        let check_moves = check_solver.generate_check_moves(&mut board);
        eprintln!("  王手数: {}", check_moves.len());

        // brute-force でも確認
        let brute_checks: Vec<String> = movegen::generate_legal_moves(&mut board)
            .into_iter()
            .filter(|m| {
                let c = board.do_move(*m);
                let gives_check = board.is_in_check(board.turn);
                board.undo_move(*m, c);
                gives_check
            })
            .map(|m| m.to_usi())
            .collect();
        eprintln!("  brute-force 王手数: {}", brute_checks.len());

        eprintln!("  {:<12} {:<14} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "Result");
        for cm in &check_moves {
            let mut after = board.clone();
            after.do_move(*cm);

            // 王手後 → AND node → 各応手を含む残り16手を探索
            let mut sub = after.clone();
            let mut solver = DfPnSolver::with_timeout(17, 250_000, 32767, 30);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut sub);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            let marker = if cm.to_usi() == "P*1g" { " ← PV" } else { "" };
            eprintln!("  {:<12} {:<14} {:<10.2} {:<10} {}{}",
                cm.to_usi(), solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, result_str, marker);
        }

        // 4. P*1g に注目: depth を変えて解く
        eprintln!("\n--- P*1g 単体 depth 別 (1M nodes) ---");
        let pawn_drop = board.move_from_usi("P*1g").unwrap();
        let mut after_pg = board.clone();
        after_pg.do_move(pawn_drop);

        for depth in [15u32, 17, 19, 21] {
            let mut sub = after_pg.clone();
            let mut solver = DfPnSolver::with_timeout(depth, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut sub);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            eprintln!("  depth={:<4} {} nodes={:<10} time={:.2}s max_ply={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 5. all_checks_refutable_recursive のバグ確認
        // P*1g 後の各応手について，次の王手の有無を確認
        eprintln!("\n--- all_checks_refutable analysis for P*1g ---");
        let pawn_drop2 = board.move_from_usi("P*1g").unwrap();
        let cap_pg = board.do_move(pawn_drop2);
        let mut def_solver = DfPnSolver::default_solver();
        let defenses = def_solver.generate_defense_moves(&mut board);
        eprintln!("  P*1g 後の応手数: {}", defenses.len());
        for def_mv in &defenses {
            let cap_d = board.do_move(*def_mv);
            let next_checks = def_solver.generate_check_moves(&mut board);
            eprintln!("  {} → 次の王手数: {} {:?}",
                def_mv.to_usi(), next_checks.len(),
                next_checks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());
            board.undo_move(*def_mv, cap_d);
        }
        board.undo_move(pawn_drop2, cap_pg);

        eprintln!("{}", "=".repeat(80));
    }

    /// TT 保護のリグレッションテスト: find_shortest モード有効時の PV 検証．
    ///
    /// complete_or_proofs 中の mid() が転置により証明済み TT エントリを
    /// 上書きしていたバグの回帰テスト．find_shortest=true(デフォルト)で
    /// PV が最長抵抗を正しく反映することを確認する．
    #[test]
    fn test_pv_follows_longest_defense() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let result = solve_tsume_with_timeout(
            sfen, Some(31), Some(2_000_000), None, None,
            Some(true), // find_shortest = true
            None, None, None,
        ).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "PV should be 9 moves (longest defense via gold interposition), got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                assert_eq!(usi_moves[0], "B*3c", "move 1: B*3c(3三角打)");
                assert_eq!(usi_moves[1], "2a2b", "move 2: g(2a→2b)(金の移動合い=最長抵抗)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 後手番不詰のケース．
    #[test]
    fn test_no_checkmate_gote() {
        // 先手玉5九，後手持ち駒: 歩 → 歩では詰まない
        let sfen = "9/9/9/9/9/9/9/9/4K4 w p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(5, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!("expected NoCheckmate, got {:?}", other),
        }
    }

    /// プロファイリング計測テスト．
    ///
    /// `cargo test -p maou_shogi --features profile test_profile_solve -- --nocapture`
    /// で実行し，各操作の時間内訳を表示する．
    #[test]
    #[cfg(feature = "profile")]
    fn test_profile_solve() {
        let cases: Vec<(&str, &str, u32, u64, u32)> = vec![
            // (name, sfen, depth, max_nodes, timeout_sec)
            ("9手詰(9te)", "6s2/6l2/9/6BBk/9/9/9/9/K8 b RPr4g3s4n3l17p 1", 31, 10_000_000, 60),
            ("11手詰(tsume2)", "4+P2kl/7s1/5R3/7B1/9/9/9/9/K8 b GNrb3g3s3n3l17p 1", 31, 10_000_000, 60),
            ("17手詰(tsume5)", "9/5Pk2/9/8R/8B/9/9/9/K8 b 2Srb4g2s4n4l17p 1", 31, 10_000_000, 120),
            ("7手詰(tsume3)", "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/K8 b R2b3g4s2n2l15p 1", 31, 10_000_000, 60),
            ("5手詰(tsume4)", "7nk/9/5R3/8p/6P2/9/9/9/K8 b SNPr2b4g3s2n4l15p 1", 31, 10_000_000, 60),
            ("9手詰(合駒)", "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1", 31, 2_000_000, 60),
            ("29手詰(tsume6)", "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1", 31, 50_000_000, 300),
        ];

        let budgets = [0u32, 16, 256];

        println!("\n{:<20} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Problem", "Budget", "Wall(ms)", "Nodes", "NPS",
            "SM_call", "SM_hit%");
        println!("{}", "-".repeat(90));

        for (name, sfen, depth, nodes, timeout) in &cases {
            for &budget in &budgets {
                let mut board = Board::new();
                board.set_sfen(sfen).unwrap();

                let mut solver = DfPnSolver::with_timeout(*depth, *nodes, 32767, *timeout as u64);
                solver.set_mate_budget(budget);

                let start = std::time::Instant::now();
                let result = solver.solve(&mut board);
                let wall_time = start.elapsed();
                solver.sync_tt_profile();

                let nodes_searched = match &result {
                    TsumeResult::Checkmate { nodes_searched, .. } => *nodes_searched,
                    TsumeResult::CheckmateNoPv { nodes_searched } => *nodes_searched,
                    TsumeResult::NoCheckmate { nodes_searched } => *nodes_searched,
                    TsumeResult::Unknown { nodes_searched } => *nodes_searched,
                };
                let wall_ms = wall_time.as_secs_f64() * 1000.0;
                let nps = if wall_ms > 0.0 { nodes_searched as f64 / (wall_ms / 1000.0) } else { 0.0 };
                let sm = &solver.profile_stats;
                let hit_pct = if sm.static_mate_count > 0 {
                    sm.static_mate_hits as f64 / sm.static_mate_count as f64 * 100.0
                } else { 0.0 };

                println!("{:<20} {:>6} {:>10.1} {:>10} {:>10.0} {:>8} {:>7.1}%  overflow:{} max_e:{}",
                    name, budget, wall_ms, nodes_searched, nps,
                    sm.static_mate_count, hit_pct,
                    sm.tt_overflow_count, sm.tt_max_entries_per_position);
            }
        }
    }

    /// 39手詰めベンチマーク(budget 別比較)．
    ///
    /// `cargo test -p maou_shogi --features profile --release bench_39te -- --nocapture --ignored`
    #[test]
    #[ignore]
    #[cfg(feature = "profile")]
    fn bench_39te_budgets() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let budgets = [0u32, 8, 16, 32, 64];

        println!("\n39手詰め(合駒) budget comparison");
        println!("{:>6} {:>10} {:>10} {:>10} {:>8} {:>7}",
            "Budget", "Wall(ms)", "Nodes", "NPS", "SM_call", "SM_hit%");
        println!("{}", "-".repeat(65));

        for &budget in &budgets {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            let mut solver = DfPnSolver::with_timeout(63, 10_000_000, 32767, 120);
            solver.set_mate_budget(budget);

            let start = std::time::Instant::now();
            let result = solver.solve(&mut board);
            let wall_time = start.elapsed();
            solver.sync_tt_profile();

            let nodes_searched = match &result {
                TsumeResult::Checkmate { nodes_searched, .. } => *nodes_searched,
                TsumeResult::CheckmateNoPv { nodes_searched } => *nodes_searched,
                TsumeResult::NoCheckmate { nodes_searched } => *nodes_searched,
                TsumeResult::Unknown { nodes_searched } => *nodes_searched,
            };
            let wall_ms = wall_time.as_secs_f64() * 1000.0;
            let nps = if wall_ms > 0.0 { nodes_searched as f64 / (wall_ms / 1000.0) } else { 0.0 };
            let sm = &solver.profile_stats;
            let hit_pct = if sm.static_mate_count > 0 {
                sm.static_mate_hits as f64 / sm.static_mate_count as f64 * 100.0
            } else { 0.0 };
            let status = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("mate in {}", moves.len()),
                _ => "NOT SOLVED".to_string(),
            };

            println!("{:>6} {:>10.1} {:>10} {:>10.0} {:>8} {:>6.1}%  {}",
                budget, wall_ms, nodes_searched, nps,
                sm.static_mate_count, hit_pct, status);
        }
    }

    /// 中合い(ちゅうあい)によって不詰みになる局面のテスト．
    ///
    /// 飛び駒の王手に対して持ち駒を間に打つ中合いが有効で，
    /// 詰みが成立しないことを検証する．
    #[test]
    fn test_chuai_no_checkmate() {
        // 中合いによって詰まない局面
        // デバッグビルドでの実行時間制約のため予算を抑制する．
        // Checkmate を返さないことが主要な検証ポイント．
        //
        // 合い駒生成の改善により探索分岐が増え，デバッグビルドでは
        // デフォルトスタック(8MB)で溢れるため専用スレッドで実行する．
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let sfen = "9/3+N1P3/2+R3p2/8k/8N/5+B3/4S4/1R1p5/9 b NPb4g3sn4l14p 1";
                solve_tsume(sfen, Some(31), Some(10_000), None).unwrap()
            })
            .unwrap()
            .join()
            .unwrap();

        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "中合いにより不詰みのはず: {:?}",
            result
        );
    }

    /// 8h8d 王手後の AND ノードで中合い(P*7d)が回避手に含まれることを確認．
    ///
    /// 合い効かずフィルタが P*7d を誤って除外していないかの診断テスト．
    #[test]
    fn test_chuai_defense_includes_pawn_drop() {
        let sfen = "9/3+N1P3/2+R3p2/8k/8N/5+B3/4S4/1R1p5/9 b NPb4g3sn4l14p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // 攻め方: 8h8d (飛車8八→8四)
        let m = board.move_from_usi("8h8d").unwrap();
        board.do_move(m);

        // AND ノード: 後手の回避手を生成
        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();

        // 全合法手と比較
        let legal = movegen::generate_legal_moves(&mut board);
        let usi_legal: Vec<String> = legal.iter().map(|m| m.to_usi()).collect();

        // P*7d が合法手に含まれること
        assert!(
            usi_legal.contains(&"P*7d".to_string()),
            "P*7d should be a legal move, got: {:?}", usi_legal
        );

        // P*7d が回避手にも含まれること(合い効かずで除外されていないこと)
        assert!(
            usi_defenses.contains(&"P*7d".to_string()),
            "P*7d should be in defense moves (中合い), but was filtered out.\n\
             defense moves: {:?}\n\
             legal moves: {:?}",
            usi_defenses, usi_legal
        );
    }

    /// 中合い発生後の局面が不詰みであることを確認するテスト．
    ///
    /// 上記 `test_chuai_no_checkmate` の局面で中合いが行われた後の
    /// 状態を直接与え，攻め方から探索しても詰みがないことを検証する．
    #[test]
    fn test_chuai_position_after_block() {
        // 中合いが発生した後の局面(不詰み)
        // デバッグビルドでの実行時間制約のため予算を抑制する．
        let sfen = "9/3+N1P3/2+R3p2/1Rp5k/8N/5+B3/4S4/3p5/9 b NPb4g3sn4l13p 1";
        let result = solve_tsume(sfen, Some(31), Some(10_000), None).unwrap();

        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "中合い後の局面は不詰みのはず: {:?}",
            result
        );
    }

    /// 39手詰め正解 PV 上の各局面における分岐数・手生成数を診断する．
    ///
    /// PV を1手ずつ進めながら，各局面で:
    /// - OR ノード(攻め方): generate_check_moves の手数
    /// - AND ノード(守備方): generate_defense_moves の手数
    /// を出力し，探索がどこでノードを浪費しているかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_pv_trace() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", // 1. ６二成桂
            "5b4c", // 2. ４三玉
            "8b9c", // 3. ９三龍
            "4c3d", // 4. ３四玉
            "1b2c", // 5. ２三銀
            "3d2c", // 6. 同玉
            "N*1e", // 7. １五桂打
            "2c3b", // 8. ３二玉
            "N*2d", // 9. ２四桂打
            "3b2b", // 10. ２二玉
            "2d1b+", // 11. １二桂成
            "2b3b", // 12. ３二玉
            "1b2b", // 13. ２二成桂
            "3b2b", // 14. 同玉
            "4f1c", // 15. １三馬
            "2b1c", // 16. 同玉
            "9c3c", // 17. ３三龍
            "1c1d", // 18. １四玉
            "3c2c", // 19. ２三龍
            "1d1e", // 20. １五玉
            "P*1f", // 21. １六歩打
            "1e1f", // 22. 同玉
            "P*1g", // 23. １七歩打
            "1f1g", // 24. 同玉
            "5g6f", // 25. ６六銀
            "1g1h", // 26. １八玉
            "2c2g", // 27. ２七龍
            "1h1i", // 28. １九玉
            "8g8i", // 29. ８九飛
            "S*6i", // 30. ６九銀打
            "8i6i", // 31. 同飛
            "6h6i+", // 32. 同歩成
            "S*2h", // 33. ２八銀打
            "1i2i", // 34. ２九玉
            "2h3g", // 35. ３七銀
            "2i3i", // 36. ３九玉
            "2g2h", // 37. ２八龍
            "3i4i", // 38. ４九玉
            "2h4h", // 39. ４八龍
        ];

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        let mut solver = DfPnSolver::default_solver();

        eprintln!("\n{:>3} {:>6} {:>5} {:>5} {:>6} {:<12} {}",
            "Ply", "Node", "Moves", "Drops", "Total", "PV Move", "Sample moves (first 10)");
        eprintln!("{}", "-".repeat(90));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let ply = i + 1;
            let is_or = (ply % 2) == 1; // 奇数手=攻め方(OR), 偶数手=守備方(AND)

            let moves = if is_or {
                solver.generate_check_moves(&mut board)
            } else {
                solver.generate_defense_moves(&mut board)
            };

            // ドロップ手のカウント
            let drop_count = moves.iter().filter(|m| m.is_drop()).count();
            let move_count = moves.len() - drop_count;

            // 正解手が手リストに含まれているか確認
            let expected_move = board.move_from_usi(usi)
                .unwrap_or_else(|| panic!("Invalid USI at ply {}: {}", ply, usi));
            let found = moves.iter().any(|m| *m == expected_move);

            // サンプル表示(ply 4 は全手, それ以外は先頭10手)
            let limit = if ply == 4 { moves.len() } else { 10 };
            let sample: Vec<String> = moves.iter().take(limit).map(|m| m.to_usi()).collect();

            let node_type = if is_or { "OR" } else { "AND" };
            let mark = if !found { " *** MISSING ***" } else { "" };

            eprintln!("{:>3} {:>6} {:>5} {:>5} {:>6} {:<12} [{}]{}",
                ply, node_type, move_count, drop_count, moves.len(),
                usi, sample.join(", "), mark);

            // 手を適用して次の局面へ
            board.do_move(expected_move);
        }

        // 最終局面が詰みかチェック
        let final_defenses = solver.generate_defense_moves(&mut board);
        eprintln!("\n最終局面(39手目後)の回避手数: {}", final_defenses.len());
        if final_defenses.is_empty() {
            eprintln!("→ 詰み!");
        } else {
            let sample: Vec<String> = final_defenses.iter().take(10).map(|m| m.to_usi()).collect();
            eprintln!("→ 回避手あり: [{}]", sample.join(", "));
        }
    }

    /// Ply 4 の AND ノードにおける各子ノードの探索難易度を診断する．
    ///
    /// 各回避手を適用した局面に対して小予算ソルブを実行し，
    /// 消費ノード数・結果を比較する．AND ノードでは 1 つの不詰み子
    /// が見つかれば十分なので，簡単な子が先に試されるべき．
    #[test]
    #[ignore]
    fn test_ply4_child_node_difficulty() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        eprintln!("\nPly 4 子ノード難易度診断 (budget=100,000 nodes, depth=37)");
        eprintln!("{:>4} {:>12} {:>8} {:>10} {:>10} {:>8}",
            "#", "Move", "Type", "Nodes", "Result", "PV len");
        eprintln!("{}", "-".repeat(65));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 攻め方視点でソルブ (残り depth=37, budget=100K)
            let mut child_solver = DfPnSolver::new(37, 100_000, 32767);
            child_solver.set_find_shortest(false);
            let result = child_solver.solve(&mut child_board);

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, pv_len) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched: _ } =>
                    (format!("MATE"), moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    (format!("MATE(nopv)"), 0),
                TsumeResult::NoCheckmate { .. } =>
                    (format!("NO_MATE"), 0),
                TsumeResult::Unknown { .. } =>
                    (format!("UNKNOWN"), 0),
            };
            let nodes = match &result {
                TsumeResult::Checkmate { nodes_searched, .. } => *nodes_searched,
                TsumeResult::CheckmateNoPv { nodes_searched } => *nodes_searched,
                TsumeResult::NoCheckmate { nodes_searched } => *nodes_searched,
                TsumeResult::Unknown { nodes_searched, .. } => *nodes_searched,
            };

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };

            eprintln!("{:>4} {:>12} {:>8} {:>10} {:>10} {:>8}{}",
                i + 1, defense.to_usi(), move_type, nodes, result_str, pv_len, marker);
        }
    }

    /// 39手詰 PV 上の合駒(S*6i)後の局面を単体ソルブし，
    /// 各応手に必要なノード数を計測する．
    ///
    /// 合駒は30手目(S*6i)で，攻め方は31手目(8i6i)で銀を取る．
    /// 取った後の局面(攻め方番)を単体で解き，ノード数を確認する．
    #[test]
    #[ignore]
    fn test_interpose_subproblem_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        // PV: 39手
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i",
        ];
        // 29手目(8g8i = 飛車8i)まで進める → 30手目が合駒局面(AND ノード)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_usi {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 29 後: 後手番(AND ノード) - ここで合駒 S*6i が最善
        assert_eq!(board.turn, Color::White);

        // AND ノードの回避手を生成
        let mut solver_tmp = DfPnSolver::default_solver();
        let defenses = solver_tmp.generate_defense_moves(&mut board);

        eprintln!("\n=== 合駒局面ブレークダウン (ply 29 後) ===");
        eprintln!("回避手数: {} (うち drop: {})",
            defenses.len(),
            defenses.iter().filter(|m| m.is_drop()).count());

        eprintln!("\n{:>4} {:>8} {:>5} {:>10} {:>10.1} {:>8}",
            "#", "Move", "Type", "Nodes", "Time(ms)", "Result");
        eprintln!("{}", "-".repeat(55));

        // 各回避手を適用後，攻め方視点でソルブ
        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 残り手数 = 39 - 30 = 9 手(合駒後)
            // ただしここで攻め方が合駒を取るので，取り後は残り 8 手
            let remaining = 10; // 余裕を持って depth=10
            let mut child_solver = DfPnSolver::new(remaining, 100_000, 32767);
            child_solver.set_find_shortest(false);
            let start = Instant::now();
            let result = child_solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let is_best = defense.to_usi() == "S*6i";
            let marker = if is_best { " ← BEST" } else { "" };
            eprintln!("{:>4} {:>8} {:>5} {:>10} {:>10.1} {:>8}{}",
                i + 1, defense.to_usi(), move_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str, marker);
        }
    }

    /// PV 上の各サブ問題(守備方の手後 = 攻め方番)を個別ソルブし，
    /// どの深さからソルブ困難になるかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_solve() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        eprintln!("\n{:>5} {:>3} {:>8} {:>10} {:>10} {:>8}",
            "After", "Typ", "Remain", "Nodes", "Time(ms)", "Result");
        eprintln!("{}", "-".repeat(62));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);

            // 全 ply をテスト(偶数=OR, 奇数=AND)
            let remaining = 39 - (i + 1);
            let or_sub = i % 2 == 1; // 奇数 i 後は攻め方番 = OR
            let budget = if remaining >= 31 { 10_000_000u64 } else { 2_000_000 };
            let mut solver = DfPnSolver::new(
                (remaining + 2) as u32, budget, 32767,
            );
            solver.set_find_shortest(false);
            let start = Instant::now();
            let mut test_board = board.clone();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (status, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let node_type = if or_sub { "OR " } else { "AND" };
            eprintln!("{:>5} {:>3} {:>8} {:>10} {:>10.1} {:>8}",
                format!("ply{}", i + 1), node_type, remaining, nodes,
                elapsed.as_secs_f64() * 1000.0, status);
        }
    }

    /// 39手詰 PV 上の各サブ問題を小予算(100K)でソルブし，
    /// どの ply でノード爆発が起きるかを特定する．
    ///
    /// 各 ply 後の局面を独立した詰将棋として解き，
    /// 消費ノード数と結果を出力する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_quick() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        eprintln!("\n{:>5} {:>3} {:>6} {:>10} {:>10} {:>10}",
            "After", "Typ", "Remain", "Budget", "Nodes", "Result");
        eprintln!("{}", "-".repeat(55));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);

            let remaining = 39 - (i + 1);
            if remaining == 0 { break; }
            let or_sub = i % 2 == 1; // 奇数 i 後は攻め方番 = OR

            // 小予算で素早くテスト
            let budget = 100_000u64;
            let mut solver = DfPnSolver::new(
                (remaining + 2) as u32, budget, 32767,
            );
            solver.set_find_shortest(false);
            let mut test_board = board.clone();
            let result = solver.solve(&mut test_board);

            let (status, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let node_type = if or_sub { "OR " } else { "AND" };
            eprintln!("{:>5} {:>3} {:>6} {:>10} {:>10} {:>10}",
                format!("ply{}", i + 1), node_type, remaining,
                budget, nodes, status);
        }
    }

    /// 39手詰で最も分岐の多い ply 4 (AND ノード，20手の応手) の
    /// 各回避手を単体ソルブし，どの分岐でノード爆発するかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply4_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        eprintln!("\n=== Ply 4 AND ノード回避手ブレークダウン ===");
        eprintln!("回避手数: {} (うち drop: {}, move: {})",
            defenses.len(),
            defenses.iter().filter(|m| m.is_drop()).count(),
            defenses.iter().filter(|m| !m.is_drop()).count());

        eprintln!("\n{:>4} {:>8} {:>5} {:>10} {:>10} {:>10}",
            "#", "Move", "Type", "Budget", "Nodes", "Result");
        eprintln!("{}", "-".repeat(55));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 残り35手分で探索 (depth=37, budget=500K)
            let mut child_solver = DfPnSolver::new(37, 500_000, 32767);
            child_solver.set_find_shortest(false);
            let result = child_solver.solve(&mut child_board);

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };
            eprintln!("{:>4} {:>8} {:>5} {:>10} {:>10} {:>10}{}",
                i + 1, defense.to_usi(), move_type,
                500_000, nodes, result_str, marker);
        }
    }

    /// 39手詰 ply 4 の全応手について，合駒は龍で取った後の局面を solve し
    /// PV(詰み筋)を比較する．
    ///
    /// 仮説: 合駒を取ると元の局面とほぼ同じ → 同じ詰み筋が繰り返される．
    /// 合駒の数だけ同難度のサブ問題が増殖しているかを確認する．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply4_pv_comparison() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        // ── Part 1: 逃げ手(盤上の手)の PV ──
        eprintln!("\n=== Part 1: 逃げ手の詰み筋 (budget=500K) ===\n");
        for (i, &defense) in defenses.iter().enumerate() {
            if defense.is_drop() { continue; }
            let mut child_board = board.clone();
            child_board.do_move(defense);

            let mut s = DfPnSolver::with_timeout(37, 500_000, 32767, 10);
            s.set_find_shortest(false);
            let result = s.solve(&mut child_board);

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };
            print_result(i + 1, &defense.to_usi(), &result, marker);
        }

        // ── Part 2: 合駒を龍で取った後の局面の PV ──
        // 合駒は 5c/6c/7c/8c 筋に打たれ，龍(9c)が取る．
        // 取った後の局面を solve し，PV を比較する．
        eprintln!("\n=== Part 2: 合駒 → 龍で取った後の詰み筋 (budget=500K) ===");
        eprintln!("(defense → capture → 玉方応手の各分岐を solve)\n");

        let interpositions: Vec<(&str, &str)> = vec![
            // (合駒, 龍の取り)
            ("B*5c", "9c5c"), ("G*5c", "9c5c"), ("S*5c", "9c5c"),
            ("N*5c", "9c5c"), ("L*5c", "9c5c"), ("P*5c", "9c5c"),
            ("L*6c", "9c6c"), ("B*6c", "9c6c"), ("N*6c", "9c6c"),
            ("P*7c", "9c7c"), ("B*7c", "9c7c"), ("N*7c", "9c7c"),
            ("P*8c", "9c8c"), ("B*8c", "9c8c"), ("N*8c", "9c8c"),
        ];

        for (idx, &(interpose_usi, capture_usi)) in interpositions.iter().enumerate() {
            let interpose = board.move_from_usi(interpose_usi).unwrap();
            let mut b = board.clone();
            b.do_move(interpose);
            let capture = b.move_from_usi(capture_usi).unwrap();
            b.do_move(capture);

            // ply 6 相当: 玉方番(AND) → 各応手後の攻め方局面を solve
            let defs = solver.generate_defense_moves(&mut b);
            eprintln!("{:>2}. {} → {} (玉方応手{}手)",
                idx + 1, interpose_usi, capture_usi, defs.len());

            for &def in defs.iter() {
                let mut bc = b.clone();
                bc.do_move(def);

                // 攻め方番: 残り33手で solve
                let mut s = DfPnSolver::with_timeout(33, 200_000, 32767, 3);
                s.set_find_shortest(false);
                let result = s.solve(&mut bc);

                let label = format!("  {} → {}", capture_usi, def.to_usi());
                print_result(0, &label, &result, "");
            }
            eprintln!();
        }
    }

    fn print_result(idx: usize, label: &str, result: &TsumeResult, marker: &str) {
        match result {
            TsumeResult::Checkmate { moves, nodes_searched } => {
                let pv_str: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                if idx > 0 {
                    eprintln!("{:>2}. {} nodes={:>7} MATE({:>2}) PV: {}{}",
                        idx, label, nodes_searched, moves.len(), pv_str.join(" "), marker);
                } else {
                    eprintln!("    {} nodes={:>7} MATE({:>2}) PV: {}{}",
                        label, nodes_searched, moves.len(), pv_str.join(" "), marker);
                }
            }
            TsumeResult::CheckmateNoPv { nodes_searched } => {
                eprintln!("    {} nodes={:>7} MATE(nopv){}", label, nodes_searched, marker);
            }
            TsumeResult::NoCheckmate { nodes_searched } => {
                eprintln!("    {} nodes={:>7} NO_MATE{}", label, nodes_searched, marker);
            }
            TsumeResult::Unknown { nodes_searched } => {
                eprintln!("    {} nodes={:>7} UNKNOWN{}", label, nodes_searched, marker);
            }
        }
    }

    /// 39手詰のボトルネック分析: 不詰み証明に時間がかかる分岐を特定する．
    ///
    /// ply 4 (AND ノード) の各応手について:
    /// 1. 応手後の OR ノード(ply 5)で生成される王手の数と各王手の結果
    /// 2. 各王手の先(ply 6 AND ノード)の回避手数と各回避手の結果
    /// を再帰的に調べ，ノード消費のホットスポットを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_bottleneck_analysis() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        // Phase 1: 各応手を 500K でソルブし，NO_MATE / UNKNOWN を特定
        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" 39手詰 ply 4 ボトルネック分析");
        eprintln!("{}", "=".repeat(80));
        eprintln!("\n--- Phase 1: ply 4 応手の概要 (budget=500K) ---\n");
        eprintln!("{:>3} {:>8} {:>5} {:>10} {:>10.1} {:>10}",
            "#", "Move", "Type", "Nodes", "Time(ms)", "Result");
        eprintln!("{}", "-".repeat(55));

        let mut hard_defenses: Vec<(Move, String, u64, String)> = Vec::new();

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            let mut s = DfPnSolver::with_timeout(37, 500_000, 32767, 10);
            s.set_find_shortest(false);
            let start = Instant::now();
            let result = s.solve(&mut child_board);
            let elapsed = start.elapsed();

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            eprintln!("{:>3} {:>8} {:>5} {:>10} {:>10.1} {:>10}",
                i + 1, defense.to_usi(), move_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str);

            // 500K 以上消費 or UNKNOWN → ボトルネック候補
            if nodes >= 200_000 {
                hard_defenses.push((
                    defense, defense.to_usi(), nodes, result_str.clone(),
                ));
            }
        }

        // Phase 2: ボトルネック応手の内部構造を分析
        // check 後は defender turn なので，各回避手を個別に attacker turn でソルブ
        eprintln!("\n--- Phase 2: ボトルネック応手を深掘り ---");
        eprintln!("(defense → check → reply → attacker 視点でソルブ)\n");

        for (defense, def_usi, parent_nodes, parent_result) in &hard_defenses {
            let mut def_board = board.clone();
            def_board.do_move(*defense);

            // ply 5: 攻め方番(OR) — 王手を列挙
            let checks = solver.generate_check_moves(&mut def_board);

            eprintln!("=== {} ({}，親ノード={}，王手数={}) ===\n",
                def_usi, parent_result, parent_nodes, checks.len());

            for (j, &check) in checks.iter().enumerate() {
                let mut check_board = def_board.clone();
                check_board.do_move(check);

                // ply 6: 守備方番(AND) — 回避手を列挙
                let defs_after = solver.generate_defense_moves(&mut check_board);
                let def_count = defs_after.len();
                let check_type = if check.is_drop() { "drop" } else { "move" };

                eprintln!("  王手 {:>2}. {} ({}) → 回避手 {} 手",
                    j + 1, check.to_usi(), check_type, def_count);

                if def_count == 0 {
                    eprintln!("    → 応手なし(詰み)\n");
                    continue;
                }

                eprintln!("  {:>4} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}",
                    "#", "Reply", "Type", "Nodes", "ms", "Result", "Chks");
                eprintln!("  {}", "-".repeat(58));

                let mut total_nodes: u64 = 0;
                let mut nm_count = 0;
                let mut mate_count = 0;
                let mut unk_count = 0;

                for (k, &reply) in defs_after.iter().enumerate() {
                    let mut reply_board = check_board.clone();
                    reply_board.do_move(reply);

                    // ply 7: 攻め方番(OR) — 正しい視点でソルブ
                    let next_checks = solver.generate_check_moves(&mut reply_board);
                    let chk_count = next_checks.len();

                    let mut s = DfPnSolver::with_timeout(33, 200_000, 32767, 5);
                    s.set_find_shortest(false);
                    let start = Instant::now();
                    let result = s.solve(&mut reply_board);
                    let elapsed = start.elapsed();

                    let reply_type = if reply.is_drop() { "drop" } else { "move" };
                    let (result_str, nodes) = match &result {
                        TsumeResult::Checkmate { moves, nodes_searched } => {
                            mate_count += 1;
                            (format!("MATE({})", moves.len()), *nodes_searched)
                        }
                        TsumeResult::CheckmateNoPv { nodes_searched } => {
                            mate_count += 1;
                            ("MATE(nopv)".into(), *nodes_searched)
                        }
                        TsumeResult::NoCheckmate { nodes_searched } => {
                            nm_count += 1;
                            ("NM".into(), *nodes_searched)
                        }
                        TsumeResult::Unknown { nodes_searched } => {
                            unk_count += 1;
                            ("UNK".into(), *nodes_searched)
                        }
                    };
                    total_nodes += nodes;

                    let heavy = if nodes >= 50_000 { " <<<" } else { "" };
                    eprintln!("  {:>4} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}{}",
                        k + 1, reply.to_usi(), reply_type, nodes,
                        elapsed.as_secs_f64() * 1000.0, result_str, chk_count, heavy);
                }
                eprintln!("  合計: {} nodes | MATE={} NM={} UNK={}\n",
                    total_nodes, mate_count, nm_count, unk_count);
            }
        }

        // Phase 3: PV 上の正解手(4c3d)後を 2 階層深掘り
        eprintln!("--- Phase 3: 正解手 4c3d → 1b2c(PV) 後の回避手分析 ---\n");
        let correct_def = board.move_from_usi("4c3d").unwrap();
        let mut correct_board = board.clone();
        correct_board.do_move(correct_def);

        let pv_check = correct_board.move_from_usi("1b2c").unwrap();
        let mut pv_board = correct_board.clone();
        pv_board.do_move(pv_check);

        let pv_defs = solver.generate_defense_moves(&mut pv_board);
        eprintln!("4c3d → 1b2c 後の回避手: {} 手", pv_defs.len());
        eprintln!("{:>3} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}",
            "#", "Reply", "Type", "Nodes", "ms", "Result", "Chks");
        eprintln!("{}", "-".repeat(55));

        for (k, &reply) in pv_defs.iter().enumerate() {
            let mut reply_board = pv_board.clone();
            reply_board.do_move(reply);

            let next_checks = solver.generate_check_moves(&mut reply_board);
            let chk_count = next_checks.len();

            let mut s = DfPnSolver::with_timeout(33, 2_000_000, 32767, 30);
            s.set_find_shortest(false);
            let start = Instant::now();
            let result = s.solve(&mut reply_board);
            let elapsed = start.elapsed();

            let reply_type = if reply.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            let is_pv = reply.to_usi() == "3d2c";
            let marker = if is_pv { " ← PV" } else { "" };
            let heavy = if nodes >= 100_000 { " <<<" } else { "" };
            eprintln!("{:>3} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}{}{}",
                k + 1, reply.to_usi(), reply_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str, chk_count, heavy, marker);
        }
    }

    /// 39手詰 ply 4 の未解決応手(500K で NO_MATE)を高予算で再調査し，
    /// 真の詰み手数・必要ノード数・分岐の特徴を分析する．
    #[test]
    #[ignore]
    fn test_tsume_39te_hard_defenses_deep() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        // 前回 500K で未解決だった応手のみ高予算で再調査
        let hard_moves = [
            ("4c3b", "king move (3b)"),
            ("4c3d", "king move (3d) [CORRECT]"),
            ("N*5c", "knight drop 5c"),
            ("P*5c", "pawn drop 5c"),
            ("N*6c", "knight drop 6c"),
            ("P*7c", "pawn drop 7c"),
            ("N*7c", "knight drop 7c"),
        ];

        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" 39手詰 ply 4: 高予算(5M)でのボトルネック応手分析");
        eprintln!("{}", "=".repeat(80));
        eprintln!("\n{:>12} {:>25} {:>10} {:>8} {:>10}",
            "Move", "Description", "Nodes", "Time(s)", "Result");
        eprintln!("{}", "-".repeat(75));

        for (usi, desc) in &hard_moves {
            let m = board.move_from_usi(usi).unwrap();
            let mut child_board = board.clone();
            child_board.do_move(m);

            let mut solver = DfPnSolver::with_timeout(37, 5_000_000, 32767, 120);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            eprintln!("{:>12} {:>25} {:>10} {:>8.1} {:>10}",
                usi, desc, nodes, elapsed.as_secs_f64(), result_str);

            // 解けた場合は PV を表示
            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!("             PV: {}", pv.join(" "));
            }
        }

        // 正解手(4c3d)の PV を辿り，各 ply での分岐数とノードを段階的に分析
        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" 正解 PV 沿いの IDS 各段階での進捗");
        eprintln!("{}", "=".repeat(80));

        let _pv_usi = [
            "4c3d", "1b2c", "3d2c", "N*1e", "2c3b", "N*2d",
            "3b2b", "2d1b+", "2b3b", "1b2b", "3b2b", "4f1c",
            "2b1c", "9c3c", "1c1d", "3c2c", "1d1e", "P*1f",
            "1e1f", "P*1g", "1f1g", "5g6f", "1g1h", "2c2g",
            "1h1i", "8g8i", "S*6i", "8i6i", "6h6i+", "S*2h",
            "1i2i", "2h3g", "2i3i", "2g2h", "3i4i", "2h4h",
        ];

        // IDS depth 5,9,13,...,41 での進捗
        let depths = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41];

        let mut pv_board = board.clone();
        let correct_def = board.move_from_usi("4c3d").unwrap();
        pv_board.do_move(correct_def);
        // 4c3d 後の局面 = ply 5 (攻め方番 OR)

        eprintln!("\n{:>6} {:>10} {:>8} {:>10}",
            "Depth", "Nodes", "Time(s)", "Result");
        eprintln!("{}", "-".repeat(40));

        for &depth in &depths {
            let mut solver = DfPnSolver::with_timeout(depth, 2_000_000, 32767, 30);
            solver.set_find_shortest(false);
            let mut test_board = pv_board.clone();
            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            eprintln!("{:>6} {:>10} {:>8.1} {:>10}", depth, nodes, elapsed.as_secs_f64(), result_str);

            if let TsumeResult::Checkmate { .. } = &result {
                break; // 解けたら終了
            }
        }
    }

    /// 全王手 NM 局面の検出テスト．
    ///
    /// N*1e → 2c1d 後の局面は全8王手が1ノードで NM(不詰)．
    ///
    /// depth=1 では構造的に NM を検出できる(全王手が即座に反証される)．
    /// depth=33 では IDS の浅い反復の NM エントリ(remaining=小)が深い反復の
    /// look_up で再利用されないため，構造的証明(REMAINING_INFINITE)に
    /// 到達できず Unknown になる場合がある．これは安全側の挙動であり，
    /// depth 制限由来の仮 NM を真の不詰に昇格しない設計の帰結である
    /// (39手詰め偽陽性防止のため)．
    #[test]
    fn test_all_checks_nm_but_solve_returns_unk() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e", "2c1d",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 8 後: 攻め方番(OR ノード), 全8王手が NM
        assert_eq!(board.turn, Color::Black);

        // depth=1 では全王手が即座に反証され NM を検出する．
        {
            let mut s = DfPnSolver::new(1, 1_000_000, 32767);
            s.set_find_shortest(false);
            let r = s.solve(&mut board);
            match &r {
                TsumeResult::NoCheckmate { .. } => {}
                other => panic!(
                    "depth=1: expected NoCheckmate, got {:?}",
                    other
                ),
            }
        }

        // depth=3 で Checkmate 偽陽性が発生しないことを検証する．
        // この局面は真の不詰であるため Checkmate を返してはならない．
        // デバッグビルドでの実行時間制約(3分)のため予算・深さを抑制する．
        let mut solver = DfPnSolver::new(3, 10_000, 32767);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {
                // 構造的 NM 検出または予算内未収束: どちらも許容
            }
            other => {
                panic!("Unexpected result: {:?}", match other {
                    TsumeResult::Checkmate { moves, nodes_searched } =>
                        format!("Checkmate({} moves, {} nodes)", moves.len(), nodes_searched),
                    TsumeResult::CheckmateNoPv { nodes_searched } =>
                        format!("CheckmateNoPv({} nodes)", nodes_searched),
                    _ => "?".to_string(),
                });
            }
        }
    }

    /// 39手詰: N*1e → 2c1d 後の OR ノード(ply 8)を深掘りし，
    /// どの王手でノード爆発が起きるかを特定する．
    ///
    /// 2c1d は N*1e の AND 子ノードで 500K budget では未解決．
    /// ply 8 後は攻め方番(OR)なので，各王手候補を 1M budget で個別ソルブし，
    /// さらに解けない王手については AND 子ノード(回避手)を個別に分析する．
    #[test]
    #[ignore]
    fn test_tsume_39te_2c1d_deep_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        // 8手進めて N*1e → 2c1d 後の局面へ
        let pv_setup = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e", "2c1d",
        ];

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 8 後: 攻め方番(OR ノード)
        assert_eq!(board.turn, Color::Black);

        let mut helper = DfPnSolver::default_solver();
        let checks = helper.generate_check_moves(&mut board);

        eprintln!("\n{}", "=".repeat(80));
        eprintln!(" 39手詰 N*1e → 2c1d 深掘り分析 (ply 8 OR ノード)");
        eprintln!(" 局面: {} (8手進めた後)", board.sfen());
        eprintln!(" 王手候補数: {} (うちドロップ: {})",
            checks.len(),
            checks.iter().filter(|m| m.is_drop()).count());
        eprintln!("{}", "=".repeat(80));

        // ── N*1e の AND 子ノード(回避手)を 1M budget で各ソルブ ──
        // ply 7 まで戻って N*1e 後の局面を作る
        let mut board_after_n1e = Board::new();
        board_after_n1e.set_sfen(sfen).unwrap();
        let pv_to_n1e = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e",
        ];
        for usi in &pv_to_n1e {
            let m = board_after_n1e.move_from_usi(usi).unwrap();
            board_after_n1e.do_move(m);
        }
        // N*1e 後: 守備方番(AND ノード)
        assert_eq!(board_after_n1e.turn, Color::White);

        let defenses = helper.generate_defense_moves(&mut board_after_n1e);

        eprintln!("\n--- N*1e 後の AND 子ノード(回避手)分析 (budget=1M) ---");
        eprintln!("回避手数: {}\n", defenses.len());
        eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
            "#", "Defense", "Type", "Nodes", "Time(s)", "Result");
        eprintln!("{}", "-".repeat(55));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board_after_n1e.clone();
            child_board.do_move(defense);

            // depth=33 (残り 39-8=31 手 + margin 2), budget=1M
            let mut solver = DfPnSolver::with_timeout(33, 1_000_000, 32767, 60);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let def_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            let heavy = if nodes >= 500_000 { " <<<" } else { "" };
            eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}{}",
                i + 1, defense.to_usi(), def_type, nodes,
                elapsed.as_secs_f64(), result_str, heavy);

            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().take(10).map(|m| m.to_usi()).collect();
                let suffix = if moves.len() > 10 { " ..." } else { "" };
                eprintln!("    PV: {}{}", pv.join(" "), suffix);
            }
        }

        // ── 2c1d 後の各王手を 1M budget で個別ソルブ ──
        eprintln!("\n--- 2c1d 後の各王手候補 (budget=1M, depth=33) ---\n");
        eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
            "#", "Check", "Type", "Nodes", "Time(s)", "Result");
        eprintln!("{}", "-".repeat(55));

        for (i, &check) in checks.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(check);

            let mut solver = DfPnSolver::with_timeout(33, 1_000_000, 32767, 60);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let check_type = if check.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
                i + 1, check.to_usi(), check_type, nodes,
                elapsed.as_secs_f64(), result_str);

            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().take(10).map(|m| m.to_usi()).collect();
                let suffix = if moves.len() > 10 { " ..." } else { "" };
                eprintln!("    PV: {}{}", pv.join(" "), suffix);
            }
        }
    }

    /// TT 診断テスト: 指定 ply の指定手で TT エントリの変化をモニタリングする．
    ///
    /// `--features tt_diag` でビルドし `cargo test --features tt_diag -- --nocapture` で実行．
    /// stderr に `[tt_diag]` プレフィックスのログが出力される．
    #[test]
    #[ignore]
    fn test_tt_diag_monitor() {
        // 39手詰め問題: PV を24手進めた局面(ply 24 = 攻め番)から開始
        // PV: 7b6b 5b4c ... P*1g 1f1g ← ここまで24手
        // この局面から ply 0 = 5g6f(攻め)，ply 1 = 応手(P*7g 等)
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV を24手進める
        let pv_24 = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
        ];
        for usi in &pv_24 {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        eprintln!("\n=== TT Diag: ply 24 局面(攻め番) → P*7g 調査 ===");
        eprintln!("SFEN: {}", board.sfen());

        // ply 1 の AND ノード(応手)をモニタリング
        // ply 0 = 5g6f(攻め)，ply 1 = 応手(P*7g 含む合駒)
        // (1) この局面で王手が生成されるか確認
        let check_solver = DfPnSolver::default_solver();
        let checks = check_solver.generate_check_moves(&mut board);
        eprintln!("Check moves from ply 24 ({}):", checks.len());
        for m in &checks {
            eprintln!("  {}", m.to_usi());
        }

        // (2) IDS の各深さでの結果を個別に確認(MIDのみ，PNSスキップ)
        // 浅い深さで不当な NoMate/disproof が出ていないか
        for depth in (3..=17).step_by(2) {
            let mut s = DfPnSolver::with_timeout(depth, 50_000, 32767, 5);
            s.set_find_shortest(false);
            s.attacker = board.turn;
            s.table.clear();
            s.nodes_searched = 0;
            s.max_ply = 0;
            s.path.clear();
            s.killer_table.clear();
            s.start_time = Instant::now();
            s.timed_out = false;
            s.next_gc_check = 100_000;
            let mut b = board.clone();
            s.mid_fallback(&mut b);
            let pk = position_key(&b);
            let att_hand = b.hand[s.attacker.index()];
            let (root_pn, root_dn, _) = s.look_up_pn_dn(pk, &att_hand, depth as u16);
            let r_str = if root_pn == 0 {
                "Mate".to_string()
            } else if root_dn == 0 {
                "NoMate".to_string()
            } else {
                format!("Unknown(pn={},dn={})", root_pn, root_dn)
            };
            eprintln!(
                "  depth={:2} → {} nodes={} max_ply={} tt_pos={}",
                depth, r_str, s.nodes_searched, s.max_ply, s.table.len(),
            );
            if root_pn == 0 {
                break;
            }
        }
    }

    /// チェーン合駒最適化の動作検証テスト．
    ///
    /// 39手詰め ply 25 AND ノード(5g6f 後)を対象に，以下の最適化が
    /// 正常に機能しているかをモニタリングする:
    ///
    /// 1. 合駒遅延展開(deferred children)の逐次活性化
    /// 2. チェーンドロップ3カテゴリ制限
    /// 3. 同一マス証明転用(cross_deduce_deferred)
    /// 4. TT ベース合駒プレフィルタ(try_prefilter_block)
    /// ply 24 の depth 問題を診断する．
    ///
    /// 非 PV 変化では PV より長い詰み手順が存在するため，
    /// `depth = remaining + 2` では不足する．depth=41(最大)で
    /// 解けるかを確認し，必要予算を推定する．
    #[test]
    #[ignore]
    fn test_1g1f_nomate_verification() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f",
        ];
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        // ply 25 の局面を構築 (5g6f 後)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for &mv in pv {
            let m = board.move_from_usi(mv).unwrap();
            board.do_move(m);
        }

        // 1g1f (玉逃げ) を指す
        let def = board.move_from_usi("1g1f").unwrap();
        board.do_move(def);

        eprintln!("=== 1g1f 後の局面 (OR node, 攻め番) ===");
        eprintln!("SFEN: {}", board.sfen());

        // 王手生成で攻め手を確認
        let mut check_solver = DfPnSolver::default_solver();
        let checks = check_solver.generate_check_moves(&mut board);
        eprintln!("Check moves: {} {:?}", checks.len(),
            checks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());

        // 段階的に深さを増やして解析
        for depth in [15u32, 21, 31, 41] {
            let mut solver = DfPnSolver::with_timeout(depth, 5_000_000, 32767, 60);
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut board);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            eprintln!("depth={}: {} nodes={} time={:.2}s TT_pos={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.table.len());
        }

        // 旧互換用: 最後の結果を使う
        let mut solver = DfPnSolver::with_timeout(15, 5_000_000, 32767, 120);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        eprintln!("Result: {:?}", match &result {
            TsumeResult::Checkmate { moves, .. } =>
                format!("Mate({})", moves.len()),
            TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
            TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
            TsumeResult::Unknown { .. } => "Unknown".to_string(),
        });
        eprintln!("Nodes: {}, TT_pos: {}", solver.nodes_searched, solver.table.len());

        // N*6g 後の各攻め手を depth=41 で解く
        {
            let mut brd_n6g = Board::new();
            brd_n6g.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = brd_n6g.move_from_usi(mv).unwrap();
                brd_n6g.do_move(m);
            }
            let def_n6g = brd_n6g.move_from_usi("N*6g").unwrap();
            brd_n6g.do_move(def_n6g);

            eprintln!("\n=== N*6g 後の各攻め手サブ問題 (depth=41, 1M) ===");
            eprintln!("SFEN: {}", brd_n6g.sfen());

            let mut gen = DfPnSolver::default_solver();
            let attacks = gen.generate_check_moves(&mut brd_n6g);
            eprintln!("Check moves: {} {:?}", attacks.len(),
                attacks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());

            for atk in &attacks {
                let mut after_atk = brd_n6g.clone();
                after_atk.do_move(*atk);

                let mut sub = DfPnSolver::with_timeout(41, 1_000_000, 32767, 60);
                sub.set_find_shortest(false);
                let start = Instant::now();
                let r = sub.solve(&mut after_atk);
                let elapsed = start.elapsed();

                let r_str = match &r {
                    TsumeResult::Checkmate { moves, .. } =>
                        format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                eprintln!("  {} → {} nodes={} time={:.2}s TT_pos={}",
                    atk.to_usi(), r_str, sub.nodes_searched,
                    elapsed.as_secs_f64(), sub.table.len());
            }
        }

        // 残り3応手も段階的深さテスト
        for def_usi in ["N*6g", "P*7g", "N*7g"] {
            let mut brd = Board::new();
            brd.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = brd.move_from_usi(mv).unwrap();
                brd.do_move(m);
            }
            let def = brd.move_from_usi(def_usi).unwrap();
            brd.do_move(def);

            eprintln!("\n=== {} 後の局面 (OR node, 攻め番) ===", def_usi);
            eprintln!("SFEN: {}", brd.sfen());

            for depth in [15u32, 21, 31, 41] {
                let mut s = DfPnSolver::with_timeout(depth, 5_000_000, 32767, 60);
                s.set_find_shortest(false);

                let start = Instant::now();
                let r = s.solve(&mut brd);
                let elapsed = start.elapsed();

                let r_str = match &r {
                    TsumeResult::Checkmate { moves, .. } =>
                        format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                eprintln!("depth={}: {} nodes={} time={:.2}s TT_pos={}",
                    depth, r_str, s.nodes_searched, elapsed.as_secs_f64(),
                    s.table.len());
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// N*6g → 8g6g 後のボトルネック局面調査(診断用)．
    #[test]
    #[ignore]
    fn test_n6g_bottleneck_diagnosis() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f",
        ];
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        // ply 25 の局面を構築 (5g6f 後)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for &mv in pv {
            let m = board.move_from_usi(mv).unwrap();
            board.do_move(m);
        }
        eprintln!("=== ply 25 AND (5g6f 後) ===");
        eprintln!("SFEN: {}", board.sfen());

        // N*6g を指す
        let n6g = board.move_from_usi("N*6g").unwrap();
        board.do_move(n6g);
        eprintln!("\n=== N*6g 後 (OR node) ===");
        eprintln!("SFEN: {}", board.sfen());

        // 8g6g(飛車で桂を取る)
        let r6g = board.move_from_usi("8g6g").unwrap();
        board.do_move(r6g);
        eprintln!("\n=== 8g6g 後 (AND node, 玉方番) ===");
        eprintln!("SFEN: {}", board.sfen());

        // この局面の情報
        let defender = board.turn();
        let attacker = defender.opponent();
        let king_sq = board.king_square(defender).unwrap();
        eprintln!("King: {}{}  turn: {:?}",
            9 - king_sq.col(), (b'a' + king_sq.row()) as char, defender);

        let checkers = board.compute_checkers_at(king_sq, attacker);
        eprintln!("Checkers: {}", checkers.count());
        for sq in checkers {
            let piece = board.squares[sq.index()];
            eprintln!("  {:?} at {}{}", piece, 9 - sq.col(), (b'a' + sq.row()) as char);
        }

        let mut solver = DfPnSolver::default_solver();

        // between / futile / chain
        let checker_sq = checkers.lsb().unwrap();
        let sliding = solver.find_sliding_checker(&board, king_sq, attacker);
        eprintln!("Sliding checker: {:?}", sliding.is_some());

        if sliding.is_some() {
            let between = attack::between_bb(checker_sq, king_sq);
            let (futile, chain) = solver.compute_futile_and_chain_squares(
                &board, &between, king_sq, checker_sq, defender, attacker,
            );
            let normal_count = between.count() - futile.count() - chain.count();
            eprintln!("Between: {}  Futile: {}  Chain: {}  Normal: {}",
                between.count(), futile.count(), chain.count(), normal_count);
            for sq in between {
                let tag = if futile.contains(sq) { "futile" }
                         else if chain.contains(sq) { "chain" }
                         else { "normal" };
                eprintln!("  {}{} = {}",
                    9 - sq.col(), (b'a' + sq.row()) as char, tag);
            }
        }

        // 防御手一覧
        let defenses = solver.generate_defense_moves(&mut board);
        eprintln!("\nDefense moves ({}):", defenses.len());
        let drops: Vec<_> = defenses.iter().filter(|m| m.is_drop()).collect();
        let non_drops: Vec<_> = defenses.iter().filter(|m| !m.is_drop()).collect();
        eprintln!("  Non-drops ({}):", non_drops.len());
        for m in &non_drops {
            eprintln!("    {}", m.to_usi());
        }
        eprintln!("  Drops ({}):", drops.len());
        for m in &drops {
            eprintln!("    {}", m.to_usi());
        }

        // N*6g 後(OR node, 攻め方手番)から解く
        let mut or_board = {
            let mut b = Board::new();
            b.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = b.move_from_usi(mv).unwrap();
                b.do_move(m);
            }
            let n6g_m = b.move_from_usi("N*6g").unwrap();
            b.do_move(n6g_m);
            b
        };

        // MID のみ(PNS skip)で解く
        eprintln!("\n=== N*6g 後 MID only (depth=41, 10M, 300s) ===");
        eprintln!("SFEN: {}", or_board.sfen());
        {
            let mut s = DfPnSolver::with_timeout(41, 10_000_000, 32767, 300);
            s.set_find_shortest(false);
            s.table.clear();
            s.nodes_searched = 0;
            s.max_ply = 0;
            s.path.clear();
            s.killer_table.clear();
            s.start_time = std::time::Instant::now();
            s.timed_out = false;
            s.attacker = or_board.turn;
            // mid_fallback を直接呼ぶ
            s.mid_fallback(&mut or_board);
            let (root_pn, root_dn) = s.look_up_board(&or_board);
            let r_str = if root_pn == 0 { "Proven" }
                        else if root_dn == 0 { "Disproven" }
                        else { "Unknown" };
            eprintln!("  → {} pn={} dn={} searched={} time={:.2}s TT={}",
                r_str, root_pn, root_dn, s.nodes_searched,
                s.start_time.elapsed().as_secs_f64(), s.table.len());
            eprintln!("  prefilter_hits={}", s.prefilter_hits);
            #[cfg(feature = "tt_diag")]
            eprintln!("  deferred: act={} enqueued={} ready={} not_ready={} cross={}",
                s.diag_mid_deferred_activations,
                s.diag_deferred_enqueued,
                s.diag_deferred_ready,
                s.diag_deferred_not_ready,
                s.diag_cross_deduce_hits);
        }

        // 8g6g 後の各防御手を個別に OR node として解く
        eprintln!("\n=== 8g6g 後 → 各防御手のサブ問題 (depth=41, 2M) ===");
        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            let mut s = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
            s.set_find_shortest(false);
            let start = std::time::Instant::now();
            let r = s.solve(&mut after_def);
            let elapsed = start.elapsed();
            let r_str = match &r {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            eprintln!("  {} → {} searched={} time={:.2}s TT={}",
                def_mv.to_usi(), r_str, s.nodes_searched, elapsed.as_secs_f64(), s.table.len());
        }

        // Unknown が出た防御手を掘り下げ(2段目: 攻め手→防御手)
        eprintln!("\n=== Unknown 防御手の攻め手別サブ問題 (depth=41, 2M) ===");
        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            let mut s_check = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
            s_check.set_find_shortest(false);
            let r_check = s_check.solve(&mut after_def);
            if !matches!(r_check, TsumeResult::Unknown { .. }) {
                continue;
            }

            eprintln!("--- {} (Unknown) → 攻め手展開 ---", def_mv.to_usi());
            let mut gen = DfPnSolver::default_solver();
            let attacks = gen.generate_check_moves(&mut after_def);
            eprintln!("  攻め手数: {}", attacks.len());
            for atk in &attacks {
                let mut after_atk = after_def.clone();
                after_atk.do_move(*atk);

                let mut s2 = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
                s2.set_find_shortest(false);
                let start2 = std::time::Instant::now();
                let r2 = s2.solve(&mut after_atk);
                let elapsed2 = start2.elapsed();
                let r2_str = match &r2 {
                    TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                eprintln!("    {} → {} searched={} time={:.2}s TT={}",
                    atk.to_usi(), r2_str, s2.nodes_searched, elapsed2.as_secs_f64(), s2.table.len());
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// 5. TT エントリ数の推移
    #[test]
    #[cfg(feature = "tt_diag")]
    fn test_chain_interpose_diagnostics() {
        use std::io::Write;
        let out_path = "/tmp/chain_interpose_diag.log";
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " チェーン合駒最適化 動作検証").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // ply 25 AND ノード(5g6f 後): 飛車8g→玉1gの横利き開き王手
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";

        let mut board = Board::new();
        board.set_sfen(and_sfen).unwrap();

        // ========================================
        // Phase 1: チェーンドロップ3カテゴリ制限の検証
        // ========================================
        writeln!(out, "\n--- Phase 1: チェーンドロップ3カテゴリ制限 ---\n").unwrap();

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let chain_bb = solver.chain_bb_cache;

        writeln!(out, "chain_bb: {:?}", chain_bb).unwrap();
        writeln!(out, "全回避手数: {}", defenses.len()).unwrap();

        // チェーンマスへのドロップを抽出
        let mut chain_drops: Vec<String> = Vec::new();
        let mut normal_drops: Vec<String> = Vec::new();
        let mut non_drops: Vec<String> = Vec::new();

        for m in &defenses {
            if m.is_drop() {
                let to = m.to_sq();
                if chain_bb.contains(to) {
                    chain_drops.push(m.to_usi());
                } else {
                    normal_drops.push(m.to_usi());
                }
            } else {
                non_drops.push(m.to_usi());
            }
        }

        writeln!(out, "チェーンマスへのドロップ: {:?}", chain_drops).unwrap();
        writeln!(out, "通常マスへのドロップ: {:?}", normal_drops).unwrap();
        writeln!(out, "駒移動: {:?}", non_drops).unwrap();

        // チェーンマスへのドロップが3カテゴリ制限に従っているか検証
        // 各マスごとにグループ化
        use std::collections::HashMap;
        let mut drops_by_sq: HashMap<String, Vec<String>> = HashMap::new();
        for m in &defenses {
            if m.is_drop() && chain_bb.contains(m.to_sq()) {
                let sq_str = format!("{}{}",
                    (b'1' + (8 - m.to_sq().col())) as char,
                    (b'a' + m.to_sq().row()) as char,
                );
                let pt = m.drop_piece_type().unwrap();
                let pt_str = match pt {
                    PieceType::Pawn => "P", PieceType::Lance => "L",
                    PieceType::Knight => "N", PieceType::Silver => "S",
                    PieceType::Gold => "G", PieceType::Bishop => "B",
                    PieceType::Rook => "R", _ => "?",
                };
                drops_by_sq.entry(sq_str).or_default().push(pt_str.to_string());
            }
        }

        let mut all_ok = true;
        for (sq, pieces) in &drops_by_sq {
            writeln!(out, "  チェーンマス {}: 駒種={:?} ({}個)", sq, pieces, pieces.len()).unwrap();
            // 3カテゴリ制限: 最大3手(前方系1 + 角1 + 桂1)
            if pieces.len() > 3 {
                writeln!(out, "  *** 異常: 3カテゴリ制限違反! {} > 3 ***", pieces.len()).unwrap();
                all_ok = false;
            }
            // 前方利き系は最弱の1つのみ
            let forward_count = pieces.iter().filter(|p| {
                matches!(p.as_str(), "P" | "L" | "S" | "G" | "R")
            }).count();
            if forward_count > 1 {
                writeln!(out, "  *** 異常: 前方利き系 {} > 1 ***", forward_count).unwrap();
                all_ok = false;
            }
        }
        writeln!(out, "3カテゴリ制限: {}", if all_ok { "OK" } else { "*** NG ***" }).unwrap();

        // ========================================
        // Phase 2: 各応手の探索とTTモニタリング
        // ========================================
        writeln!(out, "\n--- Phase 2: 応手別探索 + TT/最適化モニタリング ---\n").unwrap();

        // P*3g (短いチェーン) と P*7g (長いチェーン) を比較
        let target_moves = ["P*3g", "B*3g", "N*3g", "P*7g", "N*7g", "1g1f", "1g1h"];

        writeln!(out, "{:<10} {:<10} {:<12} {:<10} {:<17} {:<17} {:<10}",
            "Move", "Nodes", "TT_pos", "Prefilter", "Defer(MID/PNS)", "XDeduce/PNSprov", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(90)).unwrap();

        let mut phase2_cross_deduce_total: u64 = 0;
        let mut phase2_pns_proven_total: u64 = 0;

        for &mv_usi in &target_moves {
            let m = match board.move_from_usi(mv_usi) {
                Some(m) => m,
                None => {
                    writeln!(out, "{:<10} (invalid move)", mv_usi).unwrap();
                    continue;
                }
            };

            let mut after_def = board.clone();
            after_def.do_move(m);

            let def_remaining = 13; // 39 - 24 - 1(攻め) - 1(受け) = 13
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 250_000, 32767, 30,
            );
            sub_solver.set_find_shortest(false);

            let mut sub_board = after_def.clone();
            let sub_result = sub_solver.solve(&mut sub_board);

            let result_str = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };

            phase2_cross_deduce_total += sub_solver.diag_cross_deduce_hits;
            phase2_pns_proven_total += sub_solver.diag_pns_deferred_already_proven;

            let sub_proven = sub_solver.table.count_proven();
            let sub_disproven = sub_solver.table.count_disproven();
            writeln!(out, "{:<10} {:<10} {:<12} {:<10} {:<8}/{:<8} {:<8}/{:<8} {:<10}",
                mv_usi,
                sub_solver.nodes_searched,
                sub_solver.table.len(),
                sub_solver.prefilter_hits,
                sub_solver.diag_mid_deferred_activations,
                sub_solver.diag_pns_deferred_activations,
                sub_solver.diag_cross_deduce_hits,
                sub_solver.diag_pns_deferred_already_proven,
                result_str,
            ).unwrap();
            writeln!(out, "{:<10} TT: proven={}, disproven={}, total={}",
                "", sub_proven, sub_disproven, sub_solver.table.total_entries(),
            ).unwrap();
        }

        writeln!(out, "\nPhase 2 合計: cross_deduce={}, pns_proven={}",
            phase2_cross_deduce_total, phase2_pns_proven_total).unwrap();

        // ========================================
        // Phase 3: ply 24 全体の TT 推移モニタリング
        // ========================================
        writeln!(out, "\n--- Phase 3: ply 24 全体探索の TT 推移 ---\n").unwrap();

        // PV を ply 24 まで進めて OR ノードから探索
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
        ];
        let mut board24 = Board::new();
        board24.set_sfen(sfen).unwrap();
        for mv in &pv {
            let m = board24.move_from_usi(mv).unwrap();
            board24.do_move(m);
        }

        let node_limit: u64 = 500_000;
        let depth = 17u32; // remaining 15 + 2
        let mut full_solver = DfPnSolver::with_timeout(
            depth, node_limit, 32767, 60,
        );
        full_solver.set_find_shortest(false);

        let start = Instant::now();
        let full_result = full_solver.solve(&mut board24);
        let elapsed = start.elapsed();

        let result_str = match &full_result {
            TsumeResult::Checkmate { moves, .. } =>
                format!("Mate({})", moves.len()),
            TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
            TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
            TsumeResult::Unknown { .. } => "Unknown".to_string(),
        };

        let total_deferred = full_solver.diag_mid_deferred_activations
            + full_solver.diag_pns_deferred_activations;
        writeln!(out, "Result: {}", result_str).unwrap();
        writeln!(out, "Nodes: {}", full_solver.nodes_searched).unwrap();
        writeln!(out, "Time: {:.2}s", elapsed.as_secs_f64()).unwrap();
        writeln!(out, "NPS: {:.0}", full_solver.nodes_searched as f64 / elapsed.as_secs_f64()).unwrap();
        let tt_total = full_solver.table.total_entries();
        let tt_proven = full_solver.table.count_proven();
        let tt_disproven = full_solver.table.count_disproven();
        let tt_intermediate = full_solver.table.count_intermediate();
        writeln!(out, "TT positions: {}", full_solver.table.len()).unwrap();
        writeln!(out, "TT entries: {} (proven={}, disproven={}, intermediate={})",
            tt_total, tt_proven, tt_disproven, tt_intermediate).unwrap();
        writeln!(out, "Prefilter hits: {}", full_solver.prefilter_hits).unwrap();
        writeln!(out, "Deferred activations: {} (MID={}, PNS={})",
            total_deferred,
            full_solver.diag_mid_deferred_activations,
            full_solver.diag_pns_deferred_activations).unwrap();
        writeln!(out, "PNS deferred already proven: {}", full_solver.diag_pns_deferred_already_proven).unwrap();
        writeln!(out, "Cross-deduce hits (MID): {}", full_solver.diag_cross_deduce_hits).unwrap();

        // ========================================
        // Phase 4: 正常性チェック
        // ========================================
        writeln!(out, "\n--- Phase 4: 正常性チェック ---\n").unwrap();

        let mut checks_passed = 0;
        let mut checks_failed = 0;

        // Check 1: prefilter_hits > 0 (プレフィルタが動作している)
        if full_solver.prefilter_hits > 0 {
            writeln!(out, "[OK] prefilter_hits = {} (> 0)", full_solver.prefilter_hits).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] prefilter_hits = 0 (プレフィルタが動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 2: deferred_activations > 0 (遅延展開が動作している)
        if total_deferred > 0 {
            writeln!(out, "[OK] deferred_activations = {} (> 0, MID={} PNS={})",
                total_deferred,
                full_solver.diag_mid_deferred_activations,
                full_solver.diag_pns_deferred_activations).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] deferred_activations = 0 (遅延展開が動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 3: cross_deduce が per-move テストで動作している
        // 全体探索は PNS 支配のため MID cross_deduce = 0 は想定内．
        // per-move テスト(Phase 2)の合計で検証する．
        if phase2_cross_deduce_total > 0 {
            writeln!(out, "[OK] cross_deduce(Phase2合計) = {} (> 0, MID で TT 転用が動作)",
                phase2_cross_deduce_total).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] cross_deduce(Phase2合計) = 0 (同一マス証明転用が動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 4: 3カテゴリ制限
        if all_ok {
            writeln!(out, "[OK] 3カテゴリ制限: 全チェーンマスで遵守").unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] 3カテゴリ制限: 違反あり").unwrap();
            checks_failed += 1;
        }

        // Check 5: TT entries < nodes (TT がノード数と乖離していない)
        let ratio = full_solver.table.total_entries() as f64
            / full_solver.nodes_searched as f64;
        if ratio < 2.0 {
            writeln!(out, "[OK] TT entries/nodes = {:.2} (< 2.0)", ratio).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] TT entries/nodes = {:.2} (>= 2.0, TT 肥大化の疑い)", ratio).unwrap();
            checks_failed += 1;
        }

        // Check 6: deferred_activations << nodes (バルク活性化していない)
        let act_ratio = total_deferred as f64
            / full_solver.nodes_searched as f64;
        if act_ratio < 0.1 {
            writeln!(out, "[OK] deferred_act/nodes = {:.4} (< 0.1, 逐次活性化)", act_ratio).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] deferred_act/nodes = {:.4} (>= 0.1, 過剰な活性化の疑い)", act_ratio).unwrap();
            checks_failed += 1;
        }

        writeln!(out, "\n合計: {} passed, {} failed", checks_passed, checks_failed).unwrap();

        // テスト結果
        assert!(checks_failed == 0,
            "チェーン合駒最適化の動作検証に失敗: {} 件のチェックが NG (詳細: {})",
            checks_failed, out_path);

            })
            .unwrap()
            .join()
            .unwrap();
        eprintln!("結果: {}", out_path);
    }

}
