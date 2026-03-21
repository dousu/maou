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
const INTERPOSE_DN_BIAS: u32 = INF / 2;

/// 遅延合駒展開の閾値: drop 応手がこの数以上の AND ノードで適用．
///
/// 合駒(drop)が多い AND ノードでは初期 pn が膨張し，
/// 親 OR ノードがこの分岐を後回しにする問題が生じる．
/// この閾値以上の drop がある場合:
/// - MID ループで drop 子を遅延活性化する
/// - 初期 pn 計算で drop を割引く
const LAZY_INTERPOSE_THRESHOLD: u32 = 8;

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
const DEEP_DFPN_R: f32 = 0.5;

/// Deep df-pn による深さ依存の初期値バイアスを計算する．
///
/// `base` は元の初期値(応手数や王手数)，`ply` は根からの深さ．
/// 返り値は `max(base, ceil(R * ply))` で，深い位置ほど大きくなる．
#[inline]
fn depth_biased_dn(base: u32, ply: u32) -> u32 {
    let bias = ((DEEP_DFPN_R * ply as f32).ceil() as u32).max(1);
    base.max(bias)
}

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
        let total_ns = self.position_key_ns
            + self.loop_detect_ns
            + self.tt_lookup_ns
            + self.tt_store_ns
            + self.movegen_check_ns
            + self.movegen_defense_ns
            + self.do_move_ns
            + self.undo_move_ns
            + self.child_init_ns
            + self.main_loop_collect_ns;
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
            ("main_loop_collect", self.main_loop_collect_ns, self.main_loop_collect_count),
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
}

/// PNS アリーナの最大ノード数(メモリ上限)．
///
/// 1ノード ≈ 80〜120 bytes(children Vec 含む)．
/// 2M ノードで約 200〜300 MB を使用する．
const PNS_MAX_ARENA_NODES: usize = 2_000_000;

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
            if e.pn == 0 && hand_gte(hand, &e.hand) {
                return (0, e.dn, e.source);
            }
        }
        for e in entries {
            // 反証済み: 持ち駒が少ない(以下)かつ十分な深さなら再利用
            if e.dn == 0
                && hand_gte(&e.hand, hand)
                && e.remaining >= remaining
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
        let entries =
            self.tt.entry(pos_key).or_default();

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        for e in entries.iter() {
            // 証明済みエントリが支配: hand ≥ e.hand → 新エントリの持ち駒で詰み確定
            if e.pn == 0 && hand_gte(&hand, &e.hand) {
                return;
            }
            // 反証済みエントリが支配: e.hand ≥ hand かつ十分な深さ → 不詰確定
            // GHI: 経路依存の反証は経路非依存の反証に支配されない
            if e.dn == 0
                && !e.path_dependent
                && hand_gte(&e.hand, &hand)
                && e.remaining >= remaining
            {
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
                !hand_gte(&e.hand, &hand)
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, source });
            return;
        }

        if dn == 0 {
            // === 反証済みエントリの挿入 ===
            // パレートフロンティア(最大持ち駒 × 最大remaining 集合)を維持:
            // 新反証に支配される既存エントリを除去する．
            // GHI: 経路非依存の反証は経路依存の反証を支配して置換できる
            entries.retain(|e| {
                // 証明済みは保護
                if e.pn == 0 {
                    return true;
                }
                if e.dn == 0 {
                    if !path_dependent && e.path_dependent {
                        // 経路非依存の新反証は経路依存の既存反証を置換
                        return false;
                    }
                    // 反証: e.hand ≤ hand かつ e.remaining ≤ remaining → 冗長
                    return !(hand_gte(&hand, &e.hand)
                        && remaining >= e.remaining);
                }
                // 中間エントリは保護(remaining の不一致で必要になりうる)
                true
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, source });
            return;
        }

        // === 中間エントリ(pn > 0, dn > 0)の挿入 ===

        // 同一持ち駒の既存エントリを更新
        for e in entries.iter_mut() {
            if e.hand == hand {
                // 証明済み(pn=0)は上の共通チェックで return 済み
                // 反証済み(dn=0)はより深い探索でのみ上書き可能
                if e.dn == 0 && remaining <= e.remaining {
                    return;
                }
                e.pn = pn;
                e.dn = dn;
                e.remaining = remaining;
                e.source = source;
                e.path_dependent = false;
                if best_move != 0 {
                    e.best_move = best_move;
                }
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
                if e.pn == 0 && hand_gte(hand, &e.hand) {
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
    #[inline(always)]
    fn get_disproof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte(&e.hand, hand) {
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
                if e.dn == 0 && hand_gte(&e.hand, hand) {
                    return e.path_dependent;
                }
            }
        }
        false
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
                // 証明のみ保持(反証は浅い探索のコンテキスト依存性があるため除外)
                e.pn == 0
            });
            !entries.is_empty()
        });
    }


    /// TT のポジション数を返す．
    fn len(&self) -> usize {
        self.tt.len()
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
    fn merge_proven(&mut self, other: &TranspositionTable) {
        for (pos_key, entries) in &other.tt {
            for e in entries {
                if e.pn == 0 || e.dn == 0 {
                    // GHI: 経路依存フラグを保持してマージ
                    self.store_path_dep(
                        *pos_key, e.hand, e.pn, e.dn,
                        e.remaining, e.source, e.path_dependent,
                    );
                }
            }
        }
    }

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
    /// 合駒事前証明(interpose pre-solve)の1子あたりノード予算．
    ///
    /// AND ノードの合駒(drop)子ノードに対して，MID ループ前に
    /// クリーンな TT で `mid()` を呼び出し，証明/反証を試みる．
    /// 1つ目の合駒で蓄積された証明が2つ目以降に TT ヒットで伝播する．
    /// 0 にすると事前証明を無効化する．デフォルトは 256．
    interpose_pre_solve_nodes: u64,
    /// TT GC 閾値: TT のポジション数がこの値を超えると GC を実行する．
    ///
    /// 0 にすると GC を無効化する．
    /// デフォルトは 0(無効)．超長手数問題で OOM を防ぐ場合に設定する．
    /// 推奨値: 探索ノード数の 1/5〜1/2 程度(例: 100M ノードなら 20M〜50M)．
    tt_gc_threshold: usize,
    /// 合駒事前証明の再帰防止フラグ．
    ///
    /// Pre-Solve 内の mid() が再帰的に AND ノードに到達した場合，
    /// ネストした Pre-Solve によるスタックオーバーフローを防ぐ．
    in_pre_solve: bool,
    /// TT ベース合駒プレフィルタの発火回数(診断用)．
    prefilter_hits: u64,
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
            interpose_pre_solve_nodes: 256,
            in_pre_solve: false,
            prefilter_hits: 0,
            tt_gc_threshold: 0,
            next_gc_check: 0,
            killer_table: Vec::new(),
            table: TranspositionTable::new(),
            nodes_searched: 0,
            max_ply: 0,
            path: FxHashSet::default(),
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            #[cfg(feature = "profile")]
            profile_stats: ProfileStats::default(),
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

    /// 合駒事前証明のノード予算を設定する．
    ///
    /// AND ノードの合駒子ノードに対して，MID ループ前にクリーンな TT で
    /// df-pn 探索を実行し，証明/反証を試みる．
    /// 0 にすると事前証明を無効化する．デフォルトは 256．
    pub fn set_interpose_pre_solve_nodes(&mut self, v: u64) -> &mut Self {
        self.interpose_pre_solve_nodes = v;
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

    /// 転置表を参照する(位置キー＋持ち駒指定)．
    ///
    /// `remaining` は反証済みエントリの有効性判定に使用する．
    #[inline]
    fn look_up_pn_dn(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u64) {
        self.table.look_up(pos_key, hand, remaining)
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
        let (pn, dn, _source) = self.table.look_up(pk, hand, 0);
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
        self.path.clear();
        self.killer_table.clear();
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
        let pns_pv = self.pns_main(board);

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
            self.mid_fallback(board);
        }

        let (root_pn, root_dn) = self.look_up_board(board);

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
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let full_hash = board.hash;
        let pos_key = profile_timed!(self, position_key_ns, position_key_count,
            position_key(board));
        let att_hand = board.hand[self.attacker.index()];

        // ループ検出: フルハッシュで判定(持ち駒込みの完全一致)
        let in_path = profile_timed!(self, loop_detect_ns, loop_detect_count,
            self.path.contains(&full_hash));
        if in_path {
            return;
        }

        // 残り探索深さ
        let remaining = self.depth.saturating_sub(ply) as u16;

        // TT 参照: 既に閾値を超えている/証明済み/反証済みなら
        // 手生成をスキップして早期 return
        let (tt_pn, tt_dn, _) = profile_timed!(self, tt_lookup_ns, tt_lookup_count,
            self.look_up_pn_dn(pos_key, &att_hand, remaining));
        if tt_pn == 0 || tt_dn == 0 {
            return;
        }
        if tt_pn >= pn_threshold || tt_dn >= dn_threshold {
            return;
        }

        // 終端条件: 深さ制限・手数制限
        if ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            // 実際の持ち駒で不詰を記録する．
            // remaining = 0 で記録し，より深い探索での上書きを許可する．
            // PieceType::MAX_HAND_COUNT で保存すると，任意の持ち駒で不詰と扱われ，
            // 同じ局面が浅い ply で到達されたときも不詰として誤判定される．
            // att_hand を使うことで，持ち駒が異なる経路からの
            // 再到達時に TT ヒットせず，正しく再探索される．
            self.store(pos_key, att_hand, INF, 0, 0, pos_key);
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
        // AND ノードの遅延合駒展開 (lazy interposition expansion):
        // 合駒(drop)の子ノードは初期化時に deferred_children に分離し，
        // 非合駒応手(王移動・捕獲・駒移動合い)が全て証明された後に
        // children へ合流させる．これにより AND ノードの初期 pn(= sum)が
        // 大幅に低下し，親 OR ノードがこの分岐を深く探索しやすくなる．
        let mut deferred_children: ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        > = ArrayVec::new();
        // deferred が children に合流済みかどうかのフラグ
        let mut deferred_activated = or_node; // OR ノードでは常に true(遅延なし)
        // OR ノードの反証駒(init ループ中に蓄積)
        let mut init_or_disproof = PieceType::MAX_HAND_COUNT;
        // AND ノードの init フェーズ用: TT プレフィルタで証明済み合駒の証明駒蓄積
        let mut init_and_proof = [0u8; HAND_KINDS];
        let mut init_prefiltered_count: u32 = 0;
        // DFPN-E: OR ノードのエッジコスト計算用に守備側玉の位置を取得
        let defender_king_sq = if or_node {
            board.king_square(board.turn.opponent())
        } else {
            None
        };
        #[cfg(feature = "profile")]
        let _child_init_start = Instant::now();
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
                                    let dn = depth_biased_dn(n, ply + 1);
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
                                    let dn = depth_biased_dn(n, ply + 1).max(2);
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
                                let dn = depth_biased_dn(n, ply + 1);
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
                            let dn = depth_biased_dn(1, ply + 1);
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
                                    let dn = depth_biased_dn(nc, ply + 1);
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
                                    let dn = depth_biased_dn(nc, ply + 1).max(2);
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
                            let dn = depth_biased_dn(nc, ply + 1);
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
                return;
            }
            if !or_node && cdn_now == 0 {
                // AND 反証: 子の反証駒を親に伝播
                let child_dp = self
                    .table
                    .get_disproof_hand(child_pk, &child_hand);
                self.store(pos_key, child_dp, INF, 0,
                    REMAINING_INFINITE, pos_key);
                return;
            }
            // cdn_now == 0 ブロックに入るのは or_node == true のみ．
            // AND ノードは cdn_now == 0 のとき上で return 済み，
            // AND かつ cdn_now != 0 のときはここを通過して children に追加される．
            if cdn_now == 0 {
                // OR: この子は反証済み → 反証駒を蓄積
                let child_dp = self
                    .table
                    .get_disproof_hand(child_pk, &child_hand);
                let adj =
                    adjust_hand_for_move(*m, &child_dp);
                for k in 0..HAND_KINDS {
                    init_or_disproof[k] =
                        init_or_disproof[k].min(adj[k]);
                }
                continue;
            }

            // AND ノードの合駒(drop)は deferred_children に分離
            if !or_node && m.is_drop() {
                // --- TT ベース合駒プレフィルタ ---
                // 合駒を deferred_children に追加する前に，攻方の捕獲後局面を
                // メイン TT で参照する．捕獲後局面が証明済み(pn=0)なら，
                // 合駒の OR ノードは「捕獲して詰み」と確定できるため展開不要．
                // IDS の浅い反復で証明された深いレベルの合駒結果を活用し，
                // 合駒チェーンの指数的展開をボトムアップに折り畳む．
                if self.try_prefilter_block(
                    board, *m, &child_hand, remaining,
                    &mut init_and_proof,
                ) {
                    init_prefiltered_count += 1;
                    self.prefilter_hits += 1;
                } else {
                    push_move(&mut deferred_children, (
                        *m,
                        child_full_hash,
                        child_pk,
                        child_hand,
                    ));
                }
            } else {
                push_move(&mut children, (
                    *m,
                    child_full_hash,
                    child_pk,
                    child_hand,
                ));
            }
        }

        // OR ノードで全子が反証済み(children が空)
        if or_node && children.is_empty() {
            self.store(
                pos_key, init_or_disproof, INF, 0,
                remaining, pos_key,
            );
            return;
        }

        #[cfg(feature = "profile")]
        {
            self.profile_stats.child_init_ns += _child_init_start.elapsed().as_nanos() as u64;
            self.profile_stats.child_init_count += 1;
        }

        // パスに追加(フルハッシュ)
        self.path.insert(full_hash);

        // --- 合駒事前証明 (interpose pre-solve) ---
        // AND ノードの合駒(drop)子ノードに対して，クリーンな TT で
        // df-pn の mid() を予算付きで実行し，MID ループ前に証明/反証を確定する．
        //
        // 設計思想 (Lambda df-pn の simplest-first 原則):
        // 1. 一時 TT を使って1つ目の合駒を証明 → TT に証明パスが蓄積
        // 2. 2つ目以降の合駒は持ち駒優越で TT ヒット → 事実上ゼロコストで証明
        // 3. 証明済み子は children から除外 → MID ループの負荷が激減
        //
        // 既存 TT はノイズになるためリセットし，Pre-Solve 間で TT を共有する．
        // 終了後に確定エントリ(証明・反証)のみ本体 TT にマージする．
        //
        // `#[inline(never)]` の別関数に抽出し，mid() のスタックフレーム膨張を防ぐ．
        if !or_node && !deferred_children.is_empty()
            && self.interpose_pre_solve_nodes > 0 && remaining >= 3
            && !self.in_pre_solve
        {
            match self.interpose_pre_solve(
                board, &mut deferred_children, pos_key, &att_hand,
                ply, remaining,
            ) {
                PreSolveResult::Disproved(dp) => {
                    self.store(pos_key, dp, INF, 0, REMAINING_INFINITE, pos_key);
                    self.path.remove(&full_hash);
                    return;
                }
                PreSolveResult::AllProved(mut proof) if children.is_empty() => {
                    // プレフィルタで証明済みの合駒と pre-solve の結果を統合
                    for k in 0..HAND_KINDS {
                        proof[k] = proof[k].max(init_and_proof[k]);
                        proof[k] = proof[k].min(att_hand[k]);
                    }
                    self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                    self.path.remove(&full_hash);
                    return;
                }
                _ => {}
            }
        }

        // プレフィルタで全合駒が証明済み(deferred_children が空になった場合)
        if !or_node && deferred_children.is_empty() && init_prefiltered_count > 0
            && children.is_empty()
        {
            let mut p = init_and_proof;
            for k in 0..HAND_KINDS {
                p[k] = p[k].min(att_hand[k]);
            }
            self.store(pos_key, p, 0, INF, REMAINING_INFINITE, pos_key);
            self.path.remove(&full_hash);
            return;
        }

        // AND ノードで遅延合駒の活性化判定:
        // (a) 非合駒の children が空 → 遅延をキャンセルし即座に合流
        // (b) deferred_children が少数(< 8) → 効果が薄いので即座に合流
        //     少数の合駒を遅延させると，不詰み局面で合駒が鍵となる応手の
        //     発見が遅れ，かえって探索効率が低下する．
        if !or_node && !deferred_children.is_empty()
            && (children.is_empty() || (deferred_children.len() as u32) < LAZY_INTERPOSE_THRESHOLD)
        {
            children.extend(deferred_children.drain(..));
            deferred_activated = true;
        }

        // --- 単一子最適化 ---
        // 子が1つしかない場合，MID ループ(閾値計算・全子走査)をバイパスし，
        // 親の閾値をそのまま渡して直接再帰する．
        // OR ノードでは王手が1手のみ，AND ノードでは合法応手が1手のみの
        // ケースが詰将棋で頻出する．
        // AND ノードで deferred_children がある場合は単一子最適化を
        // 使用できない(証明後に deferred を活性化する必要がある)．
        if children.len() == 1 && deferred_children.is_empty() {
            let (m, child_fh, child_pk, ref child_hand) = children[0];
            loop {
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
                            let child_dp = self.table.get_disproof_hand(child_pk, child_hand);
                            let adj = adjust_hand_for_move(m, &child_dp);
                            let mut dp = init_or_disproof;
                            for k in 0..HAND_KINDS {
                                dp[k] = dp[k].min(adj[k]);
                            }
                            // GHI 対策: ループ子由来の反証は経路依存
                            // 子の反証が経路依存なら親の反証も経路依存
                            let child_path_dep = is_loop_child
                                || self.table.has_path_dependent_disproof(
                                    child_pk, child_hand,
                                );
                            if child_path_dep {
                                self.store_path_dep(
                                    pos_key, dp, INF, 0,
                                    remaining, pos_key, true,
                                );
                            } else {
                                self.store(pos_key, dp, INF, 0, remaining, pos_key);
                            }
                        }
                    } else {
                        if cdn == 0 {
                            let child_dp = self.table.get_disproof_hand(child_pk, child_hand);
                            // GHI 対策: ループ子による反証は経路依存
                            if is_loop_child {
                                self.store_path_dep(
                                    pos_key, child_dp, INF, 0,
                                    remaining, 0, true,
                                );
                            } else {
                                self.store(pos_key, child_dp, INF, 0, REMAINING_INFINITE, pos_key);
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
            return;
        }

        // SNDA 用の (source, value) ペアバッファ(ループ外で確保し再利用)
        let mut snda_pairs: Vec<(u64, u32)> = Vec::new();

        // MID ループ(証明駒/反証駒の伝播を含む)
        loop {
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

            if or_node {
                // OR ノード: min(pn), sum(dn)
                current_pn = INF;
                current_dn = 0;
                second_best = INF; // 2番目に小さい pn
                // 反証駒の交差(全子の反証駒の min)
                // init フェーズで反証済みの子から蓄積した init_or_disproof を引き継ぐ
                let mut or_disproof = init_or_disproof;
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

                    // 反証済みの子: 反証駒を蓄積
                    // adjust_hand_for_move は証明駒/反証駒共通の駒入出補正関数
                    if cdn == 0 {
                        let child_dp = self
                            .table
                            .get_disproof_hand(
                                child_pk, child_hand,
                            );
                        let adj = adjust_hand_for_move(
                            children[i].0,
                            &child_dp,
                        );
                        for k in 0..HAND_KINDS {
                            or_disproof[k] =
                                or_disproof[k].min(adj[k]);
                        }
                        // GHI 伝播: 子の反証が経路依存なら親も経路依存
                        if !self.path.contains(&child_fh)
                            && self.table.has_path_dependent_disproof(
                                child_pk, child_hand,
                            )
                        {
                            loop_child_count += 1; // path_dependent として扱う
                        }
                    }

                    if cpn < current_pn
                        || (cpn == current_pn
                            && cdn < best_pn_dn.1)
                    {
                        second_best = current_pn;
                        current_pn = cpn;
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
                    // GHI 対策: ループ子が寄与した反証は経路依存．
                    // ループ子は現在の探索パスに依存して dn=0 と見做されるため，
                    // 異なるパスの IDS 反復では無効になりうる．
                    if loop_child_count > 0 {
                        self.store_path_dep(
                            pos_key, or_disproof,
                            INF, 0,
                            remaining, pos_key, true,
                        );
                    } else {
                        self.store(
                            pos_key, or_disproof,
                            INF, 0,
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
                        // 反証駒 = 子の反証駒
                        let child_dp = self
                            .table
                            .get_disproof_hand(
                                child_pk, child_hand,
                            );
                        // GHI 対策: ループ子による反証は経路依存(path-dependent)．
                        // REMAINING_INFINITE ではなく有限 remaining で保存し，
                        // 異なる経路のより深い探索で再評価可能にする．
                        if is_loop_child {
                            self.store_path_dep(
                                pos_key, child_dp, INF, 0,
                                remaining, csrc, true,
                            );
                        } else {
                            self.store(
                                pos_key, child_dp, INF, 0,
                                REMAINING_INFINITE, csrc,
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
                    // TT 保存用: 真の min(dn)
                    if cdn < current_dn {
                        current_dn = cdn;
                    }
                    // SNDA ペア収集(source=0 は独立ノード)
                    if csrc != 0 {
                        snda_pairs.push((csrc, cpn));
                    }
                    // 子ノード選択用: 合駒(drop)にバイアスを加算
                    let effective_cdn = if m.is_drop() {
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
                    self.path.remove(&full_hash);
                    return;
                }

                // WPN: current_pn = max(cpn) + (unproven_count - 1)
                // unproven_count == 0 の場合は全子証明済み(current_pn = 0)
                if unproven_count > 0 {
                    current_pn = (max_cpn as u64)
                        .saturating_add(unproven_count as u64 - 1)
                        .min(INF as u64) as u32;
                }

                // SNDA 補正: 同一 source の子は DAG 合流の可能性
                // 重複グループの最小値分を控除して過大評価を補正
                if snda_pairs.len() >= 2 {
                    current_pn = snda_dedup(&mut snda_pairs, current_pn);
                }

                // 全子が証明済み
                if all_proved && current_pn == 0 {
                    if !deferred_activated && !deferred_children.is_empty() {
                        // 遅延合駒を活性化: children に合流させて MID ループ続行
                        children.extend(deferred_children.drain(..));
                        deferred_activated = true;
                        continue;
                    }
                    // AND ノード証明(deferred 含め全子が証明済み)
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
            let best_move16 = children[best_idx].0.to_move16();
            profile_timed!(self, tt_store_ns, tt_store_count,
                self.store_with_best_move(pos_key, att_hand, current_pn, current_dn, remaining, best_source, best_move16));

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
                let child_dn_th = eff_dn_th
                    .saturating_sub(current_dn)
                    .saturating_add(best_pn_dn.1)
                    .min(INF - 1);
                let epsilon = second_best / 4 + 1;
                let child_pn_th = eff_pn_th
                    .min(second_best.saturating_add(epsilon))
                    .min(INF - 1);
                (child_pn_th, child_dn_th)
            } else {
                let child_pn_th = eff_pn_th
                    .saturating_sub(current_pn)
                    .saturating_add(best_pn_dn.0)
                    .min(INF - 1);
                let epsilon = second_best / 4 + 1;
                let child_dn_th = eff_dn_th
                    .min(second_best.saturating_add(epsilon))
                    .min(INF - 1);
                (child_pn_th, child_dn_th)
            };

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
                let checks = self.generate_check_moves(board);
                if self.try_capture_tt_proof(
                    board, &checks,
                    remaining.saturating_sub(1),
                ) {
                    // 証明済み → MID 再帰をスキップ
                    profile_timed!(self, undo_move_ns, undo_move_count,
                        board.undo_move(m, captured));
                    continue;
                }
            }

            self.mid(
                board,
                child_pn_th,
                child_dn_th,
                ply + 1,
                !or_node,
            );
            profile_timed!(self, undo_move_ns, undo_move_count,
                board.undo_move(m, captured));
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
                // 取り後の局面が証明済み → この OR ノードは証明済み
                // 証明駒は取り後の証明駒を調整して使用
                let cap_proof = self.table.get_proof_hand(cap_pk, &cap_hand);
                let proof = adjust_hand_for_move(*check, &cap_proof);
                self.store_board_with_hand(board, &proof, 0, INF, REMAINING_INFINITE, cap_pk);
                board.undo_move(*check, captured);
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

    /// 合駒(drop)子ノードの事前証明を試みる．
    ///
    /// クリーンな一時 TT を使い，各合駒子に対してノード予算付きの
    /// `mid()` を実行する．1つ目の合駒で蓄積された証明エントリが
    /// 2つ目以降の合駒に TT ヒットで伝播するため，多くの合駒を
    /// 低コストで証明できる．
    ///
    /// 既存 TT はノイズになるため，Pre-Solve 開始時にリセットし，
    /// 完了後に確定エントリ(証明・反証)のみ本体 TT にマージする．
    ///
    /// # 戻り値
    ///
    /// - `Disproved(dp)`: 合駒子の1つが反証 → AND ノード全体が反証
    /// - `AllProved(proof)`: 全合駒子が証明済み(証明駒の和集合)
    /// - `Partial`: 一部未確定(deferred_children から証明済みを除去済み)
    ///
    /// `#[inline(never)]` により，mid() のスタックフレームに
    /// Pre-Solve のローカル変数が含まれるのを防ぐ．
    #[inline(never)]
    fn interpose_pre_solve(
        &mut self,
        board: &mut Board,
        deferred_children: &mut ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        _pos_key: u64,
        _att_hand: &[u8; HAND_KINDS],
        ply: u32,
        remaining: u16,
    ) -> PreSolveResult {
        // 一時 TT にスワップ
        let main_tt = std::mem::replace(
            &mut self.table,
            TranspositionTable::new(),
        );
        let saved_max_nodes = self.max_nodes;

        let saved_depth = self.depth;

        let mut and_proof = [0u8; HAND_KINDS];
        let mut pre_solved_indices: ArrayVec<usize, MAX_MOVES> = ArrayVec::new();

        for (i, &(ref m, _child_fh, child_pk, ref child_hand))
            in deferred_children.iter().enumerate()
        {
            let child_remaining = remaining.saturating_sub(1);

            // TT で既に証明/反証済みなら即処理
            let (cpn, cdn, _csrc) = self.look_up_pn_dn(
                child_pk, child_hand, child_remaining,
            );
            if cpn == 0 {
                // 前の合駒の Pre-Solve で TT ヒット → 証明済み
                let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                let adj = adjust_hand_for_move(*m, &child_ph);
                for k in 0..HAND_KINDS {
                    and_proof[k] = and_proof[k].max(adj[k]);
                }
                let _ = pre_solved_indices.try_push(i);
                continue;
            }
            if cdn == 0 {
                // 反証: AND ノード全体が反証
                let pre_solve_tt = std::mem::replace(
                    &mut self.table, main_tt,
                );
                self.table.merge_proven(&pre_solve_tt);
                self.depth = saved_depth;
                self.max_nodes = saved_max_nodes;
                let child_dp = pre_solve_tt.get_disproof_hand(child_pk, child_hand);
                return PreSolveResult::Disproved(child_dp);
            }

            // mid() をノード予算付きで実行
            let captured = board.do_move(*m);
            self.max_nodes = self.nodes_searched
                .saturating_add(self.interpose_pre_solve_nodes);
            self.in_pre_solve = true;
            self.mid(board, INF - 1, INF - 1, ply + 1, true);
            self.in_pre_solve = false;
            board.undo_move(*m, captured);
            self.max_nodes = saved_max_nodes;

            // 結果確認
            let (cpn, cdn, _csrc) = self.look_up_pn_dn(
                child_pk, child_hand, child_remaining,
            );
            if cpn == 0 {
                let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                let adj = adjust_hand_for_move(*m, &child_ph);
                for k in 0..HAND_KINDS {
                    and_proof[k] = and_proof[k].max(adj[k]);
                }
                let _ = pre_solved_indices.try_push(i);

                // --- 同一マス合駒の捕獲後 TT 転用 ---
                // 合駒 i が証明済みの場合，同一マスの他の合駒について
                // 捕獲後の共通局面の TT エントリで証明を転用する．
                self.cross_deduce_same_square_proofs(
                    board, deferred_children, i, remaining,
                    &mut and_proof, &mut pre_solved_indices,
                );
            } else if cdn == 0 {
                // 反証: AND ノード全体が反証
                let pre_solve_tt = std::mem::replace(
                    &mut self.table, main_tt,
                );
                self.table.merge_proven(&pre_solve_tt);
                self.depth = saved_depth;
                self.max_nodes = saved_max_nodes;
                let child_dp = pre_solve_tt.get_disproof_hand(child_pk, child_hand);
                return PreSolveResult::Disproved(child_dp);
            } else if pre_solved_indices.is_empty() {
                // 最初の合駒すら証明できない → 不詰の可能性が高い．
                // これ以上の Pre-Solve はノードの浪費なので打ち切る．
                break;
            }
        }

        // TT を復元し，確定エントリをマージ
        let pre_solve_tt = std::mem::replace(
            &mut self.table, main_tt,
        );
        self.table.merge_proven(&pre_solve_tt);
        self.depth = saved_depth;
        self.max_nodes = saved_max_nodes;

        // 証明済み子を deferred_children から除去(逆順で安全に削除)
        for &i in pre_solved_indices.iter().rev() {
            deferred_children.remove(i);
        }

        if deferred_children.is_empty() {
            PreSolveResult::AllProved(and_proof)
        } else {
            PreSolveResult::Partial
        }
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
    #[inline(never)]
    fn cross_deduce_same_square_proofs(
        &mut self,
        board: &mut Board,
        deferred_children: &ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        solved_idx: usize,
        remaining: u16,
        and_proof: &mut [u8; HAND_KINDS],
        pre_solved_indices: &mut ArrayVec<usize, MAX_MOVES>,
    ) {
        let (ref solved_move, _, _, _) = deferred_children[solved_idx];
        let target_sq = solved_move.to_sq();

        // 同一マスに未解決の合駒がなければスキップ
        let has_siblings = deferred_children.iter().enumerate().any(|(j, (mj, _, _, _))| {
            j > solved_idx && mj.to_sq() == target_sq && !pre_solved_indices.contains(&j)
        });
        if !has_siblings {
            return;
        }

        // 合駒を実行し，攻方の捕獲手を探索
        let captured_by_block = board.do_move(*solved_move);
        let legal = movegen::generate_legal_moves(board);

        // 捕獲手(ターゲットマスへの駒取り)を全て試行
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

            let solved_pt = solved_move.drop_piece_type().unwrap();
            let solved_hi = solved_pt.hand_index().unwrap();

            // 各未解決の同一マス合駒について TT 参照
            for (j, &(ref mj, _, child_pk_j, ref child_hand_j))
                in deferred_children.iter().enumerate()
            {
                if j <= solved_idx || mj.to_sq() != target_sq {
                    continue;
                }
                if pre_solved_indices.contains(&j) {
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

                    // 子 TT に証明エントリを格納(次回ループの TT チェックで使用)
                    self.table.store(
                        child_pk_j, or_ph, 0, INF,
                        remaining.saturating_sub(1), child_pk_j,
                    );

                    // AND 証明駒の更新
                    let adj = adjust_hand_for_move(*mj, &or_ph);
                    for k in 0..HAND_KINDS {
                        and_proof[k] = and_proof[k].max(adj[k]);
                    }
                    let _ = pre_solved_indices.try_push(j);
                }
            }
        }

        board.undo_move(*solved_move, captured_by_block);
    }
}

/// 合駒事前証明の結果．
enum PreSolveResult {
    /// 合駒子の1つが反証 → AND ノード全体が反証．反証駒を含む．
    Disproved([u8; HAND_KINDS]),
    /// 全合駒子が証明済み．証明駒の和集合(max)を含む．
    AllProved([u8; HAND_KINDS]),
    /// 一部未確定(証明済みの子は deferred_children から除去済み)．
    Partial,
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
        &self,
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
        &self,
        board: &mut Board,
        early_exit: bool,
    ) -> ArrayVec<Move, MAX_MOVES> {
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
            // 玉の隣接マスかつ攻め方の他駒が利いていない → 玉が取り返せる
            if king_step.contains(sq)
                && !board.is_attacked_by_excluding(sq, attacker, false, Some(checker_sq))
            {
                continue;
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
                // 通常マス: 全駒種を逆順(強い駒から)生成．
                // 強い駒の合駒を先に証明すると，攻め方が多くの持ち駒を
                // 獲得した局面が TT に蓄積され，弱い駒の合駒探索時に
                // TT ヒットで高速化される可能性がある．
                for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate().rev() {
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
    fn extract_pv_limited(&self, board: &mut Board, max_visits: u64) -> Vec<Move> {
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
        &self,
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
        &self,
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
        let saved_depth = self.depth;
        let mut ids_depth: u32 = 2;
        let total_max_nodes = self.max_nodes;
        loop {
            if ids_depth > saved_depth {
                ids_depth = saved_depth;
            }
            self.depth = ids_depth;
            self.path.clear();
            let remaining = ids_depth as u16;
            let (root_pn, _, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            if root_pn == 0 {
                break;
            }
            if ids_depth < saved_depth {
                let budget = (total_max_nodes / 16).max(1024);
                self.max_nodes = self.nodes_searched.saturating_add(budget);
            } else {
                self.max_nodes = total_max_nodes;
            }
            {
                let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if root_pn != 0 && root_dn != 0
                    && self.nodes_searched < self.max_nodes
                    && !self.timed_out
                {
                    self.mid(board, INF - 1, INF - 1, 0, true);
                }
            }
            let (root_pn2, _, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            if root_pn2 == 0 {
                break;
            }
            if self.nodes_searched >= total_max_nodes || self.timed_out {
                break;
            }
            if ids_depth >= saved_depth {
                break;
            }
            self.table.retain_proofs();
            ids_depth = ids_depth.saturating_mul(2).max(ids_depth + 2);
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
        });

        // 再利用バッファ(ループ内のアロケーション回避)
        let max_path = self.depth as usize + 2;
        let mut path: Vec<u32> = Vec::with_capacity(max_path);
        let mut captures: Vec<Piece> = Vec::with_capacity(max_path);
        let mut ancestors: FxHashSet<u64> =
            FxHashSet::with_capacity_and_hasher(max_path, Default::default());

        // PNS メインループ
        loop {
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
            // 定期タイムアウトチェック
            if self.nodes_searched & 0x3FF == 0 && self.is_timed_out() {
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

            while arena[current as usize].expanded {
                let node = &arena[current as usize];
                // OR: min(pn) の子を選択，AND: min(dn) の子を選択
                let best_child = if node.or_node {
                    *node.children.iter()
                        .min_by_key(|&&c| (arena[c as usize].pn, arena[c as usize].dn))
                        .unwrap()
                } else {
                    *node.children.iter()
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

            // リーフ展開
            let ply = (path.len() - 1) as u32;
            self.pns_expand(board, &mut arena, current, ply, &ancestors);

            // 盤面をルートに戻す
            for i in (1..path.len()).rev() {
                let child_move = arena[path[i] as usize].move_from_parent;
                board.undo_move(child_move, captures[i - 1]);
            }

            // バックアップ: 展開ノードからルートまで pn/dn を更新
            Self::pns_backup(&mut arena, current);
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
            self.store(pos_key, att_hand, INF, 0, 0, pos_key);
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
                        cdn = depth_biased_dn(nc, ply + 1);
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
                        cdn = depth_biased_dn(n, ply + 1);
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
            if !or_node && cdn == 0 {
                let child_dp = self.table.get_disproof_hand(child_pk, &child_hand);
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                arena[node_idx as usize].expanded = true;
                self.store(pos_key, child_dp, INF, 0, REMAINING_INFINITE, pos_key);
                return;
            }
            // OR ノードで子が反証済み → 子を追加せずスキップ
            if or_node && cdn == 0 {
                continue;
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
            });
            arena[node_idx as usize].children.push(child_idx);
        }

        // OR ノードで全子が反証済み(children 空)
        if or_node && arena[node_idx as usize].children.is_empty() {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            self.store(pos_key, att_hand, INF, 0, remaining, pos_key);
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
                } else if unproven == 0 {
                    (0u32, INF)
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
        &self,
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

    /// PNS ツリーの結果を TT に格納する．
    ///
    /// 証明/反証済みの中間ノードを TT に書き込み，
    /// 既存の `extract_pv()` による PV 抽出を可能にする．
    /// OR 証明ノードには TT Best Move も記録する．
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
                // 反証済みノード
                self.store(
                    node.pos_key, node.hand, INF, 0,
                    REMAINING_INFINITE, node.pos_key,
                );
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

        let solver = DfPnSolver::default_solver();
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

        let solver = DfPnSolver::default_solver();

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
    #[ignore] // 50M ノード / 300 秒タイムアウトの重いテスト
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
                // 先頭25手は固定
                let expected_prefix = [
                    "8f8g+", "7h8g", "S*7i", "8h9g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                assert_eq!(
                    &pv[..25], &expected_prefix,
                    "PV prefix mismatch (first 25 moves):\n  got:      {}\n  expected: {}",
                    pv[..25].join(" "),
                    expected_prefix.join(" "),
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
                // 3四玉(銀取り開き王手) → 31玉 → 42銀打 → 32玉 → 33飛成
                let expected = ["2d3d", "2b3a", "S*4b", "3a3b", "4c3c+"];
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {}\n  expected: {}",
                    usi_moves.join(" "),
                    expected.join(" "),
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
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

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
        let solver = DfPnSolver::default_solver();
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
            DfPnSolver::with_timeout(41, 10_000_000, 32767, 30);
        solver.set_find_shortest(false);
        let start = Instant::now();
        let result = solver.solve(&mut board);
        let elapsed = start.elapsed();
        eprintln!("39te: {} nodes, {:.1}s, max_ply={}, prefilter_hits={}",
            solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply,
            solver.prefilter_hits);

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
        let sfen = "9/3+N1P3/2+R3p2/8k/8N/5+B3/4S4/1R1p5/9 b NPb4g3sn4l14p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

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
        let solver = DfPnSolver::default_solver();
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
        let sfen = "9/3+N1P3/2+R3p2/1Rp5k/8N/5+B3/4S4/3p5/9 b NPb4g3sn4l13p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

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
        let solver = DfPnSolver::default_solver();

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

        let solver = DfPnSolver::default_solver();
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
    /// Pre-Solve に必要なノード数を計測する．
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
        let solver_tmp = DfPnSolver::default_solver();
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
            child_solver.set_interpose_pre_solve_nodes(0); // Pre-Solve 無効
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
            solver.set_interpose_pre_solve_nodes(0); // Pre-Solve 無効
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

        let solver = DfPnSolver::default_solver();
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
            child_solver.set_interpose_pre_solve_nodes(0);
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

        let solver = DfPnSolver::default_solver();
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

        let solver = DfPnSolver::default_solver();
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

}
