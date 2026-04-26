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

use crate::board::Board;
use crate::moves::Move;
use crate::types::{Piece, Square, HAND_KINDS};

/// `eprintln!` の `verbose` feature ガード版．
///
/// `verbose` feature が無効の場合はコンパイル時に完全に除去される．
/// デバッグ・分析用の進捗表示やノード情報出力に使用する．
macro_rules! verbose_eprintln {
    ($($arg:tt)*) => {
        #[cfg(feature = "verbose")]
        eprintln!($($arg)*)
    };
}

// profile_timed! マクロは profile.rs で定義．
// Rust のマクロは定義順で解決されるため，profile モジュールを先に宣言する．
#[macro_use]
mod profile;
mod entry;
mod tt;
mod solver;
mod pns;
#[cfg(test)]
mod tests;

pub use solver::{DfPnSolver, TsumeResult};
pub use pns::{solve_tsume, solve_tsume_with_timeout, solve_tsume_and_collect_pn_dn_dist};

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

/// pn/dn の 1 単位を表す定数．
///
/// 全ての pn/dn 初期値・加算定数・フロア値はこの定数の倍数で表現する．
/// PN_UNIT=1 が従来動作と等価であり，PN_UNIT を拡大することで
/// 1+ε 閾値の余裕を確保し閾値飢餓(§10.2)を緩和できる．
///
/// 完全なスケーリング要件: pn/dn の「量」を表す全ての定数に PN_UNIT を
/// 適用する必要がある．スケーリング対象は以下の通り:
/// - 初期値: TT ミス時の pn=1/dn=1，heuristic_or_pn/heuristic_and_pn 返り値
/// - 加算値: edge_cost_or/and，sacrifice_check_boost，epsilon の +1，
///   progress_floor の +1，TCA の +1
/// - フロア/バイアス: DN_FLOOR，INTERPOSE_DN_BIAS，.max(N) のリテラル
///
/// スケーリング不要: 終端値(INF, 0)，相対比率(/4, /2, *2/3)，
/// 盤面状態の比較(safe_escapes >= 4 等)，ループカウンタ．
const PN_UNIT: u32 = 16;

/// WPN スケールドサム: AND pn の非最大子寄与率 (1/2^WPN_GAMMA_SHIFT)．
///
///   AND pn = max(child_pn) + (sum(child_pn) - max(child_pn)) >> WPN_GAMMA_SHIFT
///
/// 旧式 `sum` は DAG で過大評価される．`max + (count-1)*PN_UNIT` は非最大子の
/// 変化を伝播しない．スケールドサムは実際の子 pn 値を使いつつ DAG 二重カウントを
/// SNDA と協調して割り引く中間的近似．
/// OR dn は `sum + SNDA` が適切な近似のため WPN 不要 (disproof 探索に逆効果)．
///
/// crossover 点: max = PN_UNIT * 2^WPN_GAMMA_SHIFT
///   (これより小さい max では旧式より保守的，大きい max では旧式より積極的)
/// GAMMA_SHIFT=6 → crossover = 1024 (bucket 10)
const WPN_GAMMA_SHIFT: u32 = 6;

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
/// 新値: 8 は king move の初期 dn(1)より十分大きく，
/// king move が探索されて dn が上昇した後に drop の探索が始まる程度のバイアス．
/// これにより df-pn の自然な閾値制御で king move → drop の順序が実現される．
const INTERPOSE_DN_BIAS: u32 = 8 * PN_UNIT;

// MID ループの dn 閾値フロア(スラッシング防止)は v0.24.41 で
// `DfPnSolver::param_dn_floor_mult` (デフォルト 100) に移行した．
// 旧 `const DN_FLOOR: u32 = 100 * PN_UNIT;` は削除．
// 子ノードの dn が小さすぎると MID ループが閾値超過で即座に返り，
// 進捗のない空転が発生するため，dn_threshold を最低
// `param_dn_floor_mult * PN_UNIT` まで引き上げる．

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

/// Deep df-pn (Song Zhang et al. 2017) の深さ係数 R．
///
/// 深い位置ほど初期 pn を高く設定し，浅い分岐の探索を優先させる．
/// `look_up_pn_dn` で TT ミス時に `ply > depth/2` の場合に適用される．
const DEEP_DFPN_R: u32 = 4;

/// MID ループのゼロ進捗検出閾値．
///
/// 子 `mid()` が消費するノード数が連続して0の回数がこの値を超えると
/// MID ループを脱出し，上位ノードに制御を戻す(`dn_floor` 由来の空転防止)．
const ZERO_PROGRESS_LIMIT: u32 = 16;

/// MID ループの停滞検出閾値．
///
/// best child の pn/dn と閾値が連続して変化しない回数がこの値を超えると
/// MID ループを脱出する．同じ子に同じ予算で `mid()` を呼んでも
/// 結果が変わらないケースを検出する．
const STAGNATION_LIMIT: u32 = 4;

/// `param_epsilon_denom` の depth-adaptive モードを示すセンチネル値．
///
/// この値が設定されている場合，epsilon 除数は `saved_depth_for_epsilon` に基づいて
/// 自動決定される(depth ≥ 19 → 2，それ以外 → 3)．
const EPSILON_DENOM_ADAPTIVE: u32 = 0;

/// `param_disproof_remaining_threshold` の depth-adaptive モードを示すセンチネル値 (v0.25.1)．
///
/// この値が設定されている場合，depth-limited disproof 格納閾値は
/// `outer_solve_depth` に基づいて自動決定される．
/// 詳細は `DfPnSolver::effective_disproof_remaining_threshold` 参照．
pub(super) const DISPROOF_THRESHOLD_ADAPTIVE: u16 = u16::MAX;

/// 深さ制限なし(真の証明/反証)を示す定数．
///
/// remaining_flags のビット 0-14 に格納されるため 15 ビットの最大値(0x7FFF)．
/// 実用上の depth は 31〜127 であり 32767 は十分に大きい．
const REMAINING_INFINITE: u16 = 0x7FFF;

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
        let propagated = child_min_remaining.saturating_add(1);
        propagated.min(current_remaining)
    }
}


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
    // 静かな王手: 玉との距離 + 駒打ちペナルティでコスト決定
    //
    // v0.24.65: 駒打ちの静かな王手は駒移動王手より一般に弱い
    // (攻め駒の配置を改善せず持ち駒を消費するだけ)．
    // 追加コスト PN_UNIT/2 で駒移動王手を優先させる．
    // NPS への影響: ゼロ (Move の属性判定のみ)
    let to = m.to_sq();
    let dc = (to.col() as i8 - king_sq.col() as i8).unsigned_abs();
    let dr = (to.row() as i8 - king_sq.row() as i8).unsigned_abs();
    let dist = dc.max(dr);
    let base = if dist <= 2 { PN_UNIT } else { 2 * PN_UNIT };
    let drop_penalty = if m.is_drop() { PN_UNIT / 2 } else { 0 };
    base + drop_penalty
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
        return 2 * PN_UNIT;
    }
    // 玉の逃げ・駒移動合い
    PN_UNIT
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
    2 * PN_UNIT
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
fn snda_dedup(pairs: &mut [(u32, u32)], raw_sum: u32) -> u32 {
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
    (raw_sum as u64).saturating_sub(deduction).max(PN_UNIT as u64) as u32
}

/// 持ち駒の要素ごと比較: a の全要素が b 以上なら true．
///
/// 証明駒の優越判定に使用: 持ち駒が多い方が有利(攻め方)．
///
/// SWAR (SIMD Within A Register): 7 バイトを u64 にパックし分岐なしで一括比較する．
/// 各バイトに MSB(0x80) をセットして引き算し，MSB が全て残れば全要素 a[i] >= b[i]．
/// 持ち駒値は 0-18 の範囲なので各バイトの MSB は常に 0 であり，
/// (a[i] + 128) - b[i] >= 110 となるためバイト間の桁借りは発生しない．
// SWAR パッキングは HAND_KINDS == 7 を前提とする(u64 の 7 バイトに収める)．
const _: () = assert!(HAND_KINDS == 7, "hand_gte SWAR assumes HAND_KINDS == 7");

// SWAR 比較はエンディアン非依存(両オペランドが同一バイト順序でパックされるため
// バイト単位の加減算とマスク演算は正しく動作する)が，明示性のため LE を使用する．
#[cfg(not(target_endian = "little"))]
compile_error!("hand_gte SWAR is tested only on little-endian targets");

#[inline(always)]
fn hand_gte(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
    let a_packed = u64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], 0]);
    let b_packed = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0]);
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
pub(super) fn hand_gte_forward_chain(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
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

// --- 王手生成キャッシュ (E2 最適化) ---

/// 王手生成キャッシュのサイズ(2^13 = 8192 エントリ，direct-mapped)．
const CHECK_CACHE_SIZE: usize = 8192;

/// 1エントリあたりのキャッシュ容量(典型的な王手数は 3-15)．
const CHECK_CACHE_CAPACITY: usize = 32;

/// 王手生成キャッシュの1エントリ．
struct CheckCacheEntry {
    hash: u64,
    moves: ArrayVec<Move, CHECK_CACHE_CAPACITY>,
}

impl Default for CheckCacheEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            moves: ArrayVec::new(),
        }
    }
}

/// 局面ハッシュをキーとする王手リストのキャッシュ．
///
/// `generate_check_moves` の結果を direct-mapped テーブルに保存し，
/// 同一局面への再計算を回避する．MID ループで同一局面が繰り返し
/// 出現するため，キャッシュヒット率が高い．
///
/// 内部可変性(UnsafeCell)を使用して `&self` でアクセス可能にする．
/// これにより `generate_check_moves_cached` を `&self` で呼び出せ，
/// mid() のスタックフレーム最適化を阻害しない．
pub(super) struct CheckCache {
    table: std::cell::UnsafeCell<Vec<CheckCacheEntry>>,
}

impl CheckCache {
    pub(super) fn new() -> Self {
        let mut table = Vec::with_capacity(CHECK_CACHE_SIZE);
        for _ in 0..CHECK_CACHE_SIZE {
            table.push(CheckCacheEntry::default());
        }
        Self { table: std::cell::UnsafeCell::new(table) }
    }

    /// キャッシュをクリアする．
    pub(super) fn clear(&self) {
        let table = unsafe { &mut *self.table.get() };
        for entry in table.iter_mut() {
            entry.hash = 0;
            entry.moves.clear();
        }
    }

    /// キャッシュから王手リストを取得し，ヒットした場合はコピーを返す．
    #[inline(always)]
    pub(super) fn get(&self, hash: u64) -> Option<ArrayVec<Move, CHECK_CACHE_CAPACITY>> {
        let table = unsafe { &*self.table.get() };
        let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
        let entry = &table[idx];
        if entry.hash == hash {
            Some(entry.moves.clone())
        } else {
            None
        }
    }

    /// 王手リストをキャッシュに格納する．
    #[inline(always)]
    pub(super) fn insert(&self, hash: u64, moves: &ArrayVec<Move, MAX_MOVES>) {
        if moves.len() <= CHECK_CACHE_CAPACITY {
            let table = unsafe { &mut *self.table.get() };
            let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
            let entry = &mut table[idx];
            entry.hash = hash;
            entry.moves.clear();
            for &m in moves.iter() {
                entry.moves.push(m);
            }
        }
    }
}

