//! DfPnSolver 構造体と探索コアロジック．

use arrayvec::ArrayVec;
#[cfg(feature = "visit_diag")]
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::time::{Duration, Instant};

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square, HAND_KINDS};

use super::entry::{PnsNode, PNS_MAX_ARENA_NODES};
use super::tt::TranspositionTable;
#[cfg(feature = "profile")]
use super::profile::ProfileStats;
use super::{
    adjust_hand_for_move, edge_cost_and, edge_cost_or, heuristic_dn_from_pn, heuristic_or_dn,
    hand_gte_forward_chain,
    position_key, propagate_nm_remaining, push_move, snda_dedup,
    CheckCache,
    DEEP_DFPN_R, DISPROOF_THRESHOLD_ADAPTIVE, EPSILON_DENOM_ADAPTIVE, INF, INTERPOSE_DN_BIAS,
    MAX_MOVES, PN_UNIT, REMAINING_INFINITE, STAGNATION_LIMIT, TCA_EXTEND_DENOM, WPN_GAMMA_SHIFT,
    ZERO_PROGRESS_LIMIT,
};

/// path 配列の容量．depth の最大値(41) + マージン．
const PATH_CAPACITY: usize = 48;

/// 局面ごとの mid() exit 種別内訳 (visit_diag feature)．
///
/// TT miss (biased default) か TT hit (stored value) かの区別:
///   biased default は `(biased_pn, PN_UNIT, 0)` を返すため tt_dn == PN_UNIT．
///   TT hit の場合 tt_dn は子の dn から計算された値であり，PN_UNIT と一致する
///   のは極めてまれ．よって exploration 時の `tt_dn == PN_UNIT` が TT miss の
///   実用的なシグナルとなる．
#[cfg(feature = "visit_diag")]
#[derive(Default, Clone)]
pub(super) struct VisitBreakdown {
    pub in_path: u32,
    pub proven_exit: u32,
    pub threshold_exit: u32,
    pub th_pn_default: u32,
    pub th_pn_inf: u32,
    pub th_pn_caused: u32,
    pub th_dn_caused: u32,
    pub exploration: u32,
    pub expl_pn_first: u32,    // 初回 exploration 時の tt_pn
    pub expl_dn_first: u32,    // 初回 exploration 時の tt_dn
    pub expl_pn_last: u32,     // 最新 exploration 時の tt_pn
    pub expl_dn_last: u32,     // 最新 exploration 時の tt_dn
    pub expl_pn_stuck: u32,    // tt_pn が前回 exploration と同じだった回数
    pub expl_tt_miss: u32,     // tt_dn == PN_UNIT (biased default = TT miss シグナル)
    pub expl_tt_hit: u32,      // tt_dn != PN_UNIT (TT に格納済みの値)
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

/// 捕獲後局面の proof/disproof 情報を要約する直接マップキャッシュ (v0.24.64)．
///
/// TT の hand_hash Zobrist 混合によるクラスタ分散を迂回し，`pos_key` のみで
/// O(1) lookup を実現する．`try_prefilter_block` の TT lookup 前に参照し，
/// proof/disproof ヒット時は TT lookup を完全にスキップできる．
///
/// ## エントリ構造
///
/// - `min_proof_hand`: この pos_key で詰む最弱の攻方手駒 (要素ごとの min)
/// - `max_disproof_hand`: この pos_key で詰まない最強の攻方手駒 (要素ごとの max)
///
/// ## ライフサイクル
///
/// - `solve()` 入口で全クリア
/// - IDS 反復間は保持 (proof は depth 非依存，disproof は保守的な方向)
/// - cross_deduce / prefilter / reverse_disproof_sharing で更新
/// - (v0.24.68) 容量を 64K → 256K に拡大し hash collision を削減．
///   MID proof store 時の更新は不可: MID が store する proof hand は
///   OR ノードの文脈であり，prefilter が lookup する post-capture 文脈
///   の proof hand とは異なる．min_proof_hand が不当に縮小され
///   探索パスが変化する退行が確認された (test_tsume_39te_ply24_mate15)
const PC_SUMMARY_SIZE: usize = 262144; // 256K entries, ~5.2 MB

struct PostCaptureSummary {
    /// `(pos_key, min_proof_hand, max_disproof_hand)` の直接マップ．
    /// pos_key == 0 は空エントリを表す．
    keys: Vec<u64>,
    proof_hands: Vec<[u8; HAND_KINDS]>,
    disproof_hands: Vec<[u8; HAND_KINDS]>,
}

impl PostCaptureSummary {
    fn new() -> Self {
        Self {
            keys: vec![0u64; PC_SUMMARY_SIZE],
            proof_hands: vec![[u8::MAX; HAND_KINDS]; PC_SUMMARY_SIZE],
            disproof_hands: vec![[0u8; HAND_KINDS]; PC_SUMMARY_SIZE],
        }
    }

    fn clear(&mut self) {
        self.keys.fill(0);
        self.proof_hands.fill([u8::MAX; HAND_KINDS]);
        self.disproof_hands.fill([0u8; HAND_KINDS]);
    }

    #[inline]
    fn idx(pos_key: u64) -> usize {
        (pos_key as usize) & (PC_SUMMARY_SIZE - 1)
    }

    /// proof hand を記録する．
    ///
    /// 既存エントリがある場合，新しい proof hand が forward-chain で既存を
    /// 支配する (より弱い) ときのみ上書きする．非比較な手駒同士の element-wise
    /// min は false proof を生成するリスクがあるため行わない (v0.24.67)．
    #[inline]
    fn record_proof(&mut self, pos_key: u64, proof_hand: &[u8; HAND_KINDS]) {
        let i = Self::idx(pos_key);
        if self.keys[i] == pos_key {
            // 既存エントリ: 新 proof が既存を forward-chain 支配する場合のみ更新
            if hand_gte_forward_chain(&self.proof_hands[i], proof_hand) {
                self.proof_hands[i] = *proof_hand;
            }
            // 逆方向 (既存が新を支配) や非比較の場合は既存を保持
        } else {
            // 新規 or 衝突: 上書き
            self.keys[i] = pos_key;
            self.proof_hands[i] = *proof_hand;
            self.disproof_hands[i] = [0u8; HAND_KINDS];
        }
    }

    /// disproof hand を記録する．
    ///
    /// 既存エントリがある場合，新しい disproof hand が forward-chain で既存を
    /// 支配する (より強い) ときのみ上書きする．非比較な手駒同士の element-wise
    /// max は false disproof を生成するリスクがあるため行わない (v0.24.67)．
    #[inline]
    fn record_disproof(&mut self, pos_key: u64, disproof_hand: &[u8; HAND_KINDS]) {
        let i = Self::idx(pos_key);
        if self.keys[i] == pos_key {
            // 既存エントリ: 新 disproof が既存を forward-chain 支配する場合のみ更新
            if hand_gte_forward_chain(disproof_hand, &self.disproof_hands[i]) {
                self.disproof_hands[i] = *disproof_hand;
            }
        } else {
            // 新規 or 衝突: 上書き
            self.keys[i] = pos_key;
            self.disproof_hands[i] = *disproof_hand;
            self.proof_hands[i] = [u8::MAX; HAND_KINDS];
        }
    }

    /// proof hand を lookup する．
    /// `hand ≥_fc min_proof_hand` なら proven と判定可能．
    #[inline]
    fn lookup_proof(&self, pos_key: u64) -> Option<&[u8; HAND_KINDS]> {
        let i = Self::idx(pos_key);
        if self.keys[i] == pos_key && self.proof_hands[i] != [u8::MAX; HAND_KINDS] {
            Some(&self.proof_hands[i])
        } else {
            None
        }
    }

    /// disproof hand を lookup する．
    /// `max_disproof_hand ≥_fc hand` なら disproven と判定可能．
    ///
    /// 現在は reverse\_disproof\_sharing (v0.24.61+) が TT を直接参照するため未使用．
    /// dead code (削除候補)．
    #[inline]
    #[allow(dead_code)]
    fn lookup_disproof(&self, pos_key: u64) -> Option<&[u8; HAND_KINDS]> {
        let i = Self::idx(pos_key);
        if self.keys[i] == pos_key && self.disproof_hands[i] != [0u8; HAND_KINDS] {
            Some(&self.disproof_hands[i])
        } else {
            None
        }
    }
}

/// Df-Pn ソルバー．
pub struct DfPnSolver {
    /// 最大探索手数．
    pub(super) depth: u32,
    /// 最大ノード数．
    pub(super) max_nodes: u64,
    /// 引き分け手数．
    pub(super) draw_ply: u32,
    /// 実行時間制限．
    pub(super) timeout: Duration,
    /// 転置表(固定サイズ，証明駒/反証駒対応)．
    pub(super) table: TranspositionTable,
    /// 探索ノード数．
    pub(super) nodes_searched: u64,
    /// 探索中の最大ply(デバッグ用)．
    pub(super) max_ply: u32,
    /// ply別ノード数(デバッグ用)．
    pub(super) ply_nodes: [u64; 64],
    /// ply別MIDループイテレーション数(デバッグ用)．
    pub(super) ply_iters: [u64; 64],
    /// ply別停滞ペナルティ回数(デバッグ用)．
    pub(super) ply_stag_penalties: [u64; 64],
    /// ルート局面情報(進捗追跡用)．
    pub(super) diag_root_pk: u64,
    pub(super) diag_root_hand: [u8; HAND_KINDS],
    /// 探索中のパス(ループ検出用，フルハッシュ)．
    /// 固定長配列 + 長さによるスタック実装．LIFO 規律で insert/remove する．
    /// 容量は `PATH_CAPACITY`(48)で，`depth` の最大値(41) + マージン．
    /// `depth >= PATH_CAPACITY` の場合は実行時にパニックする．
    pub(super) path: [u64; PATH_CAPACITY],
    /// パスの各ノードの pos_key (盤面ハッシュ，持ち駒除外)．
    /// ProvenTT の祖先チェックに使用する．
    pub(super) path_pos_key: [u64; PATH_CAPACITY],
    /// パスの各ノードの hand (攻め方持ち駒)．
    /// ProvenTT の祖先チェックに使用する．
    pub(super) path_hand: [[u8; HAND_KINDS]; PATH_CAPACITY],
    pub(super) path_len: usize,
    /// path 配列の高速ルックアップ用 HashSet．
    /// ループ検出の O(depth) 線形スキャンを O(1) に改善する．
    /// path[0..path_len] と常に同期する．
    pub(super) path_set: FxHashSet<u64>,
    /// 探索開始時刻．
    pub(super) start_time: Instant,
    /// タイムアウトしたかどうか．
    pub(super) timed_out: bool,
    /// 攻め方の手番色(solve 時に設定)．
    pub(super) attacker: Color,
    /// 最短手数探索を行うかどうか(デフォルト: true)．
    ///
    /// true の場合，`solve()` は `complete_or_proofs()` を呼び出して
    /// 全 OR ノードの未証明子を追加証明し，最短手順を保証する．
    /// false の場合，最初に見つかった詰み手順をそのまま返す．
    pub(super) find_shortest: bool,
    /// PV 復元フェーズで未証明子1つあたりに割り当てるノード予算(デフォルト: 1024)．
    ///
    /// 長手数の詰将棋で [`TsumeResult::CheckmateNoPv`] が返る場合，
    /// この値を増やすことで PV 復元の成功率が向上する．
    pub(super) pv_nodes_per_child: u64,
    /// TT GC 閾値: TT のエントリ数がこの値を超えると GC を実行する．
    ///
    /// 0 にすると GC を無効化する．
    /// デフォルトは 0(無効)．超長手数問題で OOM を防ぐ場合に設定する．
    /// 推奨値: 探索ノード数の 1/5〜1/2 程度(例: 100M ノードなら 20M〜50M)．
    pub(super) tt_gc_threshold: usize,
    /// 直前の `generate_defense_moves` で計算されたチェーンマスのビットボード．
    ///
    /// `mid()` 内で合駒がチェーンマスへのドロップかどうかを判定するために使用．
    /// 各 `mid()` 呼び出しで更新され，飛び駒の王手がない場合は空．
    pub(super) chain_bb_cache: Bitboard,
    /// PV 抽出が AND ノードで全 defender 子を visit budget 内に
    /// 評価し切れなかった場合に true になる．
    /// solve() はこのフラグが立っていると Mate を返さず CheckmateNoPv を
    /// 返すことで，未検証の応手が残った状態の PV を「真の最長抵抗」として
    /// 表示する soundness 違反を防ぐ．
    pub(super) pv_extraction_incomplete: bool,
    /// 王手生成キャッシュ(E2 最適化)．
    pub(super) check_cache: CheckCache,
    /// TT ベース合駒プレフィルタの発火回数(診断用)．
    pub(super) prefilter_hits: u64,
    /// 捕獲後局面の proof/disproof サマリキャッシュ (v0.24.64)．
    ///
    /// TT の hand_hash クラスタ分散を迂回し，prefilter の miss 率を改善する．
    pub(super) pc_summary: PostCaptureSummary,
    /// サマリキャッシュの proof ヒット回数 (tt_diag 診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_pc_summary_proof_hits: u64,
    /// サマリキャッシュの disproof ヒット回数 (tt_diag 診断用)．
    /// 現在は未使用．dead code (削除候補)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) diag_pc_summary_disproof_hits: u64,
    // diag_chain_inner_outer_hits: 試行不採用 (v0.24.65, soundness 違反で revert)
    /// NM 昇格の反証判定キャッシュ: 判定が false だった局面キーの集合．
    ///
    /// `depth_limit_all_checks_refutable` は局面のみに依存し探索深さに依存しないため，
    /// 一度 false と判定された局面を再判定する必要はない．
    /// MID 内部で同一局面の重複判定を回避し，パフォーマンスを改善する．
    pub(super) refutable_check_failed: FxHashSet<u64>,
    /// **F3**: OR レベル refutable 成功局面のキャッシュ (v0.25.4)．
    /// `(pos_key, outer_solve_depth)` でタグ付けし solve() 内のみで有効．
    /// `param_refut_or_success_cache=true` のときに使用．
    pub(super) refutable_check_succeeded: FxHashSet<u64>,
    /// OR ノードの子ポジション別 stale effort 追跡．
    ///
    /// 次に TT GC チェックを行うノード数．
    pub(super) next_gc_check: u64,
    /// overflow GC のクールダウン(次に GC を許可するノード数)．
    next_overflow_gc: u64,
    // === チューニング可能パラメータ ===
    /// 1+ε の epsilon 除数．
    ///
    /// - `EPSILON_DENOM_ADAPTIVE` (0): depth-adaptive モード．
    ///   `saved_depth_for_epsilon >= 19` なら 2，それ以外は 3．
    /// - 正の値: その値を固定除数として使用(テスト用)．
    pub(super) param_epsilon_denom: u32,
    /// AND pn_floor の分子(デフォルト 2: pn_floor = eff_pn_th * 2/3)．
    pub(super) param_pn_floor_numer: u32,
    /// AND pn_floor の分母(デフォルト 3)．
    pub(super) param_pn_floor_denom: u32,
    /// DN_FLOOR の PN_UNIT 倍率(デフォルト 100: DN_FLOOR = 100 * PN_UNIT)．
    pub(super) param_dn_floor_mult: u32,
    /// Deep df-pn の深さ係数 R(デフォルト 4)．
    pub(super) param_deep_dfpn_r: u32,
    /// PNS アリーナの最大ノード数 (v0.25.0)．
    ///
    /// デフォルトは `PNS_MAX_ARENA_NODES` (5M)．大きくすると spin 率が下がる
    /// 代わりにメモリ消費が増える (1 ノード ≈ 80〜120 bytes)．
    /// 例: 10M で約 800〜1,200 MB．
    pub(super) param_pns_arena_max: usize,
    /// 深さ制限反証 (depth-limited disproof) を WorkingTT に格納する
    /// 最小 `remaining` 閾値．
    ///
    /// `remaining < param_disproof_remaining_threshold` の depth-limited
    /// disproof は格納をスキップする (path_dependent と confirmed disproof は
    /// 対象外)．
    ///
    /// - `DISPROOF_THRESHOLD_ADAPTIVE` (sentinel = `u16::MAX`) の場合，
    ///   solve() 時に `outer_solve_depth` に基づいて自動決定する (v0.25.1)．
    /// - 正の u16 値: 固定閾値として使用 (テスト・チューニング用)．
    /// - 0: スキップなし (従来動作).
    ///
    /// デフォルトは `DISPROOF_THRESHOLD_ADAPTIVE` (depth-adaptive)．
    /// ply 18 ベンチマークで WorkingTT churn (87% eviction) の削減を狙いつつ，
    /// ply 24 などの shallow 問題での退行を回避する．
    pub(super) param_disproof_remaining_threshold: u16,

    /// **visit_history dominance check** (KomoringHeights `IsSuperior` 相当)．
    ///
    /// true で，子展開時に `(child_pos_key, child_hand)` が現在の探索パス
    /// `path_pos_key[i] / path_hand[i]` の祖先で支配されている場合
    /// (`hand_gte_forward_chain(ancestor_hand, child_hand)` が同一 board_key で
    /// 成立) を loop と同じく経路依存不詰として扱う．
    ///
    /// **Why:** 攻め方が同じ局面に「より少ない持ち駒」で再訪している場合，
    /// 過去のより有利な状態でも詰めなかった事実から現在も詰まないと sound に
    /// 推論できる．chain aigoma で hand 多様性が指数爆発する局面の枝刈りに効く．
    ///
    /// **soundness:** 経路依存 (path_dependent) 反証として扱われるため
    /// 永続 TT エントリは汚染されない．`is_loop_child` と同じパスで処理されるため
    /// 既存の GHI 機構と整合する．
    ///
    /// デフォルトは false (既存挙動と同等)．効果測定のため opt-in で有効化する．
    /// 関連: KomoringHeights v1.1.0 `visit_history.hpp::IsSuperior`．
    pub(super) param_use_visit_history_dominance: bool,

    /// **HandSet OR disproof 交集合演算** (KomoringHeights `local_expansion.hpp::HandSet`
    /// の DisproofHandTag 動作相当)．
    ///
    /// true で，OR ノード全子反証時 (`current_dn == 0`) の disproof_hand を
    /// 全子の `disproof_hand` の要素ごと min (交集合) で計算して store する．
    /// 既存実装は `att_hand` をそのまま渡しており，子の精密 disproof_hand
    /// 情報を活用していなかった．
    ///
    /// **Why:** 全子 (攻め手) で反証された場合，「これ以下の持ち駒では全子で詰まない」
    /// 最も強い disproof hand は子 disproof_hand の要素 min．これにより TT
    /// cross-branch ヒット率向上が期待される．
    ///
    /// **soundness:** 子の `get_disproof_hand` 自体は sound (TT lookup で
    /// `hand_gte_forward_chain` を確認済み)．要素 min はその制約を強める
    /// 方向の操作のため不正は導入しない．
    ///
    /// デフォルトは false．関連: KomoringHeights `hands.hpp::HandSet`．
    pub(super) param_use_handset_combination: bool,

    /// Tier 3 (twinkling-hatching-duckling, v0.65.0): KH 風 `DelayedMoveList`．
    ///
    /// true で，AND ノード multi-child loop で同マス合駒の chain (prev/next 双方向リスト) を
    /// 構築し，「prev が未解決なら next の子を skip」する semantics を適用する．
    /// 全合駒展開による pn 過大評価を抑止する．
    ///
    /// 関連: `delayed_move_list.rs`，`docs/plans/twinkling-hatching-duckling.md` Tier 3．
    pub(super) param_use_delayed_move_list: bool,

    /// twinkling-hatching-duckling Phase B (v0.66.0) / melodic-cascading-otter
    /// (v0.69.0): path-aware DAG 補正．KH `double_count_elimination` 相当．
    ///
    /// true で，AND multi-child loop で transposition DAG (child の TT-stored
    /// 親が現 path 上の先祖と一致する) を検出し，sum 集約から除外し
    /// max のみで集約する．
    ///
    /// 実装は runtime `parent_map` を使用 (v0.69.0)．`or_insert` で
    /// child の最初の親のみ記録．以後の同 child 再訪問では parent_map
    /// から最初の親を取得し，それが path 上にあれば DAG と判定．
    pub(super) param_use_dag_correction: bool,

    /// Phase 14 (v0.93.0): mid_v2 用 expansion stack．KH ExpansionStack 相当．
    /// EliminateDoubleCount で ancestor 走査するため．
    pub(super) mid_expansion_stack: Vec<super::mid_v2::MidLocalExpansion>,

    /// Phase 14 (v0.93.0): 各 expansion フレームで「親→子」の選択 move を記録．
    /// resolve_double_count_if_branch_root で branch_root の子を同定するため．
    /// `mid_frame_moves[i]` = mid_expansion_stack[i] フレームから次フレームへ
    /// 進んだ best_move (の move16)．
    pub(super) mid_frame_moves: Vec<u16>,

    /// melodic-cascading-otter (v0.69.0): DAG 検出用 parent_map．
    /// `child_full_hash → parent_full_hash`．mid() で子に再帰する直前に
    /// `or_insert` で挿入 (既存 entry は上書きしない)．find_known_ancestor で
    /// child から先祖チェーンを辿る際に参照．solve() 開始時に clear．
    pub(super) parent_map: rustc_hash::FxHashMap<u64, u64>,

    /// Phase 14 (v0.93.0): KH `LookUpParent` 風 metadata．
    /// child_full_hash → (child_pos_key, child_hand)．
    /// DAG correction の chain walk で各 step に TT lookup するために必要．
    pub(super) parent_meta: rustc_hash::FxHashMap<u64, (u64, [u8; HAND_KINDS])>,

    /// Phase 15 (v0.94.0): tsume_5 micro-tuning 診断用．
    /// position_fh → visit count．mid_v2 で各 entry 時にインクリメント．
    pub(super) mid_v2_visit_counts: rustc_hash::FxHashMap<u64, u32>,

    /// Phase 20 (v0.99.0): KH `min_depth` 相当の max_remaining トラッキング．
    /// (pos_key, hand_hash) → max_remaining (= shallowest ply で stored)．
    /// store 時に更新，mid_v2 の has_old_child 判定で参照．
    pub(super) max_remaining_map: rustc_hash::FxHashMap<u64, u16>,

    /// Phase 22: 1+ε 閾値 epsilon (KH デフォルト 1; PN_UNIT スケール考慮で PN_UNIT が候補)．
    pub(super) param_threshold_epsilon: u32,
    /// Phase 22: TCA extension formula を KH `max(thpn, pn+1)` 形式にするか．
    pub(super) param_tca_kh_clamp: bool,
    /// Phase 22: root level IDS (KH SearchEntry 風 1.7× threshold growth) 有効．
    pub(super) param_root_ids_enable: bool,
    /// Phase 21: deferred penalty 除数 (0=無効, 8=KH 準拠)．
    pub(super) param_deferred_penalty_denom: u32,
    /// Phase 22: deferred penalty `.max(1)` floor (KH=true)．
    pub(super) param_deferred_penalty_floor: bool,
    /// Phase 21: TCA gate を is_shallow ベースにするか (true=v0.99.0, false=旧 !is_first_visit)．
    pub(super) param_tca_use_shallow_gate: bool,
    /// Phase 21 診断: deferred penalty 発火フレーム数．
    pub(super) diag_deferred_frames: u64,
    /// Phase 21 診断: deferred penalty 合計値．
    pub(super) diag_deferred_penalty_sum: u64,
    /// Phase 21 診断: has_old_child=true (is_shallow gate)．
    pub(super) diag_tca_shallow_fire: u64,
    /// Phase 21 診断: has_old_child=false だが旧ロジックでは true になるケース．
    pub(super) diag_tca_shallow_would_fire: u64,

    /// melodic-cascading-otter (v0.69.0): DAG 検出の診断カウンタ．
    /// `param_use_dag_correction=true` の場合に動作確認用．
    /// solve() 開始時にゼロクリア．
    pub(super) diag_dag_calls: u64,
    pub(super) diag_dag_true: u64,
    pub(super) diag_dag_short_first: u64,  // step 0 で immediate_parent と一致して終了
    pub(super) diag_dag_short_none: u64,   // parent_map に未登録で終了
    pub(super) diag_dag_max_step: u32,
    pub(super) diag_dag_walks_16: u64,     // 16 step 完走したケース

    /// melodic-cascading-otter Plan B (v0.70.0): KH 流 TCA inc_flag を有効化する．
    ///
    /// true で，mid() が「loop child を検出した時に inc_flag を increment し，
    /// inc_flag > 0 の間は閾値拡張 (eff_pn_th/eff_dn_th) を maintain する」
    /// 形に変更．既存 maou は loop_child_count > 0 を **同一 mid() フレーム内で
    /// 検出された場合のみ** 適用するが，KH は inc_flag を recursion で伝播する．
    ///
    /// 効果: 深い transposition chain (29te) で，先祖フレームで検出された
    /// loop が子孫フレームの閾値にも反映され，wrong branch から早期復帰する．
    pub(super) param_use_kh_tca: bool,

    /// melodic-cascading-otter Plan B (v0.70.0): KH inc_flag の現在値．
    /// solve() 開始時に 0．mid() 内で saturating_add/sub し，
    /// mid() 終了時に min(self.inc_flag, orig_inc_flag) で巻き戻し．
    pub(super) inc_flag: u32,

    /// Plan B 診断: TCA 発火回数．
    pub(super) diag_tca_increments: u64,
    pub(super) diag_tca_decrements: u64,
    pub(super) diag_tca_extends: u64,

    /// melodic-cascading-otter 診断 (v0.71.0): root (ply 0) trace．
    /// true で，mid() の OR/AND multi-child loop が ply==0 で iterate するたびに
    /// 主要 metrics を eprintln! でダンプする (feature flag 不要)．
    ///
    /// 出力例 (interval=10000 nodes):
    /// ```
    /// [root_trace ply0 iter=42 nodes=100000] children=23 best_idx=3 best_move=S*7i
    ///   current_pn=156 current_dn=89 pn_th=200 dn_th=200
    ///   child[0] move=S*7i pn=124 dn=45 (root_visits=520)
    ///   ...
    /// ```
    pub(super) param_root_trace: bool,

    /// 診断 (v0.71.3): trace 対象の ply (default 0 = root)．
    pub(super) param_trace_ply: u32,

    /// 診断 (v0.71.4): 全 path 上の child の pn/dn を 1 dump 当たり最大何件出すか．
    pub(super) param_trace_full_children: bool,

    /// 診断 (v0.72.0): TT lookup hit rate を計測する．`param_tt_diag` で有効化．
    /// `look_up_pn_dn` 呼び出しごとに以下を更新:
    /// - `diag_tt_lookups`: 総呼び出し回数
    /// - `diag_tt_misses`: 初期値 (PN_UNIT, PN_UNIT, 0) が返った回数
    /// - `diag_tt_proven`: (0, INF, _) が返った回数
    /// - `diag_tt_disproven`: (INF, 0, _) が返った回数
    /// - `diag_tt_working`: その他 (working entry hit)．
    pub(super) param_tt_lookup_diag: bool,
    // Cell で `&self` の look_up_pn_dn からも更新可能にする
    pub(super) diag_tt_lookups: std::cell::Cell<u64>,
    pub(super) diag_tt_misses: std::cell::Cell<u64>,
    pub(super) diag_tt_proven: std::cell::Cell<u64>,
    pub(super) diag_tt_disproven: std::cell::Cell<u64>,
    pub(super) diag_tt_working: std::cell::Cell<u64>,

    /// 診断 (v0.73.0): per-depth proven 蓄積カウンタ．
    /// `store_proof_with_tag` 呼び出し時に `path_len` (= 現在の ply) を
    /// インデックスとして increment．solve() 完了後に proof がどの深さで
    /// 累積したかのヒストグラムが取れる．候補 E．
    pub(super) diag_proven_per_ply: [u64; 64],

    /// 診断 (v0.77.0): AND ノード multi-child loop 完了時の coverage 統計．
    /// `(proven_count, total_children)` のヒストグラム．
    /// proven_count ratio が低いまま loop が exit → AND coverage が不足．
    pub(super) diag_and_visit_count: u64,
    pub(super) diag_and_proven_sum: u64,    // proven_count の合計
    pub(super) diag_and_total_sum: u64,     // total_children の合計
    pub(super) diag_and_zero_proven: u64,   // proven_count == 0 で exit した回数
    pub(super) diag_and_full_proven: u64,   // proven_count == total で exit (本来は ProveAND して exit)

    /// 診断 (v0.78.0): OR scan coverage 統計 (AND と対称)．
    pub(super) diag_or_visit_count: u64,
    pub(super) diag_or_proven_count_visits: u64,  // proven child があった visits の数
    pub(super) diag_or_total_sum: u64,

    /// 診断 (v0.79.0, A): per-position revisit count．`(pos_key, or_node) → visit_count`．
    /// max 1M entries に制限してメモリ暴走防止．
    pub(super) diag_pos_visits: rustc_hash::FxHashMap<u64, u32>,
    pub(super) diag_pos_visits_capped: bool,  // 1M 超えたら collection 停止

    /// 候補 C (v0.80.0): AND node exhaustive defender prove．
    /// default false．true で AND multi-child loop の best_idx 選択を
    /// "未訪問 defender 優先 round-robin" に切り替える．具体的には:
    /// - 全 children のうち cpn != 0 (まだ proven でない) defender の中で
    ///   `self.exhaustive_and_rr_counter % unproven_count` 番目を選択
    /// - これにより同じ defender に固定せず全 defender を巡回 prove
    pub(super) param_use_exhaustive_and: bool,
    pub(super) exhaustive_and_rr_counter: u32,

    /// 候補 D (v0.81.0): per-AND-position の proven defender bitmap．
    /// `pos_key → u64 bitmap` で「どの children index がすでに proven 化されたか」
    /// を persistent 追跡．`param_use_and_proven_bitmap=true` で有効化．
    /// AND scan で bitmap 上の bit が立つ defender は **selection 候補から除外**．
    /// VPN は cpn==0 を毎回 lookup で判定するが，D はそれを memoize し
    /// remaining mismatch や TT churn による誤判定からも保護．
    pub(super) param_use_and_proven_bitmap: bool,
    pub(super) and_proven_bitmap: rustc_hash::FxHashMap<u64, u64>,

    /// melodic-cascading-otter 候補 G (v0.74.0): root child_pn_th の絶対 floor．
    /// 0 (default) で無効．> 0 で root (ply=0) の OR child_pn_th を最低
    /// この値まで引き上げる．これにより 1 つの child に深く commit するための
    /// pn 予算を保証する．推奨値の出発点: 100_000 (= 6250 * PN_UNIT)．
    pub(super) param_root_child_pn_floor: u32,

    /// melodic-cascading-otter 候補 F (v0.75.0): OR ノード best_idx 選択の
    /// mate path commitment．default false で従来挙動 (argmin pn)．true で:
    /// argmin pn の tie 時 (`pn == best_pn`) に max(dn) で tie-break する．
    /// = defender の抵抗が強い attack を優先 → 探索 commit 強化．
    pub(super) param_or_dn_tiebreak: bool,

    /// Phase 8 (v0.88.0): find_shortest 用 mate_distance 制約．
    /// `Some(md)` で look_up_pn_dn が proven 局面 (pn=0) かつ
    /// stored mate_distance > md を「未証明」として扱う (PN_UNIT, PN_UNIT)．
    /// mid_v2 が shorter mate を強制探索する．
    /// None (default) で制約なし．
    pub(super) param_max_mate_distance: Option<u16>,

    /// 診断 (v0.71.1): periodic GC (overflow-based working TT GC) を無効化する．
    /// default false (= GC fire OK)．true で `nodes_searched % 100_000 == 0` の
    /// GC トリガを skip する．catastrophic forgetting の検証用．
    pub(super) param_disable_periodic_gc: bool,

    /// 診断 (v0.71.2): IDS の浅い depth 反復を skip して full depth から開始する．
    /// default false (= 通常 IDS depth=2,4,6,...)．true で `ids_depth=saved_depth` 直行．
    /// IDS による TT 再評価 (remaining 違いで初期値復活) の影響を排除する検証用．
    pub(super) param_skip_ids_shallow: bool,

    /// root_trace の dump 間隔 (nodes)．`param_root_trace=true` のときに使用．
    pub(super) root_trace_interval: u64,
    pub(super) root_trace_next: u64,
    pub(super) root_trace_iter: u64,

    /// twinkling-hatching-duckling Phase C (v0.67.0): KH 風 per-move 差別化．
    ///
    /// true で，OR child の `edge_cost_or` を `edge_cost_or_with_support` に
    /// 切り替え．`compute_checkers_at` で to_sq の attack/defense support を算出し，
    /// 受け駒 ≥ 2 のマスを後回し，攻め支援 > 受け支援のマスを優先する．
    /// tsume_5 step-by-step 分析で ply 0 で 4.2× の差を縮める目的．
    pub(super) param_use_per_move_support: bool,

    // === M-1 refutable check fast path 改善フラグ (v0.25.4) ===
    /// **F1**: `all_checks_refutable_recursive_inner` で false 確定 check
    /// で早期 return せず，全 check を評価して store する．
    /// partial coverage の積み上げで fast path 発火率向上を狙う．
    /// trade-off: recursive cost +50%〜100% の見込み．
    pub(super) param_refut_full_eval: bool,
    /// **F2**: fast path で部分 match した場合，残り missing check のみを
    /// recursive で評価し，全 check 完成度を効率的に達成する．
    pub(super) param_refut_partial_recursion: bool,
    /// **F3**: OR レベル refutable 成功も `refutable_check_succeeded` cache
    /// に格納する．v0.24.74 で false NM の根源だったため，
    /// `(pos_key, outer_solve_depth)` でタグ付けし IDS depth ごとに分離．
    pub(super) param_refut_or_success_cache: bool,
    /// **F4**: fast path lookup を ProvenTT のみから WorkingTT も含む
    /// `look_up` に拡張．depth-limited disproof (rem>=floor) も match
    /// として count し，coverage を向上．
    pub(super) param_refut_extended_lookup: bool,
    /// solve() に渡された真の目標 depth (v0.24.66)．
    ///
    /// IDS NM 昇格判定で true depth を参照するために保持する．
    /// 0 の場合は未設定 (saved_depth をそのまま使用)．
    pub(super) outer_solve_depth: u32,
    /// 最終 IDS depth (solve 時の depth)．
    /// IDS 中に self.depth が変化するため，epsilon の depth-adaptive 判定に
    /// 最終 depth を保持する．mid_fallback の入口で設定される．
    pub(super) saved_depth_for_epsilon: u32,
    /// Killer Move テーブル(OR ノード用)．
    ///
    /// ply ごとに最大 2 つの killer move(Move16)を保持する．
    /// 閾値超過でカットオフを引き起こした手を記録し，
    /// 同じ ply の他の局面でも優先的に探索する．
    /// TT Best Move とは異なり局面に依存しない手順ヒントを提供する．
    pub(super) killer_table: Vec<[u16; 2]>,
    /// プロファイリング統計情報(`profile` feature 有効時のみ)．
    #[cfg(feature = "profile")]
    pub(super) profile_stats: ProfileStats,
    /// 直前の PNS サイクルで TT に格納された proof (pn=0) エントリ数．
    ///
    /// `pns_main_with_arena` の出口で `pns_store_to_tt` の戻り値が格納される．
    /// Frontier Variant の zero-proof early skip 判定に使用する．
    pub(super) last_pns_proof_stores: u64,
    /// K-M dual TT: all_checks_refutable_by_tt 呼び出し回数(K-M 拡張含む)．
    pub(super) diag_km_calls: u64,
    /// K-M dual TT: all_checks_refutable_by_tt で K-M cycle_root 判定が成功した回数．
    /// (confirmed disproof では refuted されず，K-M 経路で refuted された回数)
    pub(super) diag_km_hits: u64,
    /// K-M dual TT: all_checks_refutable_by_tt 全体の成功回数
    /// (K-M 経由 + confirmed disproof 経由の合計)．
    pub(super) diag_km_total_refuted: u64,
    /// 診断: PNS 空回り(pn/dn 不変)イテレーション数．
    ///
    /// 500M 予算テスト等で予算が有効に使われているかを確認する．
    /// 1 PNS イテレーションの前後で `arena[0].(pn,dn)` が完全に同一
    /// だった回数を累積する．pns_main_with_arena の出口で solver に
    /// 反映される．
    #[cfg(feature = "verbose")]
    pub(super) dbg_pns_spin_iters: u64,
    /// 診断: PNS pn/dn 変化があったイテレーション数．
    #[cfg(feature = "verbose")]
    pub(super) dbg_pns_changed_iters: u64,
    /// 診断: PNS が TT に格納した新規 proof (pn=0) エントリ数(累計)．
    ///
    /// `pns_store_to_tt` で実際に TT に書き込まれた証明ノード数．
    /// MID がこれらの proof を TT ヒットで再利用できるため，
    /// PNS の実効的な生産性を測定する指標となる．
    #[cfg(feature = "verbose")]
    pub(super) dbg_pns_proof_stores: u64,
    /// 診断: PNS サイクルごとのアリーナ新規ノード数(累計)．
    ///
    /// 各 PNS サイクルで展開された新規ノード数．アリーナが成長して
    /// いなければ PNS は既知領域を走査しているだけであり，真の空振りを示す．
    #[cfg(feature = "verbose")]
    pub(super) dbg_pns_arena_growth: u64,
    /// 診断: Frontier Variant の PNS サイクル数(累計)．
    #[cfg(feature = "verbose")]
    pub(super) dbg_pns_cycles: u64,
    /// 診断: `refutable_check_with_cache` の TT 経路ヒット数．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_tt_hits: u64,
    /// 診断: `refutable_check_with_cache` の memoize 経路ヒット数．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_memo_hits: u64,
    /// 診断: `refutable_check_with_cache` の再帰フォールバック
    /// (true を返した) 数．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_recursive_true: u64,
    /// 診断: `refutable_check_with_cache` の再帰フォールバック
    /// (false を返した・memoize した) 数．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_recursive_false: u64,
    /// 診断 (M-1): fast path 試行回数 (TT lookup の総数)．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_fast_attempts: u64,
    /// 診断 (M-1): fast path で 1 個以上の check が match した試行数．
    /// (この値が 0 なら全く重複ヒットしていない)．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_fast_partial: u64,
    /// 診断 (M-1): fast path 試行で見つかった disproof match 累計
    /// (各 attempt で N checks 中 K 個が match した場合 K を加算)．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_fast_match_total: u64,
    /// 診断 (M-1): fast path 試行で評価された check 数の累計．
    #[cfg(feature = "verbose")]
    pub(super) dbg_refut_fast_check_total: u64,
    /// TT 診断: 監視対象の ply(0 = 無効)．
    ///
    /// 指定 ply で MID ループの再帰前後に TT サイズを出力し，
    /// エントリ爆増の原因を特定する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_ply: u32,
    /// TT 診断: 監視対象の手(USI 形式，例: "P*7g")．
    ///
    /// 空文字列の場合は ply のみでフィルタする．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_move_usi: String,
    /// TT 診断: MID ループの反復回数上限(0 = 無制限)．
    ///
    /// 爆増が起きる手を特定した後，少数回の反復に絞って詳細を確認する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_max_iterations: u32,
    /// TT 診断: MID での deferred → children 逐次活性化回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_mid_deferred_activations: u64,
    /// TT 診断: PNS での deferred_drops 活性化回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_pns_deferred_activations: u64,
    /// TT 診断: PNS で活性化時に既に TT 証明済みだった回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_pns_deferred_already_proven: u64,
    /// TT 診断: cross_deduce_deferred で証明除去された合駒数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_cross_deduce_hits: u64,
    /// TT 診断: cross_deduce 外ガード `!or_node && m.is_drop()` 成立数 (v0.24.46+)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_cd_guard_and_drop: u64,
    /// TT 診断: cross_deduce 外ガード `cpn_after == 0` 成立数 (v0.24.46+)．
    /// cross_deduce_children 関数の呼び出し回数と一致する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_cd_guard_child_proven: u64,
    /// TT 診断: cross_deduce_children 内で `has_siblings == false` で早期 return した数 (v0.24.46+)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_cd_no_siblings: u64,
    /// TT 診断: cross_deduce_children の本体ループに入った数 (v0.24.46+)．
    /// `diag_cd_guard_child_proven - diag_cd_no_siblings` と等しいはず．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_cd_entered_main: u64,
    /// TT 診断: 施策 A-4 (v0.24.50+) の境界層 DN inflation 発火数．
    ///
    /// AND ノードで `remaining <= 2 && chain_bb_cache 非空` の際に chain drop
    /// (chain_king_sq or mixed_chain_king_sq パス) に BOUNDARY_CHAIN_MULT (=8)
    /// 倍の bias を加算した回数の累積．child 評価ループ内で数えるため，1 つの
    /// AND ノード訪問で複数 child に inflation が発火するとその分加算される．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_a4_inflations: u64,
    /// TT 診断 (N-7, v0.27.0): slider drop (香・角・飛) に距離比例追加ペナルティを
    /// 適用した回数．chain_king_sq / mixed_chain_king_sq / 非 chain AND の
    /// 全 AND ノードで集計する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_n7_slider_bonus: u64,
    /// TT 診断: 施策 α (v0.24.54-v0.24.72) で境界層 filter が発火した MID 数．
    /// 施策 α は v0.24.72 で不採用確定．dead code (削除候補)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) diag_alpha_x_filter_applied: u64,
    /// 施策 α (v0.24.54-v0.24.72): chain drop filter フラグ．
    /// v0.24.72 で filter 無効化後は常に false．施策 α は refutable disproof
    /// 機構 (v0.24.75+, aigoma-optimization.md §8.9) で代替されており，
    /// 再有効化の予定なし．dead code (削除候補)．
    pub(super) alpha_x_filter_active: bool,
    /// PNS 探索中に true に設定し，`look_up_pn_dn` で refutable disproof を
    /// スキップする (v0.24.75)．PNS の arena-limited false NM を防止．
    /// MID 探索では false (refutable disproof を通常の NM として使用)．
    pub(super) skip_refutable_disproof: bool,
    /// SNDA (Sequential Non-capturing Drop Avoidance): 直前の OR ノードで
    /// 攻め方が選んだ手．`Move(0)` は「前手なし」を表す．
    /// OR ノードで同駒種の非捕獲打ちを他マスで抑制するために使用する．
    pub(super) prev_attacker_move: Move,
    /// Hypothesis IDS-17 無効化フラグ (v0.27.4)．
    /// true にすると saved_depth 20-26 での depth=16→17 挿入をスキップする (IDS-17 導入前の挙動)．
    /// デフォルト false = IDS-17 有効 (depth=17 を明示的に経由)．
    pub(super) param_no_ids17: bool,
    /// solve() 呼び出し間で ProvenTT を保持するフラグ．
    /// true のとき，solve() 冒頭でWorkingTT のみクリアし ProvenTT を引き継ぐ．
    /// 逐次 backward 解析で前のソルブの ProvenTT を再利用する際に使用する．
    pub(super) preserve_proven_tt: bool,
    /// refutable check の再帰深さ (デフォルト 5)．
    pub(super) param_refutable_depth: u32,
    /// refutable check の呼び出し回数上限 (デフォルト 10,000)．
    pub(super) param_refutable_call_limit: u32,
    /// PNS 初期フェーズ専用の refutable call limit (デフォルト 500)．
    /// MID では param_refutable_call_limit (=10,000) を使用する．
    pub(super) param_pns_refutable_call_limit: u32,
    /// pns_main() 実行中のみ true．depth_limit_all_checks_refutable が
    /// PNS 専用 limit を使うかどうかを制御する．
    pub(super) in_initial_pns_phase: bool,
    /// 施策 A-6 (v0.24.54, v0.24.71 で施策α に置き換え後 v0.24.72 で無効化):
    /// 境界層 PNS 責任転嫁の残り呼出予算．施策 α が refutable disproof
    /// 機構で代替されたため再有効化の予定なし．dead code (削除候補)．
    #[allow(dead_code)]
    pub(super) a6_boundary_pns_calls_remaining: u32,
    /// TT 診断: AND ノード MID ループで deferred_children あり & all_proved=false の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_deferred_not_ready: u64,
    /// TT 診断: AND ノード MID ループで deferred_children あり & all_proved=true の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_deferred_ready: u64,
    /// TT 診断: AND ノードで prefilter 後 deferred_children に入った合駒数(累計)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_deferred_enqueued: u64,
    /// TT 診断: MID ループの総反復数(nodes_searched 以上になりうる)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_mid_loop_iters: u64,
    /// TT 診断: prefilter が remaining < 3 のためスキップされた回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_prefilter_skip_remaining: u64,
    /// TT 診断: prefilter が試行されたが TT ヒットしなかった回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_prefilter_miss: u64,
    /// TT 診断 (v0.24.58 A-fix): prefilter が TT ヒットしたものの
    /// forward-chain soundness guard により skip した回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_prefilter_fc_reject: u64,
    /// TT 診断 (v0.24.59 候補 C): multi-step cross_deduce で
    /// prefilter re-check により proven 化された異マス children 数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_multi_step_hits: u64,
    /// TT 診断 (v0.24.61): 逆方向不詰共有 (reverse disproof sharing) で
    /// post-capture level の disproof が兄弟ドロップに伝搬された回数．
    ///
    /// 強い駒の合駒が不詰 (cdn == 0) → 弱い駒の合駒も不詰 (hand_gte_forward_chain
    /// の逆方向支配) を利用し，捕獲後局面 `(pc_pk, hand_weak)` に disproof を格納
    /// した回数の累積．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_reverse_disproof_hits: u64,
    /// TT 診断 (v0.24.62): multi-step 逆方向不詰共有で異マスの兄弟ドロップに
    /// disproof が伝搬された回数．
    ///
    /// 同一マスの reverse_disproof_sharing 直後に，**異なるマス** のドロップに
    /// 対しても reverse_disproof_sharing を re-trigger して伝搬した回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_multi_step_reverse_disproof_hits: u64,
    /// TT 診断: ply ごとの MID 訪問回数(最大64手)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_ply_visits: [u64; 64],
    /// TT 診断: ply ごとの pn=0 証明ストア回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_ply_proofs: [u64; 64],
    /// TT 診断: try_capture_tt_proof の呼び出し / ヒット回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_capture_tt_calls: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_capture_tt_hits: u64,
    /// TT 診断: MID 早期リターン(閾値チェック)回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_threshold_exits: u64,
    /// TT 診断: MID 早期リターン(tt_pn==0 || tt_dn==0)回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_terminal_exits: u64,
    /// TT 診断: MID ループ内 break 原因別カウンタ．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_loop_break_proved: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_loop_break_threshold: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_loop_break_nodes: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_in_path_exits: u64,
    /// TT 診断: init フェーズでの AND 反証リターン回数(ply 別)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_init_and_disproof_exits: u64,
    /// TT 診断: 単一子最適化パス回数(ply 別)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_single_child_exits: u64,
    /// TT 診断: ノード制限によるリターン回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_node_limit_exits: u64,
    /// TT 診断: depth 境界 OR ノードで王手なし(NM store)の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_boundary_or_no_checks: u64,
    /// TT 診断: depth 境界 OR ノードで全王手 refutable(NM store)の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_boundary_or_refutable: u64,
    /// TT 診断: depth 境界 OR ノードで王手 not all refutable(仮反証，store なし)の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_boundary_or_not_refutable: u64,
    /// TT 診断: depth 境界 AND ノードのヒット回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_boundary_and_total: u64,
    /// TT 診断: depth 境界 OR ノードでの王手手数の合計(平均算出用)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_boundary_or_checks_sum: u64,
    /// TT 診断: PNS proof store の ply 分布(最大 64 手)．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_pns_proof_ply: [u64; 64],
    /// TT 診断: remaining=0 で proof 発見(仮反証回避)の回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_rem0_proof: u64,
    /// TT 診断: remaining=0 で仮反証(provisional disproof)を返した回数．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_rem0_provisional: u64,
    /// ノード制限到達時点の WorkingTT pn/dn 分布スナップショット (分析用)．
    ///
    /// max_nodes に達した最初の mid() 呼び出しで `collect_working_pn_dn_dist()` を
    /// 実行して保存する．None の場合はノード制限に達していないことを示す．
    pub(super) pn_dn_snapshot: Option<([u64; 32], [u64; 32], Vec<u64>)>,
    /// IDS 各 depth 反復終了時点の WorkingTT pn/dn 分布スナップショット列 (分析用)．
    ///
    /// `mid_fallback` 内で各 IDS depth のMID 完了後，TT 遷移 (retain_working_intermediates
    /// / clear_working) の直前に収集する．
    /// `(ids_depth, nodes_searched, elapsed_secs, pn_hist, dn_hist, joint_hist)`
    pub(super) pn_dn_per_depth: Vec<(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)>,
    /// 各局面 (board.hash) の MID 訪問回数 (visit_diag feature 時のみ存在)．
    #[cfg(feature = "visit_diag")]
    pub(super) visit_counts: FxHashMap<u64, u32>,
    /// 各局面が初めて訪問された ply (visit_diag feature 時のみ存在)．
    #[cfg(feature = "visit_diag")]
    pub(super) visit_first_ply: FxHashMap<u64, u8>,
    /// 各局面の mid() exit 種別内訳 (visit_diag feature 時のみ存在)．
    #[cfg(feature = "visit_diag")]
    pub(super) visit_breakdown: FxHashMap<u64, VisitBreakdown>,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64, draw_ply: u32) -> Self {
        Self::with_timeout(depth, max_nodes, draw_ply, 300)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    ///
    /// # Panics
    ///
    /// `depth >= PATH_CAPACITY`(48)の場合パニックする．
    pub fn with_timeout(depth: u32, max_nodes: u64, draw_ply: u32, timeout_secs: u64) -> Self {
        assert!(
            (depth as usize) < PATH_CAPACITY,
            "depth {} exceeds path capacity {}",
            depth, PATH_CAPACITY,
        );
        DfPnSolver {
            depth,
            max_nodes,
            draw_ply,
            timeout: Duration::from_secs(timeout_secs),
            find_shortest: true,
            pv_nodes_per_child: 1024,
            chain_bb_cache: Bitboard::EMPTY,
            pv_extraction_incomplete: false,
            check_cache: CheckCache::new(),
            prefilter_hits: 0,
            pc_summary: PostCaptureSummary::new(),
            #[cfg(feature = "tt_diag")]
            diag_pc_summary_proof_hits: 0,
            #[cfg(feature = "tt_diag")]
            diag_pc_summary_disproof_hits: 0,
            refutable_check_failed: FxHashSet::default(),
            refutable_check_succeeded: FxHashSet::default(),
            param_refut_full_eval: false,
            param_refut_partial_recursion: false,
            // F3 (or_success_cache): default ON (v0.25.5)．
            // ply 22 で nodes -59% / time -53% の大幅改善．
            // ply 24 (shallow) では完全 no-op (safe)．
            // full_hash keying で false positive を防止済み．
            param_refut_or_success_cache: true,
            param_refut_extended_lookup: false,
            tt_gc_threshold: 0,
            next_gc_check: 0,
            next_overflow_gc: 0,
            param_epsilon_denom: EPSILON_DENOM_ADAPTIVE,
            param_pn_floor_numer: 2,
            param_pn_floor_denom: 3,
            param_dn_floor_mult: 100,
            param_deep_dfpn_r: DEEP_DFPN_R,
            param_pns_arena_max: PNS_MAX_ARENA_NODES,
            // (v0.25.6) default: ADAPTIVE．M-A refutable depth floor (v0.25.2)
            // の導入で false-NoMate が根絶されたため，adaptive を安全に default 化．
            // depth ≤ 19: 0, depth 20-22: 1, depth ≥ 23: 3 (§3.6, M-D)．
            param_disproof_remaining_threshold: DISPROOF_THRESHOLD_ADAPTIVE,
            param_use_visit_history_dominance: true,
            // Tier 1 (twinkling-hatching-duckling, v0.64.0): default ON．
            // KH (`hands.hpp::HandSet`) と同じく常時 OR disproof_hand を要素 min で集約．
            // v0.57.0 で NPS 2.4× の効果実証済み (§10.2.27)．
            // OFF にしたい場合は `set_use_handset_combination(false)` を明示呼出．
            param_use_handset_combination: true,
            // Tier 3 (twinkling-hatching-duckling, v0.65.0): KH 風 DelayedMoveList．
            // AND ノードで同マス合駒 chain の prev が未解決なら next を skip．
            // 既存 161 fast tests + Mate(15) PV regression pass で default ON．
            param_use_delayed_move_list: true,
            // twinkling-hatching-duckling Phase B / melodic-cascading-otter (v0.69.0):
            // path-aware DAG 補正．opt-in (default false)．
            param_use_dag_correction: false,
            mid_expansion_stack: Vec::new(),
            mid_frame_moves: Vec::new(),
            parent_map: rustc_hash::FxHashMap::default(),
            parent_meta: rustc_hash::FxHashMap::default(),
            mid_v2_visit_counts: rustc_hash::FxHashMap::default(),
            max_remaining_map: rustc_hash::FxHashMap::default(),
            param_threshold_epsilon: 2,
            param_tca_kh_clamp: false,
            param_root_ids_enable: false,
            param_deferred_penalty_denom: 0,
            param_deferred_penalty_floor: false,
            param_tca_use_shallow_gate: false,
            diag_deferred_frames: 0,
            diag_deferred_penalty_sum: 0,
            diag_tca_shallow_fire: 0,
            diag_tca_shallow_would_fire: 0,
            diag_dag_calls: 0,
            diag_dag_true: 0,
            diag_dag_short_first: 0,
            diag_dag_short_none: 0,
            diag_dag_max_step: 0,
            diag_dag_walks_16: 0,
            param_use_kh_tca: false,
            inc_flag: 0,
            diag_tca_increments: 0,
            diag_tca_decrements: 0,
            diag_tca_extends: 0,
            param_root_trace: false,
            param_trace_ply: 0,
            param_trace_full_children: false,
            param_tt_lookup_diag: false,
            diag_tt_lookups: std::cell::Cell::new(0),
            diag_tt_misses: std::cell::Cell::new(0),
            diag_tt_proven: std::cell::Cell::new(0),
            diag_tt_disproven: std::cell::Cell::new(0),
            diag_tt_working: std::cell::Cell::new(0),
            diag_proven_per_ply: [0u64; 64],
            param_root_child_pn_floor: 0,
            diag_and_visit_count: 0,
            diag_and_proven_sum: 0,
            diag_and_total_sum: 0,
            diag_and_zero_proven: 0,
            diag_and_full_proven: 0,
            diag_or_visit_count: 0,
            diag_or_proven_count_visits: 0,
            diag_or_total_sum: 0,
            diag_pos_visits: rustc_hash::FxHashMap::default(),
            diag_pos_visits_capped: false,
            param_use_exhaustive_and: false,
            exhaustive_and_rr_counter: 0,
            param_use_and_proven_bitmap: false,
            and_proven_bitmap: rustc_hash::FxHashMap::default(),
            param_or_dn_tiebreak: false,
            param_max_mate_distance: None,
            root_trace_interval: 10_000,
            root_trace_next: 0,
            root_trace_iter: 0,
            param_disable_periodic_gc: false,
            param_skip_ids_shallow: false,
            // Phase C (v0.67.0): per-move attack/defense support 差別化．
            // 初期は opt-in (default false)．効果確認後 default ON 検討．
            param_use_per_move_support: false,
            saved_depth_for_epsilon: 0,
            outer_solve_depth: 0,
            killer_table: Vec::new(),
            table: TranspositionTable::new(),
            nodes_searched: 0,
            max_ply: 0,
            ply_nodes: [0; 64],
            ply_iters: [0; 64],
            ply_stag_penalties: [0; 64],
            diag_root_pk: 0,
            diag_root_hand: [0; HAND_KINDS],
            path: [0u64; PATH_CAPACITY],
            path_pos_key: [0u64; PATH_CAPACITY],
            path_hand: [[0u8; HAND_KINDS]; PATH_CAPACITY],
            path_len: 0,
            path_set: FxHashSet::default(),
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            #[cfg(feature = "profile")]
            profile_stats: ProfileStats::default(),
            last_pns_proof_stores: 0,
            diag_km_calls: 0,
            diag_km_hits: 0,
            diag_km_total_refuted: 0,
            #[cfg(feature = "verbose")]
            dbg_pns_spin_iters: 0,
            #[cfg(feature = "verbose")]
            dbg_pns_changed_iters: 0,
            #[cfg(feature = "verbose")]
            dbg_pns_proof_stores: 0,
            #[cfg(feature = "verbose")]
            dbg_pns_arena_growth: 0,
            #[cfg(feature = "verbose")]
            dbg_pns_cycles: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_tt_hits: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_memo_hits: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_recursive_true: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_recursive_false: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_fast_attempts: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_fast_partial: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_fast_match_total: 0,
            #[cfg(feature = "verbose")]
            dbg_refut_fast_check_total: 0,
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
            diag_cd_guard_and_drop: 0,
            #[cfg(feature = "tt_diag")]
            diag_cd_guard_child_proven: 0,
            #[cfg(feature = "tt_diag")]
            diag_cd_no_siblings: 0,
            #[cfg(feature = "tt_diag")]
            diag_cd_entered_main: 0,
            #[cfg(feature = "tt_diag")]
            diag_a4_inflations: 0,
            #[cfg(feature = "tt_diag")]
            diag_n7_slider_bonus: 0,
            #[cfg(feature = "tt_diag")]
            diag_alpha_x_filter_applied: 0,
            alpha_x_filter_active: false,
            skip_refutable_disproof: true, // Q-1 (v0.55.20): MID も refutable disproof をスキップ
            prev_attacker_move: Move(0),
            param_no_ids17: false,
            param_refutable_depth: Self::DEFAULT_REFUTABLE_DEPTH,
            param_refutable_call_limit: Self::DEFAULT_REFUTABLE_CALL_LIMIT,
            param_pns_refutable_call_limit: 500,
            in_initial_pns_phase: false,
            a6_boundary_pns_calls_remaining: 0,
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
            diag_prefilter_fc_reject: 0,
            #[cfg(feature = "tt_diag")]
            diag_multi_step_hits: 0,
            #[cfg(feature = "tt_diag")]
            diag_reverse_disproof_hits: 0,
            #[cfg(feature = "tt_diag")]
            diag_multi_step_reverse_disproof_hits: 0,
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
            #[cfg(feature = "tt_diag")]
            diag_boundary_or_no_checks: 0,
            #[cfg(feature = "tt_diag")]
            diag_boundary_or_refutable: 0,
            #[cfg(feature = "tt_diag")]
            diag_boundary_or_not_refutable: 0,
            #[cfg(feature = "tt_diag")]
            diag_boundary_and_total: 0,
            #[cfg(feature = "tt_diag")]
            diag_boundary_or_checks_sum: 0,
            #[cfg(feature = "tt_diag")]
            diag_pns_proof_ply: [0u64; 64],
            #[cfg(feature = "tt_diag")]
            diag_rem0_proof: 0,
            #[cfg(feature = "tt_diag")]
            diag_rem0_provisional: 0,
            pn_dn_snapshot: None,
            pn_dn_per_depth: Vec::new(),
            preserve_proven_tt: false,
            #[cfg(feature = "visit_diag")]
            visit_counts: FxHashMap::default(),
            #[cfg(feature = "visit_diag")]
            visit_first_ply: FxHashMap::default(),
            #[cfg(feature = "visit_diag")]
            visit_breakdown: FxHashMap::default(),
        }
    }

    /// solve() 呼び出し間で ProvenTT を引き継ぐフラグを設定する．
    /// true にすると solve() 冒頭で WorkingTT のみクリアし ProvenTT を保持する．
    pub fn set_preserve_proven_tt(&mut self, preserve: bool) -> &mut Self {
        self.preserve_proven_tt = preserve;
        self
    }

    /// 重複訪問レポートを返す (visit_diag feature 時のみ利用可)．
    ///
    /// 各局面の訪問回数分布と上位 `top_n` 局面をまとめた文字列を返す．
    #[cfg(feature = "visit_diag")]
    pub fn visit_summary(&self, top_n: usize) -> String {
        use std::fmt::Write as _;
        let total_visits: u64 = self.visit_counts.values().map(|&c| c as u64).sum();
        let unique_positions = self.visit_counts.len();
        let revisits = total_visits.saturating_sub(unique_positions as u64);
        let revisit_rate = if total_visits > 0 {
            100.0 * revisits as f64 / total_visits as f64
        } else {
            0.0
        };

        let mut sorted: Vec<(u64, u32, u8)> = self.visit_counts.iter()
            .map(|(&h, &cnt)| {
                let ply = self.visit_first_ply.get(&h).copied().unwrap_or(255);
                (h, cnt, ply)
            })
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // ply 別集計: (total_visits, unique_positions) per ply
        let mut ply_total = [0u64; 64];
        let mut ply_unique = [0u64; 64];
        for &(_, cnt, ply) in &sorted {
            if (ply as usize) < 64 {
                ply_total[ply as usize] += cnt as u64;
                ply_unique[ply as usize] += 1;
            }
        }

        let mut out = String::new();
        writeln!(out, "=== 重複訪問レポート ===").unwrap();
        writeln!(out, "総訪問数    : {}", total_visits).unwrap();
        writeln!(out, "ユニーク局面: {}", unique_positions).unwrap();
        writeln!(out, "重複訪問数  : {} ({:.1}%)", revisits, revisit_rate).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "--- ply 別集計 (revisit 率が高い順) ---").unwrap();
        let mut ply_rows: Vec<(usize, u64, u64)> = ply_total.iter().enumerate()
            .filter(|(_, &t)| t > 0)
            .map(|(p, &t)| {
                let u = ply_unique[p];
                (p, t, u)
            })
            .collect();
        ply_rows.sort_unstable_by(|a, b| {
            let ra = a.1.saturating_sub(a.2) * b.2;
            let rb = b.1.saturating_sub(b.2) * a.2;
            rb.cmp(&ra)
        });
        for (ply, total, unique) in &ply_rows {
            let rev = total.saturating_sub(*unique);
            let rate = if *total > 0 { 100.0 * rev as f64 / *total as f64 } else { 0.0 };
            writeln!(out, "  ply {:2}: total={:8}  unique={:7}  revisits={:7}  rate={:.1}%",
                ply, total, unique, rev, rate).unwrap();
        }
        writeln!(out).unwrap();

        writeln!(out, "--- 上位 {} 局面 (訪問回数順) ---", top_n).unwrap();
        for (rank, &(hash, cnt, ply)) in sorted.iter().take(top_n).enumerate() {
            writeln!(out, "  {:3}. hash={:#018x}  count={:6}  first_ply={}", rank + 1, hash, cnt, ply).unwrap();
        }

        // 訪問回数の分布ヒストグラム
        writeln!(out).unwrap();
        writeln!(out, "--- 訪問回数分布 ---").unwrap();
        let thresholds = [1u32, 2, 5, 10, 20, 50, 100, 500, 1000];
        let mut prev = 0u32;
        for &th in &thresholds {
            let count = sorted.iter().filter(|&&(_, c, _)| c >= prev + 1 && c <= th).count();
            writeln!(out, "  {}-{} 回: {} 局面", prev + 1, th, count).unwrap();
            prev = th;
        }
        let count_over = sorted.iter().filter(|&&(_, c, _)| c > prev).count();
        writeln!(out, "  {}+ 回: {} 局面", prev + 1, count_over).unwrap();

        out
    }

    /// 上位 `top_n` 局面の exit 種別内訳レポートを返す (visit_diag feature 時のみ利用可)．
    ///
    /// 重複訪問の原因を定量的に分解する:
    /// - `in_path`      : ループ検出 (完全除外)
    /// - `proven`       : TT proof/disproof 即 return
    /// - `threshold`    : TT 値が閾値超えで即 return
    ///   - `th_pn_def`  : その時 tt_pn == PN_UNIT (TT ミス/eviction 相当)
    ///   - `th_pn_inf`  : その時 tt_pn == INF
    ///   - `th_pn/dn`   : pn/dn どちらが原因か
    /// - `exploration`  : 実際に子展開
    ///   - `expl_pn_def`: その時 tt_pn == PN_UNIT (eviction 後の再探索)
    #[cfg(feature = "visit_diag")]
    pub fn hot_spot_summary(&self, top_n: usize) -> String {
        use std::fmt::Write as _;
        let mut sorted: Vec<(u64, u32, u8)> = self.visit_counts.iter()
            .map(|(&h, &c)| {
                let ply = self.visit_first_ply.get(&h).copied().unwrap_or(255);
                (h, c, ply)
            })
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let mut out = String::new();
        writeln!(out, "=== 上位 {} 局面の exit 種別内訳 ===", top_n).unwrap();
        // tt_miss = tt_dn==PN_UNIT (biased default シグナル = TT eviction)
        // tt_hit  = tt_dn!=PN_UNIT (WorkingTT に格納済みの値を読んだ)
        writeln!(out,
            "{:>4} {:>20}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}",
            "#", "hash", "total", "thresh",
            "expl", "tt_miss", "tt_hit", "miss%",
            "pn_1st", "dn_1st", "1stPly").unwrap();

        for (rank, &(hash, cnt, ply)) in sorted.iter().take(top_n).enumerate() {
            let bd = self.visit_breakdown.get(&hash)
                .cloned().unwrap_or_default();
            let miss_pct = if bd.exploration > 0 {
                100.0 * bd.expl_tt_miss as f64 / bd.exploration as f64
            } else { 0.0 };
            writeln!(out,
                "{:>4}. {:#018x}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>7.1}%  {:>8}  {:>8}  {:>6}",
                rank + 1, hash, cnt,
                bd.threshold_exit,
                bd.exploration, bd.expl_tt_miss, bd.expl_tt_hit, miss_pct,
                bd.expl_pn_first, bd.expl_dn_first,
                ply).unwrap();
        }
        out
    }

    /// PNS アリーナの最大ノード数を設定する (v0.25.0)．
    ///
    /// デフォルトは `PNS_MAX_ARENA_NODES` (5M)．大きくすると arena spin 率が
    /// 下がる代わりにメモリ消費が増える．`min_value` (1024) 未満は丸められる．
    pub fn set_pns_arena_max(&mut self, max_nodes: usize) {
        self.param_pns_arena_max = max_nodes.max(1024);
    }

    /// visit_history dominance check (KomoringHeights `IsSuperior` 相当) を有効化する．
    ///
    /// true で子展開時に経路上の祖先 hand が現 hand を支配する局面を
    /// 経路依存不詰として枝刈りする．chain aigoma で hand 多様性が爆発する
    /// 局面の枝刈り効果を期待．デフォルトは false．
    ///
    /// 詳細: [`DfPnSolver::param_use_visit_history_dominance`]．
    pub fn set_use_visit_history_dominance(&mut self, on: bool) {
        self.param_use_visit_history_dominance = on;
    }

    /// HandSet OR disproof 交集合演算 (KomoringHeights `HandSet::DisproofHandTag` 相当)
    /// を有効化する．
    ///
    /// true で，OR ノード全子反証時の disproof_hand を子 disproof_hand の
    /// 要素ごと min (交集合) で計算する．既存実装は `att_hand` 単一伝播のみ．
    /// デフォルトは false．
    ///
    /// 詳細: [`DfPnSolver::param_use_handset_combination`]．
    pub fn set_use_handset_combination(&mut self, on: bool) {
        self.param_use_handset_combination = on;
    }

    /// Tier 3 (twinkling-hatching-duckling, v0.65.0): KH 風 `DelayedMoveList` を
    /// 有効化/無効化する．
    ///
    /// 有効時，AND ノード multi-child loop で同マス合駒 chain を構築し，
    /// 「prev が未解決なら next を skip」する semantics を適用．
    /// 全合駒展開による pn 過大評価を抑止し，合駒可能局面のノード数削減を狙う．
    ///
    /// デフォルト OFF．関連: [`DfPnSolver::param_use_delayed_move_list`]．
    pub fn set_use_delayed_move_list(&mut self, on: bool) {
        self.param_use_delayed_move_list = on;
    }

    /// Phase B (twinkling-hatching-duckling, v0.66.0): path-aware DAG 補正を ON/OFF．
    ///
    /// 有効時，AND multi-child loop で DAG 合流子を sum 集約から除外し
    /// max のみで集約する (KH `double_count_elimination` 相当)．
    /// Plan B (v0.70.0): KH TCA inc_flag 機構の有効化．
    /// default false．有効化すると mid() loop で inc_flag を propagate し，
    /// 深い transposition chain での threshold extension を継続させる．
    pub fn set_use_kh_tca(&mut self, on: bool) -> &mut Self {
        self.param_use_kh_tca = on;
        self
    }

    /// 診断 (v0.71.1): periodic GC を無効化する．
    /// default false．catastrophic forgetting の検証用．
    /// 大規模 working TT で OOM の可能性があるので long-running 検証時注意．
    pub fn set_disable_periodic_gc(&mut self, on: bool) -> &mut Self {
        self.param_disable_periodic_gc = on;
        self
    }

    /// 診断 (v0.71.2): IDS の浅い depth 反復 (depth=2,4,6,...) を skip し，
    /// 最初から `saved_depth` full depth で MID を実行する．
    /// default false．`remaining` 違いによる TT 再評価コストを排除する検証用．
    pub fn set_skip_ids_shallow(&mut self, on: bool) -> &mut Self {
        self.param_skip_ids_shallow = on;
        self
    }

    /// 診断 (v0.71.0): per-ply mid() trace の有効化．
    /// default false．有効化すると mid() の OR/AND multi-child loop で
    /// `ply == param_trace_ply` のたびに root_trace_interval ノード経過ごとに
    /// per-iteration 状態を eprintln! でダンプする．
    pub fn set_root_trace(&mut self, on: bool, interval_nodes: u64) -> &mut Self {
        self.param_root_trace = on;
        if interval_nodes > 0 {
            self.root_trace_interval = interval_nodes;
        }
        self
    }

    /// 診断 (v0.71.3): trace 対象の ply を設定．`set_root_trace` と組合せて使用．
    /// default 0 (= root)．例: 1 で root child 入った直後 (defender's first move) を trace．
    pub fn set_trace_ply(&mut self, ply: u32) -> &mut Self {
        self.param_trace_ply = ply;
        self
    }

    /// 診断 (v0.71.4): trace 時に top-10 child のみでなく全 children を出力する．
    pub fn set_trace_full_children(&mut self, on: bool) -> &mut Self {
        self.param_trace_full_children = on;
        self
    }

    /// 診断 (v0.72.0): TT lookup hit rate を計測する．
    pub fn set_tt_lookup_diag(&mut self, on: bool) -> &mut Self {
        self.param_tt_lookup_diag = on;
        self
    }

    /// 診断 (v0.72.0): solve() 完了後に TT lookup 統計を取得．
    /// `(total, miss, proven, disproven, working)` のタプル．
    pub fn get_tt_lookup_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.diag_tt_lookups.get(),
            self.diag_tt_misses.get(),
            self.diag_tt_proven.get(),
            self.diag_tt_disproven.get(),
            self.diag_tt_working.get(),
        )
    }

    /// 診断 (v0.73.0, E): per-depth proven 蓄積ヒストグラム取得．
    /// 配列 `[u64; 64]` で index = ply, 値 = 当該 ply で proven 確定回数．
    pub fn get_proven_per_ply(&self) -> [u64; 64] {
        self.diag_proven_per_ply
    }

    /// Phase 21: deferred penalty 除数を設定 (0=無効, 8=KH 準拠)．
    pub fn set_deferred_penalty_denom(&mut self, denom: u32) -> &mut Self {
        self.param_deferred_penalty_denom = denom;
        self
    }

    /// Phase 22: deferred penalty `.max(1)` floor を設定 (KH=true)．
    pub fn set_deferred_penalty_floor(&mut self, floor: bool) -> &mut Self {
        self.param_deferred_penalty_floor = floor;
        self
    }

    /// Phase 22: 1+ε threshold epsilon を設定 (KH 1; PN_UNIT=16 試験用)．
    pub fn set_threshold_epsilon(&mut self, eps: u32) -> &mut Self {
        self.param_threshold_epsilon = eps.max(1);
        self
    }

    /// Phase 22: TCA extension formula を KH `max(thpn, pn+1)` clamp 式にするか．
    pub fn set_tca_kh_clamp(&mut self, on: bool) -> &mut Self {
        self.param_tca_kh_clamp = on;
        self
    }

    /// Phase 22: root level IDS (KH SearchEntry) を有効化．
    pub fn set_root_ids_enable(&mut self, on: bool) -> &mut Self {
        self.param_root_ids_enable = on;
        self
    }

    /// Phase 21: TCA gate モードを設定 (true=is_shallow, false=!is_first_visit)．
    pub fn set_tca_shallow_gate(&mut self, on: bool) -> &mut Self {
        self.param_tca_use_shallow_gate = on;
        self
    }

    /// Phase 21: mid_v2 visit count map を取得 (診断用)．
    pub fn get_visit_counts(&self) -> &rustc_hash::FxHashMap<u64, u32> {
        &self.mid_v2_visit_counts
    }

    /// Phase 21 診断: deferred penalty + TCA gate 統計を取得．
    /// `(deferred_frames, deferred_penalty_sum, tca_shallow_fire, tca_would_fire_old)`
    pub fn get_phase21_diag(&self) -> (u64, u64, u64, u64) {
        (
            self.diag_deferred_frames,
            self.diag_deferred_penalty_sum,
            self.diag_tca_shallow_fire,
            self.diag_tca_shallow_would_fire,
        )
    }

    /// 候補 G (v0.74.0): root child_pn_th の絶対 floor を設定．
    /// 0 で無効．推奨値: 100_000 (= 6250 * PN_UNIT)．
    /// ply=0 の OR child の pn 予算をこの値以上に引き上げ，深い commit を可能にする．
    pub fn set_root_child_pn_floor(&mut self, floor: u32) -> &mut Self {
        self.param_root_child_pn_floor = floor;
        self
    }

    /// 診断 (v0.76.0): TT の unique proven entry 数を取得．
    /// `get_proven_per_ply().sum()` (= total store 回数) と比較することで，
    /// proven entry が overwrite されていないかを確認できる．
    pub fn get_tt_proven_len(&self) -> usize {
        self.table.proven_len()
    }

    /// 診断 (v0.77.0): AND scan coverage 統計取得．
    /// `(visit_count, proven_sum, total_sum, zero_proven_count, full_proven_count)`
    /// - 平均 coverage = proven_sum / total_sum
    /// - zero_proven_count: proven=0 で exit した AND scan 回数 (proof 進捗なし)
    /// - full_proven_count: 全 children proven で exit (AND proven)
    pub fn get_and_coverage_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.diag_and_visit_count,
            self.diag_and_proven_sum,
            self.diag_and_total_sum,
            self.diag_and_zero_proven,
            self.diag_and_full_proven,
        )
    }

    /// 診断 (v0.78.0): OR scan coverage 統計取得．
    /// `(visit_count, proven_count_visits, total_sum)`
    pub fn get_or_coverage_stats(&self) -> (u64, u64, u64) {
        (
            self.diag_or_visit_count,
            self.diag_or_proven_count_visits,
            self.diag_or_total_sum,
        )
    }

    /// 診断 (v0.79.0, A): per-position visit count のヒストグラム取得．
    /// 戻り値: `(unique_positions, total_visits, capped, top_10_counts)`
    /// `top_10_counts[i]` = 上位 10 件の最多 visit 数 (降順)．
    pub fn get_pos_visit_stats(&self) -> (usize, u64, bool, Vec<u32>) {
        let unique = self.diag_pos_visits.len();
        let total: u64 = self.diag_pos_visits.values().map(|&v| v as u64).sum();
        let capped = self.diag_pos_visits_capped;
        let mut counts: Vec<u32> = self.diag_pos_visits.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.truncate(10);
        (unique, total, capped, counts)
    }

    /// 診断 (v0.79.0, A): visit count buckets でヒストグラム．
    /// `bucket[i]` = visit_count が `2^i` 以上 `2^(i+1)` 未満の position 数
    pub fn get_pos_visit_histogram(&self) -> [u64; 24] {
        let mut h = [0u64; 24];
        for &count in self.diag_pos_visits.values() {
            let bucket = if count == 0 { 0 } else {
                (32 - count.leading_zeros() - 1).min(23) as usize
            };
            h[bucket] += 1;
        }
        h
    }

    /// 候補 F (v0.75.0): OR ノード best_idx 選択の dn tie-break を有効化する．
    /// default false．true で `pn == best_pn` 同点時に `max(dn)` で tie-break．
    /// = defender の抵抗が強い attack を優先 → 探索 commit 強化．
    pub fn set_or_dn_tiebreak(&mut self, on: bool) -> &mut Self {
        self.param_or_dn_tiebreak = on;
        self
    }

    /// 候補 C (v0.80.0): AND node exhaustive defender prove を有効化．
    /// default false．true で AND multi-child loop の best_idx 選択を
    /// "全 unproven defender を順次 prove するため round-robin" に変更．
    pub fn set_use_exhaustive_and(&mut self, on: bool) -> &mut Self {
        self.param_use_exhaustive_and = on;
        self
    }

    /// 候補 D (v0.81.0): AND proven defender bitmap を有効化．
    /// default false．true で per-AND-position に proven defender index bitmap
    /// を持ち，proven 化済 defender を selection から確実に除外する．
    pub fn set_use_and_proven_bitmap(&mut self, on: bool) -> &mut Self {
        self.param_use_and_proven_bitmap = on;
        self
    }

    pub fn set_use_dag_correction(&mut self, on: bool) {
        self.param_use_dag_correction = on;
    }

    /// Phase C (twinkling-hatching-duckling, v0.67.0): per-move attack/defense
    /// support 差別化を ON/OFF．有効時 `edge_cost_or_with_support` を呼ぶ．
    pub fn set_use_per_move_support(&mut self, on: bool) {
        self.param_use_per_move_support = on;
    }

    /// melodic-cascading-otter (v0.69.0): full_hash ベースで parent_map を
    /// 辿って DAG を検出する版．KH `double_count_elimination` 風の挙動．
    ///
    /// runtime `parent_map` (child_full_hash → first-visit parent_full_hash) を
    /// 辿り，現 path 上の先祖 (immediate parent を除く) が現れたら
    /// transposition DAG と判定．
    ///
    /// `param_use_dag_correction = false` の場合は false．
    ///
    /// **2026-05-21 試行結果**: 29te root で nodes_searched が default と
    /// 完全一致．DAG 検出は動作するが sum_cpn 集約のみへの反映では探索が
    /// 変わらず効果なし．`MAX_DAG_LOOKBACK = 16` だと 500K+ で hang する
    /// 謎挙動あり (`MAX = 4` 推奨)．詳細: docs/plans/melodic-cascading-otter.md §9．
    pub(super) fn find_dag_ancestor_fh(&mut self, child_full_hash: u64) -> bool {
        if !self.param_use_dag_correction || self.path_len <= 1 {
            return false;
        }
        self.diag_dag_calls += 1;
        // v0.69.0 diagnostic: 16 ステップで explosion を確認．4 に短縮して再評価．
        const MAX_DAG_LOOKBACK: usize = 4;
        let immediate_parent = self.path[self.path_len - 1];
        let mut cur = child_full_hash;
        for step in 0..MAX_DAG_LOOKBACK {
            let parent = match self.parent_map.get(&cur) {
                Some(&p) => p,
                None => {
                    if step == 0 { self.diag_dag_short_none += 1; }
                    if (step as u32) > self.diag_dag_max_step {
                        self.diag_dag_max_step = step as u32;
                    }
                    return false;
                },
            };
            // step=0 で immediate_parent と一致 = 通常の初訪問．DAG ではない．
            if step == 0 && parent == immediate_parent {
                self.diag_dag_short_first += 1;
                return false;
            }
            // immediate parent 以外の path 上先祖と一致 → DAG．
            if parent != immediate_parent && self.path_set.contains(&parent) {
                self.diag_dag_true += 1;
                if (step as u32) > self.diag_dag_max_step {
                    self.diag_dag_max_step = step as u32;
                }
                return true;
            }
            // ループ防止 (parent_map が循環する場合)
            if parent == cur {
                if (step as u32) > self.diag_dag_max_step {
                    self.diag_dag_max_step = step as u32;
                }
                return false;
            }
            cur = parent;
        }
        self.diag_dag_walks_16 += 1;
        self.diag_dag_max_step = 16;
        false
    }

    /// **Phase 2a (swift-running-cheetah, v0.59.0)**: KH 風 flat array + linear
    /// probing `ProvenTable` の shadow-write を有効化する．
    ///
    /// `on=true` で，`store_proven` / `store_tagged_proof` / `store_refutable_disproof`
    /// の `vec.push` と同時に `ProvenTable::insert` を呼ぶ．既存 `proven_map`
    /// は primary store のまま (read path は変更なし)．Phase 2b で read path
    /// を切り替える予定．
    ///
    /// `capacity` は ProvenTable の初期 slot 数の目安．内部で `next_power_of_two`
    /// に切り上げ (最小 64)．例: 4M entries なら `1 << 22`．
    ///
    /// **注意**: 探索開始前 (`solve()` 呼出前) に設定すること．探索中の動的切替は未対応．
    ///
    /// 関連: `docs/plans/swift-running-cheetah.md`．
    pub fn set_use_kh_proven_tt(&mut self, on: bool, capacity: usize) {
        self.table.set_use_kh_proven_tt(on, capacity);
    }

    /// `ProvenTable` の `(len, proof_len, confirmed_len, refutable_len)` を返す．
    /// Phase 4a (v0.63.0): 常設化により非 Option．
    pub fn proven_table_stats(&self) -> (usize, usize, usize, usize) {
        self.table.proven_table_stats()
    }

    /// `proven_map` (primary store) の `(len, proof_len, confirmed_len, refutable_len)`
    /// を返す．Phase 2a-2 (v0.60.0) で `proven_table_stats()` と整合性を検証する
    /// 不変式テストに用いる．
    pub fn proven_map_counts(&self) -> (usize, usize, usize, usize) {
        let total = self.table.proven_len();
        let proof = self.table.proven_proof_len();
        let confirmed = self.table.proven_confirmed_len();
        let refutable = total.saturating_sub(proof).saturating_sub(confirmed);
        (total, proof, confirmed, refutable)
    }

    /// depth-limited disproof の WorkingTT 格納閾値を明示的に設定する (v0.25.0)．
    ///
    /// `remaining < threshold` の depth-limited disproof はスキップされる．
    /// `DISPROOF_THRESHOLD_ADAPTIVE` を渡すと adaptive モードに戻る．
    /// 有効値の目安: 0 (無効) / 2〜3 (深い問題向け) / 6 以上 (過剰)．
    ///
    /// **注意**: threshold > 0 は no-mate 証明 (NoCheckmate) の予算要求を
    /// 著しく増加させる場合がある．no-mate が期待される局面では default=0 を
    /// 使うか予算を増やすこと．
    pub fn set_disproof_remaining_threshold(&mut self, threshold: u16) {
        self.param_disproof_remaining_threshold = threshold;
        // adaptive センチネル以外なら即座に TT へ反映．adaptive の場合は
        // solve() 入口で outer_solve_depth に応じて計算・反映される．
        if threshold != DISPROOF_THRESHOLD_ADAPTIVE {
            self.table.set_disproof_remaining_threshold(threshold);
        }
    }

    /// retain_working_intermediates での pn/dn 上限を設定する (v0.25.7, Hypothesis 1C)．
    ///
    /// `u32::MAX` はキャップなし(デフォルト)．
    /// IDS 遷移後に保持された中間エントリの pn/dn を上限値でクリップし，
    /// 浅い探索での過大評価が深い探索の優先度を歪めるのを防ぐ．
    pub fn set_retain_pn_dn_cap(&mut self, cap: u32) {
        self.table.set_retain_pn_dn_cap(cap);
    }

    /// depth-adaptive disproof threshold を opt-in で有効化する (v0.25.1)．
    ///
    /// solve() 時に `outer_solve_depth` に基づいて閾値を自動決定する:
    /// - depth ≤ 21: 0 (スキップなし)
    /// - depth = 22:   2
    /// - depth ≥ 23 (ply 18 等): 3
    ///
    /// **注意**: adaptive は depth ≥ 23 の深い問題向けに threshold=3 を適用
    /// するため，no-mate 証明が期待される深い局面では退行しうる
    /// (`test_no_checkmate_counter_check` 等．default OFF とした理由)．
    /// 深い詰将棋だけを解く用途で明示的に有効化する．
    pub fn enable_adaptive_disproof_remaining_threshold(&mut self) {
        self.param_disproof_remaining_threshold = DISPROOF_THRESHOLD_ADAPTIVE;
        // solve() 入口で outer_solve_depth を見て TT に反映される．
    }

    /// Depth-adaptive な実効 disproof 格納閾値を返す (v0.25.1〜v0.27.2)．
    ///
    /// `param_disproof_remaining_threshold` が `DISPROOF_THRESHOLD_ADAPTIVE` の
    /// 場合，`outer_solve_depth` に基づいて閾値を自動決定する．
    ///
    /// **ポリシー (N-1 修正, v0.27.2)**:
    /// - depth ≤ 23: **1** (shallow〜中深度保護．depth=23 は chain aigoma 前の
    ///   ply 18 に対応し，threshold=3 は proven TT を 70x 破壊するため除外)
    /// - depth 24-27: **3** (chain aigoma sweet spot．ply 14 (depth=27) で実証)
    /// - depth ≥ 28: **1** (very deep．no-mate 証明の予算要求を保護．
    ///   `test_no_checkmate_counter_check` depth=31 で threshold=3 は
    ///   2M 予算不足 Unknown を引き起こすため，保守的に 1 に抑える)
    ///
    /// `param_` に非センチネル値が入っていればそれをそのまま使用 (テスト用)．
    #[inline(always)]
    pub(super) fn effective_disproof_remaining_threshold(&self) -> u16 {
        if self.param_disproof_remaining_threshold != DISPROOF_THRESHOLD_ADAPTIVE {
            return self.param_disproof_remaining_threshold;
        }
        let d = if self.outer_solve_depth > 0 {
            self.outer_solve_depth
        } else {
            self.depth
        };
        match d {
            0..=23 => 1,
            24..=27 => 3,
            _ => 1,
        }
    }

    /// 1+ε の実効 epsilon 除数を返す．
    ///
    /// `param_epsilon_denom` が `EPSILON_DENOM_ADAPTIVE` (0) の場合は
    /// `saved_depth_for_epsilon` に基づいて depth-adaptive に決定する．
    /// 正の値が設定されている場合はその値をそのまま返す(テスト用)．
    #[inline(always)]
    fn effective_eps_denom(&self) -> u32 {
        if self.param_epsilon_denom != EPSILON_DENOM_ADAPTIVE {
            self.param_epsilon_denom
        } else {
            let d = if self.saved_depth_for_epsilon > 0 {
                self.saved_depth_for_epsilon
            } else {
                self.depth
            };
            if d >= 19 { 2 } else { 3 }
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

    /// refutable check のパラメータを設定する (v0.24.76)．
    pub fn set_refutable_params(&mut self, depth: u32, call_limit: u32) -> &mut Self {
        self.param_refutable_depth = depth;
        self.param_refutable_call_limit = call_limit;
        self
    }

    /// PNS 初期フェーズ専用の refutable call limit を設定する (v0.55.13)．
    /// デフォルト 500．MID の param_refutable_call_limit (=10,000) とは独立．
    pub fn set_pns_refutable_call_limit(&mut self, limit: u32) -> &mut Self {
        self.param_pns_refutable_call_limit = limit;
        self
    }

    /// Hypothesis IDS-17 無効化検証用: saved_depth 20-26 での depth=16→17 挿入をスキップするか設定する．
    /// true にすると IDS-17 導入前の挙動 (depth=16 の次は saved_depth へ直接ジャンプ)．
    /// デフォルト false = IDS-17 有効．
    pub fn set_no_ids17(&mut self, enable: bool) -> &mut Self {
        self.param_no_ids17 = enable;
        self
    }

    /// **M-1 F1**: refutable check で全 check を必ず評価する (v0.25.4)．
    /// false 確定 check で早期 return せず，全 AND 子の disproof を store．
    pub fn set_refut_full_eval(&mut self, enable: bool) -> &mut Self {
        self.param_refut_full_eval = enable;
        self
    }

    /// **M-1 F2**: refutable check で fast path 部分 match を活用 (v0.25.4)．
    /// missing check のみを recursive で評価する．
    pub fn set_refut_partial_recursion(&mut self, enable: bool) -> &mut Self {
        self.param_refut_partial_recursion = enable;
        self
    }

    /// **M-1 F3**: OR レベル refutable 成功局面をキャッシュ (v0.25.4)．
    /// solve() 内のみで有効．clear() で IDS 透過汚染を防ぐ．
    pub fn set_refut_or_success_cache(&mut self, enable: bool) -> &mut Self {
        self.param_refut_or_success_cache = enable;
        self
    }

    /// **M-1 F4**: refutable fast path lookup を WorkingTT 含めに拡張 (v0.25.4)．
    /// depth-limited disproof (rem >= floor) も match として count．
    pub fn set_refut_extended_lookup(&mut self, enable: bool) -> &mut Self {
        self.param_refut_extended_lookup = enable;
        self
    }

    /// TT GC 閾値を設定する．
    ///
    /// TT のエントリ数がこの値を超えると GC を実行する．
    /// 0 にすると GC を無効化する．デフォルトは 0(GC 無効)．
    pub fn set_tt_gc_threshold(&mut self, v: usize) -> &mut Self {
        self.tt_gc_threshold = v;
        self
    }

    /// WorkingTT の pn/dn 分布を返す (分析用)．
    ///
    /// ノード制限到達時点のスナップショットがあればそれを返す．
    /// スナップショットがない場合は現在の WorkingTT を収集する．
    /// 返り値: (pn_hist, dn_hist, joint_hist)
    /// - pn_hist: pn 値の log2 ヒストグラム (32 バケット)
    /// - dn_hist: dn 値の log2 ヒストグラム (32 バケット)
    /// - joint_hist: (pn バケット, dn バケット) の 2D ヒストグラム (32×32 = 1024 要素)
    pub fn collect_pn_dn_dist(&self) -> ([u64; 32], [u64; 32], Vec<u64>) {
        if let Some((pn, dn, joint)) = &self.pn_dn_snapshot {
            (*pn, *dn, joint.clone())
        } else {
            self.table.collect_working_pn_dn_dist()
        }
    }

    /// IDS 各 depth 反復終了時点の WorkingTT pn/dn 分布スナップショット列を返す (分析用)．
    ///
    /// `mid_fallback` 内で各 IDS depth のMID 完了後，TT 遷移前に収集する．
    /// 返り値: `(ids_depth, nodes_searched, elapsed_secs, pn_hist, dn_hist, joint_hist)` のスライス
    pub fn collect_pn_dn_dist_per_depth(&self) -> &[(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)] {
        &self.pn_dn_per_depth
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
        // ProvenTT は v0.55.17 で FxHashMap 化されオーバーフローなし
        self.profile_stats.tt_proven_overflow_count = 0;
        self.profile_stats.tt_working_overflow_count = self.table.working_overflow_count;
        self.profile_stats.tt_overflow_no_victim_count =
            self.table.overflow_no_victim_count;
        self.profile_stats.tt_max_entries_per_position =
            self.table.max_entries_per_position;
        self.profile_stats.tt_proven_overflow_same_key_hist = [0; 9];
    }

    /// タイムアウトしたかどうかを返す．
    #[inline]
    pub(super) fn is_timed_out(&self) -> bool {
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
    /// 浅い ply は標準 df-pn と同じ `pn=1` を維持する．
    ///
    /// 本関数は探索ホットパスから呼ばれるため TT 参照は自クラスタのみとし，
    /// 近傍クラスタ走査は行わない(NPS 優先)．
    /// proof/disproof の hand\_gte 近傍走査は `has_proof`，`get_proof_hand` 等の
    /// 補助メソッド(±1 限定走査)に任せる．これらは child init や
    /// proof 伝播の際にのみ呼ばれるためホットパスへの影響が小さい．
    #[inline]
    pub(super) fn look_up_pn_dn(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u32) {
        let mut result = self.look_up_pn_dn_impl(pos_key, hand, remaining, true);
        // Phase 8 (v0.88.0) find_shortest フィルタ (legacy global state):
        // 旧来の API．Phase 9 以降の mid_v2 経路は look_up_pn_dn_md_bounded を使用．
        if let Some(max_md) = self.param_max_mate_distance {
            if result.0 == 0 {
                if let Some(stored_md) = self.table.look_up_mate_distance(pos_key, hand) {
                    if stored_md > max_md {
                        result = (PN_UNIT, PN_UNIT, 0);
                    }
                }
            }
        }
        if self.param_tt_lookup_diag {
            self.diag_tt_lookups.set(self.diag_tt_lookups.get() + 1);
            let (pn, dn, _) = result;
            if pn == PN_UNIT && dn == PN_UNIT {
                self.diag_tt_misses.set(self.diag_tt_misses.get() + 1);
            } else if pn == 0 {
                self.diag_tt_proven.set(self.diag_tt_proven.get() + 1);
            } else if dn == 0 {
                self.diag_tt_disproven.set(self.diag_tt_disproven.get() + 1);
            } else {
                self.diag_tt_working.set(self.diag_tt_working.get() + 1);
            }
        }
        result
    }


    /// Phase 9: md_budget 制約付き TT lookup (mid_v2 用).
    ///
    /// `md_budget` は「この局面から詰みまでに許される最大手数」．
    /// 例えば局面 P で `md_budget = B` なら P の proof は mate_distance <= B でなければ
    /// 有効とみなさない．KH `LookUp(len)` 相当．
    ///
    /// proven entry が見つかっても stored mate_distance > md_budget なら
    /// `(PN_UNIT, PN_UNIT, 0)` を返して mid_v2 に再探索を促す．
    ///
    /// Phase 13 (v0.92.0): md_budget が事実上無限大 (u16::MAX 周辺) の場合は
    /// mate_distance lookup を完全 skip して NPS 向上．
    #[inline]
    pub(super) fn look_up_pn_dn_md_bounded(
        &mut self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        md_budget: u16,
    ) -> (u32, u32, u32) {
        let mut result = self.look_up_pn_dn_impl(pos_key, hand, remaining, true);

        // Phase 23 (G2): KH `LookUpExact` 風 min_depth 即時更新．
        // KH `ttentry.hpp:497-498` 移植．store 時のみではなく LookUp 時にも
        // remaining が大きい (= shallower) ことを検出して `max_remaining_map`
        // を更新する．次回同 position が deeper ply で lookup されたとき
        // `is_shallow_remaining` が true を返し，TCA `inc_flag` 発火精度が向上する．
        // 旧 store-time-only 実装では同一局面が浅い ply で touched された情報が
        // 次の store まで反映されず，深い 29te で dominance signal の伝播が遅れていた．
        if remaining < REMAINING_INFINITE {
            let key = pos_key ^ self.hand_hash_for_map(hand);
            let entry = self.max_remaining_map.entry(key).or_insert(0);
            if remaining > *entry {
                *entry = remaining;
            }
        }

        // Phase 13: 大きな md_budget (e.g., u16::MAX - depth で saturating 減算後) では
        // 実用的に常に通過するので lookup overhead を skip．
        const MD_BUDGET_FILTER_THRESHOLD: u16 = 1000;
        if md_budget <= MD_BUDGET_FILTER_THRESHOLD && result.0 == 0 {
            if let Some(stored_md) = self.table.look_up_mate_distance(pos_key, hand) {
                if stored_md > md_budget {
                    // Phase 19: disproven_len check — budget 内 proof 不在が確認済みなら即 prune
                    if self.table.is_disproven_at_budget(pos_key, hand, md_budget) {
                        result = (INF, 0, 0);
                    } else {
                        // Phase 19: default (PN_UNIT, PN_UNIT) を返す．
                        // 旧 Phase 18 の working pn/dn fallback は初回 solve の値で
                        // 31手 proof 方向にガイドしてしまうため廃止．
                        result = (PN_UNIT, PN_UNIT, 0);
                    }
                }
            }
        }
        if self.param_tt_lookup_diag {
            self.diag_tt_lookups.set(self.diag_tt_lookups.get() + 1);
            let (pn, dn, _) = result;
            if pn == PN_UNIT && dn == PN_UNIT {
                self.diag_tt_misses.set(self.diag_tt_misses.get() + 1);
            } else if pn == 0 {
                self.diag_tt_proven.set(self.diag_tt_proven.get() + 1);
            } else if dn == 0 {
                self.diag_tt_disproven.set(self.diag_tt_disproven.get() + 1);
            } else {
                self.diag_tt_working.set(self.diag_tt_working.get() + 1);
            }
        }
        result
    }

    #[inline]
    fn look_up_pn_dn_impl(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        // remaining=0 (depth-limit 到達) で proof がなければ仮反証．
        // rem=0 disproof は TT に store しないため，動的に判定する．
        if remaining == 0 {
            if self.table.has_proof(pos_key, hand) {
                return (0, INF, 0);
            }
            return (INF, 0, 0);
        }
        // (v0.24.79) PNS 探索中 (skip_refutable_disproof=true) は ProvenTT
        // Pass 2 内で refutable disproof を直接スキップする版を使い，
        // disproof ヒット時のクラスタ走査を 1 回に削減する．以前は
        // look_up → is_refutable_disproof_at の 2 段スキャンだった．
        // ヒットしなかった場合は自然に (PN_UNIT, PN_UNIT, 0) が返る
        // (= 未探索ノード扱い) ため，arena-limited false NM を防止する．
        let result = if self.skip_refutable_disproof {
            self.table
                .look_up_skip_refutable(pos_key, hand, remaining, neighbor_scan)
        } else {
            self.table.look_up(pos_key, hand, remaining, neighbor_scan)
        };
        if result.0 == PN_UNIT && result.1 == PN_UNIT && result.2 == 0 {
            // TT ミス: Deep df-pn バイアスを適用(深い ply のみ)
            let ply = (self.depth as u32).saturating_sub(remaining as u32);
            let half_depth = self.depth / 2;
            if ply > half_depth {
                let biased_pn = PN_UNIT + (ply - half_depth) / self.param_deep_dfpn_r * PN_UNIT;
                (biased_pn, PN_UNIT, 0)
            } else {
                (PN_UNIT, PN_UNIT, 0)
            }
        } else {
            result
        }
    }

    /// パス上の祖先に ProvenTT の proof が存在するかチェックする．
    ///
    /// 存在すれば，現在の proof は祖先の証明に包含されるため
    /// ProvenTT への挿入は不要(探索の正確性には影響しない)．
    /// PV 復元時には WorkingTT の intermediate エントリを使用する．
    #[inline]
    fn ancestor_has_proof(&self) -> bool {
        // path[0..path_len-1] を逆順に遡る(直近の祖先から)
        // path_len-1 は自分自身なので除外
        if self.path_len < 2 { return false; }
        for i in (0..self.path_len - 1).rev() {
            if self.table.has_proof(self.path_pos_key[i], &self.path_hand[i]) {
                return true;
            }
        }
        false
    }

    /// Phase 20 (v0.99.0): max_remaining map を更新 (working entry のみ)．
    #[inline]
    fn update_max_remaining(&mut self, pos_key: u64, hand: &[u8; HAND_KINDS], pn: u32, dn: u32, remaining: u16) {
        if pn != 0 && dn != 0 {
            let key = pos_key ^ self.hand_hash_for_map(hand);
            let entry = self.max_remaining_map.entry(key).or_insert(0);
            if remaining > *entry {
                *entry = remaining;
            }
        }
    }

    /// Phase 20: max_remaining map の key hash．hand の簡易 hash．
    #[inline]
    fn hand_hash_for_map(&self, hand: &[u8; HAND_KINDS]) -> u64 {
        let mut h: u64 = 0;
        for (i, &v) in hand.iter().enumerate() {
            h ^= (v as u64).wrapping_mul(0x9e3779b97f4a7c15u64.wrapping_add((i as u64).wrapping_mul(0x517cc1b727220a95)));
        }
        h
    }

    /// Phase 20: pos_key+hand の max_remaining が current_remaining より大きいか判定．
    /// KH `min_depth < depth16` 相当: 位置が shallower ply で保存されたことを示す．
    pub(super) fn is_shallow_remaining(&self, pos_key: u64, hand: &[u8; HAND_KINDS], current_remaining: u16) -> bool {
        let key = pos_key ^ self.hand_hash_for_map(hand);
        if let Some(&max_rem) = self.max_remaining_map.get(&key) {
            max_rem > current_remaining
        } else {
            false
        }
    }

    /// visit_history dominance check (KomoringHeights `IsSuperior` 相当)．
    ///
    /// 子展開時に `(child_pos_key, child_hand)` がパス `path[0..path_len]` の
    /// **祖先** で支配されている場合 true を返す．支配の定義は
    /// 「同一 pos_key かつ祖先 hand が child hand を `hand_gte_forward_chain`
    /// で支配する」(=祖先のほうが攻め方持ち駒が多い)．
    ///
    /// **Soundness:** 攻め方が同じ局面に「より少ない (or 同じ) 持ち駒」で再訪
    /// するため，過去のより有利な状態でも詰めなかった事実から現在も詰まないと
    /// sound に推論できる．呼び出し側はこれを loop_child と同様に経路依存
    /// 反証 (path_dependent disproof) として扱うこと．
    ///
    /// **opt-in:** `param_use_visit_history_dominance == false` の場合は
    /// 常に false を返す (既存挙動と同等)．
    ///
    /// **注意:** identity (`hand` も完全一致) は既に `path_set.contains()` の
    /// ループ検出で捕捉済みなので，本関数の呼び出し側は通常 `path_set.contains`
    /// が false の子に対してのみ呼ぶ．本関数自体は identity も true を返すが，
    /// それは sound (祖先 hand == child hand も dominance に含まれる) で害は
    /// ない．
    #[inline]
    fn is_dominated_in_path(&self, child_pos_key: u64, child_hand: &[u8; HAND_KINDS]) -> bool {
        if !self.param_use_visit_history_dominance { return false; }
        if self.path_len == 0 { return false; }
        for i in 0..self.path_len {
            if self.path_pos_key[i] == child_pos_key
                && hand_gte_forward_chain(&self.path_hand[i], child_hand)
            {
                return true;
            }
        }
        false
    }

    /// GC 前に探索パス上のエントリを保護する．
    ///
    /// path 配列に記録されたルートからの全ノードの WorkingTT エントリの
    /// amount を最大値に引き上げ，GC のサンプリング閾値で除去されないようにする．
    fn mark_path_entries_for_gc_protection(&mut self) {
        for i in 0..self.path_len {
            self.table.protect_working_entry(
                self.path_pos_key[i],
                &self.path_hand[i],
            );
        }
        // root 自体も保護(path には含まれない場合がある)．
        // diag_root_pk は solve() 冒頭で必ず設定されるため，常に呼んでよい．
        self.table.protect_working_entry(
            self.diag_root_pk,
            &self.diag_root_hand,
        );
    }

    /// 転置表を更新する(位置キー＋持ち駒指定)．
    ///
    /// proof (pn=0) の場合，祖先に既に proof がある場合は格納自体をスキップする
    /// (祖先の証明に包含されるため)．table.store は pn==0 を store_proven に
    /// ルーティングするため，ProvenTT のみが影響を受け WorkingTT への
    /// 副作用はない．
    ///
    /// v0.24.72 の施策 α 不採用により，本関数は常に ABSOLUTE tag で proof を
    /// 格納する．`store_proof_with_tag` は dead code．disproof は通常の
    /// confirmed disproof として格納され，refutable disproof は
    /// `store_refutable_disproof` (v0.24.75) で別経路から格納される．
    #[inline]
    pub(super) fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
    ) {
        // proof で祖先に proof が既にある場合は ProvenTT をスキップ
        if pn == 0 && self.ancestor_has_proof() {
            return;
        }
        if pn == 0 {
            let ply = self.path_len.min(63);
            self.diag_proven_per_ply[ply] = self.diag_proven_per_ply[ply].saturating_add(1);
        }
        self.table.store(pos_key, hand, pn, dn, remaining, source);
        self.update_max_remaining(pos_key, &hand, pn, dn, remaining);
    }

    /// ベストムーブ付きで転置表を更新する．
    #[inline]
    pub(super) fn store_with_best_move(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        best_move: u16,
    ) {
        if pn == 0 && self.ancestor_has_proof() {
            return;
        }
        if pn == 0 {
            let ply = self.path_len.min(63);
            self.diag_proven_per_ply[ply] = self.diag_proven_per_ply[ply].saturating_add(1);
        }
        self.table.store_with_best_move(pos_key, hand, pn, dn, remaining, source, best_move);
        self.update_max_remaining(pos_key, &hand, pn, dn, remaining);
    }

    /// ベストムーブ + 詰み手数付きで転置表を更新する (proven entry 用)．
    ///
    /// `mate_distance` は PV 抽出の AND ノードで longest resistance 判定に
    /// 使用される．非 proven entry の場合は 0 を指定．
    #[inline]
    pub(super) fn store_with_best_move_and_distance(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        best_move: u16,
        mate_distance: u16,
    ) {
        if pn == 0 && self.ancestor_has_proof() {
            return;
        }
        if pn == 0 {
            let ply = self.path_len.min(63);
            self.diag_proven_per_ply[ply] = self.diag_proven_per_ply[ply].saturating_add(1);
        }
        self.table.store_with_best_move_and_distance(
            pos_key, hand, pn, dn, remaining, source, best_move, mate_distance,
        );
    }

    /// Tag 付き proof を転置表に格納する (v0.24.71)．
    ///
    /// **注意**: 施策 α が v0.24.72 で不採用確定したため，本関数の呼び出し
    /// 経路はすべて実行されない (filter_applied=false または
    /// child_tag!=ABSOLUTE の条件が満たされない)．dead code．
    /// refutable disproof 機構 (v0.24.75+) は別経路 (`store_refutable_disproof`)
    /// を使用するため本関数に依存しない．
    #[inline]
    pub(super) fn store_proof_with_tag(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        best_move: u16,
        mate_distance: u16,
        tag: u8,
    ) {
        if self.ancestor_has_proof() {
            return;
        }
        // 診断 (v0.73.0, E): proof 発見時の現在 ply (= path_len) を記録．
        let ply = self.path_len.min(63);
        self.diag_proven_per_ply[ply] = self.diag_proven_per_ply[ply].saturating_add(1);
        self.table.store_tagged_proof(
            pos_key, hand, best_move, mate_distance, tag, self.depth,
        );
    }

    /// TT Best Move を参照する(位置キー＋持ち駒指定)．
    #[inline]
    pub(super) fn look_up_best_move(
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
    pub(super) fn record_killer(&mut self, ply: u32, move16: u16) {
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
    pub(super) fn get_killers(&self, ply: u32) -> [u16; 2] {
        let p = ply as usize;
        if p < self.killer_table.len() {
            self.killer_table[p]
        } else {
            [0u16; 2]
        }
    }

    /// 経路依存フラグ付きで転置表を更新する(GHI 対策)．
    #[inline]
    pub(super) fn store_path_dep(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        path_dependent: bool,
    ) {
        self.table.store_path_dep(pos_key, hand, pn, dn, remaining, source, path_dependent);
    }

    /// 転置表を参照する(盤面から自動計算，事後クエリ用)．
    ///
    /// `remaining = 0` で全エントリを受け入れる．
    /// PV 抽出や結果確認など，探索外での参照に使用する．
    #[inline]
    pub(super) fn look_up_board(&self, board: &Board) -> (u32, u32) {
        let pk = position_key(board);
        let hand = &board.hand[self.attacker.index()];
        // self.depth を remaining として使用し，浅い IDS 反復の
        // 仮反証(NM)を最終結果に採用しないようにする．
        // remaining=0 だと全ての NM エントリを受け入れてしまい，
        // PNS の深さ制限内での仮反証が最終判定を汚染する．
        let (pn, dn, _source) = self.table.look_up(pk, hand, self.depth as u16, false);
        (pn, dn)
    }

    /// PV 復元用: 部分集合クラスタ走査を含む proof lookup．
    ///
    /// 通常の `look_up_board` では ProvenTT の hand_hash クラスタリングにより
    /// 証明駒と検索時の持ち駒が異なる場合に proof を見逃す．
    /// この関数は持ち駒の全部分集合クラスタを走査して proof を検出する．
    /// 探索本体では呼ばない(PV 復元のみ)．
    #[inline]
    pub(super) fn look_up_board_for_pv(&self, board: &Board) -> (u32, u32) {
        let pk = position_key(board);
        let hand = &board.hand[self.attacker.index()];
        let (pn, dn, _source) =
            self.table.look_up_proven_subset(pk, hand, self.depth as u16);
        if pn == 0 || dn == 0 {
            return (pn, dn);
        }
        // ProvenTT subset で見つからない場合は WorkingTT もチェック
        let (pn, dn, _source) = self.table.look_up(pk, hand, self.depth as u16, true);
        (pn, dn)
    }

    /// 転置表を更新する(盤面から自動計算)．
    #[inline]
    pub(super) fn store_board(
        &mut self,
        board: &Board,
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
    ) {
        if pn == 0 && self.ancestor_has_proof() { return; }
        let pk = position_key(board);
        let hand = board.hand[self.attacker.index()];
        self.table.store(pk, hand, pn, dn, remaining, source);
    }

    /// 証明駒/反証駒を指定して TT に格納する．
    pub(super) fn store_board_with_hand(
        &mut self,
        board: &Board,
        hand: &[u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
    ) {
        if pn == 0 && self.ancestor_has_proof() { return; }
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
        if self.preserve_proven_tt {
            self.table.clear_working_only();
        } else {
            self.table.clear();
        }
        self.nodes_searched = 0;
        self.diag_km_calls = 0;
        self.diag_km_hits = 0;
        self.diag_km_total_refuted = 0;
        self.pn_dn_per_depth.clear();
        self.max_ply = 0;
        self.ply_nodes = [0; 64];
        self.path_len = 0;
        self.path_set.clear();
        // melodic-cascading-otter (v0.69.0): parent_map を solve() 開始時にクリア．
        // (DAG 検出は同 solve() 内に閉じる．前回 solve() の TT 残存とは独立)．
        self.parent_map.clear();
        self.diag_dag_calls = 0;
        self.diag_dag_true = 0;
        self.diag_dag_short_first = 0;
        self.diag_dag_short_none = 0;
        self.diag_dag_max_step = 0;
        self.diag_dag_walks_16 = 0;
        // Plan B: KH TCA inc_flag は solve() 開始時に 0．
        self.inc_flag = 0;
        self.diag_tca_increments = 0;
        self.diag_tca_decrements = 0;
        self.diag_tca_extends = 0;
        self.root_trace_next = if self.param_root_trace { self.root_trace_interval } else { u64::MAX };
        self.root_trace_iter = 0;
        self.diag_tt_lookups.set(0);
        self.diag_tt_misses.set(0);
        self.diag_tt_proven.set(0);
        self.diag_tt_disproven.set(0);
        self.diag_tt_working.set(0);
        self.diag_proven_per_ply = [0u64; 64];
        self.diag_and_visit_count = 0;
        self.diag_and_proven_sum = 0;
        self.diag_and_total_sum = 0;
        self.diag_and_zero_proven = 0;
        self.diag_and_full_proven = 0;
        self.diag_or_visit_count = 0;
        self.diag_or_proven_count_visits = 0;
        self.diag_or_total_sum = 0;
        self.diag_pos_visits.clear();
        self.diag_pos_visits_capped = false;
        self.exhaustive_and_rr_counter = 0;
        self.and_proven_bitmap.clear();
        self.killer_table.clear();
        self.check_cache.clear();
        self.refutable_check_failed.clear();
        // F3: OR success cache を solve() 開始時にクリア (prev solve() の汚染回避)．
        self.refutable_check_succeeded.clear();
        self.pc_summary.clear();
        self.start_time = Instant::now();
        self.timed_out = false;
        self.next_gc_check = 100_000;
        self.attacker = board.turn;
        self.outer_solve_depth = self.depth;
        // (v0.25.1) adaptive disproof threshold: outer_solve_depth 決定後に
        // effective 値を TT へ反映する．param が非センチネルなら no-op．
        {
            let eff = self.effective_disproof_remaining_threshold();
            self.table.set_disproof_remaining_threshold(eff);
        }
        self.alpha_x_filter_active = false;
        self.prev_attacker_move = Move(0);
        /// 施策 A-6 再評価: 境界層 PNS 責任転嫁の呼出数グローバル上限．
        /// 10 回 × 5K ノード/回 = 50K ノード相当の追加予算 (solve の小さな一部)．
        const A6_BOUNDARY_PNS_MAX_CALLS: u32 = 10;
        self.a6_boundary_pns_calls_remaining = A6_BOUNDARY_PNS_MAX_CALLS;
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
        // root の pos_key / hand を GC 保護用に記録する．
        // diag_root_pk が 0 のまま GC が起動すると protect_working_entry が
        // 空振りし，root entry が evict されて探索が崩壊する．
        self.diag_root_pk = pk;
        self.diag_root_hand = att_hand;
        let (root_pn_after_pns, root_dn_after_pns, _) =
            self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
        // PNS で未解決 + 残り予算あり → MID フォールバック
        if root_pn_after_pns != 0 && root_dn_after_pns != 0
            && self.nodes_searched < self.max_nodes
            && !self.timed_out
        {
            let (rp, rd) = self.look_up_board(board);
            if rp != 0 && rd != 0
                && self.nodes_searched < self.max_nodes
                && !self.timed_out
            {
                verbose_eprintln!("[solve] MID fallback start: nodes={}", self.nodes_searched);
                self.mid_fallback(board);
                verbose_eprintln!("[solve] MID fallback end: nodes={} time={:.1}s",
                    self.nodes_searched, self.start_time.elapsed().as_secs_f64());
            }
        }

        let (root_pn, root_dn) = self.look_up_board(board);
        verbose_eprintln!("[solve] root_pn={} root_dn={} nodes={}", root_pn, root_dn, self.nodes_searched);
        if root_pn == 0 {
            // PNS アリーナから PV を抽出できた場合はそちらを優先
            // (TT ベースの extract_pv は PNS 証明パスが不完全になりうる)
            //
            // PV 抽出 visit 予算: 10M に設定する．
            // effective length 比較 (count_useless_interpose_pairs 経由) のために
            // 全 AND 子を評価する必要があり，旧 100K 予算では深い AND iteration
            // が途中で打ち切られて canonical PV を見逃すことがあった．
            //
            // コスト特性 (v0.24.29, TT-distance fast path 廃止後):
            // - 短い詰み (typical 3-15 手): proof tree が小さいため visit 数
            //   は数千〜数万程度，予算の 0.1% も消費しない
            // - 中深度 (20-30 手): 数十万〜数百万 visits，全体の数% のコスト
            // - 深い aigoma (39手詰め ply 22〜24): 数百万 visits．具体的には
            //   `test_tsume_39te_ply24_mate15_regression` (1M ノード探索予算)
            //   で PV 抽出 ~40s 程度
            //
            // 実測 (v0.24.27 → v0.24.29, fast path 廃止，backward_120m):
            // | Ply  | v0.24.27 (fast path) | v0.24.29 (slow only) |    Δ |
            // |------|----------------------|----------------------|------|
            // | 24   |              41.18s  |              41.10s  |  ~0% |
            // | **22** |            **418.91s**  |            **398.44s**  | **-4.9%** |
            // |   (Mate(17)) |              |                      |      |
            // | 38-26 |               < 0.5s |               < 0.5s |  noise |
            //
            // 重要な発見: 深い aigoma (ply 22) で v0.24.29 の方が **速い**．
            // fast path は「全 child TT lookup + distance 取得 → 失敗なら
            // slow path」という 2 パス構造だったため，fire しない場合は
            // 事前 TT lookup が純粋なオーバーヘッドになっていた．
            // chain drop が proven になる確率が低い deep aigoma 問題では
            // fast path がほぼ fire せず，廃止が高速化に寄与する．
            //
            // 浅い詰み (typical <15 手) では full dfpn suite で 140s → 156s
            // (+11%) の軽微な regression がある．これは fast path が fire
            // していた浅いケースの単純化コスト．
            //
            // ply ≤ 20 は visit 予算 10M を超える可能性があるが，そこまで
            // 深い探索では **探索側のノード予算が先に尽きる**ので PV 抽出
            // が律速要因になることはない．ply 20 は v0.24.27 でも 120M
            // ノード予算 + 1800s timeout で Unknown だった．
            // PV 抽出の visit 予算．探索ノード数に応じてスケーリングする．
            //
            // 固定 10M visits では深い探索(97M+ ノード)後に AND ノードの
            // 全 defender 評価で予算超過し MateNoPV が発生する．
            // 探索ノード数の 1/4 を基準に 10M〜50M の範囲で動的に設定する．
            // 上限 50M は PV 抽出の過大な時間消費を防止する(24M budget で
            // PV 抽出に 11 分+ かかるケースを回避)．
            let pv_visit_budget: u64 = (self.nodes_searched / 4).max(10_000_000).min(50_000_000);

            // PV 候補とその抽出が完全だったかを集める．
            // pv_extraction_incomplete フラグを extract_pv_limited 呼び出し
            // 後に確認し，最終的な Checkmate 判定で使う．
            let (final_pv, pv_complete) = if let Some(pv) = pns_pv {
                if self.find_shortest {
                    // 最短手数探索: PV 長を depth 上限にして追加証明
                    let saved_depth = self.depth;
                    self.depth = pv.len() as u32;
                    self.complete_or_proofs(board);
                    self.depth = saved_depth;
                    let final_moves = self.extract_pv_limited(board, pv_visit_budget);
                    let extraction_complete = !self.pv_extraction_incomplete;
                    let moves = if !final_moves.is_empty()
                        && final_moves.len() <= pv.len()
                    {
                        final_moves
                    } else {
                        pv
                    };
                    (moves, extraction_complete)
                } else {
                    // pns_extract_pv が直接返した PV (extract_pv_limited
                    // 経由ではない) は incomplete フラグが立たない．
                    // arena traversal で全子評価できるなら完全とみなす．
                    (pv, true)
                }
            } else {
                // アリーナ PV が取れなかった場合は TT ベースにフォールバック
                self.complete_or_proofs(board);

                let moves = self.extract_pv_limited(board, pv_visit_budget);
                let extraction_complete = !self.pv_extraction_incomplete;
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
                    let final_moves = self.extract_pv_limited(board, pv_visit_budget);
                    let final_complete = extraction_complete && !self.pv_extraction_incomplete;
                    let moves = if !final_moves.is_empty()
                        && final_moves.len() <= moves.len()
                    {
                        final_moves
                    } else {
                        moves
                    };
                    (moves, final_complete)
                } else {
                    (moves, extraction_complete)
                }
            };

            // PV 抽出が AND ノードで全 defender を評価し切れなかった場合，
            // 表示される PV が真の longest resistance である保証がないため，
            // soundness を優先して CheckmateNoPv を返す．
            if !pv_complete {
                return TsumeResult::CheckmateNoPv {
                    nodes_searched: self.nodes_searched,
                };
            }
            TsumeResult::Checkmate {
                moves: final_pv,
                nodes_searched: self.nodes_searched,
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

    /// ハイブリッド NM 昇格判定 (PNS pns_expand 用).
    ///
    /// 1. `all_checks_refutable_by_tt` で TT ベース高速判定 (~2µs/王手)
    /// 2. 失敗時は `refutable_check_failed` キャッシュを確認
    /// 3. 未キャッシュなら `depth_limit_all_checks_refutable` にフォールバック．
    ///    false の場合は pos_key をキャッシュに記録し，同一局面での
    ///    再評価を省略する．
    ///
    /// 設計意図: 39手詰め ply 20 で `depth_limit_all_checks_refutable` が
    /// PNS 時間の 99.97% を消費する律速要因だった (1,723 invocations 中
    /// 1,721 が REFUTABLE_CALL_LIMIT=10,000 に到達)．一方 29 手詰め等の
    /// 浅い問題では recursive 版が実際に NM 昇格を発生させる．
    /// このハイブリッド化で:
    ///   - 共通ケース (TT ヒット) は ~2µs
    ///   - 再帰が必要な局面は初回のみ ~66ms まで支払い，memoize で再評価回避
    ///   - 29 手詰めなど浅い問題の NM 昇格率を維持
    #[inline]
    /// **F2**: fast path 部分 match 用 helper (v0.25.4)．
    ///
    /// 全 check に対して TT lookup し，どの check が disproof として存在
    /// するかをビットマスクで返す．`checks.len() <= 64` を仮定
    /// (typical 5-15)．N 個全 match で `(1<<N)-1`．
    fn refutable_partial_match_mask(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> u64 {
        let extended = self.param_refut_extended_lookup;
        let min_remaining: u16 = if extended {
            self.effective_refutable_depth() as u16
        } else { 0 };
        let mut mask: u64 = 0;
        for (i, check) in checks.iter().enumerate().take(64) {
            let captured = board.do_move(*check);
            let pk = position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            let found = if extended {
                if self.table.has_refutable_or_confirmed_disproof(pk, &att_hand) {
                    true
                } else {
                    let (pn, dn, _) = self.table.look_up_working(
                        pk, &att_hand, min_remaining, false);
                    pn != 0 && dn == 0
                }
            } else {
                self.table.has_refutable_or_confirmed_disproof(pk, &att_hand)
            };
            board.undo_move(*check, captured);
            if found {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    pub(super) fn refutable_check_with_cache(
        &mut self,
        board: &mut Board,
        _pos_key: u64,
        checks: &[Move],
    ) -> bool {
        // F3: OR レベル success cache (有効時)
        // board.hash は pos_key + hand を含む full hash．
        // pos_key のみだと同一盤面・異 hand で false positive が起きるため
        // full_hash を使う (v0.25.5 ply 22 退行の root cause fix)．
        let full_hash = board.hash;
        if self.param_refut_or_success_cache
            && self.refutable_check_succeeded.contains(&full_hash)
        {
            #[cfg(feature = "verbose")]
            { self.dbg_refut_tt_hits += 1; }
            return true;
        }
        // Fast path: TT ベース判定 (confirmed + refutable disproof の両方を参照)
        if self.all_checks_refutable_by_tt_or_refutable(board, checks) {
            #[cfg(feature = "verbose")]
            { self.dbg_refut_tt_hits += 1; }
            // F3: 成功時に OR success cache へ記録
            if self.param_refut_or_success_cache {
                self.refutable_check_succeeded.insert(full_hash);
            }
            return true;
        }
        // Memoize: 既に false 確定の局面ならスキップ
        if self.refutable_check_failed.contains(&full_hash) {
            #[cfg(feature = "verbose")]
            { self.dbg_refut_memo_hits += 1; }
            return false;
        }
        // F2: partial fast path → missing checks のみ recursive で評価
        // matched 個数が 0 なら通常 recursive と同じ．matched > 0 のとき
        // unmatched checks のみで recursive_inner を走らせる．
        if self.param_refut_partial_recursion && checks.len() <= 64 && !checks.is_empty() {
            let mask = self.refutable_partial_match_mask(board, checks);
            let full_mask = if checks.len() == 64 {
                u64::MAX
            } else {
                (1u64 << checks.len()) - 1
            };
            if mask == full_mask {
                // 通常 fast path で既に取れているはず．万一の race condition で
                // ここに来た場合も成功扱い．
                #[cfg(feature = "verbose")]
                { self.dbg_refut_tt_hits += 1; }
                if self.param_refut_or_success_cache {
                    self.refutable_check_succeeded.insert(full_hash);
                }
                return true;
            }
            if mask != 0 {
                // 部分 match: unmatched のみ集める (typical N <= 16)
                let mut unmatched: ArrayVec<Move, 64> = ArrayVec::new();
                for (i, &c) in checks.iter().enumerate().take(64) {
                    if (mask >> i) & 1 == 0 {
                        unmatched.push(c);
                    }
                }
                // 残りの check が全て recursive で refute 可能か判定
                // (matched 部分は TT で確認済み)
                let result = self.depth_limit_all_checks_refutable(board, &unmatched);
                if result {
                    #[cfg(feature = "verbose")]
                    { self.dbg_refut_recursive_true += 1; }
                    if self.param_refut_or_success_cache {
                        self.refutable_check_succeeded.insert(full_hash);
                    }
                    return true;
                } else {
                    self.refutable_check_failed.insert(full_hash);
                    #[cfg(feature = "verbose")]
                    { self.dbg_refut_recursive_false += 1; }
                    return false;
                }
            }
            // mask == 0 なら通常 recursive へフォールスルー
        }
        // Fallback: 再帰判定 (高コスト). false ならキャッシュ．
        // true の場合は AND ノードが再帰内で ProvenTT に NM 格納済み (v0.24.74)．
        // OR ノードは ProvenTT に格納しない — PNS の backprop で
        // root_dn=0 の false NM を引き起こすため (v0.24.74 診断結果)．
        let result = self.depth_limit_all_checks_refutable(board, checks);
        if !result {
            self.refutable_check_failed.insert(full_hash);
            #[cfg(feature = "verbose")]
            { self.dbg_refut_recursive_false += 1; }
        } else {
            #[cfg(feature = "verbose")]
            { self.dbg_refut_recursive_true += 1; }
            // F3: recursive 成功時にも OR success cache へ記録
            if self.param_refut_or_success_cache {
                self.refutable_check_succeeded.insert(full_hash);
            }
        }
        result
    }

    /// 深さ制限 OR ノードの再帰的 NM 判定 (IDS の構造的不詰検証)．
    ///
    /// 全王手に対して玉方に「応手後に王手なし」または「応手後の王手が
    /// さらに反証可能」となる逃げ手が存在するかを再帰的に確認する．
    /// 全ての王手でそれが成立すれば真の不詰 (REMAINING_INFINITE) として扱える．
    /// 再帰深さは固定値 5 で制限し，分岐爆発を防止する．
    /// 呼び出し回数上限 (`REFUTABLE_CALL_LIMIT`) を超えた場合は安全側に倒して
    /// false (未証明) を返す．
    pub(super) fn depth_limit_all_checks_refutable(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        let limit = if self.in_initial_pns_phase {
            self.param_pns_refutable_call_limit
        } else {
            self.param_refutable_call_limit
        };
        let mut calls: u32 = 0;
        self.all_checks_refutable_recursive(
            board, checks, self.effective_refutable_depth(), &mut calls, limit,
        )
    }

    /// IDS target depth に応じた適応的 refutable check 再帰深さ (v0.24.77, v0.25.2 updated)．
    ///
    /// `param_refutable_depth` が 0 (EFFECTIVE_DEPTH_ADAPTIVE) の場合，
    /// outer_solve_depth (IDS target) に基づく log ベースの値を返す．
    ///
    /// 設計意図 (ユーザー提案):
    /// - self.depth (現 IDS step) ではなく outer_solve_depth (target) を使用
    ///   → IDS 中間 step 全体で d が一貫し，TT 状態の不整合を回避
    /// - log ベースで徐々に増加して飽和
    ///   → 浅い問題では小さい d (NPS 優先)
    ///   → 深い問題では大きい d (NM 検出率優先)
    ///
    /// **M-A (v0.25.2)**: target ≥ 20 に **下限フロア 8** を追加．
    /// §10.2.9 ply 20 false-NoMate 診断で，target=21 の log-adaptive d=5 では
    /// refutable check の判定が浅すぎて非 NM 局面を refutable disproof と誤判定
    /// する現象を確認．固定 depth=10 相当の対策を組み込む．
    ///
    /// **N-2 (v0.26.0)**: target 24-31 の固定フロア 8 を 6 に緩和し
    /// NPS -28% (M-A による) の一部を回収する．
    /// - target=1-19:  depth_floor=3 → d=3〜5
    /// - target=20-23: depth_floor=**8** (M-A 維持)
    /// - target=24-31: depth_floor=**6** (N-2 緩和)
    /// - target=32+:   depth_floor=3 → log_val(≥6) が dominates
    ///
    /// **O-1 (v0.55.19)**: target ≥ 32 の depth_floor を 3→8 に戻す．
    /// N-2 で target=32+ の floor を緩和しなかった (floor=3, log_val が支配 → d=6)
    /// が，39 手詰め問題 (target=35) で ProvenTT 共有時に偽 confirmed disproof が
    /// 発生する現象を確認 (ply 7 防御手プリソルブで 2c2b が 138K→500M に退行)．
    /// M-A と同様の根本原因 (d=6 が浅すぎて非 NM 局面を refutable disproof と誤判定)
    /// のため floor=8 に統一する．
    /// - target=32+: depth_floor=**8** (O-1 修正)
    ///
    /// 式: d = max(target.ilog2() + 1, depth_floor(target)).min(10)
    #[inline]
    fn effective_refutable_depth(&self) -> u32 {
        if self.param_refutable_depth != Self::EFFECTIVE_DEPTH_ADAPTIVE {
            return self.param_refutable_depth;
        }
        let target = if self.outer_solve_depth > 0 {
            self.outer_solve_depth
        } else {
            self.depth
        };
        if target == 0 {
            return 3;
        }
        let log_val = (target as u32).ilog2() + 1;
        // P-1 (v0.55.20): target≥32 の floor は 3 に戻す．
        // O-1 (v0.55.19) で floor=8 にしたが，d=8 が ProvenTT を 551K→GC まで膨張させ
        // NPS を低下させた．根本原因は floor ではなく all_checks_refutable_by_tt が
        // refutable disproof を REMAINING_INFINITE 昇格に使用していたことであり，
        // P-1 で look_up_skip_refutable に変更して修正済み．
        let depth_floor: u32 = match target {
            0..=23 => if target >= 20 { 8 } else { 3 },
            24..=31 => 6, // N-2 (v0.26.0): floor 緩和
            _ => 3,       // P-1 (v0.55.20): O-1 revert (floor=8 は bloat の原因)
        };
        log_val.max(depth_floor).min(10)
    }

    /// `param_refutable_depth = 0` は適応的 depth を意味する sentinel．
    const EFFECTIVE_DEPTH_ADAPTIVE: u32 = 0;

    /// TT ベースの NM 昇格判定(MID depth boundary 用)．
    ///
    /// 各王手後の AND ノードが ProvenTT に **confirmed** disproof として格納
    /// されているかを確認する．`look_up_skip_refutable` を使用するため
    /// refutable disproof (深さ制限付き) は無視される．
    ///
    /// # P-1 (v0.55.20): refutable disproof を REMAINING_INFINITE 昇格に使わない
    ///
    /// `look_up` (skip_refutable=false) を使うと，浅い d で生成された
    /// refutable disproof が `all_checks_refutable_by_tt=true` を引き起こし，
    /// 深さ制限ファストパスで REMAINING_INFINITE confirmed disproof が格納される．
    /// これが ProvenTT 汚染 (偽不詰) の根本原因だった．confirmed disproof のみを
    /// 参照することで，REMAINING_INFINITE 昇格の根拠を絶対知識に限定する．
    /// K-M 有効なら `(true, Some(cycle_root))`，絶対反証のみなら `(true, None)`，
    /// いずれかの check が未反証なら `(false, _)` を返す．
    pub(super) fn all_checks_refutable_by_tt(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> (bool, Option<u32>) {
        self.diag_km_calls += 1;
        let mut km_cycle_root: Option<u32> = None;
        let mut km_used = false;
        for check in checks {
            let captured = board.do_move(*check);
            let pk = position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            // P-1 (v0.55.20): confirmed disproof のみ参照する
            let (_, dn, _) = self.table.look_up_skip_refutable(pk, &att_hand, REMAINING_INFINITE, false);
            let refuted = if dn == 0 {
                true
            } else if let Some(cr) = self.table.get_km_disproof_cycle_root(
                pk, &att_hand, &self.path[..self.path_len],
            ) {
                // K-M (v0.55.23): cycle_root ∈ 現在パスなら path_dep 反証も有効
                km_cycle_root = Some(match km_cycle_root {
                    None => cr,
                    Some(prev) if prev == cr => cr,
                    _ => 0,
                });
                km_used = true;
                true
            } else {
                false
            };
            board.undo_move(*check, captured);
            if !refuted {
                return (false, None);
            }
        }
        if km_used { self.diag_km_hits += 1; }
        self.diag_km_total_refuted += 1;
        (true, km_cycle_root)
    }

    /// TT ベースの NM 昇格判定(PNS refutable check fast path 用, v0.24.75)．
    ///
    /// `all_checks_refutable_by_tt` と同様だが，ProvenTT の refutable
    /// disproof (再帰判定で格納されたもの) も含めて参照する．
    /// PNS の `refutable_check_with_cache` の高速パスとして使用する．
    fn all_checks_refutable_by_tt_or_refutable(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        #[cfg(feature = "verbose")]
        {
            self.dbg_refut_fast_attempts += 1;
            self.dbg_refut_fast_check_total += checks.len() as u64;
        }
        #[cfg(feature = "verbose")]
        let mut matched_count: u64 = 0;
        // F4: extended lookup を有効化すると WorkingTT も含めて lookup する．
        // depth-limited disproof は remaining が refutable depth 以上のものだけ
        // 安全 (refutable check が要求する深さを満たす)．
        let extended = self.param_refut_extended_lookup;
        let min_remaining: u16 = if extended {
            // refutable check 自体が探索する深さと同等以上の disproof のみ
            // を信頼．M-A の effective_refutable_depth と整合させる．
            self.effective_refutable_depth() as u16
        } else { 0 };
        for check in checks {
            let captured = board.do_move(*check);
            let pk = position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            let found = if extended {
                // F4: ProvenTT (refutable+confirmed) + WorkingTT (rem>=floor)
                if self.table.has_refutable_or_confirmed_disproof(pk, &att_hand) {
                    true
                } else {
                    let (pn, dn, _) = self.table.look_up_working(
                        pk, &att_hand, min_remaining, false);
                    pn != 0 && dn == 0
                }
            } else {
                self.table.has_refutable_or_confirmed_disproof(pk, &att_hand)
            };
            board.undo_move(*check, captured);
            if !found {
                #[cfg(feature = "verbose")]
                {
                    self.dbg_refut_fast_match_total += matched_count;
                    if matched_count > 0 {
                        self.dbg_refut_fast_partial += 1;
                    }
                }
                return false;
            }
            #[cfg(feature = "verbose")]
            { matched_count += 1; }
        }
        #[cfg(feature = "verbose")]
        {
            self.dbg_refut_fast_match_total += matched_count;
            self.dbg_refut_fast_partial += 1; // full match also counts
        }
        true
    }

    /// デフォルトの refutable check 呼び出し回数上限．
    const DEFAULT_REFUTABLE_CALL_LIMIT: u32 = 10_000;
    /// デフォルトの refutable check 再帰深さ (0 = 適応的，self.depth に基づく)．
    const DEFAULT_REFUTABLE_DEPTH: u32 = 0;

    /// `depth_limit_all_checks_refutable` の再帰本体．
    ///
    /// `depth` は残りの再帰深さ(0 で打ち切り)．各再帰レベルで
    /// 王手→応手→次の王手 を確認し，最大 `depth` 段階まで追跡する．
    /// `calls` は呼び出し回数カウンタで，`limit` 超過時は false を返す．
    ///
    /// 各王手の反証が成功した場合，AND ノードを ProvenTT に refutable
    /// disproof として格納する (v0.24.75)．TT レベルでは confirmed と
    /// 区別せず，`look_up_proven` / `all_checks_refutable_by_tt` から
    /// 可視．PNS 経路のみ `look_up_proven_skip_refutable` が bit 7 を
    /// 見て読み飛ばすことで arena-limited false NM を防止しつつ，
    /// 再帰判定結果を TT に蓄積する．
    pub(super) fn all_checks_refutable_recursive(
        &mut self,
        board: &mut Board,
        checks: &[Move],
        depth: u32,
        calls: &mut u32,
        limit: u32,
    ) -> bool {
        self.all_checks_refutable_recursive_inner(board, checks, depth, calls, limit, true)
    }

    fn all_checks_refutable_recursive_inner(
        &mut self,
        board: &mut Board,
        checks: &[Move],
        depth: u32,
        calls: &mut u32,
        limit: u32,
        store_nm: bool,
    ) -> bool {
        // F1: param_refut_full_eval=true なら false 確定 check で早期 return
        // せず，全 check を評価して残りの AND 子も store する．
        // partial coverage を完成させて将来の fast path を発火させる．
        let full_eval = self.param_refut_full_eval;
        let mut all_refuted = true;
        for check in checks {
            *calls += 1;
            if *calls > limit {
                return false;
            }
            let captured = board.do_move(*check);
            let defenses = self.generate_defense_moves(board);
            if defenses.is_empty() {
                board.undo_move(*check, captured);
                if !full_eval {
                    return false;
                }
                all_refuted = false;
                continue;
            }
            let mut has_refuting_defense = false;
            for defense in &defenses {
                let cap_d = board.do_move(*defense);
                let next_checks = self.generate_check_moves_cached(board);
                if next_checks.is_empty() {
                    board.undo_move(*defense, cap_d);
                    has_refuting_defense = true;
                    break;
                }
                // 再帰: 次の王手もすべて反証可能か確認
                // 全レベルで store_nm を伝搬 — refutable disproof として格納し
                // hand_gte 支配チェックで冗長エントリを圧縮する (v0.24.76)
                if depth > 0
                    && self.all_checks_refutable_recursive_inner(
                        board, &next_checks, depth - 1, calls, limit, store_nm,
                    )
                {
                    board.undo_move(*defense, cap_d);
                    has_refuting_defense = true;
                    break;
                }
                board.undo_move(*defense, cap_d);
            }
            // (v0.24.75) 反証成功した AND ノードを refutable disproof として
            // ProvenTT に格納．TT レベルでは通常の disproof と区別されないため
            // look_up_proven / all_checks_refutable_by_tt からは可視．
            // PNS 経路のみ look_up_proven_skip_refutable が bit 7 を見て
            // 読み飛ばすことで arena-limited false NM を防止する．
            if has_refuting_defense && store_nm {
                let and_pk = position_key(board);
                let and_hand = board.hand[self.attacker.index()];
                self.table.store_refutable_disproof(and_pk, and_hand);
            }
            board.undo_move(*check, captured);
            if !has_refuting_defense {
                if !full_eval {
                    return false;
                }
                all_refuted = false;
                // 次の check 評価へ続行 (F1)
            }
        }
        all_refuted
    }

    /// Df-Pn 探索の中核関数(文献での MID: Multiple-Iterative-Deepening に相当)．
    ///
    /// 証明数(pn)・反証数(dn)の閾値を受け取り，いずれかが閾値に達するまで
    /// 最善子ノードを再帰的に展開する．OR ノード(攻め方)では王手を生成し，
    /// AND ノード(玉方)では王手回避手を生成する．
    ///
    /// TT のキーには盤面のみのハッシュ(持ち駒除外)を使用し，
    /// 持ち駒の優越関係により TT ヒット率を向上させる．
    pub(super) fn mid(
        &mut self,
        board: &mut Board,
        pn_threshold: u32,
        dn_threshold: u32,
        ply: u32,
        or_node: bool,
    ) {
        // Plan B (v0.70.0): KH TCA inc_flag を mid() exit で `min(inc_flag, orig)` で
        // 復元する．本フレーム内で inc_flag++ した分は親フレームに leak しない．
        // (`param_use_kh_tca = false` の場合は本機構は完全に dormant)．
        let orig_inc_flag = self.inc_flag;
        // ノード制限・タイムアウトチェック
        if self.nodes_searched >= self.max_nodes {
            #[cfg(feature = "tt_diag")]
            { self.diag_node_limit_exits += 1; }
            if self.pn_dn_snapshot.is_none() {
                self.pn_dn_snapshot = Some(self.table.collect_working_pn_dn_dist());
            }
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
        // 診断 (v0.79.0, A): per-position visit count を集計．
        // (pos_key, or_node_bit) でキーとし visit count を increment．
        // 1M entries 超えたら collection 停止 (memory 暴走防止)．
        if !self.diag_pos_visits_capped {
            let pk_full = position_key(board);
            let key = (pk_full << 1) | (if or_node { 1u64 } else { 0u64 });
            *self.diag_pos_visits.entry(key).or_insert(0) += 1;
            if self.diag_pos_visits.len() > 1_000_000 {
                self.diag_pos_visits_capped = true;
            }
        }
        if (ply as usize) < 64 {
            self.ply_nodes[ply as usize] += 1;
        }
        // === Periodic GC (ProvenTT / WorkingTT 独立) ===
        // overflow カウントベースで GC をトリガする(充填率の全走査を避ける)．
        // intermediate 保護 + パス保護により，GC で探索が崩壊することはない．
        if self.nodes_searched % 100_000 == 0 && !self.param_disable_periodic_gc {
            let overflow = self.table.drain_working_overflow();

            // overflow が閾値を超え，かつ前回 GC から十分なノードが経過したら実行．
            // 連続 GC を防ぐため，GC 後は 1M ノードのクールダウンを設ける．
            if overflow > 10_000 && self.nodes_searched >= self.next_overflow_gc {
                self.mark_path_entries_for_gc_protection();
                let removed = self.table.gc_working_overflow();
                self.next_overflow_gc = self.nodes_searched + 1_000_000;
                if removed > 0 {
                    verbose_eprintln!(
                        "[overflow_gc] overflow={} removed={} working={}",
                        overflow, removed, self.table.working_len());
                }
            }

            // 1M ノードごとに ProvenTT 充填率 GC
            // confirmed disproof は永続エントリのため GC 容量判定から除外する．
            if self.nodes_searched % 1_000_000 == 0 {
                let proven_size = self.table.proven_len_for_gc();
                let proven_cap = self.table.proven_capacity();
                if proven_size > proven_cap * 7 / 10 {
                    let removed = self.table.gc_proven();
                    if removed > 0 {
                        verbose_eprintln!("[periodic_gc] proven removed={} proven={}/{}",
                            removed, self.table.proven_len(), proven_cap);
                    }
                }
                // proof GC: NPS 保護のため proof 上限を PROOF_MAP_GC_CAPACITY に制限する
                if self.table.proof_gc_needed() {
                    let removed = self.table.gc_proofs();
                    if removed > 0 {
                        eprintln!("[proof_gc] removed={} proof={} total={}",
                            removed, self.table.proven_proof_len(), self.table.proven_len());
                    }
                }
                // confirmed GC: disproof_depth 昇順 (浅い IDS = 再導出コスト安) で evict
                if self.table.confirmed_gc_needed() {
                    let removed = self.table.gc_confirmed();
                    if removed > 0 {
                        eprintln!("[confirmed_gc] removed={} confirmed={} total={}",
                            removed, self.table.proven_confirmed_len(), self.table.proven_len());
                    }
                }
            }
        }
        // Periodic progress: every 1M nodes
        #[cfg(feature = "verbose")]
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
                let total_ent = t.total_entries();
                eprintln!("[tt_diag] entries={} proof={} disproof={} inter_new={} inter_upd={} dominated={}",
                    total_ent,
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
        #[cfg(feature = "visit_diag")]
        {
            let e = self.visit_counts.entry(full_hash).or_insert(0);
            if *e == 0 {
                self.visit_first_ply.insert(full_hash, ply.min(63) as u8);
            }
            *e += 1;
        }
        let pos_key = profile_timed!(self, position_key_ns, position_key_count,
            position_key(board));
        let att_hand = board.hand[self.attacker.index()];

        // ProvenTT の ply ベース amount 用: ルートに近い proof ほど高い priority
        self.table.hint_ply = ply;

        // ループ検出: フルハッシュで判定(持ち駒込みの完全一致)
        let in_path = profile_timed!(self, loop_detect_ns, loop_detect_count,
            self.path_set.contains(&full_hash));
        if in_path {
            #[cfg(feature = "tt_diag")]
            { self.diag_in_path_exits += 1; }
            #[cfg(feature = "visit_diag")]
            { self.visit_breakdown.entry(full_hash).or_default().in_path += 1; }
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
                if remaining == 0 {
                    if tt_pn == 0 {
                        self.diag_rem0_proof += 1;
                    } else {
                        self.diag_rem0_provisional += 1;
                    }
                }
                if ply == self.diag_ply && self.diag_terminal_exits <= 3 {
                    verbose_eprintln!("[tt_diag] ply={} terminal exit: tt_pn={} tt_dn={} remaining={}",
                        ply, tt_pn, tt_dn, remaining);
                }
            }
            #[cfg(feature = "visit_diag")]
            { self.visit_breakdown.entry(full_hash).or_default().proven_exit += 1; }
            return;
        }
        if tt_pn >= pn_threshold || tt_dn >= dn_threshold {
            #[cfg(feature = "tt_diag")]
            {
                self.diag_threshold_exits += 1;
                if ply == self.diag_ply && self.diag_threshold_exits <= 3 {
                    verbose_eprintln!("[tt_diag] ply={} threshold exit: tt_pn={} tt_dn={} pn_th={} dn_th={}",
                        ply, tt_pn, tt_dn, pn_threshold, dn_threshold);
                }
            }
            #[cfg(feature = "visit_diag")]
            {
                let bd = self.visit_breakdown.entry(full_hash).or_default();
                bd.threshold_exit += 1;
                if tt_pn == PN_UNIT { bd.th_pn_default += 1; }
                else if tt_pn >= INF { bd.th_pn_inf += 1; }
                if tt_pn >= pn_threshold { bd.th_pn_caused += 1; }
                if tt_dn >= dn_threshold { bd.th_dn_caused += 1; }
            }
            return;
        }
        // 実際に子を展開する (exploration)
        #[cfg(feature = "visit_diag")]
        {
            let bd = self.visit_breakdown.entry(full_hash).or_default();
            // TT miss シグナル: biased default は (biased_pn, PN_UNIT, 0) を返す
            // ため tt_dn == PN_UNIT == 16 ならば TT miss (eviction) と判断する．
            // TT hit の場合 tt_dn は子の dn 集計値であり，PN_UNIT と一致することは稀．
            if tt_dn == PN_UNIT {
                bd.expl_tt_miss += 1;
            } else {
                bd.expl_tt_hit += 1;
            }
            if bd.exploration == 0 {
                bd.expl_pn_first = tt_pn;
                bd.expl_dn_first = tt_dn;
            } else if tt_pn == bd.expl_pn_last {
                bd.expl_pn_stuck += 1;
            }
            bd.expl_pn_last = tt_pn;
            bd.expl_dn_last = tt_dn;
            bd.exploration += 1;
        }
        // TT 診断: ply 35 で terminal exit しなかった場合の TT 状態を出力
        #[cfg(feature = "tt_diag")]
        {
            let visit_count = self.diag_ply_visits[ply as usize];
            if ply == self.diag_ply && (visit_count <= 5 || (visit_count % 1000000 == 0)) {
                let entry_count = self.table.entries_for_position(pos_key, &att_hand);
                verbose_eprintln!(
                    "[tt_diag] ply={} non-terminal entry #{}: pos_key={:#x} tt_pn={} tt_dn={} \
                     remaining={} hand={:?} tt_entries_at_key={}",
                    ply, visit_count, pos_key, tt_pn, tt_dn,
                    remaining, &att_hand, entry_count);
                if visit_count <= 3 || visit_count == 1000000 {
                    self.table.dump_entries(pos_key, &att_hand);
                }
            }
        }

        // 終端条件: 深さ制限・手数制限
        if ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            #[cfg(feature = "profile")]
            let _depth_limit_start = Instant::now();
            if or_node {
                let checks = self.generate_check_moves_cached(board);
                if checks.is_empty() {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_boundary_or_no_checks += 1; }
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key as u32);
                } else {
                    let (refuted, km_cr) = self.all_checks_refutable_by_tt(board, &checks);
                    if refuted {
                        #[cfg(feature = "tt_diag")]
                        {
                            self.diag_boundary_or_refutable += 1;
                            self.diag_boundary_or_checks_sum += checks.len() as u64;
                        }
                        if let Some(cr) = km_cr {
                            self.store_path_dep(pos_key, att_hand, INF, 0, REMAINING_INFINITE, cr, true);
                        } else {
                            self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key as u32);
                        }
                    } else {
                        // rem=0 の仮反証は TT に store しない
                        // (クラスタの 64.7% を占め overflow の主因)
                        #[cfg(feature = "tt_diag")]
                        {
                            self.diag_boundary_or_not_refutable += 1;
                            self.diag_boundary_or_checks_sum += checks.len() as u64;
                        }
                    }
                }
            } else {
                // AND ノードの深さ制限: rem=0 は TT に store しない
                #[cfg(feature = "tt_diag")]
                { self.diag_boundary_and_total += 1; }
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
                self.generate_check_moves_cached(board))
        } else {
            profile_timed!(self, movegen_defense_ns, movegen_defense_count,
                self.generate_defense_moves(board))
        };

        // SNDA (Sequential Non-capturing Drop Avoidance):
        // 直前の OR ノード手が非捕獲打ちなら，同駒種の非捕獲打ちを他マスで抑制する．
        // 根拠: path (sq1→def→sq2) と (sq2→def'→sq1) は同一局面に到達する
        // ため，後者は前者を探索した際に既にカバーされている．
        if or_node {
            let prev = self.prev_attacker_move;
            if prev.is_drop() {
                if let Some(prev_pt) = prev.drop_piece_type() {
                    let prev_to = prev.to_sq();
                    moves.retain(|m| {
                        !(m.is_drop()
                          && m.drop_piece_type() == Some(prev_pt)
                          && m.to_sq() != prev_to)
                    });
                }
            }
        }

        let save_alpha_x = self.alpha_x_filter_active;

        // 施策 α (boundary chain drop filter, v0.24.47-72): PNS→MID サイクル
        // の filter context 非伝達問題で false proof を生成するため v0.24.72
        // で不採用確定．refutable disproof 機構 (v0.24.75+) で代替済み．
        //
        // proof_tag propagation infrastructure (get_proof_tag / store_proof_with_tag
        // 等) も連動して dead code 化．以下の `if filter_applied { ... }` 分岐および
        // `if child_tag != ABSOLUTE { ... }` 分岐は条件判定のみ残るが，
        // filter_applied=false により実行経路に到達しない．
        // (削除候補: 将来のクリーンアップで一括除去可能)
        let filter_applied = false;

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
                self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key as u32);
            } else {
                // 応手なし → 詰み(証明駒 = 空)
                self.store(
                    pos_key,
                    [0; HAND_KINDS],
                    0,
                    INF,
                    REMAINING_INFINITE,
                    pos_key as u32,
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
        // K-M (v0.55.23): init フェーズで蓄積した OR 子の共通 cycle_root
        // None=未設定, Some(0)=boolean path_dep, Some(r)=K-M タグ
        let mut init_or_km_cycle_root: Option<u32> = None;
        // init フェーズでの OR 子 NM remaining の最小値
        let mut init_or_nm_min_remaining: u16 = REMAINING_INFINITE;
        // AND ノードの init フェーズ用: TT プレフィルタで証明済み合駒の証明駒蓄積
        let mut init_and_proof = [0u8; HAND_KINDS];
        // チェーン合駒コンテキストでの遅延 AND 反証
        let mut init_and_disproof_found = false;
        let mut init_and_disproof_remaining: u16 = 0;
        let mut init_and_disproof_path_dep = false;
        // 反証駒伝播用: 反証を引き起こした子の pos_key と hand を保存
        let mut init_and_disproof_child_pk: u64 = 0;
        let mut init_and_disproof_child_hand = [0u8; HAND_KINDS];
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
        #[cfg(feature = "verbose")]
        let _init_start = Instant::now();
        let is_at_depth_limit = ply + 1 >= self.depth;
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

            // look_up_pn_dn (fastpath と統合: 1 回のみ実行)
            let (mut cpn, mut cdn, _csrc) =
                self.look_up_pn_dn(child_pk, &child_hand, child_remaining);

            // 深さ制限ファストパス: 子ノードが ply+1 >= depth で即座に
            // 深さ制限に到達する場合，mid() 呼び出しを省略して直接反証を記録する．
            if is_at_depth_limit && cpn != 0 {
                // E1 最適化: proof の有無を先にチェックし，再 lookup を省略する．
                // proof があれば子は証明済み(cpn=0)．なければ store 後の
                // TT 状態は必ず disproof(dn=0)を含むため (cpn, cdn) = (INF, 0)．
                if self.table.has_proof(child_pk, &child_hand) {
                    cpn = 0;
                    cdn = INF;
                } else {
                    if !or_node {
                        // AND 親の子 = OR 局面: 王手の有無で REMAINING_INFINITE 判定
                        let checks = self.generate_check_moves_cached(board);
                        let (dl_rem, dl_km_cr) = if checks.is_empty() {
                            (REMAINING_INFINITE, None)
                        } else {
                            let (ok, cr) = self.all_checks_refutable_by_tt(board, &checks);
                            if ok { (REMAINING_INFINITE, cr) } else { (0u16, None) }
                        };
                        // REMAINING_INFINITE(真の不詰)のみ store する．
                        // rem=0 の仮反証は TT に store しない:
                        // - 同じ IDS depth の同じ深さでしか参照されない
                        // - クラスタの 64.7% を占め overflow の主因
                        // - ローカル変数 cpn/cdn で解決チェック可能
                        if dl_rem == REMAINING_INFINITE {
                            if let Some(cr) = dl_km_cr {
                                self.store_path_dep(child_pk, child_hand, INF, 0, dl_rem, cr, true);
                            } else {
                                self.store(child_pk, child_hand, INF, 0, dl_rem, child_pk as u32);
                            }
                        }
                    }
                    // rem=0 は TT に store せず，ローカル変数のみで処理
                    cpn = INF;
                    cdn = 0;
                }
            }

            #[cfg(feature = "profile")]
            let _ci_inline_start = Instant::now();
            if cpn == PN_UNIT && cdn == PN_UNIT {
                if or_node {
                    // インライン1手・3手詰め判定(AND 子ノード)
                    let defenses = self.generate_defense_moves(board);
                    if defenses.is_empty() {
                        // 応手なし → 即詰み確定(budget=0 パスでの検出)
                        self.store(child_pk, [0; HAND_KINDS], 0, INF,
                            REMAINING_INFINITE, child_pk as u32);
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
                                let checks = self.generate_check_moves_cached(board);
                                if !checks.is_empty() {
                                    self.has_mate_in_1_with(board, &checks)
                                } else {
                                    false
                                }
                            };
                            if mate {
                                self.store_board(board, 0, INF,
                                    REMAINING_INFINITE, child_pk as u32);
                            }
                            board.undo_move(*d, cap_d);
                            if !mate {
                                all_mated = false;
                                break;
                            }
                        }
                        if all_mated {
                            self.store(child_pk, child_hand, 0, INF,
                                REMAINING_INFINITE, child_pk as u32);
                        } else {
                            let n = defenses.len() as u32;
                            let mut pn = self.heuristic_and_pn(board, n);
                            // DFPN-E: エッジコスト加算
                            if let Some(ksq) = defender_king_sq {
                                if self.param_use_per_move_support {
                                    pn = pn.saturating_add(
                                        super::edge_cost_or_with_support(
                                            *m, ksq, board, board.turn.opponent(),
                                        ),
                                    );
                                } else {
                                    pn = pn.saturating_add(edge_cost_or(*m, ksq));
                                }
                            }
                            let dn = heuristic_dn_from_pn(pn);
                            self.store(child_pk, child_hand, pn, dn,
                                child_remaining, child_pk as u32);
                        }
                    } else {
                        // depth 制限超過: 応手生成なし → deep df-pn のみ適用
                        let mut pn = PN_UNIT;
                        // DFPN-E: エッジコスト加算
                        if let Some(ksq) = defender_king_sq {
                            if self.param_use_per_move_support {
                                pn = pn.saturating_add(
                                    super::edge_cost_or_with_support(
                                        *m, ksq, board, board.turn.opponent(),
                                    ),
                                );
                            } else {
                                pn = pn.saturating_add(edge_cost_or(*m, ksq));
                            }
                        }
                        let dn = heuristic_dn_from_pn(pn);
                        self.store(child_pk, child_hand, pn, dn,
                            child_remaining, child_pk as u32);
                    }
                } else {
                    // インライン王手なし/1手詰め判定 + 取り後TT参照(OR 子ノード)
                    let checks = self.generate_check_moves_cached(board);
                    if checks.is_empty() {
                        self.store(child_pk, child_hand, INF, 0,
                            REMAINING_INFINITE, child_pk as u32);
                    } else if ply + 2 < self.depth
                        && self.has_mate_in_1_with(board, &checks)
                    {
                        self.store(child_pk, child_hand, 0, INF,
                            REMAINING_INFINITE, child_pk as u32);
                    } else if ply + 2 < self.depth
                        && self.try_capture_tt_proof(
                            board, &checks, child_remaining)
                    {
                        // 取りの王手で既証明局面に到達 → 即座に証明
                    } else {
                        let nc = checks.len() as u32;
                        let (or_pn, or_se) = self.heuristic_or_pn(board, nc, child_pk);
                        let pn = or_pn.saturating_add(edge_cost_and(*m));
                        let att_in_check = board.is_in_check(board.turn);
                        let dn = heuristic_or_dn(or_se, nc, att_in_check);
                        self.store(child_pk, child_hand, pn,
                            dn, child_remaining, child_pk as u32);
                    }
                }
            }
            #[cfg(feature = "profile")]
            {
                self.profile_stats.ci_inline_ns += _ci_inline_start.elapsed().as_nanos() as u64;
                self.profile_stats.ci_inline_count += 1;
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
            // depth-limit fast path で rem=0 を TT に store しない場合，
            // ローカル変数 cpn/cdn を直接使用する．
            #[cfg(feature = "profile")]
            let _ci_resolve_start = Instant::now();
            let (cpn_now, cdn_now, _) = if cpn == INF && cdn == 0 {
                // depth-limit fast path で設定済み(TT に未 store の可能性)
                (INF, 0, 0)
            } else if cpn == 0 && cdn == INF {
                // proof 発見済み
                (0, INF, 0)
            } else {
                self.look_up_pn_dn(child_pk, &child_hand, child_remaining)
            };
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
                    REMAINING_INFINITE, pos_key as u32);
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
                        // 反証駒伝播用に子の pos_key と hand を保存
                        init_and_disproof_child_pk = child_pk;
                        init_and_disproof_child_hand = child_hand;
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
                            verbose_eprintln!("[tt_diag] ply={} init AND disproof (deferred): move={} child_rem={} parent_rem={} remaining={} path_dep={} pos_key={:#x}",
                                ply, m.to_usi(), child_nm_rem, init_and_disproof_remaining, remaining,
                                init_and_disproof_path_dep, pos_key as u32);
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
                    verbose_eprintln!("[tt_diag] ply={} init AND disproof: move={} child_rem={} parent_rem={} remaining={} path_dep={} pos_key={:#x}",
                        ply, m.to_usi(), child_nm_rem, parent_nm_remaining, remaining, is_path_dep, pos_key as u32);
                    self.diag_in_path_exits += 1;
                }
                if is_path_dep {
                    self.store_path_dep(
                        pos_key, att_hand, INF, 0,
                        parent_nm_remaining, pos_key as u32, true,
                    );
                } else {
                    // 反証駒最適化: 子の ProvenTT 反証駒 (DH_C ≥ att_hand) を AND-node に伝播．
                    // AND-node では守備側着手で att_hand 不変 → child_hand = att_hand．
                    // DH_C を保存することで将来 H ≤ DH_C のクエリが TT ヒット → 再探索削減．
                    let dh = self.table.get_disproof_hand(child_pk, &child_hand);
                    self.store(pos_key, dh, INF, 0,
                        parent_nm_remaining, pos_key as u32);
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
                            verbose_eprintln!(
                                "[tt_diag] WARNING: non-chain init AND disproof verification FAILED: \
                                 ply={} visit={} pos_key={:#x} hand={:?} stored dn=0 path_dep={} rem={} \
                                 but lookup(rem={}) returns pn={} dn={}",
                                ply, visit_count, pos_key, &att_hand, is_path_dep,
                                parent_nm_remaining, remaining, v_pn, v_dn
                            );
                            self.table.dump_entries(pos_key, &att_hand);
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
                ).map(|(r, _)| r).unwrap_or(child_remaining);
                init_or_nm_min_remaining = init_or_nm_min_remaining.min(child_nm_rem);
                // GHI 伝播: 子の反証が経路依存なら蓄積 (K-M: cycle_root も追跡)
                if let Some(cr) = self.table.get_path_dep_cycle_root(child_pk, &child_hand) {
                    init_or_path_dep = true;
                    init_or_km_cycle_root = Some(match init_or_km_cycle_root {
                        None => cr,
                        Some(prev) if prev == cr => cr,
                        _ => 0,  // 異なる cycle_root → boolean path_dep に格下げ
                    });
                }
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
            if !or_node && m.is_drop() && cpn == PN_UNIT && cdn == PN_UNIT {
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
            #[cfg(feature = "profile")]
            {
                self.profile_stats.ci_resolve_ns += _ci_resolve_start.elapsed().as_nanos() as u64;
                self.profile_stats.ci_resolve_count += 1;
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
                let checks = self.generate_check_moves_cached(board);
                if checks.is_empty() {
                    parent_nm_remaining = REMAINING_INFINITE;
                } else {
                    let (refuted, km_cr) = self.all_checks_refutable_by_tt(board, &checks);
                    if refuted {
                        parent_nm_remaining = REMAINING_INFINITE;
                        // K-M 反証を使った場合，親の反証も経路依存にする
                        if let Some(cr) = km_cr {
                            init_or_path_dep = true;
                            init_or_km_cycle_root = Some(match init_or_km_cycle_root {
                                None => cr,
                                Some(prev) if prev == cr => cr,
                                _ => 0,
                            });
                        }
                    }
                }
            }
            //
            // GHI 伝播: いずれかの子の反証が経路依存なら親も経路依存
            // OR ノード反証: att_hand で保存(TT ヒット率最大化)
            // 実際の持ち駒で不詰が確定しているため，att_hand で登録すれば
            // hand dominance によるカバー範囲が最大になる．
            if init_or_path_dep {
                // K-M (v0.55.23): 共通 cycle_root を伝播
                let cr = init_or_km_cycle_root.unwrap_or(0);
                self.store_path_dep(
                    pos_key, att_hand, INF, 0,
                    parent_nm_remaining, cr, true,
                );
            } else {
                self.store(
                    pos_key, att_hand, INF, 0,
                    parent_nm_remaining, pos_key as u32,
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
                    remaining, pos_key as u32,
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
                    init_and_disproof_remaining, pos_key as u32, true,
                );
            } else {
                // 反証駒最適化: 子の ProvenTT 反証駒 (DH_C ≥ att_hand) を AND-node に伝播．
                let dh = self.table.get_disproof_hand(
                    init_and_disproof_child_pk, &init_and_disproof_child_hand,
                );
                self.store(pos_key, dh, INF, 0,
                    init_and_disproof_remaining, pos_key as u32);
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
                        verbose_eprintln!(
                            "[tt_diag] WARNING: deferred init AND disproof verification FAILED: \
                             ply={} visit={} pos_key={:#x} hand={:?} stored dn=0 path_dep={} rem={} \
                             but lookup(rem={}) returns pn={} dn={}",
                            ply, visit_count, pos_key, &att_hand, init_and_disproof_path_dep,
                            init_and_disproof_remaining, remaining, v_pn, v_dn
                        );
                        self.table.dump_entries(pos_key, &att_hand);
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

        // パスに追加(フルハッシュ + pos_key + hand)
        debug_assert!(self.path_len < PATH_CAPACITY, "path overflow at ply={}", ply);
        self.path[self.path_len] = full_hash;
        self.path_pos_key[self.path_len] = pos_key;
        self.path_hand[self.path_len] = att_hand;
        self.path_len += 1;
        self.path_set.insert(full_hash);

        // melodic-cascading-otter (v0.69.0): parent_map に child→parent を or_insert．
        // path_len >= 2 ならば一つ前のフレームが parent．既存 entry は上書きしない
        // (最初の親のみ記録 = 再訪問時に DAG として検出可能にする)．
        if self.param_use_dag_correction && self.path_len >= 2 {
            let parent_fh = self.path[self.path_len - 2];
            self.parent_map.entry(full_hash).or_insert(parent_fh);
        }

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

        // 混合 AND ノード(非 drop 応手 + chain drop 両方あり)での
        // chain drop 距離ベース bias 適用のため，玉位置を保持する．
        //
        // 既存の chain_king_sq は全 drop が chain の場合のみ設定されるが，
        // 39te ply 22 の 5g6f 後の AND のように「玉逃げ + chain drop」が
        // 混在するケースでは chain_king_sq = None となり，chain drop は
        // 単純な INTERPOSE_DN_BIAS のみで距離ベース bias が適用されない．
        // これが外側 chain drop への探索リソース浪費の原因となっている．
        //
        // `mixed_chain_king_sq` は混合 AND でも defender king 位置を
        // 保持し，chain drop 子に対して距離ベース bias を適用する．
        let mixed_chain_king_sq =
            if chain_king_sq.is_none() && !or_node && saved_chain_bb.is_not_empty() {
                board.king_square(board.turn)
            } else {
                None
            };

// プレフィルタで全合駒が証明済み(children が空になった場合)
        if !or_node && init_prefiltered_count > 0 && children.is_empty() {
            let mut p = init_and_proof;
            for k in 0..HAND_KINDS {
                p[k] = p[k].min(att_hand[k]);
            }
            if filter_applied {
                self.store_proof_with_tag(pos_key, p, 0, 0,
                    super::entry::PROOF_TAG_FILTER_DEPENDENT);
            } else {
                self.store(pos_key, p, 0, INF, REMAINING_INFINITE, pos_key as u32);
            }
            debug_assert_eq!(self.path[self.path_len - 1], full_hash);
            self.alpha_x_filter_active = save_alpha_x;
            self.path_set.remove(&full_hash);
            self.path_len -= 1;
            return;
        }

        // Init phase duration diagnostic
        #[cfg(feature = "verbose")]
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
        // OR ノードでは王手が1手のみ，AND ノードでは合法応手が1手のみの
        // ケースが詰将棋で頻出する．
        if children.len() == 1 {
            #[cfg(feature = "tt_diag")]
            if (ply as usize) < 64 {
                self.diag_ply_proofs[ply as usize] += 1; // reuse as single-child counter
            }
            let (m, child_fh, child_pk, ref child_hand) = children[0];
            let mut _sc_iter: u64 = 0;
            // 停滞検出: 子の pn/dn が変化しなければ mid() を呼んでも無駄．
            let mut prev_cpn: u32 = 0;
            let mut prev_cdn: u32 = 0;
            let mut stagnation_count: u32 = 0;
            // single-child ループも MID ループと同じ STAGNATION_LIMIT を使用
            loop {
                _sc_iter += 1;
                // ノード制限・タイムアウトチェック
                if self.nodes_searched >= self.max_nodes || self.timed_out {
                    break;
                }

                // ループ検出: 子がパス上にある場合は (INF, 0) として扱い，
                // mid() を呼ばず即座にループ NM として処理する．
                // GHI 対策: ループ子の NM は path_dependent として store される(下流)．
                // v0.55.38: visit_history dominance check も同じパスで処理する
                // (param_use_visit_history_dominance == true のとき)．
                let is_loop_child = self.path_set.contains(&child_fh)
                    || self.is_dominated_in_path(child_pk, child_hand);
                let (cpn, cdn, _csrc) = if is_loop_child {
                    (INF, 0, 0)
                } else {
                    self.look_up_pn_dn(
                        child_pk, child_hand,
                        remaining.saturating_sub(1),
                    )
                };
                // 終端チェックを閾値チェックより先に行う (v0.55.24 fix):
                // cpn=INF かつ cdn=0 のとき閾値チェックが INF >= INF-1 で先に発火し，
                // remaining をそのまま store してしまうバグを修正する．
                if cpn == 0 || cdn == 0 {
                    // 子の証明/反証 → 親に伝播
                    if or_node {
                        if cpn == 0 {
                            let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                            let mut proof = adjust_hand_for_move(m, &child_ph);
                            for k in 0..HAND_KINDS {
                                proof[k] = proof[k].min(att_hand[k]);
                            }
                            // (v0.24.71) tag propagation: 子の tag を継承
                            let child_tag = self.table.get_proof_tag(child_pk, child_hand);
                            if child_tag != super::entry::PROOF_TAG_ABSOLUTE {
                                self.store_proof_with_tag(pos_key, proof, m.to_move16(), 0, child_tag);
                            } else {
                                self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key as u32);
                            }
                        } else {
                            // cdn == 0: 唯一の子が反証 → OR 反証
                            // att_hand で保存(TT ヒット率最大化)
                            // K-M (v0.55.23): cycle_root = child_fh as u32 (直接ループ)
                            //                 または子の cycle_root を伝播
                            let cycle_root: Option<u32> = if is_loop_child {
                                Some(child_fh as u32)
                            } else {
                                self.table.get_path_dep_cycle_root(child_pk, child_hand)
                            };
                            if let Some(cr) = cycle_root {
                                // path_dep 反証: remaining をそのまま使用
                                self.store_path_dep(
                                    pos_key, att_hand, INF, 0,
                                    remaining, cr, true,
                                );
                            } else {
                                // 絶対反証: 子の nm_remaining を伝播 (v0.55.24 fix)
                                let child_nm_rem = self.table
                                    .get_effective_disproof_info(
                                        child_pk, child_hand,
                                        remaining.saturating_sub(1),
                                    )
                                    .map(|(r, _)| r)
                                    .unwrap_or(remaining.saturating_sub(1));
                                let parent_nm_remaining =
                                    propagate_nm_remaining(child_nm_rem, remaining);
                                self.store(pos_key, att_hand, INF, 0,
                                    parent_nm_remaining, pos_key as u32);
                            }
                        }
                    } else {
                        if cdn == 0 {
                            // AND 反証: 反証駒最適化 + nm_remaining 伝播 (v0.55.24 fix)
                            // K-M (v0.55.23): cycle_root = child_fh as u32 (直接ループ)
                            //                 または子の cycle_root を伝播
                            let cycle_root: Option<u32> = if is_loop_child {
                                Some(child_fh as u32)
                            } else {
                                self.table.get_path_dep_cycle_root(child_pk, child_hand)
                            };
                            if let Some(cr) = cycle_root {
                                // path_dep 反証: remaining をそのまま使用
                                self.store_path_dep(
                                    pos_key, att_hand, INF, 0,
                                    remaining, cr, true,
                                );
                            } else {
                                // 絶対反証: 子の nm_remaining を伝播 + 反証駒最適化 (v0.55.24 fix)
                                let child_nm_rem = self.table
                                    .get_effective_disproof_info(
                                        child_pk, child_hand,
                                        remaining.saturating_sub(1),
                                    )
                                    .map(|(r, _)| r)
                                    .unwrap_or(remaining.saturating_sub(1));
                                let parent_nm_remaining =
                                    propagate_nm_remaining(child_nm_rem, remaining);
                                let dh = self.table.get_disproof_hand(child_pk, child_hand);
                                self.store(pos_key, dh, INF, 0,
                                    parent_nm_remaining, pos_key as u32);
                            }
                        } else {
                            // cpn == 0: 唯一の子が証明 → AND 証明
                            let child_ph = self.table.get_proof_hand(child_pk, child_hand);
                            let mut proof = [0u8; HAND_KINDS];
                            for k in 0..HAND_KINDS {
                                proof[k] = child_ph[k].min(att_hand[k]);
                            }
                            if filter_applied {
                                self.store_proof_with_tag(pos_key, proof, 0, 0,
                                    super::entry::PROOF_TAG_FILTER_DEPENDENT);
                            } else {
                                self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key as u32);
                            }
                        }
                    }
                    break;
                }
                if cpn >= pn_threshold || cdn >= dn_threshold {
                    self.store(pos_key, att_hand, cpn, cdn, remaining, pos_key as u32);
                    break;
                }

                let captured = profile_timed!(self, do_move_ns, do_move_count,
                    board.do_move(m));
                // SNDA: OR ノードなら prev_attacker_move を現在の手に設定して伝播する
                let _snda_saved_sc = if or_node {
                    let s = self.prev_attacker_move;
                    self.prev_attacker_move = m;
                    s
                } else { Move(0) };
                self.mid(board, pn_threshold, dn_threshold, ply + 1, !or_node);
                if or_node { self.prev_attacker_move = _snda_saved_sc; }
                profile_timed!(self, undo_move_ns, undo_move_count,
                    board.undo_move(m, captured));

                // 停滞検出: 子の pn/dn が mid() 前後で変化しない場合，
                // 閾値不足で mid() が進捗できていない．
                // STAGNATION_LIMIT 回連続で無変化ならループを脱出し
                // 親 MID に制御を戻す．
                let (post_cpn, post_cdn, _) = if is_loop_child {
                    (INF, 0, 0)
                } else {
                    self.look_up_pn_dn(
                        child_pk, child_hand,
                        remaining.saturating_sub(1),
                    )
                };
                if post_cpn == prev_cpn && post_cdn == prev_cdn {
                    stagnation_count += 1;
                    if stagnation_count >= STAGNATION_LIMIT {
                        // 停滞: 現在の pn/dn を store して脱出
                        self.store(pos_key, att_hand, post_cpn, post_cdn,
                            remaining, pos_key as u32);
                        break;
                    }
                } else {
                    stagnation_count = 0;
                }
                prev_cpn = post_cpn;
                prev_cdn = post_cdn;
            }
            debug_assert_eq!(self.path[self.path_len - 1], full_hash);
            self.alpha_x_filter_active = save_alpha_x;
            self.path_set.remove(&full_hash);
            self.path_len -= 1;
            #[cfg(feature = "tt_diag")]
            { self.diag_single_child_exits += 1; }
            return;
        }

        // SNDA 用の (source, value) ペアバッファ(ループ外で確保し再利用)
        let mut snda_pairs: Vec<(u32, u32)> = Vec::new();

        // TT 診断: このノードが監視対象 ply かどうか + 反復カウンタ
        #[cfg(feature = "tt_diag")]
        let _diag_this_node = self.diag_ply > 0 && ply == self.diag_ply;
        #[cfg(feature = "tt_diag")]
        let mut _diag_iteration: u32 = 0;
        #[cfg(feature = "tt_diag")]
        if _diag_this_node && self.diag_ply_visits[ply as usize] <= 2 {
            verbose_eprintln!(
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
                verbose_eprintln!(
                    "[tt_diag]   child[{}] move={} drop={} pn={} dn={} pos_key={:#x}",
                    i, cm.to_usi(), cm.is_drop(), cpn, cdn, cpk,
                );
            }
        }

        // MID ループ(証明駒/反証駒の伝播を含む)
        #[cfg(feature = "verbose")]
        let mut _loop_iter: u64 = 0;
        #[cfg(feature = "verbose")]
        let _loop_start_nodes = self.nodes_searched;
        #[cfg(feature = "verbose")]
        let mut _next_diag_nodes = self.nodes_searched.saturating_add(1_000_000);
        // 停滞検出: 子 mid() が消費するノード数が0(閾値で即座に返る)の
        // 連続回数を追跡．一定回数以上ゼロ進捗が続けば MID ループを脱出し，
        // 上位ノードに制御を戻す(dn_floor 由来の空転防止)．
        let mut zero_progress_count: u32 = 0;
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
        // 同一子連続選択カウンタ (スラッシング防止用幾何的閾値増幅)．
        // 同じ best_idx が選ばれるたびにインクリメントし，
        // STAGNATION_LIMIT 回ごとに閾値を 2 倍する (最大 128 倍)．
        let mut same_child_iters: u32 = 0;
        loop {
            #[cfg(feature = "verbose")]
            { _loop_iter += 1; }
            if (ply as usize) < 64 {
                self.ply_iters[ply as usize] += 1;
            }
            // ply=0 は 100K ごと，それ以外は 1M ごとに詳細診断
            #[cfg(feature = "verbose")]
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
                        for e in self.table.entries_iter(cpk, ch) {
                            eprintln!("[tt_dump]     pn={} dn={} rem={} path_dep={} hand={:?}",
                                e.pn, e.dn, e.remaining(), e.path_dependent(), &e.hand);
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
            let mut best_source: u32 = 0;

            // TCA: OR ノードでのループ子ノード数
            let mut loop_child_count: u32 = 0;
            // K-M (v0.55.23): main loop で蓄積した OR 子の共通 cycle_root
            let mut or_km_cycle_root: Option<u32> = None;
            // OR NM remaining 伝播: 全子 NM の remaining の最小値を追跡
            let mut or_nm_min_remaining: u16;

            if or_node {
                // OR ノード: min(pn), WPN-scaled sum(dn)
                current_pn = INF;
                current_dn = 0;
                second_best = INF; // 2番目に小さい pn(選択用，予算枯渇除外込み)
                let mut select_best_pn: u32 = INF; // 選択用 best pn
                // NM remaining 伝播: init フェーズの値を引き継ぐ
                or_nm_min_remaining = init_or_nm_min_remaining;
                // SNDA: (source, dn) ペアを収集し，同一 source の子は
                // sum の代わりに max で集約して過大評価を補正する
                snda_pairs.clear();
                // WPN: OR dn の max(child_dn) を追跡
                let mut max_cdn: u32 = 0;
                // HandSet (v0.56.1): 子 disproof_hand の交集合 (要素 min) 累積．
                // OR 全子反証時の disproof_hand 計算用 (path-dep child は除外)．
                let mut or_disproof_hand = [u8::MAX; HAND_KINDS];
                let mut or_disproof_initialized = false;
                // 診断 (v0.78.0): OR scan の proven child 存在を追跡．
                let mut or_scan_has_proven: bool = false;

                for (i, &(ref _m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    // v0.55.38: visit_history dominance check を path_set.contains に
                    // 併用 (H4-A 補完，OR multi-child loop も含める)．
                    let is_loop_or_dominated_child = self.path_set.contains(&child_fh)
                        || self.is_dominated_in_path(child_pk, child_hand);
                    let (cpn, cdn, csrc) =
                        if is_loop_or_dominated_child {
                            loop_child_count += 1;
                            // K-M (v0.55.23): 直接ループ → cycle_root = child_fh as u32
                            // dominance 時は child_fh が path 上にないため cycle_root として
                            // 不正だが，下流の cross-branch 再利用チェックで弾かれる (safe)．
                            let cr = child_fh as u32;
                            or_km_cycle_root = Some(match or_km_cycle_root {
                                None => cr,
                                Some(prev) if prev == cr => cr,
                                _ => 0,
                            });
                            (INF, 0, 0)
                        } else {
                            self.look_up_pn_dn(
                                child_pk, child_hand,
                                remaining.saturating_sub(1),
                            )
                        };

                    if cpn == 0 {
                        // 診断: OR scan で proven child を発見
                        or_scan_has_proven = true;
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
                        // (v0.24.71) tag propagation: 子の proof tag を継承．
                        // 子が FILTER_DEPENDENT なら親 OR も FILTER_DEPENDENT．
                        let child_tag = self.table.get_proof_tag(
                            child_pk, child_hand,
                        );
                        if child_tag != super::entry::PROOF_TAG_ABSOLUTE {
                            self.store_proof_with_tag(
                                pos_key, proof,
                                children[i].0.to_move16(), 0, child_tag,
                            );
                        } else {
                            self.store(
                                pos_key, proof, 0, INF,
                                REMAINING_INFINITE, csrc,
                            );
                        }
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
                        // GHI 伝播: 子の反証が経路依存なら親も経路依存 (K-M: cycle_root も追跡)
                        if !self.path_set.contains(&child_fh) {
                            if let Some(cr) = self.table.get_path_dep_cycle_root(
                                child_pk, child_hand,
                            ) {
                                loop_child_count += 1;
                                or_km_cycle_root = Some(match or_km_cycle_root {
                                    None => cr,
                                    Some(prev) if prev == cr => cr,
                                    _ => 0,
                                });
                            }
                        }
                        // HandSet OR disproof 交集合 (v0.56.1, opt-in)．
                        // path-dep child (loop / dominance) は実際の disproof_hand を
                        // 持たないため除外 (get_disproof_hand は child_hand を fallback で
                        // 返してしまい交集合が保守的になりすぎる)．
                        if self.param_use_handset_combination && !is_loop_or_dominated_child {
                            let child_dh = self.table.get_disproof_hand(child_pk, child_hand);
                            if !or_disproof_initialized {
                                or_disproof_hand = child_dh;
                                or_disproof_initialized = true;
                            } else {
                                for k in 0..HAND_KINDS {
                                    or_disproof_hand[k] = or_disproof_hand[k].min(child_dh[k]);
                                }
                            }
                        }
                    }

                    // True min cpn tracking (for node's proof number).
                    if cpn < current_pn {
                        current_pn = cpn;
                    }
                    // OR best_idx 選択: 主に argmin(cpn)．tie-break:
                    // - default: argmin(cdn) — defender が反証しやすい child から確認
                    // - param_or_dn_tiebreak=true (F, v0.75.0): argmax(cdn) —
                    //   defender 抵抗が強い child を優先 (mate path commitment)
                    let tie_better = if self.param_or_dn_tiebreak {
                        cdn > best_pn_dn.1
                    } else {
                        cdn < best_pn_dn.1
                    };
                    if cpn < select_best_pn
                        || (cpn == select_best_pn && tie_better)
                    {
                        second_best = select_best_pn;
                        select_best_pn = cpn;
                        best_idx = i;
                        best_pn_dn = (cpn, cdn);
                        best_source = csrc;
                    } else if cpn < second_best {
                        second_best = cpn;
                    }
                    // WPN: max(child_dn) を更新
                    if cdn > max_cdn { max_cdn = cdn; }
                    // sum(dn) を累積
                    current_dn = (current_dn as u64)
                        .saturating_add(cdn as u64)
                        .min(INF as u64)
                        as u32;
                    // SNDA ペア収集(source=0 は独立ノード → グルーピング対象外)
                    if csrc != 0 && cdn > 0 {
                        snda_pairs.push((csrc, cdn));
                    }
                }

                // 診断 (v0.78.0): OR scan の coverage 記録．
                {
                    self.diag_or_visit_count += 1;
                    self.diag_or_total_sum += children.len() as u64;
                    if or_scan_has_proven { self.diag_or_proven_count_visits += 1; }
                }

                if proved_or_disproved {
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
                    self.path_set.remove(&full_hash);
                    self.path_len -= 1;
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
                        let checks = self.generate_check_moves_cached(board);
                        if checks.is_empty() {
                            parent_nm_remaining = REMAINING_INFINITE;
                        } else {
                            let (refuted, km_cr) = self.all_checks_refutable_by_tt(board, &checks);
                            if refuted {
                                parent_nm_remaining = REMAINING_INFINITE;
                                // K-M 反証を使った場合，親の反証も経路依存にする
                                if let Some(cr) = km_cr {
                                    loop_child_count = loop_child_count.max(1);
                                    or_km_cycle_root = Some(match or_km_cycle_root {
                                        None => cr,
                                        Some(prev) if prev == cr => cr,
                                        _ => 0,
                                    });
                                }
                            }
                        }
                    }
                    //
                    // GHI 対策: ループ子または経路依存な子の反証が寄与した場合は
                    // 親の反証も経路依存．init フェーズで蓄積した init_or_path_dep
                    // も考慮する(init で反証済みの子が MID ループには残らないため)．
                    // HandSet (v0.56.1): 全子反証時の disproof_hand 計算．
                    // 子 disproof_hand の交集合 (要素 min) を att_hand で上限クリップ．
                    // OR ノード反証: 既定では att_hand で保存(TT ヒット率最大化)．
                    let hand_to_store = if self.param_use_handset_combination && or_disproof_initialized {
                        let mut h = or_disproof_hand;
                        for k in 0..HAND_KINDS {
                            h[k] = h[k].min(att_hand[k]);
                        }
                        h
                    } else {
                        att_hand
                    };
                    if loop_child_count > 0 || init_or_path_dep {
                        // K-M (v0.55.23): main loop と init フェーズの cycle_root をマージ
                        let cr = or_km_cycle_root
                            .or(init_or_km_cycle_root)
                            .unwrap_or(0);
                        self.store_path_dep(
                            pos_key, hand_to_store,
                            INF, 0,
                            parent_nm_remaining, cr, true,
                        );
                    } else {
                        self.store(
                            pos_key, hand_to_store,
                            INF, 0,
                            parent_nm_remaining, pos_key as u32,
                        );
                    }
                    // スラッシング防止(main loop): init フェーズと同じ処理
                    if parent_nm_remaining < remaining {
                        self.store(
                            pos_key, att_hand, INF - 1, 1,
                            remaining, pos_key as u32,
                        );
                    }
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
                    self.path_set.remove(&full_hash);
                    self.path_len -= 1;
                    return;
                }

                // WPN スケールドサム: OR dn の DAG 二重計数補正 (AND pn と対称)
                // OR dn = max(child_dn) + (sum(child_dn) - max(child_dn)) >> WPN_GAMMA_SHIFT
                {
                    let sum_other = (current_dn as u64).saturating_sub(max_cdn as u64);
                    current_dn = (max_cdn as u64)
                        .saturating_add(sum_other >> WPN_GAMMA_SHIFT)
                        .min(INF as u64) as u32;
                }
                // SNDA 補正: 同一 source の子は DAG 合流の可能性
                // 重複グループの最大値のみ残し重複分を控除して過大評価を補正
                // floor: SNDA の過剰控除を防ぎ，単一最大子の dn を下限とする
                if snda_pairs.len() >= 2 {
                    let snda_result = snda_dedup(&mut snda_pairs, current_dn);
                    current_dn = snda_result.max(max_cdn);
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
                // WPN: max(cpn)，未証明子の数，pn 合計を追跡
                let mut max_cpn: u32 = 0;
                let mut unproven_count: u32 = 0;
                let mut sum_cpn: u64 = 0;
                // 診断 (v0.77.0): AND scan 内の proven_count を追跡
                let mut and_scan_proven_count: u32 = 0;
                // CD-WPN: 同一マスのドロップを1グループとして数える
                // cd_sq_min_pn[sq] = そのマスへの全ドロップの min(cpn) (グループ代表値)
                let mut cd_grouped_count: u32 = 0;
                let mut drop_squares_seen: u128 = 0;
                let mut cd_sq_min_pn: [u32; 81] = [INF; 81];

                // Tier 3 (v0.65.0): DelayedMoveList による同マス合駒 chain．
                // flag ON 時のみ構築．prev chain 上に未解決の手があれば child を skip．
                let dml_opt: Option<super::delayed_move_list::DelayedMoveList> =
                    if self.param_use_delayed_move_list && children.len() >= 2 {
                        let moves_vec: Vec<Move> = children.iter().map(|c| c.0).collect();
                        let dml = super::delayed_move_list::DelayedMoveList::build(
                            &moves_vec, /*or_node=*/false,
                        );
                        Some(dml)
                    } else {
                        None
                    };
                // Pre-compute is_resolved[i] for prev chain check: child i は cpn==0 または cdn==0 で resolved．
                // dml が None なら is_resolved も使わないため空のまま．
                let is_resolved_for_dml: Vec<bool> = if dml_opt.is_some() {
                    children.iter().map(|(_, _, child_pk, child_hand)| {
                        let (cpn, cdn, _) = self.look_up_pn_dn(
                            *child_pk, child_hand,
                            remaining.saturating_sub(1),
                        );
                        cpn == 0 || cdn == 0
                    }).collect()
                } else {
                    Vec::new()
                };

                for (i, &(ref m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    // Tier 3 (v0.65.0): prev chain 上に未解決手があれば skip．
                    // 次回 mid() iteration で prev が resolved になってから再訪問される．
                    if let Some(ref dml) = dml_opt {
                        if dml.has_unresolved_prev(i, |j| is_resolved_for_dml[j]) {
                            // 未解決 prev あり: この child は今 sweep ではスキップ．
                            // ただし pn/dn 集計には参加せず all_proved も false に保つ必要あり．
                            // (skip した child を「証明済み」扱いすると AND が誤って proven 判定される)
                            all_proved = false;
                            continue;
                        }
                    }
                    // v0.55.38: visit_history dominance check は identity loop と
                    // 同じパスで処理 (経路依存反証として store される)．
                    let is_loop_child = self.path_set.contains(&child_fh)
                        || self.is_dominated_in_path(child_pk, child_hand);
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
                        // AND ノードでは守備側が着手するため att_hand は不変 (child_hand = att_hand)．
                        let child_nm_rem = self.table.get_effective_disproof_info(
                            child_pk, child_hand,
                            remaining.saturating_sub(1),
                        ).map(|(r, _)| r).unwrap_or(0);
                        let parent_nm_remaining = propagate_nm_remaining(
                            child_nm_rem, remaining);
                        // K-M (v0.55.23): AND 反証の cycle_root を子から取得・伝播
                        let cycle_root: Option<u32> = if is_loop_child {
                            Some(child_fh as u32)
                        } else {
                            self.table.get_path_dep_cycle_root(child_pk, child_hand)
                        };
                        if let Some(cr) = cycle_root {
                            self.store_path_dep(
                                pos_key, att_hand, INF, 0,
                                parent_nm_remaining, cr, true,
                            );
                        } else {
                            // 反証駒最適化: 子の ProvenTT 反証駒 (DH_C ≥ att_hand) を AND-node に伝播．
                            // DH_C を保存することで将来 H ≤ DH_C のクエリが TT ヒット → 再探索削減．
                            let dh = self.table.get_disproof_hand(child_pk, child_hand);
                            self.store(
                                pos_key, dh, INF, 0,
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
                        // 診断: AND scan の proven child を counting
                        and_scan_proven_count += 1;
                        // 候補 D (v0.81.0): proven defender を bitmap に記録．
                        // 次回 visit 時に lookup miss (remaining mismatch 等) があっても
                        // 「この index は proven 確定」として selection から除外できる．
                        if self.param_use_and_proven_bitmap && i < 64 {
                            let entry = self.and_proven_bitmap.entry(pos_key).or_insert(0);
                            *entry |= 1u64 << i;
                        }
                        // cross-deduction は all_proved パスで実行される．
                        // VPN: 証明済み子は pn=0 で sum に影響しないため，
                        // child 選択ループもスキップして効率化する．
                        continue;
                    }

                    // 候補 D (v0.81.0): bitmap で proven 化済みとされている index は
                    // selection から除外 (lookup miss の保護)．
                    if self.param_use_and_proven_bitmap && i < 64 {
                        if let Some(&bm) = self.and_proven_bitmap.get(&pos_key) {
                            if (bm & (1u64 << i)) != 0 {
                                // 既に proven 化済と判定されている → スキップ
                                and_scan_proven_count += 1;
                                continue;
                            }
                        }
                    }

                    all_proved = false;

                    // melodic-cascading-otter (v0.69.0): path-aware DAG 補正．
                    // parent_map ベース (full_hash 一致のみ)．
                    // 注: 2026-05-21 試行で 29te 効果ゼロ確認．Plan B 移行予定．
                    let is_dag_child = self.find_dag_ancestor_fh(child_fh);

                    // WPN: max(cpn) を追跡し，未証明子をカウント
                    if cpn > max_cpn {
                        max_cpn = cpn;
                    }
                    if !is_dag_child {
                        unproven_count += 1;
                        sum_cpn += cpn as u64;
                    }
                    // CD-WPN: 同一マスのドロップは1グループとして数える
                    // グループ代表値 = 同一マスへのドロップの中で最小 cpn
                    if m.is_drop() {
                        let sq_idx = m.to_sq().index();
                        let sq_bit = 1u128 << sq_idx;
                        if drop_squares_seen & sq_bit == 0 {
                            drop_squares_seen |= sq_bit;
                            cd_grouped_count += 1;
                        }
                        if cpn < cd_sq_min_pn[sq_idx] {
                            cd_sq_min_pn[sq_idx] = cpn;
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
                    //
                    // 施策 A-4 (v0.24.50): 境界層 DN inflation
                    // remaining <= 2 かつ chain aigoma 検出時，chain drop の
                    // 初期 dn バイアスを大幅に inflate して argmin 選択から
                    // 外す．pn aggregation (children pn の sum) には影響
                    // しないため false proven は生成しない (soundness-safe
                    // な move ordering 強化のみ)．詳細 benchmarks.md §10.2．
                    let boundary_inflate =
                        remaining <= 2 && !self.chain_bb_cache.is_empty();
                    const BOUNDARY_CHAIN_MULT: u32 = 8;
                    let effective_cdn = if let Some(ksq) = chain_king_sq {
                        // チェーン AND(全 drop が chain): ドロップ優先，
                        // 外側ほど後回し．
                        if m.is_drop() {
                            let to = m.to_sq();
                            let dr = (to.row() as i8 - ksq.row() as i8)
                                .unsigned_abs() as u32;
                            let dc = (to.col() as i8 - ksq.col() as i8)
                                .unsigned_abs() as u32;
                            let dist = dr.max(dc);
                            // 内側(d=1)はバイアス0，外側は距離に比例
                            let dist_bias = dist.saturating_sub(1) * PN_UNIT;
                            let base_bias = dist_bias;
                            let bias = if boundary_inflate {
                                // 施策 A-4: chain drop に追加ペナルティ
                                #[cfg(feature = "tt_diag")]
                                { self.diag_a4_inflations += 1; }
                                base_bias
                                    .saturating_mul(BOUNDARY_CHAIN_MULT)
                                    .saturating_add(INTERPOSE_DN_BIAS)
                            } else {
                                base_bias
                            };
                            cdn.saturating_add(bias)
                        } else {
                            // 非合駒: 大きなバイアスで後回し
                            cdn.saturating_add(INTERPOSE_DN_BIAS)
                        }
                    } else if let Some(ksq) = mixed_chain_king_sq {
                        // 混合 AND (非 drop 応手 + chain drop 両方あり):
                        // 非 drop 応手を先に評価し，chain drop は距離ベースで
                        // 後回しにする．無駄合 chain が PV を膨らませる
                        // 問題の緩和 (非 drop 応手で反証/proof が速い場合，
                        // chain drop の探索を最小限にする)．
                        if m.is_drop() && saved_chain_bb.contains(m.to_sq()) {
                            let to = m.to_sq();
                            let dr = (to.row() as i8 - ksq.row() as i8)
                                .unsigned_abs() as u32;
                            let dc = (to.col() as i8 - ksq.col() as i8)
                                .unsigned_abs() as u32;
                            let dist = dr.max(dc);
                            // chain drop: INTERPOSE_DN_BIAS + 距離比例加算．
                            let dist_bias = dist.saturating_sub(1) * PN_UNIT;
                            let base_bias = INTERPOSE_DN_BIAS + dist_bias;
                            let bias = if boundary_inflate {
                                // 施策 A-4: 境界層で mixed chain drop も
                                // 8 倍 inflate
                                #[cfg(feature = "tt_diag")]
                                { self.diag_a4_inflations += 1; }
                                base_bias.saturating_mul(BOUNDARY_CHAIN_MULT)
                            } else {
                                base_bias
                            };
                            cdn.saturating_add(bias)
                        } else if m.is_drop() {
                            // 非 chain drop: 通常の INTERPOSE_DN_BIAS のみ
                            cdn.saturating_add(INTERPOSE_DN_BIAS)
                        } else {
                            // 非合駒: バイアスなし(優先)
                            cdn
                        }
                    } else if m.is_drop() {
                        // 非 chain AND での drop
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

                // 診断 (v0.77.0): AND scan の coverage を記録．
                // scan 完了後の proven_count / total_children を集計．
                {
                    let total = children.len() as u64;
                    let proven = and_scan_proven_count as u64;
                    self.diag_and_visit_count += 1;
                    self.diag_and_proven_sum += proven;
                    self.diag_and_total_sum += total;
                    if proven == 0 { self.diag_and_zero_proven += 1; }
                    if proven == total && total > 0 { self.diag_and_full_proven += 1; }
                }

                if proved_or_disproved {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_loop_break_proved += 1; }
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
                    self.path_set.remove(&full_hash);
                    self.path_len -= 1;
                    return;
                }

                // WPN (Weak Proof Number) / CD-WPN 計算:
                //
                // チェーン AND: CD-WPN スケールドサムを使用．
                //   同一マスへのドロップをグループ化し，グループ代表値 = min(cpn in group)．
                //   cross-deduce で同一マスの兄弟ドロップが一括証明されるため，
                //   グループのコストはそのマスで最も小さい cpn で代表される．
                //   CD-WPN = max(rep) + (sum(rep) - max(rep)) >> WPN_GAMMA_SHIFT
                //   旧式 max + (grouped_count-1)*PN_UNIT は非最大グループの
                //   子 pn 変化を伝播しない問題があった．
                //
                // 非チェーン AND: スケールドサム WPN を使用．
                //   pn = max(cpn) + (sum(cpn) - max(cpn)) >> WPN_GAMMA_SHIFT
                if chain_king_sq.is_some() && cd_grouped_count > 0 {
                    // グループ代表値の max/sum を drop_squares_seen のビット列から収集
                    let mut cd_max_group: u32 = 0;
                    let mut cd_sum_group: u64 = 0;
                    let mut bits = drop_squares_seen;
                    while bits != 0 {
                        let sq_idx = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let rep = cd_sq_min_pn[sq_idx];
                        cd_sum_group = cd_sum_group.saturating_add(rep as u64);
                        if rep > cd_max_group { cd_max_group = rep; }
                    }
                    let sum_other = cd_sum_group.saturating_sub(cd_max_group as u64);
                    current_pn = (cd_max_group as u64)
                        .saturating_add(sum_other >> WPN_GAMMA_SHIFT)
                        .min(INF as u64) as u32;
                } else if unproven_count > 0 {
                    let sum_other = sum_cpn.saturating_sub(max_cpn as u64);
                    current_pn = (max_cpn as u64)
                        .saturating_add(sum_other >> WPN_GAMMA_SHIFT)
                        .min(INF as u64) as u32;
                }

                // SNDA 補正: 同一 source の子は DAG 合流の可能性
                // 重複グループの最小値分を控除して過大評価を補正
                //
                // WPN 圧縮後に適用するため，控除量を制限する:
                // max_cpn を下限として保証し，WPN の加算分(count-1)のみを
                // 控除対象とする．これにより過大控除を防ぎつつ
                // DAG 合流の二重カウントを補正する．
                if snda_pairs.len() >= 2 {
                    let snda_result =
                        snda_dedup(&mut snda_pairs, current_pn);
                    current_pn = snda_result.max(max_cpn);
                }

                // AND ノード証明(全子が証明済み)
                if all_proved && current_pn == 0 {
                    // 証明駒を現在の持ち駒で制限
                    for k in 0..HAND_KINDS {
                        and_proof[k] =
                            and_proof[k].min(att_hand[k]);
                    }
                    if filter_applied {
                        self.store_proof_with_tag(
                            pos_key, and_proof, 0, 0,
                            super::entry::PROOF_TAG_FILTER_DEPENDENT,
                        );
                    } else {
                        self.store(
                            pos_key, and_proof, 0, INF,
                            REMAINING_INFINITE, pos_key as u32,
                        );
                    }
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
                    self.path_set.remove(&full_hash);
                    self.path_len -= 1;
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
            // 詰み手数 (mate_distance) 計算:
            // store_pn == 0 のとき proof → 子の distance から計算．
            // OR: min(proven children distances) + 1
            // AND: max(all children distances) + 1
            // 子の distance が取得できない (0) 場合は distance=0 (未知) とする．
            let mate_dist: u16 = if store_pn == 0 {
                let compute_dist = || -> u16 {
                    if or_node {
                        // best_idx の child (proven を代表) の distance + 1
                        let best_child = &children[best_idx];
                        let best_child_pk = best_child.1;
                        let best_child_hand = &best_child.3;
                        if let Some(cd) = self.table.look_up_mate_distance(
                            best_child_pk, best_child_hand)
                        {
                            return cd.saturating_add(1);
                        }
                        0
                    } else {
                        // AND: max(children distance) + 1
                        let mut max_d: u16 = 0;
                        let mut any_unknown = false;
                        for (_m, ch_pk, _ch_fh, ch_hand) in children.iter() {
                            if let Some(cd) = self.table.look_up_mate_distance(
                                *ch_pk, ch_hand)
                            {
                                if cd > max_d { max_d = cd; }
                            } else {
                                any_unknown = true;
                                break;
                            }
                        }
                        if any_unknown { 0 } else { max_d.saturating_add(1) }
                    }
                };
                compute_dist()
            } else {
                0
            };
            // (v0.24.71) tag propagation: pn=0 の場合，tag を決定して store．
            // AND: filter_applied なら FILTER_DEPENDENT
            // OR: best child の tag を継承
            let store_tag = if store_pn == 0 {
                if !or_node && filter_applied {
                    super::entry::PROOF_TAG_FILTER_DEPENDENT
                } else if or_node {
                    let best_child = &children[best_idx];
                    self.table.get_proof_tag(best_child.1, &best_child.3)
                } else {
                    super::entry::PROOF_TAG_ABSOLUTE
                }
            } else {
                super::entry::PROOF_TAG_ABSOLUTE // non-proof: tag irrelevant
            };
            if store_pn == 0 && store_tag != super::entry::PROOF_TAG_ABSOLUTE {
                profile_timed!(self, tt_store_ns, tt_store_count,
                    self.store_proof_with_tag(
                        pos_key, att_hand, best_move16, mate_dist, store_tag));
            } else {
                profile_timed!(self, tt_store_ns, tt_store_count,
                    self.store_with_best_move_and_distance(
                        pos_key, att_hand, store_pn, store_dn,
                        remaining, best_source, best_move16, mate_dist));
            }

            // TCA (Kishimoto & Müller 2008; Kishimoto 2010): 過小評価対策
            //
            // OR ノードでループ子(path 上の子)が存在する場合，
            // 兄弟の pn/dn が過小評価されている可能性がある．
            // 閾値を加算的に拡張し，兄弟をより深く探索させる．
            // 拡張は MID ループ出口と子閾値の両方に適用する:
            // - MID 出口のみ拡張すると，子閾値が元の値に束縛され
            //   ループが空転する(attempt 2 の教訓)．
            // - 子閾値も含め加算的に拡張することで進捗を保証する．
            // Plan B (v0.70.0): KH TCA inc_flag を propagate するため，
            // loop_child_count 検出時に self.inc_flag++ してから
            // self.inc_flag > 0 ならば閾値拡張する．
            // (既存 maou は loop_child_count を frame-local にしか使わなかった)．
            if self.param_use_kh_tca && loop_child_count > 0 {
                self.inc_flag = self.inc_flag.saturating_add(1);
                self.diag_tca_increments += 1;
            }
            let extend_tca = if self.param_use_kh_tca {
                self.inc_flag > 0
            } else {
                loop_child_count > 0
            };
            let (eff_pn_th, eff_dn_th) = if extend_tca {
                if self.param_use_kh_tca {
                    self.diag_tca_extends += 1;
                }
                (
                    pn_threshold
                        .saturating_add(pn_threshold / TCA_EXTEND_DENOM)
                        .saturating_add(PN_UNIT)
                        .min(INF - 1),
                    dn_threshold
                        .saturating_add(dn_threshold / TCA_EXTEND_DENOM)
                        .saturating_add(PN_UNIT)
                        .min(INF - 1),
                )
            } else {
                (pn_threshold, dn_threshold)
            };

            // melodic-cascading-otter 診断 (v0.71.0/3): per-ply trace．
            // `ply == param_trace_ply` で root_trace_interval ノード経過ごとに
            // children pn/dn と best_idx を eprintln! でダンプ．
            if self.param_root_trace
                && ply == self.param_trace_ply
                && self.nodes_searched >= self.root_trace_next
            {
                self.root_trace_iter += 1;
                let best_move_str = if best_idx < children.len() {
                    children[best_idx].0.to_usi()
                } else {
                    "?".to_string()
                };
                // 親 path 情報 (path[0..self.path_len-1]) を出力
                let path_str: String = (0..self.path_len.saturating_sub(1))
                    .map(|i| format!("ph={:#x}", self.path[i] & 0xFFFF))
                    .collect::<Vec<_>>()
                    .join(" → ");
                eprintln!(
                    "[trace ply={} iter={} nodes={}] or={} best_idx={} best={} pn={}/{} dn={}/{} children={}",
                    ply,
                    self.root_trace_iter,
                    self.nodes_searched,
                    or_node,
                    best_idx,
                    best_move_str,
                    current_pn, eff_pn_th,
                    current_dn, eff_dn_th,
                    children.len(),
                );
                if !path_str.is_empty() {
                    eprintln!("  path: {}", path_str);
                }
                // top-10 候補のみダンプ (children 全部だと長すぎる)
                let child_rem = remaining.saturating_sub(1);
                let mut child_dump: Vec<(u32, u32, &Move, usize)> = children.iter()
                    .enumerate()
                    .map(|(i, (m, _, cpk, ch))| {
                        let (cpn, cdn, _) = self.look_up_pn_dn(*cpk, ch, child_rem);
                        (cpn, cdn, m, i)
                    })
                    .collect();
                // best_idx (min cdn for AND, min cpn for OR) でソートして先頭 10
                if or_node {
                    child_dump.sort_by_key(|&(p, _, _, _)| p);
                } else {
                    child_dump.sort_by_key(|&(_, d, _, _)| d);
                }
                let take_n = if self.param_trace_full_children { child_dump.len() } else { 10 };
                for (cpn, cdn, m, i) in child_dump.iter().take(take_n) {
                    eprintln!("  [{:>2}] {} pn={} dn={}", i, m.to_usi(), cpn, cdn);
                }
                self.root_trace_next = self.nodes_searched
                    .saturating_add(self.root_trace_interval);
            }

            // 閾値チェック(TCA 拡張済み閾値を使用)
            if current_pn >= eff_pn_th
                || current_dn >= eff_dn_th
            {
                #[cfg(feature = "tt_diag")]
                { self.diag_loop_break_threshold += 1; }
                #[cfg(feature = "tt_diag")]
                if _diag_this_node && _diag_iteration <= 2 {
                    verbose_eprintln!(
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

            // === User-configurable GC (tt_gc_threshold) ===
            // set_tt_gc_threshold() で設定された閾値に基づく GC．
            // 100K ノード毎にチェックし，TT エントリ数が閾値を超えたら
            // 75% まで縮小する．Periodic GC(上記)よりも低い閾値で
            // きめ細かくメモリ制御する．デフォルト 0 = 無効．
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
            //   threshold = second_best + second_best/3 + PN_UNIT
            //             ≈ ceil(second_best * 4/3)
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
                let child_dn_th = eff_dn_th
                    .saturating_sub(current_dn)
                    .saturating_add(best_pn_dn.1)
                    .max(self.param_dn_floor_mult * PN_UNIT)
                    .min(INF - 1);
                // OR ノード pn 閾値: 1+ε trick (sibling_based)．
                //
                // 子の pn 予算を sibling_based(second_best + ε)に制限し，
                // 不正解手から正解手への切替を全 OR ノードで強制する．
                // 自然精度 epsilon (§10.2 方針A): divide-at-unit-scale を外し，
                // 除算の自然精度を活かす．second_best=3S のとき epsilon=28，
                // sibling_based=76(4.75S) となり ~19%/level の閾値余裕を確保する．
                let epsilon_or = second_best / self.effective_eps_denom() + PN_UNIT;
                let sibling_based_or = second_best.saturating_add(epsilon_or);
                let child_pn_th = sibling_based_or.max(2 * PN_UNIT).min(INF - 1);
                // 候補 G (v0.74.0): root (ply=0) では floor を引き上げて
                // 1 つの child に深く commit させる．param=0 で無効．
                let child_pn_th = if ply == 0 && self.param_root_child_pn_floor > 0 {
                    child_pn_th.max(self.param_root_child_pn_floor).min(INF - 1)
                } else {
                    child_pn_th
                };
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
                //
                // チェーン合駒 AND ノード(chain_king_sq あり)では pn_floor を
                // DN_FLOOR(=100)に引き上げる．dn のチェーン用キャップ外し
                // (§3 最適化)と同じ発想:
                // OR 親の sibling_based が 2〜5 と極端に小さい場合でも，
                // チェーン AND の子 OR に DN_FLOOR 以上の pn 予算を保証する．
                // 標準の pn_floor = eff_pn_th / 2 では eff_pn_th=2〜5 のとき
                // pn_floor=1〜2 となり，23応手への配分が不可能(閾値飢餓 §10.4)．
                // 自然精度 pn_floor (§10.2 方針A): 除算の自然精度 + 比率 2/3．
                // AND ノードの WPN カスケード縮退を緩和する．
                // eff_pn_th の 2/3 を最低保証することで，各 AND レベルでの
                // 閾値減衰を (1/2)^N → (2/3)^N に改善する．
                // 6 AND レベルで (2/3)^6 ≈ 0.088 vs (1/2)^6 ≈ 0.016 → 5.6倍．
                // u64 に昇格して乗算オーバーフローを防止する．
                let pn_floor_raw = ((eff_pn_th as u64 * self.param_pn_floor_numer as u64
                    / self.param_pn_floor_denom as u64) as u32).max(PN_UNIT);
                let dn_floor_param = self.param_dn_floor_mult * PN_UNIT;
                let pn_floor = if chain_king_sq.is_some() {
                    dn_floor_param.max(pn_floor_raw)
                } else {
                    pn_floor_raw
                };
                // 最低進捗保証: child_pn_th は最低でも best_child.pn + PN_UNIT を
                // 保証する．これにより eff_pn_th ≈ current_pn のとき
                // child_pn_th = best_child.pn となり mid() が即座に返る
                // ゼロ進捗パターンを防止する．
                let progress_floor = best_pn_dn.0.saturating_add(PN_UNIT);
                let child_pn_th = eff_pn_th
                    .saturating_sub(current_pn)
                    .saturating_add(best_pn_dn.0)
                    .max(pn_floor)
                    .max(progress_floor)
                    .min(INF - 1);
                // 自然精度 epsilon (§10.2 方針A): OR ノードと同じく depth-adaptive．
                let epsilon = second_best / self.effective_eps_denom() + PN_UNIT;
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
                let child_dn_th = if chain_king_sq.is_some() {
                    sibling_based.max(dn_floor_param).min(INF - 1)
                } else {
                    eff_dn_th
                        .min(sibling_based.max(dn_floor_param))
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
                        verbose_eprintln!(
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

            // 候補 C (v0.80.0): AND node exhaustive defender prove．
            // AND ノード (or_node=false) かつ proven_count < total_children なら
            // round-robin で未 proven defender を選択し直す．
            // best_idx を上書きすることで標準の min(cdn) 選択を override．
            if !or_node && self.param_use_exhaustive_and && children.len() > 1 {
                let child_rem_rr = remaining.saturating_sub(1);
                // 未 proven な children のインデックスを収集
                let mut unproven_idxs: Vec<usize> = Vec::with_capacity(children.len());
                for (i, &(_, _, ch_pk, ref ch_h)) in children.iter().enumerate() {
                    let (cpn, _, _) = self.look_up_pn_dn(ch_pk, ch_h, child_rem_rr);
                    if cpn != 0 && cpn < INF {
                        unproven_idxs.push(i);
                    }
                }
                if unproven_idxs.len() >= 2 {
                    let rr = (self.exhaustive_and_rr_counter as usize)
                        % unproven_idxs.len();
                    best_idx = unproven_idxs[rr];
                    self.exhaustive_and_rr_counter =
                        self.exhaustive_and_rr_counter.wrapping_add(1);
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
                let checks = self.generate_check_moves_cached(board);
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

            // 同一子連続選択によるスラッシング防止: 幾何的閾値増幅．
            //
            // 同一 best_idx が STAGNATION_LIMIT 回連続するたびに閾値を 2 倍し，
            // 1 回の mid() 呼び出しで収束に必要な予算を段階的に確保する．
            // pn/dn が微小変化する "slow-progress thrashing" を防ぐ:
            //   - 通常の stagnation 検出: pn/dn が完全に変化しない場合のみ発火
            //   - 本機構: 同一子を繰り返し選択するだけで増幅が発動
            // 上限 128 倍 (シフト 7 ビット) でメモリ使用量を抑制する．
            let (final_child_pn_th, final_child_dn_th) = if best_idx == prev_best_idx {
                same_child_iters = same_child_iters.saturating_add(1);
                let amp_shift = (same_child_iters / STAGNATION_LIMIT).min(7);
                if amp_shift > 0 {
                    let amp = 1u32 << amp_shift;
                    (child_pn_th.saturating_mul(amp).min(INF - 1),
                     child_dn_th.saturating_mul(amp).min(INF - 1))
                } else {
                    (child_pn_th, child_dn_th)
                }
            } else {
                same_child_iters = 0;
                (child_pn_th, child_dn_th)
            };
            // 案B: TT 事前チェック — mid() を呼ぶ前に子の TT 値を確認し，
            // 既に閾値を超えていればノードカウントを消費せずにスキップする．
            // mid() 内部の threshold exit (line ~2273) と同等だが，
            // nodes_searched を消費しない点が重要．
            // WorkingTT eviction 後の再発見サイクル (pn=INF → evict → 再探索)
            // を短絡させ，無駄な 1-node 消費を排除する．
            {
                let child_pk = children[best_idx].2;
                let child_rem = remaining.saturating_sub(1);
                let (pre_pn, pre_dn, _) =
                    self.look_up_pn_dn(child_pk, &children[best_idx].3, child_rem);
                if pre_pn >= final_child_pn_th || pre_dn >= final_child_dn_th {
                    board.undo_move(m, captured);
                    if remaining > 4 {
                        zero_progress_count += 1;
                        if zero_progress_count >= ZERO_PROGRESS_LIMIT {
                            break;
                        }
                    }
                    prev_best_idx = best_idx;
                    continue;
                }
            }
            let _pre_mid_nodes = self.nodes_searched;
            // SNDA: OR ノードなら prev_attacker_move を現在の手に設定して伝播する
            let _snda_saved_ml = if or_node {
                let s = self.prev_attacker_move;
                self.prev_attacker_move = m;
                s
            } else { Move(0) };
            self.mid(
                board,
                final_child_pn_th,
                final_child_dn_th,
                ply + 1,
                !or_node,
            );
            if or_node { self.prev_attacker_move = _snda_saved_ml; }
            _prev_nodes_used = self.nodes_searched - _pre_mid_nodes;

            // 子エントリの amount を更新(ノード消費があった場合のみ)
            if _prev_nodes_used > 0 {
                let spent = (_prev_nodes_used as u64).min(u16::MAX as u64) as u16;
                self.table.update_amount(
                    children[best_idx].2,
                    &children[best_idx].3,
                    spent,
                );
            }

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
            // 最適化: _prev_nodes_used == 0 なら mid() が即 return しており
            // pn/dn は不変なので look_up を省略して前回値を再利用する．
            {
                let (cpn_after, cdn_after) = if _prev_nodes_used == 0 {
                    (prev_best_pn, prev_best_dn)
                } else {
                    let (p, d, _) = self.look_up_pn_dn(
                        children[best_idx].2,
                        &children[best_idx].3,
                        remaining.saturating_sub(1),
                    );
                    (p, d)
                };
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
                            let penalty = tt_pn.saturating_sub(current_pn).max(PN_UNIT);
                            (base.saturating_add(penalty).min(INF - 1), current_dn)
                        } else {
                            let base = current_dn.max(tt_dn);
                            let penalty = tt_dn.saturating_sub(current_dn).max(PN_UNIT);
                            (current_pn, base.saturating_add(penalty).min(INF - 1))
                        };
                        // (v0.24.71) tag propagation for stagnation store
                        let stag_tag = if stag_pn == 0 {
                            if !or_node && filter_applied {
                                super::entry::PROOF_TAG_FILTER_DEPENDENT
                            } else if or_node {
                                let best_child = &children[best_idx];
                                self.table.get_proof_tag(best_child.1, &best_child.3)
                            } else {
                                super::entry::PROOF_TAG_ABSOLUTE
                            }
                        } else {
                            super::entry::PROOF_TAG_ABSOLUTE
                        };
                        if stag_pn == 0 && stag_tag != super::entry::PROOF_TAG_ABSOLUTE {
                            self.store_proof_with_tag(
                                pos_key, att_hand, best_move16, 0, stag_tag);
                        } else {
                            self.store_with_best_move(
                                pos_key, att_hand, stag_pn, stag_dn,
                                remaining, best_source, best_move16);
                        }
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
            }

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
                verbose_eprintln!(
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
            #[cfg(feature = "verbose")]
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
                #[cfg(feature = "tt_diag")]
                { self.diag_cd_guard_and_drop += 1; }
                let (cpn_after, cdn_after, _) = self.look_up_pn_dn(
                    children[best_idx].2,
                    &children[best_idx].3,
                    remaining.saturating_sub(1),
                );
                if cpn_after == 0 {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_cd_guard_child_proven += 1; }
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

                    // 候補 C (v0.24.59): multi-step cross_deduce．
                    //
                    // cross_deduce_children は solved_move と **同一マス** の
                    // 兄弟ドロップのみを対象とするが，solved_move の sub-tree
                    // 解決時に ProvenTT に蓄積された deeper chain step の proof
                    // が **異なるマス** のドロップにも適用できる場合がある:
                    //
                    // 例 (39te cliff, ply 25 AND):
                    //   - P*5g が解決 → Rx5g の sub-tree で Rx4g, Rx3g 等の
                    //     captured position が ProvenTT に蓄積される
                    //   - 異なるマスの drop P*4g → Rx4g → **同一 pos_key**
                    //     (ply 25 からの直接 capture と P*5g sub-tree 内の
                    //     capture は board 上同一)
                    //   - ProvenTT に proof が存在すれば prefilter で即証明可能
                    //
                    // 実装: unproven な drop children 全体に prefilter を
                    // re-trigger する．cross_deduce_children 直後は ProvenTT に
                    // 新規 entry が蓄積されている確率が最も高い timing であり，
                    // prefilter のヒット率が初回訪問時より向上する．
                    //
                    // 追加で新たに proven 化された child に対しては同一マスの
                    // cross_deduce + transitive closure を連鎖的に発火させ，
                    // 波及的な証明伝搬 (cascade) を行う．
                    //
                    // コスト: children 数 N × prefilter 1 回 (movegen + TT lookup)
                    // = 23 × ~1μs = ~23μs/call．cross_deduce 自体が per-solve
                    // 数百〜数千回しか発火しないため全体コストは微小．
                    let solved_sq = m.to_sq();
                    for j in 0..children.len() {
                        let (mj, _, child_pk_j, child_hand_j) = &children[j];
                        if !mj.is_drop() { continue; }
                        // 同一マスは cross_deduce_children が処理済み
                        if mj.to_sq() == solved_sq { continue; }
                        // 既に proven なら skip
                        let (cpn_j, _, _) = self.look_up_pn_dn(
                            *child_pk_j, child_hand_j,
                            remaining.saturating_sub(1),
                        );
                        if cpn_j == 0 { continue; }
                        // prefilter re-check (dummy and_proof，init 累積不要)
                        let mut dummy_proof = [0u8; HAND_KINDS];
                        let hit = self.try_prefilter_block(
                            board, *mj, child_hand_j, remaining,
                            &mut dummy_proof,
                        );
                        if hit {
                            #[cfg(feature = "tt_diag")]
                            { self.diag_multi_step_hits += 1; }
                            // 新たに proven 化された child の同一マス兄弟にも
                            // cross_deduce を連鎖発火
                            self.cross_deduce_children(
                                board, *mj, &children, remaining,
                            );
                        }
                    }
                }

                // 逆方向不詰共有 (v0.24.61): disproven な合駒子から
                // post-capture level の disproof を兄弟ドロップに伝搬する．
                //
                // cdn_after == 0 は「この合駒に対し攻方が詰ませられない」を
                // 意味する．攻方の全応手 (捕獲を含む) が不詰のため，捕獲後
                // 局面 (pc_pk, H_disproved) も不詰である．
                //
                // forward-chain 逆方向支配: H_disproved ≥_fc H_j のとき
                // (pc_pk, H_j) も不詰 → WorkingTT に disproof を格納して
                // 兄弟 j の MID 評価で捕獲応手を即座に省略可能にする．
                if cdn_after == 0 {
                    self.reverse_disproof_sharing(
                        board, m, &children, remaining,
                    );

                    // Multi-step 逆方向不詰共有 (v0.24.62):
                    //
                    // reverse_disproof_sharing は disproved_move と **同一マス** の
                    // 兄弟ドロップのみ対象．しかし disproved_move の sub-tree 探索中
                    // に WorkingTT に蓄積された deeper chain step の disproof が
                    // **異なるマス** のドロップにも適用できる場合がある:
                    //
                    // 例 (39te, ply 25 AND):
                    //   - P*5g が disproven → Rx5g の post-capture (pc_pk, H) が
                    //     disproven として WorkingTT に蓄積される
                    //   - 異なるマスの drop P*4g → Rx4g でも同じ pc_pk に至る場合，
                    //     reverse_disproof_sharing が hand 支配で disproof を伝搬可能
                    //
                    // 実装: disproven でない全 drop children に対して
                    // reverse_disproof_sharing を re-trigger する．
                    // reverse_disproof_sharing 直後は WorkingTT に新規 disproof
                    // entry が蓄積されている確率が最も高い timing．
                    let disproved_sq = m.to_sq();
                    for j in 0..children.len() {
                        let (mj, _, child_pk_j, child_hand_j) = &children[j];
                        if !mj.is_drop() { continue; }
                        // 同一マスは reverse_disproof_sharing が処理済み
                        if mj.to_sq() == disproved_sq { continue; }
                        // 既に disproven なら skip
                        let (_, cdn_j, _) = self.look_up_pn_dn(
                            *child_pk_j, child_hand_j,
                            remaining.saturating_sub(1),
                        );
                        if cdn_j == 0 { continue; }
                        // 異マスの child j に対して reverse_disproof_sharing を試行
                        let prev_hits = {
                            #[cfg(feature = "tt_diag")]
                            { self.diag_reverse_disproof_hits }
                            #[cfg(not(feature = "tt_diag"))]
                            { 0u64 }
                        };
                        self.reverse_disproof_sharing(
                            board, *mj, &children, remaining,
                        );
                        #[cfg(feature = "tt_diag")]
                        {
                            let new_hits = self.diag_reverse_disproof_hits - prev_hits;
                            self.diag_multi_step_reverse_disproof_hits += new_hits;
                        }
                    }
                }
            }


        }

        // パスから除去
        debug_assert_eq!(self.path[self.path_len - 1], full_hash);
        self.alpha_x_filter_active = save_alpha_x;
        self.path_set.remove(&full_hash);
        self.path_len -= 1;

        // Plan B (v0.70.0): KH TCA inc_flag を mid() exit で復元．
        // 本フレーム内で `++` した分は親に leak しない (KH 流の clamp)．
        // 注: 早期 return 経路は inc_flag を変更していないため復元不要．
        if self.param_use_kh_tca {
            self.inc_flag = self.inc_flag.min(orig_inc_flag);
        }
    }

    /// 施策 A-6 再評価 (v0.24.54): 境界層 PNS 責任転嫁．
    ///
    /// **注意**: v0.24.72 の施策 α 不採用および refutable disproof 機構
    /// (v0.24.75+) の導入により，本関数は現在の solve() 経路から呼び出されない．
    /// dead code (削除候補)．以下の記述は履歴としての参考．
    ///
    /// MID の AND 境界層 (`remaining <= 2 && chain_bb_cache 非空`) で呼び出され，
    /// 通常の MID 再帰の代わりに **小規模 arena での PNS** を起動する．
    ///
    /// # v0.24.51 失敗からの改善
    ///
    /// v0.24.51 では per-call 100K ノード予算 × 数百万のユニーク境界位置 =
    /// 数百億ノードの累積 work で無限ループ状態に陥った．v0.24.54 では:
    ///
    /// - グローバル呼出数上限 (`a6_boundary_pns_calls_remaining`, 初期 100) で
    ///   総呼出回数を制限 (呼出側で実施済み)
    /// - per-call 予算を 100K → 10K ノードに縮小
    /// - total work ≈ 100 calls × 10K nodes = 1M ノード (有限)
    ///
    /// # Soundness
    ///
    /// PNS の `pns_store_to_tt` は完全証明 (pn=0) / 確定反証 (dn=0) のみを
    /// TT に書き込む既存の validated な規則で動作する．PNS の結果は
    /// ABSOLUTE tag として扱われ，Strategy X の FILTER_DEPENDENT 系統とは
    /// 独立 (PNS は filter を適用しないため常に sound)．
    ///
    /// # 動作
    ///
    /// - 専用 10K arena + `self.max_nodes` 一時 override
    /// - `pns_main_with_arena(board, &mut arena)` を呼び出す．PNS root の
    ///   `or_node` は `board.turn == self.attacker` から動的決定される
    ///   (pns.rs の v0.24.51 変更が残存．boundary = 常に AND root のはず)
    /// - PNS 完了後 solver state を restore
    /// - 結果は TT 経由で親 MID に伝搬 (通常の look_up_pn_dn)
    pub(super) fn mid_via_pns_boundary(&mut self, board: &mut Board) {
        /// 境界層 PNS arena の最大ノード数 (v0.24.51 の 100K から縮小)．
        const BOUNDARY_ARENA_NODES: usize = 5_000;
        /// 境界層 PNS ノード予算 (solver.max_nodes の一時 override)．
        const BOUNDARY_NODE_BUDGET: u64 = 5_000;

        // solver state を save
        let saved_max_nodes = self.max_nodes;

        // PNS 予算を境界用に制限
        self.max_nodes = self
            .nodes_searched
            .saturating_add(BOUNDARY_NODE_BUDGET);

        // 専用 arena を allocate．Frontier Variant とは独立．
        let mut arena: Vec<PnsNode> =
            Vec::with_capacity(BOUNDARY_ARENA_NODES);

        // PNS 起動．root の or_node は pns.rs の v0.24.51 変更により
        // board.turn から動的決定される．境界層では常に AND root．
        let _pv = self.pns_main_with_arena(board, &mut arena);

        // solver state を restore
        self.max_nodes = saved_max_nodes;
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
    pub(super) fn heuristic_and_pn(&self, board: &Board, num_defenses: u32) -> u32 {
        let defender = board.turn;
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return num_defenses * PN_UNIT,
        };

        // 玉の安全な逃げ場をカウント(ビットボード一括判定)
        // compute_king_danger は X-ray(玉を除いた占有)を使うため，
        // 玉が移動した先で飛び駒に当たるケースも正しく検出する．
        let king_moves = attack::step_attacks(defender, PieceType::King, king_sq);
        let our_occ = board.occupied[defender.index()];
        let danger = board.compute_king_danger(defender, king_sq);
        let safe_escapes = (king_moves & !our_occ & !danger).count();

        // 拡張 heuristic_and_pn: 逃げ場に基づく連続スケーリング．
        // 実験 (twinkling-hatching-duckling, v0.65.2): KH の per-move 4 カテゴリ
        // 差別化に近づける目的で safe_escapes 重みを 1/2 → 1 に拡大．
        let base = if safe_escapes == 0 {
            // 逃げ場なし: 合駒・駒取りのみ → 詰みやすい(2/3 に割引)
            (num_defenses * 2 / 3).max(1) * PN_UNIT
        } else {
            // 逃げ場あり: 応手数ベース + 逃げ場 × S
            num_defenses * PN_UNIT + safe_escapes * PN_UNIT
        };
        base
    }

    /// OR 子ノード(攻め方局面)のヒューリスティック初期 pn を計算する(df-pn+)．
    ///
    /// 標準 df-pn では OR ノードの初期 pn=1 だが，これでは全ての
    /// OR ノードが等しく「1手で詰む可能性がある」と見積もられる．
    /// 実際は玉の逃げ場が多い局面ほど詰みにくく，追い詰めに多くの手を要する．
    ///
    /// AND 親ノードの sum(pn) に直接影響し，閾値配分の精度を向上させる．
    ///
    /// # v0.43.0 変更点
    ///
    /// pn 値域を右方向に拡大して σ_ln を改善する:
    ///
    /// 1. 上限を 512S (bucket 13) から 2048S (bucket 15) に引き上げ．
    /// 2. 開放空間検出を safe_escapes に応じて段階化:
    ///    - safe_escapes=4-5, pressured=0: 1024S (bucket 14)
    ///    - safe_escapes≥6, pressured=0: 2048S (bucket 15)
    /// 3. safe_escapes=7 を 6 から分離して独立した base 値を設定．
    /// 4. num_checks=1 の乗数を ×2 から ×4 に強化
    ///    (1手しか王手がない = 詰めにくさが大幅に増す)．
    /// OR 子ノードの初期 pn と safe_escapes を返す (v0.52.0)．
    /// safe_escapes は呼び出し側で `heuristic_or_dn(se, nc, final_pn)` の計算に使う．
    /// final_pn = 返り値.0 + edge_cost + sacrifice_boost など呼び出し側で加算する．
    ///
    /// # v0.54.0 変更点
    ///
    /// `pos_key` の 3 ビットを使って pn に位置依存ジッタを加える．
    /// 同一 (safe_escapes, num_checks) の全局面が同じ pn 値に集中して
    /// KL スパイク (bucket 12 への離散集積) を生じさせるため，
    /// `×(13..20)/16` = ±20% の 8 段階ジッタで分散させる．
    pub(super) fn heuristic_or_pn(&self, board: &Board, num_checks: u32, pos_key: u64) -> (u32, u32) {
        if num_checks == 0 {
            return (INF, 0); // 王手なし → 不詰(呼び出し側で処理済みのはず)
        }

        let defender = board.turn.opponent();
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return (PN_UNIT, 0), // 玉なし: safe_escapes=0
        };

        // 玉の安全な逃げ場をカウント(ビットボード一括判定)
        let king_moves = attack::step_attacks(defender, PieceType::King, king_sq);
        let def_occ = board.occupied[defender.index()];
        let danger = board.compute_king_danger(defender, king_sq);
        let safe_escapes = (king_moves & !def_occ & !danger).count();

        // --- 開放空間逃走検出(人間的枝刈り，段階化 v0.43.0) ---
        // 玉周辺(隣接8マス)への攻め駒の利きが皆無かつ逃げ場が多い場合，
        // 人間が「玉が広い方に逃げて捕まらない」と直感するのと同様に
        // pn を引き上げて探索優先度を下げる．
        // safe_escapes の大きさに応じて詰めにくさをより細かく区別する．
        let king_adjacent = king_moves & !def_occ; // 玉が移動可能なマス(自駒除外)
        let pressured = (king_adjacent & danger).count(); // 攻め方に利かれているマス数
        let adjacent_total = king_adjacent.count(); // 移動可能マス総数

        // num_checks >= 4 のみ開放空間検出を適用．
        // num_checks=1-3 は攻め方の選択肢が少ない局面で pn 過大評価を招くため対象外とし，
        // 下の escape_base 式で処理する (不詭め検出効率を優先)．
        // num_checks >= 4 は攻め方が豊富で TT 爆発防止が必要な局面に限定する．
        if adjacent_total >= 5 && pressured == 0 && safe_escapes >= 4 && num_checks >= 2 {
            let pn = match safe_escapes {
                4 | 5 => 1024 * PN_UNIT, // bucket 14: やや広い開放空間
                _ => 2048 * PN_UNIT,     // bucket 15: 6+ 逃げ場の完全開放空間
            };
            return (pn, safe_escapes);
        }

        // heuristic_or_pn: safe_escapes × num_checks による pn 初期値マッピング
        //
        // 基本方針:
        // - safe_escapes が多い → 追い詰めに手数を要する(pn↑)
        // - num_checks が少ない → 選択肢が少なく詰みにくい(pn↑)
        //
        // safe_escapes=1-2 (大多数の39手詰め局面) は直接マッピングで粒度を上げ，
        // pn 分布が bucket 5-6 に集中するスパイクを緩和する (Case B v0.37.0)．
        // safe_escapes=3+ は escape_base を指数的にスケールし，bucket 7〜15 へ分散する
        // (Case C v0.38.0, v0.43.0: σ_ln 拡大のための値域拡大)．
        let adjusted_pn = match safe_escapes {
            1 => {
                // 逃げ場 1: checks が多いほど詰みやすい → pn を下げる
                // v0.50.0: bucket 6-7 スパイク緩和のため全ケースを +1 bucket 分シフト
                if num_checks >= 4 { 3 * PN_UNIT / 2 }   // 1.5S = 24 (~bucket 4.6, 変化なし)
                else if num_checks >= 2 { 4 * PN_UNIT }   // 4S = 64 (bucket 7, was 3S=48 bucket 6)
                else { 8 * PN_UNIT }                      // 8S = 128 (bucket 8, was 4S=64 bucket 7)
            }
            2 => {
                // 逃げ場 2: v0.50.0: 2-way → 3-way 分岐で bucket 6-7 スパイク緩和
                if num_checks >= 4 { 4 * PN_UNIT }        // 4S = 64 (bucket 7, was 3S=48 bucket 6)
                else if num_checks >= 2 { 8 * PN_UNIT }   // 8S = 128 (bucket 8, was 5S=80 bucket 7)
                else { 16 * PN_UNIT }                     // 16S = 256 (bucket 9, was 5S=80 bucket 7)
            }
            _ => {
                // safe_escapes=0, 3+: escape_base × num_checks 係数
                // 3+ は指数的スケール (bucket 7/9/11/12/13/14) で σ_ln を拡大
                let escape_base = match safe_escapes {
                    0 => PN_UNIT,              //    1S (bucket 4)
                    3 => 8 * PN_UNIT,          //    8S (bucket 7)
                    4 => 32 * PN_UNIT,         //   32S (bucket 9)
                    5 => 128 * PN_UNIT,        //  128S (bucket 11)
                    6 => 256 * PN_UNIT,        //  256S (bucket 12)
                    7 => 512 * PN_UNIT,        //  512S (bucket 13, v0.43.0 分離)
                    _ => 1024 * PN_UNIT,       // 1024S (bucket 14, 8+逃げ場)
                };
                // num_checks=1 の乗数を ×2 に戻す (v0.43.0 で ×4 に変更したが
                // 不詭め局面での pn 過大評価を招いたため pre-v0.43.0 挙動に復元)
                if num_checks >= 8 { escape_base }
                else if num_checks >= 4 { escape_base + escape_base / 4 }
                else if num_checks >= 2 { escape_base + escape_base / 2 }
                else { escape_base * 2 }  // checks=1: ×2 (pre-v0.43.0)
            }
        };

        // pos_key の 3 ビットで 8 段階 (×0.8125..×1.25) のジッタを加える (v0.54.0)．
        // 同一 (se, nc) の全局面が同値になる離散スパイクを分散させる．
        let jitter = ((pos_key >> 17) & 7) as u32;
        let pn = (adjusted_pn * (13 + jitter) / 16).max(PN_UNIT).min(2048 * PN_UNIT);
        (pn, safe_escapes)
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
    pub(super) fn try_capture_tt_proof(
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
                self.store_board_with_hand(board, &proof, 0, INF, REMAINING_INFINITE, cap_pk as u32);
                return true;
            }
            board.undo_move(*check, captured);
        }
        false
    }

    /// キャッシュ付き王手生成(E2 最適化)．
    ///
    /// 局面ハッシュをキーとして `generate_check_moves` の結果をキャッシュし，
    /// 同一局面への再計算を回避する．`CheckCache` は内部可変性(UnsafeCell)により
    /// `&self` でアクセス可能にし，mid() のスタックフレーム最適化を阻害しない．
    #[inline]
    pub(super) fn generate_check_moves_cached(
        &self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        let hash = board.hash;
        if let Some(cached) = self.check_cache.get(hash) {
            let mut result = ArrayVec::new();
            for &m in cached.iter() {
                result.push(m);
            }
            return result;
        }
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        moves
    }

    /// ビットボード演算のみで詰み判定を行い，do_move/undo_move の
    /// オーバーヘッドを回避する(cshogi の mateMoveIn1Ply 相当)．
    pub(super) fn has_mate_in_1_with(
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
                REMAINING_INFINITE, pk as u32);
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
    pub(super) fn try_prefilter_block(
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

            // サマリキャッシュによる高速判定 (v0.24.64)．
            // TT の hand_hash クラスタ分散を迂回し O(1) で proof を検出．
            if let Some(min_ph) = self.pc_summary.lookup_proof(pc_pk) {
                if hand_gte_forward_chain(&pc_hand, min_ph) {
                    // pc_hand ≥_fc min_proof_hand → 証明済み
                    #[cfg(feature = "tt_diag")]
                    { self.diag_pc_summary_proof_hits += 1; }
                    // or_ph = child_hand (conservative)
                    self.table.store(
                        child_pk, *child_hand, 0, INF,
                        remaining.saturating_sub(1), child_pk as u32,
                    );
                    let adj = adjust_hand_for_move(block_move, child_hand);
                    for k in 0..HAND_KINDS {
                        and_proof[k] = and_proof[k].max(adj[k]);
                    }
                    proved = true;
                    board.undo_move(*cap_mv, cap_piece);
                    break;
                }
            }

            // メイン TT で捕獲後局面の証明を参照．
            //
            // ## 施策 X-N 候補 A-fix (v0.24.58): 2 段階 lookup + either-or or_ph 選択
            //
            // ### v0.24.56 退行の根本原因
            //
            // v0.24.56 で `neighbor_scan=false` → `true` の 1 行変更を
            // 試みた際，test_tsume_39te_ply24_mate15_regression が Unknown
            // に退行した．原因は `or_ph = pc_ph - X_cap` 後の
            // `or_ph.min(child_hand)` clamp が forward-chain substitution
            // マッチ時に unsound な proof を child_pk に store することだった:
            //
            // - pc_ph = [0, 5, 0, 0, 0, 0, 0]  (5 lances required)
            // - pc_hand = [0, 0, 0, 0, 0, 0, 5] (5 rooks, forward-chain match)
            // - X_cap = [1, 0, 0, 0, 0, 0, 0]  (captured pawn)
            // - or_ph (pre-clamp) = [0, 5, 0, 0, 0, 0, 0]
            // - child_hand = [0, 0, 0, 0, 0, 0, 5]
            // - clamped or_ph = [0, 0, 0, 0, 0, 0, 0] ← **unsound**
            //
            // stored claim: 「child_pk は空手で詰む」．しかし実際には
            // attacker が capture した後の pc_pk with [1,0,0,0,0,0,0] は
            // pc_ph = [0,5,...] を forward-chain で満たさず
            // (deficit_lance=5, a[6]=0, 5≤0 NO)．`neighbor_scan` を
            // enable すると forward-chain matching の頻度が急増し退行が
            // 顕在化した．v0.24.55 以前は own-cluster のみで
            // substitution match が稀だったため実害が観測されていなかった．
            //
            // ### A-fix 設計: 2 段階 lookup + 適応的 neighbor_scan + either-or
            //
            // baseline の clamp 方式は own-cluster match では安定動作して
            // おり，実害が観測されていない．問題は neighbor-cluster match
            // での forward-chain substitution との噛み合わせのみ．そこで
            // lookup を 2 段階に分離し，store 時の or_ph 計算方式を
            // own/neighbor で切り替える:
            //
            // 1. **Phase 1 (own cluster only)**: `neighbor_scan=false` で
            //    自クラスタを探索．hit した場合は baseline の clamp 方式
            //    (`or_ph = min(pc_ph - X_cap, child_hand)`) を使用．
            //    own-cluster match は baseline から安定しており，clamp の
            //    soundness ギャップも 29te / 39te を通じて実害が観測されて
            //    いないため踏襲する (wider dominance を維持)．
            //
            // 2. **Phase 2 (adaptive neighbor scan)**: Phase 1 miss 時のみ
            //    実行．`proven_has_other_hand_variant` で同一 pos_key の
            //    hand バリアントが自クラスタに存在する場合のみ発火させる
            //    (user suggestion: variant が無い場所は neighbor_scan が
            //    無効なので発火させない)．hit した場合は either-or 方式
            //    で sound な or_ph を選択:
            //    - **tight 候補**: `or_ph_tight = pc_ph - X_cap` (saturating)
            //      正当性: `H ≥_fc or_ph_tight → H + X_cap ≥_fc pc_ph`
            //      (forward-chain monotonicity + componentwise dominance)
            //    - **trivial 候補**: `child_hand` 自体
            //      正当性: `H ≥_fc child_hand → H + X_cap ≥_fc pc_hand
            //      ≥_fc pc_ph` (lookup premise + fc monotonicity)
            //    - `child_hand ≥_fc or_ph_tight` の fc-check で tight を
            //      使うか trivial fallback に落とすかを決定．両者とも sound．
            //
            //    旧 clamp 方式 `min(or_ph_tight, child_hand)` は両候補の
            //    いずれでもない中間 hand となり forward-chain 支配が崩れ
            //    unsound．A-fix は 2 候補からの排他選択で soundness を保証．
            let (ppn_own, _, _) = self.table.look_up(
                pc_pk, &pc_hand, pc_remaining, false,
            );
            let mut from_neighbor = false;
            let ppn = if ppn_own == 0 {
                ppn_own
            } else if self.table.proven_has_other_hand_variant(pc_pk, &pc_hand) {
                let (ppn_nb, _, _) = self.table.look_up(
                    pc_pk, &pc_hand, pc_remaining, true,
                );
                if ppn_nb == 0 {
                    from_neighbor = true;
                }
                ppn_nb
            } else {
                ppn_own
            };
            if ppn == 0 {
                // 捕獲後局面が証明済み → 合駒の OR ノードも証明
                let pc_ph = self.table.get_proof_hand(pc_pk, &pc_hand);
                // サマリキャッシュに proof hand を記録
                self.pc_summary.record_proof(pc_pk, &pc_ph);

                // OR ノードの証明駒: 捕獲で得る駒分を差し引く
                let cap_raw = cap_mv.captured_piece_raw();
                let mut or_ph_tight = pc_ph;
                if cap_raw > 0 {
                    let piece = Piece::from_raw_u8(cap_raw);
                    if let Some(pt) = piece.piece_type() {
                        let base_pt = pt.unpromoted().unwrap_or(pt);
                        if let Some(hi) = base_pt.hand_index() {
                            or_ph_tight[hi] = or_ph_tight[hi].saturating_sub(1);
                        }
                    }
                }

                // or_ph 計算方式の選択:
                // - own cluster hit: baseline の clamp 方式．
                //   own-cluster match は forward-chain 乖離が稀で，clamp
                //   由来の soundness ギャップは実害が観測されていない．
                //   clamp 後の or_ph は `min(or_ph_tight, child_hand)` で
                //   wider dominance を提供する．
                // - neighbor cluster hit: A-fix either-or 方式．
                //   forward-chain substitution match が多いため clamp は
                //   unsound になりうる．`child_hand ≥_fc or_ph_tight` の
                //   fc-check で sound な or_ph を選択する．
                let or_ph = if from_neighbor {
                    if hand_gte_forward_chain(child_hand, &or_ph_tight) {
                        or_ph_tight
                    } else {
                        #[cfg(feature = "tt_diag")]
                        { self.diag_prefilter_fc_reject += 1; }
                        *child_hand
                    }
                } else {
                    let mut or_ph = or_ph_tight;
                    for k in 0..HAND_KINDS {
                        or_ph[k] = or_ph[k].min(child_hand[k]);
                    }
                    or_ph
                };

                // 子 TT に証明エントリを格納(後続の look_up で再利用)．
                // or_ph は unclamped の sound proof_hand: H ≥_fc or_ph →
                // H + X_cap ≥_fc pc_ph (forward-chain monotonicity + or_ph +
                // X_cap ≥ pc_ph componentwise) → child_pk 証明可能．
                self.table.store(
                    child_pk, or_ph, 0, INF,
                    remaining.saturating_sub(1), child_pk as u32,
                );

                // AND 証明駒の更新 (baseline adj 方式を踏襲)
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
    /// children 内の証明済みドロップから兄弟ドロップを TT で証明する．
    ///
    /// `cross_deduce_deferred` と同等のロジックだが，`children` を読み取り専用で
    /// 参照し，TT にエントリを格納するのみ(children からの除去は行わない)．
    /// MID ループの次の collect フェーズで cpn=0 として検出される．
    #[inline(never)]
    pub(super) fn cross_deduce_children(
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
            mj.is_drop() && mj.to_sq() == target_sq && *mj != solved_move
        });
        if !has_siblings {
            #[cfg(feature = "tt_diag")]
            { self.diag_cd_no_siblings += 1; }
            return;
        }
        #[cfg(feature = "tt_diag")]
        { self.diag_cd_entered_main += 1; }

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
        #[cfg(feature = "tt_diag")]
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
                // 施策 X-N (v0.24.55): neighbor_scan を有効化．
                //
                // v0.24.54 までは `neighbor_scan: false` で自クラスタのみ
                // 検索していたが，chain aigoma の駒種変種は hand_hash の
                // Zobrist 混合後に隣接クラスタにも分散する．proof(-1) +
                // 歩 disproof(+1) の近傍走査 (tt.rs:425-442) を活用することで
                // hand_j に対する proof 発見率を向上させ，unique (pk, hand)
                // 組合せの N を削減する．
                //
                // オーバーヘッド: look_up_proven の neighbor loop は最大
                // HAND_KINDS=7 クラスタ走査．cross_deduce は per-solve で
                // 数百〜数千回しか発火しないため全体コストは微小．
                let (ppn, _, _) = self.table.look_up(pc_pk, &hand_j, pc_remaining, true);

                if ppn == 0 {
                    let pc_ph = self.table.get_proof_hand(pc_pk, &hand_j);
                    // サマリキャッシュに proof hand を記録
                    self.pc_summary.record_proof(pc_pk, &pc_ph);
                    let mut or_ph = pc_ph;
                    or_ph[hi_j] = or_ph[hi_j].saturating_sub(1);
                    for k in 0..HAND_KINDS {
                        or_ph[k] = or_ph[k].min(child_hand_j[k]);
                    }
                    // メイン TT に証明エントリを格納
                    self.table.store(
                        *child_pk_j, or_ph, 0, INF,
                        remaining.saturating_sub(1), *child_pk_j as u32,
                    );
                    #[cfg(feature = "tt_diag")]
                    { cross_count += 1; }
                }
            }
        }

        board.undo_move(solved_move, captured_by_block);

        #[cfg(feature = "tt_diag")]
        { self.diag_cross_deduce_hits += cross_count; }

        // 施策 X-N Candidate B (v0.24.57+): Sibling forward-chain transitive closure
        //
        // 既存 cross_deduce は pc_pk への TT lookup を通じて proof を転用する
        // が，neighbor_scan を有効化してもなお Zobrist hand_hash 混合により
        // クラスタが遠い hand variant が見逃される場合がある．
        //
        // 本 phase では sibling 間の **forward-chain domination** を直接
        // check し，TT lookup を介さず proof を伝搬する．
        //
        // 安全性:
        // - D_i が self TT で proven (cpn_i = 0) → D_i の OR node は勝ち筋
        //   (attacker captures → pc_pk with pc_hand_i が proven)
        // - pc_hand_j が pc_hand_i を forward-chain dominate →
        //   pc_pk with pc_hand_j も proven (domination)
        // - 故に D_j の capture 経路も勝ち → D_j の OR node も proven
        // - or_ph = siblings[j].hand (child の実際の hand) で store すれば
        //   conservative に正しい claim となる
        //
        // 計算量: sibling 数 N (典型 ~20) に対し O(N^2) の check × 最大 N
        // round (fixpoint)．per-call 最大 ~8K ops で微小．
        self.cross_deduce_transitive_closure(children, target_sq, remaining);
    }

    /// 施策 X-N Candidate B (v0.24.57): sibling 間 forward-chain transitive closure．
    ///
    /// cross_deduce_children の末尾で呼ばれ，既存の pc_pk TT lookup では
    /// 捕捉できなかった siblings を forward-chain domination グラフ経由で
    /// proof 伝搬する．
    ///
    /// 手順:
    /// 1. target_sq への drop sibling を収集
    /// 2. 各 sibling の自己 TT 状態を check (cpn == 0 かどうか)
    /// 3. 各 sibling の `pc_hand` (= child_hand + drop_piece) を計算
    /// 4. Fixpoint loop:
    ///    - 未証明 sibling j に対し，証明済み sibling i で
    ///      `hand_gte_forward_chain(pc_hand_j, pc_hand_i)` が真なら
    ///    - `siblings[j].hand` を or_ph として TT に store し proven 化
    #[inline(never)]
    pub(super) fn cross_deduce_transitive_closure(
        &mut self,
        children: &ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        target_sq: Square,
        remaining: u16,
    ) {
        // Step 1: target_sq への drop siblings を index 収集 (最大 HAND_KINDS=7)
        // (実際のほとんどの chain aigoma では 7 未満，本番ケースは 3〜5 程度)
        let mut indices: ArrayVec<usize, HAND_KINDS> = ArrayVec::new();
        for (i, (m, _, _, _)) in children.iter().enumerate() {
            if m.is_drop() && m.to_sq() == target_sq {
                if indices.try_push(i).is_err() {
                    break; // capacity 超過は無視 (通常発生しない)
                }
            }
        }
        if indices.len() < 2 {
            return; // 兄弟なしなら伝搬対象もなし
        }

        // Step 2-3: 各 sibling の自己 TT 状態と pc_hand を計算
        let mut proven: ArrayVec<bool, HAND_KINDS> = ArrayVec::new();
        let mut pc_hands: ArrayVec<[u8; HAND_KINDS], HAND_KINDS> = ArrayVec::new();
        let sub_rem = remaining.saturating_sub(1);
        for &idx in &indices {
            let (m, _, child_pk, child_hand) = &children[idx];
            let (cpn, _, _) = self.look_up_pn_dn(*child_pk, child_hand, sub_rem);
            let _ = proven.try_push(cpn == 0);
            // pc_hand = child_hand + drop_piece_i (capture 後の attacker hand)
            let mut pc_hand = *child_hand;
            if let Some(pt) = m.drop_piece_type() {
                if let Some(hi) = pt.hand_index() {
                    pc_hand[hi] = pc_hand[hi].saturating_add(1);
                }
            }
            let _ = pc_hands.try_push(pc_hand);
        }

        // Step 4: Fixpoint transitive closure
        // 最大 N 回の外側 loop で未証明 siblings を全部伝搬
        #[cfg(feature = "tt_diag")]
        let mut closure_transfers: u64 = 0;
        loop {
            let mut changed = false;
            for j in 0..indices.len() {
                if proven[j] { continue; }
                // j の pc_hand_j が i の pc_hand_i を forward-chain dominate するか
                for i in 0..indices.len() {
                    if i == j || !proven[i] { continue; }
                    if super::hand_gte_forward_chain(&pc_hands[j], &pc_hands[i]) {
                        // D_j の OR node を proven 化
                        let (_, _, child_pk_j, child_hand_j) = &children[indices[j]];
                        // conservative: or_ph = child_hand_j (必ず dominate 可能)
                        self.table.store(
                            *child_pk_j, *child_hand_j, 0, INF,
                            sub_rem, *child_pk_j as u32,
                        );
                        proven[j] = true;
                        changed = true;
                        #[cfg(feature = "tt_diag")]
                        { closure_transfers += 1; }
                        break;
                    }
                }
            }
            if !changed { break; }
        }

        #[cfg(feature = "tt_diag")]
        { self.diag_cross_deduce_hits += closure_transfers; }
    }

    /// 逆方向不詰共有 (v0.24.61): disproven な合駒子から post-capture level の
    /// disproof を兄弟ドロップに伝搬する．
    ///
    /// # 原理
    ///
    /// AND ノードの子 (守備方ドロップ) が disproven (cdn == 0) のとき，攻方の
    /// **全ての**応手が不詰であるため，捕獲応手も不詰である:
    ///
    /// - 守備方が駒 P_i を合駒 → 攻方が P_i を捕獲 → 捕獲後局面 (pc_pk, H_i) も不詰
    /// - H_i ≥_fc H_j (強い駒を捕獲した手駒は弱い駒を捕獲した手駒を支配)
    /// - **H_i で詰まない → H_j でも詰まない** (逆方向 forward-chain 支配)
    ///
    /// この disproof を WorkingTT に格納することで，兄弟ドロップの MID 評価時に
    /// 捕獲応手が即座に TT ヒットし，そのサブツリー探索を省略できる．
    ///
    /// # Soundness
    ///
    /// - WorkingTT に depth-limited disproof (`pn=INF, dn=0, remaining=R-2`) として
    ///   格納するため ProvenTT を汚染しない
    /// - IDS depth 切替時に `clear_working()` で自然に除去される
    /// - `hand_gte_forward_chain` の正当性: 攻方のリソースが多い H_i で不詰なら，
    ///   リソースが少ない H_j でも不詰
    #[inline(never)]
    pub(super) fn reverse_disproof_sharing(
        &mut self,
        board: &mut Board,
        disproved_move: Move,
        children: &ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS]),
            MAX_MOVES,
        >,
        remaining: u16,
    ) {
        let target_sq = disproved_move.to_sq();
        let pc_remaining = remaining.saturating_sub(2);
        if pc_remaining == 0 {
            return;
        }

        // 同一マスに兄弟ドロップがなければスキップ
        let has_siblings = children.iter().any(|(mj, _, _, _)| {
            mj.is_drop() && mj.to_sq() == target_sq && *mj != disproved_move
        });
        if !has_siblings {
            return;
        }

        let disproved_pt = match disproved_move.drop_piece_type() {
            Some(pt) => pt,
            None => return,
        };
        let disproved_hi = match disproved_pt.hand_index() {
            Some(hi) => hi,
            None => return,
        };

        // 合駒を実行し，攻方の捕獲手を探索
        let captured_by_block = board.do_move(disproved_move);
        let legal = movegen::generate_legal_moves(board);
        #[cfg(feature = "tt_diag")]
        let mut disproof_count: u64 = 0;

        for cap_mv in legal.iter().filter(|mv| {
            mv.to_sq() == target_sq && mv.captured_piece_raw() > 0
        }) {
            let cap_piece = board.do_move(*cap_mv);

            if !board.is_in_check(board.turn) {
                board.undo_move(*cap_mv, cap_piece);
                continue;
            }

            let pc_pk = position_key(board);
            // base_hand: disproved_move を捕獲した後の攻方持ち駒
            // (disproved_pt を含む = 強い方の手駒)
            let base_hand = board.hand[self.attacker.index()];
            board.undo_move(*cap_mv, cap_piece);

            // Soundness ガード: (pc_pk, base_hand) が実際に disproven かを
            // TT で確認する．multi-step ループから呼ばれた場合，disproved_move
            // が未反証 (cdn != 0) の可能性があり，その場合 base_hand の disproof
            // は未確認であるため伝搬は unsound になる．(v0.24.67)
            let (_, base_dn, _) = self.table.look_up(
                pc_pk, &base_hand, pc_remaining, false,
            );
            if base_dn != 0 {
                // (pc_pk, base_hand) が disproven でない → 伝搬スキップ
                continue;
            }

            // サマリキャッシュに disproof hand を記録
            self.pc_summary.record_disproof(pc_pk, &base_hand);

            // 各兄弟ドロップについて逆方向支配を確認
            for (mj, _, _, _) in children.iter() {
                if !mj.is_drop() || mj.to_sq() != target_sq {
                    continue;
                }
                if mj.to_move16() == disproved_move.to_move16() {
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

                // 兄弟 j を捕獲した場合の攻方持ち駒を計算
                let mut hand_j = base_hand;
                hand_j[disproved_hi] = hand_j[disproved_hi].saturating_sub(1);
                hand_j[hi_j] = hand_j[hi_j].saturating_add(1);

                // 逆方向支配: base_hand (強い) ≥_fc hand_j (弱い)
                // → base_hand で不詰なら hand_j でも不詰
                if super::hand_gte_forward_chain(&base_hand, &hand_j) {
                    // 既に disproof が存在するか確認 (無駄な store を回避)
                    let (_, existing_dn, _) = self.table.look_up(
                        pc_pk, &hand_j, pc_remaining, false,
                    );
                    if existing_dn == 0 {
                        continue; // 既に disproven
                    }
                    // post-capture level に depth-limited disproof を格納
                    self.table.store(
                        pc_pk, hand_j, INF, 0,
                        pc_remaining, pc_pk as u32,
                    );
                    #[cfg(feature = "tt_diag")]
                    { disproof_count += 1; }
                }
            }
        }

        board.undo_move(disproved_move, captured_by_block);

        #[cfg(feature = "tt_diag")]
        { self.diag_reverse_disproof_hits += disproof_count; }
    }

    // chain_inner_outer_propagation (v0.24.65): 試行不採用．
    //
    // 「内側チェーンマスの全ドロップが proved → 外側も proved」の論理は
    // unsound であった．AND ノードの各子 (守備方応手) は独立した防御手であり，
    // 内側マスの合駒が proved (攻方突破可能) でも外側マスの合駒は独立に
    // 評価される必要がある．内側 proved は「攻方がその合駒を捕獲して先に
    // 進める」を意味するが，外側の合駒は別の応手として独立に存在し，
    // 内側の結果で外側を省略する論理的根拠がない．
    //
    // test_tsume_39te_ply24_mate15_regression で Mate(15) → Mate(5) の
    // soundness 違反が発生したため revert．

    // ===========================================================
    // mid_v2 (Phase 3, v0.84.0): KH SearchImpl 風の本格的 mid 実装．
    // 既存 mid() と並走．`param_use_mid_v2=true` で有効化．
    // ===========================================================

    /// Phase 3: mid_v2 本体．DfPnSolver の method として実装．
    /// callback closure と `&mut self` の衝突を避けるため，mid_v2 内で
    /// 全ての TT/board アクセスを行う．
    pub(super) fn mid_v2(
        &mut self,
        board: &mut Board,
        thpn: u32,
        thdn: u32,
        ply: u32,
        or_node: bool,
        inc_flag: &mut u32,
        md_budget: u16,
    ) -> super::mid_v2::MidSearchResult {
        use super::mid_v2::{MidLocalExpansion, MidSearchResult};

        // Phase 15 visit count 診断 (board.hash がここで使えるが path_set は後で push される)．
        *self.mid_v2_visit_counts.entry(board.hash).or_insert(0) += 1;

        // Budget / timeout チェック
        if self.nodes_searched >= self.max_nodes {
            return MidSearchResult::new_unknown(PN_UNIT, PN_UNIT);
        }
        if self.nodes_searched & 0x3FF == 0 && self.is_timed_out() {
            self.timed_out = true;
            return MidSearchResult::new_unknown(PN_UNIT, PN_UNIT);
        }
        self.nodes_searched += 1;

        // Depth limit (PATH_CAPACITY)
        if (ply as usize) >= PATH_CAPACITY {
            // 千日手相当：depth 限界で unknown 返す
            return MidSearchResult::new_unknown(PN_UNIT, PN_UNIT);
        }

        // Phase 4: 現局面の path_set に追加 (cycle 検出用)
        let pos_key_self = position_key(board);
        let att_hand_self = board.hand[self.attacker.index()];
        let full_hash_self = board.hash;
        // 既に path 上 → 千日手 → 攻め方失敗 (OR) / 防御成功 (AND)
        if self.path_set.contains(&full_hash_self) {
            return if or_node {
                MidSearchResult::new_lose(0)
            } else {
                MidSearchResult::new_win(0)
            };
        }
        // Phase 4 fix: push/pop の対称性を保証．pushed フラグで判定．
        let pushed = if self.path_len < PATH_CAPACITY {
            self.path[self.path_len] = full_hash_self;
            self.path_pos_key[self.path_len] = pos_key_self;
            self.path_hand[self.path_len] = att_hand_self;
            self.path_len += 1;
            self.path_set.insert(full_hash_self);
            true
        } else {
            // path 容量オーバーで push skip — pop も skip すべき
            false
        };

        // children 生成 (OR: check moves, AND: legal moves)
        let moves_vec: Vec<Move> = if or_node {
            self.generate_check_moves_cached(board).into_iter().collect()
        } else {
            movegen::generate_legal_moves(board)
        };

        if moves_vec.is_empty() {
            // Terminal: OR (no checks available) → 攻め方失敗 = lose
            //          AND (no defenses) → 詰み = win (mate_distance=0)
            let result = if or_node {
                MidSearchResult::new_lose(0)
            } else {
                MidSearchResult::new_win(0)  // 既に mate
            };
            // Terminal proven は best_move + mate_distance 付きで store．
            if result.pn == 0 {
                self.store_with_best_move_and_distance(
                    pos_key_self, att_hand_self, 0, INF, REMAINING_INFINITE,
                    pos_key_self as u32, 0, 0);
            } else {
                self.store(pos_key_self, att_hand_self, result.pn, result.dn,
                    REMAINING_INFINITE, pos_key_self as u32);
            }
            // path pop (push されていた場合のみ)
            if pushed {
                self.path_len -= 1;
                self.path_set.remove(&full_hash_self);
            }
            return result;
        }

        // Phase 4: per-move heuristic で children 初期 (pn, dn) を計算．
        // OR node: edge_cost_or で pn 差別化 + dn=PN_UNIT
        // AND node: edge_cost_and で pn 差別化 + dn=PN_UNIT
        // 未訪問 (TT miss) のみ heuristic を適用．訪問済みなら TT 値を使用．
        let mut initial_results: Vec<MidSearchResult> = Vec::with_capacity(moves_vec.len());
        let child_remaining = (self.depth.saturating_sub(ply + 1)) as u16;
        // OR node: target king = opponent's king (mate される側)
        // AND node: target king = own king (mate される側 = attacker の opponent と同じ)
        let target_king_sq = board.king_square(self.attacker.opponent());

        // Phase 9: 子の md_budget は自分 - 1 (1 手消費)．u16::MAX なら飽和．
        let child_md_budget = md_budget.saturating_sub(1);
        for &m in &moves_vec {
            let captured = board.do_move(m);
            let pk = position_key(board);
            let hand = board.hand[self.attacker.index()];

            // Phase 21 (v1.0.0): KH IsSuperior 相当の visit_history dominance．
            // 祖先で同一 pos_key かつ attacker hand >= child hand なら，
            // attacker が多い資源でも詰めなかった → child は不詰．
            if self.is_dominated_in_path(pk, &hand) {
                board.undo_move(m, captured);
                initial_results.push(MidSearchResult::new_lose(0));
                continue;
            }

            let (pn_tt, dn_tt, _) =
                self.look_up_pn_dn_md_bounded(pk, &hand, child_remaining, child_md_budget);

            // Phase 16 (v0.95.0): KH `CheckObviousFinalOrNode` 移植．
            // AND parent (= !or_node) で child OR が 1 手詰の場合，pn=0 即 proven．
            // TT cache がない (= 初訪問) の場合のみ check．
            // depth-aware: 小問題のみで適用 (大問題ではコスト > 利益)．
            //
            // Phase 23 (G5) で gate 撤廃を試したが，深い問題 (29te) で 1 手詰 detection が
            // move ordering を変え，初回 solve が **31 手** PV を先に発見してしまう
            // (G4 が達成した「初回 29 手」を破壊)．初回ノードは 190K→105K と減るが
            // find_shortest でも 29 手へ回復できず PV 品質が退行するため gate は維持する．
            let mut mate_in_1 = false;
            let mut mate_move_for_or: Option<Move> = None;
            if !or_node && pn_tt == PN_UNIT && dn_tt == PN_UNIT && self.depth <= 25 {
                let checks = self.generate_check_moves_cached(board);
                if !checks.is_empty() {
                    let us = board.turn;
                    if let Some(mm) = board.mate_move_in_1ply(checks.as_slice(), us) {
                        mate_in_1 = true;
                        mate_move_for_or = Some(mm);
                    }
                }
            }
            // OR 局面自体を TT に store (PV 抽出が proven を確認できるように)．
            if mate_in_1 {
                self.store_with_best_move_and_distance(
                    pk, hand, 0, INF, REMAINING_INFINITE,
                    pk as u32,
                    mate_move_for_or.map(|m| m.to_move16()).unwrap_or(0),
                    1,
                );
            }

            board.undo_move(m, captured);

            // Phase 15 (v0.94.0): clamp を problem 規模で調整．
            let clamp_value = if self.depth <= 25 { 1 } else { PN_UNIT };
            let (init_pn, init_dn, is_first) = if mate_in_1 {
                // Phase 16: 1 手詰 detection 成功．child OR proven (pn=0, dn=INF, md=1)．
                (0, INF, false)
            } else if pn_tt == PN_UNIT && dn_tt == PN_UNIT {
                let (hp, hd) = if or_node {
                    let pn_h = if let Some(ksq) = target_king_sq {
                        super::edge_cost_or(m, ksq).max(clamp_value)
                    } else {
                        PN_UNIT
                    };
                    let (pn_kh, dn_kh) = super::init_pn_dn_or_kh(board, m, self.attacker);
                    let pn_h2 = if pn_kh > PN_UNIT { pn_h.saturating_add(pn_kh - PN_UNIT) } else { pn_h };
                    (pn_h2, dn_kh)
                } else {
                    let pn_h = super::edge_cost_and(m).max(clamp_value);
                    let (_pn_kh, dn_kh) = super::init_pn_dn_and_kh(board, m, self.attacker);
                    (pn_h, dn_kh)
                };
                (hp, hd, true)
            } else {
                (pn_tt, dn_tt, false)
            };
            let is_shallow = !is_first && !mate_in_1
                && self.is_shallow_remaining(pk, &hand, child_remaining);
            let r = if mate_in_1 {
                // Phase 16: child OR proven in 1 ply, md=1．
                super::mid_v2::MidSearchResult::new_win(1)
            } else {
                let mut r = MidSearchResult::new_unknown(init_pn, init_dn);
                r.is_first_visit = is_first;
                r.is_shallow = is_shallow;
                // Phase 17 (v0.96.0): TT-cached proven の場合，mate_distance も復元．
                // 旧バグ: cached pn=0 だが md=0 のまま → 親が min_proven_or_child で md=0+1=1 を返す．
                if init_pn == 0 && init_dn == INF {
                    let captured = board.do_move(m);
                    let pk2 = position_key(board);
                    let hand2 = board.hand[self.attacker.index()];
                    if let Some(stored_md) = self.table.look_up_mate_distance(pk2, &hand2) {
                        r.mate_distance = stored_md;
                    }
                    board.undo_move(m, captured);
                }
                r
            };
            initial_results.push(r);
        }

        // Phase 14 (v0.93.0): expansion を mid_expansion_stack に push．
        // KH の ExpansionStack 相当．eliminate_double_count で祖先を辿るために必要．
        let mut expansion = MidLocalExpansion::new_with_fh(
            or_node, moves_vec, initial_results, full_hash_self);

        // Phase 23 (G4): move_brief_eval を tie-break として使う．
        // KH `SearchResultComparer` 相当: phi 同点で `mp_[i].value` 比較．
        // king_sq は side-to-move の自玉 (KH `n.KingSquare()` 相当)．
        // OR node なら attacker，AND node なら defender (= attacker.opponent)．
        let own_king_color = if or_node { self.attacker } else { self.attacker.opponent() };
        if let Some(ksq) = board.king_square(own_king_color) {
            let evals: Vec<i32> = expansion
                .moves
                .iter()
                .map(|&m| super::move_brief_eval(m, ksq, board))
                .collect();
            expansion.set_move_evals(evals);
        }

        // Phase 21: parameterized deferred penalty + TCA gate
        expansion.set_deferred_penalty_denom(self.param_deferred_penalty_denom);
        expansion.set_deferred_penalty_floor(self.param_deferred_penalty_floor);
        // Phase 22: 1+ε threshold epsilon (PN_UNIT-scaled?)
        expansion.set_threshold_epsilon(self.param_threshold_epsilon);
        let deferred = expansion.deferred_count();
        if deferred > 0 && self.param_deferred_penalty_denom > 0 {
            self.diag_deferred_frames += 1;
            let penalty = deferred / self.param_deferred_penalty_denom as usize;
            self.diag_deferred_penalty_sum += penalty as u64;
        }
        // TCA gate 診断: is_shallow vs !is_first_visit の差分
        let has_old_shallow = expansion.does_have_old_child();
        let has_old_any = expansion.results.iter().any(|r| !r.is_first_visit);
        if has_old_shallow {
            self.diag_tca_shallow_fire += 1;
        } else if has_old_any {
            self.diag_tca_shallow_would_fire += 1;
        }
        if !self.param_tca_use_shallow_gate && !has_old_shallow && has_old_any {
            expansion.recompute_has_old_child_any_revisit();
        }

        self.mid_expansion_stack.push(expansion);
        self.mid_frame_moves.push(0);
        let stack_idx = self.mid_expansion_stack.len() - 1;

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc_flag = *inc_flag;
        let mut cur_thpn = thpn;
        let mut cur_thdn = thdn;

        let mut curr = self.mid_expansion_stack[stack_idx].current_result();
        if self.mid_expansion_stack[stack_idx].does_have_old_child() {
            *inc_flag = inc_flag.saturating_add(1);
        }
        if *inc_flag > 0 {
            extend_threshold_for_mid_v2_mode(&mut cur_thpn, &mut cur_thdn, &curr, self.param_tca_kh_clamp);
        }

        // Phase 5: proof 確定時の best_move を記録．OR proven なら proof_best_move を
        // 該当 child の move に．AND proven (=全 defender proven) は最後の defender．
        let mut proof_best_move: u16 = 0;
        // Phase 6: mate_distance 伝播．OR proven via child → child.mate_distance + 1
        let mut proof_mate_distance: u16 = 0;

        while curr.pn < cur_thpn && curr.dn < cur_thdn {
            if self.nodes_searched >= self.max_nodes { break; }
            if self.mid_expansion_stack[stack_idx].empty() { break; }

            let best_move = self.mid_expansion_stack[stack_idx].best_move();
            let is_first = self.mid_expansion_stack[stack_idx].front_is_first_visit();
            // mid_frame_moves に記録 (DAG correction の branch_move 同定用)．
            self.mid_frame_moves[stack_idx] = best_move.to_move16();

            let (child_thpn, child_thdn) =
                self.mid_expansion_stack[stack_idx].front_pn_dn_thresholds(cur_thpn, cur_thdn);

            let captured = board.do_move(best_move);

            // Phase 14: DAG correction．child の parent が祖先 (immediate parent
            // 以外) と一致するなら，その祖先の expansion の branch_move sum_mask を
            // off に切替．KH `EliminateDoubleCount` + `ResolveDoubleCountIfBranchRoot` 移植．
            if self.param_use_dag_correction {
                let child_pk = position_key(board);
                let child_hand = board.hand[self.attacker.index()];
                self.eliminate_double_count_mid_v2(
                    stack_idx, board.hash, child_pk, &child_hand, full_hash_self);
            }

            let child_result = if is_first {
                let initial = *self.mid_expansion_stack[stack_idx].front_result();
                if *inc_flag > 0 { *inc_flag -= 1; }
                // Phase 16: initial.is_final() なら 1 手詰 detection 等で確定済み．
                if initial.is_final()
                    || initial.pn >= child_thpn
                    || initial.dn >= child_thdn
                {
                    initial
                } else {
                    self.mid_v2(board, child_thpn, child_thdn, ply + 1, !or_node, inc_flag,
                                child_md_budget)
                }
            } else {
                // Phase 15 (v0.94.0): TT-hit (is_first=false) child の最適化．
                // - cached が is_final (pn=0 or dn=0): TT 値をそのまま使う (recurse 不要)．
                // - cached.pn/dn が threshold 超過: 同様に skip．
                let cached = *self.mid_expansion_stack[stack_idx].front_result();
                if cached.is_final()
                    || cached.pn >= child_thpn
                    || cached.dn >= child_thdn
                {
                    cached
                } else {
                    self.mid_v2(board, child_thpn, child_thdn, ply + 1, !or_node, inc_flag,
                                child_md_budget)
                }
            };

            board.undo_move(best_move, captured);

            // KH multi_pv=1 相当: child.phi(or_node)==0 で immediate proof/refutation．
            let child_phi = child_result.phi(or_node);
            let child_delta = child_result.delta(or_node);
            if child_phi == 0 {
                // Phase 19: OR node で child が proven だが md > budget → block して次の child へ
                if or_node && md_budget != u16::MAX && child_result.mate_distance > 0 {
                    let effective_md = child_result.mate_distance.saturating_add(1);
                    if effective_md > md_budget {
                        let blocked = super::mid_v2::MidSearchResult::new_lose(
                            child_result.mate_distance);
                        self.mid_expansion_stack[stack_idx].update_best_child(blocked);
                        curr = self.mid_expansion_stack[stack_idx].current_result();
                        cur_thpn = orig_thpn;
                        cur_thdn = orig_thdn;
                        if *inc_flag > 0 {
                            extend_threshold_for_mid_v2_mode(&mut cur_thpn, &mut cur_thdn, &curr, self.param_tca_kh_clamp);
                        } else if *inc_flag == 0 && orig_inc_flag > 0 {
                            break;
                        }
                        continue;
                    }
                }
                if or_node {
                    proof_best_move = best_move.to_move16();
                    proof_mate_distance = child_result.mate_distance.saturating_add(1);
                }
                curr = if or_node {
                    super::mid_v2::MidSearchResult::new_win(proof_mate_distance)
                } else {
                    super::mid_v2::MidSearchResult::new_lose(0)
                };
                break;
            }
            let _ = child_delta;

            self.mid_expansion_stack[stack_idx].update_best_child(child_result);
            curr = self.mid_expansion_stack[stack_idx].current_result();

            // TCA threshold rollback + re-extend
            cur_thpn = orig_thpn;
            cur_thdn = orig_thdn;
            if *inc_flag > 0 {
                extend_threshold_for_mid_v2_mode(&mut cur_thpn, &mut cur_thdn, &curr, self.param_tca_kh_clamp);
            } else if *inc_flag == 0 && orig_inc_flag > 0 {
                break;
            }
        }

        *inc_flag = (*inc_flag).min(orig_inc_flag);

        // Store final result to TT
        // Phase 6: proven 時 best_move + mate_distance を store．
        let remaining = (self.depth.saturating_sub(ply)) as u16;
        if curr.pn == 0 {
            // Phase 12 (v0.91.0) 診断 (verbose feature 時のみ): AND DML 漏れ検出．
            // Phase 12 修正後はトリガしないはず．
            // OR proven: proof_best_move + proof_mate_distance を使用
            // AND proven (all children proven): 全 children から max mate_distance を取得．
            //   defender は max-resistance を選ぶ前提で AND の mate_distance = max+1．
            //   best_move は AND では「any defender」(extract_pv が iterate するため不要)．
            let (bm, md) = if or_node && proof_best_move != 0 {
                // Phase 23 (display fix): main loop で最後に proven 化した child だけでなく，
                // 既に proven 済みの兄弟 child のうち最小 mate_distance のものを比較し，
                // shorter mate があればそちらを採用する．G4 で sort 順が変わった結果，
                // root では最初に proven 化した枝が **31 手経路**で，他枝に **29 手経路**が
                // ある状況が頻発した．TT には 29 手 entries が書かれるため
                // extract_pv が PV=29 を取れていたが，root の `result.mate_distance` のみ
                // 31 のまま残り PV 長と齟齬していた．これを解消する．
                let (alt_bm, alt_md) =
                    self.mid_expansion_stack[stack_idx].min_proven_or_child();
                if alt_bm != 0 && alt_md > 0 && alt_md < proof_mate_distance {
                    (alt_bm, alt_md)
                } else {
                    (proof_best_move, proof_mate_distance)
                }
            } else if !or_node {
                // Phase 11 (v0.90.0): AND proven は max-resistance defender を選ぶ．
                self.mid_expansion_stack[stack_idx].max_resistance_defender()
            } else {
                // OR proven without main loop iteration (= initial children に既に proven 含む)．
                // Phase 17 (v0.96.0): expansion から min mate_distance を求める．
                // 旧: (0, proof_mate_distance=0) で md=0 になる soundness 違反．
                self.mid_expansion_stack[stack_idx].min_proven_or_child()
            };
            if md_budget != u16::MAX && md > md_budget {
                if pushed {
                    self.path_len -= 1;
                    self.path_set.remove(&full_hash_self);
                }
                // Phase 14: stack pop on early return．
                self.mid_expansion_stack.pop();
                self.mid_frame_moves.pop();
                return super::mid_v2::MidSearchResult::new_unknown(PN_UNIT, PN_UNIT);
            }
            curr.mate_distance = md;
            self.store_with_best_move_and_distance(
                pos_key_self, att_hand_self, 0, INF, REMAINING_INFINITE,
                pos_key_self as u32, bm, md);
        } else if curr.dn == 0 {
            // Phase 19: budget-bounded search で proven 局面の dn=0 は
            // confirmed disproof ではなく disproven_len 更新
            if md_budget != u16::MAX && md_budget <= 1000
                && self.table.has_proof(pos_key_self, &att_hand_self)
            {
                self.table.update_disproven_len(pos_key_self, &att_hand_self, md_budget);
            } else {
                self.store(pos_key_self, att_hand_self, INF, 0, REMAINING_INFINITE, pos_key_self as u32);
            }
        } else {
            self.store(pos_key_self, att_hand_self, curr.pn, curr.dn, remaining, pos_key_self as u32);
        }

        // path pop (push されていた場合のみ)
        if pushed {
            self.path_len -= 1;
            self.path_set.remove(&full_hash_self);
        }

        // Phase 14: expansion stack pop．
        self.mid_expansion_stack.pop();
        self.mid_frame_moves.pop();

        curr
    }

    /// Phase 14 (v0.93.0): EliminateDoubleCount mid_v2 版．
    ///
    /// 現フレーム (stack_idx) の best_move を実行した直後 (board.hash == child_fh) に呼ぶ．
    /// child の parent (parent_map から取得) が現在の祖先フレーム (mid_expansion_stack)
    /// の position_fh と一致するなら，その祖先フレームの分岐 move の sum_mask を
    /// off に切替えて double-counting を防ぐ．
    ///
    /// KH `expansion_stack.hpp::EliminateDoubleCount` +
    /// `local_expansion.hpp::ResolveDoubleCountIfBranchRoot` 移植．
    pub(super) fn eliminate_double_count_mid_v2(
        &mut self,
        stack_idx: usize,
        child_fh: u64,
        child_pos_key: u64,
        child_hand: &[u8; HAND_KINDS],
        immediate_parent_fh: u64,
    ) {
        // parent_map + parent_meta に or_insert．既存 entry は上書きしない．
        self.parent_map.entry(child_fh).or_insert(immediate_parent_fh);
        self.parent_meta.entry(child_fh).or_insert((child_pos_key, *child_hand));

        // KH `FindKnownAncestor` 移植 (`double_count_elimination.hpp:102-149`)．
        // 祖先 chain を walk しながら pn/dn の divergence を track．divergence が
        // kAncestorSearchThreshold (= 3 * PN_UNIT) を超えたら double-count とみなさない．
        // Phase 14: KH は kAncestorSearchThreshold = 3 * kPnDnUnit．
        // maou で sweep した結果，PN_UNIT (16) が 29te で最適 (-7%)．
        const ANCESTOR_SEARCH_THRESHOLD: u32 = PN_UNIT;
        const MAX_DAG_LOOKBACK: usize = 16;

        let mut last_pn = u32::MAX;
        let mut last_dn = u32::MAX;
        let mut pn_flag = true;
        let mut dn_flag = true;
        let mut cur_fh = child_fh;
        // or_node alternation: child is opposite of self (immediate_parent).
        // self.attacker は固定なので or_node 切替は stack frame の or_node から逆推する必要．
        // 簡易化: stack_idx + 1 が child の or_node で alternates back．
        // 実用上は anc_idx == stack_idx - dist で or_node = (stack[anc_idx].or_node)．

        for _step in 0..MAX_DAG_LOOKBACK {
            let parent_fh = match self.parent_map.get(&cur_fh) {
                Some(&p) => p,
                None => return,
            };
            // cur の pn/dn を TT lookup (divergence check 用)．
            let (cur_pk, cur_hand) = match self.parent_meta.get(&cur_fh) {
                Some(&v) => v,
                None => return,
            };
            let (cur_pn, cur_dn, _) =
                self.look_up_pn_dn_impl(cur_pk, &cur_hand, REMAINING_INFINITE, false);

            // 初訪問チェック: step=0 で parent == immediate_parent なら通常 expand．
            if cur_fh == child_fh && parent_fh == immediate_parent_fh {
                return;
            }

            // divergence check．
            if cur_dn > last_dn.saturating_add(ANCESTOR_SEARCH_THRESHOLD) {
                dn_flag = false;
            }
            if cur_pn > last_pn.saturating_add(ANCESTOR_SEARCH_THRESHOLD) {
                pn_flag = false;
            }

            // 祖先 stack に parent_fh があるか?
            for anc_idx in (0..stack_idx).rev() {
                if self.mid_expansion_stack[anc_idx].position_fh() == parent_fh {
                    let branch_or_node = self.mid_expansion_stack[anc_idx].or_node_value();
                    // KH: OR node なら dn の double-count，AND node なら pn の double-count．
                    let allowed = (branch_or_node && dn_flag) || (!branch_or_node && pn_flag);
                    if allowed {
                        let branch_move = self.mid_frame_moves[anc_idx];
                        if branch_move != 0 {
                            self.mid_expansion_stack[anc_idx]
                                .reset_sum_mask_for_move(branch_move);
                        }
                    }
                    return;
                }
            }
            if parent_fh == cur_fh { return; }
            if !pn_flag && !dn_flag { return; }
            last_pn = cur_pn;
            last_dn = cur_dn;
            cur_fh = parent_fh;
        }
    }

    /// mid_v2 用 entry point: solve() から呼ばれる．
    /// inc_flag を 0 で初期化し，root mid_v2 を呼ぶ．
    /// md_budget = u16::MAX (制約なし)．
    pub fn solve_v2(&mut self, board: &mut Board) -> super::mid_v2::MidSearchResult {
        self.solve_v2_with_budget(board, u16::MAX)
    }

    /// Phase 9 (v0.88.0): md_budget 制約付き solve_v2．
    ///
    /// `md_budget` は root から見た詰みまでの許容手数．KH SearchEntry の
    /// `MateLen len` 相当．find_shortest IDS で D-2 ずつ締めて呼ぶ．
    pub fn solve_v2_with_budget(
        &mut self, board: &mut Board, md_budget: u16,
    ) -> super::mid_v2::MidSearchResult {
        self.attacker = board.turn;
        self.start_time = std::time::Instant::now();
        self.path_len = 0;
        self.path_set.clear();
        let saved_dag = self.param_use_dag_correction;
        self.param_use_dag_correction = true;
        self.parent_map.clear();
        self.parent_meta.clear();
        self.mid_expansion_stack.clear();
        self.mid_frame_moves.clear();
        self.mid_v2_visit_counts.clear();

        // Phase 22: KH SearchEntry 風 root-level IDS．
        // 初期 thpn=thdn=PN_UNIT で開始，各反復で thresholds *= 1.7+1 に拡張．
        // KH `komoring_heights.cpp::NextPnDnThresholds`．
        // param_root_ids_enable=false で INF-1 直接 (旧挙動)．
        let result = if self.param_root_ids_enable {
            let mut thpn: u32 = PN_UNIT;
            let mut thdn: u32 = PN_UNIT;
            let mut inc_flag = 0u32;
            let mut last = super::mid_v2::MidSearchResult::new_unknown(PN_UNIT, PN_UNIT);
            loop {
                inc_flag = 0;
                let r = self.mid_v2(board, thpn, thdn, 0, true, &mut inc_flag, md_budget);
                last = r;
                if r.pn == 0 || r.dn == 0 {
                    break;
                }
                if self.nodes_searched >= self.max_nodes || self.timed_out {
                    break;
                }
                // Both at INF means overflow — shouldn't happen but be safe
                if r.pn >= INF - 1 && r.dn >= INF - 1 {
                    break;
                }
                // Multiply thresholds by 1.7 (KH formula)
                let next_pn = ((r.pn as f64) * 1.7) as u32 + 1;
                let next_dn = ((r.dn as f64) * 1.7) as u32 + 1;
                let new_thpn = thpn.max(next_pn).min(INF - 1);
                let new_thdn = thdn.max(next_dn).min(INF - 1);
                if new_thpn == thpn && new_thdn == thdn {
                    // Cannot grow further; force INF
                    thpn = INF - 1;
                    thdn = INF - 1;
                } else {
                    thpn = new_thpn;
                    thdn = new_thdn;
                }
            }
            last
        } else {
            let mut inc_flag = 0u32;
            self.mid_v2(board, INF - 1, INF - 1, 0, true, &mut inc_flag, md_budget)
        };
        self.param_use_dag_correction = saved_dag;
        result
    }

    /// mid_v2 で proven された局面から PV を抽出する．
    /// 既存 `extract_pv_limited` を流用 (TT の best_move を使う)．
    /// 返り値: (`MidSearchResult`, PV as `Vec<Move>`)
    ///
    /// Phase 23 (display fix): root の `mate_distance` を実 PV 長と一致させる．
    /// mid_v2 main loop は最初に proven 化した child のみで md を確定するため，
    /// TT walk が見つけた shorter mate と齟齬する場合があった．PV 長を真値とする．
    pub fn solve_v2_with_pv(&mut self, board: &mut Board) -> (super::mid_v2::MidSearchResult, Vec<Move>) {
        let mut result = self.solve_v2(board);
        let pv = if result.pn == 0 {
            let pv = self.extract_pv_limited(board, 100_000);
            result.mate_distance = pv.len() as u16;
            pv
        } else {
            Vec::new()
        };
        (result, pv)
    }

    /// Phase 21: mid_v2 ベースの solve を TsumeResult 互換で返す．
    /// 既存テストの `solver.solve()` → `solver.solve_via_v2()` に機械的に置換可能．
    pub fn solve_via_v2(&mut self, board: &mut Board) -> TsumeResult {
        self.table.clear();
        self.nodes_searched = 0;
        self.timed_out = false;
        self.max_remaining_map.clear();
        let (result, pv) = if self.find_shortest {
            self.solve_v2_find_shortest(board)
        } else {
            self.solve_v2_with_pv(board)
        };
        if result.pn == 0 {
            if pv.is_empty() {
                TsumeResult::CheckmateNoPv { nodes_searched: self.nodes_searched }
            } else {
                TsumeResult::Checkmate { moves: pv, nodes_searched: self.nodes_searched }
            }
        } else if result.dn == 0 {
            TsumeResult::NoCheckmate { nodes_searched: self.nodes_searched }
        } else {
            TsumeResult::Unknown { nodes_searched: self.nodes_searched }
        }
    }

    /// Phase 23 (G1): KH `SearchMainLoop` 風の iterative deepening find_shortest．
    ///
    /// KH `komoring_heights.cpp::SearchMainLoop` (L185-254) 移植．
    ///
    /// 1. `md_budget = u16::MAX` で初回 solve → 任意の proof (mate_distance = D₀)
    /// 2. 反復: `new_budget = best_d - 2` で `solve_v2_with_budget` を呼び，
    ///    shorter proof を探す．proven かつ shorter なら更新．proven 失敗または
    ///    shorter にならなければ終了．
    ///
    /// `solve_v2_with_budget` 側は md_budget を mid_v2 に貫通させ，
    /// (a) `look_up_pn_dn_md_bounded` で stored_md > budget の proven を defer，
    /// (b) `child_result.mate_distance > md_budget` の OR proven を block，
    /// (c) AND proven で md > md_budget なら unknown 復帰，
    /// により budget-bounded 探索を行う．
    ///
    /// 旧 PV-walk refinement (`refine_mate_distance`) は廃止．reason:
    /// per-child 2M budget の partial 再展開だったため，46M+ nodes 消費しても
    /// shorter PV に到達できなかった (worklog 2026-05-28 参照)．
    /// KH 風 IDS は budget が search 全体を bound するため格段に効率的．
    ///
    /// 返り値: (final `MidSearchResult`, shortest PV as `Vec<Move>`)
    pub fn solve_v2_find_shortest(
        &mut self, board: &mut Board,
    ) -> (super::mid_v2::MidSearchResult, Vec<Move>) {
        self.table.preserve_working_on_proof = true;

        // Step 1: 初回 solve (budget 無制限)．
        let nodes_before_iter0 = self.nodes_searched;
        let mut result = self.solve_v2_with_budget(board, u16::MAX);
        if result.pn != 0 {
            self.table.preserve_working_on_proof = false;
            return (result, Vec::new());
        }
        let iter0_nodes = self.nodes_searched - nodes_before_iter0;

        let mut best_pv = self.extract_pv_limited(board, 100_000);
        let mut best_d = best_pv.len() as u16;

        // Step 2: KH SearchMainLoop 風: budget を `best_d - 2` で反復．
        // 詰将棋では mate length は OR-to-move から奇数長．`-= 2` で parity 維持．
        //
        // Phase 23 Polish 1: per-iter node cap．
        // 存在しない shorter proof を探して budget 全部使い切ると無駄が大きい
        // (29te で baseline 200M nodes 観測)．iter 0 の規模に対し relative cap を
        // かけ，"既に shortest" 仮説のテストを安価に行う．確認が取れない場合は
        // (max_nodes 内で) 段階的に拡大．
        //
        // Phase 24a (v1.4.0): cap を iter0*8 → iter0*2 に縮小．
        // 根拠: shorter mate は元 proof より **小さい木**なので発見コストは iter0 未満．
        // budget=27 iter の re-walk storm (29te: 1.5M nodes / +33 misses = 純粋な
        // 再走査) を抑制する．改善時は *2 cascade で converge を許容するため，
        // 真に shorter な mate があれば段階拡大で到達できる．iter0*8 は過大だった．
        let mut per_iter_cap: u64 = (iter0_nodes.saturating_mul(2)).max(200_000);
        loop {
            if self.nodes_searched >= self.max_nodes || self.timed_out { break; }
            if best_d <= 1 { break; }
            let new_budget = best_d.saturating_sub(2);
            if new_budget == 0 { break; }

            // この iter 用に max_nodes を一時的に絞る．
            let saved_max = self.max_nodes;
            let iter_cap_target =
                self.nodes_searched.saturating_add(per_iter_cap).min(saved_max);
            self.max_nodes = iter_cap_target;
            let nodes_before = self.nodes_searched;
            let r = self.solve_v2_with_budget(board, new_budget);
            let iter_nodes = self.nodes_searched - nodes_before;
            self.max_nodes = saved_max;

            if r.pn != 0 {
                // budget 内に shorter proof が存在しないことが確認できた．stop．
                // (timed_out 経由の早期終了も含む)
                let _ = iter_nodes;
                break;
            }

            let new_pv = self.extract_pv_limited(board, 100_000);
            let new_d = new_pv.len() as u16;
            if new_d > 0 && new_d < best_d {
                best_pv = new_pv;
                best_d = new_d;
                // 改善があった → 次 iter にもう少し budget を許す (cascade で converge できるよう)．
                per_iter_cap = per_iter_cap.saturating_mul(2);
            } else {
                // proven は出たが PV 抽出が改善しなかった (例: 同じ長さ)．stop．
                break;
            }
        }

        self.table.preserve_working_on_proof = false;
        result.mate_distance = best_d;
        (result, best_pv)
    }

    /// Phase 18 (v0.97.0): find_shortest 用 backward analysis．
    ///
    /// proven ノードの正確な mate_distance を既存 proven children のみから計算し
    /// TT を更新する．mid_v2 は呼ばない (cheap recursion only)．
    ///
    /// `try_unproven=true` の場合のみ，OR ノードで未証明 child に per-child budget
    /// 付き mid_v2 を試行する．ただし再帰呼出では `try_unproven=false` にして
    /// budget を root-level alternatives に集中させる．
    pub fn refine_mate_distance(
        &mut self,
        board: &mut Board,
        or_node: bool,
        depth: u32,
        memo: &mut rustc_hash::FxHashMap<u64, u16>,
    ) -> Option<u16> {
        self.refine_mate_distance_inner(board, or_node, depth, memo, true)
    }

    fn refine_mate_distance_inner(
        &mut self,
        board: &mut Board,
        or_node: bool,
        depth: u32,
        memo: &mut rustc_hash::FxHashMap<u64, u16>,
        try_unproven: bool,
    ) -> Option<u16> {
        const PER_CHILD_BUDGET: u64 = 2_000_000;

        if depth >= 64 {
            return None;
        }
        if self.nodes_searched >= self.max_nodes || self.timed_out {
            return None;
        }
        let pk = position_key(board);
        let hand = board.hand[self.attacker.index()];
        let cache_key = board.hash;
        if let Some(&md) = memo.get(&cache_key) {
            return Some(md);
        }
        let (pn, _dn, _) = self.look_up_pn_dn(pk, &hand, REMAINING_INFINITE);
        if pn != 0 {
            return None;
        }
        let moves: Vec<Move> = if or_node {
            self.generate_check_moves_cached(board).into_iter().collect()
        } else {
            movegen::generate_legal_moves(board)
        };
        if moves.is_empty() {
            memo.insert(cache_key, 0);
            return Some(0);
        }
        if or_node {
            let mut min_md: u16 = u16::MAX;
            let mut best_move: u16 = 0;
            let mut unproven_moves: Vec<Move> = Vec::new();

            // Pass 1: refine already-proven children (recursion does NOT try mid_v2).
            for m in &moves {
                let cap = board.do_move(*m);
                let cpk = position_key(board);
                let chand = board.hand[self.attacker.index()];
                let (cpn, cdn, _) = self.look_up_pn_dn(cpk, &chand, REMAINING_INFINITE);
                if cpn == 0 {
                    if let Some(md) = self.refine_mate_distance_inner(
                        board, !or_node, depth + 1, memo, false,
                    ) {
                        if md < min_md {
                            min_md = md;
                            best_move = m.to_move16();
                        }
                    }
                } else if cdn != 0 && try_unproven {
                    unproven_moves.push(*m);
                }
                board.undo_move(*m, cap);
            }

            // Pass 2: per-child budget mid_v2 (only when try_unproven=true).
            for m in &unproven_moves {
                if min_md <= 1 { break; }
                if self.nodes_searched >= self.max_nodes || self.timed_out { break; }
                let cap = board.do_move(*m);
                let saved_max = self.max_nodes;
                self.max_nodes = self.nodes_searched.saturating_add(PER_CHILD_BUDGET)
                    .min(saved_max);
                let mut inc = 0u32;
                let _ = self.mid_v2(board, INF - 1, INF - 1, depth + 1, !or_node, &mut inc,
                                    u16::MAX);
                self.max_nodes = saved_max;
                // After mid_v2, check if now proven and refine.
                if let Some(md) = self.refine_mate_distance_inner(
                    board, !or_node, depth + 1, memo, false,
                ) {
                    if md < min_md {
                        min_md = md;
                        best_move = m.to_move16();
                    }
                }
                board.undo_move(*m, cap);
            }

            if min_md == u16::MAX {
                return None;
            }
            let new_md = min_md.saturating_add(1);
            self.store_with_best_move_and_distance(
                pk, hand, 0, INF, REMAINING_INFINITE,
                pk as u32, best_move, new_md);
            memo.insert(cache_key, new_md);
            Some(new_md)
        } else {
            let mut max_md: u16 = 0;
            let mut all_proven = true;
            let mut max_move: u16 = 0;
            for m in &moves {
                let cap = board.do_move(*m);
                let child_md = self.refine_mate_distance(board, !or_node, depth + 1, memo);
                board.undo_move(*m, cap);
                match child_md {
                    Some(md) => {
                        if md > max_md {
                            max_md = md;
                            max_move = m.to_move16();
                        }
                    }
                    None => {
                        all_proven = false;
                        break;
                    }
                }
            }
            if !all_proven {
                return None;
            }
            let new_md = max_md.saturating_add(1);
            self.store_with_best_move_and_distance(
                pk, hand, 0, INF, REMAINING_INFINITE,
                pk as u32, max_move, new_md);
            memo.insert(cache_key, new_md);
            Some(new_md)
        }
    }
}

/// mid_v2 用の TCA threshold 拡張．
fn extend_threshold_for_mid_v2(thpn: &mut u32, thdn: &mut u32, curr: &super::mid_v2::MidSearchResult) {
    extend_threshold_for_mid_v2_mode(thpn, thdn, curr, false)
}

/// Phase 22: KH style `max(thpn, pn+1)` (clamp) vs maou cumulative `thpn += pn/4+1`．
fn extend_threshold_for_mid_v2_mode(
    thpn: &mut u32, thdn: &mut u32, curr: &super::mid_v2::MidSearchResult, kh_clamp: bool,
) {
    if kh_clamp {
        if curr.pn < INF {
            let target = curr.pn.saturating_add(1).min(INF - 1);
            if *thpn < target { *thpn = target; }
        }
        if curr.dn < INF {
            let target = curr.dn.saturating_add(1).min(INF - 1);
            if *thdn < target { *thdn = target; }
        }
        return;
    }
    const EXTEND_DENOM: u32 = 4;
    if curr.pn < INF {
        let extra = curr.pn / EXTEND_DENOM + 1;
        *thpn = thpn.saturating_add(extra).min(INF - 1);
    }
    if curr.dn < INF {
        let extra = curr.dn / EXTEND_DENOM + 1;
        *thdn = thdn.saturating_add(extra).min(INF - 1);
    }
}

