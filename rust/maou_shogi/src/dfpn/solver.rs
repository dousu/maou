//! DfPnSolver 構造体と探索コアロジック．

use arrayvec::ArrayVec;
#[cfg(feature = "visit_diag")]
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::time::{Duration, Instant};

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::Move;
use crate::types::{Color, HAND_KINDS};

use super::entry::PNS_MAX_ARENA_NODES;
use super::tt::TranspositionTable;
#[cfg(feature = "profile")]
use super::profile::ProfileStats;
use super::{
    hand_gte_forward_chain,
    CheckCache,
    DEEP_DFPN_R, DISPROOF_THRESHOLD_ADAPTIVE, EPSILON_DENOM_ADAPTIVE,
    MAX_MOVES,
};

/// path 配列の容量．depth の最大値(41) + マージン．
pub(super) const PATH_CAPACITY: usize = 48;

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

    #[inline]
    fn idx(pos_key: u64) -> usize {
        (pos_key as usize) & (PC_SUMMARY_SIZE - 1)
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
    /// Phase 31: find_shortest の per-iter refinement cap を override する (0 = 旧 formula
    /// `max(iter0*2, 200_000)`)．canonical (fs=T) cost は core でなくこの ~200K refinement
    /// floor が支配する (config 不問でほぼ一定 ~200K)．budget=best_d-2 の iter は 29te で
    /// cap を使い切って give-up し iter0 の最短を信頼するだけ (実際の shorter 確認はしていない)．
    /// → cap を下げれば canonical が core に漸近する (29te は iter0 が既に最短 29 を発見)．
    /// 注: shorter mate が cap 内で見つかる他問題では非最短を返しうる (find_shortest 品質 tradeoff)．
    pub(super) param_refine_iter_cap: u64,
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
    pub(super) mid_expansion_stack: Vec<super::local_expansion::MidLocalExpansion>,

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


    /// Phase 20 (v0.99.0): KH `min_depth` 相当の max_remaining トラッキング．
    /// (pos_key, hand_hash) → max_remaining (= shallowest ply で stored)．
    /// store 時に更新，mid_v2 の has_old_child 判定で参照．
    pub(super) max_remaining_map: rustc_hash::FxHashMap<u64, u16>,

    /// Phase 22: 1+ε 閾値 epsilon (KH デフォルト 1; PN_UNIT スケール考慮で PN_UNIT が候補)．
    pub(super) param_threshold_epsilon: u32,
    /// Phase 25: first-visit 初期 pn から edge_cost を外し，純 `init_pn_dn_*_kh`
    /// (KH df-pn+ 難易度推定) のみを使う．KH は pn/dn を難易度推定に限定し move
    /// 選好は MoveBriefEvaluation tie-break に分離する．maou は edge_cost を pn に
    /// 折込むため pn が二重信号となり delta-sum 算術を歪める仮説のテスト用．
    /// move_brief_eval tie-break は decouple 時も維持される (KH と同じ ordering)．
    pub(super) param_decouple_edge_cost: bool,
    /// Phase 25: KH `CheckObviousFinalOrNode` (1 手詰 detection) を適用する最大 depth．
    /// `self.depth <= この値` のとき AND parent で child OR の 1 手詰を即 proven 化する．
    /// default 25 (29te=depth33 では OFF; G5 で gate-off は初回 105K だが PV 31 退行)．
    pub(super) param_obvious_final_max_depth: u32,
    /// Phase 25 [keystone]: find_shortest budget 探索で blocked-proven (stored_md >
    /// md_budget) の中間レンジ lookup を KH `LookUpExact` 準拠の `(PN_UNIT, INF)` に
    /// する (proven entry = 絶対的に詰みなので dn=INF は sound)．false = 旧 (PN_UNIT,
    /// PN_UNIT) リセット (subtree 全再探索)．KH dual-range の find_shortest 収束改善用．
    pub(super) param_kh_middle_range: bool,
    /// Phase 30: KH coherent scale mode．ON で mid_v2 の seed を KH `kPnDnUnit=2`
    /// スケールに揃え (pure InitialPnDn / (PN_UNIT/2))，MidLocalExpansion の
    /// 完全 KH comparer (δ tie-break) を有効化する．epsilon=1 / tca_kh_clamp /
    /// deferred_penalty(/8) / is_sum_delta と併用して KH の統合 selectivity を
    /// coherent whole として再現する (個別 piece は Phase 22-28 で全敗)．
    pub(super) param_kh_scale: bool,
    /// Phase 30: MidLocalExpansion の完全 KH comparer (δ tie-break + amount) を
    /// unit-16 でも有効化する (kh_scale とは独立)．
    pub(super) param_kh_full_comparer: bool,
    /// Phase 31 (**反証済**): kh_full_comparer の δ tie-break を **AND ノードのみ**に限定する仮説．
    /// OR の δ は王手選択を変え非最短 (Mate-31) を誘発するが deep breadth 主因は AND fan-out なので
    /// AND のみ δ を効かせれば最短 29 維持 + defender selectivity を得られると期待した．
    ///
    /// **実測 (29te, WDC fs=F)**: WDC 56,689 / fullcmp 39,324(Mate31) / fc+andonly **66,953(Mate29, +18%)**．
    /// = Mate-29 は保つ (δ-at-OR が 31 の原因と確認) が AND-δ は breadth を**増やす**．
    /// fullcmp の効率 (39K) は OR-δ 由来で 31 と不可分．かつ 31 手詰は 29 手詰より proof tree が
    /// **小さい** (39K<57K) — δ は「証明が楽な (が長い) 詰み」へ誘導する．最短 29 には使えない．default OFF．
    pub(super) param_kh_comparer_and_only: bool,
    /// Phase 30: mid_v2 の OR 子展開で合駒 capture cross-deduction を有効化する．
    /// first-visit OR 子が取り王手で既証明局面に到達するなら，hand dominance で
    /// 展開せず証明確定する (`try_capture_tt_proof`)．old `mid` の合駒共有を mid_v2 に
    /// 移植し，深い ply の合駒 fan-out (29te の breadth 主因) を削る目的．
    pub(super) param_capture_dedup: bool,
    /// Phase 31 (**反証済**): 子の seed pn を king-escape 難易度で enrich する仮説．
    /// 詰める側 (`attacker.opponent()`) の玉の safe_escapes を pn に加算し，玉の機動性を
    /// 奪う王手を seed 段階で優先させる狙いだった (KH InitialPnDn の位置難易度に相当)．
    ///
    /// **実測 (29te, fs=F)**: W+rich = 88,603 nodes / 58,056 unique (W 77K/50K **比 +15%/+16% 悪化**)．
    /// eps interaction (decisive): W+rich+eps8 = 145K(**Mate 31**), W+rich+eps32 = 463K(**Mate 31**)．
    /// → ordering が改善するなら deep commitment (eps 大) が breadth を減らすはずが逆に爆発．
    /// = rich_seed は front child を正しくしておらず，むしろ edge_cost の鋭い差別化に noise を
    /// 加えて悪化させる．**seed quality は lever でない** (kh_scale 反証と整合)．default OFF 据置．
    ///
    /// **Phase 31 再検証 (decouple seed の tie 割り)**: trace で decouple seed は多数の王手に同一
    /// (pn=32,dn=16) を与え best-child flicker を起こすと判明．rich_escape で tie を割る実験
    /// (WDC+rich) = 61,294/30,926/**Mate-31** (footprint 不減, 31 手化)．= tie を割っても「詰め手」を
    /// 先頭にしなければ探索量は減らない (escape 順は詰め手を当てない)．→ 真因は「詰め手同定 =
    /// move-ordering 品質」で，安価な heuristic では埋まらない (KH も同 heuristic)．残差は scale dynamics．
    pub(super) param_rich_seed: bool,
    /// Phase 31 (**反証済**): mid_v2 の AND ノードで一般 合駒 cross-deduction を有効化する．
    /// drop 合駒 defense が refute (child pn=0) された直後に，同一マスの兄弟 drop 合駒を
    /// `cross_deduce_children` (capture 後局面の hand-dominance proof 転用) で展開せず証明する狙い．
    ///
    /// **実測 (29te, WDC ベース)**: WDC 56,689/29,938 → WDC+xdedup 58,662/30,963 (**+3.5% 悪化**)．
    /// Mate-29 維持 (sound) だが deep AND breadth は不変 (ply19:2512→2667)．= 29te の deep AND fan-out は
    /// 同一マス drop 兄弟の proof 共有では削れない (defender breadth は king 逃げ/capture/別マス合駒が主)．
    /// Phase 28 hand-minimization 同様「reuse 強化はコストのみ増」パターン．default OFF 据置．
    pub(super) param_cross_dedup: bool,
    /// Phase 31: mid_v2 の AND ノード move 生成に PNS の 無駄合い filter を適用する．
    /// mid_v2 は従来 `movegen::generate_legal_moves` で**全**合法手 (無駄合い含む) を生成し，
    /// deep AND (ply 17+) の 合駒 fan-out が breadth 爆発の主因だった (unique gap 14× vs KH)．
    /// `generate_defense_moves_inner` (pns.rs) は futile 合駒を除外し chain マスは歩のみ生成する
    /// 既存・検証済の 無駄合い filter．KH の 駒強奪 (g_stolen_pr) 無駄合防止に相当する maou の機構．
    ///
    /// **実測 (29te, WDC fs=F)**: WDC 56,689/29,938 → WDC+muda 57,447/30,098 (**ほぼ無変化, Mate-29**)．
    /// = 29te の deep AND breadth は futile 無駄合い でなく **genuine な防御 (玉移動/駒取り) +
    /// 攻め方の drop 王手の多様性** (R2G2P 持駒) 由来で，無駄合い filter では削れない．
    /// → 「mid_v2 が 無駄合い filter を欠く」ことは事実だが 29te の breadth 主因ではない (反証)．
    pub(super) param_muda_filter: bool,
    /// Phase 22: TCA extension formula を KH `max(thpn, pn+1)` 形式にするか．
    pub(super) param_tca_kh_clamp: bool,
    /// Phase 22: root level IDS (KH SearchEntry 風 1.7× threshold growth) 有効．
    pub(super) param_root_ids_enable: bool,
    /// Phase 26 (v1.5.0): KH parity DML (非駒打ち成/不成 deferral)．**default true**．
    /// true で `build_delayed_chain` が KH `delayed_move_list.hpp` 忠実版になり，
    /// 歩/角/飛/香(rank2/8) の成・不成ペアを OR/AND 両方で chain 化して breadth を削る．
    /// false (旧挙動) は 29te depth=31 で **false NoMate** (811,241 nodes) を返していた
    /// soundness バグがあった．真因は depth-limit 偽反証の TT 汚染で，v1.6.0 の
    /// [`DfPnSolver::param_scope_disproof`] が根治した．kh_dml は guidance robustness 改善として
    /// 併用 (on+scope が 29te depth=31 で最速 162K)．
    pub(super) param_kh_dml: bool,
    /// Phase 26b (root cause fix, v1.6.0): mid_v2 の集約 disproof (curr.dn==0) を絶対
    /// (REMAINING_INFINITE = ProvenTT confirmed) ではなく **remaining scope** で store する
    /// (**default true**)．depth-limit 偽反証 (`look_up_pn_dn_impl` remaining==0 → (INF,0)) が
    /// 伝播して生じた集約 disproof が WorkingTT に scope 付きで入り，より深い ply (= remaining 大)
    /// の transposition lookup では再探索される (tt.rs:1267 `e.remaining()>=remaining`)．
    /// これで 29te depth=31 の false NoMate (811,241 nodes) を根治する．KH disproven_len scope 相当．
    /// false (旧挙動) は horizon disproof を confirmed 化し TT を汚染する soundness バグ．
    pub(super) param_scope_disproof: bool,
    /// Phase 27: KH `IsSumDeltaNode` (`initial_estimation.hpp:227`) を適用するか．
    /// true で OR ノードの香成/不成 near-duplicate child を **max 集約** に切替え，δ値の
    /// 二重計上 (= breadth 発散) を抑える．KH `local_expansion.hpp:177` の
    /// `!IsSumDeltaNode(...)` 分岐相当．maou は従来この分岐が未実装で，香車王手の多い
    /// 局面で delta が過大になり distinct-position breadth が KH より発散していた仮説の検証用．
    /// default false (baseline 不変)．効果確認後に default 化を検討する．
    pub(super) param_is_sum_delta_node: bool,
    /// Phase 28: KH 流の極小証明駒 (minimal proof hand) を mid_v2 の proof store に適用するか．
    /// true で proven 局面を「実際の攻め方持ち駒」ではなく [`super::proof_hand`] が計算する
    /// **極小証明駒** (子の要素 max + `AddIfHandGivesOtherEvasions` 補正) で store する．
    /// hand-dominance (`hand_gte_forward_chain`) の集約が強まり proof tree を圧縮する狙い
    /// (29te で KH 比 9× の bloat を縮める)．soundness-critical のため default false
    /// (baseline byte-identical)．効果・健全性確認後に default 化を検討する．
    pub(super) param_minimal_proof_hand: bool,
    /// Phase 28b: KH 流の極大反証駒 (maximal disproof hand) を mid_v2 の disproof store に
    /// 適用するか．true で不詰局面を「実際の攻め方持ち駒」ではなく [`super::proof_hand`] が計算する
    /// **極大反証駒** (子の要素 min + `RemoveIfHandGivesOtherChecks` 補正) で store する．
    /// disproof の hand-dominance が広がり失敗ライン (29te で unique 90K) を刈る狙い．
    /// soundness-critical (誤ると false-NoMate) のため default false (baseline byte-identical)．
    /// `param_minimal_proof_hand` と独立に A/B 可能．
    pub(super) param_minimal_disproof_hand: bool,
    /// Phase 29: KH 流の repetition (千日手) taint + targeted disproof scoping を使うか．
    /// true で，disproof (dn==0) が cycle に依存する (subtree に `repetition_start < 自 ply` の
    /// 千日手がある) 場合のみ scope 限定で store し，それ以外は confirmed (`REMAINING_INFINITE`)
    /// で store する．`param_scope_disproof` の**全 disproof を一律 scope 限定する再探索**
    /// (breadth 主因) を，path-dependent なものだけに精密化して置換える狙い．
    /// soundness-critical (誤ると false-NoMate) のため default false (baseline byte-identical)．
    pub(super) param_kh_repetition: bool,
    /// Phase 21: deferred penalty 除数 (0=無効, 8=KH 準拠)．
    pub(super) param_deferred_penalty_denom: u32,
    /// Phase 22: deferred penalty `.max(1)` floor (KH=true)．
    pub(super) param_deferred_penalty_floor: bool,
    /// Phase 21: TCA gate を is_shallow ベースにするか (true=v0.99.0, false=旧 !is_first_visit)．
    pub(super) param_tca_use_shallow_gate: bool,
    /// Phase 21 診断: deferred penalty 発火フレーム数．
    pub(super) diag_deferred_frames: u64,
    /// Phase 32: mid_v3 (ground-up KH コア port, unit-2 scale) 専用の exact-match TT．
    /// key = board.hash (全局面 hash, 手番・持駒含む)．mid_v2 の TT baggage を排した clean store．
    pub(super) v3_tt: rustc_hash::FxHashMap<u64, super::mid_v3::V3Entry>,
    /// mid_v3 探索パス上の board.hash → ply (千日手検出 + 参照祖先 ply 特定用)．
    pub(super) v3_path: rustc_hash::FxHashMap<u64, u32>,
    /// mid_v3 探索ノード数 (= search_v3 呼び出し回数)．
    pub(super) v3_nodes: u64,
    /// Phase 33: mid_v3 で検証済 `MidLocalExpansion` (DML/sum_mask/comparer/deferred) を
    /// per-node に駆動する (案②)．`false` = classic df-pn 集約 (sound 181K baseline)．
    /// `true` = LocalExpansion refinement + clean TT + unit-2 + 非累積 extend + root IDS の合成．
    pub(super) param_v3_local_exp: bool,
    /// Phase 36: KH `CheckObviousFinalOrNode` 先読み (AND-child OR 局面で `mate_move_in_1ply` →
    /// 詰めば absolute proof を v3_tt へ格納)．default `true`．`mate_move_in_1ply` の false
    /// mate-1 バグ (ピン軸取り返し / 逆王手 drop / 開き王手) 根治後に default 化．
    /// [[project_dfpn_domoff_none_mate1_bugs]]．
    pub(super) param_v3_lookahead: bool,
    /// Phase 33b: KH `RepetitionTable` 相当．LE path で repetition 依存の disproof を
    /// path_key でキャッシュし (clean TT は absolute 結果のみ)，sound GHI + reuse を実現する．
    pub(super) v3_rep_memo: super::repetition_memo::RepetitionMemo,
    /// Phase 33b: LE path で IsInferior dominance (KH VisitHistory) を有効化するか．
    /// RepetitionMemo + path_key で sound 化された前提で breadth を削る．
    pub(super) param_v3_dominance: bool,
    /// Phase 33b 診断: dominance 発火数 / RepetitionMemo insert / hit．
    pub(super) v3_dom_fires: u64,
    pub(super) v3_rep_inserts: u64,
    pub(super) v3_rep_hits: u64,
    /// Phase 33k: LE path で cycle-dependent disproof を path_key で RepetitionMemo にキャッシュするか
    /// (KH RepetitionTable; dominance 無しの base でも有効)．clean TT は skip する taint 付き disproof を
    /// path_key で再利用し，再降下の thrash を抑える．
    /// V3_XHAND (KH ttentry LookUpSuperior/Inferior 移植): 非 taint unknown の
    /// pn/dn を (pos_key, attacker hand) 別に保存し，child seed 時に
    /// 「現 hand 優等 → dn=max(dn, entry.dn)」「劣等 → pn=max(pn, entry.pn)」の
    /// bound 合成を行う (unknown 推定のみ変更 = final 判定に影響せず sound)．
    /// 値 = (hand, pn, dn, min_depth)．bucket cap は V3_XH_CAP．
    pub(super) v3_xh: rustc_hash::FxHashMap<u64, Vec<([u8; HAND_KINDS], u32, u32, u16)>>,
    pub(super) param_v3_xhand: bool,
    pub(super) v3_xh_hits: u64,
    pub(super) param_v3_rep_cache: bool,
    /// Phase 33c: LE path で KH EliminateDoubleCount (DAG δ 二重計上除去) を有効化するか．
    pub(super) param_v3_dag: bool,
    /// Phase 33c 診断: double-count 補正の発火数．
    pub(super) v3_dag_resets: u64,
    /// Phase 33d: LE path の hand-aware proof/disproof reuse (KH proof_hand)．
    /// pos_key (持駒抜き盤面) → [(proof_hand, len, best16)]．`hand >= proof_hand` の transposition は
    /// 同じ証明を再利用できる (攻め方が多い駒で詰むなら少ない要求でも詰む)．absolute proof のみ格納．
    pub(super) param_v3_proof_hand: bool,
    pub(super) v3_proven: rustc_hash::FxHashMap<u64, Vec<([u8; HAND_KINDS], u16, u16)>>,
    /// pos_key → [disproof_hand]．`disproof_hand >= hand` の transposition も不詰 (少ない駒で詰まないなら
    /// 更に少なくても詰まない)．absolute disproof のみ格納 (repetition は RepetitionMemo)．
    pub(super) v3_disproven: rustc_hash::FxHashMap<u64, Vec<[u8; HAND_KINDS]>>,
    /// Phase 33d 診断: hand-aware reuse hit 数．
    pub(super) v3_ph_hits: u64,
    /// Phase 33g 診断: per-ply total/unique 訪問数 (KHPLY trace と比較用)．
    pub(super) v3_ply_total: [u64; 64],
    pub(super) v3_ply_unique: [u64; 64],
    pub(super) v3_ply_seen: rustc_hash::FxHashSet<(u32, u64)>,
    /// Phase 33k 診断: KHSEL per-ply first-visit dump で各 ply を 1 度だけ出力するためのフラグ．
    pub(super) v3_sel_dumped: [u8; 8],
    /// Phase 33n 診断: V3TRACE chronological per-expansion trace のカウンタ．
    pub(super) v3_trace_cnt: u32,
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
            param_refine_iter_cap: 0,
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
            max_remaining_map: rustc_hash::FxHashMap::default(),
            param_threshold_epsilon: 2,
            param_decouple_edge_cost: false,
            param_obvious_final_max_depth: 25,
            param_kh_middle_range: false,
            param_kh_scale: false,
            param_kh_full_comparer: false,
            param_capture_dedup: false,
            param_rich_seed: false,
            param_cross_dedup: false,
            param_kh_comparer_and_only: false,
            param_muda_filter: false,
            param_tca_kh_clamp: false,
            param_root_ids_enable: false,
            param_kh_dml: true,
            param_scope_disproof: true,
            param_is_sum_delta_node: false,
            param_minimal_proof_hand: false,
            param_minimal_disproof_hand: false,
            param_kh_repetition: false,
            param_deferred_penalty_denom: 0,
            param_deferred_penalty_floor: false,
            param_tca_use_shallow_gate: false,
            diag_deferred_frames: 0,
            v3_tt: rustc_hash::FxHashMap::default(),
            v3_path: rustc_hash::FxHashMap::default(),
            v3_nodes: 0,
            param_v3_local_exp: true,
            param_v3_lookahead: true,
            v3_rep_memo: super::repetition_memo::RepetitionMemo::new(1 << 16),
            param_v3_dominance: true,
            v3_dom_fires: 0,
            v3_rep_inserts: 0,
            v3_rep_hits: 0,
            v3_xh: rustc_hash::FxHashMap::default(),
            param_v3_xhand: false,
            v3_xh_hits: 0,
            param_v3_rep_cache: false,
            param_v3_dag: true,
            v3_dag_resets: 0,
            param_v3_proof_hand: false,
            v3_proven: rustc_hash::FxHashMap::default(),
            v3_disproven: rustc_hash::FxHashMap::default(),
            v3_ph_hits: 0,
            v3_ply_total: [0; 64],
            v3_ply_unique: [0; 64],
            v3_ply_seen: rustc_hash::FxHashSet::default(),
            v3_sel_dumped: [0; 8],
            v3_trace_cnt: 0,
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

    /// Phase 26: KH parity DML (非駒打ち成/不成 deferral) を設定．
    /// 詳細: [`DfPnSolver::param_kh_dml`]．
    pub fn set_kh_dml(&mut self, on: bool) -> &mut Self {
        self.param_kh_dml = on;
        self
    }

    /// Phase 33: mid_v3 で LocalExpansion refinement を駆動するか (案②)．
    /// 詳細: [`DfPnSolver::param_v3_local_exp`]．
    pub fn set_v3_local_exp(&mut self, on: bool) -> &mut Self {
        self.param_v3_local_exp = on;
        self
    }

    /// Phase 36: mid_v3 で KH `CheckObviousFinalOrNode` 先読みを有効化するか (default true)．
    /// 詳細: [`DfPnSolver::param_v3_lookahead`]．
    pub fn set_v3_lookahead(&mut self, on: bool) -> &mut Self {
        self.param_v3_lookahead = on;
        self
    }

    /// Phase 33b: LE path で KH IsInferior dominance + RepetitionMemo を有効化する．
    /// 詳細: [`DfPnSolver::param_v3_dominance`]．**注意**: 29te では sound だが clean TT が
    /// path 依存 dominance 結果を再利用できず逆に遅くなる (119K→776K) ことが実測済 (gated, default off)．
    pub fn set_v3_dominance(&mut self, on: bool) -> &mut Self {
        self.param_v3_dominance = on;
        self
    }

    /// Phase 33c: LE path で KH EliminateDoubleCount (DAG δ 二重計上除去) を有効化する．
    /// 詳細: [`DfPnSolver::param_v3_dag`]．
    pub fn set_v3_dag(&mut self, on: bool) -> &mut Self {
        self.param_v3_dag = on;
        self
    }

    /// Phase 33d: LE path で hand-aware proof/disproof reuse (KH proof_hand) を有効化する．
    /// 詳細: [`DfPnSolver::param_v3_proof_hand`]．
    pub fn set_v3_proof_hand(&mut self, on: bool) -> &mut Self {
        self.param_v3_proof_hand = on;
        self
    }

    /// Phase 26b: 集約 disproof を remaining scope で store する (root cause fix)．
    /// 詳細: [`DfPnSolver::param_scope_disproof`]．
    pub fn set_scope_disproof(&mut self, on: bool) -> &mut Self {
        self.param_scope_disproof = on;
        self
    }

    /// Phase 25: 1 手詰 detection (CheckObviousFinalOrNode) を適用する最大 depth．
    /// 詳細: [`DfPnSolver::param_obvious_final_max_depth`]．
    pub fn set_obvious_final_max_depth(&mut self, d: u32) -> &mut Self {
        self.param_obvious_final_max_depth = d;
        self
    }

    /// Phase 25 [keystone]: find_shortest 中間レンジ lookup を KH 準拠 (PN_UNIT, INF) に．
    /// 詳細: [`DfPnSolver::param_kh_middle_range`]．
    pub fn set_kh_middle_range(&mut self, on: bool) -> &mut Self {
        self.param_kh_middle_range = on;
        self
    }

    /// Phase 25: first-visit 初期 pn の edge_cost 折込を外し，純 KH df-pn+ にする．
    /// 詳細: [`DfPnSolver::param_decouple_edge_cost`]．
    pub fn set_decouple_edge_cost(&mut self, on: bool) -> &mut Self {
        self.param_decouple_edge_cost = on;
        self
    }

    /// Phase 22: TCA extension formula を KH `max(thpn, pn+1)` clamp 式にするか．
    pub fn set_tca_kh_clamp(&mut self, on: bool) -> &mut Self {
        self.param_tca_kh_clamp = on;
        self
    }

    /// Phase 30: KH coherent scale mode を有効化 (seed unit=2 + 完全 comparer)．
    /// 詳細: [`DfPnSolver::param_kh_scale`]．
    pub fn set_kh_scale(&mut self, on: bool) -> &mut Self {
        self.param_kh_scale = on;
        self
    }

    /// Phase 30: 完全 KH comparer (δ tie-break) を unit-16 でも有効化．
    pub fn set_kh_full_comparer(&mut self, on: bool) -> &mut Self {
        self.param_kh_full_comparer = on;
        self
    }

    /// Phase 30: mid_v2 の OR 子展開で合駒 capture cross-deduction を有効化．
    pub fn set_capture_dedup(&mut self, on: bool) -> &mut Self {
        self.param_capture_dedup = on;
        self
    }

    /// Phase 31: 子の seed pn を king-escape 難易度で enrich する (default OFF)．
    pub fn set_rich_seed(&mut self, on: bool) -> &mut Self {
        self.param_rich_seed = on;
        self
    }

    /// Phase 31: mid_v2 AND ノードで一般 合駒 cross-deduction を有効化 (default OFF)．
    pub fn set_cross_dedup(&mut self, on: bool) -> &mut Self {
        self.param_cross_dedup = on;
        self
    }

    /// Phase 31: kh_full_comparer の δ tie-break を AND ノードのみに限定 (default OFF)．
    pub fn set_kh_comparer_and_only(&mut self, on: bool) -> &mut Self {
        self.param_kh_comparer_and_only = on;
        self
    }

    /// Phase 31: mid_v2 AND ノードに PNS の 無駄合い filter を適用 (default OFF)．
    pub fn set_muda_filter(&mut self, on: bool) -> &mut Self {
        self.param_muda_filter = on;
        self
    }

    /// Phase 22: root level IDS (KH SearchEntry) を有効化．
    pub fn set_root_ids_enable(&mut self, on: bool) -> &mut Self {
        self.param_root_ids_enable = on;
        self
    }

    /// Phase 27: KH `IsSumDeltaNode` (OR 香成/不成 max 集約) を有効化．
    /// 詳細: [`DfPnSolver::param_is_sum_delta_node`]．
    pub fn set_is_sum_delta_node(&mut self, on: bool) -> &mut Self {
        self.param_is_sum_delta_node = on;
        self
    }

    /// Phase 28: KH 流の極小証明駒 (minimal proof hand) を有効化．
    /// 詳細: [`DfPnSolver::param_minimal_proof_hand`]．
    pub fn set_minimal_proof_hand(&mut self, on: bool) -> &mut Self {
        self.param_minimal_proof_hand = on;
        self
    }

    /// Phase 28b: KH 流の極大反証駒 (maximal disproof hand) を有効化．
    /// 詳細: [`DfPnSolver::param_minimal_disproof_hand`]．
    pub fn set_minimal_disproof_hand(&mut self, on: bool) -> &mut Self {
        self.param_minimal_disproof_hand = on;
        self
    }

    /// Phase 29: KH 流 repetition taint + targeted disproof scoping を有効化．
    /// 詳細: [`DfPnSolver::param_kh_repetition`]．
    pub fn set_kh_repetition(&mut self, on: bool) -> &mut Self {
        self.param_kh_repetition = on;
        self
    }

    /// Phase 21: TCA gate モードを設定 (true=is_shallow, false=!is_first_visit)．
    pub fn set_tca_shallow_gate(&mut self, on: bool) -> &mut Self {
        self.param_tca_use_shallow_gate = on;
        self
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

    /// デフォルトパラメータでソルバーを生成する．
    pub fn default_solver() -> Self {
        Self::new(31, 1_048_576, 32767)
    }

    /// 最短手数探索の有無を設定する．
    ///
    /// `false` にすると最初に見つかった詰み手順をそのまま返す(高速化)．
    /// Phase 31: find_shortest の per-iter refinement cap override (0 = 旧 formula)．
    pub fn set_refine_iter_cap(&mut self, cap: u64) -> &mut Self {
        self.param_refine_iter_cap = cap;
        self
    }

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


    pub(super) fn is_dominated_in_path(&self, child_pos_key: u64, child_hand: &[u8; HAND_KINDS]) -> Option<u32> {
        if !self.param_use_visit_history_dominance { return None; }
        for i in 0..self.path_len {
            if self.path_pos_key[i] == child_pos_key
                && hand_gte_forward_chain(&self.path_hand[i], child_hand)
            {
                return Some(i as u32);
            }
        }
        None
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
        // v2.1.0: production は mid_v3 を使用する (v1 mid / legacy PNS は廃止方針)．
        // mid_v3 は max_nodes / timeout を honor し TsumeResult::{Checkmate, NoCheckmate, Unknown}
        // を返す．未解決 (budget/timeout) は Unknown (false NoMate を出さない)．
        // 注意: find_shortest は mid_v3 では未対応 (IDS は threshold 成長で canonical mate を返す)．
        self.solve_via_v3(board)
    }

    /// `param_refutable_depth = 0` は適応的 depth を意味する sentinel．
    const EFFECTIVE_DEPTH_ADAPTIVE: u32 = 0;

    /// デフォルトの refutable check 呼び出し回数上限．
    const DEFAULT_REFUTABLE_CALL_LIMIT: u32 = 10_000;
    /// デフォルトの refutable check 再帰深さ (0 = 適応的，self.depth に基づく)．
    const DEFAULT_REFUTABLE_DEPTH: u32 = 0;

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

}


