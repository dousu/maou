//! DfPnSolver 構造体と探索コアロジック．

use arrayvec::ArrayVec;
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
    adjust_hand_for_move, edge_cost_and, edge_cost_or,
    hand_gte_forward_chain,
    position_key, propagate_nm_remaining, push_move, snda_dedup,
    CheckCache,
    DEEP_DFPN_R, DISPROOF_THRESHOLD_ADAPTIVE, EPSILON_DENOM_ADAPTIVE, INF, INTERPOSE_DN_BIAS,
    MAX_MOVES, PN_UNIT, REMAINING_INFINITE, STAGNATION_LIMIT, TCA_EXTEND_DENOM,
    ZERO_PROGRESS_LIMIT,
};

/// path 配列の容量．depth の最大値(41) + マージン．
const PATH_CAPACITY: usize = 48;

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
    /// 段階的予算拡大の warmup depth リスト (v0.24.65)．
    ///
    /// 空ならば warmup なし．非空の場合，solve() 内で各 depth の
    /// warmup solve を実行して ProvenTT に proof を蓄積する．
    pub(super) warmup_depths: Vec<u32>,
    /// solve() に渡された真の目標 depth (v0.24.66)．
    ///
    /// warmup mid_fallback 内では `saved_depth` が warmup_depth に上書きされる
    /// ため，IDS NM 昇格判定で true depth を参照するために保持する．
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
    /// Warmup モード (v0.24.68-v0.24.73): true の場合，mid_fallback 入口で
    /// ProvenTT の非 proof のみ除去し WorkingTT intermediate を保持する．
    ///
    /// v0.24.73 で warmup 段間 intermediate 保持を revert．さらに
    /// v0.24.78 で `skip_warmup=true` デフォルト化により solver.rs の warmup
    /// ループ自体が実行されない (§11.6 単一スレッド方針下で warmup 不要)．
    /// 現在 `true` に設定されるのは `pns.rs` の nested warmup
    /// (IDS 内 warmup mid_fallback) のみ．
    pub(super) warmup_mode: bool,
    /// PNS 探索中に true に設定し，`look_up_pn_dn` で refutable disproof を
    /// スキップする (v0.24.75)．PNS の arena-limited false NM を防止．
    /// MID 探索では false (refutable disproof を通常の NM として使用)．
    pub(super) skip_refutable_disproof: bool,
    /// 施策 C (v0.24.77): warmup を無効化するフラグ．
    /// refutable disproof 機構が NM 蓄積を提供するため warmup は冗長．
    /// デフォルト true = warmup 無効．set_skip_warmup(false) で再有効化可能．
    pub(super) skip_warmup: bool,
    /// warmup_mode=true の IDS 最終ステップで MID に割り当てる予算の分母 (デフォルト 4 = 1/4)．
    /// 2 にすると 1/2 (1F 導入前の挙動) に戻る．テスト用．
    pub(super) param_warmup_mid_denom: u32,
    /// warmup 入口で clear_proven_non_proofs() を呼ぶかどうか (v0.27.6 デフォルト true)．
    /// true (v0.27.6 デフォルト, v0.25.5 相当): ProvenTT 非 proof を除去，WorkingTT 維持．
    /// false (v0.25.9〜v0.27.5 の旧デフォルト, 1G 有効): ProvenTT もクリアしない．
    pub(super) param_warmup_clear_proven: bool,
    /// warmup 前に clear_working_shallow(warmup_depth) を呼ぶかどうか (v0.27.6 デフォルト false)．
    /// false (v0.27.6 デフォルト, v0.25.5 相当): 前処理なし，warmup が中間エントリを活用可能．
    /// true (v0.27.3〜v0.27.5 の旧挙動): remaining ≤ warmup_depth の WorkingTT を削除．退行の原因．
    pub(super) param_use_warmup_shallow_clear: bool,
    /// Hypothesis 1E 検証用フラグ (v0.27.4)．
    /// true にすると warmup_mode=true でも outer_solve_depth を NM guard に加える (1E 導入前の挙動)．
    /// デフォルト false = 1E 有効 (warmup 内は saved_depth のみで guard)．
    pub(super) param_warmup_nm_guard_outer: bool,
    /// Hypothesis IDS-17 無効化フラグ (v0.27.4)．
    /// true にすると saved_depth 20-26 での depth=16→17 挿入をスキップする (IDS-17 導入前の挙動)．
    /// デフォルト false = IDS-17 有効 (depth=17 を明示的に経由)．
    pub(super) param_no_ids17: bool,
    /// Hypothesis 1H 検証用フラグ (v0.27.5)．
    /// true にすると warmup 前の clear_working_shallow() 呼び出しをスキップする (v0.25.5 相当)．
    /// v0.27.3 で導入した clear_working_shallow は warmup が中間エントリを活用できなくする
    /// 可能性がある．デフォルト false = 現行挙動．
    pub(super) param_no_warmup_shallow_clear: bool,
    /// refutable check の再帰深さ (デフォルト 5)．
    pub(super) param_refutable_depth: u32,
    /// refutable check の呼び出し回数上限 (デフォルト 10,000)．
    pub(super) param_refutable_call_limit: u32,
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
    /// / clear_working) の直前に収集する．(ids_depth, nodes_searched, pn_hist, dn_hist, joint_hist)
    pub(super) pn_dn_per_depth: Vec<(u32, u64, [u64; 32], [u64; 32], Vec<u64>)>,
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
            saved_depth_for_epsilon: 0,
            warmup_depths: Vec::new(),
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
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            #[cfg(feature = "profile")]
            profile_stats: ProfileStats::default(),
            last_pns_proof_stores: 0,
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
            warmup_mode: false,
            skip_refutable_disproof: false,
            skip_warmup: true,
            param_warmup_mid_denom: 4,
            param_warmup_clear_proven: true,   // v0.27.6: v0.25.5 相当をデフォルト化
            param_warmup_nm_guard_outer: false,
            param_no_ids17: false,
            param_no_warmup_shallow_clear: false, // 後方互換テスト用 (実運用では param_use_warmup_shallow_clear を使用)
            param_use_warmup_shallow_clear: false, // v0.27.6: shallow clear 無効がデフォルト
            param_refutable_depth: Self::DEFAULT_REFUTABLE_DEPTH,
            param_refutable_call_limit: Self::DEFAULT_REFUTABLE_CALL_LIMIT,
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
        }
    }

    /// PNS アリーナの最大ノード数を設定する (v0.25.0)．
    ///
    /// デフォルトは `PNS_MAX_ARENA_NODES` (5M)．大きくすると arena spin 率が
    /// 下がる代わりにメモリ消費が増える．`min_value` (1024) 未満は丸められる．
    pub fn set_pns_arena_max(&mut self, max_nodes: usize) {
        self.param_pns_arena_max = max_nodes.max(1024);
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

    /// 段階的予算拡大の warmup depth リストを設定する (v0.24.65)．
    ///
    /// 空 (デフォルト) なら warmup なし (従来の `solve()` と同一)．
    /// 例: `&[17, 21]` を設定すると `solve()` 内で depth=17 → depth=21 の
    /// 順に warmup solve を実行し，ProvenTT に proof を蓄積してから
    /// full depth で最終 solve を実行する．
    ///
    /// warmup 各段では全ノード予算の 1/(N+1) を割り当て (N = warmup 段数)，
    /// 残りを最終 solve に使用する．
    pub fn set_warmup_depths(&mut self, depths: &[u32]) -> &mut Self {
        self.warmup_depths = depths.to_vec();
        self
    }

    /// refutable check のパラメータを設定する (v0.24.76)．
    pub fn set_refutable_params(&mut self, depth: u32, call_limit: u32) -> &mut Self {
        self.param_refutable_depth = depth;
        self.param_refutable_call_limit = call_limit;
        self
    }

    /// warmup の有効/無効を設定する (v0.24.77 施策 C)．
    ///
    /// デフォルト skip=true (warmup 無効)．refutable disproof 機構が
    /// NM 蓄積を提供するため warmup は冗長で budget を無駄にする．
    /// v0.24.77 の backward_10m_warmup で ply 22 が no-warmup の Mate(17) →
    /// warmup 使用時 Unknown に退行することを確認．
    pub fn set_skip_warmup(&mut self, skip: bool) -> &mut Self {
        self.skip_warmup = skip;
        self
    }

    /// warmup_mode=true の IDS 最終ステップで MID に割り当てる予算の分母を設定する．
    /// デフォルト 4 (= 1/4，Hypothesis 1F)．2 にすると 1/2 (1F 導入前) になる．
    pub fn set_warmup_mid_denom(&mut self, denom: u32) -> &mut Self {
        self.param_warmup_mid_denom = denom.max(1);
        self
    }

    /// Hypothesis 1G 検証用: warmup_mode=true でも retain_proofs_only() を呼ぶかどうかを設定する．
    /// true にすると 1G 導入前の挙動 (ProvenTT 非 proof を warmup 前にクリア)．
    /// デフォルト false = 1G 有効 (スキップ)．
    pub fn set_warmup_clear_proven(&mut self, enable: bool) -> &mut Self {
        self.param_warmup_clear_proven = enable;
        self
    }

    /// Hypothesis 1E 検証用: warmup_mode=true で outer_solve_depth を NM guard に含めるか設定する．
    /// true にすると 1E 導入前の挙動 (nm_guard_depth = saved_depth.max(outer_solve_depth))．
    /// デフォルト false = 1E 有効 (nm_guard_depth = saved_depth のみ)．
    pub fn set_warmup_nm_guard_outer(&mut self, enable: bool) -> &mut Self {
        self.param_warmup_nm_guard_outer = enable;
        self
    }

    /// Hypothesis IDS-17 無効化検証用: saved_depth 20-26 での depth=16→17 挿入をスキップするか設定する．
    /// true にすると IDS-17 導入前の挙動 (depth=16 の次は saved_depth へ直接ジャンプ)．
    /// デフォルト false = IDS-17 有効．
    pub fn set_no_ids17(&mut self, enable: bool) -> &mut Self {
        self.param_no_ids17 = enable;
        self
    }

    /// Hypothesis 1H 検証用: warmup 前の clear_working_shallow() をスキップするか設定する．
    /// true にすると v0.25.5 相当 (warmup 前に WorkingTT 浅いエントリを削除しない)．
    /// デフォルト false = v0.27.6 以降の現行挙動 (もともとデフォルトで skip)．
    pub fn set_no_warmup_shallow_clear(&mut self, enable: bool) -> &mut Self {
        self.param_no_warmup_shallow_clear = enable;
        self
    }

    /// 旧挙動復元用: warmup 前に clear_working_shallow() を呼ぶかどうかを設定する．
    /// true にすると v0.27.3〜v0.27.5 の旧挙動 (退行の原因)．
    /// デフォルト false = v0.27.6 以降のデフォルト (clear しない = v0.25.5 相当)．
    pub fn set_use_warmup_shallow_clear(&mut self, enable: bool) -> &mut Self {
        self.param_use_warmup_shallow_clear = enable;
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
    /// 返り値: `(ids_depth, nodes_searched, pn_hist, dn_hist, joint_hist)` のスライス
    pub fn collect_pn_dn_dist_per_depth(&self) -> &[(u32, u64, [u64; 32], [u64; 32], Vec<u64>)] {
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
        self.profile_stats.tt_proven_overflow_count = self.table.proven_overflow_count;
        self.profile_stats.tt_working_overflow_count = self.table.working_overflow_count;
        self.profile_stats.tt_overflow_no_victim_count =
            self.table.overflow_no_victim_count;
        self.profile_stats.tt_max_entries_per_position =
            self.table.max_entries_per_position;
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
        self.look_up_pn_dn_impl(pos_key, hand, remaining, true)
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
        // root 自体も保護(path には含まれない場合がある)
        if self.path_len > 0 {
            self.table.protect_working_entry(
                self.diag_root_pk,
                &self.diag_root_hand,
            );
        }
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
        self.table.store(pos_key, hand, pn, dn, remaining, source);
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
        self.table.store_with_best_move(pos_key, hand, pn, dn, remaining, source, best_move);
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
        self.table.clear();
        self.nodes_searched = 0;
        self.pn_dn_per_depth.clear();
        self.max_ply = 0;
        self.ply_nodes = [0; 64];
        self.path_len = 0;
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
        let (root_pn_after_pns, root_dn_after_pns, _) =
            self.look_up_pn_dn(pk, &att_hand, self.depth as u16);

        // PNS で未解決 + 残り予算あり → warmup + MID フォールバック
        if root_pn_after_pns != 0 && root_dn_after_pns != 0
            && self.nodes_searched < self.max_nodes
            && !self.timed_out
        {
            // Adaptive warmup 予算配分 (v0.24.65):
            //
            // warmup は proof 蓄積が目的であり，full solve ほどの予算は不要．
            // 各 warmup 段に残り予算の小さな割合を割り当て，main solve に
            // 予算の大部分を残す．
            //
            // 予算戦略:
            // - 各 warmup 段: 残り予算の 10%，上限 1M (浅い段) / 2M (深い段)
            // - main solve: 残り予算の全部 (warmup が少額なので 80%+ が残る)
            //
            // 早期終了:
            // - warmup で proved (pn=0) → main solve 不要
            // - warmup が proof を 1 件も蓄積しなかった → 以降スキップ
            // (v0.24.77 施策 C) refutable disproof 機構が NM 蓄積を提供するため，
            // 従来の warmup は budget を消費するだけで効果が限定的．
            // backward_10m_warmup で ply 22 が no-warmup の Mate(17) → Unknown に退行することを確認．
            // skip_warmup フラグで明示的に無効化可能 (デフォルト true = warmup 無効)．
            if !self.warmup_depths.is_empty() && !self.skip_warmup {
                let warmup_depths = self.warmup_depths.clone();
                let final_depth = self.depth;
                let mut prev_proven_count = self.table.proven_count();
                // 前段の warmup depth を追跡し，段間の remaining shift に使用する．
                let mut prev_warmup_depth: Option<u32> = None;

                for (i, &wd) in warmup_depths.iter().enumerate() {
                    if wd >= final_depth { continue; }
                    if self.nodes_searched >= saved_max_nodes || self.timed_out {
                        break;
                    }

                    // warmup 段間の TT 管理:
                    //
                    // v0.24.68 で intermediate 保持を試みたが，浅い depth で探索
                    // した中間値が深い depth で不正に使われ false proof を生成する
                    // 問題が判明 (Mate(7) at ply 22, expected Mate(17))．
                    // fc-normalized hand hash (v0.24.70) との相互作用で，異なる
                    // hand variant の intermediate が同一クラスタに集約され，
                    // remaining shift 後の pn/dn が不正確になる．
                    //
                    // 安全のため従来の retain_proofs_only に戻す．
                    if prev_warmup_depth.is_some() {
                        self.table.retain_proofs_only();
                    }

                    let remaining_budget = saved_max_nodes - self.nodes_searched;
                    let cap = if wd <= 17 { 500_000u64 } else { 1_000_000u64 };
                    let warmup_budget = (remaining_budget / 10).min(cap).max(100_000);

                    self.max_nodes = self.nodes_searched + warmup_budget;
                    self.depth = wd;
                    verbose_eprintln!(
                        "[solve] warmup {}/{}: depth={} budget={}K nodes={}K time={:.1}s",
                        i + 1, warmup_depths.len(), wd,
                        warmup_budget / 1000,
                        self.nodes_searched / 1000,
                        self.start_time.elapsed().as_secs_f64(),
                    );
                    self.mid_fallback(board);

                    let (wp, _) = self.look_up_board(board);
                    let new_proven = self.table.proven_count();
                    let proofs_added = new_proven.saturating_sub(prev_proven_count);
                    verbose_eprintln!(
                        "[solve] warmup {} done: pn={} proofs=+{} nodes={}K time={:.1}s",
                        i + 1, wp, proofs_added,
                        self.nodes_searched / 1000,
                        self.start_time.elapsed().as_secs_f64(),
                    );

                    if wp == 0 { break; }
                    if proofs_added == 0 {
                        verbose_eprintln!("[solve] warmup {}: no proofs, skip rest", i + 1);
                        break;
                    }
                    prev_proven_count = new_proven;
                    prev_warmup_depth = Some(wd);
                    // v0.24.73: 段間の retain_proofs_only はループ先頭で実行．
                }

                self.depth = final_depth;
                self.max_nodes = saved_max_nodes;
                // warmup 完了後: WorkingTT を clear して depth-limited disproof を
                // 除去する．warmup の浅い depth で生成された depth-limited NM
                // (dn=0, remaining < INFINITE) が main solve の look_up でヒット
                // すると false NoCheckmate が発生する (v0.24.66)．
                self.table.retain_proofs_only();
            }

            // 最終 MID フォールバック (full depth)
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
        let mut calls: u32 = 0;
        self.all_checks_refutable_recursive(
            board, checks, self.effective_refutable_depth(), &mut calls,
            self.param_refutable_call_limit,
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
    /// する現象を確認．warmup で代替できたため，固定 depth=10 相当の対策を
    /// 組み込む．
    ///
    /// **N-2 (v0.26.0)**: target 24-31 の固定フロア 8 を 6 に緩和し
    /// NPS -28% (M-A による) の一部を回収する．
    /// - target=1-19:  depth_floor=3 → d=3〜5 (変更なし)
    /// - target=20-23: depth_floor=**8** (M-A 維持) → d=8 (ply 20-23 false-NM を保護)
    /// - target=24-31: depth_floor=**6** (was 8) → d=6 (より深い問題では緩和安全)
    /// - target=32+:   depth_floor=3 → log_val(≥6) が dominates (元の formula に復帰)
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
        // N-2 (v0.26.0): target 24-31 のみ floor を緩和．target ≥ 32 では log_val ≥ 6．
        let depth_floor: u32 = match target {
            0..=23 => if target >= 20 { 8 } else { 3 },
            24..=31 => 6,
            _ => 3,
        };
        log_val.max(depth_floor).min(10)
    }

    /// `param_refutable_depth = 0` は適応的 depth を意味する sentinel．
    const EFFECTIVE_DEPTH_ADAPTIVE: u32 = 0;

    /// TT ベースの NM 昇格判定(MID depth boundary 用)．
    ///
    /// 各王手後の AND ノードが ProvenTT に disproof として格納されているかを
    /// 確認する．`look_up` は confirmed / refutable 両方の disproof を返す
    /// (TT レベルではエントリ種別を区別しない)ため，MID 経路では
    /// refutable disproof も含めて NM 昇格の根拠として使用される．
    /// do_move + TT ルックアップのみで判定するため極めて高速
    /// (~2μs/王手)．TT にエントリがない場合は保守的に false を返す．
    pub(super) fn all_checks_refutable_by_tt(
        &mut self,
        board: &mut Board,
        checks: &[Move],
    ) -> bool {
        for check in checks {
            let captured = board.do_move(*check);
            let pk = position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            // look_up は confirmed/refutable disproof 両方を返す
            // (TT レベルでは種別を区別しない; PNS 側のみ
            // skip_refutable_disproof で refutable を除外する)
            let (_, dn, _) = self.table.look_up(pk, &att_hand, REMAINING_INFINITE, false);
            board.undo_move(*check, captured);
            if dn != 0 {
                return false;
            }
        }
        true
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
        if (ply as usize) < 64 {
            self.ply_nodes[ply as usize] += 1;
        }
        // === Periodic GC (ProvenTT / WorkingTT 独立) ===
        // overflow カウントベースで GC をトリガする(充填率の全走査を避ける)．
        // intermediate 保護 + パス保護により，GC で探索が崩壊することはない．
        if self.nodes_searched % 100_000 == 0 {
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
            if self.nodes_searched % 1_000_000 == 0 {
                let proven_size = self.table.proven_len();
                let proven_cap = self.table.proven_capacity();
                if proven_size > proven_cap * 7 / 10 {
                    let removed = self.table.gc_proven();
                    if removed > 0 {
                        verbose_eprintln!("[periodic_gc] proven removed={} proven={}/{}",
                            removed, self.table.proven_len(), proven_cap);
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
        let pos_key = profile_timed!(self, position_key_ns, position_key_count,
            position_key(board));
        let att_hand = board.hand[self.attacker.index()];

        // ProvenTT の ply ベース amount 用: ルートに近い proof ほど高い priority
        self.table.hint_ply = ply;

        // ループ検出: フルハッシュで判定(持ち駒込みの完全一致)
        let in_path = profile_timed!(self, loop_detect_ns, loop_detect_count,
            self.path[..self.path_len].contains(&full_hash));
        if in_path {
            #[cfg(feature = "tt_diag")]
            { self.diag_in_path_exits += 1; }
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
            return;
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
                } else if self.all_checks_refutable_by_tt(board, &checks) {
                    #[cfg(feature = "tt_diag")]
                    {
                        self.diag_boundary_or_refutable += 1;
                        self.diag_boundary_or_checks_sum += checks.len() as u64;
                    }
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key as u32);
                } else {
                    // rem=0 の仮反証は TT に store しない
                    // (クラスタの 64.7% を占め overflow の主因)
                    #[cfg(feature = "tt_diag")]
                    {
                        self.diag_boundary_or_not_refutable += 1;
                        self.diag_boundary_or_checks_sum += checks.len() as u64;
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
                        let dl_rem = if checks.is_empty() {
                            REMAINING_INFINITE
                        } else if self.all_checks_refutable_by_tt(board, &checks) {
                            REMAINING_INFINITE
                        } else {
                            0
                        };
                        // REMAINING_INFINITE(真の不詰)のみ store する．
                        // rem=0 の仮反証は TT に store しない:
                        // - 同じ IDS depth の同じ深さでしか参照されない
                        // - クラスタの 64.7% を占め overflow の主因
                        // - ローカル変数 cpn/cdn で解決チェック可能
                        if dl_rem == REMAINING_INFINITE {
                            self.store(child_pk, child_hand, INF, 0, dl_rem, child_pk as u32);
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
                                pn = pn.saturating_add(edge_cost_or(*m, ksq));
                            }
                            let dn = PN_UNIT;
                            self.store(child_pk, child_hand, pn, dn,
                                child_remaining, child_pk as u32);
                        }
                    } else {
                        // depth 制限超過: 応手生成なし → deep df-pn のみ適用
                        let mut pn = PN_UNIT;
                        // DFPN-E: エッジコスト加算
                        if let Some(ksq) = defender_king_sq {
                            pn = pn.saturating_add(edge_cost_or(*m, ksq));
                        }
                        let dn = PN_UNIT;
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
                        let pn = self.heuristic_or_pn(board, nc)
                            .saturating_add(edge_cost_and(*m));
                        let dn = PN_UNIT;
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
                    self.store(pos_key, att_hand, INF, 0,
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
                    parent_nm_remaining, pos_key as u32, true,
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
                self.store(pos_key, att_hand, INF, 0,
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
                let is_loop_child = self.path[..self.path_len].contains(&child_fh);
                let (cpn, cdn, _csrc) = if is_loop_child {
                    (INF, 0, 0)
                } else {
                    self.look_up_pn_dn(
                        child_pk, child_hand,
                        remaining.saturating_sub(1),
                    )
                };
                if cpn >= pn_threshold || cdn >= dn_threshold {
                    self.store(pos_key, att_hand, cpn, cdn, remaining, pos_key as u32);
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
                            let child_path_dep = is_loop_child
                                || self.table.has_path_dependent_disproof(
                                    child_pk, child_hand,
                                );
                            if child_path_dep {
                                self.store_path_dep(
                                    pos_key, att_hand, INF, 0,
                                    remaining, pos_key as u32, true,
                                );
                            } else {
                                self.store(pos_key, att_hand, INF, 0, remaining, pos_key as u32);
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
                                    remaining, if is_loop_child { 0 } else { pos_key as u32 }, true,
                                );
                            } else {
                                self.store(pos_key, att_hand, INF, 0, remaining, pos_key as u32);
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

                let captured = profile_timed!(self, do_move_ns, do_move_count,
                    board.do_move(m));
                self.mid(board, pn_threshold, dn_threshold, ply + 1, !or_node);
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
            // OR NM remaining 伝播: 全子 NM の remaining の最小値を追跡
            let mut or_nm_min_remaining: u16;

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
                        if self.path[..self.path_len].contains(&child_fh) {
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
                        // GHI 伝播: 子の反証が経路依存なら親も経路依存
                        if !self.path[..self.path_len].contains(&child_fh)
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
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
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
                            parent_nm_remaining, pos_key as u32, true,
                        );
                    } else {
                        self.store(
                            pos_key, att_hand,
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
                    self.path_len -= 1;
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
                    let is_loop_child = self.path[..self.path_len].contains(&child_fh);
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

                if proved_or_disproved {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_loop_break_proved += 1; }
                    debug_assert_eq!(self.path[self.path_len - 1], full_hash);
                    self.alpha_x_filter_active = save_alpha_x;
                    self.path_len -= 1;
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
                        .saturating_add((cd_grouped_count as u64 - 1) * PN_UNIT as u64)
                        .min(INF as u64) as u32;
                } else if unproven_count > 0 {
                    current_pn = (max_cpn as u64)
                        .saturating_add((unproven_count as u64 - 1) * PN_UNIT as u64)
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
            let (eff_pn_th, eff_dn_th) = if loop_child_count > 0 {
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

            let _pre_mid_nodes = self.nodes_searched;
            self.mid(
                board,
                child_pn_th,
                child_dn_th,
                ply + 1,
                !or_node,
            );
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
        self.path_len -= 1;
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
        // num_defenses ベースで既に大きな値(例: 5×S=80)になるため，
        // 逃げ場による補正は控えめに保つ．
        let base = if safe_escapes == 0 {
            // 逃げ場なし: 合駒・駒取りのみ → 詰みやすい(2/3 に割引)
            (num_defenses * 2 / 3).max(1) * PN_UNIT
        } else {
            // 逃げ場あり: 応手数ベース + 逃げ場 × S/2
            num_defenses * PN_UNIT + safe_escapes * PN_UNIT / 2
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
    pub(super) fn heuristic_or_pn(&self, board: &Board, num_checks: u32) -> u32 {
        if num_checks == 0 {
            return INF; // 王手なし → 不詰(呼び出し側で処理済みのはず)
        }

        let defender = board.turn.opponent();
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return PN_UNIT,
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
            return 8 * PN_UNIT;
        }

        // 拡張 heuristic_or_pn: S〜8S の範囲で num_checks と safe_escapes の
        // 二次元スケーリング(KomoringHeights の pn=10-80 に相当)．
        //
        // 基本方針:
        // - safe_escapes が多い → 追い詰めに手数を要する(pn↑)
        // - num_checks が少ない → 選択肢が少なく詰みにくい(pn↑)
        // - 両方が悪い場合は乗算的に大きくなる
        //
        // safe_escapes に基づくベース値(S〜4S):
        let escape_base = if safe_escapes == 0 {
            PN_UNIT // 逃げ場なし: 詰みやすい
        } else if safe_escapes == 1 {
            PN_UNIT + PN_UNIT / 2 // 1.5S
        } else if safe_escapes == 2 {
            2 * PN_UNIT // 2S
        } else if safe_escapes == 3 {
            3 * PN_UNIT // 3S
        } else {
            // safe_escapes >= 4: 4S ベース
            4 * PN_UNIT
        };

        // num_checks に基づくスケーリング(escape_base の 1.0〜2.0 倍):
        // 王手が少ないほど詰みにくい → pn を大きくする
        let adjusted_pn = if num_checks >= 8 {
            escape_base // ×1.0: 多数の王手 → ベースのまま
        } else if num_checks >= 4 {
            escape_base + escape_base / 4 // ×1.25
        } else if num_checks >= 2 {
            escape_base + escape_base / 2 // ×1.5
        } else {
            // num_checks == 1: 王手がたった1つ → ×2.0
            escape_base * 2
        };

        // 上限 8S(128): 不詰証明遅延を抑制
        adjusted_pn.min(8 * PN_UNIT)
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
}

