//! Dual フラットハッシュテーブル型転置表(Transposition Table)．
//!
//! v0.24.0: Dual TT 方式 — ProvenTT(永続エントリ)と WorkingTT(GC 対象)に分離．
//! ProvenTT には以下 3 種のエントリを格納:
//!   - proof (pn=0)
//!   - confirmed disproof (dn=0, !path_dep, remaining=INFINITE, flags bit 7=0)
//!   - refutable disproof (v0.24.75+, flags bit 7=1) —
//!     PNS からは `skip_refutable_disproof` で不可視化，MID のみ参照
//!     (aigoma-optimization.md §8.9)
//! WorkingTT には intermediate と depth-limited/path-dependent disproof を格納する．
//! これによりクラスタ飽和問題(§6.6.1)を構造的に解決する．

use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::{HAND_KINDS, PieceType};

use super::entry::{DfPnEntry, ProvenEntry};
use super::{hand_gte_forward_chain, PN_UNIT, REMAINING_INFINITE};

/// 近傍クラスタ走査の診断用グローバルカウンタ．
/// [0]: ProvenTT proof 近傍ヒット
/// [1]: ProvenTT disproof 近傍ヒット
/// [2]: WorkingTT disproof 近傍ヒット
/// [3]: 近傍走査呼び出し回数
pub(super) static NEIGHBOR_DIAG: [AtomicU64; 4] = [
    AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];

/// WorkingTT 中間エントリヒット診断カウンタ (verbose feature)．
/// [0]: exact_match ヒット回数 (hand 完全一致)
/// [1]: fc_match ヒット回数 (forward-chain dominance)
#[cfg(feature = "verbose")]
pub(super) static WORKING_DIAG: [AtomicU64; 2] = [
    AtomicU64::new(0), AtomicU64::new(0),
];

/// ProvenTT の 1 クラスタあたりのエントリ数．
///
/// 持ち駒の種類数(7) + 1 = 8 エントリとすることで，同一盤面・異なる
/// 持ち駒バリアントの proof/confirmed disproof が同一クラスタに
/// 収まりやすくなり，hand_hash 混合による proof 見逃しを軽減する．
const PROVEN_CLUSTER_SIZE: usize = 8;

/// WorkingTT の 1 クラスタあたりのエントリ数．
///
/// intermediate + depth-limited disproof 専用．
/// proof/confirmed disproof が ProvenTT に分離されたため，
/// 全エントリを working entries に使用できる．
///
/// **v0.24.27:** 6 → 8 に増加．
/// Plan D で ProvenTT から解放された 128 MB を WorkingTT の slot 拡大に再配分．
/// - 旧: 6 × 32 = 192 B/cluster (3 cache lines)
/// - 新: 8 × 32 = 256 B/cluster (4 cache lines, 整列良好)
///
/// メモリ影響 (@ 2M clusters): 384 MB → 512 MB (+128 MB，ちょうど解放分)．
/// 総 TT 消費は Plan B 以前 (896 MB) と同等に戻るが，配分が
/// ProvenTT 寄り (512+384=896) → WorkingTT 寄り (384+512=896) にシフト．
///
/// 効果: WorkingTT は overflow-limited (intermediate エントリが頻繁に衝突)．
/// slot を 33% 増やすことで overflow → eviction → 再探索サイクルを減らす．
const WORKING_CLUSTER_SIZE: usize = 8;

/// TT クラスタ数のデフォルト値(2^21 = 2M クラスタ)．
///
/// Dual TT メモリ配分 (v0.24.27):
/// - ProvenTT:  proven_num_clusters × 8 × 24B = 384 MB @ 2M clusters
/// - WorkingTT: working_num_clusters × 8 × 32B = 512 MB @ 2M clusters
const DEFAULT_NUM_CLUSTERS: usize = 1 << 21; // 2M

/// ProvenTT のクラスタ数の倍率．
/// 1 = WorkingTT と同数(2M)．
const PROVEN_CLUSTER_MULTIPLIER: usize = 1;

/// LeafDisproofTT のクラスタサイズ (4 エントリ × 16B = 64B / クラスタ = 1 cache line)．
const LEAF_CLUSTER_SIZE: usize = 4;

/// LeafDisproofTT のクラスタ数 (2^19 = 512K クラスタ)．
/// 512K × 4 entries × 16B = 32MB．WorkingTT overflow の remaining≤2 専用バッファ．
const LEAF_NUM_CLUSTERS: usize = 1 << 19;

/// LeafDisproofTT のコンパクトエントリ (16B)．
///
/// N-8 (v0.26.0): remaining ≤ 2 の depth-limited disproof を WorkingTT の
/// overflow として格納する compact entry．
/// WorkingTT の TTFlatEntry (32B) の半分のサイズで，同一メモリに 2× のエントリを保持．
///
/// - pos_key: 空スロット判定 (0 = 空)
/// - hand: hand_gte_forward_chain 比較用
/// - remaining: 格納時の remaining 値 (1 or 2)
#[derive(Clone, Copy)]
#[repr(C)]
struct TTLeafEntry {
    pos_key: u64,           // 8B
    hand: [u8; HAND_KINDS], // 7B
    remaining: u8,          // 1B (remaining 値: 1 or 2)
}

const _: () = assert!(
    std::mem::size_of::<TTLeafEntry>() == 16,
    "TTLeafEntry must be 16 bytes"
);

impl TTLeafEntry {
    const EMPTY: Self = TTLeafEntry {
        pos_key: 0,
        hand: [0; HAND_KINDS],
        remaining: 0,
    };
}

/// WorkingTT のフラットエントリ (intermediate + depth-limited/path-dep disproof)．
///
/// pos_key を含むことでクラスタ内の異なる局面を区別する．
/// pos_key = 0 は空スロットを示す(実際のハッシュ値 0 との衝突は無視可能)．
///
/// v0.24.0: DfPnEntry 圧縮(24 bytes)により TTFlatEntry は 32 bytes．
/// v0.24.24: 案 B (amount を proven entry の distance として再利用)．
/// v0.24.26: 案 D (Plan D) で proven side は別構造 `TTFlatProvenEntry` (24 bytes) へ
///           分離．WorkingTT は従来の 32 bytes を維持．
#[derive(Clone, Copy)]
#[repr(C)]
struct TTFlatEntry {
    pos_key: u64,
    entry: DfPnEntry,
}

// コンパイル時にサイズを検証
const _: () = assert!(
    std::mem::size_of::<TTFlatEntry>() == 32,
    "TTFlatEntry must be 32 bytes"
);

impl TTFlatEntry {
    const EMPTY: Self = TTFlatEntry {
        pos_key: 0,
        entry: DfPnEntry::ZERO,
    };

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.pos_key == 0
    }
}

/// ProvenTT のフラットエントリ (proof + confirmed disproof 専用)．
///
/// 案 D (Plan D): proven entry は source/remaining_flags などの
/// 探索状態を持たないため，`ProvenEntry` を使って 20 bytes (padded to 24)
/// に圧縮する．従来の `TTFlatEntry` (32 bytes) より 25% 小さく，
/// クラスタごとのキャッシュ占有を抑制する．
#[derive(Clone, Copy)]
#[repr(C)]
struct TTFlatProvenEntry {
    pos_key: u64,
    entry: ProvenEntry,
}

// コンパイル時にサイズを検証 (20 bytes raw → 8-byte alignment → 24 bytes)
const _: () = assert!(
    std::mem::size_of::<TTFlatProvenEntry>() == 24,
    "TTFlatProvenEntry must be 24 bytes"
);

impl TTFlatProvenEntry {
    const EMPTY: Self = TTFlatProvenEntry {
        pos_key: 0,
        entry: ProvenEntry::ZERO,
    };

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.pos_key == 0
    }
}

/// エントリが ProvenTT に格納されるべきかを判定する．
///
/// ProvenTT には proof (pn=0) と confirmed disproof
/// (dn=0, !path_dependent, remaining=INFINITE) を格納する．
/// confirmed disproof は IDS depth 切り替え時に
/// `clear_proven_disproofs()` で除去される(NoMate バグ対策)．
#[inline(always)]
fn is_proven_entry(pn: u32, dn: u32, remaining: u16, path_dependent: bool) -> bool {
    pn == 0 || (dn == 0 && !path_dependent && remaining == REMAINING_INFINITE)
}

/// Dual フラットハッシュテーブル型転置表(証明駒/反証駒対応)．
///
/// v0.24.0: ProvenTT + WorkingTT の 2 テーブル構成．
/// - ProvenTT: proof(pn=0) + confirmed disproof(dn=0, !path_dep, remaining=INFINITE)
/// - WorkingTT: intermediate(pn>0, dn>0) + depth-limited/path-dep disproof
///
/// ProvenTT の永続エントリが WorkingTT のクラスタを圧迫しないため，
/// クラスタ飽和問題(§6.6.1)が構造的に解消される．
pub(super) struct TranspositionTable {
    /// ProvenTT: 永続エントリ(proof + confirmed disproof)．
    /// 案 D: TTFlatProvenEntry (24 bytes) で圧縮格納．
    proven: Vec<TTFlatProvenEntry>,
    /// WorkingTT: GC 対象エントリ(intermediate + depth-limited disproof)．
    working: Vec<TTFlatEntry>,
    /// LeafDisproofTT: remaining ≤ 2 の overflow 専用 compact テーブル (N-8, v0.26.0)．
    /// WorkingTT クラスタが飽和した際の remaining ≤ 2 エントリのフォールバック先．
    /// 16B/entry で WorkingTT (32B) の半サイズ．512K clusters × 4 = 2M entries = 32MB．
    leaf_disproofs: Vec<TTLeafEntry>,
    /// ProvenTT の `num_clusters - 1`(高速 modulo 用ビットマスク)．
    proven_mask: usize,
    /// WorkingTT の `num_clusters - 1`．
    working_mask: usize,
    /// LeafDisproofTT の `num_clusters - 1`．
    leaf_mask: usize,
    /// store_proven での amount 計算に使用する現在の ply．
    /// mid() が TT store 前にセットする．ply が小さい(ルートに近い)ほど
    /// amount が高くなり，eviction 耐性が上がる．
    pub(super) hint_ply: u32,
    /// 現在の IDS depth．confirmed disproof の ProvenTT 格納時に
    /// `ProvenEntry::new_disproof(hand, ids_depth)` で確認 depth を記録する．
    /// `mid_fallback` の IDS ループで更新される．
    pub(super) current_ids_depth: u32,
    /// WorkingTT のクラスタ overflow 累積カウンタ．
    /// store 時に空きスロットが見つからず eviction が発生するたびにインクリメント．
    /// `drain_working_overflow()` でリセットし，呼び出し側が GC 判断に使用する．
    working_overflow_since_gc: u64,
    /// WorkingTT の1クラスタあたりピーク充填数（探索中の最大値）．
    pub(super) working_peak_cluster_fill: usize,
    /// Overflow 発生時のクラスタ内 distinct pos_key 数の累積(診断用)．
    pub(super) overflow_distinct_keys_sum: u64,
    /// Overflow 発生時のクラスタ内 intermediate エントリ数の累積(診断用)．
    pub(super) overflow_intermediate_sum: u64,
    /// Overflow 発生時のクラスタ内 disproof エントリ数の累積(診断用)．
    pub(super) overflow_disproof_sum: u64,
    /// Overflow サンプリング回数(診断用)．
    pub(super) overflow_sample_count: u64,
    /// Overflow 時のクラスタ内 disproof の remaining 分布(診断用)．
    /// [0]: remaining=0, [1]: remaining=1..4, [2]: remaining=5..31, [3]: remaining=INFINITE
    pub(super) overflow_disproof_remaining: [u64; 4],
    /// Overflow 時のクラスタ内 path_dependent disproof の数(診断用)．
    pub(super) overflow_disproof_path_dep: u64,
    /// TT エントリ溢れ(置換)の発生回数．
    #[cfg(feature = "profile")]
    pub(super) overflow_count: u64,
    /// ProvenTT での overflow 回数．
    #[cfg(feature = "profile")]
    pub(super) proven_overflow_count: u64,
    /// WorkingTT での overflow 回数．
    #[cfg(feature = "profile")]
    pub(super) working_overflow_count: u64,
    /// TT エントリ溢れで置換対象が見つからなかった回数．
    #[cfg(feature = "profile")]
    pub(super) overflow_no_victim_count: u64,
    /// 1 クラスタあたりのエントリ数の最大値．
    #[cfg(feature = "profile")]
    pub(super) max_entries_per_position: usize,
    // --- TT 増加診断カウンタ (verbose feature でのみ使用) ---
    #[cfg(feature = "verbose")]
    pub(super) diag_proof_inserts: u64,
    #[cfg(feature = "verbose")]
    pub(super) diag_disproof_inserts: u64,
    #[cfg(feature = "verbose")]
    pub(super) diag_intermediate_new: u64,
    #[cfg(feature = "verbose")]
    pub(super) diag_intermediate_update: u64,
    #[cfg(feature = "verbose")]
    pub(super) diag_dominated_skip: u64,
    #[cfg(feature = "verbose")]
    pub(super) diag_remaining_dist: [u64; 33],
    /// (v0.24.76) disproof 挿入の内訳: confirmed (ProvenTT), refutable (ProvenTT), working (WorkingTT)
    #[cfg(feature = "tt_diag")]
    pub(super) diag_disproof_confirmed: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_disproof_refutable: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_disproof_refutable_skip: u64,
    #[cfg(feature = "tt_diag")]
    pub(super) diag_disproof_working: u64,
    /// (v0.25.7) WorkingTT 中間エントリヒット回数: exact_match / fc_match 別カウント．
    /// retain_working_intermediates で保持されたエントリが再利用されているかを確認する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_working_intermediate_hits: [std::sync::atomic::AtomicU64; 2],
    /// (v0.25.7) 直近の retain_working_intermediates で保持されたエントリ数．
    /// IDS 遷移後の保持エントリ数を診断するために使用する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_last_retained_count: u64,
    /// (v0.25.7) 保持エントリの pn 分布 (累積): [1, 2-7, 8-63, 64-511, 512-4095, 4096+]．
    /// Hypothesis 1C (pn/dn キャップ値選定) のための診断用．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_retained_pn_dist: [u64; 6],
    /// (v0.25.7) 保持エントリの dn 分布 (累積): [1, 2-7, 8-63, 64-511, 512-4095, 4096+]．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_retained_dn_dist: [u64; 6],
    /// (v0.25.7, Hypothesis 1C) retain_working_intermediates で保持するエントリの
    /// pn/dn 上限値．u32::MAX はキャップなし(デフォルト)．
    /// 浅い IDS depth での過大評価 pn/dn が深い depth での探索優先度を歪めるのを防ぐ．
    pub(super) retain_pn_dn_cap: u32,
    /// (v0.25.0) `remaining < threshold` の depth-limited disproof は WorkingTT
    /// への格納をスキップする．デフォルト 0 (スキップなし)．
    /// `path_dependent` と `remaining == REMAINING_INFINITE` は対象外．
    /// ply 18 ベンチマークで WorkingTT churn (87% eviction) の削減を狙う．
    pub(super) disproof_remaining_threshold: u16,
    /// (v0.25.0) 閾値スキップにより格納を省略した depth-limited disproof の累計数．
    pub(super) diag_disproof_threshold_skip: u64,
    /// (N-8, v0.26.0) LeafDisproofTT への挿入数 (overflow fallback 含む)．
    pub(super) diag_leaf_inserts: u64,
    /// (N-8, v0.26.0) LeafDisproofTT からのヒット数 (look_up_working からの atomic カウント)．
    pub(super) diag_leaf_hits: std::sync::atomic::AtomicU64,
}

impl TranspositionTable {
    /// 転置表を生成する(デフォルトサイズ)．
    pub(super) fn new() -> Self {
        Self::with_clusters(DEFAULT_NUM_CLUSTERS)
    }

    /// 指定クラスタ数で転置表を生成する．
    fn with_clusters(num_clusters: usize) -> Self {
        let working_clusters = num_clusters.next_power_of_two();
        let proven_clusters = (num_clusters * PROVEN_CLUSTER_MULTIPLIER).next_power_of_two();
        let proven_total = proven_clusters * PROVEN_CLUSTER_SIZE;
        let working_total = working_clusters * WORKING_CLUSTER_SIZE;
        let leaf_total = LEAF_NUM_CLUSTERS * LEAF_CLUSTER_SIZE;
        TranspositionTable {
            proven: vec![TTFlatProvenEntry::EMPTY; proven_total],
            working: vec![TTFlatEntry::EMPTY; working_total],
            leaf_disproofs: vec![TTLeafEntry::EMPTY; leaf_total],
            proven_mask: proven_clusters - 1,
            working_mask: working_clusters - 1,
            leaf_mask: LEAF_NUM_CLUSTERS - 1,
            hint_ply: 0,
            current_ids_depth: 0,
            working_overflow_since_gc: 0,
            working_peak_cluster_fill: 0,
            overflow_distinct_keys_sum: 0,
            overflow_intermediate_sum: 0,
            overflow_disproof_sum: 0,
            overflow_sample_count: 0,
            overflow_disproof_remaining: [0; 4],
            overflow_disproof_path_dep: 0,
            #[cfg(feature = "profile")]
            overflow_count: 0,
            #[cfg(feature = "profile")]
            proven_overflow_count: 0,
            #[cfg(feature = "profile")]
            working_overflow_count: 0,
            #[cfg(feature = "profile")]
            overflow_no_victim_count: 0,
            #[cfg(feature = "profile")]
            max_entries_per_position: 0,
            #[cfg(feature = "verbose")]
            diag_proof_inserts: 0,
            #[cfg(feature = "verbose")]
            diag_disproof_inserts: 0,
            #[cfg(feature = "verbose")]
            diag_intermediate_new: 0,
            #[cfg(feature = "verbose")]
            diag_intermediate_update: 0,
            #[cfg(feature = "verbose")]
            diag_dominated_skip: 0,
            #[cfg(feature = "verbose")]
            diag_remaining_dist: [0; 33],
            #[cfg(feature = "tt_diag")]
            diag_disproof_confirmed: 0,
            #[cfg(feature = "tt_diag")]
            diag_disproof_refutable: 0,
            #[cfg(feature = "tt_diag")]
            diag_disproof_refutable_skip: 0,
            #[cfg(feature = "tt_diag")]
            diag_disproof_working: 0,
            #[cfg(feature = "tt_diag")]
            diag_working_intermediate_hits: [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ],
            #[cfg(feature = "tt_diag")]
            diag_last_retained_count: 0,
            #[cfg(feature = "tt_diag")]
            diag_retained_pn_dist: [0; 6],
            #[cfg(feature = "tt_diag")]
            diag_retained_dn_dist: [0; 6],
            retain_pn_dn_cap: u32::MAX,
            disproof_remaining_threshold: 0,
            diag_disproof_threshold_skip: 0,
            diag_leaf_inserts: 0,
            diag_leaf_hits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// WorkingTT エントリの pn/dn 分布を収集する (分析用)．
    ///
    /// 空エントリ (pos_key == 0) を除く全エントリの pn/dn を log2 バケットに分類する．
    ///
    /// 返り値: (pn_hist, dn_hist, joint_hist)
    /// - pn_hist[k]: pn が バケット k に属するエントリ数
    /// - dn_hist[k]: dn が バケット k に属するエントリ数
    /// - joint_hist[i * 32 + j]: pn がバケット i かつ dn がバケット j のエントリ数
    ///
    /// バケット定義 (32 バケット):
    /// - バケット 0: val == 0
    /// - バケット k (1..=30): 2^(k-1) <= val < 2^k
    /// - バケット 31: val >= 2^30 (INF = u32::MAX を含む)
    pub(super) fn collect_working_pn_dn_dist(&self) -> ([u64; 32], [u64; 32], Vec<u64>) {
        let mut pn_hist = [0u64; 32];
        let mut dn_hist = [0u64; 32];
        let mut joint_hist = vec![0u64; 32 * 32];

        for flat in &self.working {
            if flat.is_empty() {
                continue;
            }
            let pi = Self::pn_dn_log2_bucket(flat.entry.pn);
            let di = Self::pn_dn_log2_bucket(flat.entry.dn);
            pn_hist[pi] += 1;
            dn_hist[di] += 1;
            joint_hist[pi * 32 + di] += 1;
        }

        (pn_hist, dn_hist, joint_hist)
    }

    /// pn/dn 値を log2 バケットインデックスに変換する．
    #[inline(always)]
    fn pn_dn_log2_bucket(val: u32) -> usize {
        match val {
            0 => 0,
            u32::MAX => 31,
            v => ((32u32 - v.leading_zeros()) as usize).min(30),
        }
    }

    /// retain_working_intermediates で保持するエントリの pn/dn 上限を設定する (v0.25.7)．
    ///
    /// `u32::MAX` はキャップなし (デフォルト)．
    /// Hypothesis 1C: 浅い IDS depth での過大評価が深い depth の探索効率を下げる場合，
    /// 適切な上限でクリップすることで探索優先度の歪みを軽減できる．
    pub(super) fn set_retain_pn_dn_cap(&mut self, cap: u32) {
        self.retain_pn_dn_cap = cap;
    }

    /// depth-limited disproof の格納閾値を設定する (v0.25.0)．
    ///
    /// `remaining < threshold` の depth-limited disproof (path_dependent でなく
    /// confirmed でないもの) は WorkingTT への格納をスキップする．
    pub(super) fn set_disproof_remaining_threshold(&mut self, threshold: u16) {
        self.disproof_remaining_threshold = threshold;
    }

    // ---- ヘルパー ----

/// `pos_key == 0` をセンチネル(空スロット)と衝突させない変換．
    #[inline(always)]
    fn safe_key(pos_key: u64) -> u64 {
        pos_key | 1
    }

    /// 持ち駒からハッシュ値を計算する(Zobrist ベース)．
    ///
    /// Zobrist テーブルの持ち駒キー(color=0)を XOR して生成する．
    /// Zobrist ベースのため:
    /// - ランダムキーによる良好なハッシュ分散
    /// - 持ち駒 1 枚の増減が XOR 差分で O(1) 計算可能
    ///   → 部分集合クラスタの特定が高速
    #[inline(always)]
    fn hand_hash(hand: &[u8; HAND_KINDS]) -> u64 {
        use crate::zobrist::ZOBRIST;
        use crate::types::Color;
        let mut h = 0u64;
        for k in 0..HAND_KINDS {
            h ^= ZOBRIST.hand_hash(Color::Black, k, hand[k] as usize);
        }
        h
    }

    /// hand_hash から Zobrist 差分で変換した値を計算する．
    ///
    /// `hand[k]` が `old_val` → `new_val` に変化したときのハッシュ差分を返す．
    /// `new_hash = old_hash ^ hand_hash_diff(k, old_val, new_val)`
    #[inline(always)]
    fn hand_hash_diff(k: usize, old_val: u8, new_val: u8) -> u64 {
        use crate::zobrist::ZOBRIST;
        use crate::types::Color;
        ZOBRIST.hand_hash(Color::Black, k, old_val as usize)
            ^ ZOBRIST.hand_hash(Color::Black, k, new_val as usize)
    }

    /// hand_hash 値から直接クラスタ開始位置を計算する(ProvenTT)．
    #[inline(always)]
    fn proven_cluster_start_from_hash(&self, pos_key: u64, hh: u64) -> usize {
        ((pos_key ^ hh) as usize & self.proven_mask) * PROVEN_CLUSTER_SIZE
    }

    /// hand_hash 値から直接クラスタ開始位置を計算する(WorkingTT)．
    #[inline(always)]
    fn working_cluster_start_from_hash(&self, pos_key: u64, hh: u64) -> usize {
        ((pos_key ^ hh) as usize & self.working_mask) * WORKING_CLUSTER_SIZE
    }

    /// ProvenTT のクラスタ開始インデックス．
    /// pos_key XOR hand_hash で hand バリアントを異なるクラスタに分散する．
    #[inline(always)]
    fn proven_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::hand_hash(hand);
        ((mixed as usize) & self.proven_mask) * PROVEN_CLUSTER_SIZE
    }

    /// LeafDisproofTT のクラスタ開始インデックス (N-8)．
    ///
    /// WorkingTT と同じ fc_hand_hash を使用し，leaf_mask でクラスタを選択する．
    #[inline(always)]
    fn leaf_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::fc_hand_hash(hand);
        ((mixed as usize) & self.leaf_mask) * LEAF_CLUSTER_SIZE
    }

    /// WorkingTT のクラスタ開始インデックス．
    ///
    /// (v0.24.69) forward-chain 正規化 hand hash を使用する:
    /// Pawn(0)/Lance(1)/Rook(6) の個別カウントではなく総和で hash を計算し，
    /// forward-chain 等価な手駒変種を同一クラスタに集約する．
    /// これにより chain aigoma の intermediate エントリが共有可能になる．
    ///
    /// Knight(2)/Silver(3)/Gold(4)/Bishop(5) は forward-chain で代替不可のため
    /// 個別にhash する（従来通り）．
    #[inline(always)]
    fn working_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::fc_hand_hash(hand);
        ((mixed as usize) & self.working_mask) * WORKING_CLUSTER_SIZE
    }

    /// Forward-chain 正規化 hand hash (v0.24.69)．
    ///
    /// Pawn(0)/Lance(1)/Rook(6) の個別カウントではなく総和を単一インデックス
    /// (Pawn カウント) として hash する．これにより forward-chain 等価な手駒:
    ///   [3,2,0,0,0,0,1] と [5,0,0,0,0,0,1] と [0,0,0,0,0,0,6]
    /// が全て fc_total=6 → 同一クラスタにマッピングされる．
    ///
    /// Knight/Silver/Gold/Bishop は代替不可のため個別に hash する．
    #[inline(always)]
    fn fc_hand_hash(hand: &[u8; HAND_KINDS]) -> u64 {
        use crate::zobrist::ZOBRIST;
        use crate::types::Color;
        // Forward-chain components: pawn(0) + lance(1) + rook(6) → 総和を pawn slot で hash
        let fc_total = (hand[0] as usize) + (hand[1] as usize) + (hand[6] as usize);
        // fc_total は最大 18+4+2=24 だが，hand_hash table は MAX_HAND_STATES
        // (= 19 for pawn) なので値域 [0, 18] にクランプする．
        // fc_total 19〜24 は全て 18 と同一ハッシュ → 同一クラスタに集約される．
        // Pawn 18枚 + Lance/Rook 1枚以上の局面は実戦的に稀であり影響は軽微．
        let fc_clamped = fc_total.min(18);
        let mut h = ZOBRIST.hand_hash(Color::Black, 0, fc_clamped);
        // Non-forward-chain components: knight(2), silver(3), gold(4), bishop(5) は個別
        h ^= ZOBRIST.hand_hash(Color::Black, 2, hand[2] as usize);
        h ^= ZOBRIST.hand_hash(Color::Black, 3, hand[3] as usize);
        h ^= ZOBRIST.hand_hash(Color::Black, 4, hand[4] as usize);
        h ^= ZOBRIST.hand_hash(Color::Black, 5, hand[5] as usize);
        h
    }

    /// ProvenTT のクラスタスライスを返す(不変参照)．
    #[inline(always)]
    fn proven_cluster(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> &[TTFlatProvenEntry] {
        let start = self.proven_cluster_start(pos_key, hand);
        &self.proven[start..start + PROVEN_CLUSTER_SIZE]
    }

    /// WorkingTT のクラスタスライスを返す(不変参照)．
    #[inline(always)]
    fn working_cluster(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> &[TTFlatEntry] {
        let start = self.working_cluster_start(pos_key, hand);
        &self.working[start..start + WORKING_CLUSTER_SIZE]
    }

    /// ProvenTT から指定 pos_key+hand の proof_tag を取得する．
    ///
    /// proof エントリが見つからない場合は PROOF_TAG_ABSOLUTE を返す．
    pub(super) fn get_proof_tag(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u8 {
        let pos_key = Self::safe_key(pos_key);
        let home = self.proven_cluster(pos_key, hand);
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                return e.proof_tag();
            }
        }
        super::entry::PROOF_TAG_ABSOLUTE
    }

    /// WorkingTT のみ検索: intermediate + depth-limited/path-dep disproof．
    ///
    /// `neighbor_scan=false`: 自クラスタのみ(探索ホットパス向け)．
    /// `neighbor_scan=true`: 自クラスタ + 持ち駒±1近傍(合駒チェーン時)．
    #[inline(always)]
    pub(super) fn look_up_working(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        let pos_key = Self::safe_key(pos_key);
        let working = self.working_cluster(pos_key, hand);
        let mut exact_match: Option<(u32, u32, u32)> = None;
        // (v0.24.69) fc-dominated intermediate: hand_q ≥_fc hand_e なら
        // entry の pn/dn を保守的初期値として使用可能．
        // pn(entry) は pn(true) の上界（多い資源 → 証明が容易），
        // dn(entry) は dn(true) の下界（多い資源 → 反証が困難）．
        // exact match を優先し，なければ fc-dominated を fallback とする．
        let mut fc_match: Option<(u32, u32, u32)> = None;
        for fe in working {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && (e.remaining() >= remaining || e.path_dependent())
            {
                return (e.pn, 0, e.source);
            }
            if e.pn != 0 && e.dn != 0 {
                if e.hand == *hand {
                    exact_match = Some((e.pn, e.dn, e.source));
                } else if fc_match.is_none()
                    && hand_gte_forward_chain(hand, &e.hand)
                {
                    fc_match = Some((e.pn, e.dn, e.source));
                }
            }
        }
        if let Some(m) = exact_match {
            #[cfg(feature = "verbose")]
            { WORKING_DIAG[0].fetch_add(1, Ordering::Relaxed); }
            #[cfg(feature = "tt_diag")]
            { self.diag_working_intermediate_hits[0].fetch_add(1, Ordering::Relaxed); }
            return m;
        }
        if let Some(m) = fc_match {
            #[cfg(feature = "verbose")]
            { WORKING_DIAG[1].fetch_add(1, Ordering::Relaxed); }
            #[cfg(feature = "tt_diag")]
            { self.diag_working_intermediate_hits[1].fetch_add(1, Ordering::Relaxed); }
            return m;
        }

        // 歩(k=0)の+1のみ disproof 近傍走査．
        // disproof 近傍ヒットの60%+が歩+1(合駒チェーンの歩合い排除)．
        // 追加コストは1クラスタ(WORKING_CLUSTER_SIZE エントリ)のみで極めて軽量．
        if neighbor_scan && hand[0] < PieceType::MAX_HAND_COUNT[0] {
            let base_hh = Self::hand_hash(hand);
            let diff = Self::hand_hash_diff(0, hand[0], hand[0] + 1);
            let start = self.working_cluster_start_from_hash(pos_key, base_hh ^ diff);
            let cluster = &self.working[start..start + WORKING_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.dn == 0
                    && hand_gte_forward_chain(&e.hand, hand)
                    && (e.remaining() >= remaining || e.path_dependent())
                {
                    NEIGHBOR_DIAG[2].fetch_add(1, Ordering::Relaxed);
                    return (e.pn, 0, e.source);
                }
            }
        }

        // N-8 (v0.26.0): LeafDisproofTT を検索 (remaining ≤ 2 の overflow エントリ)．
        // WorkingTT で見つからなかった場合のみチェックし，ヒット時は (PN_UNIT, 0, 0) を返す．
        if remaining <= 2 {
            let l_start = self.leaf_cluster_start(pos_key, hand);
            let l_cluster = &self.leaf_disproofs[l_start..l_start + LEAF_CLUSTER_SIZE];
            for le in l_cluster {
                if le.pos_key != pos_key { continue; }
                if hand_gte_forward_chain(&le.hand, hand)
                    && (le.remaining as u16) >= remaining
                {
                    self.diag_leaf_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return (PN_UNIT, 0, 0);
                }
            }
        }

        (PN_UNIT, PN_UNIT, 0)
    }

    /// ProvenTT のみ検索: proof + confirmed disproof．
    ///
    /// `neighbor_scan=true` の場合，自クラスタ + ±1 近傍を走査する．
    pub(super) fn look_up_proven(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        self.look_up_proven_impl(pos_key, hand, remaining, neighbor_scan, false)
    }

    /// `look_up_proven` の refutable disproof スキップ版 (v0.24.79)．
    ///
    /// Pass 2 および歩+1 近傍の disproof 走査で `is_refutable_disproof()`
    /// なエントリをスキップする．PNS の arena-limited false NM を防ぐため，
    /// PNS 実行中の `look_up_pn_dn_impl` から呼ばれる．
    ///
    /// 以前は `look_up` + 事後 `is_refutable_disproof_at` の 2 段スキャン
    /// (ProvenTT クラスタを 2 回走査) だったが，Pass 2 内にフィルタを
    /// 統合することで disproof ヒット時のクラスタ走査を 1 回に削減．
    pub(super) fn look_up_proven_skip_refutable(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        self.look_up_proven_impl(pos_key, hand, remaining, neighbor_scan, true)
    }

    #[inline]
    fn look_up_proven_impl(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
        skip_refutable: bool,
    ) -> (u32, u32, u32) {
        let pos_key = Self::safe_key(pos_key);
        let home = self.proven_cluster(pos_key, hand);
        // Tag-aware proof lookup: non-ABSOLUTE proof は tag_depth <
        // current_ids_depth の場合にスキップする．浅い IDS step で filter 付き
        // で生成された FILTER_DEPENDENT proof が深い step で誤使用されるのを防止．
        // 全 tag が ABSOLUTE の間は no-op (behavioral change なし)．
        let ids_depth = self.current_ids_depth;
        // Pass 1: proof(pn=0) — 自クラスタ
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                let tag = e.proof_tag();
                if tag != super::entry::PROOF_TAG_ABSOLUTE
                    && e.tag_depth() < ids_depth
                {
                    continue; // stale tagged proof: skip
                }
                return (0, e.dn(), e.source());
            }
        }
        // Pass 1b: proof 近傍(-1)
        if neighbor_scan {
            NEIGHBOR_DIAG[3].fetch_add(1, Ordering::Relaxed);
            let base_hh = Self::hand_hash(hand);
            for k in 0..HAND_KINDS {
                if hand[k] == 0 { continue; }
                let diff = Self::hand_hash_diff(k, hand[k], hand[k] - 1);
                let start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
                let cluster = &self.proven[start..start + PROVEN_CLUSTER_SIZE];
                for fe in cluster {
                    if fe.pos_key != pos_key { continue; }
                    let e = &fe.entry;
                    if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                        let tag = e.proof_tag();
                        if tag != super::entry::PROOF_TAG_ABSOLUTE
                            && e.tag_depth() < ids_depth
                        {
                            continue;
                        }
                        NEIGHBOR_DIAG[0].fetch_add(1, Ordering::Relaxed);
                        return (0, e.dn(), e.source());
                    }
                }
            }
        }
        // Pass 2: confirmed disproof(dn=0) — 自クラスタ
        // 注意: ProvenTT の confirmed disproof は常に depth 非依存として
        // 格納されるため，query 側の `remaining` との比較は不要 (v0.24.53 で
        // ProvenEntry.remaining フィールドを meta にリネームし remaining()
        // accessor を削除したことで明示化)．depth 情報は flags bits 1-6 の
        // disproof_depth() に格納されており，`clear_proven_disproofs_below`
        // での選択的除去で管理する．
        let _ = remaining;
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if !e.is_proof()
                && hand_gte_forward_chain(&e.hand, hand)
            {
                if skip_refutable && e.is_refutable_disproof() { continue; }
                return (e.pn(), 0, e.source());
            }
        }
        // 歩(k=0)の+1のみ disproof 近傍走査．
        // confirmed disproof の歩+1ヒットは合駒チェーンの二歩排除に有効．
        if neighbor_scan && hand[0] < PieceType::MAX_HAND_COUNT[0] {
            let base_hh = Self::hand_hash(hand);
            let diff = Self::hand_hash_diff(0, hand[0], hand[0] + 1);
            let start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
            let cluster = &self.proven[start..start + PROVEN_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if !e.is_proof()
                    && hand_gte_forward_chain(&e.hand, hand)
                {
                    if skip_refutable && e.is_refutable_disproof() { continue; }
                    NEIGHBOR_DIAG[1].fetch_add(1, Ordering::Relaxed);
                    return (e.pn(), 0, e.source());
                }
            }
        }
        (PN_UNIT, PN_UNIT, 0)
    }

    // ---------------------------------------------------------------
    // PV 復元用: 持ち駒の部分集合クラスタを走査する proof lookup
    // ---------------------------------------------------------------
    //
    // ProvenTT は `pos_key ^ hand_hash(hand)` でクラスタを決定するため，
    // 証明駒(proof hand)と検索時の持ち駒が異なるとクラスタが一致せず
    // proof を発見できない．
    //
    // PV 復元時には性能制約が緩いため，持ち駒の全部分集合に対応する
    // クラスタを走査して proof を漏れなく検出する．
    // 部分集合数が `MAX_SUBSET_CLUSTERS` を超える場合はスキップする．

    /// 部分集合クラスタ走査の上限．
    const MAX_SUBSET_CLUSTERS: usize = 128;

    /// proven entry の mate_distance を取得する (PV 抽出 fast path 用)．
    ///
    /// 自クラスタを走査し，proof エントリが存在すれば `ProvenEntry::mate_distance()`
    /// の値を返す．`store_with_best_move_and_distance` 経由で distance が
    /// 設定されていないエントリ (flags の distance_set ビットが立っていないもの)
    /// では None を返す．
    ///
    /// PV 抽出時に AND ノードで longest resistance (最長抵抗手) の子を
    /// 選択する際に使用する．
    pub(super) fn look_up_mate_distance(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> Option<u16> {
        let pos_key = Self::safe_key(pos_key);
        let ids_depth = self.current_ids_depth;
        // 自クラスタを走査して proof entry を探す
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                // tag-aware: stale tagged proof をスキップ
                let tag = e.proof_tag();
                if tag != super::entry::PROOF_TAG_ABSOLUTE
                    && e.tag_depth() < ids_depth
                {
                    continue;
                }
                return e.mate_distance();
            }
        }
        None
    }

    pub(super) fn look_up_proven_subset(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u32) {
        // look_up_proven は自クラスタ+1枚差を既に走査済み
        let normal = self.look_up_proven(pos_key, hand, remaining, true);
        if normal.0 == 0 || normal.1 == 0 {
            return normal;
        }

        let subset_count: usize = hand.iter()
            .map(|&h| (h as usize) + 1)
            .product();
        if subset_count > Self::MAX_SUBSET_CLUSTERS {
            return normal;
        }

        let pos_key = Self::safe_key(pos_key);
        let ids_depth = self.current_ids_depth;

        // Zobrist diff で部分集合を列挙
        let mut sub_hand = [0u8; HAND_KINDS];
        let mut sub_hh = Self::hand_hash(&sub_hand);
        loop {
            let start = self.proven_cluster_start_from_hash(pos_key, sub_hh);
            let cluster = &self.proven[start..start + PROVEN_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                    // tag-aware: stale tagged proof をスキップ
                    let tag = e.proof_tag();
                    if tag != super::entry::PROOF_TAG_ABSOLUTE
                        && e.tag_depth() < ids_depth
                    {
                        continue;
                    }
                    return (0, e.dn(), e.source());
                }
            }

            // 次の部分集合へ (odometer + Zobrist diff)
            let mut advanced = false;
            for k in 0..HAND_KINDS {
                if sub_hand[k] < hand[k] {
                    let old_val = sub_hand[k];
                    sub_hand[k] += 1;
                    sub_hh ^= Self::hand_hash_diff(k, old_val, sub_hand[k]);
                    advanced = true;
                    break;
                }
                // 桁上がり: sub_hand[k] を 0 にリセット
                let old_val = sub_hand[k];
                sub_hand[k] = 0;
                sub_hh ^= Self::hand_hash_diff(k, old_val, 0);
            }
            if !advanced {
                break;
            }
        }

        (PN_UNIT, PN_UNIT, 0)
    }


    /// 統合 look_up: ProvenTT → WorkingTT の順で検索．
    ///
    /// 統合 look_up: ProvenTT → WorkingTT の順で検索．
    ///
    /// `neighbor_scan` フラグを ProvenTT/WorkingTT 両方に伝播する．
    #[inline(always)]
    pub(super) fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        let proven = self.look_up_proven(pos_key, hand, remaining, neighbor_scan);
        if proven.0 == 0 || proven.1 == 0 {
            return proven;
        }
        self.look_up_working(pos_key, hand, remaining, neighbor_scan)
    }

    /// PNS 用 look_up: ProvenTT の refutable disproof をスキップ (v0.24.79)．
    ///
    /// `look_up` と同じ意味論だが，ProvenTT Pass 2 で
    /// `is_refutable_disproof()` なエントリを読み飛ばす．PNS 実行中に
    /// solver の `skip_refutable_disproof` フラグが立っているときに
    /// 使用し，arena-limited false NM cascade を防止する．
    ///
    /// 以前は `look_up` + `is_refutable_disproof_at` の 2 段スキャンで
    /// フィルタしていたが，`look_up_proven_skip_refutable` に統合して
    /// disproof ヒット時のクラスタ走査を 1 回に削減した．
    #[inline(always)]
    pub(super) fn look_up_skip_refutable(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        neighbor_scan: bool,
    ) -> (u32, u32, u32) {
        let proven =
            self.look_up_proven_skip_refutable(pos_key, hand, remaining, neighbor_scan);
        if proven.0 == 0 || proven.1 == 0 {
            return proven;
        }
        self.look_up_working(pos_key, hand, remaining, neighbor_scan)
    }

    /// 施策 X-N A-fix (v0.24.58): 同一 pos_key の hand バリアント数を
    /// 自クラスタから推定する．
    ///
    /// 本関数は `try_prefilter_block` で `neighbor_scan` を条件付きで
    /// 有効化するために使用される．自クラスタに pos_key と一致するが
    /// hand が異なる ProvenTT エントリが 1 個以上存在する場合 `true` を返す．
    ///
    /// 原理:
    /// - ProvenTT クラスタは `pos_key ^ hand_hash(hand)` でインデックス．
    ///   異なる hand は異なるクラスタに分散するのが基本．
    /// - 自クラスタ内に pos_key 一致かつ hand 不一致のエントリがある場合，
    ///   hand_hash の衝突か同一 pos_key を異なる hand で store したかの
    ///   いずれか．いずれの場合も hand バリアントが TT 上に複数存在し，
    ///   neighbor_scan で追加 proof 発見の期待値が高い．
    /// - 真の variant 数 (全クラスタの全 hand) の正確なカウントは高価な
    ///   ため本関数は **cheap な下界推定** として機能する．
    ///
    /// cost: 自クラスタ走査 1 回 (PROVEN_CLUSTER_SIZE エントリ).
    pub(super) fn proven_has_other_hand_variant(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.is_proof()
                && fe.entry.hand != *hand
            {
                return true;
            }
        }
        false
    }

    /// 指定局面に proof エントリ(pn=0)が存在するかチェックする．
    ///
    /// 自クラスタ + 持ち駒-1の近傍クラスタ(±1限定)を走査する．
    pub(super) fn has_proof(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> bool {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.is_proof()
                && hand_gte_forward_chain(hand, &fe.entry.hand)
            {
                return true;
            }
        }
        let base_hh = Self::hand_hash(hand);
        for k in 0..HAND_KINDS {
            if hand[k] == 0 { continue; }
            let diff = Self::hand_hash_diff(k, hand[k], hand[k] - 1);
            let start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
            let cluster = &self.proven[start..start + PROVEN_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key == pos_key
                    && fe.entry.is_proof()
                    && hand_gte_forward_chain(hand, &fe.entry.hand)
                {
                    return true;
                }
            }
        }
        false
    }

    /// TT Best Move を参照する．
    ///
    /// WorkingTT → ProvenTT の順で検索．ProvenTT は hand_hash 混合インデクシング
    /// のため，異なる hand バリアントの best_move は別クラスタに格納されており
    /// 検索されない．ただし best_move は pure hint（手順最適化用）であり
    /// 正確性には影響しない．
    #[inline(always)]
    pub(super) fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        // WorkingTT (intermediate entries have best_move)
        for fe in self.working_cluster(pos_key, hand) {
            if fe.pos_key == pos_key && fe.entry.hand == *hand && fe.entry.best_move != 0 {
                return fe.entry.best_move;
            }
        }
        // ProvenTT fallback (proof/disproof may also have best_move)
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key && fe.entry.hand == *hand && fe.entry.best_move != 0 {
                return fe.entry.best_move;
            }
        }
        0
    }

    /// 転置表を更新する．
    #[inline(always)]
    pub(super) fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, 0, 0);
    }

    /// ベストムーブ付きで転置表を更新する．
    #[inline(always)]
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
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, best_move, 0);
    }

    /// ベストムーブと詰み手数付きで転置表を更新する (proven entry 用)．
    ///
    /// `mate_distance` は pn=0 のときのみ意味を持ち，この局面から詰みまでの
    /// 残り手数を保存する．PV 抽出時に AND ノードで再帰なしに longest
    /// resistance の child を選択するために使う．非 proven entry の場合は
    /// 0 を指定する．
    #[inline(always)]
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
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false,
            best_move, mate_distance);
    }

    /// 施策 X (v0.24.53+): tag 付き proof を ProvenTT に格納する．
    ///
    /// 通常の `store` 経路と異なり，明示的に proof_tag (FILTER_DEPENDENT /
    /// PROVISIONAL / ABSOLUTE) を指定して永続性を制御する．non-ABSOLUTE tag
    /// を指定した proof は `clear_proven_disproofs_below(min_depth)` で
    /// `tag_depth < min_depth` の場合に除去される．
    ///
    /// 呼出用途:
    /// - 施策 α (境界層 chain drop filter): FILTER_DEPENDENT で store
    /// - 施策 A-6 (PNS 境界責任転嫁): PROVISIONAL で store
    ///
    /// pn != 0 を指定すると panic する (tagged proof は必ず確定証明)．
    pub(super) fn store_tagged_proof(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        best_move: u16,
        mate_distance: u16,
        tag: u8,
        tag_depth: u32,
    ) {
        let pos_key = Self::safe_key(pos_key);
        let new_entry = super::entry::ProvenEntry::new_tagged_proof(
            hand, best_move, mate_distance, tag, tag_depth,
        );
        let new_priority = new_entry.amount();
        let p_start = self.proven_cluster_start(pos_key, &hand);

        // 同一 hand の proof がいれば priority 比較で上書き判定
        {
            let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
            for fe in p_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.is_proof() && fe.entry.hand == hand {
                    if new_priority >= fe.entry.amount() {
                        fe.pos_key = 0;
                    } else {
                        return; // 既存の方が価値高
                    }
                }
            }
        }

        // 空スロット or 最弱スロットに挿入
        let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
        // Phase 1: 空スロット
        for fe in p_cluster.iter_mut() {
            if fe.pos_key == 0 {
                fe.pos_key = pos_key;
                fe.entry = new_entry;
                return;
            }
        }
        // Phase 2: 最弱の eviction (eviction priority 最小の entry と置換)
        let mut min_idx: usize = 0;
        let mut min_prio = u8::MAX;
        for (i, fe) in p_cluster.iter().enumerate() {
            let prio = fe.entry.amount();
            if prio < min_prio {
                min_prio = prio;
                min_idx = i;
            }
        }
        if new_priority > min_prio {
            p_cluster[min_idx].pos_key = pos_key;
            p_cluster[min_idx].entry = new_entry;
        }
    }

    /// テスト用: tagged proof を直接ストアする (store_tagged_proof のエイリアス)．
    #[cfg(test)]
    pub(super) fn store_tagged_proof_for_test(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        best_move: u16,
        mate_distance: u16,
        tag: u8,
        tag_depth: u32,
    ) {
        self.store_tagged_proof(pos_key, hand, best_move, mate_distance, tag, tag_depth);
    }

    /// 経路依存フラグ付きで転置表を更新する．
    #[inline(always)]
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
        self.store_impl(pos_key, hand, pn, dn, remaining, source, path_dependent, 0, 0);
    }

    #[inline(always)]
    fn store_impl(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        path_dependent: bool,
        best_move: u16,
        mate_distance: u16,
    ) {
        let pos_key = Self::safe_key(pos_key);
        #[cfg(feature = "verbose")]
        let rem_idx = if remaining == REMAINING_INFINITE { 32 } else { (remaining as usize).min(31) };

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        // ProvenTT をチェック (Plan D: ProvenEntry ベース)
        // 注意: v0.24.53 以降，ProvenTT の confirmed disproof は depth 非依存
        // として格納されるため `remaining` との比較は不要．
        for fe in self.proven_cluster(pos_key, &hand) {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(&hand, &e.hand) {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
            if !e.is_proof()
                && !path_dependent
                && hand_gte_forward_chain(&e.hand, &hand)
            {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
        }

        if is_proven_entry(pn, dn, remaining, path_dependent) {
            // === ProvenTT への挿入 (Plan D: ProvenEntry を直接構築) ===
            self.store_proven(pos_key, hand, pn == 0, best_move, mate_distance,
                #[cfg(feature = "verbose")] rem_idx);
            return;
        }

        let new_entry = DfPnEntry::new(
            source, pn, dn, hand, remaining, path_dependent, best_move, 0,
        );

        if dn == 0 {
            // === B-2 (v0.25.0): 浅い remaining の depth-limited disproof をスキップ ===
            // path_dependent は GHI 情報を含むため対象外．REMAINING_INFINITE は
            // ProvenTT 経路 (is_proven_entry) で先に処理されるためここには来ない．
            if !path_dependent
                && self.disproof_remaining_threshold > 0
                && remaining < self.disproof_remaining_threshold
            {
                self.diag_disproof_threshold_skip += 1;
                return;
            }
            // === WorkingTT への depth-limited / path-dep disproof 挿入 ===
            self.store_working_disproof(pos_key, hand, remaining, new_entry,
                #[cfg(feature = "verbose")] rem_idx);
        } else {
            // === WorkingTT への intermediate 挿入 ===
            self.store_working_intermediate(pos_key, hand, pn, dn, remaining, source, best_move,
                #[cfg(feature = "verbose")] rem_idx);
        }
    }

    /// ProvenTT に proof または confirmed disproof を挿入する．
    ///
    /// **積極的エントリ削減:** 同一 pos_key の proof/disproof のうち，
    /// 同一 hand の既存エントリは新しい方で置換する．これにより ProvenTT を
    /// 恒常的に小さく保ち，クラスタ飽和と WorkingTT への圧迫を防ぐ．
    ///
    /// 案 D: ProvenEntry を直接構築する(DfPnEntry を経由しない)．
    fn store_proven(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        is_proof: bool,
        best_move: u16,
        mate_distance: u16,
        #[cfg(feature = "verbose")] rem_idx: usize,
    ) {
        let new_entry = if is_proof {
            ProvenEntry::new_proof(hand, best_move, mate_distance)
        } else {
            ProvenEntry::new_disproof(hand, self.current_ids_depth)
        };
        let new_priority = new_entry.amount();

        let p_start = self.proven_cluster_start(pos_key, &hand);

        if is_proof {
            // === 積極的削減: 同一 pos_key の proof を整理 ===
            // 被支配 proof(自 hand が支配されるもの)を除去．
            // 同一 hand の proof は amount (distance-aware が優先) で比較．
            let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
            for fe in p_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.is_proof() {
                    // 被支配 proof: hand が支配される → 除去
                    if hand_gte_forward_chain(&fe.entry.hand, &hand) {
                        fe.pos_key = 0;
                        continue;
                    }
                    // 同一 hand の proof: amount が高い方を残す
                    if fe.entry.hand == hand {
                        if new_priority >= fe.entry.amount() {
                            fe.pos_key = 0;
                        } else {
                            return; // 既存の方が価値高 → 挿入不要
                        }
                    }
                }
            }

            // WorkingTT の被支配 intermediate も除去
            let w_start = self.working_cluster_start(pos_key, &hand);
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            for fe in w_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.dn == 0 { continue; }
                if hand_gte_forward_chain(&fe.entry.hand, &hand) {
                    fe.pos_key = 0;
                }
            }
        } else {
            // === 積極的削減: 同一 pos_key・同一 hand の confirmed disproof を整理 ===
            let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
            for fe in p_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if !fe.entry.is_proof() && fe.entry.hand == hand {
                    // 同じ disproof: 新エントリで置換(方針: 最新優先)
                    fe.pos_key = 0;
                }
            }

            // WorkingTT の同一 pos_key エントリを積極的に除去(proof 以外)．
            // confirmed disproof は「この局面は不詰」を意味するため，
            // 同一 pos_key の中間エントリや depth-limited disproof は
            // confirmed disproof に包含される．hand バリアントが異なるものも
            // 除去するのは意図的: confirmed disproof の hand_gte 優越により
            // 「この hand 以上なら不詰」が保証されるため，それより弱い hand の
            // 中間エントリは不要．WorkingTT のクラスタ圧迫を防ぐトレードオフ．
            let w_start = self.working_cluster_start(pos_key, &hand);
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            for fe in w_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.pn == 0 { continue; } // proof は ProvenTT にあるが防衛的に保護
                fe.pos_key = 0;
            }
        }

        // ProvenTT に挿入
        // (p_cluster は上記の WorkingTT mutable borrow で借用が切れるため再取得)
        let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
        if let Some(slot) = p_cluster.iter_mut().find(|fe| fe.is_empty()) {
            slot.pos_key = pos_key;
            slot.entry = new_entry;
            #[cfg(feature = "verbose")] {
                if is_proof { self.diag_proof_inserts += 1; }
                else { self.diag_disproof_inserts += 1; }
                self.diag_remaining_dist[rem_idx] += 1;
            }
            #[cfg(feature = "tt_diag")]
            if !is_proof { self.diag_disproof_confirmed += 1; }
            return;
        }
        // 満杯 → ProvenTT 専用の replace
        #[cfg(feature = "profile")]
        { self.proven_overflow_count += 1; }
        if Self::replace_weakest_proven(p_cluster, pos_key, new_entry) {
            #[cfg(feature = "verbose")] {
                if is_proof { self.diag_proof_inserts += 1; }
                else { self.diag_disproof_inserts += 1; }
                self.diag_remaining_dist[rem_idx] += 1;
            }
            #[cfg(feature = "tt_diag")]
            if !is_proof { self.diag_disproof_confirmed += 1; }
        }
    }

    /// refutable check で確認された NM を ProvenTT に格納する (v0.24.75)．
    ///
    /// confirmed disproof と同じ ProvenTT クラスタに格納するが，
    /// `ProvenEntry::is_refutable_disproof()` = true のフラグ (flags bit 7) を
    /// 持つ．
    ///
    /// TT レベルの標準 lookup (`look_up_proven` / `look_up` /
    /// `has_refutable_or_confirmed_disproof`) は confirmed / refutable を
    /// 区別せず両方を返す．PNS の arena-limited false NM 防止は
    /// solver 側の `skip_refutable_disproof` フラグ (PNS 実行中に有効化)
    /// 経由で `look_up_pn_dn_impl` が `look_up_skip_refutable` を呼び，
    /// ProvenTT Pass 2 で bit 7 を読み飛ばすことで実現する．
    pub(super) fn store_refutable_disproof(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
    ) {
        let pos_key = Self::safe_key(pos_key);

        let p_start = self.proven_cluster_start(pos_key, &hand);

        // hand_gte 支配チェック: 既存エントリが新 hand を支配するなら挿入不要
        {
            let p_cluster = &self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
            for fe in p_cluster {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                // 既存 proof が支配 → 不詰ではないので挿入不要
                if e.is_proof() && hand_gte_forward_chain(&hand, &e.hand) {
                    return;
                }
                // 既存 disproof (confirmed/refutable) が支配 → 冗長なので挿入不要
                if !e.is_proof() && hand_gte_forward_chain(&e.hand, &hand) {
                    #[cfg(feature = "tt_diag")]
                    { self.diag_disproof_refutable_skip += 1; }
                    return;
                }
            }
        }

        let new_entry = ProvenEntry {
            hand,
            flags: ProvenEntry::encode_refutable_disproof_flags(self.current_ids_depth),
            best_move: 0,
            meta: 0,
        };

        // 新エントリに支配される既存 refutable disproof を除去
        let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
        for fe in p_cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            if fe.entry.is_refutable_disproof()
                && hand_gte_forward_chain(&hand, &fe.entry.hand)
            {
                fe.pos_key = 0;
            }
        }

        // 挿入
        let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];
        if let Some(slot) = p_cluster.iter_mut().find(|fe| fe.is_empty()) {
            slot.pos_key = pos_key;
            slot.entry = new_entry;
            #[cfg(feature = "tt_diag")]
            { self.diag_disproof_refutable += 1; }
            return;
        }
        // 満杯 → 最弱エントリを置換
        #[cfg(feature = "profile")]
        { self.proven_overflow_count += 1; }
        if Self::replace_weakest_proven(p_cluster, pos_key, new_entry) {
            #[cfg(feature = "tt_diag")]
            { self.diag_disproof_refutable += 1; }
        }
    }

    /// refutable disproof を lookup する (v0.24.75)．
    ///
    /// `all_checks_refutable_by_tt` 専用．通常の `look_up_proven` では
    /// refutable disproof はスキップされるため，この関数で参照する．
    /// ProvenTT クラスタ内の refutable disproof + confirmed disproof の
    /// 両方にマッチする．
    pub(super) fn has_refutable_or_confirmed_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        let pos_key = Self::safe_key(pos_key);
        let home = self.proven_cluster(pos_key, hand);
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if !e.is_proof() && hand_gte_forward_chain(&e.hand, hand) {
                return true;
            }
        }
        false
    }

/// WorkingTT に depth-limited / path-dep disproof を挿入する．
    fn store_working_disproof(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        remaining: u16,
        mut new_entry: DfPnEntry,
        #[cfg(feature = "verbose")] rem_idx: usize,
    ) {
        let path_dependent = new_entry.path_dependent();
        // depth-limited disproof の amount は低い値にして GC で
        // intermediate より先に除去されやすくする．
        // 0 にすると IDS-MID で不詰判定の情報が失われるため，
        // 最低限の保護として 1 を設定する．
        new_entry.amount = 1;

        let w_start = self.working_cluster_start(pos_key, &hand);
        let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];

        // 同じ pos_key の中間エントリと被支配反証を除去
        for fe in w_cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.pn != 0 && e.dn != 0 {
                // 中間エントリ → 除去
                fe.pos_key = 0;
                continue;
            }
            if e.dn == 0 {
                if e.path_dependent() && !path_dependent { continue; }
                if !(hand_gte_forward_chain(&hand, &e.hand) && remaining >= e.remaining()) {
                    continue;
                }
                fe.pos_key = 0;
            }
        }

        if let Some(slot) = w_cluster.iter_mut().find(|fe| fe.is_empty()) {
            slot.pos_key = pos_key;
            slot.entry = new_entry;
            #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; self.diag_remaining_dist[rem_idx] += 1; }
            #[cfg(feature = "tt_diag")] { self.diag_disproof_working += 1; }
            // ピーク充填数の追跡
            let fill = self.working[w_start..w_start + WORKING_CLUSTER_SIZE].iter()
                .filter(|fe| fe.pos_key != 0).count();
            if fill > self.working_peak_cluster_fill {
                self.working_peak_cluster_fill = fill;
            }
            return;
        }
        // クラスタ飽和 → overflow．サンプリング診断は replace 後に実行 (borrow 回避)．
        self.working_overflow_since_gc += 1;
        if Self::replace_weakest_for_disproof_in(w_cluster, pos_key, new_entry) {
            #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; self.diag_remaining_dist[rem_idx] += 1; }
            #[cfg(feature = "tt_diag")] { self.diag_disproof_working += 1; }
            return;
        }
        // NM 同士の置換
        let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
        for fe in w_cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            if fe.entry.dn == 0
                && fe.entry.remaining() != REMAINING_INFINITE
                && !hand_gte_forward_chain(&fe.entry.hand, &hand)
            {
                fe.entry = new_entry;
                #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; self.diag_remaining_dist[rem_idx] += 1; }
                #[cfg(feature = "tt_diag")] { self.diag_disproof_working += 1; }
                return;
            }
        }
        // N-8 (v0.26.0): remaining ≤ 2 かつ WorkingTT に入れなかった場合，
        // LeafDisproofTT に格納し中間エントリを圧迫しない．
        // always-replace policy でクラスタ最初の空きか，先頭スロットを上書き．
        if !new_entry.path_dependent() && remaining <= 2 {
            let l_start = self.leaf_cluster_start(pos_key, &hand);
            let l_cluster = &mut self.leaf_disproofs[l_start..l_start + LEAF_CLUSTER_SIZE];
            // 既存の同 pos_key 被支配エントリを置換，なければ空きか先頭に格納
            let leaf_entry = TTLeafEntry { pos_key, hand, remaining: remaining as u8 };
            let mut stored = false;
            for le in l_cluster.iter_mut() {
                if le.pos_key == pos_key
                    && hand_gte_forward_chain(&hand, &le.hand)
                    && remaining >= le.remaining as u16
                {
                    *le = leaf_entry;
                    stored = true;
                    break;
                }
            }
            if !stored {
                if let Some(slot) = l_cluster.iter_mut().find(|le| le.pos_key == 0) {
                    *slot = leaf_entry;
                    stored = true;
                }
            }
            if !stored {
                l_cluster[0] = leaf_entry;
            }
            self.diag_leaf_inserts += 1;
            return;
        }
        // フォールバック
        let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
        if Self::replace_weakest_in(w_cluster, pos_key, new_entry) {
            #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; self.diag_remaining_dist[rem_idx] += 1; }
        }
    }

    /// WorkingTT に intermediate エントリを挿入する．
    fn store_working_intermediate(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        best_move: u16,
        #[cfg(feature = "verbose")] rem_idx: usize,
    ) {
        let w_start = self.working_cluster_start(pos_key, &hand);
        let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];

        // 同一持ち駒の既存エントリを更新
        for fe in w_cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            let e = &mut fe.entry;
            if e.hand == hand {
                if e.dn == 0 {
                    if e.remaining() >= remaining || e.path_dependent() {
                        return;
                    }
                }
                e.pn = pn;
                e.dn = dn;
                e.set_remaining(remaining);
                e.source = source;
                e.clear_path_dependent();
                e.amount = e.amount.saturating_add(1);
                if best_move != 0 {
                    e.best_move = best_move;
                }
                #[cfg(feature = "verbose")] { self.diag_intermediate_update += 1; self.diag_remaining_dist[rem_idx] += 1; }
                return;
            }
        }

        // 新規エントリを追加
        let new_entry = DfPnEntry::new(source, pn, dn, hand, remaining, false, best_move, 0);
        if let Some(slot) = w_cluster.iter_mut().find(|fe| fe.is_empty()) {
            slot.pos_key = pos_key;
            slot.entry = new_entry;
            #[cfg(feature = "verbose")] { self.diag_intermediate_new += 1; self.diag_remaining_dist[rem_idx] += 1; }
            // ピーク充填数の追跡
            let fill = self.working[w_start..w_start + WORKING_CLUSTER_SIZE].iter()
                .filter(|fe| fe.pos_key != 0).count();
            if fill > self.working_peak_cluster_fill {
                self.working_peak_cluster_fill = fill;
            }
            #[cfg(feature = "profile")]
            {
                let count = self.working[w_start..w_start + WORKING_CLUSTER_SIZE].iter()
                    .filter(|fe| fe.pos_key == pos_key).count();
                if count > self.max_entries_per_position {
                    self.max_entries_per_position = count;
                }
            }
        } else {
            #[cfg(feature = "profile")]
            { self.overflow_count += 1; self.working_overflow_count += 1; }
            self.working_overflow_since_gc += 1;
            // 100回に1回サンプリング
            if self.working_overflow_since_gc % 100 == 1 {
                let cluster = &self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
                let mut keys = [0u64; WORKING_CLUSTER_SIZE];
                let mut n_keys = 0usize;
                let mut n_inter = 0u64;
                let mut n_disp = 0u64;
                for fe in cluster {
                    if fe.pos_key == 0 { continue; }
                    if !keys[..n_keys].contains(&fe.pos_key) {
                        if n_keys < WORKING_CLUSTER_SIZE { keys[n_keys] = fe.pos_key; n_keys += 1; }
                    }
                    if fe.entry.dn == 0 {
                        n_disp += 1;
                        let rem = fe.entry.remaining();
                        let bucket = if rem == 0 { 0 }
                            else if rem <= 4 { 1 }
                            else if rem < REMAINING_INFINITE { 2 }
                            else { 3 };
                        self.overflow_disproof_remaining[bucket] += 1;
                        if fe.entry.path_dependent() {
                            self.overflow_disproof_path_dep += 1;
                        }
                    } else {
                        n_inter += 1;
                    }
                }
                self.overflow_distinct_keys_sum += n_keys as u64;
                self.overflow_intermediate_sum += n_inter;
                self.overflow_disproof_sum += n_disp;
                self.overflow_sample_count += 1;
            }
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            if Self::replace_weakest_in(w_cluster, pos_key, new_entry) {
                #[cfg(feature = "verbose")] { self.diag_intermediate_new += 1; self.diag_remaining_dist[rem_idx] += 1; }
            }
        }
    }

    /// 指定エントリの amount を更新する．
    /// GC 前に指定局面の WorkingTT エントリの amount を最大値に引き上げる．
    ///
    /// 探索パス上のエントリを GC から保護するために使用する．
    pub(super) fn protect_working_entry(
        &mut self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) {
        let pos_key = Self::safe_key(pos_key);
        let start = self.working_cluster_start(pos_key, hand);
        let cluster = &mut self.working[start..start + WORKING_CLUSTER_SIZE];
        for fe in cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            if fe.entry.hand == *hand {
                fe.entry.amount = 255;
                return;
            }
        }
    }

    /// mid() からの帰還時に呼ばれ，探索投資量を記録する．
    /// WorkingTT のみスキャン(intermediate エントリ対象)．
    pub(super) fn update_amount(
        &mut self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        nodes_spent: u16,
    ) {
        let pos_key = Self::safe_key(pos_key);
        let start = self.working_cluster_start(pos_key, hand);
        let cluster = &mut self.working[start..start + WORKING_CLUSTER_SIZE];
        for fe in cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            if fe.entry.hand == *hand {
                fe.entry.amount = fe.entry.amount.saturating_add(nodes_spent.min(255) as u8);
                return;
            }
        }
    }

    /// クラスタ内の最弱エントリを amount ベースで置換する(テーブル非依存)．
    fn replace_weakest_in(cluster: &mut [TTFlatEntry], pos_key: u64, new_entry: DfPnEntry) -> bool {
        let mut worst_idx: Option<usize> = None;
        let mut worst_score: u32 = u32::MAX;
        let mut worst_is_foreign = false;

        for (i, fe) in cluster.iter().enumerate() {
            if fe.pos_key == 0 { continue; }
            let is_foreign = fe.pos_key != pos_key;
            let score = fe.entry.amount as u32;

            let better = match (worst_is_foreign, is_foreign) {
                (false, true) => true,
                (true, false) => false,
                _ => score < worst_score,
            };
            if better {
                worst_score = score;
                worst_idx = Some(i);
                worst_is_foreign = is_foreign;
            }
        }

        if let Some(idx) = worst_idx {
            cluster[idx].pos_key = pos_key;
            cluster[idx].entry = new_entry;
            return true;
        }
        false
    }

    /// ProvenTT 専用の置換: Plan D では ProvenEntry::amount() (distance/種別ベース)
    /// で eviction 優先度を決定する．
    ///
    /// amount の値:
    /// - proof with distance: 128..191 (長い詰み筋ほど高 priority)
    /// - proof without distance: 64
    /// - confirmed disproof: 32
    ///
    /// 低い amount のエントリを優先的に evict する．
    /// 新エントリの amount が既存の最小 amount 以上の場合のみ置換する
    /// (高 priority のエントリが低 priority に evict されないようにする)．
    fn replace_weakest_proven(
        cluster: &mut [TTFlatProvenEntry],
        pos_key: u64,
        new_entry: ProvenEntry,
    ) -> bool {
        let mut worst_idx: Option<usize> = None;
        let mut worst_amount: u8 = u8::MAX;
        let mut worst_is_foreign = false;

        for (i, fe) in cluster.iter().enumerate() {
            if fe.pos_key == 0 { continue; }
            let is_foreign = fe.pos_key != pos_key;
            let amount = fe.entry.amount();

            let better = match (worst_is_foreign, is_foreign) {
                (false, true) => true,   // foreign を優先 evict
                (true, false) => false,
                _ => amount < worst_amount,
            };
            if better {
                worst_amount = amount;
                worst_idx = Some(i);
                worst_is_foreign = is_foreign;
            }
        }

        if let Some(idx) = worst_idx {
            // 新エントリの amount が既存最弱以上の場合のみ置換
            if new_entry.amount() >= worst_amount {
                cluster[idx].pos_key = pos_key;
                cluster[idx].entry = new_entry;
                return true;
            }
        }
        false
    }

    /// WorkingTT の反証エントリ挿入用の置換．
    ///
    /// Dual TT では ProvenTT に proof/confirmed disproof が分離されているため，
    /// WorkingTT には depth-limited disproof と intermediate のみが存在する．
    /// foreign の depth-limited disproof を優先的に置換する．
    fn replace_weakest_for_disproof_in(
        cluster: &mut [TTFlatEntry],
        pos_key: u64,
        new_entry: DfPnEntry,
    ) -> bool {
        // 1st pass: foreign depth-limited disproof
        for fe in cluster.iter_mut() {
            if fe.pos_key == pos_key || fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 && fe.entry.remaining() != REMAINING_INFINITE {
                fe.pos_key = pos_key;
                fe.entry = new_entry;
                return true;
            }
        }
        // 2nd pass: foreign intermediate (lowest amount)
        let mut worst_idx: Option<usize> = None;
        let mut worst_amount: u8 = u8::MAX;
        for (i, fe) in cluster.iter().enumerate() {
            if fe.pos_key == pos_key || fe.pos_key == 0 { continue; }
            if fe.entry.pn != 0 && fe.entry.dn != 0 && fe.entry.amount < worst_amount {
                worst_amount = fe.entry.amount;
                worst_idx = Some(i);
            }
        }
        if let Some(idx) = worst_idx {
            cluster[idx].pos_key = pos_key;
            cluster[idx].entry = new_entry;
            return true;
        }
        false
    }

    /// 自クラスタ + 持ち駒-1の近傍クラスタ(±1限定)を走査する．
    pub(super) fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.is_proof()
                && hand_gte_forward_chain(hand, &fe.entry.hand)
            {
                return fe.entry.hand;
            }
        }
        let base_hh = Self::hand_hash(hand);
        for k in 0..HAND_KINDS {
            if hand[k] == 0 { continue; }
            let diff = Self::hand_hash_diff(k, hand[k], hand[k] - 1);
            let start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
            let cluster = &self.proven[start..start + PROVEN_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key == pos_key
                    && fe.entry.is_proof()
                    && hand_gte_forward_chain(hand, &fe.entry.hand)
                {
                    return fe.entry.hand;
                }
            }
        }
        *hand
    }

    /// 反証エントリが経路依存かどうかを返す．
    /// path-dep disproof は WorkingTT に格納される．
    /// 自クラスタ + 持ち駒1枚増の近傍クラスタを走査する．
    pub(super) fn has_path_dependent_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.working_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.path_dependent();
            }
        }
        // 持ち駒+1の近傍(±1限定)
        let base_hh = Self::hand_hash(hand);
        for k in 0..HAND_KINDS {
            let max_k = PieceType::MAX_HAND_COUNT[k];
            if hand[k] >= max_k { continue; }
            let diff = Self::hand_hash_diff(k, hand[k], hand[k] + 1);
            let start = self.working_cluster_start_from_hash(pos_key, base_hh ^ diff);
            let cluster = &self.working[start..start + WORKING_CLUSTER_SIZE];
            for fe in cluster {
                if fe.pos_key == pos_key
                    && fe.entry.dn == 0
                    && hand_gte_forward_chain(&fe.entry.hand, hand)
                {
                    return fe.entry.path_dependent();
                }
            }
        }
        false
    }

    /// 反証エントリの remaining を返す(±1近傍走査付き)．
    ///
    /// v0.24.53 以降: ProvenTT の confirmed disproof は常に depth 非依存
    /// (`REMAINING_INFINITE` 相当) のため，該当クラスタで hit した場合は
    /// `REMAINING_INFINITE` を返す．WorkingTT の depth-limited disproof は
    /// 従来通り `DfPnEntry::remaining()` を使う．
    pub(super) fn get_disproof_remaining(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && !fe.entry.is_proof()
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return REMAINING_INFINITE;
            }
        }
        for fe in self.working_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.remaining();
            }
        }
        // 持ち駒+1の近傍(±1限定)
        let base_hh = Self::hand_hash(hand);
        for k in 0..HAND_KINDS {
            let max_k = PieceType::MAX_HAND_COUNT[k];
            if hand[k] >= max_k { continue; }
            let diff = Self::hand_hash_diff(k, hand[k], hand[k] + 1);
            let p_start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
            for fe in &self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE] {
                if fe.pos_key == pos_key
                    && !fe.entry.is_proof()
                    && hand_gte_forward_chain(&fe.entry.hand, hand)
                {
                    return REMAINING_INFINITE;
                }
            }
            let w_start = self.working_cluster_start_from_hash(pos_key, base_hh ^ diff);
            for fe in &self.working[w_start..w_start + WORKING_CLUSTER_SIZE] {
                if fe.pos_key == pos_key
                    && fe.entry.dn == 0
                    && hand_gte_forward_chain(&fe.entry.hand, hand)
                {
                    return fe.entry.remaining();
                }
            }
        }
        0
    }

    /// lookup が実際に使用する反証エントリの情報を返す(±1近傍走査付き)．
    ///
    /// v0.24.53 以降: ProvenTT の confirmed disproof は depth 非依存として
    /// 格納されるため `remaining` 比較は不要 (常に REMAINING_INFINITE 相当)．
    /// WorkingTT の depth-limited disproof は従来通り `remaining` 比較する．
    pub(super) fn get_effective_disproof_info(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> Option<(u16, bool)> {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && !fe.entry.is_proof()
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return Some((REMAINING_INFINITE, false));
            }
        }
        for fe in self.working_cluster(pos_key, hand) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
                && (fe.entry.remaining() >= remaining || fe.entry.path_dependent())
            {
                return Some((fe.entry.remaining(), fe.entry.path_dependent()));
            }
        }
        // 持ち駒+1の近傍(±1限定)
        let base_hh = Self::hand_hash(hand);
        for k in 0..HAND_KINDS {
            let max_k = PieceType::MAX_HAND_COUNT[k];
            if hand[k] >= max_k { continue; }
            let diff = Self::hand_hash_diff(k, hand[k], hand[k] + 1);
            let p_start = self.proven_cluster_start_from_hash(pos_key, base_hh ^ diff);
            for fe in &self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE] {
                if fe.pos_key == pos_key
                    && !fe.entry.is_proof()
                    && hand_gte_forward_chain(&fe.entry.hand, hand)
                {
                    return Some((REMAINING_INFINITE, false));
                }
            }
            let w_start = self.working_cluster_start_from_hash(pos_key, base_hh ^ diff);
            for fe in &self.working[w_start..w_start + WORKING_CLUSTER_SIZE] {
                if fe.pos_key == pos_key
                    && fe.entry.dn == 0
                    && hand_gte_forward_chain(&fe.entry.hand, hand)
                    && (fe.entry.remaining() >= remaining || fe.entry.path_dependent())
                {
                    return Some((fe.entry.remaining(), fe.entry.path_dependent()));
                }
            }
        }
        None
    }

    /// 全エントリをクリアする．
    pub(super) fn clear(&mut self) {
        for fe in self.proven.iter_mut() { fe.pos_key = 0; }
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
        self.working_overflow_since_gc = 0;
    }

    /// 証明エントリ(pn=0)のみを保持する．
    /// Dual TT: WorkingTT を全クリア + ProvenTT の confirmed disproof を除去．
    pub(super) fn retain_proofs_only(&mut self) {
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
        for fe in self.proven.iter_mut() {
            if fe.pos_key != 0 && !fe.entry.is_proof() {
                fe.pos_key = 0;
            }
        }
    }

    /// WorkingTT の confirmed disproof を保持し，他を除去する．
    ///
    /// Frontier サイクル間で呼ばれ，WorkingTT の confirmed disproof
    /// (!path_dep, REMAINING_INFINITE) のみ保持する．
    /// 中間エントリ・depth-limited disproof・path-dep disproof は全て除去．
    pub(super) fn retain_proofs(&mut self) {
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            // confirmed disproof (non-path-dep, REMAINING_INFINITE) は保持
            if fe.entry.dn == 0 && !fe.entry.path_dependent()
                && fe.entry.remaining() == REMAINING_INFINITE
            { continue; }
            fe.pos_key = 0;
        }
    }

    /// ProvenTT の非 proof エントリのみを除去する（WorkingTT は触らない）．
    ///
    /// nested warmup (warmup\_mode=true) の mid\_fallback 入口で使用する．
    /// ProvenTT の confirmed disproof (浅い depth で生成されたものが深い
    /// depth を汚染するリスク) を除去し，WorkingTT intermediate は保持する．
    pub(super) fn clear_proven_non_proofs(&mut self) {
        for fe in self.proven.iter_mut() {
            if fe.pos_key != 0 && !fe.entry.is_proof() {
                fe.pos_key = 0;
            }
        }
    }

    /// WorkingTT の confirmed disproof (dn=0, !path_dep, remaining=INF) の数を返す．
    ///
    /// warmup 診断用: IDS depth 切り替え後に WorkingTT に残っている
    /// confirmed disproof 数を確認する．
    pub(super) fn count_working_confirmed_disproofs(&self) -> usize {
        self.working.iter().filter(|fe| {
            fe.pos_key != 0
                && fe.entry.dn == 0
                && !fe.entry.path_dependent()
                && fe.entry.remaining() == REMAINING_INFINITE
        }).count()
    }

    /// WorkingTT を全クリアする（IDS depth 切り替え時の強制クリア用）．
    pub(super) fn clear_working(&mut self) {
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
        // N-8: LeafDisproofTT も同時にクリア
        for le in self.leaf_disproofs.iter_mut() { le.pos_key = 0; }
        self.working_overflow_since_gc = 0;
    }

    /// warmup 前の selective クリア (v0.27.3): remaining <= threshold のエントリのみ削除し，
    /// remaining > threshold のエントリ (より深い intermediate) を保持する．
    ///
    /// 1D (v0.25.9) の全消去を置き換える．warmup_depth を threshold として渡すことで:
    /// - warmup に干渉しうる shallow intermediate (remaining <= warmup_depth) を除去 ✓
    /// - main search で再利用可能な deep intermediate (remaining > warmup_depth) を保持 ✓
    pub(super) fn clear_working_shallow(&mut self, threshold: u16) {
        for fe in self.working.iter_mut() {
            if fe.pos_key != 0 && fe.entry.remaining() <= threshold {
                fe.pos_key = 0;
            }
        }
        for le in self.leaf_disproofs.iter_mut() { le.pos_key = 0; }
        self.working_overflow_since_gc = 0;
    }

    /// IDS depth 切替時に WorkingTT から intermediate エントリを選択的に保持する (v0.24.45)．
    ///
    /// 施策 I: `test_tsume_39te_ply25_gap_diagnosis` (v0.24.44) で特定された
    /// 「IDS depth 切替時の intermediate 全消去」問題への対策．
    ///
    /// # 保持条件
    ///
    /// 以下を全て満たすエントリのみ保持する:
    /// 1. `pn > 0 && dn > 0` (intermediate; proven/disproven は既に ProvenTT にあるか除去される)
    /// 2. `!path_dependent` (GHI 不整合防止)
    /// 3. `remaining >= min_remaining` (前 IDS step の上位 N ply 分の作業のみ保持)
    /// 4. `remaining != REMAINING_INFINITE` (depth-limited のみ; 仮値は除外)
    ///
    /// # remaining の shift
    ///
    /// 保持したエントリの `remaining` に `delta_remaining` を加算する．
    /// 旧 IDS depth で計算された pn/dn は新 depth でも**下限値として有効**である
    /// (より多くの remaining = より多くの探索必要 → 旧 pn/dn は安全な初期下限)．
    /// 加算後の値が `REMAINING_INFINITE` を超える場合は保持しない．
    ///
    /// # 返り値
    ///
    /// 保持したエントリ数．0 なら通常の `clear_working()` と同じ効果．
    ///
    /// # 注意
    ///
    /// - path-dependent disproof は常に除去する (GHI 対策)
    /// - depth-limited disproof (dn=0, remaining < INFINITE) も除去する
    ///   (新 depth では無効になる可能性が高い)
    /// - 保持後は `working_overflow_since_gc` をリセットする
    pub(super) fn retain_working_intermediates(
        &mut self,
        min_remaining: u16,
        delta_remaining: u16,
    ) -> usize {
        let mut kept = 0usize;
        // [0]:0-1, [1]:2-3, [2]:4-7, [3]:8-15, [4]:16-31, [5]:32+
        #[cfg(feature = "verbose")]
        let mut rem_dist = [0usize; 6];
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            let entry = &mut fe.entry;
            let rem = entry.remaining();
            let is_intermediate = entry.pn > 0 && entry.dn > 0;
            let is_path_dep = entry.path_dependent();
            let is_depth_limited = rem < REMAINING_INFINITE;
            let new_rem = rem.saturating_add(delta_remaining);
            let keep = is_intermediate
                && !is_path_dep
                && is_depth_limited
                && rem >= min_remaining
                && new_rem < REMAINING_INFINITE;
            if keep {
                entry.set_remaining(new_rem);
                #[cfg(feature = "tt_diag")]
                {
                    // cap 適用前の値で分布を記録する
                    let pb = match entry.pn as usize {
                        1 => 0, 2..=7 => 1, 8..=63 => 2,
                        64..=511 => 3, 512..=4095 => 4, _ => 5,
                    };
                    self.diag_retained_pn_dist[pb] += 1;
                    let db = match entry.dn as usize {
                        1 => 0, 2..=7 => 1, 8..=63 => 2,
                        64..=511 => 3, 512..=4095 => 4, _ => 5,
                    };
                    self.diag_retained_dn_dist[db] += 1;
                }
                if self.retain_pn_dn_cap < u32::MAX {
                    entry.pn = entry.pn.min(self.retain_pn_dn_cap);
                    entry.dn = entry.dn.min(self.retain_pn_dn_cap);
                }
                kept += 1;
                #[cfg(feature = "verbose")]
                {
                    let b = match new_rem as usize {
                        0..=1 => 0, 2..=3 => 1, 4..=7 => 2,
                        8..=15 => 3, 16..=31 => 4, _ => 5,
                    };
                    rem_dist[b] += 1;
                }
            } else {
                fe.pos_key = 0;
            }
        }
        self.working_overflow_since_gc = 0;
        #[cfg(feature = "tt_diag")]
        { self.diag_last_retained_count = kept as u64; }
        #[cfg(feature = "verbose")]
        eprintln!(
            "[retain_intermediates] delta={} kept={} rem_dist(post-shift): \
             0-1:{} 2-3:{} 4-7:{} 8-15:{} 16-31:{} 32+:{}",
            delta_remaining, kept,
            rem_dist[0], rem_dist[1], rem_dist[2], rem_dist[3], rem_dist[4], rem_dist[5],
        );
        kept
    }

    /// WorkingTT の overflow カウンタを取得しリセットする．
    ///
    /// 前回の drain/GC/clear 以降に発生したクラスタ overflow の累積回数を返す．
    /// 呼び出し側はこの値をもとに GC の必要性を判断する．
    #[inline]
    pub(super) fn drain_working_overflow(&mut self) -> u64 {
        let count = self.working_overflow_since_gc;
        self.working_overflow_since_gc = 0;
        count
    }

    /// ProvenTT の ephemeral エントリ (confirmed disproof + non-ABSOLUTE proof)
    /// を選択的に除去する．
    ///
    /// IDS depth 切り替え時に呼び出す．2 種のエントリを一括管理する:
    ///
    /// 1. **confirmed disproof** (is_proof=0): 浅い IDS depth で REMAINING_INFINITE
    ///    として格納された confirmed disproof が深い depth の探索を汚染する
    ///    のを防ぐ (NoMate バグ対策)．`disproof_depth() < min_depth` のエントリ
    ///    のみ除去し，`min_depth` 以上のものは保持する．
    ///
    /// 2. **non-ABSOLUTE proof** (is_proof=1 かつ proof_tag != ABSOLUTE)
    ///    (施策 X, v0.24.53+): heuristic filter 下で生成された FILTER_DEPENDENT /
    ///    PROVISIONAL proof が，後続 IDS step を汚染するのを防ぐ．`tag_depth()
    ///    < min_depth` のエントリのみ除去．
    ///
    /// **ABSOLUTE proof は永続**: 完全 df-pn 探索による proof (tag=ABSOLUTE)
    /// は depth 非依存の真理として扱われ除去されない．従来の `new_proof` 経由
    /// のエントリはすべて ABSOLUTE tag で格納されるため backward compat．
    ///
    /// 施策 α (v0.24.47, revert) や A-6 (v0.24.51, revert) の失敗は，
    /// ProvenTT の REMAINING_INFINITE 不変条件下で heuristic proof が
    /// 汚染していたため．proof_tag 付きエントリと本 API の統合により
    /// soundness を保ちつつ heuristic 施策を適用可能になる．
    ///
    /// 呼出元 (pns.rs:~1704): IDS depth 遷移で `clear_proven_disproofs_below(next_ids_depth / 2)`
    /// として呼ばれる．proof/disproof 共通の閾値で動作する．
    pub(super) fn clear_proven_disproofs_below(&mut self, min_depth: u32) {
        for fe in self.proven.iter_mut() {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.is_proof() {
                // 施策 X: non-ABSOLUTE proof を tag_depth ベースで選択除去
                if e.proof_tag() != super::entry::PROOF_TAG_ABSOLUTE
                    && e.tag_depth() < min_depth
                {
                    fe.pos_key = 0;
                }
            } else {
                // 既存: confirmed disproof を disproof_depth ベースで選択除去
                if e.disproof_depth() < min_depth {
                    fe.pos_key = 0;
                }
            }
        }
    }

    /// WorkingTT の非空エントリ数を返す．
    pub(super) fn len(&self) -> usize {
        self.working.iter().filter(|fe| fe.pos_key != 0).count()
    }

    /// ProvenTT の非空エントリ数を返す．
    pub(super) fn proven_len(&self) -> usize {
        self.proven.iter().filter(|fe| fe.pos_key != 0).count()
    }

    /// ProvenTT の proof (pn=0) エントリ数を返す．
    pub(super) fn proven_count(&self) -> usize {
        self.proven.iter().filter(|fe| fe.pos_key != 0 && fe.entry.is_proof()).count()
    }

    /// 特定の position_key + hand の WorkingTT エントリを除去する (v0.24.66)．
    ///
    /// warmup mid_fallback 後に root 局面の depth-limited NM を除去して
    /// full-depth IDS での false NoCheckmate を防止する目的で使用．
    pub(super) fn clear_working_entry(&mut self, pos_key: u64, hand: &[u8; HAND_KINDS]) {
        let pos_key = Self::safe_key(pos_key);
        let start = self.working_cluster_start(pos_key, hand);
        for fe in &mut self.working[start..start + WORKING_CLUSTER_SIZE] {
            if fe.pos_key == pos_key
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                fe.pos_key = 0;
            }
        }
    }

    /// WorkingTT の非空エントリ数を返す(`len` のエイリアス)．
    pub(super) fn working_len(&self) -> usize {
        self.len()
    }

    /// TT の使用中エントリ数を返す(ProvenTT + WorkingTT 合計)．
    /// verbose/profile feature での診断用．
    #[allow(dead_code)]
    pub(super) fn total_entries(&self) -> usize {
        self.proven_len() + self.working_len()
    }

    /// ProvenTT の総スロット数を返す．
    pub(super) fn proven_capacity(&self) -> usize {
        self.proven.len()
    }

    /// TT の詳細診断情報を出力する(テスト用)．
    #[cfg(test)]
    pub(super) fn dump_overflow_diag(&self) {
        let proven_count = self.proven.iter().filter(|fe| fe.pos_key != 0).count();
        let working_count = self.working.iter().filter(|fe| fe.pos_key != 0).count();
        let proven_slots = self.proven.len();
        let working_slots = self.working.len();

        eprintln!("ProvenTT:  entries={} / {} slots ({:.1}% full)",
            proven_count, proven_slots,
            proven_count as f64 / proven_slots as f64 * 100.0);
        eprintln!("WorkingTT: entries={} / {} slots ({:.1}% full)",
            working_count, working_slots,
            working_count as f64 / working_slots as f64 * 100.0);

        #[cfg(feature = "profile")]
        {
            eprintln!("Proven overflow:  {}", self.proven_overflow_count);
            eprintln!("Working overflow: {}", self.working_overflow_count);
            eprintln!("No victim found:  {}", self.overflow_no_victim_count);
            eprintln!("Total overflow:   {}", self.overflow_count);
            eprintln!("Max entries/pos:  {}", self.max_entries_per_position);
        }
        eprintln!("Working peak cluster fill: {}", self.working_peak_cluster_fill);

        // 近傍走査診断
        let n_scans = NEIGHBOR_DIAG[3].load(Ordering::Relaxed);
        let proof_hits = NEIGHBOR_DIAG[0].load(Ordering::Relaxed);
        let disproof_proven_hits = NEIGHBOR_DIAG[1].load(Ordering::Relaxed);
        let disproof_working_hits = NEIGHBOR_DIAG[2].load(Ordering::Relaxed);
        eprintln!("Neighbor scan: calls={} proof_hits={} disproof_proven_hits={} disproof_working_hits={}",
            n_scans, proof_hits, disproof_proven_hits, disproof_working_hits);
        if n_scans > 0 {
            let total_hits = proof_hits + disproof_proven_hits + disproof_working_hits;
            eprintln!("  hit rate: {:.4}% ({}/{})",
                total_hits as f64 / n_scans as f64 * 100.0, total_hits, n_scans);
        }

        // Overflow サンプリング結果
        if self.overflow_sample_count > 0 {
            let n = self.overflow_sample_count as f64;
            eprintln!("Overflow sampling ({} samples):", self.overflow_sample_count);
            eprintln!("  avg distinct pos_keys/cluster: {:.1}", self.overflow_distinct_keys_sum as f64 / n);
            eprintln!("  avg intermediate entries/cluster: {:.1}", self.overflow_intermediate_sum as f64 / n);
            eprintln!("  avg disproof entries/cluster:  {:.1}", self.overflow_disproof_sum as f64 / n);
            let total_disp = self.overflow_disproof_sum.max(1) as f64;
            eprintln!("  disproof remaining distribution:");
            eprintln!("    rem=0:        {} ({:.1}%)", self.overflow_disproof_remaining[0],
                self.overflow_disproof_remaining[0] as f64 / total_disp * 100.0);
            eprintln!("    rem=1..4:     {} ({:.1}%)", self.overflow_disproof_remaining[1],
                self.overflow_disproof_remaining[1] as f64 / total_disp * 100.0);
            eprintln!("    rem=5..31:    {} ({:.1}%)", self.overflow_disproof_remaining[2],
                self.overflow_disproof_remaining[2] as f64 / total_disp * 100.0);
            eprintln!("    rem=INFINITE:  {} ({:.1}%)", self.overflow_disproof_remaining[3],
                self.overflow_disproof_remaining[3] as f64 / total_disp * 100.0);
            eprintln!("  path_dependent disproof: {} ({:.1}%)", self.overflow_disproof_path_dep,
                self.overflow_disproof_path_dep as f64 / total_disp * 100.0);
        }

        // ProvenTT エントリ種別
        let mut proof_count = 0u64;
        let mut confirmed_disproof_count = 0u64;
        for fe in &self.proven {
            if fe.pos_key == 0 { continue; }
            if fe.entry.is_proof() {
                proof_count += 1;
            } else {
                confirmed_disproof_count += 1;
            }
        }
        eprintln!("ProvenTT breakdown: proof={} confirmed_disproof={}", proof_count, confirmed_disproof_count);

        // WorkingTT エントリ種別
        let mut intermediate_count = 0u64;
        let mut dl_disproof_count = 0u64;
        let mut pd_disproof_count = 0u64;
        for fe in &self.working {
            if fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 {
                if fe.entry.path_dependent() {
                    pd_disproof_count += 1;
                } else {
                    dl_disproof_count += 1;
                }
            } else {
                intermediate_count += 1;
            }
        }
        eprintln!("WorkingTT breakdown: intermediate={} depth_limited_disproof={} path_dep_disproof={}",
            intermediate_count, dl_disproof_count, pd_disproof_count);

        // ProvenTT クラスタ充填分布
        let proven_clusters = proven_slots / PROVEN_CLUSTER_SIZE;
        let mut cluster_fill = [0u64; PROVEN_CLUSTER_SIZE + 1];
        for c in 0..proven_clusters {
            let start = c * PROVEN_CLUSTER_SIZE;
            let fill = self.proven[start..start + PROVEN_CLUSTER_SIZE].iter()
                .filter(|fe| fe.pos_key != 0).count();
            cluster_fill[fill] += 1;
        }
        eprintln!("ProvenTT cluster fill distribution:");
        for i in 0..=PROVEN_CLUSTER_SIZE {
            if cluster_fill[i] > 0 {
                eprintln!("  {} entries: {} clusters ({:.1}%)", i, cluster_fill[i],
                    cluster_fill[i] as f64 / proven_clusters as f64 * 100.0);
            }
        }

        // WorkingTT クラスタ充填分布
        let working_clusters = working_slots / WORKING_CLUSTER_SIZE;
        let mut wcluster_fill = [0u64; WORKING_CLUSTER_SIZE + 1];
        for c in 0..working_clusters {
            let start = c * WORKING_CLUSTER_SIZE;
            let fill = self.working[start..start + WORKING_CLUSTER_SIZE].iter()
                .filter(|fe| fe.pos_key != 0).count();
            wcluster_fill[fill] += 1;
        }
        eprintln!("WorkingTT cluster fill distribution:");
        for i in 0..=WORKING_CLUSTER_SIZE {
            if wcluster_fill[i] > 0 {
                eprintln!("  {} entries: {} clusters ({:.1}%)", i, wcluster_fill[i],
                    wcluster_fill[i] as f64 / working_clusters as f64 * 100.0);
            }
        }
    }

    /// TT GC: メモリ使用量を抑制する(WorkingTT のみ)．
    pub(super) fn gc(&mut self, target_size: usize) {
        if self.len() <= target_size {
            return;
        }
        // Phase 1: remaining が小さい中間エントリを除去
        let median_remaining = 8u16;
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 { continue; }
            if fe.entry.remaining() <= median_remaining {
                fe.pos_key = 0;
            }
        }
        if self.len() <= target_size {
            return;
        }
        // Phase 2: WorkingTT 全クリア
        self.retain_proofs();
    }

    // ---------------------------------------------------------------
    // 独立 GC: ProvenTT と WorkingTT をそれぞれ個別に GC する
    // ---------------------------------------------------------------

    /// WorkingTT の独立 GC．
    ///
    /// WorkingTT の充填率に基づいて intermediate エントリを除去する．
    /// Phase 1: amount が低い intermediate エントリを除去
    /// Phase 2: 全 intermediate エントリを除去(disproof は保護)
    /// Phase 3: WorkingTT 全クリア
    ///
    /// 返り値: 除去されたエントリ数．
    /// GC サンプリング数．
    const GC_SAMPLING_ENTRIES: usize = 10_000;

    /// GC 除去率(サンプリングした中のこの割合以下の amount を除去)．
    const GC_REMOVAL_RATIO: f64 = 0.2;

    /// WorkingTT の GC（overflow トリガ）．
    /// obsolete intermediate の除去のみ実行（disproof は保護）．
    /// disproof を除去すると再生成→再overflow→再GC のサイクルに陥るため．
    pub(super) fn gc_working_overflow(&mut self) -> usize {
        self.gc_working_sampling(false)
    }

    /// サンプリングベースの WorkingTT GC (KomoringHeights 方式)．
    ///
    /// 1. サンプリング: ストライドで GC_SAMPLING_ENTRIES 個の amount を収集
    /// 2. nth_element で除去閾値を決定(下位 GC_REMOVAL_RATIO を除去)
    /// 3. 閾値以下の全エントリを除去(intermediate も disproof も対象)
    /// 4. max_amount が大きい場合，生存エントリの amount を半減(CutAmount)
    fn gc_working_sampling(&mut self, remove_disproof: bool) -> usize {
        self.working_overflow_since_gc = 0;
        let total = self.working.len();
        if total == 0 { return 0; }

        // Phase 1: サンプリングで amount 分布を収集
        let stride = total / Self::GC_SAMPLING_ENTRIES;
        let stride = stride.max(1);
        let mut amounts: Vec<u8> = Vec::with_capacity(Self::GC_SAMPLING_ENTRIES);
        let mut idx = 0usize;
        while amounts.len() < Self::GC_SAMPLING_ENTRIES && idx < total {
            if self.working[idx].pos_key != 0 {
                amounts.push(self.working[idx].entry.amount);
            }
            idx += stride;
        }

        if amounts.is_empty() {
            return 0;
        }

        // Phase 2: nth_element で除去閾値を決定
        let pivot = ((amounts.len() as f64 * Self::GC_REMOVAL_RATIO) as usize)
            .max(1)
            .min(amounts.len() - 1);
        amounts.select_nth_unstable(pivot);
        let amount_threshold = amounts[pivot];
        let max_amount = amounts.iter().copied().max().unwrap_or(0);

        // Phase 3: 証明/反証済み局面の obsolete intermediate を除去
        //
        // ProvenTT に proof(pn=0) または confirmed disproof(dn=0) がある局面の
        // WorkingTT intermediate エントリは不要:
        // - OR ノードで子が証明済み → 親も証明 → 他の子の intermediate は不要
        // - AND ノードで子が反証済み → 親も反証 → 他の子の intermediate は不要
        // ProvenTT クラスタの参照は O(1) なので，全エントリに適用しても軽量．
        let initial = self.working_len();
        let proven_mask = self.proven_mask;
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 || fe.entry.dn == 0 { continue; } // non-intermediate skip
            if fe.entry.amount >= 255 { continue; } // パス保護エントリは除外
            // ProvenTT に proof/disproof があれば obsolete
            let pk = fe.pos_key;
            let hand = fe.entry.hand;
            let hh = Self::hand_hash(&hand);
            let p_start = ((pk ^ hh) as usize & proven_mask) * PROVEN_CLUSTER_SIZE;
            let mut is_resolved = false;
            for pfe in &self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE] {
                if pfe.pos_key != pk { continue; }
                if pfe.entry.is_proof() && hand_gte_forward_chain(&hand, &pfe.entry.hand) {
                    is_resolved = true;
                    break;
                }
                if !pfe.entry.is_proof() && hand_gte_forward_chain(&pfe.entry.hand, &hand) {
                    is_resolved = true;
                    break;
                }
            }
            if is_resolved {
                fe.pos_key = 0;
            }
        }

        if remove_disproof {
            // Phase 4: 閾値以下の disproof を除去(intermediate は保護)
            for fe in self.working.iter_mut() {
                if fe.pos_key == 0 { continue; }
                if fe.entry.pn != 0 && fe.entry.dn != 0 { continue; } // intermediate 保護
                if fe.entry.amount <= amount_threshold {
                    fe.pos_key = 0;
                }
            }

            // Phase 5: CutAmount — 生存エントリの amount を半減
            if max_amount > 32 {
                for fe in self.working.iter_mut() {
                    if fe.pos_key == 0 { continue; }
                    fe.entry.amount = (fe.entry.amount / 2).max(1);
                }
            }
        }
        let removed = initial - self.working_len();

        removed
    }

    /// ProvenTT の独立 GC．
    ///
    /// ProvenTT の充填率に基づいてエントリを除去する．
    /// confirmed disproof を優先的に除去し，proof は最後まで保護する．
    /// Phase 1: confirmed disproof のうち amount が低いものから除去
    /// Phase 2: 全 confirmed disproof を除去
    /// Phase 3: proof のうち amount が低いものから除去
    ///
    /// 返り値: 除去されたエントリ数．
    pub(super) fn gc_proven(&mut self) -> usize {
        let capacity = self.proven.len();
        let initial = self.proven_len();
        let target = capacity * 6 / 10;
        if initial <= target {
            return 0;
        }

        // Phase 1: confirmed disproof のうち amount が低いものから除去
        for threshold in [0u8, 16, 32, 64, 128] {
            for fe in self.proven.iter_mut() {
                if fe.pos_key == 0 { continue; }
                if !fe.entry.is_proof() && fe.entry.amount() <= threshold {
                    fe.pos_key = 0;
                }
            }
            if self.proven_len() <= target {
                return initial - self.proven_len();
            }
        }

        // Phase 2: 全 confirmed disproof を除去
        self.clear_proven_disproofs_below(u32::MAX);
        if self.proven_len() <= target {
            return initial - self.proven_len();
        }

        // Phase 3: proof のうち amount が低いものから除去
        for threshold in [0u8, 16, 32, 64, 128, 192] {
            for fe in self.proven.iter_mut() {
                if fe.pos_key == 0 { continue; }
                if fe.entry.is_proof() && fe.entry.amount() <= threshold {
                    fe.pos_key = 0;
                }
            }
            if self.proven_len() <= target {
                return initial - self.proven_len();
            }
        }

        initial - self.proven_len()
    }

    /// 指定局面のエントリ数を返す(診断用)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) fn entries_for_position(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let pos_key = Self::safe_key(pos_key);
        let p = self.proven_cluster(pos_key, hand).iter().filter(|fe| fe.pos_key == pos_key).count();
        let w = self.working_cluster(pos_key, hand).iter().filter(|fe| fe.pos_key == pos_key).count();
        p + w
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn dump_entries(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) {
        let pos_key = Self::safe_key(pos_key);
        verbose_eprintln!("[tt_dump] ProvenTT:");
        for (i, fe) in self.proven_cluster(pos_key, hand).iter().enumerate() {
            if fe.pos_key == pos_key {
                let e = &fe.entry;
                // ProvenEntry は常に path_dependent=false (設計上，path-dep
                // エントリは WorkingTT に格納される)．
                // v0.24.53: remaining フィールドは meta に変更されたため
                // proof_tag/tag_depth を出力する．
                verbose_eprintln!(
                    "  [P{}]: pn={} dn={} tag={} tag_depth={} path_dep=false hand={:?}",
                    i, e.pn(), e.dn(), e.proof_tag(), e.tag_depth(), &e.hand
                );
            }
        }
        verbose_eprintln!("[tt_dump] WorkingTT:");
        for (i, fe) in self.working_cluster(pos_key, hand).iter().enumerate() {
            if fe.pos_key == pos_key {
                let e = &fe.entry;
                verbose_eprintln!(
                    "  [W{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn, e.dn, e.remaining(), e.path_dependent(), &e.hand
                );
            }
        }
    }

    /// 証明済み(pn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_proven(&self) -> usize {
        self.proven.iter().filter(|fe| fe.pos_key != 0 && fe.entry.is_proof()).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す(ProvenTT + WorkingTT)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_disproven(&self) -> usize {
        let p = self.proven.iter().filter(|fe| fe.pos_key != 0 && !fe.entry.is_proof()).count();
        let w = self.working.iter().filter(|fe| fe.pos_key != 0 && fe.entry.dn == 0).count();
        p + w
    }

    /// 中間のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_intermediate(&self) -> usize {
        self.working.iter()
            .filter(|fe| fe.pos_key != 0 && fe.entry.pn > 0 && fe.entry.dn > 0)
            .count()
    }

    /// TT コンテンツの詳細分析(診断用)．
    #[cfg(feature = "verbose")]
    pub(super) fn dump_content_analysis(&self) {
        let mut proof_count: u64 = 0;
        let mut disproof_count: u64 = 0;
        let mut intermediate_count: u64 = 0;
        let mut disproof_rem: [u64; 33] = [0; 33];
        let mut inter_pn_buckets: [u64; 8] = [0; 8];
        let mut inter_rem: [u64; 33] = [0; 33];
        let mut inter_dn_buckets: [u64; 5] = [0; 5];

        // ProvenTT: proof / confirmed disproof のみ
        // v0.24.53: ProvenEntry の disproof は remaining 非保持 (常に INFINITE)
        // のため disproof_rem は常に [32] にカウントする．
        for fe in self.proven.iter() {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.is_proof() {
                proof_count += 1;
            } else {
                disproof_count += 1;
                disproof_rem[32] += 1;
            }
        }
        // WorkingTT: intermediate + depth-limited disproof
        for fe in self.working.iter() {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.pn == 0 {
                proof_count += 1; // defensive (should not happen in Plan D)
            } else if e.dn == 0 {
                disproof_count += 1;
                let ri = if e.remaining() == REMAINING_INFINITE { 32 } else { (e.remaining() as usize).min(31) };
                disproof_rem[ri] += 1;
            } else {
                intermediate_count += 1;
                let ri = if e.remaining() == REMAINING_INFINITE { 32 } else { (e.remaining() as usize).min(31) };
                inter_rem[ri] += 1;
                // バケット閾値は PN_UNIT=16 スケールに合わせる
                let pb = match e.pn {
                    0..=16 => 0, 17..=48 => 1, 49..=128 => 2, 129..=512 => 3,
                    513..=2048 => 4, 2049..=16384 => 5, 16385..=131072 => 6, _ => 7,
                };
                inter_pn_buckets[pb] += 1;
                let db = match e.dn {
                    1 => 0, 2..=5 => 1, 6..=20 => 2, 21..=100 => 3, _ => 4,
                };
                inter_dn_buckets[db] += 1;
            }
        }

        verbose_eprintln!("\n=== TT Content Analysis ===");
        verbose_eprintln!("entries: proof={} disproof={} intermediate={}",
            proof_count, disproof_count, intermediate_count);
        let dr: Vec<String> = disproof_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        verbose_eprintln!("disproof remaining: [{}]", dr.join(", "));
        let ir: Vec<String> = inter_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        verbose_eprintln!("intermediate remaining: [{}]", ir.join(", "));
        let pn_labels = ["pn≤S", "pn≤3S", "pn≤8S", "pn≤32S", "pn≤128S", "pn≤1KS", "pn≤8KS", "pn>8KS"];
        let pb: Vec<String> = inter_pn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", pn_labels[i], c))
            .collect();
        verbose_eprintln!("intermediate pn dist: [{}]", pb.join(", "));
        let dn_labels = ["dn=1", "dn=2-5", "dn=6-20", "dn=21-100", "dn=100+"];
        let db: Vec<String> = inter_dn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", dn_labels[i], c))
            .collect();
        verbose_eprintln!("intermediate dn dist: [{}]", db.join(", "));
    }

    /// プロファイル統計をリセットする．
    #[cfg(feature = "profile")]
    pub(super) fn reset_profile(&mut self) {
        self.overflow_count = 0;
        self.proven_overflow_count = 0;
        self.working_overflow_count = 0;
        self.overflow_no_victim_count = 0;
        self.max_entries_per_position = 0;
    }

    /// 指定 pos_key のエントリイテレータ(verbose 診断用)．
    /// ProvenTT + WorkingTT を chain して返す．
    #[cfg(feature = "verbose")]
    pub(super) fn entries_iter(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> impl Iterator<Item = &DfPnEntry> {
        let pos_key = Self::safe_key(pos_key);
        self.working_cluster(pos_key, hand).iter()
            .filter(move |fe| fe.pos_key == pos_key)
            .map(|fe| &fe.entry)
    }
}
