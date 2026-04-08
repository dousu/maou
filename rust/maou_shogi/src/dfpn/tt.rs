//! Dual フラットハッシュテーブル型転置表(Transposition Table)．
//!
//! v0.24.0: Dual TT 方式 — ProvenTT(永続エントリ)と WorkingTT(GC 対象)に分離．
//! ProvenTT には proof(pn=0)と confirmed disproof(dn=0, !path_dep, remaining=INFINITE)を格納し，
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
/// Dual TT メモリ配分:
/// - ProvenTT:  proven_num_clusters × 8 × 32B
/// - WorkingTT: working_num_clusters × 6 × 32B
const DEFAULT_NUM_CLUSTERS: usize = 1 << 21; // 2M

/// ProvenTT のクラスタ数の倍率．
/// 1 = WorkingTT と同数(2M)．
const PROVEN_CLUSTER_MULTIPLIER: usize = 1;

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
    /// ProvenTT の `num_clusters - 1`(高速 modulo 用ビットマスク)．
    proven_mask: usize,
    /// WorkingTT の `num_clusters - 1`．
    working_mask: usize,
    /// store_proven での amount 計算に使用する現在の ply．
    /// mid() が TT store 前にセットする．ply が小さい(ルートに近い)ほど
    /// amount が高くなり，eviction 耐性が上がる．
    pub(super) hint_ply: u32,
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
        TranspositionTable {
            proven: vec![TTFlatProvenEntry::EMPTY; proven_total],
            working: vec![TTFlatEntry::EMPTY; working_total],
            proven_mask: proven_clusters - 1,
            working_mask: working_clusters - 1,
            hint_ply: 0,
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
        }
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

    /// WorkingTT のクラスタ開始インデックス．
    /// ProvenTT と同様に pos_key ^ hand_hash で持ち駒バリアントを分散する．
    #[inline(always)]
    fn working_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::hand_hash(hand);
        ((mixed as usize) & self.working_mask) * WORKING_CLUSTER_SIZE
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
        for fe in working {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && (e.remaining() >= remaining || e.path_dependent())
            {
                return (e.pn, 0, e.source);
            }
            if exact_match.is_none()
                && e.hand == *hand
                && e.pn != 0
                && e.dn != 0
            {
                exact_match = Some((e.pn, e.dn, e.source));
            }
        }
        if exact_match.is_some() {
            return exact_match.unwrap();
        }

        // 歩(k=0)の+1のみ disproof 近傍走査．
        // disproof 近傍ヒットの60%+が歩+1(合駒チェーンの歩合い排除)．
        // 追加コストは1クラスタ(6エントリ)のみで極めて軽量．
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
        let pos_key = Self::safe_key(pos_key);
        let home = self.proven_cluster(pos_key, hand);
        // Pass 1: proof(pn=0) — 自クラスタ
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
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
                        NEIGHBOR_DIAG[0].fetch_add(1, Ordering::Relaxed);
                        return (0, e.dn(), e.source());
                    }
                }
            }
        }
        // Pass 2: confirmed disproof(dn=0) — 自クラスタ
        for fe in home {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if !e.is_proof()
                && hand_gte_forward_chain(&e.hand, hand)
                && e.remaining() >= remaining
            {
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
                    && e.remaining() >= remaining
                {
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

    /// 持ち駒の全部分集合に対応するクラスタを走査して proof を検索する．
    ///
    /// `look_up_proven` の1枚差走査で見つからない場合のフォールバック．
    /// Zobrist XOR 差分で部分集合のハッシュを逐次計算する．
    /// proven entry の mate_distance を取得する．
    ///
    /// pn=0 の proof エントリが存在し，かつ distance-aware な store で
    /// 設定されたものに限り mate_distance を返す．legacy proven entry
    /// (priority 値のみ) や 未設定の場合は None．
    /// PV 抽出時に AND ノードで longest resistance 判定に使用する．
    pub(super) fn look_up_mate_distance(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> Option<u16> {
        let pos_key = Self::safe_key(pos_key);
        // 自クラスタを走査して proof entry を探す
        for fe in self.proven_cluster(pos_key, hand) {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.is_proof() && hand_gte_forward_chain(hand, &e.hand) {
                return e.proven_distance();
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
                && e.remaining() >= remaining
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
            ProvenEntry::new_disproof(hand)
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
        }
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
            // ピーク充填数の追跡
            let fill = self.working[w_start..w_start + WORKING_CLUSTER_SIZE].iter()
                .filter(|fe| fe.pos_key != 0).count();
            if fill > self.working_peak_cluster_fill {
                self.working_peak_cluster_fill = fill;
            }
            return;
        }
        // クラスタ飽和 → overflow
        self.working_overflow_since_gc += 1;
        // サンプリングは replace 後に実行(borrow 回避)
        // replace_weakest_for_disproof
        if Self::replace_weakest_for_disproof_in(w_cluster, pos_key, new_entry) {
            #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; self.diag_remaining_dist[rem_idx] += 1; }
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
                return;
            }
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
                return fe.entry.remaining();
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
                    return fe.entry.remaining();
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
                && fe.entry.remaining() >= remaining
            {
                return Some((fe.entry.remaining(), false));
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
                    && fe.entry.remaining() >= remaining
                {
                    return Some((fe.entry.remaining(), false));
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

    /// WorkingTT を全クリアする（IDS depth 切り替え時の強制クリア用）．
    pub(super) fn clear_working(&mut self) {
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
        self.working_overflow_since_gc = 0;
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

    /// ProvenTT の confirmed disproof を除去する．
    ///
    /// IDS depth 切り替え時に呼び出す．浅い IDS depth で REMAINING_INFINITE
    /// として格納された confirmed disproof が，深い depth の探索を汚染する
    /// のを防ぐ(NoMate バグ対策)．Proof (pn=0) は影響を受けない．
    pub(super) fn clear_proven_disproofs(&mut self) {
        for fe in self.proven.iter_mut() {
            if fe.pos_key != 0 && !fe.entry.is_proof() {
                fe.pos_key = 0;
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
        self.clear_proven_disproofs();
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
                verbose_eprintln!(
                    "  [P{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn(), e.dn(), e.remaining(), e.path_dependent(), &e.hand
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
        for fe in self.proven.iter() {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.is_proof() {
                proof_count += 1;
            } else {
                disproof_count += 1;
                let ri = if e.remaining() == REMAINING_INFINITE { 32 } else { (e.remaining() as usize).min(31) };
                disproof_rem[ri] += 1;
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
