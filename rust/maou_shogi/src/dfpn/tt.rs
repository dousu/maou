//! Dual フラットハッシュテーブル型転置表(Transposition Table)．
//!
//! v0.24.0: Dual TT 方式 — ProvenTT(永続エントリ)と WorkingTT(GC 対象)に分離．
//! ProvenTT には proof(pn=0)と confirmed disproof(dn=0, !path_dep, remaining=INFINITE)を格納し，
//! WorkingTT には intermediate と depth-limited/path-dependent disproof を格納する．
//! これによりクラスタ飽和問題(§6.6.1)を構造的に解決する．

use crate::types::HAND_KINDS;

use super::entry::DfPnEntry;
use super::{hand_gte_forward_chain, INF, PN_UNIT, REMAINING_INFINITE};

/// ProvenTT の 1 クラスタあたりのエントリ数．
///
/// proof(1-2) + confirmed disproof(1-2) で典型的に 2-4 エントリ．
/// hand バリアント(合駒チェーン等)で 4 を超えるケースがあるため
/// 6 エントリに設定し，PV 復元時の proof チェーン断絶を防止する．
const PROVEN_CLUSTER_SIZE: usize = 6;

/// WorkingTT の 1 クラスタあたりのエントリ数．
///
/// intermediate + depth-limited disproof 専用．
/// proof/confirmed disproof が ProvenTT に分離されたため，
/// 6 エントリの全てを working entries に使用できる．
const WORKING_CLUSTER_SIZE: usize = 6;

/// TT クラスタ数のデフォルト値(2^21 = 2M クラスタ)．
///
/// Dual TT メモリ配分:
/// - ProvenTT:  2M × 6 × 32B = 384 MB
/// - WorkingTT: 2M × 6 × 32B = 384 MB
/// - 合計: 768 MB
const DEFAULT_NUM_CLUSTERS: usize = 1 << 21; // 2M

/// フラットテーブルの 1 エントリ．
///
/// pos_key を含むことでクラスタ内の異なる局面を区別する．
/// pos_key = 0 は空スロットを示す(実際のハッシュ値 0 との衝突は無視可能)．
///
/// v0.24.0: DfPnEntry 圧縮(24 bytes)により TTFlatEntry は 32 bytes．
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

/// エントリが ProvenTT に格納されるべきかを判定する．
#[inline(always)]
fn is_proven_entry(pn: u32, dn: u32, remaining: u16, path_dependent: bool) -> bool {
    // proof (pn=0) または confirmed disproof (dn=0, !path_dependent, remaining=INFINITE)
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
    proven: Vec<TTFlatEntry>,
    /// WorkingTT: GC 対象エントリ(intermediate + depth-limited disproof)．
    working: Vec<TTFlatEntry>,
    /// ProvenTT の `num_clusters - 1`(高速 modulo 用ビットマスク)．
    proven_mask: usize,
    /// WorkingTT の `num_clusters - 1`．
    working_mask: usize,
    /// TT エントリ溢れ(置換)の発生回数．
    #[cfg(feature = "profile")]
    pub(super) overflow_count: u64,
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
        let num_clusters = num_clusters.next_power_of_two();
        let proven_total = num_clusters * PROVEN_CLUSTER_SIZE;
        let working_total = num_clusters * WORKING_CLUSTER_SIZE;
        TranspositionTable {
            proven: vec![TTFlatEntry::EMPTY; proven_total],
            working: vec![TTFlatEntry::EMPTY; working_total],
            proven_mask: num_clusters - 1,
            working_mask: num_clusters - 1,
            #[cfg(feature = "profile")]
            overflow_count: 0,
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

    /// ProvenTT のクラスタ開始インデックス．
    #[inline(always)]
    fn proven_cluster_start(&self, pos_key: u64) -> usize {
        ((pos_key as usize) & self.proven_mask) * PROVEN_CLUSTER_SIZE
    }

    /// WorkingTT のクラスタ開始インデックス．
    #[inline(always)]
    fn working_cluster_start(&self, pos_key: u64) -> usize {
        ((pos_key as usize) & self.working_mask) * WORKING_CLUSTER_SIZE
    }

    /// ProvenTT のクラスタスライスを返す(不変参照)．
    #[inline(always)]
    fn proven_cluster(&self, pos_key: u64) -> &[TTFlatEntry] {
        let start = self.proven_cluster_start(pos_key);
        &self.proven[start..start + PROVEN_CLUSTER_SIZE]
    }

    /// WorkingTT のクラスタスライスを返す(不変参照)．
    #[inline(always)]
    fn working_cluster(&self, pos_key: u64) -> &[TTFlatEntry] {
        let start = self.working_cluster_start(pos_key);
        &self.working[start..start + WORKING_CLUSTER_SIZE]
    }

    /// 転置表を参照する(証明駒/反証駒の優越関係を利用)．
    ///
    /// Dual TT 版: ProvenTT → WorkingTT の順でスキャン．
    /// Pass 1: ProvenTT で proof(pn=0) — early return
    /// Pass 2: ProvenTT で confirmed disproof(dn=0)
    /// Pass 3: WorkingTT で depth-limited disproof + exact_match
    #[inline(always)]
    pub(super) fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u32) {
        let pos_key = Self::safe_key(pos_key);

        // Pass 1: ProvenTT — proof(pn=0) early return
        let proven = self.proven_cluster(pos_key);
        for fe in proven {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                return (0, e.dn, e.source);
            }
        }
        // Pass 2: ProvenTT — confirmed disproof(dn=0)
        for fe in proven {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && e.remaining() >= remaining
            {
                return (e.pn, 0, e.source);
            }
        }
        // Pass 3: WorkingTT — depth-limited disproof + exact_match
        let working = self.working_cluster(pos_key);
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

        exact_match.unwrap_or((PN_UNIT, PN_UNIT, 0))
    }

    /// 指定局面に proof エントリ(pn=0)が存在するかチェックする．
    ///
    /// ProvenTT のみスキャン(proof は ProvenTT に格納)．
    #[inline(always)]
    pub(super) fn has_proof(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> bool {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.pn == 0
                && hand_gte_forward_chain(hand, &fe.entry.hand)
            {
                return true;
            }
        }
        false
    }

    /// TT Best Move を参照する．
    ///
    /// WorkingTT の intermediate エントリから取得する．
    #[inline(always)]
    pub(super) fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        // WorkingTT (intermediate entries have best_move)
        for fe in self.working_cluster(pos_key) {
            if fe.pos_key == pos_key && fe.entry.hand == *hand && fe.entry.best_move != 0 {
                return fe.entry.best_move;
            }
        }
        // ProvenTT fallback (proof/disproof may also have best_move)
        for fe in self.proven_cluster(pos_key) {
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
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, 0);
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
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, best_move);
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
        source: u32,
        path_dependent: bool,
        best_move: u16,
    ) {
        let pos_key = Self::safe_key(pos_key);
        #[cfg(feature = "verbose")]
        let rem_idx = if remaining == REMAINING_INFINITE { 32 } else { (remaining as usize).min(31) };

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        // ProvenTT をチェック
        for fe in self.proven_cluster(pos_key) {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.pn == 0 && hand_gte_forward_chain(&hand, &e.hand) {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
            if e.dn == 0
                && !path_dependent
                && hand_gte_forward_chain(&e.hand, &hand)
                && e.remaining() >= remaining
            {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
        }

        let new_entry = DfPnEntry::new(source, pn, dn, hand, remaining, path_dependent, best_move, 0);

        if is_proven_entry(pn, dn, remaining, path_dependent) {
            // === ProvenTT への挿入 ===
            self.store_proven(pos_key, hand, new_entry, pn == 0,
                #[cfg(feature = "verbose")] rem_idx);
        } else if dn == 0 {
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
    fn store_proven(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        mut new_entry: DfPnEntry,
        is_proof: bool,
        #[cfg(feature = "verbose")] rem_idx: usize,
    ) {
        let p_start = self.proven_cluster_start(pos_key);
        let p_cluster = &mut self.proven[p_start..p_start + PROVEN_CLUSTER_SIZE];

        if is_proof {
            const PROOF_BONUS: u8 = 100;
            new_entry.amount = PROOF_BONUS;
            // パレートフロンティア維持: ProvenTT の被支配エントリを除去
            for fe in p_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.dn == 0 { continue; } // 反証は保護
                if hand_gte_forward_chain(&fe.entry.hand, &hand) {
                    fe.pos_key = 0;
                }
            }
            // WorkingTT の被支配 intermediate も除去
            let w_start = self.working_cluster_start(pos_key);
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            for fe in w_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.dn == 0 { continue; }
                if hand_gte_forward_chain(&fe.entry.hand, &hand) {
                    fe.pos_key = 0;
                }
            }
        } else {
            // confirmed disproof
            const DISPROOF_BONUS: u8 = 50;
            new_entry.amount = DISPROOF_BONUS;
            // WorkingTT の被支配 intermediate と弱い disproof を除去
            let w_start = self.working_cluster_start(pos_key);
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            for fe in w_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.pn == 0 { continue; }
                fe.pos_key = 0;
            }
        }

        // ProvenTT に挿入
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
        // 満杯 → replace_weakest
        if Self::replace_weakest_in(p_cluster, pos_key, new_entry) {
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
        const DISPROOF_BONUS: u8 = 50;
        let dp_amount = DISPROOF_BONUS / 2; // depth-limited は半額
        new_entry.amount = dp_amount;

        let w_start = self.working_cluster_start(pos_key);
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
            return;
        }
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
        let w_start = self.working_cluster_start(pos_key);
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
            { self.overflow_count += 1; }
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            if Self::replace_weakest_in(w_cluster, pos_key, new_entry) {
                #[cfg(feature = "verbose")] { self.diag_intermediate_new += 1; self.diag_remaining_dist[rem_idx] += 1; }
            }
        }
    }

    /// 指定エントリの amount を更新する．
    /// mid() からの帰還時に呼ばれ，探索投資量を記録する．
    /// WorkingTT のみスキャン(intermediate エントリ対象)．
    pub(super) fn update_amount(
        &mut self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        nodes_spent: u16,
    ) {
        let pos_key = Self::safe_key(pos_key);
        let start = self.working_cluster_start(pos_key);
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

    /// 証明済みエントリの証明駒を返す(ProvenTT のみ)．
    #[inline(always)]
    pub(super) fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.proven_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.pn == 0
                && hand_gte_forward_chain(hand, &fe.entry.hand)
            {
                return fe.entry.hand;
            }
        }
        *hand
    }

    /// 反証エントリが経路依存かどうかを返す．
    /// path-dep disproof は WorkingTT に格納される．
    pub(super) fn has_path_dependent_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        let pos_key = Self::safe_key(pos_key);
        // WorkingTT (path-dep disproof はここに格納)
        for fe in self.working_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.path_dependent();
            }
        }
        false
    }

    /// 反証エントリの remaining を返す．
    /// ProvenTT + WorkingTT 両方を検索する．
    pub(super) fn get_disproof_remaining(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        // ProvenTT (confirmed disproof)
        for fe in self.proven_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.remaining();
            }
        }
        // WorkingTT (depth-limited disproof)
        for fe in self.working_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.remaining();
            }
        }
        0
    }

    /// lookup が実際に使用する反証エントリの情報を返す．
    /// ProvenTT + WorkingTT 両方を検索する．
    pub(super) fn get_effective_disproof_info(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> Option<(u16, bool)> {
        let pos_key = Self::safe_key(pos_key);
        // ProvenTT (confirmed disproof: !path_dep, remaining=INFINITE)
        for fe in self.proven_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
                && fe.entry.remaining() >= remaining
            {
                return Some((fe.entry.remaining(), false));
            }
        }
        // WorkingTT (depth-limited / path-dep disproof)
        for fe in self.working_cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
                && (fe.entry.remaining() >= remaining || fe.entry.path_dependent())
            {
                return Some((fe.entry.remaining(), fe.entry.path_dependent()));
            }
        }
        None
    }

    /// 全エントリをクリアする．
    pub(super) fn clear(&mut self) {
        for fe in self.proven.iter_mut() { fe.pos_key = 0; }
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
    }

    /// 証明エントリ(pn=0)のみを保持する．
    /// Dual TT: WorkingTT を全クリア(ProvenTT の proof はそのまま)．
    /// ProvenTT の confirmed disproof も除去する．
    pub(super) fn retain_proofs_only(&mut self) {
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
        for fe in self.proven.iter_mut() {
            if fe.pos_key != 0 && fe.entry.pn != 0 {
                fe.pos_key = 0;
            }
        }
    }

    /// 確定済みエントリ(証明・経路非依存反証)を保持し，他を除去する．
    /// Dual TT: WorkingTT を全クリア(ProvenTT はそのまま)．
    pub(super) fn retain_proofs(&mut self) {
        for fe in self.working.iter_mut() { fe.pos_key = 0; }
    }

    /// 経路依存の反証エントリを除去する(WorkingTT のみ)．
    pub(super) fn remove_path_dependent_disproofs(&mut self) {
        for fe in self.working.iter_mut() {
            if fe.pos_key != 0 && fe.entry.dn == 0 && fe.entry.path_dependent() {
                fe.pos_key = 0;
            }
        }
    }

    /// 浅い反復のスラッシング防止エントリと浅い反証を除去する(WorkingTT のみ)．
    pub(super) fn remove_stale_for_ids(&mut self) {
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn >= INF - 1 && fe.entry.dn > 0 {
                fe.pos_key = 0;
                continue;
            }
            if fe.entry.dn == 0 && fe.entry.remaining() == 0 {
                fe.pos_key = 0;
            }
        }
    }

    /// TT の非空エントリ数を返す(ProvenTT + WorkingTT)．
    ///
    /// GC 閾値(`tt_gc_threshold`)はこのエントリ数ベースで比較される．
    /// Dual TT では WorkingTT のエントリ数のみを返す(GC 対象)．
    pub(super) fn len(&self) -> usize {
        self.working.iter().filter(|fe| fe.pos_key != 0).count()
    }

    /// TT の使用中エントリ数を返す(ProvenTT + WorkingTT 合計)．
    pub(super) fn total_entries(&self) -> usize {
        let proven = self.proven.iter().filter(|fe| fe.pos_key != 0).count();
        let working = self.working.iter().filter(|fe| fe.pos_key != 0).count();
        proven + working
    }

    /// TT の総スロット数を返す(ProvenTT + WorkingTT)．
    pub(super) fn capacity(&self) -> usize {
        self.proven.len() + self.working.len()
    }

    /// 浅いエントリを除去する GC(WorkingTT のみ)．
    pub(super) fn gc_shallow_entries(&mut self, remaining_threshold: u16) -> usize {
        let mut removed = 0usize;
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.remaining() <= remaining_threshold {
                fe.pos_key = 0;
                removed += 1;
            }
        }
        removed
    }

    /// amount ベースの GC(WorkingTT のみ)．
    ///
    /// Phase 1: amount=0 の中間エントリを除去
    /// Phase 2: WorkingTT 全クリア(retain_proofs 相当)
    pub(super) fn gc_by_amount(&mut self, target_size: usize) -> usize {
        let initial = self.len();
        if initial <= target_size {
            return 0;
        }
        // Phase 1: amount=0 の中間エントリを除去
        for fe in self.working.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 { continue; } // disproof は保護
            if fe.entry.amount == 0 {
                fe.pos_key = 0;
            }
        }
        if self.len() <= target_size {
            return initial - self.len();
        }
        // Phase 2: WorkingTT 全クリア
        self.retain_proofs();
        initial - self.len()
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

    /// 指定局面のエントリ数を返す(診断用)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) fn entries_for_position(&self, pos_key: u64) -> usize {
        let pos_key = Self::safe_key(pos_key);
        let p = self.proven_cluster(pos_key).iter().filter(|fe| fe.pos_key == pos_key).count();
        let w = self.working_cluster(pos_key).iter().filter(|fe| fe.pos_key == pos_key).count();
        p + w
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn dump_entries(&self, pos_key: u64) {
        let pos_key = Self::safe_key(pos_key);
        verbose_eprintln!("[tt_dump] ProvenTT:");
        for (i, fe) in self.proven_cluster(pos_key).iter().enumerate() {
            if fe.pos_key == pos_key {
                let e = &fe.entry;
                verbose_eprintln!(
                    "  [P{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn, e.dn, e.remaining(), e.path_dependent(), &e.hand
                );
            }
        }
        verbose_eprintln!("[tt_dump] WorkingTT:");
        for (i, fe) in self.working_cluster(pos_key).iter().enumerate() {
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
        self.proven.iter().filter(|fe| fe.pos_key != 0 && fe.entry.pn == 0).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_disproven(&self) -> usize {
        let p = self.proven.iter().filter(|fe| fe.pos_key != 0 && fe.entry.dn == 0).count();
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

        for fe in self.proven.iter().chain(self.working.iter()) {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.pn == 0 {
                proof_count += 1;
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
        self.overflow_no_victim_count = 0;
        self.max_entries_per_position = 0;
    }

    /// 指定 pos_key のエントリイテレータ(verbose 診断用)．
    /// ProvenTT + WorkingTT を chain して返す．
    #[cfg(feature = "verbose")]
    pub(super) fn entries_iter(&self, pos_key: u64) -> impl Iterator<Item = &DfPnEntry> {
        let pos_key = Self::safe_key(pos_key);
        self.proven_cluster(pos_key).iter()
            .chain(self.working_cluster(pos_key).iter())
            .filter(move |fe| fe.pos_key == pos_key)
            .map(|fe| &fe.entry)
    }
}
