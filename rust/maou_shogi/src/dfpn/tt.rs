//! フラットハッシュテーブル型転置表(Transposition Table)．
//!
//! `FxHashMap<u64, Vec<DfPnEntry>>` を固定サイズのフラットハッシュテーブルに置換．
//! Vec ヒープアロケーション削減とキャッシュ効率向上を目的とする．

use crate::types::HAND_KINDS;

use super::entry::DfPnEntry;
use super::{hand_gte_forward_chain, INF, PN_UNIT, REMAINING_INFINITE};

/// 1 クラスタあたりのエントリ数．
///
/// 詰将棋 TT の実測データ(14M entries / 11M positions ≈ 1.27 entries/position)
/// から，大半の局面は 1〜2 エントリで済む．
/// 6 エントリあれば証明・反証・中間のパレートフロンティアを十分に保持できる．
/// TT クラスタサイズ．
///
/// 同一盤面に対する持ち駒バリアント(proof/disproof/intermediate)の
/// 最大同時保持数を決定する．将棋の持ち駒は 7 種であり，
/// 典型的には同一盤面に 2-6 種の hand バリアントが必要．
/// サイズ 4 では hand バリアント不足で TT 飽和が早期に発生(8.3M/16M)．
/// サイズ 6 で hand バリアント + foreign 衝突に対応する．
const CLUSTER_SIZE: usize = 6;

/// デフォルトのクラスタ数(2^21 = 2M クラスタ)．
///
/// 2M クラスタ × 6 エントリ × 40 bytes ≈ 480 MB．
/// 10M ノード探索では ~11M 局面が生成されるため，
/// 11M / 1M ≈ 11 局面/クラスタの競合が発生する．
/// フラットテーブルではクラスタ内の異なる pos_key のエントリが
/// 置換対象となるが，証明/反証エントリの保護ポリシーにより
/// 重要なエントリは維持される．
/// TT クラスタ数のデフォルト値．
///
/// クラスタ数 N と格納局面数 n の関係(Poisson 近似):
/// - lambda = n / N (クラスタあたりの期待エントリ数)
/// - P(overflow) = P(X >= CLUSTER_SIZE) for X ~ Poisson(lambda)
///
/// | N | 10M entries overflow | 15M | memory |
/// |---|---------------------|-----|--------|
/// | 2M (1<<21) | 22% | N/A | 320 MB |
/// | 4M (1<<22) | 3.3% | 11% | 640 MB |
/// | 8M (1<<23) | 0.3% | 3.3% | 1280 MB |
///
/// | N(clusters) | size | 10M overflow | 12.5M | memory |
/// |-------------|------|-------------|-------|--------|
/// | 1M (1<<20) | 6 | >容量 | >容量 | 240 MB |
/// | 2M (1<<21) | 6 | 3.5% | 22% | 480 MB |
/// | 4M (1<<22) | 6 | 0.2% | 3.5% | 960 MB |
///
/// 29手詰め規模(~12.5M entries)では 2M clusters で TT 飽和が早い(6.3M)．
/// 4M clusters で TT が 12.5M まで成長し 109M ノードで解決．
const DEFAULT_NUM_CLUSTERS: usize = 1 << 21; // 2M

/// フラットテーブルの 1 エントリ．
///
/// pos_key を含むことでクラスタ内の異なる局面を区別する．
/// pos_key = 0 は空スロットを示す(実際のハッシュ値 0 との衝突は無視可能)．
#[derive(Clone, Copy)]
#[repr(C)]
struct TTFlatEntry {
    pos_key: u64,
    entry: DfPnEntry,
}

impl TTFlatEntry {
    const EMPTY: Self = TTFlatEntry {
        pos_key: 0,
        entry: DfPnEntry {
            source: 0,
            pn: 0,
            dn: 0,
            hand: [0; HAND_KINDS],
            path_dependent: false,
            remaining: 0,
            best_move: 0,
            amount: 0,
        },
    };

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.pos_key == 0
    }
}

/// フラットハッシュテーブル型転置表(証明駒/反証駒対応)．
///
/// 固定サイズのフラットテーブルで，各クラスタが `CLUSTER_SIZE` エントリを保持する．
/// キーは盤面のみのハッシュ(持ち駒除外)を使用し，
/// 同一盤面・異なる持ち駒のエントリを同一クラスタに格納する．
///
/// 参照時に持ち駒の優越関係を利用して TT ヒット率を向上させる:
/// - 証明済み(pn=0): 現在の持ち駒 >= 登録時の持ち駒 → 再利用
/// - 反証済み(dn=0): 登録時の持ち駒 >= 現在の持ち駒 → 再利用
pub(super) struct TranspositionTable {
    /// フラットエントリ配列．`table[idx * CLUSTER_SIZE .. (idx+1) * CLUSTER_SIZE]`．
    table: Vec<TTFlatEntry>,
    /// `num_clusters - 1`(高速 modulo 用ビットマスク)．
    mask: usize,
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
        // 2 のべき乗に切り上げ
        let num_clusters = num_clusters.next_power_of_two();
        let total = num_clusters * CLUSTER_SIZE;
        TranspositionTable {
            table: vec![TTFlatEntry::EMPTY; total],
            mask: num_clusters - 1,
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

    /// pos_key からクラスタの開始インデックスを計算する．
    #[inline(always)]
    fn cluster_start(&self, pos_key: u64) -> usize {
        ((pos_key as usize) & self.mask) * CLUSTER_SIZE
    }

    /// `pos_key == 0` をセンチネル(空スロット)と衝突させない変換．
    ///
    /// Zobrist ハッシュが 0 になる確率は 2^{-64} で極めて低いが，
    /// 原理的に発生しうる．ビット 0 を立てることで 0 を回避する．
    /// クラスタインデックスとクラスタ内識別の両方に `safe_key` を使用する．
    ///
    /// **副作用:** 偶数キー `k` は `k+1` にマップされるため，
    /// 異なる局面のハッシュが `(2n, 2n+1)` のペアになると同一 `safe_key` に衝突する．
    /// Zobrist ハッシュのランダム性から衝突確率は 2^{-63} であり実用上は無視可能．
    #[inline(always)]
    fn safe_key(pos_key: u64) -> u64 {
        pos_key | 1
    }

    /// 指定 pos_key のクラスタスライスを返す(不変参照)．
    #[inline(always)]
    fn cluster(&self, pos_key: u64) -> &[TTFlatEntry] {
        let start = self.cluster_start(pos_key);
        &self.table[start..start + CLUSTER_SIZE]
    }

    /// 転置表を参照する(証明駒/反証駒の優越関係を利用)．
    ///
    /// 2-pass: proof (early return) → disproof + exact_match (統合 1-pass)．
    /// proof は発見時に即座に return するため独立パスが有利．
    /// disproof と exact_match は 1 パスに統合しクラスタスキャンを削減．
    #[inline(always)]
    pub(super) fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u64) {
        let pos_key = Self::safe_key(pos_key);
        let cluster = self.cluster(pos_key);
        let mut exact_match: Option<(u32, u32, u64)> = None;

        // Pass 1: 証明(pn=0) — early return
        for fe in cluster {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                return (0, e.dn, e.source);
            }
        }
        // Pass 2: 反証(dn=0) + exact_match 統合
        for fe in cluster {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && (e.remaining >= remaining || e.path_dependent)
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

    /// TT Best Move を参照する．
    #[inline(always)]
    pub(super) fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        let cluster = self.cluster(pos_key);
        for fe in cluster {
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
        source: u64,
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
        source: u64,
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
        let pos_key = Self::safe_key(pos_key);
        #[cfg(feature = "verbose")]
        let rem_idx = if remaining == REMAINING_INFINITE { 32 } else { (remaining as usize).min(31) };
        let start = self.cluster_start(pos_key);
        let cluster = &mut self.table[start..start + CLUSTER_SIZE];

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        for fe in cluster.iter() {
            if fe.pos_key != pos_key { continue; }
            let e = &fe.entry;
            if e.pn == 0 && hand_gte_forward_chain(&hand, &e.hand) {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
            if e.dn == 0
                && !e.path_dependent
                && !path_dependent
                && hand_gte_forward_chain(&e.hand, &hand)
                && e.remaining >= remaining
            {
                #[cfg(feature = "verbose")] { self.diag_dominated_skip += 1; }
                return;
            }
        }

        if pn == 0 {
            // === 証明済みエントリの挿入 ===
            // パレートフロンティア維持: 被支配エントリを空にする
            for fe in cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.dn == 0 { continue; } // 反証は保護
                if hand_gte_forward_chain(&e.hand, &hand) {
                    fe.pos_key = 0; // 支配される → 除去
                }
            }
            // 空スロットに挿入 (proof ボーナス: amount=PROOF_BONUS)
            const PROOF_BONUS: u16 = 100;
            if let Some(slot) = cluster.iter_mut().find(|fe| fe.is_empty()) {
                slot.pos_key = pos_key;
                slot.entry = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, amount: PROOF_BONUS, source };
                #[cfg(feature = "verbose")] { self.diag_proof_inserts += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
                return;
            }
            // 空スロットなし → 異なる pos_key の最弱エントリを置換
            if self.replace_weakest(start, pos_key, DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, amount: PROOF_BONUS, source }) {
                #[cfg(feature = "verbose")] { self.diag_proof_inserts += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
            }
            return;
        }

        if dn == 0 {
            // === 反証済みエントリの挿入 ===
            // 同じ pos_key の中間エントリと被支配反証を除去
            for fe in cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.pn == 0 { continue; } // 証明は保護
                if e.dn == 0 {
                    if e.path_dependent && !path_dependent { continue; }
                    if !(hand_gte_forward_chain(&hand, &e.hand) && remaining >= e.remaining) {
                        continue;
                    }
                }
                // 中間エントリまたは被支配反証 → 除去
                fe.pos_key = 0;
            }
            // disproof ボーナス: REMAINING_INFINITE は高価値
            const DISPROOF_BONUS: u16 = 50;
            let dp_amount = if remaining == REMAINING_INFINITE { DISPROOF_BONUS } else { DISPROOF_BONUS / 2 };
            if let Some(slot) = cluster.iter_mut().find(|fe| fe.is_empty()) {
                slot.pos_key = pos_key;
                slot.entry = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, amount: dp_amount, source };
                #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
                return;
            }
            // 空スロットなし — replace_weakest_for_disproof で
            // foreign protected エントリも含めて置換を試みる
            if self.replace_weakest_for_disproof(start, pos_key,
                DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, amount: dp_amount, source })
            {
                #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
                return;
            }
            // 空スロットなし: depth-limited NM 同士で新エントリに支配される
            // 既存エントリを1つ置換する(クラスタ飽和防止)．
            // 同じ pos_key の depth-limited 反証で hand が新エントリを支配しない
            // ものを置換候補とし，クラスタが同種の NM で埋まる問題を防止する．
            let cluster = &mut self.table[start..start + CLUSTER_SIZE];
            for fe in cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.dn == 0
                    && e.remaining != REMAINING_INFINITE
                    && !hand_gte_forward_chain(&e.hand, &hand)
                {
                    // 既存 NM は新エントリの hand を支配しない → 置換
                    fe.entry = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, amount: dp_amount, source };
                    #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; }
                    #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
                    return;
                }
            }
            // それでも挿入できない場合は replace_weakest にフォールバック
            if self.replace_weakest(start, pos_key, DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, amount: dp_amount, source }) {
                #[cfg(feature = "verbose")] { self.diag_disproof_inserts += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
            }
            return;
        }

        // === 中間エントリ(pn > 0, dn > 0)の挿入 ===
        let cluster = &mut self.table[start..start + CLUSTER_SIZE];

        // 同一持ち駒の既存エントリを更新
        for fe in cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            let e = &mut fe.entry;
            if e.hand == hand {
                if e.dn == 0 {
                    if e.remaining >= remaining || e.path_dependent {
                        return;
                    }
                }
                e.pn = pn;
                e.dn = dn;
                e.remaining = remaining;
                e.source = source;
                e.path_dependent = false;
                // amount 累積: 更新のたびに 1 加算(探索投資量の近似)
                e.amount = e.amount.saturating_add(1);
                if best_move != 0 {
                    e.best_move = best_move;
                }
                #[cfg(feature = "verbose")] { self.diag_intermediate_update += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
                return;
            }
        }

        // 新規エントリを追加
        if let Some(slot) = cluster.iter_mut().find(|fe| fe.is_empty()) {
            slot.pos_key = pos_key;
            slot.entry = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, amount: 0, source };
            #[cfg(feature = "verbose")] { self.diag_intermediate_new += 1; }
            #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
            #[cfg(feature = "profile")]
            {
                let count = self.table[start..start + CLUSTER_SIZE].iter()
                    .filter(|fe| fe.pos_key == pos_key).count();
                if count > self.max_entries_per_position {
                    self.max_entries_per_position = count;
                }
            }
        } else {
            #[cfg(feature = "profile")]
            { self.overflow_count += 1; }
            if self.replace_weakest(start, pos_key, DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, amount: 0, source }) {
                #[cfg(feature = "verbose")] { self.diag_intermediate_new += 1; }
                #[cfg(feature = "verbose")] { self.diag_remaining_dist[rem_idx] += 1; }
            }
        }
    }

    /// 指定エントリの amount を更新する．
    /// mid() からの帰還時に呼ばれ，探索投資量を記録する．
    pub(super) fn update_amount(
        &mut self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        nodes_spent: u16,
    ) {
        let pos_key = Self::safe_key(pos_key);
        let start = self.cluster_start(pos_key);
        let cluster = &mut self.table[start..start + CLUSTER_SIZE];
        for fe in cluster.iter_mut() {
            if fe.pos_key != pos_key { continue; }
            if fe.entry.hand == *hand {
                fe.entry.amount = fe.entry.amount.saturating_add(nodes_spent);
                return;
            }
        }
    }

    /// クラスタ内の最弱エントリを amount ベースで置換する．
    ///
    /// 置換優先度: foreign > same_pk，amount 小 > amount 大．
    /// min_depth は GC (gc_by_amount) でのみ使用し，置換では amount のみ参照する．
    /// amount と min_depth の複合スコア（乗算・加算）は proof/disproof の
    /// amount ボーナスを弱めてしまい，重要なエントリの淘汰を招くため不採用．
    fn replace_weakest(&mut self, start: usize, pos_key: u64, new_entry: DfPnEntry) -> bool {
        let cluster = &mut self.table[start..start + CLUSTER_SIZE];
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
        #[cfg(feature = "profile")]
        { self.overflow_no_victim_count += 1; }
        false
    }

    /// 反証エントリ挿入用の置換: foreign protected エントリも対象にする．
    ///
    /// 通常の replace_weakest は proof(pn=0) と confirmed disproof(dn=0,
    /// remaining=REMAINING_INFINITE) を保護するが，クラスタが全て foreign
    /// protected で埋まっている場合に新規エントリが挿入不能になる．
    /// 反証エントリの挿入では foreign の depth-limited disproof(remaining
    /// != REMAINING_INFINITE) を優先的に置換し，それもなければ foreign の
    /// confirmed disproof を犠牲にする．同じ pos_key のエントリは保護する．
    fn replace_weakest_for_disproof(
        &mut self,
        start: usize,
        pos_key: u64,
        new_entry: DfPnEntry,
    ) -> bool {
        let cluster = &mut self.table[start..start + CLUSTER_SIZE];
        // 1st pass: foreign depth-limited disproof を探す
        for fe in cluster.iter_mut() {
            if fe.pos_key == pos_key || fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 && fe.entry.remaining != REMAINING_INFINITE {
                fe.pos_key = pos_key;
                fe.entry = new_entry;
                return true;
            }
        }
        // 2nd pass: foreign confirmed disproof を探す
        for fe in cluster.iter_mut() {
            if fe.pos_key == pos_key || fe.pos_key == 0 { continue; }
            if fe.entry.dn == 0 {
                fe.pos_key = pos_key;
                fe.entry = new_entry;
                return true;
            }
        }
        // 3rd pass: foreign proof を探す(最終手段)
        // foreign proof は別 pos_key の証明であり，evict しても現局面の
        // 探索の正確性には影響しない．df-pn の TT はキャッシュであり，
        // 喪失した proof は再探索時に再計算される．
        for fe in cluster.iter_mut() {
            if fe.pos_key == pos_key || fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 {
                fe.pos_key = pos_key;
                fe.entry = new_entry;
                return true;
            }
        }
        false
    }

    /// 証明済みエントリの証明駒を返す．
    #[inline(always)]
    pub(super) fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.cluster(pos_key) {
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
    pub(super) fn has_path_dependent_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.path_dependent;
            }
        }
        false
    }

    /// 反証エントリの remaining を返す．
    pub(super) fn get_disproof_remaining(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
            {
                return fe.entry.remaining;
            }
        }
        0
    }

    /// lookup が実際に使用する反証エントリの情報を返す．
    pub(super) fn get_effective_disproof_info(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> Option<(u16, bool)> {
        let pos_key = Self::safe_key(pos_key);
        for fe in self.cluster(pos_key) {
            if fe.pos_key == pos_key
                && fe.entry.dn == 0
                && hand_gte_forward_chain(&fe.entry.hand, hand)
                && (fe.entry.remaining >= remaining || fe.entry.path_dependent)
            {
                return Some((fe.entry.remaining, fe.entry.path_dependent));
            }
        }
        None
    }

    /// 全エントリをクリアする．
    pub(super) fn clear(&mut self) {
        for fe in self.table.iter_mut() {
            fe.pos_key = 0;
        }
    }

    /// 証明エントリ(pn=0)のみを保持する．
    pub(super) fn retain_proofs_only(&mut self) {
        for fe in self.table.iter_mut() {
            if fe.pos_key != 0 && fe.entry.pn != 0 {
                fe.pos_key = 0;
            }
        }
    }

    /// 確定済みエントリ(証明・経路非依存反証)を保持し，他を除去する．
    pub(super) fn retain_proofs(&mut self) {
        for fe in self.table.iter_mut() {
            if fe.pos_key == 0 { continue; }
            let keep = fe.entry.pn == 0
                || (fe.entry.dn == 0 && !fe.entry.path_dependent);
            if !keep {
                fe.pos_key = 0;
            }
        }
    }

    /// 経路依存の反証エントリを除去する．
    pub(super) fn remove_path_dependent_disproofs(&mut self) {
        for fe in self.table.iter_mut() {
            if fe.pos_key != 0 && fe.entry.dn == 0 && fe.entry.path_dependent {
                fe.pos_key = 0;
            }
        }
    }

    /// 浅い反復のスラッシング防止エントリと浅い反証を除去する．
    pub(super) fn remove_stale_for_ids(&mut self) {
        for fe in self.table.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 { continue; }
            if fe.entry.pn >= INF - 1 && fe.entry.dn > 0 {
                fe.pos_key = 0;
                continue;
            }
            if fe.entry.dn == 0 && fe.entry.remaining == 0 {
                fe.pos_key = 0;
            }
        }
    }

    /// TT の非空エントリ数を返す．
    ///
    /// フラットテーブルでは同一クラスタ内に異なる `pos_key` のエントリが
    /// 共存するため，正確なポジション数(異なる pos\_key の数)の取得には
    /// O(N) のフルスキャン + ハッシュセットが必要でコストが高い．
    /// 本関数は非空エントリ数を返す．大半の局面は 1 エントリのため
    /// ポジション数の近似値として使用できるが，厳密には異なる．
    ///
    /// GC 閾値(`tt_gc_threshold`)はこのエントリ数ベースで比較される．
    pub(super) fn len(&self) -> usize {
        self.table.iter().filter(|fe| fe.pos_key != 0).count()
    }

    /// TT の使用中エントリ数を返す(`len()` のエイリアス)．
    ///
    /// `capacity()` が全スロット数を返すのに対し，本関数は
    /// 非空(pos\_key != 0)のエントリ数を返す．
    pub(super) fn total_entries(&self) -> usize {
        self.len()
    }

    /// TT の総スロット数を返す．
    pub(super) fn capacity(&self) -> usize {
        self.table.len()
    }

    /// 浅いエントリを除去する GC．
    pub(super) fn gc_shallow_entries(&mut self, remaining_threshold: u16) -> usize {
        let mut removed = 0usize;
        for fe in self.table.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 { continue; }
            if fe.entry.dn == 0 && fe.entry.remaining == REMAINING_INFINITE { continue; }
            if fe.entry.remaining <= remaining_threshold {
                fe.pos_key = 0;
                removed += 1;
            }
        }
        removed
    }

    /// amount ベースの GC: amount が小さいエントリを除去する．
    ///
    /// Phase 1: amount=0 の中間エントリを除去(一度も再訪されていない)
    /// Phase 2: 全中間エントリを除去(retain_proofs)
    /// proof/disproof (dn=0 or pn=0) はボーナス付きなので生存しやすい．
    pub(super) fn gc_by_amount(&mut self, target_size: usize) -> usize {
        let initial = self.len();
        if initial <= target_size {
            return 0;
        }
        // Phase 1: amount=0 の中間エントリを除去
        for fe in self.table.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 || fe.entry.dn == 0 { continue; }
            if fe.entry.amount == 0 {
                fe.pos_key = 0;
            }
        }
        if self.len() <= target_size {
            return initial - self.len();
        }
        // Phase 2: 全中間エントリを除去
        self.retain_proofs();
        initial - self.len()
    }

    /// TT GC: メモリ使用量を抑制する．
    pub(super) fn gc(&mut self, target_size: usize) {
        if self.total_entries() <= target_size {
            return;
        }
        // Phase 1: remaining が小さい中間エントリを除去
        let median_remaining = 8u16;
        for fe in self.table.iter_mut() {
            if fe.pos_key == 0 { continue; }
            if fe.entry.pn == 0 || fe.entry.dn == 0 { continue; }
            if fe.entry.remaining <= median_remaining {
                fe.pos_key = 0;
            }
        }
        if self.total_entries() <= target_size {
            return;
        }
        // Phase 2: 全中間エントリを除去
        self.retain_proofs();
    }

    /// 指定局面のエントリ数を返す(診断用)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) fn entries_for_position(&self, pos_key: u64) -> usize {
        let pos_key = Self::safe_key(pos_key);
        self.cluster(pos_key).iter()
            .filter(|fe| fe.pos_key == pos_key).count()
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn dump_entries(&self, pos_key: u64) {
        let pos_key = Self::safe_key(pos_key);
        for (i, fe) in self.cluster(pos_key).iter().enumerate() {
            if fe.pos_key == pos_key {
                let e = &fe.entry;
                verbose_eprintln!(
                    "[tt_dump] entry[{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn, e.dn, e.remaining, e.path_dependent, &e.hand
                );
            }
        }
    }

    /// 証明済み(pn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_proven(&self) -> usize {
        self.table.iter().filter(|fe| fe.pos_key != 0 && fe.entry.pn == 0).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_disproven(&self) -> usize {
        self.table.iter().filter(|fe| fe.pos_key != 0 && fe.entry.dn == 0).count()
    }

    /// 中間のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_intermediate(&self) -> usize {
        self.table.iter()
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

        for fe in &self.table {
            if fe.pos_key == 0 { continue; }
            let e = &fe.entry;
            if e.pn == 0 {
                proof_count += 1;
            } else if e.dn == 0 {
                disproof_count += 1;
                let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                disproof_rem[ri] += 1;
            } else {
                intermediate_count += 1;
                let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                inter_rem[ri] += 1;
                let pb = match e.pn {
                    1 => 0, 2..=5 => 1, 6..=20 => 2, 21..=100 => 3,
                    101..=1000 => 4, 1001..=10000 => 5, 10001..=100000 => 6, _ => 7,
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
        let pn_labels = ["pn=1", "pn=2-5", "pn=6-20", "pn=21-100", "pn=101-1K", "pn=1K-10K", "pn=10K-100K", "pn=100K+"];
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
    #[cfg(feature = "verbose")]
    pub(super) fn entries_iter(&self, pos_key: u64) -> impl Iterator<Item = &DfPnEntry> {
        let pos_key = Self::safe_key(pos_key);
        self.cluster(pos_key).iter()
            .filter(move |fe| fe.pos_key == pos_key)
            .map(|fe| &fe.entry)
    }
}
