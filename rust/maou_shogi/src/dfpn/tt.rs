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

#[cfg(test)]
use rustc_hash::FxHashMap;

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

/// ProvenTT HashMap の GC トリガー閾値 (refutable エントリ数)．
/// この値を超えると gc_proven() が呼ばれる．
const PROVEN_MAP_GC_CAPACITY: usize = 500_000;

/// ProvenTT proof エントリの GC トリガー閾値 (NPS 保護)．
/// proof がこの値を超えると gc_proofs() が amount 昇順で evict を行う．
const PROOF_MAP_GC_CAPACITY: usize = 2_000_000;

/// gc_proofs() 後の proof エントリ目標数 (PROOF_MAP_GC_CAPACITY の 60%)．
const PROOF_MAP_GC_TARGET: usize = 1_200_000;

/// ProvenTT confirmed disproof エントリの GC トリガー閾値．
/// confirmed がこの値を超えると gc_confirmed() が disproof_depth 昇順で evict を行う．
/// depth が低い = 浅い IDS で確認 = 再導出コストが安い → 優先的に削除．
const CONFIRMED_MAP_GC_CAPACITY: usize = 2_000_000;

/// gc_confirmed() 後の confirmed エントリ目標数 (CONFIRMED_MAP_GC_CAPACITY の 60%)．
const CONFIRMED_MAP_GC_TARGET: usize = 1_200_000;

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


/// LeafDisproofTT のクラスタサイズ (4 エントリ × 16B = 64B / クラスタ = 1 cache line)．
const LEAF_CLUSTER_SIZE: usize = 4;

/// LeafDisproofTT のクラスタ数 (2^19 = 512K クラスタ)．
/// 512K × 4 entries × 16B = 32MB．WorkingTT overflow の remaining≤2 専用バッファ．
const LEAF_NUM_CLUSTERS: usize = 1 << 19;

/// IDS 引継ぎ時に pn=INF (WPN 飽和 / depth-limited) エントリの pn を
/// 値落としするキャップ値．
///
/// pn=INF のまま引き継ぐと AND ノードで pn が爆発するため (n children × INF)，
/// heuristic_or_pn の典型値域 (O(512)) に近い小さい値に置き換える．
/// dn は有限なら保持し，INF の場合のみ同値でキャップする．
const RETAIN_INF_PN_CAP: u32 = 32 * PN_UNIT;

/// pn=INF エントリを IDS 引継ぎに含める最大 delta_remaining 値．
///
/// delta が大きい IDS ステップ (例: 4→17, delta=13) では，
/// depth=D の pn=INF エントリは depth=D+delta でも依然 INF である可能性が高く，
/// 未検証のヒントを大量注入して TT を汚染する．
/// delta が小さい場合 (例: 16→17, delta=1) は「直前の depth 限界に近い」
/// エントリのみが対象となり，汚染リスクが低い．
const RETAIN_INF_MAX_DELTA: u16 = 4;

/// FrontierTT: remaining ≤ この値の intermediate エントリを専用プールに格納する (案4, v0.55.7)．
///
/// WorkingTT のクラスタ overflow による eviction から保護するため，
/// 探索フロンティア (ply が深く remaining が小さいノード) の intermediate を別プールに格納する．
/// ply=39 (depth=41, remaining=2) の 84K 回重複訪問を引き起こす eviction thrashing を解消する．
///
/// 0 = 無効 (専用プール未使用)
const FRONTIER_REMAINING_THRESHOLD: u16 = 24;

/// FrontierTT の 1 クラスタあたりのエントリ数．
const FRONTIER_CLUSTER_SIZE: usize = 8;

/// FrontierTT のクラスタ数 (WorkingTT と同数 = 2M クラスタ)．
/// 2M × 8 × 32B = 512MB．
/// WorkingTT と同数にすることで pos_key のクラスタ分散が同一となり，
/// remaining≤2 エントリが remaining>2 エントリのクラスタ衝突を受けない．
const FRONTIER_NUM_CLUSTERS: usize = DEFAULT_NUM_CLUSTERS;

/// remaining ≤ この値の intermediate 新規エントリに付与する初期 amount ブースト値 (案4, v0.55.7)．
///
/// WorkingTT eviction thrashing 対策:
/// biased-default TT miss により amount=0 で格納された remaining≤2 エントリ (depth=41 の ply=39)
/// が remaining>2 エントリのクラスタ書き込みで即座に eviction される問題を緩和する．
/// 初期 amount=FRONTIER_INITIAL_AMOUNT により，amount=0 の remaining>2 エントリより
/// eviction 耐性が高くなる．
const FRONTIER_INITIAL_AMOUNT: u8 = 32;


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

// =============================================================================
// Phase 1 of swift-running-cheetah (docs/plans/swift-running-cheetah.md):
// KH 風 flat array + linear probing の ProvenTable
//
// 既存 `proven_map: FxHashMap<u64, Vec<ProvenEntry>>` の cache locality 問題
// (NPS 12× 残差の主因可能性) を解消するため，KomoringHeights `RegularTable`
// 相当の open-addressed flat array + linear probing + amount-based GC を導入する．
//
// Phase 1: 新 struct 定義 + 基本 API (lookup/insert/length/iter_pos_key)．
// 既存 `proven_map` と並列で動作させ，caller-facing API の差し替えは Phase 2 で行う．
// =============================================================================

/// `ProvenTable` の flat-array スロット．
///
/// レイアウト (24 bytes, alignment=8):
/// - `pos_key: u64` (8B) — 局面ハッシュ (持ち駒除く)．`0` = empty slot．
/// - `entry: ProvenEntry` (12B) — 既存 proven entry を流用．
/// - padding (4B, alignment)．
///
/// `safe_key` で実 `pos_key == 0` の局面は `1` に置換してから格納する．
#[derive(Clone, Copy)]
#[repr(C)]
pub(super) struct ProvenSlot {
    /// 局面ハッシュ (持ち駒除く)．`0` = empty slot．
    pos_key: u64,
    /// proven entry (proof / confirmed / refutable disproof)．
    entry: ProvenEntry,
}

impl Default for ProvenSlot {
    fn default() -> Self {
        Self {
            pos_key: 0,
            entry: ProvenEntry::ZERO,
        }
    }
}

/// `ProvenTable::lookup` / `insert` の linear probing 最大距離．
///
/// これを超えると lookup は `None`，insert は `false` を返す．caller は
/// GC をトリガーして容量を確保する．128 は load factor 50% 時の期待距離より
/// 十分大きい．
pub(super) const PROVEN_TABLE_MAX_PROBE: usize = 128;

/// Phase 4a (v0.63.0): `TranspositionTable::new()` 経由でデフォルト初期化される
/// `ProvenTable` の容量．`1 << 20` = 1M slots ≈ 24 MB．caller は
/// `set_use_kh_proven_tt(true, capacity)` で再初期化可能 (Phase 4 以降は
/// `on` 引数は dead，capacity のみ反映)．
pub(super) const DEFAULT_PROVEN_TABLE_CAPACITY: usize = 1 << 20;

/// `ProvenTable` の tombstone (削除済み slot) を表すセンチネル `pos_key`．
///
/// `pos_key == 0` は empty slot．`pos_key == TOMBSTONE_KEY` は tombstone
/// (削除済みだが probe chain を壊さないため slot を保持)．通常の
/// `TT::safe_key` は `pos_key | 1` で奇数を返すため，`u64::MAX` (奇数だが
/// 1 bit hash 衝突確率 2^-63 と無視可能) を tombstone として利用しても
/// 実害なし．衝突した場合，該当エントリが lookup で見えないだけで
/// soundness は保たれる (proven_map が primary store)．
pub(super) const PROVEN_TABLE_TOMBSTONE_KEY: u64 = u64::MAX;

/// KomoringHeights `RegularTable` 風の flat array + linear probing ProvenTT
/// (Phase 1, swift-running-cheetah)．
///
/// ## 設計
///
/// - 固定容量 `Vec<ProvenSlot>` (size は 2 のべき乗)．
/// - インデックス: `(hash_low * size) >> 32` (Stockfish 風)．
/// - 衝突処理: linear probing (`(idx + 1) % size`)．
/// - eviction: Phase 3 で amount-based GC (sampling) を実装予定．
///
/// ## Phase 1 の範囲
///
/// 基本 API (`lookup`, `insert`, `iter_pos_key`, length 系) のみ実装．
/// caller-facing API (`look_up_proven` 等) は Phase 2 で差し替える．
/// 既存 `proven_map: FxHashMap` と並列で動作する．
///
/// 関連: `docs/plans/swift-running-cheetah.md`．
pub(super) struct ProvenTable {
    /// flat array．`entries.len()` は `2^N` (`new` で保証)．
    entries: Vec<ProvenSlot>,
    /// 全エントリ数 (O(1) カウンタ)．
    total: usize,
    /// proof エントリ数．
    proofs: usize,
    /// confirmed disproof エントリ数 (refutable は別)．
    confirmed: usize,
}

impl ProvenTable {
    /// 指定容量で空 table を生成する．`capacity` は 2 のべき乗に切り上げ
    /// (最小 64) する．
    pub(super) fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(64);
        Self {
            entries: vec![ProvenSlot::default(); cap],
            total: 0,
            proofs: 0,
            confirmed: 0,
        }
    }

    /// `pos_key == 0` を `1` に置換する `safe_key`．
    /// 既存 `TranspositionTable::safe_key` と同等．
    #[inline(always)]
    fn safe_key(pos_key: u64) -> u64 {
        if pos_key == 0 { 1 } else { pos_key }
    }

    /// Stockfish 風インデックス計算 (`hash_low * size >> 32`)．
    /// `%` を回避し division-free．
    #[inline(always)]
    fn index(&self, pos_key: u64) -> usize {
        let hash_low = pos_key & 0xFFFF_FFFF;
        ((hash_low.wrapping_mul(self.entries.len() as u64)) >> 32) as usize
    }

    /// table 容量．
    #[inline(always)]
    pub(super) fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// 総エントリ数 (proof + confirmed + refutable)．
    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.total
    }

    /// proof エントリ数．
    #[inline(always)]
    pub(super) fn proof_len(&self) -> usize {
        self.proofs
    }

    /// confirmed disproof エントリ数．
    #[inline(always)]
    pub(super) fn confirmed_len(&self) -> usize {
        self.confirmed
    }

    /// refutable disproof エントリ数 (`total - proofs - confirmed`)．
    #[inline(always)]
    pub(super) fn refutable_len(&self) -> usize {
        self.total
            .saturating_sub(self.proofs)
            .saturating_sub(self.confirmed)
    }

    /// 全エントリをクリアする．
    pub(super) fn clear(&mut self) {
        for slot in self.entries.iter_mut() {
            slot.pos_key = 0;
        }
        self.total = 0;
        self.proofs = 0;
        self.confirmed = 0;
    }

    /// `(pos_key, hand)` で完全一致するエントリを検索する．
    ///
    /// linear probing で次の空 slot (`pos_key == 0`) に到達したら `None`．
    /// 一致しない pos_key および tombstone slot は skip して継続．
    pub(super) fn lookup(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> Option<&ProvenEntry> {
        let pos_key = Self::safe_key(pos_key);
        let mut idx = self.index(pos_key);
        let n = self.entries.len();
        for _ in 0..PROVEN_TABLE_MAX_PROBE {
            let slot = &self.entries[idx];
            if slot.pos_key == 0 {
                return None;
            }
            if slot.pos_key != PROVEN_TABLE_TOMBSTONE_KEY
                && slot.pos_key == pos_key
                && slot.entry.hand == *hand
            {
                return Some(&slot.entry);
            }
            idx += 1;
            if idx >= n {
                idx = 0;
            }
        }
        None
    }

    /// Phase 4b (v0.63.0): 全 (pos_key, &ProvenEntry) を走査する iterator．
    ///
    /// 診断 (count_proven 等)，clone_proven_map (test 用) 等の大域走査で使う．
    /// 空 slot および tombstone slot は skip する．
    pub(super) fn iter_all(&self) -> impl Iterator<Item = (u64, &ProvenEntry)> + '_ {
        self.entries.iter().filter_map(|slot| {
            if slot.pos_key == 0 || slot.pos_key == PROVEN_TABLE_TOMBSTONE_KEY {
                None
            } else {
                Some((slot.pos_key, &slot.entry))
            }
        })
    }

    /// 同じ `pos_key` の全エントリを線形に走査する iterator．
    ///
    /// `hand_gte_forward_chain` 等で antichain を走査する際に使う．
    /// 空 slot (`pos_key == 0`) に到達した時点で停止．
    pub(super) fn iter_pos_key<'a>(
        &'a self,
        pos_key: u64,
    ) -> ProvenTableIter<'a> {
        let pos_key = Self::safe_key(pos_key);
        ProvenTableIter {
            table: self,
            target_pos_key: pos_key,
            idx: self.index(pos_key),
            remaining: PROVEN_TABLE_MAX_PROBE,
            done: false,
        }
    }

    /// `(pos_key, hand)` のエントリを挿入または更新する．
    ///
    /// 既存エントリ (同 hand) があれば上書き．なければ空 slot に挿入．
    /// tombstone slot は最初に遭遇したものを記憶しておき，empty に到達した
    /// 場合に再利用する (load factor を保つ)．空 slot も tombstone も
    /// 見つからず MAX_PROBE 超過なら `false` を返す (caller は GC をトリガー)．
    pub(super) fn insert(&mut self, pos_key: u64, entry: ProvenEntry) -> bool {
        let pos_key = Self::safe_key(pos_key);
        let mut idx = self.index(pos_key);
        let n = self.entries.len();
        let mut first_tombstone: Option<usize> = None;
        for _ in 0..PROVEN_TABLE_MAX_PROBE {
            let slot = &mut self.entries[idx];
            if slot.pos_key == 0 {
                // empty: tombstone があれば再利用，なければここに挿入
                let target = first_tombstone.unwrap_or(idx);
                let s = &mut self.entries[target];
                s.pos_key = pos_key;
                s.entry = entry;
                self.total += 1;
                Self::adjust_counters(&mut self.proofs, &mut self.confirmed, &entry, true);
                return true;
            }
            if slot.pos_key == PROVEN_TABLE_TOMBSTONE_KEY {
                if first_tombstone.is_none() {
                    first_tombstone = Some(idx);
                }
            } else if slot.pos_key == pos_key && slot.entry.hand == entry.hand {
                // 更新: 旧 entry のカウンタを引き，新 entry のカウンタを足す．
                Self::adjust_counters(&mut self.proofs, &mut self.confirmed, &slot.entry, false);
                slot.entry = entry;
                Self::adjust_counters(&mut self.proofs, &mut self.confirmed, &entry, true);
                return true;
            }
            idx += 1;
            if idx >= n {
                idx = 0;
            }
        }
        // 走査終了．tombstone があれば再利用．
        if let Some(target) = first_tombstone {
            let s = &mut self.entries[target];
            s.pos_key = pos_key;
            s.entry = entry;
            self.total += 1;
            Self::adjust_counters(&mut self.proofs, &mut self.confirmed, &entry, true);
            return true;
        }
        false
    }

    /// `(pos_key, hand)` のエントリを削除する (tombstone marker)．
    ///
    /// 該当エントリを発見した場合，slot の `pos_key` を
    /// `PROVEN_TABLE_TOMBSTONE_KEY` に変更し，カウンタを減算する．
    /// linear probing chain を壊さないため slot 自体は保持する．
    /// 戻り値: 削除に成功したら `true`，見つからなければ `false`．
    pub(super) fn remove(&mut self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> bool {
        let pos_key = Self::safe_key(pos_key);
        let mut idx = self.index(pos_key);
        let n = self.entries.len();
        for _ in 0..PROVEN_TABLE_MAX_PROBE {
            let slot = &mut self.entries[idx];
            if slot.pos_key == 0 {
                return false;
            }
            if slot.pos_key != PROVEN_TABLE_TOMBSTONE_KEY
                && slot.pos_key == pos_key
                && slot.entry.hand == *hand
            {
                let removed = slot.entry;
                Self::adjust_counters(
                    &mut self.proofs,
                    &mut self.confirmed,
                    &removed,
                    false,
                );
                self.total -= 1;
                slot.pos_key = PROVEN_TABLE_TOMBSTONE_KEY;
                return true;
            }
            idx += 1;
            if idx >= n {
                idx = 0;
            }
        }
        false
    }

    /// 指定 `pos_key` の全エントリを tombstone 化する．
    ///
    /// `iter_pos_key` 相当の走査で同一 `pos_key` の slot を全て削除．
    /// 戻り値は削除したエントリ数．Phase 2a-2 の `sync_proven_table_for_pos_key`
    /// で使用 (proven_map と整合させるため，差分計算より単純な
    /// 「pos_key 全消去 + proven_map から再挿入」戦略を採用)．
    pub(super) fn remove_pos_key(&mut self, pos_key: u64) -> usize {
        let pos_key = Self::safe_key(pos_key);
        let mut idx = self.index(pos_key);
        let n = self.entries.len();
        let mut removed = 0usize;
        for _ in 0..PROVEN_TABLE_MAX_PROBE {
            let slot = &mut self.entries[idx];
            if slot.pos_key == 0 {
                break;
            }
            if slot.pos_key == pos_key {
                let entry = slot.entry;
                Self::adjust_counters(
                    &mut self.proofs,
                    &mut self.confirmed,
                    &entry,
                    false,
                );
                self.total -= 1;
                slot.pos_key = PROVEN_TABLE_TOMBSTONE_KEY;
                removed += 1;
            }
            idx += 1;
            if idx >= n {
                idx = 0;
            }
        }
        removed
    }

    /// 負荷率 (実エントリ数 / 容量)．`0.0..=1.0`．Phase 3 (v0.62.0)．
    ///
    /// caller は容量逼迫の指標として参照する．KH `IsAlmostFull()` 相当の
    /// 閾値 (例: 0.94) を超えた場合に `collect_garbage` を呼ぶ運用を想定．
    #[inline(always)]
    pub(super) fn load_factor(&self) -> f64 {
        self.total as f64 / self.entries.len() as f64
    }

    /// KH 風 sampling-based amount GC (Phase 3, v0.62.0)．
    ///
    /// `removal_ratio` (`0.0..=1.0`) を target ratio として，sampling で
    /// amount の `removal_ratio` 分位点 (低い側) を threshold に決定し，
    /// `amount() <= threshold` のエントリを tombstone 化する．
    ///
    /// ## アルゴリズム
    ///
    /// 1. `SAMPLE_SIZE = 20_000` 個のエントリを stride 走査でサンプリング
    ///    (KH `transposition_table.cpp::CollectGarbage` 相当)．
    /// 2. `select_nth_unstable` で `removal_ratio` 分位点を threshold とする．
    /// 3. 全 slot 走査して `amount() <= threshold` を tombstone 化．
    ///
    /// ## `only_refutable` パラメータ
    ///
    /// - `true`: refutable disproof のみを eviction 対象 (proof / confirmed
    ///   は永続として保護)．既存 `TranspositionTable::gc_proven` と同等．
    /// - `false`: proof / confirmed / refutable 全てを対象 (KH 同等)．
    ///   Phase 4 で proven_map 削除後に proof/confirmed の GC も統合する場合
    ///   に使う．現状の Phase 3 では未使用．
    ///
    /// **戻り値**: tombstone 化したエントリ数．
    ///
    /// **注意**: Phase 3 (v0.62.0) ではこの API を追加するのみで，caller は
    /// 接続しない (proven_map が primary store のため)．Phase 4 で proven_map
    /// 削除後に caller を接続する．
    pub(super) fn collect_garbage(
        &mut self,
        removal_ratio: f64,
        only_refutable: bool,
    ) -> usize {
        if self.total == 0 {
            return 0;
        }
        let removal_ratio = removal_ratio.clamp(0.0, 1.0);
        if removal_ratio <= 0.0 {
            return 0;
        }

        const SAMPLE_SIZE: usize = 20_000;
        let n = self.entries.len();
        let stride = (n / SAMPLE_SIZE).max(1);

        let mut samples: Vec<u8> = Vec::with_capacity(SAMPLE_SIZE.min(self.total));
        let mut idx = 0usize;
        let mut visited = 0usize;
        // sample 数または slot 走査数のどちらかが上限に達するまで走査
        while samples.len() < SAMPLE_SIZE && visited < n {
            let slot = &self.entries[idx];
            if slot.pos_key != 0 && slot.pos_key != PROVEN_TABLE_TOMBSTONE_KEY {
                if !only_refutable || slot.entry.is_refutable_disproof() {
                    samples.push(slot.entry.amount());
                }
            }
            idx = (idx + stride) % n;
            visited += stride;
        }
        if samples.is_empty() {
            return 0;
        }

        // threshold: removal_ratio 分位点 (nth_element)
        let pivot = ((samples.len() as f64) * removal_ratio) as usize;
        let pivot = pivot.min(samples.len() - 1);
        samples.select_nth_unstable(pivot);
        let threshold = samples[pivot];

        // eviction pass: amount <= threshold の対象エントリを tombstone 化
        let mut removed = 0usize;
        for slot in self.entries.iter_mut() {
            if slot.pos_key == 0 || slot.pos_key == PROVEN_TABLE_TOMBSTONE_KEY {
                continue;
            }
            if only_refutable && !slot.entry.is_refutable_disproof() {
                continue;
            }
            if slot.entry.amount() <= threshold {
                let entry = slot.entry;
                Self::adjust_counters(
                    &mut self.proofs,
                    &mut self.confirmed,
                    &entry,
                    false,
                );
                self.total -= 1;
                slot.pos_key = PROVEN_TABLE_TOMBSTONE_KEY;
                removed += 1;
            }
        }
        removed
    }

    /// `proofs` / `confirmed` カウンタを `entry` 種別に応じて増減する．
    /// `add=true` で +1, `add=false` で saturating -1．
    #[inline(always)]
    fn adjust_counters(
        proofs: &mut usize,
        confirmed: &mut usize,
        entry: &ProvenEntry,
        add: bool,
    ) {
        if entry.is_proof() {
            *proofs = if add { *proofs + 1 } else { proofs.saturating_sub(1) };
        } else if !entry.is_refutable_disproof() {
            *confirmed = if add { *confirmed + 1 } else { confirmed.saturating_sub(1) };
        }
    }
}

/// `ProvenTable::iter_pos_key` の戻り値．同一 `pos_key` のエントリ群を
/// linear probing 順で走査する．
pub(super) struct ProvenTableIter<'a> {
    table: &'a ProvenTable,
    target_pos_key: u64,
    idx: usize,
    remaining: usize,
    done: bool,
}

impl<'a> Iterator for ProvenTableIter<'a> {
    type Item = &'a ProvenEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let n = self.table.entries.len();
        while self.remaining > 0 {
            self.remaining -= 1;
            let slot = &self.table.entries[self.idx];
            let cur_idx = self.idx;
            self.idx += 1;
            if self.idx >= n {
                self.idx = 0;
            }
            if slot.pos_key == 0 {
                self.done = true;
                return None;
            }
            if slot.pos_key == PROVEN_TABLE_TOMBSTONE_KEY {
                // tombstone: skip but keep probing
                continue;
            }
            if slot.pos_key == self.target_pos_key {
                let _ = cur_idx;
                return Some(&slot.entry);
            }
        }
        self.done = true;
        None
    }
}

/// Phase 4b (v0.63.0, swift-running-cheetah): proven entry の読出経路は
/// `proven_table: ProvenTable` 単独．以前は `Option<ProvenTable>` + `proven_map`
/// fallback の dispatched enum だったが，proven_map 廃止に伴いシンプルな
/// type alias に置換．
pub(super) type ProvenEntriesIter<'a> = ProvenTableIter<'a>;

/// Dual フラットハッシュテーブル型転置表(証明駒/反証駒対応)．
///
/// v0.24.0: ProvenTT + WorkingTT の 2 テーブル構成．
/// - ProvenTT: proof(pn=0) + confirmed disproof(dn=0, !path_dep, remaining=INFINITE)
/// - WorkingTT: intermediate(pn>0, dn>0) + depth-limited/path-dep disproof
///
/// ProvenTT の永続エントリが WorkingTT のクラスタを圧迫しないため，
/// クラスタ飽和問題(§6.6.1)が構造的に解消される．
pub(super) struct TranspositionTable {
    // Phase 4d (v0.63.0): ProvenTT は `proven_table: ProvenTable` 単独運用．
    // 旧 `proven_map: FxHashMap` と TT 側カウンタ
    // (proven_total_entries / proven_confirmed_entries / proven_proof_entries)
    // は廃止．ProvenTable の O(1) カウンタ (len / proof_len / confirmed_len /
    // refutable_len) を直接参照する．
    /// Phase 2 swift-running-cheetah (v0.59.0): KH 風 flat array + linear probing
    /// ProvenTable．`Some` で flag ON 状態 (新実装を使用)，`None` で flag OFF
    /// (既存 `proven_map` を使用)．設定は `set_use_kh_proven_tt(true, capacity)`
    /// で行う．既存 caller への影響なし (default は None)．
    ///
    /// 関連: `docs/plans/swift-running-cheetah.md`．
    /// Phase 4a (v0.63.0): 常設化．`Option` を排除し，`with_clusters` で
    /// `DEFAULT_PROVEN_TABLE_CAPACITY` の容量で初期化される．`set_use_kh_proven_tt`
    /// は今後 capacity 変更 (再初期化) を行う．
    proven_table: ProvenTable,
    /// WorkingTT: GC 対象エントリ(intermediate + depth-limited disproof)．
    working: Vec<TTFlatEntry>,
    /// LeafDisproofTT: remaining ≤ 2 の overflow 専用 compact テーブル (N-8, v0.26.0)．
    /// WorkingTT クラスタが飽和した際の remaining ≤ 2 エントリのフォールバック先．
    /// 16B/entry で WorkingTT (32B) の半サイズ．512K clusters × 4 = 2M entries = 32MB．
    leaf_disproofs: Vec<TTLeafEntry>,
    /// WorkingTT の `num_clusters - 1`．
    working_mask: usize,
    /// LeafDisproofTT の `num_clusters - 1`．
    leaf_mask: usize,
    /// store_proven での amount 計算に使用する現在の ply．
    /// mid() が TT store 前にセットする．ply が小さい(ルートに近い)ほど
    /// amount が高くなり，eviction 耐性が上がる．
    pub(super) hint_ply: u32,
    /// Phase 18 (v0.97.0): store_proven で WorkingTT の被支配 intermediate を削除しない．
    /// IDS find_shortest で working pn/dn を保持するために使う．
    pub(super) preserve_working_on_proof: bool,
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
    /// FrontierTT: remaining ≤ FRONTIER_REMAINING_THRESHOLD の intermediate 専用テーブル (案4, v0.55.7)．
    ///
    /// WorkingTT クラスタ overflow による eviction から保護するため，
    /// フロンティアノード (remaining が小さい = ply が深い) の intermediate を分離する．
    /// 256K clusters × 8 × 32B = 64MB．
    frontier: Vec<TTFlatEntry>,
    /// FrontierTT の `num_clusters - 1`(高速 modulo 用ビットマスク)．
    frontier_mask: usize,
    /// FrontierTT のクラスタ overflow 累積カウンタ．
    frontier_overflow_since_gc: u64,
}

impl TranspositionTable {
    /// 転置表を生成する(デフォルトサイズ)．
    pub(super) fn new() -> Self {
        Self::with_clusters(DEFAULT_NUM_CLUSTERS)
    }

    /// 指定クラスタ数で転置表を生成する．
    fn with_clusters(num_clusters: usize) -> Self {
        let working_clusters = num_clusters.next_power_of_two();
        let working_total = working_clusters * WORKING_CLUSTER_SIZE;
        let leaf_total = LEAF_NUM_CLUSTERS * LEAF_CLUSTER_SIZE;
        let frontier_total = FRONTIER_NUM_CLUSTERS * FRONTIER_CLUSTER_SIZE;
        TranspositionTable {
            proven_table: ProvenTable::new(DEFAULT_PROVEN_TABLE_CAPACITY),
            working: vec![TTFlatEntry::EMPTY; working_total],
            leaf_disproofs: vec![TTLeafEntry::EMPTY; leaf_total],
            frontier: vec![TTFlatEntry::EMPTY; frontier_total],
            working_mask: working_clusters - 1,
            leaf_mask: LEAF_NUM_CLUSTERS - 1,
            frontier_mask: FRONTIER_NUM_CLUSTERS - 1,
            hint_ply: 0,
            preserve_working_on_proof: false,
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
            frontier_overflow_since_gc: 0,
        }
    }

    /// Phase 4a (v0.63.0): `proven_table` を指定 capacity で再初期化する．
    ///
    /// 以前は opt-in flag だったが，Phase 4 で常設化．`on` 引数は dead
    /// (互換のため残置)．呼ぶと既存内容は破棄され空 table になる．
    /// 探索開始前のみ呼ぶこと．
    pub(super) fn set_use_kh_proven_tt(&mut self, _on: bool, capacity: usize) {
        self.proven_table = ProvenTable::new(capacity);
    }

    /// `ProvenTable` の `(len, proof_len, confirmed_len, refutable_len)` を返す．
    /// Phase 4a (v0.63.0): 常設化により `Option` を返さない．
    pub(super) fn proven_table_stats(&self) -> (usize, usize, usize, usize) {
        let t = &self.proven_table;
        (t.len(), t.proof_len(), t.confirmed_len(), t.refutable_len())
    }

    /// 指定 `pos_key` の proven entry を走査する iterator．Phase 4b (v0.63.0)
    /// で proven_table 単独になったため，直接 `iter_pos_key` を呼ぶシンプルな
    /// helper．`pos_key` は caller 側で `Self::safe_key` 済みであること．
    #[inline(always)]
    pub(super) fn iter_proven_entries(&self, pos_key: u64) -> ProvenEntriesIter<'_> {
        self.proven_table.iter_pos_key(pos_key)
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

    /// hand_hash 値から直接クラスタ開始位置を計算する(WorkingTT)．
    #[inline(always)]
    fn working_cluster_start_from_hash(&self, pos_key: u64, hh: u64) -> usize {
        ((pos_key ^ hh) as usize & self.working_mask) * WORKING_CLUSTER_SIZE
    }

    /// LeafDisproofTT のクラスタ開始インデックス (N-8)．
    ///
    /// WorkingTT と同じ fc_hand_hash を使用し，leaf_mask でクラスタを選択する．
    #[inline(always)]
    fn leaf_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::fc_hand_hash(hand);
        ((mixed as usize) & self.leaf_mask) * LEAF_CLUSTER_SIZE
    }

    /// FrontierTT のクラスタ開始インデックス (案4, v0.55.7)．
    ///
    /// WorkingTT と同じ fc_hand_hash を使用し，frontier_mask でクラスタを選択する．
    /// pos_key は呼び出し元で safe_key 適用済みであること．
    #[inline(always)]
    fn frontier_cluster_start(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let mixed = pos_key ^ Self::fc_hand_hash(hand);
        ((mixed as usize) & self.frontier_mask) * FRONTIER_CLUSTER_SIZE
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

        // 案4 (v0.55.7): FrontierTT を優先チェック (remaining ≤ FRONTIER_REMAINING_THRESHOLD)．
        // WorkingTT eviction thrashing を受けない専用プールの intermediate を先に探す．
        //
        // v0.55.8 修正: disproof は WorkingTT に格納されるため，frontier intermediate より優先する．
        // store_working_disproof は WorkingTT の同 pos_key intermediate を除去するが frontier は
        // 触らない — frontier に stale intermediate が残ると disproof を見落とす．
        // 対策: frontier intermediate ヒット前に WorkingTT disproof を確認し，あれば優先返却．
        if remaining <= FRONTIER_REMAINING_THRESHOLD {
            // Step 1: WorkingTT で disproof を先に確認 (frontier の stale intermediate より優先)
            let w_start = self.working_cluster_start(pos_key, hand);
            for fe in &self.working[w_start..w_start + WORKING_CLUSTER_SIZE] {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.dn == 0
                    && hand_gte_forward_chain(&e.hand, hand)
                    && (e.remaining() >= remaining || e.path_dependent())
                {
                    return (e.pn, 0, e.source);
                }
            }
            // Step 2: FrontierTT で intermediate を確認
            let f_start = self.frontier_cluster_start(pos_key, hand);
            let f_cluster = &self.frontier[f_start..f_start + FRONTIER_CLUSTER_SIZE];
            let mut f_exact: Option<(u32, u32, u32)> = None;
            let mut f_fc: Option<(u32, u32, u32)> = None;
            for fe in f_cluster {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.pn != 0 && e.dn != 0 {
                    if e.pn == u32::MAX && e.remaining() < remaining { continue; }
                    if e.hand == *hand {
                        f_exact = Some((e.pn, e.dn, e.source));
                    } else if f_fc.is_none() && hand_gte_forward_chain(hand, &e.hand) {
                        f_fc = Some((e.pn, e.dn, e.source));
                    }
                }
            }
            if let Some(m) = f_exact { return m; }
            if let Some(m) = f_fc { return m; }
            // FrontierTT ミス: disproof は WorkingTT / LeafDisproofTT で継続探索
        }

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
                // pn=INF intermediate は depth-limited 証明不能．
                // dn=0 disproof と同様に，保存時の remaining が現在の remaining
                // より浅い場合は無効 — より深い探索で再展開が必要．
                if e.pn == u32::MAX && e.remaining() < remaining { continue; }
                if e.hand == *hand {
                    exact_match = Some((e.pn, e.dn, e.source));
                } else if fc_match.is_none()
                    && hand_gte_forward_chain(hand, &e.hand)
                {
                    fc_match = Some((e.pn, e.dn, e.source));
                }
            }
        }

        // 案4 (v0.55.8): remaining > FRONTIER_REMAINING_THRESHOLD の lookup でも，
        // frontier に残る remaining≤threshold エントリを保守的推定 fallback として使用．
        // baseline では remaining≤2 エントリが WorkingTT に存在し，remaining>2 の lookup で
        // 保守的推定として返っていた．frontier 導入後は WorkingTT に存在しないため補完する．
        // pn=INF エントリは常にスキップ (frontier remaining ≤ threshold < remaining であり，
        // baseline の `e.remaining() < remaining` スキップ条件と等価)．
        if FRONTIER_REMAINING_THRESHOLD > 0
            && remaining > FRONTIER_REMAINING_THRESHOLD
            && exact_match.is_none()
        {
            let f_start = self.frontier_cluster_start(pos_key, hand);
            for fe in &self.frontier[f_start..f_start + FRONTIER_CLUSTER_SIZE] {
                if fe.pos_key != pos_key { continue; }
                let e = &fe.entry;
                if e.pn != 0 && e.dn != 0 && e.pn != u32::MAX {
                    if e.hand == *hand {
                        exact_match = Some((e.pn, e.dn, e.source));
                        break;
                    } else if fc_match.is_none() && hand_gte_forward_chain(hand, &e.hand) {
                        fc_match = Some((e.pn, e.dn, e.source));
                    }
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
    pub(super) fn look_up_proven(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u32) {
        self.look_up_proven_impl(pos_key, hand, remaining, false)
    }

    #[inline]
    fn look_up_proven_impl(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
        skip_refutable: bool,
    ) -> (u32, u32, u32) {
        let pos_key = Self::safe_key(pos_key);
        let _ = remaining;
        // Tag-aware proof lookup: non-ABSOLUTE proof は tag_depth <
        // current_ids_depth の場合にスキップする．
        let ids_depth = self.current_ids_depth;
        // Pass 1: proof(pn=0)
        for e in self.iter_proven_entries(pos_key) {
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
        // Pass 2: confirmed/refutable disproof(dn=0)
        // ProvenTT の confirmed disproof は常に depth 非依存として格納される．
        for e in self.iter_proven_entries(pos_key) {
            if !e.is_proof() && hand_gte_forward_chain(&e.hand, hand) {
                if skip_refutable && e.is_refutable_disproof() { continue; }
                return (e.pn(), 0, e.source());
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
        let proven = self.look_up_proven(pos_key, hand, remaining);
        if proven.0 == 0 || proven.1 == 0 {
            return proven;
        }
        self.look_up_working(pos_key, hand, remaining, neighbor_scan)
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
        // Phase 4b (v0.63.0): proven_table 単独．同一 hand の既存 proof の amount を比較．
        let mut dominated = false;
        let mut should_replace = false;
        for e in self.proven_table.iter_pos_key(pos_key) {
            if e.is_proof() && e.hand == hand {
                if new_priority >= e.amount() {
                    should_replace = true;
                } else {
                    dominated = true;
                }
                break; // 同一 (pos_key, hand) は最大 1 件
            }
        }
        if dominated {
            return;
        }
        if should_replace {
            // 旧エントリを除去．insert が更新も対応するため明示 remove 不要だが，
            // 念のため counters のため remove → insert で書き直す．
            self.proven_table.remove(pos_key, &hand);
        }
        self.proven_table.insert(pos_key, new_entry);
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
        // ProvenTT をチェック．Phase 2b (v0.61.0): flag ON 時は proven_table から走査．
        // 注意: v0.24.53 以降，ProvenTT の confirmed disproof は depth 非依存
        // として格納されるため `remaining` との比較は不要．
        for e in self.iter_proven_entries(pos_key) {
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

        if is_proof {
            // WorkingTT の被支配 intermediate も除去
            // Phase 18: preserve_working_on_proof 時は IDS 用 working pn/dn 保持のため skip
            if !self.preserve_working_on_proof {
                let w_start = self.working_cluster_start(pos_key, &hand);
                let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
                for fe in w_cluster.iter_mut() {
                    if fe.pos_key != pos_key { continue; }
                    if fe.entry.dn == 0 { continue; }
                    if hand_gte_forward_chain(&fe.entry.hand, &hand) {
                        fe.pos_key = 0;
                    }
                }
            }

            // Phase 4b (v0.63.0): proven_table 単独．被支配 proof を収集 → 除去
            let mut dominated = false;
            let mut to_remove: arrayvec::ArrayVec<[u8; HAND_KINDS], 16> =
                arrayvec::ArrayVec::new();
            for e in self.proven_table.iter_pos_key(pos_key) {
                if !e.is_proof() { continue; }
                // 被支配 proof: old_hand ≥ new_hand → 除去対象
                if hand_gte_forward_chain(&e.hand, &hand) {
                    let _ = to_remove.try_push(e.hand);
                    continue;
                }
                // 同一 hand の proof: amount が高い方を残す
                if e.hand == hand {
                    if new_priority >= e.amount() {
                        let _ = to_remove.try_push(e.hand);
                    } else {
                        dominated = true;
                    }
                }
            }
            for h in &to_remove {
                self.proven_table.remove(pos_key, h);
            }
            if dominated { return; }
            self.proven_table.insert(pos_key, new_entry);
        } else {
            // WorkingTT の同一 pos_key エントリを積極的に除去(proof 以外)．
            let w_start = self.working_cluster_start(pos_key, &hand);
            let w_cluster = &mut self.working[w_start..w_start + WORKING_CLUSTER_SIZE];
            for fe in w_cluster.iter_mut() {
                if fe.pos_key != pos_key { continue; }
                if fe.entry.pn == 0 { continue; } // proof は ProvenTT にあるが防衛的に保護
                fe.pos_key = 0;
            }

            // Phase 4b (v0.63.0): 同一 hand の既存 disproof (proof でない) を除去
            let mut to_remove: arrayvec::ArrayVec<[u8; HAND_KINDS], 16> =
                arrayvec::ArrayVec::new();
            for e in self.proven_table.iter_pos_key(pos_key) {
                if !e.is_proof() && e.hand == hand {
                    let _ = to_remove.try_push(e.hand);
                }
            }
            for h in &to_remove {
                self.proven_table.remove(pos_key, h);
            }
            // 新 confirmed disproof を挿入
            self.proven_table.insert(pos_key, new_entry);
        }

        #[cfg(feature = "verbose")] {
            if is_proof { self.diag_proof_inserts += 1; }
            else { self.diag_disproof_inserts += 1; }
            self.diag_remaining_dist[rem_idx] += 1;
        }
        #[cfg(feature = "tt_diag")]
        if !is_proof { self.diag_disproof_confirmed += 1; }
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
        // 案4 (v0.55.7): remaining ≤ threshold の intermediate は FrontierTT へ
        if remaining <= FRONTIER_REMAINING_THRESHOLD {
            self.store_frontier_intermediate(pos_key, hand, pn, dn, remaining, source, best_move);
            return;
        }

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

        // 新規エントリを追加 (remaining ≤ FRONTIER_REMAINING_THRESHOLD は store_frontier_intermediate でブースト付き)
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

    /// FrontierTT に intermediate エントリを挿入する (案4, v0.55.7)．
    ///
    /// pos_key は呼び出し元で safe_key 適用済みであること．
    /// WorkingTT と同様の update-or-insert-or-evict ロジックを使用する．
    fn store_frontier_intermediate(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u32,
        best_move: u16,
    ) {
        let f_start = self.frontier_cluster_start(pos_key, &hand);

        // 同一持ち駒の既存エントリを更新
        for fe in self.frontier[f_start..f_start + FRONTIER_CLUSTER_SIZE].iter_mut() {
            if fe.pos_key != pos_key { continue; }
            let e = &mut fe.entry;
            if e.hand == hand {
                if e.dn == 0 && (e.remaining() >= remaining || e.path_dependent()) {
                    return;
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
                return;
            }
        }

        // 新規エントリを追加 (初期 amount ブースト付き)
        let new_entry = DfPnEntry::new(
            source, pn, dn, hand, remaining, false, best_move, FRONTIER_INITIAL_AMOUNT,
        );
        if let Some(slot) = self.frontier[f_start..f_start + FRONTIER_CLUSTER_SIZE]
            .iter_mut()
            .find(|fe| fe.is_empty())
        {
            slot.pos_key = pos_key;
            slot.entry = new_entry;
        } else {
            self.frontier_overflow_since_gc += 1;
            let f_cluster = &mut self.frontier[f_start..f_start + FRONTIER_CLUSTER_SIZE];
            Self::replace_weakest_in(f_cluster, pos_key, new_entry);
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

    /// WorkingTT の confirmed disproof (dn=0, !path_dep, remaining=INF) の数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_working_confirmed_disproofs(&self) -> usize {
        self.working.iter().filter(|fe| {
            fe.pos_key != 0
                && fe.entry.dn == 0
                && !fe.entry.path_dependent()
                && fe.entry.remaining() == REMAINING_INFINITE
        }).count()
    }

    /// ProvenTT の ephemeral エントリ (refutable disproof + non-ABSOLUTE proof)
    /// を選択的に除去する．
    ///
    /// IDS depth 切り替え時に呼び出す．2 種のエントリを除去する:
    ///
    /// 1. **refutable disproof** (is_refutable_disproof=true): `all_checks_refutable_recursive`
    ///    由来の heuristic disproof．浅い depth で格納されたものが深い探索を
    ///    汚染するのを防ぐ．`disproof_depth() < min_depth` のエントリのみ除去．
    ///    confirmed disproof (remaining=REMAINING_INFINITE) は深さ非依存の
    ///    永続エントリのため絶対に除去しない (v0.55.28 バグ#2 修正)．
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
        // Phase 4b (v0.63.0): proven_table 直接走査．条件に合致しないエントリを remove．
        let mut to_remove: Vec<(u64, [u8; HAND_KINDS])> = Vec::new();
        for (pk, e) in self.proven_table.iter_all() {
            let keep = if e.is_proof() {
                e.proof_tag() == super::entry::PROOF_TAG_ABSOLUTE
                    || e.tag_depth() >= min_depth
            } else {
                !e.is_refutable_disproof() || e.disproof_depth() >= min_depth
            };
            if !keep {
                to_remove.push((pk, e.hand));
            }
        }
        for (pk, h) in &to_remove {
            self.proven_table.remove(*pk, h);
        }
    }

    /// WorkingTT の非空エントリ数を返す．
    pub(super) fn len(&self) -> usize {
        self.working.iter().filter(|fe| fe.pos_key != 0).count()
    }

    /// ProvenTT の非空エントリ数を返す (O(1))．
    pub(super) fn proven_len(&self) -> usize {
        self.proven_table.len()
    }

    /// proof エントリ数を返す (O(1))．
    pub(super) fn proven_proof_len(&self) -> usize {
        self.proven_table.proof_len()
    }

    /// confirmed disproof エントリ数を返す (O(1))．
    pub(super) fn proven_confirmed_len(&self) -> usize {
        self.proven_table.confirmed_len()
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

    /// 指定局面のエントリ数を返す(診断用)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) fn entries_for_position(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) -> usize {
        let pos_key = Self::safe_key(pos_key);
        let p = self.proven_map.get(&pos_key).map_or(0, |v| v.len());
        let w = self.working_cluster(pos_key, hand).iter().filter(|fe| fe.pos_key == pos_key).count();
        p + w
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn dump_entries(&self, pos_key: u64, hand: &[u8; HAND_KINDS]) {
        let pos_key = Self::safe_key(pos_key);
        verbose_eprintln!("[tt_dump] ProvenTT:");
        if let Some(vec) = self.proven_map.get(&pos_key) {
            for (i, e) in vec.iter().enumerate() {
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
        self.proven_map.values().flat_map(|v| v.iter()).filter(|e| e.is_proof()).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す(ProvenTT + WorkingTT)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_disproven(&self) -> usize {
        let p = self.proven_map.values().flat_map(|v| v.iter()).filter(|e| !e.is_proof()).count();
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
        for e in self.proven_map.values().flat_map(|v| v.iter()) {
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

// =============================================================================
// Phase 1 unit tests for swift-running-cheetah ProvenTable
// =============================================================================

#[cfg(test)]
mod proven_table_tests {
    use super::*;

    #[test]
    fn test_proven_table_basic_lookup_miss() {
        let table = ProvenTable::new(64);
        let hand = [0u8; HAND_KINDS];
        assert!(table.lookup(0x1234, &hand).is_none());
        assert_eq!(table.len(), 0);
        assert_eq!(table.proof_len(), 0);
        assert_eq!(table.confirmed_len(), 0);
        assert_eq!(table.refutable_len(), 0);
    }

    #[test]
    fn test_proven_table_capacity_rounding() {
        // capacity は 2^N に切り上げ
        assert_eq!(ProvenTable::new(60).capacity(), 64);
        assert_eq!(ProvenTable::new(64).capacity(), 64);
        assert_eq!(ProvenTable::new(100).capacity(), 128);
        // 最小は 64
        assert_eq!(ProvenTable::new(1).capacity(), 64);
    }

    #[test]
    fn test_proven_table_store_lookup() {
        let mut table = ProvenTable::new(64);
        let hand = [1u8, 0, 0, 0, 0, 0, 0];
        let entry = ProvenEntry::new_proof(hand, 0, 15);

        assert!(table.insert(0x1234, entry));
        let found = table.lookup(0x1234, &hand);
        assert!(found.is_some());
        assert!(found.unwrap().is_proof());
        assert_eq!(table.len(), 1);
        assert_eq!(table.proof_len(), 1);
    }

    #[test]
    fn test_proven_table_safe_key_collision() {
        let mut table = ProvenTable::new(64);
        let hand = [1u8, 0, 0, 0, 0, 0, 0];
        let entry = ProvenEntry::new_proof(hand, 0, 15);

        // pos_key = 0 は safe_key で 1 に置換される．
        // pos_key=0 と pos_key=1 は同じ slot に格納される．
        assert!(table.insert(0, entry));
        assert!(table.lookup(0, &hand).is_some());
        assert!(table.lookup(1, &hand).is_some());
    }

    #[test]
    fn test_proven_table_counters() {
        let mut table = ProvenTable::new(64);
        let hand1 = [1u8, 0, 0, 0, 0, 0, 0];
        let hand2 = [2u8, 0, 0, 0, 0, 0, 0];

        // 2 proof + 1 confirmed disproof を挿入
        table.insert(0x1, ProvenEntry::new_proof(hand1, 0, 15));
        table.insert(0x2, ProvenEntry::new_proof(hand2, 0, 17));
        table.insert(0x3, ProvenEntry::new_disproof(hand1, 30));

        assert_eq!(table.len(), 3);
        assert_eq!(table.proof_len(), 2);
        assert_eq!(table.confirmed_len(), 1);
        assert_eq!(table.refutable_len(), 0);
    }

    #[test]
    fn test_proven_table_iter_pos_key() {
        let mut table = ProvenTable::new(64);
        let hand1 = [1u8, 0, 0, 0, 0, 0, 0];
        let hand2 = [2u8, 0, 0, 0, 0, 0, 0];

        // 同じ pos_key で異なる hand を 2 個挿入
        table.insert(0x100, ProvenEntry::new_proof(hand1, 0, 15));
        table.insert(0x100, ProvenEntry::new_proof(hand2, 0, 17));

        let count = table.iter_pos_key(0x100).count();
        assert_eq!(count, 2);
        // 異なる pos_key のは 0 個
        assert_eq!(table.iter_pos_key(0x200).count(), 0);
    }

    #[test]
    fn test_proven_table_update_existing() {
        let mut table = ProvenTable::new(64);
        let hand = [1u8, 0, 0, 0, 0, 0, 0];

        // 最初 proof として挿入
        table.insert(0x1, ProvenEntry::new_proof(hand, 0, 15));
        assert_eq!(table.proof_len(), 1);
        assert_eq!(table.confirmed_len(), 0);

        // 同じ pos_key + hand で disproof に更新
        table.insert(0x1, ProvenEntry::new_disproof(hand, 30));
        assert_eq!(table.len(), 1, "total should remain 1 after update");
        assert_eq!(table.proof_len(), 0, "proof counter should decrement");
        assert_eq!(table.confirmed_len(), 1, "confirmed counter should increment");
    }

    #[test]
    fn test_proven_table_clear() {
        let mut table = ProvenTable::new(64);
        let hand = [1u8, 0, 0, 0, 0, 0, 0];
        table.insert(0x1, ProvenEntry::new_proof(hand, 0, 15));
        table.insert(0x2, ProvenEntry::new_disproof(hand, 30));
        assert_eq!(table.len(), 2);

        table.clear();
        assert_eq!(table.len(), 0);
        assert_eq!(table.proof_len(), 0);
        assert_eq!(table.confirmed_len(), 0);
        assert!(table.lookup(0x1, &hand).is_none());
    }

    /// Phase 2a-2 (v0.60.0): tombstone-based `remove` の基本動作．
    #[test]
    fn test_proven_table_remove_basic() {
        let mut table = ProvenTable::new(64);
        let h1 = [1u8, 0, 0, 0, 0, 0, 0];
        let h2 = [2u8, 0, 0, 0, 0, 0, 0];
        table.insert(0x1, ProvenEntry::new_proof(h1, 0, 15));
        table.insert(0x1, ProvenEntry::new_proof(h2, 0, 15));
        assert_eq!(table.len(), 2);
        assert_eq!(table.proof_len(), 2);

        assert!(table.remove(0x1, &h1));
        assert_eq!(table.len(), 1);
        assert_eq!(table.proof_len(), 1);
        // h1 はもう見えない，h2 は依然見える
        assert!(table.lookup(0x1, &h1).is_none());
        assert!(table.lookup(0x1, &h2).is_some());

        // 同じ削除を繰り返しても false
        assert!(!table.remove(0x1, &h1));
    }

    /// Phase 2a-2: `remove` で tombstone slot に新エントリが再挿入できること．
    #[test]
    fn test_proven_table_remove_then_reinsert() {
        let mut table = ProvenTable::new(64);
        let h1 = [1u8, 0, 0, 0, 0, 0, 0];
        table.insert(0x1, ProvenEntry::new_proof(h1, 0, 15));
        assert_eq!(table.len(), 1);

        assert!(table.remove(0x1, &h1));
        assert_eq!(table.len(), 0);

        // 再挿入: tombstone slot が再利用される (容量消費なし)．
        let entry2 = ProvenEntry::new_disproof(h1, 30);
        assert!(table.insert(0x1, entry2));
        assert_eq!(table.len(), 1);
        assert_eq!(table.confirmed_len(), 1);
        let found = table.lookup(0x1, &h1).unwrap();
        assert!(!found.is_proof());
    }

    /// Phase 2a-2: `remove_pos_key` で同一 pos_key の全エントリを削除．
    #[test]
    fn test_proven_table_remove_pos_key() {
        let mut table = ProvenTable::new(64);
        let h1 = [1u8, 0, 0, 0, 0, 0, 0];
        let h2 = [2u8, 0, 0, 0, 0, 0, 0];
        let h3 = [3u8, 0, 0, 0, 0, 0, 0];
        table.insert(0x100, ProvenEntry::new_proof(h1, 0, 15));
        table.insert(0x100, ProvenEntry::new_proof(h2, 0, 17));
        table.insert(0x100, ProvenEntry::new_disproof(h3, 30));
        // 他の pos_key も追加 (削除されないこと)
        table.insert(0x200, ProvenEntry::new_proof(h1, 0, 15));
        assert_eq!(table.len(), 4);

        let removed = table.remove_pos_key(0x100);
        assert_eq!(removed, 3);
        assert_eq!(table.len(), 1);
        assert_eq!(table.proof_len(), 1);
        assert_eq!(table.confirmed_len(), 0);
        // 0x200 は残る
        assert!(table.lookup(0x200, &h1).is_some());
        // 0x100 の全 hand は見えない
        assert!(table.lookup(0x100, &h1).is_none());
        assert!(table.lookup(0x100, &h2).is_none());
        assert!(table.lookup(0x100, &h3).is_none());
    }

    /// Phase 2a-2: tombstone を含む slot 列で lookup chain が壊れないこと．
    #[test]
    fn test_proven_table_lookup_skips_tombstones() {
        let mut table = ProvenTable::new(64);
        let h1 = [1u8, 0, 0, 0, 0, 0, 0];
        let h2 = [2u8, 0, 0, 0, 0, 0, 0];

        // 同じ pos_key で 2 個挿入 (linear probing で隣接 slot に配置)
        table.insert(0x100, ProvenEntry::new_proof(h1, 0, 15));
        table.insert(0x100, ProvenEntry::new_proof(h2, 0, 17));

        // 先に挿入した h1 を削除 → tombstone が残る
        assert!(table.remove(0x100, &h1));

        // h2 は依然 lookup できなければならない (tombstone を skip して継続)
        assert!(table.lookup(0x100, &h2).is_some(),
            "lookup must skip tombstones and reach h2");

        // iter_pos_key も 1 エントリ (h2) のみ返す
        let count = table.iter_pos_key(0x100).count();
        assert_eq!(count, 1, "iter_pos_key must skip tombstones");
    }

    /// Phase 3 (v0.62.0): `load_factor` の基本動作．
    #[test]
    fn test_proven_table_load_factor() {
        let mut table = ProvenTable::new(64);
        assert_eq!(table.load_factor(), 0.0);

        let h1 = [1u8, 0, 0, 0, 0, 0, 0];
        table.insert(0x1, ProvenEntry::new_proof(h1, 0, 15));
        // 1 / 64 ≈ 0.015625
        assert!((table.load_factor() - 1.0 / 64.0).abs() < 1e-9);

        // 32 個挿入で 32/64 = 0.5 を超える
        for i in 2u64..=32 {
            let h = [i as u8, 0, 0, 0, 0, 0, 0];
            table.insert(i, ProvenEntry::new_proof(h, 0, 15));
        }
        assert!(table.load_factor() >= 0.5);
        assert!(table.load_factor() <= 1.0);
    }

    /// Phase 3: `collect_garbage(only_refutable=true)` で refutable のみ evict．
    /// proof と confirmed は保護される．
    #[test]
    fn test_proven_table_collect_garbage_refutable_only() {
        let mut table = ProvenTable::new(64);
        let h_p = [1u8, 0, 0, 0, 0, 0, 0];
        let h_c = [2u8, 0, 0, 0, 0, 0, 0];
        let h_r1 = [3u8, 0, 0, 0, 0, 0, 0];
        let h_r2 = [4u8, 0, 0, 0, 0, 0, 0];

        // proof / confirmed / refutable disproof を 1 個ずつ挿入
        table.insert(0x10, ProvenEntry::new_proof(h_p, 0, 15));
        table.insert(0x20, ProvenEntry::new_disproof(h_c, 30));
        let r1 = ProvenEntry {
            hand: h_r1,
            flags: ProvenEntry::encode_refutable_disproof_flags(10),
            best_move: 0,
            meta: 0,
        };
        let r2 = ProvenEntry {
            hand: h_r2,
            flags: ProvenEntry::encode_refutable_disproof_flags(20),
            best_move: 0,
            meta: 0,
        };
        table.insert(0x30, r1);
        table.insert(0x40, r2);
        assert_eq!(table.len(), 4);
        assert_eq!(table.proof_len(), 1);
        assert_eq!(table.confirmed_len(), 1);
        assert_eq!(table.refutable_len(), 2);

        // ratio=1.0 で refutable のみ全て evict (proof/confirmed は保護)
        let removed = table.collect_garbage(1.0, true);
        assert!(removed > 0, "should evict at least one refutable");
        assert_eq!(table.proof_len(), 1, "proof は保護");
        assert_eq!(table.confirmed_len(), 1, "confirmed は保護");
        // proof + confirmed = 2 が残る
        assert_eq!(table.len(), 2);
        assert!(table.lookup(0x10, &h_p).is_some(), "proof は残る");
        assert!(table.lookup(0x20, &h_c).is_some(), "confirmed は残る");
    }

    /// Phase 3: `collect_garbage(only_refutable=false)` で全 entry を amount 順 evict．
    /// `removal_ratio=0.0` なら no-op．
    #[test]
    fn test_proven_table_collect_garbage_full() {
        let mut table = ProvenTable::new(64);
        let h_p = [1u8, 0, 0, 0, 0, 0, 0];
        let h_c = [2u8, 0, 0, 0, 0, 0, 0];

        table.insert(0x10, ProvenEntry::new_proof(h_p, 0, 15));
        table.insert(0x20, ProvenEntry::new_disproof(h_c, 30));
        assert_eq!(table.len(), 2);

        // removal_ratio=0.0: 何も削除されない
        let removed = table.collect_garbage(0.0, false);
        assert_eq!(removed, 0);
        assert_eq!(table.len(), 2);

        // removal_ratio=1.0 & only_refutable=false: 全 entry が threshold 以下になり全 evict．
        let removed = table.collect_garbage(1.0, false);
        assert_eq!(removed, 2, "all entries should be evicted at ratio=1.0");
        assert_eq!(table.len(), 0);
        assert_eq!(table.proof_len(), 0);
        assert_eq!(table.confirmed_len(), 0);
    }

    /// Phase 3: GC 後の tombstone slot に再 insert ができる．
    #[test]
    fn test_proven_table_collect_garbage_then_reinsert() {
        let mut table = ProvenTable::new(64);
        let h = [1u8, 0, 0, 0, 0, 0, 0];
        let r1 = ProvenEntry {
            hand: h,
            flags: ProvenEntry::encode_refutable_disproof_flags(10),
            best_move: 0,
            meta: 0,
        };
        table.insert(0x10, r1);
        assert_eq!(table.len(), 1);

        let removed = table.collect_garbage(1.0, true);
        assert_eq!(removed, 1);
        assert_eq!(table.len(), 0);

        // tombstone slot に新エントリを再挿入
        table.insert(0x10, ProvenEntry::new_proof(h, 0, 17));
        assert_eq!(table.len(), 1);
        assert_eq!(table.proof_len(), 1);
        assert!(table.lookup(0x10, &h).is_some());
    }
}
