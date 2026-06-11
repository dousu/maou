//! KH `repetition_table.hpp` 相当の千日手メモ (RepetitionMemo)．
//!
//! 経路ハッシュ ([`super::path_key`]) をキーに「この経路は千日手で，少なくとも `len` 手では
//! 詰まない」を記録する open-addressed (線形走査) 置換表．KH はこれにより千日手を全探索で
//! **キャッシュ**して再展開を防ぐ (maou が `scope_disproof` で再探索している breadth の代替)．
//!
//! soundness 注: false positive (hash 衝突で非千日手を千日手判定) は理論上 false-NoMate に
//! つながるが，64-bit path key の衝突は天文学的稀 (KH と同じ前提)．`len` gating
//! (`stored_len >= query_len`) で更に安全側に倒す．insert の取りこぼし・GC による削除は
//! reuse 減 (breadth) のみで不健全化しない．
//!
//! ## Generation GC (KH parity)
//!
//! KH と同じく **テーブルの高々 ~30% しか要素を格納しない**．`size/20` insert ごとに
//! generation を進め，3 generation ごとに GC で古い generation のエントリを削除 +
//! コンパクションする (`kGcKeepGeneration=3` → live ≈ (3+3)/20 = 30%)．
//! これにより線形走査の probe 長が常に短く保たれる．
//!
//! 旧実装 (固定 1<<16 + 満杯時 insert skip, GC なし) は長時間探索で飽和し，
//! **contains/insert が毎回テーブル全周 (65K slot = 1MB) を走査して NPS が崩壊する**
//! 性能バグがあった (39te 後半 5K nps の真因; 2026-06-11 gdb sampling で確定)．

/// 1 エントリ (key=0 は空)．KH `TableEntry` (16 bytes に詰める)．
#[derive(Clone, Copy)]
struct RepEntry {
    /// 経路ハッシュ値 (0 = 空)．
    key: u64,
    /// 千日手判定開始深さ (ply)．
    depth: u32,
    /// 不詰手数 (この経路は少なくとも len 手では詰まない)．
    len: u16,
    /// 置換表世代 (KH `Generation`)．GC で古い世代から削除する．
    generation: u16,
}

const EMPTY_KEY: u64 = 0;

/// 置換表全体を何 generation で管理するか (KH `kGenerationPerTableSize`)．
const GENERATION_PER_TABLE_SIZE: u64 = 20;
/// 初回の GC タイミング (KH `kInitialGcDuration`)．
const INITIAL_GC_DURATION: u16 = 6;
/// 2 回目以降の GC タイミング (KH `kGcDuration`)．
const GC_DURATION: u16 = 3;
/// GC で残す置換表世代数 (KH `kGcKeepGeneration`)．
const GC_KEEP_GENERATION: u16 = 3;

/// 千日手メモ本体．`entries.len()` は 2 の冪．
pub(super) struct RepetitionMemo {
    entries: Vec<RepEntry>,
    /// 現在の置換表世代．
    generation: u16,
    /// 現在までに insert した新規エントリ数．
    entry_count: u64,
    /// 次回 generation をインクリメントする entry_count．
    next_generation_update: u64,
    /// 次回 GC を行う generation．
    next_gc: u16,
    /// 1 generation あたりのエントリ数 (= size / 20)．
    entries_per_generation: u64,
}

impl RepetitionMemo {
    /// `capacity_pow2` 個 (2 の冪に切り上げ) のエントリで生成する．
    pub(super) fn new(capacity_pow2: usize) -> Self {
        let size = capacity_pow2.max(1).next_power_of_two();
        let mut memo = Self {
            entries: vec![
                RepEntry {
                    key: EMPTY_KEY,
                    depth: 0,
                    len: 0,
                    generation: 0,
                };
                size
            ],
            generation: 0,
            entry_count: 0,
            next_generation_update: 0,
            next_gc: 0,
            entries_per_generation: ((size as u64) / GENERATION_PER_TABLE_SIZE).max(1),
        };
        memo.reset_counters();
        memo
    }

    /// 全エントリを空にする (solve ごとに呼ぶ)．KH `Clear`．
    pub(super) fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            e.key = EMPTY_KEY;
        }
        self.reset_counters();
    }

    fn reset_counters(&mut self) {
        self.generation = 0;
        self.entry_count = 0;
        self.next_generation_update = self.entries_per_generation;
        self.next_gc = INITIAL_GC_DURATION;
    }

    /// Stockfish 流 multiply-shift で開始 index を求める (KH `StartIndex`)．
    #[inline]
    fn start_index(&self, path_key: u64) -> usize {
        let key_low = path_key & 0xffff_ffff;
        ((key_low.wrapping_mul(self.entries.len() as u64)) >> 32) as usize
    }

    #[inline]
    fn next(&self, index: usize) -> usize {
        if index + 1 >= self.entries.len() {
            0
        } else {
            index + 1
        }
    }

    /// 経路ハッシュ `path_key` に (depth, len) を記録する (KH `Insert`)．
    ///
    /// GC により occupancy は ~30% に保たれるため probe 長は常に短い．
    pub(super) fn insert(&mut self, path_key: u64, depth: u32, len: u16) {
        if path_key == EMPTY_KEY {
            return;
        }
        let mut index = self.start_index(path_key);
        loop {
            let e = self.entries[index];
            if e.key == EMPTY_KEY || e.key == path_key {
                break;
            }
            index = self.next(index);
        }

        let generation = self.generation;
        let e = &mut self.entries[index];
        if e.key == EMPTY_KEY {
            *e = RepEntry {
                key: path_key,
                depth,
                len,
                generation,
            };
            self.entry_count += 1;
            if self.entry_count >= self.next_generation_update {
                self.generation = self.generation.wrapping_add(1);
                self.next_generation_update = self.entry_count + self.entries_per_generation;
                if self.generation == self.next_gc {
                    self.collect_garbage();
                    self.next_gc = self.generation.wrapping_add(GC_DURATION);
                }
            }
        } else if len != e.len {
            // 異なる手数 → 上書き (KH と同じ)．
            e.depth = depth;
            e.len = len;
            e.generation = generation;
        } else if e.depth <= depth {
            e.depth = depth;
            e.generation = generation;
        }
    }

    /// `path_key` が記録されていて `stored_len >= len` なら (depth, stored_len) を返す (KH `Contains`)．
    pub(super) fn contains(&self, path_key: u64, len: u16) -> Option<(u32, u16)> {
        if path_key == EMPTY_KEY {
            return None;
        }
        let start = self.start_index(path_key);
        let mut index = start;
        loop {
            let e = self.entries[index];
            if e.key == EMPTY_KEY {
                return None;
            }
            if e.key == path_key && e.len >= len {
                return Some((e.depth, e.len));
            }
            index = self.next(index);
            if index == start {
                // GC により満杯は起きないはずだが，万一の無限ループ保険．
                return None;
            }
        }
    }

    /// ガベージコレクション (KH `CollectGarbage`)．
    ///
    /// 現在の世代から `GC_KEEP_GENERATION` 世代前までを残し，それより古いエントリを削除する．
    /// 歯抜けがあると線形走査が途切れるため，残ったエントリをできるだけ手前に詰める
    /// (コンパクション)．
    fn collect_garbage(&mut self) {
        let generation = self.generation;
        let erased_generation = generation.wrapping_sub(GC_KEEP_GENERATION);

        // [erased_generation, generation] の範囲外なら削除．u16 wrap に注意 (KH と同じ)．
        let should_erase = |entry_generation: u16| -> bool {
            if erased_generation < generation {
                entry_generation < erased_generation || generation < entry_generation
            } else {
                generation < entry_generation && entry_generation < erased_generation
            }
        };

        for e in self.entries.iter_mut() {
            if e.key != EMPTY_KEY && should_erase(e.generation) {
                e.key = EMPTY_KEY;
            }
        }

        // コンパクション．配列終端付近の歯抜けでアクセス不能になるエントリが微小数
        // 出得るが目をつぶる (KH と同じ)．
        for i in 0..self.entries.len() {
            if self.entries[i].key == EMPTY_KEY {
                continue;
            }
            let entry = self.entries[i];
            let mut index = self.start_index(entry.key);
            while index != i {
                if self.entries[index].key == EMPTY_KEY {
                    self.entries[index] = entry;
                    self.entries[i].key = EMPTY_KEY;
                    break;
                }
                index = self.next(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_contains() {
        let mut m = RepetitionMemo::new(64);
        m.insert(0xABCD_1234_5678_9F01, 7, 10);
        assert_eq!(m.contains(0xABCD_1234_5678_9F01, 10), Some((7, 10)));
        // stored_len(10) >= query(5) → hit
        assert_eq!(m.contains(0xABCD_1234_5678_9F01, 5), Some((7, 10)));
    }

    #[test]
    fn contains_respects_len_gating() {
        let mut m = RepetitionMemo::new(64);
        m.insert(0xDEAD_BEEF_0000_0001, 3, 8);
        // query len(12) > stored_len(8) → miss (詰みまでもっとかかるので適用不可)
        assert_eq!(m.contains(0xDEAD_BEEF_0000_0001, 12), None);
    }

    #[test]
    fn missing_key_returns_none() {
        let mut m = RepetitionMemo::new(64);
        m.insert(0x1111_1111_1111_1111, 1, 4);
        assert_eq!(m.contains(0x2222_2222_2222_2222, 1), None);
    }

    #[test]
    fn clear_empties() {
        let mut m = RepetitionMemo::new(64);
        m.insert(0x55, 2, 6);
        m.clear();
        assert_eq!(m.contains(0x55, 1), None);
    }

    #[test]
    fn collision_via_linear_probe() {
        // 小さい table で複数キーを入れても線形走査で取り出せる．
        let mut m = RepetitionMemo::new(8);
        let keys = [0xA1, 0xB2, 0xC3, 0xD4, 0xE5];
        for (i, &k) in keys.iter().enumerate() {
            m.insert(k, i as u32, 9);
        }
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(m.contains(k, 9), Some((i as u32, 9)), "key {k:#x}");
        }
    }

    #[test]
    fn gc_evicts_old_generations_and_keeps_recent() {
        // size=64 → entries_per_generation = 64/20 = 3．
        // generation は 3 insert ごとに進み，generation 6 で初回 GC →
        // keep 3 世代 (gen 4,5,6) = 直近 ~9-12 エントリのみ残る．
        let mut m = RepetitionMemo::new(64);
        let n = 30u64;
        for i in 0..n {
            // start_index 衝突を避けるため上位 32bit にも分散させる
            let key = (i + 1) | ((i + 1) << 32);
            m.insert(key, i as u32, 5);
        }
        let live = (1..=n).filter(|i| {
            let key = i | (i << 32);
            m.contains(key, 5).is_some()
        }).count();
        // 古い世代は GC で削除され，全件は残っていない．直近世代は残っている．
        assert!(live < n as usize, "GC should evict old generations (live={live})");
        let last_key = n | (n << 32);
        assert_eq!(m.contains(last_key, 5), Some(((n - 1) as u32, 5)));
    }

    #[test]
    fn saturation_does_not_full_scan() {
        // 旧実装の性能バグ回帰テスト: 容量を大きく超える insert をしても
        // GC により occupancy ~30% が保たれ，contains が短い probe で返る
        // (飽和 = 全周走査だと本テストが現実時間で終わらない規模にする)．
        let size = 1 << 12;
        let mut m = RepetitionMemo::new(size);
        for i in 0..(size as u64 * 4) {
            let key = (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            m.insert(key, 1, 3);
        }
        // 空きスロットが必ず存在する (occupancy < 100%)．
        let occupied = m.entries.iter().filter(|e| e.key != EMPTY_KEY).count();
        assert!(
            occupied <= size * 4 / 10,
            "GC should cap occupancy (~30%), got {occupied}/{size}"
        );
        // miss 側 lookup も EMPTY 終端で即返る (全周しない)．
        assert_eq!(m.contains(0xFFFF_FFFF_FFFF_FFFF, 3), None);
    }
}
