//! KH `repetition_table.hpp` 相当の千日手メモ (RepetitionMemo)．
//!
//! 経路ハッシュ ([`super::path_key`]) をキーに「この経路は千日手で，少なくとも `len` 手では
//! 詰まない」を記録する open-addressed (線形走査) 置換表．KH はこれにより千日手を全探索で
//! **キャッシュ**して再展開を防ぐ (maou が `scope_disproof` で再探索している breadth の代替)．
//!
//! soundness 注: false positive (hash 衝突で非千日手を千日手判定) は理論上 false-NoMate に
//! つながるが，64-bit path key の衝突は天文学的稀 (KH と同じ前提)．`len` gating
//! (`stored_len >= query_len`) で更に安全側に倒す．insert の取りこぼしは reuse 減 (breadth) のみで
//! 不健全化しない．
//!
//! GC は簡略化: 固定サイズ + solve ごと `clear`．満杯時の insert は skip (KH の generation GC は
//! 効果確認後に移植)．

/// 1 エントリ (key=0 は空)．KH `TableEntry`．
#[derive(Clone, Copy)]
struct RepEntry {
    /// 経路ハッシュ値 (0 = 空)．
    key: u64,
    /// 千日手判定開始深さ (ply)．
    depth: u32,
    /// 不詰手数 (この経路は少なくとも len 手では詰まない)．
    len: u16,
}

const EMPTY_KEY: u64 = 0;

/// 千日手メモ本体．`entries.len()` は 2 の冪．
pub(super) struct RepetitionMemo {
    entries: Vec<RepEntry>,
}

impl RepetitionMemo {
    /// `capacity_pow2` 個 (2 の冪に切り上げ) のエントリで生成する．
    pub(super) fn new(capacity_pow2: usize) -> Self {
        let size = capacity_pow2.max(1).next_power_of_two();
        Self {
            entries: vec![
                RepEntry {
                    key: EMPTY_KEY,
                    depth: 0,
                    len: 0,
                };
                size
            ],
        }
    }

    /// 全エントリを空にする (solve ごとに呼ぶ)．
    pub(super) fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            e.key = EMPTY_KEY;
        }
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
    pub(super) fn insert(&mut self, path_key: u64, depth: u32, len: u16) {
        if path_key == EMPTY_KEY {
            return;
        }
        let start = self.start_index(path_key);
        let mut index = start;
        // 空 or 同キーまで線形走査．満杯 (一周) なら skip．
        loop {
            let e = self.entries[index];
            if e.key == EMPTY_KEY || e.key == path_key {
                break;
            }
            index = self.next(index);
            if index == start {
                return; // 満杯: insert skip (健全性に影響なし)
            }
        }

        let e = &mut self.entries[index];
        if e.key == EMPTY_KEY {
            *e = RepEntry {
                key: path_key,
                depth,
                len,
            };
        } else if len != e.len {
            // 異なる手数 → 上書き (KH と同じ)．
            e.depth = depth;
            e.len = len;
        } else if e.depth <= depth {
            e.depth = depth;
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
                return None;
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
}
