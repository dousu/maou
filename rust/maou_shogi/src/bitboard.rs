use crate::types::Square;

/// lo部(マス0-62)の有効ビットマスク．63ビット．
const LO_MASK: u64 = (1u64 << 63) - 1;
/// hi部(マス63-80)の有効ビットマスク．18ビット．
const HI_MASK: u64 = (1u64 << 18) - 1;

/// 将棋盤のビットボード表現．
///
/// 81マスを2つのu64で表現する:
/// - lo: マス0-62 (63ビット)
/// - hi: マス63-80 (18ビット)
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
}

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard { lo: 0, hi: 0 };

    /// 全マスがセットされたビットボード．
    pub const ALL: Bitboard = Bitboard {
        lo: LO_MASK,
        hi: HI_MASK,
    };

    /// 指定マスのみセットされたビットボードを返す．
    ///
    /// 事前計算済みルックアップテーブルを使用し，分岐を排除する．
    #[inline]
    pub fn from_square(sq: Square) -> Bitboard {
        SQUARE_BB[sq.0 as usize]
    }

    /// 空かどうか．
    #[inline]
    pub fn is_empty(self) -> bool {
        (self.lo | self.hi) == 0
    }

    /// 空でないかどうか．
    #[inline]
    pub fn is_not_empty(self) -> bool {
        (self.lo | self.hi) != 0
    }

    /// 指定マスがセットされているか．
    #[inline]
    pub fn contains(self, sq: Square) -> bool {
        let idx = sq.0 as u64;
        if idx < 63 {
            (self.lo >> idx) & 1 != 0
        } else {
            (self.hi >> (idx - 63)) & 1 != 0
        }
    }

    /// マスをセットする．
    #[inline]
    pub fn set(&mut self, sq: Square) {
        let idx = sq.0 as u64;
        if idx < 63 {
            self.lo |= 1u64 << idx;
        } else {
            self.hi |= 1u64 << (idx - 63);
        }
    }

    /// マスをクリアする．
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        let idx = sq.0 as u64;
        if idx < 63 {
            self.lo &= !(1u64 << idx);
        } else {
            self.hi &= !(1u64 << (idx - 63));
        }
    }

    /// セットされたビット数を返す．
    #[inline]
    pub fn count(self) -> u32 {
        self.lo.count_ones() + self.hi.count_ones()
    }

    /// 最下位セットビットのマスを返す(非破壊)．空の場合は None．
    ///
    /// lo はマス 0..63，hi はマス 63..80 に対応するため，
    /// hi のビット位置には +63 のオフセットを加える．
    #[inline]
    pub fn lsb(self) -> Option<Square> {
        if self.lo != 0 {
            let idx = self.lo.trailing_zeros();
            debug_assert!(idx < 63, "lo bit index {} out of range", idx);
            Some(Square(idx as u8))
        } else if self.hi != 0 {
            // hi のビット位置 0 がマス 63 に対応
            let idx = self.hi.trailing_zeros() + 63;
            debug_assert!(idx <= 80, "hi bit index {} out of square range", idx);
            Some(Square(idx as u8))
        } else {
            None
        }
    }

    /// 最上位セットビットのマスを返す(非破壊)．空の場合は None．
    ///
    /// lo はマス 0..63，hi はマス 63..80 に対応するため，
    /// hi のビット位置には +63 のオフセットを加える．
    #[inline]
    pub fn msb(self) -> Option<Square> {
        if self.hi != 0 {
            // hi のビット位置 0 がマス 63 に対応するため +63
            let bit = 63 - self.hi.leading_zeros();
            let idx = bit + 63;
            debug_assert!(idx <= 80, "hi msb index {} out of square range", idx);
            Some(Square(idx as u8))
        } else if self.lo != 0 {
            let bit = 63 - self.lo.leading_zeros();
            debug_assert!(bit < 63, "lo msb index {} out of range", bit);
            Some(Square(bit as u8))
        } else {
            None
        }
    }

    /// 最下位セットビットのマスを返し，そのビットをクリアする．
    ///
    /// # Panics (debug)
    ///
    /// 空のビットボードに対して呼び出すと `debug_assert` で停止する．
    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        debug_assert!(!self.is_empty(), "pop_lsb called on empty Bitboard");
        if self.lo != 0 {
            let idx = self.lo.trailing_zeros();
            self.lo &= self.lo - 1;
            Square(idx as u8)
        } else {
            let idx = self.hi.trailing_zeros() + 63;
            self.hi &= self.hi - 1;
            Square(idx as u8)
        }
    }

    /// 指定した筋(col)の全マスがセットされたビットボードを返す．
    #[inline]
    pub fn file_mask(col: u8) -> Bitboard {
        debug_assert!(col < 9);
        FILE_MASKS[col as usize]
    }

    /// 指定した段(row)の全マスがセットされたビットボードを返す．
    #[inline]
    pub fn rank_mask(row: u8) -> Bitboard {
        debug_assert!(row < 9);
        RANK_MASKS[row as usize]
    }

    /// ビットが立っている筋すべてをマスクしたビットボードを返す．
    ///
    /// 例: 歩のビットボードを渡すと，歩が存在する筋全体がセットされた
    /// ビットボードが返る(二歩チェックに使用)．
    #[inline]
    pub fn occupied_files(self) -> Bitboard {
        // FILE_MASKS (lo=マス0-62 / hi=マス63-80 の正規約で構築済み) を
        // 単一真実として使う．旧実装は col 7 (8筋) を「lo bit63 + hi bits0-7」
        // という誤った分割で見ており，マス71 (8i) が二歩マスクから漏れていた．
        let mut result = Bitboard::EMPTY;
        let mut col = 0usize;
        while col < 9 {
            let mask = FILE_MASKS[col];
            if (self.lo & mask.lo) | (self.hi & mask.hi) != 0 {
                result.lo |= mask.lo;
                result.hi |= mask.hi;
            }
            col += 1;
        }
        result
    }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard {
            lo: self.lo & rhs.lo,
            hi: self.hi & rhs.hi,
        }
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard {
            lo: self.lo | rhs.lo,
            hi: self.hi | rhs.hi,
        }
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl std::ops::Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        // 81ビットのみ有効にする
        Bitboard {
            lo: !self.lo & (LO_MASK),
            hi: !self.hi & (HI_MASK),
        }
    }
}

impl std::ops::BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.lo &= rhs.lo;
        self.hi &= rhs.hi;
    }
}

impl std::ops::BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.lo |= rhs.lo;
        self.hi |= rhs.hi;
    }
}

/// Bitboardのイテレータ．セットされている各マスを順に返す．
pub struct BitboardIter {
    bb: Bitboard,
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    fn into_iter(self) -> BitboardIter {
        BitboardIter { bb: self }
    }
}

impl Iterator for BitboardIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.bb.is_empty() {
            None
        } else {
            Some(self.bb.pop_lsb())
        }
    }
}

// ============================================================
// 事前計算済みマスクテーブル
// ============================================================

/// 各マスに対応するビットボード．`from_square(sq)` で使用．
const SQUARE_BB: [Bitboard; 81] = {
    let mut table = [Bitboard::EMPTY; 81];
    let mut i = 0u8;
    while i < 81 {
        let idx = i as u64;
        if idx < 63 {
            table[i as usize] = Bitboard {
                lo: 1u64 << idx,
                hi: 0,
            };
        } else {
            table[i as usize] = Bitboard {
                lo: 0,
                hi: 1u64 << (idx - 63),
            };
        }
        i += 1;
    }
    table
};

/// 筋(col)ごとのビットボードマスク．`file_mask(col)` で使用．
const FILE_MASKS: [Bitboard; 9] = {
    let mut masks = [Bitboard::EMPTY; 9];
    let mut col = 0u8;
    while col < 9 {
        let start = col as u64 * 9;
        let mut lo = 0u64;
        let mut hi = 0u64;
        let mut row = 0u64;
        while row < 9 {
            let idx = start + row;
            if idx < 63 {
                lo |= 1u64 << idx;
            } else {
                hi |= 1u64 << (idx - 63);
            }
            row += 1;
        }
        masks[col as usize] = Bitboard { lo, hi };
        col += 1;
    }
    masks
};

/// 段(row)ごとのビットボードマスク．`rank_mask(row)` で使用．
const RANK_MASKS: [Bitboard; 9] = {
    let mut masks = [Bitboard::EMPTY; 9];
    let mut row = 0u8;
    while row < 9 {
        let mut lo = 0u64;
        let mut hi = 0u64;
        let mut col = 0u8;
        while col < 9 {
            let idx = col as u64 * 9 + row as u64;
            if idx < 63 {
                lo |= 1u64 << idx;
            } else {
                hi |= 1u64 << (idx - 63);
            }
            col += 1;
        }
        masks[row as usize] = Bitboard { lo, hi };
        row += 1;
    }
    masks
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_square() {
        let sq = Square(0);
        let bb = Bitboard::from_square(sq);
        assert!(bb.contains(sq));
        assert_eq!(bb.count(), 1);

        let sq = Square(80);
        let bb = Bitboard::from_square(sq);
        assert!(bb.contains(sq));
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn test_from_square_all() {
        // 全81マスでルックアップテーブルが正しいことを検証
        for i in 0..81u8 {
            let sq = Square(i);
            let bb = Bitboard::from_square(sq);
            assert!(bb.contains(sq), "from_square failed for sq={}", i);
            assert_eq!(bb.count(), 1, "count != 1 for sq={}", i);
        }
    }

    #[test]
    fn test_set_clear() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square(10));
        bb.set(Square(70));
        assert!(bb.contains(Square(10)));
        assert!(bb.contains(Square(70)));
        assert_eq!(bb.count(), 2);

        bb.clear(Square(10));
        assert!(!bb.contains(Square(10)));
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn test_file_mask() {
        let fm = Bitboard::file_mask(0);
        assert_eq!(fm.count(), 9);
        for row in 0..9 {
            assert!(fm.contains(Square::new(0, row)));
        }
    }

    #[test]
    fn test_occupied_files_all_squares() {
        // 全 81 マスについて，そのマス 1 つだけのビットボードが
        // 該当する筋全体をマークすることを検証する．
        // 回帰: 旧実装は col 7 (8筋) の分割を誤っており，マス 71 (8i) の
        // 歩を見落とし，また 8筋の occupied_files がマス 71 を含まなかった
        // (二歩チェック漏れ → P*8i の非合法生成)．
        for col in 0..9u8 {
            for row in 0..9u8 {
                let mut bb = Bitboard::EMPTY;
                bb.set(Square::new(col, row));
                let files = bb.occupied_files();
                assert_eq!(
                    files,
                    Bitboard::file_mask(col),
                    "occupied_files mismatch for col={col} row={row}"
                );
            }
        }
    }

    #[test]
    fn test_occupied_files_multiple() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::new(0, 4));
        bb.set(Square::new(7, 8)); // マス 71 (8i) — 旧実装の見落とし箇所
        bb.set(Square::new(8, 0));
        let files = bb.occupied_files();
        let expected = Bitboard::file_mask(0) | Bitboard::file_mask(7) | Bitboard::file_mask(8);
        assert_eq!(files, expected);
        assert_eq!(Bitboard::EMPTY.occupied_files(), Bitboard::EMPTY);
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square(5));
        bb.set(Square(70));
        let sq1 = bb.pop_lsb();
        assert_eq!(sq1, Square(5));
        let sq2 = bb.pop_lsb();
        assert_eq!(sq2, Square(70));
        assert!(bb.is_empty());
    }

    #[test]
    fn test_iterator() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square(0));
        bb.set(Square(40));
        bb.set(Square(80));
        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares, vec![Square(0), Square(40), Square(80)]);
    }
}
