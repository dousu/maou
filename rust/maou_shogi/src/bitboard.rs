use crate::types::Square;

/// 将棋盤のビットボード表現．
///
/// 81マスを2つのu64で表現する:
/// - lo: マス0-62 (63ビット)
/// - hi: マス63-80 (18ビット)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard {
    pub lo: u64,
    pub hi: u64,
}

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard { lo: 0, hi: 0 };

    /// 全マスがセットされたビットボード．
    pub const ALL: Bitboard = Bitboard {
        lo: (1u64 << 63) - 1,
        hi: (1u64 << 18) - 1,
    };

    /// 指定マスのみセットされたビットボードを返す．
    #[inline]
    pub fn from_square(sq: Square) -> Bitboard {
        let idx = sq.0 as u64;
        if idx < 63 {
            Bitboard {
                lo: 1u64 << idx,
                hi: 0,
            }
        } else {
            Bitboard {
                lo: 0,
                hi: 1u64 << (idx - 63),
            }
        }
    }

    /// 空かどうか．
    #[inline]
    pub fn is_empty(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    /// 空でないかどうか．
    #[inline]
    pub fn is_not_empty(self) -> bool {
        !self.is_empty()
    }

    /// 指定マスがセットされているか．
    #[inline]
    pub fn contains(self, sq: Square) -> bool {
        (self & Bitboard::from_square(sq)).is_not_empty()
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

    /// 最下位セットビットのマスを返し，そのビットをクリアする．
    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
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
    pub fn file_mask(col: u8) -> Bitboard {
        debug_assert!(col < 9);
        let start = col as u64 * 9;
        let mut bb = Bitboard::EMPTY;
        for row in 0..9u64 {
            let idx = start + row;
            if idx < 63 {
                bb.lo |= 1u64 << idx;
            } else {
                bb.hi |= 1u64 << (idx - 63);
            }
        }
        bb
    }

    /// 指定した段(row)の全マスがセットされたビットボードを返す．
    pub fn rank_mask(row: u8) -> Bitboard {
        debug_assert!(row < 9);
        let mut bb = Bitboard::EMPTY;
        for col in 0..9u8 {
            bb.set(Square::new(col, row));
        }
        bb
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
            lo: !self.lo & ((1u64 << 63) - 1),
            hi: !self.hi & ((1u64 << 18) - 1),
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
