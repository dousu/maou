use crate::types::{Color, HAND_KINDS, MAX_HAND_STATES, PIECE_TYPES_NUM, PieceType, Square};

/// Zobristテーブルの駒ID数(14駒種 × 2色 = 28)．
const ZOBRIST_BOARD_IDS: usize = PIECE_TYPES_NUM * 2;

/// Zobrist hashテーブル．
///
/// 疑似乱数で初期化する．cshogi互換のテーブルは使用せず，
/// 独自のテーブルを用いる(テスト時にhash値の一致は検証しない)．
pub struct ZobristTable {
    /// 盤上の駒: [piece_id][square]
    /// piece_id: color * PIECE_TYPES_NUM + piece_type - 1 (0..ZOBRIST_BOARD_IDS)
    board: [[u64; 81]; ZOBRIST_BOARD_IDS],
    /// 持ち駒: [color][piece_kind][count]
    /// piece_kind: 0..HAND_KINDS (歩,香,桂,銀,金,角,飛)
    hand: [[[u64; MAX_HAND_STATES]; HAND_KINDS]; 2],
    /// 手番(後手のときにXOR)
    turn: u64,
}

impl ZobristTable {
    /// テーブルを初期化する．
    pub fn new() -> ZobristTable {
        let mut rng = XorShift64(0x5D58_8B65_6C07_8965);
        let mut table = ZobristTable {
            board: [[0u64; 81]; ZOBRIST_BOARD_IDS],
            hand: [[[0u64; MAX_HAND_STATES]; HAND_KINDS]; 2],
            turn: rng.next(),
        };

        for piece_idx in 0..ZOBRIST_BOARD_IDS {
            for sq in 0..81 {
                table.board[piece_idx][sq] = rng.next();
            }
        }

        for color in 0..2 {
            for kind in 0..HAND_KINDS {
                for count in 0..MAX_HAND_STATES {
                    table.hand[color][kind][count] = rng.next();
                }
            }
        }

        table
    }

    /// 盤上の駒のハッシュ値を取得する．
    #[inline]
    pub fn board_hash(&self, color: Color, piece_type: PieceType, sq: Square) -> u64 {
        let piece_idx = color.index() * PIECE_TYPES_NUM + (piece_type as usize - 1);
        self.board[piece_idx][sq.index()]
    }

    /// 持ち駒のハッシュ値を取得する．
    #[inline]
    pub fn hand_hash(&self, color: Color, hand_index: usize, count: usize) -> u64 {
        assert!(hand_index < HAND_KINDS, "hand_index {} out of range", hand_index);
        assert!(count < MAX_HAND_STATES, "hand count {} out of range", count);
        self.hand[color.index()][hand_index][count]
    }

    /// 手番のハッシュ値を取得する．
    #[inline]
    pub fn turn_hash(&self) -> u64 {
        self.turn
    }
}

/// 簡易な疑似乱数生成器．
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// グローバルなZobristテーブル(遅延初期化)．
use std::sync::LazyLock;

pub static ZOBRIST: LazyLock<ZobristTable> = LazyLock::new(ZobristTable::new);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_table() {
        let table = ZobristTable::new();
        // 異なるマスの同じ駒で異なるハッシュ値
        let h1 = table.board_hash(Color::Black, PieceType::Pawn, Square(0));
        let h2 = table.board_hash(Color::Black, PieceType::Pawn, Square(1));
        assert_ne!(h1, h2);
        assert_ne!(h1, 0);
    }
}
