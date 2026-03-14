//! HuffmanCodedPos (HCP) — Apery/cshogi互換の32バイト局面シリアライズ．
//!
//! ## フォーマット (256ビット = 32バイト)
//!
//! - ビット0: 手番 (0=先手, 1=後手)
//! - ビット1-7: 先手玉のマス番号 (7ビット, LSBファースト)
//! - ビット8-14: 後手玉のマス番号 (7ビット, LSBファースト)
//! - ビット15以降: 79マスの駒 (玉以外) をHuffman符号化
//! - 残りビット: 持ち駒をHuffman符号化
//!
//! 盤上の各マス(玉以外)について sq=0..80 の順にスキャンし，
//! 玉のマスは飛ばして符号化する．

use crate::board::Board;
use crate::types::{Color, Piece, PieceType, Square};

/// HCPエンコード/デコード時のエラー．
#[derive(Debug, thiserror::Error)]
pub enum HcpError {
    /// ビットライターが256ビットバッファを超過した．
    #[error("HCP bit writer overflow at position {0}")]
    BitWriterOverflow(usize),
    /// ビットリーダーが256ビットバッファを超過した．
    #[error("HCP bit reader overflow at position {0}")]
    BitReaderOverflow(usize),
    /// 無効な持ち駒のHuffman符号．
    #[error("invalid HCP hand piece code: {0}")]
    InvalidHandPieceCode(String),
    /// 無効な玉のマス番号．
    #[error("invalid king square: {0}")]
    InvalidKingSquare(u8),
    /// 玉が見つからない(片玉局面など)．
    #[error("king not found for {0:?}: HCP requires both kings")]
    KingNotFound(Color),
}

/// HCPのバイト数．
pub const HCP_SIZE: usize = 32;

/// HCPバイト列．
pub type Hcp = [u8; HCP_SIZE];

// ============================================================
// Huffman符号テーブル (Apery互換)
//
// 盤上の駒: prefix + color_bit + promotion_bit
//   ただしGoldは成れないため promotion_bit なし
//
// ストリームにはLSBファーストで書き込む．
// ============================================================

/// 盤上駒のHuffmanプレフィックスと各ビット長．
///
/// 構造: prefix (可変長) + color (1ビット) + promoted (1ビット, Goldを除く)
///
/// | 駒種   | prefix (LSB first)  | prefix長 | +color | +promoted | 合計 |
/// |--------|---------------------|----------|--------|-----------|------|
/// | Empty  | 0                   | 1        | -      | -         | 1    |
/// | Pawn   | 1,0                 | 2        | +1     | +1        | 4    |
/// | Lance  | 1,1,0,0             | 4        | +1     | +1        | 6    |
/// | Knight | 1,1,1,0             | 4        | +1     | +1        | 6    |
/// | Silver | 1,1,0,1             | 4        | +1     | +1        | 6    |
/// | Gold   | 1,1,1,1,0           | 5        | +1     | -         | 6    |
/// | Bishop | 1,1,1,1,1,0         | 6        | +1     | +1        | 8    |
/// | Rook   | 1,1,1,1,1,1         | 6        | +1     | +1        | 8    |

/// 持ち駒のHuffman符号 (Apery互換)．
///
/// | 駒種   | prefix (LSB first)  | prefix長 | +color | 合計 |
/// |--------|---------------------|----------|--------|------|
/// | Pawn   | 0,0                 | 2        | +1     | 3    |
/// | Lance  | 1,0,0,0             | 4        | +1     | 5    |
/// | Knight | 1,1,0,0             | 4        | +1     | 5    |
/// | Silver | 1,0,1,0             | 4        | +1     | 5    |
/// | Gold   | 1,1,1,0             | 4        | +1     | 5    |
/// | Bishop | 1,1,1,1,1,0         | 6        | +1     | 7    |
/// | Rook   | 1,1,1,1,1,1         | 6        | +1     | 7    |

// ============================================================
// BitWriter / BitReader
// ============================================================

struct BitWriter {
    buf: [u8; HCP_SIZE],
    pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            buf: [0u8; HCP_SIZE],
            pos: 0,
        }
    }

    /// 1ビット書き込む．
    #[inline]
    fn write_bit(&mut self, bit: u8) -> Result<(), HcpError> {
        if self.pos >= 256 {
            return Err(HcpError::BitWriterOverflow(self.pos));
        }
        if bit != 0 {
            self.buf[self.pos / 8] |= 1 << (self.pos % 8);
        }
        self.pos += 1;
        Ok(())
    }

    /// 値をnビットLSBファーストで書き込む．
    #[inline]
    fn write_bits(&mut self, value: u32, n: usize) -> Result<(), HcpError> {
        for i in 0..n {
            self.write_bit(((value >> i) & 1) as u8)?;
        }
        Ok(())
    }
}

struct BitReader<'a> {
    buf: &'a [u8; HCP_SIZE],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8; HCP_SIZE]) -> Self {
        BitReader { buf, pos: 0 }
    }

    /// 1ビット読み込む．
    ///
    /// 256ビット(32バイト)を超えて読み進めた場合は `Err` を返す．
    #[inline]
    fn read_bit(&mut self) -> Result<u8, HcpError> {
        if self.pos >= 256 {
            return Err(HcpError::BitReaderOverflow(self.pos));
        }
        let bit = (self.buf[self.pos / 8] >> (self.pos % 8)) & 1;
        self.pos += 1;
        Ok(bit)
    }

    /// nビットをLSBファーストで読み込み値として返す．
    #[inline]
    fn read_bits(&mut self, n: usize) -> Result<u32, HcpError> {
        let mut value = 0u32;
        for i in 0..n {
            value |= (self.read_bit()? as u32) << i;
        }
        Ok(value)
    }
}

// ============================================================
// エンコード
// ============================================================

/// 盤上の駒をHuffman符号化してビットストリームに書き込む．
fn encode_board_piece(writer: &mut BitWriter, piece: Piece) -> Result<(), HcpError> {
    if piece.is_empty() {
        writer.write_bit(0)?;
        return Ok(());
    }

    let color = piece.color().unwrap();
    let pt = piece.piece_type().unwrap();
    let color_bit = color as u8;

    // 基本駒種 (成駒→元の駒種に戻す)
    let (base_pt, is_promoted) = if pt.is_promoted() {
        (pt.captured_to_hand(), true)
    } else {
        (pt, false)
    };

    match base_pt {
        PieceType::Pawn => {
            // prefix: 1,0 (2ビット) + color + promoted = 4ビット
            writer.write_bits(0b01, 2)?; // 1,0 in LSB first
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::Lance => {
            // prefix: 1,1,0,0 (4ビット) + color + promoted = 6ビット
            writer.write_bits(0b0011, 4)?;
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::Knight => {
            // prefix: 1,1,1,0 (4ビット) + color + promoted = 6ビット
            writer.write_bits(0b0111, 4)?;
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::Silver => {
            // prefix: 1,1,0,1 (4ビット) + color + promoted = 6ビット
            writer.write_bits(0b1011, 4)?;
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::Gold => {
            // prefix: 1,1,1,1,0 (5ビット) + color = 6ビット (成りなし)
            writer.write_bits(0b01111, 5)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Bishop => {
            // prefix: 1,1,1,1,1,0 (6ビット) + color + promoted = 8ビット
            writer.write_bits(0b011111, 6)?;
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::Rook => {
            // prefix: 1,1,1,1,1,1 (6ビット) + color + promoted = 8ビット
            writer.write_bits(0b111111, 6)?;
            writer.write_bit(color_bit)?;
            writer.write_bit(is_promoted as u8)?;
        }
        PieceType::King
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver
        | PieceType::Horse
        | PieceType::Dragon => {
            unreachable!("unexpected piece type in board encoding: {:?}", base_pt)
        }
    }
    Ok(())
}

/// 持ち駒をHuffman符号化してビットストリームに書き込む．
fn encode_hand_piece(writer: &mut BitWriter, hand_type: PieceType, color: Color) -> Result<(), HcpError> {
    let color_bit = color as u8;

    match hand_type {
        PieceType::Pawn => {
            writer.write_bits(0b00, 2)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Lance => {
            writer.write_bits(0b0001, 4)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Knight => {
            writer.write_bits(0b0011, 4)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Silver => {
            writer.write_bits(0b0101, 4)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Gold => {
            writer.write_bits(0b0111, 4)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Bishop => {
            writer.write_bits(0b011111, 6)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::Rook => {
            writer.write_bits(0b111111, 6)?;
            writer.write_bit(color_bit)?;
        }
        PieceType::King
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver
        | PieceType::Horse
        | PieceType::Dragon => {
            unreachable!("invalid hand piece type: {:?}", hand_type)
        }
    }
    Ok(())
}

/// Board を HCP (32バイト) にエンコードする．
///
/// HCPフォーマットは両玉が存在する標準局面を前提としている．
/// 片玉局面(詰将棋など)はサポート対象外であり，`Err` を返す．
pub fn to_hcp(board: &Board) -> Result<Hcp, HcpError> {
    let mut writer = BitWriter::new();

    // 1. 手番 (1ビット)
    writer.write_bit(board.turn as u8)?;

    // 2. 先手玉のマス番号 (7ビット)
    let bk_sq = find_king(board, Color::Black)?;
    writer.write_bits(bk_sq as u32, 7)?;

    // 3. 後手玉のマス番号 (7ビット)
    let wk_sq = find_king(board, Color::White)?;
    writer.write_bits(wk_sq as u32, 7)?;

    // 4. 盤上の駒 (sq=0..80, 玉は飛ばす)
    for sq in 0..81u8 {
        if sq == bk_sq || sq == wk_sq {
            continue;
        }
        encode_board_piece(&mut writer, board.squares[sq as usize])?;
    }

    // 5. 持ち駒
    // 色別にグループ化: 先手の全持ち駒, 次に後手の全持ち駒
    // 各色内の順序: 歩, 香, 桂, 銀, 金, 角, 飛
    for &color in &[Color::Black, Color::White] {
        for (i, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            let count = board.hand[color.index()][i];
            for _ in 0..count {
                encode_hand_piece(&mut writer, pt, color)?;
            }
        }
    }

    Ok(writer.buf)
}

// ============================================================
// デコード
// ============================================================

/// 盤上の駒をビットストリームからデコードする．
fn decode_board_piece(reader: &mut BitReader) -> Result<Piece, HcpError> {
    // 最初のビット: 0=空, 1=駒あり
    if reader.read_bit()? == 0 {
        return Ok(Piece::EMPTY);
    }

    // 駒あり: プレフィックスを読む
    if reader.read_bit()? == 0 {
        // prefix 1,0 → Pawn
        let color_bit = reader.read_bit()?;
        let promoted = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        let pt = if promoted == 0 {
            PieceType::Pawn
        } else {
            PieceType::ProPawn
        };
        return Ok(Piece::new(color, pt));
    }

    // 1,1,...
    if reader.read_bit()? == 0 {
        // 1,1,0,...
        if reader.read_bit()? == 0 {
            // prefix 1,1,0,0 → Lance
            let color_bit = reader.read_bit()?;
            let promoted = reader.read_bit()?;
            let color = if color_bit == 0 {
                Color::Black
            } else {
                Color::White
            };
            let pt = if promoted == 0 {
                PieceType::Lance
            } else {
                PieceType::ProLance
            };
            return Ok(Piece::new(color, pt));
        } else {
            // prefix 1,1,0,1 → Silver
            let color_bit = reader.read_bit()?;
            let promoted = reader.read_bit()?;
            let color = if color_bit == 0 {
                Color::Black
            } else {
                Color::White
            };
            let pt = if promoted == 0 {
                PieceType::Silver
            } else {
                PieceType::ProSilver
            };
            return Ok(Piece::new(color, pt));
        }
    }

    // 1,1,1,...
    if reader.read_bit()? == 0 {
        // prefix 1,1,1,0 → Knight
        let color_bit = reader.read_bit()?;
        let promoted = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        let pt = if promoted == 0 {
            PieceType::Knight
        } else {
            PieceType::ProKnight
        };
        return Ok(Piece::new(color, pt));
    }

    // 1,1,1,1,...
    if reader.read_bit()? == 0 {
        // prefix 1,1,1,1,0 → Gold
        let color_bit = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        return Ok(Piece::new(color, PieceType::Gold));
    }

    // 1,1,1,1,1,...
    if reader.read_bit()? == 0 {
        // prefix 1,1,1,1,1,0 → Bishop
        let color_bit = reader.read_bit()?;
        let promoted = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        let pt = if promoted == 0 {
            PieceType::Bishop
        } else {
            PieceType::Horse
        };
        return Ok(Piece::new(color, pt));
    }

    // prefix 1,1,1,1,1,1 → Rook
    let color_bit = reader.read_bit()?;
    let promoted = reader.read_bit()?;
    let color = if color_bit == 0 {
        Color::Black
    } else {
        Color::White
    };
    let pt = if promoted == 0 {
        PieceType::Rook
    } else {
        PieceType::Dragon
    };
    Ok(Piece::new(color, pt))
}

/// 持ち駒の1つをビットストリームからデコードする．
/// 戻り値: (駒種, 色)
fn decode_hand_piece(reader: &mut BitReader) -> Result<(PieceType, Color), HcpError> {
    if reader.read_bit()? == 0 {
        // 0,...
        if reader.read_bit()? == 0 {
            // prefix 0,0 → Pawn
            let color_bit = reader.read_bit()?;
            let color = if color_bit == 0 {
                Color::Black
            } else {
                Color::White
            };
            return Ok((PieceType::Pawn, color));
        }
        return Err(HcpError::InvalidHandPieceCode("prefix 0,1 is not defined".to_string()));
    }

    // 1,...
    if reader.read_bit()? == 0 {
        // 1,0,...
        if reader.read_bit()? == 0 {
            // 1,0,0,...
            if reader.read_bit()? == 0 {
                // prefix 1,0,0,0 → Lance
                let color_bit = reader.read_bit()?;
                let color = if color_bit == 0 {
                    Color::Black
                } else {
                    Color::White
                };
                return Ok((PieceType::Lance, color));
            }
            return Err(HcpError::InvalidHandPieceCode("prefix 1,0,0,1 is not defined".to_string()));
        }
        // 1,0,1,...
        if reader.read_bit()? == 0 {
            // prefix 1,0,1,0 → Silver
            let color_bit = reader.read_bit()?;
            let color = if color_bit == 0 {
                Color::Black
            } else {
                Color::White
            };
            return Ok((PieceType::Silver, color));
        }
        return Err(HcpError::InvalidHandPieceCode("prefix 1,0,1,1 is not defined".to_string()));
    }

    // 1,1,...
    if reader.read_bit()? == 0 {
        // 1,1,0,...
        if reader.read_bit()? == 0 {
            // prefix 1,1,0,0 → Knight
            let color_bit = reader.read_bit()?;
            let color = if color_bit == 0 {
                Color::Black
            } else {
                Color::White
            };
            return Ok((PieceType::Knight, color));
        }
        return Err(HcpError::InvalidHandPieceCode("prefix 1,1,0,1 is not defined".to_string()));
    }

    // 1,1,1,...
    if reader.read_bit()? == 0 {
        // 1,1,1,0 → Gold
        let color_bit = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        return Ok((PieceType::Gold, color));
    }

    // 1,1,1,1,...
    if reader.read_bit()? == 0 {
        return Err(HcpError::InvalidHandPieceCode("prefix 1,1,1,1,0 is not defined".to_string()));
    }

    // 1,1,1,1,1,...
    if reader.read_bit()? == 0 {
        // prefix 1,1,1,1,1,0 → Bishop
        let color_bit = reader.read_bit()?;
        let color = if color_bit == 0 {
            Color::Black
        } else {
            Color::White
        };
        return Ok((PieceType::Bishop, color));
    }

    // prefix 1,1,1,1,1,1 → Rook
    let color_bit = reader.read_bit()?;
    let color = if color_bit == 0 {
        Color::Black
    } else {
        Color::White
    };
    Ok((PieceType::Rook, color))
}

/// HCP (32バイト) から Board をデコードする．
pub fn from_hcp(hcp: &Hcp) -> Result<Board, HcpError> {
    let mut reader = BitReader::new(hcp);
    let mut board = Board::empty();

    // 1. 手番
    let turn_bit = reader.read_bit()?;
    board.turn = if turn_bit == 0 {
        Color::Black
    } else {
        Color::White
    };

    // 2. 先手玉
    let bk_sq = reader.read_bits(7)? as u8;
    if bk_sq >= 81 {
        return Err(HcpError::InvalidKingSquare(bk_sq));
    }

    // 3. 後手玉
    let wk_sq = reader.read_bits(7)? as u8;
    if wk_sq >= 81 {
        return Err(HcpError::InvalidKingSquare(wk_sq));
    }

    // 玉を配置
    board.put_piece(Square(bk_sq), Piece::new(Color::Black, PieceType::King));
    board.put_piece(Square(wk_sq), Piece::new(Color::White, PieceType::King));

    // 4. 盤上の駒
    for sq in 0..81u8 {
        if sq == bk_sq || sq == wk_sq {
            continue;
        }
        let piece = decode_board_piece(&mut reader)?;
        if !piece.is_empty() {
            board.put_piece(Square(sq), piece);
        }
    }

    // 5. 持ち駒
    // 盤上にない駒数の合計分だけHuffman符号を読む．
    // 各符号は駒種と色を自己識別するため，読み出し順序は問わない．
    let mut total_remaining = 0usize;
    for (i, &_pt) in PieceType::HAND_PIECES.iter().enumerate() {
        let max_count = PieceType::MAX_HAND_COUNT[i] as usize;
        let on_board = count_on_board(&board, PieceType::HAND_PIECES[i]);
        let remaining = max_count.checked_sub(on_board).ok_or_else(|| {
            HcpError::InvalidHandPieceCode(format!(
                "too many {:?} on board: {} exceeds max {}",
                PieceType::HAND_PIECES[i],
                on_board,
                max_count
            ))
        })?;
        total_remaining += remaining;
    }

    for _ in 0..total_remaining {
        let (decoded_pt, color) = decode_hand_piece(&mut reader)?;
        let hand_idx = decoded_pt
            .hand_index()
            .expect("decoded non-hand piece type from hand section");
        board.hand[color.index()][hand_idx] += 1;
    }

    // Zobrist hash を計算
    board.hash = board.compute_hash();

    Ok(board)
}

// ============================================================
// ヘルパー関数
// ============================================================

/// 盤上の指定した基本駒種(成駒含む)の個数を数える．
///
/// Bitboard の popcount を利用して O(1) で計算する．
fn count_on_board(board: &Board, base_pt: PieceType) -> usize {
    let mut count = 0u32;
    for color_idx in 0..2 {
        count += board.piece_bb[color_idx][base_pt as usize].count();
        if let Some(promoted) = base_pt.promoted() {
            count += board.piece_bb[color_idx][promoted as usize].count();
        }
    }
    count as usize
}

/// 指定した色の玉のマス番号を返す．
///
/// 玉が見つからない場合は `Err` を返す(片玉局面など)．
fn find_king(board: &Board, color: Color) -> Result<u8, HcpError> {
    board
        .king_square(color)
        .map(|sq| sq.0)
        .ok_or(HcpError::KingNotFound(color))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcp_hirate_roundtrip() {
        let board = Board::new();
        let hcp = to_hcp(&board).unwrap();
        let decoded = from_hcp(&hcp).unwrap();

        assert_eq!(decoded.turn, board.turn);
        for sq in 0..81 {
            assert_eq!(
                decoded.squares[sq], board.squares[sq],
                "mismatch at sq {}",
                sq
            );
        }
        for c in 0..2 {
            for h in 0..7 {
                assert_eq!(decoded.hand[c][h], board.hand[c][h]);
            }
        }
    }

    #[test]
    fn test_hcp_roundtrip_with_hand() {
        // 飛角交換後の局面
        let mut board = Board::empty();
        board
            .set_sfen("lnsgkg1nl/1r5s1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/7R1/LNSGKGSNL b Bb 5")
            .unwrap();
        let hcp = to_hcp(&board).unwrap();
        let decoded = from_hcp(&hcp).unwrap();

        assert_eq!(decoded.turn, board.turn);
        for sq in 0..81 {
            assert_eq!(
                decoded.squares[sq], board.squares[sq],
                "mismatch at sq {}",
                sq
            );
        }
        assert_eq!(decoded.hand, board.hand);
    }

    #[test]
    fn test_hcp_roundtrip_white_turn() {
        let mut board = Board::empty();
        board
            .set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2")
            .unwrap();
        let hcp = to_hcp(&board).unwrap();
        let decoded = from_hcp(&hcp).unwrap();
        assert_eq!(decoded.turn, Color::White);
        for sq in 0..81 {
            assert_eq!(decoded.squares[sq], board.squares[sq]);
        }
    }
}
