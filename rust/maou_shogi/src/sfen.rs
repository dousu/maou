use crate::piece::{hand_piece_to_sfen_char, piece_from_sfen_char, piece_to_sfen_string};
use crate::types::{Color, Piece, PieceType, Square};

/// SFEN文字列からの局面パース結果．
#[derive(Debug, Clone)]
pub struct SfenPosition {
    /// 盤面(81マス)．column-major順．
    pub squares: [Piece; 81],
    /// 手番．
    pub turn: Color,
    /// 持ち駒 [色][駒種7]．
    pub hand: [[u8; 7]; 2],
    /// 手数(SFEN4番目のフィールド)．
    pub ply: u16,
}

/// SFEN文字列をパースする．
///
/// SFEN例: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
pub fn parse_sfen(sfen: &str) -> Result<SfenPosition, String> {
    let parts: Vec<&str> = sfen.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(format!("invalid SFEN: too few fields: {}", sfen));
    }

    // 1. 盤面
    let mut squares = [Piece::EMPTY; 81];
    let rows: Vec<&str> = parts[0].split('/').collect();
    if rows.len() != 9 {
        return Err(format!("invalid SFEN board: expected 9 rows, got {}", rows.len()));
    }

    // SFENの行は上(1段目=row0)から下(9段目=row8)，
    // 各行は左(9筋=col0)から右(1筋=col8)の順．
    for (row, row_str) in rows.iter().enumerate() {
        let mut col: u8 = 0;
        let mut chars = row_str.chars().peekable();
        while let Some(ch) = chars.next() {
            if col >= 9 {
                return Err(format!("invalid SFEN board: row {} has too many pieces", row));
            }
            if ch.is_ascii_digit() {
                col += ch.to_digit(10).unwrap() as u8;
            } else if ch == '+' {
                // 成駒: 次の文字と組み合わせる
                let next_ch = chars.next().ok_or_else(|| {
                    format!("invalid SFEN: '+' at end of row {}", row)
                })?;
                let piece = piece_from_sfen_char(next_ch).ok_or_else(|| {
                    format!("invalid SFEN piece: +{}", next_ch)
                })?;
                let pt = piece.piece_type().unwrap();
                let promoted_pt = pt.promoted().ok_or_else(|| {
                    format!("invalid SFEN: cannot promote {:?}", pt)
                })?;
                let color = piece.color().unwrap();
                let sq = Square::new(8 - col, row as u8);
                squares[sq.index()] = Piece::new(color, promoted_pt);
                col += 1;
            } else {
                let piece = piece_from_sfen_char(ch).ok_or_else(|| {
                    format!("invalid SFEN piece char: {}", ch)
                })?;
                let sq = Square::new(8 - col, row as u8);
                squares[sq.index()] = piece;
                col += 1;
            }
        }
        if col != 9 {
            return Err(format!(
                "invalid SFEN board: row {} has {} columns, expected 9",
                row, col
            ));
        }
    }

    // 2. 手番
    let turn = match parts[1] {
        "b" => Color::Black,
        "w" => Color::White,
        _ => return Err(format!("invalid SFEN turn: {}", parts[1])),
    };

    // 3. 持ち駒
    let mut hand = [[0u8; 7]; 2];
    if parts[2] != "-" {
        let mut chars = parts[2].chars().peekable();
        while let Some(ch) = chars.next() {
            let count: u8;
            if ch.is_ascii_digit() {
                // 数字で始まる場合は枚数
                let mut num_str = ch.to_string();
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        num_str.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                count = num_str.parse().map_err(|_| {
                    format!("invalid SFEN hand count: {}", num_str)
                })?;
                // 次の文字が駒種
                let piece_ch = chars.next().ok_or_else(|| {
                    "invalid SFEN: number at end of hand".to_string()
                })?;
                let (color, pt) = parse_hand_piece(piece_ch)?;
                let idx = pt.hand_index().unwrap();
                hand[color.index()][idx] += count;
            } else {
                let (color, pt) = parse_hand_piece(ch)?;
                let idx = pt.hand_index().unwrap();
                hand[color.index()][idx] += 1;
            }
        }
    }

    // 4. 手数(省略可能)
    let ply = if parts.len() >= 4 {
        parts[3]
            .parse()
            .map_err(|_| format!("invalid ply number: '{}'", parts[3]))?
    } else {
        1
    };

    Ok(SfenPosition {
        squares,
        turn,
        hand,
        ply,
    })
}

fn parse_hand_piece(ch: char) -> Result<(Color, PieceType), String> {
    let (color, pt) = match ch {
        'P' => (Color::Black, PieceType::Pawn),
        'L' => (Color::Black, PieceType::Lance),
        'N' => (Color::Black, PieceType::Knight),
        'S' => (Color::Black, PieceType::Silver),
        'G' => (Color::Black, PieceType::Gold),
        'B' => (Color::Black, PieceType::Bishop),
        'R' => (Color::Black, PieceType::Rook),
        'p' => (Color::White, PieceType::Pawn),
        'l' => (Color::White, PieceType::Lance),
        'n' => (Color::White, PieceType::Knight),
        's' => (Color::White, PieceType::Silver),
        'g' => (Color::White, PieceType::Gold),
        'b' => (Color::White, PieceType::Bishop),
        'r' => (Color::White, PieceType::Rook),
        _ => return Err(format!("invalid hand piece: {}", ch)),
    };
    Ok((color, pt))
}

/// 局面をSFEN文字列に変換する．
pub fn to_sfen(squares: &[Piece; 81], turn: Color, hand: &[[u8; 7]; 2], ply: u16) -> String {
    let mut result = String::new();

    // 1. 盤面
    for row in 0..9u8 {
        if row > 0 {
            result.push('/');
        }
        let mut empty_count = 0u8;
        for col in (0..9u8).rev() {
            let sq = Square::new(col, row);
            let piece = squares[sq.index()];
            if piece.is_empty() {
                empty_count += 1;
            } else {
                if empty_count > 0 {
                    result.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                result.push_str(&piece_to_sfen_string(piece));
            }
        }
        if empty_count > 0 {
            result.push_str(&empty_count.to_string());
        }
    }

    // 2. 手番
    result.push(' ');
    result.push(match turn {
        Color::Black => 'b',
        Color::White => 'w',
    });

    // 3. 持ち駒
    result.push(' ');
    let mut has_hand = false;

    // 先手 → 後手の順(SFEN規則: 大文字が先)
    for &color in &[Color::Black, Color::White] {
        for (i, &pt) in PieceType::HAND_PIECES.iter().enumerate().rev() {
            let count = hand[color.index()][i];
            if count > 0 {
                has_hand = true;
                if count > 1 {
                    result.push_str(&count.to_string());
                }
                let ch = hand_piece_to_sfen_char(pt);
                result.push(match color {
                    Color::Black => ch,
                    Color::White => ch.to_ascii_lowercase(),
                });
            }
        }
    }
    if !has_hand {
        result.push('-');
    }

    // 4. 手数
    result.push(' ');
    result.push_str(&ply.to_string());

    result
}

/// 平手初期局面のSFEN．
pub const HIRATE_SFEN: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hirate() {
        let pos = parse_sfen(HIRATE_SFEN).unwrap();
        assert_eq!(pos.turn, Color::Black);

        // 先手の歩は7段目(row=6)に9枚
        for col in 0..9 {
            let sq = Square::new(col, 6);
            let piece = pos.squares[sq.index()];
            assert_eq!(piece.piece_type(), Some(PieceType::Pawn));
            assert_eq!(piece.color(), Some(Color::Black));
        }

        // 後手の歩は3段目(row=2)に9枚
        for col in 0..9 {
            let sq = Square::new(col, 2);
            let piece = pos.squares[sq.index()];
            assert_eq!(piece.piece_type(), Some(PieceType::Pawn));
            assert_eq!(piece.color(), Some(Color::White));
        }

        // 先手の王は5九(col=4, row=8)
        let sq = Square::new(4, 8);
        let piece = pos.squares[sq.index()];
        assert_eq!(piece.piece_type(), Some(PieceType::King));
        assert_eq!(piece.color(), Some(Color::Black));

        // 持ち駒なし
        assert_eq!(pos.hand, [[0; 7]; 2]);
    }

    #[test]
    fn test_sfen_roundtrip() {
        let pos = parse_sfen(HIRATE_SFEN).unwrap();
        let sfen = to_sfen(&pos.squares, pos.turn, &pos.hand, pos.ply);
        assert_eq!(sfen, HIRATE_SFEN);
    }

    #[test]
    fn test_sfen_with_hand() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 2P3p 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.hand[0][0], 2); // 先手の歩2枚
        assert_eq!(pos.hand[1][0], 3); // 後手の歩3枚
    }

    #[test]
    fn test_sfen_with_promoted() {
        let sfen = "lnsgkgsnl/1r5b1/pppp+Bpppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL w - 2";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.turn, Color::White);
        // +B は先手の馬(成角) at row=2, col=4
        let sq = Square::new(4, 2);
        let piece = pos.squares[sq.index()];
        assert_eq!(piece.piece_type(), Some(PieceType::Horse));
        assert_eq!(piece.color(), Some(Color::Black));
    }
}
