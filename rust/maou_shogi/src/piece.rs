use crate::types::{Color, Piece, PieceType};

/// cshogi駒IDからdomain PieceId (maou内部形式)への変換テーブル．
///
/// cshogi: 歩(1),香(2),桂(3),銀(4),角(5),飛(6),金(7),王(8) ...
/// domain: 歩(1),香(2),桂(3),銀(4),金(5),角(6),飛(7),王(8) ...
/// 角・飛・金の順序が異なる．
#[allow(dead_code)]
const CSHOGI_TO_DOMAIN: [u8; 31] = [
    0,  // 0: EMPTY
    1,  // 1: BPAWN → FU
    2,  // 2: BLANCE → KY
    3,  // 3: BKNIGHT → KE
    4,  // 4: BSILVER → GI
    6,  // 5: BBISHOP → KA
    7,  // 6: BROOK → HI
    5,  // 7: BGOLD → KI
    8,  // 8: BKING → OU
    9,  // 9: BPROPAWN → TO
    10, // 10: BPROLANCE → NKY
    11, // 11: BPROKNIGHT → NKE
    12, // 12: BPROSILVER → NGI
    13, // 13: BPROBISHOP → UMA
    14, // 14: BPROROOK → RYU
    0,  // 15: unused
    0,  // 16: unused
    15, // 17: WPAWN → FU + 14
    16, // 18: WLANCE → KY + 14
    17, // 19: WKNIGHT → KE + 14
    18, // 20: WSILVER → GI + 14
    20, // 21: WBISHOP → KA + 14
    21, // 22: WROOK → HI + 14
    19, // 23: WGOLD → KI + 14
    22, // 24: WKING → OU + 14
    23, // 25: WPROPAWN → TO + 14
    24, // 26: WPROLANCE → NKY + 14
    25, // 27: WPROKNIGHT → NKE + 14
    26, // 28: WPROSILVER → NGI + 14
    27, // 29: WPROBISHOP → UMA + 14
    28, // 30: WPROROOK → RYU + 14
];

/// cshogi駒ID(0-30)をdomain PieceId(0-28)に変換する．
///
/// 無効なID(範囲外およびcshogi未使用のID 15,16)の場合は `None` を返す．
#[allow(dead_code)]
#[inline]
pub fn cshogi_to_domain_piece_id(cshogi_piece: u8) -> Option<u8> {
    match cshogi_piece {
        15 | 16 => None,
        _ => CSHOGI_TO_DOMAIN.get(cshogi_piece as usize).copied(),
    }
}

/// SFENの駒文字からPieceを生成する．
///
/// 大文字=先手，小文字=後手．
/// '+' prefixで成駒(呼び出し側で処理)．
pub fn piece_from_sfen_char(ch: char) -> Option<Piece> {
    let (color, pt) = match ch {
        'P' => (Color::Black, PieceType::Pawn),
        'L' => (Color::Black, PieceType::Lance),
        'N' => (Color::Black, PieceType::Knight),
        'S' => (Color::Black, PieceType::Silver),
        'B' => (Color::Black, PieceType::Bishop),
        'R' => (Color::Black, PieceType::Rook),
        'G' => (Color::Black, PieceType::Gold),
        'K' => (Color::Black, PieceType::King),
        'p' => (Color::White, PieceType::Pawn),
        'l' => (Color::White, PieceType::Lance),
        'n' => (Color::White, PieceType::Knight),
        's' => (Color::White, PieceType::Silver),
        'b' => (Color::White, PieceType::Bishop),
        'r' => (Color::White, PieceType::Rook),
        'g' => (Color::White, PieceType::Gold),
        'k' => (Color::White, PieceType::King),
        _ => return None,
    };
    Some(Piece::new(color, pt))
}

/// PieceをSFEN文字列に変換する．成駒の場合は'+X'の2文字．
///
/// ヒープアロケーションを避けるため `&'static str` を返す．
#[inline]
pub fn piece_to_sfen_string(piece: Piece) -> &'static str {
    if piece.is_empty() {
        return "";
    }
    let color = piece.color().unwrap();
    let pt = piece.piece_type().unwrap();

    match (color, pt) {
        (Color::Black, PieceType::Pawn) => "P",
        (Color::Black, PieceType::Lance) => "L",
        (Color::Black, PieceType::Knight) => "N",
        (Color::Black, PieceType::Silver) => "S",
        (Color::Black, PieceType::Bishop) => "B",
        (Color::Black, PieceType::Rook) => "R",
        (Color::Black, PieceType::Gold) => "G",
        (Color::Black, PieceType::King) => "K",
        (Color::Black, PieceType::ProPawn) => "+P",
        (Color::Black, PieceType::ProLance) => "+L",
        (Color::Black, PieceType::ProKnight) => "+N",
        (Color::Black, PieceType::ProSilver) => "+S",
        (Color::Black, PieceType::Horse) => "+B",
        (Color::Black, PieceType::Dragon) => "+R",
        (Color::White, PieceType::Pawn) => "p",
        (Color::White, PieceType::Lance) => "l",
        (Color::White, PieceType::Knight) => "n",
        (Color::White, PieceType::Silver) => "s",
        (Color::White, PieceType::Bishop) => "b",
        (Color::White, PieceType::Rook) => "r",
        (Color::White, PieceType::Gold) => "g",
        (Color::White, PieceType::King) => "k",
        (Color::White, PieceType::ProPawn) => "+p",
        (Color::White, PieceType::ProLance) => "+l",
        (Color::White, PieceType::ProKnight) => "+n",
        (Color::White, PieceType::ProSilver) => "+s",
        (Color::White, PieceType::Horse) => "+b",
        (Color::White, PieceType::Dragon) => "+r",
    }
}

/// 持ち駒の駒種からSFEN文字(大文字)を返す．
///
/// 持ち駒になれない駒種(王・成駒)の場合は `None` を返す．
pub fn hand_piece_to_sfen_char(pt: PieceType) -> Option<char> {
    match pt {
        PieceType::Pawn => Some('P'),
        PieceType::Lance => Some('L'),
        PieceType::Knight => Some('N'),
        PieceType::Silver => Some('S'),
        PieceType::Gold => Some('G'),
        PieceType::Bishop => Some('B'),
        PieceType::Rook => Some('R'),
        _ => None,
    }
}

impl std::fmt::Display for Piece {
    /// SFEN形式で駒を表示する(例: `P`, `+b`, 空マスは `.`)．
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = piece_to_sfen_string(*self);
        if s.is_empty() {
            f.write_str(".")
        } else {
            f.write_str(s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cshogi_to_domain() {
        assert_eq!(cshogi_to_domain_piece_id(0), Some(0)); // EMPTY
        assert_eq!(cshogi_to_domain_piece_id(1), Some(1)); // BPAWN → FU
        assert_eq!(cshogi_to_domain_piece_id(5), Some(6)); // BBISHOP → KA
        assert_eq!(cshogi_to_domain_piece_id(6), Some(7)); // BROOK → HI
        assert_eq!(cshogi_to_domain_piece_id(7), Some(5)); // BGOLD → KI
        assert_eq!(cshogi_to_domain_piece_id(17), Some(15)); // WPAWN
        assert_eq!(cshogi_to_domain_piece_id(21), Some(20)); // WBISHOP
        assert_eq!(cshogi_to_domain_piece_id(22), Some(21)); // WROOK
        assert_eq!(cshogi_to_domain_piece_id(23), Some(19)); // WGOLD
        assert_eq!(cshogi_to_domain_piece_id(15), None); // unused
        assert_eq!(cshogi_to_domain_piece_id(16), None); // unused
        assert_eq!(cshogi_to_domain_piece_id(31), None); // invalid
    }

    #[test]
    fn test_piece_sfen_roundtrip() {
        let piece = Piece::new(Color::Black, PieceType::Pawn);
        assert_eq!(piece_to_sfen_string(piece), "P");

        let piece = Piece::new(Color::White, PieceType::Rook);
        assert_eq!(piece_to_sfen_string(piece), "r");

        let piece = Piece::new(Color::Black, PieceType::Horse);
        assert_eq!(piece_to_sfen_string(piece), "+B");
    }
}
