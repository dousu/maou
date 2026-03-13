use crate::types::{Color, Piece, PieceType};

/// cshogi駒IDからdomain PieceId (maou内部形式)への変換テーブル．
///
/// cshogi: 歩(1),香(2),桂(3),銀(4),角(5),飛(6),金(7),王(8) ...
/// domain: 歩(1),香(2),桂(3),銀(4),金(5),角(6),飛(7),王(8) ...
/// 角・飛・金の順序が異なる．
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
#[inline]
pub fn cshogi_to_domain_piece_id(cshogi_piece: u8) -> u8 {
    if (cshogi_piece as usize) < CSHOGI_TO_DOMAIN.len() {
        CSHOGI_TO_DOMAIN[cshogi_piece as usize]
    } else {
        0
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

/// PieceをSFEN文字に変換する．成駒の場合は'+X'の2文字．
pub fn piece_to_sfen_string(piece: Piece) -> String {
    if piece.is_empty() {
        return String::new();
    }
    let color = piece.color().unwrap();
    let pt = piece.piece_type().unwrap();

    let (promoted, base_pt) = if pt.is_promoted() {
        (true, pt.unpromoted().unwrap())
    } else {
        (false, pt)
    };

    let ch = match base_pt {
        PieceType::Pawn => 'P',
        PieceType::Lance => 'L',
        PieceType::Knight => 'N',
        PieceType::Silver => 'S',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::Gold => 'G',
        PieceType::King => 'K',
        PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver
        | PieceType::Horse
        | PieceType::Dragon => unreachable!("promoted piece after unpromoted(): {:?}", base_pt),
    };

    let ch = match color {
        Color::Black => ch,
        Color::White => ch.to_ascii_lowercase(),
    };

    if promoted {
        format!("+{}", ch)
    } else {
        ch.to_string()
    }
}

/// 持ち駒の駒種からSFEN文字(大文字)を返す．
pub fn hand_piece_to_sfen_char(pt: PieceType) -> char {
    match pt {
        PieceType::Pawn => 'P',
        PieceType::Lance => 'L',
        PieceType::Knight => 'N',
        PieceType::Silver => 'S',
        PieceType::Gold => 'G',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::King
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver
        | PieceType::Horse
        | PieceType::Dragon => unreachable!("not a hand piece: {:?}", pt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cshogi_to_domain() {
        assert_eq!(cshogi_to_domain_piece_id(0), 0); // EMPTY
        assert_eq!(cshogi_to_domain_piece_id(1), 1); // BPAWN → FU
        assert_eq!(cshogi_to_domain_piece_id(5), 6); // BBISHOP → KA
        assert_eq!(cshogi_to_domain_piece_id(6), 7); // BROOK → HI
        assert_eq!(cshogi_to_domain_piece_id(7), 5); // BGOLD → KI
        assert_eq!(cshogi_to_domain_piece_id(17), 15); // WPAWN
        assert_eq!(cshogi_to_domain_piece_id(21), 20); // WBISHOP
        assert_eq!(cshogi_to_domain_piece_id(22), 21); // WROOK
        assert_eq!(cshogi_to_domain_piece_id(23), 19); // WGOLD
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
