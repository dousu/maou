/// 手番．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Color {
    /// 相手の手番を返す．
    #[inline]
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    /// u8から変換する．
    #[inline]
    pub fn from_u8(v: u8) -> Option<Color> {
        match v {
            0 => Some(Color::Black),
            1 => Some(Color::White),
            _ => None,
        }
    }

    /// インデックスとして使用する．
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// 勝敗結果．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GameResult {
    BlackWin = 0,
    WhiteWin = 1,
    Draw = 2,
}

/// 駒種(色なし)．cshogi互換の順序．
///
/// cshogi順: 歩(1), 香(2), 桂(3), 銀(4), 角(5), 飛(6), 金(7), 王(8)
/// 成駒: と(9), 成香(10), 成桂(11), 成銀(12), 馬(13), 龍(14)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    // 生駒
    Pawn = 1,    // 歩
    Lance = 2,   // 香
    Knight = 3,  // 桂
    Silver = 4,  // 銀
    Bishop = 5,  // 角
    Rook = 6,    // 飛
    Gold = 7,    // 金
    King = 8,    // 王
    // 成駒
    ProPawn = 9,    // と
    ProLance = 10,  // 成香
    ProKnight = 11, // 成桂
    ProSilver = 12, // 成銀
    Horse = 13,     // 馬 (成角)
    Dragon = 14,    // 龍 (成飛)
}

// is_promoted() が King < 9 <= ProPawn を前提としていることをコンパイル時に検証
const _: () = {
    assert!(PieceType::King as u8 == 8, "is_promoted() assumes King == 8");
    assert!(
        PieceType::ProPawn as u8 == 9,
        "is_promoted() assumes ProPawn == 9"
    );
};

impl PieceType {
    /// 全駒種のスライス．
    pub const ALL: [PieceType; 14] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Gold,
        PieceType::King,
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::Horse,
        PieceType::Dragon,
    ];

    /// 持ち駒にできる駒種(打てる駒)．順序は cshogi の pieces_in_hand と同じ．
    pub const HAND_PIECES: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];

    /// 各持ち駒の最大枚数．HAND_PIECES と同じ順序．
    pub const MAX_HAND_COUNT: [u8; HAND_KINDS] = [18, 4, 4, 4, 4, 2, 2];

    /// 持ち駒のインデックスを返す(0-6)．持ち駒にできない駒種はNone．
    #[inline]
    pub fn hand_index(self) -> Option<usize> {
        match self {
            PieceType::Pawn => Some(0),
            PieceType::Lance => Some(1),
            PieceType::Knight => Some(2),
            PieceType::Silver => Some(3),
            PieceType::Gold => Some(4),
            PieceType::Bishop => Some(5),
            PieceType::Rook => Some(6),
            _ => None,
        }
    }

    /// 成れるかどうか．
    #[inline]
    pub fn can_promote(self) -> bool {
        matches!(
            self,
            PieceType::Pawn
                | PieceType::Lance
                | PieceType::Knight
                | PieceType::Silver
                | PieceType::Bishop
                | PieceType::Rook
        )
    }

    /// 成った後の駒種を返す．成れない場合はNone．
    #[inline]
    pub fn promoted(self) -> Option<PieceType> {
        match self {
            PieceType::Pawn => Some(PieceType::ProPawn),
            PieceType::Lance => Some(PieceType::ProLance),
            PieceType::Knight => Some(PieceType::ProKnight),
            PieceType::Silver => Some(PieceType::ProSilver),
            PieceType::Bishop => Some(PieceType::Horse),
            PieceType::Rook => Some(PieceType::Dragon),
            _ => None,
        }
    }

    /// 成駒の元の駒種を返す．生駒の場合はNone．
    #[inline]
    pub fn unpromoted(self) -> Option<PieceType> {
        match self {
            PieceType::ProPawn => Some(PieceType::Pawn),
            PieceType::ProLance => Some(PieceType::Lance),
            PieceType::ProKnight => Some(PieceType::Knight),
            PieceType::ProSilver => Some(PieceType::Silver),
            PieceType::Horse => Some(PieceType::Bishop),
            PieceType::Dragon => Some(PieceType::Rook),
            _ => None,
        }
    }

    /// 成駒かどうか．
    #[inline]
    pub fn is_promoted(self) -> bool {
        (self as u8) >= 9
    }

    /// u8から変換する．
    #[inline]
    pub fn from_u8(v: u8) -> Option<PieceType> {
        match v {
            1 => Some(PieceType::Pawn),
            2 => Some(PieceType::Lance),
            3 => Some(PieceType::Knight),
            4 => Some(PieceType::Silver),
            5 => Some(PieceType::Bishop),
            6 => Some(PieceType::Rook),
            7 => Some(PieceType::Gold),
            8 => Some(PieceType::King),
            9 => Some(PieceType::ProPawn),
            10 => Some(PieceType::ProLance),
            11 => Some(PieceType::ProKnight),
            12 => Some(PieceType::ProSilver),
            13 => Some(PieceType::Horse),
            14 => Some(PieceType::Dragon),
            _ => None,
        }
    }

    /// 持ち駒に取った時の駒種(成駒→生駒に戻す)を返す．
    #[inline]
    pub fn captured_to_hand(self) -> PieceType {
        self.unpromoted().unwrap_or(self)
    }
}

/// 盤上の駒(色付き)．cshogi互換のID体系．
///
/// - 0: 空マス
/// - 1-14: 先手の駒
/// - 15-16: 未使用 (cshogi互換のギャップ)
/// - 17-30: 後手の駒 (先手 + 16)
///
/// 内部値は `pub(crate)` とし，外部クレートからは `raw_u8()` アクセサで取得する．
/// 未定義値 (15, 16) の生成はクレート内部に限定されるため，
/// 通常の局面生成パス (SFEN/HCP) では安全に使用できる．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece(pub(crate) u8);

impl Piece {
    pub const EMPTY: Piece = Piece(0);

    /// 内部値を返す(外部クレート向けアクセサ)．
    #[inline]
    pub fn raw_u8(self) -> u8 {
        self.0
    }

    /// cshogi互換の白駒オフセット．
    pub const WHITE_OFFSET: u8 = 16;

    /// 色と駒種から生成する．
    #[inline]
    pub fn new(color: Color, piece_type: PieceType) -> Piece {
        let base = piece_type as u8;
        match color {
            Color::Black => Piece(base),
            Color::White => Piece(base + Self::WHITE_OFFSET),
        }
    }

    /// 空マスかどうか．
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// 色を返す．空マスまたは未定義値 (15, 16) の場合は `None`．
    ///
    /// cshogi 互換 ID 空間では 15, 16 はギャップであり，`Piece::new()` では
    /// 生成されない．`raw_u8()` 経由で外部から不正値が渡された場合は
    /// `None` を返すことで安全にフォールバックする．
    #[inline]
    pub fn color(self) -> Option<Color> {
        if self.0 == 0 {
            None
        } else if self.0 <= 14 {
            Some(Color::Black)
        } else if self.0 >= 17 && self.0 <= 30 {
            Some(Color::White)
        } else {
            // cshogi 互換ギャップ (15, 16) または範囲外
            None
        }
    }

    /// 駒種を返す．空マスの場合はNone．
    #[inline]
    pub fn piece_type(self) -> Option<PieceType> {
        if self.0 == 0 {
            return None;
        }
        let base = if self.0 >= 17 {
            self.0 - Self::WHITE_OFFSET
        } else {
            self.0
        };
        PieceType::from_u8(base)
    }
}

/// マス番号(0-80)．column-major: square = col * 9 + row．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(pub(crate) u8);

impl Square {
    pub const NUM: usize = 81;

    /// 内部値を返す(外部クレート向けアクセサ)．
    #[inline]
    pub fn raw_u8(self) -> u8 {
        self.0
    }

    /// col, rowから生成する．
    ///
    /// # Safety (論理的前提条件)
    ///
    /// `col < 9 && row < 9` であること．リリースビルドではチェックされない．
    /// 呼び出し元(SFENパーサー等)で事前にバリデーションすること．
    #[inline]
    pub fn new(col: u8, row: u8) -> Square {
        debug_assert!(col < 9 && row < 9, "Square::new: col={}, row={} out of range", col, row);
        Square(col * 9 + row)
    }

    /// 列(筋)を返す．0=1筋, 8=9筋．
    #[inline]
    pub fn col(self) -> u8 {
        self.0 / 9
    }

    /// 行(段)を返す．0=一段, 8=九段．
    #[inline]
    pub fn row(self) -> u8 {
        self.0 % 9
    }

    /// インデックスとして使用する．
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// 先手の敵陣(1-3段目, row 0-2)かどうか．
    #[inline]
    pub fn is_promotion_zone_black(self) -> bool {
        self.row() <= 2
    }

    /// 後手の敵陣(7-9段目, row 6-8)かどうか．
    #[inline]
    pub fn is_promotion_zone_white(self) -> bool {
        self.row() >= 6
    }

    /// 指定した色にとっての敵陣かどうか．
    #[inline]
    pub fn is_promotion_zone(self, color: Color) -> bool {
        match color {
            Color::Black => self.is_promotion_zone_black(),
            Color::White => self.is_promotion_zone_white(),
        }
    }

    /// 有効なマス番号かどうか．
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 < 81
    }
}

/// 定数．
pub const PIECE_TYPES_NUM: usize = 14;
/// piece_bb配列のサイズ．PieceTypeは1始まりのため，インデックス0は未使用．
pub const PIECE_BB_SIZE: usize = PIECE_TYPES_NUM + 1;
/// 持ち駒の種類数(歩,香,桂,銀,金,角,飛)．
pub const HAND_KINDS: usize = 7;
/// 持ち駒1種あたりの最大状態数(歩の最大18枚 + 0枚状態 = 19)．
pub const MAX_HAND_STATES: usize = 19;
/// 特徴平面数: 28 (盤上14駒種×2色) + 76 (持ち駒38枚×2色, 38=18歩+4香+4桂+4銀+4金+2角+2飛) = 104
pub const FEATURES_NUM: usize = PIECE_TYPES_NUM * 2 + 76;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_opponent() {
        assert_eq!(Color::Black.opponent(), Color::White);
        assert_eq!(Color::White.opponent(), Color::Black);
    }

    #[test]
    fn test_piece_new() {
        let bp = Piece::new(Color::Black, PieceType::Pawn);
        assert_eq!(bp.0, 1);
        assert_eq!(bp.color(), Some(Color::Black));
        assert_eq!(bp.piece_type(), Some(PieceType::Pawn));

        let wp = Piece::new(Color::White, PieceType::Pawn);
        assert_eq!(wp.0, 17);
        assert_eq!(wp.color(), Some(Color::White));
        assert_eq!(wp.piece_type(), Some(PieceType::Pawn));
    }

    #[test]
    fn test_piece_type_promotion() {
        assert_eq!(PieceType::Pawn.promoted(), Some(PieceType::ProPawn));
        assert_eq!(PieceType::Bishop.promoted(), Some(PieceType::Horse));
        assert_eq!(PieceType::Gold.promoted(), None);
        assert_eq!(PieceType::King.promoted(), None);
        assert_eq!(PieceType::ProPawn.unpromoted(), Some(PieceType::Pawn));
    }

    #[test]
    fn test_square() {
        let sq = Square::new(3, 5);
        assert_eq!(sq.0, 32); // 3*9 + 5
        assert_eq!(sq.col(), 3);
        assert_eq!(sq.row(), 5);
    }

    #[test]
    fn test_square_promotion_zone() {
        assert!(Square::new(0, 0).is_promotion_zone(Color::Black));
        assert!(Square::new(0, 2).is_promotion_zone(Color::Black));
        assert!(!Square::new(0, 3).is_promotion_zone(Color::Black));
        assert!(Square::new(0, 6).is_promotion_zone(Color::White));
        assert!(!Square::new(0, 5).is_promotion_zone(Color::White));
    }

    #[test]
    fn test_hand_index() {
        assert_eq!(PieceType::Pawn.hand_index(), Some(0));
        assert_eq!(PieceType::Rook.hand_index(), Some(6));
        assert_eq!(PieceType::King.hand_index(), None);
        assert_eq!(PieceType::ProPawn.hand_index(), None);
    }
}
