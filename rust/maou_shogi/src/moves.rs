use crate::types::{PieceType, Square};

/// 指し手(32-bit)．
///
/// cshogi互換の16-bitエンコーディングに追加情報を持たせる．
/// 16-bit部分:
/// - Bits 0-6: to_sq (0-80)
/// - Bits 7-13: from_sq (0-80，通常手) or 81+piece_type (駒打ち)
/// - Bit 14: promotion flag
///
/// 駒打ちの判定は from_field >= 81 で行う(cshogi互換)．
///
/// 追加情報(上位16-bit):
/// - Bits 16-20: captured piece (取った駒の cshogi ID，0=取っていない)
/// - Bits 21-24: moving piece type (動かした駒種)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move(pub(crate) u32);

/// 盤面のマス数(cshogi互換の駒打ち判定閾値)．
const SQ_SIZE: u32 = 81;

impl Move {
    pub const NONE: Move = Move(0);

    /// 内部値を返す(外部クレート向けアクセサ)．
    #[inline]
    pub fn raw_u32(self) -> u32 {
        self.0
    }

    /// 生のu32値からMoveを生成する(外部クレート向け)．
    #[inline]
    pub fn from_raw_u32(v: u32) -> Move {
        Move(v)
    }

    // 16-bit move masks
    const TO_MASK: u32 = 0x7F; // bits 0-6
    const FROM_MASK: u32 = 0x7F << 7; // bits 7-13
    const PROMOTE_FLAG: u32 = 1 << 14;

    // 32-bit extension masks
    const CAPTURED_SHIFT: u32 = 16;
    const CAPTURED_MASK: u32 = 0x1F << 16;
    const MOVING_PIECE_SHIFT: u32 = 21;

    /// 盤上の駒を移動する手を生成する(クレート内部専用)．
    ///
    /// `captured` は cshogi 互換の駒 ID (0-30)，`moving_pt` は `PieceType as u8` (0-14)．
    /// いずれも 5 ビットに収まる値であること(リリースビルドではチェックされない)．
    #[inline]
    pub(crate) fn new_move(from: Square, to: Square, promote: bool, captured: u8, moving_pt: u8) -> Move {
        debug_assert!(captured <= 30, "captured raw ID {} out of range", captured);
        debug_assert!(moving_pt <= 14, "moving_pt {} out of range", moving_pt);
        let mut v = (to.0 as u32) | ((from.0 as u32) << 7);
        if promote {
            v |= Self::PROMOTE_FLAG;
        }
        v |= (captured as u32) << Self::CAPTURED_SHIFT;
        v |= (moving_pt as u32) << Self::MOVING_PIECE_SHIFT;
        Move(v)
    }

    /// 駒を打つ手を生成する(クレート内部専用)．
    ///
    /// cshogi互換: from_field に `81 + drop_move_index` を格納する．
    /// drop_move_index: 歩=0, 香=1, 桂=2, 銀=3, 角=4, 飛=5, 金=6
    #[inline]
    pub(crate) fn new_drop(to: Square, piece_type: PieceType) -> Move {
        let idx = piece_type.drop_move_index().unwrap();
        let v = (to.0 as u32) | ((SQ_SIZE + idx) << 7);
        Move(v)
    }

    /// 移動先のマスを返す．
    #[inline]
    pub fn to_sq(self) -> Square {
        Square((self.0 & Self::TO_MASK) as u8)
    }

    /// 移動元のフィールド値を返す(駒打ちの場合は 81+piece_type)．
    #[inline]
    fn from_field(self) -> u32 {
        (self.0 & Self::FROM_MASK) >> 7
    }

    /// 移動元のマスを返す(駒打ちの場合は意味を持たない)．
    #[inline]
    pub fn from_sq(self) -> Square {
        Square(self.from_field() as u8)
    }

    /// 駒打ちかどうか(cshogi互換: from_field >= 81)．
    #[inline]
    pub fn is_drop(self) -> bool {
        self.from_field() >= SQ_SIZE
    }

    /// 移動する駒種を返す(盤上の駒の移動手のみ有効)．
    #[inline]
    pub fn moving_piece_type_raw(self) -> u8 {
        ((self.0 >> Self::MOVING_PIECE_SHIFT) & 0x1F) as u8
    }

    /// 成りかどうか．
    #[inline]
    pub fn is_promotion(self) -> bool {
        (self.0 & Self::PROMOTE_FLAG) != 0
    }

    /// 打った駒種を返す．
    ///
    /// 駒打ちでない場合は `None` を返す．
    /// cshogi互換のdrop_move_indexからPieceTypeに変換する．
    #[inline]
    pub fn drop_piece_type(self) -> Option<PieceType> {
        if !self.is_drop() {
            return None;
        }
        let idx = self.from_field() - SQ_SIZE;
        PieceType::from_drop_move_index(idx)
    }

    /// 取った駒のcshogi IDを返す(0=取っていない)．
    #[inline]
    pub fn captured_piece_raw(self) -> u8 {
        ((self.0 & Self::CAPTURED_MASK) >> Self::CAPTURED_SHIFT) as u8
    }

    /// 16-bit表現(HCPE形式)に変換する．
    #[inline]
    pub fn to_move16(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// 16-bit表現からMove(上位ビットなし)を生成する．
    /// 完全な32-bit Moveに変換するにはBoard上での解決が必要．
    #[inline]
    pub fn from_move16(move16: u16) -> Move {
        Move(move16 as u32)
    }

    /// USI文字列に変換する．
    pub fn to_usi(self) -> String {
        if self.is_drop() {
            let pt = self.drop_piece_type().unwrap();
            let ch = match pt {
                PieceType::Pawn => 'P',
                PieceType::Lance => 'L',
                PieceType::Knight => 'N',
                PieceType::Silver => 'S',
                PieceType::Gold => 'G',
                PieceType::Bishop => 'B',
                PieceType::Rook => 'R',
                _ => unreachable!(),
            };
            let to = self.to_sq();
            let file = to.col() + 1; // USI: 1-9 (col0=1筋→file1)
            let rank = (b'a' + to.row()) as char;
            format!("{}*{}{}", ch, file, rank)
        } else {
            let from = self.from_sq();
            let to = self.to_sq();
            let from_file = from.col() + 1;
            let from_rank = (b'a' + from.row()) as char;
            let to_file = to.col() + 1;
            let to_rank = (b'a' + to.row()) as char;
            let promo = if self.is_promotion() { "+" } else { "" };
            format!(
                "{}{}{}{}{}",
                from_file, from_rank, to_file, to_rank, promo
            )
        }
    }

    /// USI文字列からMoveを生成する(16-bitレベル)．
    pub fn from_usi(usi: &str) -> Option<Move> {
        let b = usi.as_bytes();
        if b.len() < 4 {
            return None;
        }

        // 駒打ち: "P*5e" format
        if b[1] == b'*' {
            let pt = match b[0] {
                b'P' => PieceType::Pawn,
                b'L' => PieceType::Lance,
                b'N' => PieceType::Knight,
                b'S' => PieceType::Silver,
                b'G' => PieceType::Gold,
                b'B' => PieceType::Bishop,
                b'R' => PieceType::Rook,
                _ => return None,
            };
            let to_file = b[2].checked_sub(b'0')?;
            let to_rank = b[3].checked_sub(b'a')?;
            if to_file < 1 || to_file > 9 || to_rank > 8 {
                return None;
            }
            let to = Square::new(to_file - 1, to_rank);
            return Some(Move::new_drop(to, pt));
        }

        // 通常の手: "7g7f" or "7g7f+"
        let from_file = b[0].checked_sub(b'0')?;
        let from_rank = b[1].checked_sub(b'a')?;
        let to_file = b[2].checked_sub(b'0')?;
        let to_rank = b[3].checked_sub(b'a')?;

        if from_file < 1 || from_file > 9 || to_file < 1 || to_file > 9 {
            return None;
        }
        if from_rank > 8 || to_rank > 8 {
            return None;
        }

        let promote = b.len() > 4 && b[4] == b'+';
        let from = Square::new(from_file - 1, from_rank);
        let to = Square::new(to_file - 1, to_rank);

        Some(Move::new_move(from, to, promote, 0, 0))
    }
}

/// cshogi互換: move16関数．
#[inline]
pub fn move16(m: Move) -> u16 {
    m.to_move16()
}

/// cshogi互換: move_to関数．
#[inline]
pub fn move_to(m: Move) -> u8 {
    m.to_sq().raw_u8()
}

/// cshogi互換: move_from関数．
///
/// 駒打ちの場合は 81+piece_type を返す(cshogi互換)．
#[inline]
pub fn move_from(m: Move) -> u8 {
    m.from_field() as u8
}

/// cshogi互換: move_to_usi関数．
#[inline]
pub fn move_to_usi(m: Move) -> String {
    m.to_usi()
}

/// cshogi互換: move_is_drop関数．
#[inline]
pub fn move_is_drop(m: Move) -> bool {
    m.is_drop()
}

/// cshogi互換: move_is_promotion関数．
#[inline]
pub fn move_is_promotion(m: Move) -> bool {
    m.is_promotion()
}

/// cshogi互換: move_drop_hand_piece関数．
///
/// cshogi の HandPiece enum 値(歩=0,香=1,桂=2,銀=3,金=4,角=5,飛=6)を返す．
/// 指し手エンコーディング内の順序(歩=0,...,角=4,飛=5,金=6)から
/// HandPiece enum 順序(歩=0,...,金=4,角=5,飛=6)に変換する．
/// 駒打ちでない場合は 0 を返す．
#[inline]
pub fn move_drop_hand_piece(m: Move) -> u8 {
    m.drop_piece_type()
        .and_then(|pt| pt.hand_index())
        .map(|idx| idx as u8)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_encoding() {
        // 7七歩(76) → 7六歩(75): from=Square(6*9+6)=60, to=Square(6*9+5)=59
        let m = Move::new_move(Square(60), Square(59), false, 0, PieceType::Pawn as u8);
        assert_eq!(m.from_sq(), Square(60));
        assert_eq!(m.to_sq(), Square(59));
        assert!(!m.is_drop());
        assert!(!m.is_promotion());
    }

    #[test]
    fn test_move_drop() {
        // P*5e: 5五歩打ち to=Square(4*9+4)=40
        let m = Move::new_drop(Square(40), PieceType::Pawn);
        assert!(m.is_drop());
        assert_eq!(m.drop_piece_type(), Some(PieceType::Pawn));
        assert_eq!(m.to_sq(), Square(40));
    }

    #[test]
    fn test_move_promotion() {
        let m = Move::new_move(Square(10), Square(1), true, 0, PieceType::Silver as u8);
        assert!(m.is_promotion());
        assert!(!m.is_drop());
    }

    #[test]
    fn test_move16_roundtrip() {
        let m = Move::new_move(Square(60), Square(59), false, 0, PieceType::Pawn as u8);
        let m16 = m.to_move16();
        let m2 = Move::from_move16(m16);
        assert_eq!(m2.to_sq(), m.to_sq());
        assert_eq!(m2.from_sq(), m.from_sq());
        assert_eq!(m2.is_drop(), m.is_drop());
        assert_eq!(m2.is_promotion(), m.is_promotion());
    }

    #[test]
    fn test_usi_normal_move() {
        // 7g7f: 7筋=col6, g=row6, f=row5
        let m = Move::from_usi("7g7f").unwrap();
        assert_eq!(m.from_sq(), Square::new(6, 6)); // 7-1=6, g=6
        assert_eq!(m.to_sq(), Square::new(6, 5)); // 7-1=6, f=5
        assert!(!m.is_promotion());

        assert_eq!(m.to_usi(), "7g7f");
    }

    #[test]
    fn test_usi_promotion() {
        let m = Move::from_usi("8h2b+").unwrap();
        assert!(m.is_promotion());
        assert_eq!(m.to_usi(), "8h2b+");
    }

    #[test]
    fn test_usi_drop() {
        let m = Move::from_usi("P*5e").unwrap();
        assert!(m.is_drop());
        assert_eq!(m.drop_piece_type(), Some(PieceType::Pawn));
        assert_eq!(m.to_usi(), "P*5e");
    }

    #[test]
    fn test_compat_functions() {
        let m = Move::new_drop(Square(40), PieceType::Pawn);
        assert!(move_is_drop(m));
        assert_eq!(move_to(m), 40);
        // Pawn: hand_index()=0
        assert_eq!(move_drop_hand_piece(m), 0);
    }

    #[test]
    fn test_cshogi_drop_encoding_compat() {
        // Verify cshogi-compatible drop move encoding.
        // Move encoding stores drops as: to | ((81 + drop_move_index) << 7)
        // drop_move_index order: P=0,L=1,N=2,S=3,B=4,R=5,G=6
        // But move_drop_hand_piece returns hand_index order: P=0,L=1,N=2,S=3,G=4,B=5,R=6
        // (matching cshogi.HPAWN=0,...,HGOLD=4,HBISHOP=5,HROOK=6)

        // R*9i: cshogi move16 = 11088 (to=80, drop_move_index=5=Rook)
        let m = Move::new_drop(Square(80), PieceType::Rook);
        assert_eq!(m.raw_u32() & 0xFFFF, 11088);
        assert!(m.is_drop());
        assert_eq!(m.drop_piece_type(), Some(PieceType::Rook));
        // hand_index: Rook=6 (cshogi HROOK=6)
        assert_eq!(move_drop_hand_piece(m), 6);
        assert_eq!(m.to_usi(), "R*9i");

        // P*5b: cshogi move16 = 10405 (to=37, drop_move_index=0=Pawn)
        let m = Move::new_drop(Square(37), PieceType::Pawn);
        assert_eq!(m.raw_u32() & 0xFFFF, 10405);
        // hand_index: Pawn=0 (cshogi HPAWN=0)
        assert_eq!(move_drop_hand_piece(m), 0);

        // B*5b: cshogi move16 = 10917 (to=37, drop_move_index=4=Bishop)
        let m = Move::new_drop(Square(37), PieceType::Bishop);
        assert_eq!(m.raw_u32() & 0xFFFF, 10917);
        // hand_index: Bishop=5 (cshogi HBISHOP=5)
        assert_eq!(move_drop_hand_piece(m), 5);

        // G*5b: cshogi move16 = 11173 (to=37, drop_move_index=6=Gold)
        let m = Move::new_drop(Square(37), PieceType::Gold);
        assert_eq!(m.raw_u32() & 0xFFFF, 11173);
        // hand_index: Gold=4 (cshogi HGOLD=4)
        assert_eq!(move_drop_hand_piece(m), 4);

        // Verify cshogi raw values decode correctly
        let m = Move::from_raw_u32(11088);
        assert!(m.is_drop());
        assert_eq!(m.drop_piece_type(), Some(PieceType::Rook));
        assert_eq!(m.to_sq(), Square(80));
    }
}
