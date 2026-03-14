use crate::bitboard::Bitboard;
use crate::types::{Color, PIECE_BB_SIZE, PieceType, Square};

/// 駒の利きを計算する．
///
/// 近接駒(歩,桂,銀,金,王,成駒のステップ部分)は事前計算テーブルから取得．
/// 飛び駒(香,角,飛,馬,龍)は事前計算レイマスクとLSB/MSBによる
/// 遮蔽検出で計算する(ループなし)．

/// 近接駒(飛び駒以外)の利きテーブル．
/// step_attacks[color][piece_type][square]
///
/// 先手(Black)は上方向(row減少)が前，後手(White)は下方向(row増加)が前．
static STEP_ATTACKS: std::sync::LazyLock<[[[Bitboard; 81]; PIECE_BB_SIZE]; 2]> =
    std::sync::LazyLock::new(init_step_attacks);

fn init_step_attacks() -> [[[Bitboard; 81]; PIECE_BB_SIZE]; 2] {
    let mut table = [[[Bitboard::EMPTY; 81]; 15]; 2];

    for sq_idx in 0..81u8 {
        let sq = Square(sq_idx);

        // 各駒種の方向ベクトル (dcol, drow)
        // 先手の場合(前=row減少方向)

        // 歩: 前1マス
        add_step(&mut table[0][PieceType::Pawn as usize], sq, &[(0, -1)]);
        add_step(&mut table[1][PieceType::Pawn as usize], sq, &[(0, 1)]);

        // 桂: 前方2マス先の左右
        add_step(
            &mut table[0][PieceType::Knight as usize],
            sq,
            &[(-1, -2), (1, -2)],
        );
        add_step(
            &mut table[1][PieceType::Knight as usize],
            sq,
            &[(-1, 2), (1, 2)],
        );

        // 銀: 前3方向+斜め後ろ2方向
        add_step(
            &mut table[0][PieceType::Silver as usize],
            sq,
            &[(-1, -1), (0, -1), (1, -1), (-1, 1), (1, 1)],
        );
        add_step(
            &mut table[1][PieceType::Silver as usize],
            sq,
            &[(-1, 1), (0, 1), (1, 1), (-1, -1), (1, -1)],
        );

        // 金(と金,成香,成桂,成銀も同じ): 前後左右+斜め前2方向
        let gold_dirs_black: [(i8, i8); 6] =
            [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)];
        let gold_dirs_white: [(i8, i8); 6] =
            [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (0, -1)];

        for &pt in &[
            PieceType::Gold,
            PieceType::ProPawn,
            PieceType::ProLance,
            PieceType::ProKnight,
            PieceType::ProSilver,
        ] {
            add_step(
                &mut table[0][pt as usize],
                sq,
                &gold_dirs_black,
            );
            add_step(
                &mut table[1][pt as usize],
                sq,
                &gold_dirs_white,
            );
        }

        // 王: 全8方向
        let king_dirs: [(i8, i8); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        // 王は色に依存しない
        add_step(&mut table[0][PieceType::King as usize], sq, &king_dirs);
        add_step(&mut table[1][PieceType::King as usize], sq, &king_dirs);

        // 馬(成角): 斜め走りは別途処理，前後左右1マスのステップ部分のみ
        let horse_step: [(i8, i8); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        add_step(
            &mut table[0][PieceType::Horse as usize],
            sq,
            &horse_step,
        );
        add_step(
            &mut table[1][PieceType::Horse as usize],
            sq,
            &horse_step,
        );

        // 龍(成飛): 前後左右走りは別途処理，斜め1マスのステップ部分のみ
        let dragon_step: [(i8, i8); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
        add_step(
            &mut table[0][PieceType::Dragon as usize],
            sq,
            &dragon_step,
        );
        add_step(
            &mut table[1][PieceType::Dragon as usize],
            sq,
            &dragon_step,
        );
    }

    table
}

fn add_step(table: &mut [Bitboard; 81], sq: Square, dirs: &[(i8, i8)]) {
    let col = sq.col() as i8;
    let row = sq.row() as i8;
    let mut bb = Bitboard::EMPTY;
    for &(dc, dr) in dirs {
        let nc = col + dc;
        let nr = row + dr;
        if nc >= 0 && nc < 9 && nr >= 0 && nr < 9 {
            bb.set(Square::new(nc as u8, nr as u8));
        }
    }
    table[sq.index()] = table[sq.index()] | bb;
}

/// 近接駒の利きを返す(飛び駒のスライド部分は含まない)．
#[inline]
pub fn step_attacks(color: Color, pt: PieceType, sq: Square) -> Bitboard {
    STEP_ATTACKS[color.index()][pt as usize][sq.index()]
}

// ============================================================
// スライド利きの事前計算
// ============================================================

/// レイ方向のインデックス．
/// column-major (sq = col*9 + row) でのインデックス増減:
///   正方向(LSB): S(+1), E(+9), NE(+8), SE(+10)
///   負方向(MSB): N(-1), W(-9), NW(-10), SW(-8)
const DIR_N: usize = 0;
const DIR_S: usize = 1;
const DIR_W: usize = 2;
const DIR_E: usize = 3;
const DIR_NW: usize = 4;
const DIR_NE: usize = 5;
const DIR_SW: usize = 6;
const DIR_SE: usize = 7;

/// 各方向の移動ベクトル (dcol, drow)．DIR_* インデックスと対応．
const DIR_VECTORS: [(i8, i8); 8] = [
    (0, -1),  // N
    (0, 1),   // S
    (-1, 0),  // W
    (1, 0),   // E
    (-1, -1), // NW
    (1, -1),  // NE
    (-1, 1),  // SW
    (1, 1),   // SE
];

/// レイマスク: 各マスから各方向に伸びるレイ(起点は含まない)．
/// RAY_MASKS[direction][square]
static RAY_MASKS: std::sync::LazyLock<[[Bitboard; 81]; 8]> =
    std::sync::LazyLock::new(init_ray_masks);

fn init_ray_masks() -> [[Bitboard; 81]; 8] {
    let mut table = [[Bitboard::EMPTY; 81]; 8];
    for dir_idx in 0..8 {
        let (dc, dr) = DIR_VECTORS[dir_idx];
        for sq_idx in 0..81u8 {
            let col = (sq_idx / 9) as i8;
            let row = (sq_idx % 9) as i8;
            let mut bb = Bitboard::EMPTY;
            let mut c = col + dc;
            let mut r = row + dr;
            while c >= 0 && c < 9 && r >= 0 && r < 9 {
                bb.set(Square::new(c as u8, r as u8));
                c += dc;
                r += dr;
            }
            table[dir_idx][sq_idx as usize] = bb;
        }
    }
    table
}

/// 正方向(インデックス増加)のレイ利きを返す．最初の遮蔽駒をLSBで検出する．
#[inline]
fn ray_attack_positive(dir: usize, sq: Square, occupied: Bitboard) -> Bitboard {
    let ray = RAY_MASKS[dir][sq.index()];
    let blockers = ray & occupied;
    match blockers.lsb() {
        Some(first) => ray ^ RAY_MASKS[dir][first.index()],
        None => ray,
    }
}

/// 負方向(インデックス減少)のレイ利きを返す．最初の遮蔽駒をMSBで検出する．
#[inline]
fn ray_attack_negative(dir: usize, sq: Square, occupied: Bitboard) -> Bitboard {
    let ray = RAY_MASKS[dir][sq.index()];
    let blockers = ray & occupied;
    match blockers.msb() {
        Some(first) => ray ^ RAY_MASKS[dir][first.index()],
        None => ray,
    }
}

/// 香車の利きを返す．事前計算レイマスクとLSB/MSBによる遮蔽検出を使用．
#[inline]
pub fn lance_attacks(color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match color {
        Color::Black => ray_attack_negative(DIR_N, sq, occupied),
        Color::White => ray_attack_positive(DIR_S, sq, occupied),
    }
}

/// 角の利きを返す．4方向のレイ利きを合成する．
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_NW, sq, occupied)
        | ray_attack_positive(DIR_NE, sq, occupied)
        | ray_attack_negative(DIR_SW, sq, occupied)
        | ray_attack_positive(DIR_SE, sq, occupied)
}

/// 飛車の利きを返す．4方向のレイ利きを合成する．
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_N, sq, occupied)
        | ray_attack_positive(DIR_S, sq, occupied)
        | ray_attack_negative(DIR_W, sq, occupied)
        | ray_attack_positive(DIR_E, sq, occupied)
}

/// 馬(成角)の利きを返す．斜め走り + 前後左右1マス．
#[inline]
pub fn horse_attacks(color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | step_attacks(color, PieceType::Horse, sq)
}

/// 龍(成飛)の利きを返す．前後左右走り + 斜め1マス．
#[inline]
pub fn dragon_attacks(color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | step_attacks(color, PieceType::Dragon, sq)
}

/// 指定した駒種・色・マスの利きを返す(占有ビットボード考慮)．
pub fn piece_attacks(color: Color, pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Lance => lance_attacks(color, sq, occupied),
        PieceType::Bishop => bishop_attacks(sq, occupied),
        PieceType::Rook => rook_attacks(sq, occupied),
        PieceType::Horse => horse_attacks(color, sq, occupied),
        PieceType::Dragon => dragon_attacks(color, sq, occupied),
        _ => step_attacks(color, pt, sq),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pawn_attacks() {
        // 先手の歩(5五 = col4,row4)の利き → 5四(col4,row3)
        let att = step_attacks(Color::Black, PieceType::Pawn, Square::new(4, 4));
        assert_eq!(att.count(), 1);
        assert!(att.contains(Square::new(4, 3)));

        // 後手の歩の利き → 5六(col4,row5)
        let att = step_attacks(Color::White, PieceType::Pawn, Square::new(4, 4));
        assert_eq!(att.count(), 1);
        assert!(att.contains(Square::new(4, 5)));
    }

    #[test]
    fn test_knight_attacks() {
        // 先手の桂(5五 = col4,row4)の利き → (col3,row2) and (col5,row2)
        let att = step_attacks(Color::Black, PieceType::Knight, Square::new(4, 4));
        assert_eq!(att.count(), 2);
        assert!(att.contains(Square::new(3, 2)));
        assert!(att.contains(Square::new(5, 2)));
    }

    #[test]
    fn test_king_attacks() {
        // 中央の王(5五)の利き → 8方向
        let att = step_attacks(Color::Black, PieceType::King, Square::new(4, 4));
        assert_eq!(att.count(), 8);

        // 角の王(1一 = col0,row0)の利き → 3方向
        let att = step_attacks(Color::Black, PieceType::King, Square::new(0, 0));
        assert_eq!(att.count(), 3);
    }

    #[test]
    fn test_rook_attacks() {
        // 空盤で飛車(5五)の利き → 上下左右合わせて 8+8-2=16マス (自分のマスを含まない)
        let att = rook_attacks(Square::new(4, 4), Bitboard::EMPTY);
        // col4: row 0-3, 5-8 = 8マス + row4: col 0-3, 5-8 = 8マス = 16
        assert_eq!(att.count(), 16);
    }

    #[test]
    fn test_lance_attacks() {
        // 先手の香(5九 = col4,row8)，空盤 → 上方向に8マス
        let att = lance_attacks(Color::Black, Square::new(4, 8), Bitboard::EMPTY);
        assert_eq!(att.count(), 8);

        // 障害物があると止まる
        let mut occ = Bitboard::EMPTY;
        occ.set(Square::new(4, 5)); // 5六に駒
        let att = lance_attacks(Color::Black, Square::new(4, 8), occ);
        assert_eq!(att.count(), 3); // row 7, 6, 5
        assert!(att.contains(Square::new(4, 5))); // 障害物のマスも含む
    }
}
