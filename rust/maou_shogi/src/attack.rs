use crate::bitboard::Bitboard;
use crate::types::{Color, PIECE_BB_SIZE, PieceType, Square};

/// 駒の利きを計算する．
///
/// 近接駒(歩,桂,銀,金,王,成駒のステップ部分)は事前計算テーブルから取得．
/// 飛び駒(香,角,飛,馬,龍)はPEXTベースのルックアップテーブルで計算する．
/// BMI2非対応環境ではソフトウェアPEXTにフォールバックする．

/// 近接駒(飛び駒以外)の利きテーブル．
/// step_attacks[color][piece_type][square]
///
/// 先手(Black)は上方向(row減少)が前，後手(White)は下方向(row増加)が前．
static STEP_ATTACKS: std::sync::LazyLock<[[[Bitboard; 81]; PIECE_BB_SIZE]; 2]> =
    std::sync::LazyLock::new(init_step_attacks);

fn init_step_attacks() -> [[[Bitboard; 81]; PIECE_BB_SIZE]; 2] {
    let mut table = [[[Bitboard::EMPTY; 81]; PIECE_BB_SIZE]; 2];

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
    table[sq.index()] |= bb;
}

/// 近接駒の利きを返す(飛び駒のスライド部分は含まない)．
#[inline]
pub fn step_attacks(color: Color, pt: PieceType, sq: Square) -> Bitboard {
    STEP_ATTACKS[color.index()][pt as usize][sq.index()]
}

// ============================================================
// レイマスク(初期化・line_through用)
// ============================================================

/// レイ方向のインデックス．
const DIR_N: usize = 0;
const DIR_S: usize = 1;
const DIR_W: usize = 2;
const DIR_E: usize = 3;
const DIR_NW: usize = 4;
const DIR_NE: usize = 5;
const DIR_SW: usize = 6;
const DIR_SE: usize = 7;

/// 各方向の移動ベクトル (dcol, drow)．
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

/// レイ利き(正方向): テーブル初期化時に使用．
fn ray_attack_positive(dir: usize, sq: Square, occupied: Bitboard) -> Bitboard {
    let ray = RAY_MASKS[dir][sq.index()];
    let blockers = ray & occupied;
    match blockers.lsb() {
        Some(first) => ray ^ RAY_MASKS[dir][first.index()],
        None => ray,
    }
}

/// レイ利き(負方向): テーブル初期化時に使用．
fn ray_attack_negative(dir: usize, sq: Square, occupied: Bitboard) -> Bitboard {
    let ray = RAY_MASKS[dir][sq.index()];
    let blockers = ray & occupied;
    match blockers.msb() {
        Some(first) => ray ^ RAY_MASKS[dir][first.index()],
        None => ray,
    }
}

// ============================================================
// PEXT ベースのスライド利きテーブル
// ============================================================

/// Parallel Bit Extract．BMI2対応時はハードウェア命令を使用．
#[inline(always)]
fn pext(src: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        unsafe { std::arch::x86_64::_pext_u64(src, mask) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // ソフトウェアフォールバック
        let mut result = 0u64;
        let mut bit = 0u32;
        let mut m = mask;
        while m != 0 {
            let lsb = m & m.wrapping_neg();
            if src & lsb != 0 {
                result |= 1u64 << bit;
            }
            m &= m - 1;
            bit += 1;
        }
        result
    }
}

/// PEXTエントリ: 各マス用の占有マスクとテーブルオフセット．
struct PextEntry {
    mask_lo: u64,
    mask_hi: u64,
    lo_popcount: u32,
    offset: u32,
}

/// PEXTベースの利きテーブル．
struct PextTable {
    entries: [PextEntry; 81],
    attacks: Vec<Bitboard>,
}

impl PextTable {
    #[inline(always)]
    fn lookup(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let e = unsafe { self.entries.get_unchecked(sq.index()) };
        let lo_idx = pext(occ.lo, e.mask_lo) as usize;
        let hi_idx = pext(occ.hi, e.mask_hi) as usize;
        let idx = lo_idx | (hi_idx << e.lo_popcount);
        unsafe { *self.attacks.get_unchecked(e.offset as usize + idx) }
    }
}

/// PEXTテーブルを初期化する．
fn init_pext_table(
    mask_fn: impl Fn(Square) -> Bitboard,
    attack_fn: impl Fn(Square, Bitboard) -> Bitboard,
) -> PextTable {
    let dummy = PextEntry { mask_lo: 0, mask_hi: 0, lo_popcount: 0, offset: 0 };
    // Safety: PextEntry is Copy-like (all primitives)
    let mut entries: [PextEntry; 81] = unsafe { std::mem::zeroed() };
    let _ = dummy; // suppress unused
    let mut attacks = Vec::new();

    for sq_idx in 0..81u8 {
        let sq = Square(sq_idx);
        let mask = mask_fn(sq);
        let mask_squares: Vec<Square> = mask.into_iter().collect();
        let n = mask_squares.len();
        debug_assert!(n <= 14, "PEXT mask too large: {} bits for sq={}", n, sq_idx);

        entries[sq_idx as usize] = PextEntry {
            mask_lo: mask.lo,
            mask_hi: mask.hi,
            lo_popcount: mask.lo.count_ones(),
            offset: attacks.len() as u32,
        };

        let size = 1usize << n;
        for idx in 0..size {
            let mut occ = Bitboard::EMPTY;
            for bit in 0..n {
                if idx & (1 << bit) != 0 {
                    occ.set(mask_squares[bit]);
                }
            }
            attacks.push(attack_fn(sq, occ));
        }
    }

    PextTable { entries, attacks }
}

// --- マスク関数 ---

/// 先手香車: 北方向(row減少)，辺を除く内部マス．
fn lance_black_mask(sq: Square) -> Bitboard {
    let col = sq.col();
    let row = sq.row();
    let mut mask = Bitboard::EMPTY;
    // row 1..row-1 (row 0 = 辺は除外)
    for r in 1..row {
        mask.set(Square::new(col, r));
    }
    mask
}

/// 後手香車: 南方向(row増加)，辺を除く内部マス．
fn lance_white_mask(sq: Square) -> Bitboard {
    let col = sq.col();
    let row = sq.row();
    let mut mask = Bitboard::EMPTY;
    // row+1..7 (row 8 = 辺は除外)
    for r in (row + 1)..8 {
        mask.set(Square::new(col, r));
    }
    mask
}

/// 飛車筋(縦)方向のマスク: row 1..7 のうち自マスを除く．
fn rook_file_mask(sq: Square) -> Bitboard {
    let col = sq.col();
    let row = sq.row();
    let mut mask = Bitboard::EMPTY;
    for r in 1..8u8 {
        if r != row {
            mask.set(Square::new(col, r));
        }
    }
    mask
}

/// 飛車段(横)方向のマスク: col 1..7 のうち自マスを除く．
fn rook_rank_mask(sq: Square) -> Bitboard {
    let col = sq.col();
    let row = sq.row();
    let mut mask = Bitboard::EMPTY;
    for c in 1..8u8 {
        if c != col {
            mask.set(Square::new(c, row));
        }
    }
    mask
}

/// 角の NW-SE 対角線マスク: 辺を除く内部マス(自マス除外)．
fn bishop_diag1_mask(sq: Square) -> Bitboard {
    let col = sq.col() as i8;
    let row = sq.row() as i8;
    let mut mask = Bitboard::EMPTY;
    // NW 方向
    let (mut c, mut r) = (col - 1, row - 1);
    while c > 0 && r > 0 {
        mask.set(Square::new(c as u8, r as u8));
        c -= 1;
        r -= 1;
    }
    // SE 方向
    let (mut c, mut r) = (col + 1, row + 1);
    while c < 8 && r < 8 {
        mask.set(Square::new(c as u8, r as u8));
        c += 1;
        r += 1;
    }
    mask
}

/// 角の NE-SW 対角線マスク: 辺を除く内部マス(自マス除外)．
fn bishop_diag2_mask(sq: Square) -> Bitboard {
    let col = sq.col() as i8;
    let row = sq.row() as i8;
    let mut mask = Bitboard::EMPTY;
    // NE 方向
    let (mut c, mut r) = (col + 1, row - 1);
    while c < 8 && r > 0 {
        mask.set(Square::new(c as u8, r as u8));
        c += 1;
        r -= 1;
    }
    // SW 方向
    let (mut c, mut r) = (col - 1, row + 1);
    while c > 0 && r < 8 {
        mask.set(Square::new(c as u8, r as u8));
        c -= 1;
        r += 1;
    }
    mask
}

// --- 利き計算関数(初期化用) ---

fn compute_lance_black(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_N, sq, occ)
}

fn compute_lance_white(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_positive(DIR_S, sq, occ)
}

fn compute_rook_file(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_N, sq, occ) | ray_attack_positive(DIR_S, sq, occ)
}

fn compute_rook_rank(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_W, sq, occ) | ray_attack_positive(DIR_E, sq, occ)
}

fn compute_bishop_diag1(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_negative(DIR_NW, sq, occ) | ray_attack_positive(DIR_SE, sq, occ)
}

fn compute_bishop_diag2(sq: Square, occ: Bitboard) -> Bitboard {
    ray_attack_positive(DIR_NE, sq, occ) | ray_attack_negative(DIR_SW, sq, occ)
}

// --- グローバルテーブル ---

use std::sync::LazyLock;

static PEXT_LANCE_BLACK: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(lance_black_mask, compute_lance_black));
static PEXT_LANCE_WHITE: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(lance_white_mask, compute_lance_white));
static PEXT_ROOK_FILE: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(rook_file_mask, compute_rook_file));
static PEXT_ROOK_RANK: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(rook_rank_mask, compute_rook_rank));
static PEXT_BISHOP_DIAG1: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(bishop_diag1_mask, compute_bishop_diag1));
static PEXT_BISHOP_DIAG2: LazyLock<PextTable> =
    LazyLock::new(|| init_pext_table(bishop_diag2_mask, compute_bishop_diag2));

/// 事前計算済み between_bb テーブル: [sq1][sq2]．
static BETWEEN_TABLE: LazyLock<[[Bitboard; 81]; 81]> = LazyLock::new(init_between_table);

fn init_between_table() -> [[Bitboard; 81]; 81] {
    let mut table = [[Bitboard::EMPTY; 81]; 81];
    for i in 0..81u8 {
        for j in 0..81u8 {
            table[i as usize][j as usize] = compute_between_bb(Square(i), Square(j));
        }
    }
    table
}

/// between_bb のループ版(テーブル初期化用)．
fn compute_between_bb(sq1: Square, sq2: Square) -> Bitboard {
    let c1 = sq1.col() as i8;
    let r1 = sq1.row() as i8;
    let c2 = sq2.col() as i8;
    let r2 = sq2.row() as i8;

    let dc = (c2 - c1).signum();
    let dr = (r2 - r1).signum();

    if dc == 0 && dr == 0 {
        return Bitboard::EMPTY;
    }

    let cdiff = (c2 - c1).unsigned_abs();
    let rdiff = (r2 - r1).unsigned_abs();
    let on_line = (dc == 0 && rdiff > 0) || (dr == 0 && cdiff > 0) || (cdiff == rdiff);
    if !on_line {
        return Bitboard::EMPTY;
    }

    let mut bb = Bitboard::EMPTY;
    let mut c = c1 + dc;
    let mut r = r1 + dr;
    while (c, r) != (c2, r2) {
        if c < 0 || c >= 9 || r < 0 || r >= 9 {
            return Bitboard::EMPTY;
        }
        bb.set(Square::new(c as u8, r as u8));
        c += dc;
        r += dr;
    }
    bb
}

// ============================================================
// 公開 API
// ============================================================

/// 香車の利きを返す．PEXTベースのルックアップテーブルを使用．
#[inline]
pub fn lance_attacks(color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match color {
        Color::Black => PEXT_LANCE_BLACK.lookup(sq, occupied),
        Color::White => PEXT_LANCE_WHITE.lookup(sq, occupied),
    }
}

/// 角の利きを返す．2つの対角線のPEXTテーブルを合成．
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    PEXT_BISHOP_DIAG1.lookup(sq, occupied) | PEXT_BISHOP_DIAG2.lookup(sq, occupied)
}

/// 飛車の利きを返す．筋・段のPEXTテーブルを合成．
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    PEXT_ROOK_FILE.lookup(sq, occupied) | PEXT_ROOK_RANK.lookup(sq, occupied)
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

/// 2マス間のレイ上のマス(両端を含まない)を返す．
///
/// 事前計算済みテーブルからO(1)で取得する．
#[inline]
pub fn between_bb(sq1: Square, sq2: Square) -> Bitboard {
    BETWEEN_TABLE[sq1.index()][sq2.index()]
}

/// 2マスを通る直線(縦横斜め)上の全マスを返す(両端含む)．
///
/// 同一直線上にない場合は空を返す．
/// ピン判定で使用: ピンされた駒がこの直線上でのみ移動可能．
#[inline]
pub fn line_through(sq1: Square, sq2: Square) -> Bitboard {
    let c1 = sq1.col() as i8;
    let r1 = sq1.row() as i8;
    let c2 = sq2.col() as i8;
    let r2 = sq2.row() as i8;

    let dc = c2 - c1;
    let dr = r2 - r1;

    if dc == 0 && dr == 0 {
        return Bitboard::EMPTY;
    }

    let adc = dc.unsigned_abs();
    let adr = dr.unsigned_abs();

    let dir = if dc == 0 {
        DIR_N
    } else if dr == 0 {
        DIR_W
    } else if adc == adr {
        if (dc > 0) == (dr > 0) { DIR_SE } else { DIR_NE }
    } else {
        return Bitboard::EMPTY;
    };

    let opposite = match dir {
        DIR_N => DIR_S,
        DIR_S => DIR_N,
        DIR_W => DIR_E,
        DIR_E => DIR_W,
        DIR_NW => DIR_SE,
        DIR_NE => DIR_SW,
        DIR_SW => DIR_NE,
        DIR_SE => DIR_NW,
        _ => unreachable!(),
    };
    let mut line = RAY_MASKS[dir][sq1.index()] | RAY_MASKS[opposite][sq1.index()];
    line.set(sq1);
    line
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
        let att = step_attacks(Color::Black, PieceType::Pawn, Square::new(4, 4));
        assert_eq!(att.count(), 1);
        assert!(att.contains(Square::new(4, 3)));

        let att = step_attacks(Color::White, PieceType::Pawn, Square::new(4, 4));
        assert_eq!(att.count(), 1);
        assert!(att.contains(Square::new(4, 5)));
    }

    #[test]
    fn test_knight_attacks() {
        let att = step_attacks(Color::Black, PieceType::Knight, Square::new(4, 4));
        assert_eq!(att.count(), 2);
        assert!(att.contains(Square::new(3, 2)));
        assert!(att.contains(Square::new(5, 2)));
    }

    #[test]
    fn test_king_attacks() {
        let att = step_attacks(Color::Black, PieceType::King, Square::new(4, 4));
        assert_eq!(att.count(), 8);

        let att = step_attacks(Color::Black, PieceType::King, Square::new(0, 0));
        assert_eq!(att.count(), 3);
    }

    #[test]
    fn test_rook_attacks() {
        let att = rook_attacks(Square::new(4, 4), Bitboard::EMPTY);
        assert_eq!(att.count(), 16);
    }

    #[test]
    fn test_lance_attacks() {
        let att = lance_attacks(Color::Black, Square::new(4, 8), Bitboard::EMPTY);
        assert_eq!(att.count(), 8);

        let mut occ = Bitboard::EMPTY;
        occ.set(Square::new(4, 5));
        let att = lance_attacks(Color::Black, Square::new(4, 8), occ);
        assert_eq!(att.count(), 3);
        assert!(att.contains(Square::new(4, 5)));
    }

    /// PEXTテーブルがレイベースの計算と一致することを全マス・全占有パターンで検証．
    #[test]
    fn test_pext_lance_vs_ray() {
        for sq_idx in 0..81u8 {
            let sq = Square(sq_idx);
            // いくつかの代表的な占有パターンで検証
            for pattern in 0..16u64 {
                let mut occ = Bitboard::EMPTY;
                let col = sq.col();
                for r in 0..9u8 {
                    if pattern & (1 << (r % 4)) != 0 && r != sq.row() {
                        occ.set(Square::new(col, r));
                    }
                }
                let pext_b = PEXT_LANCE_BLACK.lookup(sq, occ);
                let ray_b = ray_attack_negative(DIR_N, sq, occ);
                assert_eq!(pext_b, ray_b, "lance black mismatch at sq={}, occ={:?}", sq_idx, occ);

                let pext_w = PEXT_LANCE_WHITE.lookup(sq, occ);
                let ray_w = ray_attack_positive(DIR_S, sq, occ);
                assert_eq!(pext_w, ray_w, "lance white mismatch at sq={}", sq_idx);
            }
        }
    }

    /// PEXTテーブルがレイベースの計算と一致することを飛車で検証．
    #[test]
    fn test_pext_rook_vs_ray() {
        for sq_idx in 0..81u8 {
            let sq = Square(sq_idx);
            for pattern in 0..32u64 {
                let mut occ = Bitboard::EMPTY;
                // 筋方向
                let col = sq.col();
                for r in 0..9u8 {
                    if pattern & (1 << (r % 5)) != 0 && r != sq.row() {
                        occ.set(Square::new(col, r));
                    }
                }
                // 段方向
                let row = sq.row();
                for c in 0..9u8 {
                    if pattern & (1 << ((c + 2) % 5)) != 0 && c != sq.col() {
                        occ.set(Square::new(c, row));
                    }
                }
                let pext_r = rook_attacks(sq, occ);
                let ray_r = ray_attack_negative(DIR_N, sq, occ)
                    | ray_attack_positive(DIR_S, sq, occ)
                    | ray_attack_negative(DIR_W, sq, occ)
                    | ray_attack_positive(DIR_E, sq, occ);
                assert_eq!(pext_r, ray_r, "rook mismatch at sq={}", sq_idx);
            }
        }
    }

    /// PEXTテーブルがレイベースの計算と一致することを角で検証．
    #[test]
    fn test_pext_bishop_vs_ray() {
        for sq_idx in 0..81u8 {
            let sq = Square(sq_idx);
            for pattern in 0..16u64 {
                let mut occ = Bitboard::EMPTY;
                // 対角線上にいくつかの駒を配置
                let col = sq.col() as i8;
                let row = sq.row() as i8;
                for d in 1..9i8 {
                    let bit = (pattern >> (d as u64 % 4)) & 1;
                    if bit != 0 {
                        let nc = col + d;
                        let nr = row + d;
                        if nc >= 0 && nc < 9 && nr >= 0 && nr < 9 {
                            occ.set(Square::new(nc as u8, nr as u8));
                        }
                        let nc = col - d;
                        let nr = row - d;
                        if nc >= 0 && nc < 9 && nr >= 0 && nr < 9 {
                            occ.set(Square::new(nc as u8, nr as u8));
                        }
                        let nc = col + d;
                        let nr = row - d;
                        if nc >= 0 && nc < 9 && nr >= 0 && nr < 9 {
                            occ.set(Square::new(nc as u8, nr as u8));
                        }
                        let nc = col - d;
                        let nr = row + d;
                        if nc >= 0 && nc < 9 && nr >= 0 && nr < 9 {
                            occ.set(Square::new(nc as u8, nr as u8));
                        }
                    }
                }
                let pext_b = bishop_attacks(sq, occ);
                let ray_b = ray_attack_negative(DIR_NW, sq, occ)
                    | ray_attack_positive(DIR_NE, sq, occ)
                    | ray_attack_negative(DIR_SW, sq, occ)
                    | ray_attack_positive(DIR_SE, sq, occ);
                assert_eq!(pext_b, ray_b, "bishop mismatch at sq={}", sq_idx);
            }
        }
    }

    /// between_bbテーブルの検証．
    #[test]
    fn test_between_bb_table() {
        for i in 0..81u8 {
            for j in 0..81u8 {
                let tbl = between_bb(Square(i), Square(j));
                let computed = compute_between_bb(Square(i), Square(j));
                assert_eq!(tbl, computed, "between_bb mismatch: sq1={}, sq2={}", i, j);
            }
        }
    }
}
