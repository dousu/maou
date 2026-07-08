//! 指し手 → policy ラベル (0..1496) 変換．
//!
//! Python 実装 `src/maou/domain/move/label.py` の Rust 移植．NN の policy 出力
//! (1496 クラス) の index と指し手を対応付ける．ラベル体系は「移動方向 10 種 ×
//! 成/不成 + 持ち駒 7 種」を移動先座標と組み合わせ，盤の端で実現不可能な
//! 組み合わせを詰めた稠密な番号付け．
//!
//! 座標は手番視点に正規化する (後手番は `sq → 80 - sq`)．正規化後の
//! `x = sq / 9` (筋，0 = 1筋)，`y = sq % 9` (段，0 = 一段)．
//!
//! 移植の正しさは Python 正実装から生成した golden fixture
//! (`tests/fixtures/label_table_golden.txt` — 全 (from, to, promo) と全駒打ちの
//! 網羅表) との一致で検証している (`tests/parity.rs`)．

use maou_shogi::moves::Move;
use maou_shogi::types::Color;

/// policy ラベルの総数．
pub const MOVE_LABELS_NUM: usize = (start_label::HI + 81) as usize;

/// 各移動カテゴリの開始ラベル (Python `MoveCategoryStartLabel` と同一の算術)．
mod start_label {
    pub const UP: u16 = 0;
    // UP は 9 マス不要
    pub const UP_LEFT: u16 = UP + 81 - 9;
    pub const UP_RIGHT: u16 = UP_LEFT + 81 - 17;
    pub const LEFT: u16 = UP_RIGHT + 81 - 17;
    pub const RIGHT: u16 = LEFT + 81 - 9;
    pub const DOWN: u16 = RIGHT + 81 - 9;
    pub const DOWN_LEFT: u16 = DOWN + 81 - 9;
    pub const DOWN_RIGHT: u16 = DOWN_LEFT + 81 - 17;
    pub const KEIMA_LEFT: u16 = DOWN_RIGHT + 81 - 17;
    pub const KEIMA_RIGHT: u16 = KEIMA_LEFT + 81 - 41;
    pub const UP_PROMOTION: u16 = KEIMA_RIGHT + 81 - 41;
    pub const UP_LEFT_PROMOTION: u16 = UP_PROMOTION + 81 - 54;
    pub const UP_RIGHT_PROMOTION: u16 = UP_LEFT_PROMOTION + 81 - 57;
    pub const LEFT_PROMOTION: u16 = UP_RIGHT_PROMOTION + 81 - 57;
    pub const RIGHT_PROMOTION: u16 = LEFT_PROMOTION + 81 - 57;
    pub const DOWN_PROMOTION: u16 = RIGHT_PROMOTION + 81 - 57;
    pub const DOWN_LEFT_PROMOTION: u16 = DOWN_PROMOTION + 81 - 9;
    pub const DOWN_RIGHT_PROMOTION: u16 = DOWN_LEFT_PROMOTION + 81 - 32;
    pub const KEIMA_LEFT_PROMOTION: u16 = DOWN_RIGHT_PROMOTION + 81 - 32;
    pub const KEIMA_RIGHT_PROMOTION: u16 = KEIMA_LEFT_PROMOTION + 81 - 57;
    // 駒打ち: 歩香桂銀金角飛 (hand_index 順)
    pub const FU: u16 = KEIMA_RIGHT_PROMOTION + 81 - 57;
    pub const KY: u16 = FU + 81 - 9;
    pub const KE: u16 = KY + 81 - 9;
    pub const GI: u16 = KE + 81 - 18;
    pub const KI: u16 = GI + 81;
    pub const KA: u16 = KI + 81;
    pub const HI: u16 = KA + 81;
}

/// 指し手をラベルに変換する．合法手でない場合は `None`
/// (Python 実装の `IllegalMove` に対応)．
pub fn try_move_label(turn: Color, m: Move) -> Option<u16> {
    if m.is_drop() {
        let hand_piece = m.drop_piece_type()?.hand_index()? as u8;
        let to_sq = normalize(turn, m.to_sq().index() as u8);
        drop_move_label(to_sq, hand_piece)
    } else {
        let to_sq = normalize(turn, m.to_sq().index() as u8);
        let from_sq = normalize(turn, m.from_sq().index() as u8);
        board_move_label(to_sq, from_sq, m.is_promotion())
    }
}

/// 合法手をラベルに変換する．
///
/// # Panics
///
/// `m` が合法手でない場合 (合法手は必ずラベルを持つ)．
pub fn move_label(turn: Color, m: Move) -> u16 {
    try_move_label(turn, m)
        .unwrap_or_else(|| panic!("合法手はラベルに変換できるはず: {}", m.to_usi()))
}

/// 後手番なら座標を 180 度回転する．
#[inline]
fn normalize(turn: Color, sq: u8) -> u8 {
    match turn {
        Color::Black => sq,
        Color::White => 80 - sq,
    }
}

/// 盤上移動のラベル (座標は手番視点に正規化済み)．
fn board_move_label(to_sq: u8, from_sq: u8, is_promotion: bool) -> Option<u16> {
    let (to_x, to_y) = (i16::from(to_sq) / 9, i16::from(to_sq) % 9);
    let (from_x, from_y) = (i16::from(from_sq) / 9, i16::from(from_sq) % 9);
    let (dx, dy) = (to_x - from_x, to_y - from_y);
    let to_sq = i16::from(to_sq);

    let label = match (dx, dy) {
        (1, -2) => keima_left(to_sq, to_x, to_y, is_promotion)?,
        (-1, -2) => keima_right(to_sq, to_x, to_y, is_promotion)?,
        (0, y) if y < 0 => {
            // UP
            if to_y == 8 {
                return None;
            }
            if !is_promotion {
                start_label::UP as i16 + to_sq - to_x
            } else {
                if to_y >= 3 {
                    return None;
                }
                start_label::UP_PROMOTION as i16 + to_sq - to_x * 6
            }
        }
        (0, y) if y > 0 => {
            // DOWN
            if to_y == 0 {
                return None;
            }
            if !is_promotion {
                start_label::DOWN as i16 + to_sq - (to_x + 1)
            } else {
                start_label::DOWN_PROMOTION as i16 + to_sq - (to_x + 1)
            }
        }
        (x, 0) if x > 0 => {
            // LEFT
            if to_x == 0 {
                return None;
            }
            if !is_promotion {
                start_label::LEFT as i16 + to_sq - 9
            } else {
                if to_y >= 3 {
                    return None;
                }
                start_label::LEFT_PROMOTION as i16 + to_sq - to_x * 6 - 3
            }
        }
        (x, 0) if x < 0 => {
            // RIGHT
            if to_x == 8 {
                return None;
            }
            if !is_promotion {
                start_label::RIGHT as i16 + to_sq
            } else {
                if to_y >= 3 {
                    return None;
                }
                start_label::RIGHT_PROMOTION as i16 + to_sq - to_x * 6
            }
        }
        (x, y) if x > 0 && y < 0 => {
            // UP_LEFT
            if to_y == 8 || to_x == 0 {
                return None;
            }
            if !is_promotion {
                start_label::UP_LEFT as i16 + to_sq - to_x - 8
            } else {
                if to_y >= 3 {
                    return None;
                }
                start_label::UP_LEFT_PROMOTION as i16 + to_sq - to_x * 6 - 3
            }
        }
        (x, y) if x < 0 && y < 0 => {
            // UP_RIGHT
            if to_y == 8 || to_x == 8 {
                return None;
            }
            if !is_promotion {
                start_label::UP_RIGHT as i16 + to_sq - to_x
            } else {
                if to_y >= 3 {
                    return None;
                }
                start_label::UP_RIGHT_PROMOTION as i16 + to_sq - to_x * 6
            }
        }
        (x, y) if x > 0 && y > 0 => {
            // DOWN_LEFT
            if to_y == 0 || to_x == 0 {
                return None;
            }
            if !is_promotion {
                start_label::DOWN_LEFT as i16 + to_sq - to_x - 9
            } else {
                if 8 - to_y + to_x < 6 {
                    return None;
                }
                // sum(range(7 - to_x, 7)) (to_x >= 6 は 21)
                let range_sum = match to_x {
                    0 => 0,
                    1 => 6,
                    2 => 11,
                    3 => 15,
                    4 => 18,
                    5 => 20,
                    _ => 21,
                };
                start_label::DOWN_LEFT_PROMOTION as i16 + to_sq - (to_x + 1) - 2 - range_sum
            }
        }
        (x, y) if x < 0 && y > 0 => {
            // DOWN_RIGHT
            if to_y == 0 || to_x == 8 {
                return None;
            }
            if !is_promotion {
                start_label::DOWN_RIGHT as i16 + to_sq - (to_x + 1)
            } else {
                if 8 - to_y + 8 - to_x < 6 {
                    return None;
                }
                // sum(range(0, to_x - 2)) (to_x <= 2 は 0)
                let range_sum = match to_x {
                    3 => 0,
                    4 => 1,
                    5 => 3,
                    6 => 6,
                    7 => 10,
                    8 => 15,
                    _ => 0,
                };
                start_label::DOWN_RIGHT_PROMOTION as i16 + to_sq - (to_x + 1) - range_sum
            }
        }
        _ => return None,
    };
    debug_assert!((0..MOVE_LABELS_NUM as i16).contains(&label));
    Some(label as u16)
}

/// 桂馬 (左) のラベル．
fn keima_left(to_sq: i16, to_x: i16, to_y: i16, is_promotion: bool) -> Option<i16> {
    if to_y > 6 || to_x == 0 {
        return None;
    }
    if !is_promotion {
        if to_y < 2 {
            return None;
        }
        Some(start_label::KEIMA_LEFT as i16 + to_sq - (to_x + 1) * 2 - to_x * 2 - 5)
    } else {
        if to_y >= 3 {
            return None;
        }
        Some(start_label::KEIMA_LEFT_PROMOTION as i16 + to_sq - to_x * 6 - 3)
    }
}

/// 桂馬 (右) のラベル．
fn keima_right(to_sq: i16, to_x: i16, to_y: i16, is_promotion: bool) -> Option<i16> {
    if to_y > 6 || to_x == 8 {
        return None;
    }
    if !is_promotion {
        if to_y < 2 {
            return None;
        }
        Some(start_label::KEIMA_RIGHT as i16 + to_sq - (to_x + 1) * 2 - to_x * 2)
    } else {
        if to_y >= 3 {
            return None;
        }
        Some(start_label::KEIMA_RIGHT_PROMOTION as i16 + to_sq - to_x * 6)
    }
}

/// 駒打ちのラベル (座標は正規化済み，hand_piece は 歩0 香1 桂2 銀3 金4 角5 飛6)．
fn drop_move_label(to_sq: u8, hand_piece: u8) -> Option<u16> {
    let (to_x, to_y) = (i16::from(to_sq) / 9, i16::from(to_sq) % 9);
    let to_sq = i16::from(to_sq);
    let (base, offset) = match hand_piece {
        0 | 1 => {
            // 歩・香: 一段目に打てない
            if to_y == 0 {
                return None;
            }
            let base = if hand_piece == 0 {
                start_label::FU
            } else {
                start_label::KY
            };
            (base, to_sq - (to_x + 1))
        }
        2 => {
            // 桂: 一・二段目に打てない
            if to_y < 2 {
                return None;
            }
            (start_label::KE, to_sq - (to_x + 1) * 2)
        }
        3 => (start_label::GI, to_sq),
        4 => (start_label::KI, to_sq),
        5 => (start_label::KA, to_sq),
        6 => (start_label::HI, to_sq),
        _ => return None,
    };
    let label = base as i16 + offset;
    debug_assert!((0..MOVE_LABELS_NUM as i16).contains(&label));
    Some(label as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_labels_num_matches_python() {
        // Python 側 MOVE_LABELS_NUM = 1496 と一致すること
        assert_eq!(MOVE_LABELS_NUM, 1496);
    }
}
