use crate::board::Board;
use crate::types::{Color, PieceType, Square, FEATURES_NUM};

/// piece_planes (104×9×9) を生成する(先手視点)．
///
/// チャネル配置(cshogi互換):
/// [0-7]: 先手盤上駒 (歩,香,桂,銀,角,飛,金,王)
/// [8-13]: 先手成駒 (と,成香,成桂,成銀,馬,龍)
/// [14-21]: 後手盤上駒
/// [22-27]: 後手成駒
/// [28-65]: 先手持ち駒 (歩×18, 香×4, 桂×4, 銀×4, 金×4, 角×2, 飛×2)
/// [66-103]: 後手持ち駒
pub fn piece_planes(board: &Board, buf: &mut [f32]) {
    assert_eq!(buf.len(), FEATURES_NUM * 9 * 9);
    buf.fill(0.0);

    // 盤上の駒
    for sq_idx in 0..81u8 {
        let piece = board.squares[sq_idx as usize];
        if piece.is_empty() {
            continue;
        }
        let color = piece.color().unwrap();
        let pt = piece.piece_type().unwrap();

        let channel = board_piece_channel(color, pt);
        let sq = Square(sq_idx);
        // cshogi互換のインデックス: channel * 81 + col * 9 + row
        buf[channel * 81 + sq.index()] = 1.0;
    }

    // 持ち駒
    let hand_channel_offset = 28; // 先手の持ち駒チャネル開始
    fill_hand_planes(board, Color::Black, hand_channel_offset, buf);
    fill_hand_planes(board, Color::White, hand_channel_offset + 38, buf);
}

/// piece_planes_rotate (104×9×9) を生成する(後手視点)．
///
/// 盤面を180度回転し，先後を入れ替える．
pub fn piece_planes_rotate(board: &Board, buf: &mut [f32]) {
    assert_eq!(buf.len(), FEATURES_NUM * 9 * 9);
    buf.fill(0.0);

    // 盤上の駒(回転: sq → 80-sq, 先後入替)
    for sq_idx in 0..81u8 {
        let piece = board.squares[sq_idx as usize];
        if piece.is_empty() {
            continue;
        }
        let color = piece.color().unwrap();
        let pt = piece.piece_type().unwrap();

        // 色を反転してチャネルを計算
        let rotated_color = color.opponent();
        let channel = board_piece_channel(rotated_color, pt);
        // マスを180度回転
        let rotated_sq = 80 - sq_idx;
        buf[channel * 81 + rotated_sq as usize] = 1.0;
    }

    // 持ち駒(先後を入れ替える)
    let hand_channel_offset = 28;
    fill_hand_planes(board, Color::White, hand_channel_offset, buf); // 後手→先手チャネル
    fill_hand_planes(board, Color::Black, hand_channel_offset + 38, buf); // 先手→後手チャネル
}

/// 盤上の駒のチャネルインデックスを返す(cshogi順)．
fn board_piece_channel(color: Color, pt: PieceType) -> usize {
    let color_offset = match color {
        Color::Black => 0,
        Color::White => 14,
    };
    let piece_offset = match pt {
        PieceType::Pawn => 0,
        PieceType::Lance => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Bishop => 4,
        PieceType::Rook => 5,
        PieceType::Gold => 6,
        PieceType::King => 7,
        PieceType::ProPawn => 8,
        PieceType::ProLance => 9,
        PieceType::ProKnight => 10,
        PieceType::ProSilver => 11,
        PieceType::Horse => 12,
        PieceType::Dragon => 13,
    };
    color_offset + piece_offset
}

/// 持ち駒のチャネルを埋める．
fn fill_hand_planes(board: &Board, color: Color, channel_start: usize, buf: &mut [f32]) {
    // 持ち駒の各駒種のチャネル数: 歩18, 香4, 桂4, 銀4, 金4, 角2, 飛2
    let max_counts = PieceType::MAX_HAND_COUNT;
    let mut channel = channel_start;

    for (kind_idx, &max_count) in max_counts.iter().enumerate() {
        let count = board.hand[color.index()][kind_idx];
        // count枚分のチャネルを全マス1.0で埋める
        for i in 0..max_count as usize {
            if (i as u8) < count {
                // 全81マスを1.0に
                for sq in 0..81 {
                    buf[channel * 81 + sq] = 1.0;
                }
            }
            // count以上のチャネルは0.0のまま
            channel += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_planes_size() {
        let board = Board::new();
        let mut buf = vec![0.0f32; FEATURES_NUM * 9 * 9];
        piece_planes(&board, &mut buf);

        // 先手の歩(チャネル0)は7段目(row=6)に9枚
        let pawn_channel = 0;
        let mut pawn_count = 0;
        for sq in 0..81 {
            if buf[pawn_channel * 81 + sq] > 0.0 {
                pawn_count += 1;
            }
        }
        assert_eq!(pawn_count, 9, "black should have 9 pawns on board");
    }
}
