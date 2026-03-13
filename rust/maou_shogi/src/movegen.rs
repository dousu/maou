use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::Move;
use crate::types::{Color, PieceType, Square};

/// 合法手を生成する．
///
/// 以下のルールを考慮:
/// 1. 自玉への王手放置(ピン含む)
/// 2. 二歩
/// 3. 行き所のない駒
/// 4. 打ち歩詰め
///
/// 注: 連続王手の千日手はPosition側で処理する．
pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    let pseudo_moves = generate_pseudo_legal_moves(board);
    let mut legal_moves = Vec::with_capacity(pseudo_moves.len());

    for m in pseudo_moves {
        if is_legal(board, m) {
            legal_moves.push(m);
        }
    }

    legal_moves
}

/// 疑似合法手を生成する(自玉の王手放置チェックなし)．
fn generate_pseudo_legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(128);
    let us = board.turn;
    let our_occ = board.occupied[us.index()];
    let all_occ = board.all_occupied();

    // 1. 盤上の駒の移動手
    for sq_idx in 0..81u8 {
        let piece = board.squares[sq_idx as usize];
        if piece.is_empty() || piece.color() != Some(us) {
            continue;
        }
        let pt = piece.piece_type().unwrap();
        let from = Square(sq_idx);
        let attacks = attack::piece_attacks(us, pt, from, all_occ);
        // 味方の駒がいるマスは除外
        let targets = attacks & !our_occ;

        for to in targets {
            let captured_piece = board.squares[to.index()];
            let captured_raw = captured_piece.0;

            let in_promo_zone = to.is_promotion_zone(us) || from.is_promotion_zone(us);

            if pt.can_promote() && in_promo_zone {
                // 成りの手
                moves.push(Move::new_move(from, to, true, captured_raw, pt as u8));

                // 不成の手(行き所がある場合のみ)
                if !must_promote(us, pt, to) {
                    moves.push(Move::new_move(from, to, false, captured_raw, pt as u8));
                }
            } else {
                // 成れない場合は不成のみ
                // 行き所のない駒チェック(盤上の駒の移動)
                if !is_immovable(us, pt, to) {
                    moves.push(Move::new_move(from, to, false, captured_raw, pt as u8));
                }
            }
        }
    }

    // 2. 駒打ち
    let empty_squares = !all_occ & Bitboard::ALL;

    for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
        if board.hand[us.index()][hand_idx] == 0 {
            continue;
        }

        // 二歩チェック: 歩の場合，自分の歩がある筋には打てない
        let mut drop_targets = empty_squares;
        if pt == PieceType::Pawn {
            // 各筋に自分の歩があるかチェック
            let our_pawns = board.piece_bb[us.index()][PieceType::Pawn as usize];
            for col in 0..9u8 {
                let file = Bitboard::file_mask(col);
                if (our_pawns & file).is_not_empty() {
                    // この筋には打てない
                    drop_targets &= !file;
                }
            }
        }

        // 行き所のない駒の制限
        match pt {
            PieceType::Pawn | PieceType::Lance => {
                // 先手: row=0(1段目)に打てない，後手: row=8(9段目)に打てない
                let forbidden = match us {
                    Color::Black => Bitboard::rank_mask(0),
                    Color::White => Bitboard::rank_mask(8),
                };
                drop_targets &= !forbidden;
            }
            PieceType::Knight => {
                // 先手: row=0,1(1-2段目)に打てない，後手: row=7,8(8-9段目)に打てない
                let forbidden = match us {
                    Color::Black => Bitboard::rank_mask(0) | Bitboard::rank_mask(1),
                    Color::White => Bitboard::rank_mask(7) | Bitboard::rank_mask(8),
                };
                drop_targets &= !forbidden;
            }
            _ => {}
        }

        for to in drop_targets {
            moves.push(Move::new_drop(to, pt));
        }
    }

    moves
}

/// 手が合法かどうかを検証する(自玉の王手放置チェック + 打ち歩詰め)．
fn is_legal(board: &Board, m: Move) -> bool {
    let us = board.turn;

    // 打ち歩詰めチェック
    if m.is_drop() && m.drop_piece_type() == PieceType::Pawn {
        // 歩を打って王手になるかチェック
        let to = m.to_sq();
        let opp_king = board.king_square(us.opponent());
        if let Some(king_sq) = opp_king {
            // 歩の利きが相手玉にかかるか
            let pawn_attack = attack::step_attacks(us, PieceType::Pawn, to);
            if pawn_attack.contains(king_sq) {
                // 王手になる歩打ち → 詰みかどうかチェック
                if is_pawn_drop_mate(board, m) {
                    return false; // 打ち歩詰め
                }
            }
        }
    }

    // 自玉の王手放置チェック: 手を指した後に自玉に王手がかかるか
    let mut board_copy = board.clone();
    let _captured = board_copy.do_move(m);

    // do_move後は手番が交代しているので，usの玉をチェック
    let in_check = board_copy.is_in_check(us);

    // 戻す(board_copyは使い捨て)
    !in_check
}

/// 打ち歩詰めかどうかを判定する．
///
/// 歩を打って相手玉に王手 → 相手に合法手がないなら打ち歩詰め．
fn is_pawn_drop_mate(board: &Board, pawn_drop: Move) -> bool {
    let mut board_copy = board.clone();
    let _captured = board_copy.do_move(pawn_drop);

    // 相手(手番交代後の現在手番)の合法手があるかチェック
    // 1手でも見つかれば詰みではない
    let them = board_copy.turn; // 歩を打たれた側
    let pseudo_moves = generate_pseudo_legal_moves(&board_copy);

    for m in pseudo_moves {
        // この手で王手が解消されるかチェック
        let mut board_copy2 = board_copy.clone();
        let _cap = board_copy2.do_move(m);
        if !board_copy2.is_in_check(them) {
            return false; // 合法手がある → 詰みではない
        }
    }

    true // 合法手がない → 打ち歩詰め
}

/// 強制成りが必要かどうか(行き所がなくなるため)．
fn must_promote(color: Color, pt: PieceType, to: Square) -> bool {
    match (color, pt) {
        (Color::Black, PieceType::Pawn | PieceType::Lance) => to.row() == 0,
        (Color::White, PieceType::Pawn | PieceType::Lance) => to.row() == 8,
        (Color::Black, PieceType::Knight) => to.row() <= 1,
        (Color::White, PieceType::Knight) => to.row() >= 7,
        _ => false,
    }
}

/// 行き所のない駒になるか(強制成りの別の書き方，打ち駒の移動にも使用)．
fn is_immovable(color: Color, pt: PieceType, to: Square) -> bool {
    must_promote(color, pt, to)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_hirate_legal_moves_count() {
        let board = Board::new();
        let moves = generate_legal_moves(&board);
        // 平手初期局面の合法手は30手
        // 歩9枚×1 = 9, 香0, 桂0, 銀0, 金0, 角1, 飛1, 王0 (待機)
        // 実際は: 歩9 + 角0(塞がってる) + 飛0(塞がってる) + 銀0 + 金0 + 桂0 + 香0 + 王0
        // = 歩前進9手 + 角2手(USI: 8h7g, 8h9i?→いや壁) → 要確認
        // cshogiでの正解値: 30手
        assert_eq!(moves.len(), 30, "hirate legal moves should be 30, got {}", moves.len());
    }

    #[test]
    fn test_drop_pawn_nifu() {
        // 二歩テスト: 5筋に歩がある状態で5筋に歩を打てない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1")
            .unwrap();
        let moves = generate_legal_moves(&board);
        // 歩を打てるマスに5筋が含まれていないことを確認
        for m in &moves {
            if m.is_drop() && m.drop_piece_type() == PieceType::Pawn {
                assert_ne!(
                    m.to_sq().col(),
                    4,
                    "should not be able to drop pawn on file 5 (col 4)"
                );
            }
        }
    }
}
