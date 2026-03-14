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
/// 注: 連続王手の千日手は考慮しない．千日手を含む完全な合法手は
/// [`Position::legal_moves()`](crate::position::Position::legal_moves) を使用すること．
///
/// # パニック安全性
///
/// `&mut Board` を受け取るが，正常終了時は盤面を元の状態に復元する．
/// ただし内部の `do_move`/`undo_move` 間で panic が発生した場合，
/// Board が不整合な中間状態のままになる可能性がある．
pub fn generate_legal_moves(board: &mut Board) -> Vec<Move> {
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

    // 1. 盤上の駒の移動手(occupied bitboardを利用して自駒のみ走査)
    let mut our_bb = our_occ;
    while our_bb.is_not_empty() {
        let from = our_bb.pop_lsb();
        let piece = board.squares[from.index()];
        let pt = piece.piece_type().unwrap();
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
                if !must_promote(us, pt, to) {
                    moves.push(Move::new_move(from, to, false, captured_raw, pt as u8));
                }
            }
        }
    }

    // 2. 駒打ち
    let empty_squares = !all_occ;

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
fn is_legal(board: &mut Board, m: Move) -> bool {
    let us = board.turn;

    // 打ち歩詰めチェック
    if m.is_drop() && m.drop_piece_type() == Some(PieceType::Pawn) {
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
    let captured = board.do_move(m);
    let in_check = board.is_in_check(us);
    board.undo_move(m, captured);

    !in_check
}

/// 打ち歩詰めかどうかを判定する．
///
/// 歩を打って相手玉に王手 → 相手に合法手がないなら打ち歩詰め．
fn is_pawn_drop_mate(board: &mut Board, pawn_drop: Move) -> bool {
    let captured = board.do_move(pawn_drop);

    // 相手(手番交代後の現在手番)の合法手があるかチェック
    // 1手でも見つかれば詰みではない
    let them = board.turn; // 歩を打たれた側
    let pseudo_moves = generate_pseudo_legal_moves(board);

    let mut has_legal = false;
    for m in pseudo_moves {
        let cap2 = board.do_move(m);
        let evades_check = !board.is_in_check(them);
        board.undo_move(m, cap2);
        if evades_check {
            has_legal = true;
            break;
        }
    }

    board.undo_move(pawn_drop, captured);
    !has_legal // 合法手がない → 打ち歩詰め
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

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_hirate_legal_moves_count() {
        let mut board = Board::new();
        let moves = generate_legal_moves(&mut board);
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
        let moves = generate_legal_moves(&mut board);
        // 歩を打てるマスに5筋が含まれていないことを確認
        for m in &moves {
            if m.is_drop() && m.drop_piece_type() == Some(PieceType::Pawn) {
                assert_ne!(
                    m.to_sq().col(),
                    4,
                    "should not be able to drop pawn on file 5 (col 4)"
                );
            }
        }
    }

    #[test]
    fn test_tsume_complex_position() {
        // 片玉局面: 馬・歩・桂を持つ攻め方の詰将棋
        // cshogiは片玉でバグがあるため(存在しない銀の打ちを生成)，
        // 正しい合法手数はcshogi結果からphantom S*を除いた144手
        let mut board = Board::empty();
        board
            .set_sfen("7n1/5+BPk1/5N3/7P1/9/9/9/9/9 b GP2rb3g4s2n4l15p 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        assert_eq!(
            moves.len(),
            144,
            "expected 144 legal moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 銀の打ちが含まれていないことを確認(手持ちに銀がない)
        let silver_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("S*")).collect();
        assert!(
            silver_drops.is_empty(),
            "should not have silver drops (no silver in hand), but found: {:?}",
            silver_drops
        );

        // 盤上の駒の移動手を検証
        // 馬(4b): 3a, 3c, 4a, 5a, 5b, 5c, 6d, 7e, 8f, 9g
        let horse_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("4b")).collect();
        assert_eq!(horse_moves.len(), 10, "horse should have 10 moves: {:?}", horse_moves);

        // 歩(2d): 2c, 2c+(成り)
        let pawn_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("2d")).collect();
        assert_eq!(pawn_moves.len(), 2, "pawn should have 2 moves: {:?}", pawn_moves);

        // 桂(4c): 3a+(成り必須), 5a+(成り必須)
        let knight_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("4c")).collect();
        assert_eq!(knight_moves.len(), 2, "knight should have 2 moves (forced promotion): {:?}", knight_moves);
        assert!(knight_moves.iter().all(|u| u.ends_with('+')), "knight moves to row 0 must promote");

        // 二歩チェック: 2筋(col=1)と3筋(col=2)に歩があるので歩打ち不可
        let pawn_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("P*")).collect();
        assert!(
            pawn_drops.iter().all(|u| !u.starts_with("P*2") && !u.starts_with("P*3")),
            "should not have pawn drops on files 2 and 3 (nifu)"
        );
    }

    #[test]
    fn test_tsume_mixed_pieces() {
        // 片玉局面: 銀・飛・香・桂・角・金が混在する局面
        // cshogiは片玉でバグがあるため(存在しない銀の打ちを生成)，
        // 正しい合法手数はcshogi結果からphantom S*を除いた98手
        let mut board = Board::empty();
        board
            .set_sfen("9/9/9/9/9/SSSSrllll/2nbbGGGG/9/8k b R3n18p 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        assert_eq!(
            moves.len(),
            98,
            "expected 98 legal moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 銀の打ちが含まれていないことを確認(手持ちに銀がない)
        let silver_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("S*")).collect();
        assert!(
            silver_drops.is_empty(),
            "should not have silver drops (no silver in hand), but found: {:?}",
            silver_drops
        );

        // 盤上の駒の移動手を検証(34手)
        let board_moves: Vec<&String> = usi_moves.iter().filter(|u| !u.contains('*')).collect();
        assert_eq!(board_moves.len(), 34, "board moves should be 34: {:?}", board_moves);

        // 飛車打ち(64手)
        let rook_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("R*")).collect();
        assert_eq!(rook_drops.len(), 64, "rook drops should be 64: {:?}", rook_drops);

        // 歩打ちがないこと(手持ちに歩がない)
        let pawn_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("P*")).collect();
        assert!(
            pawn_drops.is_empty(),
            "should not have pawn drops (no pawn in hand), but found: {:?}",
            pawn_drops
        );

        // 桂打ちがないこと(手持ちに桂があるが打ち先がない - 全空きマスが1-2段目)
        // 注: 桂は先手の場合1-2段目(row 0-1)に打てないが，空きマスは3段目以上にもある
        let knight_drops: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("N*")).collect();
        // 桂は3段目以降の空きマスに打てる
        // 手持ちの桂3枚，空きマスのうちrow>=2のマスに打てる
        // (検証のみ，具体的数値はcshogi結果から逆算: 98 - 34 - 64 = 0 → 桂打ちなし)
        assert!(
            knight_drops.is_empty(),
            "knight drops count: {:?}",
            knight_drops
        );
    }

    #[test]
    fn test_nifu() {
        // 二歩テスト: 3筋,5筋,7筋に先手の歩がある局面
        // → これらの筋には歩を打てない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/2P1P1P2/9/4K4 b P 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 56手(48歩打ち + 8盤上)
        assert_eq!(
            moves.len(),
            56,
            "expected 56 legal moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 3筋,5筋,7筋に歩打ちがないことを確認
        let nifu_drops: Vec<&String> = usi_moves
            .iter()
            .filter(|u| u.starts_with("P*") && matches!(u.chars().nth(2), Some('3' | '5' | '7')))
            .collect();
        assert!(
            nifu_drops.is_empty(),
            "should not have pawn drops on files 3,5,7 (nifu), but found: {:?}",
            nifu_drops
        );

        // と金の筋には歩を打てることを確認(と金は二歩の対象外)
        // 3筋にと金，5筋に歩がある局面
        let mut board_tokin = Board::empty();
        board_tokin
            .set_sfen("4k4/9/9/9/9/9/2+P1P4/9/4K4 b P 1")
            .unwrap();
        let moves_tokin = generate_legal_moves(&mut board_tokin);
        let mut usi_tokin: Vec<String> = moves_tokin.iter().map(|m| m.to_usi()).collect();
        usi_tokin.sort();

        // cshogiで検証: 75手(63歩打ち + 12盤上)
        assert_eq!(
            moves_tokin.len(),
            75,
            "expected 75 legal moves with tokin, got {}\nmoves: {:?}",
            moves_tokin.len(),
            usi_tokin
        );

        // 3筋に歩打ちがあること(と金は二歩対象外)
        let p3_drops: Vec<&String> = usi_tokin
            .iter()
            .filter(|u| u.starts_with("P*3"))
            .collect();
        assert!(
            !p3_drops.is_empty(),
            "should have pawn drops on file 3 (tokin is not nifu)"
        );

        // 5筋に歩打ちがないこと(歩がある)
        let p5_drops: Vec<&String> = usi_tokin
            .iter()
            .filter(|u| u.starts_with("P*5"))
            .collect();
        assert!(
            p5_drops.is_empty(),
            "should not have pawn drops on file 5 (nifu), but found: {:?}",
            p5_drops
        );
    }

    #[test]
    fn test_uchifuzume() {
        // 打ち歩詰め局面: 後手玉1a，白桂2a，黒金2c，黒飛2d，黒玉9i
        // P*1bは打ち歩詰め(玉の逃げ場なし)なので合法手に含まれない
        let mut board = Board::empty();
        board
            .set_sfen("7nk/9/7G1/7R1/9/9/9/9/K8 b P 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 89手(P*1bが打ち歩詰めで除外)
        assert_eq!(
            moves.len(),
            89,
            "expected 89 legal moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // P*1bが含まれないことを確認(打ち歩詰め)
        assert!(
            !usi_moves.contains(&"P*1b".to_string()),
            "P*1b should be excluded (uchifuzume)"
        );

        // P*1aも含まれないことを確認(1段目に歩は打てない)
        assert!(
            !usi_moves.contains(&"P*1a".to_string()),
            "P*1a should be excluded (immovable pawn)"
        );

        // 比較: 桂がない局面ではP*1bが合法(打ち歩詰めにならない)
        let mut board_no_knight = Board::empty();
        board_no_knight
            .set_sfen("8k/9/7G1/7R1/9/9/9/9/K8 b P 1")
            .unwrap();
        let moves_no_knight = generate_legal_moves(&mut board_no_knight);
        let usi_no_knight: Vec<String> = moves_no_knight.iter().map(|m| m.to_usi()).collect();

        // cshogiで検証: 90手(P*1bが合法)
        assert_eq!(
            moves_no_knight.len(),
            90,
            "expected 90 legal moves without knight, got {}",
            moves_no_knight.len()
        );

        // P*1bが含まれることを確認(玉が2aに逃げられるので打ち歩詰めではない)
        assert!(
            usi_no_knight.contains(&"P*1b".to_string()),
            "P*1b should be legal when king can escape to 2a"
        );
    }

    #[test]
    fn test_check_evasion() {
        // 王手放置禁止テスト: 後手飛車5aが先手玉5iに王手
        // 玉の移動のみ合法(5筋を離れる手のみ)
        let mut board = Board::empty();
        board
            .set_sfen("4r4/9/9/9/9/9/9/9/4K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 4手(4h,4i,6h,6i)
        assert_eq!(
            moves.len(),
            4,
            "expected 4 check evasion moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 5筋に留まる手がないことを確認(飛車の利き上)
        assert!(
            usi_moves.iter().all(|u| !u.starts_with("5i5")),
            "king should not stay on file 5 (rook line)"
        );
    }

    #[test]
    fn test_pin() {
        // ピンテスト: 後手飛車5a，先手金5h，先手玉5i
        // 金は飛車と玉の間にピンされている → 5筋に沿った手のみ合法
        let mut board = Board::empty();
        board
            .set_sfen("4r4/9/9/9/9/9/9/4G4/4K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 5手(金5h5g + 玉4手)
        assert_eq!(
            moves.len(),
            5,
            "expected 5 legal moves with pin, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 金の合法手は5h5gのみ(5筋に沿って前進)
        let gold_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("5h")).collect();
        assert_eq!(
            gold_moves,
            vec!["5h5g"],
            "pinned gold should only move along file 5"
        );
    }

    #[test]
    fn test_pin_bishop_by_lance_vertical() {
        // 香で角と玉が縦に串刺し(ピン)
        // 先手玉5i，先手角5g，後手香5a
        // 角は縦に動けないため全ての角の手が不合法
        let mut board = Board::empty();
        board
            .set_sfen("4l4/9/9/9/9/9/4B4/9/4K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 5手(玉のみ)
        assert_eq!(
            moves.len(),
            5,
            "expected 5 legal moves (king only), got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 角の手が含まれないことを確認
        let bishop_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("5g")).collect();
        assert!(
            bishop_moves.is_empty(),
            "bishop pinned vertically by lance should have no moves: {:?}",
            bishop_moves
        );
    }

    #[test]
    fn test_pin_bishop_by_rook_horizontal() {
        // 飛で角と玉が横に串刺し(ピン)
        // 先手玉5i，先手角7i，後手飛9i
        // 角は横に動けないため全ての角の手が不合法
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/r1B1K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 5手(玉のみ)
        assert_eq!(
            moves.len(),
            5,
            "expected 5 legal moves (king only), got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 角の手が含まれないことを確認
        let bishop_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("7i")).collect();
        assert!(
            bishop_moves.is_empty(),
            "bishop pinned horizontally by rook should have no moves: {:?}",
            bishop_moves
        );
    }

    #[test]
    fn test_pin_bishop_by_bishop_diagonal() {
        // 角で角が斜めにピン
        // 先手玉5i，先手角6h，後手角8f
        // 斜めライン上の移動のみ合法(7gと8f=角取り)
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/1b7/9/3B5/4K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 6手(角2手 + 玉4手)
        assert_eq!(
            moves.len(),
            6,
            "expected 6 legal moves, got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 角はピンライン上の2手のみ(7g, 8f=角取り)
        let bishop_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("6h")).collect();
        assert_eq!(
            bishop_moves,
            vec!["6h7g", "6h8f"],
            "bishop pinned diagonally should only move along pin line"
        );
    }

    #[test]
    fn test_pin_rook_by_bishop_diagonal() {
        // 角で飛が斜めにピン
        // 先手玉5i，先手飛6h，後手角8f
        // 飛は斜めに動けないため全ての飛の手が不合法
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/1b7/9/3R5/4K4 b - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        // cshogiで検証: 4手(玉のみ)
        assert_eq!(
            moves.len(),
            4,
            "expected 4 legal moves (king only), got {}\nmoves: {:?}",
            moves.len(),
            usi_moves
        );

        // 飛の手が含まれないことを確認
        let rook_moves: Vec<&String> = usi_moves.iter().filter(|u| u.starts_with("6h")).collect();
        assert!(
            rook_moves.is_empty(),
            "rook pinned diagonally by bishop should have no moves: {:?}",
            rook_moves
        );
    }

    #[test]
    fn test_immovable_drop_knight() {
        // 桂打ち制限: 先手は1-2段目(row a,b)に桂を打てない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/4K4 b N 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();

        // cshogiで検証: 67手(玉5手 + 桂打ち62手)
        assert_eq!(
            moves.len(),
            67,
            "expected 67 legal moves, got {}",
            moves.len()
        );

        // 1-2段目への桂打ちがないことを確認
        let bad_drops: Vec<&String> = usi_moves
            .iter()
            .filter(|u| u.starts_with("N*") && matches!(u.chars().nth(3), Some('a' | 'b')))
            .collect();
        assert!(
            bad_drops.is_empty(),
            "knight drops on rows a,b should be excluded: {:?}",
            bad_drops
        );
    }

    #[test]
    fn test_immovable_drop_lance() {
        // 香打ち制限: 先手は1段目(row a)に香を打てない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/4K4 b L 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();

        // cshogiで検証: 76手(玉5手 + 香打ち71手)
        assert_eq!(
            moves.len(),
            76,
            "expected 76 legal moves, got {}",
            moves.len()
        );

        // 1段目への香打ちがないことを確認
        let bad_drops: Vec<&String> = usi_moves
            .iter()
            .filter(|u| u.starts_with("L*") && u.ends_with('a'))
            .collect();
        assert!(
            bad_drops.is_empty(),
            "lance drops on row a should be excluded: {:?}",
            bad_drops
        );
    }

    #[test]
    fn test_immovable_drop_pawn() {
        // 歩打ち制限: 先手は1段目(row a)に歩を打てない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();

        // cshogiで検証: 76手(玉5手 + 歩打ち71手)
        assert_eq!(
            moves.len(),
            76,
            "expected 76 legal moves, got {}",
            moves.len()
        );

        // 1段目への歩打ちがないことを確認
        let bad_drops: Vec<&String> = usi_moves
            .iter()
            .filter(|u| u.starts_with("P*") && u.ends_with('a'))
            .collect();
        assert!(
            bad_drops.is_empty(),
            "pawn drops on row a should be excluded: {:?}",
            bad_drops
        );
    }

    #[test]
    fn test_tsume_single_king() {
        // 片玉局面: 攻め方(先手)に玉がない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/4G4/9/9/9/9/9/9 b G 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        // 合法手が生成できること(玉がなくてもpanicしない)
        assert!(!moves.is_empty(), "tsume position should have legal moves");
    }

    #[test]
    fn test_tsume_defender_turn() {
        // 片玉局面: 受け方(後手)の手番
        let mut board = Board::empty();
        board
            .set_sfen("4k4/4G4/9/9/9/9/9/9/9 w - 1")
            .unwrap();
        let moves = generate_legal_moves(&mut board);
        // 後手は王手されているので応手が必要
        // 合法手が生成できること
        assert!(!moves.is_empty(), "defender should have escape moves");
    }
}
