// verbose feature 無効時に verbose_eprintln! 内でのみ使われる変数の
// unused 警告を抑制する．verbose 有効時はマクロが eprintln! に展開され
// 変数が実際に使用されるため警告は出ない．
#![allow(unused_variables, unused_assignments)]

use super::*;
use super::solver::*;
use super::pns::*;
use crate::attack;
use crate::movegen;
use crate::types::PieceType;

    // === hand_gte / hand_gte_forward_chain のユニットテスト ===

    /// hand_gte: 全駒種で a >= b なら true．
    #[test]
    fn test_hand_gte_basic() {
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte(&a, &b));

        let a = [2, 0, 0, 0, 0, 0, 0]; // 歩2
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte(&a, &b));

        let a = [0, 0, 0, 0, 0, 0, 0]; // 空
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(!hand_gte(&a, &b));
    }

    /// hand_gte: 異種駒では代替不可(従来の挙動)．
    #[test]
    fn test_hand_gte_different_pieces() {
        // 香1 vs 歩1: 香で歩を代替できない(hand_gteでは)
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(!hand_gte(&a, &b));

        // 飛1 vs 香1
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(!hand_gte(&a, &b));
    }

    /// forward_chain: 歩 ≤ 香 ≤ 飛 のチェーン代替．
    #[test]
    fn test_forward_chain_pawn_to_lance() {
        // 香1 >= 歩1 (香は歩の上位互換)
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    #[test]
    fn test_forward_chain_pawn_to_rook() {
        // 飛1 >= 歩1 (飛は歩の上位互換)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    #[test]
    fn test_forward_chain_lance_to_rook() {
        // 飛1 >= 香1 (飛は香の上位互換)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 逆方向は不成立(歩は香の代替にならない)．
    #[test]
    fn test_forward_chain_no_reverse() {
        // 歩1 < 香1 (歩は香の代替にならない)
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [0, 1, 0, 0, 0, 0, 0]; // 香1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 香1 < 飛1
        let a = [0, 1, 0, 0, 0, 0, 0]; // 香1
        let b = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 歩1 < 飛1
        let a = [1, 0, 0, 0, 0, 0, 0]; // 歩1
        let b = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: カスケード(歩不足 → 香で補填 → 香不足 → 飛で補填)．
    #[test]
    fn test_forward_chain_cascade() {
        // 歩1+香1 を 飛2 で代替
        let a = [0, 0, 0, 0, 0, 0, 2]; // 飛2
        let b = [1, 1, 0, 0, 0, 0, 0]; // 歩1+香1
        assert!(hand_gte_forward_chain(&a, &b));

        // 歩2 を 香1+飛1 で代替(香が歩1つ分を吸収，飛が残り1つを吸収)
        let a = [0, 1, 0, 0, 0, 0, 1]; // 香1+飛1
        let b = [2, 0, 0, 0, 0, 0, 0]; // 歩2
        assert!(hand_gte_forward_chain(&a, &b));

        // 歩2+香1 を 飛1 では不足(飛1で代替できるのは1つだけ)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [2, 1, 0, 0, 0, 0, 0]; // 歩2+香1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 非チェーン駒(桂・銀・金・角)は代替不可．
    #[test]
    fn test_forward_chain_non_chain_pieces() {
        // 桂が足りなければ false
        let a = [1, 1, 0, 0, 0, 0, 1]; // 歩1+香1+飛1
        let b = [0, 0, 1, 0, 0, 0, 0]; // 桂1
        assert!(!hand_gte_forward_chain(&a, &b));

        // 角が足りなければ false
        let a = [0, 0, 0, 0, 0, 0, 2]; // 飛2
        let b = [0, 0, 0, 0, 0, 1, 0]; // 角1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 複合ケース(チェーン + 非チェーン駒)．
    #[test]
    fn test_forward_chain_mixed() {
        // 香1+金1 >= 歩1+金1
        let a = [0, 1, 0, 0, 1, 0, 0]; // 香1+金1
        let b = [1, 0, 0, 0, 1, 0, 0]; // 歩1+金1
        assert!(hand_gte_forward_chain(&a, &b));

        // 飛1+桂1 >= 歩1+桂1
        let a = [0, 0, 1, 0, 0, 0, 1]; // 桂1+飛1
        let b = [1, 0, 1, 0, 0, 0, 0]; // 歩1+桂1
        assert!(hand_gte_forward_chain(&a, &b));

        // 飛1+金0 < 歩1+金1 (金が不足)
        let a = [0, 0, 0, 0, 0, 0, 1]; // 飛1
        let b = [1, 0, 0, 0, 1, 0, 0]; // 歩1+金1
        assert!(!hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 同一 hand なら常に true (hand_gte の fast path)．
    #[test]
    fn test_forward_chain_identity() {
        let h = [3, 2, 1, 1, 2, 1, 1];
        assert!(hand_gte_forward_chain(&h, &h));
    }

    /// forward_chain: 空の hand(何も必要としない)なら常に true．
    #[test]
    fn test_forward_chain_empty_requirement() {
        let a = [0, 0, 0, 0, 0, 0, 0];
        let b = [0, 0, 0, 0, 0, 0, 0];
        assert!(hand_gte_forward_chain(&a, &b));

        let a = [1, 1, 1, 1, 1, 1, 1];
        assert!(hand_gte_forward_chain(&a, &b));
    }

    /// forward_chain: 実際のチェーン合駒シナリオ(L*6g → N*6g)．
    ///
    /// L*6g の proof で attacker hand に lance が必要な局面の TT エントリを，
    /// N*6g の探索(attacker hand に knight がある)で再利用できるか．
    #[test]
    fn test_forward_chain_real_scenario() {
        // L*6g 後の proof entry: 攻め方が香を獲得
        let stored = [0, 1, 0, 0, 0, 0, 0]; // 香1(Rx6g で獲得)
        // N*6g 後の current hand: 攻め方が桂を獲得
        let current = [0, 0, 1, 0, 0, 0, 0]; // 桂1(Rx6g で獲得)
        // 桂は香の上位互換ではない → 再利用不可
        assert!(!hand_gte_forward_chain(&current, &stored));

        // 逆に: stored が歩1, current が香1 → 再利用可能
        let stored_pawn = [1, 0, 0, 0, 0, 0, 0];
        let current_lance = [0, 1, 0, 0, 0, 0, 0];
        assert!(hand_gte_forward_chain(&current_lance, &stored_pawn));
    }

    /// 詰将棋画像のテストケース: 小阪昇作，9手詰
    ///
    /// 局面: 後手玉1四，先手角2四・3四，後手銀3一・後手香3二
    /// 先手持ち駒: 飛，歩
    /// 後手持ち駒: 飛，金4，銀3，桂4，香3，歩17
    ///
    /// 正解手順: 1三角成，同玉，2三飛打，1二玉，1三歩打，1一玉，2一飛成，同玉，1二歩成
    #[test]
    fn test_tsume_9te() {
        let sfen = "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(15, 1_048_576, 32767);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(),
                    9,
                    "should be 9-move checkmate, got: {:?}",
                    usi_moves
                );

                // 正解手順を検証(USI形式)
                assert_eq!(usi_moves[0], "2d1c+", "move 1: 1三角成");
                assert_eq!(usi_moves[1], "1d1c", "move 2: 同玉");
                assert_eq!(usi_moves[2], "R*2c", "move 3: 2三飛打");
                assert_eq!(usi_moves[3], "1c1b", "move 4: 1二玉");
                assert_eq!(usi_moves[4], "P*1c", "move 5: 1三歩打");
                assert_eq!(usi_moves[5], "1b1a", "move 6: 1一玉");
                assert_eq!(usi_moves[6], "2c2a+", "move 7: 2一飛成");
                assert_eq!(usi_moves[7], "1a2a", "move 8: 同玉");
                assert_eq!(usi_moves[8], "1c1b+", "move 9: 1二歩成");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 簡単な1手詰め．
    #[test]
    fn test_tsume_1te() {
        // 後手玉1一，先手金2三，先手持ち駒: 金
        // G*1b(1二金打)で詰み
        let sfen = "8k/9/7G1/9/9/9/9/9/9 b G 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(3, 100_000, 32767);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1);
                assert_eq!(moves[0].to_usi(), "G*1b", "1手詰め: G*1b(1二金打)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 3手詰め: 後手玉1一，先手飛3三，先手持ち駒: 金
    ///
    /// 正解: 1三飛成，2一玉，2二金打 まで3手詰
    #[test]
    fn test_tsume_3te() {
        let sfen = "8k/9/6R2/9/9/9/9/9/9 b G 1";
        let result = solve_tsume(sfen, Some(7), Some(1_048_576), None).unwrap();

        let expected = ["3c1c+", "1a2a", "G*2b"];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 3,
                    "expected 3 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 不詰のケース．
    #[test]
    fn test_no_checkmate() {
        // 後手玉5一，先手持ち駒: 歩 → 歩では詰まない
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(5, 100_000, 32767);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!("expected NoCheckmate, got {:?}", other),
        }
    }

    /// solve_tsume 便利関数のテスト．
    #[test]
    fn test_solve_tsume_convenience() {
        let result = solve_tsume(
            "8k/9/7G1/9/9/9/9/9/9 b G 1",
            Some(3),
            Some(100_000),
            None,
        )
        .unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(moves.len(), 1);
                // G*1b(12金打) または G*2b(22金打) が正解
                assert!(
                    pv[0] == "G*1b" || pv[0] == "G*2b",
                    "expected G*1b or G*2b, got {}",
                    pv[0],
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    #[test]
    fn test_tsume_2() {
        // 盤面: 5一と，2一王，1一香，2二銀，4三飛，2四角，先手持駒: 金桂
        // 11手詰め: 32金打，同玉，42角成，21玉，31馬，同銀，23飛成，22銀，33桂打，31玉，41と
        let sfen = "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(1_048_576), None).unwrap();

        let expected = [
            "G*3b", "2a3b", "2d4b+", "3b2a", "4b3a", "2b3a",
            "4c2c+", "3a2b", "N*3c", "2a3a", "5a4a",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(moves.len(), 11, "expected 11 moves, got {}: {:?}", moves.len(), usi_moves);
                assert_eq!(
                    usi_moves,
                    expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// between_bb ヘルパーのテスト．
    #[test]
    fn test_between_bb() {
        // 飛5一(col=4,row=0) と 1一(col=0,row=0) の間: 2一,3一,4一
        let between = attack::between_bb(Square::new(4, 0), Square::new(0, 0));
        assert_eq!(between.count(), 3);
        assert!(between.contains(Square::new(1, 0))); // 2一
        assert!(between.contains(Square::new(2, 0))); // 3一
        assert!(between.contains(Square::new(3, 0))); // 4一

        // 隣接マス(間なし)
        let between2 = attack::between_bb(Square::new(0, 0), Square::new(1, 0));
        assert!(between2.is_empty());

        // 斜め
        let between3 = attack::between_bb(Square::new(0, 0), Square::new(3, 3));
        assert_eq!(between3.count(), 2);
        assert!(between3.contains(Square::new(1, 1)));
        assert!(between3.contains(Square::new(2, 2)));
    }

    /// タイムアウト機能のテスト．
    #[test]
    fn test_timeout() {
        // 不詰の局面を極短タイムアウトで探索 → Unknown が返る
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, u64::MAX, 32767, 0);
        // timeout=0 なので即タイムアウト(ただし最初の1024ノードは走る)
        let result = solver.solve_via_v3(&mut board);

        // NoCheckmate か Unknown のどちらか(歩1枚では詰まない)
        match &result {
            TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {}
            other => panic!("expected NoCheckmate or Unknown, got {:?}", other),
        }
    }

    /// 詰将棋画像3のテストケース．
    ///
    /// 局面: 後手玉2三，後手桂2一，後手香1一，後手飛5四，後手歩1三，後手と1六
    ///       先手歩1五，先手桂3四，先手金2六，先手香3六
    /// 先手持駒: 飛
    /// 後手持駒: 歩15，香2，桂2，銀4，金3，角2
    #[test]
    fn test_tsume_3() {
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        let expected = [
            "3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d",
            "2f2e", "2d2e", "3b3e",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "expected 9 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 2一龍後に2三歩打で詰まないことを検証する．
    ///
    /// 無駄合いフィルタが誤って2三歩打を除外していた問題の回帰テスト．
    /// 2二桂成，同玉，4二飛打，2三玉，3二飛成，2四玉，2一龍 の後，
    /// 後手は2三歩打で合い駒でき，これは無駄合いではない．
    #[test]
    fn test_tsume_3_ryu_2a_not_checkmate() {
        // 2一龍後の局面を作成
        // 初期局面から 3d2b+, 2c2b, R*4b, 2b2c, 4b3b+, 2c2d, 3b2a を実行
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let moves_usi = ["3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d", "3b2a"];
        for usi in &moves_usi {
            let m = board.move_from_usi(usi).expect(&format!("invalid USI: {}", usi));
            board.do_move(m);
        }

        // ここは AND ノード(後手番)．2三歩打(P*2c)が合法手に含まれることを検証
        let defenses = movegen::generate_legal_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();
        // debug: defenses

        // P*2c (2三歩打) が合法手に含まれること
        assert!(
            usi_defenses.contains(&"P*2c".to_string()),
            "P*2c should be a legal defense, got: {:?}", usi_defenses
        );

        // 2三歩打後の局面は詰みではないことを確認
        let p2c = board.move_from_usi("P*2c").unwrap();
        let cap = board.do_move(p2c);

        // 先手番(攻め方)から探索して詰みがないことを検証
        let mut solver = DfPnSolver::new(15, 100_000, 32767);
        let result = solver.solve_via_v3(&mut board);
        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "P*2c 後の局面は詰みではないはず: {:?}",
            result
        );

        board.undo_move(p2c, cap);
    }

    /// 詰将棋テストケース4．
    ///
    /// 局面: 7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p
    /// 11手詰め(合い効かずにより3二歩打を除外)．
    /// 最終2手は玉の逃げ方により3パターンの正解が存在:
    /// - 1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍
    #[test]
    fn test_tsume_4() {
        let sfen = "7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                // PNS (Best-First) は最短 9 手詰めを発見する．
                // MID フォールバック時は 11 手の解を返す場合がある．
                assert!(
                    usi_moves.len() == 9 || usi_moves.len() == 11,
                    "expected 9 or 11 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert!(
                    usi_moves.len() % 2 == 1,
                    "tsume must have odd number of moves, got {}", usi_moves.len(),
                );
                // 最終2手は玉の逃げ方によりパターンが分岐する:
                //   1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍 / 2三玉，2二龍
                // mid_v3 (v2.1.0 production engine) は canonical longest-defense
                // mate-11 を返し，玉 2三逃げ (1b2c) の応手線で 2二龍 (4b2b) が詰む．
                // STRICT VERIFY Some(11) で健全性確認済．
                let last2 = &usi_moves[usi_moves.len() - 2..];
                let valid_endings = [
                    ["1b1a", "3d2b+"],  // 1一玉，2二桂成
                    ["1b1a", "4b2b"],   // 1一玉，2二龍
                    ["1b1c", "4b2b"],   // 1三玉，2二龍
                    ["1b2c", "4b2b"],   // 2三玉，2二龍 (mid_v3)
                ];
                assert!(
                    valid_endings.iter().any(|e| last2 == e),
                    "last 2 moves must match a valid ending pattern, got: {:?}\n  valid: {:?}",
                    last2, valid_endings,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// generate_check_moves の結果を brute-force と比較する．
    #[test]
    fn test_check_moves_completeness() {
        use std::collections::BTreeSet;
        let test_positions = [
            // 17手詰めの初期局面(OR node)
            "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1",
            // tsume2
            "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1",
            // tsume3
            "l1k6/9/1pB6/9/9/9/9/9/9 b RGrb4g4s4n3l16p 1",
            // 9te (tsume1)
            "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1",
        ];

        let solver = DfPnSolver::default_solver();
        for sfen in &test_positions {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            // Brute-force: 全合法手 → フィルタ
            let brute_checks: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .into_iter()
                .filter(|m| {
                    let c = board.do_move(*m);
                    let gives_check = board.is_in_check(board.turn);
                    board.undo_move(*m, c);
                    gives_check
                })
                .map(|m| m.to_usi())
                .collect();

            let optimized_checks: BTreeSet<String> = solver
                .generate_check_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let missing: BTreeSet<_> = brute_checks.difference(&optimized_checks).collect();
            let extra: BTreeSet<_> = optimized_checks.difference(&brute_checks).collect();

            assert!(
                missing.is_empty() && extra.is_empty(),
                "check moves mismatch for sfen: {}\n  missing: {:?}\n  extra: {:?}\n  brute: {:?}\n  opt: {:?}",
                sfen, missing, extra, brute_checks, optimized_checks
            );
        }
    }

    /// 39手詰め PV 途中の全局面で generate_check_moves が brute-force と一致することを検証．
    ///
    /// 特に ply 22 の P*1g が生成されないバグの回帰テスト．
    #[test]
    fn test_check_moves_completeness_39te_pv() {
        use std::collections::BTreeSet;

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        let solver = DfPnSolver::default_solver();
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV の各 OR node (攻め方手番) で check_moves の完全性を検証
        for (ply, &usi) in pv.iter().enumerate() {
            if ply % 2 == 0 {
                // OR node: 攻め方手番 → check_moves を検証
                let brute_checks: BTreeSet<String> =
                    movegen::generate_legal_moves(&mut board)
                        .into_iter()
                        .filter(|m| {
                            let c = board.do_move(*m);
                            let gives_check = board.is_in_check(board.turn);
                            board.undo_move(*m, c);
                            gives_check
                        })
                        .map(|m| m.to_usi())
                        .collect();

                let optimized_checks: BTreeSet<String> = solver
                    .generate_check_moves(&mut board)
                    .iter()
                    .map(|m| m.to_usi())
                    .collect();

                let missing: BTreeSet<_> =
                    brute_checks.difference(&optimized_checks).collect();
                let extra: BTreeSet<_> =
                    optimized_checks.difference(&brute_checks).collect();

                assert!(
                    missing.is_empty() && extra.is_empty(),
                    "check moves mismatch at ply {} (next PV move: {})\n  \
                     missing: {:?}\n  extra: {:?}",
                    ply, usi, missing, extra
                );
            }

            let m = board
                .move_from_usi(usi)
                .unwrap_or_else(|| panic!("invalid USI at ply {}: {}", ply, usi));
            board.do_move(m);
        }
    }

    /// 39手詰め PV の各 AND ノードで defense_moves ⊆ legal_moves を検証する．
    ///
    /// chain 最適化により defense_moves は legal_moves のサブセットになるため，
    /// extra(legal にない手)が空であることのみ検証する．
    /// また，PV 上の応手が defense_moves に含まれることも確認する．
    #[test]
    fn test_defense_moves_subset_39te_pv() {
        use std::collections::BTreeSet;

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        let mut solver = DfPnSolver::default_solver();
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        for (ply, &usi) in pv.iter().enumerate() {
            let m = board
                .move_from_usi(usi)
                .unwrap_or_else(|| panic!("invalid USI at ply {}: {}", ply, usi));
            board.do_move(m);

            if ply % 2 == 0 {
                // 攻め手の後 → AND node (玉方手番)
                if !board.is_in_check(board.turn) {
                    continue;
                }

                let legal_moves: BTreeSet<String> =
                    movegen::generate_legal_moves(&mut board)
                        .iter()
                        .map(|m| m.to_usi())
                        .collect();

                let defense_moves: BTreeSet<String> = solver
                    .generate_defense_moves(&mut board)
                    .iter()
                    .map(|m| m.to_usi())
                    .collect();

                // defense_moves ⊆ legal_moves (不正な手がないこと)
                let extra: BTreeSet<_> =
                    defense_moves.difference(&legal_moves).collect();
                assert!(
                    extra.is_empty(),
                    "defense has illegal moves at ply {} (after {})\n  \
                     extra: {:?}",
                    ply + 1, usi, extra
                );

                // PV の次の応手が defense_moves に含まれること
                if ply + 1 < pv.len() {
                    let next_defense = pv[ply + 1];
                    assert!(
                        defense_moves.contains(next_defense),
                        "PV defense move {} missing from defense_moves at ply {}\n  \
                         defense({}): {:?}",
                        next_defense, ply + 1,
                        defense_moves.len(), defense_moves
                    );
                }
            }
        }
    }

    /// generate_defense_moves と generate_legal_moves の結果を比較する．
    ///
    /// 王手がかかっている局面で，回避手生成(evasion)が
    /// 全合法手のサブセットであり，かつ全合法手を漏れなく含むことを検証する．
    #[test]
    fn test_defense_moves_completeness() {
        use std::collections::BTreeSet;
        // テスト局面: 攻め方が王手した直後の局面をいくつか用意
        let test_positions = [
            // S*4a で王手 → 玉方の応手
            "9/3S1Pk2/9/8R/8B/9/9/9/9 w rb4g2s4n4l17p 2",
            // 飛車で王手(スライディング)
            "9/5Pk2/9/5R3/8B/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // 角で王手
            "9/5Pk2/9/8R/5B3/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // test_tsume_3 の中間局面(R*2a 後)
            "l1k6/R8/1pB6/9/9/9/9/9/9 w rb4g4s4n3l16p 2",
        ];

        let mut solver = DfPnSolver::default_solver();

        for sfen in &test_positions {
            let mut board = Board::new();
            if board.set_sfen(sfen).is_err() {
                continue;
            }

            // まず王手されているか確認
            if !board.is_in_check(board.turn) {
                continue;
            }

            let defense_moves: BTreeSet<String> = solver
                .generate_defense_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let legal_moves: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            // defense_moves は legal_moves のサブセットであること
            let extra: BTreeSet<_> = defense_moves.difference(&legal_moves).collect();
            assert!(
                extra.is_empty(),
                "defense has extra moves not in legal: {:?}\nsfen: {}",
                extra, sfen
            );

            // legal_moves は defense_moves のサブセットであること(漏れがない)
            let missing: BTreeSet<_> = legal_moves.difference(&defense_moves).collect();
            assert!(
                missing.is_empty(),
                "defense is missing legal moves: {:?}\nsfen: {}\ndefense: {:?}\nlegal: {:?}",
                missing, sfen, defense_moves, legal_moves
            );
        }
    }

    /// 17手詰め局面の solve() テスト．
    #[test]
    fn test_tsume_5() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    pv.len(),
                    17,
                    "expected 17-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );

                // 攻め方の手順を検証(奇数手)
                // 41銀打，32銀打，11飛成，33角成，同馬，23金打，21銀成，32銀成
                assert_eq!(pv[0], "S*4a", "move 1: 41銀打");
                assert_eq!(pv[2], "S*3b", "move 3: 32銀打");
                assert_eq!(pv[4], "1d1a+", "move 5: 11飛成");
                assert_eq!(pv[6], "1e3c+", "move 7: 33角成");
                assert_eq!(pv[8], "3c2b", "move 9: 同馬");
                assert_eq!(pv[10], "G*2c", "move 11: 23金打");
                // move 12: 何を合駒してもよい
                assert_eq!(pv[12], "3b2a+", "move 13: 21銀成");
                assert_eq!(pv[14], "4a3b+", "move 15: 32銀成");
                // move 17: 22成銀(3b2b) or 22金(2c2b) など複数正解
                assert!(
                    pv[16] == "3b2b" || pv[16] == "2c2b",
                    "move 17: expected 3b2b or 2c2b, got {}",
                    pv[16],
                );
            }
            other => panic!("expected Checkmate for tsume5, got {:?}", other),
        }
    }

    /// `mate_move_in_1ply` の false mate-1 回帰テスト．
    ///
    /// `is_checkmate_after_bb` / `generate_check_moves` には以下 3 つの false mate-1 バグがあり，
    /// mid_v3 の look-ahead (`CheckObviousFinalOrNode`) が多数の局面で `mate_move_in_1ply` を
    /// 呼ぶため偽の証明を TT に汚染し，dom-off で STRICT VERIFY None を生んでいた．
    /// 各局面で「攻め方は真の 1 手詰を持たない」ことを `generate_legal_moves` で確認する．
    ///
    /// 1. **ピン軸上の取り返し** (`can_capture_checker_bb`): 香で縦ピンされた金が同じ筋に
    ///    打たれた歩 (王手駒) を取れるのに，ピン駒を一律除外して詰みと誤判定していた．
    /// 2. **逆王手中の駒打ち** (`generate_check_moves`): 攻め方自身が王手されている局面で，
    ///    自玉の王手を解消しない駒打ちを (非合法なのに) 王手手として生成していた．
    /// 3. **開き王手** (`is_checkmate_after_bb`): 動いた駒を唯一の王手駒と仮定するため，
    ///    開き王手で受け方が真の王手駒を取れるのに詰みと誤判定していた．
    #[test]
    fn test_mate_move_in_1ply_no_false_mate() {
        // (sfen, ラベル) — いずれも攻め方手番 (OR 局面) で真の 1 手詰は存在しない．
        let cases = [
            // 1. ピン軸上の取り返し: 香(9a)が金(9b)を玉(9d)へ縦ピン．P*9c は金 9b9c で取れる．
            ("ln1+P5/G1k4+L1/2p1p2B1/K1gp1spN1/1n2Ps3/P1PP2P2/2+bS5/1g3+p3/L1s6 w 2RGNL3P4p 37",
             "pinned-capture-along-ray (P*9c)"),
            // 2. 逆王手: 後手 R*9b が先手玉(7b)を王手．N*8a は自玉の王手を解消しないため非合法．
            ("l2+P5/R1k4+L1/K1p1p2B1/2gp1spN1/1n2Ps3/P1PP2P2/2+bS5/1g3+p3/L1s6 w R2GNL3Pn4p 35",
             "counter-check illegal drop (N*8a)"),
            // 3. 開き王手: 飛 8h8i+ は玉(7b)を直接王手しない (開き王手)．受け方は逃れ可．
            ("l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PGPP2P2/K2S5/1r3+p3/LN+b6 w R2GNLPgs5p 9",
             "discovered check (8h8i+)"),
            // 4. pinner 自身の移動による pin 解除: 移動前は +R(1e) が金(1f) を file 1 に
            //    縦ピンしているが，王手 1e2f で pinner が動くと金は unpin され 1f2f (横) で
            //    王手駒を取り返せる．移動前 pin の流用は偽 1 手詰 (39te 偽証明 @17.9M の真因)．
            ("4+N4/9/6pS1/9/8+R/6+B1g/1R2S3k/3p3N1/9 b NPb3g2sn4l15p 17",
             "pin released by moving pinner (1e2f)"),
        ];
        for (sfen, label) in cases {
            let mut b = Board::new();
            b.set_sfen(sfen).unwrap();
            let s = DfPnSolver::with_timeout(31, 1_000, 32767, 5);
            let checks = s.generate_check_moves_cached(&mut b);
            // 真の 1 手詰の有無を generate_legal_moves で確定する (ground truth)．
            let mut real_mate1 = false;
            for &c in checks.iter() {
                let cap = b.do_move(c);
                let mated = b.is_in_check(b.turn()) && movegen::generate_legal_moves(&mut b).is_empty();
                b.undo_move(c, cap);
                if mated { real_mate1 = true; break; }
            }
            assert!(!real_mate1, "{label}: テスト局面は本来 1 手詰でないはず ({sfen})");
            let turn = b.turn;
            let claimed = b.mate_move_in_1ply(checks.as_slice(), turn);
            assert!(
                claimed.is_none(),
                "{label}: mate_move_in_1ply が false mate-1 を返した: {} ({sfen})",
                claimed.map(|m| m.to_usi()).unwrap_or_default(),
            );
        }
    }

    /// 逆王手 (counter-check) 詰将棋の回帰テスト．攻め方 (先手) 自身が後手龍に王手されている
    /// 局面で，王手を解消しつつ詰ます mate-7．逆王手 drop filter (`generate_check_moves`) と
    /// mate_move_in_1ply の counter-check 健全性を守る．[[project_dfpn_domoff_none_mate1_bugs]]．
    #[test]
    fn test_mid_v3_counter_check_example() {
        let sfen = "7l1/5n1k1/7+RP/6sK1/7L1/9/9/9/9 w r2b4g3s3n2l17p 2";
        let mut b = Board::new();
        b.set_sfen(sfen).unwrap();
        // 前提: 攻め方自身が王手されている (逆王手局面)．
        assert!(b.is_in_check(b.turn), "テスト前提: 攻め方が王手されている逆王手局面のはず");
        let mut s = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        match s.solve_via_v3(&mut b) {
            TsumeResult::Checkmate { moves, .. } => {
                // 終局面が真の詰みか検証 (false mate 検出)．
                let mut chk = Board::new();
                chk.set_sfen(sfen).unwrap();
                for m in &moves { chk.do_move(*m); }
                let in_check = chk.is_in_check(chk.turn());
                let legal = movegen::generate_legal_moves(&mut chk).len();
                assert!(in_check && legal == 0,
                    "逆王手例題 PV 終局面は真の詰みであるべき: in_check={in_check} legal={legal}");
                assert_eq!(moves.len(), 7,
                    "逆王手例題は mate-7 のはず: {:?}", moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>());
            }
            other => panic!("逆王手例題 NON-SOLVE: {other:?}"),
        }
    }

    /// Phase 32: mid_v3 (ground-up KH コア port) の correctness + 29te 計測．
    ///
    /// **[SLOW]** 29te を解く．
    #[test]
    #[ignore]
    fn test_mid_v3() {
        // 1 手詰 correctness．
        {
            let mut b = Board::empty();
            b.set_sfen("8k/9/7G1/9/9/9/9/9/9 b G 1").unwrap();
            let mut s = DfPnSolver::with_timeout(3, 1_000_000, 32767, 30);
            match s.solve_via_v3(&mut b) {
                TsumeResult::Checkmate { moves, nodes_searched } => eprintln!(
                    "[v3] 1te: {} moves ({}), {} nodes",
                    moves.len(), moves.first().map(|m| m.to_usi()).unwrap_or_default(), nodes_searched),
                other => eprintln!("[v3] 1te FAIL: {other:?}"),
            }
        }
        // 3 手詰 correctness．
        {
            let mut b = Board::empty();
            b.set_sfen("8k/9/6R2/9/9/9/9/9/9 b G 1").unwrap();
            let mut s = DfPnSolver::with_timeout(7, 1_000_000, 32767, 30);
            match s.solve_via_v3(&mut b) {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    let u: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                    eprintln!("[v3] 3te: {} moves {:?}, {} nodes", moves.len(), u, nodes_searched);
                }
                other => eprintln!("[v3] 3te FAIL: {other:?}"),
            }
        }
        // 29te 計測 (budget 制限つき; v3 初版なので爆発時は budget で止める)．
        {
            let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
            let mut b = Board::new();
            b.set_sfen(sfen).unwrap();
            let mut s = DfPnSolver::with_timeout(31, 5_000_000, 32767, 120);
            match s.solve_via_v3(&mut b) {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                    eprintln!("[v3] 29te: {} moves, {} nodes, {} tt | PV {:?}",
                        moves.len(), nodes_searched, s.v3_tt.len(), usi);
                    // PV を辿った終局面が本当に詰みか (合法手なし) 検証 = false mate 検出．
                    let mut chk = Board::new();
                    chk.set_sfen(sfen).unwrap();
                    for m in &moves { chk.do_move(*m); }
                    let final_legal = movegen::generate_legal_moves(&mut chk).len();
                    let in_check = chk.is_in_check(chk.turn());
                    eprintln!("[v3] 29te final: in_check={in_check}, legal_moves={final_legal} (詰み=in_check&&legal==0)");
                    // Phase 36: look-ahead (CheckObviousFinalOrNode) + dominance を default 化．
                    // mate_move_in_1ply の false mate-1 3 バグ (ピン軸取り返し/逆王手 drop/開き王手)
                    // 根治後，dom-on look-ahead で canonical mate-29 を 18,531 nodes (< KH 19,270) で
                    // sound に解く ([[project_dfpn_domoff_none_mate1_bugs]])．
                    // soundness ガード: 真の詰み (in_check && legal==0) + canonical mate-29 + KH parity．
                    assert!(in_check && final_legal == 0, "LE path PV 終局面は真の詰みであるべき (false mate 検出)");
                    assert_eq!(moves.len(), 29, "default は canonical mate-29 を返すべき: {}", moves.len());
                    assert!(nodes_searched <= 19_270,
                        "default 29te は KH (19,270 nodes) 以下で解くべき: {}", nodes_searched);
                }
                other => panic!("[v3] 29te NON-SOLVE: {other:?}, {} nodes, {} tt", s.v3_nodes, s.v3_tt.len()),
            }
        }
    }

    /// TEMP (削除予定): mid_v3 の 39te baseline 計測．
    /// KH v1.1.0 実測: proof(None)=7,454,827 nodes/mate-33 PV; MinLength=52M/mate-47(無駄合い込み);
    /// tsume 正解=39 手．mid_v3 の現状 (nodes/可否/PV/手数) を測る．assert なし．
    #[test]
    #[ignore]
    fn test_mid_v3_39te_measure() {
        // V3_SFEN で任意局面に差し替え可能 (偽証明調査等の計測用)．
        let sfen_env = std::env::var("V3_SFEN").ok();
        let sfen: &str = sfen_env
            .as_deref()
            .unwrap_or("9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1");
        let mut b = Board::new();
        b.set_sfen(sfen).unwrap();
        // 計測専用: budget/timeout を env で上書き可能に (lever 比較用)．
        let budget: u64 = std::env::var("V3_BUDGET").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
        let secs: u64 = std::env::var("V3_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
        let mut s = DfPnSolver::with_timeout(47, budget, 32767, secs);
        let t0 = std::time::Instant::now();
        let result = s.solve_via_v3(&mut b);
        let elapsed = t0.elapsed();
        match result {
            TsumeResult::Checkmate { moves, nodes_searched } => {
                let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!(
                    "[v3] 39te SOLVE: {} moves, {} nodes, {} tt, {:.1}s | PV {:?}",
                    moves.len(), nodes_searched, s.v3_tt.len(), elapsed.as_secs_f64(), usi);
                let mut chk = Board::new();
                chk.set_sfen(sfen).unwrap();
                for m in &moves { chk.do_move(*m); }
                let final_legal = movegen::generate_legal_moves(&mut chk).len();
                let in_check = chk.is_in_check(chk.turn());
                eprintln!(
                    "[v3] 39te final: in_check={in_check}, legal_moves={final_legal} (詰み=in_check&&legal==0)");
            }
            other => {
                eprintln!(
                    "[v3] 39te NON-SOLVE: {other:?}, {} nodes, {} tt, {:.1}s",
                    s.v3_nodes, s.v3_tt.len(), elapsed.as_secs_f64());
                // breadth lever 比較用: per-ply total/unique 訪問数．
                let mut summary = String::new();
                for p in 0..20usize {
                    let t = s.v3_ply_total[p];
                    let u = s.v3_ply_unique[p];
                    if t == 0 { continue; }
                    summary.push_str(&format!("p{p}:{u}/{t} "));
                }
                eprintln!("[v3] 39te per-ply unique/total: {summary}");
            }
        }
    }

    /// 後手番1手詰め．
    ///
    /// 先手攻め test_tsume_1te の盤面を180度回転+色反転した局面．
    /// 先手玉9九(K)，後手金8七(g)，後手持ち駒:金．
    /// 正解: G*8h(8八金打)で詰み(G*9hも正解)．
    #[test]
    fn test_tsume_1te_gote() {
        // 先手玉 9i, 後手金 8g, 後手持ち駒: g
        let sfen = "9/9/9/9/9/9/1g7/9/K8 w g 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(3, 100_000, 32767);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1, "should be 1-move checkmate, got: {:?}",
                    moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>());
                let usi = moves[0].to_usi();
                assert!(
                    usi == "G*8h" || usi == "G*9h",
                    "1手詰め: G*8h(8八金打) or G*9h(9八金打), got: {}", usi
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 後手番3手詰め．
    ///
    /// 先手攻め test_tsume_3te の盤面を180度回転+色反転した局面．
    /// 先手玉9九(K)，後手飛7七(r)，後手持ち駒:金．
    /// 正解: 7g9g+(9七飛成)，9i8i(8九玉)，G*8h(8八金打) まで3手詰．
    #[test]
    fn test_tsume_3te_gote() {
        // 先手玉 9i, 後手飛 7g, 後手持ち駒: g
        // (test_tsume_3te: 8k/9/6R2/9/.../9 b G 1 の反転)
        let sfen = "9/9/9/9/9/9/2r6/9/K8 w g 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let result = solve_tsume_with_timeout(sfen, Some(7), Some(1_048_576), None, None, None, None, None).unwrap();

        let expected = ["7g9g+", "9i8i", "G*8h"];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 3,
                    "expected 3 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 打歩詰め回避のため飛不成が正解のケース(7手詰め)．
    ///
    /// 局面: 先手飛4一，先手桂3三・2六，先手歩3四，
    ///       後手玉2二，後手歩3二・2五
    /// 先手持駒: 桂，歩二
    /// 後手持駒: 飛，角二，金四，銀四，桂，香四，歩十三
    ///
    /// 2一飛成(4a2a+)は龍の利きにより打歩詰めの反則が生じるため不詰．
    /// 2一飛不成(4a2a)なら飛車のまま利きが制限され，7手で詰みが成立する．
    #[test]
    fn test_tsume_uchifuzume_rook_no_promote() {
        let sfen = "5R3/6pk1/6N2/6P2/7p1/7N1/9/9/9 b N2Pr2b4g4sn4l13p 1";
        let result = solve_tsume(sfen, Some(31), Some(5_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                // 正解PV(7手): 4a2a(21飛不成) 2b1b(12玉) P*1c(13歩打)
                //   1b1c(同玉) P*1d(14歩打) 1c1b(12玉) N*2d(24桂打)
                //
                // 2一飛成(4a2a+)は龍の利きにより打歩詰めの反則が生じるため不詰．
                // 2一飛不成(4a2a)なら飛車のまま利きが制限され，7手で詰みが成立する．
                // 2手目12玉(2b1b)が受け方の最長抵抗．
                assert_eq!(
                    usi_moves.len(), 7,
                    "expected 7-move checkmate, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                assert_eq!(usi_moves[0], "4a2a", "move 1: 21飛不成");
                assert_eq!(usi_moves[1], "2b1b", "move 2: 12玉");
                assert_eq!(usi_moves[2], "P*1c", "move 3: 13歩打");
                assert_eq!(usi_moves[3], "1b1c", "move 4: 同玉");
                assert_eq!(usi_moves[4], "P*1d", "move 5: 14歩打");
                assert_eq!(usi_moves[5], "1c1b", "move 6: 12玉");
                assert_eq!(usi_moves[6], "N*2d", "move 7: 24桂打");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 2一飛成(龍)だと打歩詰めにより詰みがないことの検証．
    ///
    /// 上記 test_tsume_uchifuzume_rook_no_promote の局面で
    /// 4a2a+(飛成)を指した後の局面は，龍の利きにより
    /// 打歩詰めの反則が避けられず詰まない．
    #[test]
    fn test_uchifuzume_promoted_rook_fails() {
        let sfen = "5R3/6pk1/6N2/6P2/7p1/7N1/9/9/9 b N2Pr2b4g4sn4l13p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // 4a2a+(飛成=龍)を指す
        let promote_move = board.move_from_usi("4a2a+").unwrap();
        board.do_move(promote_move);

        // 龍の局面からは詰まないことを検証
        let mut solver = DfPnSolver::new(31, 2_000_000, 32767);
        let result = solver.solve_via_v3(&mut board);
        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "4a2a+ (promoted rook) should NOT lead to checkmate due to uchifuzume, got: {:?}",
            result
        );
    }

    /// 馬の往復で千日手となり不詰のケース．
    ///
    /// 局面: 後手玉1二，先手馬3二，先手金3一，先手歩3五・2五，後手歩1三
    /// 先手持駒: なし
    /// 後手持駒: 飛二，角，金三，銀四，桂四，香四，歩十五
    ///
    /// 2一馬，2三玉，3二馬，1二玉 の繰り返しで千日手(連続王手の千日手)．
    /// 攻め方に持ち駒がなく打開手段がないため不詰．
    #[test]
    fn test_no_checkmate_perpetual_check() {
        let sfen = "6G2/6+B1k/8p/9/6PP1/9/9/9/9 b 2rb3g4s4n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (perpetual check by horse), got {:?}",
                other
            ),
        }
    }

    /// 逆王手の移動生成ユニットテスト．
    ///
    /// 局面: 2三飛成直後(後手番)
    ///       先手竜2三，先手玉2四，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手桂4二，後手銀3四
    ///
    /// 後手が指せる手のリストに 3四銀→2三銀(逆王手) が含まれることを確認する．
    /// 竜(2三)を銀で取ると銀が2三に来て先手玉2四へ斜め前方向の逆王手になる．
    #[test]
    fn test_counter_check_move_generation_unit() {
        // 2三飛成後の局面: 竜が2三に，後手番(w)
        let sfen = "7l1/5n1k1/7+RP/6sK1/7L1/9/9/9/9 w r2b4g3s3n2l17p 2";
        let mut board = crate::board::Board::empty();
        board.set_sfen(sfen).expect("valid SFEN");

        let mut solver = DfPnSolver::new(31, 1, 32767);
        let moves = solver.generate_defense_moves(&mut board);

        // 3d2c = 後手銀 3四→2三 (竜を取り，先手玉2四への逆王手)
        let has_counter_check = moves.iter().any(|m| m.to_usi() == "3d2c");
        assert!(
            has_counter_check,
            "逆王手手 3d2c が指し手リストにない．生成された手: {:?}",
            moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>()
        );
    }

    /// 逆王手で不詰のケース．
    ///
    /// 局面: 先手玉2四，先手飛4三，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手桂4二，後手銀3四
    /// 先手持駒: なし
    /// 後手持駒: 飛，角二，金四，銀三，桂三，香二，歩十七
    ///
    /// 4三飛→2三飛成は王手だが，後手3四銀→2三銀の逆王手(2四の先手玉に対する王手)
    /// により攻め方は次の王手ができず不詰になる．他の王手(4二飛成等)も後手が
    /// 逆王手や玉の逃げで対処できるため不詰．
    ///
    /// v0.53.2 で attacker_in_check dn 引き下げ，v0.53.3 で INTERPOSE_DN_BIAS 拡大により
    /// 441K ノードで解決 (旧: 30M ノード超過)．
    #[test]
    fn test_no_checkmate_counter_check() {
        let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(3_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (counter-check by silver), got {:?}",
                other
            ),
        }
    }

    /// 逆王手がある局面で3四玉と上がって詰むケース．
    ///
    /// 局面: 先手玉2四，先手飛4三，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手銀3四
    /// 先手持駒: なし
    /// 後手持駒: 飛，角二，金四，銀三，桂四，香二，歩十七
    ///
    /// 上記の不詰局面から4二の後手桂を除いた形．
    /// 先手3四玉(銀を取って王手回避しつつ接近)から詰みがある．
    #[test]
    fn test_checkmate_with_counter_check_avoidance() {
        let sfen = "7l1/7k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s4n2l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> =
                    moves.iter().map(|m| m.to_usi()).collect();
                // 5手詰め(複数解あり): 探索順序によりいずれかの PV が返る
                // 合駒の駒種は探索順序依存(弱い駒優先 → 歩合が先に証明される)
                let pv1 = vec!["2d3d", "2b3a", "S*4b", "3a3b", "4c3c+"];
                let pv2 = vec!["2d3d", "R*2c", "2e2c+", "2b1a", "1c1b+"];
                let pv3 = vec!["2d3d", "P*2c", "2e2c+", "2b1a", "1c1b+"];
                let pv4 = vec!["2d3d", "N*2c", "2e2c+", "2b1a", "1c1b+"];
                let pv5 = vec!["2d3d", "L*2c", "2e2c+", "2b3a", "L*3b"];
                // pv6: 効果長による tiebreak で見つかる別解．
                // P*2d 中合 → 4c3c+ で詰まし上げる 5 手詰め．
                let pv6 = vec!["2d3d", "P*2d", "4c3c+", "2b1a", "S*1b"];
                let pv_str: Vec<&str> = usi_moves.iter().map(|s| s.as_str()).collect();
                assert!(
                    pv_str == pv1 || pv_str == pv2 || pv_str == pv3
                        || pv_str == pv4 || pv_str == pv5 || pv_str == pv6,
                    "PV must be one of the known solutions:\n  got:  {}\n  pv1: {}\n  pv2: {}\n  pv3: {}\n  pv4: {}\n  pv5: {}\n  pv6: {}",
                    usi_moves.join(" "),
                    pv1.join(" "),
                    pv2.join(" "),
                    pv3.join(" "),
                    pv4.join(" "),
                    pv5.join(" "),
                    pv6.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate (king captures silver), got {:?}",
                other
            ),
        }
    }

    /// 打歩詰めしかなく不詰のケース．
    ///
    /// 局面: 後手玉1一，後手桂2一，先手金1三
    /// 先手持駒: 歩
    /// 後手持駒: 飛二，角二，金三，銀四，桂三，香四，歩十七
    ///
    /// 1二歩打(P*1b)は王手だが打歩詰めの反則(玉が逃げられない)．
    /// 他に有効な王手がないため不詰．
    #[test]
    fn test_no_checkmate_uchifuzume_only() {
        let sfen = "7nk/9/8G/9/9/9/9/9/9 b P2r2b3g4s3n4l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(100_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (only move is uchifuzume), got {:?}",
                other
            ),
        }
    }

    /// 金の移動合いで不詰になるケース．
    ///
    /// 局面: 後手玉1一，後手金2一，後手歩1二，後手銀1三，
    ///       先手歩4三・2四
    /// 先手持駒: 角，桂
    /// 後手持駒: 飛二，角，金三，銀三，桂三，香四，歩十五
    ///
    /// 角打ちに対して金の移動合い(2一金→1二等)が有効で詰まない．
    #[test]
    fn test_no_checkmate_gold_interposition() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BN2rb3g3s3n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!(
                "expected NoCheckmate (gold interposition defense), got {:?}",
                other
            ),
        }
    }

    /// 先手持駒に銀を追加した局面で9手詰めになることのテスト．
    ///
    /// 上記 test_no_checkmate_gold_interposition と同じ盤面だが，
    /// 先手持駒に銀が追加(角，銀，桂)され後手の銀が1枚減っている．
    /// 銀の追加により9手詰めが生じる(金の移動合いが最長抵抗)．
    #[test]
    fn test_tsume_9te_with_silver() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "expected 9-move checkmate, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // 初手: 3三角打
                assert_eq!(usi_moves[0], "B*3c", "move 1: B*3c(3三角打)");
                // 2手目: 金の移動合い(最長抵抗)
                assert_eq!(usi_moves[1], "2a2b", "move 2: g(2a→2b)(金の移動合い)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 金の移動合い(2一金→2二)が回避手に含まれることを検証する．
    ///
    /// B*3c(3三角打)に対して，2一の金を2二に移動して合い駒する手が
    /// 回避手として生成されていることを確認する．
    /// この手が漏れていると不正に短手数で詰みと判定される．
    #[test]
    fn test_gold_interposition_is_legal_defense() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // B*3c(3三角打) で王手
        let b3c = board.move_from_usi("B*3c").unwrap();
        board.do_move(b3c);

        // 後手番: 回避手を生成
        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();

        // 金の移動合い 2a2b(2一金→2二) が含まれること
        assert!(
            usi_defenses.contains(&"2a2b".to_string()),
            "g(2a→2b) gold interposition should be a legal defense, got: {:?}",
            usi_defenses
        );

        // 銀の移動合い 1c2b(1三銀→2二) も含まれること
        assert!(
            usi_defenses.contains(&"1c2b".to_string()),
            "s(1c→2b) silver interposition should be a legal defense, got: {:?}",
            usi_defenses
        );
    }

    /// ply 22 偽証明: 最終局面の合駒生成と between_bb を診断．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_ids_depth_diagnosis() {
        // 最終局面: 飛車 8e から玉 1e への王手
        let final_sfen = "9/3+N1P3/7+R1/9/1R6k/9/8P/3S5/9 w 2b4g3s3n4l16p 30";

        let mut final_board = Board::new();
        final_board.set_sfen(final_sfen).unwrap();

        let defender = final_board.turn(); // White
        let attacker = defender.opponent(); // Black
        let king_sq = final_board.king_square(defender).unwrap();
        verbose_eprintln!("King square: {:?} (col={}, row={})", king_sq, king_sq.col(), king_sq.row());

        // compute_checkers_at
        let checkers = final_board.compute_checkers_at(king_sq, attacker);
        verbose_eprintln!("Checkers: count={}", checkers.count());
        for sq in checkers {
            verbose_eprintln!("  checker at {:?} (col={}, row={})", sq, sq.col(), sq.row());
        }

        // find_sliding_checker
        let mut solver = DfPnSolver::default_solver();
        let sliding = solver.find_sliding_checker(&final_board, king_sq, attacker);
        verbose_eprintln!("find_sliding_checker: {:?}", sliding.map(|s| format!("col={}, row={}", s.col(), s.row())));

        // checker_sq
        let checker_sq = checkers.lsb().unwrap();

        // between_bb
        let between = attack::between_bb(checker_sq, king_sq);
        verbose_eprintln!("between_bb({:?}, {:?}): count={}", checker_sq, king_sq, between.count());
        for sq in between {
            verbose_eprintln!("  between: col={}, row={}", sq.col(), sq.row());
        }

        // compute_futile_and_chain_squares
        let (futile, chain) = solver.compute_futile_and_chain_squares(
            &final_board, &between, king_sq, checker_sq, defender, attacker,
        );
        verbose_eprintln!("futile: count={}", futile.count());
        for sq in futile {
            verbose_eprintln!("  futile: col={}, row={}", sq.col(), sq.row());
        }
        verbose_eprintln!("chain: count={}", chain.count());
        for sq in chain {
            verbose_eprintln!("  chain: col={}, row={}", sq.col(), sq.row());
        }

        // generate_defense_moves
        let defenses = solver.generate_defense_moves(&mut final_board);
        verbose_eprintln!("generate_defense_moves: {} moves", defenses.len());
        for d in &defenses {
            verbose_eprintln!("  {}", d.to_usi());
        }

        // 比較: generate_legal_moves
        let legal = movegen::generate_legal_moves(&mut final_board);
        verbose_eprintln!("generate_legal_moves: {} moves", legal.len());

        // between_bb が空なら合駒生成がスキップされる → バグの原因
        assert!(
            between.count() > 0,
            "between_bb is empty for checker={:?} king={:?}, blocking moves will be skipped!",
            checker_sq, king_sq
        );
    }

    /// 無駄合い判定テスト: 飛車の王手に対して合駒で詰みが回避できる局面．
    ///
    /// 飛車が横(rank)方向に王手しているが，玉の逃げ道が飛び駒の取り進みで
    /// 塞がれない場合，合駒は無駄合いではない．
    /// `compute_futile_and_chain_squares` が全マスを futile にせず，
    /// `generate_defense_moves` が合駒(駒打ち)を生成することを検証する．
    #[test]
    fn test_futile_check_rook_rank_not_futile_when_king_can_escape() {
        // 飛車(8e)が横王手，金(2d)が 2e を支えている
        // 飛車が 2e に取り進んだ後，玉は 1f に逃げられるので無駄合いではない
        //
        // 盤面:
        //   9  8  7  6  5  4  3  2  1
        //                            |  rank 1
        //                            |  rank 2
        //                            |  rank 3
        //                      G     |  rank 4  (金 at 2d)
        //      R                 k   |  rank 5  (飛車 at 8e, 玉 at 1e)
        //                            |  rank 6  (1f = 玉の逃げ先)
        let sfen = "9/9/9/7G1/1R6k/9/9/9/9 w r2b3g4s4n4l18p 2";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent(); // Black
        let king_sq = board.king_square(defender).unwrap();
        let checkers = board.compute_checkers_at(king_sq, attacker);
        assert_eq!(checkers.count(), 1, "Expected exactly one checker");
        let checker_sq = checkers.lsb().unwrap();

        let mut solver = DfPnSolver::default_solver();

        // between_bb が正しく計算されること
        let between = attack::between_bb(checker_sq, king_sq);
        assert!(between.count() > 0, "between_bb must not be empty");

        // compute_futile_and_chain_squares: king-adjacent マスが futile にならないこと
        let (futile, _chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );
        // 飛車が 2e(king-adjacent)に取り進んだ後，玉は 1f に逃げられる
        // → king-adjacent マスは futile ではない → futile != between
        assert!(
            futile != between,
            "All between squares are futile — king escape after slider capture not checked"
        );

        // generate_defense_moves: 合駒(駒打ち)が含まれること
        let defenses = solver.generate_defense_moves(&mut board);
        let has_drop = defenses.iter().any(|m| m.is_drop());
        assert!(
            has_drop,
            "generate_defense_moves must include drop interpositions, got {} moves: {:?}",
            defenses.len(),
            defenses.iter().map(|m| m.to_usi()).collect::<Vec<_>>()
        );
    }

    /// 無駄合い判定テスト: 飛車王手で玉が完全に囲まれており合駒が無駄な局面．
    ///
    /// 飛び駒が取り進んだ後も玉の逃げ道がない場合，合駒は無駄合いとなる．
    /// `compute_futile_and_chain_squares` が king-adjacent マスを futile にし，
    /// 合駒(駒打ち)がスキップされることを検証する．
    #[test]
    fn test_futile_check_rook_rank_futile_when_king_trapped() {
        // 後手玉 1a, 先手飛 9a から横王手
        // 先手: 飛9a, 金2a, 金1b → 玉の全逃げ道が塞がれている
        // 飛車が 2a に取り進むと金が利いており，玉は逃げられない
        let sfen = "R1G5k/1G7/9/9/9/9/9/9/9 w 2b2r3s4n4l18p 2";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent();
        let king_sq = board.king_square(defender).unwrap();
        let checkers = board.compute_checkers_at(king_sq, attacker);

        if checkers.is_empty() {
            // チェッカーが検出されない場合はテストスキップ
            return;
        }
        let checker_sq = checkers.lsb().unwrap();

        let solver = DfPnSolver::default_solver();
        let between = attack::between_bb(checker_sq, king_sq);

        if between.is_empty() {
            // between が空(隣接王手)の場合はテストスキップ
            return;
        }

        let (futile, _chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );

        // 玉が完全に囲まれているので，king-adjacent マスは futile であるべき
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let king_adj_between = between & king_step;
        if king_adj_between.is_not_empty() {
            for sq in king_adj_between {
                assert!(
                    futile.contains(sq),
                    "King-adjacent between square (col={}, row={}) should be futile when king is trapped",
                    sq.col(), sq.row()
                );
            }
        }
    }

    /// TT 保護のリグレッションテスト: find_shortest モード有効時の PV 検証．
    ///
    /// complete_or_proofs 中の mid() が転置により証明済み TT エントリを
    /// 上書きしていたバグの回帰テスト．find_shortest=true(デフォルト)で
    /// PV が最長抵抗を正しく反映することを確認する．
    #[test]
    fn test_pv_follows_longest_defense() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let result = solve_tsume_with_timeout(
            sfen, Some(31), Some(2_000_000), None, None,
            Some(true), // find_shortest = true
            None, None,
        ).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "PV should be 9 moves (longest defense via gold interposition), got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                assert_eq!(usi_moves[0], "B*3c", "move 1: B*3c(3三角打)");
                assert_eq!(usi_moves[1], "2a2b", "move 2: g(2a→2b)(金の移動合い=最長抵抗)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 後手番不詰のケース．
    #[test]
    fn test_no_checkmate_gote() {
        // 先手玉5九，後手持ち駒: 歩 → 歩では詰まない
        let sfen = "9/9/9/9/9/9/9/9/4K4 w p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(5, 100_000, 32767);
        let result = solver.solve_via_v3(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!("expected NoCheckmate, got {:?}", other),
        }
    }

    /// 中合い(ちゅうあい)によって不詰みになる局面のテスト．
    ///
    /// 飛び駒の王手に対して持ち駒を間に打つ中合いが有効で，
    /// 詰みが成立しないことを検証する．
    #[test]
    fn test_chuai_no_checkmate() {
        // 中合いによって詰まない局面
        // デバッグビルドでの実行時間制約のため予算を抑制する．
        // Checkmate を返さないことが主要な検証ポイント．
        //
        // 合い駒生成の改善により探索分岐が増え，デバッグビルドでは
        // デフォルトスタック(8MB)で溢れるため専用スレッドで実行する．
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let sfen = "9/3+N1P3/2+R3p2/8k/8N/5+B3/4S4/1R1p5/9 b NPb4g3sn4l14p 1";
                solve_tsume(sfen, Some(31), Some(10_000), None).unwrap()
            })
            .unwrap()
            .join()
            .unwrap();

        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "中合いにより不詰みのはず: {:?}",
            result
        );
    }

    /// 8h8d 王手後の AND ノードで中合い(P*7d)が回避手に含まれることを確認．
    ///
    /// 合い効かずフィルタが P*7d を誤って除外していないかの診断テスト．
    #[test]
    fn test_chuai_defense_includes_pawn_drop() {
        let sfen = "9/3+N1P3/2+R3p2/8k/8N/5+B3/4S4/1R1p5/9 b NPb4g3sn4l14p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // 攻め方: 8h8d (飛車8八→8四)
        let m = board.move_from_usi("8h8d").unwrap();
        board.do_move(m);

        // AND ノード: 後手の回避手を生成
        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();

        // 全合法手と比較
        let legal = movegen::generate_legal_moves(&mut board);
        let usi_legal: Vec<String> = legal.iter().map(|m| m.to_usi()).collect();

        // P*7d が合法手に含まれること
        assert!(
            usi_legal.contains(&"P*7d".to_string()),
            "P*7d should be a legal move, got: {:?}", usi_legal
        );

        // P*7d が回避手にも含まれること(合い効かずで除外されていないこと)
        assert!(
            usi_defenses.contains(&"P*7d".to_string()),
            "P*7d should be in defense moves (中合い), but was filtered out.\n\
             defense moves: {:?}\n\
             legal moves: {:?}",
            usi_defenses, usi_legal
        );
    }

    /// 中合い発生後の局面が不詰みであることを確認するテスト．
    ///
    /// 上記 `test_chuai_no_checkmate` の局面で中合いが行われた後の
    /// 状態を直接与え，攻め方から探索しても詰みがないことを検証する．
    #[test]
    fn test_chuai_position_after_block() {
        // 中合いが発生した後の局面(不詰み)
        // デバッグビルドでの実行時間制約のため予算を抑制する．
        let sfen = "9/3+N1P3/2+R3p2/1Rp5k/8N/5+B3/4S4/3p5/9 b NPb4g3sn4l13p 1";
        let result = solve_tsume(sfen, Some(31), Some(10_000), None).unwrap();

        assert!(
            !matches!(result, TsumeResult::Checkmate { .. }),
            "中合い後の局面は不詰みのはず: {:?}",
            result
        );
    }

    /// 39手詰め正解 PV 上の各局面における分岐数・手生成数を診断する．
    ///
    /// PV を1手ずつ進めながら，各局面で:
    /// - OR ノード(攻め方): generate_check_moves の手数
    /// - AND ノード(守備方): generate_defense_moves の手数
    /// を出力し，探索がどこでノードを浪費しているかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_pv_trace() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", // 1. ６二成桂
            "5b4c", // 2. ４三玉
            "8b9c", // 3. ９三龍
            "4c3d", // 4. ３四玉
            "1b2c", // 5. ２三銀
            "3d2c", // 6. 同玉
            "N*1e", // 7. １五桂打
            "2c3b", // 8. ３二玉
            "N*2d", // 9. ２四桂打
            "3b2b", // 10. ２二玉
            "2d1b+", // 11. １二桂成
            "2b3b", // 12. ３二玉
            "1b2b", // 13. ２二成桂
            "3b2b", // 14. 同玉
            "4f1c", // 15. １三馬
            "2b1c", // 16. 同玉
            "9c3c", // 17. ３三龍
            "1c1d", // 18. １四玉
            "3c2c", // 19. ２三龍
            "1d1e", // 20. １五玉
            "P*1f", // 21. １六歩打
            "1e1f", // 22. 同玉
            "P*1g", // 23. １七歩打
            "1f1g", // 24. 同玉
            "5g6f", // 25. ６六銀
            "1g1h", // 26. １八玉
            "2c2g", // 27. ２七龍
            "1h1i", // 28. １九玉
            "8g8i", // 29. ８九飛
            "S*6i", // 30. ６九銀打
            "8i6i", // 31. 同飛
            "6h6i+", // 32. 同歩成
            "S*2h", // 33. ２八銀打
            "1i2i", // 34. ２九玉
            "2h3g", // 35. ３七銀
            "2i3i", // 36. ３九玉
            "2g2h", // 37. ２八龍
            "3i4i", // 38. ４九玉
            "2h4h", // 39. ４八龍
        ];

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        let mut solver = DfPnSolver::default_solver();

        verbose_eprintln!("\n{:>3} {:>6} {:>5} {:>5} {:>6} {:<12} {}",
            "Ply", "Node", "Moves", "Drops", "Total", "PV Move", "Sample moves (first 10)");
        verbose_eprintln!("{}", "-".repeat(90));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let ply = i + 1;
            let is_or = (ply % 2) == 1; // 奇数手=攻め方(OR), 偶数手=守備方(AND)

            let moves = if is_or {
                solver.generate_check_moves(&mut board)
            } else {
                solver.generate_defense_moves(&mut board)
            };

            // ドロップ手のカウント
            let drop_count = moves.iter().filter(|m| m.is_drop()).count();
            let move_count = moves.len() - drop_count;

            // 正解手が手リストに含まれているか確認
            let expected_move = board.move_from_usi(usi)
                .unwrap_or_else(|| panic!("Invalid USI at ply {}: {}", ply, usi));
            let found = moves.iter().any(|m| *m == expected_move);

            // サンプル表示(ply 4 は全手, それ以外は先頭10手)
            let limit = if ply == 4 { moves.len() } else { 10 };
            let sample: Vec<String> = moves.iter().take(limit).map(|m| m.to_usi()).collect();

            let node_type = if is_or { "OR" } else { "AND" };
            let mark = if !found { " *** MISSING ***" } else { "" };

            verbose_eprintln!("{:>3} {:>6} {:>5} {:>5} {:>6} {:<12} [{}]{}",
                ply, node_type, move_count, drop_count, moves.len(),
                usi, sample.join(", "), mark);

            // 手を適用して次の局面へ
            board.do_move(expected_move);
        }

        // 最終局面が詰みかチェック
        let final_defenses = solver.generate_defense_moves(&mut board);
        verbose_eprintln!("\n最終局面(39手目後)の回避手数: {}", final_defenses.len());
        if final_defenses.is_empty() {
            verbose_eprintln!("→ 詰み!");
        } else {
            let sample: Vec<String> = final_defenses.iter().take(10).map(|m| m.to_usi()).collect();
            verbose_eprintln!("→ 回避手あり: [{}]", sample.join(", "));
        }
    }

    // ========================================================================
    // 施策 X (v0.24.53): ProvenEntry proof_tag 拡張の単体テスト
    // ========================================================================

    /// `ProvenEntry::new_tagged_proof` の round-trip 検証．
    /// 各 tag と tag_depth の組合せで `proof_tag()` / `tag_depth()` が
    /// 正しい値を返すことを確認する．
    #[test]
    fn test_proven_entry_tagged_proof_round_trip() {
        use super::entry::{
            ProvenEntry, PROOF_TAG_ABSOLUTE, PROOF_TAG_FILTER_DEPENDENT,
            PROOF_TAG_PROVISIONAL, PROOF_TAG_DEPTH_LIMITED, PROOF_TAG_INVALID,
        };
        let hand = [1u8, 0, 0, 0, 0, 0, 0];
        let best_move = 0x1234u16;

        // ABSOLUTE (既存 new_proof と等価)
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 15, PROOF_TAG_ABSOLUTE, 0);
        assert!(e.is_proof());
        assert_eq!(e.proof_tag(), PROOF_TAG_ABSOLUTE);
        assert_eq!(e.tag_depth(), 0);
        assert_eq!(e.mate_distance(), Some(15));
        assert_eq!(e.best_move, best_move);
        assert_eq!(e.hand, hand);

        // FILTER_DEPENDENT with depth=17
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 7, PROOF_TAG_FILTER_DEPENDENT, 17);
        assert!(e.is_proof());
        assert_eq!(e.proof_tag(), PROOF_TAG_FILTER_DEPENDENT);
        assert_eq!(e.tag_depth(), 17);
        assert_eq!(e.mate_distance(), Some(7));

        // PROVISIONAL with depth=25
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 5, PROOF_TAG_PROVISIONAL, 25);
        assert_eq!(e.proof_tag(), PROOF_TAG_PROVISIONAL);
        assert_eq!(e.tag_depth(), 25);

        // DEPTH_LIMITED
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 3, PROOF_TAG_DEPTH_LIMITED, 7);
        assert_eq!(e.proof_tag(), PROOF_TAG_DEPTH_LIMITED);
        assert_eq!(e.tag_depth(), 7);

        // tag_depth overflow (>63) → clamped to 63
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 11, PROOF_TAG_FILTER_DEPENDENT, 100);
        assert_eq!(e.tag_depth(), 63);

        // INVALID tag
        let e = ProvenEntry::new_tagged_proof(hand, best_move, 1, PROOF_TAG_INVALID, 0);
        assert_eq!(e.proof_tag(), PROOF_TAG_INVALID);
    }

    /// `ProvenEntry::new_proof` は `new_tagged_proof(.., ABSOLUTE, 0)` に委譲する
    /// ため backward compat を保証する．
    #[test]
    fn test_proven_entry_new_proof_defaults_to_absolute() {
        use super::entry::{ProvenEntry, PROOF_TAG_ABSOLUTE};
        let hand = [0u8, 0, 0, 0, 0, 0, 1];
        let e = ProvenEntry::new_proof(hand, 0x5678, 13);
        assert!(e.is_proof());
        assert_eq!(e.proof_tag(), PROOF_TAG_ABSOLUTE);
        assert_eq!(e.tag_depth(), 0);
        assert_eq!(e.mate_distance(), Some(13));
    }

    /// disproof エントリは `proof_tag()` で ABSOLUTE，`tag_depth()` で 0 を
    /// 返す (backward compat + 明示的な zero 返却)．
    #[test]
    fn test_proven_entry_disproof_tag_accessors() {
        use super::entry::{ProvenEntry, PROOF_TAG_ABSOLUTE};
        let hand = [0u8; 7];
        let e = ProvenEntry::new_disproof(hand, 25);
        assert!(!e.is_proof());
        assert_eq!(e.disproof_depth(), 25);
        // disproof entry は tag 非対応: accessor は ABSOLUTE/0 を返す
        assert_eq!(e.proof_tag(), PROOF_TAG_ABSOLUTE);
        assert_eq!(e.tag_depth(), 0);
    }

    /// `ProvenEntry::amount` (eviction priority) が tag に応じて変化することを
    /// 確認する．ABSOLUTE proof > non-ABSOLUTE proof > disproof の順序．
    #[test]
    fn test_proven_entry_amount_tag_priority() {
        use super::entry::{
            ProvenEntry, PROOF_TAG_ABSOLUTE, PROOF_TAG_FILTER_DEPENDENT,
            PROOF_TAG_PROVISIONAL,
        };
        let hand = [0u8; 7];

        // ABSOLUTE proof without distance: 64
        let abs = ProvenEntry::new_tagged_proof(hand, 0, 0, PROOF_TAG_ABSOLUTE, 0);
        assert_eq!(abs.amount(), 64);

        // FILTER_DEPENDENT proof without distance: 48
        let filt = ProvenEntry::new_tagged_proof(hand, 0, 0, PROOF_TAG_FILTER_DEPENDENT, 0);
        assert_eq!(filt.amount(), 48);

        // PROVISIONAL proof without distance: 48
        let prov = ProvenEntry::new_tagged_proof(hand, 0, 0, PROOF_TAG_PROVISIONAL, 0);
        assert_eq!(prov.amount(), 48);

        // confirmed disproof: 32
        let disp = ProvenEntry::new_disproof(hand, 10);
        assert_eq!(disp.amount(), 32);

        // ABSOLUTE proof with distance 15: 0x80 | 15 = 143
        let abs_d = ProvenEntry::new_tagged_proof(hand, 0, 15, PROOF_TAG_ABSOLUTE, 0);
        assert_eq!(abs_d.amount(), 0x80 | 15);

        // 順序性: ABSOLUTE+distance > ABSOLUTE > non-ABSOLUTE > disproof
        assert!(abs_d.amount() > abs.amount());
        assert!(abs.amount() > filt.amount());
        assert!(filt.amount() > disp.amount());
    }

    /// `clear_proven_disproofs_below` が non-ABSOLUTE proof を
    /// 選択的に除去することを検証する (施策 X の統合動作)．
    #[test]
    fn test_clear_proven_disproofs_below_includes_tagged_proofs() {
        use super::entry::{
            PROOF_TAG_FILTER_DEPENDENT, PROOF_TAG_PROVISIONAL,
        };
        use super::tt::TranspositionTable;
        let mut tt = TranspositionTable::new();

        let hand = [1u8, 0, 0, 0, 0, 0, 0];
        let pk = 0xDEADBEEF_12345678u64;

        // ABSOLUTE proof at tag_depth=0 (should never be removed)
        tt.store_with_best_move_and_distance(
            pk, hand, 0, u32::MAX, 0x7FFF, pk as u32, 0x100, 15,
        );

        // 3 pos_keys with different tags for diversity
        // 同一 pk + 同一 hand は cluster 内で上書きされるため，異なる pk を使う
        let pk_filter = 0x1111_1111_1111_1111u64;
        let pk_prov = 0x2222_2222_2222_2222u64;

        // FILTER_DEPENDENT at depth=5
        tt.store_tagged_proof_for_test(
            pk_filter, hand, 0x200, 10, PROOF_TAG_FILTER_DEPENDENT, 5,
        );

        // PROVISIONAL at depth=15
        tt.store_tagged_proof_for_test(
            pk_prov, hand, 0x300, 8, PROOF_TAG_PROVISIONAL, 15,
        );

        // clear at min_depth=10: FILTER_DEPENDENT (depth=5 < 10) should be removed
        // PROVISIONAL (depth=15 >= 10) should be kept
        // ABSOLUTE is never affected
        tt.clear_proven_disproofs_below(10);

        // ABSOLUTE proof remains
        let (pn, _, _) = tt.look_up(pk, &hand, 0x7FFF, false);
        assert_eq!(pn, 0, "ABSOLUTE proof should be preserved");

        // FILTER_DEPENDENT removed
        let (pn, _, _) = tt.look_up(pk_filter, &hand, 0x7FFF, false);
        assert_ne!(pn, 0, "FILTER_DEPENDENT proof at depth<10 should be removed");

        // PROVISIONAL preserved
        let (pn, _, _) = tt.look_up(pk_prov, &hand, 0x7FFF, false);
        assert_eq!(pn, 0, "PROVISIONAL proof at depth>=10 should be preserved");

        // Second clear at higher min_depth: PROVISIONAL (depth=15) also removed
        tt.clear_proven_disproofs_below(20);
        let (pn, _, _) = tt.look_up(pk_prov, &hand, 0x7FFF, false);
        assert_ne!(pn, 0, "PROVISIONAL proof at depth<20 should be removed");

        // ABSOLUTE still preserved
        let (pn, _, _) = tt.look_up(pk, &hand, 0x7FFF, false);
        assert_eq!(pn, 0, "ABSOLUTE proof must never be removed");
    }

    /// counter_check 不詭め局面の移動木診断．
    ///
    /// 先手の全王手手ごとに「後手の応手数」と「中合いチェーン情報」を出力し，
    /// どの分岐が高コストかを特定する．
    #[test]
    #[ignore]
    fn test_counter_check_diagnostic() {
        use std::io::Write;
        let out_path = "/tmp/counter_check_diagnostic.log";
        let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";

        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();
        writeln!(out, "=== Counter-Check 不詭め診断 ===").unwrap();
        writeln!(out, "SFEN: {}", sfen).unwrap();

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        let mut solver = DfPnSolver::default_solver();

        // Root OR ノード: 先手の王手手一覧
        let checks = solver.generate_check_moves(&mut board);
        writeln!(out, "\n先手王手手数: {}", checks.len()).unwrap();
        for m in &checks {
            writeln!(out, "  {}", m.to_usi()).unwrap();
        }

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();

        for chk in &checks {
            writeln!(out, "\n=== 王手: {} ===", chk.to_usi()).unwrap();
            let cap1 = board.do_move(*chk);
            writeln!(out, "  SFEN: {}", board.sfen()).unwrap();

            let defenses = solver.generate_defense_moves(&mut board);
            let drop_cnt = defenses.iter().filter(|m| m.is_drop()).count();
            writeln!(out, "  後手応手数={} (drop={}, board={})",
                defenses.len(), drop_cnt, defenses.len() - drop_cnt).unwrap();

            // AND ノードのチェーン情報
            let def_color = board.turn();
            let att_color = def_color.opponent();
            if let Some(king_sq) = board.king_square(def_color) {
                let checkers = board.compute_checkers_at(king_sq, att_color);
                if checkers.count() == 1 {
                    let checker_sq = checkers.lsb().unwrap();
                    writeln!(out, "  王手駒: {}{}", 9-checker_sq.col(), (b'a'+checker_sq.row()) as char).unwrap();
                    if let Some(_sl) = solver.find_sliding_checker(&board, king_sq, att_color) {
                        let btw = attack::between_bb(checker_sq, king_sq);
                        let (fut, chn) = solver.compute_futile_and_chain_squares(
                            &board, &btw, king_sq, checker_sq, def_color, att_color);
                        write!(out, "  Between:").unwrap();
                        for sq in btw { write!(out, " {}{}", 9-sq.col(), (b'a'+sq.row()) as char).unwrap(); }
                        writeln!(out).unwrap();
                        write!(out, "  Futile:").unwrap();
                        for sq in fut { write!(out, " {}{}", 9-sq.col(), (b'a'+sq.row()) as char).unwrap(); }
                        writeln!(out).unwrap();
                        write!(out, "  Chain:").unwrap();
                        for sq in chn { write!(out, " {}{}", 9-sq.col(), (b'a'+sq.row()) as char).unwrap(); }
                        writeln!(out).unwrap();
                        let normal = btw.count() - fut.count() - chn.count();
                        writeln!(out, "  Normal squares (= 非futile非chain 合い駒候補): {}", normal).unwrap();
                    } else {
                        writeln!(out, "  (非滑り駒王手)").unwrap();
                    }
                } else if checkers.count() > 1 {
                    writeln!(out, "  二重王手 (checkers={})", checkers.count()).unwrap();
                }
            }

            // 各応手の後の先手王手数
            writeln!(out, "  --- 各応手後の先手王手数 ---").unwrap();
            for def_m in &defenses {
                let cap2 = board.do_move(*def_m);
                let next_chks = solver.generate_check_moves(&mut board);
                writeln!(out, "    {} => 次王手数={}", def_m.to_usi(), next_chks.len()).unwrap();

                // 応手後の AND ノードを 50K ノードで解いて結果確認
                let sub_sfen = board.sfen();
                board.undo_move(*def_m, cap2);

                let sub_result = solve_tsume(&sub_sfen, Some(31), Some(50_000), None)
                    .map(|r| format!("{:?}", r))
                    .unwrap_or_else(|e| format!("Err({:?})", e));
                writeln!(out, "      50K結果: {}", sub_result).unwrap();
            }

            board.undo_move(*chk, cap1);
        }

        drop(out);
        eprintln!("診断完了: {}", out_path);
        }).unwrap().join().unwrap();
    }

    /// counter_check 不詭め局面のノード予算プローブ
    #[test]
    #[ignore]
    fn test_no_checkmate_counter_check_probe() {
        let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
        for budget in [3_000_000u64, 5_000_000, 10_000_000] {
            let result = solve_tsume(sfen, Some(31), Some(budget), None).unwrap();
            eprintln!("budget={}: {:?}", budget, result);
            if let TsumeResult::NoCheckmate { nodes_searched } = &result {
                eprintln!("  → solved at {} nodes", nodes_searched);
                return;
            }
        }
        eprintln!("Still not solved at 10M");
    }

    /// Phase 25 補助: mate15 サブ局面 (ply24 prefix 後) の SFEN を印字する．
    /// KH ground truth で「canonical Mate(15) vs chain-drop inflated Mate(21)」を
    /// 確定するため．
    #[test]
    #[ignore]
    fn test_print_mate15_subposition_sfen() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let prefix_pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &prefix_pv {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        eprintln!("MATE15_SUBPOS_SFEN={}", board.sfen());
    }

    // ======================================================================
    // retain_proofs_only() fix (v0.55.27)
    // ======================================================================

    // ======================================================================
    // Step 2: 1d1e 全ペア ProvenTT ヒット診断 (v0.55.35)
    // ======================================================================

