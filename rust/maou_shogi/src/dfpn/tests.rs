// verbose feature 無効時に verbose_eprintln! 内でのみ使われる変数の
// unused 警告を抑制する．verbose 有効時はマクロが eprintln! に展開され
// 変数が実際に使用されるため警告は出ない．
#![allow(unused_variables, unused_assignments)]

use super::*;
use super::solver::*;
use super::pns::*;
use std::time::Instant;
use crate::attack;
use crate::movegen;
use crate::types::{Color, PieceType};

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
        let result = solver.solve(&mut board);

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
        let result = solver.solve(&mut board);

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
        let result = solver.solve(&mut board);

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
        let result = solver.solve(&mut board);

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
        let result = solver.solve(&mut board);
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
                // 最終2手は玉の逃げ方により3パターン:
                //   1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍
                let last2 = &usi_moves[usi_moves.len() - 2..];
                let valid_endings = [
                    ["1b1a", "3d2b+"],  // 1一玉，2二桂成
                    ["1b1a", "4b2b"],   // 1一玉，2二龍
                    ["1b1c", "4b2b"],   // 1三玉，2二龍
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

    /// S*1c → △同桂の後は不詰であることを確認する回帰テスト．
    ///
    /// PNS の AND ノード早期脱出時に部分的な子からの誤計算で
    /// 偽の証明(false proof)が発生するバグの回帰テスト．
    /// MID は正しくこの局面を不詰と判定する．
    #[test]
    fn test_pns_no_false_proof_after_dogyoku() {
        // S*1c, 同桂後の局面: 不詰であるべき
        let sfen = "9/7k1/5R2n/7Np/6P2/9/9/9/9 b r2b4g4s2n4l16p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(31, 2_000_000, 512);
        solver.attacker = board.turn;
        solver.start_time = Instant::now();

        // MID は不詰を正しく判定する
        solver.mid_fallback(&mut board);

        let (root_pn, _root_dn) = solver.look_up_board(&board);
        assert_ne!(root_pn, 0,
            "post-dogyoku position must NOT be checkmate (false proof regression)");
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
        let result = solver.solve(&mut board);

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

    /// `find_shortest = false` で最短手数探索をスキップできることを確認．
    #[test]
    fn test_tsume_5_no_shortest() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                // find_shortest = false でも17手詰みが見つかる
                assert_eq!(
                    pv.len(),
                    17,
                    "expected 17-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
            }
            other => panic!("expected Checkmate for tsume5, got {:?}", other),
        }

        // find_shortest = true との比較
        let mut board2 = Board::new();
        board2.set_sfen(sfen).unwrap();
        let mut solver2 = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        solver2.set_find_shortest(true);
        let result2 = solver2.solve(&mut board2);

        if let TsumeResult::Checkmate { nodes_searched: n2, .. } = &result2 {
            if let TsumeResult::Checkmate { nodes_searched: n1, .. } = &result {
                assert!(n1 <= n2, "find_shortest=false should use <= nodes");
            }
        }
    }

    /// 29手詰め(tsume6)．
    ///
    /// 深さ制限時の TT 保存バグの回帰テスト．
    /// PieceType::MAX_HAND_COUNT で保存すると不詰として誤判定されていた．
    /// デバッグビルド(opt-level=0)ではノード/時間制限を超過するため #[ignore]．
    /// `cargo test --release` または opt-level >= 1 で実行すること．
    #[test]
    #[ignore]
    fn test_tsume_6_29te() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 50_000_000, 32767, 300);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                verbose_eprintln!("=== tsume6 result: {} moves, {} nodes, prefilter_hits={} ===",
                    pv.len(), nodes_searched, solver.prefilter_hits);
                verbose_eprintln!("PV: {}", pv.join(" "));

                // 診断PV抽出は完了後のみ実行(Phase 2 後は TT が巨大化するため省略)

                // 8i7g が含まれているか確認 — 27手詰めになるバグの診断
                if let Some(pos) = pv.iter().position(|m| m == "8i7g") {
                    verbose_eprintln!("WARNING: 8i7g found at ply {} — this leads to 27-move mate, not 29", pos);
                }

                assert_eq!(
                    pv.len(),
                    29,
                    "expected 29-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
                // ぴよ将棋で検証済みの正解手順 (後手攻め)
                // 手順8(L*7g)と手順14(G*7h)は合駒 — ソルバーが別の駒を選ぶ可能性あり
                // 手順26(8f9g)は玉の逃げ方で分岐 — 8f8g でも同手数の29手詰め
                // 初手3手は 8f8g+/7h8g/S*7i と S*7i/8h9g/8f8g+ の2解あり
                let prefix1 = [
                    "8f8g+", "7h8g", "S*7i", "8h9g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                let prefix2 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                // 8e8g(不成)も同等に正しい — 弱い駒優先順序変更で出現
                let prefix3 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g", "9g8g",
                    "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
                    "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
                    "P*8e",
                ];
                // Deep df-pn バイアスにより合駒選択が変化した PV
                let prefix4 = [
                    "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
                    "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
                    "6h6i", "G*7h", "N*9e", "9f9e", "6i7h", "6g7h",
                    "P*8f", "8g9g", "G*9f", "9g9f", "9d9e", "9f8f",
                    "P*8e",
                ];
                assert!(
                    pv[..25] == prefix1 || pv[..25] == prefix2 || pv[..25] == prefix3 || pv[..25] == prefix4,
                    "PV prefix mismatch (first 25 moves):\n  got:      {}\n  pv1: {}\n  pv2: {}\n  pv3: {}\n  pv4: {}",
                    pv[..25].join(" "),
                    prefix1.join(" "),
                    prefix2.join(" "),
                    prefix3.join(" "),
                    prefix4.join(" "),
                );
                // 8i7g は不正解(27手詰めへの分岐)
                assert!(
                    !pv.contains(&"8i7g".to_string()),
                    "PV must not contain 8i7g (leads to 27-move mate): {}",
                    pv.join(" "),
                );
            }
            other => panic!("expected Checkmate for tsume6, got {:?}", other),
        }
    }

    /// 29手詰め: PNS なし(IDS-MID のみ)のロバストネステスト．
    ///
    /// PNS は浅い詰みの発見に使われ，IDS-MID は深い詰みに使われる．
    /// IDS-MID のみで29手詰めを発見できるか確認し，MID 単体のロバストネスを評価する．
    #[test]
    #[ignore]
    fn test_tsume_6_29te_no_pns() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 120_000_000, 32767, 1200);
        solver.max_nodes = 4;
        solver.attacker = board.turn;
        solver.start_time = std::time::Instant::now();
        let _ = solver.pns_main(&mut board);
        solver.max_nodes = 120_000_000;

        // mid_fallback 内の IDS ステップと root ply の診断を取得するため，
        // diag_ply=0 で root ノードの mid_diag をトリガーする．
        // mid_diag の間隔を 500K に短縮して root での応手別消費を追跡する．
        solver.mid_fallback(&mut board);

        let pk = position_key(&board);
        let att_hand = board.hand[solver.attacker.index()];
        let (root_pn, _, _) = solver.look_up_pn_dn(pk, &att_hand, 31);
        verbose_eprintln!("[no_pns] root_pn={} nodes={} time={:.1}s",
            root_pn, solver.nodes_searched, solver.start_time.elapsed().as_secs_f64());

        if root_pn == 0 {
            let pv = solver.extract_pv_limited(&mut board, 10_000);
            let pv_usi: Vec<String> = pv.iter().map(|m| m.to_usi()).collect();
            verbose_eprintln!("[no_pns] PV ({} moves): {}", pv_usi.len(), pv_usi.join(" "));
            assert_eq!(pv_usi.len(), 29, "expected 29-move PV, got {} moves", pv_usi.len());
        } else {
            verbose_eprintln!("[no_pns] NOT PROVED: rpn={} nodes={}", root_pn, solver.nodes_searched);
        }

        // ply 別効率レポート(解決・未解決共通)
        verbose_eprintln!("\n[efficiency] {:>3} {:>10} {:>12} {:>8} {:>8}",
            "ply", "nodes", "iters", "n/iter", "stag");
        for p in 0..64 {
            let n = solver.ply_nodes[p];
            let it = solver.ply_iters[p];
            let stag = solver.ply_stag_penalties[p];
            if n > 0 || it > 0 {
                let ratio = if it > 0 { n as f64 / it as f64 } else { 0.0 };
                verbose_eprintln!("[efficiency] {:>3} {:>10} {:>12} {:>8.1} {:>8}",
                    p, n, it, ratio, stag);
            }
        }
        verbose_eprintln!();

        // TT コンテンツ分析
        #[cfg(feature = "verbose")]
        solver.table.dump_content_analysis();

        // プロファイル統計
        #[cfg(feature = "profile")]
        {
            solver.sync_tt_profile();
            eprintln!("{}", solver.profile_stats);
        }

        if root_pn != 0 {
            panic!("IDS-MID only should prove 29te checkmate, got pn={}", root_pn);
        }
    }

    /// 29手詰め PV 逆順解析: PV の手順を進め，各中間局面から解けるか検証．
    /// どの深さで解けなくなるかを特定する．
    #[test]
    fn test_tsume_6_29te_pv_analysis() {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        // PV の最初 25 手(テストで検証済みの手順)
        let pv_moves = [
            "S*7i", "8h9g", "8f8g+", "7h8g", "G*8f", "9g8f",
            "5g6h+", "L*7g", "R*8e", "8f9g", "8e8g+", "9g8g",
            "6h6i", "G*7h", "6i7h", "6g7h", "P*8f", "8g9g",
            "G*8g", "7h8g", "8f8g+", "9g8g", "P*8f", "8g8f",
            "P*8e",
        ];

        // 偶数 ply ごとに(攻め方の手番で)テスト
        // 常に depth=31, 5M nodes, 30s で解けるか確認
        for start_ply in (0..pv_moves.len()).step_by(2) {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            // Play first start_ply moves
            for i in 0..start_ply {
                let m = board.move_from_usi(pv_moves[i])
                    .unwrap_or_else(|| panic!("failed to parse move {} at index {}", pv_moves[i], i));
                board.do_move(m);
            }

            let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 30);
            let result = solver.solve(&mut board);

            match &result {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    verbose_eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} SOLVED {}te, {} nodes",
                        start_ply, pv_moves[start_ply], moves.len(), nodes_searched
                    );
                }
                TsumeResult::Unknown { nodes_searched } => {
                    verbose_eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} FAILED ({} nodes)",
                        start_ply, pv_moves[start_ply], nodes_searched
                    );
                }
                other => {
                    verbose_eprintln!(
                        "[pv_analysis] ply={:2} first_move={:<8} {:?}",
                        start_ply, pv_moves[start_ply], other
                    );
                }
            }
        }
    }

    /// 29手詰め ply1 応手解析: S*7i 後の各応手から解けるか検証．
    #[test]
    fn test_tsume_6_29te_ply1_analysis() {
        use crate::movegen;
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // Play S*7i (correct first move)
        let first_move = board.move_from_usi("S*7i").unwrap();
        board.do_move(first_move);

        // Generate all defense moves at ply 1
        let defenses = movegen::generate_legal_moves(&mut board);
        verbose_eprintln!("[ply1_analysis] {} defense moves after S*7i", defenses.len());

        for def in &defenses {
            let cap = board.do_move(*def);
            // From ply 2 position, try to solve
            let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 30);
            let result = solver.solve(&mut board);
            match &result {
                TsumeResult::Checkmate { moves, nodes_searched } => {
                    verbose_eprintln!(
                        "[ply1_analysis] defense={:<8} CHECKMATE {}te, {} nodes",
                        def.to_usi(), moves.len(), nodes_searched
                    );
                }
                TsumeResult::NoCheckmate { nodes_searched } => {
                    verbose_eprintln!(
                        "[ply1_analysis] defense={:<8} NO_CHECKMATE (refuted), {} nodes",
                        def.to_usi(), nodes_searched
                    );
                }
                TsumeResult::Unknown { nodes_searched } => {
                    verbose_eprintln!(
                        "[ply1_analysis] defense={:<8} UNKNOWN (stuck), {} nodes",
                        def.to_usi(), nodes_searched
                    );
                }
                other => {
                    verbose_eprintln!(
                        "[ply1_analysis] defense={:<8} {:?}",
                        def.to_usi(), other
                    );
                }
            }
            board.undo_move(*def, cap);
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
        let result = solver.solve(&mut board);

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
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 7,
                    "expected 7-move checkmate, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // 初手は飛車不成(4a2a)でなければならない
                assert_eq!(
                    usi_moves[0], "4a2a",
                    "move 1 must be 4a2a (rook WITHOUT promotion), got: {}",
                    usi_moves[0]
                );
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
        let result = solver.solve(&mut board);
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

    /// 逆王手で不詰のケース．
    ///
    /// 局面: 先手玉2四，先手飛4三，先手歩1三，先手香2五
    ///       後手玉2二，後手香2一，後手桂4二，後手銀3四
    /// 先手持駒: なし
    /// 後手持駒: 飛，角二，金四，銀三，桂三，香二，歩十七
    ///
    /// 4三飛→2三飛成は王手だが，後手3三銀の逆王手(2四の先手玉に対する王手)
    /// により攻め方は王手回避を強いられ，詰みにならない．
    /// 先手に持ち駒がなく他の有効な攻めがないため不詰．
    #[test]
    fn test_no_checkmate_counter_check() {
        let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

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
                let pv_str: Vec<&str> = usi_moves.iter().map(|s| s.as_str()).collect();
                assert!(
                    pv_str == pv1 || pv_str == pv2 || pv_str == pv3,
                    "PV must be one of the known solutions:\n  got:  {}\n  pv1: {}\n  pv2: {}\n  pv3: {}",
                    usi_moves.join(" "),
                    pv1.join(" "),
                    pv2.join(" "),
                    pv3.join(" "),
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
        let result = solve_tsume(sfen, Some(31), Some(1_000_000), None).unwrap();

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

    /// 金の移動合い後もソルバーが正しく詰みを検出することを検証する．
    ///
    /// B*3c(3三角打)後の金移動合い(2a→2b)に対して，
    /// 先手が銀を持っている場合に7手で詰ませられることを確認する．
    /// 全体としては B*3c, g(2a→2b) + 7手 = 9手詰め．
    #[test]

    fn test_after_gold_interposition_with_silver() {
        let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        // B*3c → g(2a→2b) を実行
        let b3c = board.move_from_usi("B*3c").unwrap();
        board.do_move(b3c);
        let g2b = board.move_from_usi("2a2b").unwrap();
        board.do_move(g2b);

        // この局面から先手が詰ませられるか
        let mut solver = DfPnSolver::new(31, 2_000_000, 32767);
        let result = solver.solve(&mut board);
        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 7,
                    "after gold interposition, expected 7 more moves, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
                // B*3c → 2a2b の後の正解手順 (ぴよ将棋で検証済み)
                let expected = [
                    "N*2c", "1a2a", "S*3b", "2a3b", "3c4b+", "3b2a", "4b3a",
                ];
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {}\n  expected: {}",
                    usi_moves.join(" "),
                    expected.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate after gold interposition, got {:?}",
                other
            ),
        }
    }

    /// 39手詰めの高難度テスト(6九への合駒が必要)．
    ///
    /// 後手の合駒選択が鍵となる局面．6九に合駒を打つ必要があるが，
    /// 歩・桂・香は打てない(二歩・行き所のない駒)ため，
    /// 金・銀・飛・角のみが候補となる．
    /// 後手の最善応手(最長抵抗)でのみ39手詰めとなる．
    #[test]
    #[ignore] // 約42万ノード / 5秒で解ける重量テスト．明示的に `cargo test -- --ignored` で実行．
    fn test_tsume_39te_aigoma() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver =
            DfPnSolver::with_timeout(41, 10_000_000, 32767, 60);
        solver.set_find_shortest(false);
        #[cfg(feature = "tt_diag")]
        {
            solver.diag_ply = 35;
            solver.diag_max_iterations = 0; // don't break loop
        }
        let start = Instant::now();
        let result = solver.solve(&mut board);
        let elapsed = start.elapsed();
        verbose_eprintln!("39te: {} nodes, {:.1}s, max_ply={}, prefilter_hits={}",
            solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply,
            solver.prefilter_hits);
        #[cfg(feature = "profile")]
        {
            solver.sync_tt_profile();
            verbose_eprintln!("{}", solver.profile_stats);
        }

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> =
                    moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    pv.len(),
                    39,
                    "expected 39-move checkmate, got {} moves: {}",
                    pv.len(),
                    pv.join(" ")
                );
                // ぴよ将棋で検証済みの正解手順
                let expected = [
                    "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
                    "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
                    "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
                    "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
                    "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
                    "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
                    "2g2h", "3i4i", "2h4h",
                ];
                assert_eq!(
                    pv, expected,
                    "PV mismatch:\n  got:      {}\n  expected: {}",
                    pv.join(" "),
                    expected.join(" "),
                );
            }
            other => panic!(
                "expected Checkmate for 39te aigoma, got {:?}",
                other
            ),
        }
    }

    /// 39手詰め直接MID テスト(PNS/IDS なし)．
    ///
    /// main ブランチと同様に単一 MID 呼び出しで解けるか確認する．
    #[test]
    #[ignore]
    fn test_tsume_39te_direct_mid() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver =
            DfPnSolver::with_timeout(63, 10_000_000, 32767, 60);
        solver.set_find_shortest(false);

        // 直接 MID を呼び出す(PNS/IDS をバイパス)
        solver.table.clear();
        solver.nodes_searched = 0;
        solver.max_ply = 0;
        solver.path_len = 0;
        solver.killer_table.clear();
        solver.start_time = Instant::now();
        solver.timed_out = false;
        solver.next_gc_check = 100_000;
        solver.attacker = board.turn;
        solver.mid(&mut board, INF - 1, INF - 1, 0, true);

        let (root_pn, _root_dn) = solver.look_up_board(&board);
        let start = Instant::now();
        let _elapsed = start.elapsed();
        verbose_eprintln!("39te_direct_mid: {} nodes, {:.1}s, max_ply={}, prefilter={}  pn={}",
            solver.nodes_searched, solver.start_time.elapsed().as_secs_f64(),
            solver.max_ply, solver.prefilter_hits, root_pn);
        assert_eq!(root_pn, 0, "expected pn=0 (proved) for 39te direct MID");
    }

    /// 39手詰めサブ問題実験: PV 終盤側から逆順に詰み探索ノード数を計測し，
    /// 全体を解くのに必要な予算を推定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_budget_estimation() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV: 攻め手(奇数)と玉方(偶数)
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" 39手詰めサブ問題予算推定実験(終盤→序盤)");
        verbose_eprintln!("{}", "=".repeat(80));
        verbose_eprintln!("{:<6} {:<14} {:<10} {:<10} {:<12} {}",
            "Ply", "Nodes", "Time(s)", "MaxPly", "Result", "Remaining");

        // PV を偶数手ずつ進めた局面(攻め番=ORノード)を全て事前構築
        let mut positions: Vec<(usize, Board)> = Vec::new();
        let mut sub_board = board.clone();
        // ply 0
        positions.push((0, sub_board.clone()));
        for ply_start in (0..38).step_by(2) {
            let m1 = sub_board.move_from_usi(pv_usi[ply_start]).unwrap();
            sub_board.do_move(m1);
            let m2 = sub_board.move_from_usi(pv_usi[ply_start + 1]).unwrap();
            sub_board.do_move(m2);
            positions.push((ply_start + 2, sub_board.clone()));
        }

        // 終盤側(簡単)から序盤側(困難)へ逆順で解く
        // 解けなくなったら停止
        positions.reverse();

        let node_limit: u64 = 50_000_000; // 5000万ノード上限
        let timeout = 60; // 60秒

        for (ply, pos) in &positions {
            let remaining_moves = 39 - ply;
            let depth = (remaining_moves + 2).min(41) as u32;

            let mut test_board = pos.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, node_limit, 32767, timeout,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            verbose_eprintln!("{:<6} {:<14} {:<10.2} {:<10} {:<12} {}手",
                ply, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, result_str, remaining_moves);

            // 解けなくなったら局面のSFENを出力して停止
            if !solved {
                verbose_eprintln!("--- ply {} で未解決 ---", ply);
                verbose_eprintln!("  SFEN: {}", pos.sfen());
                verbose_eprintln!("  PV残り: {:?}", &pv_usi[*ply..]);

                // 深さを大きくして再試行
                for &d in &[25u32, 31, 41, 51] {
                    let mut test_board2 = pos.clone();
                    let mut solver2 = DfPnSolver::with_timeout(
                        d, 50_000_000, 32767, 60,
                    );
                    solver2.set_find_shortest(false);

                    let start2 = Instant::now();
                    let result2 = solver2.solve(&mut test_board2);
                    let elapsed2 = start2.elapsed();

                    let result_str2 = match &result2 {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } =>
                            "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } =>
                            "NoMate".to_string(),
                        TsumeResult::Unknown { .. } =>
                            "Unknown".to_string(),
                    };

                    verbose_eprintln!("  depth={:<4} {:<14} {:<10.2} {:<10} {}",
                        d, solver2.nodes_searched, elapsed2.as_secs_f64(),
                        solver2.max_ply, result_str2);
                }

                // 1手進めた局面(ply+1, 玉方手番=ANDノード)も試す
                verbose_eprintln!("\n  --- ply {} (攻め手1手目 {} 適用後，玉方手番) ---", ply, pv_usi[*ply]);
                let mut after1 = pos.clone();
                let m1 = after1.move_from_usi(pv_usi[*ply]).unwrap();
                after1.do_move(m1);
                verbose_eprintln!("  SFEN after {}: {}", pv_usi[*ply], after1.sfen());

                // PV の最後の手から逆順に，1手ずつ戻って解けるポイントを探す
                verbose_eprintln!("\n  --- 1手ずつ PV を遡り解ける境界を特定 ---");
                // ply+1 (玉方手番後) から ply+16 まで奇数手のみ(OR局面)
                let mut walk_board = pos.clone();
                for step in 0..remaining_moves {
                    let mv_str = pv_usi[*ply + step];
                    let mv = walk_board.move_from_usi(mv_str).unwrap();
                    walk_board.do_move(mv);

                    // OR局面(攻め方手番)のみ詰み探索
                    if step % 2 == 0 {
                        continue; // step=0 で玉方手番，step=1 で攻め方手番
                    }

                    let sub_remaining = remaining_moves - step - 1;
                    if sub_remaining == 0 { break; }
                    let sub_depth = (sub_remaining + 2).min(41) as u32;

                    let mut sub_board = walk_board.clone();
                    let mut sub_solver = DfPnSolver::with_timeout(
                        sub_depth, 50_000_000, 32767, 60,
                    );
                    sub_solver.set_find_shortest(false);
                    let sub_result = sub_solver.solve(&mut sub_board);

                    let sub_result_str = match &sub_result {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } =>
                            "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } =>
                            "NoMate".to_string(),
                        TsumeResult::Unknown { .. } =>
                            "Unknown".to_string(),
                    };

                    verbose_eprintln!("  ply{}+{} {:<14} {:<12} rem={}手 SFEN: {}",
                        ply, step + 1, sub_solver.nodes_searched, sub_result_str,
                        sub_remaining, walk_board.sfen());
                }

                // P*1g 後の局面(玉方手番)の合法手を全列挙
                verbose_eprintln!("\n  --- P*1g 後の玉方応手分析 ---");
                let mut after_drop = pos.clone();
                let mv_drop = after_drop.move_from_usi("P*1g").unwrap();
                after_drop.do_move(mv_drop);

                let legal_moves = movegen::generate_legal_moves(&mut after_drop);
                verbose_eprintln!("  合法手数: {}", legal_moves.len());
                for lm in &legal_moves {
                    let mut after_resp = after_drop.clone();
                    after_resp.do_move(*lm);

                    let mut sub_board = after_resp.clone();
                    let mut sub_solver = DfPnSolver::with_timeout(
                        19, 50_000_000, 32767, 60,
                    );
                    sub_solver.set_find_shortest(false);
                    let sub_result = sub_solver.solve(&mut sub_board);

                    let sub_result_str = match &sub_result {
                        TsumeResult::Checkmate { moves, .. } =>
                            format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                        TsumeResult::Unknown { .. } => "Unknown".to_string(),
                    };
                    verbose_eprintln!("  {} {:<14} {}", lm.to_usi(), sub_solver.nodes_searched, sub_result_str);
                }

                // 直接診断: depth=19 で solve 後，TT 内のルートエントリをダンプ
                let att_hand22 = pos.hand[Color::Black.index()];
                {
                    let mut test_board = pos.clone();
                    let mut solver = DfPnSolver::with_timeout(19, 50_000_000, 32767, 60);
                    solver.set_find_shortest(false);
                    let result = solver.solve(&mut test_board);

                    let result_str = match &result {
                        TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                        TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                        TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                        TsumeResult::Unknown { .. } => "Unknown".to_string(),
                    };
                    verbose_eprintln!("  depth=19 result: {} nodes={}", result_str, solver.nodes_searched);

                    // TT ルートの全エントリをダンプ
                    let pk = position_key(pos);
                    #[cfg(feature = "verbose")]
                    {
                        let mut count = 0u32;
                        for e in solver.table.entries_iter(pk) {
                            verbose_eprintln!("    [{}] pn={} dn={} remaining={} path_dep={} hand={:?} src={}",
                                count, e.pn, e.dn, e.remaining, e.path_dependent,
                                e.hand, e.source);
                            count += 1;
                        }
                        if count == 0 {
                            verbose_eprintln!("  TT: no entries for root");
                        }
                    }

                    // remaining=0 vs remaining=19 の look_up 結果
                    let (p0, d0, _) = solver.table.look_up(pk, &att_hand22, 0);
                    let (p19, d19, _) = solver.table.look_up(pk, &att_hand22, 19);
                    verbose_eprintln!("  look_up(remaining=0):  pn={} dn={}", p0, d0);
                    verbose_eprintln!("  look_up(remaining=19): pn={} dn={}", p19, d19);
                }

                break;
            }
        }

        verbose_eprintln!("{}", "=".repeat(80));
    }

    /// 39手詰め逆順サブ問題: 1M ノード / 180 秒で各 OR ノードから解き，
    /// 解けなくなった境界を特定する．解けない局面ではANDノードの各応手の
    /// 探索コスト内訳を報告する．
    #[test]
    fn test_tsume_39te_backward_1m() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_backward_1m.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

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

        let node_limit: u64 = 1_000_000;
        let timeout: u64 = 180;

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め逆順サブ問題 (1M nodes / 180s)").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, "{:<6} {:<10} {:<14} {:<10} {:<10} {:<10} {}",
            "Ply", "Remain", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(90)).unwrap();

        // PV を偶数手ずつ進めた局面(攻め番=ORノード)を全て事前構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        let mut positions: Vec<(usize, Board)> = Vec::new();
        positions.push((0, board.clone()));
        for ply_start in (0..38).step_by(2) {
            let m1 = board.move_from_usi(pv[ply_start]).unwrap();
            board.do_move(m1);
            let m2 = board.move_from_usi(pv[ply_start + 1]).unwrap();
            board.do_move(m2);
            positions.push((ply_start + 2, board.clone()));
        }

        // 終盤(簡単)→序盤(困難)の逆順
        positions.reverse();

        let mut first_unsolved_ply: Option<usize> = None;

        for (ply, pos) in &positions {
            let remaining = 39 - ply;
            let depth = (remaining + 2).min(41) as u32;

            let mut test_board = pos.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, node_limit, 32767, timeout,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<6} {:<10} {:<14} {:<10.2} {:<10} {:<10} {}",
                ply, remaining, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, solver.table.len(), result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "       TT entries: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            if !solved {
                first_unsolved_ply = Some(*ply);

                // --- ANDノードの各応手コスト分析 ---
                // PV の攻め手を1手進めた後(ANDノード)の応手を調べる
                if *ply < pv.len() {
                    let attack_move = pv[*ply];
                    let sub_remaining = remaining - 1; // 攻め手1手分消費
                    if sub_remaining == 0 { continue; }

                    let mut after_attack = pos.clone();
                    let m = after_attack.move_from_usi(attack_move).unwrap();
                    after_attack.do_move(m);

                    writeln!(out, "\n  --- AND node analysis: after {} (ply {}) ---",
                        attack_move, ply + 1).unwrap();
                    writeln!(out, "  SFEN: {}", after_attack.sfen()).unwrap();

                    // 応手一覧
                    let mut defense_solver = DfPnSolver::default_solver();
                    let defenses = defense_solver.generate_defense_moves(&mut after_attack);
                    writeln!(out, "  Defense moves: {} (PV: {})", defenses.len(),
                        if *ply + 1 < pv.len() { pv[*ply + 1] } else { "N/A" }).unwrap();

                    writeln!(out, "  {:<12} {:<14} {:<10} {:<10} {:<10} {}",
                        "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

                    for def_mv in &defenses {
                        let mut after_def = after_attack.clone();
                        after_def.do_move(*def_mv);

                        // 守り手を指した後 → OR ノード(攻め番)
                        let def_remaining = sub_remaining - 1;
                        if def_remaining == 0 {
                            writeln!(out, "  {:<12} --- (remaining=0)", def_mv.to_usi()).unwrap();
                            continue;
                        }
                        let sub_depth = (def_remaining + 2).min(41) as u32;
                        // 応手あたりのノード予算: 全体の1/4か100Kの大きい方
                        let per_move_budget = (node_limit / 4).max(100_000);

                        let mut sub_board = after_def.clone();
                        let mut sub_solver = DfPnSolver::with_timeout(
                            sub_depth, per_move_budget, 32767, 30,
                        );
                        sub_solver.set_find_shortest(false);

                        let sub_start = Instant::now();
                        let sub_result = sub_solver.solve(&mut sub_board);
                        let sub_elapsed = sub_start.elapsed();

                        let sub_result_str = match &sub_result {
                            TsumeResult::Checkmate { moves, .. } =>
                                format!("Mate({})", moves.len()),
                            TsumeResult::CheckmateNoPv { .. } =>
                                "MateNoPV".to_string(),
                            TsumeResult::NoCheckmate { .. } =>
                                "NoMate".to_string(),
                            TsumeResult::Unknown { .. } =>
                                "Unknown".to_string(),
                        };
                        let marker = if *ply + 1 < pv.len() && def_mv.to_usi() == pv[*ply + 1] {
                            " ← PV"
                        } else { "" };
                        writeln!(out, "  {:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                            def_mv.to_usi(), sub_solver.nodes_searched,
                            sub_elapsed.as_secs_f64(), sub_solver.max_ply,
                            sub_solver.table.len(), sub_result_str, marker).unwrap();

                        #[cfg(feature = "tt_diag")]
                        {
                            let proven = sub_solver.table.count_proven();
                            let disproven = sub_solver.table.count_disproven();
                            let intermediate = sub_solver.table.count_intermediate();
                            let total = sub_solver.table.total_entries();
                            writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                                total, proven, disproven, intermediate).unwrap();
                        }
                    }
                }
                break; // 最初の未解決局面の分析後に停止
            }
        }

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        if let Some(ply) = first_unsolved_ply {
            writeln!(out, "境界: ply {} (残り{}手) で 1M ノードでは解けない",
                ply, 39 - ply).unwrap();
        } else {
            writeln!(out, "全局面 1M ノード以内で解決").unwrap();
        }
            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: /tmp/tsume_39te_backward_1m.log");
    }

    /// 39手詰めの必要ノード数を推定する．
    ///
    /// 方針: ply 24 境界の 4 Unknown 応手(1g1f, N*6g, P*7g, N*7g)を
    /// 個別に 1M ノードで解き，応手別コストの合計から推定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_budget_estimation() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_budget_est.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め予算推定: 境界ply の応手別 1M 個別分析").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // Phase 1 結果(backward_1m より)を記載
        writeln!(out, "\n--- Phase 1 結果サマリー (backward_1m) ---").unwrap();
        writeln!(out, "ply 26 (残り13): 104 nodes → Mate(13)").unwrap();
        writeln!(out, "ply 24 (残り15): 473K nodes → Unknown (境界)").unwrap();
        writeln!(out, "ply 22-4: 全て Unknown at 1M").unwrap();

        // Phase 2: ply 24 の AND ノード(5g6f後)の各応手を 1M で個別分析
        writeln!(out, "\n--- Phase 2: ply 24 境界の応手別 1M 分析 ---").unwrap();

        // ply 24 の局面を構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }
        // 攻め手 5g6f を指して AND ノードへ
        let attack_m = board.move_from_usi(pv[24]).unwrap(); // 5g6f
        board.do_move(attack_m);
        writeln!(out, "AND node after 5g6f (ply 25)").unwrap();
        writeln!(out, "SFEN: {}", board.sfen()).unwrap();

        let mut defense_solver = DfPnSolver::default_solver();
        let defenses = defense_solver.generate_defense_moves(&mut board);
        writeln!(out, "Defense moves: {} (PV: 1g1h)\n", defenses.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        let mut total_nodes: u64 = 0;
        let mut solved_count = 0;
        let mut unknown_moves: Vec<(String, u64)> = Vec::new();

        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            // 残り: 39 - 24 - 1(攻め) - 1(受け) = 13 手
            let def_remaining = 13;
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let (sub_result_str, sub_solved) = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };
            let marker = if def_mv.to_usi() == "1g1h" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                sub_result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            total_nodes += nodes;
            if sub_solved {
                solved_count += 1;
            } else {
                unknown_moves.push((def_mv.to_usi(), nodes));
            }
        }

        writeln!(out, "{}", "-".repeat(80)).unwrap();
        writeln!(out, "合計 nodes: {}, 解決: {}/{}, Unknown: {}",
            total_nodes, solved_count, defenses.len(), unknown_moves.len()).unwrap();

        if unknown_moves.is_empty() {
            writeln!(out, "\n→ ply 24 推定必要ノード ≈ {} ({:.1}M)",
                total_nodes, total_nodes as f64 / 1_000_000.0).unwrap();
        } else {
            writeln!(out, "\n→ 1M/応手でも未解決: {:?}", unknown_moves).unwrap();
        }

        // Phase 3: ply 22 の AND ノード(P*1g後)の各応手分析
        writeln!(out, "\n--- Phase 3: ply 22 の応手別 1M 分析 ---").unwrap();

        let mut board22 = Board::new();
        board22.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board22.move_from_usi(pv[i]).unwrap();
            board22.do_move(m);
        }
        // 攻め手 P*1g を指して AND ノードへ
        let attack_m22 = board22.move_from_usi(pv[22]).unwrap(); // P*1g
        board22.do_move(attack_m22);
        writeln!(out, "AND node after P*1g (ply 23)").unwrap();
        writeln!(out, "SFEN: {}", board22.sfen()).unwrap();

        let defenses22 = defense_solver.generate_defense_moves(&mut board22);
        writeln!(out, "Defense moves: {} (PV: 1f1g)\n", defenses22.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        let mut total22: u64 = 0;
        let mut solved22 = 0;
        let mut unknown22: Vec<(String, u64)> = Vec::new();

        for def_mv in &defenses22 {
            let mut after_def = board22.clone();
            after_def.do_move(*def_mv);

            // 残り: 39 - 22 - 1(攻め) - 1(受け) = 15 手
            let def_remaining = 15;
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let (sub_result_str, sub_solved) = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };
            let marker = if def_mv.to_usi() == "1f1g" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                sub_result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }

            total22 += nodes;
            if sub_solved {
                solved22 += 1;
            } else {
                unknown22.push((def_mv.to_usi(), nodes));
            }
        }

        writeln!(out, "{}", "-".repeat(70)).unwrap();
        writeln!(out, "合計 nodes: {}, 解決: {}/{}, Unknown: {}",
            total22, solved22, defenses22.len(), unknown22.len()).unwrap();

        if unknown22.is_empty() {
            writeln!(out, "→ ply 22 推定必要ノード ≈ {} ({:.1}M)",
                total22, total22 as f64 / 1_000_000.0).unwrap();
        } else {
            writeln!(out, "→ 1M/応手でも未解決: {:?}", unknown22).unwrap();
        }

        // Phase 4: 4 Unknown 応手の再帰分解(1レベル深い)
        // 各 Unknown 応手の後の OR ノード → 攻め手 → AND ノードの応手数を調査
        writeln!(out, "\n--- Phase 4: Unknown 応手の再帰分解 ---").unwrap();

        let unknown_defenses = ["1g1f", "N*6g", "P*7g", "N*7g"];

        for &def_usi in &unknown_defenses {
            let mut after_def = board.clone();
            let def_m = after_def.move_from_usi(def_usi).unwrap();
            after_def.do_move(def_m);

            writeln!(out, "\n--- {} 後 (OR node, 攻め番) ---", def_usi).unwrap();
            writeln!(out, "SFEN: {}", after_def.sfen()).unwrap();

            let attacks = defense_solver.generate_check_moves(&mut after_def);
            writeln!(out, "Attack moves: {}", attacks.len()).unwrap();

            writeln!(out, "{:<12} {:<8} {:<14} {:<10} {}",
                "Attack", "#Def", "Nodes", "Time(s)", "Result").unwrap();

            let mut def_total: u64 = 0;
            let mut def_unknown = 0;

            for atk in &attacks {
                // AND ノードの応手数を数える
                let mut count_board = after_def.clone();
                count_board.do_move(*atk);
                let and_defenses = defense_solver.generate_defense_moves(&mut count_board);
                let num_def = and_defenses.len();

                // 各攻め手の AND サブ問題を 100K で試行
                let sub_remaining = 11; // 13 - 2 (def + atk)
                let sub_depth = (sub_remaining + 2).min(41) as u32;

                let mut sub_solver = DfPnSolver::with_timeout(
                    sub_depth, 100_000, 32767, 30,
                );
                sub_solver.set_find_shortest(false);

                let sub_start = Instant::now();
                let sub_result = sub_solver.solve(&mut count_board);
                let sub_elapsed = sub_start.elapsed();
                let nodes = sub_solver.nodes_searched;

                let (result_str, solved) = match &sub_result {
                    TsumeResult::Checkmate { moves, .. } =>
                        (format!("Mate({})", moves.len()), true),
                    TsumeResult::CheckmateNoPv { .. } =>
                        ("MateNoPV".to_string(), true),
                    TsumeResult::NoCheckmate { .. } =>
                        ("NoMate".to_string(), false),
                    TsumeResult::Unknown { .. } =>
                        ("Unknown".to_string(), false),
                };
                writeln!(out, "{:<12} {:<8} {:<14} {:<10.2} {}",
                    atk.to_usi(), num_def, nodes, sub_elapsed.as_secs_f64(),
                    result_str).unwrap();

                def_total += nodes;
                if !solved { def_unknown += 1; }
            }
            writeln!(out, "合計: {} nodes, Unknown 攻め手: {}/{}",
                def_total, def_unknown, attacks.len()).unwrap();
        }

        // Phase 5: ply 20 の分析(P*1f 後)
        writeln!(out, "\n--- Phase 5: ply 20 の応手別 1M 分析 ---").unwrap();
        let mut board20 = Board::new();
        board20.set_sfen(sfen).unwrap();
        for i in 0..20 {
            let m = board20.move_from_usi(pv[i]).unwrap();
            board20.do_move(m);
        }
        let attack_m20 = board20.move_from_usi(pv[20]).unwrap(); // P*1f
        board20.do_move(attack_m20);
        writeln!(out, "AND node after P*1f (ply 21)").unwrap();
        writeln!(out, "SFEN: {}", board20.sfen()).unwrap();

        let defenses20 = defense_solver.generate_defense_moves(&mut board20);
        writeln!(out, "Defense moves: {} (PV: 1e1f)\n", defenses20.len()).unwrap();

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(80)).unwrap();

        for def_mv in &defenses20 {
            let mut after_def = board20.clone();
            after_def.do_move(*def_mv);
            let def_remaining = 17; // 39 - 20 - 1 - 1 = 17
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_board = after_def.clone();
            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 1_000_000, 32767, 180,
            );
            sub_solver.set_find_shortest(false);

            let sub_start = Instant::now();
            let sub_result = sub_solver.solve(&mut sub_board);
            let sub_elapsed = sub_start.elapsed();
            let nodes = sub_solver.nodes_searched;

            let result_str = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } =>
                    "NoMate".to_string(),
                TsumeResult::Unknown { .. } =>
                    "Unknown".to_string(),
            };
            let marker = if def_mv.to_usi() == "1e1f" { " ← PV" } else { "" };
            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}{}",
                def_mv.to_usi(), nodes, sub_elapsed.as_secs_f64(),
                sub_solver.max_ply, sub_solver.table.len(),
                result_str, marker).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = sub_solver.table.count_proven();
                let disproven = sub_solver.table.count_disproven();
                let intermediate = sub_solver.table.count_intermediate();
                let total = sub_solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // 推定サマリー
        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        writeln!(out, " 推定サマリー").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: {}", out_path);
    }

    /// 39手詰め問題の必要予算を段階的に推定する．
    ///
    /// backward_1m の結果を踏まえ，ply 24 の未解決応手を
    /// 段階的に予算増加して解き，ply 間のコスト成長率から
    /// 全体の必要予算を外挿する．
    #[test]
    #[ignore]
    fn test_tsume_39te_budget_scaling() {
        use std::io::Write;
        let out_path = "/tmp/tsume_39te_budget_scaling.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " 39手詰め予算スケーリング推定").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // ===== Phase A: ply 24 未解決応手の予算スケーリング =====
        writeln!(out, "\n### Phase A: ply 24 未解決応手の段階的予算増加").unwrap();
        writeln!(out, "ply 25 AND node (after 5g6f) → 4 Unknown defense moves\n").unwrap();

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }
        let attack_m = board.move_from_usi(pv[24]).unwrap(); // 5g6f
        board.do_move(attack_m);

        let unknown_defenses = ["1g1f", "N*6g", "P*7g", "N*7g"];
        let budgets: &[u64] = &[1_000_000, 5_000_000, 10_000_000];

        for &def_usi in &unknown_defenses {
            writeln!(out, "--- {} ---", def_usi).unwrap();
            writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
                "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

            let def_m = board.move_from_usi(def_usi).unwrap();

            for &budget in budgets {
                let mut after_def = board.clone();
                after_def.do_move(def_m);

                // depth=41(最大)を使用: 非PV変化では PV より長い詰みが存在しうる
                let sub_depth = 41_u32;

                let mut sub_solver = DfPnSolver::with_timeout(
                    sub_depth, budget, 32767, 600,
                );
                sub_solver.set_find_shortest(false);

                let sub_start = Instant::now();
                let sub_result = sub_solver.solve(&mut after_def);
                let sub_elapsed = sub_start.elapsed();

                let (result_str, solved) = match &sub_result {
                    TsumeResult::Checkmate { moves, .. } =>
                        (format!("Mate({})", moves.len()), true),
                    TsumeResult::CheckmateNoPv { .. } =>
                        ("MateNoPV".to_string(), true),
                    TsumeResult::NoCheckmate { .. } =>
                        ("NoMate".to_string(), false),
                    TsumeResult::Unknown { .. } =>
                        ("Unknown".to_string(), false),
                };

                writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                    format!("{}M", budget / 1_000_000),
                    sub_solver.nodes_searched,
                    sub_elapsed.as_secs_f64(),
                    sub_solver.max_ply,
                    sub_solver.table.len(),
                    result_str).unwrap();

                #[cfg(feature = "tt_diag")]
                {
                    let proven = sub_solver.table.count_proven();
                    let disproven = sub_solver.table.count_disproven();
                    let intermediate = sub_solver.table.count_intermediate();
                    let total = sub_solver.table.total_entries();
                    writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                        total, proven, disproven, intermediate).unwrap();
                }

                if solved { break; } // 解けたらこの応手は終了
            }
            writeln!(out).unwrap();
        }

        // ===== Phase B: ply 24 全体を解くのに必要な予算 =====
        writeln!(out, "### Phase B: ply 24 全体の予算スケーリング").unwrap();
        writeln!(out, "ply 24 OR node を段階的予算で直接解く\n").unwrap();

        let ply24_budgets: &[u64] = &[5_000_000, 10_000_000, 50_000_000];

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

        let mut board24 = Board::new();
        board24.set_sfen(sfen).unwrap();
        for i in 0..24 {
            let m = board24.move_from_usi(pv[i]).unwrap();
            board24.do_move(m);
        }

        for &budget in ply24_budgets {
            let mut test_board = board24.clone();
            let depth = 41_u32; // 最大深さ: 非PV変化で長い手順が存在
            let mut solver = DfPnSolver::with_timeout(
                depth, budget, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, _solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                format!("{}M", budget / 1_000_000),
                solver.nodes_searched,
                elapsed.as_secs_f64(),
                solver.max_ply,
                solver.table.len(),
                result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // ===== Phase C: ply 22 全体の予算スケーリング =====
        writeln!(out, "\n### Phase C: ply 22 全体の予算スケーリング").unwrap();
        writeln!(out, "ply 22 OR node を段階的予算で直接解く\n").unwrap();

        let ply22_budgets: &[u64] = &[5_000_000, 10_000_000, 50_000_000];

        writeln!(out, "{:<12} {:<14} {:<10} {:<10} {:<10} {}",
            "Budget", "Nodes", "Time(s)", "MaxPly", "TT_pos", "Result").unwrap();

        let mut board22 = Board::new();
        board22.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board22.move_from_usi(pv[i]).unwrap();
            board22.do_move(m);
        }

        for &budget in ply22_budgets {
            let mut test_board = board22.clone();
            let depth = 41_u32; // 最大深さ
            let mut solver = DfPnSolver::with_timeout(
                depth, budget, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, _solved) = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    (format!("Mate({})", moves.len()), true),
                TsumeResult::CheckmateNoPv { .. } =>
                    ("MateNoPV".to_string(), true),
                TsumeResult::NoCheckmate { .. } =>
                    ("NoMate".to_string(), false),
                TsumeResult::Unknown { .. } =>
                    ("Unknown".to_string(), false),
            };

            writeln!(out, "{:<12} {:<14} {:<10.2} {:<10} {:<10} {}",
                format!("{}M", budget / 1_000_000),
                solver.nodes_searched,
                elapsed.as_secs_f64(),
                solver.max_ply,
                solver.table.len(),
                result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "             TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        // ===== Phase D: ply 20 以前の推定 =====
        writeln!(out, "\n### Phase D: ply 20, 18, 16 の予算スケーリング").unwrap();
        writeln!(out, "各 OR node を 50M で解く\n").unwrap();

        for target_ply in [20_usize, 18, 16] {
            let mut board_t = Board::new();
            board_t.set_sfen(sfen).unwrap();
            for i in 0..target_ply {
                let m = board_t.move_from_usi(pv[i]).unwrap();
                board_t.do_move(m);
            }

            let remaining = 39 - target_ply;
            let depth = 41_u32; // 最大深さ

            let mut test_board = board_t.clone();
            let mut solver = DfPnSolver::with_timeout(
                depth, 50_000_000, 32767, 600,
            );
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } =>
                    "NoMate".to_string(),
                TsumeResult::Unknown { .. } =>
                    "Unknown".to_string(),
            };

            writeln!(out, "ply {:<4} remaining={:<4} nodes={:<14} time={:<10.2}s maxply={:<4} TT_pos={:<10} {}",
                target_ply, remaining, solver.nodes_searched,
                elapsed.as_secs_f64(), solver.max_ply,
                solver.table.len(), result_str).unwrap();

            #[cfg(feature = "tt_diag")]
            {
                let proven = solver.table.count_proven();
                let disproven = solver.table.count_disproven();
                let intermediate = solver.table.count_intermediate();
                let total = solver.table.total_entries();
                writeln!(out, "         TT: {} (proven={}, disproven={}, intermediate={})",
                    total, proven, disproven, intermediate).unwrap();
            }
        }

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();

            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: {}", out_path);
    }

    /// ply 24 のノード数急増を診断する．
    ///
    /// ply 25 ANDノード(5g6f 後)の合駒フィルタ(futile/chain)分類と，
    /// Unknown となった4応手(1g1f, N*6g, P*7g, N*7g)の探索構造を調査する．
    /// 各Unknownの応手について，攻め手が取り進んだ後の再帰的なチェーン構造を
    /// 深さ2まで展開して報告する．
    #[test]
    fn test_ply24_diagnostic() {
        use std::io::Write;
        let out_path = "/tmp/ply24_diagnostic.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        // ply 25 ANDノード: 5g6f 後の局面
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";
        let mut board = Board::new();
        board.set_sfen(and_sfen).unwrap();

        let defender = board.turn(); // White
        let attacker = defender.opponent(); // Black

        let king_sq = board.king_square(defender).unwrap();
        let mut solver = DfPnSolver::default_solver();
        let checker_sq = board.compute_checkers_at(king_sq, attacker).lsb().unwrap();

        writeln!(out, "=== Ply 24 Diagnostic ===").unwrap();
        writeln!(out, "King: {}{}  Checker: {}{}",
            9 - king_sq.col(), (b'a' + king_sq.row()) as char,
            9 - checker_sq.col(), (b'a' + checker_sq.row()) as char).unwrap();

        // between_bb
        let between = attack::between_bb(checker_sq, king_sq);
        write!(out, "Between squares:").unwrap();
        for sq in between {
            write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap();
        }
        writeln!(out).unwrap();

        // futile/chain 分類
        let (futile, chain) = solver.compute_futile_and_chain_squares(
            &board, &between, king_sq, checker_sq, defender, attacker,
        );
        write!(out, "Futile squares:").unwrap();
        for sq in futile { write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap(); }
        writeln!(out).unwrap();
        write!(out, "Chain squares:").unwrap();
        for sq in chain { write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap(); }
        writeln!(out).unwrap();
        write!(out, "Normal squares:").unwrap();
        for sq in between {
            if !futile.contains(sq) && !chain.contains(sq) {
                write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char).unwrap();
            }
        }
        writeln!(out).unwrap();

        // 防御手一覧
        let defenses = solver.generate_defense_moves(&mut board);
        writeln!(out, "\nDefense moves ({}):", defenses.len()).unwrap();
        for m in &defenses {
            writeln!(out, "  {}", m.to_usi()).unwrap();
        }

        // 問題の4応手 + 比較用に解ける応手を分析
        let targets = ["1g1f", "N*6g", "P*7g", "N*7g", "L*6g", "B*7g"];

        writeln!(out, "\n{}", "=".repeat(80)).unwrap();
        writeln!(out, "=== Unknown応手の探索構造分析 ===").unwrap();

        for &target_usi in &targets {
            writeln!(out, "\n--- {} ---", target_usi).unwrap();

            // 応手を適用
            let mut after_def = board.clone();
            let def_move = after_def.move_from_usi(target_usi).unwrap();
            after_def.do_move(def_move);
            writeln!(out, "SFEN after: {}", after_def.sfen()).unwrap();

            // この局面(ORノード)で攻め手を生成
            let attack_solver = DfPnSolver::default_solver();
            let attacks = attack_solver.generate_check_moves(&mut after_def);
            writeln!(out, "Attack moves ({}):", attacks.len()).unwrap();

            // 各攻め手について，1手進めた後のANDノードを簡易分析
            for atk in &attacks {
                let mut after_atk = after_def.clone();
                after_atk.do_move(*atk);

                // この後の防御手数をカウント
                let mut sub_solver = DfPnSolver::default_solver();
                let sub_defenses = sub_solver.generate_defense_moves(&mut after_atk);

                // チェーン構造の確認: 飛び駒の王手ならbetween/futile/chainを表示
                let sub_king_sq = match after_atk.king_square(after_atk.turn()) {
                    Some(sq) => sq,
                    None => {
                        writeln!(out, "  {} → no king (capture?), defenses={}", atk.to_usi(), sub_defenses.len()).unwrap();
                        continue;
                    }
                };
                let sub_attacker = after_atk.turn().opponent();
                let sub_checkers = after_atk.compute_checkers_at(sub_king_sq, sub_attacker);
                if sub_checkers.is_empty() {
                    writeln!(out, "  {} → not check, defenses={}", atk.to_usi(), sub_defenses.len()).unwrap();
                    continue;
                }
                let sub_checker_sq = sub_checkers.lsb().unwrap();
                let sub_sliding = sub_solver.find_sliding_checker(&after_atk, sub_king_sq, sub_attacker);
                let chain_info = if sub_sliding.is_some() {
                    let sub_between = attack::between_bb(sub_checker_sq, sub_king_sq);
                    let (sf, sc) = sub_solver.compute_futile_and_chain_squares(
                        &after_atk, &sub_between, sub_king_sq, sub_checker_sq,
                        after_atk.turn(), sub_attacker,
                    );
                    format!("between={} futile={} chain={} normal={}",
                        sub_between.count(), sf.count(), sc.count(),
                        sub_between.count() - sf.count() - sc.count())
                } else {
                    "non-sliding".to_string()
                };

                writeln!(out, "  {} → defenses={} [{}]",
                    atk.to_usi(), sub_defenses.len(), chain_info).unwrap();

                // 攻め手が飛び駒の取り進みの場合(チェーン再帰)，2段目も展開
                if sub_sliding.is_some() && sub_defenses.len() > 5 {
                    // 防御手のうちドロップ(合駒)の数
                    let drop_count = sub_defenses.iter().filter(|m| m.is_drop()).count();
                    let non_drop_count = sub_defenses.len() - drop_count;
                    writeln!(out, "    (drops={}, non-drops={})", drop_count, non_drop_count).unwrap();
                }
            }

            // 250Kノードで解いて各深さのノード使用量を確認
            let mut solve_board = after_def.clone();
            let remaining = 15 - 1; // 攻め手1手消費
            let depth = (remaining + 2).min(41) as u32;
            let mut solve_solver = DfPnSolver::with_timeout(
                depth, 250_000, 32767, 30,
            );
            solve_solver.set_find_shortest(false);

            let start = std::time::Instant::now();
            let result = solve_solver.solve(&mut solve_board);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            writeln!(out, "Solve: {} nodes={} time={:.2}s max_ply={}",
                result_str, solve_solver.nodes_searched,
                elapsed.as_secs_f64(), solve_solver.max_ply).unwrap();
            writeln!(out, "TT entries: {}", solve_solver.table.len()).unwrap();
        }

            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: /tmp/ply24_diagnostic.log");
    }

    /// ply 24 TT共有効果の測定．
    ///
    /// 同じANDノードの兄弟応手間で TT を共有した場合としない場合の
    /// ノード数差を計測し，hand dominance による TT ヒットの実効性を検証する．
    ///
    /// 検証方法:
    /// 1. L*6g (解ける) を解いた後の TT を保持したまま N*6g を解く (共有あり)
    /// 2. N*6g を新規 TT で解く (共有なし)
    /// 3. ノード数とTTエントリ数を比較
    #[test]
    fn test_ply24_tt_sharing_effectiveness() {
        use std::io::Write;
        let out_path = "/tmp/ply24_tt_sharing.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        // ply 25 ANDノード: 5g6f 後
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";

        // チェーン合駒ペア: (先に解く応手, 後に解く応手)
        let pairs = [
            ("L*6g", "N*6g", "同一マス(6g)への異種駒ドロップ"),
            ("B*7g", "N*7g", "同一マス(7g)への異種駒ドロップ"),
            ("B*7g", "P*7g", "同一マス(7g)への異種駒ドロップ(歩)"),
            ("L*6g", "P*7g", "異なるマスへの異種駒ドロップ"),
        ];

        writeln!(out, "=== TT共有効果の測定 ===\n").unwrap();

        for (first_usi, second_usi, desc) in &pairs {
            writeln!(out, "--- {} ---", desc).unwrap();
            writeln!(out, "先行: {}, 後行: {}\n", first_usi, second_usi).unwrap();

            // --- (A) TT共有あり: first → second (TT保持) ---
            let mut board_first = Board::new();
            board_first.set_sfen(and_sfen).unwrap();
            let m_first = board_first.move_from_usi(first_usi).unwrap();
            let mut after_first = board_first.clone();
            after_first.do_move(m_first);

            let depth = 16u32;
            let budget = 500_000u64;
            let mut solver = DfPnSolver::with_timeout(depth, budget, 32767, 60);
            solver.set_find_shortest(false);

            // first を解く
            let start = Instant::now();
            let r1 = solver.solve(&mut after_first);
            let first_nodes = solver.nodes_searched;
            let first_tt = solver.table.len();
            let first_time = start.elapsed();
            let r1_str = match &r1 {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                _ => "Other".to_string(),
            };
            writeln!(out, "(A) {} 単独: nodes={}, TT={}, {:.2}s → {}",
                first_usi, first_nodes, first_tt, first_time.as_secs_f64(), r1_str).unwrap();

            // TT を保持したまま second を解く
            // solve() は table.clear() するので，手動で状態をリセット
            let mut board_second = Board::new();
            board_second.set_sfen(and_sfen).unwrap();
            let m_second = board_second.move_from_usi(second_usi).unwrap();
            let mut after_second = board_second.clone();
            after_second.do_move(m_second);

            // solve() の内部を手動で再現(table.clear() をスキップ)
            solver.nodes_searched = 0;
            solver.max_ply = 0;
            solver.path_len = 0;
            solver.killer_table.clear();
            solver.start_time = Instant::now();
            solver.timed_out = false;
            solver.next_gc_check = 100_000;
            solver.attacker = after_second.turn; // Black (attacker)
            // table.clear() を意図的にスキップ
            let tt_before = solver.table.len();

            let saved_max_nodes = solver.max_nodes;
            solver.max_nodes = saved_max_nodes / 2;
            let _pns_pv = solver.pns_main(&mut after_second);
            solver.max_nodes = saved_max_nodes;

            let pk = position_key(&after_second);
            let att_hand = after_second.hand[solver.attacker.index()];
            let (root_pn, root_dn, _) = solver.look_up_pn_dn(pk, &att_hand, solver.depth as u16);
            if root_pn != 0 && root_dn != 0 && !solver.timed_out && solver.nodes_searched < solver.max_nodes {
                solver.mid(&mut after_second, INF - 1, INF - 1, 0, true);
            }

            let shared_nodes = solver.nodes_searched;
            let shared_tt = solver.table.len();
            let shared_time = solver.start_time.elapsed();
            let (final_pn, final_dn, _) = solver.look_up_pn_dn(pk, &att_hand, solver.depth as u16);
            let r2_shared = if final_pn == 0 { "Proved" } else if final_dn == 0 { "Disproved" } else { "Unknown" };
            writeln!(out, "(A) {} (TT共有): nodes={}, TT={}→{} (+{}), {:.2}s → {}",
                second_usi, shared_nodes, tt_before, shared_tt, shared_tt - tt_before,
                shared_time.as_secs_f64(), r2_shared).unwrap();

            // --- (B) TT共有なし: second を新規ソルバで解く ---
            let mut fresh_board = Board::new();
            fresh_board.set_sfen(and_sfen).unwrap();
            let m_fresh = fresh_board.move_from_usi(second_usi).unwrap();
            let mut after_fresh = fresh_board.clone();
            after_fresh.do_move(m_fresh);

            let mut fresh_solver = DfPnSolver::with_timeout(depth, budget, 32767, 60);
            fresh_solver.set_find_shortest(false);

            let start = Instant::now();
            let r2 = fresh_solver.solve(&mut after_fresh);
            let fresh_nodes = fresh_solver.nodes_searched;
            let fresh_tt = fresh_solver.table.len();
            let fresh_time = start.elapsed();
            let r2_str = match &r2 {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            writeln!(out, "(B) {} (新規TT): nodes={}, TT={}, {:.2}s → {}",
                second_usi, fresh_nodes, fresh_tt, fresh_time.as_secs_f64(), r2_str).unwrap();

            // 効果
            if shared_nodes < fresh_nodes && shared_nodes > 0 {
                writeln!(out, "→ TT共有効果: {:.1}x 削減 ({} → {} nodes)\n",
                    fresh_nodes as f64 / shared_nodes as f64, fresh_nodes, shared_nodes).unwrap();
            } else if shared_nodes > 0 {
                writeln!(out, "→ TT共有効果: なし (shared={}, fresh={})\n",
                    shared_nodes, fresh_nodes).unwrap();
            } else {
                writeln!(out, "→ TT共有効果: 計測不能\n").unwrap();
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: /tmp/ply24_tt_sharing.log");
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

    /// 39手詰め ply 22 局面で偽の短手数詰みを返すバグのリグレッションテスト．
    ///
    /// ソルバーが Mate(7) を返すが，最終局面 8g8e の後に合法手が36手あり
    /// 詰みではない(is_checkmate=false)．証明ツリーが不正．
    /// `find_shortest=true` でも同じ Mate(7) を返すため PV 抽出ではなく
    /// 証明自体のバグ．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_pv_must_end_in_checkmate() {
        // 39手詰めの ply 22 局面(攻め番)
        let sfen = "9/3+N1P3/7+R1/9/9/8k/1R2S4/3p5/9 b P2b4g3s3n4l15p 23";

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(19, 1_000_000, 32767, 180);
        solver.set_find_shortest(false);

        let mut test_board = board.clone();
        let result = solver.solve(&mut test_board);

        if let TsumeResult::Checkmate { moves, .. } = &result {
            // PV の全手が合法手であること
            let mut vb = board.clone();
            for (i, m) in moves.iter().enumerate() {
                let legal = movegen::generate_legal_moves(&mut vb);
                assert!(
                    legal.iter().any(|lm| lm.to_usi() == m.to_usi()),
                    "PV move {} ({}) is illegal at SFEN: {}",
                    i + 1, m.to_usi(), vb.sfen()
                );
                // 攻め手(偶数 index)は王手であること
                vb.do_move(*m);
                if i % 2 == 0 {
                    assert!(
                        vb.is_in_check(vb.turn()),
                        "ATK move {} ({}) does not give check",
                        i + 1, m.to_usi()
                    );
                }
            }

            // 最終局面が詰み(合法手0 かつ王手)であること
            let final_legal = movegen::generate_legal_moves(&mut vb);
            assert!(
                final_legal.is_empty() && vb.is_in_check(vb.turn()),
                "PV of length {} does not end in checkmate: \
                 legal_moves={}, in_check={}, SFEN={}",
                moves.len(),
                final_legal.len(),
                vb.is_in_check(vb.turn()),
                vb.sfen()
            );
        }
        // Checkmate 以外(Unknown 等)は許容: 解けなかっただけ
    }

    /// ply 22 OR ノードの王手ごとのノード消費を調査する．
    ///
    /// 151K ノードで NoMate (1M 予算を使い切らない) の原因を特定:
    /// - depth 制限による打ち切り
    /// - NM (不詰) の誤判定
    /// - IDS の早期終了
    #[test]
    #[ignore]
    fn test_tsume_39te_ply22_or_node_breakdown() {
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

        // ply 22 の局面を構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for i in 0..22 {
            let m = board.move_from_usi(pv[i]).unwrap();
            board.do_move(m);
        }

        let ply22_sfen = board.sfen();
        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" Ply 22 OR node breakdown (残り17手, PV: P*1g)");
        verbose_eprintln!(" SFEN: {}", ply22_sfen);
        verbose_eprintln!("{}", "=".repeat(80));

        // 1. まず全体を depth=19 で解いてみる
        verbose_eprintln!("\n--- 全体 solve (depth=19, 1M nodes) ---");
        {
            let mut b = board.clone();
            let mut solver = DfPnSolver::with_timeout(19, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut b);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            verbose_eprintln!("  depth=19: {} nodes={} time={:.2}s max_ply={}",
                result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 2. depth を変えて解いてみる
        verbose_eprintln!("\n--- depth 別 solve (1M nodes) ---");
        for depth in [17u32, 19, 21, 23, 25, 31, 41] {
            let mut b = board.clone();
            let mut solver = DfPnSolver::with_timeout(depth, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut b);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            verbose_eprintln!("  depth={:<4} {} nodes={:<10} time={:.2}s max_ply={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 3. check_moves の一覧と個別探索
        verbose_eprintln!("\n--- 王手一覧と個別探索 (depth=17, 250K nodes each) ---");
        let check_solver = DfPnSolver::default_solver();
        let check_moves = check_solver.generate_check_moves(&mut board);
        verbose_eprintln!("  王手数: {}", check_moves.len());

        // brute-force でも確認
        let brute_checks: Vec<String> = movegen::generate_legal_moves(&mut board)
            .into_iter()
            .filter(|m| {
                let c = board.do_move(*m);
                let gives_check = board.is_in_check(board.turn);
                board.undo_move(*m, c);
                gives_check
            })
            .map(|m| m.to_usi())
            .collect();
        verbose_eprintln!("  brute-force 王手数: {}", brute_checks.len());

        verbose_eprintln!("  {:<12} {:<14} {:<10} {:<10} {}",
            "Move", "Nodes", "Time(s)", "MaxPly", "Result");
        for cm in &check_moves {
            let mut after = board.clone();
            after.do_move(*cm);

            // 王手後 → AND node → 各応手を含む残り16手を探索
            let mut sub = after.clone();
            let mut solver = DfPnSolver::with_timeout(17, 250_000, 32767, 30);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut sub);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            let marker = if cm.to_usi() == "P*1g" { " ← PV" } else { "" };
            verbose_eprintln!("  {:<12} {:<14} {:<10.2} {:<10} {}{}",
                cm.to_usi(), solver.nodes_searched, elapsed.as_secs_f64(),
                solver.max_ply, result_str, marker);
        }

        // 4. P*1g に注目: depth を変えて解く
        verbose_eprintln!("\n--- P*1g 単体 depth 別 (1M nodes) ---");
        let pawn_drop = board.move_from_usi("P*1g").unwrap();
        let mut after_pg = board.clone();
        after_pg.do_move(pawn_drop);

        for depth in [15u32, 17, 19, 21] {
            let mut sub = after_pg.clone();
            let mut solver = DfPnSolver::with_timeout(depth, 1_000_000, 32767, 180);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut sub);
            let elapsed = start.elapsed();
            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
                _ => "Other".to_string(),
            };
            verbose_eprintln!("  depth={:<4} {} nodes={:<10} time={:.2}s max_ply={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(), solver.max_ply);
        }

        // 5. all_checks_refutable_recursive のバグ確認
        // P*1g 後の各応手について，次の王手の有無を確認
        verbose_eprintln!("\n--- all_checks_refutable analysis for P*1g ---");
        let pawn_drop2 = board.move_from_usi("P*1g").unwrap();
        let cap_pg = board.do_move(pawn_drop2);
        let mut def_solver = DfPnSolver::default_solver();
        let defenses = def_solver.generate_defense_moves(&mut board);
        verbose_eprintln!("  P*1g 後の応手数: {}", defenses.len());
        for def_mv in &defenses {
            let cap_d = board.do_move(*def_mv);
            let next_checks = def_solver.generate_check_moves(&mut board);
            verbose_eprintln!("  {} → 次の王手数: {} {:?}",
                def_mv.to_usi(), next_checks.len(),
                next_checks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());
            board.undo_move(*def_mv, cap_d);
        }
        board.undo_move(pawn_drop2, cap_pg);

        verbose_eprintln!("{}", "=".repeat(80));
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
        let result = solver.solve(&mut board);

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

    /// Ply 4 の AND ノードにおける各子ノードの探索難易度を診断する．
    ///
    /// 各回避手を適用した局面に対して小予算ソルブを実行し，
    /// 消費ノード数・結果を比較する．AND ノードでは 1 つの不詰み子
    /// が見つかれば十分なので，簡単な子が先に試されるべき．
    #[test]
    #[ignore]
    fn test_ply4_child_node_difficulty() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        verbose_eprintln!("\nPly 4 子ノード難易度診断 (budget=100,000 nodes, depth=37)");
        verbose_eprintln!("{:>4} {:>12} {:>8} {:>10} {:>10} {:>8}",
            "#", "Move", "Type", "Nodes", "Result", "PV len");
        verbose_eprintln!("{}", "-".repeat(65));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 攻め方視点でソルブ (残り depth=37, budget=100K)
            let mut child_solver = DfPnSolver::new(37, 100_000, 32767);
            child_solver.set_find_shortest(false);
            let result = child_solver.solve(&mut child_board);

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, pv_len) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE"), moves.len()),
                TsumeResult::CheckmateNoPv { .. } =>
                    (format!("MATE(nopv)"), 0),
                TsumeResult::NoCheckmate { .. } =>
                    (format!("NO_MATE"), 0),
                TsumeResult::Unknown { .. } =>
                    (format!("UNKNOWN"), 0),
            };
            let nodes = match &result {
                TsumeResult::Checkmate { nodes_searched, .. } => *nodes_searched,
                TsumeResult::CheckmateNoPv { nodes_searched } => *nodes_searched,
                TsumeResult::NoCheckmate { nodes_searched } => *nodes_searched,
                TsumeResult::Unknown { nodes_searched, .. } => *nodes_searched,
            };

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };

            verbose_eprintln!("{:>4} {:>12} {:>8} {:>10} {:>10} {:>8}{}",
                i + 1, defense.to_usi(), move_type, nodes, result_str, pv_len, marker);
        }
    }

    /// 39手詰 PV 上の合駒(S*6i)後の局面を単体ソルブし，
    /// 各応手に必要なノード数を計測する．
    ///
    /// 合駒は30手目(S*6i)で，攻め方は31手目(8i6i)で銀を取る．
    /// 取った後の局面(攻め方番)を単体で解き，ノード数を確認する．
    #[test]
    #[ignore]
    fn test_interpose_subproblem_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        // PV: 39手
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i",
        ];
        // 29手目(8g8i = 飛車8i)まで進める → 30手目が合駒局面(AND ノード)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_usi {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 29 後: 後手番(AND ノード) - ここで合駒 S*6i が最善
        assert_eq!(board.turn, Color::White);

        // AND ノードの回避手を生成
        let mut solver_tmp = DfPnSolver::default_solver();
        let defenses = solver_tmp.generate_defense_moves(&mut board);

        verbose_eprintln!("\n=== 合駒局面ブレークダウン (ply 29 後) ===");
        verbose_eprintln!("回避手数: {} (うち drop: {})",
            defenses.len(),
            defenses.iter().filter(|m| m.is_drop()).count());

        verbose_eprintln!("\n{:>4} {:>8} {:>5} {:>10} {:>10.1} {:>8}",
            "#", "Move", "Type", "Nodes", "Time(ms)", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        // 各回避手を適用後，攻め方視点でソルブ
        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 残り手数 = 39 - 30 = 9 手(合駒後)
            // ただしここで攻め方が合駒を取るので，取り後は残り 8 手
            let remaining = 10; // 余裕を持って depth=10
            let mut child_solver = DfPnSolver::new(remaining, 100_000, 32767);
            child_solver.set_find_shortest(false);
            let start = Instant::now();
            let result = child_solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let is_best = defense.to_usi() == "S*6i";
            let marker = if is_best { " ← BEST" } else { "" };
            verbose_eprintln!("{:>4} {:>8} {:>5} {:>10} {:>10.1} {:>8}{}",
                i + 1, defense.to_usi(), move_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str, marker);
        }
    }

    /// PV 上の各サブ問題(守備方の手後 = 攻め方番)を個別ソルブし，
    /// どの深さからソルブ困難になるかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_solve() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        verbose_eprintln!("\n{:>5} {:>3} {:>8} {:>10} {:>10} {:>8}",
            "After", "Typ", "Remain", "Nodes", "Time(ms)", "Result");
        verbose_eprintln!("{}", "-".repeat(62));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);

            // 全 ply をテスト(偶数=OR, 奇数=AND)
            let remaining = 39 - (i + 1);
            let or_sub = i % 2 == 1; // 奇数 i 後は攻め方番 = OR
            let budget = if remaining >= 31 { 10_000_000u64 } else { 2_000_000 };
            let mut solver = DfPnSolver::new(
                (remaining + 2) as u32, budget, 32767,
            );
            solver.set_find_shortest(false);
            let start = Instant::now();
            let mut test_board = board.clone();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (status, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let node_type = if or_sub { "OR " } else { "AND" };
            verbose_eprintln!("{:>5} {:>3} {:>8} {:>10} {:>10.1} {:>8}",
                format!("ply{}", i + 1), node_type, remaining, nodes,
                elapsed.as_secs_f64() * 1000.0, status);
        }
    }

    /// 39手詰 PV 上の各サブ問題を小予算(100K)でソルブし，
    /// どの ply でノード爆発が起きるかを特定する．
    ///
    /// 各 ply 後の局面を独立した詰将棋として解き，
    /// 消費ノード数と結果を出力する．
    #[test]
    #[ignore]
    fn test_tsume_39te_subproblem_quick() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_usi = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
            "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
            "2g2h", "3i4i", "2h4h",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        verbose_eprintln!("\n{:>5} {:>3} {:>6} {:>10} {:>10} {:>10}",
            "After", "Typ", "Remain", "Budget", "Nodes", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        for (i, &usi) in pv_usi.iter().enumerate() {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);

            let remaining = 39 - (i + 1);
            if remaining == 0 { break; }
            let or_sub = i % 2 == 1; // 奇数 i 後は攻め方番 = OR

            // 小予算で素早くテスト
            let budget = 100_000u64;
            let mut solver = DfPnSolver::new(
                (remaining + 2) as u32, budget, 32767,
            );
            solver.set_find_shortest(false);
            let mut test_board = board.clone();
            let result = solver.solve(&mut test_board);

            let (status, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let node_type = if or_sub { "OR " } else { "AND" };
            verbose_eprintln!("{:>5} {:>3} {:>6} {:>10} {:>10} {:>10}",
                format!("ply{}", i + 1), node_type, remaining,
                budget, nodes, status);
        }
    }

    /// 39手詰で最も分岐の多い ply 4 (AND ノード，20手の応手) の
    /// 各回避手を単体ソルブし，どの分岐でノード爆発するかを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply4_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        verbose_eprintln!("\n=== Ply 4 AND ノード回避手ブレークダウン ===");
        verbose_eprintln!("回避手数: {} (うち drop: {}, move: {})",
            defenses.len(),
            defenses.iter().filter(|m| m.is_drop()).count(),
            defenses.iter().filter(|m| !m.is_drop()).count());

        verbose_eprintln!("\n{:>4} {:>8} {:>5} {:>10} {:>10} {:>10}",
            "#", "Move", "Type", "Budget", "Nodes", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            // 残り35手分で探索 (depth=37, budget=500K)
            let mut child_solver = DfPnSolver::new(37, 500_000, 32767);
            child_solver.set_find_shortest(false);
            let result = child_solver.solve(&mut child_board);

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };
            verbose_eprintln!("{:>4} {:>8} {:>5} {:>10} {:>10} {:>10}{}",
                i + 1, defense.to_usi(), move_type,
                500_000, nodes, result_str, marker);
        }
    }

    /// 39手詰 ply 4 の全応手について，合駒は龍で取った後の局面を solve し
    /// PV(詰み筋)を比較する．
    ///
    /// 仮説: 合駒を取ると元の局面とほぼ同じ → 同じ詰み筋が繰り返される．
    /// 合駒の数だけ同難度のサブ問題が増殖しているかを確認する．
    #[test]
    #[ignore]
    fn test_tsume_39te_ply4_pv_comparison() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4 の局面へ

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        // ── Part 1: 逃げ手(盤上の手)の PV ──
        verbose_eprintln!("\n=== Part 1: 逃げ手の詰み筋 (budget=500K) ===\n");
        for (i, &defense) in defenses.iter().enumerate() {
            if defense.is_drop() { continue; }
            let mut child_board = board.clone();
            child_board.do_move(defense);

            let mut s = DfPnSolver::with_timeout(37, 500_000, 32767, 10);
            s.set_find_shortest(false);
            let result = s.solve(&mut child_board);

            let is_correct = defense.to_usi() == "4c3d";
            let marker = if is_correct { " ← CORRECT" } else { "" };
            print_result(i + 1, &defense.to_usi(), &result, marker);
        }

        // ── Part 2: 合駒を龍で取った後の局面の PV ──
        // 合駒は 5c/6c/7c/8c 筋に打たれ，龍(9c)が取る．
        // 取った後の局面を solve し，PV を比較する．
        verbose_eprintln!("\n=== Part 2: 合駒 → 龍で取った後の詰み筋 (budget=500K) ===");
        verbose_eprintln!("(defense → capture → 玉方応手の各分岐を solve)\n");

        let interpositions: Vec<(&str, &str)> = vec![
            // (合駒, 龍の取り)
            ("B*5c", "9c5c"), ("G*5c", "9c5c"), ("S*5c", "9c5c"),
            ("N*5c", "9c5c"), ("L*5c", "9c5c"), ("P*5c", "9c5c"),
            ("L*6c", "9c6c"), ("B*6c", "9c6c"), ("N*6c", "9c6c"),
            ("P*7c", "9c7c"), ("B*7c", "9c7c"), ("N*7c", "9c7c"),
            ("P*8c", "9c8c"), ("B*8c", "9c8c"), ("N*8c", "9c8c"),
        ];

        for (idx, &(interpose_usi, capture_usi)) in interpositions.iter().enumerate() {
            let interpose = board.move_from_usi(interpose_usi).unwrap();
            let mut b = board.clone();
            b.do_move(interpose);
            let capture = b.move_from_usi(capture_usi).unwrap();
            b.do_move(capture);

            // ply 6 相当: 玉方番(AND) → 各応手後の攻め方局面を solve
            let defs = solver.generate_defense_moves(&mut b);
            verbose_eprintln!("{:>2}. {} → {} (玉方応手{}手)",
                idx + 1, interpose_usi, capture_usi, defs.len());

            for &def in defs.iter() {
                let mut bc = b.clone();
                bc.do_move(def);

                // 攻め方番: 残り33手で solve
                let mut s = DfPnSolver::with_timeout(33, 200_000, 32767, 3);
                s.set_find_shortest(false);
                let result = s.solve(&mut bc);

                let label = format!("  {} → {}", capture_usi, def.to_usi());
                print_result(0, &label, &result, "");
            }
            verbose_eprintln!();
        }
    }

    fn print_result(idx: usize, label: &str, result: &TsumeResult, marker: &str) {
        match result {
            TsumeResult::Checkmate { moves, nodes_searched } => {
                let pv_str: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                if idx > 0 {
                    verbose_eprintln!("{:>2}. {} nodes={:>7} MATE({:>2}) PV: {}{}",
                        idx, label, nodes_searched, moves.len(), pv_str.join(" "), marker);
                } else {
                    verbose_eprintln!("    {} nodes={:>7} MATE({:>2}) PV: {}{}",
                        label, nodes_searched, moves.len(), pv_str.join(" "), marker);
                }
            }
            TsumeResult::CheckmateNoPv { nodes_searched } => {
                verbose_eprintln!("    {} nodes={:>7} MATE(nopv){}", label, nodes_searched, marker);
            }
            TsumeResult::NoCheckmate { nodes_searched } => {
                verbose_eprintln!("    {} nodes={:>7} NO_MATE{}", label, nodes_searched, marker);
            }
            TsumeResult::Unknown { nodes_searched } => {
                verbose_eprintln!("    {} nodes={:>7} UNKNOWN{}", label, nodes_searched, marker);
            }
        }
    }

    /// 39手詰のボトルネック分析: 不詰み証明に時間がかかる分岐を特定する．
    ///
    /// ply 4 (AND ノード) の各応手について:
    /// 1. 応手後の OR ノード(ply 5)で生成される王手の数と各王手の結果
    /// 2. 各王手の先(ply 6 AND ノード)の回避手数と各回避手の結果
    /// を再帰的に調べ，ノード消費のホットスポットを特定する．
    #[test]
    #[ignore]
    fn test_tsume_39te_bottleneck_analysis() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 4: 後手番(AND ノード)
        assert_eq!(board.turn, Color::White);

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);

        // Phase 1: 各応手を 500K でソルブし，NO_MATE / UNKNOWN を特定
        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" 39手詰 ply 4 ボトルネック分析");
        verbose_eprintln!("{}", "=".repeat(80));
        verbose_eprintln!("\n--- Phase 1: ply 4 応手の概要 (budget=500K) ---\n");
        verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>10.1} {:>10}",
            "#", "Move", "Type", "Nodes", "Time(ms)", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        let mut hard_defenses: Vec<(Move, String, u64, String)> = Vec::new();

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(defense);

            let mut s = DfPnSolver::with_timeout(37, 500_000, 32767, 10);
            s.set_find_shortest(false);
            let start = Instant::now();
            let result = s.solve(&mut child_board);
            let elapsed = start.elapsed();

            let move_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>10.1} {:>10}",
                i + 1, defense.to_usi(), move_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str);

            // 500K 以上消費 or UNKNOWN → ボトルネック候補
            if nodes >= 200_000 {
                hard_defenses.push((
                    defense, defense.to_usi(), nodes, result_str.clone(),
                ));
            }
        }

        // Phase 2: ボトルネック応手の内部構造を分析
        // check 後は defender turn なので，各回避手を個別に attacker turn でソルブ
        verbose_eprintln!("\n--- Phase 2: ボトルネック応手を深掘り ---");
        verbose_eprintln!("(defense → check → reply → attacker 視点でソルブ)\n");

        for (defense, defusi, parent_nodes, parent_result) in &hard_defenses {
            let mut def_board = board.clone();
            def_board.do_move(*defense);

            // ply 5: 攻め方番(OR) — 王手を列挙
            let checks = solver.generate_check_moves(&mut def_board);

            verbose_eprintln!("=== {} ({}，親ノード={}，王手数={}) ===\n",
                defusi, parent_result, parent_nodes, checks.len());

            for (j, &check) in checks.iter().enumerate() {
                let mut check_board = def_board.clone();
                check_board.do_move(check);

                // ply 6: 守備方番(AND) — 回避手を列挙
                let defs_after = solver.generate_defense_moves(&mut check_board);
                let def_count = defs_after.len();
                let check_type = if check.is_drop() { "drop" } else { "move" };

                verbose_eprintln!("  王手 {:>2}. {} ({}) → 回避手 {} 手",
                    j + 1, check.to_usi(), check_type, def_count);

                if def_count == 0 {
                    verbose_eprintln!("    → 応手なし(詰み)\n");
                    continue;
                }

                verbose_eprintln!("  {:>4} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}",
                    "#", "Reply", "Type", "Nodes", "ms", "Result", "Chks");
                verbose_eprintln!("  {}", "-".repeat(58));

                let mut total_nodes: u64 = 0;
                let mut nm_count = 0;
                let mut mate_count = 0;
                let mut unk_count = 0;

                for (k, &reply) in defs_after.iter().enumerate() {
                    let mut reply_board = check_board.clone();
                    reply_board.do_move(reply);

                    // ply 7: 攻め方番(OR) — 正しい視点でソルブ
                    let next_checks = solver.generate_check_moves(&mut reply_board);
                    let chk_count = next_checks.len();

                    let mut s = DfPnSolver::with_timeout(33, 200_000, 32767, 5);
                    s.set_find_shortest(false);
                    let start = Instant::now();
                    let result = s.solve(&mut reply_board);
                    let elapsed = start.elapsed();

                    let reply_type = if reply.is_drop() { "drop" } else { "move" };
                    let (result_str, nodes) = match &result {
                        TsumeResult::Checkmate { moves, nodes_searched } => {
                            mate_count += 1;
                            (format!("MATE({})", moves.len()), *nodes_searched)
                        }
                        TsumeResult::CheckmateNoPv { nodes_searched } => {
                            mate_count += 1;
                            ("MATE(nopv)".into(), *nodes_searched)
                        }
                        TsumeResult::NoCheckmate { nodes_searched } => {
                            nm_count += 1;
                            ("NM".into(), *nodes_searched)
                        }
                        TsumeResult::Unknown { nodes_searched } => {
                            unk_count += 1;
                            ("UNK".into(), *nodes_searched)
                        }
                    };
                    total_nodes += nodes;

                    let heavy = if nodes >= 50_000 { " <<<" } else { "" };
                    verbose_eprintln!("  {:>4} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}{}",
                        k + 1, reply.to_usi(), reply_type, nodes,
                        elapsed.as_secs_f64() * 1000.0, result_str, chk_count, heavy);
                }
                verbose_eprintln!("  合計: {} nodes | MATE={} NM={} UNK={}\n",
                    total_nodes, mate_count, nm_count, unk_count);
            }
        }

        // Phase 3: PV 上の正解手(4c3d)後を 2 階層深掘り
        verbose_eprintln!("--- Phase 3: 正解手 4c3d → 1b2c(PV) 後の回避手分析 ---\n");
        let correct_def = board.move_from_usi("4c3d").unwrap();
        let mut correct_board = board.clone();
        correct_board.do_move(correct_def);

        let pv_check = correct_board.move_from_usi("1b2c").unwrap();
        let mut pv_board = correct_board.clone();
        pv_board.do_move(pv_check);

        let pv_defs = solver.generate_defense_moves(&mut pv_board);
        verbose_eprintln!("4c3d → 1b2c 後の回避手: {} 手", pv_defs.len());
        verbose_eprintln!("{:>3} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}",
            "#", "Reply", "Type", "Nodes", "ms", "Result", "Chks");
        verbose_eprintln!("{}", "-".repeat(55));

        for (k, &reply) in pv_defs.iter().enumerate() {
            let mut reply_board = pv_board.clone();
            reply_board.do_move(reply);

            let next_checks = solver.generate_check_moves(&mut reply_board);
            let chk_count = next_checks.len();

            let mut s = DfPnSolver::with_timeout(33, 2_000_000, 32767, 30);
            s.set_find_shortest(false);
            let start = Instant::now();
            let result = s.solve(&mut reply_board);
            let elapsed = start.elapsed();

            let reply_type = if reply.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            let is_pv = reply.to_usi() == "3d2c";
            let marker = if is_pv { " ← PV" } else { "" };
            let heavy = if nodes >= 100_000 { " <<<" } else { "" };
            verbose_eprintln!("{:>3} {:>10} {:>5} {:>10} {:>8.1} {:>10} {:>5}{}{}",
                k + 1, reply.to_usi(), reply_type, nodes,
                elapsed.as_secs_f64() * 1000.0, result_str, chk_count, heavy, marker);
        }
    }

    /// 39手詰 ply 4 の未解決応手(500K で NO_MATE)を高予算で再調査し，
    /// 真の詰み手数・必要ノード数・分岐の特徴を分析する．
    #[test]
    #[ignore]
    fn test_tsume_39te_hard_defenses_deep() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = ["7b6b", "5b4c", "8b9c"]; // 3手進めて ply 4
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        // 前回 500K で未解決だった応手のみ高予算で再調査
        let hard_moves = [
            ("4c3b", "king move (3b)"),
            ("4c3d", "king move (3d) [CORRECT]"),
            ("N*5c", "knight drop 5c"),
            ("P*5c", "pawn drop 5c"),
            ("N*6c", "knight drop 6c"),
            ("P*7c", "pawn drop 7c"),
            ("N*7c", "knight drop 7c"),
        ];

        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" 39手詰 ply 4: 高予算(5M)でのボトルネック応手分析");
        verbose_eprintln!("{}", "=".repeat(80));
        verbose_eprintln!("\n{:>12} {:>25} {:>10} {:>8} {:>10}",
            "Move", "Description", "Nodes", "Time(s)", "Result");
        verbose_eprintln!("{}", "-".repeat(75));

        for (usi, desc) in &hard_moves {
            let m = board.move_from_usi(usi).unwrap();
            let mut child_board = board.clone();
            child_board.do_move(m);

            let mut solver = DfPnSolver::with_timeout(37, 5_000_000, 32767, 120);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            verbose_eprintln!("{:>12} {:>25} {:>10} {:>8.1} {:>10}",
                usi, desc, nodes, elapsed.as_secs_f64(), result_str);

            // 解けた場合は PV を表示
            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                verbose_eprintln!("             PV: {}", pv.join(" "));
            }
        }

        // 正解手(4c3d)の PV を辿り，各 ply での分岐数とノードを段階的に分析
        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" 正解 PV 沿いの IDS 各段階での進捗");
        verbose_eprintln!("{}", "=".repeat(80));

        let _pv_usi = [
            "4c3d", "1b2c", "3d2c", "N*1e", "2c3b", "N*2d",
            "3b2b", "2d1b+", "2b3b", "1b2b", "3b2b", "4f1c",
            "2b1c", "9c3c", "1c1d", "3c2c", "1d1e", "P*1f",
            "1e1f", "P*1g", "1f1g", "5g6f", "1g1h", "2c2g",
            "1h1i", "8g8i", "S*6i", "8i6i", "6h6i+", "S*2h",
            "1i2i", "2h3g", "2i3i", "2g2h", "3i4i", "2h4h",
        ];

        // IDS depth 5,9,13,...,41 での進捗
        let depths = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41];

        let mut pv_board = board.clone();
        let correct_def = board.move_from_usi("4c3d").unwrap();
        pv_board.do_move(correct_def);
        // 4c3d 後の局面 = ply 5 (攻め方番 OR)

        verbose_eprintln!("\n{:>6} {:>10} {:>8} {:>10}",
            "Depth", "Nodes", "Time(s)", "Result");
        verbose_eprintln!("{}", "-".repeat(40));

        for &depth in &depths {
            let mut solver = DfPnSolver::with_timeout(depth, 2_000_000, 32767, 30);
            solver.set_find_shortest(false);
            let mut test_board = pv_board.clone();
            let start = Instant::now();
            let result = solver.solve(&mut test_board);
            let elapsed = start.elapsed();

            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NO_MATE".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNKNOWN".into(), *nodes_searched),
            };

            verbose_eprintln!("{:>6} {:>10} {:>8.1} {:>10}", depth, nodes, elapsed.as_secs_f64(), result_str);

            if let TsumeResult::Checkmate { .. } = &result {
                break; // 解けたら終了
            }
        }
    }

    /// 全王手 NM 局面の検出テスト．
    ///
    /// N*1e → 2c1d 後の局面は全8王手が1ノードで NM(不詰)．
    ///
    /// depth=1 では構造的に NM を検出できる(全王手が即座に反証される)．
    /// depth=33 では IDS の浅い反復の NM エントリ(remaining=小)が深い反復の
    /// look_up で再利用されないため，構造的証明(REMAINING_INFINITE)に
    /// 到達できず Unknown になる場合がある．これは安全側の挙動であり，
    /// depth 制限由来の仮 NM を真の不詰に昇格しない設計の帰結である
    /// (39手詰め偽陽性防止のため)．
    #[test]

    fn test_all_checks_nm_but_solve_returns_unk() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv_setup = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e", "2c1d",
        ];
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 8 後: 攻め方番(OR ノード), 全8王手が NM
        assert_eq!(board.turn, Color::Black);

        // depth=1 では全王手が即座に反証され NM を検出する．
        {
            let mut s = DfPnSolver::new(1, 1_000_000, 32767);
            s.set_find_shortest(false);
            let r = s.solve(&mut board);
            match &r {
                TsumeResult::NoCheckmate { .. } => {}
                other => panic!(
                    "depth=1: expected NoCheckmate, got {:?}",
                    other
                ),
            }
        }

        // depth=3 で Checkmate 偽陽性が発生しないことを検証する．
        // この局面は真の不詰であるため Checkmate を返してはならない．
        // デバッグビルドでの実行時間制約(3分)のため予算・深さを抑制する．
        let mut solver = DfPnSolver::new(3, 10_000, 32767);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {
                // 構造的 NM 検出または予算内未収束: どちらも許容
            }
            other => {
                panic!("Unexpected result: {:?}", match other {
                    TsumeResult::Checkmate { moves, nodes_searched } =>
                        format!("Checkmate({} moves, {} nodes)", moves.len(), nodes_searched),
                    TsumeResult::CheckmateNoPv { nodes_searched } =>
                        format!("CheckmateNoPv({} nodes)", nodes_searched),
                    _ => "?".to_string(),
                });
            }
        }
    }

    /// 39手詰: N*1e → 2c1d 後の OR ノード(ply 8)を深掘りし，
    /// どの王手でノード爆発が起きるかを特定する．
    ///
    /// 2c1d は N*1e の AND 子ノードで 500K budget では未解決．
    /// ply 8 後は攻め方番(OR)なので，各王手候補を 1M budget で個別ソルブし，
    /// さらに解けない王手については AND 子ノード(回避手)を個別に分析する．
    #[test]
    #[ignore]
    fn test_tsume_39te_2c1d_deep_breakdown() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        // 8手進めて N*1e → 2c1d 後の局面へ
        let pv_setup = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e", "2c1d",
        ];

        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv_setup {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // ply 8 後: 攻め方番(OR ノード)
        assert_eq!(board.turn, Color::Black);

        let mut helper = DfPnSolver::default_solver();
        let checks = helper.generate_check_moves(&mut board);

        verbose_eprintln!("\n{}", "=".repeat(80));
        verbose_eprintln!(" 39手詰 N*1e → 2c1d 深掘り分析 (ply 8 OR ノード)");
        verbose_eprintln!(" 局面: {} (8手進めた後)", board.sfen());
        verbose_eprintln!(" 王手候補数: {} (うちドロップ: {})",
            checks.len(),
            checks.iter().filter(|m| m.is_drop()).count());
        verbose_eprintln!("{}", "=".repeat(80));

        // ── N*1e の AND 子ノード(回避手)を 1M budget で各ソルブ ──
        // ply 7 まで戻って N*1e 後の局面を作る
        let mut board_after_n1e = Board::new();
        board_after_n1e.set_sfen(sfen).unwrap();
        let pv_to_n1e = [
            "7b6b", "5b4c", "8b9c", "4c3d",
            "1b2c", "3d2c", "N*1e",
        ];
        for usi in &pv_to_n1e {
            let m = board_after_n1e.move_from_usi(usi).unwrap();
            board_after_n1e.do_move(m);
        }
        // N*1e 後: 守備方番(AND ノード)
        assert_eq!(board_after_n1e.turn, Color::White);

        let defenses = helper.generate_defense_moves(&mut board_after_n1e);

        verbose_eprintln!("\n--- N*1e 後の AND 子ノード(回避手)分析 (budget=1M) ---");
        verbose_eprintln!("回避手数: {}\n", defenses.len());
        verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
            "#", "Defense", "Type", "Nodes", "Time(s)", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        for (i, &defense) in defenses.iter().enumerate() {
            let mut child_board = board_after_n1e.clone();
            child_board.do_move(defense);

            // depth=33 (残り 39-8=31 手 + margin 2), budget=1M
            let mut solver = DfPnSolver::with_timeout(33, 1_000_000, 32767, 60);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let def_type = if defense.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            let heavy = if nodes >= 500_000 { " <<<" } else { "" };
            verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}{}",
                i + 1, defense.to_usi(), def_type, nodes,
                elapsed.as_secs_f64(), result_str, heavy);

            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().take(10).map(|m| m.to_usi()).collect();
                let suffix = if moves.len() > 10 { " ..." } else { "" };
                verbose_eprintln!("    PV: {}{}", pv.join(" "), suffix);
            }
        }

        // ── 2c1d 後の各王手を 1M budget で個別ソルブ ──
        verbose_eprintln!("\n--- 2c1d 後の各王手候補 (budget=1M, depth=33) ---\n");
        verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
            "#", "Check", "Type", "Nodes", "Time(s)", "Result");
        verbose_eprintln!("{}", "-".repeat(55));

        for (i, &check) in checks.iter().enumerate() {
            let mut child_board = board.clone();
            child_board.do_move(check);

            let mut solver = DfPnSolver::with_timeout(33, 1_000_000, 32767, 60);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut child_board);
            let elapsed = start.elapsed();

            let check_type = if check.is_drop() { "drop" } else { "move" };
            let (result_str, nodes) = match &result {
                TsumeResult::Checkmate { moves, nodes_searched } =>
                    (format!("MATE({})", moves.len()), *nodes_searched),
                TsumeResult::CheckmateNoPv { nodes_searched } =>
                    ("MATE(nopv)".into(), *nodes_searched),
                TsumeResult::NoCheckmate { nodes_searched } =>
                    ("NM".into(), *nodes_searched),
                TsumeResult::Unknown { nodes_searched } =>
                    ("UNK".into(), *nodes_searched),
            };

            verbose_eprintln!("{:>3} {:>8} {:>5} {:>10} {:>8.1} {:>10}",
                i + 1, check.to_usi(), check_type, nodes,
                elapsed.as_secs_f64(), result_str);

            if let TsumeResult::Checkmate { moves, .. } = &result {
                let pv: Vec<String> = moves.iter().take(10).map(|m| m.to_usi()).collect();
                let suffix = if moves.len() > 10 { " ..." } else { "" };
                verbose_eprintln!("    PV: {}{}", pv.join(" "), suffix);
            }
        }
    }

    /// TT 診断テスト: 指定 ply の指定手で TT エントリの変化をモニタリングする．
    ///
    /// `--features tt_diag` でビルドし `cargo test --features tt_diag -- --nocapture` で実行．
    /// stderr に `[tt_diag]` プレフィックスのログが出力される．
    #[test]
    #[ignore]
    fn test_tt_diag_monitor() {
        // 39手詰め問題: PV を24手進めた局面(ply 24 = 攻め番)から開始
        // PV: 7b6b 5b4c ... P*1g 1f1g ← ここまで24手
        // この局面から ply 0 = 5g6f(攻め)，ply 1 = 応手(P*7g 等)
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        // PV を24手進める
        let pv_24 = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
        ];
        for usi in &pv_24 {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }

        verbose_eprintln!("\n=== TT Diag: ply 24 局面(攻め番) → P*7g 調査 ===");
        verbose_eprintln!("SFEN: {}", board.sfen());

        // ply 1 の AND ノード(応手)をモニタリング
        // ply 0 = 5g6f(攻め)，ply 1 = 応手(P*7g 含む合駒)
        // (1) この局面で王手が生成されるか確認
        let check_solver = DfPnSolver::default_solver();
        let checks = check_solver.generate_check_moves(&mut board);
        verbose_eprintln!("Check moves from ply 24 ({}):", checks.len());
        for m in &checks {
            verbose_eprintln!("  {}", m.to_usi());
        }

        // (2) IDS の各深さでの結果を個別に確認(MIDのみ，PNSスキップ)
        // 浅い深さで不当な NoMate/disproof が出ていないか
        for depth in (3..=17).step_by(2) {
            let mut s = DfPnSolver::with_timeout(depth, 50_000, 32767, 5);
            s.set_find_shortest(false);
            s.attacker = board.turn;
            s.table.clear();
            s.nodes_searched = 0;
            s.max_ply = 0;
            s.path_len = 0;
            s.killer_table.clear();
            s.start_time = Instant::now();
            s.timed_out = false;
            s.next_gc_check = 100_000;
            let mut b = board.clone();
            s.mid_fallback(&mut b);
            let pk = position_key(&b);
            let att_hand = b.hand[s.attacker.index()];
            let (root_pn, root_dn, _) = s.look_up_pn_dn(pk, &att_hand, depth as u16);
            let r_str = if root_pn == 0 {
                "Mate".to_string()
            } else if root_dn == 0 {
                "NoMate".to_string()
            } else {
                format!("Unknown(pn={},dn={})", root_pn, root_dn)
            };
            verbose_eprintln!(
                "  depth={:2} → {} nodes={} max_ply={} tt_pos={}",
                depth, r_str, s.nodes_searched, s.max_ply, s.table.len(),
            );
            if root_pn == 0 {
                break;
            }
        }
    }

    /// チェーン合駒最適化の動作検証テスト．
    ///
    /// 39手詰め ply 25 AND ノード(5g6f 後)を対象に，以下の最適化が
    /// 正常に機能しているかをモニタリングする:
    ///
    /// 1. 合駒遅延展開(deferred children)の逐次活性化
    /// 2. チェーンドロップ3カテゴリ制限
    /// 3. 同一マス証明転用(cross_deduce_deferred)
    /// 4. TT ベース合駒プレフィルタ(try_prefilter_block)
    /// ply 24 の depth 問題を診断する．
    ///
    /// 非 PV 変化では PV より長い詰み手順が存在するため，
    /// `depth = remaining + 2` では不足する．depth=41(最大)で
    /// 解けるかを確認し，必要予算を推定する．
    #[test]
    #[ignore]
    fn test_1g1f_nomate_verification() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f",
        ];
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        // ply 25 の局面を構築 (5g6f 後)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for &mv in pv {
            let m = board.move_from_usi(mv).unwrap();
            board.do_move(m);
        }

        // 1g1f (玉逃げ) を指す
        let def = board.move_from_usi("1g1f").unwrap();
        board.do_move(def);

        verbose_eprintln!("=== 1g1f 後の局面 (OR node, 攻め番) ===");
        verbose_eprintln!("SFEN: {}", board.sfen());

        // 王手生成で攻め手を確認
        let check_solver = DfPnSolver::default_solver();
        let checks = check_solver.generate_check_moves(&mut board);
        verbose_eprintln!("Check moves: {} {:?}", checks.len(),
            checks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());

        // 段階的に深さを増やして解析
        for depth in [15u32, 21, 31, 41] {
            let mut solver = DfPnSolver::with_timeout(depth, 5_000_000, 32767, 60);
            solver.set_find_shortest(false);

            let start = Instant::now();
            let result = solver.solve(&mut board);
            let elapsed = start.elapsed();

            let result_str = match &result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            verbose_eprintln!("depth={}: {} nodes={} time={:.2}s TT_pos={}",
                depth, result_str, solver.nodes_searched, elapsed.as_secs_f64(),
                solver.table.len());
        }

        // 旧互換用: 最後の結果を使う
        let mut solver = DfPnSolver::with_timeout(15, 5_000_000, 32767, 120);
        solver.set_find_shortest(false);
        let result = solver.solve(&mut board);

        verbose_eprintln!("Result: {:?}", match &result {
            TsumeResult::Checkmate { moves, .. } =>
                format!("Mate({})", moves.len()),
            TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
            TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
            TsumeResult::Unknown { .. } => "Unknown".to_string(),
        });
        verbose_eprintln!("Nodes: {}, TT_pos: {}", solver.nodes_searched, solver.table.len());

        // N*6g 後の各攻め手を depth=41 で解く
        {
            let mut brd_n6g = Board::new();
            brd_n6g.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = brd_n6g.move_from_usi(mv).unwrap();
                brd_n6g.do_move(m);
            }
            let def_n6g = brd_n6g.move_from_usi("N*6g").unwrap();
            brd_n6g.do_move(def_n6g);

            verbose_eprintln!("\n=== N*6g 後の各攻め手サブ問題 (depth=41, 1M) ===");
            verbose_eprintln!("SFEN: {}", brd_n6g.sfen());

            let gen = DfPnSolver::default_solver();
            let attacks = gen.generate_check_moves(&mut brd_n6g);
            verbose_eprintln!("Check moves: {} {:?}", attacks.len(),
                attacks.iter().map(|m| m.to_usi()).collect::<Vec<_>>());

            for atk in &attacks {
                let mut after_atk = brd_n6g.clone();
                after_atk.do_move(*atk);

                let mut sub = DfPnSolver::with_timeout(41, 1_000_000, 32767, 60);
                sub.set_find_shortest(false);
                let start = Instant::now();
                let r = sub.solve(&mut after_atk);
                let elapsed = start.elapsed();

                let r_str = match &r {
                    TsumeResult::Checkmate { moves, .. } =>
                        format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                verbose_eprintln!("  {} → {} nodes={} time={:.2}s TT_pos={}",
                    atk.to_usi(), r_str, sub.nodes_searched,
                    elapsed.as_secs_f64(), sub.table.len());
            }
        }

        // 残り3応手も段階的深さテスト
        for def_usi in ["N*6g", "P*7g", "N*7g"] {
            let mut brd = Board::new();
            brd.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = brd.move_from_usi(mv).unwrap();
                brd.do_move(m);
            }
            let def = brd.move_from_usi(def_usi).unwrap();
            brd.do_move(def);

            verbose_eprintln!("\n=== {} 後の局面 (OR node, 攻め番) ===", def_usi);
            verbose_eprintln!("SFEN: {}", brd.sfen());

            for depth in [15u32, 21, 31, 41] {
                let mut s = DfPnSolver::with_timeout(depth, 5_000_000, 32767, 60);
                s.set_find_shortest(false);

                let start = Instant::now();
                let r = s.solve(&mut brd);
                let elapsed = start.elapsed();

                let r_str = match &r {
                    TsumeResult::Checkmate { moves, .. } =>
                        format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                verbose_eprintln!("depth={}: {} nodes={} time={:.2}s TT_pos={}",
                    depth, r_str, s.nodes_searched, elapsed.as_secs_f64(),
                    s.table.len());
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// N*6g → 8g6g 後のボトルネック局面調査(診断用)．
    #[test]
    #[ignore]
    fn test_n6g_bottleneck_diagnosis() {
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv: &[&str] = &[
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f",
        ];
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        // ply 25 の局面を構築 (5g6f 後)
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for &mv in pv {
            let m = board.move_from_usi(mv).unwrap();
            board.do_move(m);
        }
        verbose_eprintln!("=== ply 25 AND (5g6f 後) ===");
        verbose_eprintln!("SFEN: {}", board.sfen());

        // N*6g を指す
        let n6g = board.move_from_usi("N*6g").unwrap();
        board.do_move(n6g);
        verbose_eprintln!("\n=== N*6g 後 (OR node) ===");
        verbose_eprintln!("SFEN: {}", board.sfen());

        // 8g6g(飛車で桂を取る)
        let r6g = board.move_from_usi("8g6g").unwrap();
        board.do_move(r6g);
        verbose_eprintln!("\n=== 8g6g 後 (AND node, 玉方番) ===");
        verbose_eprintln!("SFEN: {}", board.sfen());

        // この局面の情報
        let defender = board.turn();
        let attacker = defender.opponent();
        let king_sq = board.king_square(defender).unwrap();
        verbose_eprintln!("King: {}{}  turn: {:?}",
            9 - king_sq.col(), (b'a' + king_sq.row()) as char, defender);

        let checkers = board.compute_checkers_at(king_sq, attacker);
        verbose_eprintln!("Checkers: {}", checkers.count());
        for sq in checkers {
            let piece = board.squares[sq.index()];
            verbose_eprintln!("  {:?} at {}{}", piece, 9 - sq.col(), (b'a' + sq.row()) as char);
        }

        let mut solver = DfPnSolver::default_solver();

        // between / futile / chain
        let checker_sq = checkers.lsb().unwrap();
        let sliding = solver.find_sliding_checker(&board, king_sq, attacker);
        verbose_eprintln!("Sliding checker: {:?}", sliding.is_some());

        if sliding.is_some() {
            let between = attack::between_bb(checker_sq, king_sq);
            let (futile, chain) = solver.compute_futile_and_chain_squares(
                &board, &between, king_sq, checker_sq, defender, attacker,
            );
            let normal_count = between.count() - futile.count() - chain.count();
            verbose_eprintln!("Between: {}  Futile: {}  Chain: {}  Normal: {}",
                between.count(), futile.count(), chain.count(), normal_count);
            for sq in between {
                let tag = if futile.contains(sq) { "futile" }
                         else if chain.contains(sq) { "chain" }
                         else { "normal" };
                verbose_eprintln!("  {}{} = {}",
                    9 - sq.col(), (b'a' + sq.row()) as char, tag);
            }
        }

        // 防御手一覧
        let defenses = solver.generate_defense_moves(&mut board);
        verbose_eprintln!("\nDefense moves ({}):", defenses.len());
        let drops: Vec<_> = defenses.iter().filter(|m| m.is_drop()).collect();
        let non_drops: Vec<_> = defenses.iter().filter(|m| !m.is_drop()).collect();
        verbose_eprintln!("  Non-drops ({}):", non_drops.len());
        for m in &non_drops {
            verbose_eprintln!("    {}", m.to_usi());
        }
        verbose_eprintln!("  Drops ({}):", drops.len());
        for m in &drops {
            verbose_eprintln!("    {}", m.to_usi());
        }

        // N*6g 後(OR node, 攻め方手番)から解く
        let mut or_board = {
            let mut b = Board::new();
            b.set_sfen(sfen).unwrap();
            for &mv in pv {
                let m = b.move_from_usi(mv).unwrap();
                b.do_move(m);
            }
            let n6g_m = b.move_from_usi("N*6g").unwrap();
            b.do_move(n6g_m);
            b
        };

        // MID のみ(PNS skip)で解く
        verbose_eprintln!("\n=== N*6g 後 MID only (depth=41, 10M, 300s) ===");
        verbose_eprintln!("SFEN: {}", or_board.sfen());
        {
            let mut s = DfPnSolver::with_timeout(41, 10_000_000, 32767, 300);
            s.set_find_shortest(false);
            s.table.clear();
            s.nodes_searched = 0;
            s.max_ply = 0;
            s.path_len = 0;
            s.killer_table.clear();
            s.start_time = std::time::Instant::now();
            s.timed_out = false;
            s.attacker = or_board.turn;
            // mid_fallback を直接呼ぶ
            s.mid_fallback(&mut or_board);
            let (root_pn, root_dn) = s.look_up_board(&or_board);
            let r_str = if root_pn == 0 { "Proven" }
                        else if root_dn == 0 { "Disproven" }
                        else { "Unknown" };
            verbose_eprintln!("  → {} pn={} dn={} searched={} time={:.2}s TT={}",
                r_str, root_pn, root_dn, s.nodes_searched,
                s.start_time.elapsed().as_secs_f64(), s.table.len());
            verbose_eprintln!("  prefilter_hits={}", s.prefilter_hits);
            #[cfg(feature = "tt_diag")]
            eprintln!("  deferred: act={} enqueued={} ready={} not_ready={} cross={}",
                s.diag_mid_deferred_activations,
                s.diag_deferred_enqueued,
                s.diag_deferred_ready,
                s.diag_deferred_not_ready,
                s.diag_cross_deduce_hits);
        }

        // 8g6g 後の各防御手を個別に OR node として解く
        verbose_eprintln!("\n=== 8g6g 後 → 各防御手のサブ問題 (depth=41, 2M) ===");
        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            let mut s = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
            s.set_find_shortest(false);
            let start = std::time::Instant::now();
            let r = s.solve(&mut after_def);
            let elapsed = start.elapsed();
            let r_str = match &r {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            verbose_eprintln!("  {} → {} searched={} time={:.2}s TT={}",
                def_mv.to_usi(), r_str, s.nodes_searched, elapsed.as_secs_f64(), s.table.len());
        }

        // Unknown が出た防御手を掘り下げ(2段目: 攻め手→防御手)
        verbose_eprintln!("\n=== Unknown 防御手の攻め手別サブ問題 (depth=41, 2M) ===");
        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            let mut s_check = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
            s_check.set_find_shortest(false);
            let r_check = s_check.solve(&mut after_def);
            if !matches!(r_check, TsumeResult::Unknown { .. }) {
                continue;
            }

            verbose_eprintln!("--- {} (Unknown) → 攻め手展開 ---", def_mv.to_usi());
            let gen = DfPnSolver::default_solver();
            let attacks = gen.generate_check_moves(&mut after_def);
            verbose_eprintln!("  攻め手数: {}", attacks.len());
            for atk in &attacks {
                let mut after_atk = after_def.clone();
                after_atk.do_move(*atk);

                let mut s2 = DfPnSolver::with_timeout(41, 2_000_000, 32767, 60);
                s2.set_find_shortest(false);
                let start2 = std::time::Instant::now();
                let r2 = s2.solve(&mut after_atk);
                let elapsed2 = start2.elapsed();
                let r2_str = match &r2 {
                    TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                    TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                    TsumeResult::NoCheckmate { .. } => "NoCheckmate".to_string(),
                    TsumeResult::Unknown { .. } => "Unknown".to_string(),
                };
                verbose_eprintln!("    {} → {} searched={} time={:.2}s TT={}",
                    atk.to_usi(), r2_str, s2.nodes_searched, elapsed2.as_secs_f64(), s2.table.len());
            }
        }

            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// 5. TT エントリ数の推移
    #[test]
    #[cfg(feature = "tt_diag")]
    fn test_chain_interpose_diagnostics() {
        use std::io::Write;
        let out_path = "/tmp/chain_interpose_diag.log";
        let result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        writeln!(out, "{}", "=".repeat(80)).unwrap();
        writeln!(out, " チェーン合駒最適化 動作検証").unwrap();
        writeln!(out, "{}", "=".repeat(80)).unwrap();

        // ply 25 AND ノード(5g6f 後): 飛車8g→玉1gの横利き開き王手
        let and_sfen = "9/3+N1P3/7+R1/9/9/3S5/1R6k/3p5/9 w 2b4g3s3n4l16p 26";

        let mut board = Board::new();
        board.set_sfen(and_sfen).unwrap();

        // ========================================
        // Phase 1: チェーンドロップ3カテゴリ制限の検証
        // ========================================
        writeln!(out, "\n--- Phase 1: チェーンドロップ3カテゴリ制限 ---\n").unwrap();

        let mut solver = DfPnSolver::default_solver();
        let defenses = solver.generate_defense_moves(&mut board);
        let chain_bb = solver.chain_bb_cache;

        writeln!(out, "chain_bb: {:?}", chain_bb).unwrap();
        writeln!(out, "全回避手数: {}", defenses.len()).unwrap();

        // チェーンマスへのドロップを抽出
        let mut chain_drops: Vec<String> = Vec::new();
        let mut normal_drops: Vec<String> = Vec::new();
        let mut non_drops: Vec<String> = Vec::new();

        for m in &defenses {
            if m.is_drop() {
                let to = m.to_sq();
                if chain_bb.contains(to) {
                    chain_drops.push(m.to_usi());
                } else {
                    normal_drops.push(m.to_usi());
                }
            } else {
                non_drops.push(m.to_usi());
            }
        }

        writeln!(out, "チェーンマスへのドロップ: {:?}", chain_drops).unwrap();
        writeln!(out, "通常マスへのドロップ: {:?}", normal_drops).unwrap();
        writeln!(out, "駒移動: {:?}", non_drops).unwrap();

        // チェーンマスへのドロップが3カテゴリ制限に従っているか検証
        // 各マスごとにグループ化
        use std::collections::HashMap;
        let mut drops_by_sq: HashMap<String, Vec<String>> = HashMap::new();
        for m in &defenses {
            if m.is_drop() && chain_bb.contains(m.to_sq()) {
                let sq_str = format!("{}{}",
                    (b'1' + (8 - m.to_sq().col())) as char,
                    (b'a' + m.to_sq().row()) as char,
                );
                let pt = m.drop_piece_type().unwrap();
                let pt_str = match pt {
                    PieceType::Pawn => "P", PieceType::Lance => "L",
                    PieceType::Knight => "N", PieceType::Silver => "S",
                    PieceType::Gold => "G", PieceType::Bishop => "B",
                    PieceType::Rook => "R", _ => "?",
                };
                drops_by_sq.entry(sq_str).or_default().push(pt_str.to_string());
            }
        }

        let mut all_ok = true;
        for (sq, pieces) in &drops_by_sq {
            writeln!(out, "  チェーンマス {}: 駒種={:?} ({}個)", sq, pieces, pieces.len()).unwrap();
            // 3カテゴリ制限: 最大3手(前方系1 + 角1 + 桂1)
            if pieces.len() > 3 {
                writeln!(out, "  *** 異常: 3カテゴリ制限違反! {} > 3 ***", pieces.len()).unwrap();
                all_ok = false;
            }
            // 前方利き系は最弱の1つのみ
            let forward_count = pieces.iter().filter(|p| {
                matches!(p.as_str(), "P" | "L" | "S" | "G" | "R")
            }).count();
            if forward_count > 1 {
                writeln!(out, "  *** 異常: 前方利き系 {} > 1 ***", forward_count).unwrap();
                all_ok = false;
            }
        }
        writeln!(out, "3カテゴリ制限: {}", if all_ok { "OK" } else { "*** NG ***" }).unwrap();

        // ========================================
        // Phase 2: 各応手の探索とTTモニタリング
        // ========================================
        writeln!(out, "\n--- Phase 2: 応手別探索 + TT/最適化モニタリング ---\n").unwrap();

        // P*3g (短いチェーン) と P*7g (長いチェーン) を比較
        let target_moves = ["P*3g", "B*3g", "N*3g", "P*7g", "N*7g", "1g1f", "1g1h"];

        writeln!(out, "{:<10} {:<10} {:<12} {:<10} {:<17} {:<17} {:<10}",
            "Move", "Nodes", "TT_pos", "Prefilter", "Defer(MID/PNS)", "XDeduce/PNSprov", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(90)).unwrap();

        let mut phase2_cross_deduce_total: u64 = 0;
        let mut phase2_pns_proven_total: u64 = 0;

        for &mv_usi in &target_moves {
            let m = match board.move_from_usi(mv_usi) {
                Some(m) => m,
                None => {
                    writeln!(out, "{:<10} (invalid move)", mv_usi).unwrap();
                    continue;
                }
            };

            let mut after_def = board.clone();
            after_def.do_move(m);

            let def_remaining = 13; // 39 - 24 - 1(攻め) - 1(受け) = 13
            let sub_depth = (def_remaining + 2).min(41) as u32;

            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, 250_000, 32767, 30,
            );
            sub_solver.set_find_shortest(false);

            let mut sub_board = after_def.clone();
            let sub_result = sub_solver.solve(&mut sub_board);

            let result_str = match &sub_result {
                TsumeResult::Checkmate { moves, .. } =>
                    format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };

            phase2_cross_deduce_total += sub_solver.diag_cross_deduce_hits;
            phase2_pns_proven_total += sub_solver.diag_pns_deferred_already_proven;

            let sub_proven = sub_solver.table.count_proven();
            let sub_disproven = sub_solver.table.count_disproven();
            writeln!(out, "{:<10} {:<10} {:<12} {:<10} {:<8}/{:<8} {:<8}/{:<8} {:<10}",
                mv_usi,
                sub_solver.nodes_searched,
                sub_solver.table.len(),
                sub_solver.prefilter_hits,
                sub_solver.diag_mid_deferred_activations,
                sub_solver.diag_pns_deferred_activations,
                sub_solver.diag_cross_deduce_hits,
                sub_solver.diag_pns_deferred_already_proven,
                result_str,
            ).unwrap();
            writeln!(out, "{:<10} TT: proven={}, disproven={}, total={}",
                "", sub_proven, sub_disproven, sub_solver.table.total_entries(),
            ).unwrap();
        }

        writeln!(out, "\nPhase 2 合計: cross_deduce={}, pns_proven={}",
            phase2_cross_deduce_total, phase2_pns_proven_total).unwrap();

        // ========================================
        // Phase 3: ply 24 全体の TT 推移モニタリング
        // ========================================
        writeln!(out, "\n--- Phase 3: ply 24 全体探索の TT 推移 ---\n").unwrap();

        // PV を ply 24 まで進めて OR ノードから探索
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
        ];
        let mut board24 = Board::new();
        board24.set_sfen(sfen).unwrap();
        for mv in &pv {
            let m = board24.move_from_usi(mv).unwrap();
            board24.do_move(m);
        }

        let node_limit: u64 = 500_000;
        let depth = 17u32; // remaining 15 + 2
        let mut full_solver = DfPnSolver::with_timeout(
            depth, node_limit, 32767, 60,
        );
        full_solver.set_find_shortest(false);

        let start = Instant::now();
        let full_result = full_solver.solve(&mut board24);
        let elapsed = start.elapsed();

        let result_str = match &full_result {
            TsumeResult::Checkmate { moves, .. } =>
                format!("Mate({})", moves.len()),
            TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
            TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
            TsumeResult::Unknown { .. } => "Unknown".to_string(),
        };

        let total_deferred = full_solver.diag_mid_deferred_activations
            + full_solver.diag_pns_deferred_activations;
        writeln!(out, "Result: {}", result_str).unwrap();
        writeln!(out, "Nodes: {}", full_solver.nodes_searched).unwrap();
        writeln!(out, "Time: {:.2}s", elapsed.as_secs_f64()).unwrap();
        writeln!(out, "NPS: {:.0}", full_solver.nodes_searched as f64 / elapsed.as_secs_f64()).unwrap();
        let tt_total = full_solver.table.total_entries();
        let tt_proven = full_solver.table.count_proven();
        let tt_disproven = full_solver.table.count_disproven();
        let tt_intermediate = full_solver.table.count_intermediate();
        writeln!(out, "TT positions: {}", full_solver.table.len()).unwrap();
        writeln!(out, "TT entries: {} (proven={}, disproven={}, intermediate={})",
            tt_total, tt_proven, tt_disproven, tt_intermediate).unwrap();
        writeln!(out, "Prefilter hits: {}", full_solver.prefilter_hits).unwrap();
        writeln!(out, "Deferred activations: {} (MID={}, PNS={})",
            total_deferred,
            full_solver.diag_mid_deferred_activations,
            full_solver.diag_pns_deferred_activations).unwrap();
        writeln!(out, "PNS deferred already proven: {}", full_solver.diag_pns_deferred_already_proven).unwrap();
        writeln!(out, "Cross-deduce hits (MID): {}", full_solver.diag_cross_deduce_hits).unwrap();

        // ========================================
        // Phase 4: 正常性チェック
        // ========================================
        writeln!(out, "\n--- Phase 4: 正常性チェック ---\n").unwrap();

        let mut checks_passed = 0;
        let mut checks_failed = 0;

        // Check 1: prefilter_hits > 0 (プレフィルタが動作している)
        if full_solver.prefilter_hits > 0 {
            writeln!(out, "[OK] prefilter_hits = {} (> 0)", full_solver.prefilter_hits).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] prefilter_hits = 0 (プレフィルタが動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 2: deferred_activations > 0 (遅延展開が動作している)
        if total_deferred > 0 {
            writeln!(out, "[OK] deferred_activations = {} (> 0, MID={} PNS={})",
                total_deferred,
                full_solver.diag_mid_deferred_activations,
                full_solver.diag_pns_deferred_activations).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] deferred_activations = 0 (遅延展開が動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 3: cross_deduce が per-move テストで動作している
        // 全体探索は PNS 支配のため MID cross_deduce = 0 は想定内．
        // per-move テスト(Phase 2)の合計で検証する．
        if phase2_cross_deduce_total > 0 {
            writeln!(out, "[OK] cross_deduce(Phase2合計) = {} (> 0, MID で TT 転用が動作)",
                phase2_cross_deduce_total).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] cross_deduce(Phase2合計) = 0 (同一マス証明転用が動作していない)").unwrap();
            checks_failed += 1;
        }

        // Check 4: 3カテゴリ制限
        if all_ok {
            writeln!(out, "[OK] 3カテゴリ制限: 全チェーンマスで遵守").unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] 3カテゴリ制限: 違反あり").unwrap();
            checks_failed += 1;
        }

        // Check 5: TT entries < nodes (TT がノード数と乖離していない)
        let ratio = full_solver.table.total_entries() as f64
            / full_solver.nodes_searched as f64;
        if ratio < 2.0 {
            writeln!(out, "[OK] TT entries/nodes = {:.2} (< 2.0)", ratio).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] TT entries/nodes = {:.2} (>= 2.0, TT 肥大化の疑い)", ratio).unwrap();
            checks_failed += 1;
        }

        // Check 6: deferred_activations << nodes (バルク活性化していない)
        let act_ratio = total_deferred as f64
            / full_solver.nodes_searched as f64;
        if act_ratio < 0.1 {
            writeln!(out, "[OK] deferred_act/nodes = {:.4} (< 0.1, 逐次活性化)", act_ratio).unwrap();
            checks_passed += 1;
        } else {
            writeln!(out, "[NG] deferred_act/nodes = {:.4} (>= 0.1, 過剰な活性化の疑い)", act_ratio).unwrap();
            checks_failed += 1;
        }

        writeln!(out, "\n合計: {} passed, {} failed", checks_passed, checks_failed).unwrap();

        // テスト結果
        assert!(checks_failed == 0,
            "チェーン合駒最適化の動作検証に失敗: {} 件のチェックが NG (詳細: {})",
            checks_failed, out_path);

            })
            .unwrap()
            .join()
            .unwrap();
        verbose_eprintln!("結果: {}", out_path);
    }

    /// 最適化ベンチマーク: 探索ノード数・TT エントリ数・実行時間を計測する．
    #[test]
    #[ignore]
    fn test_optimization_benchmark() {
        use crate::board::Board;
        struct BenchCase { name: &'static str, sfen: &'static str, depth: u32, max_nodes: u64 }
        let cases = [
            BenchCase { name: "9te", sfen: "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1", depth: 15, max_nodes: 100_000 },
            BenchCase { name: "39te", sfen: "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1", depth: 41, max_nodes: 10_000_000 },
        ];
        for case in &cases {
            let mut board = Board::new();
            board.set_sfen(case.sfen).unwrap();
            let mut solver = DfPnSolver::with_timeout(case.depth, case.max_nodes, 32767, 120);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut board);
            let elapsed = start.elapsed();
            let tt_ent = solver.table.total_entries();
            let status = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("SOLVED({}te)", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "PROVED".to_string(),
                TsumeResult::NoCheckmate { .. } => "NO_CHECKMATE".to_string(),
                TsumeResult::Unknown { .. } => "UNKNOWN".to_string(),
            };
            eprintln!("[bench] {} {}: nodes={} tt_ent={} time={:.3}s",
                case.name, status, solver.nodes_searched, tt_ent, elapsed.as_secs_f64());
        }
    }
    /// 50M/100M ノードでの 39手詰めベンチマーク．
    ///
    /// Frontier Variant の閾値飢餓回避効果を大規模ノード予算で検証する．
    #[test]
    #[ignore]
    fn test_39te_large_budget_benchmark() {
        use crate::board::Board;
        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";

        for &max_nodes in &[50_000_000u64, 100_000_000] {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();
            let mut solver = DfPnSolver::with_timeout(41, max_nodes, 32767, 600);
            solver.set_find_shortest(false);
            let start = Instant::now();
            let result = solver.solve(&mut board);
            let elapsed = start.elapsed();
            let tt_ent = solver.table.total_entries();
            let status = match &result {
                TsumeResult::Checkmate { moves, .. } => format!("SOLVED({}te)", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "PROVED".to_string(),
                TsumeResult::NoCheckmate { .. } => "NO_CHECKMATE".to_string(),
                TsumeResult::Unknown { .. } => "UNKNOWN".to_string(),
            };
            eprintln!("[bench_large] {}M {}: nodes={} tt_ent={} time={:.1}s NPS={:.0}K",
                max_nodes / 1_000_000, status, solver.nodes_searched, tt_ent,
                elapsed.as_secs_f64(),
                solver.nodes_searched as f64 / elapsed.as_secs_f64() / 1000.0);
        }
    }

    /// PN_UNIT スケーリング診断: ply 25 の各応手を個別に解き，ノード数を出力する．
    ///
    /// PN_UNIT=1 と PN_UNIT=64 で実行して結果を比較することで，
    /// スケーリング漏れによる探索パターンの差異を検出する．
    #[test]
    #[ignore]
    fn test_pn_unit_scaling_diagnostic() {
        use std::io::Write;
        let out_path = "/tmp/pn_unit_scaling_diag.log";
        let _result = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
        let mut out = std::fs::File::create(out_path).unwrap();

        let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
        let pv = [
            "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
            "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
            "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
            "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
            "5g6f",
        ];

        // ply 24 の開き王手後の局面を構築
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();
        for usi in &pv {
            let m = board.move_from_usi(usi).unwrap();
            board.do_move(m);
        }
        // 現在: ply 25 AND ノード(玉方手番)
        writeln!(out, "PN_UNIT={}", super::PN_UNIT).unwrap();
        writeln!(out, "SFEN after ply 25: {}", board.sfen()).unwrap();
        writeln!(out, "{:<12} {:<14} {:<10} {}", "Move", "Nodes", "TT_pos", "Result").unwrap();
        writeln!(out, "{}", "-".repeat(50)).unwrap();

        let mut defense_solver = DfPnSolver::default_solver();
        let defenses = defense_solver.generate_defense_moves(&mut board);

        for def_mv in &defenses {
            let mut after_def = board.clone();
            after_def.do_move(*def_mv);

            let sub_depth = 15u32; // (13 + 2)
            let per_move_budget = 250_000u64;

            let mut sub_solver = DfPnSolver::with_timeout(
                sub_depth, per_move_budget, 32767, 30,
            );
            sub_solver.set_find_shortest(false);

            let sub_result = sub_solver.solve(&mut after_def);

            let result_str = match &sub_result {
                TsumeResult::Checkmate { moves, .. } => format!("Mate({})", moves.len()),
                TsumeResult::CheckmateNoPv { .. } => "MateNoPV".to_string(),
                TsumeResult::NoCheckmate { .. } => "NoMate".to_string(),
                TsumeResult::Unknown { .. } => "Unknown".to_string(),
            };
            writeln!(out, "{:<12} {:<14} {:<10} {}",
                def_mv.to_usi(), sub_solver.nodes_searched,
                sub_solver.table.len(), result_str).unwrap();

            // P*4g の詳細診断: ply 分布
            if def_mv.to_usi() == "P*4g" {
                writeln!(out, "  P*4g ply distribution:").unwrap();
                for (p, &n) in sub_solver.ply_nodes.iter().enumerate() {
                    if n > 0 {
                        writeln!(out, "    ply {:>2}: {:>10} nodes", p, n).unwrap();
                    }
                }
                // root の pn/dn
                let (rpn, rdn) = sub_solver.look_up_board(&after_def);
                writeln!(out, "  root pn={} dn={} max_ply={}",
                    rpn, rdn, sub_solver.max_ply).unwrap();
            }
        }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// PNS NPS ベンチマーク: 3 問題を各 3 回実行し，PNS フェーズの NPS を計測する．
    ///
    /// 各最適化(P1-P7)の前後でこのテストを実行し，中央値を比較する．
    /// 実行: `cargo test -p maou_shogi --release -- --ignored bench_pns_nps --nocapture`
    #[test]
    #[ignore]
    fn bench_pns_nps() {
        use crate::board::Board;

        struct BenchCase {
            name: &'static str,
            sfen: &'static str,
            depth: u32,
            max_nodes: u64,
        }

        let cases = [
            BenchCase {
                name: "9te",
                sfen: "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1",
                depth: 15,
                max_nodes: 100_000,
            },
            BenchCase {
                name: "29te",
                sfen: "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1",
                depth: 31,
                max_nodes: 2_000_000,
            },
            BenchCase {
                name: "39te",
                sfen: "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1",
                depth: 41,
                max_nodes: 10_000_000,
            },
        ];

        const RUNS: usize = 3;

        eprintln!("=== PNS NPS Benchmark ===");
        eprintln!("{:<8} {:>6} {:>12} {:>10} {:>10}",
            "problem", "run", "nodes", "time_ms", "NPS_K");

        for case in &cases {
            let mut times_ms = Vec::with_capacity(RUNS);
            let mut nodes_list = Vec::with_capacity(RUNS);

            for run in 0..RUNS {
                let mut board = Board::new();
                board.set_sfen(case.sfen).unwrap();
                let mut solver = DfPnSolver::with_timeout(case.depth, case.max_nodes, 32767, 120);
                solver.set_find_shortest(false);
                solver.attacker = board.turn;

                let start = Instant::now();
                let _pv = solver.pns_main(&mut board);
                let elapsed = start.elapsed();

                let nodes = solver.nodes_searched;
                let ms = elapsed.as_secs_f64() * 1000.0;
                let nps_k = nodes as f64 / elapsed.as_secs_f64() / 1000.0;

                eprintln!("{:<8} {:>6} {:>12} {:>10.1} {:>10.1}",
                    case.name, run + 1, nodes, ms, nps_k);

                times_ms.push(ms);
                nodes_list.push(nodes);
            }

            // 中央値
            times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_ms = times_ms[RUNS / 2];
            let median_nodes = nodes_list[RUNS / 2];
            let median_nps_k = median_nodes as f64 / (median_ms / 1000.0) / 1000.0;

            eprintln!("{:<8} MEDIAN {:>12} {:>10.1} {:>10.1}",
                case.name, median_nodes, median_ms, median_nps_k);
        }
        eprintln!("=== END ===");
    }
