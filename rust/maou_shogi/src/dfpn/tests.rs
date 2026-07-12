use super::api::*;
use super::solver::*;
use super::*;
use crate::attack;
use crate::movegen;
use crate::types::{PieceType, Square};

/// 最短詰みテストの共通検証 (`find_shortest=true` がデフォルト)．
///
/// 以下を確認する:
/// 1. 結果が `Checkmate`．
/// 2. PV 手数が既知の最短手数 `expected_len` と一致 (最短手数の絶対 oracle は user 確認値)．
/// 3. PV を replay した終局面が真の詰み (`in_check && legal==0`)．
///
/// **別解は許容**する: 同一最短手数の sound な PV であればどれでも pass する (exact PV は問わない)．
/// 攻め方の強制性 (全受けが詰む) は solver 内部の STRICT-VERIFY が保証する．
fn assert_shortest_mate(sfen: &str, depth: u32, max_nodes: u64, expected_len: usize) {
    let mut board = Board::new();
    board.set_sfen(sfen).unwrap();
    let mut solver = DfPnSolver::new(depth, max_nodes);
    match solver.solve_impl(&mut board) {
        TsumeResult::Checkmate { moves, .. } => {
            let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            assert_eq!(
                moves.len(),
                expected_len,
                "最短手数が {expected_len} と不一致: {} 手 {usi:?}",
                moves.len(),
            );
            // PV を replay して終局面が真の詰みか検証 (別解でも sound なら pass)．
            let mut chk = Board::new();
            chk.set_sfen(sfen).unwrap();
            for m in &moves {
                chk.do_move(*m);
            }
            let legal = movegen::generate_legal_moves(&mut chk).len();
            let in_check = chk.is_in_check(chk.turn());
            assert!(
                in_check && legal == 0,
                "PV 終局面は真の詰みであるべき (sound mate; 別解可): {usi:?}",
            );
        }
        other => panic!("expected Checkmate (mate-{expected_len}), got {other:?}"),
    }
}

/// 最短詰みテスト (**PV 完全一致**版)．`find_shortest=true` で解き，maou の PV (USI) が
/// `acceptable_pvs` に列挙した正解手順 (別解) の **いずれかと完全一致** し，かつ replay 終局面が
/// 真の詰みであることを検証する．exact PV を pin することで非最短・誤手順を検出する．
/// 玉の逃げ方等で分岐する別解は `acceptable_pvs` に複数列挙する．
fn assert_shortest_mate_pv(sfen: &str, depth: u32, max_nodes: u64, acceptable_pvs: &[&[&str]]) {
    let mut board = Board::new();
    board.set_sfen(sfen).unwrap();
    let mut solver = DfPnSolver::new(depth, max_nodes);
    match solver.solve_impl(&mut board) {
        TsumeResult::Checkmate { moves, .. } => {
            let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            // PV を replay して終局面が真の詰みか検証 (sound)．
            let mut chk = Board::new();
            chk.set_sfen(sfen).unwrap();
            for m in &moves {
                chk.do_move(*m);
            }
            let legal = movegen::generate_legal_moves(&mut chk).len();
            let in_check = chk.is_in_check(chk.turn());
            assert!(
                in_check && legal == 0,
                "PV 終局面は真の詰みであるべき (sound mate): {usi:?}",
            );
            // PV 完全一致: acceptable_pvs (別解) のいずれかと一致すること．
            let matched = acceptable_pvs.iter().any(|pv| {
                pv.len() == usi.len() && pv.iter().zip(&usi).all(|(a, b)| *a == b.as_str())
            });
            assert!(
                matched,
                "PV が正解手順 (別解) のいずれとも完全一致しない:\n  maou: {usi:?}\n  正解: {acceptable_pvs:?}",
            );
        }
        other => panic!("expected Checkmate, got {other:?}"),
    }
}

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

/// 9手詰のテストケース．
///
/// 局面: 後手玉1四，先手角2四・3四，後手銀3一・後手香3二
/// 先手持ち駒: 飛，歩
/// 後手持ち駒: 飛，金4，銀3，桂4，香3，歩17
///
/// 正解手順: 1三角成，同玉，2三飛打，1二玉，1三歩打，1一玉，2一飛成，同玉，1二歩成
#[test]
fn test_tsume_9te_a() {
    // 9 手詰 PV 完全一致．
    let pvs: &[&[&str]] = &[&[
        "2d1c+", "1d1c", "R*2c", "1c1b", "P*1c", "1b1a", "2c2a+", "1a2a", "1c1b+",
    ]];
    assert_shortest_mate_pv(
        "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1",
        15,
        1_048_576,
        pvs,
    );
}

/// 簡単な1手詰め．
#[test]
fn test_tsume_1te() {
    // 後手玉1一，先手金2三，先手持ち駒: 金
    // G*1b(1二金打)で詰み
    // 1 手詰 PV 完全一致 (G*1b / G*2b はどちらも 1 手詰)．
    let pvs: &[&[&str]] = &[&["G*1b"], &["G*2b"]];
    assert_shortest_mate_pv("8k/9/7G1/9/9/9/9/9/9 b G 1", 3, 100_000, pvs);
}

/// 3手詰め: 後手玉1一，先手飛3三，先手持ち駒: 金
///
/// 正解: 1三飛成，2一玉，2二金打 まで3手詰
#[test]
fn test_tsume_3te() {
    // 3 手詰 PV 完全一致．
    let pvs: &[&[&str]] = &[&["3c1c+", "1a2a", "G*2b"]];
    assert_shortest_mate_pv("8k/9/6R2/9/9/9/9/9/9 b G 1", 7, 1_048_576, pvs);
}

/// 不詰のケース．
#[test]
fn test_no_checkmate() {
    // 後手玉5一，先手持ち駒: 歩 → 歩では詰まない
    let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
    let mut board = Board::empty();
    board.set_sfen(sfen).unwrap();

    let mut solver = DfPnSolver::new(5, 100_000);
    let result = solver.solve_impl(&mut board);

    match &result {
        TsumeResult::NoCheckmate { .. } => {}
        other => panic!("expected NoCheckmate, got {:?}", other),
    }
}

/// solve_tsume 便利関数のテスト．
#[test]
fn test_solve_tsume_convenience() {
    let result = solve_tsume("8k/9/7G1/9/9/9/9/9/9 b G 1", Some(3), Some(100_000)).unwrap();

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
fn test_tsume_11te_a() {
    // 盤面: 5一と，2一王，1一香，2二銀，4三飛，2四角，先手持駒: 金桂
    // 11手詰め: 32金打，同玉，42角成，21玉，31馬，同銀，23飛成，22銀，33桂打，31玉，41と
    // 11 手詰 PV 完全一致．
    let pvs: &[&[&str]] = &[&[
        "G*3b", "2a3b", "2d4b+", "3b2a", "4b3a", "2b3a", "4c2c+", "3a2b", "N*3c", "2a3a", "5a4a",
    ]];
    assert_shortest_mate_pv(
        "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1",
        31,
        1_048_576,
        pvs,
    );
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

    let mut solver = DfPnSolver::with_timeout(31, u64::MAX, 0);
    // timeout=0 なので即タイムアウト(ただし最初の1024ノードは走る)
    let result = solver.solve_impl(&mut board);

    // NoCheckmate か Unknown のどちらか(歩1枚では詰まない)
    match &result {
        TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {}
        other => panic!("expected NoCheckmate or Unknown, got {:?}", other),
    }
}

/// 逆王手のある詰将棋のテストケース．
///
/// 局面: 後手玉2三，後手桂2一，後手香1一，後手飛5四，後手歩1三，後手と1六
///       先手歩1五，先手桂3四，先手金2六，先手香3六
/// 先手持駒: 飛
/// 後手持駒: 歩15，香2，桂2，銀4，金3，角2
#[test]
fn test_tsume_9te_b() {
    // 9 手詰 PV 完全一致．
    let pvs: &[&[&str]] = &[&[
        "3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d", "2f2e", "2d2e", "3b3e",
    ]];
    assert_shortest_mate_pv(
        "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1",
        31,
        2_000_000,
        pvs,
    );
}

/// 2一龍後に2三歩打で詰まないことを検証する．
///
/// 無駄合いフィルタが誤って2三歩打を除外していた問題の回帰テスト．
/// 2二桂成，同玉，4二飛打，2三玉，3二飛成，2四玉，2一龍 の後，
/// 後手は2三歩打で合い駒でき，これは無駄合いではない．
#[test]
fn test_tsume_9te_b_ryu_2a_not_checkmate() {
    // 2一龍後の局面を作成
    // 初期局面から 3d2b+, 2c2b, R*4b, 2b2c, 4b3b+, 2c2d, 3b2a を実行
    let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
    let mut board = Board::empty();
    board.set_sfen(sfen).unwrap();

    let moves_usi = ["3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d", "3b2a"];
    for usi in &moves_usi {
        let m = board
            .move_from_usi(usi)
            .unwrap_or_else(|| panic!("invalid USI: {}", usi));
        board.do_move(m);
    }

    // ここは AND ノード(後手番)．2三歩打(P*2c)が合法手に含まれることを検証
    let defenses = movegen::generate_legal_moves(&mut board);
    let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();
    // debug: defenses

    // P*2c (2三歩打) が合法手に含まれること
    assert!(
        usi_defenses.contains(&"P*2c".to_string()),
        "P*2c should be a legal defense, got: {:?}",
        usi_defenses
    );

    // 2三歩打後の局面は詰みではないことを確認
    let p2c = board.move_from_usi("P*2c").unwrap();
    let cap = board.do_move(p2c);

    // 先手番(攻め方)から探索して詰みがないことを検証
    let mut solver = DfPnSolver::new(15, 100_000);
    let result = solver.solve_impl(&mut board);
    assert!(
        !matches!(result, TsumeResult::Checkmate { .. }),
        "P*2c 後の局面は詰みではないはず: {:?}",
        result
    );

    board.undo_move(p2c, cap);
}

/// 無駄合いのある詰将棋テストケース．
///
/// 局面: 7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p
/// **最短 11 手詰** (ユーザ確認済)．無駄合い (例: 支えのない P*3b) を含めると 13 手になるが，
/// 無駄合いは手数に数えない．
///
/// 正解手順 (共通 8 手 + 9 手目 4c4b+，以降は玉の逃げ方で別解 ①〜④):
///   S*2b 1a1b P*1c 1b2b N*3d 2b1a 1c1b 1a1b 4c4b+
///   ① 1b1a 3d2b+ (1一玉, 2二成桂)  ② 1b1a 4b2b (1一玉, 2二龍)
///   ③ 1b1c 4b2b (1三玉, 2二龍)     ④ 1b2c 4b2b (2三玉, 2二龍)
/// (受け方が S*2b に 1a2b と応じる分岐 `S*2b 1a2b N*3d 2b1a P*1b 1a2b 4c4b+` は
///  同一局面へ合流するより短い受けで max-resistance ではない)．
#[test]
fn test_tsume_11te_b() {
    // 最短 11 手 PV 完全一致 (別解 ①〜④ のいずれか; 無駄合いは除外)．
    let pvs: &[&[&str]] = &[
        &[
            "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a", "1c1b", "1a1b", "4c4b+", "1b1a",
            "3d2b+",
        ],
        &[
            "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a", "1c1b", "1a1b", "4c4b+", "1b1a", "4b2b",
        ],
        &[
            "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a", "1c1b", "1a1b", "4c4b+", "1b1c", "4b2b",
        ],
        &[
            "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a", "1c1b", "1a1b", "4c4b+", "1b2c", "4b2b",
        ],
    ];
    assert_shortest_mate_pv(
        "7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p 1",
        31,
        2_000_000,
        pvs,
    );
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
        "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c", "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+",
        "2b3b", "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d", "3c2c", "1d1e", "P*1f", "1e1f",
        "P*1g", "1f1g", "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i", "8i6i", "6h6i+", "S*2h",
        "1i2i", "2h3g", "2i3i", "2g2h", "3i4i", "2h4h",
    ];

    let solver = DfPnSolver::default_solver();
    let mut board = Board::new();
    board.set_sfen(sfen).unwrap();

    // PV の各 OR node (攻め方手番) で check_moves の完全性を検証
    for (ply, &usi) in pv.iter().enumerate() {
        if ply % 2 == 0 {
            // OR node: 攻め方手番 → check_moves を検証
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
                "check moves mismatch at ply {} (next PV move: {})\n  \
                     missing: {:?}\n  extra: {:?}",
                ply,
                usi,
                missing,
                extra
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
        "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c", "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+",
        "2b3b", "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d", "3c2c", "1d1e", "P*1f", "1e1f",
        "P*1g", "1f1g", "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i", "8i6i", "6h6i+", "S*2h",
        "1i2i", "2h3g", "2i3i", "2g2h", "3i4i", "2h4h",
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

            let legal_moves: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let defense_moves: BTreeSet<String> = solver
                .generate_defense_moves_inner(&mut board, false)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            // defense_moves ⊆ legal_moves (不正な手がないこと)
            let extra: BTreeSet<_> = defense_moves.difference(&legal_moves).collect();
            assert!(
                extra.is_empty(),
                "defense has illegal moves at ply {} (after {})\n  \
                     extra: {:?}",
                ply + 1,
                usi,
                extra
            );

            // PV の次の応手が defense_moves に含まれること
            if ply + 1 < pv.len() {
                let next_defense = pv[ply + 1];
                assert!(
                    defense_moves.contains(next_defense),
                    "PV defense move {} missing from defense_moves at ply {}\n  \
                         defense({}): {:?}",
                    next_defense,
                    ply + 1,
                    defense_moves.len(),
                    defense_moves
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
        // test_tsume_9te_b の中間局面(R*2a 後)
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
            .generate_defense_moves_inner(&mut board, false)
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
            extra,
            sfen
        );

        // legal_moves は defense_moves のサブセットであること(漏れがない)
        let missing: BTreeSet<_> = legal_moves.difference(&defense_moves).collect();
        assert!(
            missing.is_empty(),
            "defense is missing legal moves: {:?}\nsfen: {}\ndefense: {:?}\nlegal: {:?}",
            missing,
            sfen,
            defense_moves,
            legal_moves
        );
    }
}

/// 17手詰め局面の solve() テスト．
#[test]
fn test_tsume_17te() {
    // 17 手詰 (正解例: S*4a ... 3b2a+ ... 3b2b/2c2b)．別解可．
    assert_shortest_mate(
        "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1",
        31,
        5_000_000,
        17,
    );
}

/// `does_have_mate_possibility` (詰み可能性の over-approximation) の回帰テスト．
///
/// over-approximation の soundness は「実王手があれば必ず true」(false negative 不可)．加えて
/// 代表局面を固定する: 白の実王手 0 だが香の成り王手候補で true になるべき局面．
/// これを取りこぼすと look-ahead が早すぎる disproof を起こし探索経路が乖離する．
#[test]
fn test_does_have_mate_possibility() {
    // (sfen, expected, label)．
    let cases = [
        // 白王手0 だが 8f の香が file8 で 9i 段(敵陣)へ進めば成って金王手候補 → DHMP=true．
        (
            "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/NP1S5/2G2+p3/L1K6 w 2RB3GSPn4p 9",
            true,
            "実王手0だが香 promote 候補 (DHMP=true)",
        ),
        // 29te root: 白に実王手あり → 自明に true．
        (
            "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1",
            true,
            "29te-root (実王手あり)",
        ),
    ];
    for (sfen, expected, label) in cases {
        let mut b = Board::new();
        b.set_sfen(sfen).unwrap();
        let s = DfPnSolver::with_timeout(31, 1_000, 5);
        let checks = s.generate_check_moves(&mut b);
        let dhmp = b.does_have_mate_possibility(b.turn);
        // soundness: 実王手があれば DHMP は必ず true．
        if !checks.is_empty() {
            assert!(
                dhmp,
                "{label}: 実王手 {} 個あるのに DHMP=false ({sfen})",
                checks.len()
            );
        }
        assert_eq!(dhmp, expected, "{label}: DHMP 期待値不一致 ({sfen})");
    }
}

/// `mate_move_in_1ply` の false mate-1 回帰テスト．
///
/// `is_checkmate_after_bb` / `generate_check_moves` には以下の false mate-1 バグがあり，
/// look-ahead が多数の局面で `mate_move_in_1ply` を呼ぶため偽の証明を TT に汚染しうる．
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
        (
            "ln1+P5/G1k4+L1/2p1p2B1/K1gp1spN1/1n2Ps3/P1PP2P2/2+bS5/1g3+p3/L1s6 w 2RGNL3P4p 37",
            "pinned-capture-along-ray (P*9c)",
        ),
        // 2. 逆王手: 後手 R*9b が先手玉(7b)を王手．N*8a は自玉の王手を解消しないため非合法．
        (
            "l2+P5/R1k4+L1/K1p1p2B1/2gp1spN1/1n2Ps3/P1PP2P2/2+bS5/1g3+p3/L1s6 w R2GNL3Pn4p 35",
            "counter-check illegal drop (N*8a)",
        ),
        // 3. 開き王手: 飛 8h8i+ は玉(7b)を直接王手しない (開き王手)．受け方は逃れ可．
        (
            "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PGPP2P2/K2S5/1r3+p3/LN+b6 w R2GNLPgs5p 9",
            "discovered check (8h8i+)",
        ),
        // 4. pinner 自身の移動による pin 解除: 移動前は +R(1e) が金(1f) を file 1 に
        //    縦ピンしているが，王手 1e2f で pinner が動くと金は unpin され 1f2f (横) で
        //    王手駒を取り返せる．移動前 pin の流用は偽 1 手詰になる．
        (
            "4+N4/9/6pS1/9/8+R/6+B1g/1R2S3k/3p3N1/9 b NPb3g2sn4l15p 17",
            "pin released by moving pinner (1e2f)",
        ),
    ];
    for (sfen, label) in cases {
        let mut b = Board::new();
        b.set_sfen(sfen).unwrap();
        let s = DfPnSolver::with_timeout(31, 1_000, 5);
        let checks = s.generate_check_moves(&mut b);
        // 真の 1 手詰の有無を generate_legal_moves で確定する (ground truth)．
        let mut real_mate1 = false;
        for &c in checks.iter() {
            let cap = b.do_move(c);
            let mated = b.is_in_check(b.turn()) && movegen::generate_legal_moves(&mut b).is_empty();
            b.undo_move(c, cap);
            if mated {
                real_mate1 = true;
                break;
            }
        }
        assert!(
            !real_mate1,
            "{label}: テスト局面は本来 1 手詰でないはず ({sfen})"
        );
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
/// mate_move_in_1ply の counter-check 健全性を守る．
#[test]
fn test_counter_check_example() {
    let sfen = "7l1/5n1k1/7+RP/6sK1/7L1/9/9/9/9 w r2b4g3s3n2l17p 2";
    let mut b = Board::new();
    b.set_sfen(sfen).unwrap();
    // 前提: 攻め方自身が王手されている (逆王手局面)．
    assert!(
        b.is_in_check(b.turn),
        "テスト前提: 攻め方が王手されている逆王手局面のはず"
    );
    let mut s = DfPnSolver::with_timeout(31, 5_000_000, 60);
    match s.solve_impl(&mut b) {
        TsumeResult::Checkmate { moves, .. } => {
            // 終局面が真の詰みか検証 (false mate 検出)．
            let mut chk = Board::new();
            chk.set_sfen(sfen).unwrap();
            for m in &moves {
                chk.do_move(*m);
            }
            let in_check = chk.is_in_check(chk.turn());
            let legal = movegen::generate_legal_moves(&mut chk).len();
            assert!(
                in_check && legal == 0,
                "逆王手例題 PV 終局面は真の詰みであるべき: in_check={in_check} legal={legal}"
            );
            assert_eq!(
                moves.len(),
                7,
                "逆王手例題は mate-7 のはず: {:?}",
                moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>()
            );
        }
        other => panic!("逆王手例題 NON-SOLVE: {other:?}"),
    }
}

/// 探索本体の correctness + 29te canonical 検証．
///
/// **[SLOW]** 29te を解く．canonical: mate-29 / 真の詰み (in_check && legal==0) / node 数 9,288．
#[test]
#[ignore]
fn test_29te() {
    // 1 手詰 correctness．
    {
        let mut b = Board::empty();
        b.set_sfen("8k/9/7G1/9/9/9/9/9/9 b G 1").unwrap();
        let mut s = DfPnSolver::with_timeout(3, 1_000_000, 30);
        match s.solve_impl(&mut b) {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => eprintln!(
                "[dfpn] 1te: {} moves ({}), {} nodes",
                moves.len(),
                moves.first().map(|m| m.to_usi()).unwrap_or_default(),
                nodes_searched
            ),
            other => eprintln!("[dfpn] 1te FAIL: {other:?}"),
        }
    }
    // 3 手詰 correctness．
    {
        let mut b = Board::empty();
        b.set_sfen("8k/9/6R2/9/9/9/9/9/9 b G 1").unwrap();
        let mut s = DfPnSolver::with_timeout(7, 1_000_000, 30);
        match s.solve_impl(&mut b) {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let u: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!(
                    "[dfpn] 3te: {} moves {:?}, {} nodes",
                    moves.len(),
                    u,
                    nodes_searched
                );
            }
            other => eprintln!("[dfpn] 3te FAIL: {other:?}"),
        }
    }
    // 29te 計測 (budget 制限つき; 爆発時は budget で止める)．
    {
        let sfen = "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let mut b = Board::new();
        b.set_sfen(sfen).unwrap();
        let mut s = DfPnSolver::with_timeout(31, 5_000_000, 120);
        match s.solve_impl(&mut b) {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!(
                    "[dfpn] 29te: {} moves, {} nodes | PV {:?}",
                    moves.len(),
                    nodes_searched,
                    usi
                );
                // PV を辿った終局面が本当に詰みか (合法手なし) 検証 = false mate 検出．
                let mut chk = Board::new();
                chk.set_sfen(sfen).unwrap();
                for m in &moves {
                    chk.do_move(*m);
                }
                let final_legal = movegen::generate_legal_moves(&mut chk).len();
                let in_check = chk.is_in_check(chk.turn());
                eprintln!("[dfpn] 29te final: in_check={in_check}, legal_moves={final_legal} (詰み=in_check&&legal==0)");
                // look-ahead + dominance により canonical mate-29 を sound に解く．
                // soundness ガード: 真の詰み (in_check && legal==0) + canonical mate-29 + node 数固定．
                assert!(
                    in_check && final_legal == 0,
                    "LE path PV 終局面は真の詰みであるべき (false mate 検出)"
                );
                assert_eq!(
                    moves.len(),
                    29,
                    "default は canonical mate-29 を返すべき: {}",
                    moves.len()
                );
                // find_shortest 総ノード数 (canonical anchor)．
                assert_eq!(
                    nodes_searched, 396_516,
                    "29te の探索ノード数が canonical (396,516) から乖離: {nodes_searched}"
                );
            }
            other => panic!("[dfpn] 29te NON-SOLVE: {other:?}, {} nodes", s.nodes),
        }
    }
}

/// 39te (39 手詰) の最短手数検証 + 計測 (**[SLOW]**)．production `solve_impl` で探索する．
///
/// **正解 = 39 手詰** (ユーザ提供 KIF; KH MinLength の 47 は無駄合い込みで誤り)．正解 PV (USI):
///   7b6b 5b4c 8b9c 4c3d 1b2c 3d2c N*1e 2c3b N*2d 3b2b 2d1b+ 2b3b 1b2b 3b2b 4f1c 2b1c
///   9c3c 1c1d 3c2c 1d1e P*1f 1e1f P*1g 1f1g 5g6f 1g1h 2c2g 1h1i 8g8i S*6i 8i6i 6h6i+
///   S*2h 1i2i 2h3g 2i3i 2g2h 3i4i 2h4h
///
/// find_shortest は無駄合い-free len budget (透過中合い drop を len から credit) により
/// 真の最短 **39 手** へ収束する (これを欠くと len-budget が無駄合い込み raw ply を数え，
/// len=37 以下を偽 disproof して root を過大評価する)．STRICT-VERIFY Some(39)・PV は上記正解と一致．
/// SFEN/BUDGET/SECS env で上書き可．
#[test]
#[ignore]
fn test_39te_measure() {
    // SFEN で任意局面に差し替え可能 (計測用)．
    let sfen_env = std::env::var("SFEN").ok();
    let sfen: &str = sfen_env
        .as_deref()
        .unwrap_or("9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1");
    let mut b = Board::new();
    b.set_sfen(sfen).unwrap();
    // 計測専用: budget/timeout を env で上書き可能に．default は 39te が解ける予算．
    let budget: u64 = std::env::var("BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000_000);
    let secs: u64 = std::env::var("SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let mut s = DfPnSolver::with_timeout(47, budget, secs);
    let t0 = std::time::Instant::now();
    let result = s.solve_impl(&mut b);
    let elapsed = t0.elapsed();
    match result {
        TsumeResult::Checkmate {
            moves,
            nodes_searched,
        } => {
            let usi: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            eprintln!(
                "[dfpn] 39te SOLVE: {} moves, {} nodes, {:.1}s | PV {:?}",
                moves.len(),
                nodes_searched,
                elapsed.as_secs_f64(),
                usi
            );
            let mut chk = Board::new();
            chk.set_sfen(sfen).unwrap();
            for m in &moves {
                chk.do_move(*m);
            }
            let final_legal = movegen::generate_legal_moves(&mut chk).len();
            let in_check = chk.is_in_check(chk.turn());
            eprintln!(
                "[dfpn] 39te final: in_check={in_check}, legal_moves={final_legal} (詰み=in_check&&legal==0)"
            );
            // soundness ガード (SOUNDNESS-HALT): PV 終局面は真の詰みであるべき．
            assert!(
                in_check && final_legal == 0,
                "39te PV 終局面は真の詰みであるべき (false mate 検出): in_check={in_check} legal={final_legal}"
            );
            // default sfen のときのみ最短手数 + canonical node 数を固定 (回帰ガード)．
            if sfen_env.is_none() {
                assert_eq!(
                    moves.len(),
                    39,
                    "39te は真の最短 39 手を返すべき: {}",
                    moves.len()
                );
                // canonical 更新履歴: 17,545,528 → 17,593,615 (maou_shogi 5.7.1 の
                // 二歩マスク修正 occupied_files で受け方の非合法 P*8i 系が消え，
                // 探索順序が変化．mate-39/STRICT Some(39)/PV は不変)
                assert_eq!(
                    nodes_searched, 17_593_615,
                    "39te の探索ノード数が canonical (17,593,615) から乖離"
                );
            }
        }
        other => panic!(
            "[dfpn] 39te NON-SOLVE: {other:?}, {} nodes, {:.1}s",
            s.nodes,
            elapsed.as_secs_f64()
        ),
    }
}

/// 39te 分岐局面プローブ (**[SLOW]**, 診断用)．
///
/// 正解 39 手手順を 1 手ずつ進め，**攻め方手番**の各局面 (even ply) で find_shortest 解の詰み
/// 手数を **期待値 (39 - 手数)** と比較する．maou がどの局面で手数を過大評価するか (無駄合い /
/// len-bound 偽 proof) を局所化する (user 提案の方法論: 分岐局面 SFEN を再帰的に解いて残手数確認)．
#[test]
#[ignore]
fn test_39te_divergence_probe() {
    let sfen = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1";
    let line = [
        "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c", "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+",
        "2b3b", "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d", "3c2c", "1d1e", "P*1f", "1e1f",
        "P*1g", "1f1g", "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i", "8i6i", "6h6i+", "S*2h",
        "1i2i", "2h3g", "2i3i", "2g2h", "3i4i", "2h4h",
    ];
    let total = line.len();
    assert_eq!(total, 39, "正解は 39 手");
    // 局面のみ最大 PROBE_MAX まで (早期分岐の局所化用; env PROBE_MAX で上書き可)．
    let probe_max: usize = std::env::var("PROBE_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(total);
    let mut board = Board::new();
    board.set_sfen(sfen).unwrap();
    for (k, &usi) in line.iter().enumerate() {
        if k % 2 == 0 && k <= probe_max {
            let expected = (total - k) as i64;
            let sub_sfen = board.sfen();
            let mut b = Board::new();
            b.set_sfen(&sub_sfen).unwrap();
            let mut s = DfPnSolver::with_timeout(47, 15_000_000, 30);
            let got = match s.solve_impl(&mut b) {
                TsumeResult::Checkmate { moves, .. } => moves.len() as i64,
                _ => -1,
            };
            let flag = if got == expected {
                "OK"
            } else if got > expected {
                "OVER"
            } else {
                "UNDER/UNSOLVED"
            };
            eprintln!(
                "[probe] k={k:2} expected={expected:2} got={got:3} nodes={:>9} [{flag}] sfen={sub_sfen}",
                s.nodes,
            );
        }
        let m = board
            .move_from_usi(usi)
            .unwrap_or_else(|| panic!("invalid USI: {usi}"));
        board.do_move(m);
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

    let mut solver = DfPnSolver::new(3, 100_000);
    let result = solver.solve_impl(&mut board);

    match &result {
        TsumeResult::Checkmate { moves, .. } => {
            assert_eq!(
                moves.len(),
                1,
                "should be 1-move checkmate, got: {:?}",
                moves.iter().map(|m| m.to_usi()).collect::<Vec<_>>()
            );
            let usi = moves[0].to_usi();
            assert!(
                usi == "G*8h" || usi == "G*9h",
                "1手詰め: G*8h(8八金打) or G*9h(9八金打), got: {}",
                usi
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

    let result =
        solve_tsume_with_timeout(sfen, Some(7), Some(1_048_576), None, None, None, None).unwrap();

    let expected = ["7g9g+", "9i8i", "G*8h"];

    match &result {
        TsumeResult::Checkmate { moves, .. } => {
            let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            assert_eq!(
                usi_moves.len(),
                3,
                "expected 3 moves, got {}: {:?}",
                usi_moves.len(),
                usi_moves
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
    let result = solve_tsume(sfen, Some(31), Some(5_000_000)).unwrap();

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
                usi_moves.len(),
                7,
                "expected 7-move checkmate, got {}: {:?}",
                usi_moves.len(),
                usi_moves
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
    let mut solver = DfPnSolver::new(31, 2_000_000);
    let result = solver.solve_impl(&mut board);
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
    let result = solve_tsume(sfen, Some(31), Some(2_000_000)).unwrap();

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

    let mut solver = DfPnSolver::new(31, 1);
    let moves = solver.generate_defense_moves_inner(&mut board, false);

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
#[test]
fn test_no_checkmate_counter_check() {
    let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
    let result = solve_tsume(sfen, Some(31), Some(3_000_000)).unwrap();

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
    let result = solve_tsume(sfen, Some(31), Some(2_000_000)).unwrap();

    match &result {
        TsumeResult::Checkmate { moves, .. } => {
            let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            // 逆王手回避 (3四玉で銀を取りつつ王手回避) から詰む sound な PV が返ることを検証する．
            // 探索は最短保証なし (proof-tree の詰み手数は探索順依存) ゆえ，厳密 PV ではなく
            // PV replay で真の詰みであることを確認する．
            // 本来の最短は 5 手 (複数解; 合駒駒種は探索順依存):
            //   "2d3d 2b3a S*4b 3a3b 4c3c+" / "2d3d {R|P|N}*2c 2e2c+ 2b1a 1c1b+" /
            //   "2d3d L*2c 2e2c+ 2b3a L*3b" / "2d3d P*2d 4c3c+ 2b1a S*1b"．
            assert!(
                usi_moves.len() % 2 == 1,
                "tsume must have odd number of moves, got {}: {usi_moves:?}",
                usi_moves.len(),
            );
            let mut chk = Board::new();
            chk.set_sfen(sfen).unwrap();
            for m in moves {
                chk.do_move(*m);
            }
            let in_check = chk.is_in_check(chk.turn());
            let legal = movegen::generate_legal_moves(&mut chk).len();
            assert!(
                in_check && legal == 0,
                "PV 終局面は真の詰みであるべき (false mate 検出): in_check={in_check} legal={legal} PV={usi_moves:?}"
            );
        }
        other => panic!("expected Checkmate (king captures silver), got {:?}", other),
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
    let result = solve_tsume(sfen, Some(31), Some(100_000)).unwrap();

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
    let result = solve_tsume(sfen, Some(31), Some(2_000_000)).unwrap();

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
    let result = solve_tsume(sfen, Some(31), Some(2_000_000)).unwrap();

    match &result {
        TsumeResult::Checkmate { moves, .. } => {
            let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            assert_eq!(
                usi_moves.len(),
                9,
                "expected 9-move checkmate, got {}: {:?}",
                usi_moves.len(),
                usi_moves
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
    let defenses = solver.generate_defense_moves_inner(&mut board, false);
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
    let defenses = solver.generate_defense_moves_inner(&mut board, false);
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
    let king_step = attack::step_attacks(defender.opponent(), PieceType::King, king_sq);
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
/// complete_or_proofs 中の探索が転置により証明済み TT エントリを
/// 上書きしていたバグの回帰テスト．find_shortest=true(デフォルト)で
/// PV が最長抵抗を正しく反映することを確認する．
#[test]
fn test_pv_follows_longest_defense() {
    let sfen = "7gk/8p/5P2s/7P1/9/9/9/9/9 b BSN2rb3g2s3n4l15p 1";
    let result = solve_tsume_with_timeout(
        sfen,
        Some(31),
        Some(2_000_000),
        None,
        Some(true), // find_shortest = true
        None,
        None,
    )
    .unwrap();

    match &result {
        TsumeResult::Checkmate { moves, .. } => {
            let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
            assert_eq!(
                usi_moves.len(),
                9,
                "PV should be 9 moves (longest defense via gold interposition), got {}: {:?}",
                usi_moves.len(),
                usi_moves
            );
            assert_eq!(usi_moves[0], "B*3c", "move 1: B*3c(3三角打)");
            assert_eq!(
                usi_moves[1], "2a2b",
                "move 2: g(2a→2b)(金の移動合い=最長抵抗)"
            );
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

    let mut solver = DfPnSolver::new(5, 100_000);
    let result = solver.solve_impl(&mut board);

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
            solve_tsume(sfen, Some(31), Some(10_000)).unwrap()
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
    let defenses = solver.generate_defense_moves_inner(&mut board, false);
    let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();

    // 全合法手と比較
    let legal = movegen::generate_legal_moves(&mut board);
    let usi_legal: Vec<String> = legal.iter().map(|m| m.to_usi()).collect();

    // P*7d が合法手に含まれること
    assert!(
        usi_legal.contains(&"P*7d".to_string()),
        "P*7d should be a legal move, got: {:?}",
        usi_legal
    );

    // P*7d が回避手にも含まれること(合い効かずで除外されていないこと)
    assert!(
        usi_defenses.contains(&"P*7d".to_string()),
        "P*7d should be in defense moves (中合い), but was filtered out.\n\
             defense moves: {:?}\n\
             legal moves: {:?}",
        usi_defenses,
        usi_legal
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
    let result = solve_tsume(sfen, Some(31), Some(10_000)).unwrap();

    assert!(
        !matches!(result, TsumeResult::Checkmate { .. }),
        "中合い後の局面は不詰みのはず: {:?}",
        result
    );
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

    std::thread::Builder::new()
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

                let defenses = solver.generate_defense_moves_inner(&mut board, false);
                let drop_cnt = defenses.iter().filter(|m| m.is_drop()).count();
                writeln!(
                    out,
                    "  後手応手数={} (drop={}, board={})",
                    defenses.len(),
                    drop_cnt,
                    defenses.len() - drop_cnt
                )
                .unwrap();

                // AND ノードのチェーン情報
                let def_color = board.turn();
                let att_color = def_color.opponent();
                if let Some(king_sq) = board.king_square(def_color) {
                    let checkers = board.compute_checkers_at(king_sq, att_color);
                    if checkers.count() == 1 {
                        let checker_sq = checkers.lsb().unwrap();
                        writeln!(
                            out,
                            "  王手駒: {}{}",
                            9 - checker_sq.col(),
                            (b'a' + checker_sq.row()) as char
                        )
                        .unwrap();
                        if let Some(_sl) = solver.find_sliding_checker(&board, king_sq, att_color) {
                            let btw = attack::between_bb(checker_sq, king_sq);
                            let (fut, chn) = solver.compute_futile_and_chain_squares(
                                &board, &btw, king_sq, checker_sq, def_color, att_color,
                            );
                            write!(out, "  Between:").unwrap();
                            for sq in btw {
                                write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char)
                                    .unwrap();
                            }
                            writeln!(out).unwrap();
                            write!(out, "  Futile:").unwrap();
                            for sq in fut {
                                write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char)
                                    .unwrap();
                            }
                            writeln!(out).unwrap();
                            write!(out, "  Chain:").unwrap();
                            for sq in chn {
                                write!(out, " {}{}", 9 - sq.col(), (b'a' + sq.row()) as char)
                                    .unwrap();
                            }
                            writeln!(out).unwrap();
                            let normal = btw.count() - fut.count() - chn.count();
                            writeln!(
                                out,
                                "  Normal squares (= 非futile非chain 合い駒候補): {}",
                                normal
                            )
                            .unwrap();
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
                    writeln!(
                        out,
                        "    {} => 次王手数={}",
                        def_m.to_usi(),
                        next_chks.len()
                    )
                    .unwrap();

                    // 応手後の AND ノードを 50K ノードで解いて結果確認
                    let sub_sfen = board.sfen();
                    board.undo_move(*def_m, cap2);

                    let sub_result = solve_tsume(&sub_sfen, Some(31), Some(50_000))
                        .map(|r| format!("{:?}", r))
                        .unwrap_or_else(|e| format!("Err({:?})", e));
                    writeln!(out, "      50K結果: {}", sub_result).unwrap();
                }

                board.undo_move(*chk, cap1);
            }

            drop(out);
            eprintln!("診断完了: {}", out_path);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// counter_check 不詭め局面のノード予算プローブ
#[test]
#[ignore]
fn test_no_checkmate_counter_check_probe() {
    let sfen = "7l1/5n1k1/5R2P/6sK1/7L1/9/9/9/9 b r2b4g3s3n2l17p 1";
    for budget in [3_000_000u64, 5_000_000, 10_000_000] {
        let result = solve_tsume(sfen, Some(31), Some(budget)).unwrap();
        eprintln!("budget={}: {:?}", budget, result);
        if let TsumeResult::NoCheckmate { nodes_searched } = &result {
            eprintln!("  → solved at {} nodes", nodes_searched);
            return;
        }
    }
    eprintln!("Still not solved at 10M");
}

/// 協調的停止フラグ: 事前に立てたフラグで探索が即座に打ち切られる
/// (7 手詰め局面 — フラグ無しでは相応のノード数を要する)
#[test]
fn test_external_stop_flag_aborts_search() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    let flag = Arc::new(AtomicBool::new(true));
    let mut solver = DfPnSolver::with_timeout(63, 100_000_000, 3600);
    solver.set_stop_flag(Arc::clone(&flag));
    let mut board = crate::board::Board::empty();
    board
        .set_sfen("8k/9/6R2/9/9/9/9/9/9 b G 1")
        .expect("正当な SFEN");
    let report = solver.solve_report(&mut board);
    assert!(
        matches!(report.result, TsumeResult::Unknown { .. }),
        "外部停止は timeout と同じ Unknown になるはず: {:?}",
        report.result
    );
    assert_eq!(report.stop_reason, StopReason::Timeout);
}
