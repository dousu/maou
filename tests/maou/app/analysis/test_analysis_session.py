"""analysis_session (棋譜解析 GUI のセッション基盤) のテスト．"""

import copy
from pathlib import Path

import pytest

from maou.app.analysis.analysis_session import (
    GameDocument,
    _move_info,
    _snapshot,
    advance_move,
    build_variation_tree,
    goto_node,
    legal_move_infos,
    load_game,
    mainline_ancestor,
    path_moves_usi,
    validate_report,
)
from maou.domain.board.shogi import Board, PieceId

RESOURCES = Path(__file__).parent / "resources"

HIRATE_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/"
    "LNSGKGSNL b - 1"
)


@pytest.fixture
def mini_document() -> GameDocument:
    """mini.csa (4 手 + 投了) から GameDocument を構築する．"""
    data = (RESOURCES / "mini.csa").read_bytes()
    return load_game(data, "csa")


class TestLoadGame:
    """load_game のテスト．"""

    def test_snapshots_structure(
        self, mini_document: GameDocument
    ) -> None:
        """スナップショットは初期局面 + 手数分生成される．"""
        doc = mini_document
        assert doc.n_moves == 4
        assert len(doc.snapshots) == 5
        assert doc.moves_usi == [
            "7g7f",
            "3c3d",
            "2g2f",
            "8c8d",
        ]
        assert doc.snapshots[0].sfen == HIRATE_SFEN
        assert doc.snapshots[0].ply == 0
        assert doc.snapshots[0].last_move is None
        assert doc.snapshots[4].ply == 4

    def test_turn_alternates(
        self, mini_document: GameDocument
    ) -> None:
        """手番が局面ごとに交互になる．"""
        turns = [s.turn for s in mini_document.snapshots]
        assert turns == ["b", "w", "b", "w", "b"]

    def test_last_move_info(
        self, mini_document: GameDocument
    ) -> None:
        """last_move に移動元/先マスと駒種が入る．"""
        info = mini_document.snapshots[1].last_move
        assert info is not None
        assert info.usi == "7g7f"
        # 7g = (7-1)*9 + (7-1) = 60, 7f = (7-1)*9 + (6-1) = 59
        assert info.from_square == 60
        assert info.to_square == 59
        assert info.is_drop is False
        assert info.piece_id == PieceId.FU

    def test_metadata(
        self, mini_document: GameDocument
    ) -> None:
        """棋譜メタデータが引き継がれる．"""
        doc = mini_document
        assert doc.input_format == "csa"
        assert doc.names == ["black_engine", "white_engine"]
        assert doc.endgame == "%TORYO"

    def test_snapshot_is_plain_data(
        self, mini_document: GameDocument
    ) -> None:
        """gr.State 用に deepcopy 可能な plain data である．"""
        clone = copy.deepcopy(mini_document)
        assert (
            clone.snapshots[2].sfen
            == mini_document.snapshots[2].sfen
        )
        board = mini_document.snapshots[1].board_id_positions
        assert len(board) == 9
        assert all(len(row) == 9 for row in board)
        assert (
            len(mini_document.snapshots[1].pieces_in_hand) == 14
        )

    def test_empty_moves_raises(self) -> None:
        """指し手ゼロの棋譜はエラー．"""
        content = (
            "V2\nN+black\nN-white\nPI\n+\n%TORYO\n"
        ).encode("utf-8")
        with pytest.raises(ValueError, match="指し手"):
            load_game(content, "csa")


class TestMoveInfo:
    """_move_info の駒打ちのテスト．"""

    def test_drop_move_info(self) -> None:
        """駒打ちは from_square=None と駒種を持つ．"""
        board = Board()
        board.set_sfen("9/9/9/9/4k4/9/9/9/4K4 b P 1")
        move = board.move_from_usi("P*5c")

        info = _move_info(board, move)

        assert info.usi == "P*5c"
        assert info.is_drop is True
        assert info.from_square is None
        assert info.drop_piece_type == 0
        assert info.piece_id == PieceId.FU
        # 5c = (5-1)*9 + (3-1) = 38
        assert info.to_square == 38


class TestValidateReport:
    """validate_report のテスト．"""

    def _matching_report(self, doc: GameDocument) -> dict:
        return {
            "positions": [
                {
                    "played_move": doc.moves_usi[i],
                    "sfen": doc.snapshots[i].sfen,
                }
                for i in range(doc.n_moves)
            ]
        }

    def test_matching_report_passes(
        self, mini_document: GameDocument
    ) -> None:
        """整合するレポートは通る．"""
        validate_report(
            mini_document,
            self._matching_report(mini_document),
        )

    def test_missing_positions_raises(
        self, mini_document: GameDocument
    ) -> None:
        """positions が無いレポートはエラー．"""
        with pytest.raises(ValueError, match="positions"):
            validate_report(mini_document, {})

    def test_length_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """局面数の不一致はエラー．"""
        report = self._matching_report(mini_document)
        report["positions"].pop()
        with pytest.raises(ValueError, match="手数"):
            validate_report(mini_document, report)

    def test_played_move_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """指し手の不一致はエラー．"""
        report = self._matching_report(mini_document)
        report["positions"][1]["played_move"] = "9c9d"
        with pytest.raises(ValueError, match="指し手"):
            validate_report(mini_document, report)

    def test_sfen_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """局面 SFEN の不一致はエラー (別対局のレポート検出)．"""
        report = self._matching_report(mini_document)
        report["positions"][0]["sfen"] = (
            "9/9/9/9/9/9/9/9/9 b - 1"
        )
        with pytest.raises(ValueError, match="SFEN"):
            validate_report(mini_document, report)


class TestVariationTree:
    """分岐木 (VariationTree) のテスト．"""

    def test_build_mainline_chain(
        self, mini_document: GameDocument
    ) -> None:
        """本譜チェーンが ply 順に構築される．"""
        tree = build_variation_tree(mini_document)
        assert len(tree.nodes) == 5
        assert tree.mainline_ids == [0, 1, 2, 3, 4]
        assert tree.current_id == 0
        root = tree.nodes[0]
        assert root.move_usi is None
        assert root.is_mainline
        assert root.children == [1]
        for ply, node_id in enumerate(tree.mainline_ids):
            node = tree.nodes[node_id]
            assert node.snapshot.ply == ply
            assert node.is_mainline

    def test_build_seeds_report(
        self, mini_document: GameDocument
    ) -> None:
        """レポートの positions[i] が本譜ノード i に載る．"""
        report = {
            "positions": [
                {"ply": i + 1, "marker": i}
                for i in range(mini_document.n_moves)
            ]
        }
        tree = build_variation_tree(mini_document, report)
        for i in range(mini_document.n_moves):
            analysis = tree.nodes[tree.mainline_ids[i]].analysis
            assert analysis is not None
            assert analysis["marker"] == i
        # 最終局面 (ply = n_moves) にはレポートが対応しない
        assert (
            tree.nodes[tree.mainline_ids[-1]].analysis is None
        )

    def test_advance_reuses_mainline_child(
        self, mini_document: GameDocument
    ) -> None:
        """本譜と同じ手は本譜ノードを再利用する．"""
        tree = build_variation_tree(mini_document)
        node = advance_move(tree, "7g7f")
        assert node.node_id == tree.mainline_ids[1]
        assert node.is_mainline
        assert len(tree.nodes) == 5  # ノードは増えない

    def test_advance_creates_branch(
        self, mini_document: GameDocument
    ) -> None:
        """本譜と違う合法手で分岐ノードが生まれる．"""
        tree = build_variation_tree(mini_document)
        node = advance_move(tree, "2g2f")
        assert not node.is_mainline
        assert node.parent_id == 0
        assert node.snapshot.ply == 1
        assert node.snapshot.turn == "w"
        assert len(tree.nodes) == 6
        # 同じ手を root からもう一度指すと同じノードを再利用する
        goto_node(tree, 0)
        node2 = advance_move(tree, "2g2f")
        assert node2.node_id == node.node_id
        assert len(tree.nodes) == 6

    def test_advance_illegal_raises(
        self, mini_document: GameDocument
    ) -> None:
        """非合法手はエラー．"""
        tree = build_variation_tree(mini_document)
        with pytest.raises(ValueError):
            advance_move(tree, "1a1b")

    def test_goto_invalid_raises(
        self, mini_document: GameDocument
    ) -> None:
        """範囲外のノード ID はエラー．"""
        tree = build_variation_tree(mini_document)
        with pytest.raises(ValueError, match="ノード"):
            goto_node(tree, 99)

    def test_path_moves_usi(
        self, mini_document: GameDocument
    ) -> None:
        """root からの USI 経路が分岐込みで得られる．"""
        tree = build_variation_tree(mini_document)
        advance_move(tree, "7g7f")  # 本譜
        advance_move(tree, "8c8d")  # 分岐 (本譜 2 手目は 3c3d)
        assert path_moves_usi(tree) == ["7g7f", "8c8d"]
        assert path_moves_usi(tree, 0) == []

    def test_mainline_ancestor(
        self, mini_document: GameDocument
    ) -> None:
        """分岐から最も近い本譜ノードに戻れる．"""
        tree = build_variation_tree(mini_document)
        advance_move(tree, "7g7f")
        advance_move(tree, "8c8d")
        advance_move(tree, "2g2f")
        ancestor = mainline_ancestor(tree)
        assert ancestor.node_id == tree.mainline_ids[1]
        # 本譜ノード自身はそのまま返る
        goto_node(tree, tree.mainline_ids[2])
        assert (
            mainline_ancestor(tree).node_id
            == tree.mainline_ids[2]
        )

    def test_tree_is_deepcopyable(
        self, mini_document: GameDocument
    ) -> None:
        """gr.State 用に deepcopy 可能な plain data である．"""
        tree = build_variation_tree(mini_document)
        advance_move(tree, "2g2f")
        clone = copy.deepcopy(tree)
        assert clone.current_id == tree.current_id
        assert len(clone.nodes) == len(tree.nodes)


class TestLegalMoveInfos:
    """legal_move_infos のテスト．"""

    def test_initial_position_moves(
        self, mini_document: GameDocument
    ) -> None:
        """平手初期局面の合法手 30 手が列挙される．"""
        infos = legal_move_infos(mini_document.snapshots[0])
        assert len(infos) == 30
        usis = {m.usi for m in infos}
        assert "7g7f" in usis
        move = next(m for m in infos if m.usi == "7g7f")
        assert move.from_square == 60  # 7g
        assert move.to_square == 59  # 7f
        assert not move.is_drop
        assert not move.is_promotion
        assert move.piece_id == PieceId.FU

    def test_promotion_flag(self) -> None:
        """成/不成の両方が is_promotion 付きで列挙される．"""
        board = Board()
        board.set_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        snapshot = _snapshot(board, 0, None)
        infos = legal_move_infos(snapshot)
        pawn_moves = [
            m
            for m in infos
            if m.from_square == 39  # 5d
        ]
        usis = {m.usi for m in pawn_moves}
        assert usis == {"5d5c", "5d5c+"}
        promo = next(m for m in pawn_moves if m.is_promotion)
        assert promo.usi == "5d5c+"

    def test_drop_moves(self) -> None:
        """駒打ちが drop_piece_type 付きで列挙される．"""
        board = Board()
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        snapshot = _snapshot(board, 0, None)
        infos = legal_move_infos(snapshot)
        drops = [m for m in infos if m.is_drop]
        assert drops
        assert all(m.drop_piece_type == 0 for m in drops)
        assert all(m.from_square is None for m in drops)
        assert all(m.usi.startswith("P*") for m in drops)
