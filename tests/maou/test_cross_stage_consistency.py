"""Cross-stage consistency verification tests.

Stage1, Stage2, Preprocess の局面データ(boardIdPositions, piecesInHand)が
同一局面に対して同じ値を生成することを検証する．

マルチステージトレーニングでは入力データが同じ意味を表現する必要があるため，
各ステージの生成パスで一致することを保証する．
"""

from __future__ import annotations

import numpy as np
import pytest

from maou.app.pre_process import feature
from maou.app.pre_process.feature import (
    make_board_id_positions,
    make_feature_from_board_state,
    make_pieces_in_hand,
)
from maou.domain.board.shogi import Board, PieceId

# PieceId → SFEN character mapping (BLACK pieces only)
_PIECE_ID_TO_SFEN: dict[int, str] = {
    PieceId.FU: "P",
    PieceId.KY: "L",
    PieceId.KE: "N",
    PieceId.GI: "S",
    PieceId.KI: "G",
    PieceId.KA: "B",
    PieceId.HI: "R",
    PieceId.OU: "K",
    PieceId.TO: "+P",
    PieceId.NKY: "+L",
    PieceId.NKE: "+N",
    PieceId.NGI: "+S",
    PieceId.UMA: "+B",
    PieceId.RYU: "+R",
}


def _make_single_piece_sfen(
    piece_id: int, row: int, col: int
) -> str:
    """単一駒のSFEN文字列を生成する．

    Args:
        piece_id: PieceId enum value (1-14)
        row: 段 (0=一段, 8=九段)
        col: 筋 (0=1筋, 8=9筋)

    Returns:
        SFEN文字列 (例: "9/9/9/9/9/9/4P4/9/9 b - 1")
    """
    sfen_char = _PIECE_ID_TO_SFEN[piece_id]
    ranks = []
    for r in range(9):
        if r != row:
            ranks.append("9")
        else:
            # SFEN rank: left = 9筋(file=8), right = 1筋(file=0)
            left_empty = 8 - col  # squares before piece
            right_empty = col  # squares after piece
            rank_str = ""
            if left_empty > 0:
                rank_str += str(left_empty)
            rank_str += sfen_char
            if right_empty > 0:
                rank_str += str(right_empty)
            ranks.append(rank_str)
    return "/".join(ranks) + " b - 1"


class TestStage1VsCshogiBoard:
    """Stage1の手動盤面構築とcshogi経由のmake_board_id_positions()の一致検証."""

    @pytest.mark.parametrize(
        "piece_id,row,col",
        [
            # 非成駒: 各駒種で代表的な位置
            (PieceId.FU, 6, 4),  # 歩: 5七
            (PieceId.KY, 8, 0),  # 香: 1九
            (PieceId.KE, 7, 1),  # 桂: 2八
            (PieceId.GI, 5, 3),  # 銀: 4六
            (PieceId.KI, 0, 3),  # 金: 4一
            (PieceId.KA, 7, 7),  # 角: 8八
            (PieceId.HI, 7, 1),  # 飛: 2八
            (PieceId.OU, 0, 4),  # 王: 5一
            # 成駒
            (PieceId.TO, 2, 5),  # と: 6三
            (PieceId.NKY, 3, 0),  # 成香: 1四
            (PieceId.NKE, 4, 8),  # 成桂: 9五
            (PieceId.NGI, 1, 6),  # 成銀: 7二
            (PieceId.UMA, 5, 2),  # 馬: 3六
            (PieceId.RYU, 3, 7),  # 龍: 8四
        ],
    )
    def test_board_positions_match(
        self, piece_id: int, row: int, col: int
    ) -> None:
        """Stage1の手動構築とcshogi経由で同一のboardIdPositionsが得られること."""
        # Stage1方式: 手動構築
        stage1_board = np.zeros((9, 9), dtype=np.uint8)
        stage1_board[row][col] = piece_id

        # Stage2/Preprocess方式: cshogi経由
        sfen = _make_single_piece_sfen(piece_id, row, col)
        board = Board()
        board.set_sfen(sfen)
        cshogi_board = make_board_id_positions(board)

        np.testing.assert_array_equal(
            stage1_board,
            cshogi_board,
            err_msg=(
                f"PieceId={PieceId(piece_id).name} at ({row},{col}), "
                f"SFEN='{sfen}'"
            ),
        )

    @pytest.mark.parametrize(
        "piece_id,row,col",
        [
            (PieceId.FU, 6, 4),
            (PieceId.KI, 3, 3),
            (PieceId.KA, 7, 7),
            (PieceId.HI, 7, 1),
            (PieceId.TO, 2, 5),
            (PieceId.UMA, 5, 2),
            (PieceId.RYU, 3, 7),
        ],
    )
    def test_feature_planes_from_stage1_data(
        self, piece_id: int, row: int, col: int
    ) -> None:
        """Stage1のデータをmake_feature_from_board_stateに通して有効な特徴量になること."""
        # Stage1方式のデータ
        board_positions = np.zeros((9, 9), dtype=np.uint8)
        board_positions[row][col] = piece_id
        pieces_in_hand = np.zeros(14, dtype=np.uint8)

        # 特徴量変換が例外なく成功すること
        features = make_feature_from_board_state(
            board_positions, pieces_in_hand
        )
        assert features.shape[0] > 0

        # (row, col) にいずれかの駒プレーンで1が立つこと
        # NOTE: make_feature_from_board_state は単一駒盤面で
        # 手番推定ヒューリスティックにより先手/後手プレーンの
        # 割り当てが変わりうるため，プレーンインデックスは指定しない
        piece_planes = features[:, row, col]
        assert piece_planes.sum() == 1, (
            f"Expected exactly 1 plane active at ({row},{col}), "
            f"got {piece_planes.sum()}"
        )

    @pytest.mark.parametrize(
        "piece_id,row,col",
        [
            (PieceId.FU, 6, 4),
            (PieceId.KI, 3, 3),
            (PieceId.KA, 7, 7),
            (PieceId.HI, 7, 1),
        ],
    )
    def test_feature_planes_stage1_vs_cshogi_match(
        self, piece_id: int, row: int, col: int
    ) -> None:
        """Stage1データとcshogi経由データで同一の特徴量プレーンが得られること."""
        # Stage1方式
        stage1_board = np.zeros((9, 9), dtype=np.uint8)
        stage1_board[row][col] = piece_id
        stage1_hand = np.zeros(14, dtype=np.uint8)
        stage1_features = make_feature_from_board_state(
            stage1_board, stage1_hand
        )

        # cshogi方式
        sfen = _make_single_piece_sfen(piece_id, row, col)
        board = Board()
        board.set_sfen(sfen)
        cshogi_board = make_board_id_positions(board)
        cshogi_hand = make_pieces_in_hand(board)
        cshogi_features = make_feature_from_board_state(
            cshogi_board, cshogi_hand
        )

        np.testing.assert_array_equal(
            stage1_features,
            cshogi_features,
            err_msg=(
                f"Feature planes differ for "
                f"PieceId={PieceId(piece_id).name} at ({row},{col})"
            ),
        )


class TestStage1VsCshogiHandPieces:
    """Stage1の持ち駒データとcshogi経由のmake_pieces_in_hand()の一致検証."""

    @pytest.mark.parametrize(
        "piece_type_idx,sfen_hand_char",
        [
            (0, "P"),  # FU
            (1, "L"),  # KY
            (2, "N"),  # KE
            (3, "S"),  # GI
            (4, "G"),  # KI
            (5, "B"),  # KA
            (6, "R"),  # HI
        ],
    )
    def test_pieces_in_hand_match(
        self, piece_type_idx: int, sfen_hand_char: str
    ) -> None:
        """Stage1の持ち駒データとcshogi経由で同一のpiecesInHandが得られること."""
        # Stage1方式: 手動構築
        stage1_hand = np.zeros(14, dtype=np.uint8)
        stage1_hand[piece_type_idx] = 1

        # cshogi方式
        sfen = f"9/9/9/9/9/9/9/9/9 b {sfen_hand_char} 1"
        board = Board()
        board.set_sfen(sfen)
        cshogi_hand = make_pieces_in_hand(board)

        np.testing.assert_array_equal(
            stage1_hand,
            cshogi_hand,
            err_msg=(
                f"piecesInHand differs for piece_type_idx={piece_type_idx} "
                f"({sfen_hand_char})"
            ),
        )


class TestStage2VsPreprocess:
    """Stage2とPreprocessが同一局面で同一データを生成することの検証.

    両者は同じ make_board_id_positions() / make_pieces_in_hand() を呼ぶため，
    HCPからの復元も含めてラウンドトリップで一致することを確認する．
    """

    @pytest.mark.parametrize(
        "sfen",
        [
            # 初期局面 (先手番)
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            # 先手番: 持ち駒あり
            "lnsgk1snl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 5",
            # 後手番
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
        ],
    )
    def test_hcp_roundtrip_produces_identical_data(
        self, sfen: str
    ) -> None:
        """HCPエンコード→デコードで同一のboardIdPositions/piecesInHandが得られること."""
        import cshogi

        # 元の局面から特徴量を生成
        board = Board()
        board.set_sfen(sfen)
        original_board_ids = make_board_id_positions(board)
        original_hand = make_pieces_in_hand(board)

        # HCPにエンコードしてから復元 (Stage2/Preprocess のパスを模倣)
        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]
        hcp = np.frombuffer(
            hcp_bytes, dtype=cshogi.HuffmanCodedPos
        )

        board2 = Board()
        board2.set_hcp(hcp)
        restored_board_ids = make_board_id_positions(board2)
        restored_hand = make_pieces_in_hand(board2)

        np.testing.assert_array_equal(
            original_board_ids,
            restored_board_ids,
            err_msg=f"boardIdPositions differs after HCP roundtrip for {sfen}",
        )
        np.testing.assert_array_equal(
            original_hand,
            restored_hand,
            err_msg=f"piecesInHand differs after HCP roundtrip for {sfen}",
        )

    @pytest.mark.parametrize(
        "sfen",
        [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
            "8k/9/9/9/9/9/9/9/K8 b - 1",
        ],
    )
    def test_feature_roundtrip_consistency(
        self, sfen: str
    ) -> None:
        """make_feature と make_feature_from_board_state の一致検証.

        cshogi直接パス(make_feature)と
        boardIdPositions経由パス(make_feature_from_board_state)が
        同じ特徴量を生成することを確認する．
        """
        board = Board()
        board.set_sfen(sfen)

        # cshogi直接パス
        direct_features = feature.make_feature(board)

        # boardIdPositions経由パス
        board_ids = make_board_id_positions(board)
        hand = make_pieces_in_hand(board)
        indirect_features = make_feature_from_board_state(
            board_ids, hand
        )

        np.testing.assert_array_equal(
            direct_features,
            indirect_features,
            err_msg=f"Feature planes differ for {sfen}",
        )


class TestAllStage1PatternsConsistency:
    """Stage1の全パターンがcshogi経由と一致するかの網羅的検証."""

    def test_all_board_patterns_match_cshogi(self) -> None:
        """Stage1の全盤上パターンでcshogi経由と一致すること."""
        from maou.domain.data.stage1_generator import (
            Stage1DataGenerator,
        )

        mismatches: list[str] = []

        for (
            pattern
        ) in Stage1DataGenerator.enumerate_board_patterns():
            # Stage1方式
            stage1_board = np.zeros((9, 9), dtype=np.uint8)
            stage1_board[pattern.row][pattern.col] = (
                pattern.piece_id
            )

            # cshogi方式
            sfen = _make_single_piece_sfen(
                pattern.piece_id, pattern.row, pattern.col
            )
            board = Board()
            board.set_sfen(sfen)
            cshogi_board = make_board_id_positions(board)

            if not np.array_equal(stage1_board, cshogi_board):
                piece_name = PieceId(pattern.piece_id).name
                mismatches.append(
                    f"{piece_name} at ({pattern.row},{pattern.col})"
                )

        assert mismatches == [], (
            f"{len(mismatches)} mismatches found: "
            + ", ".join(mismatches[:10])
        )

    def test_all_hand_patterns_match_cshogi(self) -> None:
        """Stage1の全持ち駒パターンでcshogi経由と一致すること."""
        from maou.domain.data.stage1_generator import (
            Stage1DataGenerator,
        )

        sfen_chars = ["P", "L", "N", "S", "G", "B", "R"]
        mismatches: list[str] = []

        for (
            pattern
        ) in Stage1DataGenerator.enumerate_hand_patterns():
            # Stage1方式
            stage1_hand = np.zeros(14, dtype=np.uint8)
            stage1_hand[pattern.piece_type_idx] = 1

            # cshogi方式
            sfen_char = sfen_chars[pattern.piece_type_idx]
            sfen = f"9/9/9/9/9/9/9/9/9 b {sfen_char} 1"
            board = Board()
            board.set_sfen(sfen)
            cshogi_hand = make_pieces_in_hand(board)

            if not np.array_equal(stage1_hand, cshogi_hand):
                mismatches.append(
                    f"piece_type_idx={pattern.piece_type_idx} ({sfen_char})"
                )

        assert mismatches == [], (
            f"{len(mismatches)} mismatches found: "
            + ", ".join(mismatches)
        )
