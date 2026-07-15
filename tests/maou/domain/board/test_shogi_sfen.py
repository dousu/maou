"""shogi.py の SFEN 変換 (PIECE_ID_TO_SFEN / board_id_positions_to_sfen /
Board.from_board_id_positions) のテスト．"""

import numpy as np
import pytest

from maou.domain.board.shogi import (
    HAND_PIECE_SFEN_CHARS,
    PIECE_ID_TO_SFEN,
    Board,
    PieceId,
    Turn,
    board_id_positions_to_sfen,
    piece_id_to_sfen,
)

HIRATE_BOARD_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
)


class TestPieceIdToSfen:
    def test_empty_returns_empty_string(self) -> None:
        assert piece_id_to_sfen(PieceId.EMPTY) == ""

    def test_black_pieces_are_uppercase(self) -> None:
        assert piece_id_to_sfen(PieceId.FU) == "P"
        assert piece_id_to_sfen(PieceId.OU) == "K"
        assert piece_id_to_sfen(PieceId.RYU) == "+R"

    def test_white_pieces_are_lowercase(self) -> None:
        assert piece_id_to_sfen(PieceId.FU + 14) == "p"
        assert piece_id_to_sfen(PieceId.OU + 14) == "k"
        assert piece_id_to_sfen(PieceId.RYU + 14) == "+r"

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            piece_id_to_sfen(29)
        with pytest.raises(ValueError):
            piece_id_to_sfen(-1)

    def test_mapping_covers_all_black_pieces(self) -> None:
        """変換表が全14駒種を網羅している (単一の真実)．"""
        assert set(PIECE_ID_TO_SFEN.keys()) == {
            PieceId(i) for i in range(1, 15)
        }

    def test_hand_chars_derived_from_piece_id_map(self) -> None:
        """持ち駒文字表が PIECE_ID_TO_SFEN と整合している．"""
        assert HAND_PIECE_SFEN_CHARS == [
            "P",
            "L",
            "N",
            "S",
            "G",
            "B",
            "R",
        ]


class TestBoardIdPositionsToSfen:
    def test_initial_position(self) -> None:
        """平手初期局面の配置から正しい盤面SFENが生成される．"""
        board = Board()
        sfen = board_id_positions_to_sfen(
            board.get_normalized_board_id_positions(),
            board.get_normalized_pieces_in_hand(),
        )
        assert sfen == f"{HIRATE_BOARD_SFEN} b - 1"

    def test_hand_pieces_with_counts(self) -> None:
        """複数枚の持ち駒が枚数付きで出力される．"""
        empty_board = [[0] * 9 for _ in range(9)]
        hand = [0] * 14
        hand[0] = 2  # 先手 歩x2
        hand[4] = 1  # 先手 金x1
        hand[7] = 3  # 後手 歩x3
        hand[13] = 1  # 後手 飛x1
        sfen = board_id_positions_to_sfen(empty_board, hand)
        assert sfen.split(" ")[2] == "2PG3pr"

    def test_empty_hand_is_dash(self) -> None:
        empty_board = [[0] * 9 for _ in range(9)]
        sfen = board_id_positions_to_sfen(empty_board, [0] * 14)
        assert sfen.split(" ")[2] == "-"

    def test_turn_parameter(self) -> None:
        empty_board = [[0] * 9 for _ in range(9)]
        sfen = board_id_positions_to_sfen(
            empty_board, [0] * 14, turn=Turn.WHITE
        )
        assert sfen.split(" ")[1] == "w"

    def test_accepts_numpy_arrays(self) -> None:
        board = np.zeros((9, 9), dtype=np.uint8)
        board[0][8] = PieceId.OU + 14  # 後手玉 9一
        board[8][0] = PieceId.OU  # 先手玉 1九
        hand = np.zeros(14, dtype=np.uint8)
        sfen = board_id_positions_to_sfen(board, hand)
        assert sfen == "k8/9/9/9/9/9/9/9/8K b - 1"


class TestBoardFromBoardIdPositions:
    def test_roundtrip_initial_position(self) -> None:
        """初期局面の配列表現から同一盤面のBoardが再構築される．"""
        board = Board()
        reconstructed = Board.from_board_id_positions(
            board.get_normalized_board_id_positions(),
            board.get_normalized_pieces_in_hand(),
        )
        assert (
            reconstructed.get_sfen().split(" ")[0]
            == HIRATE_BOARD_SFEN
        )
        assert reconstructed.get_turn() == Turn.BLACK

    def test_roundtrip_with_hand_and_promoted(self) -> None:
        """成駒・持ち駒込みの局面がラウンドトリップで一致する．"""
        source = Board()
        source.set_sfen(
            "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/"
            "1P1Sb4/1KG2+p3/LN7 b R2GPrgsn4p 1"
        )
        reconstructed = Board.from_board_id_positions(
            source.get_board_id_positions(),
            np.concatenate(source.get_pieces_in_hand()).astype(
                np.uint8
            ),
        )
        assert reconstructed.get_sfen() == source.get_sfen()

    def test_invalid_piece_id_raises(self) -> None:
        with pytest.raises(ValueError):
            Board.from_board_id_positions(
                [[99] * 9] * 9, [0] * 14
            )
