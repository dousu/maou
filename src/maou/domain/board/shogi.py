from __future__ import annotations

from collections.abc import Generator
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import cshogi
import numpy as np

if TYPE_CHECKING:
    import polars as pl

# Eager import for performance-critical paths
try:
    import polars as _pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False
    _pl = None  # type: ignore

MAX_PIECES_IN_HAND: list[int] = cshogi.MAX_PIECES_IN_HAND  # type: ignore
# 駒8種類，成駒6種類
PIECE_TYPES = len(cshogi.PIECE_TYPES)  # type: ignore

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

# 104
FEATURES_NUM = PIECE_TYPES * 2 + sum(MAX_PIECES_IN_HAND) * 2


class Turn(IntEnum):
    BLACK = cshogi.BLACK  # type: ignore
    WHITE = cshogi.WHITE  # type: ignore


class Result(IntEnum):
    BLACK_WIN = cshogi.BLACK_WIN  # type: ignore
    WHITE_WIN = cshogi.WHITE_WIN  # type: ignore
    DRAW = cshogi.DRAW  # type: ignore


class PieceId(IntEnum):
    EMPTY = 0
    # 歩
    FU = auto()
    # 香車
    KY = auto()
    # 桂馬
    KE = auto()
    # 銀
    GI = auto()
    # 金
    KI = auto()
    # 角
    KA = auto()
    # 飛車
    HI = auto()
    # 王
    OU = auto()
    # と金
    TO = auto()
    # 成香
    NKY = auto()
    # 成桂
    NKE = auto()
    # 成銀
    NGI = auto()
    # 馬
    UMA = auto()
    # 龍
    RYU = auto()


def move16(move: int) -> int:
    return cshogi.move16(move)  # type: ignore


def move_to(move: int) -> int:
    return cshogi.move_to(move)  # type: ignore


def move_from(move: int) -> int:
    return cshogi.move_from(move)  # type: ignore


def move_to_usi(move: int) -> str:
    return cshogi.move_to_usi(move)  # type: ignore


def move_is_drop(move: int) -> bool:
    return cshogi.move_is_drop(move)  # type: ignore


def move_is_promotion(move: int) -> bool:
    return cshogi.move_is_promotion(move)  # type: ignore


def move_drop_hand_piece(move: int) -> int:
    return cshogi.move_drop_hand_piece(move)  # type: ignore


class Board:
    def __init__(self) -> None:
        self.board = cshogi.Board()  # type: ignore

    def set_turn(self, turn: Turn) -> None:
        self.board.turn = turn.value

    def get_turn(self) -> Turn:
        return Turn(self.board.turn)

    def set_sfen(self, sfen: str) -> None:
        self.board.set_sfen(sfen)

    def get_sfen(self) -> str:
        return self.board.sfen()

    def set_hcp(self, hcp: np.ndarray) -> None:
        self.board.set_hcp(hcp)

    def to_hcp(self, array: np.ndarray) -> None:
        self.board.to_hcp(array)

    def get_legal_moves(self) -> Generator[int, None, None]:
        for move in self.board.legal_moves:
            yield move

    def get_move_from_move16(self, move16: int) -> int:
        return self.board.move_from_move16(move16)

    def push_move(self, move: int) -> None:
        self.board.push(move)

    def to_piece_planes(self, array: np.ndarray) -> None:
        self.board.piece_planes(array)

    def to_piece_planes_rotate(self, array: np.ndarray) -> None:
        self.board.piece_planes_rotate(array)

    def get_pieces_in_hand(self) -> tuple[list[int], list[int]]:
        """手番関係なく常に(先手, 後手)の順にtupleにはいっている
        歩，香車，桂馬，銀，金，角，飛車の順番
        例: ([0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0])
        """
        return self.board.pieces_in_hand

    def to_pretty_board(self) -> str:
        return str(self.board)

    def hash(self) -> int:
        return self.board.zobrist_hash()

    def get_board_id_positions_df(self) -> "pl.DataFrame":
        """Get board piece positions as 1-row Polars DataFrame．

        盤面の駒配置をPolars DataFrameで取得する．
        9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: boardIdPositions列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_board_id_positions_df()
            >>> len(df)
            1
            >>> df.schema
            {'boardIdPositions': List(List(UInt8))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: poetry add polars"
            )

        # Map cshogi piece IDs to PieceId enum values
        def map_cshogi_to_piece_id(cshogi_piece_id: int) -> int:
            if cshogi_piece_id < len(PieceId):
                return cshogi_piece_id
            else:
                return cshogi_piece_id - 16 + len(PieceId) - 1

        v_map = np.vectorize(
            map_cshogi_to_piece_id,
            otypes=[np.uint8],
        )
        positions = v_map(
            np.array(
                self.board.pieces,
                dtype=np.uint8,
            )
        ).reshape((9, 9))
        positions_list = positions.tolist()  # Fast conversion

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"boardIdPositions": [positions_list]},
            schema={"boardIdPositions": _pl.List(_pl.List(_pl.UInt8))},
        )

    def get_hcp_df(self) -> "pl.DataFrame":
        """Get HuffmanCodedPos as 1-row Polars DataFrame．

        HuffmanCodedPos形式の局面データをPolars DataFrameで取得する．
        32バイトのバイナリデータとして返す．

        Returns:
            pl.DataFrame: hcp列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            >>> df = board.get_hcp_df()
            >>> len(df)
            1
            >>> df.schema
            {'hcp': Binary}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: poetry add polars"
            )

        # Get HCP data from cshogi board
        hcp_array = np.empty(1, dtype=cshogi.HuffmanCodedPos)  # type: ignore
        self.board.to_hcp(hcp_array)
        hcp_bytes = hcp_array.tobytes()  # Convert to bytes

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"hcp": [hcp_bytes]}, schema={"hcp": _pl.Binary()}
        )

    def get_piece_planes_df(self) -> "pl.DataFrame":
        """Get piece feature planes as 1-row Polars DataFrame．

        駒の特徴平面をPolars DataFrameで取得する．
        104x9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: piecePlanes列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_piece_planes_df()
            >>> len(df)
            1
            >>> df.schema
            {'piecePlanes': List(List(List(Float32)))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: poetry add polars"
            )

        # Get piece planes from cshogi board
        planes = np.empty(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        planes.fill(0)
        self.board.piece_planes(planes)
        planes_list = planes.tolist()  # Fast conversion

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"piecePlanes": [planes_list]},
            schema={
                "piecePlanes": _pl.List(
                    _pl.List(_pl.List(_pl.Float32))
                )
            },
        )

    def get_piece_planes_rotate_df(self) -> "pl.DataFrame":
        """Get rotated piece feature planes as 1-row Polars DataFrame．

        回転された駒の特徴平面をPolars DataFrameで取得する．
        後手視点の104x9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: piecePlanes列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_piece_planes_rotate_df()
            >>> len(df)
            1
            >>> df.schema
            {'piecePlanes': List(List(List(Float32)))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: poetry add polars"
            )

        # Get rotated piece planes from cshogi board
        planes = np.empty(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        planes.fill(0)
        self.board.piece_planes_rotate(planes)
        planes_list = planes.tolist()  # Fast conversion

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"piecePlanes": [planes_list]},
            schema={
                "piecePlanes": _pl.List(
                    _pl.List(_pl.List(_pl.Float32))
                )
            },
        )
