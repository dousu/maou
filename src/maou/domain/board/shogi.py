from collections.abc import Generator
from enum import IntEnum

import cshogi
import numpy as np

MAX_PIECES_IN_HAND: list[int] = cshogi.MAX_PIECES_IN_HAND  # type: ignore
# 駒8種類，成駒6種類
PIECE_TYPES = 14

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

FEATURES_NUM = PIECE_TYPES * 2 + sum(MAX_PIECES_IN_HAND) * 2


class Turn(IntEnum):
    BLACK = cshogi.BLACK  # type: ignore
    WHITE = cshogi.WHITE  # type: ignore


class Result(IntEnum):
    BLACK_WIN = cshogi.BLACK_WIN  # type: ignore
    WHITE_WIN = cshogi.WHITE_WIN  # type: ignore
    DRAW = cshogi.DRAW  # type: ignore


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

    def get_hcp(self) -> np.ndarray:
        array = np.empty(1, dtype=cshogi.HuffmanCodedPos)  # type: ignore
        self.board.to_hcp(array)
        return array

    def to_hcp(self, array: np.ndarray) -> None:
        self.board.to_hcp(array)

    def get_legal_moves(self) -> Generator[int, None, None]:
        for move in self.board.legal_moves:
            yield move

    def get_move_from_move16(self, move16: int) -> int:
        return self.board.move_from_move16(move16)

    def push_move(self, move: int) -> None:
        self.board.push(move)

    def get_piece_planes(self) -> np.ndarray:
        array = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
        array.fill(0)
        self.to_piece_planes(array)
        return array

    def to_piece_planes(self, array: np.ndarray) -> None:
        self.board.piece_planes(array)

    def get_piece_planes_rotate(self) -> np.ndarray:
        array = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
        array.fill(0)
        self.to_piece_planes_rotate(array)
        return array

    def to_piece_planes_rotate(self, array: np.ndarray) -> None:
        self.board.piece_planes_rotate(array)

    def get_pieces_in_hand(self) -> tuple[list[int], list[int]]:
        """手番関係なく常に(先手, 後手)の順にtupleにはいっている"""
        return self.board.pieces_in_hand
