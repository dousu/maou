from __future__ import annotations

import logging

import numpy as np

from maou.domain.board import shogi
from maou.domain.board.shogi import RAW_PIECE_TO_PIECEID

logger: logging.Logger = logging.getLogger(__name__)


def _swap_piece_ids(board: np.ndarray) -> np.ndarray:
    """BLACK(1-14)とWHITE(15-28)の駒IDを入れ替える．

    正規化時に手番側の駒が常に1-14，
    相手側が15-28になるよう入れ替える．
    EMPTY(0)は変更しない．

    Args:
        board: 盤面配列(PieceId値)

    Returns:
        駒IDが入れ替えられた新しい配列
    """
    result = board.copy()
    black_mask = (result >= 1) & (result <= 14)
    white_mask = (result >= 15) & (result <= 28)
    result[black_mask] += 14
    result[white_mask] -= 14
    return result


def make_board_id_positions(board: shogi.Board) -> np.ndarray:
    """盤面の駒配置をPieceIdで表現する.
    後手番であれば180度回転し，駒IDを入れ替える．
    正規化後は手番側の駒が常に1-14，相手側が15-28となる．
    board.piecesから直接numpy fancy indexingで変換する
    (Polars DataFrame往復なし)．shapeが(9, 9)のuint8のndarrayで返す
    """
    raw_board = np.array(board.get_pieces(), dtype=np.uint8)
    converted = RAW_PIECE_TO_PIECEID[raw_board]
    board_id_positions = converted.reshape((9, 9), order="F")
    if board.get_turn() == shogi.Turn.BLACK:
        return board_id_positions
    else:
        rotated = np.rot90(board_id_positions, 2)
        return _swap_piece_ids(rotated)


def make_pieces_in_hand(board: shogi.Board) -> np.ndarray:
    """持ち駒の各駒種類の枚数を返す.
    手番の持ち駒が最初に入る
    shapeが(14,)のuint8のndarrayで返す
    """
    if board.get_turn() == shogi.Turn.BLACK:
        pieces_in_hand = board.get_pieces_in_hand()
    else:
        pieces_in_hand = tuple(
            reversed(board.get_pieces_in_hand())
        )
    return np.concatenate(pieces_in_hand).astype(np.uint8)
