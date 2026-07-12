from __future__ import annotations

import logging

import numpy as np

from maou.domain.board import shogi

logger: logging.Logger = logging.getLogger(__name__)

# Numpy lookup table for cshogi piece ID → PieceId conversion．
# board.pieces は 0-30 の値を返す（15,16は未使用）．
# np.vectorize + dict を廃止し，numpy fancy indexing で一括変換する．
_MAX_CSHOGI_PIECE_ID = 30
_PIECE_ID_TABLE = np.zeros(
    _MAX_CSHOGI_PIECE_ID + 1, dtype=np.uint8
)
_PIECE_ID_TABLE[0] = 0  # EMPTY
_PIECE_ID_TABLE[1] = 1  # BPAWN → FU
_PIECE_ID_TABLE[2] = 2  # BLANCE → KY
_PIECE_ID_TABLE[3] = 3  # BKNIGHT → KE
_PIECE_ID_TABLE[4] = 4  # BSILVER → GI
_PIECE_ID_TABLE[5] = 6  # BBISHOP → KA
_PIECE_ID_TABLE[6] = 7  # BROOK → HI
_PIECE_ID_TABLE[7] = 5  # BGOLD → KI
_PIECE_ID_TABLE[8] = 8  # BKING → OU
_PIECE_ID_TABLE[9] = 9  # BPROM_PAWN → TO
_PIECE_ID_TABLE[10] = 10  # BPROM_LANCE → NKY
_PIECE_ID_TABLE[11] = 11  # BPROM_KNIGHT → NKE
_PIECE_ID_TABLE[12] = 12  # BPROM_SILVER → NGI
_PIECE_ID_TABLE[13] = 13  # BPROM_BISHOP → UMA
_PIECE_ID_TABLE[14] = 14  # BPROM_ROOK → RYU
_PIECE_ID_TABLE[17] = 15  # WPAWN → FU + 14
_PIECE_ID_TABLE[18] = 16  # WLANCE → KY + 14
_PIECE_ID_TABLE[19] = 17  # WKNIGHT → KE + 14
_PIECE_ID_TABLE[20] = 18  # WSILVER → GI + 14
_PIECE_ID_TABLE[21] = 20  # WBISHOP → KA + 14
_PIECE_ID_TABLE[22] = 21  # WROOK → HI + 14
_PIECE_ID_TABLE[23] = 19  # WGOLD → KI + 14
_PIECE_ID_TABLE[24] = 22  # WKING → OU + 14
_PIECE_ID_TABLE[25] = 23  # WPROM_PAWN → TO + 14
_PIECE_ID_TABLE[26] = 24  # WPROM_LANCE → NKY + 14
_PIECE_ID_TABLE[27] = 25  # WPROM_KNIGHT → NKE + 14
_PIECE_ID_TABLE[28] = 26  # WPROM_SILVER → NGI + 14
_PIECE_ID_TABLE[29] = 27  # WPROM_BISHOP → UMA + 14
_PIECE_ID_TABLE[30] = 28  # WPROM_ROOK → RYU + 14


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
    converted = _PIECE_ID_TABLE[raw_board]
    board_id_positions = converted.reshape((9, 9), order="F")
    if board.get_turn() == shogi.Turn.BLACK:
        return board_id_positions
    else:
        rotated = np.rot90(board_id_positions, 2)
        return _swap_piece_ids(rotated)


# 旧名の後方互換エイリアス (slow/fast の区別は解消済み)
make_board_id_positions_fast = make_board_id_positions


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
