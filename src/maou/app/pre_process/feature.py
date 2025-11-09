import logging

import numpy as np

from maou.domain.board import shogi

logger: logging.Logger = logging.getLogger(__name__)


def make_feature(board: shogi.Board) -> np.ndarray:
    """Create feature representation of board position.

    Converts board state into neural network input features including
    piece positions and pieces in hand for both players.

    Args:
        board: Current board position

    Returns:
        Feature array with shape (FEATURES_NUM, 9, 9)
    """
    features = np.empty(
        (shogi.FEATURES_NUM, 9, 9), dtype=np.float32
    )
    features.fill(0)
    if board.get_turn() == shogi.Turn.BLACK:
        board.to_piece_planes(features)
        pieces_in_hand = board.get_pieces_in_hand()
    else:
        board.to_piece_planes_rotate(features)
        pieces_in_hand = reversed(board.get_pieces_in_hand())
    # 盤面の駒の数の分だけ最初の地点をずらす
    i = shogi.PIECE_TYPES * 2
    # 先手と後手の持ち駒数から特徴量を作成する
    for hands in pieces_in_hand:
        for num, max_num in zip(
            hands,
            shogi.MAX_PIECES_IN_HAND,
        ):
            # 全面1にする
            features[i : i + num].fill(1)
            i += max_num
    return features.astype(np.uint8)


def make_board_id_positions(board: shogi.Board) -> np.ndarray:
    """盤面の駒配置をPieceIdで表現する.
    後手番であれば180度回転する
    shapeが(9, 9)のuint8のndarrayで返す
    """
    board_id_positions = board.get_board_id_positions()
    if board.get_turn() == shogi.Turn.BLACK:
        return board_id_positions
    else:
        return np.rot90(board_id_positions, 2)


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
