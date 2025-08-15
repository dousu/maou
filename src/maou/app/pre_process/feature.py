import logging

import cshogi
import numpy as np

# 駒8種類，成駒6種類
PIECE_TYPES = 14

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

FEATURES_NUM = (
    PIECE_TYPES * 2 + sum(cshogi.MAX_PIECES_IN_HAND) * 2  # type: ignore
)

logger: logging.Logger = logging.getLogger(__name__)


def make_feature(board: cshogi.Board) -> np.ndarray:  # type: ignore
    """Create feature representation of board position.

    Converts board state into neural network input features including
    piece positions and pieces in hand for both players.

    Args:
        board: Current board position

    Returns:
        Feature array with shape (FEATURES_NUM, 9, 9)
    """
    features = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
    features.fill(0)
    if board.turn == cshogi.BLACK:  # type: ignore
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 盤面の駒の数の分だけ最初の地点をずらす
    i = PIECE_TYPES * 2
    # 先手と後手の持ち駒数から特徴量を作成する
    for hands in pieces_in_hand:
        for num, max_num in zip(
            hands,
            cshogi.MAX_PIECES_IN_HAND,  # type: ignore
        ):
            # 全面1にする
            features[i : i + num].fill(1)
            i += max_num
    return features.astype(np.uint8)
