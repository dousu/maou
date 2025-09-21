import logging
import re
from enum import IntEnum, auto
from typing import Dict, Tuple

from maou.domain.board import shogi


# 駒の動きのラベリング
# 機械学習的には局面からこの種類のクラスタリング問題になる
# 座標間の組み合わせだと実現できない組み合わせが多すぎる．
# そのため，なるべく無駄のない分類になるように移動方向と移動先の座標の組み合わせにする
# これを選択したことによって駒の種類は無視できる
# 移動方向は8近傍と桂馬を特別扱いした10種類，
# 加えて成るか成らないかの選択があるので2倍する
# さらに，持ち駒7種類をそれぞれ別の移動方向とする
# ちなみに，桂馬は唯一駒を超えられるので特殊扱いしないと一意に駒が決まらなくなる
# このラベリングだとコードを書くのは簡単だが，無駄なラベルが多い．
# 例えば，盤の右端の座標は左，左上，左下の移動方向は存在しないし，
# 上方向の移動の場合は相手の陣地を出入りするのは相手の陣地に入るだけに限定される．
# 移動先の座標は81種類
# 大まかな構成
# - 盤上移動
#   - 移動方向
#     - 成らず
#       - 移動先
#     - 成り
#       - 移動先
# - 打ち
#   - 移動先
# 移動方向の順序: UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT
# 駒の順序: 歩，香，桂，銀，金，角，飛
# このラベリングによってあらかじめラベル数の計算が必要になる
# つまり，「ラベル数の計算」と「moveからのラベル算出」の両方テストが必要になる
# KEIMA_LEFTとKEIMA_RIGHTは1段目と2段目でならないという選択肢がないのに注意
class MoveCategoryStartLabel(IntEnum):
    UP = 0
    # UPは9マス不要
    UP_LEFT = UP + 81 - 9
    # UP_LEFTは17マス不要
    UP_RIGHT = UP_LEFT + 81 - 17
    # UP_RIGHTは17マス不要
    LEFT = UP_RIGHT + 81 - 17
    # LEFTは9マス不要
    RIGHT = LEFT + 81 - 9
    # RIGHTは9マス不要
    DOWN = RIGHT + 81 - 9
    # DOWNは9マス不要
    DOWN_LEFT = DOWN + 81 - 9
    # DOWN_LEFTは17マス不要
    DOWN_RIGHT = DOWN_LEFT + 81 - 17
    # DOWN_RIGHTは17マス不要
    KEIMA_LEFT = DOWN_RIGHT + 81 - 17
    # KEIMA_LEFTは41マス不要
    KEIMA_RIGHT = KEIMA_LEFT + 81 - 41
    # KEIMA_RIGHTは41マス不要
    UP_PROMOTION = KEIMA_RIGHT + 81 - 41
    # UP_PROMOTIONは54マス不要
    UP_LEFT_PROMOTION = UP_PROMOTION + 81 - 54
    # UP_LEFT_PROMOTIONは57マス不要
    UP_RIGHT_PROMOTION = UP_LEFT_PROMOTION + 81 - 57
    # UP_RIGHT_PROMOTIONは57マス不要
    LEFT_PROMOTION = UP_RIGHT_PROMOTION + 81 - 57
    # LEFT_PROMOTIONは57マス不要
    RIGHT_PROMOTION = LEFT_PROMOTION + 81 - 57
    # RIGHT_PROMOTIONは57マス不要
    DOWN_PROMOTION = RIGHT_PROMOTION + 81 - 57
    # DOWN_PROMOTIONは9マス不要
    DOWN_LEFT_PROMOTION = DOWN_PROMOTION + 81 - 9
    # DOWN_LEFT_PROMOTIONは32マス不要
    DOWN_RIGHT_PROMOTION = DOWN_LEFT_PROMOTION + 81 - 32
    # DOWN_RIGHT_PROMOTIONは32マス不要
    KEIMA_LEFT_PROMOTION = DOWN_RIGHT_PROMOTION + 81 - 32
    # KEIMA_LEFT_PROMOTIONは57マス不要
    KEIMA_RIGHT_PROMOTION = KEIMA_LEFT_PROMOTION + 81 - 57
    # 歩打
    # KEIMA_RIGHTは57マス不要
    FU = KEIMA_RIGHT_PROMOTION + 81 - 57
    # 香打
    # 歩打は9マス不要
    KY = FU + 81 - 9
    # 桂打
    # 香打は9マス不要
    KE = KY + 81 - 9
    # 銀打
    # 桂打は18マス不要
    GI = KE + 81 - 18
    # 金打
    KI = GI + 81
    # 角打
    KA = KI + 81
    # 飛打
    HI = KA + 81


class HandPiece(IntEnum):
    # 歩打
    FU = 0
    # 香打
    KY = auto()
    # 桂打
    KE = auto()
    # 銀打
    GI = auto()
    # 金打
    KI = auto()
    # 角打
    KA = auto()
    # 飛打
    HI = auto()


MOVE_LABELS_NUM = MoveCategoryStartLabel.HI + 81

# lamdba definition for drop piece label offsets
_DROP_LABEL_OFFSETS = {
    HandPiece.FU: lambda to_sq, to_x, to_y: to_sq - (to_x + 1),
    HandPiece.KY: lambda to_sq, to_x, to_y: to_sq - (to_x + 1),
    HandPiece.KE: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1) * 2,
    HandPiece.GI: lambda to_sq, to_x, to_y: to_sq,
    HandPiece.KI: lambda to_sq, to_x, to_y: to_sq,
    HandPiece.KA: lambda to_sq, to_x, to_y: to_sq,
    HandPiece.HI: lambda to_sq, to_x, to_y: to_sq,
}

# Pre-computed sum ranges for complex promotion calculations
# For DOWN_LEFT promotion: sum(range(7 - to_x, 7)) for to_x < 6
_DOWN_LEFT_RANGE_SUMS = {
    0: 0,  # sum(range(7, 7)) = 0
    1: 6,  # sum(range(6, 7)) = 6
    2: 11,  # sum(range(5, 7)) = 5 + 6 = 11
    3: 15,  # sum(range(4, 7)) = 4 + 5 + 6 = 15
    4: 18,  # sum(range(3, 7)) = 3 + 4 + 5 + 6 = 18
    5: 20,  # sum(range(2, 7)) = 2 + 3 + 4 + 5 + 6 = 20
}

# For DOWN_RIGHT promotion: sum(range(0, to_x - 2)) for to_x > 2
_DOWN_RIGHT_RANGE_SUMS = {
    3: 0,  # sum(range(0, 1)) = 0
    4: 1,  # sum(range(0, 2)) = 0 + 1 = 1
    5: 3,  # sum(range(0, 3)) = 0 + 1 + 2 = 3
    6: 6,  # sum(range(0, 4)) = 0 + 1 + 2 + 3 = 6
    7: 10,  # sum(range(0, 5)) = 0 + 1 + 2 + 3 + 4 = 10
    8: 15,  # sum(range(0, 6)) = 0 + 1 + 2 + 3 + 4 + 5 = 15
}


class IllegalMove(Exception):
    pass


logger: logging.Logger = logging.getLogger(__name__)


# Pre-computed lookup tables for performance optimization
_COORDINATE_CACHE: Dict[int, Tuple[int, int]] = {}
_MOVE_DIRECTION_CACHE: Dict[
    Tuple[int, int], Tuple[int, int]
] = {}

# Initialize coordinate cache for all 81 squares
for sq in range(81):
    _COORDINATE_CACHE[sq] = divmod(sq, 9)

# Pre-computed promotion zone checks
_PROMOTION_ZONE_Y = frozenset([0, 1, 2])
_NO_PROMOTION_ZONE_Y = frozenset([6, 7, 8])

# Move direction constants for faster lookups
_DIRECTION_KEIMA_LEFT = (1, -2)
_DIRECTION_KEIMA_RIGHT = (-1, -2)
_DIRECTION_UP = (0, -1)
_DIRECTION_DOWN = (0, 1)
_DIRECTION_LEFT = (1, 0)
_DIRECTION_RIGHT = (-1, 0)
_DIRECTION_UP_LEFT = (1, -1)
_DIRECTION_UP_RIGHT = (-1, -1)
_DIRECTION_DOWN_LEFT = (1, 1)
_DIRECTION_DOWN_RIGHT = (-1, 1)

# Pre-computed label offset calculations for each direction
_LABEL_OFFSETS = {
    _DIRECTION_UP: lambda to_sq, to_x, to_y: to_sq - to_x,
    _DIRECTION_DOWN: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1),
    _DIRECTION_LEFT: lambda to_sq, to_x, to_y: to_sq - 9,
    _DIRECTION_RIGHT: lambda to_sq, to_x, to_y: to_sq,
    _DIRECTION_UP_LEFT: lambda to_sq, to_x, to_y: to_sq
    - to_x
    - 8,
    _DIRECTION_UP_RIGHT: lambda to_sq, to_x, to_y: to_sq - to_x,
    _DIRECTION_DOWN_LEFT: lambda to_sq, to_x, to_y: to_sq
    - to_x
    - 9,
    _DIRECTION_DOWN_RIGHT: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1),
    _DIRECTION_KEIMA_LEFT: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1) * 2
    - to_x * 2
    - 5,
    _DIRECTION_KEIMA_RIGHT: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1) * 2
    - to_x * 2,
}

# Pre-computed promotion label offsets
_PROMOTION_LABEL_OFFSETS = {
    _DIRECTION_UP: lambda to_sq, to_x, to_y: to_sq - to_x * 6,
    _DIRECTION_LEFT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6
    - 3,
    _DIRECTION_RIGHT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6,
    _DIRECTION_UP_LEFT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6
    - 3,
    _DIRECTION_UP_RIGHT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6,
    _DIRECTION_DOWN: lambda to_sq, to_x, to_y: to_sq
    - (to_x + 1),
    _DIRECTION_KEIMA_LEFT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6
    - 3,
    _DIRECTION_KEIMA_RIGHT: lambda to_sq, to_x, to_y: to_sq
    - to_x * 6,
}


def make_move_label(turn: shogi.Turn, move: int) -> int:
    """moveの教師データ作成.
    入力値は32ビット (int)のmoveとしようと思っていたが，
    いくつかの値を試したところcshogiの中でどちらか判定して
    正しく処理しているようなので16ビットでも可 (32ビットに戻しても情報量は増えない)．
    このメソッドの出す種類数はMOVE_LABELS_NUMと連動していないといけない．
    最小値は0，最大値はMOVE_LABELS_NUM-1のintを返す．
    多クラス分類問題なのでここは無駄のないアルゴリズムにしないと，
    意味のない出力クラスができてしまい，
    不要にDeepLeaningの学習効率を悪くしてしまう．
    """
    if not shogi.move_is_drop(move):  # 盤上の移動の場合
        return _process_board_move(turn, move)
    else:  # 駒打ちの場合
        return _process_drop_move(turn, move)


def _process_board_move(turn: int, move: int) -> int:
    """Process board moves using optimized lookup tables."""
    to_sq = shogi.move_to(move)
    from_sq = shogi.move_from(move)

    if turn == shogi.Turn.WHITE:
        to_sq = 80 - to_sq
        from_sq = 80 - from_sq

    # Use pre-computed coordinate cache
    to_x, to_y = _COORDINATE_CACHE[to_sq]
    from_x, from_y = _COORDINATE_CACHE[from_sq]

    # Calculate direction vector
    direction = (to_x - from_x, to_y - from_y)
    is_promotion = shogi.move_is_promotion(move)

    # Handle each direction with optimized logic
    if direction == _DIRECTION_KEIMA_LEFT:
        return _process_keima_left(
            to_sq, to_x, to_y, is_promotion
        )
    elif direction == _DIRECTION_KEIMA_RIGHT:
        return _process_keima_right(
            to_sq, to_x, to_y, is_promotion
        )
    elif direction[0] == 0 and direction[1] < 0:  # UP
        return _process_up(to_sq, to_x, to_y, is_promotion)
    elif direction[0] == 0 and direction[1] > 0:  # DOWN
        return _process_down(to_sq, to_x, to_y, is_promotion)
    elif direction[1] == 0 and direction[0] > 0:  # LEFT
        return _process_left(to_sq, to_x, to_y, is_promotion)
    elif direction[1] == 0 and direction[0] < 0:  # RIGHT
        return _process_right(to_sq, to_x, to_y, is_promotion)
    elif direction[0] > 0 and direction[1] < 0:  # UP_LEFT
        return _process_up_left(to_sq, to_x, to_y, is_promotion)
    elif direction[0] < 0 and direction[1] < 0:  # UP_RIGHT
        return _process_up_right(
            to_sq, to_x, to_y, is_promotion
        )
    elif direction[0] > 0 and direction[1] > 0:  # DOWN_LEFT
        return _process_down_left(
            to_sq, to_x, to_y, is_promotion
        )
    elif direction[0] < 0 and direction[1] > 0:  # DOWN_RIGHT
        return _process_down_right(
            to_sq, to_x, to_y, is_promotion
        )
    else:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )


def _process_keima_left(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process KEIMA_LEFT moves."""
    if to_y > 6 or to_x == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        if to_y < 2:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.KEIMA_LEFT
            + _LABEL_OFFSETS[_DIRECTION_KEIMA_LEFT](
                to_sq, to_x, to_y
            )
        )
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_KEIMA_LEFT](
                to_sq, to_x, to_y
            )
        )


def _process_keima_right(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process KEIMA_RIGHT moves."""
    if to_y > 6 or to_x == 8:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        if to_y < 2:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.KEIMA_RIGHT
            + _LABEL_OFFSETS[_DIRECTION_KEIMA_RIGHT](
                to_sq, to_x, to_y
            )
        )
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_KEIMA_RIGHT](
                to_sq, to_x, to_y
            )
        )


def _process_up(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process UP moves."""
    if to_y == 8:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.UP + _LABEL_OFFSETS[
            _DIRECTION_UP
        ](to_sq, to_x, to_y)
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.UP_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_UP](
                to_sq, to_x, to_y
            )
        )


def _process_down(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process DOWN moves."""
    if to_y == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.DOWN + _LABEL_OFFSETS[
            _DIRECTION_DOWN
        ](to_sq, to_x, to_y)
    else:
        return (
            MoveCategoryStartLabel.DOWN_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_DOWN](
                to_sq, to_x, to_y
            )
        )


def _process_left(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process LEFT moves."""
    if to_x == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.LEFT + _LABEL_OFFSETS[
            _DIRECTION_LEFT
        ](to_sq, to_x, to_y)
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.LEFT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_LEFT](
                to_sq, to_x, to_y
            )
        )


def _process_right(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process RIGHT moves."""
    if to_x == 8:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.RIGHT + _LABEL_OFFSETS[
            _DIRECTION_RIGHT
        ](to_sq, to_x, to_y)
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.RIGHT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_RIGHT](
                to_sq, to_x, to_y
            )
        )


def _process_up_left(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process UP_LEFT moves."""
    if to_y == 8 or to_x == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.UP_LEFT + _LABEL_OFFSETS[
            _DIRECTION_UP_LEFT
        ](to_sq, to_x, to_y)
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.UP_LEFT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_UP_LEFT](
                to_sq, to_x, to_y
            )
        )


def _process_up_right(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process UP_RIGHT moves."""
    if to_y == 8 or to_x == 8:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return MoveCategoryStartLabel.UP_RIGHT + _LABEL_OFFSETS[
            _DIRECTION_UP_RIGHT
        ](to_sq, to_x, to_y)
    else:
        if to_y >= 3:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )
        return (
            MoveCategoryStartLabel.UP_RIGHT_PROMOTION
            + _PROMOTION_LABEL_OFFSETS[_DIRECTION_UP_RIGHT](
                to_sq, to_x, to_y
            )
        )


def _process_down_left(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process DOWN_LEFT moves."""
    if to_y == 0 or to_x == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return (
            MoveCategoryStartLabel.DOWN_LEFT
            + _LABEL_OFFSETS[_DIRECTION_DOWN_LEFT](
                to_sq, to_x, to_y
            )
        )
    else:
        if 8 - to_y + to_x < 6:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )

        # Use pre-computed range sums for better performance
        range_sum = _DOWN_LEFT_RANGE_SUMS.get(
            to_x, 21
        )  # Default to 21 for to_x >= 6
        return (
            MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
            + to_sq
            - (to_x + 1)
            - 2
            - range_sum
        )


def _process_down_right(
    to_sq: int, to_x: int, to_y: int, is_promotion: bool
) -> int:
    """Process DOWN_RIGHT moves."""
    if to_y == 0 or to_x == 8:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    if not is_promotion:
        return (
            MoveCategoryStartLabel.DOWN_RIGHT
            + _LABEL_OFFSETS[_DIRECTION_DOWN_RIGHT](
                to_sq, to_x, to_y
            )
        )
    else:
        if 8 - to_y + 8 - to_x < 6:
            raise IllegalMove(
                "Can not transform illegal move to move label."
            )

        # Use pre-computed range sums for better performance
        if to_x > 2:
            range_sum = _DOWN_RIGHT_RANGE_SUMS[to_x]
            return (
                MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
                + to_sq
                - (to_x + 1)
                - range_sum
            )
        else:
            return (
                MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
                + to_sq
                - (to_x + 1)
            )


def _process_drop_move(turn: int, move: int) -> int:
    """Process drop moves using optimized lookup tables."""
    to_sq = shogi.move_to(move)

    if turn == shogi.Turn.WHITE:
        to_sq = 80 - to_sq

    # Use pre-computed coordinate cache
    to_x, to_y = _COORDINATE_CACHE[to_sq]
    hand_piece_raw = shogi.move_drop_hand_piece(move)
    hand_piece = HandPiece(hand_piece_raw)

    # Validate drop constraints
    if hand_piece in (HandPiece.FU, HandPiece.KY) and to_y == 0:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )
    elif hand_piece == HandPiece.KE and to_y < 2:
        raise IllegalMove(
            "Can not transform illegal move to move label."
        )

    # Calculate label using lookup table
    base_label = getattr(
        MoveCategoryStartLabel, hand_piece.name
    )
    offset = _DROP_LABEL_OFFSETS[hand_piece](to_sq, to_x, to_y)
    return base_label + offset


def make_result_value(turn: int, game_result: int) -> float:
    """Convert game result to value label for the current player.

    Args:
        turn: Current player turn (BLACK or WHITE)
        game_result: Game result (BLACK_WIN, WHITE_WIN, or DRAW)

    Returns:
        Value between -1 and 1 representing game outcome
        from current player's perspective
    """
    match (turn, game_result):
        case (shogi.Turn.BLACK, shogi.Result.BLACK_WIN):
            return 1
        case (shogi.Turn.BLACK, shogi.Result.WHITE_WIN):
            return 0.0
        case (shogi.Turn.WHITE, shogi.Result.BLACK_WIN):
            return 0.0
        case (shogi.Turn.WHITE, shogi.Result.WHITE_WIN):
            return 1
        case _:
            return 0.5


def benchmark_make_move_label(
    num_iterations: int = 10000,
) -> None:
    """Benchmark the make_move_label function performance.

    This function can be used to measure performance improvements
    after optimization changes.

    Args:
        num_iterations: Number of iterations to run for benchmark
    """
    import random
    import time

    # Generate some test moves (simplified for benchmark)
    test_moves = []
    for _ in range(100):
        # Generate random board moves
        from_sq = random.randint(0, 80)
        to_sq = random.randint(0, 80)
        if from_sq != to_sq:
            # Create a simple move (this is a simplified test)
            move = (from_sq << 7) | to_sq
            test_moves.append((shogi.Turn.BLACK, move))

    logger.info(
        f"Starting benchmark with {num_iterations} iterations..."
    )
    start_time = time.perf_counter()

    for i in range(num_iterations):
        turn, move = test_moves[i % len(test_moves)]
        try:
            make_move_label(turn, move)
        except IllegalMove:
            # Expected for some random moves
            pass

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info("Benchmark completed:")
    logger.info(f"  Total time: {elapsed_time:.4f} seconds")
    logger.info(f"  Iterations: {num_iterations}")
    time_per_call = (
        float(elapsed_time) / float(num_iterations) * 1000
    )
    logger.info(f"  Time per call: {time_per_call:.4f} ms")
    logger.info(
        f"  Calls per second: {float(num_iterations) / elapsed_time:.0f}"
    )


def make_usi_move_from_label(
    board: shogi.Board, label: int
) -> str:
    """ラベルから指し手への逆変換.

    Args:
        board: 局面情報 (maou.domain.board.shogi.Board)
        label: 指し手ラベル (0からMOVE_LABELS_NUM-1)

    Returns:
        USI形式の指し手文字列 (例: "1b1a", "S*5b")

    Raises:
        ValueError: 無効なラベル値，または無効なUSI形式の場合
    """
    if label < 0 or label >= MOVE_LABELS_NUM:
        raise ValueError(f"Invalid label: {label}")
    else:
        # Drop moves check - they start from FU
        if label >= MoveCategoryStartLabel.FU:
            usi_move = _make_drop_move_from_label(board, label)
        else:
            usi_move = _make_board_move_from_label(board, label)

    # Validate USI format: %d%w%d%w(\+)?
    # Pattern: digit + letter + digit + letter + optional '+'
    usi_pattern = r"^\d[a-i]\d[a-i]\+?$|^[A-Z]\*\d[a-i]$"

    if not re.match(usi_pattern, usi_move):
        raise ValueError(
            f"Invalid USI format: {usi_move} (expected format: %d%w%d%w(\\+)? for moves or %w*%d%w for drops)"
        )

    return usi_move


def _make_board_move_from_label(
    board: shogi.Board,
    label: int,
) -> str:
    """盤上移動の逆変換."""
    # Determine move category and offset
    if label < MoveCategoryStartLabel.UP_LEFT:
        # UP
        direction = "UP"
        offset = label - MoveCategoryStartLabel.UP
        is_promotion = False
    elif label < MoveCategoryStartLabel.UP_RIGHT:
        # UP_LEFT
        direction = "UP_LEFT"
        offset = label - MoveCategoryStartLabel.UP_LEFT
        is_promotion = False
    elif label < MoveCategoryStartLabel.LEFT:
        # UP_RIGHT
        direction = "UP_RIGHT"
        offset = label - MoveCategoryStartLabel.UP_RIGHT
        is_promotion = False
    elif label < MoveCategoryStartLabel.RIGHT:
        # LEFT
        direction = "LEFT"
        offset = label - MoveCategoryStartLabel.LEFT
        is_promotion = False
    elif label < MoveCategoryStartLabel.DOWN:
        # RIGHT
        direction = "RIGHT"
        offset = label - MoveCategoryStartLabel.RIGHT
        is_promotion = False
    elif label < MoveCategoryStartLabel.DOWN_LEFT:
        # DOWN
        direction = "DOWN"
        offset = label - MoveCategoryStartLabel.DOWN
        is_promotion = False
    elif label < MoveCategoryStartLabel.DOWN_RIGHT:
        # DOWN_LEFT
        direction = "DOWN_LEFT"
        offset = label - MoveCategoryStartLabel.DOWN_LEFT
        is_promotion = False
    elif label < MoveCategoryStartLabel.KEIMA_LEFT:
        # DOWN_RIGHT
        direction = "DOWN_RIGHT"
        offset = label - MoveCategoryStartLabel.DOWN_RIGHT
        is_promotion = False
    elif label < MoveCategoryStartLabel.KEIMA_RIGHT:
        # KEIMA_LEFT
        direction = "KEIMA_LEFT"
        offset = label - MoveCategoryStartLabel.KEIMA_LEFT
        is_promotion = False
    elif label < MoveCategoryStartLabel.UP_PROMOTION:
        # KEIMA_RIGHT
        direction = "KEIMA_RIGHT"
        offset = label - MoveCategoryStartLabel.KEIMA_RIGHT
        is_promotion = False
    elif label < MoveCategoryStartLabel.UP_LEFT_PROMOTION:
        # UP_PROMOTION
        direction = "UP"
        offset = label - MoveCategoryStartLabel.UP_PROMOTION
        is_promotion = True
    elif label < MoveCategoryStartLabel.UP_RIGHT_PROMOTION:
        # UP_LEFT_PROMOTION
        direction = "UP_LEFT"
        offset = (
            label - MoveCategoryStartLabel.UP_LEFT_PROMOTION
        )
        is_promotion = True
    elif label < MoveCategoryStartLabel.LEFT_PROMOTION:
        # UP_RIGHT_PROMOTION
        direction = "UP_RIGHT"
        offset = (
            label - MoveCategoryStartLabel.UP_RIGHT_PROMOTION
        )
        is_promotion = True
    elif label < MoveCategoryStartLabel.RIGHT_PROMOTION:
        # LEFT_PROMOTION
        direction = "LEFT"
        offset = label - MoveCategoryStartLabel.LEFT_PROMOTION
        is_promotion = True
    elif label < MoveCategoryStartLabel.DOWN_PROMOTION:
        # RIGHT_PROMOTION
        direction = "RIGHT"
        offset = label - MoveCategoryStartLabel.RIGHT_PROMOTION
        is_promotion = True
    elif label < MoveCategoryStartLabel.DOWN_LEFT_PROMOTION:
        # DOWN_PROMOTION
        direction = "DOWN"
        offset = label - MoveCategoryStartLabel.DOWN_PROMOTION
        is_promotion = True
    elif label < MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION:
        # DOWN_LEFT_PROMOTION
        direction = "DOWN_LEFT"
        offset = (
            label - MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
        )
        is_promotion = True
    elif label < MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION:
        # DOWN_RIGHT_PROMOTION
        direction = "DOWN_RIGHT"
        offset = (
            label - MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
        )
        is_promotion = True
    elif label < MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION:
        # KEIMA_LEFT_PROMOTION
        direction = "KEIMA_LEFT"
        offset = (
            label - MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
        )
        is_promotion = True
    else:
        # KEIMA_RIGHT_PROMOTION
        direction = "KEIMA_RIGHT"
        offset = (
            label - MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION
        )
        is_promotion = True

    return _decode_board_move(
        board, direction, offset, is_promotion
    )


def _decode_board_move(
    board: shogi.Board,
    direction: str,
    offset: int,
    is_promotion: bool,
) -> str:
    """盤上移動のデコード処理."""
    # Direction-specific decoding
    if direction == "UP":
        to_sq = _decode_up_move(offset, is_promotion)
    elif direction == "DOWN":
        to_sq = _decode_down_move(offset, is_promotion)
    elif direction == "LEFT":
        to_sq = _decode_left_move(offset, is_promotion)
    elif direction == "RIGHT":
        to_sq = _decode_right_move(offset, is_promotion)
    elif direction == "UP_LEFT":
        to_sq = _decode_up_left_move(offset, is_promotion)
    elif direction == "UP_RIGHT":
        to_sq = _decode_up_right_move(offset, is_promotion)
    elif direction == "DOWN_LEFT":
        to_sq = _decode_down_left_move(offset, is_promotion)
    elif direction == "DOWN_RIGHT":
        to_sq = _decode_down_right_move(offset, is_promotion)
    elif direction == "KEIMA_LEFT":
        to_sq = _decode_keima_left_move(offset, is_promotion)
    elif direction == "KEIMA_RIGHT":
        to_sq = _decode_keima_right_move(offset, is_promotion)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    from_sq, to_sq = _find_move_from_sq(
        board,
        to_sq,
        direction,
    )

    # # Handle turn-based coordinate transformation
    # if board.get_turn() == shogi.Turn.WHITE:
    #     to_sq = 80 - to_sq
    #     from_sq = 80 - from_sq

    # Convert to USI format
    to_usi = _square_to_usi(to_sq)
    from_usi = _square_to_usi(from_sq)

    if is_promotion:
        return f"{from_usi}{to_usi}+"
    else:
        return f"{from_usi}{to_usi}"


def _decode_up_move(offset: int, is_promotion: bool) -> int:
    """UP方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8)
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6
        return to_sq


def _decode_down_move(offset: int, is_promotion: bool) -> int:
    """DOWN方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8) + 1
        return to_sq
    else:
        # Promotion: any valid position
        to_sq = offset + (offset // 8) + 1
        return to_sq


def _decode_left_move(offset: int, is_promotion: bool) -> int:
    """LEFT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + 9
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6 + 9
        return to_sq


def _decode_right_move(offset: int, is_promotion: bool) -> int:
    """RIGHT方向の移動デコード."""
    if not is_promotion:
        # to_sq = offset (from _LABEL_OFFSETS)
        to_sq = offset
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6
        return to_sq


def _decode_up_left_move(
    offset: int, is_promotion: bool
) -> int:
    """UP_LEFT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8) + 9
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6 + 9
        return to_sq


def _decode_up_right_move(
    offset: int, is_promotion: bool
) -> int:
    """UP_RIGHT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8)
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6
        return to_sq


def _decode_down_left_move(
    offset: int, is_promotion: bool
) -> int:
    """DOWN_LEFT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8) + 9
        return to_sq
    else:
        # Complex promotion calculation with range sums
        if offset < 3:
            range_sum = 1
        elif offset < 7:
            range_sum = 7
        elif offset < 12:
            range_sum = 12
        elif offset < 18:
            range_sum = 16
        elif offset < 25:
            range_sum = 19
        elif offset < 33:
            range_sum = 21
        else:
            range_sum = 22 + ((offset - 33) // 8)
        to_sq = offset + range_sum + 9
        return to_sq


def _decode_down_right_move(
    offset: int, is_promotion: bool
) -> int:
    """DOWN_RIGHT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 8) + 1
        return to_sq
    else:
        # Complex promotion calculation
        if offset < 31:
            range_sum = (offset // 8) + 1
        elif offset < 37:
            range_sum = (offset // 8) + 2
        elif offset < 42:
            range_sum = (offset // 8) + 4
        elif offset < 46:
            range_sum = (offset // 8) + 7
        else:
            range_sum = (offset // 8) + 11
        to_sq = offset + range_sum
        return to_sq


def _decode_keima_left_move(
    offset: int, is_promotion: bool
) -> int:
    """KEIMA_LEFT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 5) * 4 + 11
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6 + 9
        return to_sq


def _decode_keima_right_move(
    offset: int, is_promotion: bool
) -> int:
    """KEIMA_RIGHT方向の移動デコード."""
    if not is_promotion:
        to_sq = offset + (offset // 5) * 4 + 2
        return to_sq
    else:
        # Promotion: to_y < 3
        to_sq = offset + (offset // 3) * 6
        return to_sq


def _find_move_from_sq(
    board: shogi.Board, to_sq: int, direction: str
) -> tuple[int, int]:
    for move in board.get_legal_moves():
        legal_move_to_sq = shogi.move_to(move)
        legal_move_from_sq = shogi.move_from(move)
        logger.debug(
            f"{shogi.move_to_usi(move)} ({legal_move_to_sq}, {legal_move_from_sq})"
        )

        # 駒打ちは捜索範囲外なのでスキップ
        if not 0 < legal_move_from_sq < 81:
            continue

        if board.get_turn() == shogi.Turn.WHITE:
            legal_move_to_sq = 80 - legal_move_to_sq
            legal_move_from_sq = 80 - legal_move_from_sq

        # Use pre-computed coordinate cache
        to_x, to_y = _COORDINATE_CACHE[legal_move_to_sq]
        from_x, from_y = _COORDINATE_CACHE[legal_move_from_sq]

        # Calculate direction vector
        direction_tuple = (to_x - from_x, to_y - from_y)

        if (
            direction == "KEIMA_LEFT"
            and to_sq == legal_move_to_sq
            and direction_tuple == _DIRECTION_KEIMA_LEFT
        ):
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "KEIMA_RIGHT"
            and to_sq == legal_move_to_sq
            and direction_tuple == _DIRECTION_KEIMA_RIGHT
        ):
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "UP"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] == 0
            and direction_tuple[1] < 0
        ):  # UP
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "DOWN"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] == 0
            and direction_tuple[1] > 0
        ):  # DOWN
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "LEFT"
            and to_sq == legal_move_to_sq
            and direction_tuple[1] == 0
            and direction_tuple[0] > 0
        ):  # LEFT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "RIGHT"
            and to_sq == legal_move_to_sq
            and direction_tuple[1] == 0
            and direction_tuple[0] < 0
        ):  # RIGHT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "UP_LEFT"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] > 0
            and direction_tuple[1] < 0
        ):  # UP_LEFT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "UP_RIGHT"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] < 0
            and direction_tuple[1] < 0
        ):  # UP_RIGHT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "DOWN_LEFT"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] > 0
            and direction_tuple[1] > 0
        ):  # DOWN_LEFT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
        elif (
            direction == "DOWN_RIGHT"
            and to_sq == legal_move_to_sq
            and direction_tuple[0] < 0
            and direction_tuple[1] > 0
        ):  # DOWN_RIGHT
            if board.get_turn() == shogi.Turn.WHITE:
                legal_move_to_sq = 80 - legal_move_to_sq
                legal_move_from_sq = 80 - legal_move_from_sq
            return legal_move_from_sq, legal_move_to_sq
    raise IllegalMove(
        "Can not transform illegal move to move label."
    )


def _make_drop_move_from_label(
    board: shogi.Board,
    label: int,
) -> str:
    turn = board.get_turn()
    """駒打ちの逆変換."""
    # Determine piece type
    if label < MoveCategoryStartLabel.KY:
        piece = "FU"
        offset = label - MoveCategoryStartLabel.FU
    elif label < MoveCategoryStartLabel.KE:
        piece = "KY"
        offset = label - MoveCategoryStartLabel.KY
    elif label < MoveCategoryStartLabel.GI:
        piece = "KE"
        offset = label - MoveCategoryStartLabel.KE
    elif label < MoveCategoryStartLabel.KI:
        piece = "GI"
        offset = label - MoveCategoryStartLabel.GI
    elif label < MoveCategoryStartLabel.KA:
        piece = "KI"
        offset = label - MoveCategoryStartLabel.KI
    elif label < MoveCategoryStartLabel.HI:
        piece = "KA"
        offset = label - MoveCategoryStartLabel.KA
    else:
        piece = "HI"
        offset = label - MoveCategoryStartLabel.HI

    # Decode target square based on piece type
    to_sq = _decode_drop_target(piece, offset)

    # Handle turn-based coordinate transformation
    if turn == shogi.Turn.WHITE:
        to_sq = 80 - to_sq

    # Convert to USI format
    piece_usi = {
        "FU": "P",
        "KY": "L",
        "KE": "N",
        "GI": "S",
        "KI": "G",
        "KA": "B",
        "HI": "R",
    }[piece]
    to_usi = _square_to_usi(to_sq)

    return f"{piece_usi}*{to_usi}"


def _decode_drop_target(piece: str, offset: int) -> int:
    """駒打ち対象マスのデコード."""
    if piece in ["FU", "KY"]:
        # FU, KY: to_sq - (to_x + 1) = offset, y != 0
        for to_sq in range(81):
            to_x, to_y = _COORDINATE_CACHE[
                to_sq
            ]  # Use pre-computed cache
            if (
                to_y > 0 and to_sq - (to_x + 1) == offset
            ):  # y != 0
                return to_sq
    elif piece == "KE":
        # KE: to_sq - (to_x + 1) * 2 = offset, y >= 2
        for to_sq in range(81):
            to_x, to_y = _COORDINATE_CACHE[
                to_sq
            ]  # Use pre-computed cache
            if to_y >= 2 and to_sq - (to_x + 1) * 2 == offset:
                return to_sq
    else:
        # GI, KI, KA, HI: to_sq = offset
        to_sq = offset
        if 0 <= to_sq <= 80:
            return to_sq

    raise ValueError(f"Invalid {piece} drop offset: {offset}")


def _square_to_usi(square: int) -> str:
    """盤面座標をUSI形式に変換."""
    # square 0 = 1a, square 1 = 1b, square 9 = 2a, etc.
    file = (square // 9) + 1  # 1-9
    rank = chr(ord("a") + (square % 9))  # a-i
    return f"{file}{rank}"


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    benchmark_make_move_label()
