import logging
from enum import IntEnum, auto
from typing import Dict, Tuple

import cshogi


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

# Pre-computed drop piece label offsets
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


def make_move_label(turn: int, move: int) -> int:
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
    if not cshogi.move_is_drop(move):  # type: ignore # 盤上の移動の場合
        return _process_board_move(turn, move)
    else:  # 駒打ちの場合
        return _process_drop_move(turn, move)


def _process_board_move(turn: int, move: int) -> int:
    """Process board moves using optimized lookup tables."""
    to_sq = cshogi.move_to(move)  # type: ignore
    from_sq = cshogi.move_from(move)  # type: ignore

    if turn == cshogi.WHITE:  # type: ignore
        to_sq = 80 - to_sq
        from_sq = 80 - from_sq

    # Use pre-computed coordinate cache
    to_x, to_y = _COORDINATE_CACHE[to_sq]
    from_x, from_y = _COORDINATE_CACHE[from_sq]

    # Calculate direction vector
    direction = (to_x - from_x, to_y - from_y)
    is_promotion = cshogi.move_is_promotion(move)  # type: ignore

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
    to_sq = cshogi.move_to(move)  # type: ignore

    if turn == cshogi.WHITE:  # type: ignore
        to_sq = 80 - to_sq

    # Use pre-computed coordinate cache
    to_x, to_y = _COORDINATE_CACHE[to_sq]
    hand_piece_raw = cshogi.move_drop_hand_piece(move)  # type: ignore
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
        case (cshogi.BLACK, cshogi.BLACK_WIN):  # type: ignore
            return 1
        case (cshogi.BLACK, cshogi.WHITE_WIN):  # type: ignore
            return 0.0
        case (cshogi.WHITE, cshogi.BLACK_WIN):  # type: ignore
            return 0.0
        case (cshogi.WHITE, cshogi.WHITE_WIN):  # type: ignore
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
            test_moves.append((cshogi.BLACK, move))  # type: ignore

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


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    benchmark_make_move_label()
