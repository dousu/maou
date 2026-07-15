import logging
import re
from enum import IntEnum, auto

from maou._rust.maou_search import (
    move_label as _rust_move_label,
)
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


class IllegalMove(ValueError):
    """不正な指し手のラベル変換時に発生する例外．"""

    pass


logger: logging.Logger = logging.getLogger(__name__)

# 座標キャッシュ (decode で使用): square → (筋 x, 段 y)
_COORDINATE_CACHE: dict[int, tuple[int, int]] = {
    sq: divmod(sq, 9) for sq in range(81)
}

# 桂馬方向 (decode で使用)
_DIRECTION_KEIMA_LEFT = (1, -2)
_DIRECTION_KEIMA_RIGHT = (-1, -2)


def make_move_label(turn: shogi.Turn, move: int) -> int:
    """moveの教師データ (policy ラベル 0..MOVE_LABELS_NUM-1) を作成する.

    ラベル体系 (移動方向 10 種 × 成/不成 + 持ち駒 7 種) の算出は Rust
    (maou_search::label，maou._rust.maou_search.move_label) に委譲する．
    従来 label.py が純 Python で再実装していたエンコードロジックは Rust 側と
    parity 検証済み (rust/maou_search/tests/parity.rs の label_table_golden)．
    16-bit / 32-bit のどちらの move でも下位ビットからラベルが決まる．

    ラベルに変換できない指し手は IllegalMove を送出する．
    """
    try:
        return _rust_move_label(int(turn), move)
    except ValueError as e:
        raise IllegalMove(str(e)) from e


def make_result_value(turn: int, game_result: int) -> float:
    """Convert game result to value label for the current player.

    Args:
        turn: Current player turn (BLACK or WHITE)
        game_result: Game result in HCPE convention
            (0=DRAW, 1=BLACK_WIN, 2=WHITE_WIN — shogi.Result と同値)

    Returns:
        Value in {0, 0.5, 1} representing game outcome
        from current player's perspective (1=win, 0=loss, 0.5=draw)
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
        f" direction: {direction} to_sq: {to_sq}"
        f" turn: {board.get_turn()}"
    )


def _make_drop_move_from_label(
    board: shogi.Board,
    label: int,
) -> str:
    """駒打ちの逆変換."""
    turn = board.get_turn()
    # Determine hand piece index (0-6: 歩香桂銀金角飛)
    if label < MoveCategoryStartLabel.KY:
        hand_idx = 0  # FU
        offset = label - MoveCategoryStartLabel.FU
    elif label < MoveCategoryStartLabel.KE:
        hand_idx = 1  # KY
        offset = label - MoveCategoryStartLabel.KY
    elif label < MoveCategoryStartLabel.GI:
        hand_idx = 2  # KE
        offset = label - MoveCategoryStartLabel.KE
    elif label < MoveCategoryStartLabel.KI:
        hand_idx = 3  # GI
        offset = label - MoveCategoryStartLabel.GI
    elif label < MoveCategoryStartLabel.KA:
        hand_idx = 4  # KI
        offset = label - MoveCategoryStartLabel.KI
    elif label < MoveCategoryStartLabel.HI:
        hand_idx = 5  # KA
        offset = label - MoveCategoryStartLabel.KA
    else:
        hand_idx = 6  # HI
        offset = label - MoveCategoryStartLabel.HI

    # Decode target square based on piece type
    to_sq = _decode_drop_target(hand_idx, offset)

    # Handle turn-based coordinate transformation
    if turn == shogi.Turn.WHITE:
        to_sq = 80 - to_sq

    # Convert to USI format (駒文字は shogi.HAND_PIECE_SFEN_CHARS に一本化)
    piece_usi = shogi.HAND_PIECE_SFEN_CHARS[hand_idx]
    to_usi = _square_to_usi(to_sq)

    return f"{piece_usi}*{to_usi}"


def _decode_drop_target(hand_idx: int, offset: int) -> int:
    """駒打ち対象マスのデコード.

    Args:
        hand_idx: 持ち駒インデックス (0-6: 歩香桂銀金角飛)
        offset: 駒種カテゴリ先頭からのラベルオフセット
    """
    if hand_idx <= 1:
        # FU, KY: to_sq - (to_x + 1) = offset, y != 0
        for to_sq in range(81):
            to_x, to_y = _COORDINATE_CACHE[
                to_sq
            ]  # Use pre-computed cache
            if (
                to_y > 0 and to_sq - (to_x + 1) == offset
            ):  # y != 0
                return to_sq
    elif hand_idx == 2:
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

    raise ValueError(
        f"Invalid {shogi.HAND_PIECE_SFEN_CHARS[hand_idx]} drop"
        f" offset: {offset}"
    )


def _square_to_usi(square: int) -> str:
    """盤面座標をUSI形式に変換."""
    # square 0 = 1a, square 1 = 1b, square 9 = 2a, etc.
    file = (square // 9) + 1  # 1-9
    rank = chr(ord("a") + (square % 9))  # a-i
    return f"{file}{rank}"
