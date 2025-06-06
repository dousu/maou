import logging
from enum import IntEnum, auto

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


class IllegalMove(Exception):
    pass


logger: logging.Logger = logging.getLogger(__name__)


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
    # 後手の場合の盤の回転をこのようにすると成りの情報が消えるので注意すること
    # if turn == cshogi.WHITE:
    #     move = move_rotate(move)

    if not cshogi.move_is_drop(move):  # type: ignore # 盤上の移動の場合
        # 盤の座標を右上から下，左に向かって0からはじまる数字でシリアライズする
        # cshogiでは座標を直でだせないのでどちらにしろここから座標を計算する
        to_sq = cshogi.move_to(move)  # type: ignore
        from_sq = cshogi.move_from(move)  # type: ignore
        if turn == cshogi.WHITE:  # type: ignore
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # 移動方向を計算するために座標を計算する
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        logger.debug(
            f"from:{from_sq} ({from_x}, {from_y}), to:{to_sq} ({to_x}, {to_y})"
        )

        diff_x = to_x - from_x
        diff_y = to_y - from_y
        match (diff_x, diff_y):
            case (1, -2):
                # KEIMA_LEFT
                if to_y > 6 or to_x == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("KEIMA_LEFT")
                    if to_y < 2:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return (
                        MoveCategoryStartLabel.KEIMA_LEFT
                        + to_sq
                        - (to_x + 1) * 2
                        - to_x * 2
                        - 5
                    )
                else:
                    logger.debug("KEIMA_LEFT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return (
                        MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
                        + to_sq
                        - to_x * 6
                        - 3
                    )
            case (-1, -2):
                # KEIMA_RIGHT
                if to_y > 6 or to_x == 8:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("KEIMA_RIGHT")
                    if to_y < 2:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return (
                        MoveCategoryStartLabel.KEIMA_RIGHT
                        + to_sq
                        - (to_x + 1) * 2
                        - to_x * 2
                    )
                else:
                    logger.debug("KEIMA_RIGHT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return (
                        MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION + to_sq - to_x * 6
                    )
            case (0, diff_y) if diff_y < 0:
                # UP
                if to_y == 8:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("UP")
                    return MoveCategoryStartLabel.UP + to_sq - to_x
                else:
                    logger.debug("UP PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return MoveCategoryStartLabel.UP_PROMOTION + to_sq - to_x * 6
            case (0, diff_y) if diff_y > 0:
                # DOWN
                if to_y == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("DOWN")
                    return MoveCategoryStartLabel.DOWN + to_sq - (to_x + 1)
                else:
                    logger.debug("DOWN PROMOTION")
                    return MoveCategoryStartLabel.DOWN_PROMOTION + to_sq - (to_x + 1)
            case (diff_x, 0) if diff_x > 0:
                # LEFT
                if to_x == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("LEFT")
                    return MoveCategoryStartLabel.LEFT + to_sq - 9
                else:
                    logger.debug("LEFT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return MoveCategoryStartLabel.LEFT_PROMOTION + to_sq - to_x * 6 - 3
            case (diff_x, 0) if diff_x < 0:
                # RIGHT
                if to_x == 8:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("RIGHT")
                    return MoveCategoryStartLabel.RIGHT + to_sq
                else:
                    logger.debug("RIGHT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return MoveCategoryStartLabel.RIGHT_PROMOTION + to_sq - to_x * 6
            case (diff_x, diff_y) if diff_x > 0 and diff_y < 0:
                # UP_LEFT
                if to_y == 8 or to_x == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("UP_LEFT")
                    return MoveCategoryStartLabel.UP_LEFT + to_sq - to_x - 8
                else:
                    logger.debug("UP_LEFT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return (
                        MoveCategoryStartLabel.UP_LEFT_PROMOTION + to_sq - to_x * 6 - 3
                    )
            case (diff_x, diff_y) if diff_x < 0 and diff_y < 0:
                # UP_RIGHT
                if to_y == 8 or to_x == 8:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("UP_RIGHT")
                    return MoveCategoryStartLabel.UP_RIGHT + to_sq - to_x
                else:
                    logger.debug("UP_RIGHT PROMOTION")
                    if not to_y < 3:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    return MoveCategoryStartLabel.UP_RIGHT_PROMOTION + to_sq - to_x * 6
            case (diff_x, diff_y) if diff_x > 0 and diff_y > 0:
                # DOWN_LEFT
                if to_y == 0 or to_x == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("DOWN_LEFT")
                    return MoveCategoryStartLabel.DOWN_LEFT + to_sq - to_x - 9
                else:
                    logger.debug("DOWN_LEFT PROMOTION")
                    if 8 - to_y + to_x < 6:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    if to_x < 6:
                        return (
                            MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
                            + to_sq
                            - (to_x + 1)
                            - 2
                            - sum(range(7 - to_x, 7))
                        )
                    else:
                        return (
                            MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
                            + to_sq
                            - (to_x + 1)
                            - 2
                            - 21
                        )
            case (diff_x, diff_y) if diff_x < 0 and diff_y > 0:
                # DOWN_RIGHT
                if to_y == 0 or to_x == 8:
                    raise IllegalMove("Can not transform illegal move to move label.")
                if not cshogi.move_is_promotion(move):  # type: ignore
                    logger.debug("DOWN_RIGHT")
                    return MoveCategoryStartLabel.DOWN_RIGHT + to_sq - (to_x + 1)
                else:
                    logger.debug("DOWN_RIGHT PROMOTION")
                    if 8 - to_y + 8 - to_x < 6:
                        raise IllegalMove(
                            "Can not transform illegal move to move label."
                        )
                    if to_x > 2:
                        return (
                            MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
                            + to_sq
                            - (to_x + 1)
                            - sum(range(0, to_x - 2))
                        )
                    else:
                        return (
                            MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
                            + to_sq
                            - (to_x + 1)
                        )
            case _:
                raise IllegalMove("Can not transform illegal move to move label.")

    else:  # 駒打ちの場合
        to_sq = cshogi.move_to(move)  # type: ignore

        if turn == cshogi.WHITE:  # type: ignore
            to_sq = 80 - to_sq
        # 打てない領域だけラベリングをずらすので右上からの座標を計算しておく
        # 座標を計算しておくと簡単
        to_x, to_y = divmod(to_sq, 9)
        logger.debug(f"DROP to:{to_sq} ({to_x}, {to_y})")

        match cshogi.move_drop_hand_piece(move):  # type: ignore
            case HandPiece.FU:
                logger.debug("DROP FU")
                if to_y == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                return MoveCategoryStartLabel.FU + to_sq - (to_x + 1)
            case HandPiece.KY:
                logger.debug("DROP KY")
                if to_y == 0:
                    raise IllegalMove("Can not transform illegal move to move label.")
                return MoveCategoryStartLabel.KY + to_sq - (to_x + 1)
            case HandPiece.KE:
                logger.debug("DROP KE")
                if to_y < 2:
                    raise IllegalMove("Can not transform illegal move to move label.")
                return MoveCategoryStartLabel.KE + to_sq - (to_x + 1) * 2
            case HandPiece.GI:
                logger.debug("DROP GI")
                return MoveCategoryStartLabel.GI + to_sq
            case HandPiece.KI:
                logger.debug("DROP KI")
                return MoveCategoryStartLabel.KI + to_sq
            case HandPiece.KA:
                logger.debug("DROP KA")
                return MoveCategoryStartLabel.KA + to_sq
            case HandPiece.HI:
                logger.debug("DROP HI")
                return MoveCategoryStartLabel.HI + to_sq
            case _:
                raise IllegalMove("Can not transform illegal move to move label.")


def make_result_value(turn: int, game_result: int) -> float:
    """Convert game result to value label for the current player.
    
    Args:
        turn: Current player turn (BLACK or WHITE)
        game_result: Game result (BLACK_WIN, WHITE_WIN, or DRAW)
        
    Returns:
        Value between -1 and 1 representing game outcome from current player's perspective
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
