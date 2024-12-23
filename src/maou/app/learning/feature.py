import logging
from enum import IntEnum, auto

import cshogi
import numpy as np
import torch
from cshogi import (
    Board,
    move_drop_hand_piece,
    move_from,
    move_is_drop,
    move_is_promotion,
    move_to,
)

# 駒8種類，成駒6種類
PIECE_TYPES = 14

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

FEATURES_NUM = PIECE_TYPES * 2 + sum(cshogi.MAX_PIECES_IN_HAND) * 2

# 駒の動きのパターン数
# 機械学習的には局面からこの種類のクラスタリング問題になる
# 座標間の組み合わせだと実現できない組み合わせが多すぎる．
# そのため，なるべく無駄のない分類になるように移動方向と移動先の座標の組み合わせにする
# これを選択したことによって駒の種類は無視できる
# 移動方向は8近傍と桂馬を特別扱いした10種類，
# 加えて成るか成らないかの選択があるので2倍する
# さらに，持ち駒7種類をそれぞれ別の移動方向とする
# ちなみに，桂馬は唯一駒を超えられるので特殊扱いしないと一意に駒が決まらなくなる
# 移動先の座標は81種類
MOVE_PLANES_NUM = 10 * 2 + 7
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81


class MOVE_DIRECTION(IntEnum):
    UP = 0
    UP_LEFT = auto()
    UP_RIGHT = auto()
    LEFT = auto()
    RIGHT = auto()
    DOWN = auto()
    DOWN_LEFT = auto()
    DOWN_RIGHT = auto()
    KEIMA_LEFT = auto()
    KEIMA_RIGHT = auto()
    # UP_PROMOTE = auto()
    # UP_LEFT_PROMOTE = auto()
    # UP_RIGHT_PROMOTE = auto()
    # LEFT_PROMOTE = auto()
    # RIGHT_PROMOTE = auto()
    # DOWN_PROMOTE = auto()
    # DOWN_LEFT_PROMOTE = auto()
    # DOWN_RIGHT_PROMOTE = auto()
    # KEIMMA_LEFT_PROMOTE = auto()
    # KEIMMA_RIGHT_PROMOTE = auto()


class Transform:
    logger: logging.Logger = logging.getLogger(__name__)

    def __call__(
        self,
        hcp: np.ndarray,
        move16: int,
        game_result: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        board = Board()
        board.set_hcp(hcp)

        # 入力特徴量
        features = self.__make_feature(board)

        # 教師データ
        # move label
        # この変換は必要なさそうなので行わない
        # move = move_from_move16(move16)
        move_label = self.__make_move_label(board.turn, move16)

        # result value
        result_value = self.__make_result_value(board.turn, game_result)

        return (
            torch.tensor(features, dtype=torch.float32, pin_memory=True),
            (
                torch.tensor(move_label, dtype=torch.int64, pin_memory=True),
                torch.tensor(result_value, dtype=torch.float32, pin_memory=True),
            ),
        )

    def __make_feature(self, board: Board):
        features = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
        features.fill(0)
        if board.turn == cshogi.BLACK:
            board.piece_planes(features)
            pieces_in_hand = board.pieces_in_hand
        else:
            board.piece_planes_rotate(features)
            pieces_in_hand = reversed(board.pieces_in_hand)
        # 盤面の駒の数の分だけ最初の地点をずらす
        i = PIECE_TYPES * 2
        # 先手と後手の持ち駒数から特徴量を作成する
        for hands in pieces_in_hand:
            for num, max_num in zip(hands, cshogi.MAX_PIECES_IN_HAND):
                # 全面1にする
                features[i : i + num].fill(1)
                i += max_num
        return features

    def __make_move_label(self, turn: int, move: int) -> int:
        """moveの教師データ作成.
        入力値は32ビット (int)のmoveとしようと思っていたが，
        いくつかの値を試したところcshogiの中でどちらか判定して
        正しく処理しているようなので16ビットでも可 (32ビットに戻しても情報量は増えない)．
        このメソッドの出す種類数はMOVE_LABELS_NUMと連動していないといけない．
        最小値は0，最大値はMOVE_LABELS_NUM-1のintを返す．
        多クラス分類問題なのでここは無駄のないアルゴリズムにしないと，
        意味のない出力クラスができてしまい，
        不要にDeepLeaningの学習効率を悪くしてしまう．
        現状のロジックだと本来成れない動きでも成りのラベルは存在するし，
        移動できない方向から移動するラベルも存在していて無駄が多い．
        シンプルに無駄のない正解ラベルを作るとしたらあり得る動きのUSI指し手とのマッピングを作るとかか？
        """
        # 後手の場合の盤の回転をこのようにすると成りの情報が消えるので注意すること
        # if turn == cshogi.WHITE:
        #     move = move_rotate(move)

        move_direction: int

        # 移動方向と移動後の81マスの組み合わせをシリアライズする
        # 移動方向はMOVE_DIRECTIONで定義してあり，
        # 駒を打つ場合は駒の種類を移動方向として数える．
        # シリアライズの方法としては各移動方向ごとで81ずらしていく．
        # 移動方向の種類数よりも盤面のマスの数の方が値が変更しそうにないため．
        if not move_is_drop(move):  # 盤上の移動の場合
            # 盤の座標を右上から下，左に向かって0からはじまる数字でシリアライズする
            # cshogiでは座標を直でだせないのでどちらにしろここから座標を計算する
            to_sq = move_to(move)
            from_sq = move_from(move)
            if turn == cshogi.WHITE:
                to_sq = 80 - to_sq
                from_sq = 80 - from_sq

            self.logger.debug(f"from:{from_sq}, to:{to_sq}")

            # 移動方向を計算するために座標を計算する
            to_x, to_y = divmod(to_sq, 9)
            from_x, from_y = divmod(from_sq, 9)
            diff_x = to_x - from_x
            diff_y = to_y - from_y
            if diff_y < 0:
                if diff_x == 0:
                    self.logger.debug("MOVE_DIRECTION.UP")
                    move_direction = MOVE_DIRECTION.UP
                elif diff_y == -2 and diff_x == -1:
                    self.logger.debug("MOVE_DIRECTION.KEIMA_RIGHT")
                    move_direction = MOVE_DIRECTION.KEIMA_RIGHT
                elif diff_y == -2 and diff_x == 1:
                    self.logger.debug("MOVE_DIRECTION.KEIMA_LEFT")
                    move_direction = MOVE_DIRECTION.KEIMA_LEFT
                elif diff_x < 0:
                    self.logger.debug("MOVE_DIRECTION.UP_RIGHT")
                    move_direction = MOVE_DIRECTION.UP_RIGHT
                else:  # diff_x > 0
                    self.logger.debug("MOVE_DIRECTION.UP_LEFT")
                    move_direction = MOVE_DIRECTION.UP_LEFT
            elif diff_y == 0:
                if diff_x < 0:
                    self.logger.debug("MOVE_DIRECTION.RIGHT")
                    move_direction = MOVE_DIRECTION.RIGHT
                else:  # diff_x > 0
                    self.logger.debug("MOVE_DIRECTION.LEFT")
                    move_direction = MOVE_DIRECTION.LEFT
            else:  # diff_y > 0
                if diff_x == 0:
                    self.logger.debug("MOVE_DIRECTION.DOWN")
                    move_direction = MOVE_DIRECTION.DOWN
                elif diff_x < 0:
                    self.logger.debug("MOVE_DIRECTION.DOWN_RIGHT")
                    move_direction = MOVE_DIRECTION.DOWN_RIGHT
                else:  # diff_x > 0
                    self.logger.debug("MOVE_DIRECTION.DOWN_LEFT")
                    move_direction = MOVE_DIRECTION.DOWN_LEFT

            # 移動先で成った場合
            if move_is_promotion(move):
                self.logger.debug("PROMOTION")
                move_direction += len(MOVE_DIRECTION)
        else:  # 駒打ちの場合
            self.logger.debug("DROP")
            to_sq = move_to(move)

            if turn == cshogi.WHITE:
                to_sq = 80 - to_sq

            # 成りを含めた移動方向の種類数に持ち駒の種類を足す
            move_direction = len(MOVE_DIRECTION) * 2 + move_drop_hand_piece(move)

        return move_direction * 81 + to_sq

    def __make_result_value(self, turn: int, game_result: int) -> float:
        match (turn, game_result):
            case (cshogi.BLACK, cshogi.BLACK_WIN):
                return 1
            case (cshogi.BLACK, cshogi.WHITE_WIN):
                return 0
            case (cshogi.WHITE, cshogi.BLACK_WIN):
                return 0
            case (cshogi.WHITE, cshogi.WHITE_WIN):
                return 1
            case _:
                return 0.5
