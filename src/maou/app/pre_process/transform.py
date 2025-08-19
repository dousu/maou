import logging

import numpy as np

from maou.app.pre_process.feature import make_feature
from maou.app.pre_process.label import (
    MOVE_LABELS_NUM,
    make_move_label,
    make_result_value,
)
from maou.domain.board import shogi


class Transform:
    """Transforms HCPE position data into neural network training features.

    Converts board positions, moves, and game results into feature vectors
    and labels for training Shogi AI models.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __call__(
        self,
        *,
        hcp: np.ndarray,
        move16: int,
        game_result: int,
        eval: int,
    ) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Transform HCPE data into training features.

        Args:
            hcp: Board position in HCP format
            move16: Move in 16-bit encoding
            game_result: Game result (0=draw, 1=first player wins, 2=second player wins)
            eval: Position evaluation score

        Returns:
            Tuple of (features, move_label, result_value, legal_move_mask)
        """

        # self.logger.debug(f"hcp type: {type(hcp)}")
        # self.logger.debug(f"hcp shape: {hcp.shape}")
        # self.logger.debug(f"hcp dtype: {hcp.dtype}")
        board = shogi.Board()
        board.set_hcp(hcp)

        try:
            # 入力特徴量
            features = make_feature(board)

            # 教師データ
            # move label
            # この変換は必要なさそうなので行わない
            # move = move_from_move16(move16)
            # 不正な棋譜が入っているときここは簡単にエラーになるので注意 (ratingで絞るとか？)
            # endgame statusが%TIMEUPが入っていると変なcshogi move値になっていそう
            # こういうどうしようもないのは学習から除外するためにエラー出たら何もしないという選択肢もある
            move_label = make_move_label(
                board.get_turn(), move16
            )

            # result value
            result_value = make_result_value(
                board.get_turn(), game_result
            )

            # 合法手のラベルを取得する
            legal_move_labels = [
                make_move_label(board.get_turn(), m)
                for m in board.get_legal_moves()
            ]
            legal_move_mask = self.__create_mask(
                legal_move_labels, MOVE_LABELS_NUM
            )
        except Exception:
            move = board.get_move_from_move16(move16)
            self.logger.error(
                f"cshogi move: {move} {move16}"
                f", game: {game_result}"
                f", eval: {eval}"
                f", sfen: {board.get_sfen()}"
            )
            self.logger.error(str(board))
            board.push_move(move)
            self.logger.error(str(board))
            raise

        return (
            features,
            move_label,
            result_value,
            legal_move_mask,
        )

    def __create_mask(
        self, valid_labels: list[int], num_classes: int
    ) -> np.ndarray:
        """有効なラベルのリストから対応するマスクを作成する."""
        mask = np.zeros((num_classes), dtype=np.uint8)
        mask[valid_labels] = 1
        return mask
