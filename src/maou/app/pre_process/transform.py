import logging

import cshogi
import numpy as np

from maou.app.pre_process.feature import make_feature
from maou.app.pre_process.label import make_move_label, make_result_value


class Transform:
    logger: logging.Logger = logging.getLogger(__name__)

    def __call__(
        self, *, hcp: np.ndarray, move16: int, game_result: int, eval: int
    ) -> tuple[np.ndarray, int, float]:
        self.logger.debug(f"hcp shape: {hcp.shape}")
        self.logger.debug(f"hcp dtype: {hcp.dtype}")
        board = cshogi.Board()  # type: ignore
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
            move_label = make_move_label(board.turn, move16)

            # result value
            result_value = make_result_value(board.turn, game_result)
        except Exception as e:
            move = board.move_from_move16(move16)
            self.logger.error(
                f"cshogi move: {move} {move16}"
                f", game: {game_result}"
                f", eval: {eval}"
                f", sfen: {board.sfen()}"
            )
            self.logger.error(str(board))
            board.push(move)
            self.logger.error(str(board))
            raise e

        return (
            features,
            move_label,
            result_value,
        )
