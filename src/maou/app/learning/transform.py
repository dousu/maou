import logging

import cshogi
import numpy as np
import torch

from maou.app.learning.feature import make_feature
from maou.app.learning.label import make_move_label, make_result_value


class Transform:
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, pin_memory: bool, device: torch.device):
        self.pin_memory = pin_memory
        self.device = device

    def __call__(
        self, *, hcp: np.ndarray, move16: int, game_result: int, eval: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
            torch.from_numpy(features).to(self.device),
            (
                torch.tensor(
                    move_label, dtype=torch.long, pin_memory=self.pin_memory
                ).to(self.device),
                torch.tensor(
                    result_value, dtype=torch.float32, pin_memory=self.pin_memory
                )
                .reshape((1))
                .to(self.device),
            ),
        )
