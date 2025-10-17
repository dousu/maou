import logging

import numpy as np

from maou.domain.board.shogi import Turn

logger: logging.Logger = logging.getLogger(__name__)


class Evaluation:
    @staticmethod
    def get_winrate_from_eval(turn: Turn, eval: float) -> float:
        """evalをsigmoidで勝率に変換する"""
        return 1 / (1 + np.exp(-eval))

    @staticmethod
    def get_eval_from_winrate(r: float, a: int = 600) -> float:
        r = np.clip(r, 1e-12, 1 - 1e-12)
        return -a * np.log(1 / r - 1)

    @staticmethod
    def get_winrate_from_normalized_value(
        x: float, a: int = 600
    ) -> float:
        return 1 / (1 + np.exp(-x / a))
