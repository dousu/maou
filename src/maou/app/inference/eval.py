import logging

import numpy as np

from maou.domain.board.shogi import Turn

logger: logging.Logger = logging.getLogger(__name__)


class Evaluation:
    @staticmethod
    def get_winrate_from_eval(turn: Turn, eval: float) -> float:
        """評価値は-1から1の範囲だが0-1の勝率に変換する"""
        turn_eval = eval if turn == Turn.BLACK else -eval
        return 0.5 + turn_eval / 2

    @staticmethod
    def get_eval_from_winrate(r: float, a: int = 600) -> float:
        r = np.clip(r, 1e-12, 1 - 1e-12)
        return -a * np.log(1 / r - 1)

    @staticmethod
    def get_winrate_from_normalized_value(
        x: float, a: int = 600
    ) -> float:
        return 1 / (1 + np.exp(-x / a))
