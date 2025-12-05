import logging

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class Evaluation:
    @staticmethod
    def get_winrate_from_eval(eval: float) -> float:
        """evalをsigmoidで勝率に変換する．

        Args:
            eval: モデル出力のlogit値（現在プレイヤー視点）

        Returns:
            現在プレイヤーの勝率（0-1）
        """
        return 1 / (1 + np.exp(-eval))

    @staticmethod
    def get_eval_from_winrate(r: float, a: int = 600) -> float:
        """勝率を評価値スコアに変換する．

        評価値スコアは，モデル出力のlogitを600倍したもの:
          eval = 600 * logit

        係数600は，Ponanzaという将棋AIで使われていた定数で，
        将棋AI界隈では標準的に使用される．この係数により，
        評価値が人間にとって直感的な範囲（数百～数千）になる．

        スケールの目安:
          - eval = 0:    互角（勝率50%）
          - eval = 600:  有利（勝率73%）
          - eval = 1200: 勝勢（勝率88%）
          - eval = 1800: 圧勝（勝率95%）
          - eval ≥ 3000: 勝敗がほぼ決している

        Args:
            r: 勝率（0-1）
            a: スケーリング係数（デフォルト: 600，Ponanza由来）

        Returns:
            評価値スコア（unbounded，典型的には-4000～4000）
        """
        r = np.clip(r, 1e-12, 1 - 1e-12)
        return -a * np.log(1 / r - 1)
