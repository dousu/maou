import logging

import numpy as np

from maou.app.inference.eval import Evaluation

logger: logging.Logger = logging.getLogger("TEST")


class TestEvaluation:
    def test_get_winrate_from_eval(self) -> None:
        """evalからwinrateへの変換（sigmoid）が正しいかテスト．"""
        # Test cases: (logit, expected_winrate)
        test_cases = [
            (0.0, 0.5),
            (1.0, 0.7311),  # sigmoid(1.0) ≈ 0.731
            (-1.0, 0.2689),  # sigmoid(-1.0) ≈ 0.269
            (2.0, 0.8808),  # sigmoid(2.0) ≈ 0.881
        ]

        for logit, expected in test_cases:
            winrate = Evaluation.get_winrate_from_eval(logit)
            assert np.isclose(winrate, expected, rtol=1e-3)

    def test_get_eval_from_winrate(self) -> None:
        """winrateからevalへの変換が正しいかテスト．"""
        # Test: eval = 600 * logit
        test_cases = [
            (0.5, 0.0),  # 互角
            (0.7311, 600.0),  # 有利
            (0.8808, 1200.0),  # 勝勢
        ]

        for winrate, expected_eval in test_cases:
            eval_value = Evaluation.get_eval_from_winrate(
                winrate
            )
            assert np.isclose(
                eval_value, expected_eval, rtol=1e-2
            )

    def test_round_trip_conversion(self) -> None:
        """logit → winrate → eval の往復変換が正しいかテスト．"""
        test_logits = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]

        for logit in test_logits:
            winrate = Evaluation.get_winrate_from_eval(logit)
            eval_value = Evaluation.get_eval_from_winrate(
                winrate
            )
            expected_eval = 600 * logit
            assert np.isclose(
                eval_value, expected_eval, rtol=1e-5
            )

    def test_edge_cases(self) -> None:
        """極端な値での数値安定性をテスト．"""
        extreme_logits = [-50.0, -10.0, 10.0, 50.0]

        for logit in extreme_logits:
            winrate = Evaluation.get_winrate_from_eval(logit)
            eval_value = Evaluation.get_eval_from_winrate(
                winrate
            )

            # Check no NaN or Inf
            assert not np.isnan(winrate)
            assert not np.isnan(eval_value)
            assert not np.isinf(eval_value)

            # Check winrate is in valid range
            assert 0.0 <= winrate <= 1.0

    def test_ponanza_scale(self) -> None:
        """Ponanzaスケールの数値感覚をテスト（3000-4000で勝敗決定）．"""
        # logit = 5.0 → eval = 3000 → winrate ≈ 99.3%
        winrate_3000 = Evaluation.get_winrate_from_eval(5.0)
        assert winrate_3000 > 0.99

        # logit = 6.67 → eval ≈ 4000 → winrate ≈ 99.87%
        winrate_4000 = Evaluation.get_winrate_from_eval(6.67)
        assert winrate_4000 > 0.998
