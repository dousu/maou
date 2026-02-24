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

    def test_boundary_values(self) -> None:
        """境界値（winrate = 0.0, 1.0）での動作をテスト．"""
        # winrate = 0.0 should give very negative eval
        eval_0 = Evaluation.get_eval_from_winrate(0.0)
        assert eval_0 < -10000  # Very large negative value
        assert not np.isnan(eval_0)
        assert not np.isinf(eval_0)

        # winrate = 1.0 should give very positive eval
        eval_1 = Evaluation.get_eval_from_winrate(1.0)
        assert eval_1 > 10000  # Very large positive value
        assert not np.isnan(eval_1)
        assert not np.isinf(eval_1)

    def test_clip_behavior(self) -> None:
        """clipによる数値安定化の動作をテスト．"""
        # Values outside [0, 1] should be clipped to [1e-12, 1-1e-12]
        # winrate slightly below 0 → clipped to 1e-12
        eval_neg = Evaluation.get_eval_from_winrate(-0.1)
        expected_eval_min = Evaluation.get_eval_from_winrate(
            1e-12
        )
        assert np.isclose(
            eval_neg, expected_eval_min, rtol=1e-6
        )

        # winrate slightly above 1 → clipped to 1-1e-12
        eval_over = Evaluation.get_eval_from_winrate(1.1)
        expected_eval_max = Evaluation.get_eval_from_winrate(
            1 - 1e-12
        )
        assert np.isclose(
            eval_over, expected_eval_max, rtol=1e-6
        )

    def test_array_inputs(self) -> None:
        """numpy配列入力での動作をテスト．"""
        # Test get_winrate_from_eval with array input
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        winrates_result = Evaluation.get_winrate_from_eval(
            logits
        )

        # Type assertion: result should be array
        assert isinstance(winrates_result, np.ndarray)
        winrates = winrates_result

        assert winrates.shape == logits.shape
        assert np.all((winrates >= 0.0) & (winrates <= 1.0))

        # Verify each element
        for i, logit in enumerate(logits):
            expected = Evaluation.get_winrate_from_eval(logit)
            assert np.isclose(winrates[i], expected, rtol=1e-6)

        # Test get_eval_from_winrate with array input
        winrates_input = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        evals_result = Evaluation.get_eval_from_winrate(
            winrates_input
        )

        # Type assertion: result should be array
        assert isinstance(evals_result, np.ndarray)
        evals = evals_result

        assert evals.shape == winrates_input.shape

        # Verify each element
        for i, wr in enumerate(winrates_input):
            expected = Evaluation.get_eval_from_winrate(wr)
            assert np.isclose(evals[i], expected, rtol=1e-6)

    def test_custom_scaling_factor(self) -> None:
        """デフォルト値以外のスケーリング係数aをテスト．"""
        winrate = 0.7311  # sigmoid(1.0)

        # Default: a = 600
        eval_600 = Evaluation.get_eval_from_winrate(
            winrate, a=600
        )
        assert np.isclose(eval_600, 600.0, rtol=1e-2)

        # Custom: a = 100
        eval_100 = Evaluation.get_eval_from_winrate(
            winrate, a=100
        )
        assert np.isclose(eval_100, 100.0, rtol=1e-2)

        # Custom: a = 1200
        eval_1200 = Evaluation.get_eval_from_winrate(
            winrate, a=1200
        )
        assert np.isclose(eval_1200, 1200.0, rtol=1e-2)

        # Relationship: eval should scale linearly with a
        assert np.isclose(eval_600 / eval_100, 6.0, rtol=1e-3)
        assert np.isclose(eval_1200 / eval_600, 2.0, rtol=1e-3)

    def test_documented_examples(self) -> None:
        """docstringに記載された具体例をテスト．"""
        # From docstring: eval = 0 → 50%
        winrate_0 = Evaluation.get_winrate_from_eval(0.0)
        assert np.isclose(winrate_0, 0.5, rtol=1e-3)

        # eval = 600 → 73%
        winrate_600 = Evaluation.get_winrate_from_eval(1.0)
        assert np.isclose(winrate_600, 0.73, rtol=1e-2)

        # eval = 1200 → 88%
        winrate_1200 = Evaluation.get_winrate_from_eval(2.0)
        assert np.isclose(winrate_1200, 0.88, rtol=1e-2)

        # eval = 1800 → 95%
        winrate_1800 = Evaluation.get_winrate_from_eval(3.0)
        assert np.isclose(winrate_1800, 0.95, rtol=1e-2)

        # Reverse conversion
        eval_from_50 = Evaluation.get_eval_from_winrate(0.5)
        assert np.isclose(eval_from_50, 0.0, atol=1e-6)

        eval_from_73 = Evaluation.get_eval_from_winrate(0.73)
        assert np.isclose(eval_from_73, 600.0, rtol=5e-2)

        eval_from_88 = Evaluation.get_eval_from_winrate(0.88)
        assert np.isclose(eval_from_88, 1200.0, rtol=5e-2)

        eval_from_95 = Evaluation.get_eval_from_winrate(0.95)
        assert np.isclose(eval_from_95, 1800.0, rtol=5e-2)

    def test_invalid_inputs(self) -> None:
        """無効な入力（NaN, Inf）の処理をテスト．"""
        # NaN input to get_winrate_from_eval
        winrate_nan = Evaluation.get_winrate_from_eval(np.nan)
        assert np.isnan(winrate_nan)

        # Inf input to get_winrate_from_eval
        winrate_inf = Evaluation.get_winrate_from_eval(np.inf)
        assert winrate_inf == 1.0  # sigmoid(inf) = 1.0

        winrate_neg_inf = Evaluation.get_winrate_from_eval(
            -np.inf
        )
        assert winrate_neg_inf == 0.0  # sigmoid(-inf) = 0.0

        # NaN input to get_eval_from_winrate (clipped to valid range)
        # Note: np.clip with NaN returns NaN
        eval_nan = Evaluation.get_eval_from_winrate(np.nan)
        assert np.isnan(eval_nan)

    def test_type_variations(self) -> None:
        """異なる数値型での動作をテスト．"""
        # float32 input
        logit_f32 = np.float32(1.0)
        winrate_f32 = Evaluation.get_winrate_from_eval(
            logit_f32
        )
        assert isinstance(winrate_f32, (np.floating, float))
        assert np.isclose(winrate_f32, 0.7311, rtol=1e-3)

        # float64 input
        logit_f64 = np.float64(1.0)
        winrate_f64 = Evaluation.get_winrate_from_eval(
            logit_f64
        )
        assert isinstance(winrate_f64, (np.floating, float))
        assert np.isclose(winrate_f64, 0.7311, rtol=1e-3)

        # int input (should be converted to float)
        logit_int = 1
        winrate_int = Evaluation.get_winrate_from_eval(
            logit_int
        )
        assert isinstance(winrate_int, (np.floating, float))
        assert np.isclose(winrate_int, 0.7311, rtol=1e-3)

        # Python float
        winrate_py = 0.7311
        eval_py = Evaluation.get_eval_from_winrate(winrate_py)
        assert isinstance(eval_py, (np.floating, float))
        assert np.isclose(eval_py, 600.0, rtol=1e-2)

    def test_symmetry_properties(self) -> None:
        """対称性と数学的性質をテスト．"""
        # sigmoid(-x) + sigmoid(x) should be close to 1.0
        logit = 2.5
        winrate_pos = Evaluation.get_winrate_from_eval(logit)
        winrate_neg = Evaluation.get_winrate_from_eval(-logit)
        assert np.isclose(
            winrate_pos + winrate_neg, 1.0, rtol=1e-6
        )

        # Monotonicity: larger logit → larger winrate
        logits = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        winrates_result = Evaluation.get_winrate_from_eval(
            logits
        )

        # Type assertion: result should be array
        assert isinstance(winrates_result, np.ndarray)
        winrates = winrates_result
        assert np.all(
            np.diff(winrates) > 0
        )  # Strictly increasing

        # Monotonicity: larger winrate → larger eval
        winrates_sorted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        evals_result = Evaluation.get_eval_from_winrate(
            winrates_sorted
        )

        # Type assertion: result should be array
        assert isinstance(evals_result, np.ndarray)
        evals = evals_result
        assert np.all(np.diff(evals) > 0)  # Strictly increasing
