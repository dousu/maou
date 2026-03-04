"""Tests for per-move win rate computation logic."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class TestComputeMoveWinRates:
    """Test _compute_move_win_rates domain logic."""

    def _create_store(
        self,
        tmp_path: Path,
        threshold: int = 2,
        prior_strength: float = 0.0,
    ) -> IntermediateDataStore:
        """Create IntermediateDataStore with given threshold.

        prior_strength defaults to 0.0 (no smoothing) so that existing
        tests can assert raw win rates without Beta prior adjustment.
        Production default is 5.0.
        """
        db_path = tmp_path / "test.duckdb"
        return IntermediateDataStore(
            db_path=db_path,
            position_count_threshold=threshold,
            prior_strength=prior_strength,
        )

    def test_compute_move_win_rates_normal(
        self, tmp_path: Path
    ) -> None:
        """count >= threshold: correct per-move rates."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20, 30]]
            label_values_col = [[4, 2, 1]]
            win_values_col = [[2.0, 1.0, 0.5]]
            counts = [5]

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            assert len(move_win_rates) == 1
            assert len(move_win_rates[0]) == MOVE_LABELS_NUM
            assert move_win_rates[0][10] == pytest.approx(
                0.5, rel=1e-5
            )  # 2.0/4
            assert move_win_rates[0][20] == pytest.approx(
                0.5, rel=1e-5
            )  # 1.0/2
            assert move_win_rates[0][30] == pytest.approx(
                0.5, rel=1e-5
            )  # 0.5/1
            assert move_win_rates[0][0] == pytest.approx(
                0.0
            )  # non-legal move
        finally:
            store.close()

    def test_compute_move_win_rates_fallback(
        self, tmp_path: Path
    ) -> None:
        """count < threshold: uniform 1/N distribution."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[5, 15, 25]]
            label_values_col = [[1, 1, 1]]
            win_values_col = [[1.0, 0.0, 0.0]]
            counts = [2]  # < threshold=3

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            expected_rate = 1.0 / 3  # 3 legal moves
            assert move_win_rates[0][5] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert move_win_rates[0][15] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert move_win_rates[0][25] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert move_win_rates[0][0] == pytest.approx(0.0)
            assert best_move_win_rates[0] == pytest.approx(
                0.5, rel=1e-5
            )
        finally:
            store.close()

    def test_compute_move_win_rates_mixed(
        self, tmp_path: Path
    ) -> None:
        """Batch with both normal and fallback positions."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[10, 20], [5, 15]]
            label_values_col = [[3, 2], [1, 1]]
            win_values_col = [[1.5, 1.0], [1.0, 0.0]]
            counts = [5, 1]  # first: normal, second: fallback

            (
                move_win_rates,
                best_move_win_rates,
                fallback_count,
            ) = store._compute_move_win_rates(
                indices_col,
                label_values_col,
                win_values_col,
                counts,
            )

            assert fallback_count == 1

            # First position: normal
            assert move_win_rates[0][10] == pytest.approx(
                0.5, rel=1e-5
            )  # 1.5/3
            assert move_win_rates[0][20] == pytest.approx(
                0.5, rel=1e-5
            )  # 1.0/2
            assert best_move_win_rates[0] == pytest.approx(
                0.5, rel=1e-5
            )

            # Second position: fallback (1/2 uniform, bestMoveWinRate=0.5)
            assert move_win_rates[1][5] == pytest.approx(
                0.5, rel=1e-5
            )
            assert move_win_rates[1][15] == pytest.approx(
                0.5, rel=1e-5
            )
            assert best_move_win_rates[1] == pytest.approx(
                0.5, rel=1e-5
            )
        finally:
            store.close()

    def test_compute_best_move_win_rate(
        self, tmp_path: Path
    ) -> None:
        """Correct max(rate) computation."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20, 30]]
            label_values_col = [[4, 2, 1]]
            win_values_col = [[1.0, 2.0, 0.0]]
            counts = [5]

            _, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            # max(1.0/4=0.25, 2.0/2=1.0, 0.0/1=0.0) = 1.0
            assert best_move_win_rates[0] == pytest.approx(
                1.0, rel=1e-5
            )
        finally:
            store.close()

    def test_compute_move_win_rates_single_move(
        self, tmp_path: Path
    ) -> None:
        """Edge: position with only 1 legal move."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[42]]
            label_values_col = [[3]]
            win_values_col = [[2.0]]
            counts = [3]

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            assert move_win_rates[0][42] == pytest.approx(
                2.0 / 3, rel=1e-5
            )
            assert best_move_win_rates[0] == pytest.approx(
                2.0 / 3, rel=1e-5
            )
        finally:
            store.close()

    def test_compute_move_win_rates_zero_wins(
        self, tmp_path: Path
    ) -> None:
        """All moves have 0 wins -> all rates are 0.0."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20]]
            label_values_col = [[3, 2]]
            win_values_col = [[0.0, 0.0]]
            counts = [5]

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            assert move_win_rates[0][10] == pytest.approx(0.0)
            assert move_win_rates[0][20] == pytest.approx(0.0)
            assert best_move_win_rates[0] == pytest.approx(0.0)
        finally:
            store.close()

    def test_compute_move_win_rates_threshold_boundary(
        self, tmp_path: Path
    ) -> None:
        """count == threshold -> normal (not fallback)."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[10, 20]]
            label_values_col = [[2, 1]]
            win_values_col = [[1.0, 0.5]]
            counts = [3]  # == threshold

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            # Should use normal computation, not fallback
            assert move_win_rates[0][10] == pytest.approx(
                0.5, rel=1e-5
            )  # 1.0/2
            assert move_win_rates[0][20] == pytest.approx(
                0.5, rel=1e-5
            )  # 0.5/1
            assert best_move_win_rates[0] == pytest.approx(
                0.5, rel=1e-5
            )
        finally:
            store.close()

    def test_beta_prior_shrinks_low_count_rate(
        self, tmp_path: Path
    ) -> None:
        """Beta prior shrinks a 1-win/1-play move toward 50%."""
        prior = 5.0
        store = self._create_store(
            tmp_path, threshold=2, prior_strength=prior
        )
        try:
            # Move A: 80 plays, 50 wins -> high-count
            # Move B: 1 play, 1 win -> low-count noise
            indices_col = [[10, 20]]
            label_values_col = [[80, 1]]
            win_values_col = [[50.0, 1.0]]
            counts = [100]

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            # Move A: (50+5)/(80+10) = 55/90 ≈ 0.6111
            expected_a = (50.0 + prior) / (80.0 + 2 * prior)
            assert move_win_rates[0][10] == pytest.approx(
                expected_a, rel=1e-5
            )

            # Move B: (1+5)/(1+10) = 6/11 ≈ 0.5455
            expected_b = (1.0 + prior) / (1.0 + 2 * prior)
            assert move_win_rates[0][20] == pytest.approx(
                expected_b, rel=1e-5
            )

            # Without prior, Move B would be 100% and dominate.
            # With prior, Move A has higher rate than Move B.
            assert move_win_rates[0][10] > move_win_rates[0][20]
        finally:
            store.close()

    def test_beta_prior_zero_disables_smoothing(
        self, tmp_path: Path
    ) -> None:
        """prior_strength=0.0 gives raw win rates (backward compat)."""
        store = self._create_store(
            tmp_path, threshold=2, prior_strength=0.0
        )
        try:
            indices_col = [[10, 20]]
            label_values_col = [[4, 1]]
            win_values_col = [[2.0, 1.0]]
            counts = [5]

            move_win_rates, *_ = store._compute_move_win_rates(
                indices_col,
                label_values_col,
                win_values_col,
                counts,
            )

            assert move_win_rates[0][10] == pytest.approx(
                0.5, rel=1e-5
            )  # 2/4
            assert move_win_rates[0][20] == pytest.approx(
                1.0, rel=1e-5
            )  # 1/1
        finally:
            store.close()

    def test_beta_prior_does_not_affect_fallback(
        self, tmp_path: Path
    ) -> None:
        """Fallback positions still use uniform 1/N regardless of prior."""
        prior = 5.0
        store = self._create_store(
            tmp_path, threshold=3, prior_strength=prior
        )
        try:
            indices_col = [[5, 15, 25]]
            label_values_col = [[1, 1, 1]]
            win_values_col = [[1.0, 0.0, 0.0]]
            counts = [2]  # < threshold=3 -> fallback

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            expected_rate = 1.0 / 3
            assert move_win_rates[0][5] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert move_win_rates[0][15] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert move_win_rates[0][25] == pytest.approx(
                expected_rate, rel=1e-5
            )
            assert best_move_win_rates[0] == pytest.approx(
                0.5, rel=1e-5
            )
        finally:
            store.close()

    def test_beta_prior_high_count_minimal_impact(
        self, tmp_path: Path
    ) -> None:
        """High-count moves are barely affected by prior."""
        prior = 5.0
        store = self._create_store(
            tmp_path, threshold=2, prior_strength=prior
        )
        try:
            indices_col = [[10]]
            label_values_col = [[1000]]
            win_values_col = [[600.0]]
            counts = [1000]

            move_win_rates, *_ = store._compute_move_win_rates(
                indices_col,
                label_values_col,
                win_values_col,
                counts,
            )

            # (600+5)/(1000+10) = 605/1010 ≈ 0.5990
            # Raw rate: 600/1000 = 0.6
            raw_rate = 600.0 / 1000.0
            smoothed = move_win_rates[0][10]
            assert abs(smoothed - raw_rate) < 0.002
        finally:
            store.close()

    def test_negative_prior_strength_rejected(
        self, tmp_path: Path
    ) -> None:
        """Negative prior_strength raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        with pytest.raises(
            ValueError, match="prior_strength must be >= 0.0"
        ):
            IntermediateDataStore(
                db_path=db_path,
                prior_strength=-1.0,
            )

    def test_win_rates_clipped_to_unit_interval(
        self, tmp_path: Path
    ) -> None:
        """Win rates are clipped to [0.0, 1.0] even with noisy data."""
        store = self._create_store(
            tmp_path, threshold=2, prior_strength=0.0
        )
        try:
            # win_values > label_values -> raw rate > 1.0
            indices_col = [[10]]
            label_values_col = [[2]]
            win_values_col = [[3.0]]  # 3.0/2 = 1.5 without clip
            counts = [5]

            move_win_rates, best_move_win_rates, _ = (
                store._compute_move_win_rates(
                    indices_col,
                    label_values_col,
                    win_values_col,
                    counts,
                )
            )

            assert move_win_rates[0][10] <= 1.0
            assert best_move_win_rates[0] <= 1.0
        finally:
            store.close()


class TestFallbackCountSummation:
    """Test fallback count summation across chunks (#8)."""

    def test_fallback_count_summed_across_chunks(
        self, tmp_path: Path
    ) -> None:
        """Fallback counts from multiple chunks are correctly summed."""
        db_path = tmp_path / "test.duckdb"

        store = IntermediateDataStore(
            db_path=db_path,
            position_count_threshold=3,
            prior_strength=0.0,
        )
        try:
            # Add 6 positions: 3 with count >= threshold, 3 with count < threshold
            for i in range(6):
                move_label_count = [0] * 1496
                move_label_count[10] = 1
                move_win_count = [0.0] * 1496
                move_win_count[10] = 0.5

                df = pl.DataFrame(
                    [
                        {
                            "hash_id": i,
                            "count": 5
                            if i < 3
                            else 1,  # first 3 normal, last 3 fallback
                            "win_count": 2.5 if i < 3 else 0.5,
                            "move_label_count": move_label_count,
                            "move_win_count": move_win_count,
                            "board_id_positions": [
                                [0] * 9 for _ in range(9)
                            ],
                            "pieces_in_hand": [0] * 14,
                        }
                    ]
                )
                store.add_dataframe_batch(df)

            # Finalize in chunks of 2 (3 chunks total)
            total_fallback = 0
            chunk_count = 0
            for (
                _,
                fallback_count,
            ) in store.iter_finalize_chunks_df(
                chunk_size=2, delete_after_yield=False
            ):
                total_fallback += fallback_count
                chunk_count += 1

            assert chunk_count == 3
            assert (
                total_fallback == 3
            )  # 3 positions had count < threshold

        finally:
            store.close()


class TestPriorStrengthIntegration:
    """Integration test: prior_strength flows through store to finalized output (#7)."""

    def test_prior_strength_affects_finalized_output(
        self, tmp_path: Path
    ) -> None:
        """prior_strength set at store creation affects finalize_to_dataframe output."""
        prior = 5.0
        db_path = tmp_path / "test.duckdb"
        store = IntermediateDataStore(
            db_path=db_path,
            position_count_threshold=2,
            prior_strength=prior,
        )
        try:
            move_label_count = [0] * 1496
            move_label_count[10] = 4
            move_label_count[20] = 1
            move_win_count = [0.0] * 1496
            move_win_count[10] = 2.0
            move_win_count[20] = 1.0

            df = pl.DataFrame(
                [
                    {
                        "hash_id": 42,
                        "count": 5,
                        "win_count": 3.0,
                        "move_label_count": move_label_count,
                        "move_win_count": move_win_count,
                        "board_id_positions": [
                            [0] * 9 for _ in range(9)
                        ],
                        "pieces_in_hand": [0] * 14,
                    }
                ]
            )
            store.add_dataframe_batch(df)

            result_df, _ = store.finalize_to_dataframe()
            win_rates = result_df["moveWinRate"].to_list()[0]

            # With prior=5.0:
            # idx 10: (2.0+5)/(4+10) = 7/14 = 0.5
            # idx 20: (1.0+5)/(1+10) = 6/11 ≈ 0.5455
            # Without prior (raw): idx 10 = 0.5, idx 20 = 1.0
            expected_10 = (2.0 + prior) / (4 + 2 * prior)
            expected_20 = (1.0 + prior) / (1 + 2 * prior)
            assert win_rates[10] == pytest.approx(
                expected_10, rel=1e-5
            )
            assert win_rates[20] == pytest.approx(
                expected_20, rel=1e-5
            )
            # Confirm smoothing had an effect on idx 20
            # (raw rate would be 1.0, smoothed is ~0.5455)
            assert win_rates[20] < 1.0
        finally:
            store.close()


class TestNumpyReturnTypes:
    """Regression tests: dense array methods must return numpy arrays.

    Python list[list[float]] causes ~7× more memory than numpy float32
    due to per-object overhead (28 bytes/float). Reverting to list would
    re-introduce OOM during chunked aggregation of large datasets.
    """

    def _create_store(
        self, tmp_path: Path
    ) -> IntermediateDataStore:
        db_path = tmp_path / "test.duckdb"
        return IntermediateDataStore(
            db_path=db_path,
            position_count_threshold=2,
            prior_strength=0.0,
        )

    def test_expand_and_normalize_returns_numpy(
        self, tmp_path: Path
    ) -> None:
        """_expand_and_normalize_move_labels must return np.ndarray."""
        result = IntermediateDataStore._expand_and_normalize_move_labels(
            indices_col=[[0, 1], [2]],
            values_col=[[10, 20], [5]],
            counts=[10, 5],
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, MOVE_LABELS_NUM)

    def test_compute_move_win_rates_returns_numpy(
        self, tmp_path: Path
    ) -> None:
        """_compute_move_win_rates must return np.ndarray."""
        store = self._create_store(tmp_path)
        try:
            move_win_rates, _, _ = (
                store._compute_move_win_rates(
                    indices_col=[[0, 1]],
                    label_values_col=[[4, 2]],
                    win_values_col=[[2.0, 1.0]],
                    counts=[5],
                )
            )
            assert isinstance(move_win_rates, np.ndarray)
            assert move_win_rates.dtype == np.float32
            assert move_win_rates.shape == (
                1,
                MOVE_LABELS_NUM,
            )
        finally:
            store.close()

    def test_expand_normalize_zero_count_no_error(
        self, tmp_path: Path
    ) -> None:
        """count=0 must not raise division-by-zero."""
        result = IntermediateDataStore._expand_and_normalize_move_labels(
            indices_col=[[0, 1]],
            values_col=[[10, 20]],
            counts=[0],
        )
        # count=0 row is skipped, all zeros
        assert result[0].sum() == 0.0
