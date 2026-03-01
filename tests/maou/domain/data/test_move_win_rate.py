"""Tests for per-move win rate computation logic."""

from pathlib import Path

import pytest

from maou.domain.data.intermediate_store import IntermediateDataStore
from maou.domain.move.label import MOVE_LABELS_NUM


class TestComputeMoveWinRates:
    """Test _compute_move_win_rates domain logic."""

    def _create_store(self, tmp_path: Path, threshold: int = 2) -> IntermediateDataStore:
        """Create IntermediateDataStore with given threshold."""
        db_path = tmp_path / "test.duckdb"
        return IntermediateDataStore(
            db_path=db_path,
            win_rate_threshold=threshold,
        )

    def test_compute_move_win_rates_normal(self, tmp_path: Path) -> None:
        """count >= threshold: correct per-move rates."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20, 30]]
            label_values_col = [[4, 2, 1]]
            win_values_col = [[2.0, 1.0, 0.5]]
            counts = [5]

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            assert len(move_win_rates) == 1
            assert len(move_win_rates[0]) == MOVE_LABELS_NUM
            assert move_win_rates[0][10] == pytest.approx(0.5, rel=1e-5)     # 2.0/4
            assert move_win_rates[0][20] == pytest.approx(0.5, rel=1e-5)     # 1.0/2
            assert move_win_rates[0][30] == pytest.approx(0.5, rel=1e-5)     # 0.5/1
            assert move_win_rates[0][0] == pytest.approx(0.0)                 # non-legal move
        finally:
            store.close()

    def test_compute_move_win_rates_fallback(self, tmp_path: Path) -> None:
        """count < threshold: uniform 1/N distribution."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[5, 15, 25]]
            label_values_col = [[1, 1, 1]]
            win_values_col = [[1.0, 0.0, 0.0]]
            counts = [2]  # < threshold=3

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            expected_rate = 1.0 / 3  # 3 legal moves
            assert move_win_rates[0][5] == pytest.approx(expected_rate, rel=1e-5)
            assert move_win_rates[0][15] == pytest.approx(expected_rate, rel=1e-5)
            assert move_win_rates[0][25] == pytest.approx(expected_rate, rel=1e-5)
            assert move_win_rates[0][0] == pytest.approx(0.0)
            assert best_move_win_rates[0] == pytest.approx(0.5, rel=1e-5)
        finally:
            store.close()

    def test_compute_move_win_rates_mixed(self, tmp_path: Path) -> None:
        """Batch with both normal and fallback positions."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[10, 20], [5, 15]]
            label_values_col = [[3, 2], [1, 1]]
            win_values_col = [[1.5, 1.0], [1.0, 0.0]]
            counts = [5, 1]  # first: normal, second: fallback

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            # First position: normal
            assert move_win_rates[0][10] == pytest.approx(0.5, rel=1e-5)   # 1.5/3
            assert move_win_rates[0][20] == pytest.approx(0.5, rel=1e-5)   # 1.0/2
            assert best_move_win_rates[0] == pytest.approx(0.5, rel=1e-5)

            # Second position: fallback (1/2 uniform, bestMoveWinRate=0.5)
            assert move_win_rates[1][5] == pytest.approx(0.5, rel=1e-5)
            assert move_win_rates[1][15] == pytest.approx(0.5, rel=1e-5)
            assert best_move_win_rates[1] == pytest.approx(0.5, rel=1e-5)
        finally:
            store.close()

    def test_compute_best_move_win_rate(self, tmp_path: Path) -> None:
        """Correct max(rate) computation."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20, 30]]
            label_values_col = [[4, 2, 1]]
            win_values_col = [[1.0, 2.0, 0.0]]
            counts = [5]

            _, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            # max(1.0/4=0.25, 2.0/2=1.0, 0.0/1=0.0) = 1.0
            assert best_move_win_rates[0] == pytest.approx(1.0, rel=1e-5)
        finally:
            store.close()

    def test_compute_move_win_rates_single_move(self, tmp_path: Path) -> None:
        """Edge: position with only 1 legal move."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[42]]
            label_values_col = [[3]]
            win_values_col = [[2.0]]
            counts = [3]

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            assert move_win_rates[0][42] == pytest.approx(2.0 / 3, rel=1e-5)
            assert best_move_win_rates[0] == pytest.approx(2.0 / 3, rel=1e-5)
        finally:
            store.close()

    def test_compute_move_win_rates_zero_wins(self, tmp_path: Path) -> None:
        """All moves have 0 wins -> all rates are 0.0."""
        store = self._create_store(tmp_path, threshold=2)
        try:
            indices_col = [[10, 20]]
            label_values_col = [[3, 2]]
            win_values_col = [[0.0, 0.0]]
            counts = [5]

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            assert move_win_rates[0][10] == pytest.approx(0.0)
            assert move_win_rates[0][20] == pytest.approx(0.0)
            assert best_move_win_rates[0] == pytest.approx(0.0)
        finally:
            store.close()

    def test_compute_move_win_rates_threshold_boundary(self, tmp_path: Path) -> None:
        """count == threshold -> normal (not fallback)."""
        store = self._create_store(tmp_path, threshold=3)
        try:
            indices_col = [[10, 20]]
            label_values_col = [[2, 1]]
            win_values_col = [[1.0, 0.5]]
            counts = [3]  # == threshold

            move_win_rates, best_move_win_rates = store._compute_move_win_rates(
                indices_col, label_values_col, win_values_col, counts
            )

            # Should use normal computation, not fallback
            assert move_win_rates[0][10] == pytest.approx(0.5, rel=1e-5)   # 1.0/2
            assert move_win_rates[0][20] == pytest.approx(0.5, rel=1e-5)   # 0.5/1
            assert best_move_win_rates[0] == pytest.approx(0.5, rel=1e-5)
        finally:
            store.close()
