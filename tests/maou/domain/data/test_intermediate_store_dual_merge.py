"""Tests for DuckDB dual merge UDF integration."""

from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
)


def _create_test_df(
    hash_id: int,
    count: int,
    win_count: float,
    label_indices: list[int],
    label_values: list[int],
    win_values: list[float],
) -> pl.DataFrame:
    """Create a test DataFrame with move_win_count for bulk_upsert."""
    move_label_count = [0] * 1496
    move_win_count = [0.0] * 1496
    for idx, lv, wv in zip(
        label_indices, label_values, win_values
    ):
        move_label_count[idx] = lv
        move_win_count[idx] = wv

    return pl.DataFrame(
        [
            {
                "hash_id": hash_id,
                "count": count,
                "win_count": win_count,
                "move_label_count": move_label_count,
                "move_win_count": move_win_count,
                "board_id_positions": [
                    [0] * 9 for _ in range(9)
                ],
                "pieces_in_hand": [0] * 14,
            }
        ],
        schema={
            "hash_id": pl.UInt64,
            "count": pl.Int32,
            "win_count": pl.Float64,
            "move_label_count": pl.List(pl.Int32),
            "move_win_count": pl.List(pl.Float32),
            "board_id_positions": pl.List(pl.List(pl.UInt8)),
            "pieces_in_hand": pl.List(pl.UInt8),
        },
    )


class TestIntermediateStoreDualMerge:
    """Test DuckDB dual merge UDF integration."""

    def test_bulk_upsert_with_win_values(
        self, tmp_path: Path
    ) -> None:
        """New rows inserted with correct win values."""
        db_path = tmp_path / "test.duckdb"
        store = IntermediateDataStore(
            db_path=db_path, win_rate_threshold=2
        )
        try:
            df = _create_test_df(
                hash_id=100,
                count=3,
                win_count=2.0,
                label_indices=[10, 20],
                label_values=[2, 1],
                win_values=[1.5, 0.5],
            )
            store.bulk_upsert(df)

            assert store.get_total_count() == 1

            result_df = store.finalize_to_dataframe()
            win_rates = result_df["moveWinRate"].to_list()[0]

            # rate[10] = 1.5/2 = 0.75, rate[20] = 0.5/1 = 0.5
            assert win_rates[10] == pytest.approx(
                0.75, rel=1e-5
            )
            assert win_rates[20] == pytest.approx(0.5, rel=1e-5)
        finally:
            store.close()

    def test_bulk_upsert_conflict_merge(
        self, tmp_path: Path
    ) -> None:
        """Conflicting rows merge both tracks correctly."""
        db_path = tmp_path / "test.duckdb"
        store = IntermediateDataStore(
            db_path=db_path, win_rate_threshold=2
        )
        try:
            # First upsert
            df1 = _create_test_df(
                hash_id=200,
                count=3,
                win_count=2.0,
                label_indices=[10, 20],
                label_values=[2, 1],
                win_values=[1.0, 0.5],
            )
            store.bulk_upsert(df1)

            # Second upsert with same hash_id (conflict)
            df2 = _create_test_df(
                hash_id=200,
                count=2,
                win_count=1.0,
                label_indices=[10, 30],
                label_values=[1, 1],
                win_values=[0.5, 1.0],
            )
            store.bulk_upsert(df2)

            assert store.get_total_count() == 1

            result_df = store.finalize_to_dataframe()
            win_rates = result_df["moveWinRate"].to_list()[0]

            # After merge:
            # idx 10: label=2+1=3, win=1.0+0.5=1.5, rate=1.5/3=0.5
            # idx 20: label=1, win=0.5, rate=0.5/1=0.5
            # idx 30: label=1, win=1.0, rate=1.0/1=1.0
            assert win_rates[10] == pytest.approx(0.5, rel=1e-5)
            assert win_rates[20] == pytest.approx(0.5, rel=1e-5)
            assert win_rates[30] == pytest.approx(1.0, rel=1e-5)

            # bestMoveWinRate = max(0.5, 0.5, 1.0) = 1.0
            assert result_df["bestMoveWinRate"][
                0
            ] == pytest.approx(1.0, rel=1e-5)
        finally:
            store.close()

    def test_finalize_produces_move_win_rate(
        self, tmp_path: Path
    ) -> None:
        """Output DataFrame has moveWinRate column."""
        db_path = tmp_path / "test.duckdb"
        store = IntermediateDataStore(
            db_path=db_path, win_rate_threshold=2
        )
        try:
            df = _create_test_df(
                hash_id=300,
                count=4,
                win_count=2.0,
                label_indices=[5],
                label_values=[4],
                win_values=[2.0],
            )
            store.bulk_upsert(df)

            result_df = store.finalize_to_dataframe()

            assert "moveWinRate" in result_df.columns
            assert result_df["moveWinRate"].dtype == pl.List(
                pl.Float32
            )
            assert (
                len(result_df["moveWinRate"].to_list()[0])
                == 1496
            )
        finally:
            store.close()

    def test_finalize_produces_best_move_win_rate(
        self, tmp_path: Path
    ) -> None:
        """Output DataFrame has bestMoveWinRate column."""
        db_path = tmp_path / "test.duckdb"
        store = IntermediateDataStore(
            db_path=db_path, win_rate_threshold=2
        )
        try:
            df = _create_test_df(
                hash_id=400,
                count=6,
                win_count=3.0,
                label_indices=[10, 20],
                label_values=[4, 2],
                win_values=[3.0, 1.0],
            )
            store.bulk_upsert(df)

            result_df = store.finalize_to_dataframe()

            assert "bestMoveWinRate" in result_df.columns
            assert (
                result_df["bestMoveWinRate"].dtype == pl.Float32
            )

            # best = max(3.0/4=0.75, 1.0/2=0.5) = 0.75
            assert result_df["bestMoveWinRate"][
                0
            ] == pytest.approx(0.75, rel=1e-5)
        finally:
            store.close()
