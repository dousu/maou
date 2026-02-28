"""Tests for feather file splitting and merging (Rust backend)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.domain.data.rust_io import (
    RUST_BACKEND_AVAILABLE,
    load_hcpe_df,
    merge_hcpe_feather_files,
    save_hcpe_df,
    split_hcpe_feather,
)
from maou.domain.data.schema import create_empty_hcpe_df

pytestmark = pytest.mark.skipif(
    not RUST_BACKEND_AVAILABLE,
    reason="Rust backend not available",
)


def _create_hcpe_file(
    directory: Path, filename: str, num_rows: int
) -> Path:
    """Create a test HCPE feather file with specified number of rows."""
    df = create_empty_hcpe_df(num_rows)

    rng = np.random.default_rng(42)
    ids = [f"test_id_{i}" for i in range(num_rows)]
    eval_values = rng.integers(
        -100, 100, size=num_rows
    ).tolist()
    valid_move16 = 66309
    best_moves = [valid_move16] * num_rows
    game_results = [1] * num_rows

    df = df.with_columns(
        [
            pl.Series("id", ids),
            pl.Series("eval", eval_values),
            pl.Series("bestMove16", best_moves),
            pl.Series("gameResult", game_results),
        ]
    )

    file_path = directory / filename
    directory.mkdir(parents=True, exist_ok=True)
    save_hcpe_df(df, file_path)
    return file_path


class TestSplitHcpeFeather:
    """Test feather file splitting."""

    def test_no_split_when_file_is_small(
        self, tmp_path: Path
    ) -> None:
        """Test that files smaller than rows_per_file are not split."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "small.feather", 10)

        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=100
        )

        # Should return original file path (no split needed)
        assert len(result) == 1
        assert result[0] == fp

    def test_split_into_multiple_files(
        self, tmp_path: Path
    ) -> None:
        """Test splitting a large file into multiple smaller files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "large.feather", 100)

        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=30
        )

        # 100 rows / 30 per file = 4 files (30+30+30+10)
        assert len(result) == 4

        # Verify row counts
        total_rows = 0
        for split_path in result:
            assert split_path.exists()
            df = load_hcpe_df(split_path)
            total_rows += len(df)

        assert total_rows == 100

    def test_split_preserves_schema(
        self, tmp_path: Path
    ) -> None:
        """Test that split files preserve the HCPE schema."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(
            input_dir, "schema_test.feather", 20
        )

        original_df = load_hcpe_df(fp)
        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=7
        )

        for split_path in result:
            df = load_hcpe_df(split_path)
            assert df.schema == original_df.schema, (
                f"Schema mismatch in {split_path}"
            )

    def test_split_preserves_data(self, tmp_path: Path) -> None:
        """Test that split files contain all original data."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(
            input_dir, "data_test.feather", 50
        )

        original_df = load_hcpe_df(fp)
        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=15
        )

        # Concatenate all split files
        all_dfs = [load_hcpe_df(p) for p in result]
        combined = pl.concat(all_dfs)

        assert len(combined) == len(original_df)
        # Verify eval values match (id column may be null in some rows)
        assert (
            combined["eval"].to_list()
            == original_df["eval"].to_list()
        )

    def test_invalid_rows_per_file(
        self, tmp_path: Path
    ) -> None:
        """Test that invalid rows_per_file raises ValueError."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "test.feather", 10)

        with pytest.raises(ValueError):
            split_hcpe_feather(fp, output_dir, rows_per_file=0)

        with pytest.raises(ValueError):
            split_hcpe_feather(fp, output_dir, rows_per_file=-1)

    def test_output_dir_created_automatically(
        self, tmp_path: Path
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "deeply" / "nested" / "output"
        fp = _create_hcpe_file(input_dir, "test.feather", 20)

        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=7
        )

        assert output_dir.exists()
        assert len(result) == 3  # 20/7 = 3 files (7+7+6)

    def test_exact_split_boundary(self, tmp_path: Path) -> None:
        """Test splitting when rows divide evenly."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "exact.feather", 30)

        result = split_hcpe_feather(
            fp, output_dir, rows_per_file=10
        )

        assert len(result) == 3
        for split_path in result:
            df = load_hcpe_df(split_path)
            assert len(df) == 10


class TestSplitInputFiles:
    """Test interface-level file splitting."""

    def test_split_input_files_integration(
        self, tmp_path: Path
    ) -> None:
        """Test split_input_files from interface layer."""
        from maou.interface.preprocess import (
            split_input_files,
        )

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create 2 files: one large, one small
        fp_large = _create_hcpe_file(
            input_dir, "large.feather", 50
        )
        fp_small = _create_hcpe_file(
            input_dir, "small.feather", 5
        )

        result = split_input_files(
            file_paths=[fp_large, fp_small],
            rows_per_file=20,
            split_dir=output_dir,
        )

        # large.feather (50 rows) → 3 files (20+20+10)
        # small.feather (5 rows) → 1 file (no split)
        assert len(result) == 4

    def test_split_input_files_no_split_needed(
        self, tmp_path: Path
    ) -> None:
        """Test that small files are not split."""
        from maou.interface.preprocess import (
            split_input_files,
        )

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        fps = [
            _create_hcpe_file(
                input_dir, f"small_{i}.feather", 10
            )
            for i in range(3)
        ]

        result = split_input_files(
            file_paths=fps,
            rows_per_file=100,
            split_dir=output_dir,
        )

        # No files should be split
        assert len(result) == 3
        assert result == fps


class TestMergeHcpeFeatherFiles:
    """Test feather file merging/chunking."""

    def test_merge_multiple_small_files(
        self, tmp_path: Path
    ) -> None:
        """Test merging multiple small files into one."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        fps = [
            _create_hcpe_file(
                input_dir, f"small_{i}.feather", 10
            )
            for i in range(5)
        ]

        result = merge_hcpe_feather_files(
            file_paths=fps,
            output_dir=output_dir,
            rows_per_chunk=100,
        )

        # 50 total rows, chunk size 100 → 1 file
        assert len(result) == 1
        df = load_hcpe_df(result[0])
        assert len(df) == 50

    def test_merge_into_multiple_chunks(
        self, tmp_path: Path
    ) -> None:
        """Test merging files into multiple chunks."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        fps = [
            _create_hcpe_file(
                input_dir, f"small_{i}.feather", 10
            )
            for i in range(5)
        ]

        result = merge_hcpe_feather_files(
            file_paths=fps,
            output_dir=output_dir,
            rows_per_chunk=25,
        )

        # 50 total rows, chunk size 25 → 3 files
        # (10+10=20, 10+10=20, 10=10)
        assert len(result) == 3
        total_rows = sum(len(load_hcpe_df(p)) for p in result)
        assert total_rows == 50

    def test_merge_preserves_schema(
        self, tmp_path: Path
    ) -> None:
        """Test that merged files preserve schema."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        fps = [
            _create_hcpe_file(
                input_dir, f"data_{i}.feather", 10
            )
            for i in range(3)
        ]

        original_df = load_hcpe_df(fps[0])
        result = merge_hcpe_feather_files(
            file_paths=fps,
            output_dir=output_dir,
            rows_per_chunk=100,
        )

        merged_df = load_hcpe_df(result[0])
        assert merged_df.schema == original_df.schema

    def test_merge_empty_list(self, tmp_path: Path) -> None:
        """Test merging empty file list."""
        output_dir = tmp_path / "output"

        result = merge_hcpe_feather_files(
            file_paths=[],
            output_dir=output_dir,
            rows_per_chunk=100,
        )

        assert result == []

    def test_merge_invalid_rows_per_chunk(
        self, tmp_path: Path
    ) -> None:
        """Test that invalid rows_per_chunk raises."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "test.feather", 10)

        with pytest.raises(ValueError):
            merge_hcpe_feather_files(
                file_paths=[fp],
                output_dir=output_dir,
                rows_per_chunk=0,
            )

        with pytest.raises(ValueError):
            merge_hcpe_feather_files(
                file_paths=[fp],
                output_dir=output_dir,
                rows_per_chunk=-1,
            )

    def test_merge_single_file(self, tmp_path: Path) -> None:
        """Test merging with single file returns chunked copy."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fp = _create_hcpe_file(input_dir, "single.feather", 10)

        result = merge_hcpe_feather_files(
            file_paths=[fp],
            output_dir=output_dir,
            rows_per_chunk=100,
        )

        assert len(result) == 1
        df = load_hcpe_df(result[0])
        assert len(df) == 10


class TestChunkInputFiles:
    """Test interface-level file chunking."""

    def test_chunk_many_small_files(
        self, tmp_path: Path
    ) -> None:
        """Test chunking small files from interface."""
        from maou.interface.preprocess import (
            chunk_input_files,
        )

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        fps = [
            _create_hcpe_file(input_dir, f"tiny_{i}.feather", 5)
            for i in range(10)
        ]

        result = chunk_input_files(
            file_paths=fps,
            rows_per_chunk=20,
            chunk_dir=output_dir,
        )

        # 50 total rows, chunk 20 → 3 files
        assert len(result) < len(fps)
        total = sum(len(load_hcpe_df(p)) for p in result)
        assert total == 50

    def test_chunk_single_file_no_change(
        self, tmp_path: Path
    ) -> None:
        """Test that single file is not chunked."""
        from maou.interface.preprocess import (
            chunk_input_files,
        )

        input_dir = tmp_path / "input"
        fp = _create_hcpe_file(input_dir, "single.feather", 10)

        result = chunk_input_files(
            file_paths=[fp],
            rows_per_chunk=100,
        )

        assert result == [fp]


class TestResizeInputFiles:
    """Test combined split + chunk (resize)."""

    def test_resize_splits_large_and_chunks_small(
        self, tmp_path: Path
    ) -> None:
        """Test resize handles both large and small files."""
        from maou.interface.preprocess import (
            resize_input_files,
        )

        input_dir = tmp_path / "input"
        work_dir = tmp_path / "work"

        # 1 large file (100 rows) + 5 small files (5 rows)
        fp_large = _create_hcpe_file(
            input_dir, "large.feather", 100
        )
        fps_small = [
            _create_hcpe_file(input_dir, f"tiny_{i}.feather", 5)
            for i in range(5)
        ]

        result = resize_input_files(
            file_paths=[fp_large] + fps_small,
            rows_per_file=30,
            work_dir=work_dir,
        )

        # large (100) → split into 4 (30+30+30+10)
        # 5 small (5 each=25 total) → all below threshold
        #   (15), so merged into 1 chunk
        # Total: 4 + 1 = 5 files
        total = sum(len(load_hcpe_df(p)) for p in result)
        assert total == 125
        assert len(result) < 6 + 1  # fewer than original

    def test_resize_no_op_when_all_ok(
        self, tmp_path: Path
    ) -> None:
        """Test resize is no-op when files are right size."""
        from maou.interface.preprocess import (
            resize_input_files,
        )

        input_dir = tmp_path / "input"
        work_dir = tmp_path / "work"

        fps = [
            _create_hcpe_file(input_dir, f"ok_{i}.feather", 30)
            for i in range(3)
        ]

        result = resize_input_files(
            file_paths=fps,
            rows_per_file=50,
            work_dir=work_dir,
        )

        # 30 rows each, threshold is 25 → all above, no merge
        # 30 < 50 → no split
        assert len(result) == 3

    def test_resize_empty_input(self, tmp_path: Path) -> None:
        """Test resize with empty input."""
        from maou.interface.preprocess import (
            resize_input_files,
        )

        result = resize_input_files(
            file_paths=[],
            rows_per_file=100,
        )

        assert result == []
