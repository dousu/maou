"""Tests for HCPE transform.

Updated to use DataFrame-based I/O with Polars.
Simplified from original numpy-based tests - comprehensive integration tests exist in tests/maou/integrations/.
"""

from pathlib import Path

import numpy as np
import polars as pl

from maou.app.pre_process.hcpe_transform import PreProcess
from maou.domain.data.rust_io import save_hcpe_df
from maou.domain.data.schema import create_empty_hcpe_df
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


def _create_test_hcpe_data(
    directory: Path, samples: int
) -> list[Path]:
    """Create test HCPE DataFrame files."""
    df = create_empty_hcpe_df(samples)

    # Add minimal test data
    rng = np.random.default_rng(42)
    ids = [f"test_id_{i}" for i in range(samples)]
    eval_values = rng.integers(-100, 100, size=samples).tolist()

    # Use a valid move from initial position (1g1f)
    # This is the first legal move in standard shogi position
    valid_move16 = 66309  # 0x10305
    best_moves = [valid_move16] * samples

    # Set game results (1 = first player wins)
    game_results = [1] * samples

    df = df.with_columns(
        [
            pl.Series("id", ids),
            pl.Series("eval", eval_values),
            pl.Series("bestMove16", best_moves),
            pl.Series("gameResult", game_results),
        ]
    )

    file_path = directory / "test_hcpe.feather"
    save_hcpe_df(df, file_path)

    return [file_path]


def test_preprocess_basic_transformation(
    tmp_path: Path,
) -> None:
    """Test basic HCPE preprocessing transformation."""
    input_paths = _create_test_hcpe_data(
        tmp_path / "input", samples=5
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create datasource
    datasource = FileDataSource(
        file_paths=input_paths,
        array_type="hcpe",
    )

    # Create preprocessing option
    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=1,
    )

    # Run preprocessing (without feature store)
    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
    )

    preprocessor.transform(option)

    # Verify output file was created
    output_files = list(output_dir.glob("*.feather"))
    assert len(output_files) > 0


def test_preprocess_with_multiple_input_files(
    tmp_path: Path,
) -> None:
    """Test preprocessing with multiple input files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create multiple input files
    input_paths = []
    valid_move16 = 66309  # 0x10305 (1g1f - valid move from initial position)
    for i in range(3):
        df = create_empty_hcpe_df(2)
        df = df.with_columns(
            [
                pl.Series(
                    "id", [f"file{i}_id{j}" for j in range(2)]
                ),
                pl.Series("bestMove16", [valid_move16] * 2),
                pl.Series("gameResult", [1] * 2),
            ]
        )

        file_path = input_dir / f"hcpe_{i}.feather"
        save_hcpe_df(df, file_path)
        input_paths.append(file_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create datasource
    datasource = FileDataSource(
        file_paths=input_paths,
        array_type="hcpe",
    )

    # Run preprocessing
    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=1,
    )

    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
    )

    preprocessor.transform(option)

    # Verify output was created
    output_files = list(output_dir.glob("*.feather"))
    assert len(output_files) > 0


def test_preprocess_parallel_workers(tmp_path: Path) -> None:
    """Test preprocessing with multiple workers."""
    input_paths = _create_test_hcpe_data(
        tmp_path / "input", samples=10
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    datasource = FileDataSource(
        file_paths=input_paths,
        array_type="hcpe",
    )

    # Use multiple workers
    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=2,
    )

    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
    )

    preprocessor.transform(option)

    # Verify output was created
    output_files = list(output_dir.glob("*.feather"))
    assert len(output_files) > 0


def test_preprocess_with_large_batch_size(
    tmp_path: Path,
) -> None:
    """Test preprocessing with large DuckDB batch size (A100 default)."""
    input_paths = _create_test_hcpe_data(
        tmp_path / "input", samples=10
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    datasource = FileDataSource(
        file_paths=input_paths,
        array_type="hcpe",
    )

    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=1,
    )

    # Use the A100-optimized batch size
    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
        intermediate_batch_size=50_000,
    )

    preprocessor.transform(option)

    output_files = list(output_dir.glob("*.feather"))
    assert len(output_files) > 0


def test_preprocess_with_input_splitting(
    tmp_path: Path,
) -> None:
    """Test preprocessing with input file pre-splitting."""
    from maou.domain.data.rust_io import (
        RUST_BACKEND_AVAILABLE,
    )
    from maou.interface.preprocess import split_input_files

    if not RUST_BACKEND_AVAILABLE:
        import pytest

        pytest.skip("Rust backend not available")

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a single file with 20 records
    valid_move16 = 66309
    df = create_empty_hcpe_df(20)
    df = df.with_columns(
        [
            pl.Series(
                "id", [f"split_test_id_{i}" for i in range(20)]
            ),
            pl.Series("bestMove16", [valid_move16] * 20),
            pl.Series("gameResult", [1] * 20),
        ]
    )
    file_path = input_dir / "large_hcpe.feather"
    save_hcpe_df(df, file_path)

    # Split into files of 7 rows each
    split_dir = tmp_path / "split"
    split_paths = split_input_files(
        file_paths=[file_path],
        rows_per_file=7,
        split_dir=split_dir,
    )

    # Should produce 3 files (7+7+6)
    assert len(split_paths) == 3

    # Create datasource from split files
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    datasource = FileDataSource(
        file_paths=split_paths,
        array_type="hcpe",
    )

    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=1,
    )

    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
    )

    preprocessor.transform(option)

    output_files = list(output_dir.glob("*.feather"))
    assert len(output_files) > 0
