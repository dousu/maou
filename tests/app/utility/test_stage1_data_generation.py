import tempfile
from pathlib import Path

from maou.app.utility.stage1_data_generation import (
    Stage1DataGenerationConfig,
    Stage1DataGenerationUseCase,
)
from maou.domain.data.rust_io import load_stage1_df
from maou.domain.data.schema import get_stage1_polars_schema


class TestStage1DataGenerationUseCase:
    """Test Stage 1 data generation use case."""

    def test_execute_creates_output_file(self) -> None:
        """Test use case creates output .feather file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = Stage1DataGenerationConfig(
                output_dir=output_dir
            )

            use_case = Stage1DataGenerationUseCase()
            result = use_case.execute(config)

            # Check result
            assert result["total_patterns"] == 1105
            assert (
                "stage1_data.feather" in result["output_file"]
            )

            # Check file exists
            output_file = Path(result["output_file"])
            assert output_file.exists()

    def test_execute_saves_valid_data(self) -> None:
        """Test saved data can be loaded and is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = Stage1DataGenerationConfig(
                output_dir=output_dir
            )

            use_case = Stage1DataGenerationUseCase()
            result = use_case.execute(config)

            # Load saved data
            output_file = Path(result["output_file"])
            df = load_stage1_df(output_file)

            # Validate schema
            assert df.schema == get_stage1_polars_schema()

            # Validate record count
            assert len(df) == 1105

    def test_execute_creates_nested_directory(self) -> None:
        """Test use case creates nested directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "directories"
            config = Stage1DataGenerationConfig(
                output_dir=output_dir
            )

            use_case = Stage1DataGenerationUseCase()
            result = use_case.execute(config)

            # Check directory was created
            assert output_dir.exists()
            assert output_dir.is_dir()

            # Check file exists
            output_file = Path(result["output_file"])
            assert output_file.exists()

    def test_execute_validates_data_quality(self) -> None:
        """Test generated data has expected quality properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = Stage1DataGenerationConfig(
                output_dir=output_dir
            )

            use_case = Stage1DataGenerationUseCase()
            result = use_case.execute(config)

            # Load and validate data
            output_file = Path(result["output_file"])
            df = load_stage1_df(output_file)

            # All IDs should be unique
            assert df["id"].n_unique() == len(df)

            # All reachable squares should be binary
            for squares in df["reachableSquares"].to_list():
                for row in squares:
                    for val in row:
                        assert val in (0, 1), (
                            f"Non-binary value: {val}"
                        )

            # Board positions should be 9x9
            for board in df["boardIdPositions"].to_list():
                assert len(board) == 9
                for row in board:
                    assert len(row) == 9

            # Pieces in hand should be 14 elements
            for pieces in df["piecesInHand"].to_list():
                assert len(pieces) == 14
