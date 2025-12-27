"""Tests for data I/O service.

Updated to use DataFrame-based methods with Polars.
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.app.common.data_io_service import (
    DataIOService,
)
from maou.domain.data.array_io import DataIOError
from maou.domain.data.schema import (
    create_empty_hcpe_df,
    create_empty_preprocessing_df,
)


class TestDataIOService:
    """Test DataIOService functionality with DataFrame-based I/O."""

    def test_load_dataframe_hcpe(self) -> None:
        """Test loading of HCPE DataFrame."""
        # Create test HCPE data
        hcpe_df = create_empty_hcpe_df(3)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("eval", [100, -50, 0]),
            pl.Series("id", ["test1", "test2", "test3"]),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_hcpe_game.feather"

            # Save using service
            DataIOService.save_dataframe(
                hcpe_df, file_path, array_type="hcpe"
            )

            # The .feather file should exist
            assert file_path.exists()

            # Load using service
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="hcpe"
            )

            # Verify data
            assert loaded_df.shape == hcpe_df.shape
            assert loaded_df["eval"].to_list() == [100, -50, 0]
            assert loaded_df["id"].to_list() == [
                "test1",
                "test2",
                "test3",
            ]

    def test_load_dataframe_preprocessing(self) -> None:
        """Test loading of preprocessing DataFrame."""
        # Create test preprocessing data
        prep_df = create_empty_preprocessing_df(2)

        # Create sample move labels (normalized)
        move_label_1 = np.zeros(2187, dtype=np.float16)
        move_label_1[50] = 1.0
        move_label_2 = np.zeros(2187, dtype=np.float16)
        move_label_2[100] = 1.0

        prep_df = prep_df.with_columns([
            pl.Series(
                "moveLabel",
                [move_label_1.tolist(), move_label_2.tolist()],
            ),
            pl.Series("resultValue", [1.0, 0.0]),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = (
                Path(temp_dir) / "test_preprocessing_data.feather"
            )

            # Save using service
            DataIOService.save_dataframe(
                prep_df, file_path, array_type="preprocessing"
            )

            # The .feather file should exist
            assert file_path.exists()

            # Load using service
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="preprocessing"
            )

            # Verify data
            assert loaded_df.shape == prep_df.shape
            assert loaded_df["resultValue"].to_list() == [1.0, 0.0]

    def test_save_dataframe_hcpe(self) -> None:
        """Test saving of HCPE DataFrame."""
        hcpe_df = create_empty_hcpe_df(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_save_hcpe.feather"

            # Save using service
            DataIOService.save_dataframe(
                hcpe_df, file_path, array_type="hcpe"
            )

            # Verify file was created
            assert file_path.exists()

            # Load and verify
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="hcpe"
            )

            assert loaded_df.shape == hcpe_df.shape

    def test_save_dataframe_preprocessing(self) -> None:
        """Test saving of preprocessing DataFrame."""
        prep_df = create_empty_preprocessing_df(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = (
                Path(temp_dir) / "test_save_preprocessing.feather"
            )

            # Save using service
            DataIOService.save_dataframe(
                prep_df, file_path, array_type="preprocessing"
            )

            # Verify file was created
            assert file_path.exists()

            # Load and verify
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="preprocessing"
            )

            assert loaded_df.shape == prep_df.shape

    def test_roundtrip_hcpe_dataframe(self) -> None:
        """Test roundtrip save/load for HCPE DataFrame preserves data."""
        hcpe_df = create_empty_hcpe_df(5)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("eval", [10, -20, 30, -40, 50]),
            pl.Series(
                "id", ["id1", "id2", "id3", "id4", "id5"]
            ),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "roundtrip_hcpe.feather"

            # Save
            DataIOService.save_dataframe(
                hcpe_df, file_path, array_type="hcpe"
            )

            # Load
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="hcpe"
            )

            # Verify
            assert loaded_df["eval"].to_list() == [
                10,
                -20,
                30,
                -40,
                50,
            ]
            assert loaded_df["id"].to_list() == [
                "id1",
                "id2",
                "id3",
                "id4",
                "id5",
            ]

    def test_roundtrip_preprocessing_dataframe(self) -> None:
        """Test roundtrip save/load for preprocessing DataFrame."""
        prep_df = create_empty_preprocessing_df(2)

        # Set some board positions
        board_1 = [[i + j for j in range(9)] for i in range(9)]
        board_2 = [
            [(i * 9 + j) % 30 for j in range(9)]
            for i in range(9)
        ]

        prep_df = prep_df.with_columns([
            pl.Series("boardIdPositions", [board_1, board_2]),
            pl.Series("resultValue", [0.5, -0.5]),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = (
                Path(temp_dir) / "roundtrip_preprocessing.feather"
            )

            # Save
            DataIOService.save_dataframe(
                prep_df, file_path, array_type="preprocessing"
            )

            # Load
            loaded_df = DataIOService.load_dataframe(
                file_path, array_type="preprocessing"
            )

            # Verify
            assert loaded_df["resultValue"].to_list() == [
                0.5,
                -0.5,
            ]
            assert (
                loaded_df["boardIdPositions"].to_list()[0] == board_1
            )
            assert (
                loaded_df["boardIdPositions"].to_list()[1] == board_2
            )

    def test_load_dataframe_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        with pytest.raises(DataIOError):
            DataIOService.load_dataframe(
                "non_existent_file.feather", array_type="hcpe"
            )

    def test_load_dataframe_invalid_type(self) -> None:
        """Test error handling for invalid array type."""
        hcpe_df = create_empty_hcpe_df(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.feather"

            DataIOService.save_dataframe(
                hcpe_df, file_path, array_type="hcpe"
            )

            # Should raise error with invalid type
            with pytest.raises(DataIOError):
                DataIOService.load_dataframe(
                    file_path,
                    array_type="invalid",  # type: ignore
                )

    def test_save_dataframe_creates_parent_directory(
        self,
    ) -> None:
        """Test that save_dataframe creates parent directories if needed."""
        hcpe_df = create_empty_hcpe_df(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path
            file_path = (
                Path(temp_dir)
                / "subdir1"
                / "subdir2"
                / "test.feather"
            )

            # Parent directories should be created automatically
            DataIOService.save_dataframe(
                hcpe_df, file_path, array_type="hcpe"
            )

            assert file_path.exists()
            assert file_path.parent.exists()


class TestDataFrameBytes:
    """Test byte serialization methods."""

    def test_load_dataframe_from_bytes_hcpe(self) -> None:
        """Test loading HCPE DataFrame from bytes."""
        hcpe_df = create_empty_hcpe_df(2)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("eval", [100, -50]),
        ])

        # Convert to bytes
        bytes_data = DataIOService.save_dataframe_to_bytes(
            hcpe_df, array_type="hcpe"
        )

        # Load from bytes
        loaded_df = DataIOService.load_dataframe_from_bytes(
            bytes_data, array_type="hcpe"
        )

        assert loaded_df["eval"].to_list() == [100, -50]

    def test_load_dataframe_from_bytes_preprocessing(
        self,
    ) -> None:
        """Test loading preprocessing DataFrame from bytes."""
        prep_df = create_empty_preprocessing_df(1)
        prep_df = prep_df.with_columns([
            pl.Series("resultValue", [0.75]),
        ])

        # Convert to bytes
        bytes_data = DataIOService.save_dataframe_to_bytes(
            prep_df, array_type="preprocessing"
        )

        # Load from bytes
        loaded_df = DataIOService.load_dataframe_from_bytes(
            bytes_data, array_type="preprocessing"
        )

        assert loaded_df["resultValue"].to_list() == [0.75]

    def test_bytes_roundtrip_preserves_data(self) -> None:
        """Test that bytes serialization roundtrip preserves data."""
        hcpe_df = create_empty_hcpe_df(3)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("id", ["a", "b", "c"]),
            pl.Series("eval", [1, 2, 3]),
        ])

        # Roundtrip
        bytes_data = DataIOService.save_dataframe_to_bytes(
            hcpe_df, array_type="hcpe"
        )
        loaded_df = DataIOService.load_dataframe_from_bytes(
            bytes_data, array_type="hcpe"
        )

        # Verify
        assert loaded_df["id"].to_list() == ["a", "b", "c"]
        assert loaded_df["eval"].to_list() == [1, 2, 3]
