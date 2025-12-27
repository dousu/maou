"""Tests for interface data I/O module.

Updated to use DataFrame-based methods with Polars.
The interface now provides load_df_from_bytes/save_df_to_bytes for cloud storage integration.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.schema import (
    create_empty_hcpe_df,
    create_empty_preprocessing_df,
)
from maou.interface.data_io import (
    load_df_from_bytes,
    save_df_to_bytes,
)


class TestInterfaceDataIO:
    """Test interface layer DataFrame I/O functions."""

    def test_save_and_load_hcpe_bytes(self) -> None:
        """Test saving and loading HCPE DataFrame to/from bytes."""
        hcpe_df = create_empty_hcpe_df(5)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("eval", [10, 20, 30, 40, 50]),
            pl.Series("id", ["a", "b", "c", "d", "e"]),
        ])

        # Save to bytes
        bytes_data = save_df_to_bytes(hcpe_df, array_type="hcpe")

        # Load from bytes
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="hcpe"
        )

        # Verify
        assert loaded_df.shape == hcpe_df.shape
        assert loaded_df["eval"].to_list() == [10, 20, 30, 40, 50]
        assert loaded_df["id"].to_list() == [
            "a",
            "b",
            "c",
            "d",
            "e",
        ]

    def test_save_and_load_preprocessing_bytes(self) -> None:
        """Test saving and loading preprocessing DataFrame to/from bytes."""
        prep_df = create_empty_preprocessing_df(3)
        prep_df = prep_df.with_columns([
            pl.Series("id", [100, 200, 300]),
            pl.Series("resultValue", [0.5, -0.5, 0.0]),
        ])

        # Save to bytes
        bytes_data = save_df_to_bytes(
            prep_df, array_type="preprocessing"
        )

        # Load from bytes
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="preprocessing"
        )

        # Verify
        assert loaded_df.shape == prep_df.shape
        assert loaded_df["id"].to_list() == [100, 200, 300]
        assert loaded_df["resultValue"].to_list() == [
            0.5,
            -0.5,
            0.0,
        ]

    def test_hcpe_bytes_roundtrip_preserves_data(self) -> None:
        """Test that HCPE roundtrip preserves all data."""
        hcpe_df = create_empty_hcpe_df(2)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("eval", [500, -250]),
            pl.Series("id", ["x", "y"]),
        ])

        # Roundtrip
        bytes_data = save_df_to_bytes(hcpe_df, array_type="hcpe")
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="hcpe"
        )

        # Verify exact match
        assert loaded_df["eval"].to_list() == [500, -250]
        assert loaded_df["id"].to_list() == ["x", "y"]

    def test_preprocessing_bytes_roundtrip_preserves_data(
        self,
    ) -> None:
        """Test that preprocessing roundtrip preserves all data."""
        prep_df = create_empty_preprocessing_df(4)

        # Set some board positions
        board_positions = [
            [[j for j in range(9)] for _ in range(9)]
            for _ in range(4)
        ]

        prep_df = prep_df.with_columns([
            pl.Series("boardIdPositions", board_positions),
            pl.Series("resultValue", [0.1, 0.2, 0.3, 0.4]),
        ])

        # Roundtrip
        bytes_data = save_df_to_bytes(
            prep_df, array_type="preprocessing"
        )
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="preprocessing"
        )

        # Verify
        assert loaded_df["resultValue"].to_list() == [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
        assert (
            loaded_df["boardIdPositions"].to_list()
            == board_positions
        )

    def test_bytes_compression_effectiveness(self) -> None:
        """Test that bytes serialization produces compact output."""
        # Create a moderately sized DataFrame
        prep_df = create_empty_preprocessing_df(100)

        bytes_data = save_df_to_bytes(
            prep_df, array_type="preprocessing"
        )

        # Bytes should be reasonably sized (not megabytes for empty data)
        assert len(bytes_data) < 100_000  # Less than 100KB

    def test_empty_dataframe_bytes_roundtrip(self) -> None:
        """Test that empty DataFrames can be serialized and deserialized."""
        hcpe_df = create_empty_hcpe_df(0)

        bytes_data = save_df_to_bytes(hcpe_df, array_type="hcpe")
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="hcpe"
        )

        assert len(loaded_df) == 0
        assert loaded_df.schema == hcpe_df.schema

    def test_stage1_bytes_roundtrip(self) -> None:
        """Test stage1 DataFrame bytes serialization."""
        from maou.domain.data.schema import (
            create_empty_stage1_df,
        )

        stage1_df = create_empty_stage1_df(2)
        stage1_df = stage1_df.with_columns([
            pl.Series("id", [1, 2]),
        ])

        bytes_data = save_df_to_bytes(
            stage1_df, array_type="stage1"
        )
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="stage1"
        )

        assert loaded_df["id"].to_list() == [1, 2]

    def test_stage2_bytes_roundtrip(self) -> None:
        """Test stage2 DataFrame bytes serialization."""
        from maou.domain.data.schema import (
            create_empty_stage2_df,
        )

        stage2_df = create_empty_stage2_df(3)
        stage2_df = stage2_df.with_columns([
            pl.Series("id", [10, 20, 30]),
        ])

        bytes_data = save_df_to_bytes(
            stage2_df, array_type="stage2"
        )
        loaded_df = load_df_from_bytes(
            bytes_data, array_type="stage2"
        )

        assert loaded_df["id"].to_list() == [10, 20, 30]

    def test_bytes_can_be_written_to_file(self) -> None:
        """Test that bytes output can be written to file and read back."""
        hcpe_df = create_empty_hcpe_df(1)
        hcpe_df = hcpe_df.with_columns([
            pl.Series("id", ["test"]),
        ])

        bytes_data = save_df_to_bytes(hcpe_df, array_type="hcpe")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.bytes"

            # Write bytes to file
            file_path.write_bytes(bytes_data)

            # Read back
            read_bytes = file_path.read_bytes()
            loaded_df = load_df_from_bytes(
                read_bytes, array_type="hcpe"
            )

            assert loaded_df["id"].to_list() == ["test"]
