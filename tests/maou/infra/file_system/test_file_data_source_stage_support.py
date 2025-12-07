"""Tests for FileDataSource stage1 and stage2 array type support."""

import tempfile
from pathlib import Path

import numpy as np

from maou.domain.data.schema import (
    create_empty_stage1_array,
    create_empty_stage2_array,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


class TestFileDataSourceStage1Support:
    """Test FileDataSource support for stage1 array type."""

    def test_load_stage1_data(self) -> None:
        """Test loading stage1 data from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            size = 10
            test_data = create_empty_stage1_array(size)

            # Fill with test values
            for i in range(size):
                test_data["id"][i] = i
                test_data["boardIdPositions"][i] = (
                    np.random.randint(
                        0, 40, size=(9, 9), dtype=np.uint8
                    )
                )
                test_data["piecesInHand"][i] = (
                    np.random.randint(
                        0, 19, size=(14,), dtype=np.uint8
                    )
                )
                test_data["reachableSquares"][i] = (
                    np.random.randint(
                        0, 2, size=(9, 9), dtype=np.uint8
                    )
                )

            # Save to file
            file_path = Path(tmpdir) / "stage1_test.npy"
            np.save(file_path, test_data)

            # Load using FileDataSource
            datasource = FileDataSource(
                file_paths=[file_path],
                array_type="stage1",
                bit_pack=False,
                cache_mode="mmap",
            )

            # Verify length
            assert len(datasource) == size

            # Verify data integrity
            for i in range(size):
                record = datasource[i]
                assert record["id"] == test_data["id"][i]
                assert np.array_equal(
                    record["boardIdPositions"],
                    test_data["boardIdPositions"][i],
                )
                assert np.array_equal(
                    record["piecesInHand"],
                    test_data["piecesInHand"][i],
                )
                assert np.array_equal(
                    record["reachableSquares"],
                    test_data["reachableSquares"][i],
                )

    def test_load_stage1_data_memory_mode(self) -> None:
        """Test loading stage1 data with cache_mode='memory'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = 5
            test_data = create_empty_stage1_array(size)

            for i in range(size):
                test_data["id"][i] = i
                test_data["reachableSquares"][i, 0, :] = (
                    1  # First row reachable
                )

            file_path = Path(tmpdir) / "stage1_test.npy"
            np.save(file_path, test_data)

            # Load with memory mode
            datasource = FileDataSource(
                file_paths=[file_path],
                array_type="stage1",
                bit_pack=False,
                cache_mode="memory",
            )

            assert len(datasource) == size

            # Verify first row is reachable
            for i in range(size):
                record = datasource[i]
                assert np.all(
                    record["reachableSquares"][0, :] == 1
                )


class TestFileDataSourceStage2Support:
    """Test FileDataSource support for stage2 array type."""

    def test_load_stage2_data(self) -> None:
        """Test loading stage2 data from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            size = 10
            test_data = create_empty_stage2_array(size)

            # Fill with test values
            from maou.app.pre_process.label import (
                MOVE_LABELS_NUM,
            )

            for i in range(size):
                test_data["id"][i] = i
                test_data["boardIdPositions"][i] = (
                    np.random.randint(
                        0, 40, size=(9, 9), dtype=np.uint8
                    )
                )
                test_data["piecesInHand"][i] = (
                    np.random.randint(
                        0, 19, size=(14,), dtype=np.uint8
                    )
                )
                # Set some legal moves
                num_legal = np.random.randint(20, 50)
                legal_indices = np.random.choice(
                    MOVE_LABELS_NUM, num_legal, replace=False
                )
                test_data["legalMovesLabel"][
                    i, legal_indices
                ] = 1

            # Save to file
            file_path = Path(tmpdir) / "stage2_test.npy"
            np.save(file_path, test_data)

            # Load using FileDataSource
            datasource = FileDataSource(
                file_paths=[file_path],
                array_type="stage2",
                bit_pack=False,
                cache_mode="mmap",
            )

            # Verify length
            assert len(datasource) == size

            # Verify data integrity
            for i in range(size):
                record = datasource[i]
                assert record["id"] == test_data["id"][i]
                assert np.array_equal(
                    record["boardIdPositions"],
                    test_data["boardIdPositions"][i],
                )
                assert np.array_equal(
                    record["piecesInHand"],
                    test_data["piecesInHand"][i],
                )
                assert np.array_equal(
                    record["legalMovesLabel"],
                    test_data["legalMovesLabel"][i],
                )

    def test_load_stage2_data_memory_mode(self) -> None:
        """Test loading stage2 data with cache_mode='memory'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = 5
            test_data = create_empty_stage2_array(size)

            for i in range(size):
                test_data["id"][i] = i
                # Set first 30 moves as legal
                test_data["legalMovesLabel"][i, :30] = 1

            file_path = Path(tmpdir) / "stage2_test.npy"
            np.save(file_path, test_data)

            # Load with memory mode
            datasource = FileDataSource(
                file_paths=[file_path],
                array_type="stage2",
                bit_pack=False,
                cache_mode="memory",
            )

            assert len(datasource) == size

            # Verify first 30 moves are legal
            for i in range(size):
                record = datasource[i]
                assert np.all(
                    record["legalMovesLabel"][:30] == 1
                )
                assert np.all(
                    record["legalMovesLabel"][30:] == 0
                )


class TestFileDataSourceSpliterStageSupport:
    """Test FileDataSourceSpliter with stage array types."""

    def test_train_test_split_stage1(self) -> None:
        """Test train/test split with stage1 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = 20
            test_data = create_empty_stage1_array(size)

            for i in range(size):
                test_data["id"][i] = i

            file_path = Path(tmpdir) / "stage1_test.npy"
            np.save(file_path, test_data)

            # Create spliter
            spliter = FileDataSource.FileDataSourceSpliter(
                file_paths=[file_path],
                array_type="stage1",
                bit_pack=False,
                cache_mode="mmap",
            )

            # Split into train/test
            train_ds, test_ds = spliter.train_test_split(
                test_ratio=0.2
            )

            # Verify sizes
            assert len(train_ds) == 16  # 80% of 20
            assert len(test_ds) == 4  # 20% of 20

            # Verify total is preserved
            assert len(train_ds) + len(test_ds) == size

    def test_train_test_split_stage2(self) -> None:
        """Test train/test split with stage2 data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = 20
            test_data = create_empty_stage2_array(size)

            for i in range(size):
                test_data["id"][i] = i

            file_path = Path(tmpdir) / "stage2_test.npy"
            np.save(file_path, test_data)

            # Create spliter
            spliter = FileDataSource.FileDataSourceSpliter(
                file_paths=[file_path],
                array_type="stage2",
                bit_pack=False,
                cache_mode="mmap",
            )

            # Split into train/test
            train_ds, test_ds = spliter.train_test_split(
                test_ratio=0.25
            )

            # Verify sizes
            assert len(train_ds) == 15  # 75% of 20
            assert len(test_ds) == 5  # 25% of 20

            # Verify total is preserved
            assert len(train_ds) + len(test_ds) == size
