"""Tests for Stage 1 and Stage 2 datasets."""

import numpy as np
import torch

from maou.app.learning.dataset import (
    Stage1Dataset,
    Stage2Dataset,
)
from maou.domain.data.schema import (
    create_empty_stage1_array,
    create_empty_stage2_array,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class MockDataSource:
    """Mock data source for testing."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._array[idx]

    def __len__(self) -> int:
        return len(self._array)


class TestStage1Dataset:
    """Test Stage1Dataset for reachable squares training."""

    def test_initialization(self) -> None:
        """Test dataset initialization."""
        size = 10
        array = create_empty_stage1_array(size)
        datasource = MockDataSource(array)

        dataset = Stage1Dataset(datasource=datasource)

        assert len(dataset) == size

    def test_len(self) -> None:
        """Test __len__ returns correct size."""
        size = 20
        array = create_empty_stage1_array(size)
        datasource = MockDataSource(array)

        dataset = Stage1Dataset(datasource=datasource)

        assert len(dataset) == size

    def test_getitem_returns_correct_structure(self) -> None:
        """Test __getitem__ returns (features, target) tuple."""
        size = 5
        array = create_empty_stage1_array(size)
        datasource = MockDataSource(array)

        dataset = Stage1Dataset(datasource=datasource)

        features, target = dataset[0]

        # Features should be tuple of (board_tensor, pieces_in_hand_tensor)
        assert isinstance(features, tuple)
        assert len(features) == 2

        # Target should be tensor
        assert isinstance(target, torch.Tensor)

    def test_getitem_feature_shapes(self) -> None:
        """Test __getitem__ returns correct feature shapes."""
        size = 5
        array = create_empty_stage1_array(size)
        datasource = MockDataSource(array)

        dataset = Stage1Dataset(datasource=datasource)

        features, _ = dataset[0]
        board_tensor, pieces_in_hand_tensor = features

        # Board tensor should be (9, 9)
        assert board_tensor.shape == (9, 9)
        assert board_tensor.dtype == torch.uint8

        # Pieces in hand tensor should be (14,)
        assert pieces_in_hand_tensor.shape == (14,)
        assert pieces_in_hand_tensor.dtype == torch.uint8

    def test_getitem_target_shape(self) -> None:
        """Test __getitem__ returns correct target shape."""
        size = 5
        array = create_empty_stage1_array(size)
        datasource = MockDataSource(array)

        dataset = Stage1Dataset(datasource=datasource)

        _, target = dataset[0]

        # Target should be flattened (81,) and float
        assert target.shape == (81,)
        assert target.dtype == torch.float32

    def test_getitem_target_is_flattened(self) -> None:
        """Test target reachable squares are flattened from (9,9) to (81,)."""
        size = 5
        array = create_empty_stage1_array(size)

        # Set specific pattern in reachable squares
        array["reachableSquares"][0] = np.eye(
            9, dtype=np.uint8
        )  # Diagonal

        datasource = MockDataSource(array)
        dataset = Stage1Dataset(datasource=datasource)

        _, target = dataset[0]

        # Should be flattened
        assert target.shape == (81,)

        # Check diagonal pattern is preserved
        expected = torch.from_numpy(
            np.eye(9, dtype=np.uint8).flatten()
        ).float()
        assert torch.equal(target, expected)

    def test_getitem_multiple_samples(self) -> None:
        """Test __getitem__ works for multiple samples."""
        size = 10
        array = create_empty_stage1_array(size)

        # Set different patterns for different samples
        for i in range(size):
            array["reachableSquares"][i, 0, :] = (
                1  # First row reachable
            )
            array["boardIdPositions"][i] = np.random.randint(
                0, 40, size=(9, 9)
            )
            array["piecesInHand"][i] = np.random.randint(
                0, 19, size=(14,)
            )

        datasource = MockDataSource(array)
        dataset = Stage1Dataset(datasource=datasource)

        # Access all samples
        for i in range(size):
            features, target = dataset[i]

            assert features[0].shape == (9, 9)  # board
            assert features[1].shape == (14,)  # hand
            assert target.shape == (81,)  # flattened target

            # First 9 elements should be 1 (first row)
            assert torch.all(target[:9] == 1.0)

    def test_getitem_preserves_data_values(self) -> None:
        """Test __getitem__ preserves data values correctly."""
        size = 5
        array = create_empty_stage1_array(size)

        # Set specific values
        board_values = np.random.randint(
            0, 40, size=(9, 9), dtype=np.uint8
        )
        hand_values = np.random.randint(
            0, 19, size=(14,), dtype=np.uint8
        )
        reachable_values = np.random.randint(
            0, 2, size=(9, 9), dtype=np.uint8
        )

        array["boardIdPositions"][0] = board_values
        array["piecesInHand"][0] = hand_values
        array["reachableSquares"][0] = reachable_values

        datasource = MockDataSource(array)
        dataset = Stage1Dataset(datasource=datasource)

        features, target = dataset[0]
        board_tensor, hand_tensor = features

        # Check values are preserved
        assert torch.equal(
            board_tensor, torch.from_numpy(board_values)
        )
        assert torch.equal(
            hand_tensor, torch.from_numpy(hand_values)
        )
        assert torch.equal(
            target,
            torch.from_numpy(
                reachable_values.flatten()
            ).float(),
        )


class TestStage2Dataset:
    """Test Stage2Dataset for legal moves training."""

    def test_initialization(self) -> None:
        """Test dataset initialization."""
        size = 10
        array = create_empty_stage2_array(size)
        datasource = MockDataSource(array)

        dataset = Stage2Dataset(datasource=datasource)

        assert len(dataset) == size

    def test_len(self) -> None:
        """Test __len__ returns correct size."""
        size = 20
        array = create_empty_stage2_array(size)
        datasource = MockDataSource(array)

        dataset = Stage2Dataset(datasource=datasource)

        assert len(dataset) == size

    def test_getitem_returns_correct_structure(self) -> None:
        """Test __getitem__ returns (features, target) tuple."""
        size = 5
        array = create_empty_stage2_array(size)
        datasource = MockDataSource(array)

        dataset = Stage2Dataset(datasource=datasource)

        features, target = dataset[0]

        # Features should be tuple of (board_tensor, pieces_in_hand_tensor)
        assert isinstance(features, tuple)
        assert len(features) == 2

        # Target should be tensor
        assert isinstance(target, torch.Tensor)

    def test_getitem_feature_shapes(self) -> None:
        """Test __getitem__ returns correct feature shapes."""
        size = 5
        array = create_empty_stage2_array(size)
        datasource = MockDataSource(array)

        dataset = Stage2Dataset(datasource=datasource)

        features, _ = dataset[0]
        board_tensor, pieces_in_hand_tensor = features

        # Board tensor should be (9, 9)
        assert board_tensor.shape == (9, 9)
        assert board_tensor.dtype == torch.uint8

        # Pieces in hand tensor should be (14,)
        assert pieces_in_hand_tensor.shape == (14,)
        assert pieces_in_hand_tensor.dtype == torch.uint8

    def test_getitem_target_shape(self) -> None:
        """Test __getitem__ returns correct target shape."""
        size = 5
        array = create_empty_stage2_array(size)
        datasource = MockDataSource(array)

        dataset = Stage2Dataset(datasource=datasource)

        _, target = dataset[0]

        # Target should be (MOVE_LABELS_NUM,) and float
        assert target.shape == (MOVE_LABELS_NUM,)
        assert target.dtype == torch.float32

    def test_getitem_target_is_binary(self) -> None:
        """Test target legal moves are binary (0 or 1)."""
        size = 5
        array = create_empty_stage2_array(size)

        # Set some moves as legal
        legal_moves = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
        legal_moves[:50] = 1  # First 50 moves are legal
        array["legalMovesLabel"][0] = legal_moves

        datasource = MockDataSource(array)
        dataset = Stage2Dataset(datasource=datasource)

        _, target = dataset[0]

        # All values should be 0 or 1
        assert torch.all((target == 0) | (target == 1))

        # First 50 should be 1, rest should be 0
        assert torch.all(target[:50] == 1.0)
        assert torch.all(target[50:] == 0.0)

    def test_getitem_multiple_samples(self) -> None:
        """Test __getitem__ works for multiple samples."""
        size = 10
        array = create_empty_stage2_array(size)

        # Set different legal move patterns
        for i in range(size):
            num_legal = np.random.randint(20, 60)
            legal_indices = np.random.choice(
                MOVE_LABELS_NUM, num_legal, replace=False
            )
            array["legalMovesLabel"][i, legal_indices] = 1
            array["boardIdPositions"][i] = np.random.randint(
                0, 40, size=(9, 9)
            )
            array["piecesInHand"][i] = np.random.randint(
                0, 19, size=(14,)
            )

        datasource = MockDataSource(array)
        dataset = Stage2Dataset(datasource=datasource)

        # Access all samples
        for i in range(size):
            features, target = dataset[i]

            assert features[0].shape == (9, 9)  # board
            assert features[1].shape == (14,)  # hand
            assert target.shape == (MOVE_LABELS_NUM,)

            # Should have some legal moves
            assert target.sum() > 0

    def test_getitem_preserves_data_values(self) -> None:
        """Test __getitem__ preserves data values correctly."""
        size = 5
        array = create_empty_stage2_array(size)

        # Set specific values
        board_values = np.random.randint(
            0, 40, size=(9, 9), dtype=np.uint8
        )
        hand_values = np.random.randint(
            0, 19, size=(14,), dtype=np.uint8
        )
        legal_moves = np.random.randint(
            0, 2, size=(MOVE_LABELS_NUM,), dtype=np.uint8
        )

        array["boardIdPositions"][0] = board_values
        array["piecesInHand"][0] = hand_values
        array["legalMovesLabel"][0] = legal_moves

        datasource = MockDataSource(array)
        dataset = Stage2Dataset(datasource=datasource)

        features, target = dataset[0]
        board_tensor, hand_tensor = features

        # Check values are preserved
        assert torch.equal(
            board_tensor, torch.from_numpy(board_values)
        )
        assert torch.equal(
            hand_tensor, torch.from_numpy(hand_values)
        )
        assert torch.equal(
            target, torch.from_numpy(legal_moves).float()
        )

    def test_getitem_sparse_legal_moves(self) -> None:
        """Test dataset handles sparse legal moves (realistic scenario)."""
        size = 10
        array = create_empty_stage2_array(size)

        # Realistic: only 20-40 legal moves out of 2187
        for i in range(size):
            num_legal = np.random.randint(20, 40)
            legal_indices = np.random.choice(
                MOVE_LABELS_NUM, num_legal, replace=False
            )
            array["legalMovesLabel"][i, legal_indices] = 1

        datasource = MockDataSource(array)
        dataset = Stage2Dataset(datasource=datasource)

        for i in range(size):
            _, target = dataset[i]

            # Should have 20-40 legal moves
            num_legal = target.sum().item()
            assert 20 <= num_legal <= 40
