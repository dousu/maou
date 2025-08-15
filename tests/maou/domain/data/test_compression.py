"""Tests for bit packing compression utilities."""

import numpy as np
import pytest

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.compression import (
    FEATURES_PACKED_SIZE,
    FEATURES_SHAPE,
    LEGAL_MOVES_PACKED_SIZE,
    LEGAL_MOVES_SHAPE,
    CompressionError,
    get_compression_stats,
    pack_features_array,
    pack_legal_moves_mask,
    pack_preprocessing_record,
    unpack_features_array,
    unpack_legal_moves_mask,
    unpack_preprocessing_fields,
)
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)


class TestFeaturesPacking:
    """Test features array bit packing."""

    def test_pack_valid_features(self) -> None:
        """Test packing valid features array."""
        # Create valid features array with only 0s and 1s
        features = np.random.choice(
            [0, 1], size=FEATURES_SHAPE
        ).astype(np.uint8)

        packed = pack_features_array(features)

        assert packed.shape == (FEATURES_PACKED_SIZE,)
        assert packed.dtype == np.uint8

    def test_unpack_features_roundtrip(self) -> None:
        """Test features pack/unpack roundtrip."""
        # Create deterministic test data
        features = np.zeros(FEATURES_SHAPE, dtype=np.uint8)
        features[0, 0, 0] = 1
        features[10, 4, 4] = 1
        features[-1, -1, -1] = 1

        packed = pack_features_array(features)
        unpacked = unpack_features_array(packed)

        assert np.array_equal(features, unpacked)
        assert unpacked.shape == FEATURES_SHAPE
        assert unpacked.dtype == np.uint8

    def test_pack_invalid_features_shape(self) -> None:
        """Test packing features with invalid shape."""
        invalid_features = np.zeros((10, 9, 9), dtype=np.uint8)

        with pytest.raises(
            CompressionError, match="Invalid features shape"
        ):
            pack_features_array(invalid_features)

    def test_pack_invalid_features_dtype(self) -> None:
        """Test packing features with invalid dtype."""
        invalid_features = np.zeros(
            FEATURES_SHAPE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError, match="Invalid features dtype"
        ):
            pack_features_array(invalid_features)

    def test_pack_features_invalid_values(self) -> None:
        """Test packing features with invalid values."""
        invalid_features = np.full(
            FEATURES_SHAPE, 2, dtype=np.uint8
        )

        with pytest.raises(
            CompressionError,
            match="contains values other than 0 or 1",
        ):
            pack_features_array(invalid_features)

    def test_unpack_invalid_packed_features_shape(self) -> None:
        """Test unpacking packed features with invalid shape."""
        invalid_packed = np.zeros(100, dtype=np.uint8)

        with pytest.raises(
            CompressionError,
            match="Invalid packed features shape",
        ):
            unpack_features_array(invalid_packed)

    def test_unpack_invalid_packed_features_dtype(self) -> None:
        """Test unpacking packed features with invalid dtype."""
        invalid_packed = np.zeros(
            FEATURES_PACKED_SIZE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError,
            match="Invalid packed features dtype",
        ):
            unpack_features_array(invalid_packed)


class TestLegalMovesPacking:
    """Test legal moves mask bit packing."""

    def test_pack_valid_legal_moves(self) -> None:
        """Test packing valid legal moves mask."""
        legal_moves = np.random.choice(
            [0, 1], size=LEGAL_MOVES_SHAPE
        ).astype(np.uint8)

        packed = pack_legal_moves_mask(legal_moves)

        assert packed.shape == (LEGAL_MOVES_PACKED_SIZE,)
        assert packed.dtype == np.uint8

    def test_unpack_legal_moves_roundtrip(self) -> None:
        """Test legal moves pack/unpack roundtrip."""
        # Create deterministic test data
        legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.uint8
        )
        legal_moves[0] = 1
        legal_moves[100] = 1
        legal_moves[-1] = 1

        packed = pack_legal_moves_mask(legal_moves)
        unpacked = unpack_legal_moves_mask(packed)

        assert np.array_equal(legal_moves, unpacked)
        assert unpacked.shape == LEGAL_MOVES_SHAPE
        assert unpacked.dtype == np.uint8

    def test_pack_invalid_legal_moves_shape(self) -> None:
        """Test packing legal moves with invalid shape."""
        invalid_legal_moves = np.zeros(1000, dtype=np.uint8)

        with pytest.raises(
            CompressionError, match="Invalid legal moves shape"
        ):
            pack_legal_moves_mask(invalid_legal_moves)

    def test_pack_invalid_legal_moves_dtype(self) -> None:
        """Test packing legal moves with invalid dtype."""
        invalid_legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError, match="Invalid legal moves dtype"
        ):
            pack_legal_moves_mask(invalid_legal_moves)

    def test_pack_legal_moves_invalid_values(self) -> None:
        """Test packing legal moves with invalid values."""
        invalid_legal_moves = np.full(
            LEGAL_MOVES_SHAPE, 2, dtype=np.uint8
        )

        with pytest.raises(
            CompressionError,
            match="contains values other than 0 or 1",
        ):
            pack_legal_moves_mask(invalid_legal_moves)

    def test_unpack_invalid_packed_legal_moves_shape(
        self,
    ) -> None:
        """Test unpacking packed legal moves with invalid shape."""
        invalid_packed = np.zeros(100, dtype=np.uint8)

        with pytest.raises(
            CompressionError,
            match="Invalid packed legal moves shape",
        ):
            unpack_legal_moves_mask(invalid_packed)

    def test_unpack_invalid_packed_legal_moves_dtype(
        self,
    ) -> None:
        """Test unpacking packed legal moves with invalid dtype."""
        invalid_packed = np.zeros(
            LEGAL_MOVES_PACKED_SIZE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError,
            match="Invalid packed legal moves dtype",
        ):
            unpack_legal_moves_mask(invalid_packed)


class TestPreprocessingRecordPacking:
    """Test preprocessing record packing."""

    def test_pack_preprocessing_record(self) -> None:
        """Test packing a single preprocessing record."""
        # Create a valid preprocessing record
        record_array = create_empty_preprocessing_array(1)
        record = record_array[0]

        # Set some test data
        record["features"][:10, :5, :5] = 1
        record["legalMoveMask"][:100] = 1

        packed_features, packed_legal_moves = (
            pack_preprocessing_record(record)
        )

        assert packed_features.shape == (FEATURES_PACKED_SIZE,)
        assert packed_legal_moves.shape == (
            LEGAL_MOVES_PACKED_SIZE,
        )
        assert packed_features.dtype == np.uint8
        assert packed_legal_moves.dtype == np.uint8

    def test_unpack_preprocessing_fields_roundtrip(
        self,
    ) -> None:
        """Test preprocessing fields pack/unpack roundtrip."""
        # Create a valid preprocessing record
        record_array = create_empty_preprocessing_array(1)
        record = record_array[0]

        # Set deterministic test data
        record["features"][0, 0, 0] = 1
        record["features"][50, 4, 4] = 1
        record["legalMoveMask"][0] = 1
        record["legalMoveMask"][500] = 1

        packed_features, packed_legal_moves = (
            pack_preprocessing_record(record)
        )
        unpacked_features, unpacked_legal_moves = (
            unpack_preprocessing_fields(
                packed_features, packed_legal_moves
            )
        )

        assert np.array_equal(
            record["features"], unpacked_features
        )
        assert np.array_equal(
            record["legalMoveMask"], unpacked_legal_moves
        )


class TestCompressionStats:
    """Test compression statistics."""

    def test_get_compression_stats(self) -> None:
        """Test compression statistics calculation."""
        stats = get_compression_stats()

        # Check structure
        assert "features" in stats
        assert "legal_moves" in stats
        assert "total" in stats

        # Check features stats
        features_stats = stats["features"]
        assert (
            features_stats["original_size"]
            == FEATURES_NUM * 9 * 9
        )
        assert (
            features_stats["packed_size"]
            == FEATURES_PACKED_SIZE
        )
        assert (
            features_stats["compression_ratio"]
            == (FEATURES_NUM * 9 * 9) / FEATURES_PACKED_SIZE
        )

        # Check legal moves stats
        legal_moves_stats = stats["legal_moves"]
        assert (
            legal_moves_stats["original_size"]
            == MOVE_LABELS_NUM
        )
        assert (
            legal_moves_stats["packed_size"]
            == LEGAL_MOVES_PACKED_SIZE
        )
        assert (
            legal_moves_stats["compression_ratio"]
            == MOVE_LABELS_NUM / LEGAL_MOVES_PACKED_SIZE
        )

        # Check total stats
        total_stats = stats["total"]
        expected_original = (
            FEATURES_NUM * 9 * 9 + MOVE_LABELS_NUM
        )
        expected_packed = (
            FEATURES_PACKED_SIZE + LEGAL_MOVES_PACKED_SIZE
        )
        assert total_stats["original_size"] == expected_original
        assert total_stats["packed_size"] == expected_packed
        assert (
            total_stats["compression_ratio"]
            == expected_original / expected_packed
        )

    def test_compression_ratio_is_approximately_8x(
        self,
    ) -> None:
        """Test that compression ratio is approximately 8x."""
        stats = get_compression_stats()

        # Check that we're getting close to 8x compression
        assert 7.8 <= stats["total"]["compression_ratio"] <= 8.2


class TestPerformance:
    """Test compression performance."""

    def test_features_compression_performance(self) -> None:
        """Test features compression performance."""
        # Create random binary features
        features = np.random.choice(
            [0, 1], size=FEATURES_SHAPE
        ).astype(np.uint8)

        # Time multiple pack/unpack cycles
        import time

        start_time = time.perf_counter()
        for _ in range(100):
            packed = pack_features_array(features)
            unpacked = unpack_features_array(packed)
        end_time = time.perf_counter()

        # Should complete 100 cycles quickly (less than 1 second)
        assert end_time - start_time < 1.0

        # Verify correctness
        assert np.array_equal(features, unpacked)

    def test_legal_moves_compression_performance(self) -> None:
        """Test legal moves compression performance."""
        # Create random binary legal moves
        legal_moves = np.random.choice(
            [0, 1], size=LEGAL_MOVES_SHAPE
        ).astype(np.uint8)

        # Time multiple pack/unpack cycles
        import time

        start_time = time.perf_counter()
        for _ in range(100):
            packed = pack_legal_moves_mask(legal_moves)
            unpacked = unpack_legal_moves_mask(packed)
        end_time = time.perf_counter()

        # Should complete 100 cycles quickly (less than 1 second)
        assert end_time - start_time < 1.0

        # Verify correctness
        assert np.array_equal(legal_moves, unpacked)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_features(self) -> None:
        """Test compression of all-zeros features."""
        features = np.zeros(FEATURES_SHAPE, dtype=np.uint8)

        packed = pack_features_array(features)
        unpacked = unpack_features_array(packed)

        assert np.array_equal(features, unpacked)

    def test_all_ones_features(self) -> None:
        """Test compression of all-ones features."""
        features = np.ones(FEATURES_SHAPE, dtype=np.uint8)

        packed = pack_features_array(features)
        unpacked = unpack_features_array(packed)

        assert np.array_equal(features, unpacked)

    def test_all_zeros_legal_moves(self) -> None:
        """Test compression of all-zeros legal moves."""
        legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.uint8
        )

        packed = pack_legal_moves_mask(legal_moves)
        unpacked = unpack_legal_moves_mask(packed)

        assert np.array_equal(legal_moves, unpacked)

    def test_all_ones_legal_moves(self) -> None:
        """Test compression of all-ones legal moves."""
        legal_moves = np.ones(LEGAL_MOVES_SHAPE, dtype=np.uint8)

        packed = pack_legal_moves_mask(legal_moves)
        unpacked = unpack_legal_moves_mask(packed)

        assert np.array_equal(legal_moves, unpacked)

    def test_sparse_data(self) -> None:
        """Test compression of sparse data (mostly zeros)."""
        features = np.zeros(FEATURES_SHAPE, dtype=np.uint8)
        # Set only a few positions to 1
        features[0, 0, 0] = 1
        features[FEATURES_NUM // 2, 4, 4] = 1
        features[-1, -1, -1] = 1

        legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.uint8
        )
        legal_moves[0] = 1
        legal_moves[MOVE_LABELS_NUM // 2] = 1
        legal_moves[-1] = 1

        # Test features
        packed_features = pack_features_array(features)
        unpacked_features = unpack_features_array(
            packed_features
        )
        assert np.array_equal(features, unpacked_features)

        # Test legal moves
        packed_legal_moves = pack_legal_moves_mask(legal_moves)
        unpacked_legal_moves = unpack_legal_moves_mask(
            packed_legal_moves
        )
        assert np.array_equal(legal_moves, unpacked_legal_moves)
