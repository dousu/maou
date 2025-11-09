import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.compression import (
    BOARD_ID_POSITIONS_SHAPE,
    LEGAL_MOVES_PACKED_SIZE,
    LEGAL_MOVES_SHAPE,
    PIECES_IN_HAND_SHAPE,
    CompressionError,
    get_compression_stats,
    pack_legal_moves_mask,
    pack_preprocessing_record,
    unpack_legal_moves_mask,
    unpack_preprocessing_fields,
)
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)


class TestLegalMovesPacking:
    """Test legal moves mask bit packing."""

    def test_pack_valid_legal_moves(self) -> None:
        legal_moves = np.random.choice(
            [0, 1], size=LEGAL_MOVES_SHAPE
        ).astype(np.uint8)

        packed = pack_legal_moves_mask(legal_moves)

        assert packed.shape == (LEGAL_MOVES_PACKED_SIZE,)
        assert packed.dtype == np.uint8

    def test_unpack_legal_moves_roundtrip(self) -> None:
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
        invalid_legal_moves = np.zeros(1000, dtype=np.uint8)

        with pytest.raises(
            CompressionError, match="Invalid legal moves shape"
        ):
            pack_legal_moves_mask(invalid_legal_moves)

    def test_pack_invalid_legal_moves_dtype(self) -> None:
        invalid_legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError, match="Invalid legal moves dtype"
        ):
            pack_legal_moves_mask(invalid_legal_moves)

    def test_pack_legal_moves_invalid_values(self) -> None:
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
        invalid_packed = np.zeros(100, dtype=np.uint8)

        with pytest.raises(
            CompressionError,
            match="Invalid packed legal moves shape",
        ):
            unpack_legal_moves_mask(invalid_packed)

    def test_unpack_invalid_packed_legal_moves_dtype(
        self,
    ) -> None:
        invalid_packed = np.zeros(
            LEGAL_MOVES_PACKED_SIZE, dtype=np.float32
        )

        with pytest.raises(
            CompressionError,
            match="Invalid packed legal moves dtype",
        ):
            unpack_legal_moves_mask(invalid_packed)


class TestPreprocessingRecordPacking:
    """Test preprocessing record field handling."""

    def test_pack_preprocessing_record(self) -> None:
        record_array = create_empty_preprocessing_array(1)
        record = record_array[0]

        record["boardIdPositions"] = np.arange(
            81, dtype=np.uint8
        ).reshape(BOARD_ID_POSITIONS_SHAPE)
        record["piecesInHand"] = np.arange(14, dtype=np.uint8)

        board_positions, pieces_in_hand = (
            pack_preprocessing_record(record)
        )

        assert board_positions.shape == BOARD_ID_POSITIONS_SHAPE
        assert pieces_in_hand.shape == PIECES_IN_HAND_SHAPE
        assert board_positions.dtype == np.uint8
        assert pieces_in_hand.dtype == np.uint8

    def test_unpack_preprocessing_fields_roundtrip(
        self,
    ) -> None:
        board_positions = np.arange(81, dtype=np.uint8).reshape(
            BOARD_ID_POSITIONS_SHAPE
        )
        pieces_in_hand = np.arange(14, dtype=np.uint8)

        unpacked_board, unpacked_hand = (
            unpack_preprocessing_fields(
                board_positions,
                pieces_in_hand,
            )
        )

        assert np.array_equal(board_positions, unpacked_board)
        assert np.array_equal(pieces_in_hand, unpacked_hand)

    def test_pack_preprocessing_record_invalid_shape(
        self,
    ) -> None:
        wrong_dtype = np.dtype(
            [
                ("boardIdPositions", np.uint8, (8, 9)),
                ("piecesInHand", np.uint8, (14,)),
            ]
        )
        wrong_record = np.zeros((), dtype=wrong_dtype)

        with pytest.raises(
            CompressionError,
            match="Invalid boardIdPositions shape",
        ):
            pack_preprocessing_record(wrong_record)  # type: ignore[arg-type]

    def test_unpack_preprocessing_fields_invalid_shape(
        self,
    ) -> None:
        board_positions = np.zeros((8, 9), dtype=np.uint8)
        pieces_in_hand = np.zeros(14, dtype=np.uint8)

        with pytest.raises(
            CompressionError,
            match="Invalid boardIdPositions shape",
        ):
            unpack_preprocessing_fields(
                board_positions, pieces_in_hand
            )


class TestCompressionStats:
    """Test compression statistics."""

    def test_get_compression_stats(self) -> None:
        stats = get_compression_stats()

        assert "boardIdPositions" in stats
        assert "piecesInHand" in stats
        assert "total" in stats

        board_stats = stats["boardIdPositions"]
        assert board_stats["original_size"] == 81
        assert board_stats["packed_size"] == 81
        assert board_stats["compression_ratio"] == 1.0

        hand_stats = stats["piecesInHand"]
        assert hand_stats["original_size"] == 14
        assert hand_stats["packed_size"] == 14
        assert hand_stats["compression_ratio"] == 1.0

        total_stats = stats["total"]
        assert total_stats["original_size"] == 95
        assert total_stats["packed_size"] == 95
        assert total_stats["compression_ratio"] == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_legal_moves(self) -> None:
        legal_moves = np.zeros(
            LEGAL_MOVES_SHAPE, dtype=np.uint8
        )

        packed = pack_legal_moves_mask(legal_moves)
        unpacked = unpack_legal_moves_mask(packed)

        assert np.array_equal(legal_moves, unpacked)

    def test_all_ones_legal_moves(self) -> None:
        legal_moves = np.ones(LEGAL_MOVES_SHAPE, dtype=np.uint8)

        packed = pack_legal_moves_mask(legal_moves)
        unpacked = unpack_legal_moves_mask(packed)

        assert np.array_equal(legal_moves, unpacked)
