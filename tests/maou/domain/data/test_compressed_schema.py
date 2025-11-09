import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.compression import (
    BOARD_ID_POSITIONS_SHAPE,
    PIECES_IN_HAND_SHAPE,
)
from maou.domain.data.schema import (
    SchemaValidationError,
    create_empty_packed_preprocessing_array,
    get_packed_preprocessing_dtype,
    get_schema_info,
    validate_compressed_preprocessing_array,
)


class TestCompressedPreprocessingSchema:
    """Test compressed preprocessing schema definition."""

    def test_get_compressed_preprocessing_dtype(self) -> None:
        dtype = get_packed_preprocessing_dtype()

        assert dtype.names is not None

        expected_fields = {
            "id",
            "boardIdPositions",
            "piecesInHand",
            "moveLabel",
            "resultValue",
        }
        assert set(dtype.names) == expected_fields

        assert dtype["id"] == np.dtype("uint64")
        assert (
            dtype["boardIdPositions"].shape
            == BOARD_ID_POSITIONS_SHAPE
        )
        assert (
            dtype["piecesInHand"].shape == PIECES_IN_HAND_SHAPE
        )
        assert dtype["moveLabel"].shape == (MOVE_LABELS_NUM,)
        assert dtype["resultValue"] == np.dtype("float16")

    def test_create_empty_compressed_preprocessing_array(
        self,
    ) -> None:
        size = 10
        array = create_empty_packed_preprocessing_array(size)

        assert len(array) == size
        assert array.dtype == get_packed_preprocessing_dtype()
        assert array["boardIdPositions"].shape == (
            size,
            *BOARD_ID_POSITIONS_SHAPE,
        )
        assert array["piecesInHand"].shape == (
            size,
            *PIECES_IN_HAND_SHAPE,
        )


class TestCompressedPreprocessingValidation:
    """Test compressed preprocessing array validation."""

    def test_validate_valid_compressed_array(self) -> None:
        array = create_empty_packed_preprocessing_array(5)

        array["boardIdPositions"] = np.arange(
            5 * 81, dtype=np.uint8
        ).reshape(
            5,
            *BOARD_ID_POSITIONS_SHAPE,
        )
        array["piecesInHand"] = np.arange(
            5 * 7, dtype=np.uint8
        ).reshape(
            5,
            *PIECES_IN_HAND_SHAPE,
        )
        array["moveLabel"] = np.array(
            [
                np.bincount(arr, minlength=MOVE_LABELS_NUM)
                for arr in [
                    [0],
                    [100],
                    [500],
                    [1000],
                    [MOVE_LABELS_NUM - 1],
                ]
            ]
        )
        array["resultValue"] = [0.0, 0.25, 0.5, 0.75, 1.0]

        assert (
            validate_compressed_preprocessing_array(array)
            is True
        )

    def test_validate_invalid_dtype(self) -> None:
        wrong_array = np.zeros(
            5, dtype=[("id", "U128"), ("eval", "int16")]
        )

        with pytest.raises(
            SchemaValidationError, match="Invalid dtype"
        ):
            validate_compressed_preprocessing_array(wrong_array)

    def test_validate_non_array(self) -> None:
        with pytest.raises(
            SchemaValidationError,
            match="Expected numpy ndarray",
        ):
            validate_compressed_preprocessing_array(
                "not an array"  # type: ignore[arg-type]
            )

    def test_validate_move_label_out_of_range(self) -> None:
        array = create_empty_packed_preprocessing_array(1)
        array["moveLabel"][0][0] = 1.1

        with pytest.raises(
            SchemaValidationError,
            match="moveLabel values out of range",
        ):
            validate_compressed_preprocessing_array(array)

    def test_validate_result_value_out_of_range(self) -> None:
        array = create_empty_packed_preprocessing_array(2)
        array["resultValue"] = [-0.1, 1.1]

        with pytest.raises(
            SchemaValidationError,
            match="resultValue values out of range",
        ):
            validate_compressed_preprocessing_array(array)

    def test_validate_empty_array(self) -> None:
        array = create_empty_packed_preprocessing_array(0)

        assert (
            validate_compressed_preprocessing_array(array)
            is True
        )

    def test_validate_board_shape(self) -> None:
        array = create_empty_packed_preprocessing_array(1)
        expected_shape = (9, 9)
        actual_shape = array["boardIdPositions"].shape[1:]

        assert actual_shape == expected_shape


class TestSchemaInfo:
    """Test schema information helper."""

    def test_schema_info_contains_board_fields(self) -> None:
        info = get_schema_info()

        preprocessing = info["packed_preprocessing"]
        fields = preprocessing["fields"]

        assert "boardIdPositions" in fields
        assert "piecesInHand" in fields
        assert (
            "Board position identifiers"
            in fields["boardIdPositions"]
        )
        assert "Pieces in hand" in fields["piecesInHand"]
