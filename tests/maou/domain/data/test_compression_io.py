import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.data.array_io import (
    _convert_to_packed_format,
    load_preprocessing_array,
    save_preprocessing_array,
)
from maou.domain.data.schema import (
    convert_array_from_packed_format,
    create_empty_packed_preprocessing_array,
    create_empty_preprocessing_array,
)

logger: logging.Logger = logging.getLogger("TEST")


def zobrist_like_hash(value: bytes | str | int) -> np.uint64:
    if not isinstance(value, bytes):
        value = str(value).encode()
    h = hashlib.sha256(value).digest()
    return np.frombuffer(h[:8], dtype=np.uint64)[0]


class TestCompressionConversion:
    """Test conversion between standard and compressed formats."""

    def create_test_preprocessing_array(
        self, size: int = 3
    ) -> np.ndarray:
        array = create_empty_preprocessing_array(size)

        for i in range(size):
            array[i]["id"] = zobrist_like_hash(i)
            array[i]["moveLabel"] = np.bincount(
                [50 * i], minlength=MOVE_LABELS_NUM
            )
            array[i]["resultValue"] = 0.5
            array[i]["boardIdPositions"] = (
                np.arange(81, dtype=np.uint8).reshape(9, 9) + i
            )
            array[i]["piecesInHand"] = (
                np.arange(14, dtype=np.uint8) + i
            )

        return array

    def test_convert_to_packed_format(self) -> None:
        standard_array = self.create_test_preprocessing_array(2)

        packed_array = _convert_to_packed_format(standard_array)

        assert len(packed_array) == len(standard_array)
        assert (
            packed_array.dtype
            == create_empty_packed_preprocessing_array(1).dtype
        )

        assert np.array_equal(
            packed_array["id"], standard_array["id"]
        )
        assert np.array_equal(
            packed_array["moveLabel"],
            standard_array["moveLabel"],
        )
        assert np.array_equal(
            packed_array["resultValue"],
            standard_array["resultValue"],
        )

        assert "boardIdPositions" in packed_array.dtype.names
        assert "piecesInHand" in packed_array.dtype.names

    def test_convert_from_packed_format(self) -> None:
        standard_array = self.create_test_preprocessing_array(2)
        packed_array = _convert_to_packed_format(standard_array)

        reconstructed_array = convert_array_from_packed_format(
            packed_array
        )

        assert len(reconstructed_array) == len(standard_array)
        assert reconstructed_array.dtype == standard_array.dtype

        assert np.array_equal(
            reconstructed_array["boardIdPositions"],
            standard_array["boardIdPositions"],
        )
        assert np.array_equal(
            reconstructed_array["piecesInHand"],
            standard_array["piecesInHand"],
        )
        assert np.array_equal(
            reconstructed_array["moveLabel"],
            standard_array["moveLabel"],
        )
        assert np.array_equal(
            reconstructed_array["resultValue"],
            standard_array["resultValue"],
        )

    def test_roundtrip_conversion(self) -> None:
        original_array = self.create_test_preprocessing_array(5)

        packed = _convert_to_packed_format(original_array)
        reconstructed = convert_array_from_packed_format(packed)

        field_names = original_array.dtype.names
        if field_names is not None:
            for field in field_names:
                assert np.array_equal(
                    original_array[field], reconstructed[field]
                ), f"Field {field} differs"


class TestBitPackedFileSaveLoad:
    """Test saving and loading bit-packed files."""

    def create_test_data(self) -> np.ndarray:
        array = create_empty_preprocessing_array(10)

        for i in range(len(array)):
            array[i]["id"] = zobrist_like_hash(
                f"record_{i:03d}"
            )
            array[i]["moveLabel"] = np.bincount(
                [50 * i], minlength=MOVE_LABELS_NUM
            )
            array[i]["resultValue"] = float(i) / 10.0
            array[i]["boardIdPositions"] = (
                np.arange(81, dtype=np.uint8).reshape(9, 9) + i
            )
            array[i]["piecesInHand"] = (
                np.arange(14, dtype=np.uint8) + i
            )

        return array

    def test_save_load_bit_packed_raw(
        self, tmp_path: Path
    ) -> None:
        original_array = self.create_test_data()

        file_path = tmp_path / "test_data.packed"

        save_preprocessing_array(
            original_array, file_path, bit_pack=True
        )

        assert file_path.exists()

        loaded_array = load_preprocessing_array(
            file_path, bit_pack=True
        )

        assert loaded_array.dtype == original_array.dtype
        field_names = original_array.dtype.names
        if field_names is not None:
            for field in field_names:
                assert np.array_equal(
                    original_array[field], loaded_array[field]
                ), f"Field {field} differs"

    def test_load_without_unpacking(
        self, tmp_path: Path
    ) -> None:
        original_array = self.create_test_data()

        file_path = tmp_path / "test_data.packed"

        save_preprocessing_array(
            original_array, file_path, bit_pack=True
        )

        loaded_array = load_preprocessing_array(
            file_path,
            bit_pack=False,
        )

        assert (
            loaded_array.dtype
            == create_empty_packed_preprocessing_array(1).dtype
        )
        assert np.array_equal(
            loaded_array["id"], original_array["id"]
        )
        assert np.array_equal(
            loaded_array["moveLabel"],
            original_array["moveLabel"],
        )

    def test_save_with_invalid_mode(self) -> None:
        array = self.create_test_data()

        with pytest.raises(TypeError):
            save_preprocessing_array(  # type: ignore[arg-type]
                array, Path("dummy"), mode="invalid"
            )
