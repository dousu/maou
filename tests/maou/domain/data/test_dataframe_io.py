"""Tests for DataFrame byte serialization．"""

import time
from datetime import date

import polars as pl
import pytest

from maou.domain.data.dataframe_io import (
    load_df_from_bytes,
    load_hcpe_df_from_bytes,
    load_preprocessing_df_from_bytes,
    save_df_to_bytes,
    save_hcpe_df_to_bytes,
    save_preprocessing_df_to_bytes,
)
from maou.domain.data.schema import (
    get_hcpe_polars_schema,
    get_preprocessing_polars_schema,
)


class TestHCPEDataFrameSerialization:
    """Test HCPE DataFrame serialization to bytes．"""

    def test_save_load_hcpe_df_roundtrip(self):
        """Test DataFrame → bytes → DataFrame preserves data．"""
        schema = get_hcpe_polars_schema()

        # Create test DataFrame
        data = {
            "hcp": [bytes([i % 256 for _ in range(32)]) for i in range(100)],
            "eval": [i - 50 for i in range(100)],
            "bestMove16": [i * 10 for i in range(100)],
            "gameResult": [i % 3 for i in range(100)],
            "id": [f"id_{i:08d}" for i in range(100)],
            "partitioningKey": [date(2025, 12, 25) for _ in range(100)],
            "ratings": [[1500, 1500] for _ in range(100)],
            "endgameStatus": ["Toryo" for _ in range(100)],
            "moves": [100 + i for i in range(100)],
        }

        df = pl.DataFrame(data, schema=schema)

        # Serialize to bytes
        bytes_data = save_hcpe_df_to_bytes(df)

        # Deserialize from bytes
        df_loaded = load_hcpe_df_from_bytes(bytes_data)

        # Verify data integrity
        assert len(df_loaded) == 100
        assert df_loaded.schema == schema
        assert df_loaded["id"][0] == "id_00000000"
        assert df_loaded["eval"][0] == -50
        assert df_loaded["bestMove16"][0] == 0
        assert df_loaded["hcp"][0] == data["hcp"][0]

    def test_compression_reduces_size(self):
        """Verify LZ4 compression reduces byte size．"""
        schema = get_hcpe_polars_schema()

        # Create repetitive data (compresses well)
        data = {
            "hcp": [b"X" * 32 for _ in range(1000)],
            "eval": [100] * 1000,
            "bestMove16": [1234] * 1000,
            "gameResult": [1] * 1000,
            "id": [f"id_{i:08d}" for i in range(1000)],
            "partitioningKey": [date(2025, 12, 25)] * 1000,
            "ratings": [[1500, 1500] for _ in range(1000)],
            "endgameStatus": ["Toryo"] * 1000,
            "moves": [120] * 1000,
        }

        df = pl.DataFrame(data, schema=schema)

        # Serialize to bytes
        bytes_data = save_hcpe_df_to_bytes(df)

        # Check size (should be significantly smaller than uncompressed)
        # Estimated uncompressed: ~60KB for 1000 records
        # With LZ4 compression: should be < 30KB
        assert len(bytes_data) < 30000, f"Compressed size: {len(bytes_data)} bytes"

    def test_serialization_performance(self):
        """Benchmark: <1ms for 10K rows．"""
        schema = get_hcpe_polars_schema()

        # Create 10K records
        data = {
            "hcp": [bytes([i % 256 for _ in range(32)]) for i in range(10000)],
            "eval": [i - 5000 for i in range(10000)],
            "bestMove16": [i % 10000 for i in range(10000)],
            "gameResult": [i % 3 for i in range(10000)],
            "id": [f"id_{i:08d}" for i in range(10000)],
            "partitioningKey": [date(2025, 12, 25) for _ in range(10000)],
            "ratings": [[1500 + i % 500, 1500 - i % 500] for i in range(10000)],
            "endgameStatus": ["Toryo" for _ in range(10000)],
            "moves": [100 + i % 100 for i in range(10000)],
        }

        df = pl.DataFrame(data, schema=schema)

        # Benchmark serialization
        start = time.perf_counter()
        bytes_data = save_hcpe_df_to_bytes(df)
        serialize_time = time.perf_counter() - start

        # Benchmark deserialization
        start = time.perf_counter()
        df_loaded = load_hcpe_df_from_bytes(bytes_data)
        deserialize_time = time.perf_counter() - start

        # Verify performance
        # Note: This may vary based on hardware, so we use a generous threshold
        assert serialize_time < 0.01, f"Serialization took {serialize_time:.4f}s"
        assert deserialize_time < 0.01, f"Deserialization took {deserialize_time:.4f}s"

        print(f"Serialization: {serialize_time:.4f}s, Deserialization: {deserialize_time:.4f}s")


class TestPreprocessingDataFrameSerialization:
    """Test preprocessing DataFrame serialization to bytes．"""

    def test_save_load_preprocessing_df_roundtrip(self):
        """Test DataFrame → bytes → DataFrame preserves data．"""
        schema = get_preprocessing_polars_schema()

        # Create test DataFrame
        from maou.app.pre_process.label import MOVE_LABELS_NUM
        import numpy as np

        data = {
            "id": list(range(100)),
            "boardIdPositions": [
                np.arange(81, dtype=np.uint8).reshape(9, 9).tolist() for _ in range(100)
            ],
            "piecesInHand": [np.arange(14, dtype=np.uint8).tolist() for _ in range(100)],
            "moveLabel": [
                np.random.rand(MOVE_LABELS_NUM).astype(np.float32).tolist()
                for _ in range(100)
            ],
            "resultValue": [float(i % 2) for i in range(100)],
        }

        df = pl.DataFrame(data, schema=schema)

        # Serialize to bytes
        bytes_data = save_preprocessing_df_to_bytes(df)

        # Deserialize from bytes
        df_loaded = load_preprocessing_df_from_bytes(bytes_data)

        # Verify data integrity
        assert len(df_loaded) == 100
        assert df_loaded.schema == schema
        assert df_loaded["id"][0] == 0
        assert df_loaded["resultValue"][0] == 0.0


class TestGenericDataFrameSerialization:
    """Test generic save_df_to_bytes and load_df_from_bytes functions．"""

    def test_save_load_with_array_type_hcpe(self):
        """Test generic functions with array_type='hcpe'．"""
        schema = get_hcpe_polars_schema()

        data = {
            "hcp": [b"A" * 32 for _ in range(10)],
            "eval": [0] * 10,
            "bestMove16": [0] * 10,
            "gameResult": [0] * 10,
            "id": [f"id_{i}" for i in range(10)],
            "partitioningKey": [date(2025, 12, 25) for _ in range(10)],
            "ratings": [[1500, 1500] for _ in range(10)],
            "endgameStatus": ["Toryo"] * 10,
            "moves": [100] * 10,
        }

        df = pl.DataFrame(data, schema=schema)

        # Use generic functions
        bytes_data = save_df_to_bytes(df, array_type="hcpe")
        df_loaded = load_df_from_bytes(bytes_data, array_type="hcpe")

        assert len(df_loaded) == 10

    def test_save_load_with_array_type_preprocessing(self):
        """Test generic functions with array_type='preprocessing'．"""
        schema = get_preprocessing_polars_schema()

        from maou.app.pre_process.label import MOVE_LABELS_NUM
        import numpy as np

        data = {
            "id": list(range(10)),
            "boardIdPositions": [
                np.arange(81, dtype=np.uint8).reshape(9, 9).tolist() for _ in range(10)
            ],
            "piecesInHand": [np.arange(14, dtype=np.uint8).tolist() for _ in range(10)],
            "moveLabel": [
                np.random.rand(MOVE_LABELS_NUM).astype(np.float32).tolist()
                for _ in range(10)
            ],
            "resultValue": [0.0] * 10,
        }

        df = pl.DataFrame(data, schema=schema)

        # Use generic functions
        bytes_data = save_df_to_bytes(df, array_type="preprocessing")
        df_loaded = load_df_from_bytes(bytes_data, array_type="preprocessing")

        assert len(df_loaded) == 10

    def test_invalid_array_type_raises_error(self):
        """Test that invalid array_type raises ValueError．"""
        schema = get_hcpe_polars_schema()

        data = {
            "hcp": [b"A" * 32],
            "eval": [0],
            "bestMove16": [0],
            "gameResult": [0],
            "id": ["id_0"],
            "partitioningKey": [date(2025, 12, 25)],
            "ratings": [[1500, 1500]],
            "endgameStatus": ["Toryo"],
            "moves": [100],
        }

        df = pl.DataFrame(data, schema=schema)

        with pytest.raises(ValueError, match="Unsupported array_type"):
            save_df_to_bytes(df, array_type="invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unsupported array_type"):
            load_df_from_bytes(b"dummy", array_type="invalid")  # type: ignore
