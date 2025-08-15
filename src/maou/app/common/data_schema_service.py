import logging
from typing import Literal

import numpy as np

from maou.domain.data.schema import (
    convert_array_from_packed_format,
    convert_record_from_packed_format,
    get_hcpe_dtype,
    get_packed_preprocessing_dtype,
    get_preprocessing_dtype,
    numpy_dtype_to_bigquery_type,
)

logger: logging.Logger = logging.getLogger(__name__)


class DataSchemaService:
    """Service class for data schema operations.

    Provides a unified interface for converting numpy dtypes to BigQuery types.
    """

    @staticmethod
    def numpy_dtype_to_bigquery_type(
        *,
        numpy_type: np.dtype,
    ) -> str:
        """Convert numpy dtype to BigQuery type.

        Args:
            numpy_type: Numpy dtype to convert

        Returns:
            str: Corresponding BigQuery type
        """
        return numpy_dtype_to_bigquery_type(numpy_type)

    @staticmethod
    def get_dtype(
        *,
        array_type: Literal["hcpe", "preprocessing"],
        bit_pack: bool,
    ) -> np.dtype:
        if array_type == "hcpe":
            return get_hcpe_dtype()
        elif array_type == "preprocessing":
            if bit_pack:
                return get_packed_preprocessing_dtype()
            return get_preprocessing_dtype()
        else:
            logger.error(f"Unknown array type '{array_type}'")
            raise ValueError(
                f"Unknown array type '{array_type}'"
            )

    @staticmethod
    def convert_array_from_packed_format(
        *,
        compressed_array: np.ndarray,
        array_type: Literal["preprocessing"],
    ) -> np.ndarray:
        if array_type == "preprocessing":
            return convert_array_from_packed_format(
                compressed_array=compressed_array
            )
        else:
            logger.error(f"Unknown array type '{array_type}'")
            raise ValueError(
                f"Unknown array type '{array_type}'"
            )

    @staticmethod
    def convert_record_from_packed_format(
        *,
        compressed_record: np.ndarray,
        array_type: Literal["preprocessing"],
    ) -> np.ndarray:
        if array_type == "preprocessing":
            return convert_record_from_packed_format(
                compressed_record=compressed_record
            )
        else:
            logger.error(f"Unknown array type '{array_type}'")
            raise ValueError(
                f"Unknown array type '{array_type}'"
            )
