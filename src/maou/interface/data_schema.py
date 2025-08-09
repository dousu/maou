import logging
from typing import Literal

import numpy as np

from maou.app.common.data_schema_service import (
    DataSchemaService,
)

logger: logging.Logger = logging.getLogger(__name__)


def numpy_dtype_to_bigquery_type(numpy_type: np.dtype) -> str:
    """Convert numpy dtype to BigQuery type."""
    return DataSchemaService.numpy_dtype_to_bigquery_type(
        numpy_type=numpy_type
    )


def get_dtype(
    array_type: Literal["hcpe", "preprocessing"],
    bit_pack: bool = False,
) -> np.dtype:
    return DataSchemaService.get_dtype(
        array_type=array_type, bit_pack=bit_pack
    )
