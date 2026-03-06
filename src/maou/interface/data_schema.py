"""Data schema interface for infrastructure layer.

This module provides schema and conversion functions
that can be used by infrastructure layer components while maintaining
Clean Architecture dependency rules.
"""

import logging
from typing import Literal

import numpy as np

from maou.app.common.data_schema_service import (
    DataSchemaService,
)
from maou.domain.data.schema import (
    convert_hcpe_df_to_numpy as convert_hcpe_df_to_numpy,  # noqa: F401
    convert_preprocessing_df_to_numpy as convert_preprocessing_df_to_numpy,  # noqa: F401
    convert_stage1_df_to_numpy as convert_stage1_df_to_numpy,  # noqa: F401
    convert_stage2_df_to_numpy as convert_stage2_df_to_numpy,  # noqa: F401
    get_hcpe_dtype as get_hcpe_dtype,  # noqa: F401
    get_hcpe_polars_schema as get_hcpe_polars_schema,  # noqa: F401
    get_preprocessing_dtype as get_preprocessing_dtype,  # noqa: F401
    get_preprocessing_polars_schema as get_preprocessing_polars_schema,  # noqa: F401
    get_stage1_dtype as get_stage1_dtype,  # noqa: F401
    get_stage2_dtype as get_stage2_dtype,  # noqa: F401
)
from maou.domain.move.label import (
    MOVE_LABELS_NUM as MOVE_LABELS_NUM,
)  # noqa: F401

logger: logging.Logger = logging.getLogger(__name__)


def numpy_dtype_to_bigquery_type(numpy_type: np.dtype) -> str:
    """Convert numpy dtype to BigQuery type."""
    return DataSchemaService.numpy_dtype_to_bigquery_type(
        numpy_type=numpy_type
    )


def get_dtype(
    array_type: Literal[
        "hcpe", "preprocessing", "stage1", "stage2"
    ],
    bit_pack: bool = False,
) -> np.dtype:
    return DataSchemaService.get_dtype(
        array_type=array_type,
        bit_pack=bit_pack,
    )
