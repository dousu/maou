"""Centralized data I/O operations for Maou project.

This module provides standardized save and load functions for Polars DataFrames
using Arrow IPC format (.feather files) with proper error handling and compression.
"""

import logging


class DataIOError(Exception):
    """Raised when data I/O operations fail."""

    pass


class DataIOArrayTypeError(Exception):
    """Raised when data I/O operations fail."""

    pass


logger: logging.Logger = logging.getLogger(__name__)


# ============================================================================
# Polars DataFrame I/O Functions (Rust Backend)
# ============================================================================

# Re-export Polars I/O functions from rust_io module for convenience
# This allows users to import from array_io directly:
#   from maou.domain.data.array_io import save_hcpe_df, load_hcpe_df

try:
    from maou.domain.data.rust_io import (
        RUST_BACKEND_AVAILABLE,
        load_hcpe_df,
        load_preprocessing_df,
        load_stage1_df,
        load_stage2_df,
        save_hcpe_df,
        save_preprocessing_df,
        save_stage1_df,
        save_stage2_df,
    )

    __all__ = [
        # Polars DataFrame I/O functions
        "save_hcpe_df",
        "load_hcpe_df",
        "save_preprocessing_df",
        "load_preprocessing_df",
        "save_stage1_df",
        "load_stage1_df",
        "save_stage2_df",
        "load_stage2_df",
        "RUST_BACKEND_AVAILABLE",
    ]
except ImportError as e:
    logger.warning(
        f"Rust backend not available, Polars I/O functions disabled: {e}"
    )
    RUST_BACKEND_AVAILABLE = False

    __all__ = [
        "RUST_BACKEND_AVAILABLE",
    ]
