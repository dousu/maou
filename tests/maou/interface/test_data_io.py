"""Tests for interface data I/O module.

DEPRECATED: These tests are for numpy-based I/O which has been removed.
The project now uses Polars DataFrames exclusively with .feather format.
"""

import pytest


@pytest.mark.skip(
    reason="Numpy-based I/O has been removed. Use DataFrame-based methods instead."
)
class TestInterfaceDataIO:
    """Deprecated numpy-based tests."""

    def test_deprecated(self) -> None:
        """Placeholder for removed numpy tests."""
        pass
