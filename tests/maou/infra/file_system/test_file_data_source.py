"""Tests for the file-system based data source.

DEPRECATED: Tests use numpy array-based I/O which has been removed.
TODO: Update tests to use DataFrame-based methods with .feather files.
"""

import pytest


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
def test_deprecated() -> None:
    """Placeholder for deprecated tests."""
    pass
