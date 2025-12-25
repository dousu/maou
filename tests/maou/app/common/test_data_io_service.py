"""Tests for data I/O service.

DEPRECATED: Tests use numpy-based I/O which has been removed.
TODO: Update tests to use DataFrame-based methods.
"""

import pytest


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
class TestDataIOService:
    """Test DataIOService functionality."""

    def test_deprecated(self) -> None:
        """Placeholder."""
        pass


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    def test_deprecated(self) -> None:
        """Placeholder."""
        pass
