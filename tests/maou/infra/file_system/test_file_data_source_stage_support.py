"""Tests for FileDataSource stage1 and stage2 array type support.

DEPRECATED: Tests use numpy array-based I/O with .npy files.
TODO: Update tests to use DataFrame-based methods with .feather files.
"""

import pytest


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
class TestFileDataSourceStage1Support:
    """Test FileDataSource support for stage1 array type."""

    def test_deprecated(self) -> None:
        """Placeholder."""
        pass


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
class TestFileDataSourceStage2Support:
    """Test FileDataSource support for stage2 array type."""

    def test_deprecated(self) -> None:
        """Placeholder."""
        pass


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
class TestFileDataSourceSpliterStageSupport:
    """Test FileDataSourceSpliter stage support."""

    def test_deprecated(self) -> None:
        """Placeholder."""
        pass
