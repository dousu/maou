"""Proxy module re-exporting domain data layer tests.

This allows running the suite with ``pytest tests/domain`` while keeping the
canonical test implementations in ``tests/maou/domain``.
"""

from ..maou.domain.data.test_compressed_schema import *  # noqa: F401,F403
from ..maou.domain.data.test_compression import *  # noqa: F401,F403
from ..maou.domain.data.test_compression_io import *  # noqa: F401,F403
from ..maou.domain.data.test_io import *  # noqa: F401,F403
from ..maou.domain.data.test_schema import *  # noqa: F401,F403

__all__ = [name for name in globals() if name.startswith("Test")]
