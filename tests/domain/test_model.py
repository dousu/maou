"""Proxy module re-exporting domain model layer tests."""

from ..maou.domain.model.test_mlp_mixer import *  # noqa: F401,F403
from ..maou.domain.model.test_resnet import *  # noqa: F401,F403
from ..maou.domain.model.test_vision_transformer import *  # noqa: F401,F403

__all__ = [name for name in globals() if name.startswith("Test")]
