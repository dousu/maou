"""Utilities for compiling training modules with consistent options."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Final,
    Optional,
    Tuple,
    Type,
    cast,
)

import torch
from torch._dynamo.exc import (
    BackendCompilerFailed,
    TorchRuntimeError,
)

if TYPE_CHECKING:
    from torch._inductor.exc import (
        InductorError as InductorErrorType,
    )
else:  # pragma: no cover - optional import for CUDA builds only
    try:
        from torch._inductor.exc import (
            InductorError as InductorErrorType,
        )
    except (
        Exception
    ):  # pragma: no cover - CPU builds omit torch._inductor

        class InductorErrorType(RuntimeError):
            """Fallback when torch._inductor is unavailable."""

            pass


InductorError = InductorErrorType


logger = logging.getLogger(__name__)

_DYNAMIC_COMPILATION: Final[bool] = False
_FALLBACK_EXCEPTIONS: Final[Tuple[Type[Exception], ...]] = (
    BackendCompilerFailed,
    TorchRuntimeError,
    InductorError,
    RuntimeError,
)


def compile_module(
    module: torch.nn.Module, *, dynamic: Optional[bool] = None
) -> torch.nn.Module:
    """Return a ``torch.compile`` wrapped module.

    Args:
        module: Module to compile with TorchDynamo.
        dynamic: Optional override for the dynamic shape flag. When ``None`` the
            module falls back to the repository-wide default.
    """

    dynamic_flag = (
        _DYNAMIC_COMPILATION if dynamic is None else dynamic
    )

    try:
        compiled = torch.compile(module, dynamic=dynamic_flag)
    except _FALLBACK_EXCEPTIONS as error:
        logger.warning(
            "torch.compile failed with %s, falling back to eager execution",
            error,
        )
        return module
    return cast(torch.nn.Module, compiled)
