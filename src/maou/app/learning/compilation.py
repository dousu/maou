"""Utilities for compiling training modules with consistent options."""

from __future__ import annotations

import logging
from typing import Final, Tuple, Type, cast

import torch
from torch._dynamo.exc import BackendCompilerFailed, TorchRuntimeError

try:  # pragma: no cover - optional import for CUDA builds only
    from torch._inductor.exc import InductorError
except Exception:  # pragma: no cover - CPU builds omit torch._inductor
    InductorError = RuntimeError


logger = logging.getLogger(__name__)

_DYNAMIC_COMPILATION: Final[bool] = False
_FALLBACK_EXCEPTIONS: Final[Tuple[Type[Exception], ...]] = (
    BackendCompilerFailed,
    TorchRuntimeError,
    InductorError,
    RuntimeError,
)


def compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return a ``torch.compile`` wrapped module with static shapes."""

    try:
        compiled = torch.compile(module, dynamic=_DYNAMIC_COMPILATION)
    except _FALLBACK_EXCEPTIONS as error:
        logger.warning(
            "torch.compile failed with %s, falling back to eager execution", error
        )
        return module
    return cast(torch.nn.Module, compiled)
