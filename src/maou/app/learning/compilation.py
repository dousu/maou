"""Utilities for compiling training modules with consistent options."""

from __future__ import annotations

from typing import Final, cast

import torch


_DYNAMIC_COMPILATION: Final[bool] = True


def compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return a ``torch.compile`` wrapped module with static shapes."""

    compiled = torch.compile(module, dynamic=_DYNAMIC_COMPILATION)
    return cast(torch.nn.Module, compiled)
