from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from maou.app.learning.compilation import compile_module


def test_compile_module_enables_dynamic_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_kwargs: Dict[str, Any] = {}

    def _fake_compile(
        module: torch.nn.Module, **kwargs: Any
    ) -> torch.nn.Module:
        recorded_kwargs.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    module = torch.nn.Linear(4, 4)
    compiled = compile_module(module)

    assert compiled is module
    assert recorded_kwargs == {"dynamic": True}
