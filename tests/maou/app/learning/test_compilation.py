"""Tests for the torch.compile helper utilities."""

from __future__ import annotations

from typing import Any

import torch

from maou.app.learning import compilation
from maou.app.learning.compilation import compile_module


class _DummyModule(torch.nn.Module):
    def forward(
        self, *inputs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, ...]:
        return tuple(inputs)


def test_compile_module_uses_static_shapes(monkeypatch) -> None:
    """``compile_module`` should disable dynamic shape support."""

    module = _DummyModule()
    compiled_module = torch.nn.Sequential(torch.nn.Identity())

    def _fake_compile(
        target: torch.nn.Module, *, dynamic: bool
    ) -> torch.nn.Module:
        assert target is module
        assert dynamic is False
        return compiled_module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    result = compile_module(module)

    assert result is compiled_module


def test_compile_module_allows_dynamic_override(
    monkeypatch,
) -> None:
    """Callers can opt into dynamic shape compilation when required."""

    module = _DummyModule()
    compiled_module = torch.nn.Sequential(torch.nn.Identity())

    def _fake_compile(
        target: torch.nn.Module, *, dynamic: bool
    ) -> torch.nn.Module:
        assert target is module
        assert dynamic is True
        return compiled_module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    result = compile_module(module, dynamic=True)

    assert result is compiled_module


def test_compile_module_falls_back_on_failure(
    monkeypatch,
) -> None:
    """Errors from ``torch.compile`` should trigger a safe fallback."""

    module = _DummyModule()
    warnings: list[str] = []

    def _raise_compile(
        *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        raise RuntimeError("compilation failure")

    def _collect_warning(
        message: str, *args: Any, **kwargs: Any
    ) -> None:
        formatted = message % args if args else message
        warnings.append(formatted)

    monkeypatch.setattr(torch, "compile", _raise_compile)

    monkeypatch.setattr(
        compilation.logger, "warning", _collect_warning
    )

    result = compile_module(module)

    assert result is module
    assert warnings
    assert "falling back to eager execution" in warnings[0]
