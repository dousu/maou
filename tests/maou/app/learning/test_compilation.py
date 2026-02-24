"""Tests for the torch.compile helper utilities."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from maou.app.learning import compilation
from maou.app.learning.compilation import compile_module


class _DummyModule(torch.nn.Module):
    def forward(
        self, *inputs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, ...]:
        return tuple(inputs)


def test_compile_module_uses_static_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
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


class _SimpleLinear(torch.nn.Module):
    """ウォームアップテスト用のシンプルなモジュール．"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_warmup_compiled_model_triggers_forward() -> None:
    """warmup がフォワードパスを実行し学習モードを復元すること．"""
    from maou.app.learning.compilation import (
        warmup_compiled_model,
    )

    model = _SimpleLinear()
    model.train()

    dummy = torch.randn(2, 4)
    warmup_compiled_model(model, dummy)

    # 学習モードが復元されていること
    assert model.training


def test_warmup_compiled_model_restores_eval_mode() -> None:
    """warmup 前に eval だったモデルは eval のまま復元されること．"""
    from maou.app.learning.compilation import (
        warmup_compiled_model,
    )

    model = _SimpleLinear()
    model.eval()

    dummy = torch.randn(2, 4)
    warmup_compiled_model(model, dummy)

    assert not model.training


def test_warmup_compiled_model_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warmup のフォワードパスが失敗してもモードが復元されること．"""
    from maou.app.learning.compilation import (
        warmup_compiled_model,
    )

    model = _SimpleLinear()
    model.train()

    def _raise_forward(
        self: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        raise RuntimeError("forward failed")

    monkeypatch.setattr(
        _SimpleLinear, "forward", _raise_forward
    )

    dummy = torch.randn(2, 4)
    # RuntimeError は _FALLBACK_EXCEPTIONS に含まれるため，
    # 例外は捕捉されて警告ログが出力される
    warmup_compiled_model(model, dummy)

    assert model.training
