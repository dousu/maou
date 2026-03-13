"""Tests for value target mode utilities."""

from __future__ import annotations

import pytest
import torch

from maou.app.learning.value_targets import (
    ValueTargetMode,
    resolve_value_targets,
)


def test_result_value_mode_returns_labels_unchanged() -> None:
    """RESULT_VALUEモードではlabels_valueがそのまま返される．"""
    labels_value = torch.tensor(
        [[0.6], [0.3], [0.8]], dtype=torch.float32
    )
    result = resolve_value_targets(
        labels_value,
        mode=ValueTargetMode.RESULT_VALUE,
    )
    assert torch.equal(result, labels_value)


def test_result_value_mode_ignores_move_win_rate() -> None:
    """RESULT_VALUEモードではmove_win_rateが渡されても無視される．"""
    labels_value = torch.tensor(
        [[0.5], [0.7]], dtype=torch.float32
    )
    move_win_rate = torch.tensor(
        [[0.1, 0.9, 0.3], [0.4, 0.6, 0.8]],
        dtype=torch.float32,
    )
    result = resolve_value_targets(
        labels_value,
        mode=ValueTargetMode.RESULT_VALUE,
        move_win_rate=move_win_rate,
    )
    assert torch.equal(result, labels_value)


def test_best_move_win_rate_mode_returns_max() -> None:
    """BEST_MOVE_WIN_RATEモードではmoveWinRateの最大値が返される．"""
    labels_value = torch.tensor(
        [[0.5], [0.3]], dtype=torch.float32
    )
    move_win_rate = torch.tensor(
        [[0.1, 0.9, 0.3], [0.4, 0.6, 0.2]],
        dtype=torch.float32,
    )
    result = resolve_value_targets(
        labels_value,
        mode=ValueTargetMode.BEST_MOVE_WIN_RATE,
        move_win_rate=move_win_rate,
    )
    expected = torch.tensor([[0.9], [0.6]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_best_move_win_rate_mode_raises_without_move_win_rate() -> (
    None
):
    """BEST_MOVE_WIN_RATEモードでmove_win_rateがNoneの場合はValueError．"""
    labels_value = torch.tensor([[0.5]], dtype=torch.float32)
    with pytest.raises(ValueError, match="move_win_rate"):
        resolve_value_targets(
            labels_value,
            mode=ValueTargetMode.BEST_MOVE_WIN_RATE,
            move_win_rate=None,
        )


def test_best_move_win_rate_mode_preserves_dtype() -> None:
    """出力のdtypeがlabels_valueのdtypeと一致する．"""
    labels_value = torch.tensor(
        [[0.5], [0.3]], dtype=torch.float16
    )
    move_win_rate = torch.tensor(
        [[0.1, 0.9], [0.4, 0.6]], dtype=torch.float32
    )
    result = resolve_value_targets(
        labels_value,
        mode=ValueTargetMode.BEST_MOVE_WIN_RATE,
        move_win_rate=move_win_rate,
    )
    assert result.dtype == torch.float16


def test_best_move_win_rate_mode_empty_move_win_rate() -> None:
    """move_win_rateが空テンソル(N, 0)の場合はRuntimeError．"""
    labels_value = torch.tensor(
        [[0.5], [0.3]], dtype=torch.float32
    )
    move_win_rate = torch.empty((2, 0), dtype=torch.float32)
    with pytest.raises((RuntimeError, IndexError)):
        resolve_value_targets(
            labels_value,
            mode=ValueTargetMode.BEST_MOVE_WIN_RATE,
            move_win_rate=move_win_rate,
        )


def test_value_target_mode_enum_values() -> None:
    """Enumの値がCLIオプションと一致する．"""
    assert ValueTargetMode.RESULT_VALUE.value == "result-value"
    assert (
        ValueTargetMode.BEST_MOVE_WIN_RATE.value
        == "best-move-win-rate"
    )
    assert len(ValueTargetMode) == 2
