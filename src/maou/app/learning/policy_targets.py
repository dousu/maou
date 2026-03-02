"""Utility functions for working with policy target distributions."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch


class PolicyTargetMode(str, Enum):
    """Policy教師信号のモード．

    Attributes:
        MOVE_LABEL: 棋譜中の選択頻度(moveLabel)をそのまま使用する．
        WIN_RATE: 指し手別勝率(moveWinRate)を正規化して使用する．
        WEIGHTED: moveLabel × moveWinRate を正規化して使用する．
    """

    MOVE_LABEL = "move-label"
    WIN_RATE = "win-rate"
    WEIGHTED = "weighted"


def build_policy_targets(
    labels_policy: torch.Tensor,
    legal_move_mask: Optional[torch.Tensor],
    *,
    mode: PolicyTargetMode,
    move_win_rate: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Policy教師信号のモードに応じた分布を構築する．

    Args:
        labels_policy: 棋譜由来のpolicyラベル(moveLabel).
        legal_move_mask: 合法手マスク．
        mode: 教師信号モード．
        move_win_rate: 指し手別勝率．WIN_RATE/WEIGHTEDモードで必須．
        dtype: 出力テンソルのdtype．
        device: 出力テンソルのdevice．

    Returns:
        正規化済みpolicy分布テンソル．

    Raises:
        ValueError: WIN_RATE/WEIGHTEDモードで``move_win_rate``がNoneの場合．
    """
    if mode == PolicyTargetMode.MOVE_LABEL:
        return normalize_policy_targets(
            labels_policy,
            legal_move_mask,
            dtype=dtype,
            device=device,
        )

    if move_win_rate is None:
        msg = (
            f"move_win_rate is required for mode={mode.value!r}, "
            "but got None"
        )
        raise ValueError(msg)

    target_dtype = dtype or labels_policy.dtype
    target_device = device or labels_policy.device

    if mode == PolicyTargetMode.WIN_RATE:
        targets = move_win_rate.to(
            device=target_device, dtype=target_dtype
        )
    else:
        # WEIGHTED: labels_policy * move_win_rate
        targets = labels_policy.to(
            device=target_device, dtype=target_dtype
        ) * move_win_rate.to(
            device=target_device, dtype=target_dtype
        )

    if legal_move_mask is not None:
        targets = targets * legal_move_mask.to(
            device=target_device, dtype=target_dtype
        )

    return _safe_normalize(targets)


def normalize_policy_targets(
    labels_policy: torch.Tensor,
    legal_move_mask: Optional[torch.Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a normalized policy target distribution.

    Args:
        labels_policy: Raw policy supervision targets.
        legal_move_mask: Optional mask for filtering illegal moves.
        dtype: Desired dtype for the normalized tensor.
        device: Desired device for the normalized tensor.

    Returns:
        Normalized policy distribution with zero mass assigned to illegal moves.
    """

    target_dtype = dtype or labels_policy.dtype
    target_device = device or labels_policy.device
    targets = labels_policy.to(
        device=target_device, dtype=target_dtype
    )

    if legal_move_mask is not None:
        targets = targets * legal_move_mask.to(
            device=target_device, dtype=target_dtype
        )

    return _safe_normalize(targets)


def _safe_normalize(targets: torch.Tensor) -> torch.Tensor:
    """行方向に安全な正規化を行う．合計が0の行はそのまま返す．"""
    target_sum = targets.sum(dim=1, keepdim=True)
    safe_sum = torch.where(
        target_sum > 0,
        target_sum,
        torch.ones_like(target_sum),
    )
    return targets / safe_sum
