"""Utility functions for working with policy target distributions."""

from __future__ import annotations

from typing import Optional

import torch


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
    targets = labels_policy.to(device=target_device, dtype=target_dtype)

    if legal_move_mask is not None:
        targets = targets * legal_move_mask.to(
            device=target_device, dtype=target_dtype
        )

    target_sum = targets.sum(dim=1, keepdim=True)
    safe_sum = torch.where(
        target_sum > 0,
        target_sum,
        torch.ones_like(target_sum),
    )
    return targets / safe_sum

