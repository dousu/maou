"""Tests for policy target normalization utilities."""

from __future__ import annotations

import pytest
import torch

from maou.app.learning.policy_targets import (
    PolicyTargetMode,
    build_policy_targets,
    normalize_policy_targets,
)


def test_normalize_policy_targets_without_mask() -> None:
    labels = torch.tensor(
        [[0.25, 0.75, 1.0], [2.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    normalized = normalize_policy_targets(
        labels, legal_move_mask=None
    )

    expected = torch.tensor(
        [[0.125, 0.375, 0.5], [0.5, 0.25, 0.25]],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)


def test_normalize_policy_targets_with_mask_and_dtype() -> None:
    labels = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    normalized = normalize_policy_targets(
        labels,
        mask,
        dtype=torch.float64,
    )

    expected = torch.tensor(
        [[0.25, 0.0, 0.75], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    assert normalized.dtype == torch.float64
    assert torch.allclose(normalized, expected)
    assert torch.count_nonzero(normalized[1]).item() == 0


class TestBuildPolicyTargets:
    """Tests for build_policy_targets()."""

    def test_move_label_mode_delegates_to_normalize(
        self,
    ) -> None:
        """MOVE_LABEL mode should produce same result as normalize_policy_targets."""
        labels = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float32
        )
        mask = torch.tensor(
            [[1.0, 1.0, 1.0]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            mask,
            mode=PolicyTargetMode.MOVE_LABEL,
        )
        expected = normalize_policy_targets(labels, mask)
        assert torch.allclose(result, expected)

    def test_win_rate_mode_normalizes_win_rate(self) -> None:
        """WIN_RATE mode should normalize move_win_rate."""
        labels = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32
        )
        win_rate = torch.tensor(
            [[0.6, 0.3, 0.1]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            legal_move_mask=None,
            mode=PolicyTargetMode.WIN_RATE,
            move_win_rate=win_rate,
        )
        expected = torch.tensor(
            [[0.6, 0.3, 0.1]], dtype=torch.float32
        )
        assert torch.allclose(result, expected)

    def test_weighted_mode_multiplies_and_normalizes(
        self,
    ) -> None:
        """WEIGHTED mode should multiply labels * win_rate and normalize."""
        labels = torch.tensor(
            [[2.0, 0.0, 1.0]], dtype=torch.float32
        )
        win_rate = torch.tensor(
            [[0.5, 0.3, 0.5]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            legal_move_mask=None,
            mode=PolicyTargetMode.WEIGHTED,
            move_win_rate=win_rate,
        )
        # labels * win_rate = [1.0, 0.0, 0.5], sum = 1.5
        # normalized = [1.0/1.5, 0.0, 0.5/1.5] ≈ [0.6667, 0.0, 0.3333]
        expected = torch.tensor(
            [[1.0 / 1.5, 0.0, 0.5 / 1.5]],
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-4)

    def test_win_rate_mode_with_legal_mask(self) -> None:
        """WIN_RATE mode should apply legal_move_mask."""
        labels = torch.tensor(
            [[1.0, 1.0, 1.0]], dtype=torch.float32
        )
        win_rate = torch.tensor(
            [[0.6, 0.3, 0.1]], dtype=torch.float32
        )
        mask = torch.tensor(
            [[1.0, 0.0, 1.0]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            mask,
            mode=PolicyTargetMode.WIN_RATE,
            move_win_rate=win_rate,
        )
        # After mask: [0.6, 0.0, 0.1], sum = 0.7
        expected = torch.tensor(
            [[0.6 / 0.7, 0.0, 0.1 / 0.7]],
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-4)

    def test_win_rate_mode_raises_without_move_win_rate(
        self,
    ) -> None:
        """WIN_RATE mode should raise ValueError when move_win_rate is None."""
        labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        with pytest.raises(
            ValueError, match="move_win_rate is required"
        ):
            build_policy_targets(
                labels,
                legal_move_mask=None,
                mode=PolicyTargetMode.WIN_RATE,
                move_win_rate=None,
            )

    def test_weighted_mode_raises_without_move_win_rate(
        self,
    ) -> None:
        """WEIGHTED mode should raise ValueError when move_win_rate is None."""
        labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        with pytest.raises(
            ValueError, match="move_win_rate is required"
        ):
            build_policy_targets(
                labels,
                legal_move_mask=None,
                mode=PolicyTargetMode.WEIGHTED,
                move_win_rate=None,
            )

    def test_move_label_mode_ignores_move_win_rate(
        self,
    ) -> None:
        """MOVE_LABEL mode should work even when move_win_rate is provided."""
        labels = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        win_rate = torch.tensor(
            [[0.5, 0.5]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            legal_move_mask=None,
            mode=PolicyTargetMode.MOVE_LABEL,
            move_win_rate=win_rate,
        )
        expected = normalize_policy_targets(
            labels, legal_move_mask=None
        )
        assert torch.allclose(result, expected)

    def test_all_zero_win_rate_safe_division(self) -> None:
        """All-zero move_win_rate should not cause division by zero."""
        labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        win_rate = torch.tensor(
            [[0.0, 0.0]], dtype=torch.float32
        )

        result = build_policy_targets(
            labels,
            legal_move_mask=None,
            mode=PolicyTargetMode.WIN_RATE,
            move_win_rate=win_rate,
        )
        # All zeros -> safe_sum = 1.0, result = [0.0, 0.0]
        expected = torch.tensor(
            [[0.0, 0.0]], dtype=torch.float32
        )
        assert torch.allclose(result, expected)
