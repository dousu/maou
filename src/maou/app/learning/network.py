"""Neural network architectures used for shogi learning workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

import torch
from torch import nn

from maou.domain.board.shogi import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.model.resnet import (
    BottleneckBlock,
    ResNet as DomainResNet,
)


class HeadlessNetwork(nn.Module):
    """Headless shogi backbone that wraps the domain ResNet."""

    def __init__(
        self,
        *,
        num_channels: int = FEATURES_NUM,
        board_size: Tuple[int, int] = (9, 9),
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        pooling: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len(layers) != 4 or len(strides) != 4 or len(out_channels) != 4:
            msg = "ResNet requires four stages for layers, strides, and out_channels."
            raise ValueError(msg)

        expansion = getattr(block, "expansion", 1)

        self.backbone: DomainResNet = DomainResNet(
            block=block,
            in_channels=num_channels,
            layers=list(layers),
            strides=list(strides),
            list_out_channels=list(out_channels),
        )
        self.pool: nn.Module = (
            pooling if pooling is not None else nn.AdaptiveAvgPool2d((1, 1))
        )
        self._embedding_dim = out_channels[-1] * expansion
        self._num_channels = num_channels
        self._board_size = board_size

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the pooled backbone features."""

        return self._embedding_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled convolutional features from the shared backbone."""

        self._validate_inputs(x)
        features = self.backbone(x)
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Alias of :meth:`forward_features` for convenience."""

        return self.forward_features(x)

    def _validate_inputs(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            msg = (
                "ResNet expects inputs of shape (batch, channels, height, width)."
            )
            raise ValueError(msg)

        _, channels, height, width = x.shape
        if channels != self._num_channels:
            msg = (
                f"Expected {self._num_channels} channels but received "
                f"{channels}."
            )
            raise ValueError(msg)
        if (height, width) != self._board_size:
            msg = (
                "Input board dimensions must match the configured board size. "
                f"Expected {self._board_size} but received {(height, width)}."
            )
            raise ValueError(msg)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load only backbone parameters, ignoring head-specific entries."""

        if hasattr(self, "policy_head") or hasattr(self, "value_head"):
            return super().load_state_dict(
                state_dict, strict=strict, assign=assign
            )

        backbone_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        return super().load_state_dict(
            backbone_state, strict=strict, assign=assign
        )


class PolicyHead(nn.Module):
    """Policy head projecting backbone features to move logits."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_policy_classes: int = MOVE_LABELS_NUM,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module]
        if hidden_dim is None:
            layers = [nn.Linear(input_dim, num_policy_classes)]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_policy_classes),
            ]
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return policy logits for the provided features."""

        return self.head(features)


class ValueHead(nn.Module):
    """Value head projecting backbone features to a scalar output.

    Note: This head outputs logits (raw scores), not probabilities.
    Use BCEWithLogitsLoss for training, which combines sigmoid and BCE loss
    for numerical stability and compatibility with mixed precision training.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module]
        if hidden_dim is None:
            # Output logits directly (no Sigmoid)
            layers = [nn.Linear(input_dim, 1)]
        else:
            # Output logits directly (no Sigmoid)
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            ]
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a scalar value logit for the provided features.

        Note: Output is a logit (raw score), not a probability.
        Apply sigmoid during inference: torch.sigmoid(value_logit)
        """

        return self.head(features)


class Network(HeadlessNetwork):
    """Dual-head shogi network built on a ResNet backbone."""

    def __init__(
        self,
        *,
        num_policy_classes: int = MOVE_LABELS_NUM,
        num_channels: int = FEATURES_NUM,
        board_size: Tuple[int, int] = (9, 9),
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        pooling: nn.Module | None = None,
        policy_hidden_dim: int | None = None,
        value_hidden_dim: int | None = None,
    ) -> None:
        super().__init__(
            num_channels=num_channels,
            board_size=board_size,
            block=block,
            layers=layers,
            strides=strides,
            out_channels=out_channels,
            pooling=pooling,
        )
        self.policy_head = PolicyHead(
            input_dim=self.embedding_dim,
            num_policy_classes=num_policy_classes,
            hidden_dim=policy_hidden_dim,
        )
        self.value_head = ValueHead(
            input_dim=self.embedding_dim,
            hidden_dim=value_hidden_dim,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy and value predictions for the given features."""

        features = self.forward_features(x)
        policy_logits = self.policy_head(features)
        value_logit = self.value_head(features)
        return policy_logits, value_logit
