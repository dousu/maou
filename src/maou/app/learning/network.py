"""Shogi-optimized MLP-Mixer based policy and value network."""

from __future__ import annotations

import torch
import torch.nn as nn

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.mlp_mixer import ShogiMLPMixer


class HeadlessNetwork(nn.Module):
    """Shared Shogi MLP-Mixer backbone without policy/value heads."""

    def __init__(
        self,
        *,
        num_channels: int = FEATURES_NUM,
        num_tokens: int = 81,
        embed_dim: int = 256,
        token_dim: int = 128,
        channel_dim: int = 1024,
        depth: int = 16,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()
        self.backbone: ShogiMLPMixer = ShogiMLPMixer(
            num_classes=None,
            num_channels=num_channels,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            token_dim=token_dim,
            channel_dim=channel_dim,
            depth=depth,
            dropout_rate=dropout_rate,
        )
        self._embedding_dim = embed_dim

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the pooled backbone features."""

        return self._embedding_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled token features from the shared backbone."""

        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward_features` for convenience."""

        return self.forward_features(x)


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
    """Value head projecting backbone features to a scalar output."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module]
        if hidden_dim is None:
            layers = [nn.Linear(input_dim, 1)]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            ]
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a scalar value prediction for the provided features."""

        return self.head(features)


class Network(HeadlessNetwork):
    """Dual-head shogi network that shares a Shogi MLP-Mixer backbone.

    The shared mixer extracts a global representation from the 9x9 feature
    planes. Separate policy and value heads consume this representation and can
    optionally introduce hidden projections for additional capacity.

    Args:
        num_policy_classes: Number of classes returned by the policy head.
        num_channels: Number of input feature channels. Defaults to the shogi
            board representation (:data:`FEATURES_NUM`).
        num_tokens: Number of spatial tokens (``height × width``).
        token_dim: Hidden dimension of the token mixing MLP.
        channel_dim: Hidden dimension of the channel mixing MLP.
        depth: Number of Mixer blocks stacked in the backbone.
        policy_hidden_dim: Optional hidden dimension inserted in the policy
            head. When ``None`` the head is a single linear layer.
        value_hidden_dim: Optional hidden dimension inserted in the value head.
            When ``None`` the head is a single linear layer.
    """

    def __init__(
        self,
        *,
        num_policy_classes: int = MOVE_LABELS_NUM,
        num_channels: int = FEATURES_NUM,
        num_tokens: int = 81,
        embed_dim: int = 256,
        token_dim: int = 128,
        channel_dim: int = 1024,
        depth: int = 16,
        dropout_rate: float = 0.15,
        policy_hidden_dim: int | None = None,
        value_hidden_dim: int | None = None,
    ) -> None:
        super().__init__(
            num_channels=num_channels,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            token_dim=token_dim,
            channel_dim=channel_dim,
            depth=depth,
            dropout_rate=dropout_rate,
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
