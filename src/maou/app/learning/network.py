"""Lightweight MLP-Mixer based policy and value network."""

from __future__ import annotations

import torch
import torch.nn as nn

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.mlp_mixer import LightweightMLPMixer


class Network(nn.Module):
    """Dual-head shogi network that shares a Lightweight MLP-Mixer backbone.

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
        token_dim: int = 64,
        channel_dim: int = 256,
        depth: int = 4,
        policy_hidden_dim: int | None = None,
        value_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone: LightweightMLPMixer = LightweightMLPMixer(
            num_classes=None,
            num_channels=num_channels,
            num_tokens=num_tokens,
            token_dim=token_dim,
            channel_dim=channel_dim,
            depth=depth,
        )

        policy_layers: list[nn.Module]
        if policy_hidden_dim is None:
            policy_layers = [nn.Linear(num_channels, num_policy_classes)]
        else:
            policy_layers = [
                nn.Linear(num_channels, policy_hidden_dim),
                nn.GELU(),
                nn.Linear(policy_hidden_dim, num_policy_classes),
            ]
        self.policy_head: nn.Module = nn.Sequential(*policy_layers)

        value_layers: list[nn.Module]
        if value_hidden_dim is None:
            value_layers = [nn.Linear(num_channels, 1)]
        else:
            value_layers = [
                nn.Linear(num_channels, value_hidden_dim),
                nn.GELU(),
                nn.Linear(value_hidden_dim, 1),
            ]
        self.value_head: nn.Module = nn.Sequential(*value_layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy and value predictions for the given features."""

        features = self.backbone.forward_features(x)
        policy_logits = self.policy_head(features)
        value_logit = self.value_head(features)
        return policy_logits, value_logit

