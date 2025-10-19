"""MLP-Mixer network tailored for compact 9x9 feature maps.

The :class:`LightweightMLPMixer` expects input tensors of shape
``(batch_size, 104, 9, 9)`` and returns logits shaped ``(batch_size, num_classes)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class _FeedForward(nn.Module):
    """Two-layer feed-forward MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _MixerBlockConfig:
    num_tokens: int
    num_channels: int
    token_dim: int
    channel_dim: int


class _MixerBlock(nn.Module):
    """A single Mixer block with token and channel mixing stages."""

    def __init__(self, config: _MixerBlockConfig) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(config.num_channels)
        self.token_mlp = _FeedForward(
            config.num_tokens, config.token_dim
        )
        self.channel_norm = nn.LayerNorm(config.num_channels)
        self.channel_mlp = _FeedForward(
            config.num_channels, config.channel_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = residual + y

        residual = x
        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        return residual + y


class LightweightMLPMixer(nn.Module):
    """MLP-Mixer for 9x9 spatial inputs with 104 channels.

    Args:
        num_classes: Number of output classes for the classifier head.
        num_channels: Channel dimension of each token. Defaults to ``104``.
        num_tokens: Number of tokens after flattening the spatial dimensions.
            Defaults to ``81`` (from ``9 × 9``).
        token_dim: Hidden dimension of the token mixing MLP.
        channel_dim: Hidden dimension of the channel mixing MLP.
        depth: Number of Mixer blocks stacked in the encoder.

    The ``forward`` method accepts an optional ``token_mask`` (shape ``B × T``) to
    exclude tokens from the pooled representation and can return the per-token
    embeddings when ``return_tokens`` is set to ``True``.

    """

    def __init__(
        self,
        num_classes: int | None,
        *,
        num_channels: int = 104,
        num_tokens: int = 81,
        token_dim: int = 64,
        channel_dim: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_channels = num_channels

        block_config = _MixerBlockConfig(
            num_tokens=num_tokens,
            num_channels=num_channels,
            token_dim=token_dim,
            channel_dim=channel_dim,
        )
        self.blocks = nn.ModuleList(
            _MixerBlock(block_config) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(num_channels)
        if num_classes is None:
            self.head: nn.Linear | None = None
        else:
            self.head = nn.Linear(num_channels, num_classes)

    def _flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = height * width
        if tokens != self.num_tokens:
            raise ValueError(
                "Spatial dimensions do not match the configured number of tokens"
            )
        if channels != self.num_channels:
            raise ValueError(
                "Input channels do not match the configured channel dimension"
            )
        x = x.view(batch_size, channels, tokens)
        return x.transpose(1, 2)

    def forward_features(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Return pooled token features prior to the classifier head."""

        tokens = self._flatten_tokens(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        if token_mask is not None:
            if token_mask.shape != (
                tokens.size(0),
                self.num_tokens,
            ):
                msg = "token_mask must be shaped (batch_size, num_tokens)"
                raise ValueError(msg)
            mask = token_mask.to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / denom
        else:
            pooled = tokens.mean(dim=1)

        if return_tokens:
            return pooled, tokens
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_tokens:
            features, tokens = self.forward_features(
                x, token_mask, return_tokens=True
            )
        else:
            features = self.forward_features(x, token_mask)
            tokens = None

        if self.head is None:
            if return_tokens:
                assert tokens is not None
                return features, tokens
            return features

        logits = self.head(features)
        if return_tokens:
            assert tokens is not None
            return logits, tokens
        return logits
