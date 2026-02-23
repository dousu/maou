"""MLP-Mixer network tailored for compact 9x9 feature maps.

The :class:`ShogiMLPMixer` expects input tensors of shape
``(batch_size, 104, 9, 9)`` and returns logits shaped ``(batch_size, num_classes)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from maou.domain.model.tracing import is_tracing


class _FeedForward(nn.Module):
    """Two-layer feed-forward MLP with GELU activation and dropout."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = F.dropout(
            x, p=self.dropout_rate, training=self.training
        )
        x = self.fc2(x)
        x = F.dropout(
            x, p=self.dropout_rate, training=self.training
        )
        return x


@dataclass
class _MixerBlockConfig:
    num_tokens: int
    num_channels: int
    token_dim: int
    channel_dim: int
    dropout_rate: float


class _MixerBlock(nn.Module):
    """A single Mixer block with token and channel mixing stages."""

    def __init__(self, config: _MixerBlockConfig) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(config.num_channels)
        self.token_mlp = _FeedForward(
            config.num_tokens,
            config.token_dim,
            config.dropout_rate,
        )
        self.channel_norm = nn.LayerNorm(config.num_channels)
        self.channel_mlp = _FeedForward(
            config.num_channels,
            config.channel_dim,
            config.dropout_rate,
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


class ShogiMLPMixer(nn.Module):
    """MLP-Mixer tailored to 9×9 shogi planes with rich regularization.

    Args:
        num_classes: Number of output classes for the classifier head.
        num_channels: Number of input feature channels (default ``104``).
        num_tokens: Number of spatial tokens after flattening ``height × width``.
        embed_dim: Channel dimension used inside the Mixer blocks.
        token_dim: Hidden dimension of the token mixing MLP.
        channel_dim: Hidden dimension of the channel mixing MLP.
        depth: Number of Mixer blocks stacked in the encoder.
        dropout_rate: Dropout probability applied throughout the network.

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
        embed_dim: int = 256,
        token_dim: int = 128,
        channel_dim: int = 1024,
        depth: int = 16,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout_rate <= 1.0:
            msg = "dropout_rate must lie within [0.0, 1.0]"
            raise ValueError(msg)

        self.num_tokens = num_tokens
        self.input_channels = num_channels
        self.num_channels = embed_dim
        self.dropout_rate = dropout_rate

        block_config = _MixerBlockConfig(
            num_tokens=num_tokens,
            num_channels=embed_dim,
            token_dim=token_dim,
            channel_dim=channel_dim,
            dropout_rate=dropout_rate,
        )
        self.blocks = nn.ModuleList(
            _MixerBlock(block_config) for _ in range(depth)
        )
        self.input_norm = nn.LayerNorm(num_channels)
        self.embedding = nn.Linear(num_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        if num_classes is None:
            self.head: nn.Linear | None = None
        else:
            self.head = nn.Linear(embed_dim, num_classes)

    def _flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = height * width
        if not is_tracing():
            if tokens != self.num_tokens:
                msg = "Spatial dimensions do not match the configured number of tokens"
                raise ValueError(msg)
            if channels != self.input_channels:
                msg = "Input channels do not match the configured channel dimension"
                raise ValueError(msg)
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
        tokens = self.input_norm(tokens)
        tokens = self.embedding(tokens)
        tokens = F.dropout(
            tokens, p=self.dropout_rate, training=self.training
        )
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        tokens = F.dropout(
            tokens, p=self.dropout_rate, training=self.training
        )

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

    def get_freezable_groups(self) -> list[nn.Module]:
        """Return mixer block groups ordered from lowest to highest.

        Returns:
            list[nn.Module]: Individual mixer blocks (depth groups, default 16).
        """
        return list(self.blocks)

    def get_output_norm(self) -> nn.Module | None:
        """Return the final LayerNorm applied after mixer blocks.

        Returns:
            nn.Module: The output LayerNorm.
        """
        return self.norm

    def preprocess_for_blocks(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """入力テンソルをブロックが期待するトークン形式に変換する．

        ``forward_features()`` 内のブロック前処理と同じロジック．
        flatten → input_norm → embedding → dropout の順に処理する．

        Args:
            x: バックボーン入力テンソル (batch, channels, H, W)

        Returns:
            トークンテンソル (batch, num_tokens, embed_dim)
        """
        tokens = self._flatten_tokens(x)
        tokens = self.input_norm(tokens)
        tokens = self.embedding(tokens)
        tokens = F.dropout(
            tokens, p=self.dropout_rate, training=self.training
        )
        return tokens


def print_model_summary(model: ShogiMLPMixer) -> None:
    """Print a concise summary with parameter count for the given model."""

    param_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(
        "ShogiMLPMixer Summary\n"
        f"Parameters: {param_count:,}\n"
        f"Mixer depth: {len(model.blocks)}\n"
        f"Embedding dimension: {model.num_channels}"
    )
