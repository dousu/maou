"""Vision Transformer model definitions for shogi board evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ShogiViTConfig:
    """Configuration for :class:`ShogiVisionTransformer`."""

    input_channels: int = 104
    board_size: int = 9
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_layers: int = 6
    dropout: float = 0.1
    attention_dropout: float = 0.1

    @property
    def num_tokens(self) -> int:
        """Return the number of tokens (board cells) processed by the model."""

        return self.board_size * self.board_size


class ViTEncoderBlock(nn.Module):
    """Single Vision Transformer encoder block."""

    def __init__(self, config: ShogiViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=config.dropout)
        mlp_hidden = int(config.embed_dim * config.mlp_ratio)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(mlp_hidden, config.embed_dim),
            nn.Dropout(p=config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the encoder block to the given token embeddings."""

        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class ShogiVisionTransformer(nn.Module):
    """Vision Transformer tailored for shogi board evaluation."""

    def __init__(
        self, config: ShogiViTConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or ShogiViTConfig()
        self.token_projection = nn.Linear(
            self.config.input_channels, self.config.embed_dim
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1, self.config.num_tokens, self.config.embed_dim
            )
        )
        self.embedding_dropout = nn.Dropout(
            p=self.config.dropout
        )
        self.encoder: nn.ModuleList = nn.ModuleList(
            [
                ViTEncoderBlock(self.config)
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable parameters."""

        nn.init.trunc_normal_(
            self.positional_embedding, std=0.02
        )
        nn.init.trunc_normal_(
            self.token_projection.weight, std=0.02
        )
        if self.token_projection.bias is not None:
            nn.init.zeros_(self.token_projection.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that returns a scalar evaluation for each shogi board."""

        batch_size = x.shape[0]
        tokens = x.permute(0, 2, 3, 1).reshape(
            batch_size,
            self.config.num_tokens,
            self.config.input_channels,
        )
        tokens = self.token_projection(tokens)
        tokens = tokens + self.positional_embedding
        tokens = self.embedding_dropout(tokens)
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        output = self.head(pooled)
        return output.squeeze(-1)


__all__ = ["ShogiVisionTransformer", "ShogiViTConfig"]
