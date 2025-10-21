"""Vision Transformer tailored for compact board evaluation tasks.

The :class:`VisionTransformer` consumes board tensors shaped
``(batch_size, channels, height, width)`` and predicts a scalar evaluation per
board using the flattened cells as tokens.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisionTransformerConfig:
    """Configuration bundle for :class:`VisionTransformer`."""

    input_channels: int = 104
    board_size: int = 9
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_layers: int = 6
    dropout: float = 0.1
    attention_dropout: float = 0.0
    use_head: bool = True

    @property
    def num_tokens(self) -> int:
        """Number of tokens (board cells) processed by the transformer."""

        return self.board_size * self.board_size


class ViTEncoderBlock(nn.Module):
    """Single Vision Transformer encoder block."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = _FlashSelfAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
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
        attn_output = self.attn(x)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class _FlashSelfAttention(nn.Module):
    """Self-attention layer that prefers FlashAttention when available."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = (
                "embed_dim must be divisible by num_heads for multi-head attention"
            )
            raise ValueError(msg)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # FlashAttention kernels require dropout to be disabled. Enforce at runtime.
        if dropout != 0.0:
            msg = "FlashAttention requires attention dropout to be zero"
            raise ValueError(msg)
        self.dropout_p = float(dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        use_flash = (
            x.device.type == "cuda"
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
        )
        kernel_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=True,
            )
            if use_flash
            else nullcontext()
        )
        try:
            with kernel_context:
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p,
                    is_causal=False,
                )
        except RuntimeError:
            if not use_flash:
                raise
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=True,
                enable_math=True,
            ):
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p,
                    is_causal=False,
                )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.proj(attn)


class VisionTransformer(nn.Module):
    """Vision Transformer designed around small board-shaped inputs."""

    def __init__(self, config: VisionTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisionTransformerConfig()
        self.embedding_dim = self.config.embed_dim
        self.token_projection = nn.Linear(
            self.config.input_channels, self.config.embed_dim
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.config.num_tokens, self.config.embed_dim)
        )
        self.embedding_dropout = nn.Dropout(p=self.config.dropout)
        self.encoder: nn.ModuleList = nn.ModuleList(
            [ViTEncoderBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.head: Optional[nn.Linear]
        if self.config.use_head:
            self.head = nn.Linear(self.config.embed_dim, 1)
        else:
            self.head = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable parameters."""

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.trunc_normal_(self.token_projection.weight, std=0.02)
        if self.token_projection.bias is not None:
            nn.init.zeros_(self.token_projection.bias)
        if self.head is not None:
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def _encode_tokens(self, x: Tensor) -> Tensor:
        """Project inputs into token embeddings and run encoder blocks."""

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
        return tokens

    def forward_features(self, x: Tensor) -> Tensor:
        """Return pooled token features prior to the prediction head."""

        tokens = self._encode_tokens(x)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled

    def forward(self, x: Tensor) -> Tensor:
        """Return a scalar evaluation for each board tensor in ``x``."""

        if self.head is None:
            msg = (
                "VisionTransformer configured without a prediction head cannot "
                "be called directly. Use `forward_features` instead."
            )
            raise RuntimeError(msg)

        pooled = self.forward_features(x)
        output = self.head(pooled)
        return output.squeeze(-1)
