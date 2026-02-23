"""Neural network architectures used for shogi learning workflows."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, Tuple, Union

import torch
from torch import nn

from maou.domain.board import shogi
from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.protocol import FreezableBackbone
from maou.domain.model.resnet import (
    BottleneckBlock,
)
from maou.domain.model.resnet import ResNet as DomainResNet
from maou.domain.model.tracing import is_tracing
from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)
from maou.domain.move.label import MOVE_LABELS_NUM

BackboneArchitecture = Literal["resnet", "mlp-mixer", "vit"]
BACKBONE_ARCHITECTURES: tuple[BackboneArchitecture, ...] = (
    "resnet",
    "mlp-mixer",
    "vit",
)

DEFAULT_BOARD_VOCAB_SIZE = 256
BOARD_EMBEDDING_DIM = 32
DEFAULT_HAND_PROJECTION_DIM = 32


ModelInputs = Union[torch.Tensor, Sequence[torch.Tensor]]
PIECES_IN_HAND_VECTOR_SIZE = len(shogi.MAX_PIECES_IN_HAND) * 2


class HeadlessNetwork(nn.Module):
    """Headless shogi backbone supporting ResNet, MLP-Mixer, and ViT."""

    def __init__(
        self,
        *,
        board_vocab_size: int = DEFAULT_BOARD_VOCAB_SIZE,
        embedding_dim: int = BOARD_EMBEDDING_DIM,
        hand_projection_dim: int = 0,
        board_size: Tuple[int, int] = (9, 9),
        architecture: BackboneArchitecture = "resnet",
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (
            64,
            128,
            256,
            512,
        ),
        pooling: nn.Module | None = None,
        architecture_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.architecture: BackboneArchitecture = architecture
        self.board_vocab_size = board_vocab_size
        self._board_size = board_size
        self._embedding_channels = embedding_dim
        self._hand_projection_dim = hand_projection_dim
        self.embedding = nn.Embedding(
            board_vocab_size, embedding_dim
        )

        # hand_projection_dim > 0 の場合のみ projection レイヤーを作成
        self._hand_projection: nn.Linear | None
        if hand_projection_dim > 0:
            self._hand_projection = nn.Linear(
                PIECES_IN_HAND_VECTOR_SIZE,
                hand_projection_dim,
            )
        else:
            self._hand_projection = None

        config: dict[str, Any] = dict(architecture_config or {})

        # Total input channels to backbone (board embedding + hand projection)
        backbone_input_channels = (
            embedding_dim + hand_projection_dim
        )

        if architecture == "resnet":
            if (
                len(layers) != 4
                or len(strides) != 4
                or len(out_channels) != 4
            ):
                msg = (
                    "ResNet requires four stages for layers, strides, and "
                    "out_channels."
                )
                raise ValueError(msg)

            expansion = getattr(block, "expansion", 1)
            self.backbone: (
                DomainResNet | ShogiMLPMixer | VisionTransformer
            ) = DomainResNet(
                block=block,
                in_channels=backbone_input_channels,
                layers=list(layers),
                strides=list(strides),
                list_out_channels=list(out_channels),
            )
            self.pool: nn.Module = (
                pooling
                if pooling is not None
                else nn.AdaptiveAvgPool2d((1, 1))
            )
            self._embedding_dim = out_channels[-1] * expansion
        elif architecture == "mlp-mixer":
            mixer_kwargs: dict[str, Any] = {
                "num_classes": None,
                "num_channels": backbone_input_channels,
                "num_tokens": board_size[0] * board_size[1],
            }
            mixer_kwargs.update(config)
            self.backbone = ShogiMLPMixer(**mixer_kwargs)
            self.pool = nn.Identity()
            self._embedding_dim = self.backbone.num_channels
        elif architecture == "vit":
            if board_size[0] != board_size[1]:
                msg = "Vision Transformer requires square board dimensions."
                raise ValueError(msg)
            vit_kwargs: dict[str, Any] = {
                "input_channels": backbone_input_channels,
                "board_size": board_size[0],
                "use_head": False,
            }
            vit_kwargs.update(config)
            vit_config = VisionTransformerConfig(**vit_kwargs)
            self.backbone = VisionTransformer(vit_config)
            self.pool = nn.Identity()
            self._embedding_dim = self.backbone.embedding_dim
        else:  # pragma: no cover - defensive branch
            msg = f"Unsupported backbone architecture `{architecture}`."
            raise ValueError(msg)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the pooled backbone features."""

        return self._embedding_dim

    @property
    def hand_projection_dim(self) -> int:
        """持ち駒射影の次元数を返す．"""
        return self._hand_projection_dim

    def forward_features(self, x: ModelInputs) -> torch.Tensor:
        """Return pooled convolutional features from the shared backbone.

        Args:
            x: 盤面テンソル，または (盤面, 持ち駒) のシーケンス
        """
        board_tensor, hand_tensor = self._separate_inputs(x)
        embedded_board = self._prepare_inputs(board_tensor)

        if self._hand_projection is not None:
            combined = self._combine_board_and_hand(
                embedded_board, hand_tensor
            )
        else:
            combined = embedded_board

        if self.architecture == "resnet":
            features = self.backbone(combined)
            pooled = self.pool(features)
            return torch.flatten(pooled, 1)

        backbone_forward = getattr(
            self.backbone, "forward_features", None
        )
        if backbone_forward is None:
            msg = "Configured backbone does not implement forward_features."
            raise RuntimeError(msg)
        forward_fn: Callable[[torch.Tensor], torch.Tensor] = (
            backbone_forward
        )
        return forward_fn(combined)

    def forward(
        self, x: ModelInputs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Alias of :meth:`forward_features` for convenience."""

        return self.forward_features(x)

    def freeze_except_last_n(self, n: int) -> int:
        """Freeze backbone except the last n groups, plus always freeze embedding/pool.

        Strategy: freeze ALL backbone parameters first, then selectively
        unfreeze the last n groups. This ensures non-group parameters
        (e.g. input_norm, token_projection) are always frozen.

        Args:
            n: Number of trailing backbone groups to keep trainable.
                0 = freeze all backbone parameters.
                Values exceeding total groups are clamped
                (= only non-group params frozen).

        Returns:
            int: Total number of frozen parameters.

        Raises:
            TypeError: If backbone does not implement get_freezable_groups().
        """
        logger = logging.getLogger(__name__)

        if not isinstance(self.backbone, FreezableBackbone):
            msg = (
                f"Backbone {type(self.backbone).__name__} does not "
                f"implement get_freezable_groups()."
            )
            raise TypeError(msg)

        # 1. Always freeze embedding and pool
        for module in [self.embedding, self.pool]:
            for param in module.parameters():
                param.requires_grad = False

        # 2. Freeze ALL backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. Selectively unfreeze last N groups
        groups = self.backbone.get_freezable_groups()
        total = len(groups)
        n_clamped = min(n, total)
        freeze_count = total - n_clamped

        if n > total:
            logger.info(
                "trainable_layers=%d exceeds total groups=%d, "
                "clamped to %d.",
                n,
                total,
                total,
            )

        for group in groups[freeze_count:]:
            for param in group.parameters():
                param.requires_grad = True

        # Count frozen parameters
        frozen_params = sum(
            1 for p in self.parameters() if not p.requires_grad
        )
        return frozen_params

    @staticmethod
    def _separate_inputs(
        inputs: ModelInputs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """盤面テンソルと持ち駒テンソルを分離する．

        単一の ``torch.Tensor`` が渡された場合は持ち駒なしとして扱い，
        ``(board, hand)`` のシーケンスが渡された場合はそれぞれを返す．

        Args:
            inputs: 盤面テンソル，または (盤面, 持ち駒) のシーケンス

        Returns:
            (board_tensor, hand_tensor | None) のタプル
        """
        if isinstance(inputs, torch.Tensor):
            return inputs, None
        if isinstance(inputs, Sequence):
            if len(inputs) != 2:
                msg = "Expected inputs to contain board and pieces_in_hand tensors."
                raise ValueError(msg)
            board_tensor, pieces_tensor = inputs[0], inputs[1]
            if not isinstance(board_tensor, torch.Tensor):
                msg = "Board input must be a torch.Tensor."
                raise TypeError(msg)
            if pieces_tensor is None:
                return board_tensor, None
            if not isinstance(pieces_tensor, torch.Tensor):
                msg = "Pieces-in-hand input must be a torch.Tensor."
                raise TypeError(msg)
            return board_tensor, pieces_tensor
        msg = f"Unsupported input structure: {type(inputs)!r}"
        raise TypeError(msg)

    def _combine_board_and_hand(
        self,
        embedded_board: torch.Tensor,
        hand_tensor: torch.Tensor | None,
    ) -> torch.Tensor:
        """盤面 embedding と持ち駒 projection を結合する．

        Args:
            embedded_board: 盤面の embedding テンソル (batch, channels, H, W)
            hand_tensor: 持ち駒ベクトル (batch, PIECES_IN_HAND_VECTOR_SIZE) または None

        Returns:
            結合済みテンソル (batch, channels + hand_projection_dim, H, W)
        """
        assert self._hand_projection is not None

        batch_size = embedded_board.shape[0]
        height, width = (
            embedded_board.shape[2],
            embedded_board.shape[3],
        )

        if hand_tensor is not None:
            # Project hand features to hand_projection_dim
            projected = self._hand_projection(
                hand_tensor.to(
                    dtype=embedded_board.dtype,
                    device=embedded_board.device,
                )
            )

            # Architecture-specific hand feature integration
            if self.architecture == "resnet":
                # ResNet: Expand hand features to all spatial positions
                hand_features = projected.view(
                    batch_size,
                    self._hand_projection_dim,
                    1,
                    1,
                ).expand(-1, -1, height, width)
            else:
                # MLP-Mixer/ViT: Place hand features only at position (0, 0)
                hand_features = torch.zeros(
                    batch_size,
                    self._hand_projection_dim,
                    height,
                    width,
                    dtype=embedded_board.dtype,
                    device=embedded_board.device,
                )
                hand_features[:, :, 0, 0] = projected
        else:
            # Create zero padding if no hand information provided
            hand_features = torch.zeros(
                batch_size,
                self._hand_projection_dim,
                height,
                width,
                dtype=embedded_board.dtype,
                device=embedded_board.device,
            )

        return torch.cat([embedded_board, hand_features], dim=1)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        input_type, tensor = self._validate_inputs(x)
        if input_type == "embedded":
            return tensor

        board_tensor = tensor.to(torch.long)
        embedded = self.embedding(board_tensor)
        return embedded.permute(0, 3, 1, 2).contiguous()

    def _validate_inputs(
        self, x: torch.Tensor
    ) -> tuple[Literal["board", "embedded"], torch.Tensor]:
        # NOTE: x.dim() や channels による分岐は静的形状モード
        # (dynamic=False)では specialization として正しく動作する．
        # dynamic=True に切り替えた場合，これらの分岐が
        # graph break を引き起こす可能性がある．
        tracing = is_tracing()

        if x.dim() == 3:
            if not tracing:
                height, width = x.shape[1:]
                if (height, width) != self._board_size:
                    msg = (
                        "Input board dimensions must match the configured board size. "
                        f"Expected {self._board_size} but received {(height, width)}."
                    )
                    raise ValueError(msg)
                if torch.is_floating_point(x):
                    msg = "Board identifiers must be integral tensors."
                    raise ValueError(msg)
            return "board", x

        if x.dim() == 4:
            _, channels, height, width = x.shape
            if not tracing:
                if (height, width) != self._board_size:
                    msg = (
                        "Input board dimensions must match the configured board size. "
                        f"Expected {self._board_size} but received {(height, width)}."
                    )
                    raise ValueError(msg)
            if channels == 1:
                if not tracing and torch.is_floating_point(x):
                    msg = "Board identifiers must be integral tensors."
                    raise ValueError(msg)
                return "board", x.squeeze(1)
            if channels == self._embedding_channels:
                return "embedded", x

        if not tracing:
            msg = (
                "Inputs must be integer board IDs with shape (batch, 9, 9) or "
                "embedded features shaped (batch, channels, 9, 9)."
            )
            raise ValueError(msg)
        return "board", x

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load only backbone parameters, ignoring head-specific entries."""

        if hasattr(self, "policy_head") or hasattr(
            self, "value_head"
        ):
            return super().load_state_dict(
                state_dict, strict=strict, assign=assign
            )

        backbone_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
            or key.startswith("embedding.")
            or key.startswith("_hand_projection.")
        }
        return super().load_state_dict(
            backbone_state, strict=strict, assign=assign
        )

    @property
    def input_channels(self) -> int:
        """Return the number of channels produced by the board embedding."""

        return self._embedding_channels


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


class ReachableSquaresHead(nn.Module):
    """Head for predicting reachable squares (9×9 binary output).

    This head is used in Stage 1 training to learn which board squares
    pieces can move to. Outputs logits for each of the 81 board squares.

    The model learns basic piece movement rules without considering
    other pieces or complex game state.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        board_size: Tuple[int, int] = (9, 9),
        hidden_dim: int | None = None,
    ) -> None:
        """Initialize ReachableSquaresHead.

        Args:
            input_dim: Dimension of input features from backbone
            board_size: Board dimensions (default: (9，9) for Shogi)
            hidden_dim: Optional hidden layer dimension.
                If None，uses single linear layer.
        """
        super().__init__()
        self.board_size = board_size
        output_dim = (
            board_size[0] * board_size[1]
        )  # 81 for 9×9 board

        layers: list[nn.Module]
        if hidden_dim is None:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            ]
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return reachable squares logits for the provided features.

        Args:
            features: Backbone features (batch，input_dim)

        Returns:
            Logits for each board square (batch，81)

        Note: Output is logits，not probabilities.
        Use BCEWithLogitsLoss for training.
        Apply sigmoid during inference: torch.sigmoid(logits)
        """
        return self.head(features)


class LegalMovesHead(nn.Module):
    """Head for predicting legal moves (multi-label binary classification).

    This head is used in Stage 2 training to learn which moves are legal
    in a given position. Outputs logits for MOVE_LABELS_NUM classes (1496).

    Unlike the policy head (which outputs a probability distribution)，
    this performs multi-label classification where multiple moves can be
    legal simultaneously. Each move is treated as an independent binary
    classification problem.

    Supports optional dropout regularization when using a hidden layer.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_move_labels: int = MOVE_LABELS_NUM,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LegalMovesHead.

        Args:
            input_dim: Dimension of input features from backbone
            num_move_labels: Number of move label classes
                (default: MOVE_LABELS_NUM = 1496)
            hidden_dim: Optional hidden layer dimension.
                If None，uses single linear layer.
            dropout: Dropout rate for regularization (0.0-1.0).
                Only effective when hidden_dim is not None.
                Default: 0.0 (no dropout)
        """
        super().__init__()
        layers: list[nn.Module]
        if hidden_dim is None:
            layers = [nn.Linear(input_dim, num_move_labels)]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(
                nn.Linear(hidden_dim, num_move_labels)
            )
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return legal moves logits for the provided features.

        Args:
            features: Backbone features (batch，input_dim)

        Returns:
            Logits for each move label (batch，MOVE_LABELS_NUM)

        Note: Output is logits for multi-label classification.
        Use BCEWithLogitsLoss for training.
        Apply sigmoid during inference: torch.sigmoid(logits)
        """
        return self.head(features)


class Network(HeadlessNetwork):
    """Dual-head shogi network built on selectable backbones."""

    def __init__(
        self,
        *,
        num_policy_classes: int = MOVE_LABELS_NUM,
        board_vocab_size: int = DEFAULT_BOARD_VOCAB_SIZE,
        embedding_dim: int = BOARD_EMBEDDING_DIM,
        hand_projection_dim: int = DEFAULT_HAND_PROJECTION_DIM,
        board_size: Tuple[int, int] = (9, 9),
        architecture: BackboneArchitecture = "resnet",
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (
            64,
            128,
            256,
            512,
        ),
        pooling: nn.Module | None = None,
        policy_hidden_dim: int | None = None,
        value_hidden_dim: int | None = None,
        architecture_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            board_vocab_size=board_vocab_size,
            embedding_dim=embedding_dim,
            hand_projection_dim=hand_projection_dim,
            board_size=board_size,
            architecture=architecture,
            block=block,
            layers=layers,
            strides=strides,
            out_channels=out_channels,
            pooling=pooling,
            architecture_config=architecture_config,
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
        self, x: ModelInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy and value predictions for the given features."""

        features = self.forward_features(x)
        policy_logits = self.policy_head(features)
        value_logit = self.value_head(features)
        return policy_logits, value_logit

    def forward_features(self, x: ModelInputs) -> torch.Tensor:
        """Return pooled features from the shared backbone."""
        return super().forward_features(x)
