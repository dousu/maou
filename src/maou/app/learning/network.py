"""Neural network architectures used for shogi learning workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, Tuple, Union

import torch
from torch import nn

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board import shogi
from maou.domain.model.resnet import (
    BottleneckBlock,
    ResNet as DomainResNet,
)
from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


BackboneArchitecture = Literal["resnet", "mlp-mixer", "vit"]
BACKBONE_ARCHITECTURES: tuple[BackboneArchitecture, ...] = (
    "resnet",
    "mlp-mixer",
    "vit",
)

DEFAULT_BOARD_VOCAB_SIZE = 256
BOARD_EMBEDDING_DIM = 32


ModelInputs = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
PIECES_IN_HAND_VECTOR_SIZE = len(shogi.MAX_PIECES_IN_HAND) * 2


class HeadlessNetwork(nn.Module):
    """Headless shogi backbone supporting ResNet, MLP-Mixer, and ViT."""

    def __init__(
        self,
        *,
        board_vocab_size: int = DEFAULT_BOARD_VOCAB_SIZE,
        embedding_dim: int = BOARD_EMBEDDING_DIM,
        board_size: Tuple[int, int] = (9, 9),
        architecture: BackboneArchitecture = "resnet",
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        pooling: nn.Module | None = None,
        architecture_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.architecture: BackboneArchitecture = architecture
        self.board_vocab_size = board_vocab_size
        self._board_size = board_size
        self._embedding_channels = embedding_dim
        self.embedding = nn.Embedding(board_vocab_size, embedding_dim)
        config: dict[str, Any] = dict(architecture_config or {})

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
            self.backbone: DomainResNet | ShogiMLPMixer | VisionTransformer = (
                DomainResNet(
                    block=block,
                    in_channels=embedding_dim,
                    layers=list(layers),
                    strides=list(strides),
                    list_out_channels=list(out_channels),
                )
            )
            self.pool: nn.Module = (
                pooling if pooling is not None else nn.AdaptiveAvgPool2d((1, 1))
            )
            self._embedding_dim = out_channels[-1] * expansion
        elif architecture == "mlp-mixer":
            mixer_kwargs: dict[str, Any] = {
                "num_classes": None,
                "num_channels": embedding_dim,
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
                "input_channels": embedding_dim,
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled convolutional features from the shared backbone."""

        inputs = self._prepare_inputs(x)
        if self.architecture == "resnet":
            features = self.backbone(inputs)
            pooled = self.pool(features)
            return torch.flatten(pooled, 1)

        backbone_forward = getattr(self.backbone, "forward_features", None)
        if backbone_forward is None:
            msg = (
                "Configured backbone does not implement forward_features."
            )
            raise RuntimeError(msg)
        forward_fn: Callable[[torch.Tensor], torch.Tensor] = backbone_forward
        return forward_fn(inputs)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Alias of :meth:`forward_features` for convenience."""

        return self.forward_features(x)

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
        if x.dim() == 3:
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
            if (height, width) != self._board_size:
                msg = (
                    "Input board dimensions must match the configured board size. "
                    f"Expected {self._board_size} but received {(height, width)}."
                )
                raise ValueError(msg)
            if channels == 1:
                if torch.is_floating_point(x):
                    msg = "Board identifiers must be integral tensors."
                    raise ValueError(msg)
                return "board", x.squeeze(1)
            if channels == self._embedding_channels:
                return "embedded", x

        msg = (
            "Inputs must be integer board IDs with shape (batch, 9, 9) or "
            "embedded features shaped (batch, channels, 9, 9)."
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
            if key.startswith("backbone.") or key.startswith("embedding.")
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


class Network(HeadlessNetwork):
    """Dual-head shogi network built on selectable backbones."""

    def __init__(
        self,
        *,
        num_policy_classes: int = MOVE_LABELS_NUM,
        board_vocab_size: int = DEFAULT_BOARD_VOCAB_SIZE,
        embedding_dim: int = BOARD_EMBEDDING_DIM,
        board_size: Tuple[int, int] = (9, 9),
        architecture: BackboneArchitecture = "resnet",
        block: type[nn.Module] = BottleneckBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
        out_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        pooling: nn.Module | None = None,
        policy_hidden_dim: int | None = None,
        value_hidden_dim: int | None = None,
        architecture_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            board_vocab_size=board_vocab_size,
            embedding_dim=embedding_dim,
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
        self._hand_projection = nn.Linear(
            PIECES_IN_HAND_VECTOR_SIZE, self.input_channels
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
        board_tensor, hand_tensor = self._separate_inputs(x)
        embedded_board = self._prepare_inputs(board_tensor)
        if hand_tensor is not None:
            projected = self._hand_projection(
                hand_tensor.to(
                    dtype=embedded_board.dtype, device=embedded_board.device
                )
            ).view(embedded_board.shape[0], self.input_channels, 1, 1)
            embedded_board = embedded_board + projected
        return super().forward_features(embedded_board)

    @staticmethod
    def _separate_inputs(
        inputs: ModelInputs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(inputs, tuple):
            if len(inputs) != 2:
                msg = (
                    "Expected inputs to be a tuple of (board, pieces_in_hand)."
                )
                raise ValueError(msg)
            return inputs[0], inputs[1]
        return inputs, None
