"""Neural network architectures used for shogi learning workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, Tuple

import torch
from torch import nn

from maou.domain.board.shogi import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
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


class HeadlessNetwork(nn.Module):
    """Headless shogi backbone supporting ResNet, MLP-Mixer, and ViT."""

    def __init__(
        self,
        *,
        num_channels: int = FEATURES_NUM,
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
        self._num_channels = num_channels
        self._board_size = board_size
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
                    in_channels=num_channels,
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
                "num_channels": num_channels,
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
                "input_channels": num_channels,
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

        self._validate_inputs(x)
        if self.architecture == "resnet":
            features = self.backbone(x)
            pooled = self.pool(features)
            return torch.flatten(pooled, 1)

        backbone_forward = getattr(self.backbone, "forward_features", None)
        if backbone_forward is None:
            msg = (
                "Configured backbone does not implement forward_features."
            )
            raise RuntimeError(msg)
        forward_fn: Callable[[torch.Tensor], torch.Tensor] = backbone_forward
        return forward_fn(x)

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
    """Dual-head shogi network built on selectable backbones."""

    def __init__(
        self,
        *,
        num_policy_classes: int = MOVE_LABELS_NUM,
        num_channels: int = FEATURES_NUM,
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
            num_channels=num_channels,
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
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy and value predictions for the given features."""

        features = self.forward_features(x)
        policy_logits = self.policy_head(features)
        value_logit = self.value_head(features)
        return policy_logits, value_logit
