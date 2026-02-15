"""Tests for architecture_config parameter propagation in ModelFactory."""

import torch

from maou.app.learning.network import HeadlessNetwork, Network
from maou.app.learning.setup import ModelFactory


class TestModelFactoryArchitectureConfig:
    """ModelFactory.create_shogi_backbone/create_shogi_model with architecture_config."""

    device = torch.device("cpu")

    def test_resnet_backbone_ignores_architecture_config(
        self,
    ) -> None:
        """ResNetバックボーンはarchitecture_configを無視してもエラーにならない."""
        backbone = ModelFactory.create_shogi_backbone(
            self.device,
            architecture="resnet",
            architecture_config={"embed_dim": 256},
        )
        assert isinstance(backbone, HeadlessNetwork)

    def test_resnet_model_ignores_architecture_config(
        self,
    ) -> None:
        """ResNetモデルはarchitecture_configを無視してもエラーにならない."""
        model = ModelFactory.create_shogi_model(
            self.device,
            architecture="resnet",
            architecture_config={"embed_dim": 256},
        )
        assert isinstance(model, Network)

    def test_vit_backbone_with_architecture_config(
        self,
    ) -> None:
        """ViTバックボーンがarchitecture_configでカスタマイズされる."""
        backbone = ModelFactory.create_shogi_backbone(
            self.device,
            architecture="vit",
            architecture_config={
                "embed_dim": 256,
                "num_layers": 3,
            },
        )
        assert isinstance(backbone, HeadlessNetwork)
        assert backbone.embedding_dim == 256

    def test_vit_backbone_num_layers(self) -> None:
        """ViTバックボーンのnum_layersがarchitecture_configで指定可能."""
        backbone = ModelFactory.create_shogi_backbone(
            self.device,
            architecture="vit",
            architecture_config={"num_layers": 3},
        )
        groups = backbone.backbone.get_freezable_groups()
        assert len(groups) == 3

    def test_vit_model_with_architecture_config(self) -> None:
        """ViTモデルがarchitecture_configでカスタマイズされる."""
        model = ModelFactory.create_shogi_model(
            self.device,
            architecture="vit",
            architecture_config={
                "embed_dim": 256,
                "num_layers": 3,
                "num_heads": 4,
            },
        )
        assert isinstance(model, Network)

    def test_backbone_none_architecture_config(self) -> None:
        """architecture_config=Noneはデフォルト動作."""
        backbone = ModelFactory.create_shogi_backbone(
            self.device,
            architecture="resnet",
            architecture_config=None,
        )
        assert isinstance(backbone, HeadlessNetwork)

    def test_mlp_mixer_backbone_with_architecture_config(
        self,
    ) -> None:
        """MLP-Mixerバックボーンがarchitecture_configでカスタマイズされる."""
        backbone = ModelFactory.create_shogi_backbone(
            self.device,
            architecture="mlp-mixer",
            architecture_config={"depth": 4},
        )
        assert isinstance(backbone, HeadlessNetwork)
        groups = backbone.backbone.get_freezable_groups()
        assert len(groups) == 4
