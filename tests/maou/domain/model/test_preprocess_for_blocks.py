"""Tests for preprocess_for_blocks() method on each backbone."""

import torch

from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.resnet import BottleneckBlock, ResNet
from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


class TestResNetPreprocess:
    """ResNet の preprocess_for_blocks テスト．"""

    def test_identity(self) -> None:
        """ResNet の preprocess_for_blocks が入力をそのまま返すことを検証する．"""
        model = ResNet(
            block=BottleneckBlock,
            in_channels=64,
            layers=[2, 2, 2, 2],
            strides=[1, 2, 2, 2],
            list_out_channels=[16, 32, 64, 64],
        )

        x = torch.randn(2, 64, 9, 9)
        out = model.preprocess_for_blocks(x)

        assert torch.equal(x, out)


class TestMixerPreprocess:
    """ShogiMLPMixer の preprocess_for_blocks テスト．"""

    def test_output_shape(self) -> None:
        """出力が (B, 81, embed_dim) であることを検証する．"""
        model = ShogiMLPMixer(
            num_classes=None,
            num_channels=64,
            embed_dim=128,
            depth=4,
        )

        x = torch.randn(2, 64, 9, 9)
        out = model.preprocess_for_blocks(x)

        assert out.shape == (2, 81, 128)

    def test_eval_mode_no_dropout(self) -> None:
        """eval モードで dropout が無効化されることを検証する．"""
        model = ShogiMLPMixer(
            num_classes=None,
            num_channels=64,
            embed_dim=128,
            depth=4,
            dropout_rate=0.5,
        )
        model.eval()

        x = torch.randn(2, 64, 9, 9)
        out1 = model.preprocess_for_blocks(x)
        out2 = model.preprocess_for_blocks(x)

        assert torch.equal(out1, out2)


class TestViTPreprocess:
    """VisionTransformer の preprocess_for_blocks テスト．"""

    def test_output_shape(self) -> None:
        """出力が (B, 81, embed_dim) であることを検証する．"""
        config = VisionTransformerConfig(
            input_channels=64,
            board_size=9,
            embed_dim=128,
            num_heads=4,
            num_layers=4,
        )
        model = VisionTransformer(config)

        x = torch.randn(2, 64, 9, 9)
        out = model.preprocess_for_blocks(x)

        assert out.shape == (2, 81, 128)
