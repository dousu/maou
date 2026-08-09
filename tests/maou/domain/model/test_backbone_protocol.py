"""FreezableBackbone プロトコルの適合テスト．

``preprocess_for_blocks`` と ``forward_features`` はプロトコルに
宣言されており，バックボーンを走らせる側は getattr プローブなしに
呼べる．3実装のいずれかが欠けたらここで落ちる．
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.protocol import FreezableBackbone
from maou.domain.model.resnet import BottleneckBlock, ResNet
from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)

_IN_CHANNELS = 8
_BOARD = 9


def _make_resnet(pooling: nn.Module | None = None) -> ResNet:
    return ResNet(
        block=BottleneckBlock,
        in_channels=_IN_CHANNELS,
        layers=[1, 1, 1, 1],
        strides=[1, 2, 2, 2],
        list_out_channels=[4, 8, 16, 32],
        pooling=pooling,
    )


def _make_mixer() -> ShogiMLPMixer:
    return ShogiMLPMixer(
        num_classes=None,
        num_channels=_IN_CHANNELS,
        num_tokens=_BOARD * _BOARD,
        depth=1,
    )


def _make_vit() -> VisionTransformer:
    return VisionTransformer(
        VisionTransformerConfig(
            input_channels=_IN_CHANNELS,
            board_size=_BOARD,
            use_head=False,
            num_layers=1,
            num_heads=2,
            embed_dim=16,
        )
    )


@pytest.fixture(
    params=["resnet", "mlp-mixer", "vit"],
    ids=["resnet", "mlp-mixer", "vit"],
)
def backbone(request: pytest.FixtureRequest) -> nn.Module:
    if request.param == "resnet":
        return _make_resnet()
    if request.param == "mlp-mixer":
        return _make_mixer()
    return _make_vit()


def test_backbones_satisfy_freezable_backbone(
    backbone: nn.Module,
) -> None:
    """3実装とも FreezableBackbone として扱える．"""
    assert isinstance(backbone, FreezableBackbone)


def test_forward_features_returns_pooled_vector(
    backbone: nn.Module,
) -> None:
    """forward_features は (batch, dim) の特徴ベクトルを返す．

    ResNet だけ特徴マップを返すといった不揃いがあると，
    HeadlessNetwork 側で再びアーキテクチャ分岐が必要になる．
    """
    x = torch.zeros(
        (2, _IN_CHANNELS, _BOARD, _BOARD), dtype=torch.float32
    )

    protocol_backbone: FreezableBackbone = backbone  # type: ignore[assignment]
    features = protocol_backbone.forward_features(x)

    assert features.ndim == 2
    assert features.size(0) == 2


def test_preprocess_for_blocks_feeds_freezable_groups(
    backbone: nn.Module,
) -> None:
    """preprocess_for_blocks の出力を先頭グループが受け取れる．"""
    x = torch.zeros(
        (2, _IN_CHANNELS, _BOARD, _BOARD), dtype=torch.float32
    )

    protocol_backbone: FreezableBackbone = backbone  # type: ignore[assignment]
    preprocessed = protocol_backbone.preprocess_for_blocks(x)
    first_group = protocol_backbone.get_freezable_groups()[0]

    assert first_group(preprocessed) is not None


def test_resnet_forward_features_uses_injected_pooling() -> (
    None
):
    """差し替えたプーリングが forward_features で実際に使われる．

    プーリングの持ち主を HeadlessNetwork から ResNet へ移したとき
    に，``pooling`` 引数が黙って無視されるようになるのを防ぐ．
    """
    calls: list[torch.Tensor] = []

    class RecordingPool(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return nn.functional.adaptive_avg_pool2d(x, (1, 1))

    backbone = _make_resnet(pooling=RecordingPool())
    x = torch.zeros(
        (2, _IN_CHANNELS, _BOARD, _BOARD), dtype=torch.float32
    )

    features = backbone.forward_features(x)

    assert len(calls) == 1
    assert features.shape == (2, 32 * BottleneckBlock.expansion)
