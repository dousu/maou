"""Tests for backbone get_freezable_groups() method."""

from torch import nn

from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.protocol import FreezableBackbone
from maou.domain.model.resnet import (
    ResidualBlock,
    ResNet,
)
from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


def _create_resnet() -> ResNet:
    """Create a small ResNet for testing."""
    return ResNet(
        block=ResidualBlock,
        in_channels=32,
        layers=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
        list_out_channels=[32, 64, 128, 256],
    )


def _create_mlp_mixer(depth: int = 4) -> ShogiMLPMixer:
    """Create a small MLP-Mixer for testing."""
    return ShogiMLPMixer(
        num_channels=32,
        num_tokens=81,
        depth=depth,
        num_classes=None,
    )


def _create_vit(num_layers: int = 3) -> VisionTransformer:
    """Create a small VisionTransformer for testing."""
    config = VisionTransformerConfig(
        input_channels=32,
        board_size=9,
        num_layers=num_layers,
        use_head=False,
    )
    return VisionTransformer(config)


# --- ResNet ---


def test_resnet_implements_freezable_backbone_protocol() -> (
    None
):
    resnet = _create_resnet()

    assert isinstance(resnet, FreezableBackbone)


def test_resnet_returns_four_groups() -> None:
    resnet = _create_resnet()

    groups = resnet.get_freezable_groups()

    assert len(groups) == 4
    assert groups[0] is resnet.layer1
    assert groups[1] is resnet.layer2
    assert groups[2] is resnet.layer3
    assert groups[3] is resnet.layer4


def test_resnet_groups_are_nn_sequential() -> None:
    resnet = _create_resnet()

    groups = resnet.get_freezable_groups()

    for group in groups:
        assert isinstance(group, nn.Sequential)


# --- MLP-Mixer ---


def test_mlp_mixer_implements_freezable_backbone_protocol() -> (
    None
):
    mixer = _create_mlp_mixer()

    assert isinstance(mixer, FreezableBackbone)


def test_mlp_mixer_returns_correct_number_of_groups() -> None:
    depth = 4
    mixer = _create_mlp_mixer(depth=depth)

    groups = mixer.get_freezable_groups()

    assert len(groups) == depth


def test_mlp_mixer_groups_match_blocks() -> None:
    mixer = _create_mlp_mixer(depth=4)

    groups = mixer.get_freezable_groups()

    for i, group in enumerate(groups):
        assert group is mixer.blocks[i]


# --- VisionTransformer ---


def test_vit_implements_freezable_backbone_protocol() -> None:
    vit = _create_vit()

    assert isinstance(vit, FreezableBackbone)


def test_vit_returns_correct_number_of_groups() -> None:
    num_layers = 3
    vit = _create_vit(num_layers=num_layers)

    groups = vit.get_freezable_groups()

    assert len(groups) == num_layers


def test_vit_groups_match_encoder_blocks() -> None:
    vit = _create_vit(num_layers=3)

    groups = vit.get_freezable_groups()

    for i, group in enumerate(groups):
        assert group is vit.encoder[i]


# --- Cross-architecture ordering ---


def test_resnet_groups_ordered_low_to_high() -> None:
    """Verify the first group corresponds to the lowest (earliest) layer."""
    resnet = _create_resnet()

    groups = resnet.get_freezable_groups()

    assert groups[0] is resnet.layer1


def test_mlp_mixer_groups_ordered_low_to_high() -> None:
    """Verify the first group corresponds to the lowest (earliest) block."""
    mixer = _create_mlp_mixer(depth=4)

    groups = mixer.get_freezable_groups()

    assert groups[0] is mixer.blocks[0]


def test_vit_groups_ordered_low_to_high() -> None:
    """Verify the first group corresponds to the lowest (earliest) encoder block."""
    vit = _create_vit(num_layers=3)

    groups = vit.get_freezable_groups()

    assert groups[0] is vit.encoder[0]
