"""Tests for HeadlessNetwork.freeze_except_last_n() method."""

import pytest

from maou.app.learning.network import HeadlessNetwork
from maou.domain.model.resnet import ResidualBlock


def _create_network(architecture: str) -> HeadlessNetwork:
    """Create a HeadlessNetwork with small config for testing."""
    if architecture == "resnet":
        return HeadlessNetwork(
            architecture="resnet",
            embedding_dim=32,
            block=ResidualBlock,
            layers=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            out_channels=(32, 64, 128, 256),
        )
    elif architecture == "mlp-mixer":
        return HeadlessNetwork(
            architecture="mlp-mixer",
            embedding_dim=32,
            architecture_config={"depth": 4, "embed_dim": 64},
        )
    elif architecture == "vit":
        return HeadlessNetwork(
            architecture="vit",
            embedding_dim=32,
            architecture_config={
                "num_layers": 3,
                "embed_dim": 64,
            },
        )
    raise ValueError(f"Unknown architecture: {architecture}")


# --- Full freeze (n=0) ---


@pytest.mark.parametrize(
    "architecture", ["resnet", "mlp-mixer", "vit"]
)
def test_full_freeze_sets_all_params_to_not_trainable(
    architecture: str,
) -> None:
    """n=0 should freeze ALL parameters (embedding, pool, backbone)."""
    net = _create_network(architecture)

    net.freeze_except_last_n(0)

    for name, param in net.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} should be frozen with n=0"
        )


# --- n=total_groups (no backbone freeze) ---


@pytest.mark.parametrize(
    "architecture", ["resnet", "mlp-mixer", "vit"]
)
def test_unfreeze_all_groups_keeps_group_params_trainable(
    architecture: str,
) -> None:
    """n=total_groups should keep all group params trainable, but embedding/pool frozen."""
    net = _create_network(architecture)
    groups = net.backbone.get_freezable_groups()
    total_groups = len(groups)

    net.freeze_except_last_n(total_groups)

    # All group params should be trainable
    for group in groups:
        for name, param in group.named_parameters():
            assert param.requires_grad, (
                f"Group param {name} should be trainable with n={total_groups}"
            )

    # Embedding and pool should still be frozen
    for param in net.embedding.parameters():
        assert not param.requires_grad, (
            "Embedding should be frozen regardless of n"
        )
    for param in net.pool.parameters():
        assert not param.requires_grad, (
            "Pool should be frozen regardless of n"
        )


# --- Partial freeze (ResNet-specific) ---


def test_resnet_partial_freeze_last_2_groups() -> None:
    """n=2 should freeze layer1/2 and keep layer3/4 trainable."""
    net = _create_network("resnet")

    net.freeze_except_last_n(2)

    # layer1 and layer2 should be frozen
    for param in net.backbone.layer1.parameters():
        assert not param.requires_grad, (
            "layer1 should be frozen with n=2"
        )
    for param in net.backbone.layer2.parameters():
        assert not param.requires_grad, (
            "layer2 should be frozen with n=2"
        )

    # layer3 and layer4 should be trainable
    for param in net.backbone.layer3.parameters():
        assert param.requires_grad, (
            "layer3 should be trainable with n=2"
        )
    for param in net.backbone.layer4.parameters():
        assert param.requires_grad, (
            "layer4 should be trainable with n=2"
        )

    # Embedding and pool should be frozen
    for param in net.embedding.parameters():
        assert not param.requires_grad
    for param in net.pool.parameters():
        assert not param.requires_grad


# --- Clamp test ---


@pytest.mark.parametrize(
    "architecture", ["resnet", "mlp-mixer", "vit"]
)
def test_clamp_n_exceeding_total_groups(
    architecture: str,
) -> None:
    """n=999 should work without exception, same effect as n=total_groups."""
    net = _create_network(architecture)
    groups = net.backbone.get_freezable_groups()

    frozen_count = net.freeze_except_last_n(999)

    # All group params should be trainable (clamped to total)
    for group in groups:
        for param in group.parameters():
            assert param.requires_grad

    # Should still return a valid frozen count
    assert frozen_count > 0


# --- Non-group params always frozen (MLP-Mixer) ---


@pytest.mark.parametrize("n", [0, 2, 4, 999])
def test_mlp_mixer_non_group_params_always_frozen(
    n: int,
) -> None:
    """MLP-Mixer's input_norm, embedding, and norm should always be frozen."""
    net = _create_network("mlp-mixer")

    net.freeze_except_last_n(n)

    for param in net.backbone.input_norm.parameters():
        assert not param.requires_grad, (
            f"backbone.input_norm should be frozen with n={n}"
        )
    for param in net.backbone.embedding.parameters():
        assert not param.requires_grad, (
            f"backbone.embedding should be frozen with n={n}"
        )
    for param in net.backbone.norm.parameters():
        assert not param.requires_grad, (
            f"backbone.norm should be frozen with n={n}"
        )


# --- Non-group params always frozen (ViT) ---


@pytest.mark.parametrize("n", [0, 2, 3, 999])
def test_vit_non_group_params_always_frozen(n: int) -> None:
    """ViT's token_projection, positional_embedding, and norm should always be frozen."""
    net = _create_network("vit")

    net.freeze_except_last_n(n)

    for param in net.backbone.token_projection.parameters():
        assert not param.requires_grad, (
            f"backbone.token_projection should be frozen with n={n}"
        )
    assert (
        not net.backbone.positional_embedding.requires_grad
    ), (
        f"backbone.positional_embedding should be frozen with n={n}"
    )
    for param in net.backbone.norm.parameters():
        assert not param.requires_grad, (
            f"backbone.norm should be frozen with n={n}"
        )


# --- Frozen count returned ---


@pytest.mark.parametrize(
    "architecture", ["resnet", "mlp-mixer", "vit"]
)
def test_frozen_count_returned_is_positive(
    architecture: str,
) -> None:
    """freeze_except_last_n should return a positive count of frozen params."""
    net = _create_network(architecture)

    frozen_count = net.freeze_except_last_n(0)

    assert frozen_count > 0


@pytest.mark.parametrize(
    "architecture", ["resnet", "mlp-mixer", "vit"]
)
def test_full_freeze_freezes_more_than_partial(
    architecture: str,
) -> None:
    """Full freeze (n=0) should freeze more params than partial freeze (n=1)."""
    net_full = _create_network(architecture)
    net_partial = _create_network(architecture)

    frozen_full = net_full.freeze_except_last_n(0)
    frozen_partial = net_partial.freeze_except_last_n(1)

    assert frozen_full > frozen_partial
