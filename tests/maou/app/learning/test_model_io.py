"""Tests for model I/O utilities."""

import torch

from maou.app.learning.model_io import ModelIO
from maou.app.learning.setup import ModelFactory


def test_format_parameter_count_millions() -> None:
    """Test parameter count formatting for millions."""
    assert ModelIO.format_parameter_count(1_234_567) == "1.2m"
    assert ModelIO.format_parameter_count(1_000_000) == "1m"
    assert ModelIO.format_parameter_count(2_500_000) == "2.5m"


def test_format_parameter_count_thousands() -> None:
    """Test parameter count formatting for thousands."""
    assert ModelIO.format_parameter_count(45_000) == "45k"
    assert ModelIO.format_parameter_count(1_000) == "1k"
    assert ModelIO.format_parameter_count(1_234) == "1.2k"


def test_format_parameter_count_small() -> None:
    """Test parameter count formatting for small numbers."""
    assert ModelIO.format_parameter_count(123) == "123"
    assert ModelIO.format_parameter_count(999) == "999"
    assert ModelIO.format_parameter_count(1) == "1"


def test_generate_model_tag() -> None:
    """Test model tag generation."""
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )

    tag = ModelIO.generate_model_tag(model, "resnet")

    # Tag should start with architecture name
    assert tag.startswith("resnet-")

    # Tag should end with parameter count (format: XXm or XXk)
    param_part = tag.split("-")[1]
    assert param_part.endswith("m") or param_part.endswith("k")


def test_generate_model_tag_different_architectures() -> None:
    """Test model tag generation for different architectures."""
    device = torch.device("cpu")

    # ResNet
    resnet_model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )
    resnet_tag = ModelIO.generate_model_tag(
        resnet_model, "resnet"
    )
    assert resnet_tag.startswith("resnet-")

    # MLP-Mixer
    mlp_model = ModelFactory.create_shogi_model(
        device, architecture="mlp-mixer"
    )
    mlp_tag = ModelIO.generate_model_tag(mlp_model, "mlp-mixer")
    assert mlp_tag.startswith("mlp-mixer-")

    # ViT
    vit_model = ModelFactory.create_shogi_model(
        device, architecture="vit"
    )
    vit_tag = ModelIO.generate_model_tag(vit_model, "vit")
    assert vit_tag.startswith("vit-")
