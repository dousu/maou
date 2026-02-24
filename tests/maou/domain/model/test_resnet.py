"""Tests for the ResNet building blocks and backbone."""

import torch
from torch import nn

from maou.domain.model.resnet import (
    BottleneckBlock,
    ResidualBlock,
    ResNet,
)


def test_residual_block_preserves_shape_without_downsample() -> (
    None
):
    block = ResidualBlock(in_channels=16, out_channels=16)
    x = torch.randn(2, 16, 8, 8)

    out = block(x)

    assert out.shape == x.shape


def test_residual_block_downsamples_when_configured() -> None:
    downsample = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(32),
    )
    block = ResidualBlock(
        in_channels=16,
        out_channels=32,
        stride=2,
        downsample=downsample,
    )
    x = torch.randn(1, 16, 8, 8)

    out = block(x)

    assert out.shape == (1, 32, 4, 4)


def test_bottleneck_block_expands_channels() -> None:
    downsample = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(256),
    )
    block = BottleneckBlock(
        in_channels=64,
        out_channels=256,
        stride=2,
        downsample=downsample,
    )
    x = torch.randn(1, 64, 16, 16)

    out = block(x)

    assert out.shape == (1, 256, 8, 8)


def test_resnet_forward_shape_basic_configuration() -> None:
    model = ResNet(
        block=ResidualBlock,
        in_channels=64,
        layers=[2, 2, 2, 2],
        strides=[1, 2, 2, 2],
        list_out_channels=[64, 128, 256, 512],
    )
    x = torch.randn(2, 64, 56, 56)

    out = model(x)

    assert out.shape == (2, 512, 7, 7)


def test_resnet_forward_shape_bottleneck_configuration() -> (
    None
):
    model = ResNet(
        block=BottleneckBlock,
        in_channels=64,
        layers=[3, 4, 6, 3],
        strides=[1, 2, 2, 2],
        list_out_channels=[64, 128, 256, 512],
    )
    x = torch.randn(1, 64, 56, 56)

    out = model(x)

    assert out.shape == (1, 2048, 7, 7)
