from typing import Optional

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class ResidualBlock(nn.Module):
    """残渣ブロック.
    pre-activationの方が制度がいいらしいが一旦はpost-activationで。
    あとWide Residual dropoutも入れた方がいいらしい。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        downsample: Optional[nn.Module] = None,
        kernel_size: _size_2_t = 3,
        padding: _size_2_t = 1,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ダウンサンプリングがない場合はNoneがはいる
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.downsample:
            identity = self.downsample(x)

        # Convolution
        out = self.conv1(x)

        # Batch Normalization
        out = self.bn1(out)

        # ReLU
        out = self.relu(out)

        # Convolution
        out = self.conv2(out)

        # Batch Normalization
        out = self.bn2(out)

        # Skip Connection
        out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual Network.
    ここではResNetのコア部分 (中間層)だけ定義する。
    入力層や出力層は各種タスクによって、
    必要な処理が違うのでapp層で定義することにした。
    例えば、画像を識別するようなタスクでは入力層で
    畳み込みやMaxPoolを使って計算量を減らすといいらしい。
    """

    def __init__(
        self,
        block: type[nn.Module],
        in_channels: int,
        layers: list[int],
        strides: list[_size_2_t],
        list_out_channels: list[int],
    ):
        super(ResNet, self).__init__()
        self.block_in_channels = in_channels

        # 各層を構築
        # stride=2で空間サイズ縮小
        self.layer1 = self._make_layer(
            block, list_out_channels[0], layers[0], stride=strides[0]
        )
        self.layer2 = self._make_layer(
            block, list_out_channels[1], layers[1], stride=strides[0]
        )
        self.layer3 = self._make_layer(
            block, list_out_channels[2], layers[2], stride=strides[0]
        )
        self.layer4 = self._make_layer(
            block, list_out_channels[3], layers[3], stride=strides[0]
        )

    def _make_layer(
        self,
        block: type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: _size_2_t = 1,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or self.block_in_channels != out_channels:
            # スキップ接続のためのダウンサンプルを作成
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.block_in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            block(
                in_channels=self.block_in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        )
        # 次のブロックの入力チャンネルを更新
        self.block_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
