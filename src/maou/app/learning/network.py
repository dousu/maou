import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.network.resnet import ResNet


class Network(nn.Module):
    """Deep Learning Network.
    コア部分はResNetで出力層で二つのヘッドがあり，
    PolicyとValueの2つの評価値を返す．
    このNetwork構成はかなり改善余地がある．
    基本的に将棋は最適なラベルがデータソースに振られていない問題が大きい．
    そのため，dropoutとかノイズに強くなる工夫をNetwork自体に入れた方がいい．
    将棋には合法手があり，それ以外は反則負けになる．
    ラベル出力時には合法手以外の手はすべて0にするような工夫があってもよさそう．
    これらのような改善を入れる前に意味のない出力ラベルがあるのでそこを改善するのが先．
    そうしないと，dead neurons前提になってdropout等がうまく機能しないように思える．
    """

    def __init__(
        self,
        block: type[nn.Module],
        in_channels: int,
        layers: list[int],
        strides: list[_size_2_t],
        list_out_channels: list[int],
    ):
        super(Network, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 入力層
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=list_out_channels[0],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(list_out_channels[0])

        # 中間層
        self.resnet = ResNet(
            block=block,
            in_channels=list_out_channels[0],
            layers=layers,
            strides=strides,
            list_out_channels=list_out_channels,
        )

        # 出力層
        # policyとvalueのネットワークはまるまる同じにしてヘッドの設定だけ変えて出しわける
        # policyとvalueにもうちょっと畳み込み層とかを挟んだ方がいい説はある
        # policy head
        self.policy_head = PolicyHead(list_out_channels[3], MOVE_LABELS_NUM)

        # value head
        self.value_head = ValueHead(list_out_channels[3])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.
        Policyの評価とValueの評価を返すのでtupleで2つ返している．
        """
        # 入力層
        x = self.conv1(x)
        x = self.relu(self.norm1(x))

        # 中間層 (ResNet)
        x = self.resnet(x)

        # 出力層
        # policy head
        policy = self.policy_head(x)

        # value head
        value = self.value_head(x)

        return policy, value


class PolicyHead(nn.Module):
    """PolicyHead.
    単にLinearだけ使ってこう言いうのでもいいかも．
    nn.Sequential(
        nn.Linear(in_channels, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    """

    def __init__(self, in_channels: int, num_classes: int):
        super(PolicyHead, self).__init__()
        # 128にしているのは勘
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """PolicyHead.
    単にLinearだけ使ってこう言いうのでもいいかも．
    nn.Sequential(
        nn.Linear(in_channels, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    """

    def __init__(self, in_channels: int):
        super(ValueHead, self).__init__()
        # 128にしているのは勘
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
