import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchinfo import summary

from maou.app.learning.dataset import KifDataset
from maou.app.learning.feature import FEATURES_NUM
from maou.app.learning.network import Network
from maou.app.learning.transform import Transform
from maou.domain.loss.loss_fn import GCELoss
from maou.domain.network.resnet import ResidualBlock


class Learning:
    """Learning.
    機械学習のトレーニングを行うユースケースを表現する．
    損失関数の設計では，
    レーティングにより重みづけしたり，
    ラベルノイズに強い損失関数を選ぶといった工夫が必要かも．
    """

    logger: logging.Logger = logging.getLogger(__name__)
    device: torch.device
    checkpoint_dir: Optional[Path]
    resume_from: Optional[Path]
    model: torch.nn.Module

    def __init__(self, gpu: Optional[int] = None):
        if gpu is not None:
            self.device = torch.device(gpu)
            self.pin_memory = True
        else:
            self.device = torch.device("cpu")
            self.pin_memory = False

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        input_paths: list[Path]
        compilation: Optional[bool] = None
        test_ratio: Optional[float] = None
        epoch: Optional[int] = None
        batch_size: Optional[int] = None
        dataloader_workers: Optional[int] = None
        gce_parameter: Optional[float] = None
        policy_loss_ratio: Optional[float] = None
        value_loss_ratio: Optional[float] = None
        learning_ratio: Optional[float] = None
        momentum: Optional[float] = None
        checkpoint_dir: Optional[Path] = None
        resume_from: Optional[Path] = None
        log_dir: Optional[Path] = None
        model_dir: Optional[Path] = None

    def learn(self, option: LearningOption) -> None:
        """機械学習を行う."""

        # モデルをコンパイルするかどうか (デフォルトTrue)
        if option.compilation is not None:
            compilation = option.compilation
        else:
            compilation = True

        # テスト割合設定 (デフォルト0.25)
        if option.test_ratio is not None:
            test_ratio = option.test_ratio
        else:
            test_ratio = 0.25

        # エポック設定 (デフォルト10)
        if option.epoch is not None:
            self.epoch = option.epoch
        else:
            self.epoch = 10

        # バッチサイズ設定 (デフォルト1000)
        if option.batch_size is not None:
            batch_size = option.batch_size
        else:
            batch_size = 1000

        # DataLoaderのワーカー数設定 (デフォルト2)
        if option.dataloader_workers is not None:
            dataloader_workers = option.dataloader_workers
        else:
            dataloader_workers = 2

        # 損失関数のパラメータ設定 (デフォルト0.7)
        if option.gce_parameter is not None:
            gce_parameter = option.gce_parameter
        else:
            gce_parameter = 0.7

        # policy損失関数のパラメータ設定 (デフォルト1)
        if option.policy_loss_ratio is not None:
            self.policy_loss_ratio = option.policy_loss_ratio
        else:
            self.policy_loss_ratio = 1

        # value損失関数のパラメータ設定 (デフォルト1)
        if option.value_loss_ratio is not None:
            self.value_loss_ratio = option.value_loss_ratio
        else:
            self.value_loss_ratio = 1

        # オプティマイザのパラメータ設定 (デフォルト0.01)
        if option.learning_ratio is not None:
            learning_ratio = option.learning_ratio
        else:
            learning_ratio = 0.01

        # オプティマイザのパラメータ設定momemtum (デフォルト0.9)
        if option.momentum is not None:
            momentum = option.momentum
        else:
            momentum = 0.9

        # チェックポイントの書き込み先設定 (デフォルトNone)
        if option.checkpoint_dir is not None:
            self.checkpoint_dir = option.checkpoint_dir
        else:
            self.checkpoint_dir = None

        # 学習開始に利用するチェックポイントファイル設定 (デフォルトNone)
        if option.resume_from is not None:
            self.resume_from = option.resume_from
        else:
            self.resume_from = None

        # SummaryWriterの書き込み先設定 (デフォルト./logs)
        if option.log_dir is not None:
            self.log_dir = option.log_dir
        else:
            self.log_dir = Path("./logs")

        # 学習後のモデルの書き込み先設定 (デフォルト./models)
        if option.model_dir is not None:
            self.model_dir = option.model_dir
        else:
            self.model_dir = Path("./models")

        # 指定されたディレクトリが存在しない場合は自動で作成する
        if self.checkpoint_dir is not None and not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()
        if self.log_dir is not None and not self.log_dir.exists():
            self.log_dir.mkdir()
        if self.model_dir is not None and not self.model_dir.exists():
            self.model_dir.mkdir()

        self.logger.info("start learning")
        input_train: list[Path]
        input_test: list[Path]
        input_train, input_test = train_test_split(
            option.input_paths, test_size=test_ratio
        )

        # datasetに特徴量と正解ラベルを作成する変換を登録する
        feature = Transform(pin_memory=self.pin_memory)
        dataset_train: Dataset = KifDataset(input_train, feature)
        dataset_test: Dataset = KifDataset(input_test, feature)

        # dataloader
        # 前処理は軽めのはずなのでワーカー数は一旦固定にしてみる
        self.training_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            pin_memory=self.pin_memory,
        )
        self.validation_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=self.pin_memory,
        )

        # モデル定義
        # チャンネル数はてきとうに256まで増やしてる
        # strideは勘で設定している (2にして計算量減らす)
        # チャンネル数も適当にちょっとずつあげてみた
        # あんまりチャンネル数多いと計算量多くなりすぎるので少しだけ
        channels = 256
        model = Network(
            ResidualBlock,
            FEATURES_NUM,
            [2, 2, 2, 2],
            [1, 2, 2, 2],
            [
                FEATURES_NUM + int((channels - FEATURES_NUM) / 15),
                FEATURES_NUM + int((channels - FEATURES_NUM) / 15 * 3),
                FEATURES_NUM + int((channels - FEATURES_NUM) / 15 * 7),
                channels,
            ],
        )
        self.logger.info(
            str(summary(model, input_size=(batch_size, FEATURES_NUM, 9, 9)))
        )
        if compilation:
            compiled_model = torch.compile(model)
            self.model = compiled_model  # type: ignore
        else:
            self.model = model
        self.model.to(self.device)
        # ヘッドが二つあるので2つ損失関数を設定する
        # 損失を単純に加算するのかどうかは議論の余地がある
        # policyの損失関数は合法手以外を無視して損失を計算しない設計も考えられる
        self.loss_fn_policy = GCELoss(q=gce_parameter)
        self.loss_fn_value = torch.nn.BCEWithLogitsLoss()
        # SGD+Momentum
        # weight_decayは特に根拠なし
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_ratio,
            momentum=momentum,
            weight_decay=0.0001,
        )
        self.__train()

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter) -> float:
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, (labels_policy, labels_value) = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs_policy, outputs_value = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.policy_loss_ratio * self.loss_fn_policy(
                outputs_policy, labels_policy
            ) + self.value_loss_ratio * self.loss_fn_value(outputs_value, labels_value)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                self.logger.info("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def __train(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(self.log_dir / "training_log_{}".format(timestamp))
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0

        # resume from checkpoint
        if self.resume_from is not None:
            # 本当はdictにいろいろ保存することというか，
            # なんでも読み込めること自体があんまりよくない
            # https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
            checkpoint: dict = torch.load(self.resume_from)
            # チェックポイントはなんらかの障害で意図せず学習が止まることの保険
            # そのため，epoch_numberは引継ぎする
            epoch_number = checkpoint["epoch_number"] + 1
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["opt_state"])

        for epoch in range(epoch_number, EPOCHS):
            self.logger.info("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, (vlabels_policy, vlabels_value) = vdata
                    voutputs_policy, voutputs_value = self.model(vinputs)
                    vloss = self.policy_loss_ratio * self.loss_fn_policy(
                        voutputs_policy, vlabels_policy
                    ) + self.value_loss_ratio * self.loss_fn_value(
                        voutputs_value, vlabels_value
                    )
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            self.logger.info("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = self.model_dir / "model_{}_{}.pt".format(
                    timestamp, epoch_number
                )
                torch.save(self.model.state_dict(), model_path)

            # checkpoint
            if self.checkpoint_dir is not None:
                checkpoint_path = self.checkpoint_dir / "model_{}_{}.checkpoint".format(
                    timestamp, epoch_number
                )
                torch.save(
                    {
                        # epoch_numberはepochより1進んでいるのでそのままいれる
                        "epoch_number": epoch_number,
                        "model_state": self.model.state_dict(),
                        "opt_state": self.optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

            epoch_number += 1
