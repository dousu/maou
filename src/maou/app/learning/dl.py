import abc
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchinfo import summary
from tqdm.auto import tqdm

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.network import Network
from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.transform import Transform
from maou.domain.loss.loss_fn import MaskedGCELoss
from maou.domain.network.resnet import ResidualBlock


class CloudStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def upload_from_local(self, *, local_path: Path, cloud_path: str) -> None:
        pass


class LearningDataSource(DataSource):
    class DataSourceSpliter(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def train_test_split(self, test_ratio: float) -> tuple[DataSource, DataSource]:
            pass


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

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        datasource: LearningDataSource.DataSourceSpliter
        datasource_type: str
        compilation: bool
        test_ratio: float
        epoch: int
        batch_size: int
        dataloader_workers: int
        gce_parameter: float
        policy_loss_ratio: float
        value_loss_ratio: float
        learning_ratio: float
        momentum: float
        checkpoint_dir: Optional[Path] = None
        resume_from: Optional[Path] = None
        log_dir: Path
        model_dir: Path

    def __init__(
        self,
        *,
        gpu: Optional[str] = None,
        cloud_storage: Optional[CloudStorage] = None,
    ):
        if gpu is not None and gpu != "cpu":
            self.device = torch.device(gpu)
            self.logger.info(f"Use GPU {torch.cuda.get_device_name(self.device)}")
            self.pin_memory = False
            torch.set_float32_matmul_precision("high")
        else:
            self.logger.info("Use CPU")
            self.device = torch.device("cpu")
            self.pin_memory = False
        self.__cloud_storage = cloud_storage

    def learn(self, option: LearningOption) -> Dict[str, str]:
        """機械学習を行う."""
        self.logger.info("start learning")
        learning_result: Dict[str, str] = {}

        # 入力とテスト用のデータソース取得
        input_datasource, test_datasource = option.datasource.train_test_split(
            test_ratio=option.test_ratio
        )

        dataset_train: KifDataset
        dataset_test: KifDataset
        if option.datasource_type == "hcpe":
            # datasetに特徴量と正解ラベルを作成する変換を登録する
            feature = Transform()
            dataset_train = KifDataset(
                datasource=input_datasource,
                transform=feature,
                pin_memory=self.pin_memory,
                device=self.device,
            )
            dataset_test = KifDataset(
                datasource=test_datasource,
                transform=feature,
                pin_memory=self.pin_memory,
                device=self.device,
            )
        elif option.datasource_type == "preprocess":
            dataset_train = KifDataset(
                datasource=input_datasource,
                pin_memory=self.pin_memory,
                device=self.device,
            )
            dataset_test = KifDataset(
                datasource=test_datasource,
                pin_memory=self.pin_memory,
                device=self.device,
            )
        else:
            raise ValueError(f"Data source type `{option.datasource_type}` is invalid.")

        # dataloader
        # 前処理は軽めのはずなのでワーカー数は一旦固定にしてみる
        self.training_loader = DataLoader(
            dataset_train,
            batch_size=option.batch_size,
            shuffle=True,
            num_workers=option.dataloader_workers,
            pin_memory=self.pin_memory,
        )
        self.validation_loader = DataLoader(
            dataset_test,
            batch_size=option.batch_size,
            shuffle=False,
            num_workers=option.dataloader_workers,
            pin_memory=self.pin_memory,
        )
        self.logger.info(f"Train: {len(self.training_loader)} batches/epoch")
        self.logger.info(f"Test: {len(self.validation_loader)} batches/epoch")

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
            str(summary(model, input_size=(option.batch_size, FEATURES_NUM, 9, 9)))
        )
        if option.compilation:
            compiled_model = torch.compile(model)
            self.model = compiled_model  # type: ignore
            self.logger.info("Finished model compilation")
        else:
            self.model = model
        self.model.to(self.device)
        # ヘッドが二つあるので2つ損失関数を設定する
        # 損失を単純に加算するのかどうかは議論の余地がある
        # policyの損失関数は合法手以外を無視して損失を計算しない設計も考えられる
        self.loss_fn_policy = MaskedGCELoss(q=option.gce_parameter)
        self.loss_fn_value = torch.nn.BCEWithLogitsLoss()
        # SGD+Momentum
        # weight_decayは特に根拠なし
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=option.learning_ratio,
            momentum=option.momentum,
            weight_decay=0.0001,
        )
        self.policy_loss_ratio = option.policy_loss_ratio
        self.value_loss_ratio = option.value_loss_ratio
        self.log_dir = option.log_dir
        self.epoch = option.epoch
        self.model_dir = option.model_dir
        self.resume_from = option.resume_from
        self.checkpoint_dir = option.checkpoint_dir
        self.__train()

        learning_result["Data Samples"] = (
            f"Training: {len(dataset_train)}, Test: {len(dataset_test)}"
        )
        learning_result["Option"] = str(option)
        learning_result["Result"] = "Finish"

        return learning_result

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter) -> float:
        running_loss = 0.0
        last_loss = 0.0

        # 記録するiteration数
        # 1/10で10回は記録する設定
        record_num = len(self.training_loader) // 10

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(self.training_loader)):
            # Every data instance is an input + label pair
            inputs, (labels_policy, labels_value, legal_move_mask) = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs_policy, outputs_value = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.policy_loss_ratio * self.loss_fn_policy(
                outputs_policy, labels_policy, legal_move_mask
            ) + self.value_loss_ratio * self.loss_fn_value(outputs_value, labels_value)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % record_num == record_num - 1:
                last_loss = running_loss / record_num  # loss avg per batch
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

            test_accuracy_policy = 0.0
            test_accuracy_value = 0.0

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, (vlabels_policy, vlabels_value, vlegal_move_mask) = vdata
                    voutputs_policy, voutputs_value = self.model(vinputs)
                    vloss = self.policy_loss_ratio * self.loss_fn_policy(
                        voutputs_policy, vlabels_policy, vlegal_move_mask
                    ) + self.value_loss_ratio * self.loss_fn_value(
                        voutputs_value, vlabels_value
                    )
                    running_vloss += vloss

                    test_accuracy_policy += self.__policy_accuracy(
                        voutputs_policy, vlabels_policy
                    )
                    test_accuracy_value += self.__value_accuracy(
                        voutputs_value, vlabels_value
                    )

            avg_vloss = running_vloss / (i + 1)

            self.logger.info("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            avg_accuracy_policy = test_accuracy_policy / (i + 1)
            avg_accuracy_value = test_accuracy_value / (i + 1)
            self.logger.info(
                "ACCURACY policy {} value {}".format(
                    avg_accuracy_policy, avg_accuracy_value
                )
            )
            writer.add_scalars(
                "Accuracy",
                {"Policy": avg_accuracy_policy, "Value": avg_accuracy_value},
                epoch_number + 1,
            )

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
                if self.__cloud_storage is not None:
                    self.__cloud_storage.upload_from_local(
                        local_path=model_path, cloud_path=str(model_path)
                    )

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
                if self.__cloud_storage is not None:
                    self.__cloud_storage.upload_from_local(
                        local_path=checkpoint_path, cloud_path=str(checkpoint_path)
                    )

            epoch_number += 1

    # 方策の正解率
    def __policy_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> float:
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    # 価値の正解率
    def __value_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> float:
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)
