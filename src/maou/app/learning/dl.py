import abc
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchinfo import summary
from tqdm.auto import tqdm

from maou.app.learning.dataset import DataSource
from maou.app.learning.setup import TrainingSetup
from maou.app.pre_process.feature import FEATURES_NUM


class CloudStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def upload_from_local(self, *, local_path: Path, cloud_path: str) -> None:
        pass

    @abc.abstractmethod
    def upload_folder_from_local(
        self,
        *,
        local_folder: Path,
        cloud_folder: str,
        extensions: Optional[list[str]] = None,
    ) -> None:
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
        pin_memory: bool
        enable_prefetch: bool
        prefetch_factor: int
        gce_parameter: float
        policy_loss_ratio: float
        value_loss_ratio: float
        learning_ratio: float
        momentum: float
        resume_from: Optional[Path] = None
        start_epoch: int = 0
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
            torch.set_float32_matmul_precision("high")
        else:
            self.logger.info("Use CPU")
            self.device = torch.device("cpu")

        # マルチプロセシング開始方法の設定（WindowsやCUDA使用時の安定化）
        # デバイス設定後に実行してCUDA使用判定を正確に行う
        self._setup_multiprocessing()

        self.__cloud_storage = cloud_storage

    def learn(self, option: LearningOption) -> Dict[str, str]:
        """機械学習を行う."""
        self.logger.info("start learning")
        torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
        learning_result: Dict[str, str] = {}

        # 入力とテスト用のデータソース取得
        input_datasource, test_datasource = option.datasource.train_test_split(
            test_ratio=option.test_ratio
        )

        # Setup training components using shared setup module
        device_config, dataloaders, model_components = (
            TrainingSetup.setup_training_components(
                training_datasource=input_datasource,
                validation_datasource=test_datasource,
                datasource_type=option.datasource_type,
                gpu=None,  # Device already set in __init__
                batch_size=option.batch_size,
                dataloader_workers=option.dataloader_workers,
                pin_memory=option.pin_memory,
                enable_prefetch=option.enable_prefetch,
                prefetch_factor=option.prefetch_factor,
                gce_parameter=option.gce_parameter,
                learning_ratio=option.learning_ratio,
                momentum=option.momentum,
            )
        )

        self.training_loader, self.validation_loader = dataloaders

        # Dataset information already logged in TrainingSetup

        # Get model components from setup (model already moved to self.device in setup)
        model = model_components.model

        # Move model to the correct device if needed (setup uses its own device
        # detection)
        model.to(self.device)

        self.logger.info(
            str(summary(model, input_size=(option.batch_size, FEATURES_NUM, 9, 9)))
        )

        if option.compilation:
            compiled_model = torch.compile(model)
            self.model = compiled_model  # type: ignore
            self.logger.info("Finished model compilation")
        else:
            self.model = model

        # Use loss functions and optimizer from setup
        self.loss_fn_policy = model_components.loss_fn_policy
        self.loss_fn_value = model_components.loss_fn_value

        # Create new optimizer with the same settings but for the final model
        self.optimizer = optim.SGD(
            self.model.parameters(),
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
        self.start_epoch = option.start_epoch
        self.__train()

        learning_result["Data Samples"] = (
            f"Training: {len(self.training_loader.dataset)}, "  # type: ignore
            f"Test: {len(self.validation_loader.dataset)}"  # type: ignore
        )
        learning_result["Option"] = str(option)
        learning_result["Result"] = "Finish"

        return learning_result

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter) -> float:
        running_loss = 0.0
        last_loss = 0.0

        # 記録するiteration数
        # 1/10で10回は記録する設定，最低でも1にする
        record_num = max(1, len(self.training_loader) // 10)

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(self.training_loader)):
            # Every data instance is an input + label pair
            inputs, (labels_policy, labels_value, legal_move_mask) = data

            # GPU転送（DataLoaderのpin_memoryと非同期転送を活用）
            inputs = inputs.to(self.device, non_blocking=True)
            labels_policy = labels_policy.to(self.device, non_blocking=True)
            labels_value = labels_value.to(self.device, non_blocking=True)
            legal_move_mask = legal_move_mask.to(self.device, non_blocking=True)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs_policy, outputs_value = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.policy_loss_ratio * self.loss_fn_policy(
                outputs_policy, labels_policy, legal_move_mask
            ) + self.value_loss_ratio * self.loss_fn_value(outputs_value, labels_value)
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
        summary_writer_log_dir = self.log_dir / "training_log_{}".format(timestamp)
        writer = SummaryWriter(summary_writer_log_dir)
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0

        # resume from checkpoint
        if self.resume_from is not None:
            self.model.load_state_dict(torch.load(self.resume_from))

        # start epoch設定
        epoch_number = self.start_epoch

        for _ in range(epoch_number, EPOCHS):
            self.logger.info("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, writer)

            # 学習ごとに各層のパラメータを記録
            for name, param in self.model.named_parameters():
                try:
                    param_np = param.detach().cpu().numpy()
                    writer.add_histogram(
                        f"parameters/{name}",
                        param_np,
                        epoch_number + 1,
                    )
                    if param.grad is not None:
                        grad_np = param.grad.detach().cpu().numpy()
                        writer.add_histogram(
                            f"gradients/{name}", grad_np, epoch_number + 1
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to log histogram for {name}: {e}")

            running_vloss = 0.0

            test_accuracy_policy = 0.0
            test_accuracy_value = 0.0

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in tqdm(enumerate(self.validation_loader)):
                    vinputs, (vlabels_policy, vlabels_value, vlegal_move_mask) = vdata

                    # GPU転送（DataLoaderのpin_memoryと非同期転送を活用）
                    vinputs = vinputs.to(self.device, non_blocking=True)
                    vlabels_policy = vlabels_policy.to(self.device, non_blocking=True)
                    vlabels_value = vlabels_value.to(self.device, non_blocking=True)
                    vlegal_move_mask = vlegal_move_mask.to(
                        self.device, non_blocking=True
                    )

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
                    timestamp, epoch_number + 1
                )
                self.logger.info("Saving model to {}".format(model_path))
                torch.save(self.model.state_dict(), model_path)
                if self.__cloud_storage is not None:
                    self.logger.info("Uploading model to cloud storage")
                    self.__cloud_storage.upload_from_local(
                        local_path=model_path, cloud_path=str(model_path)
                    )

            # SummaryWriterのイベントをGCSに保存する
            if self.__cloud_storage is not None:
                self.logger.info("Uploading tensorboard logs to cloud storage")
                self.__cloud_storage.upload_folder_from_local(
                    local_folder=summary_writer_log_dir,
                    cloud_folder="tensorboard",
                )

            epoch_number += 1

        writer.close()

    # 方策の正解率
    def __policy_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> float:
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    # 価値の正解率
    def __value_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> float:
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)

    def _setup_multiprocessing(self) -> None:
        """
        マルチプロセシング開始方法を設定する．
        プラットフォームとCUDA使用状況に応じて最適な方法を選択し，
        DataLoaderのマルチプロセシング安定性を向上させる．
        """
        import platform

        import torch.multiprocessing as mp

        try:
            # 現在の開始方法を取得
            current_method = mp.get_start_method(allow_none=True)

            # プラットフォーム別の推奨設定
            if platform.system() == "Windows":
                # Windowsでは常にspawnを使用
                if current_method != "spawn":
                    mp.set_start_method("spawn", force=True)
                    self.logger.info(
                        "Set multiprocessing start method to 'spawn' for Windows"
                    )
            elif (
                torch.cuda.is_available()
                and hasattr(self, "device")
                and self.device.type == "cuda"
            ):
                # CUDA使用時はspawnが安全
                if current_method != "spawn":
                    try:
                        mp.set_start_method("spawn", force=True)
                        self.logger.info(
                            "Set multiprocessing start method to 'spawn' for CUDA"
                        )
                    except RuntimeError as e:
                        # 既に設定済みの場合は警告のみ
                        self.logger.warning(
                            f"Could not set multiprocessing method: {e}"
                        )
            else:
                # Linux/macOSでCPU使用時はforkのままでも問題なし
                if current_method is None:
                    self.logger.info(
                        f"Using default multiprocessing start method: "
                        f"{mp.get_start_method()}"
                    )
                else:
                    self.logger.info(
                        f"Using current multiprocessing start method: {current_method}"
                    )

        except Exception as e:
            # マルチプロセシング設定に失敗した場合は警告のみ
            self.logger.warning(f"Failed to configure multiprocessing: {e}")
            self.logger.info("Continuing with default multiprocessing settings")
