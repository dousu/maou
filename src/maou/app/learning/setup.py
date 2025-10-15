"""
共通学習セットアップ機能モジュール．
training_benchmark.py と dl.py の重複コードを統一化．
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.network import Network
from maou.app.pre_process.transform import Transform
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.loss.loss_fn import GCEwithNegativePenaltyLoss
from maou.domain.network.resnet import BottleneckBlock


def default_worker_init_fn(worker_id: int) -> None:
    """
    デフォルトのワーカー初期化関数．
    """
    import random

    import numpy as np

    # 再現性のためのシード設定（ワーカーごとに異なるシードを使用）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@dataclass
class DeviceConfig:
    """デバイス設定の結果."""

    device: torch.device
    pin_memory: bool


@dataclass
class ModelComponents:
    """モデル関連コンポーネント."""

    model: Network
    loss_fn_policy: GCEwithNegativePenaltyLoss
    loss_fn_value: torch.nn.Module
    optimizer: optim.SGD


class DeviceSetup:
    """デバイス設定の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def setup_device(
        cls,
        gpu: Optional[str] = None,
        pin_memory: Optional[bool] = None,
    ) -> DeviceConfig:
        """GPU/CPUデバイスの設定."""
        if gpu is not None and gpu != "cpu":
            device = torch.device(gpu)
            cls.logger.info(
                f"Using GPU: {torch.cuda.get_device_name(device)}"
            )
            torch.set_float32_matmul_precision("high")
        else:
            device = torch.device("cpu")
            cls.logger.info("Using CPU")

        # Set pin_memory default based on device
        if pin_memory is None:
            pin_memory = device.type == "cuda"

        return DeviceConfig(
            device=device, pin_memory=pin_memory
        )


class DatasetFactory:
    """データセット作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def create_datasets(
        cls,
        training_datasource: DataSource,
        validation_datasource: DataSource,
        datasource_type: str,
    ) -> Tuple[KifDataset, KifDataset]:
        """学習・検証用データセットの作成."""

        # Validate datasource type
        if datasource_type not in ("hcpe", "preprocess"):
            raise ValueError(
                f"Data source type `{datasource_type}` is invalid."
            )

        # Create transform based on datasource type
        if datasource_type == "hcpe":
            transform = Transform()
        else:
            transform = None

        # Create base datasets
        dataset_train = KifDataset(
            datasource=training_datasource, transform=transform
        )
        dataset_validation = KifDataset(
            datasource=validation_datasource,
            transform=transform,
        )

        return dataset_train, dataset_validation


class DataLoaderFactory:
    """DataLoader作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def create_dataloaders(
        cls,
        dataset_train: KifDataset,
        dataset_validation: KifDataset,
        batch_size: int,
        dataloader_workers: int,
        pin_memory: bool,
        prefetch_factor: int = 2,
    ) -> Tuple[DataLoader, DataLoader]:
        """学習・検証用DataLoaderの作成."""

        # Worker initialization function
        worker_init_fn = (
            default_worker_init_fn
            if dataloader_workers > 0
            else None
        )

        # Training DataLoader
        training_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            persistent_workers=dataloader_workers > 0,
            prefetch_factor=prefetch_factor
            if dataloader_workers > 0
            else None,
            drop_last=True,
            timeout=120 if dataloader_workers > 0 else 0,
            worker_init_fn=worker_init_fn,
        )

        # Validation DataLoader
        validation_loader = DataLoader(
            dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            persistent_workers=dataloader_workers > 0,
            prefetch_factor=prefetch_factor
            if dataloader_workers > 0
            else None,
            drop_last=False,  # validationでは全データを使用
            timeout=120 if dataloader_workers > 0 else 0,
            worker_init_fn=worker_init_fn,
        )

        cls.logger.info(
            f"Training: {len(training_loader)} batches"
        )
        cls.logger.info(
            f"Validation: {len(validation_loader)} batches"
        )

        return training_loader, validation_loader


class ModelFactory:
    """モデル作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def create_shogi_model(
        cls, device: torch.device
    ) -> Network:
        """将棋特化のBottleneckBlockモデルを作成."""

        # 将棋特化の「広く浅い」BottleneckBlock構成
        # 各層のボトルネック幅（3x3 convolution層のチャンネル数）
        # expansion=4により実際の出力は4倍: [96, 192, 384, 576]
        bottleneck_width = [24, 48, 96, 144]

        model = Network(
            BottleneckBlock,  # 効率的なBottleneckアーキテクチャを使用
            FEATURES_NUM,  # 入力特徴量チャンネル数
            [
                2,
                2,
                2,
                1,
            ],  # 将棋特化: 広く浅い構成でパターン認識を重視
            [
                1,
                2,
                2,
                2,
            ],  # 各層のstride（2で特徴マップサイズ半減）
            bottleneck_width,  # 幅重視: 多様な戦術要素を並列学習
        )

        model.to(device)
        cls.logger.info(
            f"Created Shogi-optimized BottleneckBlock model ({str(device)})"
        )

        return model


class LossOptimizerFactory:
    """損失関数・オプティマイザ作成の共通化."""

    @classmethod
    def create_loss_functions(
        cls,
        gce_parameter: float = 0.1,
    ) -> Tuple[GCEwithNegativePenaltyLoss, torch.nn.Module]:
        """方策・価値用の損失関数ペアを作成."""
        loss_fn_policy = GCEwithNegativePenaltyLoss(
            q=gce_parameter
        )
        loss_fn_value = torch.nn.BCEWithLogitsLoss()
        return loss_fn_policy, loss_fn_value

    @classmethod
    def create_optimizer(
        cls,
        model: torch.nn.Module,
        learning_ratio: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
    ) -> optim.SGD:
        """SGDオプティマイザを作成."""
        return optim.SGD(
            model.parameters(),
            lr=learning_ratio,
            momentum=momentum,
            weight_decay=weight_decay,
        )


class TrainingSetup:
    """学習セットアップの統合クラス."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def setup_training_components(
        cls,
        training_datasource: DataSource,
        validation_datasource: DataSource,
        datasource_type: str,
        gpu: Optional[str] = None,
        compilation: bool = False,
        batch_size: int = 256,
        dataloader_workers: int = 4,
        pin_memory: Optional[bool] = None,
        prefetch_factor: int = 2,
        gce_parameter: float = 0.1,
        learning_ratio: float = 0.01,
        momentum: float = 0.9,
    ) -> Tuple[
        DeviceConfig,
        Tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """学習に必要な全コンポーネントをセットアップ."""

        cls.logger.info("Setting up training components")

        # Torch config
        torch.autograd.set_detect_anomaly(
            mode=True, check_nan=True
        )

        # Device setup
        device_config = DeviceSetup.setup_device(
            gpu, pin_memory
        )

        # Dataset creation
        dataset_train, dataset_validation = (
            DatasetFactory.create_datasets(
                training_datasource,
                validation_datasource,
                datasource_type,
            )
        )

        # DataLoader creation
        training_loader, validation_loader = (
            DataLoaderFactory.create_dataloaders(
                dataset_train,
                dataset_validation,
                batch_size,
                dataloader_workers,
                device_config.pin_memory,
                prefetch_factor,
            )
        )

        # Model creation
        if compilation:
            model = torch.compile(
                ModelFactory.create_shogi_model(
                    device_config.device
                )
            )
        else:
            model = ModelFactory.create_shogi_model(
                device_config.device
            )

        # Loss functions and optimizer
        loss_fn_policy, loss_fn_value = (
            LossOptimizerFactory.create_loss_functions(
                gce_parameter
            )
        )
        optimizer = LossOptimizerFactory.create_optimizer(
            model,  # type: ignore
            learning_ratio,
            momentum,
        )

        model_components = ModelComponents(
            model=model,  # type: ignore
            loss_fn_policy=loss_fn_policy,
            loss_fn_value=loss_fn_value,
            optimizer=optimizer,
        )

        cls.logger.info("Training components setup completed")

        return (
            device_config,
            (training_loader, validation_loader),
            model_components,
        )
