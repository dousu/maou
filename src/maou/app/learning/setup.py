"""
共通学習セットアップ機能モジュール．
training_benchmark.py と dl.py の重複コードを統一化．
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.network import HeadlessNetwork, Network
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.app.pre_process.transform import Transform
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.resnet import BottleneckBlock


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

    model: torch.nn.Module
    loss_fn_policy: torch.nn.Module
    loss_fn_value: torch.nn.Module
    optimizer: torch.optim.Optimizer


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
        drop_last_train: bool = True,
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
            drop_last=drop_last_train,
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
    def create_shogi_backbone(
        cls, device: torch.device
    ) -> HeadlessNetwork:
        """方策・価値ヘッドを含まないResNetバックボーンを作成."""

        backbone = HeadlessNetwork(
            num_channels=FEATURES_NUM,
            board_size=(9, 9),
            block=BottleneckBlock,
            layers=(2, 2, 2, 2),
            strides=(1, 2, 2, 2),
            out_channels=(64, 128, 256, 512),
        )

        backbone.to(device)
        cls.logger.info(
            "Created shogi-optimized ResNet backbone (%s)",
            str(device),
        )

        return backbone

    @classmethod
    def create_shogi_model(
        cls, device: torch.device
    ) -> Network:
        """将棋特化のResNetモデルを作成."""

        model = Network(
            num_policy_classes=MOVE_LABELS_NUM,
            num_channels=FEATURES_NUM,
            board_size=(9, 9),
            block=BottleneckBlock,
            layers=(2, 2, 2, 2),
            strides=(1, 2, 2, 2),
            out_channels=(64, 128, 256, 512),
        )

        model.to(device)
        cls.logger.info(
            f"Created shogi-optimized ResNet model ({str(device)})"
        )

        return model


class LossOptimizerFactory:
    """損失関数・オプティマイザ作成の共通化."""

    @classmethod
    def create_loss_functions(
        cls,
        gce_parameter: float = 0.1,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """方策・価値用の損失関数ペアを作成．

        Value loss関数としてBCEWithLogitsLossを使用．
        二峰性分布（0と1に集中）のデータに対してMSELossは平均値予測が
        最適解となるため，BCEWithLogitsLossの方が適切．

        BCEWithLogitsLossはSigmoidとBCE lossを組み合わせた関数で，
        以下の利点がある:
        - 数値的により安定（log-sum-exp trick）
        - Mixed precision training（autocast）と互換性がある
        - Value headはlogitsを出力し，損失関数内部でSigmoidが適用される
        """
        _ = gce_parameter
        loss_fn_policy = torch.nn.KLDivLoss(reduction="batchmean")
        # BCEWithLogitsLoss: Value headはlogitsを出力
        # Sigmoid + BCE lossを内部で実行（数値的に安定，autocast対応）
        loss_fn_value = torch.nn.BCEWithLogitsLoss()
        return loss_fn_policy, loss_fn_value

    @classmethod
    def create_optimizer(
        cls,
        model: torch.nn.Module,
        learning_ratio: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> torch.optim.Optimizer:
        """オプティマイザを作成."""
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        modules_by_name = dict(model.named_modules())
        normalization_modules = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.LayerNorm,
        )

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = modules_by_name.get(module_name, model)

            if isinstance(parent_module, normalization_modules):
                no_decay_params.append(param)
            elif param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        if not decay_params:
            raise ValueError(
                "No parameters found for the weight decay parameter group."
            )

        if not no_decay_params:
            raise ValueError(
                "No parameters found for the no-weight-decay parameter group."
            )

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer_key = optimizer_name.lower()

        if optimizer_key == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=learning_ratio,
                betas=betas,
                eps=eps,
            )

        if optimizer_key == "sgd":
            return torch.optim.SGD(
                param_groups,
                lr=learning_ratio,
                momentum=momentum,
            )

        raise ValueError(
            f"Unsupported optimizer `{optimizer_name}`. "
            "Expected 'adamw' or 'sgd'."
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
        batch_size: int = 256,
        dataloader_workers: int = 4,
        pin_memory: Optional[bool] = None,
        prefetch_factor: int = 2,
        gce_parameter: float = 0.1,
        learning_ratio: float = 0.01,
        momentum: float = 0.9,
        optimizer_name: str = "adamw",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
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
            optimizer_name=optimizer_name,
            betas=(optimizer_beta1, optimizer_beta2),
            eps=optimizer_eps,
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
