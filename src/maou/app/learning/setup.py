"""
共通学習セットアップ機能モジュール．
training_benchmark.py と dl.py の重複コードを統一化．
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

try:
    from torch.optim.lr_scheduler import LRScheduler
except (
    ImportError
):  # pragma: no cover - PyTorch < 2.0 compatibility
    from torch.optim.lr_scheduler import (  # type: ignore
        _LRScheduler as LRScheduler,
    )

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.network import (
    BOARD_EMBEDDING_DIM,
    DEFAULT_BOARD_VOCAB_SIZE,
    BackboneArchitecture,
    HeadlessNetwork,
    Network,
)
from maou.app.pre_process.transform import Transform
from maou.domain.model.resnet import BottleneckBlock
from maou.domain.move.label import MOVE_LABELS_NUM


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
    lr_scheduler: Optional[LRScheduler] = None


LR_SCHEDULER_DISPLAY_NAMES: Dict[str, str] = {
    "warmup_cosine_decay": "Warmup+CosineDecay",
    "cosine_annealing_lr": "CosineAnnealingLR",
}


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
        cache_transforms: bool = False,
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
        cache_enabled = (
            cache_transforms and transform is not None
        )
        dataset_train = KifDataset(
            datasource=training_datasource,
            transform=transform,
            cache_transforms=cache_enabled,
        )
        dataset_validation = KifDataset(
            datasource=validation_datasource,
            transform=transform,
            cache_transforms=cache_enabled,
        )

        return dataset_train, dataset_validation


class DataLoaderFactory:
    """DataLoader作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @staticmethod
    def _clamp_workers(
        requested_workers: int,
        n_files: int,
        label: str,
        logger: logging.Logger,
    ) -> int:
        """ワーカー数をファイル数で制限する．

        ストリーミングモードでは各ワーカーが1つ以上のファイルを担当するため，
        ファイル数を超えるワーカーは不要かつ有害(アイドルワーカーの一斉終了が
        GPUプリフェッチャーのデッドロックを引き起こす)．

        Args:
            requested_workers: 要求されたワーカー数
            n_files: データセットのファイル数
            label: ログ出力用ラベル(例: "training", "validation")
            logger: ロガー

        Returns:
            制限後のワーカー数
        """
        if n_files <= 0:
            return 0
        if requested_workers <= 0:
            return 0
        effective = min(requested_workers, n_files)
        if effective < requested_workers:
            logger.info(
                "Clamped %s workers from %d to %d "
                "(limited by file count)",
                label,
                requested_workers,
                effective,
            )
        return effective

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
        """学習・検証用DataLoaderの作成.

        ワーカー数はデータセットサイズで自動的に制限される．
        データセットのサンプル数がワーカー数より少ない場合，
        余剰ワーカーがサンプルを受け取れず無駄になるため，
        ``min(dataloader_workers, len(dataset))`` に制限する．
        """

        # ワーカー数をデータセットサイズで制限
        train_workers = min(
            dataloader_workers, len(dataset_train)
        )
        val_workers = min(
            dataloader_workers, len(dataset_validation)
        )

        if train_workers < dataloader_workers:
            cls.logger.info(
                "Clamped training workers from %d to %d "
                "(limited by dataset size)",
                dataloader_workers,
                train_workers,
            )
        if val_workers < dataloader_workers:
            cls.logger.info(
                "Clamped validation workers from %d to %d "
                "(limited by dataset size)",
                dataloader_workers,
                val_workers,
            )

        # Worker initialization function (per-loader)
        train_worker_init_fn = (
            default_worker_init_fn
            if train_workers > 0
            else None
        )
        val_worker_init_fn = (
            default_worker_init_fn if val_workers > 0 else None
        )

        # Training DataLoader
        training_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=pin_memory,
            persistent_workers=train_workers > 0,
            prefetch_factor=prefetch_factor
            if train_workers > 0
            else None,
            drop_last=drop_last_train,
            timeout=120 if train_workers > 0 else 0,
            worker_init_fn=train_worker_init_fn,
        )

        # Validation DataLoader
        validation_loader = DataLoader(
            dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=pin_memory,
            persistent_workers=val_workers > 0,
            prefetch_factor=prefetch_factor
            if val_workers > 0
            else None,
            drop_last=False,  # validationでは全データを使用
            timeout=120 if val_workers > 0 else 0,
            worker_init_fn=val_worker_init_fn,
        )

        cls.logger.info(
            f"Training: {len(training_loader)} batches"
        )
        cls.logger.info(
            f"Validation: {len(validation_loader)} batches"
        )

        return training_loader, validation_loader

    @classmethod
    def create_streaming_dataloaders(
        cls,
        train_dataset: IterableDataset,
        val_dataset: IterableDataset,
        dataloader_workers: int,
        pin_memory: bool,
        prefetch_factor: int = 2,
        n_train_files: int = 0,
        n_val_files: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """Streaming用DataLoader作成．

        StreamingDatasetがバッチ単位でTensorをyieldするため，
        DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．

        ワーカー数はファイル数で制限される．ファイル数を超えるワーカーは
        アイドル状態で即座に終了し，GPUプリフェッチャーのデッドロックを
        引き起こすため．

        Args:
            train_dataset: 学習用IterableDataset
            val_dataset: 検証用IterableDataset
            dataloader_workers: 要求されたworkerプロセス数
            pin_memory: pinned memoryを有効にするか
            prefetch_factor: 各workerの先読みバッチ数
            n_train_files: 学習データのファイル数(ワーカー数制限用)
            n_val_files: 検証データのファイル数(ワーカー数制限用)

        Returns:
            (training_loader, validation_loader) のタプル
        """
        train_workers = cls._clamp_workers(
            dataloader_workers,
            n_train_files,
            "training",
            cls.logger,
        )
        val_workers = cls._clamp_workers(
            dataloader_workers,
            n_val_files,
            "validation",
            cls.logger,
        )

        train_worker_init_fn = (
            default_worker_init_fn
            if train_workers > 0
            else None
        )
        val_worker_init_fn = (
            default_worker_init_fn if val_workers > 0 else None
        )

        training_loader = DataLoader(
            train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=train_workers,
            pin_memory=pin_memory,
            persistent_workers=train_workers > 0,
            prefetch_factor=prefetch_factor
            if train_workers > 0
            else None,
            timeout=120 if train_workers > 0 else 0,
            worker_init_fn=train_worker_init_fn,
        )

        validation_loader = DataLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=pin_memory,
            persistent_workers=val_workers > 0,
            prefetch_factor=prefetch_factor
            if val_workers > 0
            else None,
            timeout=120 if val_workers > 0 else 0,
            worker_init_fn=val_worker_init_fn,
        )

        if hasattr(train_dataset, "__len__"):
            cls.logger.info(
                "Streaming Training: %d batches",
                len(train_dataset),  # type: ignore[arg-type]
            )
        if hasattr(val_dataset, "__len__"):
            cls.logger.info(
                "Streaming Validation: %d batches",
                len(val_dataset),  # type: ignore[arg-type]
            )

        return training_loader, validation_loader


class ModelFactory:
    """モデル作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def create_shogi_backbone(
        cls,
        device: torch.device,
        *,
        architecture: BackboneArchitecture = "resnet",
        architecture_config: dict[str, Any] | None = None,
        hand_projection_dim: int | None = None,
    ) -> HeadlessNetwork:
        """方策・価値ヘッドを含まないバックボーンを作成."""
        from maou.app.learning.network import (
            DEFAULT_HAND_PROJECTION_DIM,
        )

        if hand_projection_dim is None:
            hand_projection_dim = DEFAULT_HAND_PROJECTION_DIM

        backbone = HeadlessNetwork(
            board_vocab_size=DEFAULT_BOARD_VOCAB_SIZE,
            embedding_dim=BOARD_EMBEDDING_DIM,
            hand_projection_dim=hand_projection_dim,
            board_size=(9, 9),
            architecture=architecture,
            block=BottleneckBlock,
            layers=(2, 2, 2, 2),
            strides=(1, 2, 2, 2),
            out_channels=(64, 128, 256, 512),
            architecture_config=architecture_config,
        )

        backbone.to(device)
        cls.logger.info(
            "Created %s backbone (%s)",
            architecture,
            str(device),
        )

        return backbone

    @classmethod
    def create_shogi_model(
        cls,
        device: torch.device,
        *,
        architecture: BackboneArchitecture = "resnet",
        hand_projection_dim: int | None = None,
        architecture_config: dict[str, Any] | None = None,
    ) -> Network:
        """将棋特化モデルを作成."""
        from maou.app.learning.network import (
            DEFAULT_HAND_PROJECTION_DIM,
        )

        if hand_projection_dim is None:
            hand_projection_dim = DEFAULT_HAND_PROJECTION_DIM

        model = Network(
            num_policy_classes=MOVE_LABELS_NUM,
            board_vocab_size=DEFAULT_BOARD_VOCAB_SIZE,
            embedding_dim=BOARD_EMBEDDING_DIM,
            hand_projection_dim=hand_projection_dim,
            board_size=(9, 9),
            architecture=architecture,
            block=BottleneckBlock,
            layers=(2, 2, 2, 2),
            strides=(1, 2, 2, 2),
            out_channels=(64, 128, 256, 512),
            architecture_config=architecture_config,
        )

        model.to(device)
        cls.logger.info(
            "Created shogi model with %s backbone (%s)",
            architecture,
            str(device),
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
        loss_fn_policy = torch.nn.KLDivLoss(
            reduction="batchmean"
        )
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

            module_name = (
                name.rsplit(".", 1)[0] if "." in name else ""
            )
            parent_module = modules_by_name.get(
                module_name, model
            )

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
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
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


class WarmupCosineDecayScheduler(LRScheduler):
    """Linear warmup followed by cosine decay scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0.0,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive for LR scheduling."
            )
        if warmup_epochs < 0:
            raise ValueError(
                "warmup_epochs must be non-negative for LR scheduling."
            )

        warmup_epochs = min(warmup_epochs, max_epochs)

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr

        super().__init__(optimizer)

        # Ensure the optimizer starts with the warmup-adjusted learning rate
        # before the first training iteration runs. Without this adjustment the
        # first epoch would use the unscaled base learning rate and the warmup
        # schedule would be shifted by one epoch. By explicitly setting
        # ``last_epoch`` to the initial epoch and synchronising the parameter
        # groups, we align the scheduler's state with the intended warm start.
        self.last_epoch = 0
        initial_lrs = self.get_lr()
        for param_group, lr in zip(
            self.optimizer.param_groups, initial_lrs
        ):
            param_group["lr"] = lr
        self._last_lr = initial_lrs

    def get_lr(self) -> List[float]:
        """Return the learning rate for the current epoch."""

        epoch_index = self.last_epoch

        if (
            self.warmup_epochs > 0
            and epoch_index < self.warmup_epochs
        ):
            warmup_progress = (
                epoch_index + 1
            ) / self.warmup_epochs
            return [
                base_lr * warmup_progress
                for base_lr in self.base_lrs
            ]

        decay_epochs = max(
            self.max_epochs - self.warmup_epochs, 1
        )
        decay_progress = min(
            max(epoch_index - self.warmup_epochs, 0)
            / decay_epochs,
            1.0,
        )
        cosine_scale = 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]


class SchedulerFactory:
    """Utility factory for constructing learning rate schedulers."""

    logger: logging.Logger = logging.getLogger(__name__)
    DEFAULT_WARMUP_RATIO: float = 0.1

    @classmethod
    def create_scheduler(
        cls,
        optimizer: torch.optim.Optimizer,
        *,
        lr_scheduler_name: Optional[str] = None,
        max_epochs: int = 1,
    ) -> Optional[LRScheduler]:
        """Create a scheduler for the given optimizer."""

        if lr_scheduler_name is None:
            return None

        normalized_name = lr_scheduler_name.strip()
        if not normalized_name:
            return None

        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive for LR scheduling."
            )

        if normalized_name == "warmup_cosine_decay":
            warmup_epochs = max(
                1,
                math.ceil(
                    max_epochs * cls.DEFAULT_WARMUP_RATIO
                ),
            )
            warmup_epochs = min(warmup_epochs, max_epochs)
            cls.logger.info(
                "Using Warmup+CosineDecay scheduler (warmup_epochs=%d)",
                warmup_epochs,
            )
            return WarmupCosineDecayScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
            )

        if normalized_name == "cosine_annealing_lr":
            cls.logger.info(
                "Using CosineAnnealingLR scheduler (T_max=%d)",
                max_epochs,
            )
            return CosineAnnealingLR(
                optimizer, T_max=max_epochs
            )

        supported = ", ".join(
            LR_SCHEDULER_DISPLAY_NAMES.values()
        )
        raise ValueError(
            "Unsupported learning rate scheduler. "
            f"Supported options are: {supported}"
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
        cache_transforms: Optional[bool] = None,
        gpu: Optional[str] = None,
        model_architecture: BackboneArchitecture = "resnet",
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
        lr_scheduler_name: Optional[str] = None,
        max_epochs: int = 1,
        detect_anomaly: bool = False,
        architecture_config: dict[str, Any] | None = None,
    ) -> Tuple[
        DeviceConfig,
        Tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """学習に必要な全コンポーネントをセットアップ."""

        cls.logger.info("Setting up training components")

        # Torch config
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(
                mode=True, check_nan=True
            )

        # Device setup
        device_config = DeviceSetup.setup_device(
            gpu, pin_memory
        )

        # Dataset creation
        cache_transforms_enabled = (
            cache_transforms
            if cache_transforms is not None
            else datasource_type == "hcpe"
        )
        dataset_train, dataset_validation = (
            DatasetFactory.create_datasets(
                training_datasource,
                validation_datasource,
                datasource_type,
                cache_transforms=cache_transforms_enabled,
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
            device_config.device,
            architecture=model_architecture,
            architecture_config=architecture_config,
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

        lr_scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            lr_scheduler_name=lr_scheduler_name,
            max_epochs=max_epochs,
        )

        model_components = ModelComponents(
            model=model,  # type: ignore
            loss_fn_policy=loss_fn_policy,
            loss_fn_value=loss_fn_value,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        cls.logger.info("Training components setup completed")

        return (
            device_config,
            (training_loader, validation_loader),
            model_components,
        )
