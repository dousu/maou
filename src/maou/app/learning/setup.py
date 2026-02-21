"""
共通学習セットアップ機能モジュール．
training_benchmark.py と dl.py の重複コードを統一化．
"""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
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
from maou.domain.model.resnet import BottleneckBlock
from maou.domain.move.label import MOVE_LABELS_NUM

logger = logging.getLogger(__name__)


def _log_worker_memory(
    worker_id: int,
    label: str,
    level: int = logging.INFO,
) -> None:
    """ワーカープロセスのRSSメモリ使用量をログ出力する．

    Args:
        worker_id: ワーカーID
        label: ログラベル(例: "init", "after_first_file")
        level: ログレベル
    """
    _logger = logging.getLogger(__name__)
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / (
            1024 * 1024
        )
    except (ImportError, OSError):
        return

    _logger.log(
        level,
        "Worker %d memory [%s]: RSS=%.0fMB",
        worker_id,
        label,
        rss_mb,
    )


def default_worker_init_fn(worker_id: int) -> None:
    """デフォルトのワーカー初期化関数．

    spawn コンテキストで新規プロセスとして起動されるため，
    シード設定とライフサイクルログを行う．
    """
    import random

    import numpy as np

    # 再現性のためのシード設定（ワーカーごとに異なるシードを使用）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    logger.info(
        "Worker %d initialized (pid=%d, seed=%d)",
        worker_id,
        os.getpid(),
        worker_seed,
    )
    _log_worker_memory(worker_id, "init")


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
        cache_transforms: bool = False,
    ) -> Tuple[KifDataset, KifDataset]:
        """学習・検証用データセットの作成."""

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


_DECOMPRESSION_FACTOR: float = 4.0
"""LZ4の一般的展開倍率(2-4倍)の上限値．"""

_SAFETY_MARGIN: float = 1.5
"""DataFrame + numpy配列共存の安全マージン．"""

_FALLBACK_PER_WORKER_MB: float = 200.0
"""ファイルサイズ情報が利用できない場合のデフォルト値．"""


def _estimate_per_worker_mb(
    file_paths: list[Path] | None,
    logger: logging.Logger,
) -> float:
    """データファイルサイズからワーカーあたりのメモリ消費量を推定する．

    圧縮済みファイルの平均サイズに展開倍率と安全マージンを乗じて
    1ワーカーあたりのメモリ使用量を算出する．
    ファイルパスが未指定またはすべて存在しない場合はフォールバック値を返す．

    Args:
        file_paths: データファイルパスのリスト
        logger: ロガー

    Returns:
        ワーカーあたりの推定メモリ消費量(MB)
    """
    if not file_paths:
        return _FALLBACK_PER_WORKER_MB

    file_sizes = []
    for fp in file_paths:
        try:
            file_sizes.append(fp.stat().st_size)
        except OSError:
            continue

    if not file_sizes:
        logger.warning(
            "No accessible data files found; "
            "using fallback per_worker_mb=%.0f",
            _FALLBACK_PER_WORKER_MB,
        )
        return _FALLBACK_PER_WORKER_MB

    avg_compressed_mb = (
        sum(file_sizes) / len(file_sizes) / (1024**2)
    )
    estimated = (
        avg_compressed_mb
        * _DECOMPRESSION_FACTOR
        * _SAFETY_MARGIN
    )
    per_worker_mb = max(estimated, _FALLBACK_PER_WORKER_MB)

    logger.info(
        "Dynamic per_worker_mb=%.0f "
        "(avg_file=%.1fMB, files=%d/%d accessible)",
        per_worker_mb,
        avg_compressed_mb,
        len(file_sizes),
        len(file_paths),
    )

    return per_worker_mb


def _estimate_max_workers_by_memory(
    pin_memory: bool,
    logger: logging.Logger,
    file_paths: list[Path] | None = None,
) -> int:
    """システムの利用可能メモリからワーカー数の上限を推定する．

    各DataLoaderワーカーはArrowファイルを読み込むため，
    一定のメモリを消費する．利用可能メモリの一部を
    DataLoaderワーカーに割り当て，安全なワーカー数を算出する．

    ファイルパスが渡された場合，圧縮ファイルの実サイズから
    ワーカーあたりのメモリ消費量を動的に推定する．
    展開倍率(LZ4: 4.0)と安全マージン(1.5)を考慮する．

    Args:
        pin_memory: pinned memory が有効か
        logger: ロガー
        file_paths: データファイルパスのリスト(動的メモリ推定用)

    Returns:
        メモリベースのワーカー数上限(最小1)
    """
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (
            1024 * 1024
        )
    except ImportError:
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_mb = (pages * page_size) / (1024 * 1024)
            available_mb = total_mb * 0.6
        except (ValueError, OSError):
            logger.warning(
                "Cannot determine available memory; "
                "skipping memory-based worker limit"
            )
            return 64  # 実質的に無制限

    # ワーカーに割当可能なメモリ: 利用可能メモリの50%
    worker_budget_mb = available_mb * 0.5

    # ファイルサイズベースの動的メモリ推定
    per_worker_mb = _estimate_per_worker_mb(file_paths, logger)
    if pin_memory:
        per_worker_mb += 50.0

    max_workers = max(1, int(worker_budget_mb / per_worker_mb))

    logger.info(
        "Memory-based worker limit: %d "
        "(available=%.0fMB, budget=%.0fMB, per_worker=%.0fMB, "
        "note: estimate may differ from actual usage)",
        max_workers,
        available_mb,
        worker_budget_mb,
        per_worker_mb,
    )

    return max_workers


def _check_shm_size(
    num_workers: int,
    batch_size: int | None,
    prefetch_factor: int,
    logger: logging.Logger,
) -> None:
    """Linux環境で /dev/shm の空き容量を確認し不足時に警告する．

    DataLoaderのワーカー間通信は共有メモリ(/dev/shm)を使用する．
    Docker等で /dev/shm サイズが制限されている場合，ワーカーが
    クラッシュする原因となる．

    Args:
        num_workers: DataLoaderのワーカー数
        batch_size: バッチサイズ(Noneの場合はストリーミングモード)
        prefetch_factor: プリフェッチファクター
        logger: ロガー
    """
    import sys

    if sys.platform != "linux" or num_workers <= 0:
        return

    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        return

    try:
        import os

        stat = os.statvfs("/dev/shm")
        shm_available_mb = (stat.f_bavail * stat.f_frsize) / (
            1024 * 1024
        )
    except OSError:
        return

    # threshold概算: batch_size × 154KB × num_workers × prefetch_factor
    # 154KB = Stage 3の1バッチあたりの入力テンソルサイズ概算
    _BYTES_PER_SAMPLE_KB = 154
    effective_batch_size = (
        batch_size if batch_size is not None else 1024
    )
    threshold_mb = (
        effective_batch_size
        * _BYTES_PER_SAMPLE_KB
        * num_workers
        * prefetch_factor
        / 1024
    )

    if shm_available_mb < threshold_mb:
        logger.warning(
            "/dev/shm available space (%.0fMB) is below "
            "estimated requirement (%.0fMB) for %d workers "
            "with prefetch_factor=%d. "
            "Consider increasing /dev/shm size "
            "(e.g. docker run --shm-size=8g) or "
            "reducing --dataloader-workers.",
            shm_available_mb,
            threshold_mb,
            num_workers,
            prefetch_factor,
        )


class DataLoaderFactory:
    """DataLoader作成の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @staticmethod
    def _clamp_workers(
        requested_workers: int,
        n_files: int,
        label: str,
        logger: logging.Logger,
        *,
        memory_limit: int | None = None,
    ) -> int:
        """ワーカー数をファイル数およびメモリ制約で制限する．

        ストリーミングモードでは各ワーカーが1つ以上のファイルを担当するため，
        ファイル数を超えるワーカーは不要かつ有害(アイドルワーカーの一斉終了が
        GPUプリフェッチャーのデッドロックを引き起こす)．

        Args:
            requested_workers: 要求されたワーカー数
            n_files: データセットのファイル数
            label: ログ出力用ラベル(例: "training", "validation")
            logger: ロガー
            memory_limit: メモリベースのワーカー数上限(Noneで無制限)

        Returns:
            制限後のワーカー数
        """
        if n_files <= 0:
            return 0
        if requested_workers <= 0:
            return 0
        effective = min(requested_workers, n_files)
        if memory_limit is not None and memory_limit > 0:
            effective = min(effective, memory_limit)
        if effective < requested_workers:
            logger.info(
                "Clamped %s workers from %d to %d "
                "(file_count=%d, memory_limit=%s)",
                label,
                requested_workers,
                effective,
                n_files,
                memory_limit,
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
        file_paths: list[Path] | None = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Streaming用DataLoader作成．

        StreamingDatasetがバッチ単位でTensorをyieldするため，
        DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．

        ワーカー数はファイル数およびシステムメモリで制限される．
        ファイル数を超えるワーカーはアイドル状態で即座に終了し，
        GPUプリフェッチャーのデッドロックを引き起こすため．
        また，各ワーカーがArrowファイルを独立に読み込むため，
        利用可能メモリに基づく上限も適用する．

        ストリーミングワーカーは Rust FFI (Polars/Arrow) を呼び出すため，
        multiprocessing_context="spawn" を使用する．fork/forkserver では
        jemalloc の内部状態が子プロセスに継承され segfault する．

        Args:
            train_dataset: 学習用IterableDataset
            val_dataset: 検証用IterableDataset
            dataloader_workers: 要求されたworkerプロセス数
            pin_memory: pinned memoryを有効にするか
            prefetch_factor: 各workerの先読みバッチ数
            n_train_files: 学習データのファイル数(ワーカー数制限用)
            n_val_files: 検証データのファイル数(ワーカー数制限用)
            file_paths: データファイルパスのリスト(動的メモリ推定用)

        Returns:
            (training_loader, validation_loader) のタプル
        """
        memory_limit = _estimate_max_workers_by_memory(
            pin_memory=pin_memory,
            logger=cls.logger,
            file_paths=file_paths,
        )
        train_workers = cls._clamp_workers(
            dataloader_workers,
            n_train_files,
            "training",
            cls.logger,
            memory_limit=memory_limit,
        )
        val_workers = cls._clamp_workers(
            dataloader_workers,
            n_val_files,
            "validation",
            cls.logger,
            memory_limit=memory_limit,
        )

        # spawn コンテキストでのワーカー上限を適用
        # 各ワーカーが独立にPython+Polars+Rustを初期化するため，
        # 過多なワーカーはメモリ圧迫と起動遅延を招く
        _MAX_SPAWN_WORKERS = 8
        if train_workers > _MAX_SPAWN_WORKERS:
            cls.logger.info(
                "Clamped training streaming workers from %d to %d "
                "(spawn context limit)",
                train_workers,
                _MAX_SPAWN_WORKERS,
            )
            train_workers = _MAX_SPAWN_WORKERS
        if val_workers > _MAX_SPAWN_WORKERS:
            cls.logger.info(
                "Clamped validation streaming workers from %d to %d "
                "(spawn context limit)",
                val_workers,
                _MAX_SPAWN_WORKERS,
            )
            val_workers = _MAX_SPAWN_WORKERS

        _check_shm_size(
            num_workers=max(train_workers, val_workers),
            batch_size=None,
            prefetch_factor=prefetch_factor,
            logger=cls.logger,
        )

        train_worker_init_fn = (
            default_worker_init_fn
            if train_workers > 0
            else None
        )
        val_worker_init_fn = (
            default_worker_init_fn if val_workers > 0 else None
        )

        # ストリーミングモードでは spawn を使用する．
        # Polars/Rust (jemalloc) が初期化済みプロセスを fork() すると，
        # 子プロセスが不整合なアロケータ状態を継承し segfault する．
        # forkserver も fork() でデーモンを生成するため同様に危険．
        # spawn は os.exec() で完全に新しいプロセスを生成するため安全．
        mp_context: str | None = (
            "spawn" if train_workers > 0 else None
        )
        mp_context_val: str | None = (
            "spawn" if val_workers > 0 else None
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
            timeout=0,
            worker_init_fn=train_worker_init_fn,
            multiprocessing_context=mp_context,
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
            timeout=0,
            worker_init_fn=val_worker_init_fn,
            multiprocessing_context=mp_context_val,
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
    """Linear warmup followed by cosine decay scheduler.

    Per-step(バッチ単位)でステップし，エポック境界でのLRジャンプを防ぐ．
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(
                "total_steps must be positive for LR scheduling."
            )
        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be non-negative for LR scheduling."
            )

        warmup_steps = min(warmup_steps, total_steps)

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer)

        # Ensure the optimizer starts with the warmup-adjusted learning rate
        # before the first training iteration runs. Without this adjustment the
        # first step would use the unscaled base learning rate and the warmup
        # schedule would be shifted by one step. By explicitly setting
        # ``last_epoch`` to the initial step and synchronising the parameter
        # groups, we align the scheduler's state with the intended warm start.
        self.last_epoch = 0
        initial_lrs = self.get_lr()
        for param_group, lr in zip(
            self.optimizer.param_groups, initial_lrs
        ):
            param_group["lr"] = lr
        self._last_lr = initial_lrs

    def get_lr(self) -> List[float]:
        """Return the learning rate for the current step."""

        step = self.last_epoch  # PyTorch convention

        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_progress = (step + 1) / self.warmup_steps
            return [
                base_lr * warmup_progress
                for base_lr in self.base_lrs
            ]

        decay_steps = max(
            self.total_steps - self.warmup_steps, 1
        )
        decay_progress = min(
            max(step - self.warmup_steps, 0) / decay_steps,
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
        steps_per_epoch: int = 1,
    ) -> Optional[LRScheduler]:
        """Create a per-step scheduler for the given optimizer.

        Args:
            optimizer: 対象のオプティマイザ
            lr_scheduler_name: スケジューラ名
            max_epochs: 最大エポック数
            steps_per_epoch: 1エポックあたりのバッチ数
        """

        if lr_scheduler_name is None:
            return None

        normalized_name = lr_scheduler_name.strip()
        if not normalized_name:
            return None

        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive for LR scheduling."
            )

        total_steps = max_epochs * steps_per_epoch

        if normalized_name == "warmup_cosine_decay":
            warmup_steps = max(
                steps_per_epoch,
                math.ceil(
                    total_steps * cls.DEFAULT_WARMUP_RATIO
                ),
            )
            warmup_steps = min(warmup_steps, total_steps)
            cls.logger.info(
                "Using Warmup+CosineDecay scheduler "
                "(warmup_steps=%d, total_steps=%d)",
                warmup_steps,
                total_steps,
            )
            return WarmupCosineDecayScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )

        if normalized_name == "cosine_annealing_lr":
            cls.logger.info(
                "Using CosineAnnealingLR scheduler (T_max=%d)",
                total_steps,
            )
            return CosineAnnealingLR(
                optimizer, T_max=total_steps
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
            else False
        )
        dataset_train, dataset_validation = (
            DatasetFactory.create_datasets(
                training_datasource,
                validation_datasource,
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
            steps_per_epoch=len(training_loader),
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
