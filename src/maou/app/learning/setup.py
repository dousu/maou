"""
共通学習セットアップ機能モジュール．
training_benchmark.py と dl.py の重複コードを統一化．
"""

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

import polars as pl

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.network import (
    BackboneArchitecture,
    HeadlessNetwork,
    Network,
)
from maou.domain.data.arrow_format import scan_row_count

logger = logging.getLogger(__name__)


def _log_worker_memory(
    worker_id: int,
    label: str,
    level: int = logging.DEBUG,
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
    import time

    import numpy as np

    start = time.monotonic()

    # 再現性のためのシード設定（ワーカーごとに異なるシードを使用）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    elapsed = time.monotonic() - start
    logger.debug(
        "Worker %d initialized (pid=%d, seed=%d, init_time=%.1fs)",
        worker_id,
        os.getpid(),
        worker_seed,
        elapsed,
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
    lr_scheduler: LRScheduler | None = None


SUPPORTED_LR_SCHEDULER_KEYS: tuple[str, ...] = (
    "warmup_cosine_decay",
    "cosine_annealing_lr",
)
"""``create_scheduler`` が受理する正規化済みスケジューラ名．

以前は表示名 ("Warmup+CosineDecay" 等) との対応表だったが，表示名は
どこからも読まれていなかった (正準の対応表は
``maou.interface.learn.SUPPORTED_LR_SCHEDULERS``)．未対応時の
エラーはここに並ぶキーを案内する．
"""


class DeviceSetup:
    """デバイス設定の共通化."""

    logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def setup_device(
        cls,
        gpu: str | None = None,
        pin_memory: bool | None = None,
    ) -> DeviceConfig:
        """GPU/CPUデバイスの設定."""
        if gpu is not None and gpu != "cpu":
            device = torch.device(gpu)
            # get_device_name は CUDA 専用．"mps" のような非 CUDA
            # デバイス名を渡されると例外になるため型で分岐する．
            if device.type == "cuda":
                cls.logger.info(
                    f"Using GPU: {torch.cuda.get_device_name(device)}"
                )
            else:
                cls.logger.info(f"Using device: {device}")
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
    ) -> tuple[KifDataset, KifDataset]:
        """学習・検証用データセットの作成."""

        dataset_train = KifDataset(
            datasource=training_datasource,
        )
        dataset_validation = KifDataset(
            datasource=validation_datasource,
        )

        return dataset_train, dataset_validation


_PEAK_IMAGES: float = 2.0
"""ワーカーが同時に持つ「展開後 1 ファイル分」のイメージ数．

Rust の ``load_feather`` は record batch の ``Vec`` を集めてから
``concat_batches`` で 1 つに複製するので読込中に 2 枚，変換中は
Polars DataFrame + numpy (ColumnarBatch ≈ 0.75 枚) で 1.75 枚になる．
前ファイルの ColumnarBatch は次を読む前に手放す
(``StreamingFileSource.iter_files_columnar_subset``) ので，これ以上は重ならない．
"""

_PEAK_MARGIN: float = 1.15
"""展開後サイズから見積もるときの安全マージン (アロケータの断片化・torch の
ベースライン分)．2026-09-21 の Colab G4 (176GB) では旧見積が 1 ファイル
1.3GB としていた worker が実際は 34GiB (= 展開後 12GB × 2.85) に達し，
5 worker で memory cgroup の OOM になった (Explore 調査 + dmesg)．
"""

_DECOMPRESSION_FACTOR: float = 4.0
"""スキーマから展開後サイズが分からないときに使う LZ4 の展開倍率．

固定幅でない列 (可変長 List / 文字列) を含むファイルだけがこの経路に落ちる．
maou が書く前処理済 / HCPE は FixedSizeList なので通常は使わない．
"""

_SAFETY_MARGIN: float = 4.0
"""展開倍率経路の安全マージン．

展開倍率は一般値でしかない (前処理済の ``moveWinRate`` 1496×f32 は約 68 倍に
展開される) ので，``_WORKER_MEMORY_BUDGET_FRACTION`` を 0.4 から 0.8 に上げた
分を打ち消して従来と同じ上限に保つ．
"""

_WORKER_MEMORY_BUDGET_FRACTION: float = 0.8
"""DataLoader ワーカー群に割り当てる利用可能メモリの割合．

残りはメインプロセス (モデル・pinned buffer・CUDA context・Python) 用に確保する．
旧値 0.4 は per-worker の見積誤差 (LZ4 4 倍固定) を吸収するための値で，
展開後サイズをスキーマと行数から直接見積もるようになったので 0.8 に上げた
(176GB の VM で 12GB/ファイルなら 5 worker，15GB の DevContainer で
小ファイルなら従来どおり)．
"""

_FALLBACK_PER_WORKER_MB: float = 200.0
"""ファイルサイズ情報が利用できない場合のデフォルト値．"""

_SIZE_SAMPLE_LIMIT: int = 64
"""平均ファイルサイズの推定に使う stat() の最大回数．

用途は DataLoader ワーカー数を決めるための平均サイズ1つだけであり，
学習開始前の同期パスで数万ファイルを stat() する価値はない．
先頭に偏らないよう等間隔サンプリングする．
"""


def _sample_for_size_estimate(
    file_paths: list[Path],
) -> list[Path]:
    """サイズ推定用に等間隔でファイルを間引く．

    Args:
        file_paths: データファイルパスのリスト

    Returns:
        高々 ``_SIZE_SAMPLE_LIMIT`` 件の等間隔サンプル
    """
    total = len(file_paths)
    if total <= _SIZE_SAMPLE_LIMIT:
        return file_paths
    step = total / _SIZE_SAMPLE_LIMIT
    return [
        file_paths[int(i * step)]
        for i in range(_SIZE_SAMPLE_LIMIT)
    ]


_FIXED_WIDTH_BYTES: dict[pl.DataType, int] = {
    pl.Boolean(): 1,
    pl.Int8(): 1,
    pl.UInt8(): 1,
    pl.Int16(): 2,
    pl.UInt16(): 2,
    pl.Int32(): 4,
    pl.UInt32(): 4,
    pl.Float32(): 4,
    pl.Int64(): 8,
    pl.UInt64(): 8,
    pl.Float64(): 8,
}


def _dtype_row_bytes(
    dtype: pl.DataType | pl.datatypes.DataTypeClass,
) -> int | None:
    """固定幅 dtype の 1 行あたりバイト数 (固定幅でなければ ``None``)．

    Arrow の FixedSizeList は Polars では ``pl.Array`` (入れ子も可) に
    なるので再帰で畳む．可変長 List / 文字列は展開後サイズが行数から
    決まらないので ``None`` を返し，呼び出し側が展開倍率の経路に落とす．
    """
    if isinstance(dtype, pl.Array):
        inner = _dtype_row_bytes(dtype.inner)
        return None if inner is None else inner * dtype.size
    for fixed, width in _FIXED_WIDTH_BYTES.items():
        if dtype == fixed:
            return width
    return None


def _uncompressed_file_mb(file_path: Path) -> float | None:
    """feather の展開後サイズ (MB) をメタデータだけから求める．

    スキーマ (``pl.read_ipc_schema``) と行数 (``scan_row_count``，File 形式
    ならメタデータ読み) の積なのでデータ本体は読まない．固定幅でない列を
    含む，または読めないときは ``None``．
    """
    try:
        schema = pl.read_ipc_schema(file_path)
        row_bytes = 0
        for dtype in schema.values():
            width = _dtype_row_bytes(dtype)
            if width is None:
                return None
            row_bytes += width
        rows = scan_row_count(file_path)
    except (
        OSError,
        TypeError,
        ValueError,
        pl.exceptions.PolarsError,
    ):
        return None
    return row_bytes * rows / (1024**2)


def _estimate_per_worker_mb(
    file_paths: list[Path] | None,
    logger: logging.Logger,
) -> float:
    """データファイルからワーカーあたりのメモリ消費量を推定する．

    各ワーカーは担当ファイルを 1 つずつ丸ごと展開するので，ピークは
    **最大の 1 ファイルの展開後サイズ** × ``_PEAK_IMAGES`` × ``_PEAK_MARGIN``．
    展開後サイズはスキーマと行数から求める (``_uncompressed_file_mb``)．
    固定幅でない列を含むなど求まらないファイルは，圧縮サイズ ×
    ``_DECOMPRESSION_FACTOR`` × ``_SAFETY_MARGIN`` で代用する．
    ファイルパスが未指定またはすべて読めない場合はフォールバック値を返す．

    サンプルは高々 ``_SIZE_SAMPLE_LIMIT`` 件の等間隔 (全ファイルを stat() しない)．

    Args:
        file_paths: データファイルパスのリスト
        logger: ロガー

    Returns:
        ワーカーあたりの推定メモリ消費量(MB)
    """
    if not file_paths:
        return _FALLBACK_PER_WORKER_MB

    sampled = _sample_for_size_estimate(file_paths)
    estimates: list[float] = []
    n_exact = 0
    max_uncompressed_mb = 0.0
    max_compressed_mb = 0.0
    for fp in sampled:
        try:
            compressed_mb = fp.stat().st_size / (1024**2)
        except OSError:
            continue
        uncompressed_mb = _uncompressed_file_mb(fp)
        if uncompressed_mb is not None:
            n_exact += 1
            max_uncompressed_mb = max(
                max_uncompressed_mb, uncompressed_mb
            )
            estimates.append(
                uncompressed_mb * _PEAK_IMAGES * _PEAK_MARGIN
            )
        else:
            max_compressed_mb = max(
                max_compressed_mb, compressed_mb
            )
            estimates.append(
                compressed_mb
                * _DECOMPRESSION_FACTOR
                * _SAFETY_MARGIN
            )

    if not estimates:
        logger.warning(
            "No accessible data files found; "
            "using fallback per_worker_mb=%.0f",
            _FALLBACK_PER_WORKER_MB,
        )
        return _FALLBACK_PER_WORKER_MB

    per_worker_mb = max(max(estimates), _FALLBACK_PER_WORKER_MB)

    logger.info(
        "Dynamic per_worker_mb=%.0f "
        "(max_uncompressed=%.1fMB from schema x rows for %d files, "
        "max_compressed=%.1fMB x %.0f for %d files; "
        "%d/%d accessible of %d total)",
        per_worker_mb,
        max_uncompressed_mb,
        n_exact,
        max_compressed_mb,
        _DECOMPRESSION_FACTOR * _SAFETY_MARGIN,
        len(estimates) - n_exact,
        len(estimates),
        len(sampled),
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

    ファイルパスが渡された場合，最大ファイルの展開後サイズ (スキーマ × 行数)
    からワーカーあたりのメモリ消費量を動的に推定する (``_estimate_per_worker_mb``)．
    割当は利用可能メモリの ``_WORKER_MEMORY_BUDGET_FRACTION`` に制限する．

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

    # ワーカーに割当可能なメモリ (残りはメインプロセス用に確保)
    worker_budget_mb = (
        available_mb * _WORKER_MEMORY_BUDGET_FRACTION
    )

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
        ファイル数を超えるワーカーは不要かつ有害(アイドルワーカーがリソースを
        消費する)．

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
        collate_fn: Callable | None = None,
    ) -> tuple[DataLoader, DataLoader]:
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
        batch_size: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Streaming用DataLoader作成．

        StreamingDatasetがバッチ単位でTensorをyieldするため，
        DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．

        ワーカー数はファイル数およびシステムメモリで制限される．
        ファイル数を超えるワーカーはアイドル状態で即座に終了するため．
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
            batch_size: StreamingDataset が yield するバッチのサイズ．
                /dev/shm 見積り用のみに使用する(DataLoader 自体は
                batch_size=None)．None の場合は概算フォールバック値を使う．

        Returns:
            (training_loader, validation_loader) のタプル
        """
        # pin_memory はストリーミングモードでもそのまま使用する．
        # DataPrefetcher 除去後は pin_memory_thread が spawn コンテキストで
        # 安全に動作する(PyTorch issue #130610 参照)．

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

        _check_shm_size(
            num_workers=max(train_workers, val_workers),
            batch_size=batch_size,
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

        # spawn + persistent_workers でのデッドロックを検出するため
        # 有限タイムアウトを設定する(spawn起動+大ファイル読込を考慮)．
        _STREAMING_TIMEOUT = 300 if train_workers > 0 else 0

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
            timeout=_STREAMING_TIMEOUT,
            worker_init_fn=train_worker_init_fn,
            multiprocessing_context=mp_context,
        )

        _STREAMING_TIMEOUT_VAL = 300 if val_workers > 0 else 0

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
            timeout=_STREAMING_TIMEOUT_VAL,
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

        # board_vocab_size / embedding_dim / board_size / block /
        # layers / strides / out_channels は HeadlessNetwork の既定値
        # をそのまま使う．ここで再掲するとネットワーク側の既定を
        # 変更したときに片方だけ古い値が残る．
        backbone = HeadlessNetwork(
            hand_projection_dim=hand_projection_dim,
            architecture=architecture,
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

        # num_policy_classes 以下のバックボーン形状は Network の
        # 既定値をそのまま使う (create_shogi_backbone と同じ理由)．
        model = Network(
            hand_projection_dim=hand_projection_dim,
            architecture=architecture,
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
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
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
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> torch.optim.Optimizer:
        """オプティマイザを作成."""
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
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

            if (
                isinstance(parent_module, normalization_modules)
                or param.ndim <= 1
            ):
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

    @staticmethod
    def compute_effective_lr(
        learning_rate: float,
        actual_batch_size: int,
        base_batch_size: int,
        logger: logging.Logger | None = None,
    ) -> float:
        """バッチサイズに基づく LR sqrt スケーリングを計算する．

        actual_batch_size > base_batch_size の場合，
        effective_lr = learning_rate * sqrt(actual / base) を返す．

        Args:
            learning_rate: ベース学習率．
            actual_batch_size: 実際のバッチサイズ．
            base_batch_size: 基準バッチサイズ(デフォルト 256)．
            logger: ロガー(スケーリング情報の出力用)．

        Returns:
            スケーリング後の学習率．
        """
        if actual_batch_size > base_batch_size:
            scale = math.sqrt(
                actual_batch_size / base_batch_size
            )
            effective_lr = learning_rate * scale
            if logger is not None:
                logger.info(
                    "LR sqrt scaling: base_lr=%.6f, scale=%.2f, "
                    "effective_lr=%.6f",
                    learning_rate,
                    scale,
                    effective_lr,
                )
            return effective_lr
        return learning_rate


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

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return the learning rate for the current step.

        PyTorch 2.11 で基底 ``LRScheduler.get_lr`` の返り値型が
        ``list[float | Tensor]`` に変更されたため，override も同じ型に
        揃える(``list`` は不変なので明示的に注釈する)．実際に返すのは
        float のみ．
        """

        step = self.last_epoch  # PyTorch convention

        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_progress = (step + 1) / self.warmup_steps
            warmup_lrs: list[float | torch.Tensor] = [
                base_lr * warmup_progress
                for base_lr in self.base_lrs
            ]
            return warmup_lrs

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

        decay_lrs: list[float | torch.Tensor] = [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]
        return decay_lrs


class SchedulerFactory:
    """Utility factory for constructing learning rate schedulers."""

    logger: logging.Logger = logging.getLogger(__name__)
    DEFAULT_WARMUP_RATIO: float = 0.1

    @classmethod
    def create_scheduler(
        cls,
        optimizer: torch.optim.Optimizer,
        *,
        lr_scheduler_name: str | None = None,
        max_epochs: int = 1,
        steps_per_epoch: int = 1,
    ) -> LRScheduler | None:
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

        # 受理されるのは正規化後のキーであって表示名ではない．
        # 表示名 ("Warmup+CosineDecay" 等) を案内すると，
        # そのまま指定したユーザーが再び同じエラーに当たる．
        supported = ", ".join(SUPPORTED_LR_SCHEDULER_KEYS)
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
        gpu: str | None = None,
        model_architecture: BackboneArchitecture = "resnet",
        batch_size: int = 256,
        dataloader_workers: int = 4,
        pin_memory: bool | None = None,
        prefetch_factor: int = 2,
        learning_ratio: float = 0.01,
        momentum: float = 0.9,
        optimizer_name: str = "adamw",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        lr_scheduler_name: str | None = None,
        max_epochs: int = 1,
        detect_anomaly: bool = False,
        architecture_config: dict[str, Any] | None = None,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
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
        dataset_train, dataset_validation = (
            DatasetFactory.create_datasets(
                training_datasource,
                validation_datasource,
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
            LossOptimizerFactory.create_loss_functions()
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
