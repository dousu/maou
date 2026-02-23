"""ステージ別コンポーネント生成ファクトリ．

各学習ステージ(Stage1/Stage2)のデータパイプラインおよび
モデル・オプティマイザ等のコンポーネント一式を生成する．
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader

from maou.app.learning.dataset import (
    Stage1Dataset,
    Stage2Dataset,
)
from maou.app.learning.multi_stage_training import (
    Stage1DatasetAdapter,
    Stage2DatasetAdapter,
    pre_stage_collate_fn,
)
from maou.app.learning.setup import (
    DataLoaderFactory,
    LossOptimizerFactory,
    SchedulerFactory,
)
from maou.domain.loss.loss_fn import (
    LegalMovesLoss,
    ReachableSquaresLoss,
)

if TYPE_CHECKING:
    from maou.app.learning.dl import LearningDataSource
    from maou.app.learning.network import HeadlessNetwork
    from maou.app.learning.streaming_dataset import (
        StreamingDataSource,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageDataPipeline:
    """ステージ学習用のデータパイプライン．

    DataLoader と損失関数をまとめて保持する．
    """

    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]
    loss_fn: torch.nn.Module


@dataclass(frozen=True)
class StageComponents:
    """ステージ学習に必要なコンポーネント一式．

    モデル，DataLoader，損失関数，オプティマイザ，
    学習率スケジューラをまとめて保持する．
    """

    model: torch.nn.Module
    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]


class StageComponentFactory:
    """各ステージのコンポーネントを生成するファクトリクラス．"""

    @staticmethod
    def create_stage1_data_pipeline(
        datasource: LearningDataSource,
        *,
        batch_size: int,
        pos_weight: float = 1.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> StageDataPipeline:
        """Stage1 用のデータパイプラインを生成する．

        Args:
            datasource: 学習データソース．
            batch_size: バッチサイズ．
            pos_weight: 正例の重み．
            num_workers: DataLoader のワーカー数．
            pin_memory: ピンメモリを使用するかどうか．
            prefetch_factor: プリフェッチファクター．

        Returns:
            Stage1 のデータパイプライン．
        """
        train_ds, _ = datasource.train_test_split(
            test_ratio=0.0
        )
        raw_dataset = Stage1Dataset(datasource=train_ds)
        dataset = Stage1DatasetAdapter(raw_dataset)

        dl_kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=pre_stage_collate_fn,
        )
        if prefetch_factor is not None and num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
        dataloader = DataLoader(dataset, **dl_kwargs)

        loss_fn = ReachableSquaresLoss(pos_weight=pos_weight)
        return StageDataPipeline(
            train_dataloader=dataloader,
            val_dataloader=None,
            loss_fn=loss_fn,
        )

    @staticmethod
    def create_stage1_streaming_data_pipeline(
        streaming_source: StreamingDataSource,
        *,
        batch_size: int,
        pos_weight: float = 1.0,
    ) -> StageDataPipeline:
        """Stage1 用のストリーミングデータパイプラインを生成する．

        Args:
            streaming_source: ストリーミングデータソース．
            batch_size: バッチサイズ．
            pos_weight: 正例の重み．

        Returns:
            Stage1 のストリーミングデータパイプライン．
        """
        from maou.app.learning.streaming_dataset import (
            Stage1StreamingAdapter,
            StreamingStage1Dataset,
        )

        raw_dataset = StreamingStage1Dataset(
            streaming_source=streaming_source,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset = Stage1StreamingAdapter(raw_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        loss_fn = ReachableSquaresLoss(pos_weight=pos_weight)
        return StageDataPipeline(
            train_dataloader=dataloader,
            val_dataloader=None,
            loss_fn=loss_fn,
        )

    @staticmethod
    def create_stage2_data_pipeline(
        datasource: LearningDataSource,
        *,
        batch_size: int,
        pos_weight: float = 1.0,
        gamma_pos: float = 0.0,
        gamma_neg: float = 0.0,
        clip: float = 0.0,
        test_ratio: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> StageDataPipeline:
        """Stage2 用のデータパイプラインを生成する．

        Args:
            datasource: 学習データソース．
            batch_size: バッチサイズ．
            pos_weight: 正例の重み．
            gamma_pos: Asymmetric Focal Loss の正例ガンマ．
            gamma_neg: Asymmetric Focal Loss の負例ガンマ．
            clip: Asymmetric Focal Loss のクリップ値．
            test_ratio: テストデータの割合．
            num_workers: DataLoader のワーカー数．
            pin_memory: ピンメモリを使用するかどうか．
            prefetch_factor: プリフェッチファクター．

        Returns:
            Stage2 のデータパイプライン．
        """
        train_ds, val_ds = datasource.train_test_split(
            test_ratio=test_ratio
        )

        raw_train = Stage2Dataset(datasource=train_ds)
        train_dataset = Stage2DatasetAdapter(raw_train)
        dl_kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=pre_stage_collate_fn,
        )
        if prefetch_factor is not None and num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
        train_dataloader = DataLoader(
            train_dataset, **dl_kwargs
        )

        val_dataloader: Optional[DataLoader] = None
        if test_ratio > 0.0 and val_ds is not None:
            raw_val = Stage2Dataset(datasource=val_ds)
            val_dataset = Stage2DatasetAdapter(raw_val)
            val_dl_kwargs: dict = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=pre_stage_collate_fn,
            )
            if prefetch_factor is not None and num_workers > 0:
                val_dl_kwargs["prefetch_factor"] = (
                    prefetch_factor
                )
            val_dataloader = DataLoader(
                val_dataset, **val_dl_kwargs
            )

        loss_fn = LegalMovesLoss(
            pos_weight=pos_weight,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        return StageDataPipeline(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
        )

    @staticmethod
    def create_stage2_streaming_data_pipeline(
        streaming_source: StreamingDataSource,
        *,
        batch_size: int,
        pos_weight: float = 1.0,
        gamma_pos: float = 0.0,
        gamma_neg: float = 0.0,
        clip: float = 0.0,
        test_ratio: float = 0.0,
        dataloader_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ) -> StageDataPipeline:
        """Stage2 用のストリーミングデータパイプラインを生成する．

        Args:
            streaming_source: ストリーミングデータソース．
            batch_size: バッチサイズ．
            pos_weight: 正例の重み．
            gamma_pos: Asymmetric Focal Loss の正例ガンマ．
            gamma_neg: Asymmetric Focal Loss の負例ガンマ．
            clip: Asymmetric Focal Loss のクリップ値．
            test_ratio: テストデータの割合．
            dataloader_workers: DataLoader のワーカー数．
            pin_memory: ピンメモリを使用するかどうか．
            prefetch_factor: プリフェッチファクター．

        Returns:
            Stage2 のストリーミングデータパイプライン．
        """
        from torch.utils.data import IterableDataset

        from maou.app.learning.streaming_dataset import (
            Stage2StreamingAdapter,
            StreamingStage2Dataset,
        )

        raw_dataset = StreamingStage2Dataset(
            streaming_source=streaming_source,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset = Stage2StreamingAdapter(raw_dataset)

        _EmptyDataset = type(
            "_EmptyDataset",
            (IterableDataset,),
            {"__iter__": lambda self: iter([])},
        )
        train_dataloader, _ = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=dataset,
                val_dataset=_EmptyDataset(),
                dataloader_workers=dataloader_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                n_train_files=len(streaming_source.file_paths),
                n_val_files=0,
                file_paths=streaming_source.file_paths,
            )
        )

        loss_fn = LegalMovesLoss(
            pos_weight=pos_weight,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        return StageDataPipeline(
            train_dataloader=train_dataloader,
            val_dataloader=None,
            loss_fn=loss_fn,
        )

    @staticmethod
    def create_stage1_components(
        datasource: LearningDataSource,
        backbone: HeadlessNetwork,
        device: torch.device,
        *,
        batch_size: int,
        learning_rate: float,
        pos_weight: float = 1.0,
        lr_scheduler_name: Optional[str] = None,
        compilation: bool = False,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        total_epochs: int = 1,
    ) -> StageComponents:
        """Stage1 用のコンポーネント一式を生成する．

        Args:
            datasource: 学習データソース．
            backbone: バックボーンネットワーク．
            device: 使用デバイス．
            batch_size: バッチサイズ．
            learning_rate: 学習率．
            pos_weight: 正例の重み．
            lr_scheduler_name: 学習率スケジューラ名．
            compilation: コンパイルを有効にするかどうか．
            optimizer_name: オプティマイザ名．
            momentum: モメンタム．
            num_workers: DataLoader のワーカー数．
            pin_memory: ピンメモリを使用するかどうか．
            prefetch_factor: プリフェッチファクター．
            total_epochs: 総エポック数．

        Returns:
            Stage1 のコンポーネント一式．
        """
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                datasource,
                batch_size=batch_size,
                pos_weight=pos_weight,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )
        )

        from maou.app.learning.multi_stage_training import (
            Stage1ModelAdapter,
        )
        from maou.app.learning.network import (
            ReachableSquaresHead,
        )

        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        model: torch.nn.Module = Stage1ModelAdapter(
            backbone, head
        )
        model.to(device)

        optimizer = LossOptimizerFactory.create_optimizer(
            model,
            learning_rate,
            momentum,
            optimizer_name=optimizer_name,
        )
        lr_scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            lr_scheduler_name=lr_scheduler_name,
            max_epochs=total_epochs,
            steps_per_epoch=len(pipeline.train_dataloader),
        )
        return StageComponents(
            model=model,
            train_dataloader=pipeline.train_dataloader,
            val_dataloader=pipeline.val_dataloader,
            loss_fn=pipeline.loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    @staticmethod
    def create_stage2_components(
        datasource: LearningDataSource,
        backbone: HeadlessNetwork,
        device: torch.device,
        *,
        batch_size: int,
        learning_rate: float,
        pos_weight: float = 1.0,
        gamma_pos: float = 0.0,
        gamma_neg: float = 0.0,
        clip: float = 0.0,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.0,
        test_ratio: float = 0.0,
        lr_scheduler_name: Optional[str] = None,
        compilation: bool = False,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        total_epochs: int = 1,
    ) -> StageComponents:
        """Stage2 用のコンポーネント一式を生成する．

        Args:
            datasource: 学習データソース．
            backbone: バックボーンネットワーク．
            device: 使用デバイス．
            batch_size: バッチサイズ．
            learning_rate: 学習率．
            pos_weight: 正例の重み．
            gamma_pos: Asymmetric Focal Loss の正例ガンマ．
            gamma_neg: Asymmetric Focal Loss の負例ガンマ．
            clip: Asymmetric Focal Loss のクリップ値．
            head_hidden_dim: ヘッドの隠れ層次元数．
            head_dropout: ヘッドのドロップアウト率．
            test_ratio: テストデータの割合．
            lr_scheduler_name: 学習率スケジューラ名．
            compilation: コンパイルを有効にするかどうか．
            optimizer_name: オプティマイザ名．
            momentum: モメンタム．
            num_workers: DataLoader のワーカー数．
            pin_memory: ピンメモリを使用するかどうか．
            prefetch_factor: プリフェッチファクター．
            total_epochs: 総エポック数．

        Returns:
            Stage2 のコンポーネント一式．
        """
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                datasource,
                batch_size=batch_size,
                pos_weight=pos_weight,
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                clip=clip,
                test_ratio=test_ratio,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )
        )

        from maou.app.learning.multi_stage_training import (
            Stage2ModelAdapter,
        )
        from maou.app.learning.network import LegalMovesHead

        head = LegalMovesHead(
            input_dim=backbone.embedding_dim,
            hidden_dim=head_hidden_dim
            if head_hidden_dim and head_hidden_dim > 0
            else None,
            dropout=head_dropout,
        )
        model: torch.nn.Module = Stage2ModelAdapter(
            backbone, head
        )
        model.to(device)

        optimizer = LossOptimizerFactory.create_optimizer(
            model,
            learning_rate,
            momentum,
            optimizer_name=optimizer_name,
        )
        lr_scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            lr_scheduler_name=lr_scheduler_name,
            max_epochs=total_epochs,
            steps_per_epoch=len(pipeline.train_dataloader),
        )
        return StageComponents(
            model=model,
            train_dataloader=pipeline.train_dataloader,
            val_dataloader=pipeline.val_dataloader,
            loss_fn=pipeline.loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
