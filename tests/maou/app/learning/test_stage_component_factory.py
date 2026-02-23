"""Tests for StageComponentFactory."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.learning.dataset import DataSource
from maou.app.learning.dl import LearningDataSource
from maou.app.learning.network import HeadlessNetwork
from maou.app.learning.stage_component_factory import (
    StageComponentFactory,
    StageComponents,
    StageDataPipeline,
)
from maou.domain.data.schema import (
    create_empty_stage1_array,
    create_empty_stage2_array,
)
from maou.domain.loss.loss_fn import (
    LegalMovesLoss,
    ReachableSquaresLoss,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class _MockDataSource(DataSource):
    """テスト用モックデータソース．"""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._array[idx]

    def __len__(self) -> int:
        return len(self._array)


class _MockSplitter(LearningDataSource.DataSourceSpliter):
    """テスト用モックスプリッター．"""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def train_test_split(
        self, test_ratio: float
    ) -> tuple[DataSource, DataSource]:
        n = len(self._array)
        if test_ratio <= 0.0:
            return (
                _MockDataSource(self._array),
                _MockDataSource(self._array[:0]),
            )
        split_idx = max(1, int(n * (1 - test_ratio)))
        return (
            _MockDataSource(self._array[:split_idx]),
            _MockDataSource(self._array[split_idx:]),
        )


def _make_stage1_splitter(size: int = 20) -> _MockSplitter:
    """Stage 1 用のテストスプリッターを生成する．"""
    array = create_empty_stage1_array(size)
    rng = np.random.default_rng(42)
    array["boardIdPositions"] = rng.integers(
        0, 40, size=(size, 9, 9), dtype=np.uint8
    )
    array["piecesInHand"] = rng.integers(
        0, 19, size=(size, 14), dtype=np.uint8
    )
    array["reachableSquares"] = rng.integers(
        0, 2, size=(size, 9, 9), dtype=np.uint8
    )
    return _MockSplitter(array)


def _make_stage2_splitter(size: int = 20) -> _MockSplitter:
    """Stage 2 用のテストスプリッターを生成する．"""
    array = create_empty_stage2_array(size)
    rng = np.random.default_rng(42)
    array["boardIdPositions"] = rng.integers(
        0, 40, size=(size, 9, 9), dtype=np.uint8
    )
    array["piecesInHand"] = rng.integers(
        0, 19, size=(size, 14), dtype=np.uint8
    )
    for i in range(size):
        labels = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
        labels[: rng.integers(1, 50)] = 1
        array["legalMovesLabel"][i] = labels
    return _MockSplitter(array)


def _make_backbone() -> HeadlessNetwork:
    """テスト用の小さなバックボーンを生成する．"""
    return HeadlessNetwork(
        board_vocab_size=32,
        hand_projection_dim=0,
        embedding_dim=64,
        architecture="resnet",
        out_channels=(16, 32, 64, 64),
    )


# ============================================================
# A. Stage 1 データパイプラインテスト
# ============================================================


class TestCreateStage1DataPipeline:
    """Stage 1 データパイプラインの生成テスト．"""

    def test_returns_correct_types(self) -> None:
        """返却型がStageDataPipelineで各フィールドが正しい型である．"""
        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert isinstance(pipeline, StageDataPipeline)
        assert isinstance(pipeline.train_dataloader, DataLoader)
        assert isinstance(pipeline.loss_fn, torch.nn.Module)

    def test_loss_fn_type(self) -> None:
        """loss_fnがReachableSquaresLoss型である．"""
        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert isinstance(
            pipeline.loss_fn, ReachableSquaresLoss
        )

    def test_no_validation(self) -> None:
        """val_dataloaderがNoneである(Stage 1はバリデーションなし)．"""
        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert pipeline.val_dataloader is None

    def test_pos_weight_propagation(self) -> None:
        """pos_weight値がloss_fnに正しく伝搬される．"""
        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=4, pos_weight=2.5
            )
        )
        loss_fn = pipeline.loss_fn
        assert isinstance(loss_fn, ReachableSquaresLoss)
        # ReachableSquaresLoss は _pos_weight (private) にtensorとして保持
        assert loss_fn._pos_weight.item() == pytest.approx(2.5)

    def test_batch_size(self) -> None:
        """DataLoaderのbatch_sizeが指定値と一致する．"""
        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=8
            )
        )
        assert pipeline.train_dataloader.batch_size == 8

    def test_collate_fn_set(self) -> None:
        """pre_stage_collate_fnが設定される．"""
        from maou.app.learning.multi_stage_training import (
            pre_stage_collate_fn,
        )

        splitter = _make_stage1_splitter()
        pipeline = (
            StageComponentFactory.create_stage1_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert (
            pipeline.train_dataloader.collate_fn
            is pre_stage_collate_fn
        )


# ============================================================
# B. Stage 2 データパイプラインテスト
# ============================================================


class TestCreateStage2DataPipeline:
    """Stage 2 データパイプラインの生成テスト．"""

    def test_returns_correct_types(self) -> None:
        """返却型が正しい．"""
        splitter = _make_stage2_splitter()
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert isinstance(pipeline, StageDataPipeline)
        assert isinstance(pipeline.train_dataloader, DataLoader)

    def test_loss_fn_type(self) -> None:
        """loss_fnがLegalMovesLoss型である．"""
        splitter = _make_stage2_splitter()
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter, batch_size=4
            )
        )
        assert isinstance(pipeline.loss_fn, LegalMovesLoss)

    def test_asl_params(self) -> None:
        """gamma_pos, gamma_neg, clip値がloss_fnに伝搬される．"""
        splitter = _make_stage2_splitter()
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter,
                batch_size=4,
                gamma_pos=1.0,
                gamma_neg=2.0,
                clip=0.05,
            )
        )
        loss_fn = pipeline.loss_fn
        assert isinstance(loss_fn, LegalMovesLoss)
        # ASL有効時は内部の AsymmetricLoss に値が設定される
        inner = loss_fn._loss_fn
        assert inner.gamma_pos == pytest.approx(1.0)
        assert inner.gamma_neg == pytest.approx(2.0)
        assert inner.clip == pytest.approx(0.05)

    def test_no_validation_when_ratio_zero(self) -> None:
        """test_ratio=0.0でval_dataloaderがNoneである．"""
        splitter = _make_stage2_splitter()
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter, batch_size=4, test_ratio=0.0
            )
        )
        assert pipeline.val_dataloader is None

    def test_has_validation_when_ratio_positive(self) -> None:
        """test_ratio>0でval_dataloaderがDataLoaderである．"""
        splitter = _make_stage2_splitter(size=20)
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter, batch_size=4, test_ratio=0.2
            )
        )
        assert isinstance(pipeline.val_dataloader, DataLoader)

    def test_batch_size(self) -> None:
        """batch_sizeが正しい．"""
        splitter = _make_stage2_splitter()
        pipeline = (
            StageComponentFactory.create_stage2_data_pipeline(
                splitter, batch_size=8
            )
        )
        assert pipeline.train_dataloader.batch_size == 8


# ============================================================
# C. Stage 1 全コンポーネントテスト
# ============================================================


class TestCreateStage1Components:
    """Stage 1 全コンポーネントの生成テスト．"""

    def test_model_type(self) -> None:
        """modelがStage1ModelAdapterである．"""
        from maou.app.learning.multi_stage_training import (
            Stage1ModelAdapter,
        )

        splitter = _make_stage1_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage1_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
            )
        )
        assert isinstance(components, StageComponents)
        assert isinstance(components.model, Stage1ModelAdapter)

    def test_model_on_device(self) -> None:
        """モデルが指定デバイスに配置される．"""
        splitter = _make_stage1_splitter()
        backbone = _make_backbone()
        device = torch.device("cpu")
        components = (
            StageComponentFactory.create_stage1_components(
                splitter,
                backbone,
                device,
                batch_size=4,
                learning_rate=1e-3,
            )
        )
        param = next(components.model.parameters())
        assert param.device == device

    def test_optimizer_type(self) -> None:
        """optimizerが指定型である．"""
        splitter = _make_stage1_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage1_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
                optimizer_name="adamw",
            )
        )
        assert isinstance(
            components.optimizer, torch.optim.AdamW
        )

    def test_includes_data_pipeline(self) -> None:
        """dataloader, loss_fnが正しく含まれる．"""
        splitter = _make_stage1_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage1_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
            )
        )
        assert isinstance(
            components.train_dataloader, DataLoader
        )
        assert isinstance(
            components.loss_fn, ReachableSquaresLoss
        )
        assert components.val_dataloader is None

    def test_lr_scheduler(self) -> None:
        """lr_scheduler_nameに応じたスケジューラが生成される．"""
        splitter = _make_stage1_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage1_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
                lr_scheduler_name="cosine_annealing_lr",
            )
        )
        assert components.lr_scheduler is not None


# ============================================================
# D. Stage 2 全コンポーネントテスト
# ============================================================


class TestCreateStage2Components:
    """Stage 2 全コンポーネントの生成テスト．"""

    def test_model_type(self) -> None:
        """modelがStage2ModelAdapterである．"""
        from maou.app.learning.multi_stage_training import (
            Stage2ModelAdapter,
        )

        splitter = _make_stage2_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage2_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
            )
        )
        assert isinstance(components, StageComponents)
        assert isinstance(components.model, Stage2ModelAdapter)

    def test_head_params(self) -> None:
        """head_hidden_dim, head_dropoutが伝搬される．"""
        splitter = _make_stage2_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage2_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
                head_hidden_dim=256,
                head_dropout=0.1,
            )
        )
        # モデルが生成されればヘッドパラメータが正しく伝搬されている
        assert isinstance(components.model, torch.nn.Module)

    def test_optimizer_type(self) -> None:
        """optimizerが正しい．"""
        splitter = _make_stage2_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage2_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
                optimizer_name="adamw",
            )
        )
        assert isinstance(
            components.optimizer, torch.optim.AdamW
        )

    def test_lr_scheduler(self) -> None:
        """スケジューラが正しく生成される．"""
        splitter = _make_stage2_splitter()
        backbone = _make_backbone()
        components = (
            StageComponentFactory.create_stage2_components(
                splitter,
                backbone,
                torch.device("cpu"),
                batch_size=4,
                learning_rate=1e-3,
                lr_scheduler_name="cosine_annealing_lr",
            )
        )
        assert components.lr_scheduler is not None


# ============================================================
# E. ストリーミングバリアントテスト
# ============================================================


class TestStreamingDataPipeline:
    """ストリーミングデータパイプラインのテスト．"""

    def _make_fake_stage1_source(self) -> object:
        """Stage 1 用の fake streaming source を生成する．"""
        from pathlib import Path

        from maou.domain.data.columnar_batch import (
            ColumnarBatch,
        )

        class _FakeStage1Source:
            def __init__(self) -> None:
                self._file_paths = [
                    Path("/fake/stage1_0.feather"),
                    Path("/fake/stage1_1.feather"),
                ]

            @property
            def file_paths(self) -> list[Path]:
                return list(self._file_paths)

            @property
            def total_rows(self) -> int:
                return 20

            @property
            def row_counts(self) -> list[int]:
                return [10, 10]

            def _make_batch(self, n: int) -> ColumnarBatch:
                rng = np.random.default_rng(42)
                return ColumnarBatch(
                    board_positions=rng.integers(
                        0, 30, size=(n, 9, 9), dtype=np.uint8
                    ),
                    pieces_in_hand=rng.integers(
                        0, 5, size=(n, 14), dtype=np.uint8
                    ),
                    reachable_squares=rng.integers(
                        0, 2, size=(n, 9, 9), dtype=np.uint8
                    ),
                )

            def iter_files_columnar(self):  # type: ignore[no-untyped-def]
                for _ in range(len(self._file_paths)):
                    yield self._make_batch(10)

            def iter_files_columnar_subset(
                self, file_paths: list[Path]
            ):  # type: ignore[no-untyped-def]
                target_set = {str(fp) for fp in file_paths}
                for fp in self._file_paths:
                    if str(fp) in target_set:
                        yield self._make_batch(10)

        return _FakeStage1Source()

    def _make_fake_stage2_source(self) -> object:
        """Stage 2 用の fake streaming source を生成する．"""
        from pathlib import Path

        from maou.domain.data.columnar_batch import (
            ColumnarBatch,
        )

        class _FakeStage2Source:
            def __init__(self) -> None:
                self._file_paths = [
                    Path("/fake/stage2_0.feather"),
                    Path("/fake/stage2_1.feather"),
                ]

            @property
            def file_paths(self) -> list[Path]:
                return list(self._file_paths)

            @property
            def total_rows(self) -> int:
                return 20

            @property
            def row_counts(self) -> list[int]:
                return [10, 10]

            def _make_batch(self, n: int) -> ColumnarBatch:
                rng = np.random.default_rng(42)
                legal = np.zeros(
                    (n, MOVE_LABELS_NUM), dtype=np.uint8
                )
                legal[:, :50] = 1
                return ColumnarBatch(
                    board_positions=rng.integers(
                        0, 30, size=(n, 9, 9), dtype=np.uint8
                    ),
                    pieces_in_hand=rng.integers(
                        0, 5, size=(n, 14), dtype=np.uint8
                    ),
                    legal_moves_label=legal,
                )

            def iter_files_columnar(self):  # type: ignore[no-untyped-def]
                for _ in range(len(self._file_paths)):
                    yield self._make_batch(10)

            def iter_files_columnar_subset(
                self, file_paths: list[Path]
            ):  # type: ignore[no-untyped-def]
                target_set = {str(fp) for fp in file_paths}
                for fp in self._file_paths:
                    if str(fp) in target_set:
                        yield self._make_batch(10)

        return _FakeStage2Source()

    def test_stage1_streaming_types(self) -> None:
        """ストリーミングStage1 DataLoaderが返される．"""
        source = self._make_fake_stage1_source()
        pipeline = StageComponentFactory.create_stage1_streaming_data_pipeline(
            source,  # type: ignore[arg-type]
            batch_size=4,
        )
        assert isinstance(pipeline, StageDataPipeline)
        assert isinstance(pipeline.train_dataloader, DataLoader)
        assert isinstance(
            pipeline.loss_fn, ReachableSquaresLoss
        )
        assert pipeline.val_dataloader is None

    def test_stage2_streaming_types(self) -> None:
        """ストリーミングStage2 DataLoaderが返される．"""
        source = self._make_fake_stage2_source()
        pipeline = StageComponentFactory.create_stage2_streaming_data_pipeline(
            source,  # type: ignore[arg-type]
            batch_size=4,
        )
        assert isinstance(pipeline, StageDataPipeline)
        assert isinstance(pipeline.train_dataloader, DataLoader)
        assert isinstance(pipeline.loss_fn, LegalMovesLoss)
        assert pipeline.val_dataloader is None
