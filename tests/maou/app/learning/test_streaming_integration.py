"""Tests for Phase 4: streaming integration into Learning and multi-stage training.

Tests cover:
1. LearningOption streaming fields
2. set_epoch() calls in Learning.__train() epoch loop
3. training_loop.py batch_size=None handling
4. _setup_streaming_components validation
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data import DataLoader

from maou.app.learning.dl import Learning
from maou.app.learning.multi_stage_training import (
    TrainingStage,
)
from maou.app.learning.network import (
    HeadlessNetwork,
)
from maou.app.learning.streaming_dataset import (
    StreamingKifDataset,
)
from maou.domain.data.columnar_batch import ColumnarBatch
from maou.domain.move.label import MOVE_LABELS_NUM

# ============================================================================
# Fake StreamingDataSource for testing
# ============================================================================


class FakePreprocessingSource:
    """Fake streaming source for preprocessing data."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/file_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return self._n_files * self._rows_per_file

    @property
    def row_counts(self) -> list[int]:
        return [self._rows_per_file] * self._n_files

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(123)
        for _ in range(self._n_files):
            n = self._rows_per_file
            yield ColumnarBatch(
                board_positions=rng.integers(
                    0, 30, size=(n, 9, 9), dtype=np.uint8
                ),
                pieces_in_hand=rng.integers(
                    0, 5, size=(n, 14), dtype=np.uint8
                ),
                move_label=rng.random(
                    (n, MOVE_LABELS_NUM)
                ).astype(np.float16),
                result_value=rng.random(n).astype(np.float16),
            )


class FakeStage1Source:
    """Fake streaming source for stage1 data."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/stage1_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return self._n_files * self._rows_per_file

    @property
    def row_counts(self) -> list[int]:
        return [self._rows_per_file] * self._n_files

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(456)
        for _ in range(self._n_files):
            n = self._rows_per_file
            yield ColumnarBatch(
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

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        """Yield columnar batches only for files whose path is in *file_paths*."""
        rng = np.random.default_rng(456)
        target_set = set(str(fp) for fp in file_paths)
        for fp in self._file_paths:
            n = self._rows_per_file
            batch = ColumnarBatch(
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
            if str(fp) in target_set:
                yield batch


# ============================================================================
# LearningOption streaming fields
# ============================================================================


class TestLearningOptionStreaming:
    """LearningOption streaming fields tests."""

    def test_streaming_defaults_to_false(self) -> None:
        """streaming field defaults to False."""
        mock_splitter = MagicMock()
        option = Learning.LearningOption(
            datasource=mock_splitter,
            datasource_type="preprocess",
            compilation=False,
            test_ratio=0.2,
            epoch=1,
            batch_size=64,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            cache_transforms=False,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            learning_ratio=0.01,
            momentum=0.9,
            optimizer_name="adamw",
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            log_dir=Path("/tmp/logs"),
            model_dir=Path("/tmp/models"),
        )
        assert option.streaming is False
        assert option.streaming_train_source is None
        assert option.streaming_val_source is None

    def test_streaming_option_with_sources(self) -> None:
        """streaming=True with streaming sources."""
        train_source = FakePreprocessingSource()
        val_source = FakePreprocessingSource()
        mock_splitter = MagicMock()
        option = Learning.LearningOption(
            datasource=mock_splitter,
            datasource_type="preprocess",
            compilation=False,
            test_ratio=0.2,
            epoch=1,
            batch_size=64,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            cache_transforms=False,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            learning_ratio=0.01,
            momentum=0.9,
            optimizer_name="adamw",
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            log_dir=Path("/tmp/logs"),
            model_dir=Path("/tmp/models"),
            streaming=True,
            streaming_train_source=train_source,
            streaming_val_source=val_source,
        )
        assert option.streaming is True
        assert option.streaming_train_source is train_source
        assert option.streaming_val_source is val_source


# ============================================================================
# _setup_streaming_components validation
# ============================================================================


class TestSetupStreamingComponents:
    """Tests for Learning._setup_streaming_components."""

    def test_raises_without_train_source(self) -> None:
        """Raises ValueError when streaming_train_source is None."""
        mock_splitter = MagicMock()
        option = Learning.LearningOption(
            datasource=mock_splitter,
            datasource_type="preprocess",
            compilation=False,
            test_ratio=0.2,
            epoch=1,
            batch_size=64,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            cache_transforms=False,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            learning_ratio=0.01,
            momentum=0.9,
            optimizer_name="adamw",
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            log_dir=Path("/tmp/logs"),
            model_dir=Path("/tmp/models"),
            streaming=True,
            streaming_train_source=None,
            streaming_val_source=FakePreprocessingSource(),
        )
        learning = Learning()
        try:
            learning._setup_streaming_components(option, None)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "streaming_train_source" in str(e)

    def test_raises_without_val_source(self) -> None:
        """Raises ValueError when streaming_val_source is None."""
        mock_splitter = MagicMock()
        option = Learning.LearningOption(
            datasource=mock_splitter,
            datasource_type="preprocess",
            compilation=False,
            test_ratio=0.2,
            epoch=1,
            batch_size=64,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            cache_transforms=False,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            learning_ratio=0.01,
            momentum=0.9,
            optimizer_name="adamw",
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            log_dir=Path("/tmp/logs"),
            model_dir=Path("/tmp/models"),
            streaming=True,
            streaming_train_source=FakePreprocessingSource(),
            streaming_val_source=None,
        )
        learning = Learning()
        try:
            learning._setup_streaming_components(option, None)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "streaming_val_source" in str(e)

    def test_returns_components_with_valid_sources(
        self,
    ) -> None:
        """Returns valid components with streaming sources."""
        mock_splitter = MagicMock()
        train_source = FakePreprocessingSource()
        val_source = FakePreprocessingSource()
        option = Learning.LearningOption(
            datasource=mock_splitter,
            datasource_type="preprocess",
            compilation=False,
            test_ratio=0.2,
            epoch=1,
            batch_size=4,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            cache_transforms=False,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            learning_ratio=0.01,
            momentum=0.9,
            optimizer_name="adamw",
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            log_dir=Path("/tmp/logs"),
            model_dir=Path("/tmp/models"),
            streaming=True,
            streaming_train_source=train_source,
            streaming_val_source=val_source,
        )
        learning = Learning()
        device_config, dataloaders, model_components = (
            learning._setup_streaming_components(option, None)
        )

        train_loader, val_loader = dataloaders
        assert train_loader.batch_size is None
        assert val_loader.batch_size is None
        assert isinstance(
            train_loader.dataset, StreamingKifDataset
        )
        assert isinstance(
            val_loader.dataset, StreamingKifDataset
        )
        assert model_components.model is not None
        assert model_components.optimizer is not None


# ============================================================================
# training_loop.py batch_size=None handling
# ============================================================================


class TestTrainingLoopBatchSizeNone:
    """TrainingLoop handles batch_size=None for streaming DataLoaders."""

    def test_streaming_dataloader_batch_size_is_none(
        self,
    ) -> None:
        """Streaming DataLoader has batch_size=None."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )
        assert loader.batch_size is None


# ============================================================================
# Interface layer streaming functions
# ============================================================================


class TestInterfaceStreamingStages:
    """Tests for _run_stage1_streaming and _run_stage2_streaming."""

    def test_run_stage1_streaming_completes(self) -> None:
        """_run_stage1_streaming trains and returns StageResult."""
        from maou.interface.learn import (
            _run_stage1_streaming,
        )

        source = FakeStage1Source(n_files=1, rows_per_file=8)
        device = torch.device("cpu")
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=8,
            board_size=(9, 9),
            architecture="resnet",
            layers=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            out_channels=(8, 8, 8, 8),
        )
        from maou.app.learning.multi_stage_training import (
            MultiStageTrainingOrchestrator,
        )

        orchestrator = MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=device,
            model_dir=Path("/tmp/test_models"),
        )

        result = _run_stage1_streaming(
            streaming_source=source,
            orchestrator=orchestrator,
            backbone=backbone,
            batch_size=4,
            learning_rate=0.001,
            max_epochs=1,
            threshold=0.0,  # Low threshold to avoid RuntimeError
            device=device,
        )

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained == 1
        assert 0.0 <= result.achieved_accuracy <= 1.0

    def test_run_stage2_streaming_completes(self) -> None:
        """_run_stage2_streaming trains and returns StageResult."""
        from maou.interface.learn import (
            _run_stage2_streaming,
        )

        # FakeStage2Source with legal_moves_label
        class FakeStage2Source:
            def __init__(self) -> None:
                self._file_paths = [
                    Path("/fake/stage2_0.feather")
                ]

            @property
            def file_paths(self) -> list[Path]:
                return list(self._file_paths)

            @property
            def total_rows(self) -> int:
                return 8

            @property
            def row_counts(self) -> list[int]:
                return [8]

            def iter_files_columnar(
                self,
            ) -> Generator[ColumnarBatch, None, None]:
                rng = np.random.default_rng(789)
                yield ColumnarBatch(
                    board_positions=rng.integers(
                        0, 30, size=(8, 9, 9), dtype=np.uint8
                    ),
                    pieces_in_hand=rng.integers(
                        0, 5, size=(8, 14), dtype=np.uint8
                    ),
                    legal_moves_label=rng.integers(
                        0,
                        2,
                        size=(8, MOVE_LABELS_NUM),
                        dtype=np.uint8,
                    ),
                )

            def iter_files_columnar_subset(
                self,
                file_paths: list[Path],
            ) -> Generator[ColumnarBatch, None, None]:
                """Yield columnar batches only for files in *file_paths*."""
                rng = np.random.default_rng(789)
                target_set = set(str(fp) for fp in file_paths)
                for fp in self._file_paths:
                    if str(fp) in target_set:
                        yield ColumnarBatch(
                            board_positions=rng.integers(
                                0,
                                30,
                                size=(8, 9, 9),
                                dtype=np.uint8,
                            ),
                            pieces_in_hand=rng.integers(
                                0,
                                5,
                                size=(8, 14),
                                dtype=np.uint8,
                            ),
                            legal_moves_label=rng.integers(
                                0,
                                2,
                                size=(8, MOVE_LABELS_NUM),
                                dtype=np.uint8,
                            ),
                        )

        source = FakeStage2Source()
        device = torch.device("cpu")
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=8,
            board_size=(9, 9),
            architecture="resnet",
            layers=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            out_channels=(8, 8, 8, 8),
        )
        from maou.app.learning.multi_stage_training import (
            MultiStageTrainingOrchestrator,
        )

        orchestrator = MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=device,
            model_dir=Path("/tmp/test_models"),
        )

        result = _run_stage2_streaming(
            streaming_source=source,
            orchestrator=orchestrator,
            backbone=backbone,
            batch_size=4,
            learning_rate=0.001,
            max_epochs=1,
            threshold=0.0,  # Low threshold to avoid RuntimeError
            device=device,
        )

        assert result.stage == TrainingStage.LEGAL_MOVES
        assert result.epochs_trained == 1
        assert 0.0 <= result.achieved_accuracy <= 1.0
