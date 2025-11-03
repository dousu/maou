import numpy as np
import pytest
import torch

from maou.app.learning.dataset import DataSource
from maou.app.learning.setup import (
    TrainingSetup,
    WarmupCosineDecayScheduler,
)
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.mlp_mixer import ShogiMLPMixer


class DummyPreprocessedDataSource(DataSource):
    def __init__(self, length: int = 4) -> None:
        dtype = np.dtype(
            [
                ("features", np.float32, (FEATURES_NUM, 9, 9)),
                (
                    "legalMoveMask",
                    np.float32,
                    (MOVE_LABELS_NUM,),
                ),
                ("moveLabel", np.float32, (MOVE_LABELS_NUM,)),
                ("resultValue", np.float32),
            ]
        )
        self._data = np.zeros(length, dtype=dtype)
        self._data["legalMoveMask"] = 1.0
        self._data["moveLabel"] = 1.0

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def test_training_setup_uses_adamw_optimizer() -> None:
    datasource = DummyPreprocessedDataSource(length=4)

    _, _, model_components = (
        TrainingSetup.setup_training_components(
            training_datasource=datasource,
            validation_datasource=datasource,
            datasource_type="preprocess",
            gpu="cpu",
            batch_size=2,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            optimizer_name="adamw",
            optimizer_beta1=0.85,
            optimizer_beta2=0.98,
            optimizer_eps=1e-7,
        )
    )

    optimizer = model_components.optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.85, 0.98)
    assert optimizer.defaults["eps"] == 1e-07
    assert model_components.lr_scheduler is None


def test_training_setup_supports_mlp_mixer_backbone() -> None:
    datasource = DummyPreprocessedDataSource(length=4)

    _, _, model_components = (
        TrainingSetup.setup_training_components(
            training_datasource=datasource,
            validation_datasource=datasource,
            datasource_type="preprocess",
            gpu="cpu",
            batch_size=2,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            optimizer_name="adamw",
            optimizer_beta1=0.85,
            optimizer_beta2=0.98,
            optimizer_eps=1e-7,
            model_architecture="mlp-mixer",
        )
    )

    assert isinstance(
        model_components.model.backbone, ShogiMLPMixer
    )
    assert model_components.lr_scheduler is None


def test_training_setup_creates_warmup_cosine_decay_scheduler() -> (
    None
):
    datasource = DummyPreprocessedDataSource(length=8)

    _, _, model_components = (
        TrainingSetup.setup_training_components(
            training_datasource=datasource,
            validation_datasource=datasource,
            datasource_type="preprocess",
            gpu="cpu",
            batch_size=2,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            optimizer_name="adamw",
            optimizer_beta1=0.85,
            optimizer_beta2=0.98,
            optimizer_eps=1e-7,
            lr_scheduler_name="warmup_cosine_decay",
            max_epochs=20,
        )
    )

    scheduler = model_components.lr_scheduler
    assert isinstance(scheduler, WarmupCosineDecayScheduler)

    optimizer = model_components.optimizer
    base_lr = scheduler.base_lrs[0]
    initial_lr = optimizer.param_groups[0]["lr"]

    assert initial_lr == pytest.approx(base_lr * 0.5)

    optimizer.step()
    scheduler.step()
    warmup_completed_lr = scheduler.get_last_lr()[0]

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    decay_lr = scheduler.get_last_lr()[0]

    assert warmup_completed_lr == pytest.approx(base_lr)
    assert decay_lr < base_lr


def test_training_setup_creates_cosine_annealing_scheduler() -> (
    None
):
    datasource = DummyPreprocessedDataSource(length=8)

    _, _, model_components = (
        TrainingSetup.setup_training_components(
            training_datasource=datasource,
            validation_datasource=datasource,
            datasource_type="preprocess",
            gpu="cpu",
            batch_size=2,
            dataloader_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            optimizer_name="adamw",
            optimizer_beta1=0.85,
            optimizer_beta2=0.98,
            optimizer_eps=1e-7,
            lr_scheduler_name="cosine_annealing_lr",
            max_epochs=5,
        )
    )

    scheduler = model_components.lr_scheduler
    assert isinstance(
        scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert scheduler.T_max == 5
