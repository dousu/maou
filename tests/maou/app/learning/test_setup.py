import logging
import numpy as np
import pytest
import torch

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.setup import (
    DatasetFactory,
    TrainingSetup,
    WarmupCosineDecayScheduler,
)
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.model.mlp_mixer import ShogiMLPMixer

class DummyPreprocessedDataSource(DataSource):
    def __init__(self, length: int = 4) -> None:
        dtype = np.dtype(
            [
                ("boardIdPositions", np.uint8, (9, 9)),
                ("piecesInHand", np.uint8, (14,)),
                ("moveLabel", np.float16, (MOVE_LABELS_NUM,)),
                ("resultValue", np.float16),
            ]
        )
        self._data = np.zeros(length, dtype=dtype)
        self._data["boardIdPositions"] = np.eye(9, dtype=np.uint8)
        self._data["moveLabel"] = np.float16(1.0)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)




class DummyHCPEDataSource(DataSource):
    def __init__(self, length: int = 4) -> None:
        dtype = np.dtype(
            [
                ("hcp", np.uint8, (2,)),
                ("bestMove16", np.uint16),
                ("gameResult", np.uint8),
                ("eval", np.int16),
            ]
        )
        self._data = np.zeros(length, dtype=dtype)
        self._data["bestMove16"] = np.arange(length, dtype=np.uint16)
        self._data["gameResult"] = np.ones(length, dtype=np.uint8)
        self._data["eval"] = np.zeros(length, dtype=np.int16)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class RecordingTransform:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(
        self, *, hcp: np.ndarray, move16: int, game_result: int, eval: int
    ) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
        self.calls.append(int(move16))
        board = np.full((9, 9), move16, dtype=np.uint8)
        pieces = np.full((14,), move16, dtype=np.uint8)
        move_label = int(move16 % MOVE_LABELS_NUM)
        result_value = float(game_result)
        legal_move_mask = np.zeros((MOVE_LABELS_NUM,), dtype=np.uint8)
        legal_move_mask[move_label] = 1
        return board, pieces, move_label, result_value, legal_move_mask
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


def test_kifdataset_caches_transform_results_when_enabled(caplog: pytest.LogCaptureFixture) -> None:
    datasource = DummyHCPEDataSource(length=3)
    transform = RecordingTransform()
    with caplog.at_level(logging.INFO, logger=KifDataset.logger.name):
        dataset = KifDataset(
            datasource=datasource,
            transform=transform,
            cache_transforms=True,
        )

    assert len(transform.calls) == len(datasource)
    first_sample = dataset[0]
    second_sample = dataset[0]
    assert first_sample[0][0] is second_sample[0][0]
    assert first_sample[0][1] is second_sample[0][1]
    assert first_sample[1][0] is second_sample[1][0]
    assert first_sample[1][1] is second_sample[1][1]
    assert first_sample[1][2] is second_sample[1][2]
    assert torch.equal(first_sample[0][0], second_sample[0][0])
    assert torch.equal(first_sample[1][0], second_sample[1][0])
    assert len(transform.calls) == len(datasource)


def test_training_setup_enables_cache_for_hcpe_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_flags: list[bool] = []

    class _StubDataset:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            board = torch.zeros((9, 9), dtype=torch.long)
            pieces = torch.zeros((14,), dtype=torch.float32)
            move_label = torch.zeros((MOVE_LABELS_NUM,), dtype=torch.float32)
            result_value = torch.zeros((1,), dtype=torch.float32)
            legal_mask = torch.zeros((MOVE_LABELS_NUM,), dtype=torch.float32)
            return (board, pieces), (move_label, result_value, legal_mask)

    def fake_create_datasets(
        cls,
        training_datasource: DataSource,
        validation_datasource: DataSource,
        datasource_type: str,
        cache_transforms: bool = False,
    ) -> tuple[_StubDataset, _StubDataset]:
        cache_flags.append(cache_transforms)
        stub = _StubDataset()
        return stub, stub

    monkeypatch.setattr(
        DatasetFactory,
        "create_datasets",
        classmethod(fake_create_datasets),
    )

    datasource = DummyPreprocessedDataSource(length=4)
    TrainingSetup.setup_training_components(
        training_datasource=datasource,
        validation_datasource=datasource,
        datasource_type="hcpe",
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

    assert cache_flags == [True]


def test_training_setup_respects_cache_transforms_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_flags: list[bool] = []

    class _StubDataset:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            board = torch.zeros((9, 9), dtype=torch.long)
            pieces = torch.zeros((14,), dtype=torch.float32)
            move_label = torch.zeros((MOVE_LABELS_NUM,), dtype=torch.float32)
            result_value = torch.zeros((1,), dtype=torch.float32)
            legal_mask = torch.zeros((MOVE_LABELS_NUM,), dtype=torch.float32)
            return (board, pieces), (move_label, result_value, legal_mask)

    def fake_create_datasets(
        cls,
        training_datasource: DataSource,
        validation_datasource: DataSource,
        datasource_type: str,
        cache_transforms: bool = False,
    ) -> tuple[_StubDataset, _StubDataset]:
        cache_flags.append(cache_transforms)
        stub = _StubDataset()
        return stub, stub

    monkeypatch.setattr(
        DatasetFactory,
        "create_datasets",
        classmethod(fake_create_datasets),
    )

    datasource = DummyPreprocessedDataSource(length=4)
    TrainingSetup.setup_training_components(
        training_datasource=datasource,
        validation_datasource=datasource,
        datasource_type="hcpe",
        cache_transforms=False,
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

    assert cache_flags == [False]


def test_training_setup_disables_anomaly_detection_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_set_detect_anomaly(*, mode: bool, check_nan: bool) -> None:
        calls.append((mode, check_nan))

    monkeypatch.setattr(
        torch.autograd,
        "set_detect_anomaly",
        fake_set_detect_anomaly,
    )

    datasource = DummyPreprocessedDataSource(length=4)
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

    assert calls == []


def test_training_setup_enables_anomaly_detection_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_set_detect_anomaly(*, mode: bool, check_nan: bool) -> None:
        calls.append((mode, check_nan))

    monkeypatch.setattr(
        torch.autograd,
        "set_detect_anomaly",
        fake_set_detect_anomaly,
    )

    datasource = DummyPreprocessedDataSource(length=4)
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
        detect_anomaly=True,
    )

    assert calls == [(True, True)]
