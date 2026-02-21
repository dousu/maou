import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.setup import (
    DataLoaderFactory,
    DatasetFactory,
    ModelFactory,
    TrainingSetup,
    WarmupCosineDecayScheduler,
    _estimate_max_workers_by_memory,
)
from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.move.label import MOVE_LABELS_NUM


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
        self._data["boardIdPositions"] = np.eye(
            9, dtype=np.uint8
        )
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
        self._data["bestMove16"] = np.arange(
            length, dtype=np.uint16
        )
        self._data["gameResult"] = np.ones(
            length, dtype=np.uint8
        )
        self._data["eval"] = np.zeros(length, dtype=np.int16)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class RecordingTransform:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(
        self,
        *,
        hcp: np.ndarray,
        move16: int,
        game_result: int,
        eval: int,
    ) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
        self.calls.append(int(move16))
        board = np.full((9, 9), move16, dtype=np.uint8)
        pieces = np.full((14,), move16, dtype=np.uint8)
        move_label = int(move16 % MOVE_LABELS_NUM)
        result_value = float(game_result)
        legal_move_mask = np.zeros(
            (MOVE_LABELS_NUM,), dtype=np.uint8
        )
        legal_move_mask[move_label] = 1
        return (
            board,
            pieces,
            move_label,
            result_value,
            legal_move_mask,
        )


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

    # Per-step scheduler: steps_per_epoch=4, total_steps=80, warmup_steps=8
    optimizer = model_components.optimizer
    base_lr = scheduler.base_lrs[0]
    initial_lr = optimizer.param_groups[0]["lr"]

    # Step 0: warmup_progress = 1/8 = 0.125
    assert initial_lr == pytest.approx(base_lr * (1 / 8))
    assert initial_lr < base_lr  # Still in warmup

    # Advance through warmup (7 more steps to complete 8 warmup steps)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    warmup_completed_lr = scheduler.get_last_lr()[0]

    # Step 7 (last warmup): warmup_progress = 8/8 = 1.0
    assert warmup_completed_lr == pytest.approx(base_lr)

    # Advance into decay phase
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    decay_lr = scheduler.get_last_lr()[0]

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
    # Per-step: T_max = max_epochs * steps_per_epoch = 5 * 4 = 20
    assert scheduler.T_max == 20


def test_kifdataset_caches_transform_results_when_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    datasource = DummyHCPEDataSource(length=3)
    transform = RecordingTransform()
    with caplog.at_level(
        logging.INFO, logger=KifDataset.logger.name
    ):
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


def test_training_setup_enables_cache_for_hcpe_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_flags: list[bool] = []

    class _StubDataset:
        def __len__(self) -> int:
            return 4

        def __getitem__(
            self, idx: int
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]:
            board = torch.zeros((9, 9), dtype=torch.long)
            pieces = torch.zeros((14,), dtype=torch.float32)
            move_label = torch.zeros(
                (MOVE_LABELS_NUM,), dtype=torch.float32
            )
            result_value = torch.zeros(
                (1,), dtype=torch.float32
            )
            legal_mask = torch.zeros(
                (MOVE_LABELS_NUM,), dtype=torch.float32
            )
            return (board, pieces), (
                move_label,
                result_value,
                legal_mask,
            )

    def fake_create_datasets(
        cls: type[DatasetFactory],
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


def test_training_setup_respects_cache_transforms_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_flags: list[bool] = []

    class _StubDataset:
        def __len__(self) -> int:
            return 4

        def __getitem__(
            self, idx: int
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]:
            board = torch.zeros((9, 9), dtype=torch.long)
            pieces = torch.zeros((14,), dtype=torch.float32)
            move_label = torch.zeros(
                (MOVE_LABELS_NUM,), dtype=torch.float32
            )
            result_value = torch.zeros(
                (1,), dtype=torch.float32
            )
            legal_mask = torch.zeros(
                (MOVE_LABELS_NUM,), dtype=torch.float32
            )
            return (board, pieces), (
                move_label,
                result_value,
                legal_mask,
            )

    def fake_create_datasets(
        cls: type[DatasetFactory],
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


def test_training_setup_disables_anomaly_detection_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_set_detect_anomaly(
        *, mode: bool, check_nan: bool
    ) -> None:
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


def test_training_setup_enables_anomaly_detection_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_set_detect_anomaly(
        *, mode: bool, check_nan: bool
    ) -> None:
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


def test_create_shogi_backbone_uses_default_hand_projection_dim() -> (
    None
):
    """create_shogi_backbone uses DEFAULT_HAND_PROJECTION_DIM when not specified."""
    backbone = ModelFactory.create_shogi_backbone(
        torch.device("cpu")
    )

    assert backbone._hand_projection_dim == 32


def test_create_shogi_backbone_input_channels_include_hand() -> (
    None
):
    """Backbone input_channels equals embedding_dim + hand_projection_dim."""
    backbone = ModelFactory.create_shogi_backbone(
        torch.device("cpu")
    )

    # embedding_dim=32, hand_projection_dim=32 -> backbone_input_channels=64
    expected_channels = (
        backbone._embedding_channels
        + backbone._hand_projection_dim
    )
    assert expected_channels == 64


def test_create_shogi_backbone_explicit_zero_hand_projection() -> (
    None
):
    """Passing hand_projection_dim=0 disables hand projection."""
    backbone = ModelFactory.create_shogi_backbone(
        torch.device("cpu"),
        hand_projection_dim=0,
    )

    assert backbone._hand_projection_dim == 0
    assert backbone._hand_projection is None


@pytest.mark.parametrize(
    "architecture", ["resnet", "vit", "mlp-mixer"]
)
def test_create_shogi_backbone_shape_matches_full_model(
    architecture: str,
) -> None:
    """Backbone state_dict shapes match the backbone portion of full model."""

    backbone = ModelFactory.create_shogi_backbone(
        torch.device("cpu"),
        architecture=architecture,
    )
    model = ModelFactory.create_shogi_model(
        torch.device("cpu"),
        architecture=architecture,
    )

    backbone_state = backbone.state_dict()
    model_state = model.state_dict()

    for key, tensor in backbone_state.items():
        assert key in model_state, (
            f"Key {key} missing from full model"
        )
        assert tensor.shape == model_state[key].shape, (
            f"Shape mismatch for {key}: "
            f"backbone {list(tensor.shape)} vs model {list(model_state[key].shape)}"
        )


# ---------------------------------------------------------------------------
# DataLoaderFactory._clamp_workers tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workers", "n_files", "expected"),
    [
        pytest.param(4, 37, 4, id="workers_less_than_files"),
        pytest.param(12, 4, 4, id="workers_greater_than_files"),
        pytest.param(4, 4, 4, id="workers_equal_to_files"),
        pytest.param(12, 1, 1, id="single_file"),
        pytest.param(0, 10, 0, id="zero_workers"),
        pytest.param(12, 0, 0, id="zero_files"),
    ],
)
def test_clamp_workers(
    workers: int, n_files: int, expected: int
) -> None:
    logger = logging.getLogger("test_clamp_workers")
    result = DataLoaderFactory._clamp_workers(
        requested_workers=workers,
        n_files=n_files,
        label="test",
        logger=logger,
    )
    assert result == expected


def test_create_streaming_dataloaders_applies_different_worker_counts() -> (
    None
):
    """Train and val loaders get different num_workers when file counts differ."""

    class _MinimalIterableDataset(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _MinimalIterableDataset()
    val_ds = _MinimalIterableDataset()

    with patch(
        "maou.app.learning.setup._estimate_max_workers_by_memory",
        return_value=64,
    ):
        train_loader, val_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataloader_workers=8,
                pin_memory=False,
                prefetch_factor=2,
                n_train_files=8,
                n_val_files=3,
            )
        )

    assert train_loader.num_workers == 7
    assert val_loader.num_workers == 1


def _make_kifdataset(n_samples: int) -> KifDataset:
    """テスト用の最小KifDatasetを作成する．"""
    ds = DummyPreprocessedDataSource(length=n_samples)
    return KifDataset(datasource=ds)


def test_create_dataloaders_clamps_val_workers() -> None:
    """validation datasetが小さい場合，valワーカー数が制限される．"""
    train_ds = _make_kifdataset(1000)
    val_ds = _make_kifdataset(3)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=32,
            dataloader_workers=12,
            pin_memory=False,
        )
    )

    assert train_loader.num_workers == 12
    assert val_loader.num_workers == 3


def test_create_dataloaders_no_clamping_when_dataset_large() -> (
    None
):
    """データセットサイズ >= workers の場合，ワーカー数は変更されない．"""
    train_ds = _make_kifdataset(1000)
    val_ds = _make_kifdataset(1000)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=32,
            dataloader_workers=4,
            pin_memory=False,
        )
    )

    assert train_loader.num_workers == 4
    assert val_loader.num_workers == 4


def test_create_dataloaders_clamps_both() -> None:
    """train/val 両方のデータセットが小さい場合，両方制限される．"""
    train_ds = _make_kifdataset(2)
    val_ds = _make_kifdataset(3)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=1,
            dataloader_workers=12,
            pin_memory=False,
        )
    )

    assert train_loader.num_workers == 2
    assert val_loader.num_workers == 3


def test_create_dataloaders_zero_workers() -> None:
    """workers=0 の場合，clamping後も0のまま．"""
    train_ds = _make_kifdataset(100)
    val_ds = _make_kifdataset(100)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=32,
            dataloader_workers=0,
            pin_memory=False,
        )
    )

    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0


def test_create_dataloaders_single_sample_dataset() -> None:
    """データセットサイズ=1 の場合，ワーカー数が1に制限される．"""
    train_ds = _make_kifdataset(1)
    val_ds = _make_kifdataset(1)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=1,
            dataloader_workers=12,
            pin_memory=False,
        )
    )

    assert train_loader.num_workers == 1
    assert val_loader.num_workers == 1


# --- Track 1-2: メモリベースワーカー制限テスト ---


class _FakeVirtualMemory:
    """psutil.virtual_memory() のモック用．"""

    def __init__(self, available_mb: float) -> None:
        self.available = int(available_mb * 1024 * 1024)
        self.total = self.available * 2
        self.percent = 50.0


@pytest.mark.parametrize(
    ("available_mb", "pin_memory", "expected"),
    [
        pytest.param(
            8000.0,
            False,
            20,
            id="8GB_no_pin",
        ),
        pytest.param(
            8000.0,
            True,
            16,
            id="8GB_pin",
        ),
        pytest.param(
            2000.0,
            False,
            5,
            id="2GB_no_pin",
        ),
        pytest.param(
            500.0,
            False,
            1,
            id="500MB_no_pin",
        ),
    ],
)
def test_estimate_max_workers_by_memory(
    available_mb: float,
    pin_memory: bool,
    expected: int,
) -> None:
    """メモリ量モックによるワーカー上限推定の検証．"""
    logger = logging.getLogger("test_estimate_max_workers")
    fake_vm = _FakeVirtualMemory(available_mb)

    with patch("psutil.virtual_memory", return_value=fake_vm):
        result = _estimate_max_workers_by_memory(
            pin_memory=pin_memory,
            logger=logger,
        )
    assert result == expected


def test_estimate_max_workers_by_memory_returns_at_least_one() -> (
    None
):
    """利用可能メモリが極端に少ない場合でも最小1を返す．"""
    logger = logging.getLogger("test_estimate_min")
    fake_vm = _FakeVirtualMemory(50.0)

    with patch("psutil.virtual_memory", return_value=fake_vm):
        result = _estimate_max_workers_by_memory(
            pin_memory=False,
            logger=logger,
        )
    assert result >= 1


@pytest.mark.parametrize(
    ("workers", "n_files", "memory_limit", "expected"),
    [
        pytest.param(12, 20, 8, 8, id="memory_limit_binding"),
        pytest.param(12, 5, 8, 5, id="file_limit_binding"),
        pytest.param(4, 20, 8, 4, id="requested_binding"),
        pytest.param(12, 20, None, 12, id="no_memory_limit"),
        pytest.param(12, 20, 0, 12, id="zero_memory_limit"),
    ],
)
def test_clamp_workers_with_memory_limit(
    workers: int,
    n_files: int,
    memory_limit: int | None,
    expected: int,
) -> None:
    """_clamp_workersのmemory_limitパラメータ動作検証．"""
    logger = logging.getLogger("test_clamp_memory")
    result = DataLoaderFactory._clamp_workers(
        requested_workers=workers,
        n_files=n_files,
        label="test",
        logger=logger,
        memory_limit=memory_limit,
    )
    assert result == expected


# --- Fix 1: ファイルサイズベースの動的メモリ推定テスト ---


class _FakeStat:
    """Path.stat() のモック用．"""

    def __init__(self, size_bytes: int) -> None:
        self.st_size = size_bytes


def _make_fake_path(
    size_mb: float, *, exists: bool = True
) -> Path:
    """テスト用の疑似Pathを作成する．"""
    from pathlib import Path
    from unittest.mock import MagicMock

    fp = MagicMock(spec=Path)
    if exists:
        fp.stat.return_value = _FakeStat(
            int(size_mb * 1024 * 1024)
        )
    else:
        fp.stat.side_effect = OSError("file not found")
    return fp


def test_estimate_per_worker_mb_100mb_files() -> None:
    """100MBファイル → per_worker_mb=600(pin_memory=True で650)．"""
    from maou.app.learning.setup import _estimate_per_worker_mb

    logger = logging.getLogger("test_per_worker")
    paths = [_make_fake_path(100.0) for _ in range(5)]
    result = _estimate_per_worker_mb(paths, logger)
    # 100 * 4.0 * 1.5 = 600.0
    assert result == pytest.approx(600.0, rel=1e-3)


def test_estimate_per_worker_mb_10mb_files() -> None:
    """10MBファイル → per_worker_mb=200(下限適用)．"""
    from maou.app.learning.setup import _estimate_per_worker_mb

    logger = logging.getLogger("test_per_worker")
    paths = [_make_fake_path(10.0) for _ in range(5)]
    result = _estimate_per_worker_mb(paths, logger)
    # 10 * 4.0 * 1.5 = 60.0 → max(60, 200) = 200.0
    assert result == pytest.approx(200.0, rel=1e-3)


def test_estimate_per_worker_mb_no_paths() -> None:
    """ファイルパスなし → フォールバック200MB．"""
    from maou.app.learning.setup import _estimate_per_worker_mb

    logger = logging.getLogger("test_per_worker")
    result = _estimate_per_worker_mb(None, logger)
    assert result == pytest.approx(200.0, rel=1e-3)


def test_estimate_per_worker_mb_all_missing() -> None:
    """全ファイル不存在 → フォールバック200MB．"""
    from maou.app.learning.setup import _estimate_per_worker_mb

    logger = logging.getLogger("test_per_worker")
    paths = [
        _make_fake_path(100.0, exists=False) for _ in range(3)
    ]
    result = _estimate_per_worker_mb(paths, logger)
    assert result == pytest.approx(200.0, rel=1e-3)


def test_estimate_per_worker_mb_mixed_existing() -> None:
    """存在するファイルのみで計算(存在しないファイル混在)．"""
    from maou.app.learning.setup import _estimate_per_worker_mb

    logger = logging.getLogger("test_per_worker")
    paths = [
        _make_fake_path(100.0, exists=True),
        _make_fake_path(100.0, exists=False),
        _make_fake_path(100.0, exists=True),
    ]
    result = _estimate_per_worker_mb(paths, logger)
    # avg = 100MB, 100 * 4.0 * 1.5 = 600.0
    assert result == pytest.approx(600.0, rel=1e-3)


def test_estimate_max_workers_with_file_paths() -> None:
    """ファイルサイズ100MB + pin_memory → per_worker_mb=650．"""
    logger = logging.getLogger("test_max_workers_file")
    fake_vm = _FakeVirtualMemory(50000.0)  # 50GB available
    paths = [_make_fake_path(100.0) for _ in range(10)]

    with patch("psutil.virtual_memory", return_value=fake_vm):
        result = _estimate_max_workers_by_memory(
            pin_memory=True,
            logger=logger,
            file_paths=paths,
        )
    # budget = 50000 * 0.5 = 25000, per_worker = 600 + 50 = 650
    # max_workers = 25000 / 650 = 38.46 → 38
    assert result == 38


def test_estimate_max_workers_fallback_no_file_paths() -> None:
    """ファイルパスなし + pin_memory → per_worker_mb=250(フォールバック)．"""
    logger = logging.getLogger("test_max_workers_fallback")
    fake_vm = _FakeVirtualMemory(8000.0)

    with patch("psutil.virtual_memory", return_value=fake_vm):
        result = _estimate_max_workers_by_memory(
            pin_memory=True,
            logger=logger,
            file_paths=None,
        )
    # budget = 8000 * 0.5 = 4000, per_worker = 200 + 50 = 250
    # max_workers = 4000 / 250 = 16
    assert result == 16


# --- Fix 3: /dev/shm サイズチェックテスト ---


class _FakeStatvfs:
    """os.statvfs のモック用．"""

    def __init__(self, available_mb: float) -> None:
        self.f_frsize = 4096
        self.f_bavail = int(
            available_mb * 1024 * 1024 / self.f_frsize
        )


def test_check_shm_size_warns_when_insufficient() -> None:
    """shm空き不足時にWARNINGが出力されること．"""
    from maou.app.learning.setup import _check_shm_size

    logger = logging.getLogger("test_shm")
    # 100MB available, but threshold will be much higher
    fake_statvfs = _FakeStatvfs(100.0)

    with (
        patch("sys.platform", "linux"),
        patch("os.statvfs", return_value=fake_statvfs),
        patch.object(Path, "exists", return_value=True),
        patch.object(logger, "warning") as mock_warn,
    ):
        _check_shm_size(
            num_workers=8,
            batch_size=None,
            prefetch_factor=2,
            logger=logger,
        )

    mock_warn.assert_called_once()
    assert "/dev/shm" in mock_warn.call_args[0][0]


def test_check_shm_size_no_warn_when_sufficient() -> None:
    """shm空き十分な場合はWARNINGが出力されないこと．"""
    from maou.app.learning.setup import _check_shm_size

    logger = logging.getLogger("test_shm")
    # 16GB available - should be more than enough
    fake_statvfs = _FakeStatvfs(16000.0)

    with (
        patch("sys.platform", "linux"),
        patch("os.statvfs", return_value=fake_statvfs),
        patch.object(Path, "exists", return_value=True),
        patch.object(logger, "warning") as mock_warn,
    ):
        _check_shm_size(
            num_workers=8,
            batch_size=None,
            prefetch_factor=2,
            logger=logger,
        )

    mock_warn.assert_not_called()


def test_check_shm_size_skipped_on_non_linux() -> None:
    """Linux以外ではチェックがスキップされること．"""
    from maou.app.learning.setup import _check_shm_size

    logger = logging.getLogger("test_shm")

    with (
        patch("sys.platform", "darwin"),
        patch.object(logger, "warning") as mock_warn,
    ):
        _check_shm_size(
            num_workers=8,
            batch_size=None,
            prefetch_factor=2,
            logger=logger,
        )

    mock_warn.assert_not_called()


def test_check_shm_size_skipped_with_zero_workers() -> None:
    """ワーカー数0ではチェックがスキップされること．"""
    from maou.app.learning.setup import _check_shm_size

    logger = logging.getLogger("test_shm")

    with patch.object(logger, "warning") as mock_warn:
        _check_shm_size(
            num_workers=0,
            batch_size=None,
            prefetch_factor=2,
            logger=logger,
        )

    mock_warn.assert_not_called()


# --- Fix 4: ワーカーメモリ使用量ログテスト ---


class _FakeMemoryInfo:
    """psutil.Process().memory_info() のモック用．"""

    def __init__(self, rss_mb: float) -> None:
        self.rss = int(rss_mb * 1024 * 1024)


def test_log_worker_memory_info_level() -> None:
    """_log_worker_memory がINFOレベルで正しい形式のログを出力すること．"""
    from maou.app.learning.setup import _log_worker_memory

    fake_mem = _FakeMemoryInfo(512.0)
    mock_process = patch(
        "psutil.Process",
        return_value=type(
            "FakeProcess",
            (),
            {"memory_info": lambda self: fake_mem},
        )(),
    )

    with mock_process:
        with patch(
            "maou.app.learning.setup.logging"
        ) as mock_logging:
            mock_logger = mock_logging.getLogger.return_value
            _log_worker_memory(0, "init")

    mock_logger.log.assert_called_once()
    args = mock_logger.log.call_args[0]
    assert args[0] == logging.INFO
    assert "Worker 0" in args[1] % args[2:]
    assert "init" in args[1] % args[2:]


def test_log_worker_memory_debug_level() -> None:
    """_log_worker_memory がDEBUGレベルで正しい形式のログを出力すること．"""
    from maou.app.learning.setup import _log_worker_memory

    fake_mem = _FakeMemoryInfo(1024.0)
    mock_process = patch(
        "psutil.Process",
        return_value=type(
            "FakeProcess",
            (),
            {"memory_info": lambda self: fake_mem},
        )(),
    )

    with mock_process:
        with patch(
            "maou.app.learning.setup.logging"
        ) as mock_logging:
            mock_logger = mock_logging.getLogger.return_value
            _log_worker_memory(
                3, "after_first_file", level=logging.DEBUG
            )

    mock_logger.log.assert_called_once()
    args = mock_logger.log.call_args[0]
    assert args[0] == logging.DEBUG
    assert "Worker 3" in args[1] % args[2:]
    assert "after_first_file" in args[1] % args[2:]


# --- Fix G: multiprocessing_context="spawn" テスト ---


def test_streaming_dataloaders_use_spawn() -> None:
    """ストリーミングDataLoaderがspawnコンテキストを使用すること．"""

    class _MinimalIterableDataset2(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _MinimalIterableDataset2()
    val_ds = _MinimalIterableDataset2()

    with patch(
        "maou.app.learning.setup._estimate_max_workers_by_memory",
        return_value=64,
    ):
        train_loader, val_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataloader_workers=4,
                pin_memory=False,
                prefetch_factor=2,
                n_train_files=4,
                n_val_files=4,
            )
        )

    # ストリーミングモードではspawnが使われること
    assert train_loader.multiprocessing_context is not None
    assert (
        "spawn"
        in type(
            train_loader.multiprocessing_context
        ).__name__.lower()
    )
    assert val_loader.multiprocessing_context is not None
    assert (
        "spawn"
        in type(
            val_loader.multiprocessing_context
        ).__name__.lower()
    )
    # persistent_workersはTrue(spawnで安全)
    assert train_loader.persistent_workers is True
    assert val_loader.persistent_workers is True


def test_streaming_dataloaders_no_mp_context_with_zero_workers() -> (
    None
):
    """ワーカー数0ではmultiprocessing_contextが設定されないこと．"""

    class _MinimalIterableDataset3(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _MinimalIterableDataset3()
    val_ds = _MinimalIterableDataset3()

    with patch(
        "maou.app.learning.setup._estimate_max_workers_by_memory",
        return_value=64,
    ):
        train_loader, val_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataloader_workers=0,
                pin_memory=False,
                prefetch_factor=2,
                n_train_files=4,
                n_val_files=4,
            )
        )

    assert train_loader.multiprocessing_context is None
    assert val_loader.multiprocessing_context is None


def test_non_streaming_dataloaders_no_mp_context() -> None:
    """非ストリーミングDataLoaderにはmultiprocessing_contextが設定されないこと．"""
    train_ds = _make_kifdataset(100)
    val_ds = _make_kifdataset(100)

    train_loader, val_loader = (
        DataLoaderFactory.create_dataloaders(
            dataset_train=train_ds,
            dataset_validation=val_ds,
            batch_size=32,
            dataloader_workers=2,
            pin_memory=False,
        )
    )

    # 非ストリーミングモードではmultiprocessing_contextなし
    assert train_loader.multiprocessing_context is None


# --- Fix 2: ストリーミングDataLoaderのタイムアウト検証 ---


def test_streaming_dataloaders_timeout_zero() -> None:
    """ストリーミングDataLoaderの timeout が 0 であることを検証する．"""

    class _TimeoutTestDataset(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _TimeoutTestDataset()
    val_ds = _TimeoutTestDataset()

    with patch(
        "maou.app.learning.setup._estimate_max_workers_by_memory",
        return_value=64,
    ):
        train_loader, val_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataloader_workers=4,
                pin_memory=False,
                prefetch_factor=2,
                n_train_files=4,
                n_val_files=4,
            )
        )

    # ストリーミングDataLoaderのtimeoutは0
    # (ストリーミングモードではタイムアウト管理をアプリケーション側で行う)
    assert train_loader.timeout == 0
    assert val_loader.timeout == 0


# --- Fix 3: ストリーミングワーカー数警告テスト ---


@pytest.mark.parametrize(
    ("train", "val", "expected"),
    [
        pytest.param(3, 3, (3, 3), id="under_cap"),
        pytest.param(6, 6, (6, 2), id="over_cap_val_reduced"),
        pytest.param(10, 2, (7, 1), id="train_alone_exceeds"),
        pytest.param(8, 4, (7, 1), id="val_min_guarantee"),
        pytest.param(1, 1, (1, 1), id="minimal"),
        pytest.param(8, 0, (8, 0), id="val_zero_no_data"),
        pytest.param(0, 0, (0, 0), id="both_zero"),
        pytest.param(0, 4, (0, 4), id="train_zero_val_exists"),
        pytest.param(
            12, 0, (8, 0), id="val_zero_train_exceeds"
        ),
    ],
)
def test_cap_total_workers(
    train: int,
    val: int,
    expected: tuple[int, int],
) -> None:
    """合計ワーカー数キャップのロジックを検証する．"""
    result = DataLoaderFactory._cap_total_workers(
        train,
        val,
        8,
        logging.getLogger("test"),
    )
    assert result == expected


@pytest.mark.parametrize(
    (
        "requested",
        "n_train",
        "n_val",
        "expected_train",
        "expected_val",
    ),
    [
        pytest.param(6, 100, 100, 6, 2, id="cap_applied"),
        pytest.param(3, 100, 100, 3, 3, id="under_cap"),
    ],
)
def test_streaming_dataloaders_cap_total_spawn_workers(
    requested: int,
    n_train: int,
    n_val: int,
    expected_train: int,
    expected_val: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """create_streaming_dataloaders で合計ワーカー数キャップが適用されることを検証する．"""

    class _CapTestDataset(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _CapTestDataset()
    val_ds = _CapTestDataset()

    maou_logger = logging.getLogger("maou")
    original_propagate = maou_logger.propagate
    maou_logger.propagate = True
    try:
        with (
            patch(
                "maou.app.learning.setup._estimate_max_workers_by_memory",
                return_value=64,
            ),
            caplog.at_level(logging.INFO),
        ):
            train_loader, val_loader = (
                DataLoaderFactory.create_streaming_dataloaders(
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    dataloader_workers=requested,
                    pin_memory=False,
                    prefetch_factor=2,
                    n_train_files=n_train,
                    n_val_files=n_val,
                )
            )
    finally:
        maou_logger.propagate = original_propagate

    assert train_loader.num_workers == expected_train
    assert val_loader.num_workers == expected_val


def test_streaming_dataloaders_pin_memory_passthrough() -> None:
    """ストリーミングDataLoaderのpin_memoryが引数通りに設定されることを検証する．

    DataPrefetcher除去後はpin_memory_threadがspawnコンテキストで
    安全に動作するため，pin_memoryをそのまま渡す．
    """

    class _PinMemoryTestDataset(IterableDataset):
        def __iter__(self) -> Iterator[None]:  # type: ignore[override]
            return iter([])

    train_ds = _PinMemoryTestDataset()
    val_ds = _PinMemoryTestDataset()

    with patch(
        "maou.app.learning.setup._estimate_max_workers_by_memory",
        return_value=64,
    ):
        train_loader, val_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_ds,
                val_dataset=val_ds,
                dataloader_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                n_train_files=10,
                n_val_files=10,
            )
        )

    # pin_memory=True がそのまま渡される
    assert train_loader.pin_memory is True
    assert val_loader.pin_memory is True
