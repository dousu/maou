import numpy as np
import torch

from maou.app.learning.dataset import DataSource
from maou.app.learning.setup import TrainingSetup
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM


class DummyPreprocessedDataSource(DataSource):
    def __init__(self, length: int = 4) -> None:
        dtype = np.dtype(
            [
                ("features", np.float32, (FEATURES_NUM, 9, 9)),
                ("legalMoveMask", np.float32, (MOVE_LABELS_NUM,)),
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

    _, _, model_components = TrainingSetup.setup_training_components(
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

    optimizer = model_components.optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.85, 0.98)
    assert optimizer.defaults["eps"] == 1e-07
