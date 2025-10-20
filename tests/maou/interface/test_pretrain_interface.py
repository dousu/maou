from pathlib import Path

import numpy as np
import torch

from maou.app.learning.setup import ModelFactory

from maou.domain.data.schema import create_empty_preprocessing_array
from maou.interface.data_io import save_array
from maou.infra.file_system.file_data_source import FileDataSource
from maou.interface.pretrain import pretrain


def _create_dummy_datasource(
    directory: Path, samples: int
) -> FileDataSource.FileDataSourceSpliter:
    array = create_empty_preprocessing_array(samples)
    rng = np.random.default_rng(123)
    array["features"] = (
        (rng.random((samples, 104, 9, 9)) > 0.5).astype(np.uint8)
    )
    file_path = directory / "preprocessing.npy"
    save_array(
        array,
        file_path,
        array_type="preprocessing",
        bit_pack=False,
    )
    return FileDataSource.FileDataSourceSpliter(
        file_paths=[file_path],
        array_type="preprocessing",
        bit_pack=False,
    )


def test_pretrain_persists_state_dict(tmp_path: Path) -> None:
    output_path = tmp_path / "state.pt"
    datasource = _create_dummy_datasource(tmp_path, samples=8)

    result = pretrain(
        datasource=datasource,
        config_path=None,
        output_path=output_path,
        epochs=1,
        batch_size=4,
        hidden_dim=32,
    )

    assert isinstance(result, str)
    assert output_path.exists()
    state_dict = torch.load(output_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert state_dict
    assert not any(key.startswith("decoder") for key in state_dict)
    model = ModelFactory.create_shogi_model(torch.device("cpu"))
    model.load_state_dict(state_dict)
    assert "saved state_dict" in result.lower()
