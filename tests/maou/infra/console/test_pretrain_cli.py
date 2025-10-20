from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

from maou.domain.data.schema import create_empty_preprocessing_array
from maou.interface.data_io import save_array
from maou.infra.console.pretrain_cli import pretrain
from maou.app.learning.setup import ModelFactory


def _write_preprocessing_file(directory: Path, samples: int) -> None:
    array = create_empty_preprocessing_array(samples)
    rng = np.random.default_rng(321)
    array["features"] = (
        (rng.random((samples, 104, 9, 9)) > 0.5).astype(np.uint8)
    )
    save_array(
        array,
        directory / "preprocessing.npy",
        array_type="preprocessing",
        bit_pack=False,
    )


def test_pretrain_cli(tmp_path: Path) -> None:
    _write_preprocessing_file(tmp_path, samples=6)
    output_path = tmp_path / "weights.pt"

    runner = CliRunner()
    result = runner.invoke(
        pretrain,
        [
            "--input-dir",
            str(tmp_path),
            "--output-path",
            str(output_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "32",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    state_dict = torch.load(output_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert state_dict
    assert not any(key.startswith("decoder") for key in state_dict)
    model = ModelFactory.create_shogi_model(torch.device("cpu"))
    model.load_state_dict(state_dict)
    assert "saved state_dict" in result.output.lower()
