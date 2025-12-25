from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from maou.app.learning import masked_autoencoder
from maou.app.learning.setup import ModelFactory
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)
from maou.infra.console.pretrain_cli import pretrain
from maou.interface.data_io import save_array


def _write_preprocessing_file(
    directory: Path, samples: int
) -> None:
    array = create_empty_preprocessing_array(samples)
    rng = np.random.default_rng(321)
    array["boardIdPositions"] = rng.integers(
        0, 30, size=(samples, 9, 9), dtype=np.uint8
    )
    array["piecesInHand"] = rng.integers(
        0, 2, size=(samples, 14), dtype=np.uint8
    )
    move_label = rng.random(
        (samples, array["moveLabel"].shape[1])
    )
    move_label /= move_label.sum(axis=1, keepdims=True)
    array["moveLabel"] = move_label.astype(np.float16)
    array["resultValue"] = rng.random(samples).astype(
        np.float16
    )
    save_array(
        array,
        directory / "preprocessing.npy",
        array_type="preprocessing",
        bit_pack=False,
    )


def test_pretrain_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_preprocessing_file(tmp_path, samples=6)
    output_path = tmp_path / "weights.pt"

    compile_called = False
    compile_kwargs: Dict[str, Any] = {}

    def _fake_compile(
        module: torch.nn.Module, **kwargs: Any
    ) -> torch.nn.Module:
        nonlocal compile_called
        compile_called = True
        compile_kwargs.update(kwargs)
        return module

    monkeypatch.setattr(
        masked_autoencoder.torch, "compile", _fake_compile
    )

    runner = CliRunner()
    result = runner.invoke(
        pretrain,
        [
            "--input-dir",
            str(tmp_path),
            "--input-cache-mode",
            "memory",
            "--output-path",
            str(output_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--dataloader-workers",
            "0",
            "--prefetch-factor",
            "1",
            "--pin-memory",
            "--cache-transforms",
            "--hidden-dim",
            "32",
            "--compilation",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    state_dict = torch.load(output_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert state_dict
    assert not any(
        key.startswith("decoder") for key in state_dict
    )
    backbone = ModelFactory.create_shogi_backbone(
        torch.device("cpu")
    )
    backbone.load_state_dict(state_dict)
    assert "saved state_dict" in result.output.lower()
    assert compile_called
    assert compile_kwargs.get("dynamic") is True
