from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from maou.app.learning import masked_autoencoder
from maou.app.learning.setup import ModelFactory
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.interface.data_io import save_array
from maou.interface.pretrain import pretrain


def _create_dummy_datasource(
    directory: Path, samples: int
) -> FileDataSource.FileDataSourceSpliter:
    array = create_empty_preprocessing_array(samples)
    rng = np.random.default_rng(123)
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
        datasource_type="preprocess",
        config_path=None,
        output_path=output_path,
        epochs=1,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=1,
        hidden_dim=32,
    )

    assert isinstance(result, str)
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
    backbone.load_state_dict(state_dict, strict=False)
    assert "saved state_dict" in result.lower()


def test_pretrain_compilation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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

    output_path = tmp_path / "state.pt"
    datasource = _create_dummy_datasource(tmp_path, samples=4)

    pretrain(
        datasource=datasource,
        datasource_type="preprocess",
        config_path=None,
        output_path=output_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=1,
        hidden_dim=32,
        compilation=True,
    )

    assert compile_called
    assert compile_kwargs.get("dynamic") is True


def test_pretrain_cache_transforms_default_hcpe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: Dict[str, Any] = {}

    def _fake_run(
        self: masked_autoencoder.MaskedAutoencoderPretraining,
        options: masked_autoencoder.MaskedAutoencoderPretraining.Options,
    ) -> str:
        captured["cache_transforms"] = options.cache_transforms
        return "ok"

    monkeypatch.setattr(
        masked_autoencoder.MaskedAutoencoderPretraining,
        "run",
        _fake_run,
    )

    datasource = _create_dummy_datasource(tmp_path, samples=4)

    result = pretrain(
        datasource=datasource,
        datasource_type="hcpe",
        config_path=None,
        output_path=None,
        epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=1,
        hidden_dim=32,
    )

    assert result == "ok"
    assert captured["cache_transforms"] is True
