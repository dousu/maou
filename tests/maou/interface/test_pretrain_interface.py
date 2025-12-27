"""Tests for pretrain interface.

Updated to use DataFrame-based methods with Polars.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import pytest
import torch

from maou.app.learning import masked_autoencoder
from maou.app.learning.setup import ModelFactory
from maou.domain.data.rust_io import save_preprocessing_df
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.interface.pretrain import pretrain


def _create_dummy_datasource(
    directory: Path, samples: int
) -> FileDataSource.FileDataSourceSpliter:
    """Create dummy preprocessing datasource with DataFrame."""
    df = create_empty_preprocessing_df(samples)
    rng = np.random.default_rng(123)

    # Create random board positions
    board_positions = [
        rng.integers(0, 30, size=(9, 9), dtype=np.uint8).tolist()
        for _ in range(samples)
    ]

    # Create random pieces in hand
    pieces_in_hand = [
        rng.integers(0, 2, size=14, dtype=np.uint8).tolist()
        for _ in range(samples)
    ]

    # Create normalized move labels
    move_labels = []
    for _ in range(samples):
        label = rng.random(MOVE_LABELS_NUM)
        label = label / label.sum()
        move_labels.append(label.astype(np.float32).tolist())

    # Create random result values
    result_values = rng.random(samples).astype(np.float16).tolist()

    # Update DataFrame
    df = df.with_columns([
        pl.Series("boardIdPositions", board_positions),
        pl.Series("piecesInHand", pieces_in_hand),
        pl.Series("moveLabel", move_labels),
        pl.Series("resultValue", result_values),
    ])

    # Save to .feather file
    file_path = directory / "preprocessing.feather"
    save_preprocessing_df(df, file_path)

    return FileDataSource.FileDataSourceSpliter(
        file_paths=[file_path],
        array_type="preprocessing",
        bit_pack=False,
    )


def test_pretrain_persists_state_dict(tmp_path: Path) -> None:
    """Test that pretrain saves a state dict to the specified path."""
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
    """Test that compilation flag triggers torch.compile."""
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
    """Test that cache_transforms defaults to True for HCPE datasource."""
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
