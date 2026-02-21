"""Tests for pretrain CLI.

Updated to use DataFrame-based methods with Polars.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import pytest
import torch
from click.testing import CliRunner

from maou.app.learning import masked_autoencoder
from maou.app.learning.setup import ModelFactory
from maou.domain.data.rust_io import save_preprocessing_df
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.console.pretrain_cli import pretrain


def _write_preprocessing_file(
    directory: Path, samples: int
) -> None:
    """Write preprocessing DataFrame to file."""
    df = create_empty_preprocessing_df(samples)
    rng = np.random.default_rng(321)

    # Create random board positions
    board_positions = [
        rng.integers(
            0, 30, size=(9, 9), dtype=np.uint8
        ).tolist()
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
    result_values = (
        rng.random(samples).astype(np.float16).tolist()
    )

    # Update DataFrame
    df = df.with_columns(
        [
            pl.Series("boardIdPositions", board_positions),
            pl.Series("piecesInHand", pieces_in_hand),
            pl.Series("moveLabel", move_labels),
            pl.Series("resultValue", result_values),
        ]
    )

    # Save to .feather file
    save_preprocessing_df(
        df, directory / "preprocessing.feather"
    )


def test_pretrain_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test pretrain CLI with DataFrame-based input."""
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
            "--input-path",
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
        torch.device("cpu"),
        hand_projection_dim=0,
    )
    backbone.load_state_dict(state_dict)
    assert "saved state_dict" in result.output.lower()
    assert compile_called
    assert compile_kwargs.get("dynamic") is True
