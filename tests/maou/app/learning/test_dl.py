"""Tests for the learning module utilities."""

from __future__ import annotations

from typing import Mapping, cast

import pytest
import torch

from maou.app.learning.callbacks import ValidationMetrics
from maou.app.learning.dl import Learning
from maou.app.learning.network import HeadlessNetwork, Network


class _DummyCompiledModule(torch.nn.Module):
    """Minimal stand-in for ``torch.compile`` wrapped modules."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._orig_mod(*args, **kwargs)


class _RecordingWriter:
    """In-memory SummaryWriter replacement used for logging tests."""

    def __init__(self) -> None:
        self.histograms: list[tuple[str, int]] = []
        self.scalar_groups: list[
            tuple[str, dict[str, float], int]
        ] = []
        self.scalars: list[tuple[str, float, int]] = []

    def add_histogram(
        self, tag: str, values: torch.Tensor, global_step: int
    ) -> None:
        self.histograms.append((tag, global_step))

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ) -> None:
        self.scalar_groups.append(
            (main_tag, dict(tag_scalar_dict), global_step)
        )

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int
    ) -> None:
        self.scalars.append(
            (tag, float(scalar_value), global_step)
        )


def _assert_state_dict_equality(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> None:
    for key, tensor in first.items():
        assert torch.equal(tensor, second[key])


def test_load_resume_state_dict_adds_prefix_for_compiled_models() -> (
    None
):
    """State dicts saved before compilation can be loaded into compiled models."""

    source_model = Network()
    target_model = Network()
    compiled_model = _DummyCompiledModule(target_model)

    learning = Learning()
    learning.model = cast(Network, compiled_model)

    state_dict = source_model.state_dict()

    learning._load_resume_state_dict(state_dict)

    _assert_state_dict_equality(
        state_dict, target_model.state_dict()
    )


def test_load_resume_state_dict_without_compilation() -> None:
    """Pure PyTorch modules continue to load state dicts without modification."""

    source_model = Network()
    target_model = Network()

    learning = Learning()
    learning.model = target_model

    state_dict = source_model.state_dict()

    learning._load_resume_state_dict(state_dict)

    _assert_state_dict_equality(
        state_dict, target_model.state_dict()
    )


def test_load_resume_state_dict_requires_complete_state() -> (
    None
):
    """Checkpoints missing head weights should raise a descriptive error."""

    source_model = HeadlessNetwork()
    target_model = Network()

    learning = Learning()
    learning.model = target_model

    state_dict = source_model.state_dict()

    with pytest.raises(RuntimeError):
        learning._load_resume_state_dict(state_dict)


def test_format_parameter_count_generates_human_readable_labels() -> (
    None
):
    """Learning._format_parameter_count should generate readable suffixes."""

    assert Learning._format_parameter_count(4_000_000) == "4m"
    assert Learning._format_parameter_count(1_250_000) == "1.2m"
    assert Learning._format_parameter_count(125_000) == "125k"
    assert Learning._format_parameter_count(512) == "512"


def test_histogram_logging_can_be_filtered_and_sampled() -> (
    None
):
    """Histogram logging honors frequency and module filters."""

    learning = Learning()
    learning.model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )
    learning.tensorboard_histogram_frequency = 2
    learning.tensorboard_histogram_modules = ("0.weight",)

    writer = _RecordingWriter()

    learning._log_parameter_histograms(writer, epoch_number=0)
    assert writer.histograms == []

    learning._log_parameter_histograms(writer, epoch_number=1)
    assert writer.histograms == [("parameters/0.weight", 2)]


def test_scalar_metrics_still_logged_when_histograms_disabled() -> (
    None
):
    """Scalar TensorBoard metrics remain available when histograms are off."""

    learning = Learning()
    learning.model = torch.nn.Linear(2, 2, bias=False)
    learning.tensorboard_histogram_frequency = 0
    learning.tensorboard_histogram_modules = None

    writer = _RecordingWriter()
    metrics = ValidationMetrics(
        policy_cross_entropy=0.5,
        value_brier_score=0.25,
        policy_top5_accuracy=0.75,
        policy_f1_score=0.7,
        value_high_confidence_rate=0.6,
    )

    learning._log_parameter_histograms(writer, epoch_number=0)
    assert writer.histograms == []

    learning._log_epoch_metrics(
        writer=writer,
        metrics=metrics,
        avg_loss=0.4,
        avg_vloss=0.3,
        epoch_number=0,
        learning_rate=1e-3,
    )

    assert any(
        main_tag == "Validation Loss Metrics"
        for main_tag, _, _ in writer.scalar_groups
    )
    assert any(
        tag == "Learning Rate" for tag, _, _ in writer.scalars
    )
