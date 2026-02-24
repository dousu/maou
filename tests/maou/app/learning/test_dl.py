"""Tests for the learning module utilities."""

from __future__ import annotations

import torch

from maou.app.learning.callbacks import ValidationMetrics
from maou.app.learning.dl import Learning


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


def test_format_parameter_count_generates_human_readable_labels() -> (
    None
):
    """ModelIO.format_parameter_count should generate readable suffixes."""
    from maou.app.learning.model_io import ModelIO

    assert ModelIO.format_parameter_count(4_000_000) == "4m"
    assert ModelIO.format_parameter_count(1_250_000) == "1.2m"
    assert ModelIO.format_parameter_count(125_000) == "125k"
    assert ModelIO.format_parameter_count(512) == "512"


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
