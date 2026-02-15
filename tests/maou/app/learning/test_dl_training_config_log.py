"""Tests for training configuration summary log in Learning."""

import logging
from unittest.mock import MagicMock

import pytest

from maou.app.learning.dl import Learning


def _make_learning_for_config_log(
    *,
    trainable_layers: int | None = None,
    freeze_backbone: bool = False,
) -> Learning:
    """Create a Learning instance for config log testing."""
    learning = Learning()
    learning.freeze_backbone = freeze_backbone
    learning.trainable_layers = trainable_layers
    return learning


def _enable_propagation() -> tuple[logging.Logger, bool]:
    """Context helper to temporarily enable maou logger propagation for caplog."""
    maou_logger = logging.getLogger("maou")
    original = maou_logger.propagate
    maou_logger.propagate = True
    return maou_logger, original


class TestLogTrainingConfig:
    """Tests for _log_training_config method."""

    def _make_config(self, **overrides: object) -> MagicMock:
        """Create a mock config with default values."""
        defaults = dict(
            model_architecture="resnet",
            optimizer_name="adamw",
            learning_ratio=0.001,
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            lr_scheduler_name=None,
            batch_size=256,
            epoch=100,
            dataloader_workers=4,
            gce_parameter=0.7,
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            datasource_type="hcpe",
            input_cache_mode="mmap",
        )
        defaults.update(overrides)
        config = MagicMock()
        for k, v in defaults.items():
            setattr(config, k, v)
        return config

    def test_no_freeze_shows_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Freeze: none is shown when no freezing."""
        learning = _make_learning_for_config_log(
            trainable_layers=None, freeze_backbone=False
        )
        config = self._make_config()

        maou_logger, original = _enable_propagation()
        try:
            with caplog.at_level(logging.INFO):
                learning._log_training_config(
                    config, None, "resnet-1.2m"
                )
        finally:
            maou_logger.propagate = original

        assert "Freeze: none" in caplog.text

    def test_trainable_layers_shown(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Freeze section shows trainable_layers value."""
        learning = _make_learning_for_config_log(
            trainable_layers=2, freeze_backbone=False
        )
        config = self._make_config(
            lr_scheduler_name="cosine_annealing",
            batch_size=128,
            epoch=50,
        )

        maou_logger, original = _enable_propagation()
        try:
            with caplog.at_level(logging.INFO):
                learning._log_training_config(
                    config, None, "resnet-1.2m-tl2"
                )
        finally:
            maou_logger.propagate = original

        assert "Freeze: trainable_layers=2" in caplog.text

    def test_vit_architecture_config_in_model_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ViT architecture_config params appear in Model line."""
        learning = _make_learning_for_config_log()
        config = self._make_config(model_architecture="vit")

        arch_config = {
            "embed_dim": 256,
            "num_layers": 8,
            "num_heads": 8,
        }

        maou_logger, original = _enable_propagation()
        try:
            with caplog.at_level(logging.INFO):
                learning._log_training_config(
                    config, arch_config, "vit-19m"
                )
        finally:
            maou_logger.propagate = original

        assert "embed_dim=256" in caplog.text
        assert "num_layers=8" in caplog.text
        assert "num_heads=8" in caplog.text

    def test_resnet_no_vit_params(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ResNet model line does not include ViT params."""
        learning = _make_learning_for_config_log()
        config = self._make_config(
            model_architecture="resnet",
            optimizer_name="sgd",
            learning_ratio=0.01,
        )

        maou_logger, original = _enable_propagation()
        try:
            with caplog.at_level(logging.INFO):
                learning._log_training_config(
                    config, None, "resnet-1.2m"
                )
        finally:
            maou_logger.propagate = original

        assert "embed_dim" not in caplog.text
        assert "Model: resnet-1.2m" in caplog.text

    def test_config_header_and_footer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Config summary includes header and footer markers."""
        learning = _make_learning_for_config_log()
        config = self._make_config()

        maou_logger, original = _enable_propagation()
        try:
            with caplog.at_level(logging.INFO):
                learning._log_training_config(
                    config, None, "resnet-1.2m"
                )
        finally:
            maou_logger.propagate = original

        assert "=== Training Configuration ===" in caplog.text
        assert "==============================" in caplog.text
