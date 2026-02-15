"""Tests for TensorBoard hparams recording in Learning."""

from unittest.mock import MagicMock

from maou.app.learning.callbacks import ValidationMetrics
from maou.app.learning.dl import Learning


def _make_learning_with_config(
    *,
    trainable_layers: int | None = None,
    freeze_backbone: bool = False,
    model_architecture: str = "resnet",
    architecture_config: dict | None = None,
    learning_ratio: float = 0.001,
    optimizer_name: str = "adamw",
    lr_scheduler_name: str | None = None,
    batch_size: int = 256,
    epoch: int = 100,
    gce_parameter: float = 0.7,
    policy_loss_ratio: float = 1.0,
    value_loss_ratio: float = 1.0,
) -> Learning:
    """Create a Learning instance with config and architecture_config set."""
    learning = Learning()
    learning.freeze_backbone = freeze_backbone
    learning.trainable_layers = trainable_layers
    learning.architecture_config = architecture_config

    # Create a mock config
    config = MagicMock()
    config.trainable_layers = trainable_layers
    config.freeze_backbone = freeze_backbone
    config.model_architecture = model_architecture
    config.learning_ratio = learning_ratio
    config.optimizer_name = optimizer_name
    config.lr_scheduler_name = lr_scheduler_name
    config.batch_size = batch_size
    config.epoch = epoch
    config.gce_parameter = gce_parameter
    config.policy_loss_ratio = policy_loss_ratio
    config.value_loss_ratio = value_loss_ratio
    learning.config = config

    return learning


def _make_metrics(
    *,
    policy_top5_accuracy: float = 0.85,
    policy_cross_entropy: float = 2.1,
    value_brier_score: float = 0.15,
    policy_f1_score: float = 0.5,
    value_high_confidence_rate: float = 0.7,
) -> ValidationMetrics:
    """Create ValidationMetrics for testing."""
    return ValidationMetrics(
        policy_cross_entropy=policy_cross_entropy,
        value_brier_score=value_brier_score,
        policy_top5_accuracy=policy_top5_accuracy,
        policy_f1_score=policy_f1_score,
        value_high_confidence_rate=value_high_confidence_rate,
    )


class TestLogHparams:
    """Tests for _log_hparams method."""

    def test_hparams_called_with_correct_dict(self) -> None:
        """add_hparams() is called with expected hparam and metric dicts."""
        learning = _make_learning_with_config(
            trainable_layers=2,
            freeze_backbone=False,
            model_architecture="resnet",
        )
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m-tl2",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        writer.add_hparams.assert_called_once()
        call_args = writer.add_hparams.call_args
        hparam_dict = call_args[0][0]
        metric_dict = call_args[0][1]

        assert hparam_dict["trainable_layers"] == 2
        assert hparam_dict["model_architecture"] == "resnet"
        assert hparam_dict["model_tag"] == "resnet-1.2m-tl2"
        assert hparam_dict["optimizer_name"] == "adamw"

        assert metric_dict["hparam/best_vloss"] == 0.5
        assert (
            metric_dict["hparam/policy_top5_accuracy"] == 0.85
        )

    def test_trainable_layers_none_becomes_negative_one(
        self,
    ) -> None:
        """trainable_layers=None is recorded as -1 sentinel."""
        learning = _make_learning_with_config(
            trainable_layers=None,
        )
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        hparam_dict = writer.add_hparams.call_args[0][0]
        assert hparam_dict["trainable_layers"] == -1

    def test_vit_architecture_config_included(self) -> None:
        """ViT-specific params from architecture_config are added."""
        learning = _make_learning_with_config(
            model_architecture="vit",
            architecture_config={
                "embed_dim": 256,
                "num_layers": 8,
                "num_heads": 8,
            },
        )
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="vit-19m",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        hparam_dict = writer.add_hparams.call_args[0][0]
        assert hparam_dict["vit_embed_dim"] == 256
        assert hparam_dict["vit_num_layers"] == 8
        assert hparam_dict["vit_num_heads"] == 8

    def test_resnet_no_vit_keys(self) -> None:
        """ResNet architecture does not include ViT keys."""
        learning = _make_learning_with_config(
            model_architecture="resnet",
            architecture_config=None,
        )
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        hparam_dict = writer.add_hparams.call_args[0][0]
        assert "vit_embed_dim" not in hparam_dict
        assert "vit_num_layers" not in hparam_dict
        assert "vit_num_heads" not in hparam_dict

    def test_run_name_is_dot(self) -> None:
        """run_name='.' is passed to avoid sub-run directory."""
        learning = _make_learning_with_config()
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        call_kwargs = writer.add_hparams.call_args[1]
        assert call_kwargs["run_name"] == "."

    def test_lr_scheduler_none_becomes_string(self) -> None:
        """lr_scheduler_name=None is recorded as 'none'."""
        learning = _make_learning_with_config(
            lr_scheduler_name=None,
        )
        writer = MagicMock()
        metrics = _make_metrics()

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m",
            best_vloss=0.5,
            final_metrics=metrics,
        )

        hparam_dict = writer.add_hparams.call_args[0][0]
        assert hparam_dict["lr_scheduler_name"] == "none"

    def test_metric_dict_keys(self) -> None:
        """All expected metric keys are present."""
        learning = _make_learning_with_config()
        writer = MagicMock()
        metrics = _make_metrics(
            policy_cross_entropy=2.1,
            value_brier_score=0.15,
        )

        learning._log_hparams(
            writer=writer,
            model_tag="resnet-1.2m",
            best_vloss=0.42,
            final_metrics=metrics,
        )

        metric_dict = writer.add_hparams.call_args[0][1]
        assert set(metric_dict.keys()) == {
            "hparam/best_vloss",
            "hparam/policy_top5_accuracy",
            "hparam/policy_cross_entropy",
            "hparam/value_brier_score",
        }
        assert metric_dict["hparam/best_vloss"] == 0.42
        assert metric_dict["hparam/policy_cross_entropy"] == 2.1
        assert metric_dict["hparam/value_brier_score"] == 0.15
