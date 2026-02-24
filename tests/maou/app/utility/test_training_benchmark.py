"""Tests for training benchmark components."""

from __future__ import annotations

import pytest
import torch

from maou.app.utility.training_benchmark import (
    BenchmarkResult,
    TrainingBenchmarkConfig,
    TrainingBenchmarkUseCase,
)
from maou.interface import utility_interface


class TestBenchmarkResult:
    """BenchmarkResult dataclass のテスト."""

    def _make_result(
        self,
        *,
        warmup_time: float = 34.0,
        warmup_batches: int = 5,
        measured_time: float = 4.75,
        measured_batches: int = 95,
        actual_average_batch_time: float = 0.05,
        data_load_method: str = "map-style",
    ) -> BenchmarkResult:
        """テスト用の BenchmarkResult を生成する."""
        return BenchmarkResult(
            total_epoch_time=warmup_time + measured_time,
            average_batch_time=0.04,
            actual_average_batch_time=actual_average_batch_time,
            total_batches=warmup_batches + measured_batches,
            warmup_time=warmup_time,
            warmup_batches=warmup_batches,
            measured_time=measured_time,
            measured_batches=measured_batches,
            data_loading_time=0.01,
            gpu_transfer_time=0.005,
            forward_pass_time=0.015,
            loss_computation_time=0.002,
            backward_pass_time=0.005,
            optimizer_step_time=0.003,
            final_loss=0.5,
            average_loss=0.6,
            samples_per_second=640.0,
            batches_per_second=20.0,
            data_load_method=data_load_method,
        )

    def test_warmup_fields_present(self) -> None:
        """ウォームアップ関連フィールドが正しく設定される."""
        result = self._make_result()
        assert result.warmup_time == 34.0
        assert result.warmup_batches == 5
        assert result.measured_time == 4.75
        assert result.measured_batches == 95

    def test_actual_average_batch_time_excludes_warmup(
        self,
    ) -> None:
        """actual_average_batch_time がウォームアップを除外した値."""
        result = self._make_result(
            warmup_time=34.0,
            measured_time=4.75,
            measured_batches=95,
            actual_average_batch_time=4.75 / 95,
        )
        assert result.actual_average_batch_time == 4.75 / 95

    def test_estimated_epoch_time_excludes_warmup(self) -> None:
        """推定エポック時間がウォームアップを含まない定常速度で算出される.

        format_timing_summary 内の推定式:
            estimated = actual_average_batch_time * total_batches_in_dataset
        ウォームアップ時間は含めない（初回エポック限りのコスト）．
        """
        total_batches_in_dataset = 10000
        avg_batch_time = 0.05  # 50ms/batch

        result = self._make_result(
            warmup_time=34.0,
            actual_average_batch_time=avg_batch_time,
        )

        # 推定式（format_timing_summary と同じ）
        estimated = (
            result.actual_average_batch_time
            * total_batches_in_dataset
        )

        # 500秒（warmup 34秒は含まない）
        assert (
            estimated
            == avg_batch_time * total_batches_in_dataset
        )
        assert estimated == 500.0

    def test_to_dict_includes_warmup_fields(self) -> None:
        """to_dict にウォームアップ関連フィールドが含まれる."""
        result = self._make_result()
        d = result.to_dict()
        assert "warmup_time" in d
        assert "warmup_batches" in d
        assert "measured_time" in d
        assert "measured_batches" in d
        assert d["warmup_time"] == 34.0
        assert d["warmup_batches"] == 5.0
        assert d["measured_time"] == 4.75
        assert d["measured_batches"] == 95.0

    def test_zero_warmup(self) -> None:
        """warmup_batches=0 の場合のフィールド値."""
        result = self._make_result(
            warmup_time=0.0,
            warmup_batches=0,
            measured_time=10.0,
            measured_batches=100,
            actual_average_batch_time=0.1,
        )
        assert result.warmup_time == 0.0
        assert result.warmup_batches == 0
        assert result.total_epoch_time == 10.0

    def test_data_load_method_default(self) -> None:
        """data_load_method のデフォルト値が map-style."""
        result = self._make_result()
        assert result.data_load_method == "map-style"

    def test_data_load_method_streaming(self) -> None:
        """data_load_method に streaming を設定できる."""
        result = self._make_result(data_load_method="streaming")
        assert result.data_load_method == "streaming"

    def test_to_dict_includes_data_load_method(self) -> None:
        """to_dict に data_load_method が含まれる."""
        result = self._make_result(data_load_method="streaming")
        d = result.to_dict()
        assert "data_load_method" in d
        assert d["data_load_method"] == "streaming"


class TestTrainingBenchmarkConfig:
    """TrainingBenchmarkConfig dataclass のテスト."""

    def test_warmup_batches_default_is_10(self) -> None:
        """warmup_batches デフォルトが10."""
        config = TrainingBenchmarkConfig()
        assert config.warmup_batches == 10

    def test_streaming_defaults(self) -> None:
        """streaming関連フィールドのデフォルト値."""
        config = TrainingBenchmarkConfig()
        assert config.streaming is False
        assert config.streaming_train_source is None
        assert config.streaming_val_source is None

    def test_datasource_optional(self) -> None:
        """datasource が None を許容する."""
        config = TrainingBenchmarkConfig(datasource=None)
        assert config.datasource is None


class TestBenchmarkTrainingValidation:
    """Interface層バリデーションのテスト."""

    def test_non_streaming_without_datasource_raises_value_error(
        self,
    ) -> None:
        """streaming=False かつ datasource=None で ValueError."""
        with pytest.raises(
            ValueError, match="datasource is required"
        ):
            utility_interface.benchmark_training(
                datasource=None,
                streaming=False,
            )

    def test_streaming_without_train_source_raises_value_error(
        self,
    ) -> None:
        """streaming=True かつ streaming_train_source=None で ValueError."""
        with pytest.raises(
            ValueError,
            match="streaming_train_source is required",
        ):
            utility_interface.benchmark_training(
                datasource=None,
                streaming=True,
                streaming_train_source=None,
            )


def test_single_epoch_benchmark_accepts_training_loop_class() -> (
    None
):
    """SingleEpochBenchmark が training_loop_class パラメータを受け取れることを確認する．"""
    from unittest.mock import MagicMock

    from maou.app.learning.training_loop import (
        RawLogitsTrainingLoop,
        TrainingLoop,
    )
    from maou.app.utility.training_benchmark import (
        SingleEpochBenchmark,
    )

    model = MagicMock()
    device = torch.device("cpu")
    optimizer = MagicMock()
    loss_policy = MagicMock()
    loss_value = MagicMock()

    # Default should be TrainingLoop
    benchmark = SingleEpochBenchmark(
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn_policy=loss_policy,
        loss_fn_value=loss_value,
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
    )
    assert benchmark.training_loop_class is TrainingLoop

    # Should accept RawLogitsTrainingLoop
    benchmark_s2 = SingleEpochBenchmark(
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn_policy=loss_policy,
        loss_fn_value=loss_value,
        policy_loss_ratio=1.0,
        value_loss_ratio=0.0,
        training_loop_class=RawLogitsTrainingLoop,
    )
    assert (
        benchmark_s2.training_loop_class
        is RawLogitsTrainingLoop
    )


def test_training_benchmark_config_stage_defaults() -> None:
    """TrainingBenchmarkConfig の新フィールドのデフォルト値を確認する．"""
    from maou.app.utility.training_benchmark import (
        TrainingBenchmarkConfig,
    )

    config = TrainingBenchmarkConfig()
    assert config.stage == 3
    assert config.stage1_datasource is None
    assert config.stage2_datasource is None
    assert config.stage2_streaming_train_source is None
    assert config.stage2_streaming_val_source is None
    assert config.stage12_lr_scheduler_name is None
    assert config.stage12_compilation is False
    assert config.stage1_pos_weight == 1.0
    assert config.stage2_pos_weight == 1.0
    assert config.stage2_gamma_pos == 0.0
    assert config.stage2_gamma_neg == 0.0
    assert config.stage2_clip == 0.0
    assert config.stage2_hidden_dim == 128
    assert config.stage2_head_dropout == 0.0
    assert config.stage2_test_ratio == 0.2


def test_training_benchmark_config_stage1() -> None:
    """Stage 1 用 TrainingBenchmarkConfig が正しく生成されることを確認する．"""
    from unittest.mock import MagicMock

    from maou.app.utility.training_benchmark import (
        TrainingBenchmarkConfig,
    )

    mock_datasource = MagicMock()
    config = TrainingBenchmarkConfig(
        stage=1,
        stage1_datasource=mock_datasource,
        stage1_pos_weight=2.0,
    )
    assert config.stage == 1
    assert config.stage1_datasource is mock_datasource
    assert config.stage1_pos_weight == 2.0


def test_training_benchmark_config_stage2() -> None:
    """Stage 2 用 TrainingBenchmarkConfig が正しく生成されることを確認する．"""
    from unittest.mock import MagicMock

    from maou.app.utility.training_benchmark import (
        TrainingBenchmarkConfig,
    )

    mock_datasource = MagicMock()
    config = TrainingBenchmarkConfig(
        stage=2,
        stage2_datasource=mock_datasource,
        stage2_pos_weight=3.0,
        stage2_gamma_neg=2.0,
        stage2_clip=0.02,
        stage2_hidden_dim=256,
        stage2_head_dropout=0.1,
    )
    assert config.stage == 2
    assert config.stage2_datasource is mock_datasource
    assert config.stage2_pos_weight == 3.0
    assert config.stage2_gamma_neg == 2.0
    assert config.stage2_clip == 0.02
    assert config.stage2_hidden_dim == 256
    assert config.stage2_head_dropout == 0.1


def test_stage1_benchmark_warmup_adjustment() -> None:
    """warmup_batches >= total_batches の場合に自動調整されることを確認する．"""
    # Test the adjustment logic directly
    warmup_batches = 10
    estimated_batches = 5  # Small dataset like Stage 1

    effective_warmup = min(
        warmup_batches, estimated_batches - 2
    )
    effective_warmup = max(0, effective_warmup)

    assert effective_warmup == 3  # 5 - 2 = 3

    # Edge case: very small dataset
    warmup_batches = 10
    estimated_batches = 1

    effective_warmup = min(
        warmup_batches, estimated_batches - 2
    )
    effective_warmup = max(0, effective_warmup)

    assert effective_warmup == 0  # max(0, 1-2) = 0

    # Normal case: enough batches
    warmup_batches = 5
    estimated_batches = 100

    effective_warmup = min(
        warmup_batches, estimated_batches - 2
    )
    effective_warmup = max(0, effective_warmup)

    assert effective_warmup == 5  # No adjustment needed


class TestTrainingBenchmarkConfigNewFields:
    """新規追加フィールドのデフォルト値テスト."""

    def test_architecture_config_default(self) -> None:
        """architecture_config のデフォルト値が None であること."""
        config = TrainingBenchmarkConfig()
        assert config.architecture_config is None

    def test_freeze_backbone_default(self) -> None:
        """freeze_backbone のデフォルト値が False であること."""
        config = TrainingBenchmarkConfig()
        assert config.freeze_backbone is False

    def test_trainable_layers_default(self) -> None:
        """trainable_layers のデフォルト値が None であること."""
        config = TrainingBenchmarkConfig()
        assert config.trainable_layers is None

    def test_stage1_batch_size_default(self) -> None:
        """stage1_batch_size のデフォルト値が None であること."""
        config = TrainingBenchmarkConfig()
        assert config.stage1_batch_size is None

    def test_stage2_batch_size_default(self) -> None:
        """stage2_batch_size のデフォルト値が None であること."""
        config = TrainingBenchmarkConfig()
        assert config.stage2_batch_size is None

    def test_stage1_batch_size_override(self) -> None:
        """stage1_batch_size の上書きが反映されること."""
        config = TrainingBenchmarkConfig(stage1_batch_size=512)
        assert config.stage1_batch_size == 512

    def test_stage2_batch_size_override(self) -> None:
        """stage2_batch_size の上書きが反映されること."""
        config = TrainingBenchmarkConfig(stage2_batch_size=128)
        assert config.stage2_batch_size == 128

    def test_architecture_config_with_vit(self) -> None:
        """ViT architecture_config が正しく設定されること."""
        config = TrainingBenchmarkConfig(
            model_architecture="vit",
            architecture_config={
                "embed_dim": 256,
                "num_layers": 4,
            },
        )
        assert config.architecture_config == {
            "embed_dim": 256,
            "num_layers": 4,
        }


class TestBenchmarkTrainingNewValidation:
    """新規パラメータのバリデーションテスト."""

    def test_negative_trainable_layers_raises(self) -> None:
        """trainable_layers が負の値でエラーになること."""
        with pytest.raises(
            ValueError, match="trainable_layers"
        ):
            utility_interface.benchmark_training(
                trainable_layers=-1,
            )

    def test_zero_stage1_batch_size_raises(self) -> None:
        """stage1_batch_size が 0 でエラーになること."""
        with pytest.raises(
            ValueError, match="stage1_batch_size"
        ):
            utility_interface.benchmark_training(
                stage1_batch_size=0,
            )

    def test_zero_stage2_batch_size_raises(self) -> None:
        """stage2_batch_size が 0 でエラーになること."""
        with pytest.raises(
            ValueError, match="stage2_batch_size"
        ):
            utility_interface.benchmark_training(
                stage2_batch_size=0,
            )


class TestResolveTrainableLayers:
    """TrainingBenchmarkUseCase._resolve_trainable_layers のテスト."""

    def setup_method(self) -> None:
        """テスト用の use case インスタンスを作成."""
        self.use_case = TrainingBenchmarkUseCase()

    def test_both_unset_returns_none(self) -> None:
        """両方未設定の場合 None を返すこと."""
        config = TrainingBenchmarkConfig()
        assert (
            self.use_case._resolve_trainable_layers(config)
            is None
        )

    def test_freeze_backbone_only_returns_zero(self) -> None:
        """freeze_backbone のみ True の場合 0 を返すこと."""
        config = TrainingBenchmarkConfig(freeze_backbone=True)
        assert (
            self.use_case._resolve_trainable_layers(config) == 0
        )

    def test_trainable_layers_only(self) -> None:
        """trainable_layers のみ設定の場合その値を返すこと."""
        config = TrainingBenchmarkConfig(trainable_layers=3)
        assert (
            self.use_case._resolve_trainable_layers(config) == 3
        )

    def test_both_set_prefers_trainable_layers(self) -> None:
        """両方設定の場合 trainable_layers を優先すること."""
        config = TrainingBenchmarkConfig(
            freeze_backbone=True, trainable_layers=2
        )
        assert (
            self.use_case._resolve_trainable_layers(config) == 2
        )


class TestApplyLayerFreezing:
    """TrainingBenchmarkUseCase._apply_layer_freezing のテスト."""

    def setup_method(self) -> None:
        """テスト用の use case インスタンスを作成."""
        self.use_case = TrainingBenchmarkUseCase()

    def test_freeze_network_stage3(self) -> None:
        """Stage 3 モデル (Network) でのフリーズ."""
        from maou.app.learning.setup import ModelFactory

        model = ModelFactory.create_shogi_model(
            torch.device("cpu")
        )
        # 凍結前は全パラメータが学習可能
        trainable_before = sum(
            1 for p in model.parameters() if p.requires_grad
        )
        assert trainable_before > 0

        self.use_case._apply_layer_freezing(
            model, trainable_layers=0
        )

        # backbone + hand_projection が凍結される
        # policy/value head は学習可能のまま
        trainable_after = sum(
            1 for p in model.parameters() if p.requires_grad
        )
        assert trainable_after < trainable_before
        assert trainable_after > 0  # head のパラメータは残る

    def test_freeze_stage1_adapter(self) -> None:
        """Stage 1 アダプタ (backbone wrapper) でのフリーズ."""
        from maou.app.learning.multi_stage_training import (
            Stage1ModelAdapter,
        )
        from maou.app.learning.network import (
            ReachableSquaresHead,
        )
        from maou.app.learning.setup import ModelFactory

        backbone = ModelFactory.create_shogi_backbone(
            torch.device("cpu")
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        model = Stage1ModelAdapter(backbone, head)

        self.use_case._apply_layer_freezing(
            model, trainable_layers=0
        )

        # backbone のパラメータが凍結されている
        frozen_count = sum(
            1
            for p in backbone.parameters()
            if not p.requires_grad
        )
        assert frozen_count > 0

    def test_unsupported_model_does_not_freeze(self) -> None:
        """freeze_except_last_n も backbone も持たないモデルでは凍結されないこと."""
        model = torch.nn.Linear(10, 10)
        trainable_before = sum(
            1 for p in model.parameters() if p.requires_grad
        )

        # Should not raise, just log a warning and return
        self.use_case._apply_layer_freezing(
            model, trainable_layers=0
        )

        # Parameters should remain unchanged (no freezing applied)
        trainable_after = sum(
            1 for p in model.parameters() if p.requires_grad
        )
        assert trainable_after == trainable_before


class TestValidationWarningLogic:
    """Validation 警告ロジックのテスト (タスク#2)．

    execute() 内の validation 分岐を検証する．
    重いコンポーネント(モデル生成，学習)はモックで代替する．
    """

    def _make_fake_benchmark_result(self) -> BenchmarkResult:
        """ダミーの BenchmarkResult を生成する．"""
        return BenchmarkResult(
            total_epoch_time=1.0,
            average_batch_time=0.1,
            actual_average_batch_time=0.1,
            total_batches=10,
            warmup_time=0.2,
            warmup_batches=2,
            measured_time=0.8,
            measured_batches=8,
            data_loading_time=0.01,
            gpu_transfer_time=0.005,
            forward_pass_time=0.05,
            loss_computation_time=0.01,
            backward_pass_time=0.02,
            optimizer_step_time=0.01,
            final_loss=0.5,
            average_loss=0.6,
            samples_per_second=100.0,
            batches_per_second=10.0,
            data_load_method="map-style",
        )

    def _run_execute_with_patches(
        self,
        config: TrainingBenchmarkConfig,
    ) -> str:
        """パッチ適用済みの execute() を実行し JSON 文字列を返す．"""
        from unittest.mock import MagicMock, patch

        use_case = TrainingBenchmarkUseCase()
        fake_result = self._make_fake_benchmark_result()

        # setup メソッドのモック返却値
        device_config = MagicMock()
        device_config.device = torch.device("cpu")
        device_config.pin_memory = False

        train_loader = MagicMock()
        train_loader.__len__ = MagicMock(return_value=10)
        val_loader = MagicMock()
        val_loader.__len__ = MagicMock(return_value=5)
        dataloaders = (train_loader, val_loader)

        model_components = MagicMock()
        model_components.model = MagicMock()
        model_components.optimizer = MagicMock()
        model_components.lr_scheduler = None

        setup_return = (
            device_config,
            dataloaders,
            model_components,
        )

        # SingleEpochBenchmark のモック
        mock_benchmark = MagicMock()
        mock_benchmark.benchmark_epoch.return_value = (
            fake_result
        )
        mock_benchmark.benchmark_validation.return_value = (
            fake_result
        )

        with (
            patch.object(
                use_case,
                "_setup_stage1_components",
                return_value=setup_return,
            ),
            patch.object(
                use_case,
                "_setup_stage2_components",
                return_value=setup_return,
            ),
            patch.object(
                use_case,
                "_setup_stage2_streaming_components",
                return_value=setup_return,
            ),
            patch(
                "maou.app.utility.training_benchmark.SingleEpochBenchmark",
                return_value=mock_benchmark,
            ),
            patch(
                "maou.app.utility.training_benchmark.compile_module",
                side_effect=lambda m: m,
            ),
        ):
            return use_case.execute(config)

    def test_stage1_run_validation_emits_warning(
        self,
    ) -> None:
        """Stage 1 + run_validation=True → validation_skipped が JSON 出力に含まれる．"""
        import json

        config = TrainingBenchmarkConfig(
            stage=1,
            run_validation=True,
        )
        result_json = self._run_execute_with_patches(config)
        result = json.loads(result_json)

        # validation_skipped キーが出力に含まれる
        assert (
            "validation_skipped" in result["benchmark_results"]
        )
        assert (
            "Stage 1"
            in result["benchmark_results"]["validation_skipped"]
        )
        # ValidationSummary は出力に含まれない
        assert (
            "ValidationSummary"
            not in result["benchmark_results"]
        )

    def test_stage2_run_validation_no_test_ratio_emits_warning(
        self,
    ) -> None:
        """Stage 2 + run_validation=True + stage2_test_ratio=0 → validation_skipped が出力される．"""
        import json

        config = TrainingBenchmarkConfig(
            stage=2,
            run_validation=True,
            stage2_test_ratio=0.0,
        )
        result_json = self._run_execute_with_patches(config)
        result = json.loads(result_json)

        assert (
            "validation_skipped" in result["benchmark_results"]
        )
        assert (
            "test-ratio"
            in result["benchmark_results"]["validation_skipped"]
        )
        assert (
            "ValidationSummary"
            not in result["benchmark_results"]
        )

    def test_stage2_run_validation_with_test_ratio_executes(
        self,
    ) -> None:
        """Stage 2 + run_validation=True + stage2_test_ratio>0 → validation が実行される．"""
        import json

        config = TrainingBenchmarkConfig(
            stage=2,
            run_validation=True,
            stage2_test_ratio=0.2,
        )
        result_json = self._run_execute_with_patches(config)
        result = json.loads(result_json)

        # validation_skipped がない
        assert (
            "validation_skipped"
            not in result["benchmark_results"]
        )
        # ValidationSummary が存在する
        assert (
            "ValidationSummary" in result["benchmark_results"]
        )

    def test_validation_skipped_json_output_stage1(
        self,
    ) -> None:
        """Stage 1 スキップ時 JSON に validation_skipped キーと理由が含まれる．"""
        import json

        config = TrainingBenchmarkConfig(
            stage=1,
            run_validation=True,
        )
        result_json = self._run_execute_with_patches(config)
        result = json.loads(result_json)

        skipped = result["benchmark_results"].get(
            "validation_skipped"
        )
        assert skipped is not None
        assert "Stage 1" in skipped
        assert "ignored" in skipped.lower()

    def test_validation_skipped_json_output_stage2_no_ratio(
        self,
    ) -> None:
        """Stage 2 + test_ratio=0 スキップ時 JSON に validation_skipped キーと理由が含まれる．"""
        import json

        config = TrainingBenchmarkConfig(
            stage=2,
            run_validation=True,
            stage2_test_ratio=0.0,
        )
        result_json = self._run_execute_with_patches(config)
        result = json.loads(result_json)

        skipped = result["benchmark_results"].get(
            "validation_skipped"
        )
        assert skipped is not None
        assert "stage2-test-ratio" in skipped
