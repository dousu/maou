"""Tests for learn_multi_stage Stage 3 implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from maou.app.learning.multi_stage_training import (
    StageResult,
    TrainingStage,
)
from maou.interface.learn import (
    StageDataConfig,
    _find_latest_backbone_checkpoint,
)


class TestFindLatestBackboneCheckpoint:
    """_find_latest_backbone_checkpoint() のテスト."""

    def test_prefers_stage2_over_stage1(
        self, tmp_path: Path
    ) -> None:
        """stage2 checkpointがstage1より優先される."""
        (
            tmp_path / "stage1_backbone_20260101_000000.pt"
        ).touch()
        (
            tmp_path / "stage2_backbone_20260102_000000.pt"
        ).touch()

        result = _find_latest_backbone_checkpoint(tmp_path)
        assert result is not None
        assert "stage2_backbone_" in result.name

    def test_falls_back_to_stage1(self, tmp_path: Path) -> None:
        """stage2がない場合はstage1にフォールバック."""
        (
            tmp_path / "stage1_backbone_20260101_000000.pt"
        ).touch()

        result = _find_latest_backbone_checkpoint(tmp_path)
        assert result is not None
        assert "stage1_backbone_" in result.name

    def test_returns_none_when_no_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """checkpointがない場合はNone."""
        result = _find_latest_backbone_checkpoint(tmp_path)
        assert result is None

    def test_returns_latest_stage2(
        self, tmp_path: Path
    ) -> None:
        """複数のstage2 checkpointがある場合は最新を返す."""
        (
            tmp_path / "stage2_backbone_20260101_000000.pt"
        ).touch()
        (
            tmp_path / "stage2_backbone_20260102_120000.pt"
        ).touch()

        result = _find_latest_backbone_checkpoint(tmp_path)
        assert result is not None
        assert "20260102_120000" in result.name

    def test_ignores_non_backbone_files(
        self, tmp_path: Path
    ) -> None:
        """backbone以外のcheckpointファイルは無視する."""
        (
            tmp_path
            / "stage1_reachable_head_20260101_000000.pt"
        ).touch()
        (
            tmp_path
            / "stage2_legal_moves_head_20260101_000000.pt"
        ).touch()

        result = _find_latest_backbone_checkpoint(tmp_path)
        assert result is None


class TestLearnMultiStageStage3:
    """learn_multi_stage() のStage 3動作テスト."""

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage3_calls_learn(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_learn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """stage='3'がlearn()を呼び出す."""
        from maou.interface.learn import learn_multi_stage

        # Setup mocks
        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        # Create mock datasource via StageDataConfig
        mock_datasource = MagicMock()
        stage3_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="hcpe",
        )

        learn_multi_stage(
            stage="3",
            stage3_data_config=stage3_config,
            model_dir=tmp_path,
        )

        mock_learn.assert_called_once()
        call_kwargs = mock_learn.call_args
        assert (
            call_kwargs.kwargs.get("datasource")
            is mock_datasource
            or call_kwargs[1].get("datasource")
            is mock_datasource
        )

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage3_passes_trainable_layers(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_learn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """trainable_layersがStage 3に転送される."""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        mock_datasource = MagicMock()
        stage3_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="hcpe",
        )

        learn_multi_stage(
            stage="3",
            stage3_data_config=stage3_config,
            trainable_layers=2,
            model_dir=tmp_path,
        )

        call_kwargs = mock_learn.call_args
        assert call_kwargs.kwargs.get("trainable_layers") == 2

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn._find_latest_backbone_checkpoint"
    )
    @patch("maou.interface.learn._run_stage2")
    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage_all_uses_saved_backbone(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        mock_run_stage2: MagicMock,
        mock_find_checkpoint: MagicMock,
        mock_learn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """stage='all'がsaved backboneをStage 3に渡す."""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )

        mock_backbone = MagicMock()
        mock_backbone.embedding_dim = 512
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        # Mock _run_stage1/2 to return StageResult-like objects
        from maou.app.learning.multi_stage_training import (
            StageResult,
            TrainingStage,
        )

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )
        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        checkpoint_path = tmp_path / "stage2_backbone_test.pt"
        mock_find_checkpoint.return_value = checkpoint_path

        mock_s1_ds = MagicMock()
        mock_s2_ds = MagicMock()
        mock_s3_ds = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_s1_ds,
            array_type="stage1",
        )
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_s2_ds,
            array_type="stage2",
        )
        s3_config = StageDataConfig(
            create_datasource=lambda: mock_s3_ds,
            array_type="hcpe",
        )

        learn_multi_stage(
            stage="all",
            stage1_data_config=s1_config,
            stage2_data_config=s2_config,
            stage3_data_config=s3_config,
            model_dir=tmp_path,
        )

        call_kwargs = mock_learn.call_args
        assert (
            call_kwargs.kwargs.get("resume_backbone_from")
            == checkpoint_path
        )

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage3_passes_architecture_config(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_learn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """architecture_configがStage 3に転送される."""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        mock_datasource = MagicMock()
        stage3_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="hcpe",
        )

        config = {"embed_dim": 256, "num_layers": 3}
        learn_multi_stage(
            stage="3",
            stage3_data_config=stage3_config,
            model_dir=tmp_path,
            architecture_config=config,
        )

        call_kwargs = mock_learn.call_args
        assert (
            call_kwargs.kwargs.get("architecture_config")
            == config
        )
        # Also check that backbone was created with the config
        mock_model_factory.create_shogi_backbone.assert_called_once()
        factory_kwargs = (
            mock_model_factory.create_shogi_backbone.call_args
        )
        assert (
            factory_kwargs.kwargs.get("architecture_config")
            == config
        )


class TestStageSpecificBatchSize:
    """Stage別バッチサイズのフォールバック解決テスト．"""

    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage1_batch_size_overrides_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage1-batch-size指定時はその値がStage 1に渡される．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage1",
        )

        learn_multi_stage(
            stage="1",
            stage1_data_config=s1_config,
            batch_size=256,
            stage1_batch_size=32,
            model_dir=tmp_path,
        )

        mock_run_stage1.assert_called_once()
        call_kwargs = mock_run_stage1.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32

    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage1_batch_size_falls_back_to_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage1-batch-size未指定時はbatch_sizeにフォールバック．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage1",
        )

        learn_multi_stage(
            stage="1",
            stage1_data_config=s1_config,
            batch_size=256,
            stage1_batch_size=None,
            model_dir=tmp_path,
        )

        mock_run_stage1.assert_called_once()
        call_kwargs = mock_run_stage1.call_args
        assert call_kwargs.kwargs.get("batch_size") == 256

    @patch("maou.interface.learn._run_stage2")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage2_batch_size_overrides_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage2: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage2-batch-size指定時はその値がStage 2に渡される．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage2",
        )

        learn_multi_stage(
            stage="2",
            stage2_data_config=s2_config,
            batch_size=256,
            stage2_batch_size=64,
            model_dir=tmp_path,
        )

        mock_run_stage2.assert_called_once()
        call_kwargs = mock_run_stage2.call_args
        assert call_kwargs.kwargs.get("batch_size") == 64

    @patch("maou.interface.learn._run_stage2")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage2_batch_size_falls_back_to_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage2: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage2-batch-size未指定時はbatch_sizeにフォールバック．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage2",
        )

        learn_multi_stage(
            stage="2",
            stage2_data_config=s2_config,
            batch_size=256,
            stage2_batch_size=None,
            model_dir=tmp_path,
        )

        mock_run_stage2.assert_called_once()
        call_kwargs = mock_run_stage2.call_args
        assert call_kwargs.kwargs.get("batch_size") == 256

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn._find_latest_backbone_checkpoint"
    )
    @patch("maou.interface.learn._run_stage2")
    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage3_always_uses_global_batch_size(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        mock_run_stage2: MagicMock,
        mock_find_checkpoint: MagicMock,
        mock_learn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Stage 3はstage固有のbatch_sizeを持たず，globalを使用．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_backbone.embedding_dim = 512
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )
        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        checkpoint_path = tmp_path / "stage2_backbone_test.pt"
        mock_find_checkpoint.return_value = checkpoint_path

        mock_s1_ds = MagicMock()
        mock_s2_ds = MagicMock()
        mock_s3_ds = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_s1_ds,
            array_type="stage1",
        )
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_s2_ds,
            array_type="stage2",
        )
        s3_config = StageDataConfig(
            create_datasource=lambda: mock_s3_ds,
            array_type="hcpe",
        )

        learn_multi_stage(
            stage="all",
            stage1_data_config=s1_config,
            stage2_data_config=s2_config,
            stage3_data_config=s3_config,
            batch_size=256,
            stage1_batch_size=32,
            stage2_batch_size=64,
            model_dir=tmp_path,
        )

        # Stage 1 should use stage1_batch_size=32
        s1_kwargs = mock_run_stage1.call_args
        assert s1_kwargs.kwargs.get("batch_size") == 32

        # Stage 2 should use stage2_batch_size=64
        s2_kwargs = mock_run_stage2.call_args
        assert s2_kwargs.kwargs.get("batch_size") == 64

        # Stage 3 should use global batch_size=256
        mock_learn.assert_called_once()
        s3_kwargs = mock_learn.call_args
        assert s3_kwargs.kwargs.get("batch_size") == 256


class TestStageSpecificLearningRate:
    """Stage別学習率のフォールバック解決テスト．"""

    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage1_learning_rate_overrides_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage1-learning-rate指定時はその値がStage 1に渡される．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage1",
        )

        learn_multi_stage(
            stage="1",
            stage1_data_config=s1_config,
            learning_rate=0.001,
            stage1_learning_rate=0.0001,
            model_dir=tmp_path,
        )

        mock_run_stage1.assert_called_once()
        call_kwargs = mock_run_stage1.call_args
        assert call_kwargs.kwargs.get("learning_rate") == 0.0001

    @patch("maou.interface.learn._run_stage1")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage1_learning_rate_falls_back_to_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage1: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage1-learning-rate未指定時はlearning_rateにフォールバック．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage1.return_value = StageResult(
            stage=TrainingStage.REACHABLE_SQUARES,
            achieved_accuracy=0.99,
            final_loss=0.01,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s1_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage1",
        )

        learn_multi_stage(
            stage="1",
            stage1_data_config=s1_config,
            learning_rate=0.001,
            stage1_learning_rate=None,
            model_dir=tmp_path,
        )

        mock_run_stage1.assert_called_once()
        call_kwargs = mock_run_stage1.call_args
        assert call_kwargs.kwargs.get("learning_rate") == 0.001

    @patch("maou.interface.learn._run_stage2")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage2_learning_rate_overrides_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage2: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage2-learning-rate指定時はその値がStage 2に渡される．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage2",
        )

        learn_multi_stage(
            stage="2",
            stage2_data_config=s2_config,
            learning_rate=0.001,
            stage2_learning_rate=0.005,
            model_dir=tmp_path,
        )

        mock_run_stage2.assert_called_once()
        call_kwargs = mock_run_stage2.call_args
        assert call_kwargs.kwargs.get("learning_rate") == 0.005

    @patch("maou.interface.learn._run_stage2")
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage2_learning_rate_falls_back_to_global(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_orchestrator_cls: MagicMock,
        mock_run_stage2: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stage2-learning-rate未指定時はlearning_rateにフォールバック．"""
        from maou.interface.learn import learn_multi_stage

        mock_device_config = MagicMock()
        mock_device_config.device = MagicMock()
        mock_device_config.device.type = "cpu"
        mock_device_setup.setup_device.return_value = (
            mock_device_config
        )
        mock_backbone = MagicMock()
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_run_stage2.return_value = StageResult(
            stage=TrainingStage.LEGAL_MOVES,
            achieved_accuracy=0.95,
            final_loss=0.05,
            epochs_trained=5,
            threshold_met=True,
        )

        mock_datasource = MagicMock()
        s2_config = StageDataConfig(
            create_datasource=lambda: mock_datasource,
            array_type="stage2",
        )

        learn_multi_stage(
            stage="2",
            stage2_data_config=s2_config,
            learning_rate=0.001,
            stage2_learning_rate=None,
            model_dir=tmp_path,
        )

        mock_run_stage2.assert_called_once()
        call_kwargs = mock_run_stage2.call_args
        assert call_kwargs.kwargs.get("learning_rate") == 0.001
