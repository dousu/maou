"""Tests for learn_multi_stage Stage 3 implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from maou.interface.learn import (
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

        # Create mock datasource
        mock_datasource = MagicMock()
        mock_datasource.datasource.array_type = "hcpe"

        learn_multi_stage(
            stage="3",
            stage3_datasource=mock_datasource,
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
        mock_datasource.datasource.array_type = "hcpe"

        learn_multi_stage(
            stage="3",
            stage3_datasource=mock_datasource,
            trainable_layers=2,
            model_dir=tmp_path,
        )

        call_kwargs = mock_learn.call_args
        assert call_kwargs.kwargs.get("trainable_layers") == 2

    @patch("maou.interface.learn.learn")
    @patch(
        "maou.interface.learn._find_latest_backbone_checkpoint"
    )
    @patch(
        "maou.interface.learn.MultiStageTrainingOrchestrator"
    )
    @patch("maou.interface.learn.DataLoader")
    @patch("maou.interface.learn.Stage2Dataset")
    @patch("maou.interface.learn.Stage1Dataset")
    @patch("maou.interface.learn.ModelFactory")
    @patch("maou.interface.learn.DeviceSetup")
    def test_stage_all_uses_saved_backbone(
        self,
        mock_device_setup: MagicMock,
        mock_model_factory: MagicMock,
        mock_stage1_dataset: MagicMock,
        mock_stage2_dataset: MagicMock,
        mock_dataloader: MagicMock,
        mock_orchestrator_cls: MagicMock,
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
        import torch

        mock_backbone = MagicMock()
        mock_backbone.embedding_dim = 512
        # Provide real parameters so torch.optim.Adam doesn't error
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        mock_backbone.parameters.return_value = [dummy_param]
        mock_model_factory.create_shogi_backbone.return_value = mock_backbone
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all_stages.return_value = {}
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_learn.return_value = '{"Result": "Finish"}'

        checkpoint_path = tmp_path / "stage2_backbone_test.pt"
        mock_find_checkpoint.return_value = checkpoint_path

        mock_s1 = MagicMock()
        mock_s1.datasource.array_type = "stage1"
        mock_s2 = MagicMock()
        mock_s2.datasource.array_type = "stage2"
        mock_s3 = MagicMock()
        mock_s3.datasource.array_type = "hcpe"

        learn_multi_stage(
            stage="all",
            stage1_datasource=mock_s1,
            stage2_datasource=mock_s2,
            stage3_datasource=mock_s3,
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
        mock_datasource.datasource.array_type = "hcpe"

        config = {"embed_dim": 256, "num_layers": 3}
        learn_multi_stage(
            stage="3",
            stage3_datasource=mock_datasource,
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
