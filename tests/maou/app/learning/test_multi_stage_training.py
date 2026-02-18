"""Tests for multi-stage training loop with HeadlessNetwork backbone."""

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from maou.app.learning.multi_stage_training import (
    MultiStageTrainingOrchestrator,
    SingleStageTrainingLoop,
    StageConfig,
    TrainingStage,
)
from maou.app.learning.network import (
    PIECES_IN_HAND_VECTOR_SIZE,
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)


def _make_dummy_dataloader(
    *,
    batch_size: int = 4,
    num_samples: int = 8,
    board_vocab_size: int = 32,
    board_size: tuple[int, int] = (9, 9),
    target_dim: int = 81,
    hand_dim: int = PIECES_IN_HAND_VECTOR_SIZE,
) -> DataLoader:
    """ダミーの (board_tensor, hand_tensor), target データローダーを作成する．

    Stage 1/2 のデータセットは ((board_tensor, hand_tensor), target) を返す．
    """
    board_tensor = torch.randint(
        0, board_vocab_size, (num_samples, *board_size)
    )
    hand_tensor = torch.randn(num_samples, hand_dim)
    targets = torch.zeros(num_samples, target_dim)

    dataset = TensorDataset(board_tensor, hand_tensor, targets)

    def collate_fn(
        batch: list[tuple[torch.Tensor, ...]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        boards = torch.stack([b[0] for b in batch])
        hands = torch.stack([b[1] for b in batch])
        tgts = torch.stack([b[2] for b in batch])
        return (boards, hands), tgts

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


class TestSingleStageTrainingLoopWithHeadlessNetwork:
    """_train_epoch が HeadlessNetwork で正常に動作するテスト．"""

    def test_train_epoch_stage1_completes(self) -> None:
        """Stage 1: _train_epoch が HeadlessNetwork + ReachableSquaresHead で完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_train_epoch_stage2_completes(self) -> None:
        """Stage 2: _train_epoch が HeadlessNetwork + LegalMovesHead で完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        from maou.domain.move.label import MOVE_LABELS_NUM

        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_forward_features_receives_model_inputs(
        self,
    ) -> None:
        """forward_features が ModelInputs (テンソルまたはタプル) を受け取ることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )

        board_tensor = torch.randint(0, 32, (2, 9, 9))
        features = backbone.forward_features(board_tensor)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 2
        assert features.shape[1] == backbone.embedding_dim

    def test_run_stage1_e2e(self) -> None:
        """Stage 1 E2E: SingleStageTrainingLoop.run が StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)

    def test_run_stage2_e2e(self) -> None:
        """Stage 2 E2E: SingleStageTrainingLoop.run が StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        from maou.domain.move.label import MOVE_LABELS_NUM

        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.LEGAL_MOVES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)

    def test_optimizer_includes_all_parameters(self) -> None:
        """SingleStageTrainingLoopがbackbone+headの全パラメータでoptimizerを生成する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        # optimizerのパラメータ数 = backbone + head のパラメータ数
        optimizer_param_count = sum(
            1
            for group in loop.optimizer.param_groups
            for _ in group["params"]
        )
        expected_param_count = len(
            list(backbone.parameters())
        ) + len(list(head.parameters()))
        assert optimizer_param_count == expected_param_count

        # learning_rateが正しく設定されている
        assert loop.optimizer.param_groups[0]["lr"] == 1e-3


class TestSingleStageTrainingLoopWithHandProjection:
    """hand_projection_dim > 0 の HeadlessNetwork で Stage 1/2 が動作するテスト．"""

    def test_train_epoch_stage1_with_hand_projection(
        self,
    ) -> None:
        """Stage 1: hand_projection_dim > 0 で _train_epoch が完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_train_epoch_stage2_with_hand_projection(
        self,
    ) -> None:
        """Stage 2: hand_projection_dim > 0 で _train_epoch が完走する．"""
        from maou.domain.move.label import MOVE_LABELS_NUM

        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_run_stage1_e2e_with_hand_projection(
        self,
    ) -> None:
        """Stage 1 E2E: hand_projection_dim > 0 で StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)


class TestThresholdErrorMessages:
    """Threshold未達時のエラーメッセージフォーマットを検証するテスト．"""

    def _make_orchestrator(
        self,
        tmp_path: Path,
    ) -> MultiStageTrainingOrchestrator:
        """テスト用のオーケストレーターを作成する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        return MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=torch.device("cpu"),
            model_dir=tmp_path / "checkpoints",
        )

    def _make_stage_config(
        self,
        stage: TrainingStage,
        threshold: float = 0.99,
    ) -> StageConfig:
        """テスト用のStageConfigを作成する．"""
        if stage == TrainingStage.REACHABLE_SQUARES:
            target_dim = 81
        else:
            from maou.domain.move.label import MOVE_LABELS_NUM

            target_dim = MOVE_LABELS_NUM

        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=target_dim
        )
        return StageConfig(
            stage=stage,
            max_epochs=1,
            accuracy_threshold=threshold,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

    def test_stage1_error_message_format(
        self, tmp_path: Path
    ) -> None:
        """Stage 1 threshold未達時のエラーメッセージに必要な情報が含まれることを検証する．"""
        orchestrator = self._make_orchestrator(tmp_path)
        config = self._make_stage_config(
            TrainingStage.REACHABLE_SQUARES,
            threshold=0.99,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Stage 1 failed to meet accuracy threshold",
        ) as exc_info:
            orchestrator.run_all_stages(
                stage1_config=config,
                save_checkpoints=False,
            )

        msg = str(exc_info.value)
        assert "after" in msg and "epochs" in msg
        assert "achieved:" in msg
        assert "required:" in msg
        assert "of target)" in msg
        assert "--stage1-max-epochs" in msg
        assert "--stage1-threshold" in msg

    def test_stage2_error_message_format(
        self, tmp_path: Path
    ) -> None:
        """Stage 2 threshold未達時のエラーメッセージに必要な情報が含まれることを検証する．"""
        orchestrator = self._make_orchestrator(tmp_path)
        stage1_config = self._make_stage_config(
            TrainingStage.REACHABLE_SQUARES,
            threshold=0.0,
        )
        stage2_config = self._make_stage_config(
            TrainingStage.LEGAL_MOVES,
            threshold=0.99,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Stage 2 failed to meet accuracy threshold",
        ) as exc_info:
            orchestrator.run_all_stages(
                stage1_config=stage1_config,
                stage2_config=stage2_config,
                save_checkpoints=False,
            )

        msg = str(exc_info.value)
        assert "after" in msg and "epochs" in msg
        assert "achieved:" in msg
        assert "required:" in msg
        assert "of target)" in msg
        assert "--stage2-max-epochs" in msg
        assert "--stage2-threshold" in msg


class TestStage2F1Metric:
    """Stage2のF1メトリック計算を検証するテスト．"""

    def test_stage2_f1_metric_computation(self) -> None:
        """既知のlogits/targetsでF1スコアが正しく計算されることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        from maou.domain.move.label import MOVE_LABELS_NUM

        head = LegalMovesHead(input_dim=backbone.embedding_dim)

        # 手計算可能なtargetsを作成: 最初の10ラベルが合法手
        num_samples = 4
        targets = torch.zeros(num_samples, MOVE_LABELS_NUM)
        targets[:, :10] = 1.0  # 全サンプルで最初の10手が合法

        board_tensor = torch.randint(0, 32, (num_samples, 9, 9))
        hand_tensor = torch.randn(
            num_samples, PIECES_IN_HAND_VECTOR_SIZE
        )

        dataset = TensorDataset(
            board_tensor, hand_tensor, targets
        )

        def collate_fn(
            batch: list[tuple[torch.Tensor, ...]],
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor], torch.Tensor
        ]:
            boards = torch.stack([b[0] for b in batch])
            hands = torch.stack([b[1] for b in batch])
            tgts = torch.stack([b[2] for b in batch])
            return (boards, hands), tgts

        dataloader = DataLoader(
            dataset,
            batch_size=num_samples,
            collate_fn=collate_fn,
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=1,
            accuracy_threshold=0.0,  # 閾値チェックを無効化
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        # F1スコアは0〜1の範囲
        assert 0.0 <= result.achieved_accuracy <= 1.0
        # ランダム重みの初期状態ではF1は低いはず（全部0予測に近い）
        # 重要: accuracyなら~98%になるが，F1なら低くなることを検証
        assert result.achieved_accuracy < 0.9, (
            f"F1 should be low with random weights, got {result.achieved_accuracy:.4f}. "
            "If this is high, the metric may still be element-wise accuracy."
        )

    def test_stage2_f1_known_values(self) -> None:
        """F1の計算ロジックを直接テストする（ネットワーク不使用）．"""
        # TP=5, FP=2, FN=3 のケース
        # Precision = 5/(5+2) = 5/7 ≈ 0.714
        # Recall = 5/(5+3) = 5/8 = 0.625
        # F1 = 2 * 0.714 * 0.625 / (0.714 + 0.625) ≈ 0.667
        predictions = torch.tensor(
            [
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ]
            ]
        )
        targets = torch.tensor(
            [
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ]
        )

        pred_bool = predictions
        tgt_bool = targets
        tp = (pred_bool & tgt_bool).float().sum(dim=1)
        fp = (pred_bool & ~tgt_bool).float().sum(dim=1)
        fn = (~pred_bool & tgt_bool).float().sum(dim=1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
        )

        expected_precision = 5.0 / 7.0
        expected_recall = 5.0 / 8.0
        expected_f1 = (
            2
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
        )

        assert abs(f1.item() - expected_f1) < 1e-5, (
            f"F1 mismatch: got {f1.item():.6f}, expected {expected_f1:.6f}"
        )

    def test_stage1_still_uses_accuracy(self) -> None:
        """Stage1ではF1ではなくelement-wise accuracyが使われることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )

        # ターゲットを全て0にすることで，ランダムweightsでもaccuracyが高くなる
        # （element-wise accuracyなら0予測で~50%は正解する）
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.0,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        # Stage1はelement-wise accuracy: ターゲットが全0なので高い精度になるはず
        assert result.achieved_accuracy > 0.3, (
            f"Stage1 accuracy unexpectedly low: {result.achieved_accuracy:.4f}"
        )


class TestLRSqrtScaling:
    """sqrt LRスケーリングのテスト．"""

    def test_no_scaling_when_batch_sizes_equal(self) -> None:
        """base_batch_size == actual_batch_sizeの場合，LRは変更されない．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            base_batch_size=256,
            actual_batch_size=256,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        actual_lr = loop.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(1e-3)

    def test_sqrt_scaling_applied(self) -> None:
        """actual_batch_size > base_batch_sizeの場合，sqrt scalingが適用される．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            base_batch_size=256,
            actual_batch_size=4096,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        expected_lr = 1e-3 * math.sqrt(
            4096 / 256
        )  # = 1e-3 * 4.0
        actual_lr = loop.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_lr)

    def test_no_scaling_when_smaller_batch(self) -> None:
        """actual_batch_size < base_batch_sizeの場合，LRは変更されない．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            base_batch_size=256,
            actual_batch_size=128,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        actual_lr = loop.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(1e-3)


class TestTensorAccumulation:
    """GPU tensor累積と.item()累積の結果等価性テスト．"""

    def test_loss_accumulation_equivalence(self) -> None:
        """テンソル累積が.item()累積と同じ結果を返す．"""
        losses = [
            torch.tensor(x) for x in [0.5, 0.3, 0.7, 0.2, 0.4]
        ]

        # .item() pattern
        total_item = 0.0
        for loss in losses:
            total_item += loss.item()

        # tensor pattern
        total_tensor = torch.tensor(0.0, dtype=torch.float64)
        for loss in losses:
            total_tensor += loss.detach()

        assert total_tensor.item() == pytest.approx(total_item)

    def test_accuracy_accumulation_equivalence(self) -> None:
        """精度メトリクスのテンソル累積が等価．"""
        corrects = [
            torch.tensor(x)
            for x in [10.0, 15.0, 12.0, 8.0, 20.0]
        ]

        total_item = 0.0
        for c in corrects:
            total_item += c.item()

        total_tensor = torch.tensor(0.0, dtype=torch.float64)
        for c in corrects:
            total_tensor += c

        assert total_tensor.item() == pytest.approx(total_item)

    def test_f1_accumulation_equivalence(self) -> None:
        """F1スコアのテンソル累積が等価．"""
        f1_sums = [
            torch.tensor(x)
            for x in [0.8, 0.9, 0.75, 0.85, 0.95]
        ]

        total_item = 0.0
        for f1 in f1_sums:
            total_item += f1.item()

        total_tensor = torch.tensor(0.0, dtype=torch.float64)
        for f1 in f1_sums:
            total_tensor += f1

        assert total_tensor.item() == pytest.approx(total_item)

    def test_fp64_precision_for_large_accumulation(
        self,
    ) -> None:
        """FP64 accumulatorがFP32の整数表現限界(2^24)を超えても精度を維持する．"""
        n = 20_000_000  # 20M — exceeds FP32 exact integer limit (2^24 ≈ 16.7M)
        value_per_step = 1.0

        # FP64 accumulation
        acc_fp64 = torch.tensor(0.0, dtype=torch.float64)
        # Simulate by adding in chunks to avoid slow loop
        chunk = torch.tensor(
            value_per_step * 1_000_000, dtype=torch.float64
        )
        for _ in range(n // 1_000_000):
            acc_fp64 += chunk

        assert acc_fp64.item() == pytest.approx(float(n))

        # When accumulating individual 1.0s past 2^24, FP32 rounds
        # 16777217 → 16777216. We demonstrate this with a targeted check.
        limit = torch.tensor(
            2**24, dtype=torch.float32
        )  # 16777216
        one = torch.tensor(1.0, dtype=torch.float32)
        result_fp32 = (
            limit + one
        )  # Should be 16777217 but FP32 can't represent it
        assert float(result_fp32) == float(
            limit
        )  # FP32 precision loss!

        # FP64 handles it correctly
        limit_fp64 = torch.tensor(2**24, dtype=torch.float64)
        one_fp64 = torch.tensor(1.0, dtype=torch.float64)
        result_fp64 = limit_fp64 + one_fp64
        assert float(result_fp64) == 2**24 + 1  # FP64 is exact


class TestStage12Compilation:
    """Stage 1/2のtorch.compile適用テスト．"""

    def test_compilation_separate_backbone_and_head(
        self,
    ) -> None:
        """backboneとheadが個別にコンパイルされエラーにならない．"""
        from maou.app.learning.compilation import compile_module

        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )

        # compile_module should not raise (falls back to eager on CPU if needed)
        compiled_backbone = compile_module(backbone)
        compiled_head = compile_module(head)

        assert compiled_backbone is not None
        assert compiled_head is not None
        assert isinstance(compiled_backbone, torch.nn.Module)
        assert isinstance(compiled_head, torch.nn.Module)

    def test_compilation_disabled_by_default(self) -> None:
        """compilation=Falseの場合，モデルはそのまま使用される．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            compilation=False,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        # Model should be the original HeadlessNetwork, not a compiled wrapper
        assert isinstance(loop.model, HeadlessNetwork)


class TestSchedulerIntegration:
    """SingleStageTrainingLoopのスケジューラ統合テスト．"""

    def test_scheduler_created_when_configured(self) -> None:
        """lr_scheduler_nameが設定されている場合，スケジューラが生成される．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=10,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            lr_scheduler_name="warmup_cosine_decay",
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        assert loop.lr_scheduler is not None

    def test_no_scheduler_when_none(self) -> None:
        """lr_scheduler_nameがNoneの場合，スケジューラは生成されない．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=10,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            lr_scheduler_name=None,
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        assert loop.lr_scheduler is None

    def test_scheduler_step_called_per_epoch(self) -> None:
        """各epoch終了後にLRが変化する（スケジューラが動作している）．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=3,
            accuracy_threshold=1.0,  # Unreachable threshold, ensure all epochs run
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            learning_rate=1e-3,
            lr_scheduler_name="warmup_cosine_decay",
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        initial_lr = loop.optimizer.param_groups[0]["lr"]

        # Run training — scheduler should step after each epoch
        loop.run()

        final_lr = loop.optimizer.param_groups[0]["lr"]

        # After 3 epochs of warmup+cosine decay, LR should have changed
        # (warmup_epochs = max(1, ceil(3*0.1)) = 1, so epoch 0 is warmup)
        assert initial_lr != final_lr
