"""Tests for multi-stage training loop with HeadlessNetwork backbone."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from maou.app.learning.multi_stage_training import (
    MultiStageTrainingOrchestrator,
    Stage1DatasetAdapter,
    StageConfig,
    TrainingStage,
    TruncatedStageModel,
    pre_stage_collate_fn,
)
from maou.app.learning.network import (
    PIECES_IN_HAND_VECTOR_SIZE,
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)

if TYPE_CHECKING:
    from maou.app.learning.stage_component_factory import (
        StageComponents,
    )


class _PairedDataset(Dataset):
    """TensorDataset を ((board, hand), target) 形式で返すラッパー．

    Stage1DatasetAdapter でラップするために，
    per-sample で (inputs, target) のタプルを返す必要がある．
    """

    def __init__(
        self,
        board: torch.Tensor,
        hand: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._board = board
        self._hand = hand
        self._targets = targets

    def __len__(self) -> int:
        return self._board.size(0)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (
            self._board[idx],
            self._hand[idx],
        ), self._targets[idx]


def _make_dummy_dataloader(
    *,
    batch_size: int = 4,
    num_samples: int = 8,
    board_vocab_size: int = 32,
    board_size: tuple[int, int] = (9, 9),
    target_dim: int = 81,
    hand_dim: int = PIECES_IN_HAND_VECTOR_SIZE,
    stage: TrainingStage = TrainingStage.REACHABLE_SQUARES,
    wrap_stage1_adapter: bool = False,
) -> DataLoader:
    """ダミーの (board_tensor, hand_tensor), target データローダーを作成する．

    Stage 1 は ((board_tensor, hand_tensor), target) を返す．
    Stage 2 は TrainingLoop 互換形式 ((board_tensor, hand_tensor), (target, dummy_value, None)) を返す．

    wrap_stage1_adapter=True の場合，Stage 1 のデータセットを
    Stage1DatasetAdapter でラップし，TrainingLoop 互換の 3-tuple 形式
    ((board, hand), (target, dummy_value, None)) を返す．
    """
    board_tensor = torch.randint(
        0, board_vocab_size, (num_samples, *board_size)
    )
    hand_tensor = torch.randn(num_samples, hand_dim)
    targets = torch.zeros(num_samples, target_dim)

    if (
        wrap_stage1_adapter
        and stage == TrainingStage.REACHABLE_SQUARES
    ):
        paired_dataset: Dataset = _PairedDataset(
            board_tensor, hand_tensor, targets
        )
        adapted_dataset = Stage1DatasetAdapter(paired_dataset)

        return DataLoader(
            adapted_dataset,
            batch_size=batch_size,
            collate_fn=pre_stage_collate_fn,
        )

    if stage == TrainingStage.LEGAL_MOVES:
        from maou.app.learning.multi_stage_training import (
            Stage2DatasetAdapter,
        )

        paired_dataset = _PairedDataset(
            board_tensor, hand_tensor, targets
        )
        adapted_dataset = Stage2DatasetAdapter(paired_dataset)
        return DataLoader(
            adapted_dataset,
            batch_size=batch_size,
            collate_fn=pre_stage_collate_fn,
        )

    dataset = TensorDataset(board_tensor, hand_tensor, targets)

    def collate_fn(
        batch: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        boards = torch.stack([b[0] for b in batch])
        hands = torch.stack([b[1] for b in batch])
        tgts = torch.stack([b[2] for b in batch])
        return (boards, hands), tgts

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


class TestBackboneForwardFeatures:
    """HeadlessNetwork.forward_features() のテスト．"""

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
        return StageConfig(
            stage=stage,
            max_epochs=1,
            accuracy_threshold=threshold,
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
        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Stage 1 failed to meet accuracy threshold",
        ) as exc_info:
            orchestrator.run_all_stages(
                stage1_components=components,
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
        stage1_components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        stage2_config = self._make_stage_config(
            TrainingStage.LEGAL_MOVES,
            threshold=0.99,
        )
        stage2_components = _make_stage_components(
            stage=TrainingStage.LEGAL_MOVES,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Stage 2 failed to meet accuracy threshold",
        ) as exc_info:
            orchestrator.run_all_stages(
                stage1_components=stage1_components,
                stage1_config=stage1_config,
                stage2_components=stage2_components,
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


# ============================================================
# Production パステスト (run_stage1/2_with_training_loop)
# ============================================================


def _make_backbone(
    *,
    board_vocab_size: int = 32,
    hand_projection_dim: int = 0,
) -> HeadlessNetwork:
    """テスト用の小規模 HeadlessNetwork を生成する．"""
    return HeadlessNetwork(
        board_vocab_size=board_vocab_size,
        hand_projection_dim=hand_projection_dim,
        embedding_dim=64,
        architecture="resnet",
        out_channels=(16, 32, 64, 64),
    )


def _make_stage_config(
    *,
    stage: TrainingStage = TrainingStage.REACHABLE_SQUARES,
    max_epochs: int = 1,
    accuracy_threshold: float = 0.0,
) -> StageConfig:
    """テスト用のスリム StageConfig を生成する(3フィールド)．"""
    return StageConfig(
        stage=stage,
        max_epochs=max_epochs,
        accuracy_threshold=accuracy_threshold,
    )


def _make_stage_components(
    *,
    backbone: HeadlessNetwork | None = None,
    stage: TrainingStage = TrainingStage.REACHABLE_SQUARES,
    batch_size: int = 4,
    num_samples: int = 8,
    learning_rate: float = 1e-3,
    lr_scheduler_name: str | None = None,
    board_vocab_size: int = 32,
    hand_dim: int = PIECES_IN_HAND_VECTOR_SIZE,
    max_epochs: int = 1,
) -> "StageComponents":
    """テスト用の StageComponents を生成する．"""
    from maou.app.learning.multi_stage_training import (
        Stage1ModelAdapter,
        Stage2ModelAdapter,
    )
    from maou.app.learning.setup import SchedulerFactory
    from maou.app.learning.stage_component_factory import (
        StageComponents,
    )
    from maou.domain.move.label import MOVE_LABELS_NUM

    if backbone is None:
        backbone = _make_backbone(
            board_vocab_size=board_vocab_size
        )

    if stage == TrainingStage.REACHABLE_SQUARES:
        target_dim = 81
    else:
        target_dim = MOVE_LABELS_NUM

    dataloader = _make_dummy_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        board_vocab_size=board_vocab_size,
        target_dim=target_dim,
        hand_dim=hand_dim,
        stage=stage,
        wrap_stage1_adapter=(
            stage == TrainingStage.REACHABLE_SQUARES
        ),
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    if stage == TrainingStage.REACHABLE_SQUARES:
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model: torch.nn.Module = Stage1ModelAdapter(
            backbone, head
        )
    else:
        head = LegalMovesHead(
            input_dim=backbone.embedding_dim,
        )
        model = Stage2ModelAdapter(backbone, head)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )

    lr_scheduler = SchedulerFactory.create_scheduler(
        optimizer,
        lr_scheduler_name=lr_scheduler_name,
        max_epochs=max_epochs,
        steps_per_epoch=len(dataloader),
    )

    return StageComponents(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=None,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


class TestRunStage1WithTrainingLoop:
    """run_stage1_with_training_loop() の直接テスト．"""

    def test_stage1_e2e_completes(self) -> None:
        """Stage 1 の E2E 訓練が完走し，正しい StageResult を返す．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
        )
        device = torch.device("cpu")

        result, head = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert 0.0 <= result.achieved_accuracy <= 1.0
        assert isinstance(result.final_loss, float)
        assert result.final_loss > 0.0
        assert isinstance(head, ReachableSquaresHead)

    def test_stage1_with_hand_projection(self) -> None:
        """hand_projection_dim > 0 で E2E 訓練が完走する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        backbone = _make_backbone(hand_projection_dim=8)
        components = _make_stage_components(
            backbone=backbone,
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        device = torch.device("cpu")

        result, head = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(head, ReachableSquaresHead)

    def test_stage1_uses_accuracy_metric(self) -> None:
        """Stage 1 が element-wise accuracy メトリックを使用する．

        ターゲットが全ゼロのため，ランダム重みでも
        accuracy は一定以上の値(> 0.3)になるはず．
        """
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        device = torch.device("cpu")

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        # 全ゼロターゲットに対してランダム重みでも
        # accuracy > 0.3 が期待される
        assert result.achieved_accuracy > 0.3

    def test_stage1_threshold_early_stopping(self) -> None:
        """accuracy_threshold=0.0 で即座に threshold_met=True になる．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=5,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=5,
            accuracy_threshold=0.0,
        )
        device = torch.device("cpu")

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.threshold_met is True
        # threshold=0.0 なので 1 エポックで達成
        assert result.epochs_trained == 1

    def test_stage1_streaming_set_epoch(self) -> None:
        """streaming DataLoader で set_epoch が呼ばれることを確認する．"""
        from unittest.mock import MagicMock

        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=1.0,
        )
        device = torch.device("cpu")

        # DataLoader の dataset に set_epoch を追加
        mock_set_epoch = MagicMock()
        components.train_dataloader.dataset.set_epoch = (
            mock_set_epoch
        )

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.epochs_trained == 2
        assert mock_set_epoch.call_count == 2
        mock_set_epoch.assert_any_call(0)
        mock_set_epoch.assert_any_call(1)

    def test_stage1_non_streaming_no_error(self) -> None:
        """非 streaming DataLoader でエラーが発生しないことを確認する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        device = torch.device("cpu")

        # set_epoch がないデータセットでもエラーなし
        assert not hasattr(
            components.train_dataloader.dataset, "set_epoch"
        )

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.epochs_trained >= 1

    def test_stage1_with_scheduler(self) -> None:
        """lr_scheduler 指定時に訓練が完走する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            lr_scheduler_name="warmup_cosine_decay",
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=1.0,
        )
        device = torch.device("cpu")

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.epochs_trained == 2
        assert isinstance(result.final_loss, float)

    def test_stage1_without_scheduler(self) -> None:
        """lr_scheduler=None で訓練が完走する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
            lr_scheduler_name=None,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        device = torch.device("cpu")

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.epochs_trained >= 1

    def test_stage1_compilation_disabled(self) -> None:
        """compilation=False (components に含まれない) で訓練が正常に完走する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage1_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        config = _make_stage_config(
            stage=TrainingStage.REACHABLE_SQUARES,
        )
        device = torch.device("cpu")

        result, _ = run_stage1_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.epochs_trained >= 1


class TestRunStage2WithTrainingLoop:
    """run_stage2_with_training_loop() の直接テスト．"""

    def test_stage2_e2e_completes(self) -> None:
        """Stage 2 の E2E 訓練が完走し，正しい StageResult を返す．"""
        from maou.app.learning.multi_stage_training import (
            run_stage2_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=2,
        )
        config = _make_stage_config(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=2,
        )
        device = torch.device("cpu")

        result, head = run_stage2_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.stage == TrainingStage.LEGAL_MOVES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert 0.0 <= result.achieved_accuracy <= 1.0
        assert isinstance(result.final_loss, float)
        assert result.final_loss > 0.0
        assert isinstance(head, LegalMovesHead)

    def test_stage2_with_hand_projection(self) -> None:
        """hand projection 付きで Stage 2 E2E が完走する．"""
        from maou.app.learning.multi_stage_training import (
            run_stage2_with_training_loop,
        )

        backbone = _make_backbone(hand_projection_dim=8)
        components = _make_stage_components(
            backbone=backbone,
            stage=TrainingStage.LEGAL_MOVES,
        )
        config = _make_stage_config(
            stage=TrainingStage.LEGAL_MOVES,
        )
        device = torch.device("cpu")

        result, head = run_stage2_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert result.stage == TrainingStage.LEGAL_MOVES
        assert result.epochs_trained >= 1
        assert isinstance(head, LegalMovesHead)

    def test_stage2_uses_f1_metric(self) -> None:
        """Stage 2 が F1 メトリックを使用し，ランダム重みで低い値を返す．"""
        from maou.app.learning.multi_stage_training import (
            run_stage2_with_training_loop,
        )

        components = _make_stage_components(
            stage=TrainingStage.LEGAL_MOVES,
        )
        config = _make_stage_config(
            stage=TrainingStage.LEGAL_MOVES,
        )
        device = torch.device("cpu")

        result, _ = run_stage2_with_training_loop(
            components=components,
            config=config,
            device=device,
        )

        assert 0.0 <= result.achieved_accuracy <= 1.0
        # ランダム重み + 全ゼロターゲットで F1 は低い
        assert result.achieved_accuracy < 0.9


class TestEffectiveLRComputation:
    """LossOptimizerFactory.compute_effective_lr() のユニットテスト．"""

    def test_no_scaling_when_equal(self) -> None:
        """base == actual の場合，LR は変化しない．"""
        from maou.app.learning.setup import (
            LossOptimizerFactory,
        )

        result = LossOptimizerFactory.compute_effective_lr(
            learning_rate=1e-3,
            actual_batch_size=256,
            base_batch_size=256,
        )
        assert result == pytest.approx(1e-3)

    def test_sqrt_scaling_applied(self) -> None:
        """actual > base の場合，sqrt スケーリングが適用される．"""
        from maou.app.learning.setup import (
            LossOptimizerFactory,
        )

        result = LossOptimizerFactory.compute_effective_lr(
            learning_rate=1e-3,
            actual_batch_size=4096,
            base_batch_size=256,
        )
        expected = 1e-3 * math.sqrt(4096 / 256)
        assert result == pytest.approx(expected)

    def test_no_scaling_when_smaller(self) -> None:
        """actual < base の場合，LR は変化しない．"""
        from maou.app.learning.setup import (
            LossOptimizerFactory,
        )

        result = LossOptimizerFactory.compute_effective_lr(
            learning_rate=1e-3,
            actual_batch_size=128,
            base_batch_size=256,
        )
        assert result == pytest.approx(1e-3)


class TestTruncatedStageModelFreeze:
    """TruncatedStageModel のバックボーン凍結テスト．"""

    def test_trailing_groups_frozen(self) -> None:
        """trainable_layers 個の末尾グループが requires_grad=False になる．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        trainable_layers = 2

        _model = TruncatedStageModel(
            backbone, head, trainable_layers=trainable_layers
        )

        groups = backbone.backbone.get_freezable_groups()
        total = len(groups)
        n_use = total - trainable_layers

        # 使用グループ (前半) は requires_grad=True
        for i, group in enumerate(groups[:n_use]):
            for name, param in group.named_parameters():
                assert param.requires_grad, (
                    f"Group {i} param {name} should be trainable"
                )

        # 除外グループ (後半) は requires_grad=False
        for i, group in enumerate(groups[n_use:]):
            for name, param in group.named_parameters():
                assert not param.requires_grad, (
                    f"Frozen group {n_use + i} param {name} "
                    "should have requires_grad=False"
                )

    def test_head_remains_trainable(self) -> None:
        """TruncatedStageModel のヘッドは凍結されない．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )

        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        for name, param in model.head.named_parameters():
            assert param.requires_grad, (
                f"Head param {name} should be trainable"
            )

    def test_trainable_layers_propagated(self) -> None:
        """trainable_layers=1 でも正しく partial backbone が構成される．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )

        groups = backbone.backbone.get_freezable_groups()
        total = len(groups)

        model = TruncatedStageModel(
            backbone, head, trainable_layers=1
        )

        # partial_backbone は (total - 1) 個のグループから成る
        assert len(model.partial_backbone) == total - 1
