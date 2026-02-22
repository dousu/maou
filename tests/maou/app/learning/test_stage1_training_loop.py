"""Tests for Stage 1 training loop components."""

from __future__ import annotations

import torch
import torch.nn as nn

from maou.app.learning.callbacks import (
    Stage1AccuracyCallback,
    TrainingContext,
)
from maou.app.learning.multi_stage_training import (
    Stage1DatasetAdapter,
    Stage1ModelAdapter,
)
from maou.app.learning.network import (
    HeadlessNetwork,
    ReachableSquaresHead,
)
from maou.app.learning.streaming_dataset import (
    Stage1StreamingAdapter,
)
from maou.app.learning.training_loop import Stage1TrainingLoop
from maou.domain.loss.loss_fn import ReachableSquaresLoss


class TestStage1ModelAdapterOutputFormat:
    """Stage1ModelAdapter の forward 出力形式を検証するテスト．"""

    def test_stage1_model_adapter_output_format(self) -> None:
        """forward() が (policy_logits, dummy_value) を正しい shape で返す．"""
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
        adapter = Stage1ModelAdapter(backbone, head)

        batch_size = 4
        board = torch.randint(0, 32, (batch_size, 9, 9))
        hand = torch.randn(batch_size, 14)

        policy_logits, dummy_value = adapter((board, hand))

        assert policy_logits.shape == (batch_size, 81)
        assert dummy_value.shape == (batch_size, 1)
        torch.testing.assert_close(
            dummy_value,
            torch.zeros(batch_size, 1),
        )


class TestStage1TrainingLoopComputePolicyLoss:
    """Stage1TrainingLoop._compute_policy_loss の検証テスト．"""

    def test_stage1_training_loop_compute_policy_loss(
        self,
    ) -> None:
        """_compute_policy_loss が log_softmax なしで ReachableSquaresLoss を呼ぶ．"""
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
        model = Stage1ModelAdapter(backbone, head)

        loss_fn = ReachableSquaresLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3
        )

        loop = Stage1TrainingLoop(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            loss_fn_policy=loss_fn,
            loss_fn_value=nn.MSELoss(),
            policy_loss_ratio=1.0,
            value_loss_ratio=0.0,
        )

        batch_size = 4
        logits = torch.randn(batch_size, 81)
        targets = torch.zeros(batch_size, 81)
        targets[:, :10] = 1.0

        context = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(
                torch.zeros(batch_size, 9, 9),
                torch.zeros(batch_size, 14),
            ),
            labels_policy=targets,
            labels_value=torch.zeros(batch_size, 1),
            legal_move_mask=None,
            outputs_policy=logits,
            loss=torch.tensor(0.0),
        )

        loss = loop._compute_policy_loss(context)

        # Loss should be a scalar tensor
        assert loss.dim() == 0
        assert loss.item() > 0.0

        # Verify: same result as calling ReachableSquaresLoss directly
        expected_loss = loss_fn(logits, targets)
        torch.testing.assert_close(loss, expected_loss)


class TestStage1AccuracyCallback:
    """Stage1AccuracyCallback の精度計算を検証するテスト．"""

    def test_stage1_accuracy_callback(self) -> None:
        """sigmoid(logits) > 0.5 の閾値で精度が正しく計算される．"""
        callback = Stage1AccuracyCallback()

        batch_size = 2
        # logits = [2.0, -2.0, 2.0] -> sigmoid -> [0.88, 0.12, 0.88]
        # predictions: [True, False, True]
        # targets: [1.0, 0.0, 1.0] -> [True, False, True]
        # -> all correct -> accuracy = 1.0
        logits = torch.tensor(
            [[2.0, -2.0, 2.0], [2.0, -2.0, 2.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        context = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(
                torch.zeros(batch_size, 9, 9),
                torch.zeros(batch_size, 14),
            ),
            labels_policy=targets,
            labels_value=torch.zeros(batch_size, 1),
            legal_move_mask=None,
            outputs_policy=logits,
            loss=torch.tensor(0.5),
        )

        callback.on_batch_end(context)

        accuracy = callback.get_epoch_accuracy()
        assert accuracy == 1.0

    def test_stage1_accuracy_callback_partial(self) -> None:
        """一部不正解の場合に精度が正しく計算される．"""
        callback = Stage1AccuracyCallback()

        # logits = [2.0, 2.0, -2.0] -> sigmoid -> [0.88, 0.88, 0.12]
        # predictions: [True, True, False]
        # targets: [1.0, 0.0, 1.0] -> [True, False, True]
        # correct: [True, False, False] -> 1/3
        logits = torch.tensor(
            [[2.0, 2.0, -2.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        context = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(torch.zeros(1, 9, 9), torch.zeros(1, 14)),
            labels_policy=targets,
            labels_value=torch.zeros(1, 1),
            legal_move_mask=None,
            outputs_policy=logits,
            loss=torch.tensor(0.5),
        )

        callback.on_batch_end(context)

        accuracy = callback.get_epoch_accuracy()
        assert abs(accuracy - 1.0 / 3.0) < 1e-6


class TestStage1DatasetAdapter:
    """Stage1DatasetAdapter のアダプタ変換を検証するテスト．"""

    def test_stage1_dataset_adapter(self) -> None:
        """__getitem__ が 3-tuple (inputs, (targets, dummy_value, None)) を返す．"""

        class FakeStage1Dataset:
            """2-tuple を返すモック Stage1Dataset．"""

            def __init__(self, size: int = 5) -> None:
                self._size = size

            def __len__(self) -> int:
                return self._size

            def __getitem__(
                self, idx: int
            ) -> tuple[
                tuple[torch.Tensor, torch.Tensor], torch.Tensor
            ]:
                board = torch.randint(0, 32, (9, 9))
                hand = torch.randn(14)
                targets = torch.zeros(81)
                targets[:10] = 1.0
                return (board, hand), targets

        fake_dataset = FakeStage1Dataset(size=5)
        adapter = Stage1DatasetAdapter(fake_dataset)

        assert len(adapter) == 5

        inputs, label_tuple = adapter[0]

        # inputs should be (board, hand)
        board, hand = inputs
        assert board.shape == (9, 9)
        assert hand.shape == (14,)

        # label_tuple should be (targets, dummy_value, None)
        targets, dummy_value, legal_mask = label_tuple
        assert targets.shape == (81,)
        assert dummy_value.shape == (1,)
        torch.testing.assert_close(
            dummy_value,
            torch.zeros(1, dtype=torch.float32),
        )
        assert legal_mask is None

    def test_stage1_dataset_adapter_len(self) -> None:
        """__len__ が元の dataset の長さと同じ．"""

        class FakeStage1Dataset:
            """テスト用のモック．"""

            def __init__(self, size: int) -> None:
                self._size = size

            def __len__(self) -> int:
                return self._size

            def __getitem__(
                self, idx: int
            ) -> tuple[
                tuple[torch.Tensor, torch.Tensor], torch.Tensor
            ]:
                return (
                    torch.zeros(9, 9),
                    torch.zeros(14),
                ), torch.zeros(81)

        for size in [0, 1, 10, 100]:
            fake_dataset = FakeStage1Dataset(size=size)
            adapter = Stage1DatasetAdapter(fake_dataset)
            assert len(adapter) == size


class TestStage1StreamingAdapter:
    """Stage1StreamingAdapter のアダプタ変換を検証するテスト．"""

    def test_stage1_streaming_adapter(self) -> None:
        """__iter__ が 3-tuple (inputs, (targets, dummy_value, None)) を yield する．"""
        batch_size = 4

        class FakeStreamingStage1Dataset:
            """2-tuple をイテレートで返すモック StreamingStage1Dataset．"""

            def __init__(self) -> None:
                self._epoch: int = 0

            def __len__(self) -> int:
                return 3

            def __iter__(self):  # type: ignore[no-untyped-def]
                for _ in range(3):
                    board = torch.randint(
                        0, 32, (batch_size, 9, 9)
                    )
                    hand = torch.randn(batch_size, 14)
                    targets = torch.zeros(batch_size, 81)
                    targets[:, :10] = 1.0
                    yield (board, hand), targets

            def set_epoch(self, epoch: int) -> None:
                self._epoch = epoch

        fake_dataset = FakeStreamingStage1Dataset()
        adapter = Stage1StreamingAdapter(fake_dataset)

        assert len(adapter) == 3

        batches = list(adapter)
        assert len(batches) == 3

        for inputs, label_tuple in batches:
            board, hand = inputs
            assert board.shape == (batch_size, 9, 9)
            assert hand.shape == (batch_size, 14)

            targets, dummy_value, legal_mask = label_tuple
            assert targets.shape == (batch_size, 81)
            assert dummy_value.shape == (batch_size, 1)
            torch.testing.assert_close(
                dummy_value,
                torch.zeros(batch_size, 1, dtype=torch.float32),
            )
            assert legal_mask is None

    def test_stage1_streaming_adapter_set_epoch(self) -> None:
        """set_epoch() が元の dataset に委譲される．"""

        class FakeStreamingStage1Dataset:
            """set_epoch を持つモック．"""

            def __init__(self) -> None:
                self.epoch: int = -1

            def __len__(self) -> int:
                return 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter([])

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

        fake_dataset = FakeStreamingStage1Dataset()
        adapter = Stage1StreamingAdapter(fake_dataset)

        adapter.set_epoch(42)
        assert fake_dataset.epoch == 42

        adapter.set_epoch(0)
        assert fake_dataset.epoch == 0


class TestStage1AccuracyCallbackReset:
    """Stage1AccuracyCallback の reset 動作を検証するテスト．"""

    def test_stage1_accuracy_callback_reset(self) -> None:
        """reset() 後に新しいデータのみが精度に反映される．"""
        callback = Stage1AccuracyCallback()

        # First batch: all correct
        logits_1 = torch.tensor(
            [[2.0, -2.0, 2.0]],
            dtype=torch.float32,
        )
        targets_1 = torch.tensor(
            [[1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        context_1 = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(torch.zeros(1, 9, 9), torch.zeros(1, 14)),
            labels_policy=targets_1,
            labels_value=torch.zeros(1, 1),
            legal_move_mask=None,
            outputs_policy=logits_1,
            loss=torch.tensor(0.5),
        )
        callback.on_batch_end(context_1)

        # Second batch: all correct
        logits_2 = torch.tensor(
            [[-2.0, 2.0, -2.0]],
            dtype=torch.float32,
        )
        targets_2 = torch.tensor(
            [[0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        context_2 = TrainingContext(
            batch_idx=1,
            epoch_idx=0,
            inputs=(torch.zeros(1, 9, 9), torch.zeros(1, 14)),
            labels_policy=targets_2,
            labels_value=torch.zeros(1, 1),
            legal_move_mask=None,
            outputs_policy=logits_2,
            loss=torch.tensor(0.3),
        )
        callback.on_batch_end(context_2)

        # Before reset: accuracy = 1.0 (all 6 elements correct)
        assert callback.get_epoch_accuracy() == 1.0

        # Reset
        callback.reset()

        # After reset: no data yet
        assert callback.get_epoch_accuracy() == 0.0

        # Third batch: 1 out of 3 correct
        # logits = [-2.0, -2.0, 2.0] -> sigmoid -> [0.12, 0.12, 0.88]
        # predictions: [False, False, True]
        # targets: [1.0, 0.0, 1.0] -> [True, False, True]
        # correct: [False, True, True] -> 2/3
        logits_3 = torch.tensor(
            [[-2.0, -2.0, 2.0]],
            dtype=torch.float32,
        )
        targets_3 = torch.tensor(
            [[1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        context_3 = TrainingContext(
            batch_idx=0,
            epoch_idx=1,
            inputs=(torch.zeros(1, 9, 9), torch.zeros(1, 14)),
            labels_policy=targets_3,
            labels_value=torch.zeros(1, 1),
            legal_move_mask=None,
            outputs_policy=logits_3,
            loss=torch.tensor(0.7),
        )
        callback.on_batch_end(context_3)

        # After reset + one batch: accuracy = 2/3 (only new data)
        accuracy = callback.get_epoch_accuracy()
        assert abs(accuracy - 2.0 / 3.0) < 1e-6


class TestGetPostfix:
    """get_postfix() メソッドのテスト."""

    def test_stage1_accuracy_callback_get_postfix(self) -> None:
        """Stage1AccuracyCallback.get_postfix() が正しい dict を返す."""
        from maou.app.learning.callbacks import (
            Stage1AccuracyCallback,
            TrainingContext,
        )

        callback = Stage1AccuracyCallback()
        batch_size = 4
        logits = torch.tensor([[2.0] * 81] * batch_size)
        targets = torch.ones(batch_size, 81)
        loss = torch.tensor(0.05)

        context = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(torch.zeros(batch_size, 9, 9), torch.zeros(batch_size, 14)),
            labels_policy=targets,
            labels_value=torch.zeros(batch_size, 1),
            legal_move_mask=None,
            outputs_policy=logits,
            loss=loss,
        )
        callback.on_batch_end(context)

        postfix = callback.get_postfix()
        assert postfix is not None
        assert "acc" in postfix
        assert "loss" in postfix
        assert postfix["acc"].endswith("%")

    def test_stage2_f1_callback_get_postfix(self) -> None:
        """Stage2F1Callback.get_postfix() が正しい dict を返す."""
        from maou.app.learning.callbacks import (
            Stage2F1Callback,
            TrainingContext,
        )

        callback = Stage2F1Callback()
        batch_size = 4
        num_classes = 1496
        logits = torch.randn(batch_size, num_classes)
        targets = torch.zeros(batch_size, num_classes)
        targets[:, :10] = 1.0
        loss = torch.tensor(0.1)

        context = TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=(torch.zeros(batch_size, 9, 9), torch.zeros(batch_size, 14)),
            labels_policy=targets,
            labels_value=torch.zeros(batch_size, 1),
            legal_move_mask=None,
            outputs_policy=logits,
            loss=loss,
        )
        callback.on_batch_end(context)

        postfix = callback.get_postfix()
        assert postfix is not None
        assert "f1" in postfix
        assert "loss" in postfix

    def test_get_postfix_returns_none_when_no_data(self) -> None:
        """データなし時に get_postfix() が None を返す."""
        from maou.app.learning.callbacks import (
            Stage1AccuracyCallback,
            Stage2F1Callback,
        )

        assert Stage1AccuracyCallback().get_postfix() is None
        assert Stage2F1Callback().get_postfix() is None

    def test_base_callback_get_postfix_returns_none(self) -> None:
        """BaseCallback.get_postfix() がデフォルトで None を返す."""
        from maou.app.learning.callbacks import BaseCallback

        assert BaseCallback().get_postfix() is None
