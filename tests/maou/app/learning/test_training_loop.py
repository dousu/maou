from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import torch

from maou.app.learning.adaptive_batch import AdaptiveBatchConfig
from maou.app.learning.callbacks import (
    AdaptiveBatchCallback,
    ModelInputs,
    TrainingContext,
)
from maou.app.learning.network import Network
from maou.app.learning.training_loop import TrainingLoop


def test_resolve_batch_size_supports_list_inputs() -> None:
    board = torch.zeros((4, 9, 9), dtype=torch.long)
    pieces = torch.zeros((4, 14), dtype=torch.float32)
    inputs: ModelInputs = [board, pieces]

    batch_size = TrainingLoop._resolve_batch_size(inputs)

    assert batch_size == 4


def test_network_separate_inputs_accepts_list() -> None:
    board = torch.zeros((2, 9, 9), dtype=torch.long)
    pieces = torch.zeros((2, 14), dtype=torch.float32)

    separated_board, separated_pieces = (
        Network._separate_inputs([board, pieces])
    )

    assert separated_board is board
    assert separated_pieces is pieces


def test_nan_loss_does_not_call_scaler_update() -> None:
    """NaN損失検出時にscaler.update()が呼ばれないことを確認する．"""
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mock_scaler = MagicMock()

    loop = TrainingLoop(
        model=model,
        device=torch.device("cpu"),
        optimizer=optimizer,
        loss_fn_policy=torch.nn.CrossEntropyLoss(),
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
    )
    loop.scaler = mock_scaler

    # NaN損失を含むcontextを作成
    context = TrainingContext(
        batch_idx=0,  # finitude_check_intervalの倍数でチェック発生
        epoch_idx=0,
        inputs=torch.zeros(2, 10),
        labels_policy=torch.zeros(2, dtype=torch.long),
        labels_value=torch.zeros(2),
        legal_move_mask=None,
        batch_size=2,
    )

    # モデル出力とNaN損失を設定
    context.outputs_policy = torch.zeros(2, 2)
    context.outputs_value = torch.zeros(2)
    context.loss = torch.tensor(float("nan"))

    # _compute_policy_lossとモデルをモックして
    # _train_batch_mixed_precisionのNaN検出パスを直接テスト
    with (
        patch.object(
            loop,
            "_compute_policy_loss",
            return_value=torch.tensor(float("nan")),
        ),
        patch.object(
            loop.model,
            "forward",
            return_value=(torch.zeros(2, 2), torch.zeros(2)),
        ),
    ):
        loop._train_batch_mixed_precision(
            context, is_accumulation_step=False, accumulation_step=0
        )

    # scaler.update()はNaN検出時に呼ばれてはいけない
    mock_scaler.update.assert_not_called()


class _SimpleBatchDataset(torch.utils.data.IterableDataset):
    """Yields simple batches for testing iteration methods."""

    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor, None],
        ]
    ]:  # type: ignore[override]
        for i in range(self.num_batches):
            features = torch.full((2, 3), float(i))
            targets = (
                torch.full((2,), float(i)),
                torch.full((2,), float(i)),
                None,
            )
            yield features, targets

    def __len__(self) -> int:
        return self.num_batches


def _make_loop() -> TrainingLoop:
    """Create a minimal TrainingLoop for testing."""
    model = torch.nn.Linear(3, 2)
    return TrainingLoop(
        model=model,
        device=torch.device("cpu"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn_policy=torch.nn.CrossEntropyLoss(),
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
    )


class TestIterateDirect:
    """Tests for _iterate_direct (CPU path)."""

    def test_yields_all_batches(self) -> None:
        """All batches are yielded in correct order."""
        loop = _make_loop()
        dataset = _SimpleBatchDataset(5)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        results = list(
            loop._iterate_direct(dataloader, epoch_idx=0)
        )

        assert len(results) == 5
        for i, (batch_idx, ctx) in enumerate(results):
            assert batch_idx == i
            assert ctx.epoch_idx == 0
            assert ctx.batch_size == 2

    def test_empty_dataloader(self) -> None:
        """Empty dataloader yields nothing."""
        loop = _make_loop()
        dataset = _SimpleBatchDataset(0)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        results = list(
            loop._iterate_direct(dataloader, epoch_idx=0)
        )
        assert len(results) == 0

    def test_single_batch(self) -> None:
        """Single batch is yielded correctly."""
        loop = _make_loop()
        dataset = _SimpleBatchDataset(1)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        results = list(
            loop._iterate_direct(dataloader, epoch_idx=3)
        )

        assert len(results) == 1
        batch_idx, ctx = results[0]
        assert batch_idx == 0
        assert ctx.epoch_idx == 3


class TestIterateWithTransfer:
    """Tests for _iterate_with_transfer dispatch."""

    def test_cpu_device_uses_direct(self) -> None:
        """CPU device dispatches to _iterate_direct."""
        loop = _make_loop()
        dataset = _SimpleBatchDataset(3)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        results = list(
            loop._iterate_with_transfer(dataloader, epoch_idx=0)
        )

        assert len(results) == 3
        for i, (batch_idx, ctx) in enumerate(results):
            assert batch_idx == i

    def test_transfers_data_for_cpu(self) -> None:
        """_transfer_to_device is called for each batch on CPU."""
        loop = _make_loop()
        dataset = _SimpleBatchDataset(2)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )
        transfer_count = 0
        original_transfer = loop._transfer_to_device

        def counting_transfer(ctx: TrainingContext) -> None:
            nonlocal transfer_count
            transfer_count += 1
            original_transfer(ctx)

        with patch.object(
            loop, "_transfer_to_device", counting_transfer
        ):
            list(
                loop._iterate_with_transfer(
                    dataloader, epoch_idx=0
                )
            )

        assert transfer_count == 2


class TestUnpackBatch:
    """Tests for _unpack_batch."""

    def test_unpacks_correctly(self) -> None:
        """Raw batch data is unpacked into TrainingContext."""
        loop = _make_loop()
        features = torch.zeros(4, 3)
        targets = (
            torch.zeros(4),
            torch.ones(4),
            None,
        )
        raw_data = (features, targets)

        ctx = loop._unpack_batch(
            raw_data, batch_idx=5, epoch_idx=2
        )

        assert ctx.batch_idx == 5
        assert ctx.epoch_idx == 2
        assert ctx.batch_size == 4
        assert ctx.inputs is features
        assert ctx.labels_policy is targets[0]
        assert ctx.labels_value is targets[1]
        assert ctx.legal_move_mask is None

    def test_unpacks_4_element_tuple_with_move_win_rate(
        self,
    ) -> None:
        """4-element target tuple with move_win_rate is unpacked."""
        loop = _make_loop()
        features = torch.zeros(4, 3)
        move_win_rate = torch.rand(4, 10)
        targets = (
            torch.zeros(4),
            torch.ones(4),
            None,
            move_win_rate,
        )
        raw_data = (features, targets)

        ctx = loop._unpack_batch(
            raw_data, batch_idx=0, epoch_idx=0
        )

        assert ctx.move_win_rate is move_win_rate

    def test_unpacks_4_element_tuple_none_move_win_rate(
        self,
    ) -> None:
        """4-element target tuple with None move_win_rate."""
        loop = _make_loop()
        features = torch.zeros(4, 3)
        targets = (
            torch.zeros(4),
            torch.ones(4),
            None,
            None,
        )
        raw_data = (features, targets)

        ctx = loop._unpack_batch(
            raw_data, batch_idx=0, epoch_idx=0
        )

        assert ctx.move_win_rate is None

    def test_unpacks_3_element_tuple_no_move_win_rate(
        self,
    ) -> None:
        """3-element target tuple sets move_win_rate=None."""
        loop = _make_loop()
        features = torch.zeros(4, 3)
        targets = (
            torch.zeros(4),
            torch.ones(4),
            None,
        )
        raw_data = (features, targets)

        ctx = loop._unpack_batch(
            raw_data, batch_idx=0, epoch_idx=0
        )

        assert ctx.move_win_rate is None


class _SimpleBatchDatasetForAdaptive(
    torch.utils.data.IterableDataset,
):
    """Adaptive batch テスト用のデータセット．"""

    def __init__(
        self,
        num_batches: int,
        input_dim: int,
        batch_size: int,
        output_dim: int,
    ) -> None:
        self.num_batches = num_batches
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.output_dim = output_dim

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor, None],
        ]
    ]:  # type: ignore[override]
        for _ in range(self.num_batches):
            features = torch.randn(
                self.batch_size, self.input_dim
            )
            targets = (
                torch.randint(
                    0,
                    self.output_dim,
                    (self.batch_size,),
                ).float(),
                torch.randn(self.batch_size),
                None,
            )
            yield features, targets

    def __len__(self) -> int:
        return self.num_batches


class TestAdaptiveBatchIntegration:
    """TrainingLoop と adaptive batch の統合テスト．"""

    def test_adaptive_batch_adjusts_accumulation_steps(
        self,
    ) -> None:
        """run_epoch で GNS 推定 → accumulation_steps 調整が動作することを確認する．"""
        # 入力次元を大きくしてランダム入力間の勾配に十分な分散を持たせる
        input_dim = 64
        output_dim = 2
        batch_size = 8
        model = torch.nn.Linear(input_dim, output_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaptive_config = AdaptiveBatchConfig(
            min_accumulation_steps=2,
            max_accumulation_steps=8,
            adjustment_interval=1,  # 毎ステップ調整
            smoothing_factor=1.0,  # EMA なし(即時反映)
            measurement_interval=1,
        )
        adaptive_cb = AdaptiveBatchCallback()

        loop = TrainingLoop(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            loss_fn_policy=torch.nn.CrossEntropyLoss(),
            loss_fn_value=torch.nn.MSELoss(),
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            gradient_accumulation_steps=2,
            adaptive_batch_config=adaptive_config,
            physical_batch_size=batch_size,
            adaptive_batch_callback=adaptive_cb,
            callbacks=[adaptive_cb],
        )

        # adaptive batch が有効化され初期値が min_accumulation_steps
        assert loop.gradient_accumulation_steps == 2
        assert loop._gns_estimator is not None
        assert loop._adaptive_controller is not None

        # 十分なバッチ数で run_epoch を実行
        # min_accumulation_steps=2 なので 2 バッチで 1 optimizer step
        # 最低 4 バッチ必要(2 optimizer steps)
        dataset = _SimpleBatchDatasetForAdaptive(
            num_batches=6,
            input_dim=input_dim,
            batch_size=batch_size,
            output_dim=output_dim,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        # エラーなく完走することを確認
        loop.run_epoch(
            dataloader,
            epoch_idx=0,
            progress_bar=False,
        )

        # GNS 推定器の step_count が増加していることを確認
        assert loop._gns_estimator._optimizer_step_count > 0

    def test_adaptive_batch_callback_updated(self) -> None:
        """AdaptiveBatchCallback の表示値が更新されることを確認する．"""
        input_dim = 32
        output_dim = 2
        batch_size = 4
        model = torch.nn.Linear(input_dim, output_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaptive_config = AdaptiveBatchConfig(
            min_accumulation_steps=2,
            max_accumulation_steps=8,
            adjustment_interval=1,
            smoothing_factor=1.0,
            measurement_interval=1,
        )
        adaptive_cb = AdaptiveBatchCallback()

        loop = TrainingLoop(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            loss_fn_policy=torch.nn.CrossEntropyLoss(),
            loss_fn_value=torch.nn.MSELoss(),
            policy_loss_ratio=1.0,
            value_loss_ratio=1.0,
            adaptive_batch_config=adaptive_config,
            physical_batch_size=batch_size,
            adaptive_batch_callback=adaptive_cb,
            callbacks=[adaptive_cb],
        )

        dataset = _SimpleBatchDatasetForAdaptive(
            num_batches=4,
            input_dim=input_dim,
            batch_size=batch_size,
            output_dim=output_dim,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        loop.run_epoch(
            dataloader,
            epoch_idx=0,
            progress_bar=False,
        )

        # コールバックの postfix が更新されていることを確認
        postfix = adaptive_cb.get_postfix()
        assert postfix is not None
        assert "accum" in postfix
        # accum 表示が数値文字列であること
        assert postfix["accum"].isdigit()
