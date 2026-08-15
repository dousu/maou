"""GradientNoiseScaleEstimator のテスト．"""

import math

import pytest
import torch
from torch import nn

from maou.app.learning.gradient_noise_scale import (
    GNSEstimate,
    GradientNoiseScaleEstimator,
)


def _make_simple_model() -> nn.Module:
    """テスト用の簡易モデルを作成する．"""
    return nn.Linear(10, 2, bias=False)


class TestGradientNoiseScaleEstimator:
    """GradientNoiseScaleEstimator のテスト．"""

    def test_basic_gns_estimation(self) -> None:
        """2 micro-batch で GNS が推定できることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # Simulate 2 micro-batches with gradient accumulation
        # Micro-batch 0
        model.zero_grad()
        x0 = torch.randn(32, 10)
        y0 = model(x0)
        loss0 = y0.sum() / 2  # Normalize by accum steps
        loss0.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        # Micro-batch 1
        x1 = torch.randn(32, 10)
        y1 = model(x1)
        loss1 = y1.sum() / 2
        loss1.backward()
        estimator.on_backward_end(model, accumulation_step=1)

        # Compute GNS
        result = estimator.compute(model)

        assert result is not None
        assert isinstance(result, GNSEstimate)
        assert result.b_noise > 0
        assert result.physical_batch_size == 32
        assert result.micro_batch_count == 2

    def test_single_step_returns_none(self) -> None:
        """micro-batch が 1 つしかない場合は None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        model.zero_grad()
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        result = estimator.compute(model)
        assert result is None

    def test_measurement_interval(self) -> None:
        """measurement_interval=2 で交互にスキップされることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=2,
        )

        def _run_cycle() -> GNSEstimate | None:
            model.zero_grad()
            for step in range(2):
                x = torch.randn(32, 10)
                y = model(x)
                loss = y.sum() / 2
                loss.backward()
                estimator.on_backward_end(
                    model, accumulation_step=step
                )
            return estimator.compute(model)

        # Cycle 1: step_count=0, 0%2==0 → measured
        result1 = _run_cycle()
        assert result1 is not None

        # Cycle 2: step_count=1, 1%2!=0 → skipped
        result2 = _run_cycle()
        assert result2 is None

        # Cycle 3: step_count=2, 2%2==0 → measured
        result3 = _run_cycle()
        assert result3 is not None

    def test_reset_between_cycles(self) -> None:
        """compute 後に内部状態がリセットされることを確認する．

        検証したいのは連続する 2 サイクルが独立に計算できることなので，
        計測を毎サイクル走らせる必要がある．``measurement_interval``
        の既定は 5 (host-device 同期を減らすため) であり，既定に任せると
        2 サイクル目が計測をスキップして本題が検証できない．
        """
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=1,
        )

        # Run first cycle
        model.zero_grad()
        for step in range(3):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum() / 3
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )
        result1 = estimator.compute(model)
        assert result1 is not None

        # Run second cycle - should work independently
        model.zero_grad()
        for step in range(3):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum() / 3
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )
        result2 = estimator.compute(model)
        assert result2 is not None

    def test_no_grad_returns_none(self) -> None:
        """勾配がない場合に None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # No backward call - no gradients
        result = estimator.compute(model)
        assert result is None

    def test_identical_gradients_returns_none(self) -> None:
        """全 micro-batch の勾配が同一(ratio <= 1.0)のとき None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # 同一入力で同一勾配を生成
        x = torch.randn(32, 10)

        model.zero_grad()
        y0 = model(x)
        loss0 = y0.sum() / 2
        loss0.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        y1 = model(x)
        loss1 = y1.sum() / 2
        loss1.backward()
        estimator.on_backward_end(model, accumulation_step=1)

        result = estimator.compute(model)
        # ratio = K * S / G ≈ 1.0 (同一勾配 → ノイズなし)
        # ratio <= 1.0 のため None が返る
        assert result is None

    def test_prev_grads_none_at_nonzero_step(self) -> None:
        """accumulation_step > 0 で _prev_grads が None の場合にスキップされることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # step=0 をスキップして step=1 から呼ぶ
        # (非有限損失で step=0 の backward がスキップされた場合に相当)
        model.zero_grad()
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum() / 2
        loss.backward()
        estimator.on_backward_end(model, accumulation_step=1)

        # _prev_grads が None のため on_backward_end は何も蓄積しない
        # compute は micro_batch_count < 2 で None を返す
        result = estimator.compute(model)
        assert result is None

    def test_param_count_mismatch_resets(self) -> None:
        """on_backward_end 中にモデルパラメータ数が変わった場合にリセットされることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # step=0: 正常に勾配を記録
        model.zero_grad()
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum() / 2
        loss.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        # step=1: パラメータ数が異なるモデルで呼ぶ
        model2 = nn.Sequential(
            nn.Linear(10, 5, bias=False),
            nn.Linear(5, 2, bias=False),
        )
        model2.zero_grad()
        x2 = torch.randn(32, 10)
        y2 = model2(x2)
        loss2 = y2.sum() / 2
        loss2.backward()
        estimator.on_backward_end(model2, accumulation_step=1)

        # パラメータ数不一致でリセットされ，compute は None を返す
        result = estimator.compute(model2)
        assert result is None

    def test_inf_mean_grad_norm_returns_none(self) -> None:
        """mean_grad_norm_sq が inf の場合に None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # 2 micro-batch で勾配を蓄積
        model.zero_grad()
        for step in range(2):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum() / 2
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )

        # param.grad を inf に設定して mean_grad_norm_sq = inf を引き起こす
        for param in model.parameters():
            if param.grad is not None:
                param.grad.fill_(float("inf"))

        result = estimator.compute(model)
        assert result is None


class _PythonFloatReferenceEstimator:
    """device スカラー累算を入れる前の ``.item()`` 逐次加算を写した参照実装．

    audits backlog Deferred 7 (`2026-08-08 app/learning`) の修正は
    「パラメータ毎の host-device 同期を無くしても計測値は 1 bit も
    変わらない」ことを前提にしている．そのクラス分類 (P4，挙動不変)
    を**主張ではなく結果**にするため，修正前の算術をそのまま写した
    このクラスを基準として突き合わせる．

    ここを本体と共通化してはならない — 共通化すると「同じコードは
    同じ答えを出す」ことしか言えなくなり，テストが空虚になる．
    """

    def __init__(
        self,
        *,
        physical_batch_size: int,
        measurement_interval: int = 5,
    ) -> None:
        self._physical_batch_size = physical_batch_size
        self._measurement_interval = max(
            1, measurement_interval
        )
        self._sum_micro_norm_sq: float = 0.0
        self._prev_grads: list[torch.Tensor | None] | None = (
            None
        )
        self._micro_batch_count: int = 0
        self._optimizer_step_count: int = 0

    @property
    def should_measure(self) -> bool:
        return (
            self._optimizer_step_count
            % self._measurement_interval
            == 0
        )

    def on_backward_end(
        self, model: nn.Module, accumulation_step: int
    ) -> None:
        if not self.should_measure:
            return

        if accumulation_step == 0:
            micro_norm_sq = 0.0
            prev_grads: list[torch.Tensor | None] = []
            has_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    has_grad = True
                    g = param.grad.detach()
                    micro_norm_sq += g.pow(2).sum().item()
                    prev_grads.append(g.clone())
                else:
                    prev_grads.append(None)
            if not has_grad:
                return
            self._sum_micro_norm_sq += micro_norm_sq
            self._prev_grads = prev_grads
        else:
            if self._prev_grads is None:
                return
            micro_norm_sq = 0.0
            new_prev: list[torch.Tensor | None] = []
            params = list(model.parameters())
            if len(self._prev_grads) != len(params):
                self._reset()
                return
            for idx, param in enumerate(params):
                if param.grad is not None:
                    g = param.grad.detach()
                    prev = self._prev_grads[idx]
                    if prev is not None:
                        micro_norm_sq += (
                            (g - prev).pow(2).sum().item()
                        )
                    else:
                        micro_norm_sq += g.pow(2).sum().item()
                    new_prev.append(g.clone())
                else:
                    new_prev.append(None)
            self._sum_micro_norm_sq += micro_norm_sq
            self._prev_grads = new_prev

        self._micro_batch_count = accumulation_step + 1

    def compute(
        self, model: nn.Module
    ) -> tuple[float, float, float, int, int] | None:
        should_compute = self.should_measure
        self._optimizer_step_count += 1

        if not should_compute or self._micro_batch_count < 2:
            self._reset()
            return None

        mean_grad_norm_sq = 0.0
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                mean_grad_norm_sq += (
                    param.grad.detach().pow(2).sum().item()
                )

        if not has_grad:
            self._reset()
            return None

        if (
            not math.isfinite(mean_grad_norm_sq)
            or mean_grad_norm_sq < 1e-30
        ):
            self._reset()
            return None

        k = self._micro_batch_count
        s = self._sum_micro_norm_sq
        g = mean_grad_norm_sq
        b = self._physical_batch_size

        if s > 1e30 or math.isnan(s):
            self._reset()
            return None

        ratio = k * s / g
        if ratio <= 1.0:
            self._reset()
            return None

        b_noise = b * k / (k - 1) * (ratio - 1)
        self._reset()
        return (b_noise, s, g, k, b)

    def _reset(self) -> None:
        self._sum_micro_norm_sq = 0.0
        self._prev_grads = None
        self._micro_batch_count = 0


def _make_multi_param_model() -> nn.Module:
    """パラメータテンソルが複数ある(=同期回数が本来パラメータ数に比例する)モデル．"""
    return nn.Sequential(
        nn.Linear(16, 12),
        nn.Tanh(),
        nn.Linear(12, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
    )


def _install_sync_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """``Tensor.item`` / ``Tensor.tolist`` の呼び出しを記録する．

    どちらもデバイス上の値を host へ落とすので，CUDA では
    host-device 同期になる．CPU 上でも**呼ばれたかどうか**は同じに
    観測できるため，GPU 無しで同期の不在を直接検証できる．

    Returns:
        呼ばれたメソッド名が順に積まれるリスト(``clear()`` で区間を切る)．
    """
    calls: list[str] = []
    real_item = torch.Tensor.item
    real_tolist = torch.Tensor.tolist

    def counting_item(self: torch.Tensor) -> object:
        calls.append("item")
        return real_item(self)

    def counting_tolist(self: torch.Tensor) -> object:
        calls.append("tolist")
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    return calls


class TestDeviceScalarAccumulation:
    """host-device 同期を除去しても計測値が変わらないことの回帰テスト．

    audits backlog Deferred 7 (`2026-08-08 app/learning`) の消化．
    """

    def test_bit_identical_to_python_float_reference(
        self,
    ) -> None:
        """device 累算の結果が旧 ``.item()`` 逐次加算と bit 単位で一致する．

        ``_accumulate_device_scalar`` が float64 で累算することが要で，
        float32 に落とすとこのテストは落ちる．
        """
        torch.manual_seed(20260815)
        model = _make_multi_param_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=1,
        )
        reference = _PythonFloatReferenceEstimator(
            physical_batch_size=32,
            measurement_interval=1,
        )

        compared = 0
        for _ in range(3):
            model.zero_grad()
            for step in range(4):
                loss = (
                    model(torch.randn(8, 16)).pow(2).sum() / 4
                )
                loss.backward()
                estimator.on_backward_end(
                    model, accumulation_step=step
                )
                reference.on_backward_end(
                    model, accumulation_step=step
                )

            actual = estimator.compute(model)
            expected = reference.compute(model)

            assert (actual is None) == (expected is None)
            if actual is None or expected is None:
                continue
            compared += 1
            # 近似ではなく厳密一致を要求する — 「挙動不変」の主張は
            # 許容誤差では担保できない．
            assert actual.sum_micro_norm_sq == expected[1], (
                f"S mismatch: {actual.sum_micro_norm_sq!r} "
                f"!= {expected[1]!r}"
            )
            assert actual.mean_grad_norm_sq == expected[2], (
                f"G mismatch: {actual.mean_grad_norm_sq!r} "
                f"!= {expected[2]!r}"
            )
            assert actual.b_noise == expected[0], (
                f"b_noise mismatch: {actual.b_noise!r} "
                f"!= {expected[0]!r}"
            )
            assert actual.micro_batch_count == expected[3]
            assert actual.physical_batch_size == expected[4]

        assert compared > 0, (
            "全 cycle で None が返っており等価性を比較できていない"
        )

    def test_bit_identical_with_half_precision_gradients(
        self,
    ) -> None:
        """float16 勾配でも一致する(``.to(float64)`` と ``.item()`` の等価性)．"""
        torch.manual_seed(4649)
        # torch は param と grad の dtype 一致を強制するので，
        # 勾配を half にするにはモデルごと half にする．
        model = _make_multi_param_model().half()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=16,
            measurement_interval=1,
        )
        reference = _PythonFloatReferenceEstimator(
            physical_batch_size=16,
            measurement_interval=1,
        )

        for step in range(3):
            for param in model.parameters():
                param.grad = torch.randn(
                    param.shape, dtype=torch.float16
                )
            estimator.on_backward_end(
                model, accumulation_step=step
            )
            reference.on_backward_end(
                model, accumulation_step=step
            )

        actual = estimator.compute(model)
        expected = reference.compute(model)

        assert actual is not None
        assert expected is not None
        assert actual.sum_micro_norm_sq == expected[1]
        assert actual.mean_grad_norm_sq == expected[2]
        assert actual.b_noise == expected[0]

    def test_no_host_sync_during_backward_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``on_backward_end`` が一度も host へ値を落とさない．

        これが Deferred 7 の本題である — 旧実装はパラメータテンソル
        ごとに ``.item()`` を呼んでいたので，このテストは 1 micro-batch
        目で落ちる．
        """
        torch.manual_seed(20260815)
        model = _make_multi_param_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=1,
        )
        syncs = _install_sync_counter(monkeypatch)

        model.zero_grad()
        for step in range(4):
            loss = model(torch.randn(8, 16)).pow(2).sum() / 4
            loss.backward()
            syncs.clear()
            estimator.on_backward_end(
                model, accumulation_step=step
            )
            assert syncs == [], (
                f"on_backward_end(step={step}) が "
                f"host 同期を起こした: {syncs}"
            )

    def test_compute_syncs_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``compute`` の同期は 1 回だけで，パラメータ数に依存しない．"""
        torch.manual_seed(20260815)
        model = _make_multi_param_model()
        assert len(list(model.parameters())) > 2, (
            "同期回数がパラメータ数に比例しないことを見るには"
            "パラメータが複数必要"
        )
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=1,
        )

        model.zero_grad()
        for step in range(4):
            loss = model(torch.randn(8, 16)).pow(2).sum() / 4
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )

        syncs = _install_sync_counter(monkeypatch)
        result = estimator.compute(model)

        assert result is not None
        assert len(syncs) == 1, (
            f"compute の host 同期が 1 回ではない: {syncs}"
        )
