import logging
from collections.abc import Iterator, Sequence
from typing import cast

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from maou.app.learning.adaptive_batch import (
    AdaptiveBatchConfig,
    AdaptiveBatchController,
)
from maou.app.learning.callbacks import (
    AdaptiveBatchCallback,
    ModelInputs,
    TrainingCallback,
    TrainingContext,
)
from maou.app.learning.gradient_noise_scale import (
    GradientNoiseScaleEstimator,
)
from maou.app.learning.policy_targets import (
    PolicyTargetMode,
    build_policy_targets,
)
from maou.app.learning.value_targets import (
    ValueTargetMode,
    resolve_value_targets,
)


class TrainingLoop:
    """Generic training loop with callback support.

    Args:
        model: 学習対象のモデル．
        device: 計算デバイス(cpu / cuda)．
        optimizer: オプティマイザ．
        loss_fn_policy: 方策損失関数．
        loss_fn_value: 価値損失関数．
        policy_loss_ratio: 方策損失の重み係数．
        value_loss_ratio: 価値損失の重み係数．
        callbacks: 学習コールバックのリスト．
        logger: ロガー．None の場合はモジュールデフォルトを使用．
        gradient_accumulation_steps: 勾配蓄積ステップ数．
            adaptive_batch_config が設定されている場合は無視される．
        policy_target_mode: 方策ターゲットの計算方式．
        value_target_mode: 価値ターゲットの計算方式．
        adaptive_batch_config: GNS ベース adaptive batch の設定．
            指定時は physical_batch_size も必須．
        physical_batch_size: DataLoader の物理バッチサイズ．
            adaptive_batch_config と合わせて使用する．
        adaptive_batch_callback: Adaptive batch 表示用コールバック．
            callbacks に含まれていなければ自動追加される．
        gns_estimator: 外部で生成済みの GNS 推定器．
            adaptive_controller と共に渡すとエポック間で
            EMA 状態と accumulation_steps を引き継ぐ．
            片方だけ渡した場合は警告付きで両方新規作成される．
        adaptive_controller: 外部で生成済みの adaptive batch コントローラ．
            gns_estimator と共に渡す必要がある．
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: torch.nn.Module,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
        callbacks: list[TrainingCallback] | None = None,
        logger: logging.Logger | None = None,
        gradient_accumulation_steps: int = 1,
        policy_target_mode: PolicyTargetMode = PolicyTargetMode.WIN_RATE,
        value_target_mode: ValueTargetMode = ValueTargetMode.RESULT_VALUE,
        adaptive_batch_config: AdaptiveBatchConfig
        | None = None,
        physical_batch_size: int | None = None,
        adaptive_batch_callback: AdaptiveBatchCallback
        | None = None,
        gns_estimator: GradientNoiseScaleEstimator
        | None = None,
        adaptive_controller: AdaptiveBatchController
        | None = None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn_policy = loss_fn_policy
        self.loss_fn_value = loss_fn_value
        self.policy_loss_ratio = policy_loss_ratio
        self.value_loss_ratio = value_loss_ratio
        self.callbacks = callbacks or []
        self.logger = logger or logging.getLogger(__name__)
        self._cuda_sync_enabled = False
        self.policy_target_mode = policy_target_mode
        self.value_target_mode = value_target_mode

        # Gradient accumulation設定
        self.gradient_accumulation_steps = max(
            1, gradient_accumulation_steps
        )

        # Accumulation cycle カウンタ (動的 steps 変更に対応)
        self._accumulation_counter: int = 0

        # Adaptive batch size 設定
        self._gns_estimator: (
            GradientNoiseScaleEstimator | None
        ) = None
        self._adaptive_controller: (
            AdaptiveBatchController | None
        ) = None
        self._adaptive_callback: (
            AdaptiveBatchCallback | None
        ) = None

        if adaptive_batch_config is not None:
            if physical_batch_size is None:
                msg = (
                    "physical_batch_size is required "
                    "when adaptive_batch_config is set"
                )
                raise ValueError(msg)

            if gradient_accumulation_steps > 1:
                self.logger.warning(
                    "gradient_accumulation_steps=%d は "
                    "adaptive batch 有効時は無視されます "
                    "(min_accumulation_steps=%d を使用)",
                    gradient_accumulation_steps,
                    adaptive_batch_config.min_accumulation_steps,
                )

            # 外部から渡された controller/estimator を使う場合は
            # EMA 状態と current_steps を引き継ぐ(エポック間の継続)
            _has_controller = adaptive_controller is not None
            _has_estimator = gns_estimator is not None
            if _has_controller != _has_estimator:
                self.logger.warning(
                    "gns_estimator と adaptive_controller は"
                    "両方同時に渡す必要があります．"
                    "片方のみ渡された場合は両方とも新規作成します"
                )
            if _has_controller and _has_estimator:
                assert adaptive_controller is not None
                self._adaptive_controller = adaptive_controller
                self._gns_estimator = gns_estimator
                self.gradient_accumulation_steps = adaptive_controller.current_accumulation_steps
            else:
                self.gradient_accumulation_steps = (
                    adaptive_batch_config.min_accumulation_steps
                )
                self._gns_estimator = GradientNoiseScaleEstimator(
                    physical_batch_size=physical_batch_size,
                    measurement_interval=adaptive_batch_config.measurement_interval,
                )
                self._adaptive_controller = (
                    AdaptiveBatchController(
                        config=adaptive_batch_config,
                        physical_batch_size=physical_batch_size,
                    )
                )

            self._adaptive_callback = adaptive_batch_callback
            if self._adaptive_callback is not None:
                # callbacks に含まれていなければ自動追加
                # (get_postfix() による tqdm 表示のため callbacks にも必要)
                if (
                    self._adaptive_callback
                    not in self.callbacks
                ):
                    self.callbacks.append(
                        self._adaptive_callback
                    )
                assert self._adaptive_controller is not None
                self._adaptive_callback.update_display(
                    self._adaptive_controller.smoothed_gns,
                    self.gradient_accumulation_steps,
                )

            self.logger.info(
                "Adaptive batch enabled: accum_steps %d-%d, "
                "adjustment_interval=%d, physical_bs=%d",
                adaptive_batch_config.min_accumulation_steps,
                adaptive_batch_config.max_accumulation_steps,
                adaptive_batch_config.adjustment_interval,
                physical_batch_size,
            )
        elif self.gradient_accumulation_steps > 1:
            self.logger.info(
                "Gradient accumulation enabled: %d steps "
                "(effective batch size multiplied by %d)",
                gradient_accumulation_steps,
                gradient_accumulation_steps,
            )

        # Loss finitude check interval (reduces GPU sync points)
        self._finitude_check_interval: int = 100

        # Mixed precision training用のGradScalerを初期化（GPU使用時のみ）
        if self.device.type == "cuda":
            self.scaler: GradScaler | None = GradScaler("cuda")
        else:
            self.scaler = None

    def run_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        *,
        max_batches: int | None = None,
        enable_profiling: bool = False,
        progress_bar: bool = True,
        train_mode: bool = True,
        force_cuda_sync: bool | None = None,
    ) -> None:
        """Run a single epoch of training or validation."""
        previous_sync_state = self._cuda_sync_enabled
        resolved_sync_state = (
            force_cuda_sync
            if force_cuda_sync is not None
            else enable_profiling
        )
        self._cuda_sync_enabled = resolved_sync_state

        if (
            self.device.type == "cuda"
            and previous_sync_state != resolved_sync_state
        ):
            self.logger.debug(
                "CUDA synchronization %s for epoch %d (profiling=%s, forced=%s)",
                "enabled"
                if resolved_sync_state
                else "disabled",
                epoch_idx,
                enable_profiling,
                force_cuda_sync,
            )

        # モデルのモード設定
        self.model.train(train_mode)

        # エポック境界で accumulation カウンタをリセット
        # バッチ数が gradient_accumulation_steps の倍数でない場合，
        # 前エポック末尾の不完全な cycle の stale 勾配が混入するのを防ぐ
        if train_mode:
            self._accumulation_counter = 0

        # エポック開始のコールバック
        for callback in self.callbacks:
            callback.on_epoch_start(epoch_idx)

        # PyTorchプロファイラーの設定
        profiler = None
        if enable_profiling:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./profiler_logs"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()

        try:
            transfer_iter = self._iterate_with_transfer(
                dataloader, epoch_idx
            )
            dataloader_iter = (
                tqdm(
                    transfer_iter,
                    desc="Training"
                    if train_mode
                    else "Validation",
                    total=len(dataloader),
                )
                if progress_bar
                else transfer_iter
            )

            for batch_idx, context in dataloader_iter:
                if (
                    max_batches is not None
                    and batch_idx >= max_batches
                ):
                    break

                # バッチ開始のコールバック
                for callback in self.callbacks:
                    callback.on_batch_start(context)

                if train_mode:
                    # 学習モード：勾配計算あり
                    self._train_batch(context)
                else:
                    # 評価モード：勾配計算なし
                    self._eval_batch(context)

                # バッチ終了のコールバック
                for callback in self.callbacks:
                    callback.on_batch_end(context)

                # tqdm postfix 更新
                if isinstance(dataloader_iter, tqdm):
                    postfix: dict[str, str] = {}
                    for callback in self.callbacks:
                        cb_postfix = callback.get_postfix()
                        if cb_postfix is not None:
                            postfix.update(cb_postfix)
                    if postfix:
                        dataloader_iter.set_postfix(postfix)

                # プロファイラーのステップ
                if profiler is not None:
                    profiler.step()

        finally:
            if (
                self.device.type == "cuda"
                and previous_sync_state
                != self._cuda_sync_enabled
            ):
                self.logger.debug(
                    "CUDA synchronization restored to %s after epoch %d",
                    "enabled"
                    if previous_sync_state
                    else "disabled",
                    epoch_idx,
                )
            self._cuda_sync_enabled = previous_sync_state
            if profiler is not None:
                profiler.stop()

        # エポック終了のコールバック
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_idx)

    def _transfer_to_device(
        self, context: TrainingContext
    ) -> None:
        """Transfer data to device with callback hooks."""
        for callback in self.callbacks:
            callback.on_data_transfer_start(context)

        if self.device.type == "cuda":
            # GPU転送（DataLoaderのpin_memoryと非同期転送を活用）
            context.inputs = self._move_inputs_to_device(
                context.inputs,
                self.device,
            )
            context.labels_policy = context.labels_policy.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            context.labels_value = context.labels_value.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if context.legal_move_mask is not None:
                context.legal_move_mask = (
                    context.legal_move_mask.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                )
            if context.move_win_rate is not None:
                context.move_win_rate = (
                    context.move_win_rate.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                )

            # GPU同期（正確な転送時間測定のため）
            self._maybe_synchronize("post_data_transfer")

        for callback in self.callbacks:
            callback.on_data_transfer_end(context)

    def _iterate_with_transfer(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
    ) -> Iterator[tuple[int, TrainingContext]]:
        """Iterate dataloader with device transfer.

        CUDAデバイスではストリームオーバーラップによりH2D転送と
        計算を並行して実行する．CPUでは同期的に転送する．
        """
        if self.device.type == "cuda":
            yield from self._iterate_cuda_overlap(
                dataloader, epoch_idx
            )
        else:
            yield from self._iterate_direct(
                dataloader, epoch_idx
            )

    def _iterate_cuda_overlap(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
    ) -> Iterator[tuple[int, TrainingContext]]:
        """Iterate with CUDA stream overlap for async H2D transfer.

        別のCUDAストリームでH2D転送を行い，デフォルトストリームでの
        計算とオーバーラップさせることでスループットを向上させる．
        """
        stream = torch.cuda.Stream()
        data_iter = iter(dataloader)

        # 最初のバッチは同期転送
        try:
            first_raw = next(data_iter)
        except StopIteration:
            return

        current_ctx = self._unpack_batch(
            first_raw, batch_idx=0, epoch_idx=epoch_idx
        )
        self._transfer_to_device(current_ctx)

        batch_idx = 0
        for next_raw in data_iter:
            next_ctx = self._unpack_batch(
                next_raw,
                batch_idx=batch_idx + 1,
                epoch_idx=epoch_idx,
            )
            with torch.cuda.stream(stream):
                # NOTE: _transfer_to_device内のコールバック(on_data_transfer_start/end)は
                # 転送ストリーム上で実行される．将来のコールバックでCUDA操作を行うと
                # 誤ったストリームにスケジュールされるため注意．
                self._transfer_to_device(next_ctx)

            # デフォルトストリームで現在のバッチを学習（H2D転送とオーバーラップ）
            yield batch_idx, current_ctx

            # 次バッチの転送完了を待機
            stream.synchronize()
            current_ctx = next_ctx
            batch_idx += 1

        # 最後のバッチを返す
        yield batch_idx, current_ctx

    def _iterate_direct(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
    ) -> Iterator[tuple[int, TrainingContext]]:
        """Direct iteration with synchronous transfer (CPU path)."""
        for batch_idx, raw_batch in enumerate(dataloader):
            ctx = self._unpack_batch(
                raw_batch,
                batch_idx=batch_idx,
                epoch_idx=epoch_idx,
            )
            self._transfer_to_device(ctx)
            yield batch_idx, ctx

    def _unpack_batch(
        self,
        data: tuple[
            ModelInputs,
            tuple[torch.Tensor, ...],
        ],
        batch_idx: int,
        epoch_idx: int,
    ) -> TrainingContext:
        """Unpack raw dataloader output into a TrainingContext."""
        inputs, targets = data
        labels_policy = targets[0]
        labels_value = targets[1]
        legal_move_mask: torch.Tensor | None = (
            targets[2] if len(targets) > 2 else None
        )
        move_win_rate: torch.Tensor | None = None
        if len(targets) > 3:
            move_win_rate = targets[3]

        # ValueTargetModeに応じてvalue教師信号を切り替え
        resolved_value = resolve_value_targets(
            labels_value,
            mode=self.value_target_mode,
            move_win_rate=move_win_rate,
        )

        batch_size = self._resolve_batch_size(inputs)
        return TrainingContext(
            batch_idx=batch_idx,
            epoch_idx=epoch_idx,
            inputs=inputs,
            labels_policy=labels_policy,
            labels_value=resolved_value,
            legal_move_mask=legal_move_mask,
            batch_size=batch_size,
            move_win_rate=move_win_rate,
        )

    def _train_batch(self, context: TrainingContext) -> None:
        """Train a single batch with gradient computation.

        Adaptive batch 不変条件:
            gradient_accumulation_steps の変更は _maybe_update_adaptive_batch
            (accumulation cycle 完了時のみ呼ばれる)で行われるため，
            同一 cycle 内で steps が変わることはない．
            ただし cycle 完了判定は現在の steps に基づくため，
            steps が前 cycle より減少した場合，カウンタが新しい steps を
            超えていると即座に cycle が完了する(勾配は正しく蓄積済み)．
            この遷移 cycle では _micro_batch_count が新しい steps を
            超えるが，GNS 算出は実際に蓄積された K を使うため
            推定値に小さな誤差が生じる．次 cycle からは正しい K で計測される．
        """
        # カウンタベースの accumulation step 管理
        accumulation_step = self._accumulation_counter
        is_accumulation_step = accumulation_step < (
            self.gradient_accumulation_steps - 1
        )

        # 勾配蓄積サイクルの最初のステップでのみ勾配をクリア
        if accumulation_step == 0:
            self.optimizer.zero_grad()

        # Mixed precision training with autocast
        if self.scaler is not None:
            skipped = self._train_batch_mixed_precision(
                context, is_accumulation_step, accumulation_step
            )
        else:
            skipped = self._train_batch_full_precision(
                context, is_accumulation_step, accumulation_step
            )

        # accumulation カウンタ更新
        # 非有限損失で backward がスキップされた場合は
        # 勾配がクリアされているためサイクルを最初からやり直す
        if skipped:
            self._accumulation_counter = 0
        elif not is_accumulation_step:
            self._accumulation_counter = 0
        else:
            self._accumulation_counter += 1

    def _train_batch_mixed_precision(
        self,
        context: TrainingContext,
        is_accumulation_step: bool,
        accumulation_step: int,
    ) -> bool:
        """Train batch with mixed precision.

        Returns:
            True if the batch was skipped (e.g. non-finite loss).
        """
        if self.scaler is None:
            raise RuntimeError(
                "GradScaler is None but mixed precision training was requested"
            )

        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        with autocast(self.device.type):
            context.outputs_policy, context.outputs_value = (
                self.model(context.inputs)
            )

            # 損失計算
            for callback in self.callbacks:
                callback.on_loss_computation_start(context)

            policy_loss = self._compute_policy_loss(context)
            value_loss = self.loss_fn_value(
                context.outputs_value, context.labels_value
            )
            # Gradient accumulation: 損失を蓄積ステップ数で正規化
            # adaptive batch では steps が動的に変わるが，変更は
            # cycle 完了時のみ行われるため，同一 cycle 内では一貫した
            # 正規化係数が使用される
            context.loss = (
                self.policy_loss_ratio * policy_loss
                + self.value_loss_ratio * value_loss
            ) / self.gradient_accumulation_steps
            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        self._maybe_synchronize("post_forward_mixed_precision")

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # Check loss finitude periodically to reduce GPU sync points
        if (
            context.batch_idx % self._finitude_check_interval
            == 0
        ):
            loss_is_finite = torch.isfinite(context.loss)
            if not bool(loss_is_finite.item()):
                policy_loss_is_finite = torch.isfinite(
                    policy_loss
                )
                value_loss_is_finite = torch.isfinite(
                    value_loss
                )
                self.logger.warning(
                    "Non-finite loss detected (epoch=%d, batch=%d): loss=%s "
                    "policy_loss=%s (finite=%s) value_loss=%s (finite=%s)",
                    context.epoch_idx,
                    context.batch_idx,
                    context.loss.detach(),
                    policy_loss.detach(),
                    bool(policy_loss_is_finite.item()),
                    value_loss.detach(),
                    bool(value_loss_is_finite.item()),
                )
                self.optimizer.zero_grad(set_to_none=True)
                # GNS estimator をリセット: backward() がスキップされるため
                # on_backward_end() と compute() が呼ばれず stale データが残る
                if self._gns_estimator is not None:
                    self._gns_estimator.reset_cycle()
                return True

        # 逆伝播
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)

        if context.loss is None:
            raise RuntimeError(
                "Loss computation failed - context.loss is None"
            )

        self.scaler.scale(context.loss).backward()

        self._maybe_synchronize("post_backward_mixed_precision")

        # GNS 計測: backward 後に micro-batch 勾配統計を収集
        # mixed precision ではスケール済み勾配を使用する．
        # compute() も unscale_ 前に呼ぶため S と G が同じスケールになり，
        # 比率計算で scale² が約分される．
        if self._gns_estimator is not None:
            self._gns_estimator.on_backward_end(
                self.model, accumulation_step
            )

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # Gradient accumulation: 蓄積ステップの最後でのみオプティマイザを実行
        if not is_accumulation_step:
            # GNS 推定: unscale_ 前に計算する．
            # on_backward_end で収集した S(micro-batch 勾配ノルム和)と
            # compute() で計算する G(蓄積済み勾配ノルム)が共にスケール済み
            # の状態であるため，B_noise = b*K/(K-1)*(K*S/G - 1) の比率計算で
            # scale² が約分され正しい推定値が得られる．
            # unscale_ 後に compute すると S(スケール済み) vs G(スケール解除済み)
            # で scale² の誤差が混入するため不可．
            self._maybe_update_adaptive_batch()

            # unscale → クリッピング → step
            self.scaler.unscale_(self.optimizer)

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # オプティマイザステップ
            for callback in self.callbacks:
                callback.on_optimizer_step_start(context)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._maybe_synchronize(
                "post_optimizer_step_mixed_precision"
            )

            for callback in self.callbacks:
                callback.on_optimizer_step_end(context)

        return False

    def _move_inputs_to_device(
        self,
        inputs: ModelInputs,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> ModelInputs:
        def move(
            value: object, path: tuple[int, ...]
        ) -> object:
            if isinstance(value, torch.Tensor):
                target_dtype = self._infer_input_dtype(
                    path, value
                )
                if (
                    target_dtype is not None
                    and value.dtype != target_dtype
                ):
                    return value.to(
                        device=device,
                        dtype=target_dtype,
                        non_blocking=non_blocking,
                    )
                return value.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            if isinstance(value, tuple):
                return tuple(
                    move(item, path + (index,))
                    for index, item in enumerate(value)
                )
            if isinstance(value, list):
                return [
                    move(item, path + (index,))
                    for index, item in enumerate(value)
                ]
            return value

        moved = move(inputs, tuple())
        return cast(ModelInputs, moved)

    @staticmethod
    def _infer_input_dtype(
        index_path: tuple[int, ...], tensor: torch.Tensor
    ) -> torch.dtype | None:
        if not index_path:
            return None

        root_index = index_path[0]
        if root_index == 0 and not torch.is_floating_point(
            tensor
        ):
            return torch.long
        if root_index == 1 and tensor.dtype != torch.float32:
            return torch.float32
        return None

    @staticmethod
    def _resolve_batch_size(inputs: ModelInputs) -> int:
        if isinstance(inputs, torch.Tensor):
            return int(inputs.size(0))
        if isinstance(inputs, Sequence):
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    return int(element.size(0))
                if isinstance(element, Sequence):
                    try:
                        return TrainingLoop._resolve_batch_size(
                            element
                        )
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                    ):
                        continue
            msg = "Unable to resolve batch size from nested inputs."
            raise TypeError(msg)
        msg = f"Unsupported input type for batch size resolution: {type(inputs)!r}"
        raise TypeError(msg)

    def _train_batch_full_precision(
        self,
        context: TrainingContext,
        is_accumulation_step: bool,
        accumulation_step: int,
    ) -> bool:
        """Train batch with full precision.

        Returns:
            True if the batch was skipped (e.g. non-finite loss).
        """
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        context.outputs_policy, context.outputs_value = (
            self.model(context.inputs)
        )

        self._maybe_synchronize("post_forward_full_precision")

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        policy_loss = self._compute_policy_loss(context)
        value_loss = self.loss_fn_value(
            context.outputs_value, context.labels_value
        )
        # Gradient accumulation: 損失を蓄積ステップ数で正規化
        # adaptive batch では steps が動的に変わるが，変更は
        # cycle 完了時のみ行われるため，同一 cycle 内では一貫した
        # 正規化係数が使用される
        context.loss = (
            self.policy_loss_ratio * policy_loss
            + self.value_loss_ratio * value_loss
        ) / self.gradient_accumulation_steps
        for callback in self.callbacks:
            callback.on_loss_computation_end(context)

        # Check loss finitude periodically to reduce GPU sync points
        if (
            context.batch_idx % self._finitude_check_interval
            == 0
        ):
            loss_is_finite = torch.isfinite(context.loss)
            if not bool(loss_is_finite.item()):
                policy_loss_is_finite = torch.isfinite(
                    policy_loss
                )
                value_loss_is_finite = torch.isfinite(
                    value_loss
                )
                self.logger.warning(
                    "Non-finite loss detected (epoch=%d, batch=%d): loss=%s "
                    "policy_loss=%s (finite=%s) value_loss=%s (finite=%s)",
                    context.epoch_idx,
                    context.batch_idx,
                    context.loss.detach(),
                    policy_loss.detach(),
                    bool(policy_loss_is_finite.item()),
                    value_loss.detach(),
                    bool(value_loss_is_finite.item()),
                )
                self.optimizer.zero_grad(set_to_none=True)
                # GNS estimator をリセット(mixed precision パスと同様)
                if self._gns_estimator is not None:
                    self._gns_estimator.reset_cycle()
                return True

        # 逆伝播
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)

        if context.loss is None:
            raise RuntimeError(
                "Loss computation failed - context.loss is None"
            )

        context.loss.backward()

        self._maybe_synchronize("post_backward_full_precision")

        # GNS 計測: backward 後に micro-batch 勾配統計を収集
        if self._gns_estimator is not None:
            self._gns_estimator.on_backward_end(
                self.model, accumulation_step
            )

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # Gradient accumulation: 蓄積ステップの最後でのみオプティマイザを実行
        if not is_accumulation_step:
            # GNS 推定と adaptive batch 調整(クリッピング前に計算し S と G を同じ基準で比較)
            self._maybe_update_adaptive_batch()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # オプティマイザステップ
            for callback in self.callbacks:
                callback.on_optimizer_step_start(context)

            self.optimizer.step()

            self._maybe_synchronize(
                "post_optimizer_step_full_precision"
            )

            for callback in self.callbacks:
                callback.on_optimizer_step_end(context)

        return False

    def _eval_batch(self, context: TrainingContext) -> None:
        """Evaluate a single batch without gradient computation."""
        with torch.no_grad():
            # Mixed precision for validation (memory efficiency)
            if self.scaler is not None:
                self._eval_batch_mixed_precision(context)
            else:
                self._eval_batch_full_precision(context)

    def _eval_batch_mixed_precision(
        self, context: TrainingContext
    ) -> None:
        """Evaluate batch with mixed precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        with autocast(self.device.type):
            context.outputs_policy, context.outputs_value = (
                self.model(context.inputs)
            )

            # 損失計算
            for callback in self.callbacks:
                callback.on_loss_computation_start(context)

            policy_loss = self._compute_policy_loss(context)
            context.loss = (
                self.policy_loss_ratio * policy_loss
                + self.value_loss_ratio
                * self.loss_fn_value(
                    context.outputs_value, context.labels_value
                )
            )

            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        self._maybe_synchronize(
            "post_eval_forward_mixed_precision"
        )

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 評価モードでは逆伝播とオプティマイザステップはスキップ
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)
            callback.on_backward_pass_end(context)
            callback.on_optimizer_step_start(context)
            callback.on_optimizer_step_end(context)

    def _maybe_update_adaptive_batch(self) -> None:
        """GNS 推定値に基づいて gradient_accumulation_steps を調整する．

        毎 optimizer step で呼ばれる．GNS 推定値が得られなかった場合も
        controller.update(None) を呼び出し，adjustment_interval が
        全 optimizer step に対する間隔として正しく機能するようにする．
        """
        if (
            self._gns_estimator is None
            or self._adaptive_controller is None
        ):
            return

        estimate = self._gns_estimator.compute(self.model)
        b_noise = (
            estimate.b_noise if estimate is not None else None
        )

        new_steps = self._adaptive_controller.update(b_noise)
        if new_steps != self.gradient_accumulation_steps:
            self.gradient_accumulation_steps = new_steps

        # AdaptiveBatchCallback の表示を更新
        if self._adaptive_callback is not None:
            self._adaptive_callback.update_display(
                self._adaptive_controller.smoothed_gns,
                self.gradient_accumulation_steps,
                low_noise=self._gns_estimator.last_compute_was_low_noise,
            )

    def _maybe_synchronize(self, reason: str) -> None:
        if self.device.type != "cuda":
            return

        if not self._cuda_sync_enabled:
            return

        torch.cuda.synchronize()

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "torch.cuda.synchronize() executed (%s)",
                reason,
            )

    def _eval_batch_full_precision(
        self, context: TrainingContext
    ) -> None:
        """Evaluate batch with full precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        context.outputs_policy, context.outputs_value = (
            self.model(context.inputs)
        )

        self._maybe_synchronize(
            "post_eval_forward_full_precision"
        )

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        policy_loss = self._compute_policy_loss(context)
        context.loss = (
            self.policy_loss_ratio * policy_loss
            + self.value_loss_ratio
            * self.loss_fn_value(
                context.outputs_value, context.labels_value
            )
        )

        for callback in self.callbacks:
            callback.on_loss_computation_end(context)

        # 評価モードでは逆伝播とオプティマイザステップはスキップ
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)
            callback.on_backward_pass_end(context)
            callback.on_optimizer_step_start(context)
            callback.on_optimizer_step_end(context)

    def _compute_policy_loss(
        self, context: TrainingContext
    ) -> torch.Tensor:
        if context.outputs_policy is None:
            raise RuntimeError(
                "Policy outputs are required before computing the loss"
            )

        if context.legal_move_mask is not None:
            mask_bool = context.legal_move_mask.bool()
            # 全ゼロマスク行の防御: log_softmax([-inf,...]) = NaN を防止
            has_legal = mask_bool.any(dim=1)
            if not has_legal.all():
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Found %d samples with all-zero legal_move_mask; "
                    "skipping masking for those samples to avoid NaN",
                    int((~has_legal).sum().item()),
                )
                # 全ゼロ行はマスクを適用しない(元のlogitsを保持)
                safe_mask = mask_bool | ~has_legal.unsqueeze(1)
                masked_logits = (
                    context.outputs_policy.masked_fill(
                        ~safe_mask, float("-inf")
                    )
                )
            else:
                masked_logits = (
                    context.outputs_policy.masked_fill(
                        ~mask_bool, float("-inf")
                    )
                )
        else:
            masked_logits = context.outputs_policy

        policy_log_probs = torch.nn.functional.log_softmax(
            masked_logits,
            dim=1,
        )
        policy_targets = build_policy_targets(
            context.labels_policy,
            context.legal_move_mask,
            mode=self.policy_target_mode,
            move_win_rate=context.move_win_rate,
            dtype=policy_log_probs.dtype,
            device=policy_log_probs.device,
        )
        context.policy_target_distribution = (
            policy_targets.detach()
        )
        return self.loss_fn_policy(
            policy_log_probs, policy_targets
        )


class RawLogitsTrainingLoop(TrainingLoop):
    """BCEWithLogitsLoss 系の損失関数用 TrainingLoop サブクラス．

    Stage 3 の ``_compute_policy_loss`` は ``log_softmax`` +
    ``normalize_policy_targets`` で方策分布を正規化するが，
    Stage 1 (``ReachableSquaresLoss``) と Stage 2 (``LegalMovesLoss``)
    は生 logits に対する BCEWithLogitsLoss であるため，
    これらの前処理をバイパスする．
    """

    def _compute_policy_loss(
        self, context: TrainingContext
    ) -> torch.Tensor:
        """生logitsを直接 loss_fn_policy に渡す."""
        if context.outputs_policy is None:
            raise RuntimeError(
                "Policy outputs are required before computing the loss"
            )
        return self.loss_fn_policy(
            context.outputs_policy, context.labels_policy
        )


Stage1TrainingLoop = RawLogitsTrainingLoop
"""Stage 1 も同じ raw logits パスを使用するためエイリアスを提供．"""
