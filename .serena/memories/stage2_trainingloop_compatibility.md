# Stage 2 と TrainingLoop の互換性：データフォーマット変換メカニズム

## 概要

Stage 2 (Legal Moves) が TrainingLoop と互換性を持つために，複数の層でデータフォーマット変換が行われている．

---

## 1. Stage2Dataset の出力フォーマット

**ファイル:** `/workspaces/maou/src/maou/app/learning/dataset.py` (行 372-444)

```python
class Stage2Dataset(Dataset, Sized):
    def __getitem__(self, idx: int) -> tuple[
        tuple[torch.Tensor, torch.Tensor],  # features (2要素タプル)
        torch.Tensor,                        # target (1要素テンソル)
    ]:
```

**返り値:**
- `features`: `(board_tensor, pieces_in_hand_tensor)` の 2 要素タプル
- `target`: `legal_moves_tensor` の単一テンソル (MOVE_LABELS_NUM 次元)

つまり Stage2Dataset は **2 要素タプル** を返す:
```
(
    (board_tensor, hand_tensor),    # features
    legal_moves_tensor              # target (BCEWithLogitsLoss用の生ラベル)
)
```

---

## 2. DataLoader による自動バッチング

**ファイル:** `/workspaces/maou/src/maou/interface/learn.py` (行 669-675)

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    # collate_fn は指定されていない → デフォルトcollate_fnを使用
)
```

**重要:** collate_fn が指定されていない．
PyTorch のデフォルト collate_fn は，Dataset の返り値を batch_size でスタックする．

Stage2Dataset の返り値が 2 要素なので，collate_fn の出力も **2 要素タプル** になる:

```
(
    (batched_boards, batched_hands),        # batched features
    batched_legal_moves                     # batched target
)
```

---

## 3. TrainingLoop._unpack_batch() の期待値

**ファイル:** `/workspaces/maou/src/maou/app/learning/training_loop.py` (行 314-339)

```python
def _unpack_batch(
    self,
    data: tuple[
        ModelInputs,
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor | None
        ],
    ],
    batch_idx: int,
    epoch_idx: int,
) -> TrainingContext:
    """Unpack raw dataloader output into a TrainingContext."""
    (
        inputs,
        (labels_policy, labels_value, legal_move_mask),
    ) = data
```

**期待値:** **3 要素タプル**
```
(
    inputs,
    (
        labels_policy,       # torch.Tensor
        labels_value,        # torch.Tensor
        legal_move_mask      # torch.Tensor | None
    )
)
```

---

## 4. データフォーマット不一致の解決

### 問題
- Stage2Dataset: 2 要素タプル
- TrainingLoop._unpack_batch(): 3 要素タプルを期待

### 解決策: Stage2StreamingAdapter

**ファイル:** `/workspaces/maou/src/maou/app/learning/streaming_dataset.py` (行 603-636)

```python
class Stage2StreamingAdapter(IterableDataset):
    """StreamingStage2Dataset を TrainingLoop の入力形式に変換するアダプタ．
    
    StreamingStage2Dataset は ``((board, hand), legal_moves)`` を yield するが，
    TrainingLoop._unpack_batch() は
    ``((board, hand), (labels_policy, labels_value, legal_move_mask))``
    を期待する．このアダプタがダミーの value ラベルと None マスクを挿入する．
    """

    def __iter__(self) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, None],
        ]
    ]:
        for inputs, targets in self._dataset:
            dummy_value = torch.zeros(
                targets.shape[0], 1, dtype=torch.float32
            )
            yield (inputs, (targets, dummy_value, None))
```

**変換の流れ:**
1. StreamingStage2Dataset: `((board, hand), legal_moves)` を yield
2. Stage2StreamingAdapter: 3 要素タプルに変換
   ```
   ((board, hand), (legal_moves, dummy_value_zeros, None))
   ```
3. TrainingLoop._unpack_batch(): このフォーマットを受け取り，正常にアンパック

---

## 5. 通常の (非ストリーミング) Stage2 の場合

**ファイル:** `/workspaces/maou/src/maou/interface/learn.py` (行 663-707)

通常の Stage2 では，明示的な適応層がない．その代わり以下のフローが成立:

1. **Stage2Dataset** が返す値:
   ```python
   (
       (board_tensor, pieces_in_hand_tensor),
       legal_moves_tensor
   )
   ```

2. **DataLoader のデフォルト collate_fn** が 2 要素タプルのバッチを作成

3. **SingleStageTrainingLoop に渡す**
   - SingleStageTrainingLoop は `run_stage2_with_training_loop()` 経由で作成される
   - この関数内で **Stage2TrainingLoop** が使用される

### run_stage2_with_training_loop() の流れ

**ファイル:** `/workspaces/maou/src/maou/app/learning/multi_stage_training.py` (行 663-855)

```python
def run_stage2_with_training_loop(
    *,
    backbone: HeadlessNetwork,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, LegalMovesHead]:
    
    # Stage2Dataset は通常の 2 要素タプルを返す
    # しかし config.dataloader は既に設定済みで，Stage2Dataset から来たもの
```

**重要な発見:** Stage2Dataset を直接 DataLoader に入れて，
Stage2TrainingLoop に渡すと，2 要素のタプルが DataLoader から来る．

---

## 6. Stage2TrainingLoop による処理

**ファイル:** `/workspaces/maou/src/maou/app/learning/training_loop.py` (行 816-835)

```python
class Stage2TrainingLoop(TrainingLoop):
    """Stage 2 (Legal Moves) 用の TrainingLoop サブクラス．
    
    Stage 3 の ``_compute_policy_loss`` は ``log_softmax`` +
    ``normalize_policy_targets`` で方策分布を正規化するが，
    Stage 2 の ``LegalMovesLoss`` は生logitsに対するBCEWithLogitsLoss
    であるため，これらの前処理をバイパスする．
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
```

---

## 7. 通常の Stage2 でのデータ不一致の実際の解決

調査結果から，通常の (非ストリーミング) Stage2 では以下が成立:

### パターン A: 通常の SingleStageTrainingLoop (Stage 1用)
- Stage1Dataset: 2 要素 `(features, target_tensor)`
- DataLoader: 2 要素バッチ
- SingleStageTrainingLoop._train_epoch(): 2 要素をアンパック
  ```python
  for batch_idx, (inputs, targets) in enumerate(self.config.dataloader):
  ```

### パターン B: 新しい Stage2TrainingLoop (Stage 2用)
- Stage2Dataset: 2 要素 `(features, target_tensor)`
- DataLoader: 2 要素バッチ
- TrainingLoop._unpack_batch(): **3 要素を期待** ← 問題

**解決策がないように見えるが...**

実際には，Stage 2 ストリーミングモードでのみ Stage2StreamingAdapter が使用される:

**ファイル:** `/workspaces/maou/src/maou/interface/learn.py` (行 846-872)

```python
raw_dataset = StreamingStage2Dataset(
    streaming_source=streaming_source,
    batch_size=batch_size,
    shuffle=True,
)
dataset = Stage2StreamingAdapter(raw_dataset)  # <- ここで変換

dataloader, _ = (
    DataLoaderFactory.create_streaming_dataloaders(
        train_dataset=dataset,  # Stage2StreamingAdapter を入れる
        ...
    )
)
```

---

## 8. StreamingStage2Dataset の出力フォーマット

**ファイル:** `/workspaces/maou/src/maou/app/learning/streaming_dataset.py` (行 435-600)

```python
class StreamingStage2Dataset(IterableDataset):
    """IterableDataset版のStage2Dataset(バッチ単位yield)．"""

    def __iter__(self) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
    ]:
        # _yield_stage2_batches() を呼び出し
```

**返り値:** 2 要素タプル (Stage2Dataset と同じ)
```
((board_tensor, pieces_tensor), legal_moves_tensor)
```

バッチ単位で yield される (IterableDataset なので DataLoader の batch_size=None で使用):

**ファイル:** `/workspaces/maou/src/maou/app/learning/streaming_dataset.py` (行 777-831)

```python
def _yield_stage2_batches(
    columnar_batch: ColumnarBatch,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Generator[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    None,
    None,
]:
    # 各バッチは 2 要素タプルで yield される
    yield (
        (board_tensor, pieces_tensor),
        legal_moves_tensor,
    )
```

---

## 9. データフォーマット変換の全体図

```
通常の Stage2 (非ストリーミング):
  Stage2Dataset.__getitem__()
  ↓ returns 2-tuple
  ((board, hand), legal_moves)
  ↓ (DataLoader batch_size=256)
  Batched: ((batched_boards, batched_hands), batched_legal_moves)
  ↓ (SingleStageTrainingLoop でアンパック)
  inputs, targets = batch
  ↓ (2要素)
  正常に処理

ストリーミング Stage2:
  StreamingStage2Dataset.__iter__()
  ↓ yields 2-tuple batches
  ((board, hand), legal_moves)
  ↓ (Stage2StreamingAdapter でラッピング)
  ((board, hand), (legal_moves, dummy_value, None))
  ↓ (DataLoader batch_size=None で through)
  ((board, hand), (legal_moves, dummy_value, None))
  ↓ (TrainingLoop._unpack_batch でアンパック)
  inputs, (labels_policy, labels_value, legal_move_mask) = data
  ↓ (3要素)
  正常に処理
```

---

## 10. TrainingLoop._unpack_batch の全体コード

**ファイル:** `/workspaces/maou/src/maou/app/learning/training_loop.py` (行 314-339)

```python
def _unpack_batch(
    self,
    data: tuple[
        ModelInputs,
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor | None
        ],
    ],
    batch_idx: int,
    epoch_idx: int,
) -> TrainingContext:
    """Unpack raw dataloader output into a TrainingContext."""
    (
        inputs,
        (labels_policy, labels_value, legal_move_mask),
    ) = data
    batch_size = self._resolve_batch_size(inputs)
    return TrainingContext(
        batch_idx=batch_idx,
        epoch_idx=epoch_idx,
        inputs=inputs,
        labels_policy=labels_policy,
        labels_value=labels_value,
        legal_move_mask=legal_move_mask,
        batch_size=batch_size,
    )
```

---

## 11. SingleStageTrainingLoop でのアンパック (比較用)

**ファイル:** `/workspaces/maou/src/maou/app/learning/multi_stage_training.py` (行 389-402)

```python
for batch_idx, (inputs, targets) in enumerate(
    self.config.dataloader
):
    # Move to device
    board_tensor, hand_tensor = inputs
    board_tensor = board_tensor.to(
        self.device, non_blocking=True
    )
    hand_tensor = (
        hand_tensor.to(self.device, non_blocking=True)
        if hand_tensor is not None
        else None
    )
    targets = targets.to(self.device, non_blocking=True)
```

**アンパック方法:**
- 2 要素で単純アンパック: `(inputs, targets)`
- targets は直接 loss_fn に渡される (多ラベル BCE)

---

## 12. Stage2ModelAdapter の役割

**ファイル:** `/workspaces/maou/src/maou/app/learning/multi_stage_training.py` (行 79-116)

```python
class Stage2ModelAdapter(torch.nn.Module):
    """Stage 2 用のモデルアダプタ．
    
    HeadlessNetwork と LegalMovesHead をラップし，
    TrainingLoop が期待する ``(policy, value)`` の2タプルを返す．
    ``value`` 出力はダミーゼロテンソルで，value loss は ``value_loss_ratio=0.0`` で無視される．
    """

    def __init__(
        self,
        backbone: HeadlessNetwork,
        head: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """フォワードパスを実行し，(policy, dummy_value) を返す．"""
        features = self.backbone.forward_features(inputs)
        logits = self.head(features)
        dummy_value = torch.zeros(
            logits.shape[0], 1, device=logits.device
        )
        return logits, dummy_value
```

**役割:**
- model(inputs) が `(policy_output, value_output)` の 2-tuple を返す
- TrainingLoop が self.model() の出力を期待する形式に合わせる
- value_output はダミーで，TrainingLoop 内で value_loss_ratio=0.0 で無視される

---

## 13. TrainingLoop での損失計算 (Stage2特有)

**ファイル:** `/workspaces/maou/src/maou/app/learning/training_loop.py` (行 816-835)

Stage2TrainingLoop オーバーライド:

```python
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
```

比較: 通常の TrainingLoop (Stage 3):
```python
# log_softmax + normalize_policy_targets を適用
policy_log_probs = torch.nn.functional.log_softmax(
    masked_logits,
    dim=1,
)
policy_targets = normalize_policy_targets(...)
return self.loss_fn_policy(
    policy_log_probs, policy_targets
)
```

Stage 2 では正規化を**バイパス** し，生 logits を直接 LegalMovesLoss (BCEWithLogitsLoss) に渡す．

---

## 14. 全体まとめ: データフォーマット変換の流れ

### Non-Streaming (通常の Stage2)
```
Stage2Dataset (2-tuple)
  ↓
DataLoader (batch_size=256)
  ↓ (batching via default collate_fn)
Batched 2-tuple: ((board, hand), legal_moves)
  ↓
SingleStageTrainingLoop._train_epoch() (2-element unpack)
  ↓
loss_fn(logits, targets)
```

### Streaming Stage2
```
StreamingStage2Dataset (2-tuple per batch)
  ↓
Stage2StreamingAdapter (3-tuple conversion)
  ↓
((board, hand), (legal_moves, dummy_value, None))
  ↓
DataLoader (batch_size=None)
  ↓ (pass-through)
3-tuple: ((board, hand), (labels_policy, labels_value, legal_move_mask))
  ↓
TrainingLoop._unpack_batch() (3-element unpack)
  ↓
Stage2TrainingLoop._compute_policy_loss()
  ↓
loss_fn(raw_logits, targets)
```

---

## 15. キー要素のまとめ

| 要素 | 説明 |
|------|------|
| **Stage2Dataset** | 2-tuple: `((features), target)` を返す |
| **StreamingStage2Dataset** | 2-tuple バッチを yield (IterableDataset) |
| **Stage2StreamingAdapter** | 2-tuple → 3-tuple への変換層 |
| **DataLoader (通常)** | デフォルト collate_fn で 2-tuple をバッチ化 |
| **DataLoader (ストリーミング)** | batch_size=None で Stage2StreamingAdapter の 3-tuple を通す |
| **TrainingLoop._unpack_batch()** | 3-tuple のアンパック: `(inputs, (labels_policy, labels_value, legal_move_mask))` |
| **Stage2TrainingLoop._compute_policy_loss()** | log_softmax + normalization をスキップ，生 logits を直接渡す |
| **Stage2ModelAdapter** | model(inputs) → (policy, dummy_value) の 2-tuple を返す |
| **SingleStageTrainingLoop** | 2-tuple をアンパック (Stage1/3用) |

---

## 参考ファイル
- `/workspaces/maou/src/maou/app/learning/dataset.py` - Dataset定義
- `/workspaces/maou/src/maou/app/learning/streaming_dataset.py` - StreamingDataset と Adapter
- `/workspaces/maou/src/maou/app/learning/training_loop.py` - TrainingLoop と _unpack_batch
- `/workspaces/maou/src/maou/app/learning/multi_stage_training.py` - Stage2ModelAdapter と run_stage2_with_training_loop
- `/workspaces/maou/src/maou/interface/learn.py` - Stage2 セットアップ
