---
status: applied
applied_in: 03b61ad
date: 2026-08-09
target: [docs/loss-functions.md, docs/rust-backend.md, docs/adr-001-dataloader-multiprocessing-optimization.md]
risk: low
reversibility: trivial
---

# `legal_move_mask` を targets タプルから外したことによる文書ドリフト

## Trigger

`/audit-backlog` の T3-2 (`audits/coverage.md` Deferred backlog,
[2026-08-08 app/learning](../audits/2026-08-08-src-maou-app-learning.md)
Deferred 8) の修正．`KifDataset` / `StreamingKifDataset` が
`legal_move_mask` を yield するのをやめ，`TrainingLoop._unpack_batch()`
が受け取る targets タプルの並びが
`(labels_policy, labels_value, move_win_rate)` になった．

これで 3 つの文書が実装と食い違う．いずれも**修正前の動作を正確に
説明していた**文書であり，コードが動いた瞬間に誤りになった類の
ドリフトである．

## Proposed change

### 1. `docs/loss-functions.md`

#### 1-a. L112-114

before:
```markdown
前処理パイプラインで棋譜から出現率マップ(ソフトターゲット)を計算し，
確率分布として保存する．`normalize_policy_targets` で
`legal_move_mask` を適用した上で確率分布に正規化する
(ただし現行のマスクはダミーであり実質的な絞り込みは起きない．後述)．
```

after:
```markdown
前処理パイプラインで棋譜から出現率マップ(ソフトターゲット)を計算し，
確率分布として保存する．`normalize_policy_targets` は
`legal_move_mask` を受け取れるが，現行のデータ経路はマスクを
供給しないため素の正規化になる(後述)．
```

#### 1-b. L129-149 (「合法手マスキング — 機構はあるが**現在は発動していない**」節)

before:
```markdown
しかし **Stage 3 に供給される `legal_move_mask` は全要素 1 のダミー**である
(`dataset.py` / `streaming_dataset.py` の `torch.ones_like`)．前処理出力スキーマに
合法手の情報が無いためで，結果として:

- `masked_fill` は恒等変換になり，`log_softmax` は **1496 次元全体**で正規化される
- `normalize_policy_targets` の `targets * mask` も 1 倍で無変更
- **`legal_move_mask=None` を渡した場合と勾配まで完全に一致する**

つまりこの節が想定していた「有効次元が1496→~20に縮小される」効果は**得られていない**．
モデルは非合法手のlogitsを押し下げる学習に勾配を費やしている．
```

after:
```markdown
しかし **どのデータ経路も `legal_move_mask` を供給しない**．前処理出力
スキーマに合法手の情報が無いためで，`TrainingLoop._unpack_batch()` は
`legal_move_mask=None` を立てる．結果として:

- `masked_fill` の分岐に入らず，`log_softmax` は **1496 次元全体**で正規化される
- `normalize_policy_targets` はマスク無しで正規化する
- 上のマスキング経路は丸ごと休眠している

つまりこの節が想定していた「有効次元が1496→~20に縮小される」効果は**得られていない**．
モデルは非合法手のlogitsを押し下げる学習に勾配を費やしている．

2026-08-09 まで `dataset.py` / `streaming_dataset.py` は
`torch.ones_like(moveLabel)` の全 1 マスクを実際に作って targets に
入れていた．勾配は `None` の場合と完全に一致する一方，バッチ毎に
moveLabel と同じサイズ (B=1024 で約 9MB) を PCIe 上に流し，消費側の
5 つのカーネルを no-op として通していたため，データ側からは外した．
`TrainingContext.legal_move_mask` と `_compute_policy_loss` の
マスク分岐は，本物の合法手マスクを流す経路が将来できたときのために
残してある．
```

#### 1-c. L199

before:
```markdown
- Stage 3: Policy損失に合法手マスキングを追加(デフォルトで有効，`legal_move_mask` がない場合は従来動作)
```

after:
```markdown
- Stage 3: Policy損失に合法手マスキング機構を追加(現行データ経路はマスクを供給しないため従来動作)
```

### 2. `docs/rust-backend.md` L768-773

before:
```python
# Training loop works identically
for features, targets in dataloader:
    board, pieces = features
    # moveWinRate 列があれば 4 要素 (旧データでは 3 要素)
    move_label, result_value, legal_move_mask, *rest = targets
    move_win_rate = rest[0] if rest else None
    # ... training code ...
```

after:
```python
# Training loop works identically
for features, targets in dataloader:
    board, pieces = features
    # moveWinRate 列があれば 3 要素 (無ければ 2 要素)
    move_label, result_value, *rest = targets
    move_win_rate = rest[0] if rest else None
    # ... training code ...
```

### 3. `docs/adr-001-dataloader-multiprocessing-optimization.md` L124-137

ADR は決定時点の記録なので本文は書き換えず，コード例が現行の契約と
違うことだけを直後に注記する．

before:
```markdown
    # GPU上でモデル実行
    outputs_policy, outputs_value = model(inputs)
    # 学習処理...
```

after:
```markdown
    # GPU上でモデル実行
    outputs_policy, outputs_value = model(inputs)
    # 学習処理...
```

> **注 (2026-08-09)**: 上のコード例は当時の targets タプルである．
> 現在 `legal_move_mask` はどのデータ経路も供給せず，targets は
> `(labels_policy, labels_value, move_win_rate)` (3 要素目は省略可)
> になっている．経緯は [loss-functions.md](loss-functions.md) の
> 「合法手マスキング」節を参照．本 ADR の主題である
> DataLoader/GPU 転送の決定自体は有効である．

## Motivation

この 3 箇所はいずれも「修正前の動作を正しく説明していた」文書である．
特に `loss-functions.md` L139 の「全要素 1 のダミー」は，まさに今回
消した `torch.ones_like` を指しており，コードが変わった瞬間に事実で
なくなる．`rust-backend.md` のコード例は読者がそのままコピーする
入口 (`PolarsDataFrameSource` を文書化している唯一の場所) なので，
放置すると `move_win_rate` を `legal_move_mask` として受け取る —
これは `_unpack_batch` で実際に踏みかけた罠と同じ形である．

## Alternatives considered

1. **文書を触らない．**
   コードだけ直して doc は次の `/audit-and-fix docs` に任せる．却下:
   `rust-backend.md` のコード例は公開 API の唯一の使用例で，誤った
   まま残すと読者が壊れたアンパックを書く．しかも本 run の変更が
   原因なので，同じ run で直すのが筋である．
2. **ADR も含めて全部書き換える．**
   却下: ADR は決定時点の記録であり，過去の記述を現在形に書き換える
   と「その決定が何を前提にしていたか」が失われる．注記に留める．
3. **`loss-functions.md` の「合法手マスキング」節ごと削除する．**
   却下: マスキング機構はコード上に残しており，本物の合法手マスクを
   流す経路ができたときに読む文書である．削除すると機構が存在する
   ことすら分からなくなる．

## What this enables

`legal_move_mask` の現況 (「全 1 ダミーが流れている」ではなく
「そもそも供給されない」) が 1 箇所で読めるようになる．将来
本物のマスクを実装する人が，何を再接続すればよいか
(`_unpack_batch` の targets 契約と `_compute_policy_loss` の分岐)
を文書から辿れる．

## What this constrains

targets タプルの並びを再び変えるときは，この 3 文書 (+ ADR 注記) を
同時に直す必要がある．`docs/rust-backend.md` のコード例は
`PolarsDataFrameSource` の実質的な契約テストでもあるので，
`tests/maou/app/learning/test_polars_datasource.py` と食い違わせては
ならない．

## Rollback plan

3 ファイルの該当箇所を revert するだけ．コードとの整合は失われるが，
文書のみの変更なので実行時の影響はない．コード側を戻す場合は
`dataset.py` / `streaming_dataset.py` の `torch.ones_like` と
`_unpack_batch` の index 2 の解釈を同時に戻す必要がある
(片方だけ戻すと `move_win_rate` がマスクとして読まれる)．
