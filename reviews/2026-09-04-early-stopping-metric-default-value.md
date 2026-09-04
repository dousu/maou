---
status: applied          # pending | approved | applied | rejected
applied_in: 86492370
target: docs/commands/learn_model.md
---

# `--early-stopping-metric` の既定を `value` にする

## Trigger

`worklog/2026-09-04-103853.md`．探索値 28.6% 投与の判定を held-out 8 倍
(6,785 局) で取った際に，**監視指標の取り違えによる差が，教師信号そのものを
変えて得られた差より大きい**ことが測れた．

コード側の既定変更は `feat(learning): --early-stopping-metric の既定を
value にする` で入る (`dl.py` / `interface/learn.py` ×2 / `console/learn_model.py`)．
本提案はそれに伴う `docs/commands/learn_model.md` の記述更新．

## Why

`total` は policy と value の合算だが，**合算は policy に支配される**．
実測 (vit-19.8m / 38.5M 行 / stage3):

| | 値 |
|---|---|
| 検証 total @ ep20 | 2.5680 |
| うち policy CE | 2.0609 (**約 8 割**) |

policy は ep25 まで単調改善を続けるので，合算の最小点は value head の
最小点より **7 epoch 後ろ**へずれた (value 底 ep13 / 合算底 ep20)．

その 2 点を held-out 較正 (2026-03-02〜17 / 6,785 局 / 対局単位 bootstrap) で
測った結果:

| checkpoint | ECE | ΔECE vs 基準 ep11 | 95% CI | |
|---|---|---|---|---|
| 28.6% ep13 (value の底) | **0.0285** | −0.0066 | [−0.0105, +0.0008] | ns |
| 28.6% ep20 (合算の底) | **0.0513** | +0.0162 | [+0.0106, +0.0215] | **有意** |

**同一 run で監視指標を取り違えるだけで ECE が 1.8 倍**になる．これは有意で，
240 時間かけた探索値蓄積が生んだ差 (有意ですらない −0.0066) より大きい．

`total` を既定にしておく限り，較正を追う実験は毎回この差を取り落とす．

## Proposed change

`docs/commands/learn_model.md` の `--early-stopping-metric` 行を差し替える．

**現在**:

```
| `--early-stopping-metric [total\|value\|policy]` | `total` | early stopping と**チェックポイント保存**が監視する検証指標．`total` は合算損失，`value` は Brier score，`policy` は交差エントロピー．2 つの head は過学習の速さが違うため，**合算値の最小はどちらの head の最小とも一致しない** (実測: value は epoch 11 で底，policy は epoch 21 まで改善，合算は epoch 16 で底)．較正を追うときは `value` を選ぶと held-out ECE が下がるが policy を等価分だけ失う．詳細は [docs/design/training-quality/](../design/training-quality/index.md) §5.2． |
```

**変更後**:

```
| `--early-stopping-metric [total\|value\|policy]` | `value` | early stopping と**チェックポイント保存**が監視する検証指標．`value` は Brier score，`policy` は交差エントロピー，`total` は合算損失．2 つの head は過学習の速さが違うため，**合算値の最小はどちらの head の最小とも一致しない**．さらに **`total` は policy に支配される** (実測: 検証 total 2.5680 のうち policy CE が 2.0609 = 約 8 割) ため，policy が単調改善を続ける間は合算の最小点が value の最小点より後ろへずれる (実測で 7 epoch)．その 2 点を held-out 較正で測ると **ECE 0.0285 (value の底) 対 0.0513 (合算の底)** で有意差があり，**監視指標の取り違えによる差のほうが教師信号を変えて得られた差より大きかった**ため，既定を `value` にしている．指し手予測を追うときは `policy`，従来どおりの折衷が要るときは `total` を明示する．詳細は [docs/design/training-quality/](../design/training-quality/index.md) §5.2． |
```

**変更点は 3 つ**:
1. 既定列 `total` → `value`
2. 旧測定値 (value ep11 / policy ep21 / 合算 ep16) を新実測へ更新
3. 「`value` を選ぶと held-out ECE が下がるが **policy を等価分だけ失う**」を削除．
   この等価交換の主張は今回の測定では裏づけられていない (policy 側の損失量を
   測っていない)．測っていない対称性を書かない

## Risk

- **中**: 既定の挙動変更なので，既存のスクリプトが `--early-stopping-metric` を
  明示していない場合に停止 epoch と保存 checkpoint が変わる．3 択は維持するので
  `total` を明示すれば従来どおり．
- compass § VETOES 「レバーは A/B で『より強い』を確認してから既定化」に対しては，
  **user の明示判断** (2026-09-04) を例外条項として適用．今回測ったのは較正 (ECE)
  であって**棋力ではない**点は本文に残さない (docs はコマンドの説明であり，
  campaign の測定履歴は worklog/compass が持つ)．

## Alternatives considered

- **既定を `total` のまま，head ごとの最良 epoch を要約ログに出す** — VETO に触れず
  取り落としを可視化できるが，**毎回人間が読んで選び直す必要が残る**．user が
  既定変更を選択した．
- **`value` 監視時の best checkpoint も併せて保存する** — 追加保存のみで停止挙動は
  不変だが，保存物が倍になりクラウド費用と混乱が増える．
