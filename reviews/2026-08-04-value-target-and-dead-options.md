---
title: value 教師を resultValue 固定にし，デッド CLI オプションを整理する
date: 2026-08-04
status: applied
applied_in: PENDING_SHA
target:
  - docs/adr-005-move-win-rates.md
  - docs/commands/learn_model.md
  - docs/commands/pre_process.md
  - docs/commands/utility_benchmark_training.md
risk: medium
reversibility: easy
---

# 提案: value 教師を `resultValue` 固定にし，未配線オプションを削除する

## 背景

floodgate 棋譜で学習したモデルが将棋モデルとして正しくない挙動を示した
(平手初期局面の勝率 0.71 / 互角の中盤局面で 0.13)．学習経路を監査した結果，
教師信号の構成に構造的な問題が見つかった．

### `max(moveWinRate)` の上方バイアス (winner's curse)

`--value-target-mode best-move-win-rate` は `moveWinRate` の行ごとの最大値を
value 教師にする．**複数のノイズ推定の最大値**なので構造的に上振れする．
真の勝率が全手 0.500 の局面でのシミュレーション:

| 対局数 | 異手数 | value 教師 | 上方バイアス |
|---|---|---|---|
| 1,000 | 30 | 0.620 | +0.120 |
| 10,000 | 30 | 0.619 | +0.119 |
| 50,000 | 30 | 0.589 | +0.089 |

学習済みモデルの平手初期局面の出力は 0.6146 (旧) / 0.7066 (新) で，
この帯と整合する．

### 粒度でも `resultValue` に劣る

「`resultValue` は 0/1 しか返さないので `best-move-win-rate` で粒度を上げる」
という当初の意図は達成されていなかった．出現回数ごとの実測:

```
    出現回数            勝敗の内訳  result-value  bmwr(raw)  bmwr(neutral)
       1             0勝1敗         0.000      0.000          0.500
       1             1勝0敗         1.000      1.000          0.500
       2             1勝1敗         0.500      1.000          0.500
       3             2勝1敗         0.667      0.545          0.545
       5             4勝1敗         0.800      0.615          0.615
     100           51勝49敗         0.510      0.514          0.514
```

- 出現 1 回では両者は**完全に同一** (0.0/1.0)
- 出現 2 回以上では `resultValue` のほうが粒度が高い (Beta 平滑化で圧縮されない)
- **出現 2 回で 1 勝 1 敗 (真値 0.5) が `bmwr(raw-outcome)` では 1.000** になる

### policy 教師の全ゼロ行

`--win-rate-fallback raw-outcome` + `--policy-target-mode win-rate/weighted`
では，手番側が負けた閾値未満局面の `moveWinRate` が全要素 0 になり，
正規化後の policy 教師も行和 0 になる．KLDivLoss はその行に損失を生まないため
**勾配がゼロ**．実データ (floodgate golden 棋譜 342 局面) で **44.2%** が該当した．

`reduction="batchmean"` の分母はバッチサイズのままなので，**報告される
policy loss が死んだ行の分だけ小さく見える** (4/8 行が全ゼロなら 7.311 → 3.655)．
計器が問題を隠していた．

## 変更内容

### コード (src/)

1. **`--value-target-mode` を削除**．value 教師は `resultValue`
   (= `win_count / count`) 固定．`value_targets.py` ごと削除した
2. **`--win-rate-fallback` へ改名** (旧 `--best-move-win-rate-fallback`)．
   選択肢も `uniform` → `neutral` に改めた．`--value-target-mode` の削除で
   このオプションが制御するのは policy 教師 (`moveWinRate`) と可視化用の
   `bestMoveWinRate` 列だけになり，旧名が実態と合わなくなったため．
   `uniform` も 0.5 (中立値) を書くのであって一様分布ではない
3. **検証メトリクス `policy_empty_target_rate` を追加**．policy 教師が
   全ゼロ (= 勾配を生まない) 行の割合を報告する
4. **`pre-process` に警告を追加**．`raw-outcome` 選択時に policy 側の帰結を
   ログに出す
5. **デッドオプション 6 個を削除**: `--gce-parameter` (`setup.py` で破棄されて
   いたが hparams とログには出るので効いているように見えた) /
   `--start-epoch` / `--resume-policy-head-from` / `--resume-value-head-from` /
   `--tensorboard-histogram-frequency` / `--tensorboard-histogram-module`

### ドキュメント

- `docs/adr-005-move-win-rates.md` §5 を「`resultValue` 固定」に改訂し，
  `max(moveWinRate)` を「将来検討」から「却下」へ．上記の実測値を追記
- `docs/commands/learn_model.md` — 削除した 7 オプションの行を除去し，
  changelog に 2026-08-04 の項を追加
- `docs/commands/pre_process.md` — 改名を反映し，`raw-outcome` の policy 副作用と
  `policy_empty_target_rate` での監視方法を明記
- `docs/commands/utility_benchmark_training.md` — 削除したオプションの記述を除去

## 検討したが採用しなかった案

- **HCPE の `eval` 列を value 教師に混ぜる**: 出現 1 回の局面に連続値の粒度を
  与えられる唯一の手段だったが，**floodgate の評価値はプレイしている
  クライアントの仕様に依存する**ため見送り (user 判断)．棋譜ソースを変えると
  学習が変質する信号は教師に入れない
- **`--value-loss-ratio` を上げる**: 損失値の比 (KLDiv 7.31 vs BCE 0.693) から
  提案したが撤回した．損失の大きさは勾配の強さではなく，KLDiv の 7.31 は
  `log(1496)` = ラベル空間の広さを反映しているにすぎない．BCE は 0/1 ラベルの
  エントロピー未満に下がらないので比は学習中に変動する．さらに**観測症状
  (value の過信) と逆向き**で，上げれば過学習が悪化する
- **train/val を対局単位で分割**: 前処理が Zobrist hash で全コーパス横断に
  集約するため，集約後に対局の同一性は復元できない (user 指摘)．前段で
  HCPE を分けてから別々に前処理すれば可能だが，別途の課題とする

## リスク

- **中**: value 教師が変わるため，**前処理データの再生成と再学習が必要**．
  棋力への影響は未測定
- 削除した 6 オプションは指定しても何も起きなかったため，動作の変化は
  「指定するとエラーになる」ことのみ
- `--best-move-win-rate-fallback` の改名と `uniform` → `neutral` は破壊的変更．
  既存スクリプトの書き換えが要る
- CLI 表面の破壊的変更のため minor bump (`maou 0.75.0` → `0.76.0`)．
  0.x では破壊的変更を minor で扱う既存慣行に従う
- 逆行は容易 (いずれも追加し直せる)
