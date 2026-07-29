---
title: warmup を予算の外で前払いする位置づけを設計 doc に明記する
date: 2026-07-29
status: applied
applied_in: 613f5c6
target:
  - docs/design/position-search/index.md
risk: low
reversibility: easy
---

# 提案: 「初回推論の固定費をどこで払うか」を設計 doc に節として起こす

## Trigger

user 指示「searchのときだけwarm_upを予算に入れないようにすることはできますか？
USIエンジンとして稼働する場合やCSAサーバでの対局時はtensorRTエンジンの生成を
予算に入れるのは正しいですが，searchコマンドの時は指定した秒数を探索してほしい
ので別計算にしてほしい」．

実機で踏んだ事象 (Colab L4 / TensorRT cold cache):

```
Bestmove: 1g1f
Eval: -16578.61
Stats: playouts=0 ... elapsed_ms=175 warmup_ms=32227 stop=time_limit
```

`maou search --time-ms 3000` に対しエンジンビルドが 32.2 秒かかり，予算を
使い切って playout が 0 になった．`Bestmove` は合法手生成順の先頭
(`select_best_root_index`，search.rs:349) で，評価に基づく手ではない．

## 現状の設計記述

`docs/design/position-search/index.md` は時間予算の起点について明示的な節を
持たない．実装側 (`rust/maou_search/src/search.rs:1446-1461`) には

> `budget_start` は**呼び出し側が `go` を受け取った時刻**．[...]
> ただし**期限は budget_start 起点**で，warmup も予算に数える．

とあり，回帰テスト `test_wall_clock_stays_within_time_budget`
(search.rs:2542) が「起点が warmup の後だと予算をはみ出す」過去のバグを
記録している．つまり **warmup を予算に数えるのは対局経路にとって必須**で，
これを緩めてはならない．

一方で CLI (`maou search`) は対局経路ではなく計測・検査の道具であり，
「指定した秒数だけ探索する」が期待される挙動．

## 提案する記述 (§5.1 として新設)

> 起票時は §3.5 としていたが，その番号は既存 (木の表現とメモリ)．
> 予算の話なので §5「予算 API と停止」の配下に置く．

> ### 3.5 初回推論の固定費 (warmup) をどこで払うか
>
> ONNX Runtime の TensorRT EP は**最初の推論時に**エンジンをビルドする
> (cold cache で数十秒)．CUDA EP もコンテキスト初期化を初回に行う．
>
> 時間予算の期限は `budget_start` (= 呼び出し側が `go` を受けた時刻) 起点で
> 張る．これは対局経路の要件であり緩めない — GUI/サーバは指し手送信から
> `bestmove` 受信までを消費時間として計るため，予算の外で払った時間は
> そのまま持ち時間の超過になる (§ 回帰テスト
> `test_wall_clock_stays_within_time_budget`)．
>
> したがって**固定費は「1 手ぶんの予算を張る前」に前払いする**．
> `maou_search::warmup` (平手初期局面を 1 回評価) を各経路の適切な位置で
> 呼ぶ:
>
> | 経路 | 前払いの位置 | 根拠 |
> |---|---|---|
> | USI | `isready` 処理中 (`readyok` を返す前) | GUI は `isready` に時間制限を課さない |
> | CSA (floodgate) | 対局ループ開始前 (プロセス内 1 回) | 連続対局で共有 |
> | 自己対局 | driver 起動時 (全対局で共有) | 同上 |
> | CLI `maou search` | 探索開始前 | 予算 = 探索時間．壁時計は warmup + 予算 |
> | `SearchEngine` (棋譜解析) | コンストラクタ | 局面ごとに予算を配分するため，初手だけ削られるのを防ぐ |
>
> **CLI と対局経路で予算の意味が違う**ことは意図的である．CLI は計測・検査の
> 道具で「N ミリ秒*探索*する」が期待される挙動．対局経路は「`go` から N
> ミリ秒以内に `bestmove` を返す」が要件．予算セマンティクス自体
> (`budget_start` 起点) は両者で共通で，違うのは前払いの有無だけ．

## 却下した代替案

- **`SearchLimits` に `exclude_warmup` フラグを足す**: 予算セマンティクスに
  分岐が増え，対局経路で誤って有効化すると切れ負けに直結する．前払いは
  呼び出し側で完結するので探索側に選択肢を持たせる必要がない．
- **`maou search` に `--warmup/--no-warmup` を露出する**: 前払いしない側に
  用途がない (playouts=0 になるだけ)．オプションを増やす価値がない．

## 影響

- 実装は済んでいる (この提案は durable doc への反映のみ)．
- `maou search` の壁時計は `warmup + --time-ms` になる．cold cache の
  TensorRT では初回のみ数十秒伸びる．
- 対局経路 (USI / CSA / 自己対局) の予算挙動は**不変**．
