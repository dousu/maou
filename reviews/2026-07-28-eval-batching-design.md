---
status: pending
date: 2026-07-28
target:
  - docs/design/position-search/eval-batching.md (新規)
  - docs/design/position-search/index.md (§3.3 の「global collector は未決」を更新)
  - docs/design/usi-engine/verification.md (§5 の aggregator 採否を書き換え)
  - docs/commands/search.md (`--pad-buckets` 行の追加)
  - docs/commands/selfplay.md (`--pad-buckets` 行の追加)
risk: medium
reversibility: easy
---

# 提案: 評価バッチング設計 — 「対局間 aggregator」を主目標から降ろし，bit-identical な GPU 浪費除去を先に置く

## Trigger

user 指示「GPU の推論性能を最大限に引き出す aggregator を実装する．**絶対的な指標は
Elo**．GPU 性能を上げた結果ほかが犠牲になって強くならない可能性も考慮せよ」．

着手にあたり `maou_search` の評価経路を読み，CPU (MockEvaluator) で機構を実測した
ところ，**§5.1 が想定していた「対局をまたいで評価要求をまとめれば約 2 倍」という
筋書きは成立しない**ことが分かった．本レビューはその測定と，代わりに取るべき設計を
提案する．

## 1. Elo に効かない経路を先に潰す (user 指摘への回答)

「GPU を速くしても強くならない」には**構造的に 2 つの形**がある．設計はこれを
分離することから始める．

- **罠 1 — 利得が対局間にしか出ない**: 対局をまたいで評価をまとめると self-play の
  スループットは上がるが，**実戦は 1 局しかないので棋力は 1 ミリも上がらない**．
  さらに A/B ハーネスは A と B が同一プロセスで同一 aggregator を共有するため，
  **A/B の Elo 差としても現れない**．得られるのは学習データ生成レートと実験の
  wall clock だけ (= 間接的 Elo)．
- **罠 2 — 対局内で埋めると質が落ちる**: バッチを埋めるには backprop 前により多くの
  葉を集める必要があり，集めた葉はすべて virtual loss を背負った古い統計で選ばれる．
  in-flight 葉 B 個 / 木のノード数 N に対し **B/N が「PUCT が壊れている割合」**で，
  B を上げるほど 1 playout あたりの質が落ちる．

## 2. 測定 (CPU / MockEvaluator / threads=1 起点)

`cargo run --release -p maou_search --example nps_bench`．平手 startpos，
leaf_mate off / root_dfpn off，seed 42．**NPS は探索コアの上限値であり GPU とは
比較しない**．読むのは fill% (`avg_batch ÷ batch_size`) と `eval_batches`
(= GPU 呼び出し回数)．

### 2.1 threads=1 では衝突は起きていない

| playouts(要求) | batch | 実 playouts | avg_batch | fill% | collis% |
|---|---|---|---|---|---|
| 800 | 256 | 1,054 | 210.80 | **82.3** | 0.09 |
| 6,400 | 256 | 6,430 | 247.31 | 96.6 | 0.02 |
| 51,200 | 256 | 51,230 | 254.88 | **99.6** | 0.00 |

800 playouts の 82.3% は `(30 + 256×4) / 5 / 256` に一致する — **最初の 1 バッチだけが
木の広さ (平手 root の合法手 30) で頭打ち**になっているだけで，衝突ではない．
⇒ **「collision がバッチを切って充填を落とす」は threads=1 では棄却**．

副次発見: `--playouts 800 --batch 256` は実際には **1,054 playouts** 走る (+32%)．
バッチを満たしてから予算判定するため．予算の意味に効くので記録する．

### 2.2 threads を増やすと壊れる (`--threads 2` が 4 割遅い理由)

51,200 playouts / batch 256:

| threads | avg_batch | fill% | collis% | **eval_batches** |
|---|---|---|---|---|
| 1 | 254.88 | **99.6** | 0.00 | **201** |
| 2 | 69.18 | **27.0** | 1.42 | **743** (3.7 倍) |
| 4 | 16.79 | **6.6** | 6.56 | **3,052** (15.2 倍) |

threads=2 の衝突は 730 回に対し eval_batches は 743 — **ほぼ 1 バッチにつき 1 回の
衝突**で，`search.rs:1271` の `break` が毎バッチを平均 69 件で打ち切っている．
TensorRT は `pad_to = batch_size` (`backend.rs:49-53` → `onnx.rs:166`) なので
**GPU コストは呼び出し回数に比例**する．これが verification.md §5.1 の
「threads 2 はどの並列度でも約 4 割遅い」の機構．

### 2.3 対局内 aggregator の上限利得は約 2.5% しかない

T スレッドの部分バッチを 1 回にまとめても，得られるのは最良でも threads=1 と
同じ 255 件/回．**aggregator が threads=1 に対して上乗せできるのは，CPU の葉収集と
GPU 推論のオーバーラップ分だけ**である．

- 収集コスト: mock 実測 412k playouts/s ⇒ 256 葉あたり **≤ 0.62 ms**
- GPU バッチ: L4 実測から約 **25 ms**

⇒ **オーバーラップの上限は約 2.5%**．**「aggregator が 1 局の中で 2 倍を出す」は
成立しない**．

## 3. では 2 倍差はどこから来るのか (分解)

self-play 4,400 p/s vs 単発 30 秒 10,010 p/s の差を，コード上の固定費で分解する
(GPU 実測での確認は未了 — §6):

| 要因 | 機構 | 見積 |
|---|---|---|
| **完了検知のポーリング** | `backend.rs:220-230` が `sleep(100ms)` してから `is_finished()` を見る ⇒ 1 手あたり**平均 50ms の死に時間** | 約 1.2× |
| **root 評価が 1 件バッチ** | `search.rs:1489`．TRT では 1 件でも 256 件分のコスト．1 手あたり playout バッチ 4 回 + root 1 回 ⇒ **GPU の約 20% が 1 局面の評価** | 約 1.2× |
| **バッチ充填 82.3%** | §2.1．reroot 直後の最初のバッチが木の広さで頭打ち | 約 1.2× |

いずれも **対局間 aggregator とは無関係**で，かつ **探索の挙動を 1 ビットも変えずに
除去できる**．

## 4. 提案する設計 — 変更を 2 クラスに分ける

| | 検証方法 | 罠 2 の有無 |
|---|---|---|
| **クラス 1: 探索挙動が bit-identical な高速化** | throughput 実測 + bit-identical 回帰テスト．**棋力 A/B 不要** — Elo は doubling 換算で導出 | **構造的に発生しない** |
| **クラス 2: 探索挙動を変えるレバー** | **持ち時間制の Elo A/B が必須**．n の見積りは §5 | 発生する．測って判定 |

### クラス 1 (先に全部やる)

1. **探索完了の通知化** — `backend.rs` の monitor を channel の切断で完了検知し，
   `POLL_INTERVAL` は進捗ポーリング間隔としてのみ使う．observer の駆動間隔は不変．
2. **`pad_to` のバケット化** — 実バッチ長を次のバケット (例 1/8/32/64/128/256) へ
   切り上げる．root 1 件評価と部分バッチの浪費が消える．
3. (2 の後) **root 評価を最初の playout バッチに相乗り**させる余地の検討．

### クラス 2 (クラス 1 の結果を見てから採否)

4. **バッチサイズの動的化** (user 提案) — B を木のノード数 N に比例させ B/N を一定に
   保つ．**これは速度レバーではなく質レバー**である．`pad_to` 固定の下では
   「呼び出し回数が増えてコストが増える」だけなので，**2 が入るまで動かせない**
   (user 指摘のとおり)．バケット集合にスナップさせれば TRT の shape 数も増えない．
5. **衝突時の収集継続** (`search.rs:1271` の `break` 撤廃) と **対局内/対局間
   aggregator** — 2 と 4 の後に，まだ充填が足りない場合のみ．

**対局間 aggregator は「self-play データ生成レートと実験 wall clock の改善」として
別枠で評価する**．棋力目標の主経路から降ろす．

## 5. 検証プロトコル (罠 2 を検出できる形)

クラス 2 のレバーは以下 3 点セットで測る．E1 を省くと「速くなったが強くなっていない」
の原因が質の劣化か速度不足か区別できない (compass: 区別できないものに名前を付けない)．

| | 設定 | 測るもの |
|---|---|---|
| E1 | **等 playout** A/B | **質の劣化のみ** (速度は無関係)．期待 ≤ 0 |
| E2 | **等時間** A/B | **正味の Elo** = 速度利得 − 質劣化 ← 判定はこれ |
| E3 | throughput 比 → 208 Elo/doubling で予測 | E2 実測と突き合わせて換算モデル自体を検証 |

- **A/B は `--parallel 1` で回す**．並列対局にすると 1 手の wall clock が他対局の
  GPU 負荷に依存し，持ち時間制の測定が汚染される．
- **n の見積り**: n=40 の検出限界は約 150 Elo (compass TRIPWIRE)．±50 Elo を
  95% で分離するには**ペア局面込みでも n ≈ 90-190 局**が要る．クラス 2 に着手する
  前に，この対局数を払う価値があるかを期待効果から判断する．

## 6. 本レビュー時点で**検証できていない**こと (誠実な限界)

- **§3 の分解はすべて GPU 未確認**．CPU の mock 測定は fill/衝突/呼び出し回数という
  **構造量**しか保証しない．
- **棋力の主張は一切していない**．MockEvaluator は prior/value が擬似乱数で，
  探索が収束すべき信号が存在しないため，**bestmove 一致率は質の証拠にならない**．
  ローカル CPU は 23 p/s なので実ネットでの Elo 測定は原理的に不可能．
- **TRT の shape ごとのエンジン構築コストとキャッシュ挙動は未測定** (バケット数だけ
  engine が要る)．
- **バケット化で出力が bit-identical のままかは未確認**．現在の「並列度を変えても
  bit-identical」(§5.1) は **`pad_to` 固定で shape が常に同一**であることに依存して
  いる可能性があり，バケット化はこの再現性を壊し得る．**これはバケット化採用の
  前提条件として GPU で先に確認する**．

## docs/commands への追記内容 (承認時にそのまま適用)

`docs/commands/search.md` の `--trt-cache-dir` 行の直後，および
`docs/commands/selfplay.md` の同位置へ:

```
| `--pad-buckets` / `--no-pad-buckets` | `--no-pad-buckets` | TensorRT の padding を `--batch-size` 固定でなく 2 冪バケットへ切り上げる．固定 padding では 1 件の root 評価も 1 バッチ分の推論コストを払うが，バケット化は shape ごとにエンジンビルドが増え，数値が変わり得る (計測用トグル)． |
```

## 影響

- docs/design/position-search/eval-batching.md を新規作成 (本設計の置き場)．
- index.md §3.3 の「将来: global collector (未決 — per-thread で足りるか実測で判断)」
  を，本測定の結論 (対局内利得 ≤ 約 2.5%) で更新．
- verification.md §5 の「aggregator は次 campaign の課題」を，分解結果と
  クラス 1/2 の切り分けで書き換え．

## 代替案と却下理由

- **予定どおり対局間 aggregator から着手する**: 却下．§2.3 の測定により棋力への
  寄与が構造的に無いことが分かったため．self-play 高速化としては後で再評価する．
- **`Evaluator` trait を submit/await に作り替える**: 却下 (現時点)．`search.rs` /
  `Shared` / ライフタイムに広く波及するのに，得られる上限が §2.3 の約 2.5%．
  クラス 1 を入れた後の再測定で必要性が示されてからにする．
