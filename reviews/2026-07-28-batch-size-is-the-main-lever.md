---
status: applied
applied_in: 292fbd2, 01e1ba4
date: 2026-07-28
target:
  - docs/design/position-search/eval-batching.md (§3-§5 を GPU 実測で全面改訂)
  - docs/design/usi-engine/verification.md (§5 の分解を訂正 + batch_size A/B 手順を追加
    + 手順中の `--batch-size 256` を 64 へ，10 箇所)
  - docs/commands/selfplay.md (`--ab-mode batch` / `--batch-size-b` の追加 + 推奨値)
  - docs/commands/search.md (推奨値)
  - src/maou/infra/console/{selfplay,search_board,usi}.py (`--batch-size` の help
    "use around 256 on GPU" → 64．**既定値 8 は据え置き**)
  - docs/design/position-search/benchmarking.md (North-star 計測手順の
    `--batch-size 256` → 64．**user 追加指示 2026-07-28**)
  - docs/performance.md は**対象外** — `benchmark-dataloader` /
    `benchmark-training` の `--batch-size 256` は**学習ミニバッチ**であり
    評価バッチサイズとは別概念 (同様に utility_benchmark_training.md も対象外)
risk: medium
reversibility: easy
---

# 提案: GPU 実測に基づく eval-batching 設計の訂正 — 主レバーは `--batch-size` だった

## Trigger

PR #412 マージ後の wheel で GPU 検証 (①-B + 持ち時間スイープ) を実施した．
**CPU (MockEvaluator) で立てた仮説のうち 3 つが実測で覆り**，同時に
**既定の `--batch-size 256` が最適でない**ことが分かった．
eval-batching.md は現状のままだと誤った結論を残すため，全面的に訂正する．

## 1. 覆った仮説 (3 件)

### 1.1 「threads=1 では collision がバッチ充填を壊さない」→ **誤り**

CPU では 800 playouts / batch 256 で fill 82.3% / collisions 0.09% だったが，
**GPU (実 NN) では fill 35.1% / collisions 9 回**だった．

数字は 1 つの機構を指す — 「最初の 9 バッチが約 71 件で打ち切られ，以降は満杯」:

- 800 playouts: `9×71.3 + 256 = 897.7` → 実測 **898**
- 6,400 playouts: `9×71.3 + 23×256 = 6,529.7` → 実測 **6,530**

**原因は mock の prior がほぼ均一なこと**．実 NN の prior は鋭く
(2g2f 0.413 / 7g7f 0.237)，選択が上位手に集中するため同じ葉へ何度も降りる．
⇒ **「構造量だから mock でも転移する」という判断が誤りだった**．転移したのは
fill の**定常値** (長い探索で 99.5%) だけで，**立ち上がりは転移しない**．

### 1.2 「GPU 呼び出しに固定費がある」→ **誤り** (中間で立てて撤回)

800 playouts の 2 点から `cost(n) ≈ 10.3 + 0.052n` (固定費 10ms) と推定したが，
batch_size を振った直接測定で否定された:

| batch (= padded) | ms/call |
|---|---|
| 8 | 1.13 |
| 32 | 2.85 |
| 64 | 5.39 |
| 128 | 10.95 |
| 256 | 25.47 |

```
cost(n) ≈ 0.15 + 0.084·n  [ms]     ← 固定費はほぼゼロ
```

① の長い探索 (`cost(256) = 24.05`) とも整合する．
⇒ **「呼び出し回数を減らすことが最重要」という中間結論も撤回**．
固定費が無い以上，効くのは **padding を減らすことだけ**である．

### 1.3 「対局間 aggregator の上限利得は約 2.5%」→ **根拠が誤り**

§3.3 は「threads=1 の fill が 99.6% だから束ねる余地がない」と論じたが，
その 99.6% は 51,200 playouts の**定常値**であり，実配置の短い探索では
fill 31-55% である．結論 (棋力には効かない = 罠 1) は変わらないが，
**理由が違う**ので書き直す．

## 2. 主レバーは `--batch-size` だった

### 2.1 単発 6,400 playouts (`maou search`)

| batch | fill | nps |
|---|---|---|
| 8 | 100% | 7,096 |
| 32 | 99% | 11,131 |
| **64** | 97% | **11,459** |
| 128 | 91% | 10,685 |
| 256 | 80% | 8,004 |

### 2.2 持ち時間モード 30s+0.5s (各 4 局)

| batch | throughput | vs 256 |
|---|---|---|
| 32 | 9,918 p/s | +29.7% |
| **64** | **10,257 p/s** | **+34.1%** |
| 128 | 10,039 p/s | +31.3% |
| 256 | 7,646 p/s | — |

**独立な 2 つの regime で batch 64 が最速**，既定の 256 は **34-43% 遅い**．
32/64/128 はプラトー (±2%) で 256 だけが落ちる．
同一 session の再現性も確認済み (このスイープの batch 256 = 7,646 p/s と
③ の持ち時間 8 局 batch 256 = 7,557 p/s が 1.2% 差)．

機構は §1.2 の cost モデルで説明できる — コストが padded 長に比例するので
**大きいバッチにしても GPU 効率は上がらず，充填しきれない分の padding を
捨てるだけ**になる (256 で fill 80%，64 で 97%)．

### 2.3 速度と質が同じ方向を向いている

`maou search` 6,400 playouts の root visits:

| batch | 2g2f | 7g7f |
|---|---|---|
| 8 | 4,556 | 969 |
| 256 | 3,999 | 1,527 |

batch を下げるほど **top-1 に集中**する = in-flight 葉が減って virtual loss の
歪みが小さい．CPU の mock 実験で見た平坦化が実 NN で再現した．
⇒ **罠 2 (GPU を速くして質を犠牲にする) が発生しない稀なケース**．

## 3. `--pad-buckets` の位置づけ変更

`--pad-buckets` は GPU で **+34.2%** (batch 256 / 800 playouts) を確認したが，
**batch 64 の上位互換にはならない**:

| | `--pad-buckets` (batch 256) | `--batch-size 64` |
|---|---|---|
| 速度 | +34% | +34% (同等) |
| 質 | 変わらず | **向上** |
| bit-identical | **壊れる** | **保たれる** |
| TRT エンジン | 9 個 + warmup 事前ビルド実装が必要 | 1 個 |

batch 64 では fill 97% でバケット化の余地がほぼ無い．
⇒ **`--pad-buckets` は既定化せず計測用トグルのまま残す**．

### bit-identical が壊れる証拠 (記録)

同一局面・同一モデルで `pad_to` の shape が変わると root 評価の出力が変わる:

| 手 | `--no-pad-buckets` (padded 256) | `--pad-buckets` (padded 1) |
|---|---|---|
| 2g2f | 0.4132 | **0.4151** |
| 7g7f | 0.2373 | **0.2346** |

PV も 7 手目から分岐した．**compass の invariant「並列度を変えても
bit-identical」は `pad_to` 固定で shape が常に同一であることに依存していた**．

## 4. クラス 1 の累計 (持ち時間モード = 実戦 regime)

| | throughput |
|---|---|
| §4.6 (改善前, batch 256) | 7,006.5 p/s |
| + 完了通知化 (de15d54) | 7,557 (+7.9%) |
| + batch 64 | **10,257 (+46.4% 累計)** |

**1 doubling ≈ +140 Elo [+51, +229]** (n=24 で更新済．旧 +208 [+52, +364] は
使わない) を当てると **+77 Elo 級 [+28, +126]**．

なお batch 64 の実測 Elo は **+137** (§5.1) で，この throughput 換算 (+59 Elo) の
2.3 倍だった — 速度だけでなく **1 playout あたりの質**も上がっているため．
throughput からの換算は棋力の**下界**として扱うのが妥当である．

## 5. 提案する次のステップ — `--ab-mode batch` で A/B を回す

compass VETO 「レバーは『より強い』ことを A/B で確認してから既定化」に従い，
**既定値は変更せず**，A/B のための計測手段だけを実装する (本 PR)．

- `AbMode::Batch` を追加．A = `--batch-size` / B = `--batch-size-b`
  (既定 A の 4 倍)．差分は `batch_size` ただ 1 つ
- **持ち時間モードで回すこと** — 固定 playout 予算では速度差が棋力差にならない
- `--ab-mode` の選択肢は **Rust `AbMode` / Python `AB_MODES` / click `Choice`
  の 3 箇所**にあり，過去に片方だけ足して踏んでいる．3 箇所の一致を pin する
  回帰テスト `tests/maou/app/usi/test_ab_modes.py` を追加した

n の見積り (実行前): +59 Elo を 95% で検出するには **n ≈ 140 局**と見ていたが，
実際は効果が大きく **n=48 で t = +3.19** と決着した (§5.1)．

## 5.1 A/B 実測結果 (2026-07-28, 48 局) — **A = batch 64 の勝ち**

`--ab-mode batch --batch-size 64 --batch-size-b 256 --clock-ms 30000 --inc-ms 500
--max-moves 256 --opening-random-plies 8 --seed 1 --alternate-colors --parallel 1
--threads 1 --no-root-dfpn --no-leaf-mate`:

```
A result: 33W 0D 15L / A score: 68.8% (Wilson 95% CI [54.7%, 80.1%])
A Elo: +137 [+33, +241]
paired: 24 pairs, mean +0.375 (SE 0.118), t = +3.19, A ahead in 10/24 (tied 13)
time left at end (avg): A 5.5s / B 5.4s
```

**t = +3.19 / CI 下限 +33** でゼロを明確に超える．色バランス (先手 23 / 後手 25) も
残り持ち時間 (A 5.5s / B 5.4s) も公平で，n=48 で決着した．

**予測 (+59 Elo) を 2.3 倍上回った**．予測は throughput +34.1% を換算しただけの値で，
差分の **+78 Elo 相当は 1 playout あたりの質の向上**と解釈できる — batch を下げると
in-flight 葉が減って virtual loss の歪みが小さくなる (§2.3 の visits 分布) ことが，
棋力として実測された．**罠 2 の逆**が起きており，質の寄与のほうが速度の寄与より
大きい可能性がある．

なお **1 doubling あたり Elo は n=24 で +140 [+51, +229] に更新済** (旧 +208 [+52,
+364])．本レビュー中の Elo 換算はすべて 140 で計算し直してある．

### 5.1.1 追試 A/B (2026-07-28, 48 局): A = batch 32 / B = batch 64

`--batch-size 32 --batch-size-b 64 --seed 2` (他は §5.1 と同一):

```
A result: 22W 1D 25L / A score: 46.9% (Wilson 95% CI [33.5%, 60.7%])
A Elo: -22 [-119, +75]
paired: 24 pairs, mean -0.062 (SE 0.142), t = -0.44, A ahead in 5/24 (tied 12)
time left at end (avg): A 6.0s / B 5.6s
```

**有意差なし** (t = −0.44，CI が 0 をまたぐ)．点推定はわずかに 64 側．
n=48 の分解能は ±75 Elo なので「同等」と断定はできないが，
**推奨値を 64 とする判断には十分**である (速度も 64 が最速)．

### 5.1.2 機構: batch_size は「充填が飽和する最小値」が最適

2 本の A/B と fill を重ねると説明が閉じる:

| batch | fill | 状態 |
|---|---|---|
| 8 | 100% | 充填は満点だが **GPU が遊ぶ** (7,096 nps) |
| 32 / **64** | 99% / 97% | **飽和点 = 最適**．両者に棋力差なし |
| 128 | 91% | わずかに padding 損 |
| 256 | 80% | **padding 損が大きい** → 64 に −137 Elo |

256 が悪いのは padding 損が主因で，64 以下では fill が 97-100% なので損が無く，
質の差も出ない．**プラトーの内側では速度も質も飽和している**．

⇒ **一般則: `batch_size` は「充填が飽和する最小値」に置く**．それより大きいと
padding を捨て，小さいと GPU が遊ぶ．GPU/TRT + ViT 19.8M fp16 / L4 では **64**．

## 5.2 変更対象は「既定値」ではなく「推奨値」だった

調査の結果，**コードの既定は 8** (`EngineConfig::default()` / CLI `default=8`) で，
256 は **CLI ヘルプの推奨文と docs の手順 10 箇所**が指定している値だった．

| | 現状 | 実測での位置 |
|---|---|---|
| コード既定 | 8 | GPU では 64 比 **−38%** (7,096 vs 11,459 nps)．**CPU では未測定** |
| 推奨値 (help + docs) | 256 | **A/B で 64 に −137 Elo と判定** |
| 実測最適 | — | **64** (32/128 とはプラトー内) |

**提案: 推奨値を 256 → 64 に直し，コード既定 8 は据え置く．**

既定 8 を 64 へ変えない理由: CPU は TRT の padding 損が無くコストが実 items に比例
するため，速度メリットが無いまま質だけ落とす可能性がある．**CPU-only 動作は compass
VETO の回帰条件**であり，未測定のまま既定を GPU 前提の値へ倒すべきでない．
GPU 利用者は `--tensorrt --cuda --model-path` を明示する必要がある以上，
`--batch-size` も明示する運用で一貫する．

## 6. 未検証事項

- **持ち時間モードのスイープは各 4 局で交絡がある** — plies が 462〜621 と
  ばらつき，対局の長さ (局面の性質) が throughput に混ざっている．
  32/64/128 が固まり 256 だけ外れるパターンは一貫しているが，n は小さい
- **32 vs 64 は n=48 で分離できなかった** (§5.1.1，t = −0.44)．分解能は
  ±75 Elo なので「同等」と断定はできない．推奨値 64 の選択には影響しない
- **CPU での最適 batch_size は未測定**．既定 8 を据え置く根拠は「TRT の padding
  損が無いので速度メリットが無い」という機構的推論であって実測ではない
- **質の向上は visits 分布からの示唆にとどまる**．等 playout A/B (E1) で
  測っていない
- **Elo 換算係数 208 が n=12** — task #2 (n ≥ 24) が未了

## docs/commands/selfplay.md への追記内容 (承認時にそのまま適用)

`--playouts-b` 行の直後へ:

```
| `--batch-size-b N` | `--batch-size` × 4 | Evaluation batch size for player B (`--ab-mode batch`). |
```

`--ab-mode` 行の選択肢に `batch` を追加し，説明へ:
「`batch` = evaluation batch size; run it under the clock, since a fixed
playout budget turns speed into no strength difference」

## 代替案と却下理由

- **A/B なしで既定値を 64 に変更する**: 却下．速度と質が同方向で棋力向上は
  ほぼ確実だが，compass VETO が「A/B で確認してから既定化」を求めている．
  user の選択も A/B 実施 (選択肢 1)．
- **`--pad-buckets` を既定化する**: 却下．batch 64 で同等の速度が
  再現性を失わず得られるため，bit-identical を捨てる理由がない．
- **`--batch-size` を動的化する (木のサイズに比例)**: 保留．固定値 64 で
  プラトーに乗っているので，動的化の上積みは小さい見込み．batch A/B の
  結果を見てから判断する．
