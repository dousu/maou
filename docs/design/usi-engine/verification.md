# USI エンジンの検証手順 (GPU / GUI)

[設計本体](index.md) の未決事項のうち，**DevContainer (CPU) では原理的に
閉じられないもの**の手順書．CPU で閉じた項目の根拠は index.md §12 を見ること．

| 残件 | 必要な環境 | 状態 | 手順 |
|---|---|---|---|
| 未決 1 TimeStrategy の定数 | GPU (探索速度が要る) | **決着 2026-07-27** (horizon 40 据え置き) | [§4](#4-未決-1-timestrategy-の想定残り手数) |
| 未決 5 バッチ aggregator | GPU | **決着 2026-07-28** (棋力の主経路ではない．主レバーは `--batch-size` で，64 が 256 に +137 Elo) | [§5](#5-未決-5-バッチ-aggregator-の採否) |
| 未決 2 keep-alive の既定値 | GUI 実機 | **未実施 (将来課題)** | [§8](#8-gui-実機検証-将来課題-未実施) |
| 空回りの予算開放 | CPU で判定可 | **棄却 2026-07-27** (実 playout +1.34%) | [§4.5](#45-空回りの予算開放---ab-mode-spin--cpu-スクリーニングで棄却) |
| 確定済み子の選択除外 | GPU (発火量) | **棄却 2026-07-27** (実探索 +1.8% / throughput +0.35%) | [§4.6](#46-確定済み子の選択除外---ab-mode-proven--gpu-発火測定で棄却) |

手順は再実行できる形で残してある (別モデル・別 GPU で測り直すときに使う)．

**GPU 環境は Colab (L4) のみ**という前提で，検証は **Release `latest` の
事前ビルド wheel** で行う (Rust ツールチェイン不要)．A/B ハーネスは
`maou selfplay --ab-mode` として wheel に入っている (Rust example
`selfplay_ab` は同じ `maou_usi::ab` を呼ぶ薄いラッパーで，数値の定義は同一)．

---

## 1. 環境構築 (Colab / 事前ビルド wheel)

wheel の取得と provider の解決 (`ldconfig`) は
[docs/design/position-search/benchmarking.md §4](../position-search/benchmarking.md)
の手順をそのまま使う (セル 0-2)．`maou search` ではなく `maou selfplay` /
`maou-usi` を叩く点だけが違う．

検証モデルは学習済みの
**`model_20260725_044443_vit-19.8m_32_fp16.onnx`** (ViT 19.8M / fp16 / IR 9)
を `/content/model_fp16.onnx` へ置く．棋力に依存する判定 (§3・§4・§6) は
mock 評価器や極小モデルでは意味を持たない．

GPU 実行の共通フラグ:

```bash
--model-path /content/model_fp16.onnx \
--tensorrt --cuda --threads 1 --batch-size 64 \
--trt-cache-dir /content/trt_cache
```

- **wheel を入れ替えるときは `--force-reinstall --no-deps` を併用する**．
  Rust のみの修正では `pyproject.toml` の版数が動かないため，Release `latest`
  が更新されても pip が「同版数」と見なして**入れ替えないことがある**．
  修正前バイナリで計測して誤結論を出すのを防ぐため，dfpn の挙動が関わる計測
  では先に preflight を通すこと (モデル不要):

  ```
  position sfen 1g1+N+N1+P1l/4+B4/4Np+P2/l1p1p1ppk/3PsnP1p/+r4P3/1pPG3PL/p2S1SGbP/SRK6 b GLPp 141
  go mate 10000
  ```

  f967499 以降は `checkmate 5b4a 1d1c L*1d` を返す．`checkmate timeout` が
  返ったら**修正前の wheel** なので入れ替えをやり直す (この局面は
  `test_no_false_proof_when_check_capture_clears_nifu` で pin 済み)．
- **`--threads 1` が最適** (実測 2026-07-27): threads 2 はどの並列度でも
  約 4 割遅い．1 手 800 playouts 級の短い探索では探索内スレッド並列が
  オーバーヘッドにしかならない (単発 30 秒探索のような長い探索でのみ有効)．
- 実効速度 (L4 / ViT 19.8M fp16 / threads 1): **単発 30 秒 = 10,909
  playouts/秒，500ms = 8,101 playouts/秒**．短い探索ほど遅いのは木が育つ
  まで評価バッチが埋まらないため．
- 単発 30 秒は **約 10,000-10,900 playouts/秒**．同一 session 内で 4 連続
  (threads 2 / 1 各 2 回) 回すと **10,010-10,096 = ±0.4% で安定**し，
  `--threads 1` と `2` の差も 0.5% 以内 (**長い単発探索では 2 スレッド目が
  効かない** — 自己対局の短い探索で threads 2 が約 4 割遅いのはバッチが
  埋まらない regime の話)．
- 一度だけ **4,595 playouts/秒 (2.2 倍低下)** を観測したが**再現しなかった**
  (一過性)．会計の水増しではない (空回りは 0.008% = 303,171 playouts に対し
  24 件，`elapsed_ms` も 30,031 / 30,065 でほぼ同一)．当時は
  **GPU 側か バッチ充填率かを区別できなかった**:
  TensorRT は `pad_to = batch_size` で毎バッチを padding するため，
  **充填率が下がると GPU の仕事量が同じでも `playouts`/`nps` が下がる**．
  ⇒ 切り分けのため `maou search` の Stats に `eval_batches` / `avg_batch` /
  `collisions` を出すようにした (`maou 0.60.2`)．**外れ値を見たときは
  `avg_batch ÷ --batch-size` (充填率) と `collisions` を必ず併記すること**
  — 充填率が一定なら GPU 側，落ちていれば充填側 (`--threads` を上げたときの
  衝突が主因になりやすい)．
  ⇒ いずれにせよ **絶対値を session をまたいで比較してはいけない．throughput 比を
  出す計測では，比較対象の天井 (単発 30 秒) を同じ session 内で測り直すこと**．
  **物理上限のゲート値としては約 11,000 を使う** (§4.2 の水増し検知 — 上限側の
  ゲートなので session 変動に影響されない)．
- **TensorRT の要否 (実測 2026-07-27，同一 session 内で連続測定)**:

  | regime | TRT+CUDA | CUDA のみ | 判定 |
  |---|---|---|---|
  | 単発 30 秒 (バッチが埋まる) | **4,595 p/s** | 2,901 p/s | TRT が **1.58 倍**速い |
  | 自己対局 800 playouts/手 (埋まらない) | 2,153.9 p/s | **2,292.9 p/s** | CUDA が 6.5% 速い |

  TRT は固定 shape が要るため毎バッチを `batch_size` へ padding する
  (`pad_to = batch_size`)．短い探索ではバッチが埋まらず padding が無駄になるが，
  **バッチが埋まる regime では TRT が明確に速い**ので，実配置 (1 手数千 playouts)
  と aggregator の将来像では **TRT を維持する**．自己対局の 2 run は EP 由来の
  数値差で局が分岐するため 6.5% は参考値．

- TensorRT の初回エンジンビルドは **バッチ shape ごとに数十秒〜数分**．
  `--trt-cache-dir` を必ず指定し，同じセッション内で使い回す．
- **計測の前にキャッシュを温める** (§2 の smoke を先に 1 回通す)．初回
  ビルドが計測区間に入ると playouts/秒 が過小に出る．

## 2. 事前確認 — 探索速度の実測 (以降の設定はこの値から決める)

```python
# 1 局面探索の NPS (warmup はエンジンビルドを計測区間から外す)
!maou search --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --time-ms 30000 --root-dfpn \
    --trt-cache-dir /content/trt_cache
```

```python
# 自己対局 1 局の smoke (対局経路が GPU で通ることの確認 + TRT キャッシュ温め)
!maou selfplay --games 1 --playouts 800 --max-moves 64 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache
```

得られた **playouts/秒 (以下 `NPS`)** を控える．CPU (DevContainer) の実測は
ViT 19.8M / 1 スレッド / batch 8 で **約 23 playouts/秒**で，これが未決 1 を
CPU で閉じられない理由だった．

## 3. ハーネスの再較正 (`--ab-mode budget`)

**レバーの A/B より先に通す健全性確認**．予算の多い側が勝たなければ，この
driver で棋力差を測ること自体が成立していない．同時に **GPU の予算域での
1 doubling あたり Elo** を測り直す (CPU で得た「1 doubling ≈ 208 Elo」は
16→64 playouts の極低予算域の値で，高予算域へは外挿できない)．

```python
!maou selfplay --games 24 --ab-mode budget --playouts 800 --playouts-b 200 \
    --opening-random-plies 8 --seed 1 --max-moves 256 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache \
    --output /content/ab_budget.jsonl
```

判定:

- **A が有意に勝つこと** (`paired` の t 値が明確に正，`A ahead in` が過半)．
  勝たない場合は以降の A/B を回しても意味がないので，先に原因を調べる．
- `A Elo` を 2 doubling (800 vs 200) で割った値が，その予算域での
  1 doubling あたり Elo．§4・§6 の期待値計算に使う．

### 3.1 実測結果 (2026-07-28, Colab L4 / 6400 vs 1600 / 12 局 / threads 2 / max-moves 512)

**ハーネスは健全**: `A result: 11W 0D 1L` / `A score: 91.7% (CI [64.6%, 98.5%])` /
`A Elo: +417 [+105, +729]` / `paired: 6 pairs, mean +0.833, t = +5.00`．
dfpn on/off の 2 run で **A の成績が完全に一致**した (どちらも 11W-1L)．

⇒ **1 doubling あたり +208 Elo [+52, +364]** (2 doubling で +417)．

> **旧値「1 doubling ≈ 60 Elo」(2026-07-27, 24 局, +120 [-23, +264]) は
> 信用しないこと**．当時は**公称予算の 13.6% しか消化していなかった** (下記
> 3.2) ため，A に約束した 4 倍の探索量が実際には届いておらず，棋力差が圧縮
> されていた．予算が届くようになった本測定の点推定は **3.5 倍大きい**．
> ただし n=12 の CI は広く旧値の CI と重なる ([+52,+364] vs [-12,+132]) ので，
> **確定には n ≥ 24 が要る** (この予算だと約 35 分/24 局)．

### 3.2 「公称予算の 13.6% しか消化しない」— 現行ビルドでは再現しない

同一設定で消費率を測り直した (消費 = 実 playout + 空回り = 予算判定の分母):

| 構成 | 実 playout/手 | 空回り/手 | 消費率 | 空回り比 | throughput |
|---|---|---|---|---|---|
| dfpn on | 2,628 | 913 | **88.5%** | 25.8% | 3,999.6 p/s |
| dfpn off (`--no-root-dfpn --no-leaf-mate`) | 3,504 | 372 | **96.9%** | 9.6% | 5,154.4 p/s |
| 旧測定 (2026-07-27) | — | — | **13.6%** | (空回り込み) | 1,997.2 p/s |

**旧測定の 13.6% は再現しない**．dfpn は残差 (96.9% → 88.5% = 8.4 ポイント) を
説明する — 詰みを証明すると `RootProven` で探索を打ち切るため — が，**旧測定の
86% 欠損を説明できる要因ではない**．旧測定と本測定の間に入った変更のどれが
効いたかは特定できていないが，**現行ビルドでは予算はほぼ満額届いている**ので
未解明現象としては閉じる．再発したら**まず消費率 (実 + 空回り ÷ 公称) を出す**
こと — 空回り分離前は「消費されたのか水増しなのか」が読めなかった．

副産物: **dfpn 併走のコストは throughput −22.4%** (5,154.4 → 3,999.6 p/s)．
空回りも 9.6% → 25.8% と 2.7 倍になる (証明済みノードが増える = 再訪が増える)．

## 4. 未決 1: TimeStrategy の想定残り手数

`--ab-mode horizon` は **持ち時間モード** (実時計を回して TimeStrategy に
1 手の予算を決めさせる) で A/B する．A = `--horizon`，B = `--horizon-b` で，
それ以外は同一設定．壁時計で消費を測るため **`--parallel 1` 限定**．

### 4.1 regime ゲート (先に通すこと)

CPU での 3 回の失敗はすべて **regime を外したこと**が原因で，レバーの効果
以前の問題だった (worklog 2026-07-26)．次の 3 条件を満たす時計設定でのみ
結果を採用する:

1. **時計を踏み越えていない** — `reasons` の `timeout` が 0〜1 局
   (多ければ時計設定が短すぎる)．
2. **時計が実際に効く** — `time left at end (avg)` が **A と B で明確に
   違い**，かつ初期持ち時間の大半を残していない (残しているなら「多く使う側
   が単純に得」なだけでトレードオフが発生していない)．
3. **早期終了しない** — `reasons` が `resign` に支配されていない
   (`--resign-value 0` で投了を切る)．

> **`総 playouts ÷ 総手数` はゲートに使わない** (2026-07-27 に判明)．
> 引き分け終端・千日手・証明済み局面が近いと，探索は新しい葉を開かずに
> 終端 backprop だけを回すため，同じ wall clock で playout 数が 27,000 →
> 260,000 まで膨らむ．**サマリの `throughput:` が NN 評価の物理上限
> (L4 / ViT 19.8M で約 11,000 playouts/秒) を超えていたら水増しを疑うこと**．

実測 (2026-07-27, L4 / 40 局 / 30s+0.5s / `checkmate` 39 局): `throughput:`
**668,449 playouts/秒** = 物理上限 (約 11,000) の **約 61 倍**．実 NN 評価から
逆算すると本物の playout は全体の **約 1%** しかない．同じ分母を使う
`subtree reuse:` の割合も **0.8%** へ潰れる (実測済みの 18-20% と乖離) ので，
**reuse 率の異常な低さは水増しの二次シグナルとして使える**．

会計修正後に持ち時間モードで直接測った空回り比は **99.3%** (1 手あたり実 playout
6,223 に対し空回り 828,081) で，この逆算値と一致した (§4.6)．

この水増しは `maou_search` 0.23.0 / `maou_usi` 0.15.0 で解消した．`throughput:`
は実 playout のみを分子に取り，空回りは `terminal spin:` 行に分離される．
**修正前の run の `playouts` と直接比較しないこと** — 同じ探索でも報告値が
1-2 桁小さくなる．

初期持ち時間の目安:

```
初期持ち時間 [秒] ≈ horizon × (目標 playouts/手 ÷ NPS)
```

例: `NPS = 8,101`，目標 4,000 playouts/手 → 1 手 0.5 秒 → `--inc-ms 500`
+ `--clock-ms 30000` (初手は A = 30000/40+500 = 1,250ms / B = 2,000ms)．

**`--max-moves 512` (電竜戦値) を必須とする**: 256 以下だと終盤が上記の
playout 膨張に汚染され，時計がニセの探索に食われる．

### 4.2 実行

```python
!maou selfplay --games 40 --ab-mode horizon \
    --clock-ms 30000 --inc-ms 500 --horizon 40 --horizon-b 20 \
    --resign-value 0 --max-moves 512 --opening-random-plies 8 --seed 1 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache \
    --output /content/ab_horizon.jsonl
```

### 4.3 判定

- **機構の発火**: `time left at end (avg)` が A/B で明確に違うこと．同じなら
  horizon の違いが時間配分に出ていない = レバーが効いていないので，勝率を
  読んではいけない (設計 §12 の A/B tripwire)．
- **効果**: `paired` の平均と t 値を第一に見る (Wilson CI は n=40 では
  ±15% あり，~150 Elo 級しか検出できない)．§3 で得た 1 doubling あたり
  Elo から期待値を出し，符号と桁が整合するかを見る．
- **決着の書き方**: 有意差なしなら「この regime では差が出ない → 既定
  (horizon 40) を据え置く」で閉じてよい．**据え置きも決着**であり，根拠
  (regime ゲートを通した上での測定) を worklog に残すことが要件．
- 既定値を変える場合は `TimeStrategyConfig::horizon_moves` (rust/maou_usi)
  と [docs/commands/usi.md](../../commands/usi.md) を同時に更新する．

### 4.4 実測結果 (2026-07-27, Colab L4 / ViT 19.8M fp16 / 30s+0.5s / 40 局)

| 比較 | A のスコア | Elo | paired | 終局時残り |
|---|---|---|---|---|
| 40 vs 20 | 62.5% (CI [47.0, 75.8]) | +89 [-21, +198] | mean +0.250, **t = +1.75**, A 優位 7/20 (同着 11) | A 5.5s / B 1.6s |
| 60 vs 40 | 41.2% (CI [27.4, 56.6]) | -61 [-169, +46] | mean -0.175, **t = -2.10**, A 優位 **0/20** (同着 16) | A 10.1s / B 6.5s |

**結論: 既定 `horizon_moves = 40` を据え置く**．どちらへ動かしても悪化する
(20 へ = -89 Elo / 60 へ = -61 Elo)．終局時の残り時間が horizon 20/40/60 で
1.6s / 5.5-6.5s / 10.1s と単調に増えており，レバーが時間配分を実際に変えて
いることも確認できている (regime ゲート 2 の合格)．40 vs 50 級の細かい調整は
30 Elo 差の検出に n≈400 局を要するため未検証．

### 4.4.1 手数カーブ (`--ab-mode timecurve`) — 既定 on (暫定)

**結論: 山を変換期に置いた形を既定 on にした．ただし有意水準には達して
いない** — 追試で覆り得る前提で扱うこと．

| 設定 | n | A スコア | Elo | paired t |
|---|---|---|---|---|
| 山=中盤 (ply 55 / 1.8 / 開 0.7 / 終 1.0) | 40 | 33.8% | −117 [−229, −5] | **−2.37** |
| 同上 (再現) | 10 | 35.0% | −108 [−320, +105] | −1.50 |
| **山=変換期 (ply 100 / 2.5 / 開 0.3 / 終 1.2)** | **20** | **65.0%** | **+108 [−47, +262]** | **+1.15** |

中盤へ寄せると 50 局で 34%，変換期へ寄せると 20 局で 65%．**約 220 Elo の
開き**があり，符号の反転自体は 2 系列で一貫している．一方
**「変換期版が一律配分より強い」は t = +1.15 / CI が 0 を含み未確定**．
実測のペア分散から，t = 2.0 には **30 ペア = 60 局**が要る (今回 10 ペア，
mean +0.300 / SD 0.822)．**この追試は未実施** — user 判断で先に既定化した．

#### 4.4.1.1 何を測って何が分かったか

**発火量の予測と実測が一致した** (シミュレーションは実装の `allocate` を
直接叩いたもの):

| ply 帯 | 予測 A/B | 実測 A/B |
|---|---|---|
| 9-30 | 0.76 | 0.75 |
| 31-60 | 0.89 | 0.90 |
| 61-90 | 1.28 | 1.25 |
| 91-110 | 1.30 | 1.20 |

**GPU 時間を使う前にこのプローブを回すこと**．最初の形は実効 0.93〜1.15
倍しか動いておらず，公称値 (0.7〜1.8) を信じると発火量を 3 倍見誤る．

**中立な再解析 (analyze-game) で分かったこと** (1 局の詳細分析):

- 序盤 (ply 9-30) の平均 winrate loss は A 0.0004 / B 0.0013 = **序盤の
  探索時間はほぼ無価値**．削る原資にしてよい．
- 着手品質の優劣が時間配分の境目と一致する: A は時間の多い 31-90 で優れ
  (0.0192 vs 0.0351)，少ない 91 以降で劣る (0.0118 vs 0.0041)．
- **同じ winrate loss でも局面によって値段が違う**．敗者側の最大失着
  (0.2855 / 0.2397) はいずれも既に劣勢のときに出ており勝敗を変えていない．
  勝勢を変換している最中の小さな失着だけが試合を決めた．

#### 4.4.1.2 手順 (追試 / パラメータ変更時)

```
maou selfplay --ab-mode timecurve --clock-ms 300000 --inc-ms 10000 \
  --games <n> --model-path <onnx> --parallel 1 \
  --threads 1 --batch-size 64 --tensorrt --cuda --trt-cache-dir <dir> \
  --output ab.jsonl --kifu-dir kifu/
```

regime ゲート: **持ち時間モード必須** (固定 playout 予算では配分が無い) /
**GPU で測る** (CPU 22 p/s では中盤の増分が棋力差になる前に飽和する) /
**発火量を先に確認** (`--kifu-dir` の JSONL を ply 帯で集計する)．
パラメータは A/B の両者に同じ値が入り，差は on/off だけ — 山を変えたい
場合は A/B ごと回し直す (A 側だけ変えて比較しない)．

#### 4.4.1.3 未決

- **追試 40 局**で t = 2.0 級まで詰める (未実施)．
- 変換期版は**まだ時間を余らせている** (終局残り A 62.7s / B 56.6s)．
  さらに寄せる用量 (頂点 4.0 / 序盤 0.2) は未検証．
- 変換期版で **228 手・267 手の長い将棋**が現れ 1 局が入玉宣言勝ちで
  終わった．終盤が厚くなったことで入玉を目指す展開が成立した可能性がある．
  floodgate の 512 手ルールとの相性は未確認．
- **詰まされる側の盲点**は予算では埋まらない (別課題)．400k playouts
  (対局中の 4 倍) を与えても，A 側の局面からは自分が詰まされていることが
  見えず，B 側の同じ局面からは詰みが見えた．AND ノードの証明が OR ノード
  より難しいという構造的な問題．

## 4.5 空回りの予算開放 (`--ab-mode spin`) — CPU スクリーニングで棄却

**GPU A/B を回す前に CPU で機構の発火量を測ったところ，効果の上限が検出限界を
2 桁下回ることが判明したため，現時点では GPU 時間を使わない**．

測定 (mock 評価器 / 1 局 40 手 / 400 playouts/手 / `--opening-random-plies 6`
で開局を 3 通り):

| seed | relief off: 実 playout / 空回り | relief on: 実 playout / 空回り |
|---|---|---|
| 1 | 12,677 / 931 | 12,828 / 10,771 |
| 2 | 12,645 / 973 | 12,825 / 9,114 |
| 3 | 12,659 / 954 | 12,833 / 8,687 |
| 平均 | 12,660 / 953 | 12,829 (**+1.34%**) / 9,524 (**10 倍**) |

レバー自体は発火している (空回りが 10 倍 = 予算を消費しなくなった分だけ走査が
伸びた) が，**実 playout は +1.34% しか増えない**．空回りは予算を奪っている
のではなく，**その時点で開ける葉が無い**ことの症状だから — 予算を足しても
PUCT の降下先は終端のままで，新しい葉にはならない．引き分け地平が 2 手先の
極端な局面 (`--max-moves 2`) では実 playout が全く増えず，
`stop=spin_exhausted` で止まるだけだった．

期待効果の見積り: 実効予算 +1.34% ≈ log2(1.0134) × 60 Elo ≈ **+1.2 Elo**
(当時の換算値 1 doubling ≈ 60 Elo)．**§3.1 で更新した 208 Elo/doubling で
換算しても +4.1 Elo** で，n=40 の検出限界 ~150 Elo に対して 1-2 桁小さい
ままなので**結論は変わらない**．走査が 10 倍に
なる分の wall clock 増 (実測 +4%) を考えると期待値は負．

**次に試すべきレバーは会計ではなく「選択」側**: 確定済み (proven) の子を PUCT の
候補から外し，全子が確定した親を確定として畳む (MCTS-Solver 相当)．これなら
空回りの降下自体が消えるので，同じ wall clock で実 playout が増える — 持ち時間
モードでも効く (会計の開放は時計が拘束条件のモードでは原理的に無効)．
千日手終端・詰み終端は既に `mark_terminal` + `propagate_proven_from` で確定化
されているため候補から外せる．**深さ上限超過だけは `mark_terminal` しない**
(reroot で深さが変わると stale になるため) ので別扱いが要る — 絶対手数基準の
`max_moves_to_draw` は reroot 不変なので確定化できる可能性がある．実装したら
**本節と同じ手順で発火量 (実 playout の増分) を先に測り**，有意に増えることを
確認してから GPU A/B に進むこと．

## 4.6 確定済み子の選択除外 (`--ab-mode proven`) — GPU 発火測定で棄却

§4.5 で棄却した予算開放の代わりに，**選択**側で空回りを消すレバー．確定済み
(詰み・千日手・確定伝播済み) の子を PUCT の候補から外し，全子が確定した親を
その場で確定化する．会計と違い**降下そのものが消える**ので，時計が拘束条件の
持ち時間モードでも効く．

**CPU 発火確認 (mock / 千日手終端が支配的な局面 / 400 playouts/手)**: 1 手あたり
実 playout 312 → **376 (+20.5%)**，空回り 91 → **27 (-70%)**，throughput
1,039 → **1,244 playouts/秒**．§4.5 のレバーと違い空回りが実探索へ転換している．

**注意 — 効く局面と効かない局面がある**: 空回りの源が **(a) 確定済み終端
(詰み・千日手)** なら効く．**(b) 深さ上限・最大手数の超過**なら効かない — この
打ち切りは `mark_terminal` しない (reroot で深さが変わると stale になるため) ので
確定化できず，候補から外せない．実測でも平手 40 手・`--max-moves 40` の局面
(空回りの大半が (b)) では実 playout が **1 件も変わらなかった**．GPU 実測の 98%
空回りは終盤の詰み・千日手由来 (`--max-moves 512` に対し終局 105 手) なので (a) に
当たると考えられるが，**未確認**．

### 手順

1. **発火量 (A/B より先)**: レバー on/off の**素の 2 run** を比較する
   (`--ab-mode` はサマリが A/B 両者を合算するので発火量が読めない)．

   ```python
   for f in ("", "--skip-proven"):
       !maou selfplay --games 4 --playouts 800 --max-moves 512 \
           --opening-random-plies 8 --seed 1 {f} \
           --model-path /content/model_fp16.onnx --tensorrt --cuda \
           --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache
   ```

   **1 手あたりに直して比べる** (`playouts ÷ plies`)．局数・手数が変わるため総量は
   比較にならない．実 playout/手 が有意に増えていなければ，その分布では空回りが
   (b) 由来なので A/B を回さない．

2. **棋力 A/B** (発火が確認できた場合のみ):

   ```python
   !maou selfplay --games 40 --ab-mode proven --playouts 800 \
       --resign-value 0 --max-moves 512 --opening-random-plies 8 --seed 1 \
       --model-path /content/model_fp16.onnx --tensorrt --cuda \
       --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache \
       --output /content/ab_proven.jsonl
   ```

   判定は §4.3 と同じ (paired の平均と t 値を第一に，n=40 の検出限界 ~150 Elo を
   踏まえる)．§4.5 の換算 (1 doubling ≈ 60 Elo) で発火量から期待値を出し，符号と
   桁が整合するかを見る．

3. 有意に強ければ既定 on 化と USI option 化を別 PR で起票する．

### GPU 実測 (2026-07-27, Colab L4 / ViT 19.8M fp16 / 各 4 局) — **棄却**

手順 1 (発火量) を 2 つの regime で測った．**どちらも棋力 A/B に値する発火量では
なかった**ため，A/B は実施しない．

固定予算 (`--playouts 800` / `--max-moves 512`):

| | 実 playout/手 | 空回り/手 | 空回り比 | throughput |
|---|---|---|---|---|
| off (473 手) | 694.6 | 113.0 | 14.0% | 3,168.7 p/s |
| on (752 手) | 698.9 (**+0.6%**) | 82.9 (**−26.7%**) | 10.6% | 3,124.6 p/s |

持ち時間モード (`--clock-ms 30000 --inc-ms 500`; 126 件の偽証明を観測したのと
同じ regime．**会計修正後に空回り比率を直接測った初めての値**):

| | 実 playout/手 | 空回り/手 | 空回り比 | throughput |
|---|---|---|---|---|
| off (523 手) | 6,223 | **828,081** | **99.3%** | 7,006.5 p/s |
| on (444 手) | 6,334 (**+1.8%**) | 406,541 (**−50.9%**) | 98.5% | 7,031.3 p/s (**+0.35%**) |

**空回りを半減させても実探索は +1.8%，throughput は +0.35%**．期待効果は
log2(1.018) × 60 ≈ **+1.5 Elo** で，n=40 の検出限界 ~150 Elo の 2 桁下
(n≈400 でも検出できない)．

### なぜ効かないのか — 空回りは探索速度を奪っていない

1 手 0.89 秒で 828,081 回の走査 = 約 **93 万 traversal/秒** を CPU が回している
一方，実 playout は **7,000/秒** で律速は **GPU の評価バッチ充填**．空回りの走査は
GPU が評価している間の**遊んでいる CPU 時間**で起きるため，半減させても wall clock
上の実探索はほとんど増えない．

⇒ **空回りは計測上の見かけの問題であり，性能上の問題ではなかった**．この campaign の
実質的な成果は**会計の分離** (`throughput` / `nps` / `carried_visits` が実探索量を
表すようになったこと) であり，レバーは 2 つとも棄却する (計測器として既定 off で
残す)．探索速度の伸びしろは §5.1 が特定済みの**バッチ aggregator (約 2 倍)** 側に
ある．

### 着手選択への副作用 (確認済み)

確定済みの子は訪問が伸びないため robust child では選ばれなくなる．そのため有効時は
`best_root_index` が**確定値で上書き判定**する (確定値 > robust child の推定 q なら
確定側)．「確実な引き分け」を「不確実な劣勢」と取り違えないためで，逆に推定が
引き分けより良ければ上書きしない
(`test_skip_proven_children_prefers_sure_draw_over_worse_guess` /
`..._keeps_better_unproven_move` で pin)．千日手模様の指し手選択が変わるので，
**棋力 A/B の前に千日手判定の回帰 (`reasons` の `repetition` 件数) も見ること**．

## 4.7 受け方向の詰み探索 (`--ab-mode defmate`) — GPU 検証

環境構築は §1 のとおり (wheel + `ldconfig` + モデル配置)．以下は
**その続きから流す Colab の python セル列**．

### セル 1 — 共通設定と pin 局面

```python
# 4.7-1. 共通設定．以降のセルはこの変数を使う．
import json, re, subprocess, statistics, time

MODEL = "/content/model_fp16.onnx"
GPU = ["--model-path", MODEL, "--tensorrt", "--cuda",
       "--threads", "1", "--batch-size", "64",
       "--trt-cache-dir", "/content/trt_cache"]

# 自己対局で実際に踏んだ局面 (game_0006)．
BLUNDER = "l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1NR3/1G1PPg3/1S2K1+b2/LN7 b BSgsnl8p 117"
MATED   = "l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1N4/1G1PPR3/1S2K1+b2/LN1s5 b BGSgnl8p 119"
QUIET   = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

def run_search(sfen, time_ms=11000, defensive=True):
    """maou search を叩いて bestmove / 候補 / Stats を返す."""
    cmd = (["maou", "search", "--sfen", sfen, "--time-ms", str(time_ms)]
           + GPU + (["--defensive-mate"] if defensive else ["--no-defensive-mate"]))
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    best = re.search(r"^Bestmove:\s*(\S+)", out, re.M)
    winrate = re.search(r"^WinRate:\s*([\d.]+)", out, re.M)
    stats = dict(re.findall(r"(\w+)=([\w.-]+)", re.search(r"^Stats:.*", out, re.M).group(0)))
    cands = re.findall(r"^(\S+) \(visits=(\d+),.*?(proven=\w+)?\)$", out, re.M)
    return {"best": best.group(1) if best else None,
            "winrate": float(winrate.group(1)) if winrate else None,
            "stats": stats, "cands": cands, "raw": out}

print("ok")
```

TensorRT 有効時は teardown 回避のため destructor を走らせずに終了するが
(§8.5)，**出力は flush 済みで exit code も 0** なので `subprocess` で問題なく
拾える．

### セル 2 — Stage A: 機構が実機で発火するか

```python
# 4.7-2. Stage A: pin 局面 3 つで発火と着手を確認する (各 11 秒 × 4 回 ≒ 1 分)．
def check(label, cond, detail):
    print(f"[{'OK ' if cond else 'NG '}] {label}: {detail}")
    return cond

ok = True

# A-1 敗着局面 (まだ受かる / 合法手 4)．
r = run_search(BLUNDER); rb = run_search(BLUNDER, defensive=False)
n_loss = sum(1 for c in r["cands"] if c[2])
ok &= check("A-1 bestmove", r["best"] == "5h6i", f"{r['best']} (対照 {rb['best']})")
ok &= check("A-1 除外手数", n_loss == 3, f"proven=loss が {n_loss} 手")
ok &= check("A-1 発火", int(r["stats"]["filtered_root_moves"]) >= 1,
            f"filtered_root_moves={r['stats']['filtered_root_moves']} "
            f"defensive_mates={r['stats']['defensive_mates']}")
for mv, v, pv in r["cands"]:
    print(f"      {mv:6s} visits={int(v):>8,} {pv or ''}")

# A-2 被詰み局面 (もう詰んでいる)．
r = run_search(MATED); rb = run_search(MATED, defensive=False)
ok &= check("A-2 勝率", r["winrate"] == 0.0, f"{r['winrate']} (対照 {rb['winrate']})")
ok &= check("A-2 発火", int(r["stats"]["defensive_mates"]) >= 1,
            f"defensive_mates={r['stats']['defensive_mates']}")

# A-3 静かな局面 (偽陽性ゼロ)．
r = run_search(QUIET)
ok &= check("A-3 偽陽性なし",
            int(r["stats"]["defensive_mates"]) == 0
            and int(r["stats"]["filtered_root_moves"]) == 0,
            f"defensive_mates={r['stats']['defensive_mates']} "
            f"filtered_root_moves={r['stats']['filtered_root_moves']}")

print("\nStage A:", "PASS" if ok else "FAIL — 下の判定表を参照")
```

| 判定 | 期待 | 外れたときに疑うこと |
|---|---|---|
| A-1 bestmove = `5h6i` | 唯一の受かる手 | 対照と同じ手なら未発火か gate 外れ |
| A-1 `proven=loss` が 3 手 | 4f4g(mate-41) / 5h5i(mate-1) / 5h6h(mate-29) | 予算不足なら `--time-ms` を上げる |
| A-2 `WinRate = 0.0` | 被詰みの正直な報告 | 対照は 0.49 前後の互角 |
| A-3 両カウンタ 0 | 偽陽性ゼロ | **0 でなければ即調査** (偽の被詰みは自滅につながる) |

**`filtered_root_moves` が 3 未満でも異常ではない** — 木の proven 伝播
(leaf-mate 経由) が先に確定させた手はフィルタが上書きしないため内訳は動く．
**見るべきは「除外が 3 手」と「bestmove = 5h6i」**．

DevContainer / mock 評価器での実測 (参考値．GPU では内訳が動く):

```
Bestmove: 5h6i
5h6i (visits=990,    winrate=0.4846)               ← 選ばれた
5h6h (visits=541509, winrate=0.4987, proven=loss)
4f4g (visits=482872, winrate=0.4988, proven=loss)  ← 対局で選んだ敗着
5h5i (visits=1744,   winrate=0.1197, proven=loss)
Stats: ... leaf_mates=6439 defensive_mates=19 filtered_root_moves=2 ...
```

**visits では敗着が 50 万回で圧倒的なのに 990 回の手が選ばれる** — MCTS の
訪問分布では敗着が勝っており，確定値の除外だけが手を変えている．
A-2 は `WinRate 0.0000` / `defensive_mates=50` / `filtered_root_moves=0`
(既に詰みなら候補選別はしない = 設計どおり)．

### セル 3 — Stage B: CPU 競合で探索速度を奪わないか

```python
# 4.7-3. Stage B: 静かな局面で nps を on/off 比較する (各 15 秒 × 2 回)．
import os
print("vCPU:", os.cpu_count(),
      "/ 常駐スレッド = MCTS 1 + 攻め dfpn 1 + 受け dfpn 1 + leaf-mate 1 = 4")

on  = run_search(QUIET, time_ms=15000, defensive=True)
off = run_search(QUIET, time_ms=15000, defensive=False)
nps_on, nps_off = int(on["stats"]["nps"]), int(off["stats"]["nps"])
drop = (1 - nps_on / nps_off) * 100
print(f"nps on={nps_on:,} off={nps_off:,} → 低下 {drop:.1f}%")
print(f"充填率 on={float(on['stats']['avg_batch'])/64:.2f} "
      f"off={float(off['stats']['avg_batch'])/64:.2f} "
      f"(外れ値を見たら充填率と collisions を必ず併記する — §1)")
print("判定:", "許容 (<5%)" if drop < 5 else "要調査 (>=5%) — 並列度は上げない")
```

**静かな局面で測る**こと．gate 該当は全局面の 7.6% (終盤 21.6%) なので，
静かな局面での低下はそのまま「常時払う税」になる．
5% 以上なら `--defensive-mate-threads` を上げずに Stage C へ進む．

### セル 4 — Stage C: 棋力 A/B

```python
# 4.7-4. Stage C: A/B 40 局．持ち時間モード必須 / --parallel 1 必須 / GPU 必須．
!maou selfplay --ab-mode defmate --clock-ms 300000 --inc-ms 10000 \
    --games 40 --parallel 1 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache \
    --output /content/ab_defmate.jsonl --kifu-dir /content/kifu_defmate
```

regime ゲート (先に通すこと):

- **持ち時間モード必須** — 固定 playout 予算では敗着フィルタが使える壁時計が
  変わる (フィルタは MCTS 終了の停止フラグで打ち切られる)
- **`--parallel 1` 必須** — フィルタが CPU を使うので同時対局は互いを歪める
- **GPU 必須** — CPU 22 p/s では手の分布が別物になる

**期待効果の見積り**: CPU 全数走査では 30 局中 7 局 (23%) に「避けられた
強制詰みの敗着」があった．1 局あたり平均 0.23 手の改善であり，
**n=40 が検出できる ~150 Elo 級に届くかは未知**．有意にならなくても
機構が効いていないことにはならない — だから Stage D を併せて見る．

### セル 5 — Stage D: 棋譜による直接確認 (Elo より感度が高い)

```python
# 4.7-5. Stage D: A/B の棋譜を走査し「避けられた敗着」を A/B で数える．
#   1. 合法手 <= 16 を gate  2. 各合法手を指した後を攻め方向 dfpn で判定
#   3. 「指した手が敗着 かつ 安全な代替手が存在した」を数える
import glob, re
from maou._rust import maou_shogi as S

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
STAGES = [1_000, 50_000, 500_000]   # production と同じ段階予算

def scan(path):
    """1 局を走査し，避けられた敗着の明細を返す (A/B の割当は CSA の N+ から)."""
    text = open(path).read()
    m = re.search(r"^N\+(.*)$", text, re.M)
    black_is_a = (m.group(1).strip().upper() == "A") if m else True
    rec = S.parse_csa_str(text)[0]
    b = S.PyBoard(); b.set_sfen(rec.sfen or START)
    hits = []
    for i, mv in enumerate(rec.moves):
        legal = b.legal_moves()
        if len(legal) <= 16:
            played = S.move_to_usi(mv)
            unresolved, losing, safe = list(legal), [], []
            for budget in STAGES:
                nxt = []
                for m2 in unresolved:
                    b.push(m2); child = b.sfen(); b.pop()
                    # find_shortest=False 必須 — 既定 (True) は最小性を確定
                    # できないと Unknown を返し，長手数の詰みを見逃す
                    r = S.solve_tsume(child, nodes=budget,
                                      find_shortest=False, timeout_secs=60)
                    (losing if r.status == "checkmate" else
                     safe if r.status == "no_checkmate" else nxt).append(m2)
                unresolved = nxt
                if not unresolved:
                    break
            if played in [S.move_to_usi(x) for x in losing] and safe:
                side = "A" if ((i + 1) % 2 == 1) == black_is_a else "B"
                hits.append({"ply": i + 1, "side": side, "played": played,
                             "safe": [S.move_to_usi(x) for x in safe][:3]})
        b.push(mv)
    return hits

tally = {"A": 0, "B": 0}
for path in sorted(glob.glob("/content/kifu_defmate/game_*.csa")):
    for h in scan(path):
        tally[h["side"]] += 1
        print(f"  {path.split('/')[-1]} ply{h['ply']:3d} [{h['side']}] "
              f"{h['played']} → 安全手 {h['safe']}")
print(f"\n避けられた敗着: A={tally['A']} B={tally['B']}")
print("判定:", "機構は効いている" if tally["A"] < tally["B"] else "要調査")
```

判定: **A 側の件数が B 側より少なければ機構は効いている**．Elo が有意で
なくても，この差は直接観測できる (30 局の事前走査では 7 件あった)．

### 4.7.1 実測結果 (2026-08-01/02, Colab L4 / ViT 19.8M fp16 / 300s+10s / 40 局)

**Stage A: PASS / Stage B: コスト実質ゼロ / Stage C・D: 効果を示さず**．
ただし **C・D は「敗着を除外するが，その手に探索の 99.4% を注ぎ込む」版で
測ったもの**であり，修正後に再測定が要る (下記)．

| Stage | 結果 |
|---|---|
| A-1 | bestmove **5h6i** (対照 **4f4g** = 実際に負けた敗着)．`proven=loss` 3 手 / `filtered_root_moves=2` / `defensive_mates=350` |
| A-2 | **`WinRate 0.0`** (対照 0.3615) / `defensive_mates=372` |
| A-3 | 両カウンタ **0** (偽陽性なし) |
| B | vCPU 12．**nps 11,923 (on) vs 11,814 (off) = −0.9% (むしろ微増)**．充填率は両者 1.00 |
| C | A **45.0%** (15W 6D 19L) / **−35 Elo [−141, +72]** / t = −0.81 / 20 ペアで A 優位 4 (同着 11) |
| D | 避けられた敗着 **A=1 / B=1** (4,561 局面中 gate 該当 478 = 10.5%) |

#### 何が分かったか

**(1) 機構は実機で正しく動く**．A-1 の対照が決定的で，**受け方向を切ると
実際に負けた敗着 4f4g をそのまま選ぶ**．コストは静かな局面で測って −0.9% =
実質ゼロ (12 vCPU では常駐 4 スレッドが競合しない)．

**(2) しかし playout を敗着へ流し続けていた**．A-1 の内訳:

```
5h6i   visits=      24        ← 実際に選ばれた手
4f4g   visits= 123,306 proven=loss
5h6h   visits=     793 proven=loss
5h5i   visits=       1 proven=loss
```

**root 訪問の 99.4% が「負けと証明済みの手」に注がれ，選ばれた手は 24 visits
しか探索されていない**．敗着フィルタは `collect_result` (探索終了後) でしか
効いておらず，木にマークしていなかったため．修正後は同じ局面で
**600,584 visits (79.7%)** が生きている手に入る．**C の A/B はこの欠陥を
抱えた版の測定値**なので，敗着回避の利得が着手品質の劣化で相殺されている
可能性が高い．

**(3) Stage D の指標には既知の欠陥がある**．走査は履歴を持たない dfpn で
「安全」を判定するため，**連続王手の千日手による負けを見落とす**．
実例: game_0000 ply131 は走査が「6i7i で受かる」と判定したが，対局中の
engine は `-30000` を `T0` で報告した．受け方向ソルバ自体は 50〜500,000
ノードのいずれでも詰みを申告しない (偽証明ではない — 回帰テスト
`test_defensive_no_false_proof_ply131` で固定) ため，engine の判断は
subtree 再利用で積み上がった木の証明由来であり，**探索線上の千日手を
含み得る**．したがって **D の件数は上界**として読むこと．

**(4) この regime では A/B が原理的に underpowered**．D の基準率が
40 局で高々 2 件 (上界) であり，1 局あたり 0.05 手．事前見積り 0.23 手は
別条件の 30 局から採ったもので**過大だった**．n=40 が検出できる ~150 Elo
級には遠く及ばない．

#### 再測定の手順

1. **Stage A を先に流す** — A-1 で `5h6i` の visits が全体の過半になること
   (24 visits のままなら修正が効いていない)
2. Stage C を同条件で再実行する
3. Stage D は**上界**として読み，千日手を判定に含める改良を別途行う

---

### 記録先

結果は §9 のとおり worklog + compass へ．A/B が有意でなかった場合も
**「発火したが Elo 差は検出限界未満」と「発火しなかった」を必ず区別して
書くこと** — 前者は追試の対象，後者は実装のバグである．

## 5. 未決 5: バッチ aggregator の採否

同時対局数を振って **wall clock ベースの playouts/秒** (`throughput:` 行) を
比較する．CPU では評価器の `Mutex<Session>` が上限で `parallel 1/2/4` が
`64/65/65 playouts/秒` (完全に頭打ち) だった．

```python
for p in (1, 2, 4, 8):
    !maou selfplay --games 8 --parallel {p} --playouts 800 --max-moves 120 \
        --model-path /content/model_fp16.onnx --tensorrt --cuda \
        --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache --quiet
```

判定規則:

- **スケールする** (parallel 4 で 3 倍以上) → GPU では Mutex 直列化が上限に
  ならない ⇒ **aggregator は不要** (未決 5 を「見送り」で確定)．
- **頭打ち** (parallel 4 で 1.5 倍未満) → GPU が遊んでいる ⇒ **採用検討**．
  併せて `nvidia-smi dmon` などで GPU 利用率を見て「直列化で遊んでいる」
  ことを確認してから，次 campaign の課題として起票する．
- 中間 (1.5〜3 倍) は `--batch-size` を振って上限がバッチ側か直列化側かを
  切り分ける (バッチを上げて改善するなら aggregator の余地がある)．

### 5.1 実測結果 (2026-07-27, Colab L4 / 8 局 / 800 playouts/手 / dfpn off)

| threads | parallel 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 4,682 | 5,644 | 5,882 | 5,879 playouts/秒 |
| 2 | 2,767 | — | 3,062 | — |

#### 会計修正後の再計測 (2026-07-27, 同一設定．空回り分離後)

上表は**空回り込みの値**だったため取り直した (`maou_search` 0.23.0 以降は
`playouts` が実探索量のみ)．

| parallel | 実 playout/秒 | 旧会計へ換算 | 上表の記録値 | 差 |
|---|---|---|---|---|
| 1 | 4,399.8 | 4,696.9 | 4,681.6 | +0.3% |
| 2 | 5,331.6 | 5,692.3 | 5,643.5 | +0.9% |
| 4 | 5,530.5 | 5,905.2 | 5,882.4 | +0.4% |
| 8 | 5,536.4 | 5,909.7 | 5,879.2 | +0.5% |

- 空回りは消費予算の **6.34%** のみ (`--playouts 800` の短い探索では空回りが
  育たない — 持ち時間モードの 99.3% と対照的 §4.6)．**旧会計へ換算すると旧値を
  1% 以内で再現**するので同一 regime の再測定として成立している．
- スケール上限 **1.257×** (旧 1.256×) で不変．単発探索との差は **1.97×**
  (旧換算 1.86×) → **「約 2 倍の余地」は会計修正後も維持**．
  なお天井を**同じ session 内で測った値** (10,095 p/s) で割ると **1.82×**．
  session 変動が 2 倍あるため (§1)，この比は必ず同一 session の天井で出すこと．
- `game time` (局ごとの合計) は parallel 1 → 8 で 163.6s → 1,037.2s = **6.3 倍**．
  wall clock は 1.26 倍しか縮まないので，**共有評価器での直列化**が可視化されている．
- 全設定で plies 864 / playouts 724,528 / 空回り 49,056 が **bit-identical** =
  並列度は探索結果を変えない．

**結論 (2026-07-28 再改訂): 対局間 aggregator は棋力の主経路ではない．
主レバーは `--batch-size` だった**．

伸びしろ約 2 倍は当初「対局をまたいで評価要求をまとめれば取れる」と読んで
いたが，GPU 実測で否定された (詳細:
[position-search/eval-batching.md](../position-search/eval-batching.md)):

- **TensorRT のコストは padding 後の長さにほぼ比例する** — `cost(n) ≈ 0.15 +
  0.084·n` [ms] で **固定費はほぼ無い** (batch 8/32/64/128/256 で
  1.13/2.85/5.39/10.95/25.47 ms/call)．したがって**呼び出し回数を減らしても
  得はなく，効くのは padding を減らすことだけ**である．
- **`--batch-size 256` は充填率 80% で padding を捨てていた**．batch 64 は
  fill 97% で，**単発 6,400 playouts で 11,459 vs 8,004 nps**，
  **持ち時間 30s+0.5s で 10,257 vs 7,646 p/s (+34.1%)**．
- **棋力 A/B で決着** (§5.2): batch 64 が batch 256 に **+137 Elo [+33, +241]**
  (48 局 / t = +3.19)．batch 32 vs 64 は有意差なし (t = −0.44) で，
  **充填が飽和する最小値 = 64 が最適**．
- 対局間 aggregator を降ろす根拠も差し替わった．旧稿は「threads=1 の充填率が
  99.6% だから束ねる余地がない」としたが，その 99.6% は 51,200 playouts の
  **定常値**で，実配置の短い探索では 31-55% である．**降ろす理由は罠 1 —
  利得が対局間にしか出ず，実戦 (1 局) にも A/B の Elo 差にも現れない**こと．
- **threads 2 が約 4 割遅い**のは，衝突 1 回でバッチ収集を打ち切るため
  1 バッチが平均 69 件になり，GPU 呼び出し回数がそのままコストになるから
  (threads 1/2/4 = 201/743/3,052 回)．

**当面の方針は棋力を直接狙うこと**．対局間 aggregator は **self-play の
データ生成レートと実験 wall clock の改善**として別枠で残す
([eval-batching.md §7](../position-search/eval-batching.md))．

なお dfpn 併走のコストは wall +18% / throughput -15% で，実配置で切る理由はない．

### 5.2 `--batch-size` の A/B (`--ab-mode batch`)

バッチサイズは**速度と探索の質の両方**に効く — 大きいバッチは padding を捨てる
一方，in-flight の葉が増えて virtual loss が PUCT を歪める．**必ず持ち時間モードで
回すこと**: 固定 playout 予算では両者とも同じ playout 数を使うので，速度差が
棋力差にならない．

```python
!maou selfplay --games 48 --parallel 1 --ab-mode batch \
    --batch-size 64 --batch-size-b 256 \
    --clock-ms 30000 --inc-ms 500 --max-moves 256 \
    --opening-random-plies 8 --seed 1 --alternate-colors \
    --no-root-dfpn --no-leaf-mate \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --trt-cache-dir /content/trt_cache \
    --output /content/ab_batch.jsonl
```

#### 実測結果 (2026-07-28, Colab L4 / 各 48 局)

| A / B | A Elo | paired t | 判定 |
|---|---|---|---|
| **64 / 256** | **+137 [+33, +241]** | **+3.19** | **64 の勝ち** |
| 32 / 64 | −22 [−119, +75] | −0.44 | 有意差なし |

64 vs 256 は 33W 0D 15L (68.8%)．色バランス (先手 23 / 後手 25) も残り持ち時間
(A 5.5s / B 5.4s) も公平．

**予測を 2.3 倍上回った**．throughput +34.1% の換算値は +59 Elo
(1 doubling ≈ +140) で，差分の **+78 Elo 相当は 1 playout あたりの質の向上**と
解釈できる．**throughput からの Elo 換算は棋力の下界として扱うこと** —
バッチを縮める変更では質の向上分が上乗せされる．

#### 一般則

| batch | fill | 状態 |
|---|---|---|
| 8 | 100% | 充填は満点だが **GPU が遊ぶ** (7,096 nps) |
| 32 / **64** | 99% / 97% | **飽和点 = 最適**．両者に棋力差なし |
| 128 | 91% | わずかに padding 損 |
| 256 | 80% | **padding 損が大きい** → 64 に −137 Elo |

⇒ **`batch_size` は「充填が飽和する最小値」に置く**．それより大きいと padding を
捨て，小さいと GPU が遊ぶ．**別モデル・別 GPU では測り直すこと** (最適値は
モデルの計算量と GPU の並列度で決まる)．

**コードの既定値 8 は据え置いている**: CPU は padding が無くコストが実 items に
比例するため速度メリットが無く，質だけ落とす可能性がある．CPU での最適値は未測定．


## 6. subtree 再利用の GPU 実挙動 (`--ab-mode subtree`)

CPU では「探索手の 90% で reroot 成功・引き継ぎは playout の 18-20%」が
実測済み (`--parallel 1` / 実モデル)．GPU (TRT) でも同じ発火量になるかを
確認する — mock 評価器では 1.2% しか出ず，mock だけ見ると無効と誤判定する．

```python
!maou selfplay --games 24 --ab-mode subtree --playouts 800 \
    --opening-random-plies 8 --seed 1 --max-moves 256 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache
```

`subtree reuse:` 行の引き継ぎ率が CPU と同水準 (18-20%) なら，CPU で出した
「on 継続」の結論が GPU でも通る．大きく外れたら worklog に記録して原因
(TRT 経路のバッチ待ちで再利用が効かない等) を追う．

## 7. USI プロトコルの headless smoke (GUI なし)

GUI を使わずに確認できる範囲 (`go mate` / ponder / keep-alive / TRT 初回
ビルド中の `readyok` 待ち) はここで潰す．**一括 pipe は使わない** —
`quit` が先に届いて `stop` が立ち，探索が 0 playout で終わる (既知の罠)．
必ず応答を待ってから次を送る:

```bash
# GUI 機・Colab のどちらでも同じスクリプトを使う (既定は mock 評価器なので
# モデル転送前でも疎通だけ確認できる)．起動に失敗したら欠落 DLL を名指しする
python scripts/usi_smoke.py \
    --model-path /content/model_fp16.onnx \
    --keep-alive 200
```

`--engine` で実行ファイルを指定できる (既定 `maou-usi`)．Windows なら
`--engine <venv>\Scripts\maou-usi.exe`．`[isready] N.Ns / KeepAlive 空行 M 行`
を出すので，**M を数えてから** GUI 側の確認へ進む — M = 0 のまま GUI で
「何も起きなかった」を見ても，無視されたのか発火していないのかが区別できない．

確認項目:

- `isready` の所要 (TRT 初回ビルド) と，その間に **空行が流れているか**
  (`KeepAlive` の実動作．GUI が無害に無視するかは実機でしか判らない = 未決 2)．
- `bestmove ... ponder ...` が付くこと，`ponderhit` 後に `bestmove` が返ること．
- `go mate` が `checkmate <手順>` を返すこと．

## 8. GUI 実機検証 (実施済み — 2026-07-28)

**環境**: Windows / **ShogiHome** (winget 経由で導入) / AMD Ryzen 5 3500U
(4C8T / 15W モバイル APU) / RAM 5.9GB / CPU 推論 (GPU なし)．
エンジンは Actions artifact の Windows wheel を venv へ pip install し，
`<venv>\Scripts\maou-usi.exe` を直接登録した (引数なしエントリポイントなので
bat ラッパーは不要)．

GUI の入手は **winget を使う** — 将棋 GUI はいずれも未署名なので，ブラウザで
ダウンロードすると Mark of the Web が付いて SmartScreen の警告が出る．GUI を
変えても避けられないので，変えるべきは入手経路の方:

```powershell
winget install sunfish-shogi.shogihome
winget install shogixyz.ShogiGUI
```

チェックリスト:

- [x] エンジン登録 (`maou-usi` を引数なしで起動) と `usi` → `usiok` →
      option 一覧の表示．
- [x] **`KeepAlive` の空行を GUI が無害に無視するか** — **ShogiHome は無視した**．
      `KeepAlive 200` で `isready` 中に空行 2 行が流れ，GUI はそのまま
      `usinewgame` へ進み `close=0` で正常終了．⇒ 既定を **on (5000ms)** に
      反転 (未決 2 決着)．他の GUI は未確認．
- [ ] **TRT 初回エンジンビルド中に `readyok` を待てるか** — CPU 機なので
      該当せず未確認．GPU 機での確認が要る．
- [ ] **`OpeningScript` が実サーバ/GUI 経由で正しく消化されるか** — 未実施．
- [x] ponder の実挙動 — GUI は `go ponder` を送る．ただし**予想手はほぼ当たらない**
      (1 手 24 playouts 級では当然) ので毎手 `stop` が飛ぶ．`stop` から
      `bestmove` までは `evaluate_batch` が中断不能なぶん遅れる (下記)．
- [x] 1 局を最後まで完走 (詰みまで)．投了・入玉宣言・千日手の表示は今回の
      対局では発生せず未確認．

**実測 (この機械)**: `maou search` 30 秒 / batch 8 / threads 1 で
`playouts=172 nps=6 avg_batch=7.8 warmup_ms=270`．同条件の DevContainer は
`playouts=676 nps=22`．**3.7 倍遅い**．ORT の推論スレッドは 1 に固定
(`backend.rs`) で `Mutex<Session>` 直列なので，**コア数は速度に寄与しない**．
Zen+ が AVX2 を内部 128bit 幅で実装していること + 15W の持続クロックで説明が付く．
**この速度では棋力の評価はできない** (1 手 180 playouts 級)．§8 が見るのは
GUI 互換性なので目的には支障しない．

**この検証で見つかった不具合**: 秒読み 5 秒の対局で切れ負けした．エンジンの
期限判定が warmup の後から始まり，かつ中断不能な `evaluate_batch` を予約して
いなかったため，壁時計が予算を超えていた．超過量は `batch_size ÷ nps` に比例
するので GPU (6ms 級) では露見せず，6 playouts 秒の機械で初めて出た．
修正済み (maou_search 0.26.1)．**回避策としては `NetworkDelay` を 2500 に
上げると安定した**．

## 8.5 既知の課題 (検証中に判明)

いずれも worklog/2026-07-27-140516.md に詳細．

- **dfpn の偽証明アラート — 解決済み (f967499 / PR #408)**．
  `[dfpn] STRICT VERIFY None (偽証明/不完全)` が 40 局で 9 件，同設定の追試で
  126 件出ていた．真因は `can_interpose_bb` の二歩判定が**移動前**の歩
  ビットボードを見ていたこと (王手手が守備の歩を取るとその筋の二歩が解ける)
  で，TT 再利用による GHI ではない (TT は solve をまたいで再利用されない)．
  soundness 違反 (偽の 1 手詰) と完全性の欠落 (本物の詰みの取りこぼし) の
  両方を起こしていた．**追認 (2026-07-27, 同一コマンドで 40 局)**:
  `STRICT VERIFY None` **0 件** / `FALSE MATE` **0 件** / 検証器の発火
  `sound_checks` **28,177,735 件**．
  再発の監視は `MATE1PLY_VERIFY=1` を付けて自己対局を回す
  (**`sound_checks` の総和が 0 でないことを必ず確認する** — 検証器が発火して
  いなければ「0 件」は無意味)．統計行は solve ごとに出るので総和を取ること．
  `DFPN_STATS=1` は併用しない (成功時の行が solve ごとに出て log が溢れる)．
- **終端再訪による探索の空回り** — 引き分け終端・千日手・証明済み局面が
  近いと新しい葉を開かずカウントだけ回り，時間制では予算を空回りに使う．
  実戦 (電竜戦 512 手上限の終盤・千日手模様) でも起きる効率問題．
- **`corrupted double-linked list`** — **並列とは無関係**．単発の
  `maou search --tensorrt --cuda` (parallel 1) でも，`Stats`・盤面の出力後の
  **プロセス終了時**に発生した (2026-07-27)．従来は `--parallel 4` 終了時の 1 回
  だけだったため対局 driver・並列度を疑っていたが，どちらも無関係と判明．
  数値は出力後なので有効．DevContainer の mock では再現しない → ONNX Runtime /
  TensorRT EP の teardown が疑わしい．**最小再現 (約 30 秒)**:

  ```bash
  maou search --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 2 --batch-size 64 --time-ms 3000 --trt-cache-dir /content/trt_cache
  ```

  **切り分け完了 (2026-07-27)**: 同一コマンド 3 回で **3/3 決定的**に再現し，
  `--tensorrt` を外した `--cuda` のみでは **0/3**．**TensorRT EP 固有**で確定
  (CUDA EP も同じ provider 共有ライブラリ機構で動的ロードされるが壊れないので，
  「静的 ORT コア + 動的 provider の二重 ORT」説は否定された)．
  **回避を実装済み** (`maou 0.60.1`): `maou search` / `maou selfplay` /
  `maou-usi` は **TensorRT 有効時のみ**，全出力を flush した後に destructor を
  経由せずプロセスを終える (`common.exit_skipping_teardown`)．終了コードが
  SIGABRT にならないので GUI/対局サーバからクラッシュと見なされない．
  **原因は外部ライブラリ側なので根治はしていない** (`ort` / onnxruntime-gpu の
  更新で再確認する)．

## 9. 結果の記録先

- 数値と失敗した試行は `worklog/YYYY-MM-DD-HHMMSS.md` (JST，追記不可)．
- 「再導出しない結論」は `scratchpad/compass.md` の Invariants へ．
- 既定値やドキュメントを変える場合のみ `reviews/*.md` を起票する
  (`CLAUDE.md` / `docs/` を変更するときの必須手順)．
- 未決事項の状態は [index.md §12](index.md) の表を更新する．
