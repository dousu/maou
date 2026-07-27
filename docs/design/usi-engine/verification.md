# USI エンジンの検証手順 (GPU / GUI)

[設計本体](index.md) の未決事項のうち，**DevContainer (CPU) では原理的に
閉じられないもの**の手順書．CPU で閉じた項目の根拠は index.md §12 を見ること．

| 残件 | 必要な環境 | 状態 | 手順 |
|---|---|---|---|
| 未決 1 TimeStrategy の定数 | GPU (探索速度が要る) | **決着 2026-07-27** (horizon 40 据え置き) | [§4](#4-未決-1-timestrategy-の想定残り手数) |
| 未決 5 バッチ aggregator | GPU | **決着 2026-07-27** (現行構成では採用しない) | [§5](#5-未決-5-バッチ-aggregator-の採否) |
| 未決 2 keep-alive の既定値 | GUI 実機 | **未実施 (将来課題)** | [§8](#8-gui-実機検証-将来課題-未実施) |

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
--tensorrt --cuda --threads 1 --batch-size 256 \
--trt-cache-dir /content/trt_cache
```

- **`--threads 1` が最適** (実測 2026-07-27): threads 2 はどの並列度でも
  約 4 割遅い．1 手 800 playouts 級の短い探索では探索内スレッド並列が
  オーバーヘッドにしかならない (単発 30 秒探索のような長い探索でのみ有効)．
- 実効速度 (L4 / ViT 19.8M fp16 / threads 1): **単発 30 秒 = 10,909
  playouts/秒，500ms = 8,101 playouts/秒**．短い探索ほど遅いのは木が育つ
  まで評価バッチが埋まらないため．

- TensorRT の初回エンジンビルドは **バッチ shape ごとに数十秒〜数分**．
  `--trt-cache-dir` を必ず指定し，同じセッション内で使い回す．
- **計測の前にキャッシュを温める** (§2 の smoke を先に 1 回通す)．初回
  ビルドが計測区間に入ると playouts/秒 が過小に出る．

## 2. 事前確認 — 探索速度の実測 (以降の設定はこの値から決める)

```python
# 1 局面探索の NPS (warmup はエンジンビルドを計測区間から外す)
!maou search --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 256 --time-ms 30000 --root-dfpn \
    --trt-cache-dir /content/trt_cache
```

```python
# 自己対局 1 局の smoke (対局経路が GPU で通ることの確認 + TRT キャッシュ温め)
!maou selfplay --games 1 --playouts 800 --max-moves 64 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache
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
    --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache \
    --output /content/ab_budget.jsonl
```

判定:

- **A が有意に勝つこと** (`paired` の t 値が明確に正，`A ahead in` が過半)．
  勝たない場合は以降の A/B を回しても意味がないので，先に原因を調べる．
- `A Elo` を 2 doubling (800 vs 200) で割った値が，その予算域での
  1 doubling あたり Elo．§4・§6 の期待値計算に使う．

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
    --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache \
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

## 5. 未決 5: バッチ aggregator の採否

同時対局数を振って **wall clock ベースの playouts/秒** (`throughput:` 行) を
比較する．CPU では評価器の `Mutex<Session>` が上限で `parallel 1/2/4` が
`64/65/65 playouts/秒` (完全に頭打ち) だった．

```python
for p in (1, 2, 4, 8):
    !maou selfplay --games 8 --parallel {p} --playouts 800 --max-moves 120 \
        --model-path /content/model_fp16.onnx --tensorrt --cuda \
        --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache --quiet
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

**結論: 現行構成では aggregator を採用しない．ただし伸びしろは確認された**．
並列は **1.26 倍で頭打ち**で判定規則の「採用検討」側だが，単発の長い探索は
10,909 playouts/秒 出るため律速は GPU 飽和ではなく**評価バッチの充填**．
対局をまたいで評価要求をまとめれば約 2 倍の余地がある → **次 campaign の
課題として起票**．なお **threads 2 はどの並列度でも約 4 割遅い** (短い探索
では探索内スレッド並列がオーバーヘッド)．dfpn 併走のコストは wall +18% /
throughput -15% で，実配置で切る理由はない．

## 6. subtree 再利用の GPU 実挙動 (`--ab-mode subtree`)

CPU では「探索手の 90% で reroot 成功・引き継ぎは playout の 18-20%」が
実測済み (`--parallel 1` / 実モデル)．GPU (TRT) でも同じ発火量になるかを
確認する — mock 評価器では 1.2% しか出ず，mock だけ見ると無効と誤判定する．

```python
!maou selfplay --games 24 --ab-mode subtree --playouts 800 \
    --opening-random-plies 8 --seed 1 --max-moves 256 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache
```

`subtree reuse:` 行の引き継ぎ率が CPU と同水準 (18-20%) なら，CPU で出した
「on 継続」の結論が GPU でも通る．大きく外れたら worklog に記録して原因
(TRT 経路のバッチ待ちで再利用が効かない等) を追う．

## 7. USI プロトコルの headless smoke (GUI なし)

GUI を使わずに確認できる範囲 (`go mate` / ponder / keep-alive / TRT 初回
ビルド中の `readyok` 待ち) はここで潰す．**一括 pipe は使わない** —
`quit` が先に届いて `stop` が立ち，探索が 0 playout で終わる (既知の罠)．
必ず応答を待ってから次を送る:

```python
%%writefile /content/usi_smoke.py
import subprocess, time

def send(p, line):
    p.stdin.write(line + "\n"); p.stdin.flush(); print(">", line)

def wait(p, token, timeout=600.0):
    """token で始まる行が来るまで読み，途中の行 (空行含む) も表示する."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = p.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited")
        print("<", repr(line.rstrip("\n")))
        if line.startswith(token):
            return line.rstrip("\n")
    raise TimeoutError(token)

p = subprocess.Popen(["maou-usi"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, bufsize=1)
send(p, "usi"); wait(p, "usiok")
for opt in [
    "setoption name ModelPath value /content/model_fp16.onnx",
    "setoption name UseTensorRT value true",
    "setoption name UseCuda value true",
    "setoption name TrtCacheDir value /content/trt_cache",
    "setoption name KeepAlive value 5000",   # 未決 2: 空行の生存通知
    "setoption name USI_Ponder value true",
]:
    send(p, opt)
t0 = time.monotonic(); send(p, "isready"); wait(p, "readyok")
print(f"isready took {time.monotonic() - t0:.1f}s")   # TRT 初回ビルド込み

send(p, "position startpos")
send(p, "go btime 30000 wtime 30000 binc 500 winc 500")
best = wait(p, "bestmove")            # "bestmove <手> [ponder <予想手>]"
tokens = best.split()
assert len(tokens) >= 4, f"ponder 予想手が付かない: {best}"
# GUI と同じ手順: 自分の手 + 予想した相手の手を進めてから go ponder
send(p, f"position startpos moves {tokens[1]} {tokens[3]}")
send(p, "go ponder btime 30000 wtime 30000 binc 500 winc 500")
time.sleep(2.0); send(p, "ponderhit"); wait(p, "bestmove")

# go mate: 先手 5三歩 + 持駒金 / 後手 5一玉 = G*5b の 1 手詰め
send(p, "position sfen 4k4/9/4P4/9/9/9/9/9/9 b G 1")
send(p, "go mate 10000"); wait(p, "checkmate")

send(p, "quit"); p.wait(timeout=30)
print("smoke ok")
```

```python
!python /content/usi_smoke.py
```

確認項目:

- `isready` の所要 (TRT 初回ビルド) と，その間に **空行が流れているか**
  (`KeepAlive` の実動作．GUI が無害に無視するかは実機でしか判らない = 未決 2)．
- `bestmove ... ponder ...` が付くこと，`ponderhit` 後に `bestmove` が返ること．
- `go mate` が `checkmate <手順>` を返すこと．

## 8. GUI 実機検証 (将来課題 — 未実施)

**現時点で GUI を動かせる環境がないため未実施**．Colab では GUI (将棋所 /
ShogiGUI / ShogiHome) を動かせないので，§7 の headless smoke で代替できない
項目だけがここに残る．

環境要件: GUI を動かせるデスクトップ環境 (Windows / Linux)．GPU は必須では
ないが，TRT 初回ビルドの待ち時間を実機で見るには GPU 機が望ましい．

チェックリスト:

- [ ] エンジン登録 (`maou-usi` を引数なしで起動．[usi.md](../../commands/usi.md)
      の登録手順) と `usi` → `usiok` → option 一覧の表示．
- [ ] **`KeepAlive` の空行を GUI が無害に無視するか** — 無視するなら既定を
      on にできる (**未決 2 の判断はこれだけが根拠になる**)．壊れる GUI が
      あるなら既定 off のまま，該当 GUI 名を docs に残す．
- [ ] **TRT 初回エンジンビルド中に `readyok` を待てるか** (GUI 側の
      タイムアウトに引っかからないか)．`TrtCacheDir` 指定で 2 回目が短縮
      されることも確認．
- [ ] **`OpeningScript` が実サーバ/GUI 経由で正しく消化されるか** (電竜戦
      HWT の玉往復ハンデ)．指定局面方式で手数付きの局面を渡された場合に
      再発火しないこと (手数 1 ガード) も確認．
- [ ] ponder の実挙動 (GUI が `go ponder` を送るか，`ponderhit` /
      `stop` 後の応答が速いか)．
- [ ] 1 局を最後まで完走 (投了・入玉宣言・千日手の表示が GUI と食い違わない)．

## 8.5 既知の課題 (検証中に判明．未解決)

いずれも worklog/2026-07-27-140516.md に詳細．

- **dfpn の偽証明アラート** — `[dfpn] STRICT VERIFY None (偽証明/不完全)`
  が 40 局で 9 件，同設定の追試で 126 件．**実害は出ていない** (STRICT 検証
  が最終権威なので `Unknown` に落ち，偽の詰みは指されない) が，探索が pn=0 を
  誤って立てる頻度としては看過できない．原因候補は TT 再利用による GHI．
- **終端再訪による探索の空回り** — 引き分け終端・千日手・証明済み局面が
  近いと新しい葉を開かずカウントだけ回り，時間制では予算を空回りに使う．
  実戦 (電竜戦 512 手上限の終盤・千日手模様) でも起きる効率問題．
- **`corrupted double-linked list`** — Colab の `--parallel 4` 実行の
  **終了時**に 1 回発生 (サマリ出力後なので数値は有効)．DevContainer の
  mock・parallel 4 × 12 回では再現せず → ONNX Runtime / TensorRT EP の終了
  処理側が疑わしい．切り分けは `--cuda` のみ (TRT 無し) で再現するか．

## 9. 結果の記録先

- 数値と失敗した試行は `worklog/YYYY-MM-DD-HHMMSS.md` (JST，追記不可)．
- 「再導出しない結論」は `scratchpad/compass.md` の Invariants へ．
- 既定値やドキュメントを変える場合のみ `reviews/*.md` を起票する
  (`CLAUDE.md` / `docs/` を変更するときの必須手順)．
- 未決事項の状態は [index.md §12](index.md) の表を更新する．
