# USI エンジンの検証手順 (GPU / GUI)

[設計本体](index.md) の未決事項のうち，**DevContainer (CPU) では原理的に
閉じられないもの**の手順書．CPU で閉じた項目の根拠は index.md §12 を見ること．

| 残件 | 必要な環境 | 手順 |
|---|---|---|
| 未決 1 TimeStrategy の定数 | GPU (探索速度が要る) | [§4](#4-未決-1-timestrategy-の想定残り手数) |
| 未決 5 バッチ aggregator | GPU | [§5](#5-未決-5-バッチ-aggregator-の採否) |
| 未決 2 keep-alive の既定値 | GUI 実機 | [§8](#8-gui-実機検証-将来課題-未実施) — **将来課題** |

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
--tensorrt --cuda --threads 2 --batch-size 256 \
--trt-cache-dir /content/trt_cache
```

- TensorRT の初回エンジンビルドは **バッチ shape ごとに数十秒〜数分**．
  `--trt-cache-dir` を必ず指定し，同じセッション内で使い回す．
- **計測の前にキャッシュを温める** (§2 の smoke を先に 1 回通す)．初回
  ビルドが計測区間に入ると playouts/秒 が過小に出る．

## 2. 事前確認 — 探索速度の実測 (以降の設定はこの値から決める)

```python
# 1 局面探索の NPS (warmup はエンジンビルドを計測区間から外す)
!maou search --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 2 --batch-size 256 --time-ms 30000 --root-dfpn \
    --trt-cache-dir /content/trt_cache
```

```python
# 自己対局 1 局の smoke (対局経路が GPU で通ることの確認 + TRT キャッシュ温め)
!maou selfplay --games 1 --playouts 800 --max-moves 64 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 2 --batch-size 256 --trt-cache-dir /content/trt_cache
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
    --threads 2 --batch-size 256 --trt-cache-dir /content/trt_cache \
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

1. **早期終了しない** — `reasons` が `resign` / `checkmate` に支配されて
   いない (`--resign-value 0` で投了を切る)．
2. **`min_think` に張り付かない** — `総 playouts ÷ 総手数` が
   `NPS × min_think` を十分上回る (張り付くと両者とも同じ最小予算になり，
   horizon の違いが消える)．
3. **時計が実際に効く** — `time left at end` が初期持ち時間の大半を残して
   いない (残しているなら「多く使う側が単純に得」なだけでトレードオフが
   発生していない)．

初期持ち時間の目安:

```
初期持ち時間 [秒] ≈ horizon × (目標 playouts/手 ÷ NPS)
```

例: `NPS = 2000`，目標 1600 playouts/手，horizon 40 → 32 秒 + 加算 0.5 秒．

### 4.2 実行

```python
!maou selfplay --games 40 --ab-mode horizon \
    --clock-ms 32000 --inc-ms 500 --horizon 40 --horizon-b 20 \
    --resign-value 0 --max-moves 256 --opening-random-plies 8 --seed 1 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 2 --batch-size 256 --trt-cache-dir /content/trt_cache \
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

## 5. 未決 5: バッチ aggregator の採否

同時対局数を振って **wall clock ベースの playouts/秒** (`throughput:` 行) を
比較する．CPU では評価器の `Mutex<Session>` が上限で `parallel 1/2/4` が
`64/65/65 playouts/秒` (完全に頭打ち) だった．

```python
for p in (1, 2, 4, 8):
    !maou selfplay --games 8 --parallel {p} --playouts 800 --max-moves 120 \
        --model-path /content/model_fp16.onnx --tensorrt --cuda \
        --threads 2 --batch-size 256 --trt-cache-dir /content/trt_cache --quiet
```

判定規則:

- **スケールする** (parallel 4 で 3 倍以上) → GPU では Mutex 直列化が上限に
  ならない ⇒ **aggregator は不要** (未決 5 を「見送り」で確定)．
- **頭打ち** (parallel 4 で 1.5 倍未満) → GPU が遊んでいる ⇒ **採用検討**．
  併せて `nvidia-smi dmon` などで GPU 利用率を見て「直列化で遊んでいる」
  ことを確認してから，次 campaign の課題として起票する．
- 中間 (1.5〜3 倍) は `--batch-size` を振って上限がバッチ側か直列化側かを
  切り分ける (バッチを上げて改善するなら aggregator の余地がある)．

## 6. subtree 再利用の GPU 実挙動 (`--ab-mode subtree`)

CPU では「探索手の 90% で reroot 成功・引き継ぎは playout の 18-20%」が
実測済み (`--parallel 1` / 実モデル)．GPU (TRT) でも同じ発火量になるかを
確認する — mock 評価器では 1.2% しか出ず，mock だけ見ると無効と誤判定する．

```python
!maou selfplay --games 24 --ab-mode subtree --playouts 800 \
    --opening-random-plies 8 --seed 1 --max-moves 256 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 2 --batch-size 256 --trt-cache-dir /content/trt_cache
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

## 9. 結果の記録先

- 数値と失敗した試行は `worklog/YYYY-MM-DD-HHMMSS.md` (JST，追記不可)．
- 「再導出しない結論」は `scratchpad/compass.md` の Invariants へ．
- 既定値やドキュメントを変える場合のみ `reviews/*.md` を起票する
  (`CLAUDE.md` / `docs/` を変更するときの必須手順)．
- 未決事項の状態は [index.md §12](index.md) の表を更新する．
