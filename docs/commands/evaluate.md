# `maou evaluate`

## Overview

- 任意の SFEN 局面をニューラルネットワークで **0 手読み** 評価する．探索は
  行わず，NN 推論を 1 回だけ実行して policy 上位手・評価値・勝率・盤面を出力する．
- **推論は Rust (`maou_search`) が実行する**．Python 側は CLI のオプション
  受け渡しと表示整形だけを担い，モデルのロードも forward pass も行わない
  (`maou search` / `maou usi` / `maou selfplay` と同一の評価器を共有する)．
- ONNX Runtime を静的リンクした Rust wheel が推論を担うため，**CPU 実行に
  追加の Python 依存は不要**．CUDA / TensorRT Execution Provider を使う場合
  のみ extra が要る (下記 § Execution Provider)．

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--sfen STRING` | ✅ | 評価する局面 (駒配置・手番・持ち駒・手数)．Rust の `maou_shogi` が受理する SFEN が使える． |
| `--model-path PATH` | — | ONNX モデルファイルパス．**未指定時は決定論的な mock 評価器**が使われる (API 疎通確認用．出力の中身に意味はない)．`click.Path(exists=True)` が存在を先に検査する． |
| `--num-moves INT` | default `5` | 返す候補手の数 (0 以上)．合法手数より大きい値は合法手数に丸められる． |
| `--cuda/--no-cuda` | default `--no-cuda` | CUDA Execution Provider を有効にする (`onnx-cuda` feature 付き wheel が必要)．`--model-path` が必須． |
| `--tensorrt/--no-tensorrt` | default `--no-tensorrt` | TensorRT Execution Provider を有効にする (`onnx-tensorrt` feature 付き wheel が必要)．`--model-path` が必須． |
| `--trt-cache-dir PATH` | default `None` | TensorRT エンジンキャッシュの保存先．初回のエンジンビルドを次回以降再利用する． |

オプション体系は `maou search` と揃えてある (同じ評価器を同じフラグで駆動する)．

## Execution flow

1. **CLI 検証** — `src/maou/infra/console/evaluate_board.py` が
   `--num-moves` の範囲と，`--cuda` / `--tensorrt` に `--model-path` が
   伴うことを検査する．
2. **interface 委譲** — `src/maou/interface/infer.py` が
   `maou._rust.maou_search.evaluate` を呼ぶ．
3. **Rust 推論** — `rust/maou_rust/src/maou_search.rs` の `evaluate` が
   局面から合法手を生成し，`maou_search::OnnxEvaluator`
   (または mock) で 1 局面を評価する．**推論中は GIL を解放する**．
4. **結果整形** — policy 事前確率の降順に候補手を並べ，勝率を
   `maou_search::eval::winrate_to_eval` で評価値へ変換して返す．Python 側は
   受け取った値を文字列に整形するだけ．

## Policy の規約

候補手は**合法手のみ**で，policy logits を**合法手の中で softmax 正規化**した
事前確率の降順に並ぶ．これは MCTS (`maou search` / USI) が使う事前確率と
同一の規約であり，非合法ラベルが出力に混ざることはない．

出力形式は `<USI 指し手> (<事前確率>)` のカンマ区切り:

```
Policy: 2g2f (0.4158), 7g7f (0.2350), 1g1f (0.0613), 6i7h (0.0604), 3i4h (0.0402)
```

合法手が無い局面 (詰み) では `Policy: (no legal moves)`，`--num-moves 0` を
指定した場合は `Policy: (suppressed; N legal moves)` になる．

## Validation and guardrails

- `--model-path` は `click.Path(exists=True)` で存在を先に検査するため，
  GPU コンテキストや TensorRT エンジンが作られる前に誤りが分かる．
- `--cuda` / `--tensorrt` を `--model-path` なしで指定すると
  `UsageError` になる (mock 評価器に EP は無意味なため)．
- `--num-moves` は `click.IntRange(min=0)` で負値を弾く．
- SFEN の妥当性は Rust 側 (`maou_shogi`) が検証し，駒配置や持ち駒の記述が
  壊れていれば例外になる．
- `onnx` feature 無しでビルドされた wheel で `--model-path` を指定すると，
  ビルド方法を案内する `RuntimeError` になる．
- TensorRT 使用時はプロセス終了時の EP teardown がヒープを壊すため，出力後に
  デストラクタを経由せず終了する (`maou search` と同じ回避策)．

## Execution Provider

| 実行環境 | 追加依存 | 備考 |
| --- | --- | --- |
| CPU | なし | Rust wheel が ONNX Runtime を静的リンクする |
| CUDA | `uv sync --extra onnx-gpu-infer` | `onnx-cuda` feature 付き wheel が必要 |
| TensorRT | `uv sync --extra tensorrt-infer` | `onnx-tensorrt` feature 付き wheel が必要．`--trt-cache-dir` の指定を推奨 |

## 評価値の解釈

### 評価値スコア（Eval）について

評価値スコアは，モデル出力の logit を600倍したものです:

```
eval = 600 × logit = 600 × ln(勝率 / (1 - 勝率))
```

係数600は，Ponanzaという著名な将棋AIで使われていた定数で，将棋AIコミュニティでは標準的に使用されています．この係数により，評価値が人間にとって直感的な範囲（数百～数千）になります．

**スケールの目安**:
- `eval = 0`: 互角（勝率50%）
- `eval = ±600`: やや有利/不利（勝率73%/27%）
- `eval = ±1200`: 有利/不利（勝率88%/12%）
- `eval = ±1800`: 勝勢/敗勢（勝率95%/5%）
- `eval ≥ ±3000`: 勝敗がほぼ決している

勝率 0 / 1 でも発散しないよう内側にクランプするため，評価値は約 ±16570 で飽和します．

この変換は `maou_search::eval::winrate_to_eval` の**単一実装**で，`maou evaluate` / `maou search` / `maou analyze-game` / USI の `score cp` が共有します．

### 評価値と手番の関係

評価値は常に**手番側(現在プレイヤー)の視点**で出力されます．

- 先手番で評価した場合: 正の値 = 先手有利，負の値 = 後手有利
- 後手番で評価した場合: 正の値 = 後手有利，負の値 = 先手有利

同じ局面でもSFENの手番指定(`b` / `w`)を変えると，評価値の符号が反転します．
これは，入力特徴量がニューラルネットワークに渡される前に手番側視点に正規化されるためです．

> [!NOTE]
> 将棋局面の評価としては一貫して先手側からの評価を用います．これによって，モンテカルロツリーサーチ等で先手番で指しているときは正の数値の最大化，後手番で指しているときは負の数値の最大化を目指すことが目標になります．

**注意**: この評価値は将棋の「点数」（駒の価値）とは異なります．あくまで「勝ちやすさ」を表す指標です．

## Example invocation

```bash
# CPU (追加依存なし)
uv run maou evaluate \
  --model-path artifacts/eval.onnx \
  --num-moves 7 \
  --sfen "lnsgkgsnl/1r5b1/p1pppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1"

# TensorRT
uv run maou evaluate \
  --model-path artifacts/eval.onnx \
  --tensorrt --cuda \
  --trt-cache-dir .trt-cache \
  --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
```

出力例:

```
Policy: 2g2f (0.4158), 7g7f (0.2350), 1g1f (0.0613), 6i7h (0.0604), 3i4h (0.0402)
Eval: 279.93
WinRate: 0.6146
後手の持駒：なし
  ９ ８ ７ ６ ５ ４ ３ ２ １
...
```

## Implementation references

- CLI 定義 — `src/maou/infra/console/evaluate_board.py`
- interface アダプタ — `src/maou/interface/infer.py`
- Rust 推論 — `rust/maou_rust/src/maou_search.rs` (`evaluate`),
  `rust/maou_search/src/onnx.rs` (`OnnxEvaluator`),
  `rust/maou_search/src/eval.rs` (`winrate_to_eval`)
- 盤面表示 — `src/maou/domain/board/shogi.py`
