# maou utility search-values

## Overview

floodgate の既存局面をそのまま探索して，**局面ごとの value 教師**を作る．
出力は `maou pre-process --search-value-path` に渡して `resultValue` を
差し替える．

### なぜ要るのか

現在の value 教師は HCPE の `gameResult` から作られるので，
**1 対局に属する約 110 局面が全部同じ 0/1 を持つ**．
「この局面はどの対局のものか」を思い出せば学習データには当たるが，
その近道は未知の対局では 1 ビットも稼げない．

実測でも記憶の痕跡は手数とともに増える (学習期間内と held-out の Brier 比が
ply 0-19 で 1.08，ply 120+ で **2.31**)．中終盤の局面はほぼ一意なので
対局の同定が容易であり，機序と一致する．

**探索値は同一対局の中でも局面ごとに異なる**ので，この近道が効かなくなる．
背景と測定は [docs/design/training-quality/](../design/training-quality/index.md)
§2.5 / §3.3 / §5.3 を参照．

policy 教師 (floodgate の実指し手) には手を触れない．局面の分布も変わらない．

### 選定はラベルと独立

絞り込みには**手数 (`--min-ply`) と重複しか使わない**．
「モデルが外している局面を選ぶ」は能動学習として魅力的だが，学習分布が
モデルの誤りへ偏り較正測定の前提が壊れるので採らない．

同一局面 (同じ Zobrist hash) は 1 回だけ探索する．前処理は hash で集約する
ので 2 回目以降は無駄になる．

### 部分適用できる

局面は Zobrist hash で対応づけられ，これは `maou pre-process` が集約に使う
キーと同じである．**出力に無い局面は対局結果由来の値のまま残る**ので，
GPU 予算に収まる範囲だけ探索すれば良い．

### 蓄積の保証 (`--resume`)

`--resume` を付けて繰り返し実行すれば，**既に探索した局面は二度と探索されず，
重複なく貯まっていく**．

- 走査時に既存出力の hash を除外する
- 同一局面は 1 ファイル内でもファイルをまたいでも 1 回だけ探索する
- 連結時に id で一意化する (重複時は後勝ち)．
  重複したまま前処理へ渡すと**左 join が行を複製して学習データが壊れる**ため，
  `pre-process` 側でも一意化して行数不変を検証する

**`--resume` も `--overwrite` も無いまま既存の出力を指すとエラーになる．**
出力は数日分の GPU 時間そのものなので，取り違えで捨てさせない．
既存の出力が無ければフラグ無しでも `--resume` 単体でも通るので，
**スクリプトには `--resume` を固定で書いておけば初回から中断・再開まで
同じコマンドで済む**．

**`--resume` と `--overwrite` の同時指定はエラー**である．
「続きから」と「作り直し」は両立せず，暗黙の優先順位を持たせると
作り直したつもりの実行で古い値が残る．新しく始めるときは出力を手で消すか
`--overwrite` を 1 回だけ使う．

```bash
# 新しい campaign を始めるとき (1 回だけ．手で消しても同じ)
maou utility search-values ... --overwrite

# 以降は毎回これ (中断したらそのまま再実行)
maou utility search-values ... --resume
```

中断すると最後のフラッシュ以降 (`--flush-interval` 局面ぶんが上限) だけ
再探索する．書き出しは一時ファイル経由の原子的な置換なので，
フラッシュ中にクラッシュしても出力が壊れることはない．

出力を入力ディレクトリの配下へ置いてもよい (走査から自動的に除外される)．

### 注意

HCPE は局面のみを持ち指し手履歴を持たないため，SFEN へ復元した時点で
**千日手の文脈は失われる**．元の対局で千日手絡みだった局面は，
その文脈なしに評価される．

### 検証データには適用しない

**探索値は training 側の前処理にだけ渡す．** 検証側は対局結果 (0/1) のまま
残すこと．理由は 3 つある．

1. **循環する** — 検証の教師がモデル自身の探索出力になると，
   「自分の探索を真似るほど良いスコア」になり汎化の指標として機能しない
2. **North-star とずれる** — 最終目標は実際の対局結果に対する較正である．
   検証を対局結果のままにすれば，検証損失は「未知の対局の実際の勝敗を
   どれだけ当てられるか」を測り続ける
3. **過去の測定と比較できなくなる** — `value_brier_score` は 0/1 教師に対する
   値なので，検証側を変えると Step 2 の基準
   (`docs/design/training-quality/` §2.5 の epoch 11) と比較できなくなり，
   **反証テスト自体が成立しなくなる**

training と validation で教師の意味が違ってよい．検証損失の絶対値は
前回と比べにくくなるが，`value_brier_score` はどちらも 0/1 相手なので
直接比較でき，early stopping も同一 run 内の相対比較なので機能する．

なお `scripts/measure_calibration.py` は HCPE の `gameResult` を直接使い
前処理を経由しないので，held-out 較正の測定はこの選択の影響を受けない．

## Usage

```bash
# GPU で 100 万局面ぶんを探索する
# (中断したら同じコマンドを再実行する — --resume で続きから貯まる)
maou utility search-values \
    --input-path hcpe_train/ \
    --output-path search_values.feather \
    --model-path model.onnx \
    --min-ply 60 --max-positions 1000000 --playouts 800 \
    --batch-size 64 --tensorrt --cuda --trt-cache-dir trt_cache/ --resume

# training 側の前処理にだけ渡す
maou pre-process --input-path hcpe_train/ --output-dir pre_train/ \
    --search-value-path search_values.feather

# validation 側は素のまま (対局結果を教師に残す)
maou pre-process --input-path hcpe_val/ --output-dir pre_val/
```

## CLI options

| Option | Default | Description |
|---|---|---|
| `--input-path PATH` | required | HCPE (`.feather`) のディレクトリまたはファイル．再帰的に走査する． |
| `--output-path PATH` | required | 出力する feather (`id` / `searchWinRate` / `playouts` / `stop`)． |
| `--model-path PATH` | optional | ONNX モデル．未指定なら決定論的な mock 評価器 (API 検証用．**値に意味は無い**)． |
| `--min-ply INT` | `60` | この手数以上の局面のみ対象にする．記憶は中終盤に集中し，序盤の局面は多数の対局で共有されて教師が平均化されるので手を入れる必要が無い． |
| `--max-positions INT` | `0` | 対象局面数の上限 (`0` で無制限)．GPU 予算に合わせる．標本抽出は `--seed` で決まりラベルに依存しない． |
| `--seed INT` | `0` | `--max-positions` が効くときの標本抽出の乱数種． |
| `--playouts INT` | `800` | 1 局面あたりの playout 上限． |
| `--time-ms INT` | optional | 1 局面あたりの時間上限 (ミリ秒)． |
| `--threads INT` | `1` | 探索スレッド数． |
| `--batch-size INT` | `8` | 評価バッチサイズ．GPU では 64 以上． |
| `--root-dfpn / --no-root-dfpn` | `True` | ルート並行 dfpn 詰み探索． |
| `--leaf-mate / --no-leaf-mate` | `True` | MCTS の葉の短手詰み探索． |
| `--cuda / --no-cuda` | `False` | CUDA Execution Provider (`onnx-cuda` 付き wheel が必要)． |
| `--tensorrt / --no-tensorrt` | `False` | TensorRT Execution Provider (`onnx-tensorrt` 付き wheel が必要)． |
| `--trt-cache-dir PATH` | optional | TensorRT エンジンキャッシュ保存先． |
| `--flush-interval INT` | `500` | この局面数ごとに途中結果を書き出す．**中断してもそこまでの結果が残り** `--resume` で続きから再開できる．数十万局面を数日かけて回す用途では必須． |
| `--resume` | `False` | `--output-path` に既にある局面を飛ばして残りを追記する．中断した実行はそのまま再実行できる． |
| `--overwrite` | `False` | 既存の `--output-path` を破棄して作り直す．`--resume` も `--overwrite` も無いまま既存の出力を指すと**エラー**になる．`--resume` との**同時指定もエラー**． |

## 進行状況の確認

長時間かかるので，2 段階とも tqdm の進捗バーを出す．

```
Scanning HCPE: 100%|██████████| 39/39 [00:41<00:00,  1.06s/file]
Searching positions:  12%|█▏  | 122093/1000000 [2:41:07<19:19:02, 12.6pos/s, flushed=122000, mean_wr=0.512]
```

- **Scanning HCPE**: 対象局面の走査 (hash のみ集める段)
- **Searching positions**: 探索本体．**残り時間 (ETA) と 局面/秒**が出る．
  `flushed` は最後に書き出した局面数，`mean_wr` はここまでの探索勝率の平均

`mean_wr` が 0.5 から大きく離れ続ける場合は対象局面の偏りか探索の異常を疑う．

## Output

| カラム | 型 | 内容 |
|---|---|---|
| `id` | `UInt64` | Zobrist hash．前処理出力の `id` と同じキー． |
| `searchWinRate` | `Float32` | 手番側から見た探索の勝率 (0-1)．`resultValue` と同じ規約． |
| `playouts` | `Int32` | 実際に消化した playout 数． |
| `stop` | `String` | 探索の停止理由 (`playout_limit` / `root_proven` など)． |

## Cost

L4 (10,257 playouts/s, batch 64, fp16 TRT+CUDA) で 800 playouts/局面なら
**約 0.078 秒/局面**．木の再利用が効かない独立探索なので自己対局より
1 局面あたりは割高だが，**必要な局面だけ選べる**．

| 局面数 | L4 時間 (800 playouts) | 同 (200 playouts) |
|---|---|---|
| 100k | 2.2 時間 | 0.5 時間 |
| 1M | 22 時間 | 5.4 時間 |
| 19M (ply≥60 の全量) | 411 時間 | 103 時間 |
