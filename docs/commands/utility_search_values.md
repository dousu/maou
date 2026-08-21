# maou utility search-values

## Overview

floodgate の既存局面をそのまま探索して，**局面ごとの value 教師**を作る．
出力は **シャードのディレクトリ** (`part_NNNNNNNN.feather`) で，
`maou pre-process --search-value-path` に渡して `resultValue` を差し替える．

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
    --output-path search_values/ \
    --model-path model.onnx \
    --min-ply 60 --max-positions 1000000 --playouts 800 \
    --batch-size 64 --tensorrt --cuda --trt-cache-dir trt_cache/ --resume

# training 側の前処理にだけ渡す
maou pre-process --input-path hcpe_train/ --output-dir pre_train/ \
    --search-value-path search_values/

# 別々に貯めた出力をまとめて渡すこともできる (--search-value-path は複数指定可)
maou pre-process --input-path hcpe_train/ --output-dir pre_train/ \
    --search-value-path search_values/ --search-value-path search_values_g4/

# validation 側は素のまま (対局結果を教師に残す)
maou pre-process --input-path hcpe_val/ --output-dir pre_val/
```

### 旧形式 (単一 feather) からの移行

0.86.0 より前は `--output-path` が単一ファイルだった．ファイルを渡すと
移行手順付きの**エラー**になるので，ディレクトリへ移してから `--resume` する．

```bash
mkdir -p search_values/
mv search_values.feather search_values/
maou utility search-values --output-path search_values/ --resume ...
```

`--resume` はシャード名に限らず配下の feather を全部読むので，移した旧ファイルは
そのまま既探索として効く．新しい行は `part_00000001.feather` から書かれる．

## CLI options

| Option | Default | Description |
|---|---|---|
| `--input-path PATH` | required | HCPE (`.feather`) のディレクトリまたはファイル．再帰的に走査する． |
| `--output-path PATH` | required | **シャードを書き出すディレクトリ**．確定シャードは `part_NNNNNNNN.feather` (`id` / `searchWinRate` / `playouts` / `stop` / `elapsedMs` / `warmupMs`)．`--flush-interval` ごとの途中結果は小さな `pending_NNNNNNNN.feather` として足され，`--shard-rows` に達した時点で 1 枚の `part_` へまとめられて消える．**累積全体を書き直さない**ので，1 回の書き込みコストが行数によらず一定になる (旧実装は書き込み量が行数の二乗で伸び，18.7M 行では合計 5.9 時間に達した)．単一ファイルを渡すと移行手順付きの**エラー**になる． |
| `--model-path PATH` | optional | ONNX モデル．未指定なら決定論的な mock 評価器 (API 検証用．**値に意味は無い**)． |
| `--min-ply INT` | `60` | この手数以上の局面のみ対象にする．記憶は中終盤に集中し，序盤の局面は多数の対局で共有されて教師が平均化されるので手を入れる必要が無い． |
| `--max-positions INT` | `0` | 対象局面数の上限 (`0` で無制限)．GPU 予算に合わせる．標本抽出は `--seed` で決まりラベルに依存しない． |
| `--seed INT` | `0` | `--max-positions` が効くときの標本抽出の乱数種． |
| `--playouts INT` | `800` | 1 局面あたりの playout 上限． |
| `--time-ms INT` | optional | 1 局面あたりの時間上限 (ミリ秒)． |
| `--threads INT` | `1` | 探索スレッド数． |
| `--batch-size INT` | `8` | 評価バッチサイズ．GPU では 64 以上． |
| `--node-capacity INT` | `--playouts` から導出 (`2 × playouts + 4096`) | 探索木のノードプール容量．ノードは**未展開の子へ降りた playout 1 回につき 1 個**しか確保されないので，必要数は playout 予算で上から押さえられる (実測: `--playouts 800` で `nodes_used` = 801)．Rust 既定の 2^20 は約 1,300 倍にあたり，1 ノード約 48 B なので**約 50MB を局面ごとに確保し直す**．この用途は保持木を引き継がないため毎回払われ，しかも**そのコストは `elapsedMs` でなく `warmupMs` に乗る**．絞っても探索は変わらない — プールの GC は**枯渇したときにしか**走らないため，必要数を上回っていれば木は同一になる (実測で `winrate` / `playouts` / `stop` / 最善手が一致)．上回れているかは要約の `gc_runs` が示す (0 なら不変)． |
| `--root-dfpn / --no-root-dfpn` | `True` | ルート並行 dfpn 詰み探索． |
| `--root-dfpn-nodes INT` | Rust 既定 (2,000,000) | ルート dfpn のノード予算．`--min-ply 60` は戦術的に濃い局面を狙って選ぶので，詰み探索が壁時計を支配しうる．下げて切り分ける． |
| `--root-dfpn-depth INT` | Rust 既定 | ルート dfpn の深さ上限． |
| `--leaf-mate / --no-leaf-mate` | `True` | MCTS の葉の短手詰み探索． |
| `--leaf-mate-nodes INT` | Rust 既定 (50) | leaf-mate 1 回あたりのノード予算． |
| `--leaf-mate-threads INT` | Rust 既定 (1) | leaf-mate 専用スレッド数． |
| `--defensive-mate / --no-defensive-mate` | Rust 既定 | 受け方向の詰み探索 (root 敗着フィルタ)．**局面ごとの CPU 側の仕事**． |
| `--defensive-mate-threads INT` | Rust 既定 | root 敗着フィルタの並列度． |
| `--pad-buckets / --no-pad-buckets` | Rust 既定 (固定 padding) | TensorRT の評価バッチを `--batch-size` へ固定 padding せず 2 冪バケットへ切り上げる．**1 局面 1 探索で毎回 root から立ち上げるので序盤の葉は少なく，固定 padding だと 1 件の評価が `--batch-size` 件分のコストを払う**． |
| `--cuda / --no-cuda` | `False` | CUDA Execution Provider (`onnx-cuda` 付き wheel が必要)． |
| `--tensorrt / --no-tensorrt` | `False` | TensorRT Execution Provider (`onnx-tensorrt` 付き wheel が必要)． |
| `--trt-cache-dir PATH` | optional | TensorRT エンジンキャッシュ保存先． |
| `--flush-interval INT` | `500` | この局面数ごとに途中結果を書き出す．**中断してもそこまでの結果が残り** `--resume` で続きから再開できる．書き出しは `pending_*.feather` を 1 枚足すだけで，既存には触れない． |
| `--shard-rows INT` | `5000000` | 確定シャード 1 枚に収める目標行数．**1 行あたり実測 19.4 B** なので既定は約 97MB．大きくするほどファイル数は減るが，中断時に `pending_*` として残る量も増える．なお**実行の終わりでは目標行数に届かなくても確定させる**ので，1 回の実行につき最低 1 枚の `part_` ができる (端数を持ち越さないのは，`--resume` のたびに `--shard-rows` が同じとは限らず，値が変われば「既存の確定シャードも分割し直すのか」という問題になるため)． |
| `--resume` | `False` | `--output-path` 配下に既にある局面を飛ばして残りを追記する．**前回の実行が残した `pending_*` も引き継いで**次の確定シャードにまとめる．連番は既存の最大値の次から振るので，途中のシャードを手で消しても既存を上書きしない．中断した実行はそのまま再実行できる． |
| `--overwrite` | `False` | 既存の `--output-path` ディレクトリを削除して作り直す．シャードが 1 枚でも生き残ると次の `--resume` が古い値を拾うため，ディレクトリごと消す．`--resume` も `--overwrite` も無いまま既存の出力を指すと**エラー**になる．`--resume` との**同時指定もエラー**． |

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
| `elapsedMs` | `Int32` | 1 局面あたりの探索時間 (ミリ秒)．**律速の切り分け用**．0.82.0 以前の出力には無く null になる． |
| `warmupMs` | `Int32` | 1 局面あたりの**計測区間外**コスト (ミリ秒)．root の同期評価とノードプール確保．**`elapsedMs` と足して初めて 1 局面の総コストになる**．0.96.0 以前の出力には無く null になる． |

## Cost (実測)

**公称 playouts/s からの割り算を根拠にしない．** 当初この節は
「10,257 playouts/s ÷ 800 = 0.078 秒/局面 ⇒ 1M = 22 時間」と書いていたが，
これは compass § TRIPWIRE「公称パラメータを信じない」に反しており，
GPU によって 2.7 倍ずれる．

| GPU | 実測 | 1 局面あたり |
|---|---|---|
| G4 | 1M = **22 時間** (Colab Pro の連続 24 時間に収まる) | 0.079 秒 |
| L4 | 300k = **18 時間** | **0.216 秒** |

**この用途は自己対局と違って木の再利用が効かず，毎回 root から探索を立ち上げる**
ため，公称のスループットは出ない．

### `elapsedMs` だけで外挿しない

上の節は「公称 playouts/s からの割り算を根拠にしない」と言っているが，**実測値を
使ってなお外す道がある** — `elapsedMs` は探索本体だけで，**1 局面の総コストでは
ない**．root の同期評価とノードプールの確保は計測区間の外にあり `warmupMs` に
乗る．そしてこの用途は保持木を引き継がないので，**それは局面ごとに払われる**．

L4 での実測 (2026-08-20，warmup 後の 50 局面，`--playouts 800`):

| dfpn TT プール | `elapsedMs` median | 備考 |
|---|---|---|
| 無効 (`DFPN_TT_POOL_BYTES=0`) | 184.0 ms | 局面ごとの TT 確保 352MB が覆っている |
| 有効 (既定) | **75.5 ms** | 800 playouts なので **10,596 playouts/s** |

有効時の 75.5 ms は `docs/performance.md` の L4 batch 64 = 10,257 playouts/s と
一致する．**184 ms がこれを覆い隠していた**．

ここから `5.3M × 75.5 ms = 111 時間` と出したくなるが，これは**下限**である．
`warmupMs` を足していないためで，真の値は `111 時間 × (1 + warmupMs / 75.5)`
になる．DevContainer (mock 評価器，`--playouts 800`) では:

| `--node-capacity` | `warmupMs` median | `elapsedMs` median |
|---|---|---|
| Rust 既定 2^20 | **23 ms** | 21 ms |
| `--playouts` から導出 (5,696) | **0 ms** | 22 ms |

**局面あたりコストの過半が `elapsedMs` の外にあり，`elapsedMs` だけを見ていると
この 2 倍差はまったく見えない**．導出値が既定になったので現在は前者を踏まない
が，**所要時間を見積もるときは両方を足すこと**．

## 律速の切り分け

出力には `playouts` / `stop` / `elapsedMs` / `warmupMs` を記録しているので，
**本番実行そのものが計測になる**．別途 A/B を組まなくても後から追える．

```bash
uv run python -c "
import polars as pl
import glob
# how='diagonal' は必須: 列は版数で増えてきたので (elapsedMs は 0.82.0,
# warmupMs は 0.97.0)，蓄積の途中で版数をまたいだ出力は列構成が揃わない
d = pl.concat([pl.read_ipc(f, memory_map=False)
               for f in sorted(glob.glob('search_values/*.feather'))],
              how='diagonal')
print(d.group_by('stop').len().sort('len', descending=True).to_dicts())
print('playouts 中央値', d['playouts'].median(),
      '/ 800 未満の割合', (d['playouts'] < 800).mean())
print('elapsedMs 中央値', d['elapsedMs'].median(),
      '/ warmupMs 中央値', d['warmupMs'].median())"
```

- `stop` が `root_proven` 中心 / `playouts` が 800 に届かない
  → **時間は playout でなく詰み探索に行っている**．
  `--root-dfpn-nodes` を下げる，`--no-defensive-mate` を試す
- `playouts` が 800 に張り付いていて `elapsedMs` が大きい
  → **評価バッチが埋まっていない可能性**．`--pad-buckets` /
  `--threads` を上げる / `--batch-size` を下げるを試す
- `warmupMs` が `elapsedMs` に対して無視できない
  → **探索の外に固定費がある**．`--node-capacity` を明示していれば
  外して大きすぎないか見る．残りは root の同期評価 1 回分

いずれも**教師の質を変えうる**ので，速度だけで既定を決めない
(詰みが証明された局面の探索値は 0/1 の真値になり教師として最良)．

`elapsedMs` は 0.82.0 で追加した．**それ以前の出力には無いが，読み込み時に
null で補われるので `--resume` はそのまま継続できる**．
