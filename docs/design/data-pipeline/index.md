# CLI コマンド間データパイプライン

## この文書の役割

`maou` の各 CLI コマンドは単体で見ると「入力パスを受け取り出力ディレクトリ
へ書く」だけに見える．実際には **あるコマンドの出力が別のコマンドの入力で
ある** 有向グラフを構成しており，効率性・一貫性の問題の多くはコマンド単体
ではなくこの **辺 (edge)** の上に存在する．

個別コマンドの仕様は [docs/commands/](../../commands/) にある．本書は
それらを繋ぐ辺だけを扱い，コマンド単体の説明は繰り返さない．
`CLAUDE.md` § "Data Pipeline" は**形式** (Arrow IPC / Polars / Rust I/O)
を述べるもので，本書とは別の層を扱う．

## パイプライングラフ

```
utility fetch-floodgate --output-dir D
    → D/YYYY/MM/DD/*.csa
        ├→ utility split-kifu --input-path D --train-dir/--val-dir
        │      → 同じ .csa の train/val ミラーツリー (copy|symlink|hardlink)
        └→ hcpe-convert --input-path D --input-format csa

selfplay --kifu-dir K
    → K/game_{NNNN}.csa
        └→ hcpe-convert / analyze-game / analyze-gui

hcpe-convert --output-dir H
    → H/hcpe_chunk{NNNN}.feather                     [schema: hcpe]
        ├→ pre-process --input-path H
        ├→ utility generate-stage2-data --input-path H
        ├→ utility search-values --input-path H
        └→ visualize --array-type hcpe

utility search-values --output-path S.feather
    → S.feather (id, searchWinRate, playouts, stop)
        └→ pre-process --search-value-path S.feather   ※学習側のみ

pre-process --output-dir P
    → P/transformed_chunk{NNNN}.feather              [schema: preprocessing]
        ├→ learn-model --stage3-data-path / --stage3-validation-data-path
        ├→ build-game-graph --input-path P
        ├→ utility benchmark-dataloader / benchmark-training
        └→ visualize --array-type preprocessing

utility generate-stage1-data --output-dir S1
    → S1/stage1_data.feather                         [schema: stage1]
        └→ learn-model --stage1-data-path S1

utility generate-stage2-data --input-path H --output-dir S2
    → S2/stage2[_chunk{NNNN}].feather                [schema: stage2]
        └→ learn-model --stage2-data-path S2

build-game-graph --output-dir G
    → G/nodes.feather, G/edges.feather, G/metadata.json
        └→ visualize --input-path G --array-type game-graph

learn-model --model-dir M
    → M/model_{id}_{tag}_{epoch}.onnx (+ _fp16.onnx, + 分割 .pt)
        ├→ usi / selfplay / floodgate / search / evaluate / analyze-game
        └→ utility search-values --model-path      ← ループを閉じる辺

analyze-game --output R.json
    └→ analyze-gui --report R.json
```

ループは `learn-model` → `search-values` → `pre-process` → `learn-model`
で閉じる．学習済みモデルで探索した勝率を教師信号に混ぜ直す経路であり，
本パイプラインで唯一の巡回辺である．

## 成果物一覧

| 成果物 | 生成 | 命名・レイアウト | 消費 | 決定箇所 |
|---|---|---|---|---|
| Floodgate 棋譜 | `utility fetch-floodgate` | `<output-dir>/YYYY/MM/DD/*.csa` | `split-kifu`, `hcpe-convert` | `app/fetcher/floodgate_fetcher.py:203-205` |
| 自己対局棋譜 | `selfplay --kifu-dir` | `game_{NNNN}.csa` | `hcpe-convert`, `analyze-game` | `infra/console/selfplay.py:640-652` |
| HCPE | `hcpe-convert --output-dir` | `hcpe_chunk{NNNN}.feather` (元の個別 `.feather` は merge 後に削除) | `pre-process`, `generate-stage2-data`, `search-values`, `visualize` | `app/converter/hcpe_converter.py:200-203,217-242`; パターンは `rust/maou_io/src/arrow_io.rs:224` |
| search value | `utility search-values --output-path` | 単一ファイル．拡張子は `.feather`/`.arrow` のみ許可．一時ファイル + `os.replace` の原子的書き込み | `pre-process --search-value-path` | `app/pre_process/search_value.py:144,718-728,866-870` |
| preprocessing | `pre-process --output-dir` | `transformed_chunk{NNNN}.feather` | `learn-model`(stage3), `build-game-graph`, `benchmark-*`, `visualize` | `app/pre_process/hcpe_transform.py:263,574` |
| stage1 | `utility generate-stage1-data` | `stage1_data.feather` (固定名) | `learn-model --stage1-data-path` | `app/utility/stage1_data_generation.py:54` |
| stage2 | `utility generate-stage2-data` | 単一なら `stage2.feather`，分割時は `stage2_chunk{NNNN}.feather` | `learn-model --stage2-data-path` | `app/utility/stage2_data_generation.py:29,274-280` |
| game graph | `build-game-graph --output-dir` | `nodes.feather` / `edges.feather` / `metadata.json` | `visualize --array-type game-graph` | `interface/game_graph_io.py:21-23` |
| モデル | `learn-model --model-dir` | `model_{id}_{tag}_{epoch}.onnx`，`…_fp16.onnx`，分割 `.pt` | 全エンジン系コマンド + `search-values --model-path` | `app/learning/model_io.py:448,544` |

すべてのデータ成果物は Arrow IPC (LZ4 圧縮)．書き込みは
`domain/data/rust_io.py` の `save_*_df` を経由する
(唯一の例外は `search_value.py:868` の直接 `write_ipc`)．
列定義は `domain/data/schema.py` が単一の出所であり，
ファイルと schema を結びつける `array_type` リテラルは
`infra/file_system/file_data_source.py` が正本
(`docs/architecture.md:158-160`)．

## 典型的なエンドツーエンド手順

```bash
# 1. 棋譜取得 → train/val 分割
maou utility fetch-floodgate --start-date 2024-01-01 --end-date 2024-12-31 \
    --output-dir data/kifu
maou utility split-kifu --input-path data/kifu --ext .csa \
    --train-dir data/kifu_train --val-dir data/kifu_val

# 2. 棋譜 → HCPE
maou hcpe-convert --input-path data/kifu_train --input-format csa \
    --output-dir data/hcpe_train

# 3. (任意) 学習済みモデルで探索勝率を収集
maou utility search-values --input-path data/hcpe_train \
    --model-path models/model_x.onnx --output-path data/search_values.feather

# 4. HCPE → preprocessing
maou pre-process --input-path data/hcpe_train --output-dir data/pre_train \
    --search-value-path data/search_values.feather

# 5. 補助ステージのデータ
maou utility generate-stage1-data --output-dir data/stage1
maou utility generate-stage2-data --input-path data/hcpe_train \
    --output-dir data/stage2

# 6. 学習
maou learn-model --stage all \
    --stage1-data-path data/stage1 --stage2-data-path data/stage2 \
    --stage3-data-path data/pre_train --model-dir models
```

`--search-value-path` は学習データ側にのみ適用する．検証データへ同じ
探索勝率を混ぜると評価が汚染される
([utility_search_values.md](../../commands/utility_search_values.md))．

## 本書の鮮度を保つ規約

本書は列挙を含むため，放置すれば `docs/` の他の設計文書と同じように
腐る．腐敗を**検出可能**にするため，次を守る．

1. **辺は必ず「生成コマンド → 成果物 → 消費コマンド」の三つ組で書く．**
   成果物だけ，コマンドだけの記述は検証できない．
2. **命名・レイアウトの主張には `file:line` を添える．** 主張に真偽値が
   あることが，`/audit-and-fix` step 4b がこれを検証できる条件である．
3. **新しいコマンド・成果物・ステージを足したら本書に辺を足す．**
   欠けた辺は step 2.5e の finding であり，コマンド側を直す理由には
   ならない．
4. 本書は**辺だけ**を扱う．コマンドのオプション表は
   `docs/commands/<command>.md` の管轄であり，ここに複製しない
   (複製は必ず片方が先に腐る)．
