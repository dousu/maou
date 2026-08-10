---
status: pending
applied_in:
date: 2026-08-10
target:
  - docs/design/data-pipeline/index.md
  - CLAUDE.md
  - audits/README.md
risk: low
reversibility: trivial (新規1ファイル + 2箇所の追記)
---

# CLI コマンド間データパイプライン文書の新設と，監査記録への sweep 節の追加

## Trigger

ユーザーから `/audit-and-fix` の構造的な穴の指摘があった．

> モジュールを跨ぐコードの一貫性も維持あるいは改善する仕組みが欲しい．
> 学習やデータのパイプラインでの効率性，一貫性，単純化課題の発見もしたい．
> CLI コマンドを跨いだデータの利用を踏まえて最適化する視点が欲しいが，
> 既存の audit-and-fix コマンドでは発見できない穴となっている．
> CLI コマンド間でどのようなデータパイプラインが構築，想定されているか
> ドキュメントがあれば，データパイプライン視点での最適化の難易度が
> 下がるかもしれない．

指摘は正しい．`/audit-and-fix` は path 単位で，Hard constraints に
「Stay inside the target path」があり，steps 1–2 (`/code-review`,
`/simplify`) はいずれも `<path>` 配下のファイルを個別に評価する．
**2つのモジュールの「関係」にしか存在しない欠陥は，どの単一 path 監査から
も見えない**．各コピーは単体では正しいからである．

コマンド本体 `.claude/commands/audit-and-fix.md` は `CLAUDE.md` / `docs/`
配下ではないため本提案の対象外 (`2026-08-09-audit-backlog-command.md`
と同じ整理) であり，step 2.5「Cross-module consistency sweep」の追加は
別途適用済み．本提案はそれが依存する **durable doc 側** 2点を扱う．

## Motivation

step 2.5 を書くにあたり実際に repo 全体を sweep したところ，穴が仮説では
なく実在することが確認できた．以下はすべて `733ff31` で検証済み．

**1. 共有ヘルパーが存在し，呼び出し側が0である．**
`src/maou/infra/console/common.py:164` の
`validate_cloud_provider_exclusivity()` は `__all__` (`:55`) に公開されて
いるが，`src/` `tests/` 全体で呼び出し箇所が **0**．同じ検査が
`hcpe_convert.py:183`，`pre_process.py:363` と `:504`，`utility.py:1163`
に手書きで4回複製され，メッセージは既に分岐している
("Cannot use multiple cloud providers simultaneously." /
"…for input simultaneously." / "…for output simultaneously.")．
ヘルパーが防ぐために書かれたはずの drift が，ヘルパーの外で起きている．
この4パスのどれを単体監査しても発見できない．

**2. 同一概念の出力ディレクトリ初期化が3実装あり，1つが劣化している．**
`interface/converter.py:58` と `interface/preprocess.py:32` の
`output_dir_init()` は docstring まで含めてほぼ同一
(`mkdir(parents=True, exist_ok=True)`)．一方
`interface/learn.py:180` の `dir_init()` は `d.mkdir()` のみで
`parents` も `exist_ok` もない．結果として
`hcpe-convert --output-dir a/b/c` は成功し，
`learn-model --model-dir a/b/c` は `FileNotFoundError` になる．

**3. 入力パス解決の意味論が実際に食い違う．**
共有ヘルパー `infra/file_system/path_utils.py:31` `collect_files()` は
`ext in f.suffixes` で照合し，未ソートで，空でも例外を投げない．
これをバイパスする独自 glob が
`app/pre_process/search_value.py:166`，
`interface/search_value_interface.py:83`，
`app/utility/stage2_data_generation.py:90` にある．
照合規則が違うため**同じ `--input-path` に対して集合が異なる**:
`ext in f.suffixes` は `data.feather.bak` に一致し，
`glob("**/*.feather")` は一致しない (検証済み)．
空入力時の挙動も `ValueError` / `FileNotFoundError` / 無言の空リストの
3通りに分かれている．

**4. パイプライン辺の文書が存在しない．**
`docs/commands/` は20コマンドを個別に網羅しているが index がなく，
「どのコマンドの出力がどのコマンドの入力か」は4辺
(`utility_fetch_floodgate.md:8`，`utility_split_kifu.md:53-78`，
`utility_search_values.md:105-125`，`selfplay.md:26`) にしか書かれて
いない．`CLAUDE.md:15-20` の "Data Pipeline" 節は**形式**
(Arrow IPC / Polars / Rust I/O) の説明であって，コマンドのグラフでは
ない．未記載の辺には，ループを閉じる最重要辺
`learn-model --model-dir` (`.onnx`) → `utility search-values --model-path`
→ `pre-process --search-value-path` が含まれる．

4 は 1–3 と性質が違う．1–3 は「見つけて直す」対象だが，4 は
**探索コストそのもの**である．パイプライン視点の監査は，毎回
「どのコマンドがこの成果物を読むのか」を repo 全体から再導出することから
始まる．その導出結果を committed な文書にしておけば，step 2.5e は
グラフを引くのではなく **グラフを検証する** 作業になり，難易度が下がる．

## Proposed change

### (A) 新規 `docs/design/data-pipeline/index.md`

既存の `docs/design/<topic>/index.md` 規約に従う．全文:

````markdown
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
2. **命名・レイアウトの主張には `file:line` を添える.** 主張に真偽値が
   あることが，`/audit-and-fix` step 4b がこれを検証できる条件である．
3. **新しいコマンド・成果物・ステージを足したら本書に辺を足す.**
   欠けた辺は step 2.5e の finding であり，コマンド側を直す理由には
   ならない．
4. 本書は**辺だけ**を扱う．コマンドのオプション表は
   `docs/commands/<command>.md` の管轄であり，ここに複製しない
   (複製は必ず片方が先に腐る)．
````

### (B) `CLAUDE.md` § "Documentation Links" に1行追加

`| 教師信号の質 (policy/value 改善) | … |` の直後に挿入:

```markdown
| CLI 間データパイプライン | [docs/design/data-pipeline/](docs/design/data-pipeline/index.md) |
```

これにより `/audit-and-fix` step 4a の探索
(「CLAUDE.md's Documentation Links table and every
`docs/design/*/index.md`」) から自動的に到達可能になり，step 2.5e が
「文書があれば先に読む」と書ける根拠になる．

### (C) `audits/README.md` § "Record shape" に1節追加

`## Applied` の直前に挿入:

```markdown
## Cross-module sweep
<step 2.5 で導出した sweep key と，各 key の結果．finding だけでなく
**clean だった key も書く** — 「調べて一貫していた」は次の隣接 path 監査が
同じ Explore sweep を再実行しないための結果である．意図的な分岐は理由と
ともにここに記録する．>
```

理由: sweep の主要な成果物の一つは「調べた結果一貫していた」という
**否定的結果**であり，既存の5節 (Resume point / Applied / Deferred /
Doc findings / Out of scope) のどれにも収まらない．Out of scope に
書くと「path 外の未解決事項」の意味が壊れる．

## Alternatives considered

- **`/audit-pipeline` を別コマンドとして新設する.**
  `/audit-backlog` の前例に倣う案．単位が path でなく「辺」になるので
  概念的には綺麗だが，(a) ユーザーの要求は明示的に
  「audit-and-fix コマンドで」であり，(b) コマンドを増やすと
  `coverage.md` の主表に載らない記録種別がもう1つ増えて
  ledger の運用が重くなる．step 2.5e という **lens** として
  `/audit-and-fix` に内包する方が，既存の out-of-scope backlog へ
  そのまま流せる分だけ軽い．必要になれば後から切り出せる．
- **パイプライン文書を書かず，step 2.5e に毎回グラフを導出させる.**
  動作はする．ただし辺の導出は repo 全体の grep であり，
  監査のたびに同じコストを払う．しかも導出結果は記録されないので，
  次の監査も同じことをする．文書化はこのコストを一度だけにする．
- **`docs/commands/` に index を足すだけにする.**
  安いが，index はコマンドの一覧であって辺のグラフではない．
  「`learn-model` の `.onnx` を `search-values` が読む」のような
  ループ辺は，コマンド一覧からは出てこない．
- **`audits/README.md` を変更せず sweep 結果を Out of scope に書く.**
  README を触らずに済むが，「path 外の未解決 finding」という
  Out of scope の意味が「調べて clean だった key」で薄まり，
  backlog 行の生成規則が曖昧になる．

## What this enables

- `/audit-and-fix` step 2.5e が，グラフを**引く**のではなく
  **検証する**作業になる．パイプライン視点の監査の初期コストが消える．
- 未記載の辺・腐った命名規則が step 4b の通常の drift finding として
  自動的に検出対象になる ((A) の規約2により，主張に真偽値があるため)．
- 新規参加者と将来のセッションが，`--input-path` に何を渡すべきかを
  20個のコマンド文書を横断せずに知れる．
- sweep の否定的結果が記録に残り，隣接 path の監査が同じ
  `Explore` sweep を再実行しなくなる．

## What this constrains

- 新しいコマンド・成果物・パイプラインステージを追加したら
  `docs/design/data-pipeline/index.md` に辺を足す義務が生じる
  (規約3)．これは `docs/commands/` の既存義務と同種の負担である．
- (A) の規約4により，本書はオプション表を持てない．
  パイプライン文書だけ読んでもコマンドは実行できず，
  `docs/commands/` との併読が前提になる．これは意図的な
  トレードオフ (重複は必ず片方が先に腐る) だが，制約ではある．

## Rollback plan

- (A) `git rm docs/design/data-pipeline/index.md`
- (B) `CLAUDE.md` の追加1行を削除
- (C) `audits/README.md` の追加1節を削除

いずれも独立に元へ戻せる．コードには一切影響しない．

## 本提案に含まれないもの (別途対応)

Motivation 1–3 で確認した **実在の不整合そのもの** は，本提案の対象では
ない．これらは durable doc ではなくコードの欠陥であり，
`/audit-and-fix src/maou/infra/console` および
`/audit-and-fix src/maou/interface` を step 2.5 込みで走らせたときに，
通常の finding として triage・backlog 化されるべきものである．
検証済みの `file:line` は上記 Motivation に残してあるので，
その監査は再導出から始めなくてよい．
