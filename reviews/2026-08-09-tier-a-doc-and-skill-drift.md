---
status: applied
applied_in: 3600b32
date: 2026-08-09
target:
  - docs/commands/utility_benchmark_training.md
  - docs/commands/visualize.md
  - .claude/skills/type-safety-enforcer/SKILL.md
  - .claude/skills/qa-pipeline-automation/SKILL.md
risk: low
reversibility: easy
---

# Tier A: benchmark-training の scheduler 行更新と skill の line-length 修正

## Trigger

`audits/coverage.md` の out-of-scope backlog から Tier A (優先度最高)
を消化する作業で発生した2種類のドキュメント編集．

1. **コード修正に追随する doc 更新**．backlog 行
   「[2026-08-08 app/learning] `src/maou/infra/console`」を解消した
   結果，`docs/commands/utility_benchmark_training.md:31` が説明して
   いた欠陥が存在しなくなった．
2. **エージェント向け skill の陳腐化**．backlog 行
   「[2026-08-08 game_graph] `.claude/skills/type-safety-enforcer/SKILL.md`」．
   調査中に同じ誤りが sibling skill にもあることを発見したので併せて
   提案する (backlog は 1 ファイルしか挙げていない)．

## 1. docs/commands/utility_benchmark_training.md:31

### 現状

```
| `--stage12-lr-scheduler CHOICE` | optional | Stage 1/2 ベンチマークの LR スケジューラ．実際に構築できるのは `warmup_cosine_decay` のみ．CLI の `Choice` には `cosine_annealing` / `step` も残っているが，`normalize_lr_scheduler_name` が受理しないため実行時に `ValueError` になる (別名表: `interface/learn.py:66-82`)．未指定なら `--lr-scheduler` を継承． |
```

この行は 2026-08-08 の `app/learning` 監査で「バグを正確に説明する」
形に更新された．コード側を直したので，説明が現実と合わなくなる．

### コード側の変更 (本提案とは別コミット)

`src/maou/infra/console/utility.py:487` の `click.Choice` は
ハードコードした `["warmup_cosine_decay", "cosine_annealing", "step"]`
を捨て，`learn.SUPPORTED_LR_SCHEDULERS` のキーと表示名の両方から
導出するようにした．これで

- 動かない `cosine_annealing` / `step` が消える
- `cosine_annealing_lr` (`SchedulerFactory.create_scheduler`
  (`setup.py:1071`) が実装済み) が初めて指定可能になる
- 既に動いていた `warmup_cosine_decay` は引き続き受理される
- 一覧が別名表から乖離しえなくなる (同じ導出を使う sibling は
  `utility.py:832` の `--lr-scheduler` と `learn_model.py:468` の
  `--stage12-lr-scheduler`)

### 提案

```
| `--stage12-lr-scheduler CHOICE` | optional | Stage 1/2 ベンチマークの LR スケジューラ．`learn.SUPPORTED_LR_SCHEDULERS` から導出され，正準キー (`warmup_cosine_decay`, `cosine_annealing_lr`) と表示名 (`Warmup+CosineDecay`, `CosineAnnealingLR`) の両方を受理する．未指定なら `--lr-scheduler` を継承． |
```

## 2. docs/commands/visualize.md:182

### 現状

```
- **局面統計**: Zobrist Hash，勝率(手番視点)，最善手勝率，深さ，分岐数，
  定跡(一致する定跡がある場合のみ「定跡名(カテゴリ)」形式で表示)
```

### コード側の変更 (本提案とは別コミット)

`src/maou/interface/game_graph_visualization.py` の
`get_opening_name` が，グラフのルートが平手初期局面でない場合に
定跡照合を行わなくなった (`_root_is_startpos`)．定跡データベースの
全エントリは平手初期局面からの指し手列として定義されているため，
`--initial-sfen <中盤局面>` で構築したグラフの初手 `5g5f` が
「先手中飛車」と誤ラベルされていた．

ユーザに見える変化: `--initial-sfen` を指定して構築したグラフでは
定跡行が出なくなる (誤った定跡名が出なくなる)．

### 提案

```
- **局面統計**: Zobrist Hash，勝率(手番視点)，最善手勝率，深さ，分岐数，
  定跡(一致する定跡がある場合のみ「定跡名(カテゴリ)」形式で表示)．
  定跡データベースは平手初期局面からの指し手列で定義されているため，
  `build-game-graph --initial-sfen` で平手以外から構築したグラフでは
  定跡行は表示されない
```

## 3. .claude/skills/type-safety-enforcer/SKILL.md:14

### 現状

```
**Line length**: 88 characters maximum
```

`pyproject.toml:220` は `line-length = 64`．`docs/code-quality.md:96-100`
は 64 を正しく記述し，旧 `.flake8` の `max-line-length = 88` が
2026-08-04 の flake8 廃止で失効したことも注記している．
エージェントはこの skill を読んで作業するため，88 桁を前提に書いた
コードは毎回 `ruff format` で書き換えられる．

### 提案

```
**Line length**: `ruff format` が 64 桁で整形する
(`[tool.ruff] line-length = 64`)．`E501` は ruff の既定集合に含まれない
のでハードな上限チェックは無く，コメント・文字列・URL は超過しうる
(詳細: `docs/code-quality.md`)
```

## 4. .claude/skills/qa-pipeline-automation/SKILL.md (3箇所)

`type-safety-enforcer` を確認した際に見つけた同一の陳腐化．
backlog には記載されていないが，同じ 88 桁の誤りで，同じ影響がある．

| 行 | 現状 | 提案 |
|---|---|---|
| 22 | `Normalize code style to 88-character line limit:` | `Normalize code style to the configured line limit (64):` |
| 55 | `- ✓ Code formatted to 88 characters` | `- ✓ Code formatted to 64 characters` |
| 64 | `**Code Style**: 88-character line limit enforced` | `**Code Style**: 64-character line limit (`ruff format`; ハードな `E501` チェックは無い)` |

## Risk

低い．4ファイルとも記述のみで実行経路に影響しない．
1 と 2 は既にコミット済みのコード変更に追随させるものなので，
適用しないと doc が誤りになる (現状はどちらも「修正前の挙動」として
正しい)．

## Reversibility

容易．いずれも単一行〜数行の差分．

## 判断が必要な点

`.claude/skills/` は CLAUDE.md § MUST rules の
「`CLAUDE.md` / `docs/` を承認済み `reviews/*.md` なしに編集しない」
の文面には含まれない．ただし 2026-08-08 の game_graph 監査記録が
「Needs its own `reviews/` proposal」と判断しているため，それに従って
本提案に含めた．今後 `.claude/skills/` を明示的に対象に含めるか否かは
別途 CLAUDE.md 側で決める余地がある (本提案の範囲外)．
