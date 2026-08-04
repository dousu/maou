---
title: docs/ 配下に残る Poetry 前提の記述を uv に揃える
date: 2026-08-04
status: applied
applied_in: 3651dd2
target:
  - docs/dependency_management.md
  - docs/visualization/usage.md
  - docs/browser-automation.md
  - docs/testing-guide.md
  - docs/code-quality.md
  - docs/visualization/api.md
  - docs/performance.md
  - docs/newcomer-guide.md
  - docs/visualization/design.md
  - docs/visualization/UI_UX_REDESIGN.md
  - docs/git-workflow.md
  - docs/command_design_template.md
  - docs/commands/learn_model.md
  - docs/commands/pre_process.md
  - docs/commands/hcpe_convert.md
  - docs/commands/utility_benchmark_dataloader.md
risk: low
reversibility: trivial
---

# 提案: `docs/` 配下の Poetry 前提記述を uv に揃える

## 背景

このプロジェクトは uv 管理である．根拠:

- `pyproject.toml` に `[tool.uv]` / `[tool.uv.sources]` / `[[tool.uv.index]]` があり，
  `[tool.poetry.*]` は**一切存在しない**
- ロックファイルは `uv.lock`．`poetry.lock` は存在しない
- `.pre-commit-config.yaml` は `astral-sh/uv-pre-commit` の `uv-lock` フックと
  `uv run pytest` / `uv run mypy` を使う
- `scripts/dev-init.sh` が `uv sync --extra cpu --extra visualize --group dev` を実行する
- devcontainer に poetry は入っていない (`which poetry` が空)
- `CLAUDE.md` も uv で統一されている

にもかかわらず `docs/` 配下には Poetry 前提の記述が **112 行 / 21 ファイル**残っている
(2026-08-04 時点，`git grep -ic poetry -- docs/ scripts/ .claude/`)．
`docs/dependency_management.md` に至っては本文全体が Poetry の extras 解説であり，
**そこに書かれた `pyproject.toml` の断片は現在の実ファイルと構造が違う**．
新規参加者が `docs/newcomer-guide.md` や `docs/dependency_management.md` の手順を
そのまま実行すると `poetry: command not found` で最初につまずく．

同じ不整合のうち `README.md` / `AGENTS.md` / `.codex/` (計 6 ファイル) は
`docs/` 外のため本提案とは別に既に修正済み．本提案は `docs/` ゲート対象分を扱う．

## 変更内容

対象を 3 分類する．**すべてを一律 sed してはならない**のが本提案の要点．

### Tier A — 手順書 (機械的置換, 15 ファイル / 約 88 行)

読者がコピペして実行することを意図した現行手順．`poetry run` → `uv run` の
機械的置換で足りる．

| ファイル | 件数 |
|---|---|
| `docs/visualization/usage.md` | 20 |
| `docs/browser-automation.md` | 19 |
| `docs/testing-guide.md` | 13 |
| `docs/code-quality.md` | 12 |
| `docs/visualization/api.md` | 4 |
| `docs/performance.md` | 4 |
| `docs/newcomer-guide.md` | 4 |
| `docs/visualization/design.md` | 3 |
| `docs/visualization/UI_UX_REDESIGN.md` | 1 |
| `docs/git-workflow.md` | 1 |
| `docs/command_design_template.md` | 1 |
| `docs/commands/*.md` (4 ファイル) | 各 1 |

### Tier B — `docs/dependency_management.md` (全面改稿, 17 行)

置換では済まない．本文の骨格が「Poetry の extras とは」であり，
掲載されている `pyproject.toml` の断片が**現在の実ファイルに存在しない構造**を
説明しているため，読者を能動的に誤らせる:

| ドキュメントの記述 | `pyproject.toml` の実際 |
|---|---|
| `[tool.poetry.dependencies]` に `optional = true` で追加 | `[project.optional-dependencies]` に extra 名で直接列挙 |
| `[tool.poetry.extras]` セクションで extra を定義 | 同上 (別セクションは無い) |
| `[[tool.poetry.source]]` で PyTorch index を指定 | `[[tool.uv.index]]` + `[tool.uv.sources]` の `explicit = true` |
| (記述なし) | `[tool.uv] conflicts` で `cpu` / `cuda` / `mpu` の排他を宣言 |
| (記述なし) | `[dependency-groups] dev` (Poetry の `--group dev` とは別機構) |

コマンド対応:

- `poetry install` → `uv sync`
- `poetry install -E cpu -E gcp` / `--extras` → `uv sync --extra cpu --extra gcp`
- `poetry update` → `uv lock --upgrade && uv sync`
- `poetry update パッケージ名` → `uv lock --upgrade-package パッケージ名 && uv sync`
- `poetry add plotly` → `uv add plotly`
- `poetry env info` → `uv python find` (uv はプロジェクト直下に `.venv` を作る)
- L84 のシェル変数 `INSTALL_CMD="poetry install -E $GPU_TYPE -E $CLOUD_PROVIDER"`
  → `uv sync --extra "$GPU_TYPE" --extra "$CLOUD_PROVIDER"`

あわせて，本文の extra 内容の記述も実態とずれているため直す
(例: `cpu` / `cuda` の説明に `tensorboard` が挙がっているが
現在の `[project.optional-dependencies]` に `tensorboard` は無い．
また `cpu-infer` / `onnx-gpu-infer` / `tensorrt-infer` / `fetch` の 4 extra が未記載)．

### Tier C — 時点記録 (書き換えない, 3 ファイル / 7 行)

| ファイル | 件数 | 扱い |
|---|---|---|
| `docs/TRAINING_INVESTIGATION_REPORT.md` | 2 | **何もしない** |
| `docs/adr-001-dataloader-multiprocessing-optimization.md` | 4 | 時点注記のみ |
| `docs/adr-002-disk-based-preprocessing-with-sqlite.md` | 1 | 時点注記のみ |

`reviews/2026-08-04-training-report-stale-commands.md` (applied, `ccc6d29`) で
確立した先例に従う:

> 調査レポートは「その時点で何を実行したか」の記録であり，コマンドを
> 書き換えると**再現性の記録が壊れる**．

ADR も「✅ Accepted - 2025-06-17 実装完了」「✅ Accepted - 2025-10-13 実装完了」と
明記された決定記録であり，同じ理由で本文は書き換えない．
冒頭に「当時のコマンドであり現在は uv を使う」旨の注記ブロックを 1 つ足すに留める．

`docs/TRAINING_INVESTIGATION_REPORT.md` は `ccc6d29` で既に注記済みのため
追加作業は不要 (残る 2 行はその注記が意図的に保存している本文中のコマンド例)．

### 本提案の対象外 (`docs/` 外につきゲートなし)

同じ campaign だが review 承認を待たずに直せる．Tier A と同じ機械的置換:

- `scripts/verify_bce_training.py` (1 行, docstring 内の実行例)
- `.claude/skills/visualize-screenshot-checker/SKILL.md` (1 行)

## 選択の理由

**一律 sed 案 (A) を却下した．** `docs/dependency_management.md` の
`pyproject.toml` 断片は `poetry` 文字列を含まない行にも誤りが波及しており
(セクション名・`optional = true` の作法・index 指定方法)，
`poetry` → `uv` の置換だけでは「uv と書かれた Poetry の説明」という
より悪い状態になる．また ADR / 調査レポートまで置換すると Tier C の
先例に正面から反する．

**Tier B を別 PR に分ける案 (B) も検討したが却下した．** 分けると
「uv に統一済み」と読める中途状態が残り，`dependency_management.md` だけが
Poetry のまま取り残される期間が生じる．112 行はまとめて処理できる規模である．

## リスク

- **低**: ドキュメントの記述のみ．`src/` / `rust/` は触らないため
  バージョン bump は不要．挙動に影響しない．逆行は容易
- Tier B の改稿内容は `pyproject.toml` の現状 (2026-08-04, `version = "0.76.0"`) に
  依存する．extra 構成が変われば追随が必要
- `docs/commands/*.md` は `scripts/check-cli-docs.sh` (pre-commit フック) の
  検査対象．置換後にフックが通ることを確認する

## 検証

```bash
# Tier A/B 適用後: docs/ 配下に残る poetry は Tier C の注記文言のみになるはず
git grep -in "poetry" -- docs/

# CLI ドキュメント整合性フック
uv run pre-commit run check-cli-docs --all-files
```
