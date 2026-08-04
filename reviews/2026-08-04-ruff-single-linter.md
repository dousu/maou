---
title: lint/format を ruff に一本化し (isort・flake8 廃止)，ruff デフォルト全ルールを採用する
date: 2026-08-04
status: pending
applied_in:
target:
  - CLAUDE.md
  - docs/code-quality.md
  - docs/git-workflow.md
risk: low
reversibility: moderate
---

# 提案: lint/format を ruff に一本化する

## 背景

`reviews/2026-08-04-poetry-to-uv-docs.md` (applied, `3651dd2`) で
パッケージ管理を uv に統一した．今回はその lint/format 版にあたる．

このリポジトリは同じ責務のツールを 3 つ抱えていた:

| ツール | 状態 (2026-08-04 調査時点) |
|---|---|
| ruff | pre-commit で強制．実質的な唯一のゲート |
| isort | pre-commit で強制．`[tool.isort] profile=black, line_length=64` の 2 設定のみ |
| flake8 | **pre-commit に未登録．かつ既に fail していた** (`analysis_gui.py` の F401 4 件，`path_suggestions.py` の E501) |

flake8 は誰も通していない死んだゲートであり，ユーザ確認により
「完全に廃止している想定だった」ことが判明した．

isort については，ruff の `I` ルールと**同時に有効化できない**ことを実測した．
ruff で 22 ファイルを整形した直後に isort が 17 ファイルを差し戻すため，
pre-commit 上で互いに書き換え合う．

さらに ruff 0.16 のデフォルトルール集合が大幅に拡張され，
同一コードに対し 484 件が新規検出される状態だった
(`reviews` 対象外の `pyproject.toml` 側で `select` 固定により一旦回避済み)．

## 変更内容

**すべて `docs/` / `CLAUDE.md` の記述のみ．コードの変更は本提案の対象外**
(コード側は同一 PR 内で扱う)．

### 1. QA パイプラインから isort を除去

3 ファイルに以下のコマンド列が記載されている:

```
uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/ && uv run mypy src/
```

`uv run isort src/` を除去する．import 順は `ruff check --fix` の
`I` ルールが担当するため，パイプラインの実効内容は変わらない．

| ファイル | 該当 |
|---|---|
| `CLAUDE.md` | L166 (Development Guidelines > Git Workflow) |
| `docs/code-quality.md` | L20, L46 |
| `docs/git-workflow.md` | L7 |

### 2. flake8 の記述を削除

`docs/code-quality.md` の L17 (`uv run flake8 src/`) と
L48 (`3. **Linting Issues**: uv run flake8 src/`) を削除し，
lint は `uv run ruff check src/` に一本化する旨を記す．

### 3. `docs/code-quality.md` に版上げ手順を追記

ruff のデフォルト集合はマイナー更新で変わりうる．
hook 引数が `--fix` のため，放置すると版上げのたびに未レビューの
一括自動書き換えが走る (実際 0.15→0.16 で 245 件が 97 ファイルに適用された)．

以下の検証手順を明記する:

```bash
uv tool run ruff@<新版> check --show-settings --isolated \
  | sed -n '/linter.rules.enabled/,/^]/p' | grep -oE '\([A-Z]+[0-9]+\)'
```

新旧を比較し，増分ルールを意図的に採否してから版を上げる．

## 選択の理由

**isort を残して ruff の `I` を除外する案 (A) を却下した．**
このリポジトリの isort 設定は `profile=black` と `line_length=64` の
2 項目のみで，どちらも ruff 側で表現できる．一方 isort を残すと
`I` の 23 件だけが恒久的に採用できず，「全ルールを採用する」という
今回の目的と正面から矛盾する．移行コストは 19 ファイルの機械的差分
(空行と同一モジュール import の統合) で意味の変更はないと実測済み．

**flake8 を pre-commit に追加して生かす案 (B) も却下した．**
ruff は pyflakes (F) を包含しており，flake8 を生かすと
`# noqa` の管轄が二重化する (実際 `external` 設定での回避を一度試みたが，
ユーザ確認により廃止済みが正であることが判明した)．

## リスク

- **低**: ドキュメントの記述のみ．逆行は容易
- ただし本提案と同一 PR のコード変更 (isort 廃止・ruff 全ルール採用) は
  484 件の指摘対応を含み，逆行は moderate
- `docs/code-quality.md` の版上げ手順は ruff の CLI 仕様
  (`--show-settings`) に依存する．将来 ruff 側で出力形式が変われば追随が必要

## 検証

```bash
# ドキュメントに isort / flake8 が残っていないこと
git grep -in "isort\|flake8" -- CLAUDE.md docs/

# パイプラインが実際に通ること
uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/
```
