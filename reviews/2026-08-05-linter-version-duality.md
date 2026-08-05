---
title: linter/formatter の版が pre-commit と dev 依存で二重化する問題を文書化する
date: 2026-08-05
status: pending
applied_in:
target:
  - docs/code-quality.md
risk: low
reversibility: trivial
---

# 提案: linter/formatter の「版の二重化」を文書化する

## 背景

`reviews/2026-08-04-ruff-single-linter.md` (applied, `b737720`) で
lint/format を ruff に一本化した際，コードレビューで実バグが検出された:

`.pre-commit-config.yaml` の `ruff-pre-commit` は **hook 専用の隔離 venv に
自前で ruff を入れる**．一方 `pyproject.toml` の dev 依存にも ruff がある．
この 2 つは独立に版が決まるため食い違いうる:

| | 版を決めるもの | 当時の版 |
|---|---|---|
| pre-commit hook | `.pre-commit-config.yaml` の `rev` | 0.16.1 |
| `uv run ruff` | `pyproject.toml` / `uv.lock` | 0.15.2 |

ruff は `select` を明示していない場合 **適用ルール集合 = ruff の版**である．
0.16.1 の既定は E402 を含まないが 0.15.2 の既定 (E4/E7/E9/F) は含むため，
0.16.1 基準で `# noqa: E402` を剥がした結果，
**`uv run ruff check` (0.15.2) だけが 9 件のエラーを出す**状態になった．
これは CLAUDE.md・AGENTS.md・docs/code-quality.md が
「コミット前に実行せよ」と書いている当のコマンドである．

## この問題の一般性

**ruff 固有ではない．** pre-commit の `repo:` 指定フックは原則として
自分専用の環境にツールを入れるため，同じツールを dev 依存にも持つと
必ず二重化する．挙動が版に依存するツールほど影響が大きい:

- **linter**: 既定ルールセットが版で変わる (ruff, pylint)
- **formatter**: 整形結果が版で変わる (black, ruff format)
- **型チェッカ**: 推論と既定の厳格さが版で変わる (mypy, pyright)

このリポジトリでは `mypy` / `pytest` を最初から `repo: local` +
`uv run` で回しており，二重化していない．今回 ruff だけが
`repo: https://...` 形式で例外的に二重化していた．

## 変更内容

`docs/code-quality.md` に節を追加する．内容:

1. **原則**: 挙動が版に依存する linter/formatter/型チェッカは
   `repo: local` + `uv run <tool>` で回し，版は `pyproject.toml` に一本化する
2. **理由**: `repo:` 形式は hook 専用環境にツールを入れるため
   dev 依存と二重化し，版がずれると同じコードに対する結果が食い違う
3. **副次的な利点**: local hook には `rev` が無いため
   `pre-commit autoupdate` の対象外になり，版上げは
   `pyproject.toml` の差分としてのみ現れる (レビュー可能になる)
4. **コスト**: `uv run` 経由の起動オーバーヘッドは実測 約 90ms/回
5. **例外**: 版に挙動が依存しない汎用フック
   (`trailing-whitespace`, `check-yaml` 等) は `repo:` 形式のままでよい

## 選択の理由

**`pre-commit autoupdate --repo` によるホワイトリスト方式 (案 A) を却下した．**
bot による版上げは止められるが，**二重化そのものは残る**ため，
人間が `pyproject.toml` の版を上げた場合や `uv lock --upgrade` を打った場合に
同じズレが再発する．またホワイトリストなので repo を追加するたびに
workflow 側にも追記が必要で，忘れると黙って更新対象から外れる．

**版一致を検査するスクリプトを足す案 (案 B) も却下した．**
一度実装したが，自ら作った二重化を検査で守る構図になる．
local hook なら二重化自体が消えるため検査が不要になる．

## リスク

- **低**: ドキュメントの追記のみ
- local hook 化そのものは `reviews/2026-08-04-ruff-single-linter.md` の
  適用範囲外だったため，本提案と同一 PR でコード側も変更する

## 検証

```bash
# hook と直接実行で同じ結果になること
uv run pre-commit run ruff-check --all-files
uv run ruff check src/ tests/ scripts/
```
