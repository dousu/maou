# Code Quality Guide

## lint / format は ruff に一本化している

整形・lint・import 順のすべてを ruff が担当する．
**isort と flake8 は 2026-08-04 に廃止した** (`reviews/2026-08-04-ruff-single-linter.md`)．

- isort は ruff の `I` ルールと同時に有効化できない
  (19-22 ファイルで整形が食い違い，pre-commit 上で互いに書き換え合う)
- flake8 は pre-commit に未登録で，かつ既に fail していた死んだゲートだった

### linter/formatter は local hook で回す (版の二重化を避ける)

**原則: 挙動が版に依存するツール (linter / formatter / 型チェッカ) は
`repo: local` + `uv run <tool>` で回し，版は `pyproject.toml` に一本化する．**

pre-commit の `repo: https://...` 形式のフックは，**hook 専用の隔離環境に
自前でツールを入れる**．同じツールを dev 依存にも持つと版が二重化し，
独立に動くため食い違う．

2026-08-04 に実際に踏んだ例:

| | 版を決めるもの | 当時の版 |
|---|---|---|
| pre-commit hook | `.pre-commit-config.yaml` の `rev` | 0.16.1 |
| `uv run ruff` | `pyproject.toml` / `uv.lock` | 0.15.2 |

ruff は `select` 未指定なら**適用ルール集合 = 版**である．0.16.1 の既定は
E402 を含まないが 0.15.2 の既定は含む．0.16.1 基準で `# noqa: E402` を
剥がした結果，**`uv run ruff check` だけが 9 件のエラーを出す**状態になった．
これはこのドキュメントが「コミット前に実行せよ」と書いている当のコマンドである．

**ruff 固有の問題ではない．** black / pylint / mypy / pyright など，
既定ルールや推論が版で変わるツールはすべて同じ構図になる．
本リポジトリは `mypy` / `pytest` を最初から local hook で回しており，
ruff だけが例外的に二重化していた．

副次的な利点として，local hook には `rev` が無いため
`pre-commit autoupdate` の対象外になり，版上げは `pyproject.toml` の
差分としてのみ現れる (bot が黙って上げることがなくなる)．
`uv run` 経由の起動オーバーヘッドは実測で約 90ms/回．

**Rust も同じ原則で回している．** `cargo fmt --all` と
`cargo clippy --workspace --all-targets -- -D warnings` は 2026-08-12 に
local hook として登録した．hook 側が独自に Rust toolchain を入れると
`rust-toolchain` の版と二重化し，ローカルの `cargo fmt` / `cargo clippy` と
pre-commit の結果が食い違う — ruff で踏んだのと同じ構図である．
**版の一本化先だけが Python と異なり，Rust では `pyproject.toml` ではなく
toolchain の指定になる．**

補足が 3 点ある．

- `cargo fmt` を `--check` ではなく自動整形にしてあるのは Python 側の
  `ruff format` に揃えるため (未整形を残すと，無関係な PR に整形差分が
  混ざる)．
- clippy は `-D warnings` で回す．落ちないゲートは守られないためで，
  warning 0 を維持できなくなったときは閾値を緩めるのではなく，warning を
  消すか `#[allow]` に理由コメントを付けて明示する．`--all-targets` を
  付けているのはテストコードも対象にするためで，導入時に見つかった
  warning 4 件はすべてテスト側にあった (lib だけ見ていると気付けない)．
- `cargo fmt` / `cargo clippy` はファイル引数ではなく crate 単位で動くので
  `pass_filenames: false` が必須．

`trailing-whitespace` や `check-yaml` のように**挙動が版に依存しない**
汎用フックは `repo:` 形式のままでよい．

### ruff の版を上げるときの必須手順

ruff の適用ルールは**デフォルト集合に追随している** (`pyproject.toml` に
`select` を書いていない)．これはデフォルトがファミリ全体ではなく厳選された
サブセットであり (`E` は E722/E902 のみ，`D` は D419 のみ など)，
`select` で再現しようとすると 413 に対し 586 ルールと大幅に超過するため．

デフォルト集合はマイナー更新で変わりうる．実際 0.15 → 0.16 では
同一コードに **484 件**が新規検出された．pre-commit の hook 引数が `--fix`
のため，気付かずに上げると**未レビューの一括自動書き換え**が走る
(0.16 では 245 件が 97 ファイルに適用される状態だった)．

MUST 版を上げる前に新旧の有効ルールを比較し，増分を意図的に採否すること:

```bash
# 有効ルール一覧を出して新旧で diff する
uv tool run ruff@<新版> check --show-settings --isolated \
  | sed -n '/linter.rules.enabled/,/^]/p' | grep -oE '\([A-Z]+[0-9]+\)'

# 実際の検出件数を確認する
uv tool run ruff@<新版> check src/ tests/ scripts/ --statistics
```

## Python Tools

### Essential Commands

```bash
# Type checking (required before commits)
uv run mypy src/

# Code formatting (import 順の整列も ruff が担当する)
uv run ruff format src/
uv run ruff check src/ --fix

# Complete quality pipeline (run before commits)
uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/
```

### Pre-commit Hooks
```bash
uv run bash scripts/pre-commit.sh    # Install hooks
uv run pre-commit run --all-files    # Run manually
```

## Code Quality Standards

### Required Standards
- **Type hints**: Required for all functions, methods, and class attributes
- **Docstrings**: Required for all public APIs
- **Line length**: `ruff format` が 64 桁で整形する
  (`[tool.ruff] line-length = 64`)．ただしコメント・文字列・URL は
  formatter が分割できないため超過しうる．`E501` は ruff の既定集合に
  含まれないので**ハードな上限チェックは無い**
  (旧 `.flake8` の `max-line-length = 88` は 2026-08-04 の flake8 廃止で失効)
- **Function size**: Functions must be focused and small
- **Architecture**: Follow Clean Architecture dependency rules

## Pre-commit Hook Enforcement

**CRITICAL:** NEVER skip pre-commit hooks when running `git push` or `git commit`. The hooks enforce code quality standards and must always run. Do NOT use `--no-verify` flag unless explicitly requested by the user.

## Error Resolution Order

When encountering CI failures, resolve issues in this order:

1. **Code Formatting**: `uv run ruff format src/ && uv run ruff check src/ --fix`
2. **Type Errors**: `uv run mypy src/`
3. **Linting Issues**: `uv run ruff check src/`
4. **Test Failures**: `uv run pytest --tb=short`
