---
status: pending
applied_in:
date: 2026-08-12
target: [docs/code-quality.md]
risk: low
reversibility: trivial
---

# `docs/code-quality.md` の local hook 原則が Python のツールしか挙げていない

## Trigger

`audits/coverage.md` § Out-of-scope backlog の NEW-1 行
([2026-08-12 backlog auto-band-and-n1](../audits/2026-08-12-backlog-auto-band-and-n1.md))．
`/audit-backlog` (2026-08-12) で HEAD に対して再検証した．

## 現状

`docs/code-quality.md` §「linter/formatter は local hook で回す
(版の二重化を避ける)」は原則を

> 挙動が版に依存するツール (linter / formatter / 型チェッカ) は
> `repo: local` + `uv run <tool>` で回し，版は `pyproject.toml` に一本化する．

と書き，例として ruff / mypy / pytest を挙げる．2026-08-12 に rustfmt が
**同じ原則の 2 例目**として `.pre-commit-config.yaml` に入った (PR #474)．

`.pre-commit-config.yaml:64-81` には理由が書かれている:

```yaml
# rustfmt も ruff と同じ理由で local hook にする．
# hook 側が独自に Rust toolchain を入れると rust-toolchain の版と
# 二重化し，ローカルの `cargo fmt` と pre-commit の結果が食い違う．
```

しかし doc 側からは，(a) Rust の整形が pre-commit で強制されていること，
(b) 版の一本化先が Rust では `pyproject.toml` ではなく toolchain である
ことのどちらも読み取れない．

## なぜ P2 (drift correction) ではないか

訂正後の本文が現行コードから**一意に決まらない**．節に何を書くかは，
少なくとも次の 3 通りがあり得る:

1. 既存の Python 中心の記述に Rust の 1 段落を足す (下の案)．
2. 節を「言語ごとの表」に再構成し ruff / mypy / rustfmt を並べる．
3. Rust 固有事情 (版の一本化先が `pyproject.toml` ではなく
   `rust-toolchain`) を別節に切り出す．

いずれも現行コードと矛盾しない．したがって **新しい指針の追加**であり，
CLAUDE.md の standing approval (drift 訂正のみ) の外にある．承認を待つ．

## Before / After (案 1)

### `:44` の直前 (「`trailing-whitespace` や `check-yaml` のように…」の段落の前) に追記

```diff
+**Rust も同じ原則で回している．** `cargo fmt --all` は 2026-08-12 に
+local hook として登録した．hook 側が独自に Rust toolchain を入れると
+`rust-toolchain` の版と二重化し，ローカルの `cargo fmt` と pre-commit の
+結果が食い違う — ruff で踏んだのと同じ構図である．版の一本化先だけが
+Python と異なり，Rust では `pyproject.toml` ではなく toolchain の指定に
+なる．
+
+`--check` ではなく自動整形にしてあるのは Python 側の `ruff format` に
+揃えるため (未整形を残すと，無関係な PR に整形差分が混ざる)．
+`cargo fmt` は crate 単位で動くので `pass_filenames: false` が必須．
+
```

## 判断してほしい点

- 案 1 (追記) でよいか，案 2 (言語ごとの表への再構成) を採るか．
- 表に再構成する場合，clippy の扱い (`audits/coverage.md` NEW-2 行 /
  この run の PR) を先に決めてから書いた方が二度手間にならない．
