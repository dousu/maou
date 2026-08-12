# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles.

### Core Components
- **Domain Layer**: Business logic and entities (network models, loss functions, parsers)
- **App Layer**: Use cases (converter, learning, pre-processing)
- **Interface Layer**: Adapters between app and infrastructure
- **Infrastructure Layer**: External systems (cloud storage, databases, logging)

### Data Pipeline
- **Data Format**: Arrow IPC (.feather) with LZ4 compression
- **Data Processing**: Polars DataFrames
- **I/O Backend**: Rust (PyO3 + maturin) for high-performance file I/O
- **Legacy Support**: Numpy .npy format still supported

## Critical Rules (MUST)

### Architecture
- MUST maintain dependency flow: `infra → interface → app → domain`
- MUST NOT introduce circular dependencies between layers
- MUST NOT import heavy optional dependencies (torch, cloud SDKs) at module
  level in `src/maou/infra/console/` or any path a light command reaches —
  use PEP 562 `__getattr__` lazy resolution
- MUST NOT leave a module-level assignment for a name resolved lazily by
  `__getattr__` (the global shadows the hook and leaks the pre-probe value)

### Code Quality
- MUST add type hints to all code
- MUST add docstrings to public APIs
- MUST run pre-commit hooks (NEVER use `--no-verify`)

### Serena MCP
- MUST NOT call multiple Serena MCP tools in parallel (to prevent OOM in memory-constrained DevContainer)
- MUST call Serena tools sequentially, one at a time

### Versioning (Python)
- MUST bump version in `pyproject.toml` when modifying files under `src/`
- MUST follow semantic versioning: `fix:` → patch, `feat:` → minor, breaking change → major
- MUST NOT push changes to `src/` without a corresponding version bump
- Version is the single source of truth in `pyproject.toml` (`version_provider = "pep621"`)

### Versioning (Rust crates)
- MUST bump version in the **owning** `Cargo.toml` when modifying files under `rust/<crate>/`
  - The owning manifest is the nearest ancestor `Cargo.toml` with a `version` field.
    Resolve it from the changed file, never from a list here — crates added later are
    then covered automatically. `ls rust/` is the current set (as of 2026-08-08:
    `maou_convert`, `maou_index`, `maou_io`, `maou_rust` (PyO3 bindings), `maou_search`,
    `maou_shogi`, `maou_usi`).
- MUST follow semantic versioning independently per crate: `fix:` → patch, `feat:` → minor, breaking change → major
- MUST NOT push changes to `rust/<crate>/` without a corresponding version bump in that crate's `Cargo.toml`
- Rust crate versions are independent of the Python package version in `pyproject.toml`

### Forbidden Actions
- MUST NOT use pip directly (use `uv` only)
- MUST NOT create `__init__.py` unless absolutely necessary
- MUST NOT skip pre-commit hooks
- MUST NOT commit secrets (.env, credentials)

### Documentation
- MUST update `docs/commands/` when modifying CLI commands or options in `src/maou/infra/console/`
- MUST create a new `docs/commands/<command-name>.md` when adding a new CLI command
- MUST remove `docs/commands/<command-name>.md` when removing a CLI command
- MUST follow the existing documentation format (Overview + CLI options tables)

## Repository-Centric Memory Architecture (MUST)

Long-term memory lives in the repository, not in the conversation.
Full spec: [docs/memory-architecture.md](docs/memory-architecture.md).

### Files

| Path | Role | Committed |
|---|---|---|
| `reviews/YYYY-MM-DD-<title>.md` | Proposals + audit trail. `status:` in frontmatter. | yes |
| `scratchpad/current.md` | Authoritative current state. | no (`.gitignore`d) |
| `scratchpad/compass.md` | Always-loaded binding layer. Fixed sections: VETOES → TRIPWIRES → North-star → Invariants[scope] → REFUTED → 環境リファレンス. ≤ ~9KB. | no (`.gitignore`d) |
| `worklog/YYYY-MM-DD-HHMMSS.md` | One file per checkpoint, JST, immutable. | no (`.gitignore`d) |
| `~/.claude/.../memory/` (auto-memory) | `feedback_*.md` process rules ONLY (advisory). NOT campaign state; no new `project_*.md`. | n/a (per-machine) |
| `audits/coverage.md` | Repo-wide audit ledger: per-path status, resume point, open items. | yes |
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. | yes |
| `audits/YYYY-MM-DD-backlog-<slug>.md` | One record per `/audit-backlog` run (`kind: backlog`). Consumes individual deferred / out-of-scope findings; does **not** earn a main-table row. | yes |
| `.claude/commands/checkpoint-context.md` | Writer of campaign working memory (`worklog/`, `scratchpad/`). | yes |
| `.claude/commands/resume-context.md` | Read-only resume. | yes |
| `.claude/commands/audit-and-fix.md` | Writer of `audits/` (path 単位の監査). | yes |
| `.claude/commands/audit-backlog.md` | Writer of `audits/` (deferred / out-of-scope の個別消化)．判断コスト P1-P6 で分類し，P1-P3 (自動帯) はユーザに聞かずに修正・PR・マージする． | yes |

### MUST rules

- MUST NOT edit `CLAUDE.md` / `docs/` without an **approved** `reviews/*.md`
  proposal. Draft it `status: pending`; **on explicit user approval the
  model applies the edit itself and commits**, then sets `status: applied`
  + `applied_in: <sha>` (approval is the safeguard against *silent* edits
  — it is not tied to any one command). `/checkpoint-context` step 5 and
  `/audit-and-fix` step 8 both reconcile proposals; either may take the
  approval.
- **Standing approval — drift corrections only.** `/audit-backlog` の
  P2 クラス (`.claude/commands/audit-backlog.md` § "priority is decision
  cost") が **drift correction** と判定した durable-doc 修正は，この項が
  与える恒久承認により，その run 内で適用してよい．`reviews/*.md` の
  提案は依然として MUST — 承認の往復だけを省く．
  判定基準は 1 つ: **訂正後の本文が現行コードから一意に決まるか**．
  一意なら適用 (例: doc が `.npy` と書き，writer は `.feather` を書く)．
  一意でないなら — 新しい指針，節の再構成，規則の追加，複数の書き方が
  あり得る記述 — P2 ではないので判断帯に落ち，通常どおり承認を待つ．
  迷った場合は待つ側に倒す．この恒久承認は `/audit-backlog` にのみ
  適用され，`/audit-and-fix` step 8 には及ばない (拡張は別提案とする)．
- MUST treat `worklog/*.md` as immutable. Each `/checkpoint-context`
  creates a **new** file — never edit a previous one.
- MUST preserve failed attempts, reasoning, and uncertainty in every
  checkpoint entry. Do not over-summarize.
- MUST run `/resume-context` at the start of any session inheriting
  cleared context, before acting.
- MUST run `/checkpoint-context` before any context reset, long break,
  or handoff.
- MUST use JST (`Asia/Tokyo`) for all timestamps and filenames.
- MUST load `scratchpad/compass.md` at `/resume-context` and treat its
  Invariants as binding guardrails (evaluate values against the SHA for
  staleness).
- MUST curate `scratchpad/compass.md` at every `/checkpoint-context`:
  update North-star numbers (or write "unchanged"), add new do-not-redo
  conclusions, delete/edit overturned invariants, and evict when over the
  size cap (≤ ~9KB byte cap, env-reference 込). Never append-only.
- MUST refuse `/checkpoint-context` on an uncommitted `src/`/`rust/` tree
  (dirty-tree gate — commit + pre-commit + version bump first). Only an
  explicit `--allow-dirty` overrides.
- MUST keep campaign do-not-redo conclusions in `scratchpad/compass.md`
  (binding) ONLY; MUST NOT mirror them into `~/.claude` auto-memory (a
  "background, may-be-outdated" channel — mirroring licenses
  re-litigation). Auto-memory holds only the `feedback_*.md` process rules;
  MUST NOT author new `project_*.md`.
- MUST file a `reviews/*.md` ONLY for committed durable-doc targets
  (`CLAUDE.md` / `docs/`); `rust/`/`src/` algorithmic tuning + lever
  rejections go to `worklog/` + `compass.md`, never `reviews/`.
- MUST surface `compass.md` § 🚫 VETOES and § 🚦 TRIPWIRES FIRST and
  verbatim (Confirmed-binding) at `/resume-context`, and commit every
  `reviews/` `status:` transition immediately (audit trail).
- MUST record every `/audit-and-fix` **and `/audit-backlog`** run in
  `audits/` and commit it — `/audit-and-fix` writes a ledger row + record
  file; `/audit-backlog` writes a `kind: backlog` record and MUST NOT
  write a main-table row (a row there claims the whole path was audited).
  `audits/` is the only cross-session record of repo-wide audit coverage,
  and unlike `scratchpad/` it survives container reclamation. An
  interrupted run MUST still write its resume point before the session
  ends.
- MUST treat `audits/coverage.md` § "Open findings backlog" as the **only**
  live worklist of open findings, deferred and out-of-scope alike:
  `/audit-and-fix` MUST append a row for every finding it leaves open, and
  a run that resolves one MUST delete its row. MUST NOT read a record's
  `## Deferred` / `## Out of scope` section to decide what work remains —
  a record is an immutable account whose Deferred section stays true after
  the finding ships, so treating it as a worklist re-surfaces resolved
  findings forever. MUST NOT amend a record to carry state (no `RESOLVED`
  markers, no Deferred→Applied moves, no renumbering); a **correction** of
  a wrong diagnosis is the only permitted amendment.
- MUST keep `audits/` independent of the campaign memory (`worklog/` /
  `scratchpad/compass.md`). The campaign layer tracks one
  stable-environment measurement campaign; `audits/` tracks traversal of
  the tree. Do not mirror one into the other.

## Code Exploration Policy (MUST)

コードベースの調査・探索には，MUST use `Task` tool with `subagent_type=Explore`.

### Covered Operations
- ファイル検索（Glob/Grep）を複数回行う調査
- 複数ファイルを読んでコードを理解する作業
- エラー原因の調査
- 実装方法を決めるための既存コード調査

### MUST NOT: Direct Multi-file Exploration
以下の操作を直接行うことを禁止:
- 調査目的での連続的なGrep/Glob実行
- 複数ファイルを順次Readして回る探索
- トラブルシューティング時のコード調査

### Exceptions (Direct Access Allowed)
1. **ユーザーが明示的にファイルパスを指定** - 「src/foo.pyを読んで」
2. **単一ファイルの特定行を確認** - エラーメッセージの「file:line」参照
3. **Exploreで特定済みファイルへのアクセス** - 既知の場所への編集

### Decision Criteria
- 「どこにあるか分からない」→ Explore必須
- 「このファイルのこの部分」→ 直接アクセス可

## Development Guidelines (SHOULD)

### Tool Priority (Serena MCP)
SHOULD prefer Serena tools for token efficiency:
1. `find_symbol` - Find definitions
2. `find_referencing_symbols` - Find usages
3. Glob/Grep - File/text search
4. Read - Full file (last resort)

### Package Management
- SHOULD use `uv sync` for dependencies
- SHOULD use `uv add` for new packages
- SHOULD use `uv run` for script execution

### Git Workflow
- SHOULD follow commit format: `feat|fix|docs|refactor|test|perf: message`
- SHOULD run QA pipeline before commit:
  `uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/`

### Agent Teams
- SHOULD limit team agents to 2 or fewer in 8GB RAM DevContainer environments
- SHOULD consider memory impact when adding agents (each agent spawns its own MCP server processes)

### Testing
- SHOULD write tests for new features
- SHOULD write regression tests for bug fixes
- Test path: `src/maou/{layer}/{module}/file.py` → `tests/maou/{layer}/{module}/test_file.py`

### 重いテスト (Rust dfpn) — release ビルド必須

doc コメントに `**[SLOW]**` フラグがついている Rust テストは全て `#[ignore]` 属性を持ち，
debug ビルドでは release 比約 6 倍遅く，ノード/時間制限を超過してテストが失敗する場合がある．
これらのテストを実行する際は MUST `--release --ignored` フラグを付けること:

```bash
cargo test --release -p maou_shogi -- <test_name> --nocapture --ignored
```

現在 `[SLOW]` フラグがついている主なテスト (`rust/maou_shogi/src/dfpn/tests.rs`):

| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_29te` | - | mid 1te/3te/29te canonical (396,516 nodes (find_shortest 総数) / mate-29 / STRICT Some(29)) |
| `test_39te_measure` | 30M nodes (default) | 39te canonical (17,593,615 nodes / mate-39 / STRICT Some(39) / canonical PV) |
| `test_counter_check_example` | - | 逆王手詰将棋 mate-7 健全性 |
| `test_counter_check_diagnostic` | - | 診断用ログ出力 |
| `test_no_checkmate_counter_check_probe` | 10M nodes | ノード予算プローブ |
| `test_false_proof_hunt` | 50 nodes/局面 (leaf-mate 相当) | 偽証明 (`STRICT VERIFY None`) ハンター．ランダム対局から局面生成．env `SEED`/`GAMES`/`PLIES`/`NODES`/`MAXHITS` |

dfpn の **偽 1 手詰 (soundness 違反)** を疑うときは `MATE1PLY_VERIFY=1` を付ける．
1 手詰と申告した手を実 replay で検証し，偽なら
`[mate1ply] FALSE MATE site=... m=... sfen=...` を出す
(production が通る fused / near2 / cached full scan の全経路をカバー)．
`test_false_proof_hunt` と併用すると広域スイープになる:

```bash
SEED=1 GAMES=200 MATE1PLY_VERIFY=1 cargo test --release -p maou_shogi -- \
  --test-threads=1 --ignored --nocapture dfpn::tests::test_false_proof_hunt
```

dfpn テストは各々が大きな置換表 (TT) を alloc するため，**MUST `--test-threads=1`** で実行すること．
default の並列実行は memory 制約 DevContainer (8GB) で OOM → `signal: 15 SIGTERM` となり，
assertion failure でなくプロセス kill として現れる (コード回帰と誤認しやすい)．

```bash
cargo test --release -p maou_shogi -- --test-threads=1
```

## Quick Reference

### Common Commands
```bash
# Dependencies
uv sync                                    # Base install
uv sync --extra cpu                        # CPU PyTorch
uv sync --extra cuda                       # CUDA PyTorch
uv sync --extra cpu --extra visualize      # Testing with visualization
uv sync --extra cuda --extra visualize     # Full development

# Development
uv run pytest                              # Run tests
uv run maturin develop                     # Build Rust extension
uv run maou --help                         # CLI help

# Rust tests
cargo test -p maou_shogi                                          # 通常テスト (debug)
cargo test --release -p maou_shogi -- <test_name> --nocapture     # [SLOW] テスト (release 必須)
cargo test --release -p maou_shogi -- --ignored --nocapture       # #[ignore] テスト (release 必須)
```

### Japanese Writing Rules (日本語記述規則)
- 読点: `，` (全角コンマ)
- 句点: `．` (全角ピリオド)
- 括弧: `()` (半角のみ)

## Documentation Links

| Topic | Document |
|-------|----------|
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Testing | [docs/testing-guide.md](docs/testing-guide.md) |
| Code Quality | [docs/code-quality.md](docs/code-quality.md) |
| Rust Backend | [docs/rust-backend.md](docs/rust-backend.md) |
| maou_shogi 設計思想 | [docs/design/maou-shogi-concept.md](docs/design/maou-shogi-concept.md) |
| 詰将棋ソルバー設計 | [docs/design/tsume-solver/](docs/design/tsume-solver/index.md) |
| 1局面探索エンジン設計 | [docs/design/position-search/](docs/design/position-search/index.md) |
| 教師信号の質 (policy/value 改善) | [docs/design/training-quality/](docs/design/training-quality/index.md) |
| CLI 間データパイプライン | [docs/design/data-pipeline/](docs/design/data-pipeline/index.md) |
| Performance | [docs/performance.md](docs/performance.md) |
| LR Tuning | [docs/learning-rate-tuning.md](docs/learning-rate-tuning.md) |
| Git Workflow | [docs/git-workflow.md](docs/git-workflow.md) |
| CLI Commands | [docs/commands/](docs/commands/) |
| Shogi Visualization | [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) |

### ⚠️ Visualization 実装時の必読ドキュメント

`maou visualize` や将棋盤描画に関する実装を行う前に，
**必ず** [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) を読むこと．

将棋の座標系は一般的な row-major 配列とは異なり，`square = col * 9 + row` である．
この規則を理解せずに実装すると，駒の位置や矢印の方向が90度回転するバグが発生する．
