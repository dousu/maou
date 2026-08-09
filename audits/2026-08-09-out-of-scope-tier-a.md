---
path: src/maou/infra/console, src/maou/interface, .claude/skills
kind: backlog
scope: other
level: high
status: done
started: 2026-08-09
last_sha: b38b36c
---

# Audit — out-of-scope backlog, Tier A

**これは path 監査ではない．** `audits/coverage.md` の out-of-scope
backlog から個別の finding 3件だけを消化した記録である．対象パスは
どれも全体監査されていないので，ledger のメインテーブルに `done` 行は
書かない (書くと未監査パスを監査済みと誤って主張することになる)．

この run が `/audit-backlog` (`.claude/commands/audit-backlog.md`) の
motivating example であり，同コマンドはこの run で判明した3つの運用
ギャップを埋めるために書かれた．

## Resume point

_(complete — no resume point)_

消化した3件は全て resolved．QA は下の Environment notes の通り
**全て実行済みで pass**．

## Consumed

| # | Source record | Target | 消化 |
|---|---|---|---|
| A1 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Out-of-scope 1 | `src/maou/infra/console` | `064806e` |
| A2 | [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Out-of-scope 1 | `src/maou/interface` | `cc10790` |
| A3 | [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Out-of-scope 3 | `.claude/skills/` | `3600b32` |

優先度づけは 22件 (deferred 13 + out-of-scope 9) 全件を HEAD (`d7d046f`)
に対して再検証した上で 6 段階に並べ，ユーザが Tier A を選択した．
残りのランク付けは下の「Backlog remaining」を参照．

## Applied

### `cc10790` — A2 (`pyproject.toml` 0.82.4 → 0.82.5, patch)

- `src/maou/interface/game_graph_visualization.py` — `_root_is_startpos()`
  を追加し，`get_opening_name` はルートが平手初期局面でないとき
  `_DEFAULT_OPENINGS` と照合しない．`_startpos_sfen_fields()`
  (`functools.cache`) が `Board().get_sfen()` の盤面・手番・持駒
  フィールドを返す (手数フィールドは局面の同一性に無関係なので除外)．
- `tests/maou/interface/test_game_graph_visualization.py` — 回帰テスト
  6件．`_build_tree_with_chuo_hisha()` (初手 `5g5f`，
  `_DEFAULT_OPENINGS` に単独一致する唯一の1手パターン) と
  `_MIDDLEGAME_SFEN`．move16 定数と中盤 SFEN の妥当性自体も
  テストで固定した．

### `064806e` — A1 (`pyproject.toml` 0.82.5 → 0.82.6, patch)

- `src/maou/infra/console/utility.py` — `--stage12-lr-scheduler` の
  `click.Choice` を `learn.SUPPORTED_LR_SCHEDULERS` から導出
  (正準キー + 表示名の両方を受理; `normalize_lr_scheduler_name` が
  どちらも解決する)．`help` に `--lr-scheduler` を継承する旨を追記．
- `tests/maou/infra/console/test_cli_option_compatibility.py` —
  「広告した選択肢が全て正規化を通る」性質テストを
  `learn-model` / `benchmark-training` 両方に対して追加
  (`auto`/`none` は learn-model 側の制御値なので除外)．

### `3600b32` — A3 + doc 追随 (doc-only, 版上げなし)

`reviews/2026-08-09-tier-a-doc-and-skill-drift.md` (承認済み，
`applied_in: 3600b32`，frontmatter 更新は `88e8ac1`)．

- `.claude/skills/type-safety-enforcer/SKILL.md:14` — 88桁 → 64桁．
- `.claude/skills/qa-pipeline-automation/SKILL.md` 3箇所 — 同じ 88桁．
  **backlog 未記載**で，A3 の確認中に sibling を見て発見した．
- `docs/commands/utility_benchmark_training.md:31` — 「`cosine_annealing`
  / `step` は `ValueError`」という欠陥の記述だったので A1 に追随．
- `docs/commands/visualize.md:182` — 定跡行が平手ルートのグラフに
  限定されたことを追記 (A2 に追随)．

## Corrections to the source records

**この run で最も価値のある発見．** record の診断は正しかったが，
示唆した修正は本番を壊すものだった．

**A2 (game_graph record Out-of-scope 1).** record は
「sibling `export_sfen_path` (:738) は `_initial_sfen` で分岐しており，
これが抜けの証拠」としていた．素直に読むと修正は
`if self._initial_sfen is not None: return None` になる．
これは**通常の平手グラフで定跡表示を全滅させる回帰**だった:

- `build_game_graph.py:184-189` は `initial_sfen` 未指定 (平手) でも
  `Board().get_sfen()` に解決して `metadata.json` に書く．
- `game_graph_server.py:297` はそれをそのまま
  `GameGraphVisualizationInterface(initial_sfen=...)` に渡す．
- したがって**本番経路で `_initial_sfen` が `None` になることはない**．
  sibling の `is not None` 分岐は常に `position sfen` 側を通っており，
  `position startpos` 側は本番では死んでいる．`get_initial_sfen()` が
  `"startpos"` を返すこともない．

教訓: **値の出自 (producer) を読まずに consumer 側のガードを書かない．**
record は「どこを見るか」には信頼できるが「何をするか」には信頼できない．
この落とし穴自体を `test_resolved_startpos_sfen_still_matches` で固定した．

**A1 (app/learning record Out-of-scope 1).** 診断は正確だった．
record が触れていなかった点: `cosine_annealing_lr` は
`SchedulerFactory.create_scheduler` (`setup.py:1071`) に実装済みで，
CLI から選べなかっただけだった．また同じ「別名表から導出する」修正は
sibling 2箇所 (`utility.py` の `--lr-scheduler`, `learn_model.py:468`)
が既に採用済みで，`utility.py:487` だけがハードコードの取り残しだった．

**A3 (game_graph record Out-of-scope 3).** 診断は正確だが**不完全**
だった — 同じ 88桁の誤りが `qa-pipeline-automation/SKILL.md` の3箇所
にもあり，record は 1 ファイルしか挙げていなかった．

## Re-triaged

なし．選択した3件は全て resolved．

## Deferred

なし (この run で新たに判断待ちになった項目はない)．

## Doc findings

`reviews/2026-08-09-tier-a-doc-and-skill-drift.md` — 4ファイル，
**承認され `3600b32` で適用済み**，frontmatter は `88e8ac1` で
`status: applied` に更新．内訳は上の Applied を参照．

## Out of scope

この run が生んだ新規 finding は，`/audit-backlog` コマンド新設に伴う
durable doc 追随のみで，これは backlog ではなく
`reviews/2026-08-09-audit-backlog-command.md` (承認待ち) で扱う:

- `CLAUDE.md` の Files テーブルと MUST rules に `/audit-backlog` を反映．
- `audits/README.md` に (a) deferred 項目の到達性の議論，(b) `done`
  record の追記注釈規約 (`916e874` が immutability 規約と矛盾している
  ことの解消) を追加．

## Backlog remaining

Tier A 消化後の残り 19件 (deferred 13 + out-of-scope 6)．
ランク付けは `/audit-backlog` の rubric に対応する:

- **T3 (判断待ち，contained)**: `interface` の `stage2_test_ratio` 未読
  パラメータ (app/learning Out-of-scope 2 + Deferred 10) /
  `domain/model` の `FreezableBackbone` 型の穴 (同 3 + Deferred 11) /
  `game_graph/schema.py:41-79` のデッドコード keep-delete
  (game_graph Deferred 1) / `domain/data/polars_tensor.py` の 4関数
  caller ゼロ (app/learning Out-of-scope 4 + Deferred 9) /
  `streaming_dataset.py:604` の `__len__` が LR scheduler を汚染
  (同 Deferred 1)
- **T4 (大規模リファクタ)**: app/learning Deferred 2 (stage1/2 の
  ~400行クローン) / 3 (6 adapter クラス) / 4 (`callbacks.py` の
  `_ensure_device` 6重複)
- **T5 (この環境で検証不能 — GPU 実機必須)**: app/learning Deferred
  5 / 6 / 7 / 8
- **T6 (新規執筆・未確認)**: `openings.py` が無文書 (game_graph
  Out-of-scope 4) / `docs/architecture.md` に `game_graph` の居場所が
  ない (同 5) / `app/game_graph/query.py:184-194` 未確認 (同 2)

T1 / T2 は空になった．

## Environment notes

**QA 状況．**

| チェック | 結果 |
|---|---|
| `ruff format --check src/ tests/` | **pass** (285 files) |
| `ruff check src/ tests/` | **pass** |
| `mypy src/` | **pass** (134 files) |
| `pytest` 対象2ファイル | **56 passed** |
| `pytest tests/maou/{infra/console,interface,domain/game_graph}` | **363 passed, 3 skipped** |
| `pytest` 全体 | **1725 passed, 54 skipped** (100.6s) |

**回帰テストの非空虚性を実証済み** (修正を無効化して失敗を確認し復元):

- A2: `_root_is_startpos()` ガードを削除すると
  `test_middlegame_root_does_not_match` と
  `test_gote_root_does_not_match` が失敗する．同時に
  `test_startpos_root_matches_opening` /
  `test_resolved_startpos_sfen_still_matches` は**通り続ける**ので，
  修正が過剰抑制していないことも同時に確認できる．
- A1: `click.Choice` を元のハードコード一覧に戻すと
  `test_stage12_lr_scheduler_choices_are_all_resolvable[benchmark-training]`
  が失敗し，`ValueError: Unsupported learning rate scheduler` を再現する．

`uv run` はこのコンテナで使用不可 (前 run と同じ理由)．
`uv sync --extra cpu --no-install-project` は成功したが，
`maou._rust` は含まれないため `maturin develop --release` が必要だった
(**31m05s**; `patchelf` 未インストールの rpath 警告は無害)．
QA は `.venv` のツールを直接叩いて実施した．

`.git/hooks/pre-commit` はこのコンテナに**未インストール**なので
commit ではフックが走っていない．`uv run` ベースのフック
(`test`, `mypy`, `ruff-check`, `ruff-format`, `uv-lock`) は
そもそもこの環境では走らない．

**プロセス上の発見 (`/audit-backlog` 新設の直接の動機).**

1. **deferred 項目に到達経路がない．** `coverage.md` は「11 deferred」と
   件数だけを持ち，項目本体は record の中にしかない．
   `audits/README.md` が out-of-scope について述べている retrieval の
   議論 (「a per-run record is read only when someone opens that specific
   path」) が deferred には適用されていなかった．13件が事実上不可視
   だった．
2. **record の immutability が実運用と矛盾．** README は `done` record を
   immutable とするが，`916e874` が `done` record の Deferred を Applied
   に移している．追記注釈方式で両立させる案を提案中．
3. **消化 run の記録先が未定義．** メインテーブルに `done` 行を書けない
   (未監査パスを監査済みと主張してしまう) ため，`kind: backlog` の
   record + backlog 表からのリンクという形を採った．
