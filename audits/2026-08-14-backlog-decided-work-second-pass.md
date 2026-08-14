---
kind: backlog
date: 2026-08-14
level: medium
path:
  - src/maou/infra/object_storage
  - src/maou/infra/console
  - src/maou/app/learning
  - src/maou/app/utility
  - audits
last_sha: a0690f0
---

# `/audit-backlog` 2026-08-14 — 決定済み作業の 2 巡目 + 台帳の retrieval bug

前 run ([cache-knob-removal](2026-08-14-backlog-cache-knob-removal.md)) が
得た設計判断のうち**未実装で残っていた 3 件**が入口．そのうち 2 件
(O5(c) / Deferred 3) を実装・出荷した．「決定を行に書けば次 run が通常
作業として拾える」が **2 run 連続で機能した**．

あわせて，その仕組み自体を壊していた**台帳のセル数バグ**を修復した
(step 1d)．

## Classification

10 行を再検証: **stale 0 / changed shape 1 / confirmed 9**．

| ID | backlog 行 | クラス | クラスを決めたテスト | ゲート | 結果 |
|---|---|---|---|---|---|
| B1 | (新規) `coverage.md` のセル数バグ | **P1** | 出荷物に触れない (`audits/` のみ) | — | 出荷 |
| B2 | O5 (c) bundling ノブ削除 | **P6** | 3 コマンドから公開オプションが 2 つ消える | なし (G4 は前 run で retire) | 出荷 |
| B3 | Deferred 3 アダプタ 6→3 統合 | **P4** | Stage2 streaming が `set_epoch` ガードを獲得し挙動が変わる．データ互換は不変，公開名は別名で温存 | G2 → 取り込んで解消 | 出荷 |
| B4 | D14(b) + N6-2 HCPE 専用化 | **P6** | 公開 ABC の改名 | **G2** | 残置 |
| B5 | O9 決定的ハッシュ置換 | **P4** | 同じ呼び出しで返る行が変わる | **G1** + キー列未決 | 残置 |
| B6 | D13 columnar 直接スライス | **P4/P6** | 未決 | **G2 + G4** | 設計判断を質問 |
| B7 | Deferred 2 Stage パイプライン統合 | **P4** | 未決 | **G3 + G4** | 設計判断を質問 |
| B8 | Deferred 5 休眠中の host sync | **P4** | GPU | **G1** | 設計判断を質問 |
| B9 | Deferred 6 `stream.synchronize()` | **P4** | GPU | **G1** | 同上 (1 問が 3 行を governs) |
| B10 | Deferred 7 GNS の `.item()` | **P4** | GPU | **G1** | 同上 |

### changed shape (1 件) — B2

O5 行が列挙していた「残る作業」が**過小**だった．再検証で判明した
追加分:

- `object_storage/data_source.py:415-416` の docstring 2 行 (行が挙げた
  4 構築経路 + 3 pass-through には入っていない)
- 読み手のいない `bundle_cache` / `bundle_id` (`:88-89` の型注釈と
  `:133-134` の初期化)．行はこれに触れていない
- `docs/rust-backend.md:698-701` (行は `docs/commands/` 3 本しか挙げて
  いない)
- `.claude/skills/` 配下 **4 ファイル** — `cloud-integration-tests`
  (3 箇所)，`benchmark-execution` (4 箇所 + skill description)，
  `data-pipeline-validator` (2 箇所)．いずれも実行可能なコマンド例と
  して削除済みフラグを載せていた

診断と向きは不変なので **stale ではなく changed shape**．

### confirmed だが行番号がずれていたもの (4 件)

前 run の `bdda7b5` が `file_data_source.py` / `bq_data_source.py` /
`console/*.py` を触ったため:

| 行 | 記録の位置 | HEAD での位置 |
|---|---|---|
| D13 | `_columnar_batch_to_structured_array` `:553-634`，`np.empty` `:584`，`_columnar_to_structured_record` `:471-494`，`FileManager.get_items` `:496-509`，`__getitem__` `:692-696`，`get_items` `:701-712` | `:408-`，`:439`，`:338-359`，`:363-`，`:310-`，`:550-` |
| D14(b) | `iter_batches` `:539`，`iter_batches_df` `:726` | `:394`，`:575` |
| N6-2 | bq override `:740`，file override `:726` | `:737`，`:575` (obj storage `:480` は前 run の訂正どおり) |
| O9 | `__get_total_rows` def `:213`，`__fetch_from_bigquery` def `:345`，再サンプル `:396-405`，`list_rows` `:415-419`，`get_page` `:486` | `:212`，`:344`，`:397-404`，`:413-417`，`:485` |

`app/learning` 系 (Deferred 2/3/5/6/7) は 2026-08-09 以降ファイルが
触られていないため，行番号は完全に一致した．

## Consumed

| 行 | 由来 | 対象 | 出荷内容 |
|---|---|---|---|
| O5 (残り (c)) | [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) | `infra/object_storage` + `infra/console` | bundling ノブ 2 つを CLI 3 コマンド・4 構築経路・3 pass-through・docstring・死んだ `bundle_cache`/`bundle_id` から削除．doc 4 本 + skill 3 本を更新．不活性テストを「存在しない」テストへ置換 |
| Deferred 3 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `app/learning` | アダプタ 6 クラスを 3 つに統合 (旧 6 名は別名)．`_build_model` の `isinstance` 分岐を除去．型アサーション 2 本を挙動アサーションへ書き換え |

**O5 行はこれで全消化** — (a)(b)(d) は前の 3 run で，(c) が本 run．

## Applied

| commit | 内容 |
|---|---|
| `588ffae` | `audits/coverage.md` — Deferred 3 行と O9 行の 4 セル目を本文へ畳む (B1) |
| `232358e` | `feat!:` bundling ノブ削除 (B2)．version `0.90.0` → `0.91.0` |
| `aad00d9` | `refactor(learning):` アダプタ統合 (B3)．version `0.91.0` → `0.91.1` |

主要な `file:line`:

- `src/maou/infra/object_storage/data_source.py` — `enable_bundling` /
  `bundle_size_gb` / `bundle_cache` / `bundle_id` を全除去，
  `__download_all_to_local()` は引数なしへ
- `src/maou/infra/console/pre_process.py` / `utility.py` — click option
  3 組 (pre-process 1 + benchmark 2)，関数引数 3 組，pass-through 6 組
- `src/maou/app/learning/multi_stage_training.py:111` `StageModelAdapter`，
  `:157` `StageDatasetAdapter`，`:195-198`/`:227-230` 別名 4 本
- `src/maou/app/learning/streaming_dataset.py:685`
  `StageStreamingAdapter` (`set_epoch` は `hasattr` ガード付き)，
  `:738-741` 別名 2 本
- `src/maou/app/learning/stage_component_factory.py:866-874` —
  `isinstance` 分岐 (旧 `:876-882`) を単一構築へ

### 回帰テスト

- `tests/maou/infra/object_storage/test_bundling_knobs_absent.py`
  (旧 `test_bundling_knobs_inert.py` を置換) — 3 層のコンストラクタ・
  モジュール本文・CLI の 3 方向から不在を固定
- `tests/maou/app/learning/test_stage_adapters_merged.py` (新規 10 本) —
  別名 6 本の同一性，`set_epoch` ガードの**両方向** (無い dataset で
  落ちない / 有る dataset へは委譲する)，head 型を見ないこと
- `tests/maou/app/learning/test_stage_component_factory.py` —
  `test_model_type` × 2 を `test_model_wraps_the_stage{1,2}_head` へ．
  型アサーションは別名化で識別力を失うため，**掴んでいる head の型**と
  **forward の出力** (policy shape + value が全ゼロ) で固定し直した

### 非空虚性の確認

| 無効化した内容 | 落ちたテスト |
|---|---|
| `StageStreamingAdapter.set_epoch` のガードを外す | `test_set_epoch_is_guarded_for_datasets_without_it` (`AttributeError`) |
| `create_stage1_components` に Stage 2 の head を作らせる | `TestCreateStage1Components::test_model_wraps_the_stage1_head` |

2 つ目が重要 — **書き換え前の `isinstance` アサーションはこの無効化を
検出できない** (別名なのでどちらも通る)．書き換えがテストの識別力を
回復させたことの証拠になっている．

## Decisions asked

このセクションは 3d の質問と回答が入る (質問を上げた時点で追記)．

## In flight

なし (指定ブランチ 1 本の制約により run 全体が 1 PR)．

## Re-triaged

- **B4 (D14(b) + N6-2)** — 決定済みで人間待ちではないが，**G2 の解消
  (テスト 3 ファイルを `StreamingHcpeDataSource` へ寄せる) が B2+B3 と
  同じ all-or-nothing PR に載せるには重い**と判断して見送った．行番号を
  HEAD に更新した以外，判断は不変．**次 run の筆頭候補** — 決定は済んで
  いるので通常作業として着手できる．
- **B5 (O9)** — G1 に加え，決定 (b) 決定的ハッシュ条件の**キー列が
  未調査**．「現行スキーマに一意キーの保証があるか」を先に確かめないと
  実装の形が決まらない．今回の 4 問枠には入らなかった (下記)．

## Corrections to the source records

**なし．** changed shape 1 件 (B2 の作業一覧が過小) は記録ではなく
`coverage.md` の行の記述の問題で，しかも行が挙げた内容自体は正しく，
不足していただけ．行は本 run で削除されるので訂正先が残らない
(この記録の § Classification が代わりの account になる)．

## Doc findings

[`reviews/2026-08-14-bundling-knobs-removed-and-stage-adapters-merged.md`](../reviews/2026-08-14-bundling-knobs-removed-and-stage-adapters-merged.md)
— `status: applied`．**P2 の恒久承認**で適用した (訂正後の本文が現行
コードから一意に決まる: オプションは存在しない / 生成されるクラスは
1 つ)．対象は `docs/commands/pre_process.md` /
`utility_benchmark_dataloader.md` / `utility_benchmark_training.md` /
`docs/rust-backend.md`．

**据え置いたもの** (P2 に該当しないと判断):

- `docs/stage2-speed-investigation.md` — 特定時点の調査報告．当時の
  クラス名を書いていること自体は誤りではなく，歴史的記述の扱いは
  書き手の判断なので訂正後の本文が一意に決まらない
- `.claude/skills/gh-pr` / `pr-preparation-checks` /
  `feature-branch-setup` の "array bundling" — **架空の PR / ブランチ名の
  例**であって CLI の記述ではない

## Out of scope

新規所見は **B1 のみ** — `audits/coverage.md` の Deferred 3 行と O9 行が
Item セルを閉じたあとに追記されており，3 列のヘッダに対して 4 セル目を
作っていた．GFM は余剰セルを捨てるため，両行の「2026-08-14 にユーザが
設計判断を回答」以降 (決定内容・G4 retire・残る作業) が**描画時に
丸ごと消えていた**．

これは step 1d が言う retrieval bug そのもので，しかも**皮肉な形**を
している: G4 を retire する仕組みが，適用された 4 行のうち 2 行で
不可視だった．同 run 内で修復済み (`588ffae`) なので backlog 行は
起票していない．

## Environment notes

- **torch がこの環境に無かった** — 前 run の設計判断 Q2 で
  `tests/conftest.py` から torch の skip が外れたため，`app/learning` を
  触る作業は collection error になる．`uv sync --extra cpu` で
  `torch 2.11.0+cpu` を導入して解消した (CLAUDE.md § Quick Reference の
  手順どおり)．**G3 は発生していない**
- **GPU は無い** — Deferred 5/6/7 の G1 は不変
- **BigQuery は無い** — O9 の G1 は不変
- QA: `ruff format` / `ruff check --fix` / `mypy src/` すべて通過．
  `pytest` は `tests/maou/app/learning` + `tests/maou/app/utility`
  (653 passed / 1 skipped) と全体を実行
