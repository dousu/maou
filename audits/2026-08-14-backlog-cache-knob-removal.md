---
kind: backlog
date: 2026-08-14
path:
  - src/maou/infra/file_system/file_data_source.py
  - src/maou/infra/bigquery/bq_data_source.py
  - src/maou/infra/console/pre_process.py
  - src/maou/infra/console/utility.py
  - src/maou/infra/console/learn_model.py
  - src/maou/infra/utility/benchmark_polars_io.py
  - src/maou/interface/learn.py
  - src/maou/app/learning/dl.py
  - src/maou/domain/data/columnar_batch.py
  - README.md
  - docs/commands/pre_process.md
  - docs/commands/utility_benchmark_dataloader.md
  - docs/commands/utility_benchmark_training.md
  - docs/rust-backend.md
scope: python + docs
level: medium
last_sha: 155cc73
---

# `/audit-backlog` — 決定済み設計判断 2 件を実装し，新たに 3 件の判断を得た

`audits/coverage.md` の backlog 12 行 (Deferred 7 + Out-of-scope 5) を
全て HEAD (`155cc73`) に対して再検証した．**自動帯は空** (9 run 連続) で，
12 行すべてが P4 以上だった．

この run の特徴は，**前 run が得た設計判断のうち未実装だった 3 件が
「人間待ちではないただの作業」として残っていた**ことである．前 run
(`3dea4e8`) は step 3d を初めて行使して 4 問すべてを設計判断に充て，
4 件の回答を得たが実装まで到達したのは 1 件だけだった．残り 3 件が
この run の入口になった．

前 run が「決定を行に書く」ことの価値として予告した
「次 run は通常の作業として拾える」が，実際に機能したことを示す最初の
run である．

## 再検証 (step 2)

12 行すべてを開いた．**stale 0 / changed shape 0 / confirmed 12**．

前 run (`2312f65`) 以降に `main` へ入ったのは前 run 自身の 4 commit
(`b5f2348` / `3e0a498` / `d669eca` / `3dea4e8`) だけで，触れたのは
`tests/conftest.py` / `tests/test_conftest_optional_deps.py` /
`docs/testing-guide.md` / `reviews/` / `audits/` のみ．よって backlog が
指す `src/` の行番号は 1 つも動いていない．実際，Deferred 2/3/5/6/7 /
D13 / O9 / N6-2 / D14(b) の引用箇所を直接開いて全て一致を確認した．

**行番号の更新は無し** — 8 run 続いていた「行がずれる」現象が初めて
起きなかった run である (前 run が `src/` に触れていないため)．

## Classification (step 3a)

**自動帯 (P1-P3) は空．** 12 行すべてが P4 以上だった．

| ID | 行 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|
| B1 | D5 `cache_mode` | P6 | 公開パラメータ + CLI オプションが消える | — (前 run で G4 retire) |
| B2 | O5(d) `learn-model` の非対称 | P6 | B1 と同一の変更が governs | — (同上) |
| B3 | O5(a) local-cache flag/dir | P6 | `--input-local-cache` が消える | — (同上) |
| B4 | O5(c) bundling ノブ削除 | P6 | CLI オプションが消える | G4 |
| B5 | D10+D11(1) `total_pages()` の意味 | P4 | 戻り値の意味が変わる | G4 → **B1 で消滅** |
| B6 | D14(b) 二重 ABC | P6 | 公開の継承関係が消える | G2 (前 run で G4 retire) |
| B7 | N6-2 基底 `iter_batches_df` | P6 | 改名 | G2 (同上) |
| B8 | D13 columnar スライス | P4/P6 | 決め次第で公開名が消える | G2, G4 |
| B9 | O9 BigQuery サンプリング | P4 | 同じ呼び出しで返る行が変わる | G4, G1 |
| B10 | Deferred 2 Stage1/2 統合 | P4 | ~730 行の挙動保存リファクタ | G3, G4 |
| B11 | Deferred 3 アダプタ 6→3 | P4 | 公開名 6 つ + テスト書き換え | G2, G4 |
| B12 | Deferred 5 per-batch 同期 | P4 | dormant | G1 |
| B13 | Deferred 6 `stream.synchronize()` | P4 | GPU 意味論の変更 | G1 |
| B14 | Deferred 7 GNS 同期 | P4 | 数値等価性の確認が要る | G1 |

### B5 を「消滅」と判定した根拠

D10+D11(1) は「`total_pages()` はファイル数を返すが
`cache_mode="memory"` の `iter_batches` は 1 個しか yield しない」という
食い違いで，前 run の待ち行列の**筆頭**だった．前 run 自身が
「Q1 の回答 (`cache_mode` 削除) がこの行を消滅させる可能性がある」と
書いており，それを再検証で確認した:

- 結合を起こす分岐は `file_data_source.py:301-308` の
  `if self.cache_mode == "memory" and self.total_pages > 1:` **1 箇所だけ**
- `iter_batches` が 1 batch を返すのは `_concatenated_array` /
  `_concatenated_columnar` が非 None のときだけで，これらは
  `_concatenate_numpy` `:356` / `_concatenate_columnar` `:391` でしか
  代入されず，その 2 つの production 呼び出し元は上の 1 箇所のみ

よって B1 を出荷すると食い違いは**構造的に起き得なくなる**．
「ファイル数と yield 数のどちらを意味させるか」という決めは不要になった．

## Applied

**PR [#501](https://github.com/dousu/maou/pull/501)** / commit `bdda7b5`．
`pyproject.toml` **0.89.12 → 0.90.0** (breaking change / 0.x なので minor)．

### Q1 の実装 — `cache_mode` ノブごと削除 (D5 + O5(d))

| ファイル | 変更 |
|---|---|
| `file_data_source.py` | `CacheMode` 型別名，`FileDataSource` / `FileManager` / `FileDataSourceSpliter` の `cache_mode` 引数，正規化と検証，`_concatenate_numpy` / `_concatenate_columnar` / `_warn_if_oom_risk` / `OOM_WARNING_THRESHOLD_GB`，`_concatenated_array` / `_concatenated_columnar` とその読み出し (`get_item` / `iter_batches`) を削除．`iter_batches` の docstring に「yield 数は `total_pages` と常に一致する」を明記 |
| `interface/learn.py` | `learn()` `:286` と `learn_multi_stage()` `:1017` の `input_cache_mode` 引数，`:513-517` の検証，2 つの pass-through，docstring 2 箇所を削除 |
| `app/learning/dl.py` | `Learning.LearningOption.input_cache_mode` フィールドと，設定ログの `cache=` 部分を削除 |
| `console/utility.py` | `--input-cache-mode` の click option 宣言 2 本 (benchmark-dataloader / benchmark-training)，`mmap` deprecation の正規化ブロック 2 本，コールバック引数 2 つ，pass-through / リテラル 4 箇所を削除 |
| `console/learn_model.py` | `cache_mode="file"` リテラル 2 箇所と `_s3_cache` / `_c` の lambda 捕捉を削除 |
| `infra/utility/benchmark_polars_io.py` | `cache_mode="memory"` (このリポジトリで唯一の無条件 `memory` 利用箇所) を削除 |
| `domain/data/columnar_batch.py` | `nbytes` の docstring が削除済みの警告を現存機能として説明していたので，過去形に改め **production の呼び出し元が無くなった旨を明記** |

### Q4 の実装 — ローカルキャッシュの有効化を dir に一本化 (O5(a))

| ファイル | 変更 |
|---|---|
| `bigquery/bq_data_source.py` | `BigQueryDataSourceSpliter` / `PageManager` / `BigQueryDataSource` の `use_local_cache` 引数を削除．`self.use_local_cache = local_cache_dir is not None` へ導出に変更 (`:123`)．`local_cache_dir is None` の `ValueError` は構造的に到達不能になったので削除．なぜ 1 箇所に寄せるのかをコメントで明記 |
| `console/pre_process.py` | `--input-local-cache` の option 宣言，コールバック引数，`use_local_cache=` の pass-through を削除 |
| `console/utility.py` | 同じ 3 点を benchmark 2 コマンド分削除 |

`use_local_cache` **属性そのものは `PageManager` 内部に残る** — LRU の
要否 (`:139`)，全ダウンロードの起動 (`:209`)，退避の短絡 (`:242`)，
`get_page` の分岐 (`:493`) が今も読む．消したのは**外から与える手段**
だけである．

### 回帰テスト

全 8 本．**いずれも修正を無効化すると失敗することを実測**した (非空虚性)．

| テスト | 何を固定するか | 無効化して失敗を確認した方法 |
|---|---|---|
| `TestBatchCountMatchesTotalPages` (3 本) | yield 数 == `total_pages()`，batch 名がファイル名，hcpe 経路も同じ | `iter_batches` に結合経路を再導入 → 3 本とも失敗 |
| `test_input_cache_mode_is_gone_everywhere` | 3 コマンドすべてに option が無い | (option 復活で失敗するのは自明だが未実測) |
| `test_no_command_advertises_a_cache_mode_option` | `cache-mode` を名に含む option がゼロ | 同上 |
| `test_input_local_cache_flag_is_gone` | bool flag が 3 コマンドに無い | `pre_process.py` に option を戻す → 失敗 |
| `test_input_local_cache_dir_is_the_only_switch` | `input-local-cache*` が dir 1 本だけ | 同上 → 失敗 |
| `test_local_cache_is_dir_driven.py` (3 本) | `use_local_cache` を外から渡せない / dir から導出 | 導出を `bool(local_cache_dir)` に変更 → 失敗 |

**削除したテスト**:

- `test_file_data_source_oom_estimate.py` (ファイルごと) — 対象の
  `_concatenate_*` / `_warn_if_oom_risk` が消えた．なお
  `ColumnarBatch.nbytes` 自体の回帰テストは
  `test_columnar_batch.py::TestColumnarBatchNbytes` に残っており，
  「列挙せず `dataclasses.fields` から導出する」規律は今も固定されている
- `test_cli_option_compatibility.py` の `--input-cache-mode` 一貫性テスト
  3 本 — 「存在すること」を固定していたので「存在しないこと」を固定する
  テストへ置き換えた
- `test_learn_model_passes_cache_mode` — 元から `@pytest.mark.skip`
  ("Needs update for .feather files") で，対象の option も消えた

`test_file_data_source_memory_cache` / `test_file_data_source_mmap_cache`
は単一ファイルで走らせており**元から挙動が同一**だったので，
`test_file_data_source_repeated_access` / `test_file_data_source_single_file`
へ改名して意味を実態に合わせた．

## Decisions asked (step 3d)

`AskUserQuestion` を 1 回．**受理 1 問 + 設計判断 3 問**．
前 run が 4 問すべてを設計判断に充てたのに対し，この run は
出荷物ができたので 1 枠を受理に使った (3d の予算配分どおり)．

### Q1 — PR #501 の受理 (受理)

提示した選択肢: (1) 両方マージする (推奨) / (2) `cache_mode` の廃止だけ /
(3) local-cache の一本化だけ / (4) マージしない．

無効になるコマンドラインを表で具体的に示した (`--input-cache-mode` と
`--input-local-cache`，および「dir 無しで flag だけ」が
`ValueError` 停止から**キャッシュ無効での正常実行**に変わる点)．
データ互換性の破壊が無いことも明示した．

**ユーザの回答: (1) 両方マージする．** → マージ済み．

### Q2 — 不活性な bundling ノブの去就 (設計判断)

**settles: O5(c)**

提示した選択肢: (1) ノブごと削除 (推奨) / (2) バンドリングを実装する /
(3) deprecation 警告を出して残す / (4) 現状維持．

**ユーザの回答: (1) ノブごと削除．**

**この run では実装していない．** 理由は予算ではなく**scope の誠実さ**で
ある — PR #501 は「cache 系ノブ 2 つの廃止」という明示した内容で受理を
得ており，そこへ 3 つ目の P6 変更を後から足すのは silent widening に
なる．O5 行に決定を書き G4 を retire した．次 run の筆頭作業．

### Q3 — BigQuery サンプリングの非再現性 (設計判断)

**settles: O9**

提示した選択肢: (1) 一時テーブルへ実体化 (推奨) / (2) 決定的ハッシュ条件へ
置き換える / (3) `sample_ratio` とページングの併用を拒否 / (4) 現状維持．

**ユーザの回答: (2) 決定的ハッシュ条件に置き換える** — 推奨とは違う側．
状態 (一時テーブルとその寿命管理) を持たない方を選好した形になる．

**この run では実装していない** (G1: BigQuery が無い)．O9 行に決定を書き
G4 を retire，**G1 は据え置き**．行には実装時に要る 4 点
(キー列の決め / COUNT とページで同じ条件式を使う / クラスタ経路も同様に /
`ORDER BY` 不在は別途) を書き出した．**キー列の決めは未解決** — 現行
スキーマに高カーディナリティの一意キーがあるかは調べていないので，
次 run はそこから始まる．

### Q4 — アダプタ 6 クラスの統合 (設計判断)

**settles: Deferred 3**

提示した選択肢: (1) 統合，`set_epoch` ガードは**付ける**側で揃える (推奨) /
(2) 統合，ガードは**消す**側で揃える / (3) Model 対と Dataset 対だけ統合 /
(4) 現状維持．

**ユーザの回答: (1) 統合，ガードは付ける側で揃える．**

前 run が changed shape として拾った「Stage 1 は `hasattr` で守るが
Stage 2 は無防備」という挙動差に，安全側で決着した．Stage 2 側の挙動が
変わる (落ちなくなる) ので P4 は不変．**G4 は retire，G2 は残る** —
`test_stage_component_factory.py:297`/`:398` の `isinstance` アサーションの
書き換えとセットでしか出荷できない．

### 予算に入らなかった設計判断 (次 run の待ち行列，3d のランク順)

1. **B8 — D13 columnar スライスの根本解決**．前 run から順位が 1 つ
   上がった (筆頭だった B5 が消滅したため)．ただし依然として
   **D14(b)/N6-2 の実装後の方が形が定まる** (`FileDataSource` の
   ABC 構成が変わる)．次 run が D14(b)+N6-2 を実装するなら，その後に問う．
2. **B10 — Deferred 2 (Stage1/2 パイプラインの統合)**．答えが出ても
   1 run では出荷できない規模 (~730 行, G3)．Deferred 3 の統合が
   先に入れば footprint が縮む可能性がある．

**B12 / B13 / B14 (Deferred 5 / 6 / 7) は今回も待ち行列に入れていない**
— 設計の分岐ではなく「向きは判っているが GPU が無くて検証できない」
類なので，問うても work が unblock しない (前 run と同じ判断)．

## In flight

**なし．** 判断帯の PR は #501 の 1 本だけで，質問は同一セッション内で
回答を得てマージした．

## Re-triaged

**5 行** — 人間待ち or 環境待ちのまま，文言だけ鋭くして残した:
D13 / D14(b) / N6-2 / Deferred 2 / Deferred 5 / Deferred 6 / Deferred 7．

うち **D14(b) と N6-2 は「decided」側** (前 run で G4 retire 済み) で，
この run では**着手しなかった**．理由は G2 (テスト 3 ファイルの移行と
セット) に加え，B1/B3 と同じ `console/` 系を触るので同一 run に載せると
diff が読めなくなること．**次 run の筆頭候補**として残す．

## Corrections to the source records

**なし．**

再検証で記録の診断が誤っていた例は無かった．行番号のずれも 0 だった
(前 run が `src/` に触れていないため)．

## Doc findings

`reviews/2026-08-14-cache-knobs-removed.md` — **status: applied**．

**P2 の恒久承認で適用した．** 削除された option の行を消す以外の書き方が
無く，訂正後の本文が現行コードから一意に決まる drift correction である．

対象は `README.md` (読み込み方式の節)，
`docs/commands/utility_benchmark_dataloader.md` と
`utility_benchmark_training.md` (`--input-cache-mode` の行を削除，
BigQuery cache knobs 行から `--input-local-cache` を除去)，
`docs/rust-backend.md` (コード例から `cache_mode="file",` を除去)．

`docs/commands/pre_process.md` にも 1 文足した．
`scripts/check-cli-docs.sh` が `pre_process.py` の変更時に同 doc の
ステージングを要求するためだが，**この doc には誤りが無かった** (bool
flag を一度も載せていない)．書いたのは「ローカルキャッシュは
`--input-local-cache-dir` を渡すと有効になり，別の on/off flag は無い」
という，コードから一意に決まる 1 文である．

**触らなかった**: `docs/commands/learn_model.md:189-191` の 2026-02-22 の
変更履歴 (過去の事実として今も正しい)．

## Out of scope

この run が新たに気づいた所見 **1 件**．coverage.md には**起票していない**
— 理由は下記のとおり，既存行の実装で自然に解決するため．

- **`ColumnarBatch.nbytes` の production 呼び出し元がゼロになった．**
  唯一の呼び出し元だった `_concatenate_columnar` を削除したため，
  現在の参照はテスト (`test_columnar_batch.py::TestColumnarBatchNbytes`)
  だけである．削除するか残すかは判断が要るが，**O5(c) や D13 の実装で
  メモリ見積りの呼び出し元が戻る可能性がある**ので，独立した行として
  起票するより，そのときに再評価する方が安い．docstring に
  「現在 production の呼び出し元は無い」と明記して，次の読み手が
  同じ調査を繰り返さないようにした．

## Environment notes

- **コンテナは再作成されていた** (9 run 連続)．venv は site-packages
  2 エントリの空から始まり，`uv sync --extra cpu` に約 6 分かかった．
  Rust 拡張 (`maou._rust`) は同期後に import 可能で
  `maturin develop` は不要だった．
- **QA は全て実行できた — G3 は無し．**
  `uv run ruff format src/ tests/` / `ruff check --fix` / `mypy src/`
  (135 files) はいずれも green．
  全スイート **1976 passed, 53 skipped** (89.67s)．
  pre-commit のフルスイート + `check-cli-docs` を通してコミットした
  (`--no-verify` は不使用)．
- **CI は PR にテストを走らせない** (`claude-code-review.yml` は
  `workflow_dispatch` のみ)．したがって上の実測が唯一の検証である．
- **G1 は Deferred 5/6/7 と O9 に残る** — GPU も BigQuery もこの環境に
  無い．O9 は今回設計判断を得たので，G4 は消えて G1 だけが残る形に
  なった．
