# Audit coverage ledger

One row per path touched by `/audit-and-fix`. Shape, status vocabulary,
and protocol: [README.md](README.md).

**This table lists only what has been audited.** It is not a plan and not
an inventory of remaining *paths* — to see which paths are left, compare
against the tree (`ls src/maou/*/`, `ls rust/`, `find docs -name '*.md'`),
which is always current where a checked-in list would not be.

Remaining *findings* are a different question, and they **are** inventoried
here: see § "Open findings backlog" below. That is the live worklist; the
per-run records are immutable accounts and are never read to decide what
work remains.

| Path | Scope | Status | Level | Last SHA | Record | Open items |
|---|---|---|---|---|---|---|
| `src/maou/domain/game_graph` | python | done | high | `2686689` | [2026-08-08](2026-08-08-src-maou-domain-game-graph.md) | 0 |
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 6 deferred |
| `src/maou/infra/file_system` | python | done | high | `1c6a442` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | 2 deferred |

## Blocked

_(none)_

<!-- Rows move here only while status is `blocked`, with the blocker and
     what would unblock it. Keep the main table for in-progress/done. -->

## Open findings backlog — the single live worklist

The two tables below are **the** authority on what audit work remains.
Both `/audit-and-fix` and `/audit-backlog` gather candidate work from
here and **only** from here.

**Why this file and not the records.** A per-run record is read only when
someone opens that specific path — so a finding left there is visible
exactly to the audit least able to act on it. This file is read at the
start of every run.

**Why the records are not also consulted.** A record is an *immutable
account of one run at one time*: its Deferred section says "as of that
run, this was deferred", and that stays true forever even after the
finding is fixed. Reading records for open work therefore re-surfaces
resolved findings on every run, with no way to remove them — the ledger
would never shrink. Deleting a row here is what makes a finding
*consumed*, and it is the only mechanism that does.

**Protocol (both tables).**
- **Before auditing a path**, check both tables for rows whose target
  falls inside it and fold them into the run.
- **At the end of a run**, append a row for every finding left open —
  deferred (inside the path) and out-of-scope (outside it) alike.
  Writing it only into the run's record buries it.
- **When a finding is resolved, delete its row.** The resolving record is
  the durable account. Do not delete a row that was merely re-triaged —
  sharpen its text instead.

Records of runs that resolved rows deleted from here:

- [2026-08-09 backlog tier-a](2026-08-09-out-of-scope-tier-a.md)
- [2026-08-09 backlog contained-fixes](2026-08-09-backlog-contained-fixes.md)
- [2026-08-09 backlog streaming-len-and-docs](2026-08-09-backlog-streaming-len-and-docs.md)
- [2026-08-09 backlog tier-3-contracts](2026-08-09-backlog-tier-3-contracts.md)
- [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md)
- [2026-08-12 backlog n5-fill-accepted](2026-08-12-backlog-n5-fill-accepted.md)
- [2026-08-12 backlog auto-band-and-n1](2026-08-12-backlog-auto-band-and-n1.md)
- [2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md)
- [2026-08-13 backlog columnar-dedup-and-split-seed](2026-08-13-backlog-columnar-dedup-and-split-seed.md)
- [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md)
  ([PR #492](https://github.com/dousu/maou/pull/492) — D8+D9 / N3 / N-1 /
  N5-1 の 4 行を削除．D14 は **(a) だけ**を消化したので行は残り，(b) の
  記述に縮めてある)
- [2026-08-13 backlog oom-estimate-and-bq-contract](2026-08-13-backlog-oom-estimate-and-bq-contract.md)
  (**行の削除なし** — 消化した 4 件はいずれも行の一部だけで，かつ PR が
  未マージ．O5 は (c) の doc drift だけ，D5 は見積り部分だけ，
  D10+D11 は (2) だけを消化し，各行をその旨に縮めてある．N6-1 は PR が
  マージされたら削除する)

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:422`/`:571`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:646`/`:735`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279` (file untouched since the record, so those still hold) and `_yield_stage1_batches`/`_yield_stage2_batches` (`streaming_dataset.py:851`/`:911`). **~400-line refactor of the multi-stage training path — architecturally significant.** (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 3 | `src/maou/app/learning` | Six adapter classes are three duplicated pairs. `Stage1ModelAdapter`/`Stage2ModelAdapter` (`multi_stage_training.py:111`/`:240`) differ in **zero** characters; `Stage1DatasetAdapter`/`Stage2DatasetAdapter` (`:151`/`:183`) in one type annotation; `Stage1StreamingAdapter`/`Stage2StreamingAdapter` (`streaming_dataset.py:721`/`:686`) in a redundant `hasattr` guard. Merging also deletes the `isinstance` dispatch + `TypeError` arm at `stage_component_factory.py:876-882`, which exists only to choose between two identical classes. Six public names referenced from tests — should land as its own reviewed change. (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 4 | `src/maou/app/learning` | `callbacks.py` — `_ensure_device` written six times (`:238`, `:362`, `:1044`, `:1433`, `:1558`, `:1705`), plus three copies of the loss-accumulator scaffolding (in `Stage2F1Callback`, `Stage1AccuracyCallback`, `Stage3LossCallback` — the record's `:1375`/`:1499`/`:1652` are now those class bodies, +37 lines). `ValidationCallback` hand-lists the same 13 accumulator tensors in three places (`__init__` / `_ensure_device` / `reset`) — the exact shape that produces "new metric added, never moved to GPU, never reset" defects. Base-class extraction across the module's metric hub (~250 → ~120 lines). (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1110` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per batch, to guard a warning. **Now dormant, not fixed** (2026-08-09, backlog `tier-3-contracts`): the record's premise "Stage 3 always ships a `legal_move_mask`, so the branch is always taken" no longer holds — no data path supplies a mask, so `_compute_policy_loss` never enters the masking arm and the sync does not execute. The code is unchanged and the stall returns the moment a real legal-move mask is wired in; fix it **then**, together with whatever produces the mask, and measure on GPU. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 6 | `src/maou/app/learning` | `training_loop.py:460` `stream.synchronize()` blocks the host, defeating much of the prefetch it implements. `wait_stream()` gives the same ordering guarantee device-side without stalling the CPU, and the `record_stream()` added in `073adbd` already covers the allocator hazard. **Second untested GPU-semantics change stacked on the first** — validate both together on real hardware. |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D5 | `src/maou/infra/file_system` | **見積りの数え落としは 2026-08-13 に消化済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md))．合計は `ColumnarBatch.nbytes` として `dataclasses.fields` から導出するようになり，警告は `_warn_if_oom_risk` に一本化，閾値は `OOM_WARNING_THRESHOLD_GB` に定数化された．**残り**: `cache_mode` の altitude 本体．両モードとも `__init__` で全ロードし差は結合の有無だけ，`total_pages<=1` なら完全同一．`_concatenate_numpy`/`_concatenate_columnar` は入力を保持したまま結合するので**ピーク2×**で，警告は全ロード後・倍化直前にしか出ない (見積りが正しくなっても手遅れであることは変わらない)．**見送り理由**: ノブ廃止は O5 と一体． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D10+D11 | `src/maou/infra/file_system` | **D10(a) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．記録が言う「削除」は誤りで，`iter_files_columnar` は `app/learning/streaming_dataset.py:199` の `StreamingSource` プロトコルが宣言するメンバかつテスト 10 本以上から呼ばれるので，削除すると公開名が消える (P6)．`_subset` への委譲に留めた．**残り**: (1) `FileDataSource.total_pages()`(`:898`) はファイル数を返すが `cache_mode="memory"` の `iter_batches` は 1個しか yield しない — **2026-08-12 の再検証で dormant と判明**．`hcpe_transform.py:679` の唯一の caller が受け取るのは `StreamingHcpeDataSource` (`console/pre_process.py:494` が構築) で，そちらの `total_pages()` は `len(self._file_paths)` を返し `iter_batches` も 1 ファイル 1 batch なので tqdm は正しい．`FileDataSource.total_pages()` の production caller は**ゼロ** (テストも無い)．食い違い自体は残っているので，「ファイル数」と「yield 数」のどちらを意味させるかの決めは，この経路に caller が戻るときに要る．**(2) 行数スキャンの逐次実行は 2026-08-13 に消化済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md))．`scan_row_counts` は File 形式のメタデータ読みだけを `ThreadPoolExecutor` に載せる 2 相構成になった (Stream 形式は全読みが要りピークメモリがワーカー数倍になるので逐次のまま)．**便益は未測定** (G1: 数百ファイルのネットワークストレージがこの環境に無い)．**残るのは (1) の意味の決めだけ**． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**2026-08-13 の再検証**: `FileDataSource.get_items` の呼び出し側は**ゼロ** (`FileManager.get_items` への内部委譲だけ) なので (b) 単独では **dormant**．実害は (a) の根本解決に踏み込んだときに初めて出るため，「(b) を先に直す」選択肢は実質無い．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D14 (残り) | `src/maou/infra/file_system` | **(a) 行数スキャンの共有化は 2026-08-13 に出荷済み** ([PR #492](https://github.com/dousu/maou/pull/492), [記録](2026-08-13-backlog-scan-share-and-abc.md))．`domain/data/arrow_format.scan_row_counts` を 2 実装が引くようになり，`StreamingHcpeDataSource` は per-file カウントを `row_counts` で公開する．**残り**: **(b) `FileDataSource` が2つの ABC を着ている．** `preprocess.DataSource` 側の役割は `StreamingHcpeDataSource` に移った (`console/pre_process.py:489-492`) のに継承が残り，`hcpe` を `FileManager` の columnar 機構に通すための `_use_columnar` 分岐 (`:376`, `:605`, `:729`) を生かし続けている．外す場合は `infra/utility/benchmark_polars_io.py:419-451` の対応が要るので path 外の編集を伴う． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D15 | `src/maou/infra/file_system` | **`.feather` で終わる中途書き込みファイルを誰も守っていない．** step 2.5 key 6 の sweep で，`.gstmp`/`.tmp`/`.partial`/`.crc` については露出サイトがゼロと確認できた一方，**中断した `save_*_df` や in-place な `rsync`/`gsutil cp` が残す「拡張子が `.feather` のまま不完全なファイル」**はどのフィルタも通過する．`path_utils.py:10-13` が記述する `OSError: failed to fill whole buffer` がまさにこれ．`_is_temp_artifact` は末尾拡張子リストで判定するので**原理的に捕捉できず**，size/footer 検査が要る．**2026-08-13 の再検証で分かったこと**: 行数スキャン (`domain/data/arrow_format.scan_row_count`) は全ファイルの Arrow footer を読むので，途中書きファイルは**そこで既に落ちている** — 行が引く `OSError: failed to fill whole buffer` がまさにそれ．つまり「素通しして壊れたデータを読む」のではなく「原因を指さないメッセージで落ちる」が実態．したがって判断は「全ファイルに検査を足すか」ではなく，**「既に落ちているこの失敗を，どこでどう分かりやすくするか」**に寄せられる (前者より格段に安い)．**この行の存在理由**: 元は record の § Cross-module sweep (worklist ではない節) にしか書いておらず，backlog 行がないため `/audit-backlog` から永久に不可視だった (2026-08-10 のユーザー指摘で発見)．運用上のリスクとして実在するかは要判断． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O5 | `src/maou/infra/console` + `src/maou/infra/object_storage` | cache/ノブ系の意味の分裂．(a) `--input-local-cache` は BigQuery にしか渡されず S3/GCS 分岐 (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None` で判定 → **`maou pre-process --input-s3 --input-local-cache` は無言の no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算 (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅 (`object_storage/data_source.py:122,260-265`) と別物．(c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ (`object_storage/data_source.py:199` の docstring が明言し，`:288-320` は各 `.feather` を無条件に個別保存する) で既定値も層をまたいで不一致 (`:45` `True` / `:102`,`:394` `False`)．**(c) の doc drift 部分は 2026-08-13 に消化済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md), [提案](../reviews/2026-08-13-bundling-knobs-are-no-ops.md)) — docs 3 本に「受理するが効果なし」と明記した．**ノブ自体の削除 (P6) と既定値の不一致は残る** ((a)(b)(d) と同じ「どちらがキャッシュを有効にするのか」の決めと一体)．(d) `learn-model` には `--input-cache-mode` が存在せず `"file"` 決め打ち (`learn_model.py:796,820,847`)． **2026-08-12 の再検証で (a) の記述を訂正**: 「無言の no-op」ではない．`--input-local-cache` (bool flag) と `--input-local-cache-dir` (str) は別のオプションで，S3/GCS の elif が見ているのは**後者**である (`pre_process.py:419`,`:451`)．dir を渡さずに `--input-s3` だけ指定すると elif を全て外れて最後の `else` に落ち，**「Please specify an input source (file path, BigQuery table, GCS bucket, or S3 bucket)」という誤誘導エラーで停止する** (`:497-501`)．黙って無視されるより気付きやすいが，メッセージが原因を指していない．したがって直す向きは「flag を S3/GCS へ渡す」ではなく，**bool flag と dir のどちらがキャッシュを有効にするのかを層をまたいで一致させる**ことになる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:222-243` — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(`:405-420`) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．同じ行集合とは限らず件数も一致しないため `indicies` の範囲 (`:652-655`) と `get_page` が返せる実体がずれる．ファイル系ソースは常に厳密．`:235` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N4 | `tests/maou/infra/file_system` | **torch 無しの環境で `infra/file_system` のテストが丸ごと消える．** `test_file_data_source.py` は `file_data_source.py` → `interface/learn` → torch の 連鎖で，torch 未導入だと**モジュールごと** skip される (`SKIPPED [1] ...: optional dependency 'torch' is not installed`)．**2026-08-12 に再測**: base 環境で **57 passed + 3 skipped**，`uv sync --extra cpu` で **90 passed** (記録当時の 52/83 から増えている)．**実害を確認済み**: 同日の `/audit-backlog` run はこのパッケージを変更したため，CPU extra を入れてからでないと QA が空振りする状態だった (入れずに回すと変更が 無検証のまま緑に見える)．CI/開発環境が base extra だけだと同じことが起きる．**判断が要る点は未解決**: optional dependency に依存しない薄いテストへ切り出すか，最低限 CPU extra を必須にするか．**2026-08-13 に 4 run 連続で実害を確認** — 同日の 2 本目の run ([記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md)) はコンテナが再作成されて **venv が空**から始まり，`uv sync --extra cpu` に約 7 分・`maturin develop --release` に 12 分を要した．この run も 4 件中 2 件が `infra/file_system` に触れているので，入れずに QA を回していたら変更が無検証のまま緑に見えていた． |
| [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md) N6-1 | `src/maou/infra/bigquery` | **`BigQueryDataSource.iter_batches` の返り値が基底の宣言と違う．** `bq_data_source.py:710` は `Generator[tuple[str, pl.DataFrame], None, None]` を返すが，基底 (`hcpe_transform.py:77`) の宣言は `np.ndarray`．`PreProcess` は `hcpe_transform.py:683`/`:839` でこれを `_process_single_array(data: np.ndarray)` に渡し，そこは `data["hcp"]` / `np.ascontiguousarray` をする．さらに `BigQueryDataSource` は `iter_batches_df` を override しないので，基底の既定実装 (`:117` の `array.dtype.names`) に `pl.DataFrame` が渡って `AttributeError` になる．**`__getitem__` について同じ形の不具合が O1 として既に直っており (`tests/maou/infra/bigquery/test_bq_get_item_contract.py`)，`iter_batches` にだけ同じ手当てがされていない．** N3 の ABC 化では捕まらない (ABC はメソッドの存在を見るが型は見ない)．**2026-08-13 に修正を出荷済み・PR 未マージ** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md))．**記録が挙げていた G1 は撤回した** — O1 の前例 (`PageManager` を `object.__new__` で作り `get_page` だけ差し替える) で BigQuery 実環境なしに検証できる．**PR がマージされたらこの行を削除すること**． |
| [2026-08-13 backlog oom-estimate-and-bq-contract](2026-08-13-backlog-oom-estimate-and-bq-contract.md) N7 | `tests/maou/infra/utility` | **既存の mypy エラー 1 件が Python を触る全コミットをブロックしていた．** `test_benchmark_polars_io.py:30` の `capsys` に型注釈が無く `no-untyped-def` になる．`.pre-commit-config.yaml` の mypy hook は `pass_filenames: false` + `args: ["src/", "tests/"]` なので，変更ファイルに関係なく毎回 `src/` と `tests/` の 296 ファイルを見る — つまり **1 行でも Python を触ると pre-commit が落ちる**状態だった．`mypy src/` (135 ファイル) だけを回して緑を確認するのでは検出できない．**修正は [PR #493](https://github.com/dousu/maou/pull/493) に同梱済み** (注釈は pytest の API から一意: `pytest.CaptureFixture[str]`)．**PR がマージされたらこの行を削除すること**．PR が閉じられた場合は次に Python を触る run が即座に踏むので，行が無くても自然に再発見されるが，記録として残す．なぜ HEAD に入り得たのか (`e7c5d3e` / `d0c4984` のどちらかが mypy の差分キャッシュをすり抜けた可能性) は未調査． |
| [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md) N6-2 | `src/maou/app/pre_process` | **基底の具象 `iter_batches_df` が HCPE スキーマ決め打ち．** `hcpe_transform.py:86-139` の既定実装は `get_hcpe_polars_schema()` を直に引くので，`preprocessing` 型のソースに対しては黙って誤動作する．現状 production の caller はすべて override 側 (`FileDataSource` / `ObjectStorageDataSource`) を通るので **dormant**．「HCPE 専用と明記して名前を変える」か「array_type で分岐させる」かの判断が要る．**2026-08-13 の再検証で dormant の度合いが増した**: N6-1 の修正で `BigQueryDataSource` も `iter_batches_df` を override したため，基底の既定実装を通る production 経路は**ゼロ**になった．直す動機は「将来の実装者が既定実装を踏む」ことだけ． |
