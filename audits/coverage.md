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
| `src/maou/infra/file_system` | python | done | high | `1c6a442` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | 8 deferred |

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
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D2 | `src/maou/infra/file_system` | 行レベル split が無 seed (`file_data_source.py:189`)．全呼び出し側が seed を渡さない (`dl.py:244`, `stage_component_factory.py:99,196` ほか) ため，再開した学習が前回の検証行で訓練する．**見送り理由**: 既定 seed を入れると分割が変わり，同じ判断が path 外の複製2件 (O2) にも及ぶ．**2026-08-13 の `/audit-backlog` で P4 として修正 — PR #TBD で処理中 (未マージ)**．`interface/learn.DEFAULT_SPLIT_SEED` を 3 実装が引く形にしたので O2 の 2 件も同時に解消する．PR がマージされたらこの行を消すこと． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D3+D4 | `src/maou/infra/file_system` | **2026-08-12 に再検証して大幅に縮小した**．記録が挙げる「ディスパッチ4箇所」のうち，`file_data_source.py:559-564` の**呼び出しごとの dict 再構築**と `:905-919` の if/elif は既に解消済み (前の run で module 級 dict へ整理された)．**残り2点**: (a) 3 entry の変換テーブルが 2 モジュールに重複 (`file_data_source.py:50-56` `_DF_TO_COLUMNAR_CONVERTERS` と `streaming_file_source.py:42-48` `_COLUMNAR_CONVERTERS` が同内容)．(b) columnar→structured 変換器が2本ほぼ同一 (`file_data_source.py:582` `_columnar_to_structured_record` / `:704` `_columnar_batch_to_structured_array`，差は `np.empty(1)` + `[idx]` か `np.empty(n)` + 全体かだけ)．**2026-08-13 の `/audit-backlog` で (a)(b) とも修正済み — PR #TBD で処理中 (未マージ)**．(a) は `domain/data/columnar_batch.COLUMNAR_CONVERTERS` へ 1 本化，(b) は `_columnar_batch_to_structured_array(batch, row=idx)` への一本化．PR がマージされたらこの行を消すこと． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D5 | `src/maou/infra/file_system` | `cache_mode` の altitude．両モードとも `__init__` で全ロードし (`:321-422`) 差は結合の有無だけ (`:428-436`)，`total_pages<=1` なら完全同一．`_concatenate_numpy`(`:445`)/`_concatenate_columnar`(`:481`) は入力を保持したまま結合するので**ピーク2×**．OOM 警告 (`:447-458`,`:489-520`) は全ロード後・倍化直前に出るうえ見積りが `move_win_rate` を数え落とす (40GB を 18GB と報告し 32GB 閾値に掛からない)．**見送り理由**: ノブ廃止は O5 と一体． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D8+D9 (残り) | `src/maou/infra/file_system` | **`file_level_split` の削除は 2026-08-12 に出荷済み** ([2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md))．**残り**: `train_test_split`(`file_data_source.py:100-106`) が `list(range(N))` を作るため，5000万行で索引だけ約 2.6GB が同時生存する．`np.random.Generator.permutation` なら C ループ 1 回で済む．**2026-08-13 に依存の向きを確定**: D2 が入るまで公開経路に seed を固定する手段が無かったので，守るべき再現可能な分割は存在しなかった．D2 (PR #TBD) がマージされた**後**は分割値が契約になるため，permutation 化はそれを 1 度だけ意図的に壊す独立の P4 になる — **D2 の後に判断すること**．同時に見つかった障害: `bq_data_source.py:641` の `indicies` は `list[int] | None` 宣言で `np.asarray` を通さずそのまま代入される (`:694`)．`file_system`/`object_storage` は `np.asarray` するので，ndarray を返すと BigQuery だけ型注釈と実体がずれる．3 実装の受け口を先に揃える必要がある． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D10+D11 | `src/maou/infra/file_system` | **D10(a) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．記録が言う「削除」は誤りで，`iter_files_columnar` は `app/learning/streaming_dataset.py:199` の `StreamingSource` プロトコルが宣言するメンバかつテスト 10 本以上から呼ばれるので，削除すると公開名が消える (P6)．`_subset` への委譲に留めた．**残り**: (1) `FileDataSource.total_pages()`(`:898`) はファイル数を返すが `cache_mode="memory"` の `iter_batches` は 1個しか yield しない — **2026-08-12 の再検証で dormant と判明**．`hcpe_transform.py:679` の唯一の caller が受け取るのは `StreamingHcpeDataSource` (`console/pre_process.py:494` が構築) で，そちらの `total_pages()` は `len(self._file_paths)` を返し `iter_batches` も 1 ファイル 1 batch なので tqdm は正しい．`FileDataSource.total_pages()` の production caller は**ゼロ** (テストも無い)．食い違い自体は残っているので，「ファイル数」と「yield 数」のどちらを意味させるかの決めは，この経路に caller が戻るときに要る．(2) 行数スキャン (`_ensure_row_counts`) は逐次かつファイルごとに `open()` 2回で，ネットワーク越し500ファイルでは起動レイテンシの支配項 (D14(a) と同時に直すのが自然)． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D14 | `src/maou/infra/file_system` | **(a) 行数スキャンの共有化．** `streaming_file_source._ensure_row_counts` と `streaming_hcpe_source._ensure_row_counts` の例外安全性の差は `8c1417e` で解消したが，**両者は依然として別実装**であり，安全なのは構造が違う結果の偶然にすぎない．`_scan_row_counts(paths) -> list[int]` を共有し，`StreamingHcpeDataSource` はそのリストを保持して `_total_rows` を導出すべき (現在は per-file カウントを捨てているので sharding 用の `row_counts` を提供できない)．**(b) `FileDataSource` が2つの ABC を着ている．** `preprocess.DataSource` 側の役割は `StreamingHcpeDataSource` に移った (`console/pre_process.py:489-492`) のに継承が残り，`hcpe` を `FileManager` の columnar 機構に通すための `_use_columnar` 分岐 (`:376`, `:605`, `:729`) を生かし続けている．外す場合は `infra/utility/benchmark_polars_io.py:419-451` の対応が要るので path 外の編集を伴う． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D15 | `src/maou/infra/file_system` | **`.feather` で終わる中途書き込みファイルを誰も守っていない．** step 2.5 key 6 の sweep で，`.gstmp`/`.tmp`/`.partial`/`.crc` については露出サイトがゼロと確認できた一方，**中断した `save_*_df` や in-place な `rsync`/`gsutil cp` が残す「拡張子が `.feather` のまま不完全なファイル」**はどのフィルタも通過する．`path_utils.py:10-13` が記述する `OSError: failed to fill whole buffer` がまさにこれ．`_is_temp_artifact` は末尾拡張子リストで判定するので**原理的に捕捉できず**，size/footer 検査が要る．**この行の存在理由**: 元は record の § Cross-module sweep (worklist ではない節) にしか書いておらず，backlog 行がないため `/audit-backlog` から永久に不可視だった (2026-08-10 のユーザー指摘で発見)．運用上のリスクとして実在するかは要判断． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O5 | `src/maou/infra/console` + `src/maou/infra/object_storage` | cache/ノブ系の意味の分裂．(a) `--input-local-cache` は BigQuery にしか渡されず S3/GCS 分岐 (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None` で判定 → **`maou pre-process --input-s3 --input-local-cache` は無言の no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算 (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅 (`object_storage/data_source.py:122,260-265`) と別物．(c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ (`object_storage/data_source.py:212-213` の docstring が明言) で既定値も層をまたいで不一致 (`:45` `True` / `:115`,`:407` `False`)．(d) `learn-model` には `--input-cache-mode` が存在せず `"file"` 決め打ち (`learn_model.py:796,820,847`)． **2026-08-12 の再検証で (a) の記述を訂正**: 「無言の no-op」ではない．`--input-local-cache` (bool flag) と `--input-local-cache-dir` (str) は別のオプションで，S3/GCS の elif が見ているのは**後者**である (`pre_process.py:419`,`:451`)．dir を渡さずに `--input-s3` だけ指定すると elif を全て外れて最後の `else` に落ち，**「Please specify an input source (file path, BigQuery table, GCS bucket, or S3 bucket)」という誤誘導エラーで停止する** (`:497-501`)．黙って無視されるより気付きやすいが，メッセージが原因を指していない．したがって直す向きは「flag を S3/GCS へ渡す」ではなく，**bool flag と dir のどちらがキャッシュを有効にするのかを層をまたいで一致させる**ことになる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:222-243` — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(`:405-420`) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．同じ行集合とは限らず件数も一致しないため `indicies` の範囲 (`:652-655`) と `get_page` が返せる実体がずれる．ファイル系ソースは常に厳密．`:235` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N3 | `src/maou/app/learning` + `src/maou/app/pre_process` | **`@abc.abstractmethod` が不活性．** `dataset.py:45` と `hcpe_transform.py:62` は `@abc.abstractmethod` を付けながら基底に `abc.ABCMeta`/`abc.ABC` を使っていないため，未実装・型違いの実装が**構築時に一切捕まらない**．O1 (`BigQueryDataSource.__getitem__` が `pl.DataFrame` を返していた) が実行時まで露見しなかった根本原因で，O1 を直しても残る．`ABC` を継承させると現存の非準拠実装が全て構築時エラーになるため，何が壊れるかを洗う必要がある (だから O1 と一緒に直せなかった)． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N4 | `tests/maou/infra/file_system` | **torch 無しの環境で `infra/file_system` のテストが丸ごと消える．** `test_file_data_source.py` は `file_data_source.py` → `interface/learn` → torch の 連鎖で，torch 未導入だと**モジュールごと** skip される (`SKIPPED [1] ...: optional dependency 'torch' is not installed`)．**2026-08-12 に再測**: base 環境で **57 passed + 3 skipped**，`uv sync --extra cpu` で **90 passed** (記録当時の 52/83 から増えている)．**実害を確認済み**: 同日の `/audit-backlog` run はこのパッケージを変更したため，CPU extra を入れてからでないと QA が空振りする状態だった (入れずに回すと変更が 無検証のまま緑に見える)．CI/開発環境が base extra だけだと同じことが起きる．**判断が要る点は未解決**: optional dependency に依存しない薄いテストへ切り出すか，最低限 CPU extra を必須にするか． |
| [2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md) N-1 | `rust/maou_io` + `src/maou/domain/data` | **polars が書いた `.feather` と Rust writer が書いた `.feather` は結合できない．** `merge_hcpe_feather_files` が `It is not possible to concatenate arrays of different data types (BinaryView, LargeBinary)` で落ちる．polars 1.38 は `Binary` 列を **BinaryView** で書き，`maou_io` の `save_feather` (arrow-rs) は **LargeBinary** で書くため．**File/Stream 形式とは無関係** — polars 書きの File 形式同士でも Rust 書きと混ぜれば落ちることを実測で確認した (2026-08-12)．`pre-process --input-split-rows` は入力を chunk するので，**writer の違う `.feather` が入力ディレクトリに混在すると停止する**．直す向きは (a) `merge_feather_files` が結合前に schema を正規化する，(b) 書き手を 1 本に揃える，のどちらかで，**どちらを採るかは決めが要る** (BinaryView は polars 側の既定なので (b) は polars 経由の書き出しを全部止めることになる)． |
| [2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md) N-2 | `src/maou/infra/utility/benchmark_polars_io.py` | **`.npy` との性能比較を維持する意味があるか．** doc 側の `.npy` 記述は 2026-08-12 にユーザ判断 (「もう使う予定はない」) で削除した (`reviews/2026-08-12-npy-legacy-support-status.md`, 案 1)．一方コードでは `benchmark_polars_io.py:35-55` が `.npy` の save/load を持ち続けている．CLI からは到達せず (`infra/console/` と `interface/` からの参照ゼロ)，`.feather` との I/O 性能比較にのみ使う．**2026-08-13 の再検証で問いが変わった**: これは保守方針の問いではなく**バグ**だった．`benchmark_datasource_iteration` が `.npy` パスを `FileDataSource` に渡すため `Only .feather files are supported` で必ず落ち，`docs/performance.md:72` が案内する `python -m maou.infra.utility.benchmark_polars_io` は一度も完走していない (到達性を `infra/console/`+`interface/` だけで測り `docs/` を見落としていた)．**ユーザ判断で「`.npy` を全削除」を選択し，P6 として修正 — PR #TBD で処理中 (未マージ)**．同時に 2 つ目の破損も出た (テストデータが preprocessing スキーマに追い付かず `moveWinRate`/`bestMoveWinRate` 欠落で `KeyError`)．PR がマージされたらこの行を消すこと． |
| [2026-08-13 backlog columnar-dedup-and-split-seed](2026-08-13-backlog-columnar-dedup-and-split-seed.md) N5-1 | `src/maou/infra/utility/benchmark_polars_io.py` | **ベンチのテストデータがスキーマから導出されていない．** `_create_hcpe_test_data_polars` / `_create_preprocessing_test_data_polars` は列名も shape (9x9 / 14 / `MOVE_LABELS_NUM`) もハードコードで，スキーマが変わると polars 内部の `KeyError` として現れる (実際 `moveWinRate`/`bestMoveWinRate` の 2 列が欠けたまま気付かれずにいた)．2026-08-13 に `_assert_covers_schema` で**列の欠落だけ**は名指しで落ちるようにしたが，dtype や shape の drift は依然捕まらない．スキーマからの生成に寄せるか，このガードで十分とするかの判断が要る． |
| [2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md) N-3 | `docs/adr-001-dataloader-multiprocessing-optimization.md` | `:143` の 2026-08-09 付の注が targets を「`(labels_policy, labels_value, move_win_rate)` (3 要素目は省略可)」と書いているが，`moveWinRate` を preprocessing の structured dtype に載せた (この run，PR #487) ことで **`KifDataset` は常に 3 要素を返すようになった** (`dataset.py:137-142` の分岐が preprocessing 経路では常に真)．`dataset.py` の 2 要素側の分岐は防御的に残っているが到達しない．**日付入りの ADR 注をどう扱うか** (当時の記述として残す / 追記する / 書き換える) の判断が要るので P2 ではない．**2026-08-13 に提案を起票** (`reviews/2026-08-13-adr-001-targets-note.md`, `status: pending`, 案 1 を推奨)．doc は未編集．承認されたら適用してこの行を消すこと． |
