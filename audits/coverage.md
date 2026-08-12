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
| `src/maou/infra/file_system` | python | done | high | `1c6a442` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | 10 deferred |

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
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D1 | `src/maou/infra/file_system` + `src/maou/domain/data` | **`moveWinRate` が structured record に載らない．** `domain/data/schema.py:136` `get_preprocessing_dtype()` にフィールドがなく，`file_data_source.py:612`/`:734` も `batch.move_win_rate` を読まない → `policy_targets.py:57` が `ValueError`．**CLI 既定の `--policy-target-mode win_rate` で `learn-model --no-streaming` が初回ステップで落ちる** (streaming 経路は通る)．同時に `(N,1496) float32` が preprocessing 常駐 RAM の約66% を占めて一度も読まれない．**見送り理由**: 根本原因が path 外の domain 層，`get_preprocessing_dtype` は4層6箇所以上から使用．「dtype に足す」(streaming と parity) と「変換直後に捨てる」(RAM 66%削減) は別方向で設計判断が要る． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D2 | `src/maou/infra/file_system` | 行レベル split が無 seed (`file_data_source.py:189`)．全呼び出し側が seed を渡さない (`dl.py:244`, `stage_component_factory.py:99,196` ほか) ため，再開した学習が前回の検証行で訓練する．**見送り理由**: 既定 seed を入れると分割が変わり，同じ判断が path 外の複製2件 (O2) にも及ぶ． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D3+D4 | `src/maou/infra/file_system` | array_type→loader ディスパッチがパッケージ内に4箇所 (`file_data_source.py:42-63`，`:559-564` は**呼び出しごとに dict 再構築**，`:905-919` の if/elif，`streaming_file_source.py:34-48`)．加えて columnar→structured 変換器が2本ほぼ同一 (`:612`/`:734`，各約60行，差は `np.empty(1)`/`np.empty(n)` のみ)．D1 の修正時に2箇所直す必要があるのはこれが理由． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D5 | `src/maou/infra/file_system` | `cache_mode` の altitude．両モードとも `__init__` で全ロードし (`:321-422`) 差は結合の有無だけ (`:428-436`)，`total_pages<=1` なら完全同一．`_concatenate_numpy`(`:445`)/`_concatenate_columnar`(`:481`) は入力を保持したまま結合するので**ピーク2×**．OOM 警告 (`:447-458`,`:489-520`) は全ロード後・倍化直前に出るうえ見積りが `move_win_rate` を数え落とす (40GB を 18GB と報告し 32GB 閾値に掛からない)．**見送り理由**: ノブ廃止は O5 と一体． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D6+D7 | `src/maou/infra/file_system` | 死んだ状態と死枝: `_FileEntry.memmap`/`.dtype` は placeholder (columnar 分岐は `:369` で `np.dtype("uint8")` という嘘の値を入れる)，`memmap_arrays`(`:266`)・`_last_file_idx`(`:276`)・`bit_pack`(`:257`) は未読．`memmap_arrays` は `object_storage/data_source.py:135,178` では生きた機構で，同名が片方だけ化石化．`iter_batches_df:890` の `isinstance(pl.DataFrame)` は到達不能なので常に else へ落ち `.feather` を**全件再読込**する． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D8+D9 | `src/maou/infra/file_system` | `file_level_split`(`:119`) は避けるはずの全ロードを払わないと呼べない — `FileDataSourceSpliter.__init__:89` が `FileManager` を構築＝全ロードし，そこから `file_paths`/`array_type` の2つしか使わない (`interface/learn.py:1306-1312` が「Stage 3 で ~123GB，spawn worker で OOM kill」と明記)．**production caller ゼロ**，テストのみ．また `train_test_split`(`:100-106`) が `list(range(N))` を作り 5000万行で索引だけ約2.6GB が同時生存 (`np.random.Generator.permutation` なら C ループ1回，ただし seed 固定時の分割値が変わる)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D10+D11 | `src/maou/infra/file_system` | **D10(a) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．記録が言う「削除」は誤りで，`iter_files_columnar` は `app/learning/streaming_dataset.py:199` の `StreamingSource` プロトコルが宣言するメンバかつテスト 10 本以上から呼ばれるので，削除すると公開名が消える (P6)．`_subset` への委譲に留めた．**残り**: (1) `total_pages()`(`:922`) はファイル数を返すが `cache_mode="memory"` の `iter_batches` は 1個しか yield しない (`:704-716`) ため `hcpe_transform.py:677,683` の `tqdm` が 1/N で止まる — 2026-08-12 に **P4** と判定 (公開メソッドの戻り値が変わる)．caller は `hcpe_transform.py:679` の 1 箇所だけなので影響は限定的だが，「ファイル数」と「yield 数」のどちらを意味させるかの決めが要る．(2) 行数スキャン (`_ensure_row_counts`) は逐次かつファイルごとに `open()` 2回で，ネットワーク越し500ファイルでは起動レイテンシの支配項 (D14(a) と同時に直すのが自然)． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D12 | `src/maou/infra/file_system` | **(a)(b)(e)(f) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．(a) の desync ハザードは変換器をループ前に非 None へ確定させて分岐ごと消した．(b) の入れ子は兄弟ハンドラへ移した (伝播する例外の型と文言は据え置き)．(e) の恒真条件は縮約し，`test_init_hcpe_rejected` のメッセージ検査も通ることを確認した．(f) の `log_level` 変数は削除した．**残り**: (c) `file_data_source.py` の `cumulative_rows` は `sum(lengths)` の二重計算，`milestone_interval` は7行3分岐で `logger.info` を間引くだけ (行番号は PR #456 でずれているので再取得が要る)．(d) `total_pages`/`total_rows`/`file_paths` はいずれも `_file_entries` から導出可能な重複状態 — (a) の desync ハザードは消えたので緊急性は下がったが，重複状態そのものは残る．property 化は公開属性の意味を変えないので **P3** 相当だが，`FileManager` の属性を読む path 外の caller を洗う必要がある． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D14 | `src/maou/infra/file_system` | **(a) 行数スキャンの共有化．** `streaming_file_source._ensure_row_counts` と `streaming_hcpe_source._ensure_row_counts` の例外安全性の差は `8c1417e` で解消したが，**両者は依然として別実装**であり，安全なのは構造が違う結果の偶然にすぎない．`_scan_row_counts(paths) -> list[int]` を共有し，`StreamingHcpeDataSource` はそのリストを保持して `_total_rows` を導出すべき (現在は per-file カウントを捨てているので sharding 用の `row_counts` を提供できない)．**(b) `FileDataSource` が2つの ABC を着ている．** `preprocess.DataSource` 側の役割は `StreamingHcpeDataSource` に移った (`console/pre_process.py:489-492`) のに継承が残り，`hcpe` を `FileManager` の columnar 機構に通すための `_use_columnar` 分岐 (`:376`, `:605`, `:729`) を生かし続けている．外す場合は `infra/utility/benchmark_polars_io.py:419-451` の対応が要るので path 外の編集を伴う． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D15 | `src/maou/infra/file_system` | **`.feather` で終わる中途書き込みファイルを誰も守っていない．** step 2.5 key 6 の sweep で，`.gstmp`/`.tmp`/`.partial`/`.crc` については露出サイトがゼロと確認できた一方，**中断した `save_*_df` や in-place な `rsync`/`gsutil cp` が残す「拡張子が `.feather` のまま不完全なファイル」**はどのフィルタも通過する．`path_utils.py:10-13` が記述する `OSError: failed to fill whole buffer` がまさにこれ．`_is_temp_artifact` は末尾拡張子リストで判定するので**原理的に捕捉できず**，size/footer 検査が要る．**この行の存在理由**: 元は record の § Cross-module sweep (worklist ではない節) にしか書いておらず，backlog 行がないため `/audit-backlog` から永久に不可視だった (2026-08-10 のユーザー指摘で発見)．運用上のリスクとして実在するかは要判断． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O5 | `src/maou/infra/console` + `src/maou/infra/object_storage` | cache/ノブ系の意味の分裂．(a) `--input-local-cache` は BigQuery にしか渡されず S3/GCS 分岐 (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None` で判定 → **`maou pre-process --input-s3 --input-local-cache` は無言の no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算 (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅 (`object_storage/data_source.py:122,260-265`) と別物．(c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ (`object_storage/data_source.py:212-213` の docstring が明言) で既定値も層をまたいで不一致 (`:45` `True` / `:115`,`:407` `False`)．(d) `learn-model` には `--input-cache-mode` が存在せず `"file"` 決め打ち (`learn_model.py:796,820,847`)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O7 | `rust/maou_io` + `rust/maou_index` | Arrow IPC 形式判定の fallback 方向が Python と Rust で逆．Python (`streaming_file_source.py:230-247`, `domain/data/dataframe_io.py:19,35`) は「File か?」を問い既定 Stream，Rust (`rust/maou_io/src/arrow_io.rs:71-92`) は「Stream か?」を問い既定 File → Arrow 0.15 以前の Stream ファイルは `scan_row_count` が成功し `load_feather` が footer エラーで失敗する．さらに `rust/maou_index/src/index.rs:205-217` は**判定なし**で常に File 前提なので，Stream 形式 `.feather` は他の全経路で読めるのに visualize の索引構築だけ失敗する．**Rust 側を触るので該当 crate の `Cargo.toml` バージョン bump が要る．** |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O8 | `src/maou/interface` | `preprocess.py:181` の `len(pl.scan_ipc(fp).collect())` が**全列を実体化**する (`scan_row_count` は `select(pl.len())`)．1496幅のリスト列を持つ preprocessing ではメタデータ読みと全ロードの差．さらに `:186` の裸の `except Exception:` が Stream 形式ファイルの失敗を飲み込み，`ok_files` 扱いで `chunk_input_files` から黙って漏れる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:222-243` — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(`:405-420`) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．同じ行集合とは限らず件数も一致しないため `indicies` の範囲 (`:652-655`) と `get_page` が返せる実体がずれる．ファイル系ソースは常に厳密．`:235` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O10 | `src/maou/domain/data` | `dataframe_io.py:19,35` が Arrow マジック定数と File/Stream 判定を再定義．`streaming_file_source.py:230,233` と同値・同幅・同 fallback で **IDENTICAL**，入力型が `bytes` と `Path` で違うだけ．低優先の重複だが，O7 で fallback 方針を触るなら同時に片付ける対象． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N1 | `src/maou/infra/console` + `src/maou/interface` | **Stage 3 のファイル分割の既定が層で食い違う．** console 側 `learn_model.py:884` は `test_ratio` 未指定時に **0.1**，interface 側 `learn.py:307` は **0.2** を使う．streaming 経路では分割済みソースを console が作るので interface 側の値は使われず潜在的だが，`--no-streaming` に切り替えた瞬間に検証割合が 10% から 20% へ黙って変わる．どちらに寄せるかは検証設計の判断 (既定を変えると全ユーザの検証割合が動く)． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N2 | `src/maou/infra/bigquery` | **キャッシュ完了検証が，検証の要る場合だけスキップされる．** `bq_data_source.__download_all_to_local` は冒頭で全ページのキャッシュ存在を確認し，揃っていれば早期 return する (`:456-462`)．そのため末尾の「何件作られたか」の検証 (`:481-491`) は，**一部だけ欠けている / 壊れている**という最も検証が要るケースを通らない．O6 (glob の拡張子違い) の回帰テストを書く過程で発見． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N3 | `src/maou/app/learning` + `src/maou/app/pre_process` | **`@abc.abstractmethod` が不活性．** `dataset.py:45` と `hcpe_transform.py:62` は `@abc.abstractmethod` を付けながら基底に `abc.ABCMeta`/`abc.ABC` を使っていないため，未実装・型違いの実装が**構築時に一切捕まらない**．O1 (`BigQueryDataSource.__getitem__` が `pl.DataFrame` を返していた) が実行時まで露見しなかった根本原因で，O1 を直しても残る．`ABC` を継承させると現存の非準拠実装が全て構築時エラーになるため，何が壊れるかを洗う必要がある (だから O1 と一緒に直せなかった)． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N4 | `tests/maou/infra/file_system` | **torch 無しの環境で `infra/file_system` のテストが丸ごと消える．** `test_file_data_source.py` は `file_data_source.py` → `interface/learn` → torch の連鎖で，torch 未導入だと**モジュールごと** skip される (`uv sync` のみの環境では 52 passed + 3 skipped と表示され，緑に見える)．`uv sync --extra cpu` を入れると 83 passed になる．CI/開発環境が base extra だけだと，このパッケージの変更が無検証で通る．optional dependency に依存しない薄いテストへ切り出すか，最低限 CPU extra を必須にするかの判断が要る． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N6 | `.pre-commit-config.yaml` | **`cargo fmt` が pre-commit で強制されていない．** Python 側は `ruff-format` / `ruff-check` の local hook があるのに Rust 側は整形も lint も hook が無く，`cargo fmt --check --all` が汚れたまま蓄積する (2026-08-12 時点で 8 件検出，PR #464 で解消済み)．**この行の存在理由**: 未整形を残すと「触るたびに検出し直して，その PR と無関係な整形差分を混ぜるか無視するかを毎回選ばされる」ため (2026-08-12 のユーザ指摘)．**やること**: `.pre-commit-config.yaml` に local hook として `cargo fmt --all -- --check` (または `--all` で自動整形) を追加する．`types_or: [rust]` で対象を絞れる．**判断が要る点**: (a) `--check` で落とすか自動整形するか — Python 側は `ruff-format` が**自動整形**なので揃えるなら後者．(b) `cargo clippy` も同時に入れるか — 現状 clippy も hook に無く，入れると既存の warning 数だけ初回コストが出る (本 run では `-p maou_shogi -p maou_usi --all-targets` で warning 0 を確認済みだが workspace 全体は未確認)．(c) hook は `language: system` + `pass_filenames: false` にする必要がある (`cargo fmt` はファイル引数ではなく crate 単位で動くため)．`.pre-commit-config.yaml` の ruff local hook のコメントが「hook 側が独自にツールを入れると版が二重化する」と警告しているので，rustfmt も同じ理由で local hook (toolchain 共有) にすべき． |
