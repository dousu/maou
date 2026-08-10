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
| `src/maou/infra/file_system` | python | done | high | `1c6a442` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | 7 deferred |

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
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D10+D11 | `src/maou/infra/file_system` | `iter_files_columnar`(`streaming_file_source.py:161`) は `_subset` から計測ログを抜いた二重実装で production は後者しか呼ばない．`total_pages()`(`:922`) はファイル数を返すが `cache_mode="memory"` の `iter_batches` は 1個しか yield しない (`:704-716`) ため `hcpe_transform.py:677,683` の `tqdm` が 1/N で止まる．行数スキャン (`_ensure_row_counts`) は逐次かつファイルごとに `open()` 2回で，ネットワーク越し500ファイルでは起動レイテンシの支配項． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O1 | `src/maou/infra/bigquery` + `src/maou/app/learning` | **`BigQueryDataSource` が learn 側の契約を破り実行時に落ちる．** `bq_data_source.py:659` の `__getitem__` は `pl.DataFrame` の1行を返すが `dataset.py:46-52` は `np.ndarray` を要求 → `KifDataset.__getitem__`(`dataset.py:87`) の `data.dtype.names` が **`AttributeError` でバッチ0から落ちる**．再現: `maou utility benchmark-dataloader --input-dataset-id … --input-table-name …` (`utility.py:318`→`utility_interface.py:103`→`dataloader_benchmark.py:93`)．`benchmark-training` も同配線 (`utility.py:1298`)．`iter_batches_df` も未 override で継承既定 (`hcpe_transform.py:86-140`) が同様に落ちる．構築時に捕まらないのは `dataset.py:45` と `hcpe_transform.py:62` が `@abc.abstractmethod` を付けながら `abc.ABCMeta`/`abc.ABC` を使っておらずマーカーが**不活性**だから． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O2 | `src/maou/infra/object_storage` + `src/maou/infra/bigquery` | `__train_test_split` の複製2件 (`object_storage/data_source.py:86`, `bq_data_source.py:81`) が互いに文字単位で同一のまま `random.seed(seed)` でグローバル RNG を汚染する．`8c1417e` で `infra/file_system` 側だけ `random.Random(seed)` に直したので **修正が既に乖離している**．現状 seed を渡す呼び出し側がないので休眠中． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O3 | `src/maou/domain/data` | `columnar_batch.py:91,101,111,121,131` — optional 列の有無を `batches[0]` だけで判定しつつ内包表記は全要素を走るため，不一致時に列が**短いまま結合**され行対応が崩れる (例外なし) か**黙って落ちる**．`moveWinRate` を持つ/持たない preprocessing ファイルが混在すると `file_data_source.py:527` 経由で到達 (`--input-cache-mode memory`)．現状は D1 により誰も読まないので休眠だが，**消費者が1人増えた瞬間に発火する**．正しい形は `all(b.<field> is not None for b in batches)` か不一致時の明示 raise． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O4 | `src/maou/infra/console` | `learn_model.py:876-892` の inline file split が `test_ratio or 0.1` なので **`--test-ratio 0.0` が黙って 0.1 になる** (検証分割なしを要求しても10%取られる)．seed 42 決め打ちで `file_level_split(seed=None)` の非再現方針と食い違う．同じ算術が `utility.py:1211-1229`, `:1256-1272` にもあり計4箇所． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O5 | `src/maou/infra/console` + `src/maou/infra/object_storage` | cache/ノブ系の意味の分裂．(a) `--input-local-cache` は BigQuery にしか渡されず S3/GCS 分岐 (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None` で判定 → **`maou pre-process --input-s3 --input-local-cache` は無言の no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算 (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅 (`object_storage/data_source.py:122,260-265`) と別物．(c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ (`object_storage/data_source.py:212-213` の docstring が明言) で既定値も層をまたいで不一致 (`:45` `True` / `:115`,`:407` `False`)．(d) `learn-model` には `--input-cache-mode` が存在せず `"file"` 決め打ち (`learn_model.py:796,820,847`)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O6 | `src/maou/infra/bigquery` | `bq_data_source.py:483-493` が `glob("*.npy")` で検証するが `__save_to_local`(`:285-293`) は `.feather` を書く → **毎回** "Created 0 local cache files" と "No local cache files were created. This might indicate a problem." が出る．ログのみだが，まさにその操作を報告する場面で誤導する． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O7 | `rust/maou_io` + `rust/maou_index` | Arrow IPC 形式判定の fallback 方向が Python と Rust で逆．Python (`streaming_file_source.py:230-247`, `domain/data/dataframe_io.py:19,35`) は「File か?」を問い既定 Stream，Rust (`rust/maou_io/src/arrow_io.rs:71-92`) は「Stream か?」を問い既定 File → Arrow 0.15 以前の Stream ファイルは `scan_row_count` が成功し `load_feather` が footer エラーで失敗する．さらに `rust/maou_index/src/index.rs:205-217` は**判定なし**で常に File 前提なので，Stream 形式 `.feather` は他の全経路で読めるのに visualize の索引構築だけ失敗する．**Rust 側を触るので該当 crate の `Cargo.toml` バージョン bump が要る．** |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O8 | `src/maou/interface` | `preprocess.py:181` の `len(pl.scan_ipc(fp).collect())` が**全列を実体化**する (`scan_row_count` は `select(pl.len())`)．1496幅のリスト列を持つ preprocessing ではメタデータ読みと全ロードの差．さらに `:186` の裸の `except Exception:` が Stream 形式ファイルの失敗を飲み込み，`ok_files` 扱いで `chunk_input_files` から黙って漏れる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:222-243` — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(`:405-420`) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．同じ行集合とは限らず件数も一致しないため `indicies` の範囲 (`:652-655`) と `get_page` が返せる実体がずれる．ファイル系ソースは常に厳密．`:235` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O11 | `docs/commands/pre_process.md` (fix と一緒に `src/maou/app/pre_process` 監査で) | **`pre_process.md` が pre-process の出力を `.npy` シャードと説明している**が，実際は `transformed_chunk{NNNN}.feather` (Arrow IPC / `hcpe_transform.py:263,574`)．該当箇所 `:6`, `:22`, `:32`, `:34`, `:66`, `:90`．`docs/adr-004-arrow-ipc-migration.md` の移行から漏れた記述．**本 run では直していない**: 2026-08-10 の承認済み提案 (`1c6a442`) は `infra/file_system` を説明する主張だけを対象にしており，これは pre-process 自身の出力形式についての別のドリフト．durable doc なので `reviews/` 提案 + 承認が必要． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O10 | `src/maou/domain/data` | `dataframe_io.py:19,35` が Arrow マジック定数と File/Stream 判定を再定義．`streaming_file_source.py:230,233` と同値・同幅・同 fallback で **IDENTICAL**，入力型が `bytes` と `Path` で違うだけ．低優先の重複だが，O7 で fallback 方針を触るなら同時に片付ける対象． |
