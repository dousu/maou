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
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 5 deferred |
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
  および
  [2026-08-13 backlog truncated-feather-diagnostics](2026-08-13-backlog-truncated-feather-diagnostics.md)
  ([PR #493](https://github.com/dousu/maou/pull/493) — 2 つの run が
  同じ指定ブランチに積まれたため 1 本の PR にまとまっている)．
  **N6-1 / N7 / D15 の 3 行を削除．** ユーザが判断帯の 2 点を承認して
  マージした — P4 (BigQuery `iter_batches`) は「元々壊れていたので修正で
  問題ない」，P3+G1 (行数スキャンの並列化) は「測れていなくて問題ない」．
  O5 / D5 / D10+D11 は**行の一部だけ**の消化なので行は残り，それぞれ
  (c) の doc drift / 見積り部分 / (2) を消化済みと明記して縮めてある．
  D15 は "changed shape" で，記録の「size/footer 検査が要る」が誤り
  だったので [元記録](2026-08-10-src-maou-infra-file-system.md) に訂正を
  追記した．N6-2 行の前 run の記述も訂正済み．
- [2026-08-13 backlog callback-accumulator-table](2026-08-13-backlog-callback-accumulator-table.md)
  ([PR #494](https://github.com/dousu/maou/pull/494) — **Deferred 4 の 1 行を削除**)．
  14 行を再検証して stale 0 / changed shape 2 / confirmed 12．自動帯に
  入ったのは Deferred 4 だけで，残る 13 行はすべてゲート付きか判断帯
  だったため，質問を上げずに文言だけ鋭くして残した．Deferred 3 と O9 は
  "changed shape" — Deferred 3 は[元記録](2026-08-08-src-maou-app-learning.md)
  に訂正を追記した．新規所見 N8 を起票．
- [2026-08-13 backlog bundling-knobs-and-loss-aliasing](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)
  ([PR #495](https://github.com/dousu/maou/pull/495) — **N8 の 1 行を削除**)．
  指定ブランチ 1 本の制約でクラス毎の PR 分割ができず，自動帯と判断帯が
  同じ PR に同居したため，レビュー単位は commit が担った．ユーザが内容を
  確認して承認・マージ (`95f31e3`)．14 行を再検証して stale 0 /
  changed shape 4 / confirmed 10．自動帯は P2 1 件 + P3 2 件，判断帯は
  N8 と新規所見 1 件．O5 は**行の一部だけ** ((b) と (c) の既定値) の消化な
  ので行は残り，残り (a)(d) + ノブ削除の記述に縮めてある．
  **元記録への訂正 3 件** — Deferred 2 (4 本目の挙動軸)，
  Deferred 3 (前 run の訂正 (i) 自体が誤り: `else` 腕は到達不能)，
  D14b/D10+D11 (path 外の障害とテスト不在の見立てが誤り)．新規所見 N9 を起票．
- [2026-08-14 backlog diagnostics-and-npy-remnants](2026-08-14-backlog-diagnostics-and-npy-remnants.md)
  (**N9 の 1 行を削除**)．指定ブランチ 1 本の制約でクラス毎の PR 分割が
  できず，自動帯 3 件と判断帯 1 件が同じ PR に同居したため，レビュー単位は
  commit が担った．14 行を再検証して stale 0 / changed shape 2 /
  confirmed 12．自動帯は P1 1 件 (テストの黙殺を可視化) + P2 1 件
  (新規に気づいた `.npy` 併存記述の drift) + P3 1 件 (pre-process の
  入力エラー文言)．N9 は「`.feather` 版へ書き直す」側が**修理ではなく
  書き下ろし**だと判明 (`Network` / `LossOptimizerFactory` /
  `KifDataset` の 3 API が全て変わっており，glob 以外に 3 つの
  破綻がある) したため削除の向きで出荷したが，「今は残して後で別物を
  書く」選択は利用者のものなので G4 は retire せず PR に判断を載せた．
  **ユーザが同一セッション内で「削除する」と回答**したため，そのまま
  マージした ([PR #498](https://github.com/dousu/maou/pull/498))．
  N4 と O5 は**行の一部だけ**の消化なので行は残り，それぞれ「黙って
  消える部分」「メッセージが原因を指していない部分」を消化済みと明記
  して縮めてある．元記録への訂正は**なし** (changed shape 2 件は
  いずれもコードが動いた結果で，診断の誤りではない)．

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:422`/`:571`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:646`/`:735`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279` (file untouched since the record, so those still hold) and `_yield_stage1_batches`/`_yield_stage2_batches` (`streaming_dataset.py:851`/`:911`). **~400-line refactor of the multi-stage training path — architecturally significant.** (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) **2026-08-13 の再検証で 2 点を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．(i) 「head/callback/metric getter とログ 2 本しか違わない」は不完全 — **4 本目の軸**として使う `TrainingLoop` サブクラスが違う (`Stage1TrainingLoop` `:451` vs `RawLogitsTrainingLoop` `:600`)．装飾ではなく挙動の軸なので，「差分は装飾だけ」という前提で統合を設計すると取り落とす．(ii) 工場の「38 行の byte-identical な末尾」は過大で，完全一致は **28 行** (`:705-732`/`:795-822`)．38 行の窓には `stage_name="Stage 1"`/`"Stage 2"` (`:702`/`:792`) が入る．現在の行: run 関数は `:422-568`/`:571-717`，`dataset.py` の対は `:242-316`/`:319-391` (記録の `:202`/`:279` は現在 `_numpy_to_tensor`)，`streaming_dataset.py` は `:848-905`/`:908-962`．統合の footprint は合計 ~600 行．**P4 + G3** (この環境で等価性を示せない) **+ G4**． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 3 | `src/maou/app/learning` | Six adapter classes are three duplicated pairs. `Stage1ModelAdapter`/`Stage2ModelAdapter` (`multi_stage_training.py:111`/`:240`) differ in **zero** characters; `Stage1DatasetAdapter`/`Stage2DatasetAdapter` (`:151`/`:183`) in one type annotation; `Stage1StreamingAdapter`/`Stage2StreamingAdapter` (`streaming_dataset.py:721`/`:686`) in a redundant `hasattr` guard. Merging also deletes the `isinstance` dispatch + `TypeError` arm at `stage_component_factory.py:876-882`, which exists only to choose between two identical classes. Six public names referenced from tests — should land as its own reviewed change. (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) **2026-08-13 の再検証で 2 点を訂正** ([記録](2026-08-13-backlog-callback-accumulator-table.md))．(i) `stage_component_factory.py:876-882` の `isinstance` 分岐は「同一の 2 クラスを選び分けるためだけ」ではない — `else` 腕が**未対応の head 型を `TypeError` で弾く検証**を担っており，統合時に落とすとこれまで拒否されていた head 型が黙って通る．(ii) 両名を同一クラスの別名にすると `test_stage_component_factory.py:297`/`:398` の `isinstance` アサーションが**どちらも通ってしまい，2 本のテストが識別力を失う** (Stage 1 のテストが Stage 2 のアダプタを構築しても緑になる)．したがって統合は**テストの書き換えとセット**でしか出荷できず，見かけの「0 文字差」より高くつく — **P4 + G2**． **2026-08-13 の再検証で上の (i) 自体を撤回** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．`_build_model` は private で呼び出し元が `:695` と `:785` の 2 つしか無く，**どちらも呼び出しの 7-10 行前に head を自分で構築している** (`ReachableSquaresHead(...)` `:688` / `LegalMovesHead(...)` `:777`)．第 3 の head 型が到達する経路は無く `else` 腕 (`:878-882`) は**到達不能**で，`"Unsupported head type"` を参照するテストもゼロ．よって「これまで拒否されていた head 型が黙って通る」は起こらない — 失われるのは何も守っていない防御コードだけ．**(ii) のテスト識別力の話は今も成立し，結論 (P4 + G2) は不変**．また「0 文字差」は文字通りには偽で，一致するのは**本体 26 行**のみ (クラス名と docstring 3 行が違う)．`Stage1StreamingAdapter`(`:721-757`)/`Stage2StreamingAdapter`(`:685-718`) はメソッド順序も違う． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1110` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per batch, to guard a warning. **Now dormant, not fixed** (2026-08-09, backlog `tier-3-contracts`): the record's premise "Stage 3 always ships a `legal_move_mask`, so the branch is always taken" no longer holds — no data path supplies a mask, so `_compute_policy_loss` never enters the masking arm and the sync does not execute. The code is unchanged and the stall returns the moment a real legal-move mask is wired in; fix it **then**, together with whatever produces the mask, and measure on GPU. **2026-08-13 に dormant の証明を強化** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): `TrainingContext` の構築箇所は `training_loop.py:516` の 1 つだけで，その 3 行上 `:507` に `legal_move_mask: torch.Tensor | None = None` と**ハードコードされている**．`src/` 全体に産出者はゼロ (`stage2_data_generation.py:247` の Rust `legal_move_masks()` はデータ生成用で `TrainingContext` に届かない)．`_unpack_batch` の docstring `:497-502` も「将来経路のために残している」と明言．**P4 + G1**． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 6 | `src/maou/app/learning` | `training_loop.py:460` `stream.synchronize()` blocks the host, defeating much of the prefetch it implements. `wait_stream()` gives the same ordering guarantee device-side without stalling the CPU, and the `record_stream()` added in `073adbd` already covers the allocator hazard. **2026-08-13 の再検証で "changed shape"** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 「2 つ目の未検証 GPU 意味論変更が 1 つ目に積み重なる」という見立ては**もう当たらない**．`record_stream()` は既に入っており (`self._record_stream(next_ctx, compute_stream)` `:454`，`compute_stream = torch.cuda.current_stream()` `:422`，ヘルパ `:750-780`)，073adbd の分は出荷済み．`record_stream()` (allocator の再利用ハザード) と `wait_stream()` (ホスト側ブロック) は別問題を解くので，**残っているのはホスト側ブロック 1 点だけ**で当初より狭い．**P4 + G1** (GPU が要る)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D5 | `src/maou/infra/file_system` | **見積りの数え落としは 2026-08-13 に消化・マージ済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md))．合計は `ColumnarBatch.nbytes` として `dataclasses.fields` から導出するようになり，警告は `_warn_if_oom_risk` に一本化，閾値は `OOM_WARNING_THRESHOLD_GB` に定数化された．**残り**: `cache_mode` の altitude 本体．両モードとも `__init__` で全ロードし差は結合の有無だけ，`total_pages<=1` なら完全同一．`_concatenate_numpy`/`_concatenate_columnar` は入力を保持したまま結合するので**ピーク2×**で，警告は全ロード後・倍化直前にしか出ない (見積りが正しくなっても手遅れであることは変わらない)．**見送り理由**: ノブ廃止は O5 と一体． **2026-08-13 に 4 主張とも再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 全ロードは `:202-285` で無条件，モード差は `:301-308` の `cache_mode == "memory" and total_pages > 1` だけ，2× ピークは `_concatenate_numpy` `:334-363` / `_concatenate_columnar` `:365-400` (入力リストを結合越しに保持し `:362-363`/`:399-400` で解放)，警告 `_warn_if_oom_risk` `:317-332` (閾値 `:46`) は各 concatenate の**先頭**で呼ばれる．**P6 + G4**．なお `scripts/benchmark_file_datasource.py` が `cache_mode="mmap"` を既定にしていて実行不能だった件は，この run で**スクリプトごと削除して解消**した (`.npy` 時代の遺物)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D10+D11 | `src/maou/infra/file_system` | **D10(a) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．記録が言う「削除」は誤りで，`iter_files_columnar` は `app/learning/streaming_dataset.py:199` の `StreamingSource` プロトコルが宣言するメンバかつテスト 10 本以上から呼ばれるので，削除すると公開名が消える (P6)．`_subset` への委譲に留めた．**残り**: (1) `FileDataSource.total_pages()`(**`:775`** — 記録の `:898` から移動．2026-08-13 に再確認) はファイル数を返すが `cache_mode="memory"` の `iter_batches` は 1個しか yield しない — **2026-08-12 の再検証で dormant と判明**．`hcpe_transform.py:679` の唯一の caller が受け取るのは `StreamingHcpeDataSource` (`console/pre_process.py:494` が構築) で，そちらの `total_pages()` は `len(self._file_paths)` を返し `iter_batches` も 1 ファイル 1 batch なので tqdm は正しい．`FileDataSource.total_pages()` の production caller は**ゼロ** (テストも無い)．食い違い自体は残っているので，「ファイル数」と「yield 数」のどちらを意味させるかの決めは，この経路に caller が戻るときに要る．**(2) 行数スキャンの逐次実行は 2026-08-13 に消化・マージ済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md))．`scan_row_counts` は File 形式のメタデータ読みだけを `ThreadPoolExecutor` に載せる 2 相構成になった (Stream 形式は全読みが要りピークメモリがワーカー数倍になるので逐次のまま)．**便益は未測定** (G1: 数百ファイルのネットワークストレージがこの環境に無い)．**残るのは (1) の意味の決めだけ**． **2026-08-13 の再検証で (1) の記述を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 「テストも無い」は**誤り**．`FileDataSource` を `PreProcess` に渡す形で間接的に実行しているテストが 3 ファイルある (`test_hcpe_transform.py:79`/`:92`/`:305`/`:331`，`test_app_hcpe_transform.py:185-190`/`:259-264`，`test_convert_and_preprocess.py:226-231`/`:328`/`:430`/`:534`)．ただしいずれも `cache_mode` を渡さず `"file"` モードで走るので，**食い違い自体は未実行**．つまり「テストが無い」ではなく「テストはあるが問題の条件を踏んでいない」— 直す際に要るのは `cache_mode="memory"` を踏むケースの追加である．production caller ゼロは今も正しい (唯一の caller `hcpe_transform.py:688` が受け取るのは `StreamingHcpeDataSource`，現在は `infra/file_system/streaming_hcpe_source.py:25` に移動)． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 3 箇所とも健在 (`:150` / `:188-192` / `:246-248`)．加えて `g.clone()` (`:151`/`:193`) が勾配の全複製をパラメータ毎に確保する．緩和策の `should_measure` (`:115-122`) はあるが **`measurement_interval` の既定は `1`** (`:85`) なので，caller が上書きしない限り毎 optimizer step で走る．値は `compute()` → `b_noise` → `training_loop.py:1024-1031` で Python float として消費され `gradient_accumulation_steps` を書き換えるため，controller 側を tensor 値に作り替えないと同期は遅延できない．**P4 + G1**． |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**2026-08-13 の再検証**: `FileDataSource.get_items` の呼び出し側は**ゼロ** (`FileManager.get_items` への内部委譲だけ) なので (b) 単独では **dormant**．実害は (a) の根本解決に踏み込んだときに初めて出るため，「(b) を先に直す」選択肢は実質無い．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． **2026-08-13 の再検証で数値を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 機構は健在だが記録の数字が古い．1 サンプルは **3,089B ではなく 9,081B** (`get_preprocessing_dtype` に `moveWinRate` (f32×1496 = 5,984B) が入ったため)，コピーは **6 フィールドではなく 8 書き込み** (`:586-587` の `id` ゼロ埋め含む)．`np.empty(n, dtype)` は `:584`，`torch.from_numpy` は `dataset.py:230` ではなく **`:239`** (「ゼロコピー」を謳うコメントは `:95`)．batch 1024 なら約 **9.3MB** の memcpy/バッチ．hcpe 経路は view を返すので影響なし — 割り当ては columnar 型に限る．(b) の caller ゼロも再確認 (`tests/` にもヒット無し)．**P4/P6 + G2 + G4**． **2026-08-14 の再検証で機構の所在を訂正** ([記録](2026-08-14-backlog-diagnostics-and-npy-remnants.md))．`__getitem__` 自身はもう確保をしていない — `:692-696` の薄い委譲 (`return self.__file_manager.get_item(...)`) になり，実体は `_columnar_batch_to_structured_array` (**`:553-634`**) へ移った (`get_item` → `_columnar_to_structured_record` `:471-494` 経由)．`np.empty(n, dtype=self._structured_dtype)` は `:584` で行番号は一致するが，**フィールド書き込みは `:586-632`** に広がっており，前の記述の `:586-587` は狭すぎる (8 書き込みという数は正しい)．`get_items` (外側 `:701-712`) が `FileManager.get_items` (`:496-509`) へ委譲し，そこが `[self.get_item(idx) for idx in indices]` で回しているのも不変．結論は不変． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D14 (残り) | `src/maou/infra/file_system` | **(a) 行数スキャンの共有化は 2026-08-13 に出荷済み** ([PR #492](https://github.com/dousu/maou/pull/492), [記録](2026-08-13-backlog-scan-share-and-abc.md))．`domain/data/arrow_format.scan_row_counts` を 2 実装が引くようになり，`StreamingHcpeDataSource` は per-file カウントを `row_counts` で公開する．**残り**: **(b) `FileDataSource` が2つの ABC を着ている．** `preprocess.DataSource` 側の役割は `StreamingHcpeDataSource` に移った (`console/pre_process.py:489-492`) のに継承が残り，`hcpe` を `FileManager` の columnar 機構に通すための `_use_columnar` 分岐 (`:376`, `:605`, `:729`) を生かし続けている．**2026-08-13 の再検証で「path 外の障害」の記述を撤回** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．`benchmark_polars_io.py:419-451` は `print_summary` の docstring と print 行で `FileDataSource` とは無関係．実使用は `:386-390` (構築) と `:394` (`iter_batches_df`) だが，**`iter_batches_df` は `FileDataSource` 自身の具象メソッド** (`file_data_source.py:726`) なので ABC を外しても壊れない．**真の障害はテスト群** — `test_hcpe_transform.py:79`/`:92`/`:305`/`:331`，`test_app_hcpe_transform.py:185-190`/`:259-264`，`test_convert_and_preprocess.py:226-231`/`:328`/`:430`/`:534` が `FileDataSource` を `PreProcess(datasource:)` に渡している．また `_use_columnar` の現在位置は `:165`,`:171`,`:182`,`:236`,`:241`,`:305`,`:456`,`:539` で，**preprocess の役割に属するのは `:539` と `:726` だけ**．`:241`/`:456` は `learn.LearningDataSource` 側を支えるので**外しても退役しない** — 「継承を外せば分岐がまとめて消える」という見積りは過大．**P6 + G2 + G4**． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O5 | `src/maou/infra/console` + `src/maou/infra/object_storage` | cache/ノブ系の意味の分裂．(a) `--input-local-cache` は BigQuery にしか渡されず S3/GCS 分岐 (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None` で判定 → **`maou pre-process --input-s3 --input-local-cache` は無言の no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算 (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅 (`object_storage/data_source.py:122,260-265`) と別物．(c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ (`object_storage/data_source.py:199` の docstring が明言し，`:288-320` は各 `.feather` を無条件に個別保存する) で既定値も層をまたいで不一致 (`:45` `True` / `:102`,`:394` `False`)．**(c) の doc drift 部分は 2026-08-13 に消化・マージ済み** (`/audit-backlog`, [記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md), [提案](../reviews/2026-08-13-bundling-knobs-are-no-ops.md)) — docs 3 本に「受理するが効果なし」と明記した．**ノブ自体の削除 (P6) と既定値の不一致は残る** ((a)(b)(d) と同じ「どちらがキャッシュを有効にするのか」の決めと一体)．(d) `learn-model` には `--input-cache-mode` が存在せず `"file"` 決め打ち (`learn_model.py:796,820,847`)． **2026-08-12 の再検証で (a) の記述を訂正**: 「無言の no-op」ではない．`--input-local-cache` (bool flag) と `--input-local-cache-dir` (str) は別のオプションで，S3/GCS の elif が見ているのは**後者**である (`pre_process.py:419`,`:451`)．dir を渡さずに `--input-s3` だけ指定すると elif を全て外れて最後の `else` に落ち，**「Please specify an input source (file path, BigQuery table, GCS bucket, or S3 bucket)」という誤誘導エラーで停止する** (`:497-501`)．黙って無視されるより気付きやすいが，メッセージが原因を指していない．したがって直す向きは「flag を S3/GCS へ渡す」ではなく，**bool flag と dir のどちらがキャッシュを有効にするのかを層をまたいで一致させる**ことになる． **(b) の doc drift と (c) の既定値不一致は 2026-08-13 に消化・マージ済み** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md), [PR #495](https://github.com/dousu/maou/pull/495))．(b): `object_storage/data_source.py` の `max_cached_bytes` docstring が「キャッシュの上限サイズ」と説明していたが，このクラスに退避機構は無く唯一の読み出し (`:250`) は `max_cached_bytes / max_workers` を並列DLの 1 チャンク上限にするだけ．docstring に「キャッシュ上限ではない / 退避予算は BigQuery 側の同名引数」と明記した．(c): `DataSourceSpliter.__init__` の `enable_bundling` 既定だけが `True` (`:45`) で他 2 層と CLI の `False` と食い違っていたのを `False` に揃えた．両ノブがリポジトリ全体で**一度も判定に使われていない**ことを AST で固定する characterization test を追加．**残る**: (a) bool flag と dir の一致，(d) `learn-model` の `--input-cache-mode` 不在，**およびノブ自体の削除 (P6 + G4)**． **2026-08-14 に (a) の「メッセージが原因を指していない」部分だけを消化・出荷** ([記録](2026-08-14-backlog-diagnostics-and-npy-remnants.md))．`describe_missing_input_options()` を追加し，どの入力ソースが指定されていてどの companion オプションが欠けているかを名指しするようにした (何も指定しない呼び出しは従来の文言のまま)．**分岐条件も送出する例外も変えていない**ので (a) の**決め** — bool flag と dir のどちらがキャッシュを有効にするのか — は手つかずで残る． **同日の再検証で (d) の記述を訂正**: `learn_model.py:847` はもう文字列リテラル `"file"` ではなく `cache_mode=_c` で，`_c=_s3_cache` は lambda の既定引数として束縛され `_s3_cache = "file"` は `:838` にある．`:796`/`:820` はリテラルのまま．**効果は同じ** (上書き手段が無い) なので (d) の結論は不変． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:212-234` (記録の `:222-243` から移動) — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(**`:395-411`**) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．ファイル系ソースは常に厳密．`:226` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． **2026-08-13 の再検証で記述を強めた** ([記録](2026-08-13-backlog-callback-accumulator-table.md)): 非決定性は**二重**である．(i) `TABLESAMPLE SYSTEM` はクエリごとに独立評価なので数えた行集合と返す行集合が別物，(ii) その上で `LIMIT/OFFSET` を **`ORDER BY` 無しの再サンプル結果**に掛けているので，**同じ `page_num` を 2 回引いても同じ行が返る保証が無い**．「件数がずれる」ではなく**再現性が無い**． **修正の向きが 3 つに割れている** (G4 相当): (a) サンプルを一時テーブルへ 1 度実体化してページングする，(b) `FARM_FINGERPRINT` 等の決定的ハッシュ条件へ置き換える (キー列の決めが要る)，(c) `sample_ratio` とページングの併用を拒否する．**G1**: BigQuery がこの環境に無く，実際の非決定性は fake client では再現できない． **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 行が移動して `__get_total_rows` は `:213-234` (サンプル分岐 `:215-223`)，`total_pages` の導出は `:202-204`，ページ側の再サンプルは `__fetch_from_bigquery` (def `:345`) の `:396-405` (`start_index = page_num * batch_size` `:398`)．クラスタ/パーティション経路も `tablesample_clause` `:380-383` を `:385-389` に差し込むので同じ欠陥を弱い形で持つ．非サンプル経路は `list_rows(start_index=…)` (`:415-419`) なので無傷．**緩和材料**: `get_page` (`:486-520`) がページをキャッシュするため 1 run 内では通常 1 回しか引かれず，非再現性は退避時と run 跨ぎで表面化する． |
| [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md) N4 | `tests/maou/infra/file_system` | **torch 無しの環境で `infra/file_system` のテストが丸ごと消える．** `test_file_data_source.py` は `file_data_source.py` → `interface/learn` → torch の 連鎖で，torch 未導入だと**モジュールごと** skip される (`SKIPPED [1] ...: optional dependency 'torch' is not installed`)．**2026-08-12 に再測**: base 環境で **57 passed + 3 skipped**，`uv sync --extra cpu` で **90 passed** (記録当時の 52/83 から増えている)．**実害を確認済み**: 同日の `/audit-backlog` run はこのパッケージを変更したため，CPU extra を入れてからでないと QA が空振りする状態だった (入れずに回すと変更が 無検証のまま緑に見える)．CI/開発環境が base extra だけだと同じことが起きる．**判断が要る点は未解決**: optional dependency に依存しない薄いテストへ切り出すか，最低限 CPU extra を必須にするか．**2026-08-13 に 4 run 連続で実害を確認** — 同日の 2 本目の run ([記録](2026-08-13-backlog-oom-estimate-and-bq-contract.md)) はコンテナが再作成されて **venv が空**から始まり，`uv sync --extra cpu` に約 7 分・`maturin develop --release` に 12 分を要した．この run も 4 件中 2 件が `infra/file_system` に触れているので，入れずに QA を回していたら変更が無検証のまま緑に見えていた． **2026-08-13 の 5 本目でも再確認** ([記録](2026-08-13-backlog-callback-accumulator-table.md))．**影響は `infra/file_system` に閉じない**: この run が触れたのは `app/learning/callbacks.py` で，base 環境では `import torch` が失敗して `tests/maou/app/learning/` が**丸ごと skip される**．CPU extra を入れて初めて 1178 件が走った．つまり「torch 依存のテストが黙って消える」のはパッケージ固有の話ではなく，**torch を import する全テストに共通**する． **2026-08-13 の 6 本目でも再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): コンテナが再作成され **venv は site-packages 2 エントリの空**から始まった．`uv sync --extra cpu` に約 7 分，`maturin develop --release` に **7 分 4 秒**を要し，入れ終えて初めて全 1956 件が走った (入れる前は `pytest` すら不在)．連鎖の実装も再確認 — `tests/conftest.py:19-21` の `_OPTIONAL_DEPS` と `pytest_make_collect_report` (`:86-107`) が collect 失敗を **collector 粒度**で skip に書き換えるため，モジュールごと消える．**判断が要る点は未解決** (薄いテストへの切り出し vs CPU extra 必須化) — **P4 + G4**． **2026-08-14 に「黙って」の部分だけを消化・出荷** ([記録](2026-08-14-backlog-diagnostics-and-npy-remnants.md))．`tests/conftest.py` に `pytest_terminal_summary` を足し，collect 段で丸ごと落ちたモジュールを依存名ごとに件数とパス付きで列挙するようにした．全依存が入っている環境では accumulator が空なので何も出ない．**残るのは決めそのもの** — 薄いテストへの切り出しか CPU extra 必須化か (**P4 + G4** 継続)．なお同日 7 本目の run でもコンテナは再作成され，venv は site-packages 2 エントリの空から始まった (`uv sync --extra cpu` が要った)． |
| [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md) N6-2 | `src/maou/app/pre_process` | **基底の具象 `iter_batches_df` が HCPE スキーマ決め打ち．** `hcpe_transform.py:86-139` の既定実装は `get_hcpe_polars_schema()` を直に引くので，`preprocessing` 型のソースに対しては黙って誤動作する．現状 production の caller はすべて override 側 (`FileDataSource` / `ObjectStorageDataSource`) を通るので **dormant**．「HCPE 専用と明記して名前を変える」か「array_type で分岐させる」かの判断が要る．**2026-08-13 の再検証で dormant の度合いが増した**: N6-1 の修正で `BigQueryDataSource` も `iter_batches_df` を override したため，基底の既定実装を**呼ぶ** production 経路はゼロになった．**ただし同日 3 本目の run で上の記述を訂正**: ゼロなのは*呼び出し*であって*継承*ではない．`StreamingHcpeDataSource` (`console/pre_process.py:494` が構築する production の pre-process ソース) は `preprocess.DataSource` を継承しつつ `iter_batches_df` を override していないので，基底の既定実装を**着ている**．しかもそれは hcpe 専用クラスなので，**HCPE 決め打ちの既定実装はそこでは正しい**．この訂正は判断に効く — 「HCPE 専用と明記する」方向の根拠が強まり，「array_type で分岐させる」方向は分岐を要する継承者が居ない分だけ弱まる．ただし前者は行の文言上**改名 (P6) を含む**ので 2 案はまだ 1 案に潰れていない (G4 継続)． **2026-08-13 に再確認，行範囲を更新** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 既定実装は現在 **`:95-148`** (記録の `:86-139` から移動)，`get_hcpe_polars_schema()` を引くのは `:114-118`，`pl.DataFrame(data, schema=schema)` は `:147`．override 状況も再確認 — `BigQueryDataSource` (`:740`，docstring `:743-756` が本行を明示的に参照)，`ObjectStorageDataSource` (`:475`)，`FileDataSource` (`:726`) は override 済み，`StreamingHcpeDataSource` (`streaming_hcpe_source.py:25`) のみ override せず既定を着ている (hcpe 専用クラスなので正しい)．この配置は `tests/maou/app/pre_process/test_datasource_abc.py:75-100` が固定している (`iter_batches_df` を敢えて非 abstract に保ち `StreamingHcpeDataSource` を構築可能にする)． |
