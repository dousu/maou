---
path: src/maou/infra/file_system
scope: python
level: high
status: done
started: 2026-08-10
last_sha: 1c6a442
---

# Audit — src/maou/infra/file_system

## Resume point

_(完了．path 全体をカバーした．)_

step 0-10 をすべて実行．対象5ファイル (`file_data_source.py` 917,
`streaming_file_source.py` 272, `streaming_hcpe_source.py` 136,
`path_utils.py` 72→90, `file_system.py` 31) を全件レビュー済み．
scope class は python 単独 (非 `.py` ファイルなし)．

**このrunは step 2.5 (cross-module consistency sweep) の初回実行**でも
あり，2.5 の挙動そのものが検証対象だった．結果は § Cross-module sweep．

未適用の finding が多数残っているが，これは path が未カバーだからでは
なく，**確認済みの上で意図的に適用を見送った**もの (§ Deferred)．
再開ではなく `/audit-backlog` で個別に消化するのが正しい経路．

## Cross-module sweep

step 2.5a で `<path>` のコードから導出した sweep key は7つ．
`Explore` で `src/` + `rust/` 全体を掃いた結果．

| # | Sweep key | 判定 |
|---|---|---|
| 1 | `array_type` リテラル / schema 選択 | **CONSISTENT** (下記) |
| 2 | Arrow IPC 形式判定・行数カウント | **DIVERGENT** → O7 |
| 3 | `total_rows` / 行数メモ化 | **CONSISTENT** (invalidation 方針) + 1件 DIVERGENT → O9 |
| 4 | `train_test_split` | **DIVERGENT** → O2, O4 |
| 5 | cache-mode / materialization ノブ | **DIVERGENT** → O5 |
| 6 | 一時ファイル除外 (`_TEMP_ARTIFACT_SUFFIXES`) | **CONSISTENT** (下記) |
| 7 | DataSource ABC 準拠 | **DIVERGENT** → O1 |

### clean だった key (再調査を防ぐための記録)

- **key 1 — `array_type` のメンバ集合は完全に一致している．**
  Python 15箇所 (`file_data_source.py:83,206,231,791`,
  `streaming_file_source.py:65`, `domain/data/dataframe_io.py:175,208`,
  `interface/data_io.py:98,128`, `interface/data_schema.py:61`,
  `app/common/data_io_service.py:45,115,184,229`,
  `app/common/data_schema_service.py:42`,
  `app/learning/polars_datasource.py:65` ほか手書きリスト6箇所) と
  Rust `rust/maou_index/src/index.rs:27-32` の enum が
  すべて `{hcpe, preprocessing, stage1, stage2}` で一致．
  共有ヘルパーは存在せず **15回 re-type されている**が，
  ズレてはいない．`docs/architecture.md:158-160` の
  「`file_data_source.py` の Literal が正準」は **正確**．
- **`visualize` の 5メンバ `click.Choice` は DELIBERATE (バグではない)．**
  当初これを divergence 候補と見ていたが，`game-graph` は
  4段の独立したガードすべてで捕捉され，4メンバ Literal に到達不能:
  (1) `visualize.py:108` で分岐し `:158` で `return`，
  (2) `gradio_server.py:3854` で分岐し `:3870` で `return`，
  (3) `gradio_server.py:398` で `search_index`/`viz_interface` を
  `None` に固定，(4) `gradio_server.py:1049,1105-1129` が
  `_build_index_background` 前に `return`．
  5メンバ側は「どの可視化 UI か」，4メンバ側は「どのレコード schema か」
  という **別の問いに答えている**．フラグ名を共有しているだけ．
  → **今後 divergence として再提起しないこと．**
- **key 6 — 一時ファイル除外は実質的に非問題．**
  `_is_temp_artifact` が守るのは `ext is None` 分岐のみ
  (`ext` 指定時は `f.suffix == ext` が先に弾く)．独自 glob の
  `search_value.py:166` / `search_value_interface.py:84` /
  `stage2_data_generation.py:90` /
  `rust/maou_index/src/path_scanner.rs:104-140` は
  いずれも終端パターン照合なので **同じ artifact を構造的に除外して
  いる**．露出サイトはゼロ．
  ただし **誰も守っていないクラス** が1つある: `.feather` で終わる
  中途書き込みファイル (中断した `save_*_df`，in-place な `rsync`)．
  suffix リストでは原理的に捕捉できず，size/footer 検査が要る．

## Applied

`8c1417e` (step 1 — 4件, 回帰テスト各1):

1. `streaming_file_source.py:126` — `_ensure_row_counts` がループ**前**に
   `self._row_counts = []` を代入しつつ `is not None` で memo 判定して
   いたため，スキャン途中の例外で打ち切られたカウントが恒久的に残り
   `total_rows` が過少申告 → `steps_per_epoch` が狂う．ローカルに積んで
   完了後に代入 (`streaming_hcpe_source` 側の安全な形に合わせた)．
2. `path_utils.py:53,65` — `ext in p.suffixes` を `p.suffix == ext` へ．
   従来は `train.feather.bak` が収集を通過し，後段
   `FileManager.__init__:316` の `file_path.suffix != ".feather"` 検査で
   実行時 `ValueError` になっていた．収集側と読込側の規則を一致させた．
3. `path_utils.py:61` — ディレクトリ走査結果を `sorted()` に．呼び出し側
   (`learn_model.py`, `utility.py`, `file_level_split`) は固定 seed で
   shuffle して train/val を分けるが，`glob` 順は scandir 依存なので
   「seed 固定なら分割が再現する」が成立していなかった．
   **注意: 既存の seeded split の結果が一度だけ変わる．**
4. `file_data_source.py:195` — `random.seed()` によるモジュールグローバル
   RNG 汚染を `random.Random(seed)` へ．同じ Mersenne Twister を同じ
   seed で初期化するので **分割結果は不変**．

`64afa41` (step 3-4 — docstring のみ，挙動不変):

5. japanese-doc-validator: `、`×2, `（）`×4 を `，` / `()` へ．
6. `file_data_source.py:809` — `bit_pack` を「ビットパッキングを使用するか」
   と説明していたが実際は未使用．`:246` の正しい記述に合わせ，
   既定値が `True`(:799) / `False`(:86) と食い違っている事実も明記．
7. 同 `cache_mode` — 常駐量の軸だと読める記述を，両モードとも全ロード
   する事実に合わせた．

## Deferred

`<path>` 内で確認したが適用しなかったもの．

**Correction** (2026-08-10, `7ec3933` 時点): 本節の冒頭は当初
「**すべて backlog に行あり**」と書いていたが，**これは誤りだった**．
step 2 (`/simplify`) が返した約17件を D1-D11 に**統合**する過程で，
以下が行を持たないまま落ちていた — `/audit-backlog` は
`coverage.md` の2表**だけ**を読むので，これらは永久に不可視だった:

- 到達不能な numpy converter 群と `else` 欠落による desync ハザード
- 入れ子 `try/except` の再ラップ，進捗記録の二重計算，派生状態の重複
- `streaming_file_source.py:81-93` の恒真な第2検証，`log_level` の間接
- `__getitem__` のホットパスコスト，`get_items` の非バッチ化，
  `_STRUCTURED_DTYPES` と `data_schema.get_dtype` の重複
- `_scan_row_counts` の共有化，`preprocess.DataSource` ABC の除去

さらに，**§ Cross-module sweep に書いた「`.feather` で終わる中途書き込み
ファイルを誰も守っていない」も行がなかった** — あの節は clean な key を
記録する場所であって worklist ではないため，同じ理由で不可視だった．

いずれも `coverage.md` の Deferred backlog に **D12-D15** として追加済み．
本節の D1-D11 の記述自体は正しく，取り消しではない．

**教訓 (統合は行を消す)**: finding を1行にまとめると読みやすくなるが，
まとめきれなかった分が黙って消える．行数を惜しむべきではない．

- **D1. `moveWinRate` が structured record に載らない (最重要).**
  `domain/data/schema.py:136` の `get_preprocessing_dtype()` に
  `moveWinRate` フィールドがなく，`_columnar_to_structured_record`
  (`file_data_source.py:612`) も `_columnar_batch_to_structured_array`
  (`:734`) も `batch.move_win_rate` を読まない．結果:
  `context.move_win_rate` が `None` → `policy_targets.py:57` が
  `ValueError` を送出．**CLI 既定の `--policy-target-mode win_rate` で
  `learn-model --no-streaming` が初回ステップで落ちる**
  (streaming 経路は `streaming_dataset.py:832-843` が渡すので通る)．
  さらに efficiency 面: `move_win_rate` は `(N,1496) float32` で
  preprocessing の常駐 RAM の **約66%** (行あたり 9,073B 中 5,984B) を
  占めながら，このモジュールでは一度も読まれない．
  **見送り理由**: 根本原因が `<path>` 外の `domain/data/schema.py` にあり，
  `get_preprocessing_dtype` は domain/app/interface/infra の4層で6箇所以上
  から使われる．「dtype に足す」(streaming と parity) と
  「変換直後に捨てる」(RAM 66%削減) は **別方向の修正**で，どちらを取るかは
  設計判断．
- **D2. 行レベル split が無 seed.** `file_data_source.py:189`
  `__train_test_split` は `seed=None` 既定で，全呼び出し側が seed を渡さない
  (`dl.py:244`, `stage_component_factory.py:99,196`,
  `training_benchmark.py:1334`, `utility_interface.py:103`)．
  再開した学習が前回の検証行で訓練する．
  **見送り理由**: 既定 seed を入れると分割が変わる．同じ判断が
  `<path>` 外の複製2件にも及ぶ (O2) ので，まとめて決めるべき．
- **D3. array_type→loader ディスパッチがパッケージ内に3つ+α.**
  `file_data_source.py:42-63` (module table), `:559-564`
  (`_load_feather` が**呼び出しごとに dict を再構築**，関数内 import 付き),
  `:905-919` (`iter_batches_df` の if/elif 梯子),
  `streaming_file_source.py:34-48` (module 定数)．
  stage 追加時に4箇所の同期が要る．
- **D4. columnar→structured 変換器が2本ほぼ同一** (`:612` と `:734`,
  各約60行)．差は `np.empty(1)` / `np.empty(n)` と代入形だけ．
  `ColumnarBatch.slice` を使えば片方は1行になる．
  D1 の修正時に2箇所直す必要があるのはこれが理由．
- **D5. `cache_mode` の altitude.** `"file"` / `"memory"` はどちらも
  `__init__` で全ファイルをロードし (`:321-422`)，差は結合の有無だけ
  (`:428-436`)．`total_pages <= 1` なら完全に同一．しかも
  `_concatenate_numpy` (`:445`) / `_concatenate_columnar` (`:481`) は
  入力リストを保持したまま `np.concatenate` するので **ピーク2×**．
  OOM 警告 (`:447-458`, `:489-520`) は全ロード後・倍化直前に計算されるので
  **手遅れになってから出る**うえ，見積りが `move_win_rate` を数え落として
  いる (6フィールド列挙，最大フィールドが欠落) → 40GB 常駐を 18GB と
  報告し 32GB 閾値に掛からない．
  **見送り理由**: ノブ自体の廃止は `interface/learn.py`,
  `console/utility.py`, `app/learning/dl.py`, CLI に跨る (O5)．
- **D6. 死んだ/placeholder 状態.** `_FileEntry.memmap`・`.dtype` は
  placeholder (columnar 分岐は `:369` で `np.dtype("uint8")` という
  **嘘の値**を入れる)，`memmap_arrays` (`:266`) は append も read も
  されない，`_last_file_idx` (`:276`) は「最適化」と注記されつつ未読，
  `bit_pack` (`:257`) は保存のみ．
  `memmap_arrays` は `infra/object_storage/data_source.py:135,178` では
  **生きた機構**であり，同じ名前が片方だけ化石化している．
- **D7. `iter_batches_df` の死枝と再読込.** `:890` の
  `isinstance(entry.cached_array, pl.DataFrame)` は到達不能
  (`cached_array` は ndarray か `None` のみ) なので常に else 側に落ち，
  `FileManager.__init__` が既に読んだ `.feather` を**全件再読込**する．
  唯一の production caller は `infra/utility/benchmark_polars_io.py:451`．
- **D8. `file_level_split` は避けるはずの全ロードを払わないと呼べない.**
  `FileDataSourceSpliter.__init__` (`:89`) が `FileManager` を構築＝全ロード
  し，`file_level_split` はそこから `file_paths` と `array_type` の2つしか
  使わない．`interface/learn.py:1306-1312` が「Stage 3 で ~123GB，
  spawn worker 起動時に OOM kill される」と明記している．
  production は呼ばず `learn_model.py:876-892` で同じ算術をインライン再実装
  (O4)．**production caller ゼロ**，テストのみ．
- **D9. `train_test_split` が `list(range(N))` を作る.** `:100-106`．
  5000万行で Python list ≈1.8GB + スライス2本 + 変換後の int64 配列
  ≈ 索引だけで約2.6GB が同時生存．`np.random.Generator.permutation` なら
  C ループ1回．**seed 固定時の分割値が変わる**ので挙動変更．
- **D10. 重複イテレータと `total_pages` の不整合.**
  `iter_files_columnar` (`streaming_file_source.py:161`) は
  `iter_files_columnar_subset` から計測ログを抜いただけの二重実装で，
  production は後者しか呼ばない．また `total_pages()` (`:922`) は
  ファイル数を返すが `cache_mode="memory"` の `iter_batches` は
  `("concatenated", …)` を **1個だけ** yield する (`:704-716`)．
  `hcpe_transform.py:677,683` がこれを `tqdm(total=…)` に渡すので
  当該組み合わせで進捗が 1/N のまま動かない．
- **D11. 行数スキャンが逐次 + ファイルごとに `open()` 2回.**
  `_ensure_row_counts` は `scan_row_count` を直列実行し，各回で
  マジックバイト用に開いて閉じ，`pl.scan_ipc` で開き直す．
  I/O bound で GIL を離すので並列化可能．gcsfuse/ネットワーク越しの
  500ファイルでは起動レイテンシの支配項．

## Doc findings

**3 stale + 3 wrong** → `reviews/2026-08-10-file-system-docs-drift.md`
(`da6044e` で提案，**本 run で承認・適用** → `status: applied`,
`applied_in: 1c6a442`)．

- `.npy` と書かれているが Arrow IPC (`utility_benchmark_dataloader.md:19`,
  `utility_benchmark_training.md:30`)
- bit-pack が機能すると書かれている (同上 + `pre_process.md:20`)．
  オプション自身の help は `"[Deprecated] ... has no effect."`
- `cache_mode` を常駐量の軸として説明 (`utility_benchmark_dataloader.md:23`,
  `utility_benchmark_training.md:36`)
- `docs/rust-backend.md:677` の例が `cache_mode="mmap"` を渡しており，
  `file_data_source.py:254-261` の検証で **ValueError になる**
  (CLI は正規化してから渡すので CLI 経由では起きない)

**正確と確認したもの** (再検査不要): `docs/architecture.md:158-160` の
「`array_type` Literal の正準定義は `file_data_source.py`」．
step 2.5 key 1 で Python 15箇所 + Rust enum を照合済み．
今日追加した `docs/design/data-pipeline/index.md` の同趣旨の記述も同様に正確．

## Out of scope

`<path>` 外のサイトを含むため本 run では修正しなかったもの．
**すべて backlog に行あり．**

- **O1. `BigQueryDataSource` が learn 側の契約を破り，実行時に落ちる.**
  `bq_data_source.py:659` の `__getitem__` は `pl.DataFrame` の1行を返すが
  `dataset.py:46-52` は `np.ndarray` を要求．`KifDataset.__getitem__`
  (`dataset.py:87`) が最初のサンプルで `data.dtype.names` を呼ぶため
  **`AttributeError` でバッチ0から落ちる**．
  再現: `maou utility benchmark-dataloader --input-dataset-id … --input-table-name …`
  (`utility.py:318` → `utility_interface.py:103` → `dataloader_benchmark.py:93`)．
  `benchmark-training` も同配線 (`utility.py:1298`)．
  さらに `iter_batches_df` を override しておらず，継承した既定実装
  (`hcpe_transform.py:86-140`) が `array.dtype.names` を呼んで同様に落ちる．
  **構築時に捕まらない理由**: `dataset.py:45` と `hcpe_transform.py:62` は
  `@abc.abstractmethod` を付けながら `abc.ABCMeta` / `abc.ABC` を使って
  いないのでマーカーが**不活性**．
- **O2. `__train_test_split` の複製2件がグローバル RNG を汚染.**
  `infra/object_storage/data_source.py:86` と
  `infra/bigquery/bq_data_source.py:81` は互いに文字単位で同一，かつ
  `random.seed(seed)` のまま．`8c1417e` で `<path>` 側だけ直したので
  **修正が既に乖離している**．現状 seed を渡す呼び出し側はないので休眠．
- **O3. `columnar_batch.py:91` の optional 列判定が `batches[0]` のみ.**
  ゲートは要素0を見るのに内包表記は全要素を走るため，不一致時に
  列が **短いまま結合** (行対応が崩れる) か **黙って落ちる**．
  `moveWinRate` を持つファイルと持たないファイルが混在した
  preprocessing ディレクトリで `file_data_source.py:527` 経由で到達
  (`--input-cache-mode memory`)．現状は D1 により誰も読まないので
  顕在化していないが，**消費者が1人増えた瞬間に発火する**．
- **O4. `learn_model.py:876-892` の inline split.**
  `test_ratio or 0.1` なので `--test-ratio 0.0` が **黙って 0.1 になる**
  (利用者が検証分割なしを要求しても10%取られる)．また seed 42 決め打ちで，
  `file_level_split(seed=None)` (非再現) と方針が食い違う．
  同じ算術が `utility.py:1211-1229`, `:1256-1272` にもあり計4箇所．
- **O5. cache/ノブ系の意味の分裂.**
  (a) `--input-local-cache` は BigQuery にしか渡されず，S3/GCS 分岐
  (`pre_process.py:417-431,449-463`) は `input_local_cache_dir is not None`
  で判定するため **`maou pre-process --input-s3 --input-local-cache` は無言の
  no-op**．(b) `--input-max-cached-bytes` は BigQuery では LRU 退避予算
  (`bq_data_source.py:118,245-271`)，object storage では並列DLのチャンク幅
  (`object_storage/data_source.py:122,260-265`) と**別物**．
  (c) `--input-enable-bundling`/`--input-bundle-size-gb` は死んだノブ
  (`object_storage/data_source.py:212-213` の docstring が明言)，かつ既定値が
  層をまたいで不一致 (`:45` は `True`，`:115`/`:407` は `False`)．
  (d) `learn-model` には `--input-cache-mode` が **存在せず** `"file"` 決め打ち
  (`learn_model.py:796,820,847`)．ベンチマーク側にだけ露出している．
- **O6. `bq_data_source.py:483-493` が `.npy` を数えて `.feather` を書く.**
  `__save_to_local` (`:285-293`) は `.feather` を書くのに検証は
  `glob("*.npy")`．結果 **毎回** "Created 0 local cache files" と
  "No local cache files were created. This might indicate a problem." が出る．
  ログのみだが，まさにその操作を報告する場面で誤導する．
- **O7. Arrow IPC 形式判定の fallback 方向が Python と Rust で逆.**
  Python (`streaming_file_source.py:230-247`,
  `domain/data/dataframe_io.py:19,35`) は「File か?」を問い既定 Stream，
  Rust (`rust/maou_io/src/arrow_io.rs:71-92`) は「Stream か?」を問い既定 File．
  Arrow 0.15 以前の Stream ファイルは `scan_row_count` が成功し
  `load_feather` が footer エラーで失敗する．
  加えて `rust/maou_index/src/index.rs:205-217` は **判定なし**で常に
  File 前提なので，Stream 形式の `.feather` は他の全経路で読めるのに
  visualize の索引構築だけ失敗する．
- **O8. `interface/preprocess.py:181` の行数取得.**
  `len(pl.scan_ipc(fp).collect())` は **全列を実体化**する
  (`scan_row_count` は `select(pl.len())`)．1496幅のリスト列を持つ
  preprocessing ではメタデータ読みと全ロードの差．さらに `:186` の
  裸の `except Exception:` が Stream 形式ファイルの失敗を飲み込み，
  `ok_files` 扱いで `chunk_input_files` から黙って漏れる．
- **O9. `bq_data_source.py:222-243` の `sample_ratio` 二重抽出.**
  `total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが，
  `__fetch_from_bigquery` (`:405-420`) はページごとに**別々に引き直す**
  `TABLESAMPLE` を発行する．同じ行集合とは限らず，件数も一致しない．
  ファイル系ソースは常に厳密なので，ここだけ挙動が違う．
  また `:235` の `num_rows` はテーブルメタデータでストリーミング挿入に
  遅れる．
- **O10. `dataframe_io.py:19` が Arrow マジック定数を再定義.**
  `streaming_file_source.py:230` と同値・同幅・同 fallback で **IDENTICAL**．
  入力型が `Path` と `bytes` で違うだけ．定数だけでも共有すべき．

## 補足: このrunが step 2.5 について示したこと

- 2.5 は **path 監査では原理的に見えない欠陥を実際に出した**．
  O1 (BigQuery の ABC 違反 → 実行時 `AttributeError`) は
  `bq_data_source.py` を単体監査しても「ABC が不活性なので落ちない」で
  終わり，`dataset.py` を単体監査しても「契約は書いてある」で終わる．
  両端を突き合わせて初めて出る．
- 2.5c の分類は **機能した**．`visualize` の5メンバ `click.Choice` は
  一見 divergence だが，4段のガードを追跡して DELIBERATE と判定できた．
  「似ているから直す」に堕ちていない．
- 2.5d も機能した．key 1 は「共有ヘルパーがなく15回 re-type されている」
  という重複を検出したが，**メンバ集合は一致している**ので
  DIVERGENT ではなく低優先の重複として正しく格下げされた．
- コストは実測で `Explore` 1回・約8.6分・約196k トークン (level `high`)．
  1428行の path に対しては steps 1-2 より重い．`low`/`medium` で
  key を絞る設計 (Usage 節) は妥当だった．
