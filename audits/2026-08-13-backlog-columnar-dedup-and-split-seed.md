---
kind: backlog
date: 2026-08-13
path:
  - src/maou/domain/data/columnar_batch.py
  - src/maou/interface/data_io.py
  - src/maou/interface/learn.py
  - src/maou/infra/file_system/file_data_source.py
  - src/maou/infra/file_system/streaming_file_source.py
  - src/maou/infra/object_storage/data_source.py
  - src/maou/infra/bigquery/bq_data_source.py
  - src/maou/infra/utility/benchmark_polars_io.py
  - docs/adr-001-dataloader-multiprocessing-optimization.md
level: medium
last_sha: 05654ba
---

# `/audit-backlog` — columnar 変換の重複 (D3+D4)，分割 seed (D2)，`.npy` ベンチ (N-2)

`coverage.md` の backlog 2 表から **21 行** (deferred 11 + out-of-scope 10)
を拾い，HEAD `05654ba` に対して全件を再検証した．**stale はゼロ** —
21 件すべてが今も成立する．うち 1 件 (N-2) は **changed shape** で，
記録が書いていたより深刻だった (下記 § Corrections)．

> **数え直しの訂正**: run の途中では「19 行 (deferred 11 +
> out-of-scope 8)」と報告していたが，out-of-scope 表を 2 行数え落として
> いた．分類自体は 21 行すべてを覆っている (下の表 4 + 17 = 21) ので，
> 誤っていたのは合計値だけである．§ Reconciliation は訂正後の数字．

## Classification

判断コスト P1-P6 + ゲートで分類した．`P6 → P1` の順に評価し，最初に
引っ掛かったクラスを採る．

### 消化した 3 件 + doc 1 件

| ID | backlog 行 | 対象 | クラス | そのクラスに決めたテスト | ゲート |
|---|---|---|---|---|---|
| P3-1 | D3+D4 (a)+(b) | `infra/file_system` | **P3** | 受理する全入力に対し，返るレコードも `iter_batches` の配列も不変．経路だけが変わる．characterization test で固定 (§ Applied) | なし |
| P4-1 | D2 | `infra/{file_system,object_storage,bigquery}` | **P4** | どの行が訓練でどの行が検証かが変わる．既存の成果物は読めるまま，既存の起動オプションも有効なので P5/P6 ではない | G2 を**解除**．修正は 3 つの splitter と `interface/learn.py` の定数 1 個で閉じ，protocol も呼び出し側も触らない |
| P6-1 | N-2 | `infra/utility` | **P6** | 公開関数 4 本 (`save_hcpe_array` ほか) を削除し，documented command の出力形式が変わる | G4 を**解除** — この run でユーザに聞き，「`.npy` を全削除」の回答を得た |
| doc-1 | N-3 | `docs/adr-001-…` | 判断帯 | **P2 の一意性テストに落ちる**: 日付入り ADR 注の扱いに 3 通りの妥当な書き方があり，訂正後の本文がコードから一意に決まらない | 提案のみ．doc は**編集していない** |

### 再検証で判断帯に留まった 17 件

| backlog 行 | クラス相当 | ゲート | この run で留めた理由 |
|---|---|---|---|
| app/learning Deferred 2 | P3 | **G3** | Stage 1/2 パイプラインの ~400 行リファクタ．「挙動不変」を主張するには学習経路の等価性確認が要るが，この環境に GPU が無く長時間の実測もできない |
| app/learning Deferred 3 | P6 | **G4** | アダプタ 6 クラスの統合は公開名を消す．記録自身が「独立した reviewed change にすべき」と書く |
| app/learning Deferred 4 | P3 | **G3** | `callbacks.py` の基底抽出 ~250→120 行．Deferred 2 と同じ理由 |
| app/learning Deferred 5 | P4 | **G1** | dormant のまま (マスクを供給する経路が無い)．実際のマスクを配線する変更と一緒に GPU で測るべき |
| app/learning Deferred 6 | P4 | **G1** | `wait_stream()` 化は GPU セマンティクスの変更．実機検証が要る |
| app/learning Deferred 7 | P4 | **G1** | GNS の device scalar 化は数値等価性の確認が要る |
| D5 | P4 | **G4** | `cache_mode` ノブの廃止は O5 と一体という記録の判断を，再検証でも覆せなかった |
| D8+D9 (残り) | P4 | **G4** (**内容を更新**) | § Re-triaged 参照 |
| D10+D11 | P4 | **G4** | (1) `FileDataSource.total_pages()` は production caller ゼロのまま dormant．(2) 行数スキャンの逐次化は D14(a) と同時が自然 |
| D13 | P4 | **G2** | `__getitem__` の per-sample `np.empty` の根本解決は `app/learning/dataset.py` と ABC を触る |
| D14 | P3/P6 | **G2** | (a) 行数スキャン共有は 2 モジュール + `StreamingHcpeDataSource` に及ぶ．(b) ABC 2 枚の解消は `benchmark_polars_io` の対応を伴う |
| D15 | P4 | **G4** | 不完全な `.feather` を size/footer 検査で弾く是非そのものが要判断 (運用上のリスクの実在性) |
| O5 | P6 | 判断帯 | CLI 契約の変更を 4 つの小問 (a)-(d) に分けて決める必要がある |
| O9 | P4 | **G1** | BigQuery の `TABLESAMPLE` 不整合は BQ 実環境が無いと確認できない |
| N3 | P4/P6 | 判断帯 | `ABC` 継承を入れると現存の非準拠実装が構築時エラーになる．何が壊れるかの洗い出しが先 |
| N4 | P1 | **G4** | この run でも再現した (§ Environment notes)．「薄いテストへ切り出す」か「CPU extra を必須にする」かの決めが未解決 |
| N-1 | P5 | **G4** | polars(BinaryView) と Rust writer(LargeBinary) のどちらに寄せるかが未決．(b) を採ると polars 経由の書き出しを全部止めることになる |

## Consumed

| backlog 行 | 対象 | 出荷したもの | 行の扱い |
|---|---|---|---|
| D3+D4 | `infra/file_system` | `22da510` | **削除** |
| D2 | 3 データソース | `ecc7a30` | **削除** |
| N-2 | `infra/utility` | `d0c4984` | **削除** |
| N-3 | `docs/adr-001-…` | `33b3573` (提案 `reviews/2026-08-13-adr-001-targets-note.md` は `applied`) | **削除** |

**ユーザがこのセッション中に 3 件とも回答した**ため，4 行とも消化して
行を削除した (5e: セッション中に回答があれば適用して 5d の条件でマージ
する)．回答は「seed は 0 固定で問題ない」「`.npy` 削除は意図どおり」
「adr-001 は案 1」．

## Applied

| 変更 | 場所 | commit |
|---|---|---|
| `COLUMNAR_CONVERTERS` を domain に 1 本化し，2 つの infra モジュールが引くようにした | `domain/data/columnar_batch.py:301`，`interface/data_io.py:14`，`infra/file_system/file_data_source.py:21`，`streaming_file_source.py:28` | `22da510` |
| `_columnar_to_structured_record` を `_columnar_batch_to_structured_array(batch, row=idx)` の薄いラッパにした | `infra/file_system/file_data_source.py:498` | `22da510` |
| `DEFAULT_SPLIT_SEED` を追加し 3 つの splitter が引くようにした | `interface/learn.py:61`，`file_data_source.py:85`，`object_storage/data_source.py:73`，`bq_data_source.py:73` | `ecc7a30` |
| `.npy` の I/O ヘルパ 4 本と 3 ベンチの numpy 半分・比較部を削除 | `infra/utility/benchmark_polars_io.py` | `d0c4984` |
| `_assert_covers_schema` を追加し，テストデータのスキーマ追従漏れを名指しで落とすようにした | `infra/utility/benchmark_polars_io.py:42` | `d0c4984` |

### 回帰テスト

| テスト | 何を押さえるか |
|---|---|
| `test_single_record_matches_batch_conversion` | `get_item` と `iter_batches` の一致 (P3 の「挙動不変」の根拠．全フィールドが行ごとに異なるデータを使い，1 フィールド潰すと落ちることを確認) |
| `test_negative_index_still_selects_the_last_row` | **trap**: 素朴な `arr[idx:idx+1]` は `idx=-1` で空になる．元の `arr[idx]` の負インデックス挙動を固定 |
| `test_columnar_converter_table_is_shared` | 2 モジュールが同一オブジェクトを引くこと (dict が再び 2 本に割れない) |
| `test_public_split_passes_the_default_seed` | 3 実装とも公開経路で seed を渡すこと |
| `test_file_system_split_is_stable_across_instances` | 実ファイルで，再開に相当する 2 回目の構築が同じ分割になること |
| `test_main_runs_to_completion` / `test_no_npy_artifacts_are_written` / `test_npy_helpers_are_gone` | documented command が完走すること，`.npy` を書かないこと，ヘルパが復活していないこと |

非空虚性は 3 件とも実測: 修正を潰すと該当テストが落ち，戻すと通る．

## In flight

| PR | クラス | base | 未決の問い |
|---|---|---|---|
_(なし)_ — [#491](https://github.com/dousu/maou/pull/491) が抱えていた
3 つの問いはすべてこのセッション中に回答され，解決済み．

**class ごとに PR を分けていない**理由: このセッションは
`claude/audit-backlog-ejdnzm` という designated branch を渡されており，
「NEVER push to a different branch without explicit permission」の指示が
ある．`/audit-backlog` 5a は designated branch の指示が勝つと定めるので，
1 ブランチ 1 PR に畳んだ．commit は class ごとに分けてあるので，
review 単位は commit 境界で読める．

その帰結として，**自動帯 (P3) も `main` に着地していない**．通常なら
P3 は無条件で settled として stack 最下段からマージされるが，同じ PR に
判断帯が同居するため，ユーザの判断を待つ．

## Re-triaged

**D8+D9 (残り) — `train_test_split` の `list(range(N))`.**
記録は「`np.random.Generator.permutation` にすると seed 固定時の分割値が
変わるので D2 と同じ決めに帰着する」と書いていた．再検証で**依存の向きが
はっきりした**: この run で D2 (既定 seed) が入るまでは，そもそも seed を
固定する手段が公開経路に無かったので「守るべき再現可能な分割」は存在
しなかった．D2 が入った**後**は分割値が契約になるため，permutation への
切り替えはそれを 1 度だけ意図的に壊す変更になる．したがって **D2 の PR が
マージされてから，独立した P4 として判断する**のが正しい順序．

同時に，記録に無かった障害を 1 つ見つけた: `bq_data_source.py:641` の
`indicies` は `list[int] | None` と宣言され `np.asarray` を通さず
そのまま代入される (`:694`)．`file_system` と `object_storage` は
`np.asarray` するので，permutation の ndarray を返すと **BigQuery だけ
型注釈と実体がずれる**．切り替えるときはこの 3 実装の受け口を先に
揃える必要がある．

**N4 — torch 無し環境でのテスト消失.** この run でも再現した．背景は
§ Environment notes．記録が書く「未解決の判断点」はそのままだが，
**実害が 2 run 連続で出ている**という事実を行に足した．

## Corrections to the source records

`audits/2026-08-12-backlog-arrow-format-and-clippy.md` の N-2 に対する訂正
(記録本体には追記済み):

N-2 は「使う予定のない形式との比較ベンチを残す意味があるか」という
**保守方針の問い**として記録されていた．実際には
`benchmark_datasource_iteration` が `.npy` パスを `FileDataSource` に
渡しており，`FileManager.__init__` の拡張子ガード
(`Only .feather files are supported`) に必ず当たる．つまり
`docs/performance.md:72` が案内する
`python -m maou.infra.utility.benchmark_polars_io` は
**一度も最後まで走ったことがない**．記録は「残すか消すか」を問うていたが，
正しい問いは「壊れているものをどう直すか」だった．

**この種の主張を疑う指針**: 「CLI から到達しない」は「実行されない」を
意味しない．`docs/` が案内する `python -m` 形式の入口は CLI 参照の
grep に掛からないので，到達性を測るときは `docs/` も走査対象に含める
こと．

さらに，テストを書いて初めて **2 つ目の破損**が出た:
`_create_preprocessing_test_data_polars` のハードコード dict が
preprocessing スキーマに追い付いておらず (`moveWinRate`,
`bestMoveWinRate` の 2 列が欠落)，polars 内部の `KeyError` で落ちる．
**「壊れている」と分かっている経路には，直す前に必ずテストを回すこと** —
1 つ目の破損を直しただけでは完走しなかった．

## Doc findings

| ファイル | status | 適用したか |
|---|---|---|
| `reviews/2026-08-13-adr-001-targets-note.md` | `pending` | **していない**．P2 の一意性テストに落ちる (日付入り ADR 注の扱いに 3 案) ため，CLAUDE.md の standing approval は及ばない．提案だけを PR に載せ，承認を待つ |

`.npy` 削除に伴う doc drift は**発生しなかった**: `docs/performance.md` は
コマンド例のみで `.npy` に言及していない (`grep -n "npy" docs/performance.md`
の結果はゼロ)．

## Out of scope

この run が新たに気付いたもの．`coverage.md` の backlog にも行を足した．

- **N5-1**: `benchmark_polars_io` のテストデータがスキーマから導出されて
  いない．今回は `_assert_covers_schema` で「欠けたら名指しで落ちる」まで
  詰めたが，値の shape (9x9 / 14 / `MOVE_LABELS_NUM`) は依然ハードコード
  で，型が変わる drift は捕まえられない．スキーマからの生成に寄せるかは
  判断が要る．

## Reconciliation (6d)

触れた項目 + 新規発見 = **22** (backlog 21 行 + 新規 1)

- **resolved** (行削除): **4** — D3+D4 / D2 / N-2 / N-3
- **in flight**: **0**
- **re-triaged** (行保持・文面を鋭くした): **2** — D8+D9 / N4
- **そのまま保持** (再検証で confirmed，文面変更なし): **15** —
  app/learning Deferred 2-7 (6) / D5 / D10+D11 / D13 / D14 / D15 / O5 /
  O9 / N3 / N-1 (9)
- **new row**: **1** — N5-1
- **not a finding**: **0**

4 + 0 + 2 + 15 + 1 = **22** ✓

backlog 行数: **21 → 18** (21 − 4 + 1)．

`coverage.md` 本表の `Open items` も更新した:
`src/maou/infra/file_system` を `8 deferred` → `3 deferred`．
**8 という数字はこの run の前から既に古かった** — 削除前の時点で
deferred 表の infra/file_system 行は 5 本しかなかった．実数に合わせた．

## Environment notes

- **G3 は発生しなかった**: `uv sync --extra cpu` が通り (Rust 拡張の
  maturin ビルドに ~15 分)，ruff / mypy / pytest 全件をこの環境で実行
  できた．全件 **1868 passed, 34 skipped**．
- **N4 の実害を再確認**: コンテナ初期状態の `.venv` は空で，
  `--extra cpu` を入れるまで `polars` すら import できなかった．
  base extra だけで回すと `infra/file_system` のテストが丸ごと skip
  され，この run の変更が無検証のまま緑に見える状態だった．
- `uv lock` は `tensorrt-cu12-libs` のビルドを伴うため初回だけ ~50 秒
  かかる．`uv run` は version 変更後に暗黙の再解決を試みるので，
  bump 直後のテストは `uv run --no-sync` で回した．
- **G1**: GPU / BigQuery / S3 / GCS は無い．app/learning の Deferred
  5/6/7 と O9 はこの環境では確認できない．
