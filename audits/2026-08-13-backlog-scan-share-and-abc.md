---
kind: backlog
date: 2026-08-13
path:
  - src/maou/domain/data/arrow_format.py
  - src/maou/infra/file_system/streaming_file_source.py
  - src/maou/infra/file_system/streaming_hcpe_source.py
  - src/maou/infra/utility/benchmark_polars_io.py
  - src/maou/interface/learn.py
  - src/maou/infra/file_system/file_data_source.py
  - src/maou/infra/object_storage/data_source.py
  - src/maou/infra/bigquery/bq_data_source.py
  - src/maou/app/learning/dataset.py
  - src/maou/app/pre_process/hcpe_transform.py
  - rust/maou_io/src/arrow_io.rs
level: medium
last_sha: 7bcb830
---

# `/audit-backlog` — 行数スキャン共有 (D14a)，ベンチのスキーマ導出 (N5-1)，分割索引 (D8+D9)，`DataSource` の ABC 化 (N3)，feather writer 混在 (N-1)

`coverage.md` の backlog 2 表から **18 行** (deferred 9 + out-of-scope 9)
を拾い，HEAD `7bcb830` に対して全件を再検証した．

- **stale: 0** — 18 件すべてが今も成立する
- **changed shape: 2** — D8+D9 と N3．どちらも「記録が deferral の
  理由として挙げた条件が，この run までに消えていた」型の変化 (§ Classification)
- **confirmed: 16**

うち **5 件**を消化し，[PR #492](https://github.com/dousu/maou/pull/492)
に載せた．PR を出したあと**ユーザが同一セッション中に「別途選択が要る
箇所がなければマージしてよい」と回答**したため，5d の条件 (この環境の
QA が通っている / GitHub の check が緑 / stack の上に未解決が無い) を
確認してマージした．

## Classification

判断コスト P1-P6 + ゲートで分類した．`P6 → P1` の順に評価し，最初に
引っ掛かったクラスを採る．

### 消化した 5 件

| ID | backlog 行 | 対象 | クラス | そのクラスに決めたテスト | ゲート |
|---|---|---|---|---|---|
| P3-1 | D14 **(a)** | `domain/data` + `infra/file_system` | **P3** | 受理する全入力に対し `__len__` / `total_pages` / `iter_batches` の値が不変．差はログ文言と経路だけ | **G2 を解除** (下記) |
| P3-2 | N5-1 | `infra/utility` | **P3** | ベンチの出力は計測値 (診断) のみ．書く成果物は自身が消す一時ファイル | **G4 が残る** (下記)．したがって自動帯ではなく判断帯 |
| P4-1 | D8+D9 (残り) | `infra/{file_system,object_storage,bigquery}` | **P4** | どの行が訓練/検証かが変わる．既存成果物は読めるまま，既存の起動オプションも有効なので P5/P6 ではない | **G4 を解除** |
| P4-2 | N3 | `app/learning` + `app/pre_process` | **P4** | 非準拠クラスの構築が `TypeError` になる = 観測可能な挙動変化．データ互換も CLI 契約も壊さない | **G4 を解除** |
| P4-3 | N-1 | `rust/maou_io` | **P4** (記録の **P5 から降格**) | 既存 `.feather` は読めるまま，**同一 writer だけの入力は出力の型も従来どおり**にする設計なので P5 ではない | **G4 を解除** |

**N5-1 の分類は run の途中で自分で訂正した．** 作業前に会話へ出した表では
「P3・ゲートなし = 自動帯」と書いたが，実装方針を詰める過程で
**polars スキーマは長さを持たない** (固定長は Arrow / structured dtype
側の情報) ことが分かり，「スキーマから導出する」が 1 通りに定まらない
ことが判明した．行自身が「スキーマからの生成に寄せるか，このガードで
十分とするかの判断が要る」と書いているとおりで，これは G4 そのもの．
**fail-safe の向きは上**なので判断帯へ移した．

### ゲート解除の根拠

| 行 | 元のゲート | 解除の理由 |
|---|---|---|
| D14(a) | **G2** (「2 モジュール + `StreamingHcpeDataSource` に及ぶ」) | 波及先は `domain/data/arrow_format` と `infra/file_system` の 2 モジュールだけで，どちらも item が名指しする範囲内．公開呼び出し側は変わらない．**`interface/preprocess.py:192` の 3 本目のループは共有しなかった** — あちらはファイル単位で例外を握って skip する別の意味論なので，寄せると挙動が変わる |
| D8+D9 | **G4** | 行が明記する前提「D2 がマージされた**後**に独立した P4 として判断すること」が，#491 のマージで満たされた |
| N3 | **G4** (「何が壊れるか洗う必要がある」) | 洗い出しをこの run で完了 (下記 § 洗い出し結果)．非準拠は**テストの 1 クラスだけ** |
| N-1 | **G4** ((a)/(b) の二択が未決) | 二択は残るが，(a) は 60 行で閉じ，しかも **(b) を採る余地を潰さない** (混在入力への防御として残せる)．one-check 原則の「分岐は違うが差分が小さく作り直しが安い」場合にあたるので，推奨案 (a) を書いて代案を PR 本文に置いた |

### N3 の洗い出し結果 (G4 を外した実体)

`Explore` で `src/` と `tests/` を AST 走査し，2 つの基底の全サブクラスを
列挙した．

- **`app/learning/dataset.DataSource`**: 全サブクラスが準拠．
  `dl.py:71 LearningDataSource` は両メソッドとも未定義だが
  **production では一度も構築されない**中間抽象クラスなので，ABC 化で
  構築不能になるのは意図どおり → **壊れるものはゼロ**．
- **`app/pre_process/hcpe_transform.DataSource`**: 抽象メソッドは 2 本では
  なく **3 本** — `total_pages` が具象 `iter_batches_df` の**下**にあるため
  見落とされていた．非準拠は
  `tests/maou/app/pre_process/test_search_value.py:573 _NoDataSource`
  **ちょうど 1 件**で，`total_pages` を持たないまま `:601` で構築されて
  いた (docstring が「データソースは触られない」と書いており，それが
  見逃しの理由になっていた)．
- **基底を直接 `DataSource()` する箇所**: ゼロ．
- **duck typing の実装** (`polars_datasource.PolarsDataFrameSource` など)
  は基底を継承していないので ABC の影響を受けない．

### 再検証で判断帯に留まった 13 件

| backlog 行 | クラス相当 | ゲート | この run で留めた理由 |
|---|---|---|---|
| app/learning Deferred 2 | P3 | **G3** | Stage 1/2 パイプラインの ~400 行リファクタ．「挙動不変」を主張するには学習経路の等価性確認が要るが GPU が無い |
| app/learning Deferred 3 | P6 | **G4** | アダプタ 6 クラスの統合は公開名を消す．記録自身が独立した reviewed change を要求 |
| app/learning Deferred 4 | P3 | **G3** | `callbacks.py` の基底抽出 ~250→120 行．Deferred 2 と同じ理由 |
| app/learning Deferred 5 | P4 | **G1** | dormant のまま (マスクを供給する経路が無い)．実マスクの配線と一緒に GPU で測るべき |
| app/learning Deferred 6 | P4 | **G1** | `wait_stream()` 化は GPU セマンティクスの変更 |
| app/learning Deferred 7 | P4 | **G1** | GNS の device scalar 化は数値等価性の確認が要る |
| D5 | P4 | **G4** | `cache_mode` ノブの廃止は O5 と一体という記録の判断を，再検証でも覆せなかった |
| D10+D11 | P4 | **G4** | (1) `FileDataSource.total_pages()` は production caller **今もゼロ**で dormant．(2) は文面を鋭くした (§ Re-triaged) |
| D13 | P4 | **G2** | `__getitem__` の per-sample `np.empty` の根本解決は `app/learning/dataset.py` と ABC を触る．文面を鋭くした (§ Re-triaged) |
| D14 **(b)** | P6 | **G2** | ABC 2 枚の解消は `benchmark_polars_io` の対応を伴う．(a) だけ消化した |
| D15 | P4 | **G4** | 不完全な `.feather` を size/footer 検査で弾く是非そのものが要判断 |
| O5 | P6 | 判断帯 | CLI 契約の変更を 4 つの小問 (a)-(d) に分けて決める必要がある |
| O9 | P4 | **G1** | BigQuery の `TABLESAMPLE` 不整合は BQ 実環境が無いと確認できない |
| N4 | P1 | **G4** | 「薄いテストへ切り出す」か「CPU extra を必須にする」かが未決．**この run でも実害が出た** (§ Environment notes) |

## Consumed

| backlog 行 | 対象 | 出荷したもの | 行の扱い |
|---|---|---|---|
| D8+D9 (残り) | 3 データソース | `7dcf993` | **削除** |
| N3 | `app/{learning,pre_process}` | `b652d5e` | **削除** |
| N-1 | `rust/maou_io` | `1a393ff` | **削除** |
| N5-1 | `infra/utility` | `e7c5d3e` | **削除** |
| D14 **(a)** | `domain/data` + `infra/file_system` | `b3568f9` | **行は残す** — (b) が未着手なので，行を (b) の記述に縮めた |

**消化は 5 件だが削除した行は 4 行**である．D14 は 1 行に (a) と (b) の
2 つの finding が入っており，(a) だけが出荷されたため．行ごと消すと
(b) が backlog から見えなくなる．

## Applied

| 変更 | 場所 | commit |
|---|---|---|
| `scan_row_counts` を domain に新設し，2 つの streaming source が引くようにした | `domain/data/arrow_format.py:88`，`streaming_file_source.py:109`，`streaming_hcpe_source.py:62` | `b3568f9` |
| `StreamingHcpeDataSource` が per-file カウントを捨てるのをやめ `row_counts` で公開 | `streaming_hcpe_source.py:97` | `b3568f9` |
| ベンチのテストデータを polars スキーマ (列集合) + structured dtype (長さ) から導出．`_assert_covers_schema` は役目を引き継がれて削除 | `infra/utility/benchmark_polars_io.py:43-198` | `e7c5d3e` |
| `train_test_split_indices` を新設し，3 実装の同一な private 複製を廃止．索引を ndarray 化 | `interface/learn.py:66`，`file_data_source.py:74`，`object_storage/data_source.py:61`，`bq_data_source.py:62` | `7dcf993` |
| BigQuery の `indicies` を `np.asarray(..., int64)` に通す (他 2 実装と同じ受け口) | `bq_data_source.py:690` | `7dcf993` |
| 2 つの `DataSource` 基底を `abc.ABC` 化，非準拠テストクラスに `total_pages` を追加 | `app/learning/dataset.py:45`，`app/pre_process/hcpe_transform.py:63`，`tests/.../test_search_value.py:573` | `b652d5e` |
| `consolidate_batches` がスキーマの食い違うときだけ view/非 view を正規化 | `rust/maou_io/src/arrow_io.rs:32,64` | `1a393ff` |

### 回帰テスト

| テスト | 何を押さえるか |
|---|---|
| `TestScanRowCounts` | 入力順・per-file 一致 (**P3「挙動不変」の根拠**)・空入力・例外時に部分結果を返さないこと |
| `TestSharedRowCountScan` | `__len__` が per-file スキャンの合計と一致 (挙動不変の根拠)，`row_counts` の公開とコピー性，再スキャンしないこと，**例外時に memo しないこと** (以前は `StreamingFileSource` 側にしかテストが無く，hcpe 側が安全なのは構造の偶然だった) |
| `TestSchemaDerivedTestData` | 生成 DataFrame がスキーマと完全一致，List の長さが structured dtype 由来，文字列が宣言幅を超えない．**trap**: 長さ不明の List 列が増えたら**列名を名指しして**落ちること |
| `TestSplitIndices` | `random` と numpy 双方のグローバル RNG を汚さないこと，seed 再現性，seed が効くこと，過不足のない分割，**ndarray/int64 を返すこと**，端の値 |
| `test_public_split_uses_the_shared_helper_with_the_default_seed` / `test_private_split_copies_are_gone` | 3 実装が共有ヘルパを既定 seed で呼び，private 複製が復活しないこと |
| `TestBigQueryIndiciesAcceptance` | list / ndarray / 既定のいずれでも int64 の ndarray になること |
| `TestDataSourceIsAbstract` / `test_datasource_abc.py` | 両基底が `ABCMeta`，**抽象メソッド集合が 3 本** (`total_pages` を数え落とすと落ちる)，未実装は構築時 `TypeError`，全部埋めた実装は従来どおり構築可，**具象の既定 `iter_batches_df` が abstract 扱いされないこと** |
| `arrow_io::tests::consolidate_*` (Rust 6 件) / `test_feather_writer_mix.py` (Python 5 件) | 混在の結合，バイト列の保存，入力順で出力型が変わらないこと，**スキーマ一致時は正規化しないこと**，view と無関係な型の食い違いは従来どおり拒否すること |

**非空虚性は 5 件すべてで実測**した (修正を潰すと該当テストが落ち，戻すと通る)．

- `scan_row_counts` を「例外を握って部分結果を返す」形に潰す → 3 件が落ちる
- ベンチの List 長を `14` にハードコード → 3 件，名指しガードを 2 つとも潰す → trap テストが落ちる
- `np.random.permutation` (レガシーグローバル) に潰す → numpy RNG のテスト，`.tolist()` に潰す → ndarray のテスト
- `abc.ABC` の継承を外す → 7 件
- `normalize_view_types` の呼び出しを外す → Rust 4 件 + Python 2 件

## In flight

**なし** — [#492](https://github.com/dousu/maou/pull/492) が抱えていた
問い (**train/test の分割値が 1 度だけ変わることを受け入れるか**,
`7dcf993`) は同一セッション中にユーザが回答し，解決済み．

**class ごとに PR を分けていない**理由: このセッションは
`claude/audit-backlog-c06z52` という designated branch を渡されており，
「NEVER push to a different branch without explicit permission」の指示が
ある．`/audit-backlog` 5a は designated branch の指示が勝つと定めるので，
1 ブランチ 1 PR に畳んだ．commit は class ごとに分けてあるので，review
単位は commit 境界で読める．

その帰結として，**自動帯 (P3-1) も単独では `main` に着地しない**．
判断帯が同じ PR に同居するため，ユーザの判断を待つ．

**ユーザへの質問は 1 度も出していない．** 5 件とも「書くべきコードが
1 通りに決まる」ので，one-check 原則どおり判断は PR 上で 1 回だけ
受けてもらう形にした．

## Re-triaged

**D10+D11 (2) — 行数スキャンの逐次実行.** 記録は「D14(a) と同時に直すのが
自然」と書いていた．D14(a) (共有化) がこの run で入ったので，**その依存は
解消した**: 並列化は共有された `scan_row_counts` の 1 箇所を直せば
2 つの streaming source に同時に効く．行の文面をそう更新した．
(1) の `FileDataSource.total_pages()` は production caller ゼロのままで
dormant なので，判断は「この経路に caller が戻るとき」に据え置き．

**D13 (b) — `get_items` がバッチ化していない.** 再検証で
`FileDataSource.get_items` の**呼び出し側がゼロ**であることが分かった
(`FileManager.get_items` への内部委譲だけ)．つまり (b) 単独では
**dormant** で，実害は (a) の根本解決 (`KifDataset` が `ColumnarBatch` を
直接スライスする) に踏み込んだときに初めて出る．行にその事実を足した —
「(b) を先に直す」選択肢が実質無いことが分かるように．

**N4 — torch 無し環境でのテスト消失.** この run でも再現した
(§ Environment notes)．**実害が 3 run 連続**で出ている事実を行に足した．

## Corrections to the source records

**なし．** 5 件とも記録の診断は正しく，修正の向きも (N-1 の (a)/(b) の
選択を除いて) 記録が書いたとおりだった．N3 の「洗い出しが要る」という
判断も正しく，この run はその洗い出しを実行しただけである．

ゲートの解除 (G2/G4) は診断の誤りではなく，**deferral の前提条件が
その後の変更で消えた**ことによる．記録は書かれた時点で正しいので，
訂正は追記していない (6b: 訂正は診断・修正案が**誤っていた**ときだけ)．

## Doc findings

**なし．** 5 件の変更に対応する durable-doc の drift は発生しなかった．

- `docs/commands/learn_model.md` は `--test-ratio` の意味を書くが，行単位
  分割の**アルゴリズムや再現性**には触れていないので P4-1 で古びない
- `docs/commands/hcpe_convert.md` は `merge_hcpe_feather_files` に触れるが，
  writer の型制約には触れていないので N-1 で古びない
- `docs/rust-backend.md` の `PolarsDataFrameSource` の例は基底を継承しない
  duck typing なので N3 の影響を受けない (`grep` と全件テストの両方で確認)
- `docs/performance.md` はベンチのコマンド例のみでスキーマに触れない

したがって `reviews/*.md` の提案は**この run では 1 件も起票していない**．

## Out of scope

この run が新たに気付いたもの．`coverage.md` の backlog にも行を足した．

- **N6-1**: `bq_data_source.py:710` の `iter_batches` は
  `Generator[tuple[str, pl.DataFrame], None, None]` を返すが，基底
  (`hcpe_transform.py:77`) の宣言は `np.ndarray`．`PreProcess` は
  `hcpe_transform.py:683`/`:839` でこれを `_process_single_array(data:
  np.ndarray)` に渡し，そこは `data["hcp"]` / `np.ascontiguousarray` を
  する．さらに `BigQueryDataSource` は `iter_batches_df` を override
  しないので，基底の既定実装 (`:117` で `array.dtype.names`) に
  `pl.DataFrame` が渡って `AttributeError` になる．
  **`__getitem__` について同じ形の不具合が O1 として既に直っており
  (`tests/maou/infra/bigquery/test_bq_get_item_contract.py`)，
  `iter_batches` にだけ同じ手当てがされていない．**
  N3 (ABC 化) では捕まらない — ABC は存在を見るが型は見ないため．
  BigQuery 実環境が無い (G1) ので本 run では直していない．
- **N6-2**: 基底の具象 `iter_batches_df` (`hcpe_transform.py:86`) は
  `get_hcpe_polars_schema()` 決め打ちで，`preprocessing` 型のソースに
  対しては黙って誤動作する．現状 production の caller はすべて override
  側 (`FileDataSource` / `ObjectStorageDataSource`) を通るので dormant．
  「HCPE 専用と明記して名前を変える」か「array_type で分岐させる」かの
  判断が要る．

## Reconciliation (6d)

触れた項目 + 新規発見 = **20** (backlog 18 行 + 新規 2)

- **resolved** (fix マージ済): **5** — D8+D9 / N3 / N-1 / N5-1 は
  行を削除．**D14(a) は行を残した** — 同じ行の (b) が未着手のため，
  行を (b) の記述に縮めた
- **in flight**: **0**
- **re-triaged** (行保持・文面を鋭くした): **3** — D10+D11 / D13 / N4
- **そのまま保持** (再検証で confirmed，文面変更なし): **10** —
  app/learning Deferred 2-7 (6) / D5 / D15 / O5 / O9
- **new row**: **2** — N6-1 / N6-2
- **not a finding**: **0**

5 + 0 + 3 + 10 + 2 = **20** ✓

backlog 行数: **18 → 16** (18 − 4 + 2)．**消化は 5 件・削除は 4 行**で
数が食い違うのは，D14 の 1 行が (a) と (b) の 2 つの finding を抱えて
いたため (§ Consumed)．

`coverage.md` 本表の `Open items`: `src/maou/infra/file_system` を
`3 deferred` → `2 deferred` (D8+D9 の行を削除)．
`src/maou/app/learning` は `6 deferred` のまま．

## Environment notes

- **G3 は発生しなかった**: `uv sync --extra cpu` が通り (~6.7 分)，
  ruff / mypy / pytest / cargo test / cargo clippy / cargo fmt を
  すべてこの環境で実行できた．全件 **1905 passed, 54 skipped** +
  Rust 38 passed + doc-tests 3 passed．
- **N4 の実害を 3 run 連続で確認**: コンテナ初期状態の `.venv` は空で，
  `--extra cpu` を入れるまで `polars` すら import できない．base extra
  だけで回すと `infra/file_system` のテストが丸ごと skip され，この run
  の変更 (5 件中 3 件がこのパッケージに触る) が無検証のまま緑に見える
  状態だった．
- **Rust の再ビルドは安い**: `target/` が温まっていれば
  `cargo build -p maou_io` が 20 秒，`maturin develop --uv` が 15 秒．
  Python から Rust 側の修正を検証するには maturin の再ビルドが要る
  (非空虚性の確認でも 2 往復した)．
- **G1**: GPU / BigQuery / S3 / GCS は無い．app/learning の Deferred
  5/6/7，O9，新規の N6-1 はこの環境では確認できない．
