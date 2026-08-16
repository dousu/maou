---
kind: backlog
date: 2026-08-16
path:
  - src/maou/app/learning/dataset.py
  - src/maou/app/learning/polars_datasource.py
  - src/maou/infra/file_system/file_data_source.py
scope: python
level: medium
last_sha: f617055
record_sha: 654986c
---

# backlog consumption — 学習 ABC への列アクセサ追加 (D13 の完了)

`/audit-backlog` (2026-08-16, `bg1urj`)．前 run
([2026-08-15 gns-sync-and-batch-api](2026-08-15-backlog-gns-sync-and-batch-api.md))
が deferred backlog を空にして残した **out-of-scope 2 行**で始まった run
である．

**開始時点で G4 ゼロ・未回答の設計判断ゼロ**という状態は 2 run 連続で，
2 行とも「決定済み，あとは作業」だった．本 run はそのうち **D13 を全消化**
した — 4 run にわたり「次 run の先頭候補」と名指しされ続けた行である．

**本 run の実質は「設計判断を行に書いておくと，後の run が通常作業として
拾える」の 3 例目**であり，かつ**その仕組みが最後まで走り切った初めての
例**である．D13 は 2026-08-14 に「`ColumnarBatch` を直接スライスする」，
2026-08-15 に「writeability は `_explode_list_column` で保証する」，
同 8-15 に「口は ABC に列アクセサを追加する」と **3 段階の設計判断**を
積んでおり，本 run は 1 問も設計を問うことなく実装だけを行っている．

指定ブランチ 1 本の制約でクラス毎の PR 分割ができないため，**レビュー
単位は commit が担う**．collapsed run なので台帳の行削除は**同じ PR の
中**に入れてある (`.claude/commands/audit-backlog.md` 6a の separability
test — 削除と修正が分岐し得ないため)．

## Classification

2 行を再検証して **stale 0 / changed shape 0 / confirmed 2**．
**行番号の移動はゼロ** (前 run 以降 `src/` へのコミットが無かったため —
3 run 連続)．**自動帯は空** (17 run 連続)．

| ID | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|
| D13 | **P4** | ABC への**追加**メソッドで既定実装を持つ．公開名の削除も再署名も無いので P6 ではない．dtype も `.feather` の版も不変なので P5 ではない．返るテンソルが `ColumnarBatch` とストレージを共有するようになる (従来はサンプル毎のコピー) ので**挙動は変わり**，P3 ではない | **G2** — 行の対象は `infra/file_system` だが，修正は `app/learning/dataset.py` と ABC に触る |
| O9 | **P4** | 同じ `page_num` を 2 回引いて返る行が変わる．既存データは読めるし既存の起動も有効 | **G1 (縮小形)** — 再現性は fake client で CI に載るが，実クエリの最終確認には BigQuery 実環境が要る |

### G2 の解し方

D13 の G2 は 4 run にわたり「作業量が枠に入らない」と読まれてきたが，
`.claude/commands/audit-backlog.md` step 4a の 2 つの解のうち
**「隣を取り込んで判断帯の PR にする」**を選んだ．取り込んだのは
`app/learning/dataset.py` (ABC と 2 つの消費側) で，結合は
「ABC に口が無い限り infra 側だけでは何もできない」という構造的なもの
であり，見積りではなく設計から決まる．拡大そのものはユーザに問うている
(§ Decisions asked Q1)．

## Consumed

| 行 | 由来の記録 | 対象 | 出荷したもの | commit |
|---|---|---|---|---|
| **D13** | [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) | `infra/file_system` + `app/learning` | 学習 `DataSource` ABC への列アクセサ，`FileDataSource` の実装，`KifDataset` / `_StageDataset` の利用 | `7c0157b` |

### D13 の残る作業がどう閉じたか

行が挙げていた残作業は (2)(3)(4) の 3 つだった．

- **(2) `LearningDataSource` への列アクセサ追加と `KifDataset` 側の
  スライス経路の新設** — 出荷した．ただし**口を生やした場所は
  `LearningDataSource` ではなく `DataSource`** である．行の記述が
  2 つの名前を混同していた: `LearningDataSource` (`app/learning/dl.py:71`)
  は `DataSourceSpliter` を足すだけの中間抽象で，`__getitem__` /
  `__len__` を宣言しているのは `DataSource`
  (`app/learning/dataset.py:45`) の方である．消費側の `KifDataset` は
  `DataSource` で型付けされているので，`LearningDataSource` に置くと
  **消費側から型安全に呼べない**．`LearningDataSource` は
  `DataSource` を継承しているので，決定 (「ABC に列アクセサを追加する」)
  の実質は満たされている．
- **(3) `FileManager.get_item` の縮退** — `get_items` は前 run で削除
  済み．`get_item` (と `_columnar_to_structured_record`) は**残すのが
  正しい終点**であって先送りではない: `__getitem__` は ABC の契約なので
  `FileDataSource` は実装し続ける必要があり，columnar 分岐を落とすと
  columnar ソースの `__getitem__` が壊れる．本 run 後，この経路は
  **production から到達しなくなった** (`dataset.py` の 2 箇所が
  `columnar_record` を通るため) — hcpe 経路とテストだけが通る．
- **(4) hcpe 経路との分岐整理** — `columnar_record` が
  `_use_columnar` False で `None` を返し，呼び出し側が
  `__getitem__` へ落ちる形にした．hcpe は
  `cached_array[local_idx]` が既にビューなので，辞書を組み直す意味が
  無いことを docstring に明記してある．

`assert self._structured_dtype is not None` が `_use_columnar` False の
経路が残る限り落とせない点は不変で，これは所見ではなく注記である．

## Applied

### (1) 列アクセサの追加 (`7c0157b`，D13)

`src/maou/app/learning/dataset.py`:

- `DataSource.columnar_record(idx) -> dict[str, np.ndarray] | None`
  (`:68`) — 既定 `None`．**抽象メソッドにはしていない**: すると
  `BigQueryDataSource` / `ObjectStorageDataSource` とテストの fake 7 本
  が全て構築不能になる．戻り値の契約 (キー集合・shape・writeable・
  ビューでよいこと) を docstring に書いた．
- `_columnar_capable()` (`:108`) — 契約の実装有無を**構築時に一度だけ**
  ABC への `isinstance` で判定する．`hasattr` を使わないのは，口の所在を
  ABC 1 箇所に固定するためである (ユーザが 2026-08-15 に (b) duck-type
  検出を却下した理由そのもの)．
- `_record_fields()` (`:138`) — 速い口があればその辞書を，無ければ
  structured array を返す．**どちらも `obj[name]` でフィールドが取れる**
  ので消費側は 1 本のコードで両方を読める．名前集合だけ取り方が違う
  (辞書のキー / `dtype.names`) ので別に返す．
- `KifDataset.__init__` / `__getitem__`，`_StageDataset.__init__` /
  `__getitem__` が上記を使う．
- `_structured_field_to_tensor` が `KeyError` も `ValueError` に揃える
  (辞書は欠けたキーで `KeyError`，structured array は `ValueError`)．

`src/maou/infra/file_system/file_data_source.py`:

- `FileManager.columnar_record()` (`:367`) / `_columnar_batch_to_record()`
  (`:396`) — 後者は `_columnar_batch_to_structured_array` と**同じ
  dtype 駆動のループ**で，違いは書き込み先だけ (`np.empty` への memcpy
  か，ビューを辞書に入れるか)．
- `FileDataSource.columnar_record()` (`:672`) — `self.indicies` の写像を
  `__getitem__` と同じに適用し，範囲外は同じく `IndexError`．

### 2 つの罠

**キー集合を dtype 側に固定した理由**．素直に書くと「batch が実際に
持っている列」で辞書を組みたくなるが，`KifDataset` は列の有無を
`dtype.names` で判定して**教師の要素数を決めている**．preprocessing の
structured dtype は `moveWinRate` を無条件に含み，`moveWinRate` 列を
持たない旧 `.feather` では structured 経路がゼロ埋めして 3 要素を返す．
batch 駆動で組むと速い口だけが 2 要素になり，**同じデータで教師の形が
経路によって変わる**．dtype 駆動にして，供給されない列は structured
経路と同じくゼロで埋めた．

**1 次元列の shape**．`source[row]` は `(N,)` の列に対して numpy スカラー
を返すので，`resultValue` が 0 次元配列でなくなる．`.item()` は
どちらでも通るため**値の比較では検出できない**．長さ 1 のスライスを
取ってから先頭軸を落とす形にして，ビューのまま structured array と
shape を揃えた．

### (2) `PolarsDataFrameSource` の削除 (`654986c`，Q2 の実装)

**P6** (公開名の削除)．`src/` を消すので版を上げる — 直前の同種の削除
(`380866c` の `get_items`) にならい `feat!:` + **minor** bump とした
(1.0 以前なので破壊的変更の置き場は minor である)．

- `src/maou/app/learning/polars_datasource.py` を削除 (365 行)．
- `tests/maou/app/learning/test_polars_datasource.py` を削除 (232 行)．
- `docs/rust-backend.md` の
  「PyTorch Dataset with Polars DataFrames (Phase 5)」節を削除 (61 行)．
- 不在を固定する回帰テストを 1 本追加 (`get_items` の
  `test_batch_retrieval_api_stays_removed` と同じ形)．

**呼び出し側がゼロであることの再確認**: `src/` 全体でのヒットは定義
ファイル自身のみ．`benchmark_polars_io.py:386` の `polars_datasource` は
**ローカル変数名**であって `FileDataSource` のインスタンスである
(同名の別物 — grep だけで判断すると誤る)．

## Decisions asked

`AskUserQuestion` は **受理 1 問 + 設計判断 1 問**．**開始時点では設計判断が
ゼロ** (G4 の行が無い) だったが，D13 の実装中に**新しい未定点が 1 つ表面化**
したのでそちらに 1 枠を充てた．2 件とも回答を得ている．

| # | 種類 | 問うたこと | 選択肢 | ユーザの回答 |
|---|---|---|---|---|
| Q1 | 受理 | D13 の修正 (PR #508) を受け入れるか | (a) マージする **(推奨)** / (b) 現状維持 (PR を閉じる) | **(a) マージする** |
| Q2 | 設計判断 | `PolarsDataFrameSource` の扱い | (a) 削除する **(推奨)** / (b) `DataSource` を実装させる / (c) 現状のまま残す | **(a) 削除する** |

**Q1 が受理を問う形になった理由**．`.claude/commands/audit-backlog.md` の
split test の**両方の半分が成立しなかった** — 口の設計は 2026-08-15 に
決まっており，実装の分岐は残っていない．したがって「書いてから受理を問う」
側が正しい．

**Q2 が設計判断になった理由**．`PolarsDataFrameSource` は
`_PolarsRow` / `_PolarsField` / `_FakeDtype` の 3 つの shim で structured
array のフィールドアクセスを**模して**おり，ABC を継承していない．本 run が
学習 ABC に列アクセサを足したことで，このクラスが `KifDataset` 側に
`isinstance` ガードを要求する**唯一の理由**になった — 「消すか，契約を
実装させるか，模造のまま残すか」は 3 つの異なる製品であり，どれを選ぶかは
利用者のものである．

**Q2 が governs する範囲**: この 1 つの回答で，`polars_datasource.py` の
削除と，`docs/rust-backend.md` の Phase 5 節の去就と，`isinstance` ガードを
将来落とせるかどうかが同時に決まる．

**Q2 は同 run 内で実装した** (`654986c`) — 削除が 365 + 232 行 + doc 61 行で
contained だったため．§ Applied (2) を参照．

**予算に入らなかった設計判断は無い** (4 問枠に対し 2 問で足りた)．

## In flight

**なし**．3d の 2 問とも**同一セッション内でユーザが回答した**ので，
判断帯の PR を開いたまま引き継ぐ状態は発生していない．PR #508 は
Q1 の回答 (マージする) を受けて `main` へマージした．

なお D13 の行は**この PR の中で削除してある** (collapsed run なので
削除と修正が分岐し得ない — 6a の separability test)．PR ごと閉じれば
行も戻る構成だった．

## Re-triaged

**O9** (`src/maou/infra/bigquery`) — 行は残す．文言を鋭くした点:

- **再検証で行番号の移動はゼロ** (`__get_total_rows` `:212`，
  `__fetch_from_bigquery` の def `:344`，ページ側の再サンプル
  `:395-404`，`tablesample_clause` `:380-382` と差し込み `:384-388`，
  `get_page` `:485`)．欠陥の構造も不変．
- **設計・テスト土台・並び順とも決定済みで人間待ちではない**．前 run が
  最後の未定点 (iv) を潰しているので，未定点は**ゼロ**である．
- 着手しなかったのは **(0) fake BigQuery client のテスト土台の新設**が
  D13 の枠と同居できなかったからで，それだけの理由である．G1 は縮小形の
  まま (fake で再現性は CI に載るが，実クエリの最終確認には実環境が要る)．

この行は**「決定済み，あとは作業」の在庫が 1 件だけ残っている**状態で
あり，次 run の唯一の候補である．

## Corrections to the source records

**なし**．D13 の記録の診断は正しかった．行の記述に 1 点の不正確さ
(口を生やす先を `LearningDataSource` と書いていたが，消費側が型付け
されているのは `DataSource`) があったが，これは**記録ではなく backlog 行
の側**の記述であり，行はこの run で削除されるので訂正の置き場が無い．
本記録の § Consumed に理由ごと残した．

## Doc findings

**D13 (1) の側は drift なし**．`docs/` を横断して確認したが，学習
`DataSource` の抽象メソッド集合を列挙している durable doc は存在しない．
`docs/rust-backend.md:727-729` が触れているのは `iter_batches()` (本 run で
不変)，`docs/adr-003-training-performance-optimization-attempts.md` は
バッチ取得 API の**過去形の記録**なので編集しない (`get_items` の削除は
前 run で済んでおり，ADR の記述を偽にしていない)．

**Q2 の実装で 1 件発生**:
[reviews/2026-08-16-rust-backend-polars-dataset-section.md](../reviews/2026-08-16-rust-backend-polars-dataset-section.md)
— `docs/rust-backend.md` の Phase 5 節の削除．**status: applied**
(`applied_in: 654986c`)．**P2 の standing approval で適用した** — 節の構成が
丸ごと `PolarsDataFrameSource` に従属しており，クラスが無くなった以上
「節を残す書き方」が存在しないので，訂正後の本文 (= 削除) は一意に決まる．

**置き換えの案内文は書いていない**．「Polars DataFrame を学習に載せる
正しい入口は `FileDataSource` である」といった文はあれば有用だが，
どこに何行でどの経路を推して書くかに複数の書き方があり**一意には
決まらない** — それは新しい指針であって drift correction ではないので，
P2 の 2 つ目のテストに落ちる．必要なら別提案とする．

## Out of scope

**新規所見 N11 を 1 件起票した** (`coverage.md` の out-of-scope backlog)．

**N11** — `src/maou/domain/data/polars_tensor.py` の
`polars_row_to_hcpe_arrays` (モジュール全体で 40 行) が **Q2 の削除で
到達不能になった**．削除前も唯一の呼び出し側が `polars_datasource.py`
だったため，`src/` `tests/` ともに呼び出し側はゼロである．
`src/maou/app/common/data_io_service.py:35` に
"Direct integration with PyTorch via polars_tensor module" という
**docstring の言及だけ**が残り，指す先が無い記述になっている．

**同じ run で黙って広げなかった理由**は step 4a の scope 境界である．
ユーザが 3d で判断したのは `PolarsDataFrameSource` の扱いであって当該
モジュールではない．3d は 1 run に 1 回しか問わないので，2 度目を
問う代わりに**行として起票して次 run に渡す**方を選んだ (step 4a の
2 つの解のうち「re-triage」側)．向きは (a) モジュールごと削除 /
(b) 残す の 2 つで，**P6 + G4**．削除するなら 40 行 + docstring 1 行で
contained である．

## Environment notes

- `uv sync --extra cpu` で torch (2.11.0+cpu) を導入．既定の
  `UV_HTTP_TIMEOUT=30` では依存取得がタイムアウトするので
  `UV_HTTP_TIMEOUT=300` を付けた (前 run と同じ)．
- **G3 は発生していない** — QA は全て実行できた．
  `ruff format` / `ruff check` / `mypy` (135 files, no issues) /
  `pytest` (**2040 passed, 53 skipped**)．
- `gradio` 未導入のため `tests/maou/infra/visualization/test_indexing_status.py`
  の 1 モジュールが収集されない．本 run の変更とは無関係
  (`visualize` 系は `DataSource` を通らない)．
- 無効化テストは **9 通り**行い，全て意図した本数だけ落ちることを確認:
  速い口の無効化 (8 本落ちる) / キー集合を batch 駆動へ (6) /
  1 次元列をスカラーで返す (3) / 負インデックスの正規化を外す (1) /
  ビューをコピーへ (2) / `indicies` の写像を無視 (1) /
  `_columnar_capable` を無条件に (1) / `dtype.names` ガードを外す (1) /
  `_StageDataset` を遅い経路のまま (1)．
  最後の 1 つは**最初は空虚だった** — 値の一致だけを見ていたので
  `_StageDataset` が遅い経路に留まっても通っていた．
  `np.shares_memory` による経路の直接観測を足して落ちるようにした．
- **無効化テストの手順そのものに 1 度失敗した**．最初の周回で各無効化の
  後始末に `git checkout <file>` を使ってしまい，**まだ commit していない
  修正ごと HEAD に戻していた**ので，2 周目以降は「無効化を当てた状態」では
  なく「修正が無い状態」を測っていた (落ちた本数は多かったので気付きにくい)．
  修正済みファイルを別途退避してから復元する形に改め，9 通りすべてを
  測り直している．本記録の本数はやり直した後のものである．
- git の pre-commit hook がコンテナに入っていなかったので
  `uv run pre-commit run --from-ref f617055 --to-ref HEAD` で明示実行し，
  全 hook Passed (trim / end-of-files / toml / large-files / uv-lock /
  test / mypy / check-cli-docs / ruff-check / ruff-format)．

## Reconciliation (6d)

本 run が触れた項目 + 新規所見 = **4**

| 内訳 | 数 | 中身 |
|---|---|---|
| resolved (行を削除，修正と同じ PR) | 1 | D13 |
| in flight | 0 | — |
| decided (行に決定を書き G4 retire，未実装) | 0 | — |
| re-triaged (行を残し文言を鋭く) | 1 | O9 |
| new row | 1 | N11 (`polars_tensor.py` の到達不能化) |
| not a finding | 1 | Q2 の `PolarsDataFrameSource` — **同 run 内で決定と実装の両方が済んだ**ので行にならない |

`2 = 1 (resolved) + 1 (re-triaged)`，これに新規 1 行と行にならなかった
1 件が加わる．**backlog 行は 2 → 2** (D13 が消え N11 が入った) で
数の上では動いていないが，**動いたのは行数ではなく完了した作業である** —
4 run にわたり先頭候補だった行が消え，代わりに入ったのは 40 行の
contained な残骸である．

## 追記 — ユーザの回答 (3d)

**Q1 「マージする」** — PR #508 を `main` へマージした．本 run の
`main` への マージはこの 1 回だけである．

**Q2 「削除する」** — `PolarsDataFrameSource` を削除した (`654986c`)．
設計判断を得た同じ run 内で実装まで到達しているので，行は起こしていない．
却下されたのは (b)「`DataSource` を実装させる」(`__getitem__` が
`np.ndarray` でなく `_PolarsRow` を返すため契約に合わせるには戻り値の
作り直しか `type: ignore` が要り，shim 3 つはそのまま残る) と
(c)「現状のまま残す」(`isinstance` ガードが恒久的に必要になる)．

**この run が `isinstance` ガードを残した理由**は不変である — Q2 は
`PolarsDataFrameSource` を消したが，`tests/maou/app/learning/test_stage_datasets.py:17`
の `MockDataSource` も ABC を継承しない duck-typed なソースであり，
`KifDataset` / `_StageDataset` に渡されている．ガードを落とせるのは
duck-typed な呼び出し側がゼロになったときである．
