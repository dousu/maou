---
kind: backlog
date: 2026-08-15
level: medium
path:
  - src/maou/infra/file_system
  - audits
last_sha: a552821
---

# `/audit-backlog` 2026-08-15 — 未初期化メモリの出荷と，決定済み設計が実装できないと判った 1 件

**backlog の 5 行すべてが「決定済み，あとは作業」か「G1/G2/G3 が塞いで
いる」かのどちらかで，G4 は 1 つも無い状態で始まった 2 回目の run**
である．先頭候補は前 run が名指ししていた **D13**．

結果として本 run が出荷したのは backlog 行ではなく，**D13 の再検証中に
発見した新規所見 N10** — `np.empty` の未初期化メモリが訓練ターゲットと
して torch に渡っていた，という所見 — である．そして D13 自身は
**"changed shape"**: 決定済みの設計が，記録の処方のままでは**本番データで
確実に落ちる**ことが判り，新しい設計判断が要ることが判明した．

「記録は**どこを見るか**の手がかりであって，**何をするか**の指示では
ない」という原則が，本 run では 2 度効いた — 1 度は D13 の処方に対して，
1 度は Deferred 2 の 2026-08-13 の訂正に対して．

## Classification

自動帯 (P1-P3) は **空** — 13 run 連続．G4 が付いた行は開始時点で
**ゼロ**だったが，D13 の再検証で**実質的に 1 つ再発した**．

| ID | backlog 行 | 対象 | 再検証 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|---|---|
| N10 | *(新規)* | `infra/file_system` | 新規 (実証済み) | **P4** | 旧データで返る値が未初期化メモリ→定義値 (ゼロ) に変わる．既存の `.feather` は読めるまま，既存の呼び出しも有効なので P5/P6 ではない．「返る値が変わる」ので P3 でもない | なし |
| B-1 | D13 | `infra/file_system` + `app/learning` | **changed shape** | **P4/P6** | 分岐前なので確定しない | **G2** + **新しい設計判断 (G4 相当)** |
| B-2 | Deferred 2 | `app/learning` | **changed shape** (前提の訂正) | **P4** | 挙動は変わりうるがデータも呼び出しも壊れない | **G3** |
| B-3 | Deferred 5 | `app/learning` | confirmed (行 +7) | **P4** | 同上 | **G1** |
| B-4 | Deferred 7 | `app/learning` | confirmed (行番号一致) | **P4** | 同上 | **G1** |
| B-5 | O9 | `infra/bigquery` | confirmed (行 −1〜−4) | **P4** | 同じ `page_num` で返る行が変わる | **G1** |

再検証: **confirmed 3 / changed shape 2 / stale 0**，＋新規 1．

### N10 のクラス判定を P3 にしなかった理由

P3 の試験は「プログラムが既に受け付ける**あらゆる**入力について，
書き出す成果物と返す値が変わらない」である．`moveWinRate` 列を持つ
現行データについては何も変わらない (下記の characterization test が
その根拠) が，**列を持たない旧データについては返る値が変わる** —
未初期化メモリからゼロへ．変更前の値が「不定」であっても，それは
「変わらない」ことの証明にはならない．fail-safe は上向きなので **P4**．

## Consumed

| 行 | 由来 | 対象 | 出荷したもの | commit |
|---|---|---|---|---|
| *(新規 N10)* | 本 run | `src/maou/infra/file_system` | `ColumnarBatch` が供給しない列のゼロ埋め + フィールド写しの dtype 駆動一本化 | `033d49f` |

backlog 行は **1 行も削除していない** (5 行すべてが残る)．N10 は本 run
内で起票・消化したので行を作っていない．

## Applied

### N10 — `np.empty` の未初期化メモリが訓練ターゲットへ漏れていた

`src/maou/infra/file_system/file_data_source.py:453` (`np.empty(n, dtype)`)
とそれに続くフィールドごとの条件付き代入．

**機構**: `_columnar_batch_to_structured_array` は structured array を
`np.empty` で確保したうえで，各フィールドを
`if batch.<field> is not None and "<name>" in dtype_names:` の形で
**条件付きに**書いていた．一方 `get_preprocessing_dtype()`
(`domain/data/schema.py`) は **`moveWinRate` を無条件に含む**．そして
`convert_preprocessing_df_to_columnar` は
`if "moveWinRate" in df.columns` の場合だけ `move_win_rate` を設定し，
`ColumnarBatch` の docstring 自身が「**旧データでは `None`**」と
明言している．

したがって `moveWinRate` 列を持たない旧 `.feather` から作った
`ColumnarBatch` では，**`moveWinRate` フィールドが `np.empty` の
未初期化メモリのまま返る**．実測 (4 行 × 1496):

```
moveWinRate all-zero?  False
min/max/mean: nan nan nan
nan count: 2
```

**被害**: `KifDataset.__getitem__` (`app/learning/dataset.py:146-150`) は
`_has_move_win_rate` を **dtype 名から**判定するので `True` になり，
この未初期化メモリ (NaN を含む) を `move_win_rate_tensor` として
**訓練ターゲットのタプルに入れて**返す．`iter_batches` 経路も同じ
変換を通るので同様．

**修正**: 写しを dtype 駆動の単一ループに一本化し，batch が供給しない
列はゼロで埋める．

- `file_data_source.py` — フィールドごとの 8 ブロックを削除し，
  `for name in dtype_names:` の 1 ループへ．`source is None` の枝が
  ゼロ埋めで，`id` (ColumnarBatch に対応物が無い) も同じ枝を通る
  (従来どおりゼロ)．
- `_STRUCTURED_TO_COLUMNAR_FIELD` (module level) — dtype 名 →
  `ColumnarBatch` 属性名の明示表．名前の対応は**機械的に導けない**
  (`board_positions` を camelCase 化しても `boardIdPositions` には
  ならない) ため表で持つ．
- `_columnar_field` (staticmethod) — 表を引いて列を返す．無ければ
  `None`．

**なぜ `np.zeros` にしなかったか**: `np.zeros` の 1 語置換でも同じ
バグは消えるが，サンプル 1 件ごとに 9,081B の memset がホットパスに
乗る (D13 が消そうとしているコストと同じ場所)．供給されない列だけを
ゼロ埋めすれば，全列が揃う通常経路では追加コストがゼロになる．

### 回帰テスト (`tests/maou/infra/file_system/test_file_data_source.py`)

`TestColumnarFieldsAreNeverUninitialized` の 4 本．

| テスト | 何を固定するか |
|---|---|
| `test_missing_move_win_rate_is_zero_filled` | 旧データ (列なし) で `moveWinRate` が NaN を含まずゼロであること |
| `test_missing_field_is_zero_filled_for_single_record` | 1 件取得経路 (`row=idx`) でも同じであること |
| `test_id_is_still_zero_filled` | 表に載せていない `id` がゼロ埋め側へ落ちること |
| `test_every_columnar_field_is_mapped` | `ColumnarBatch` の全 dataclass フィールドが表に載っていること |

最後の 1 本が「同じ形で二度壊れない」性質を担う: 表を更新し忘れると
その列は「batch が供給しない列」として**黙ってゼロ埋め**され，訓練
ターゲットが静かに全ゼロになる — 未初期化メモリよりは安全だが依然
誤りで，かつ検知手段が他に無い．

**非空虚性の確認**: ゼロ埋めの枝を外して (`id` だけ従来どおりゼロに
する形に戻して) 実行し，`test_missing_move_win_rate_is_zero_filled` が
`moveWinRate に NaN が漏れている` で落ちることを確認した．未初期化
メモリの内容は本来不定なので，テストは NaN で埋めた同サイズの
バッファを確保してすぐ手放す (`_dirty_the_heap`) ことで，直後の
`np.empty` が同じ領域を拾うよう仕向けている．
`test_missing_field_is_zero_filled_for_single_record` は無効化しても
通った (1 行ぶんの確保では領域の再利用が起きなかった) ため，これは
副次的な固定であり，決定的なのは 1 本目である．

**「挙動不変」の根拠 (通常経路)**: 既存の
`test_single_record_matches_batch_conversion` が，1 件取得と一括変換が
全フィールドで一致することを修正前後どちらでも固定している
characterization test であり，ループへの一本化が全列の揃った経路を
変えていないことの根拠になっている．

## Decisions asked

`AskUserQuestion` を **1 回**上げた．内訳は**受理 1 問 + 設計判断 1 問**．

### Q1 — 受理 (PR #504)

> PR #504 (新規所見 N10) をマージしてよいですか？

選択肢: **「マージする (推奨)」** / 「ゼロ埋めではなく『列が無い』
として扱う」/ 「現状維持」．

第 2 案が却下されなかった場合に何が違うかを明記した: ゼロ埋めは
`KifDataset` に**全ゼロの `moveWinRate` を訓練ターゲットとして渡す**
ので 3 要素タプルのまま．「列が無いとして扱う」なら 2 要素タプルを
返すべきで，そのためには列の有無を datasource から `KifDataset` へ
伝える経路が要る (`app/learning` に波及するので G2)．**未初期化メモリ
よりはどちらもましである**という点では両案は一致しており，差は
「全ゼロを学習させてよいか」だけである．

### Q2 — 設計判断 (D13)

> `ColumnarBatch` の writeability をどこで確立するか

選択肢: **(a) `domain` の `_explode_list_column` が常に writeable を
返す (推奨)** / (b) `infra` の `FileManager` がロード後に read-only
フィールドだけコピーする / (c) `app` の `KifDataset` が read-only を
受け入れる / (d) D13 を落とす．

**この問いを立てた理由**: 2026-08-14 の設計判断「`KifDataset` が
`ColumnarBatch` を直接スライスする」は有効だが，**そのままでは本番
データで実装できない**ことが再検証で判った (下記 § Re-triaged)．
決定の下で実装が 3 案に割れており，差分が材料的に異なる — (a) は
`domain` の 1 関数，(b) は `infra` のロード経路とピークメモリ，
(c) は `app` の安全チェックの緩和 — ので，外すとレビューごと捨てる
ことになる．**書く前に問うた**．

**settles**: D13 行．答えが出れば G2 の作業量だけが残り，通常作業に
なる．

### 予算に入らなかった設計判断 (次 run の待ち行列)

本 run は 2 問しか使っていないので**待ち行列に積み残しは無い**．
残り 3 行 (Deferred 2 / 5 / 7 / O9) はいずれも設計判断待ちではなく，
環境ゲート (G1/G3) 待ちである．ただし 1 点だけ，次 run が問う価値の
ある論点がある:

1. **Deferred 2** — 「4 本目の軸」が誤りと判った結果，2026-08-14 の
   決定「`TrainingLoop` サブクラスは戦略として注入する」は**前提を
   失っている**．統合そのものの決定は有効なので**問い直しは必須では
   ない**が，「注入機構は不要でよいか」を 1 問使って確認する価値は
   ある．G3 (~585 行の等価性) は動かない．

## In flight

- **PR #504** (base: `main`) — 本 run の全内容．指定ブランチ 1 本の
  制約でクラス毎の分割ができないため，N10 の修正と `audits/` の
  台帳・記録が同じ PR に同居する．レビュー単位は commit が担う．
  待っているもの: Q1 の受理．

backlog 行を 1 つも削除していないため，6a の分離可能性の試験は
本 run では発火しない (削除が無いので分岐が無い)．

## Re-triaged

### D13 — 決定済みの設計が，記録の処方のままでは実装できない

これが本 run の主要な発見である．2026-08-14 にユーザが決めた
「`KifDataset` が `ColumnarBatch` を直接スライスする」は方向としては
有効だが，**そのまま書くと本番データで `ValueError` を投げる**．

**機構**: `_explode_list_column` (`domain/data/schema.py:782`) の
fast path は

```python
result = col.to_numpy().reshape(n, *shape)
if result.dtype != dtype:
    result = result.astype(dtype)
return result
```

で，**polars 側の dtype が目標 dtype と一致すると `astype` が
スキップされる**．polars の `to_numpy()` は Arrow バッファの
**read-only ビュー**を返すので，一致した列は `writeable=False` の
まま `ColumnarBatch` に入る．preprocessing の polars スキーマ
(`schema.py:457-475`) では:

| 列 | polars | 目標 | `astype` | writeable |
|---|---|---|---|---|
| `boardIdPositions` | `List(List(UInt8))` | `uint8` | **無し** | **False** |
| `piecesInHand` | `List(UInt8)` | `uint8` | **無し** | **False** |
| `moveWinRate` | `List(Float32)` | `float32` | **無し** | **False** |
| `moveLabel` | `List(Float32)` | `float16` | 有り | True |
| `resultValue` | `Float32` | `float16` | 有り | True |

そして `KifDataset._numpy_to_tensor` (`app/learning/dataset.py:232-238`)
は read-only を**設計として** `ValueError` で撥ねる
(「Ensure preprocessing files are opened via copy-on-write memory
mapping so tensors can share storage」)．

**なぜ今まで露見しなかったか**: 現行の経路は `np.empty` の structured
array に**全部コピーしてから**返すので，read-only 性はそこで消える．
直接スライスにした瞬間に初めて表面化する．

**なぜテストでは捕まらないか** — こちらの方が重要である．テストが
Python の list から組む DataFrame は int64/float64 になるため，
`astype` が必ず挟まって全フィールドがコピー (writeable) になる．
つまり**直接スライス経路はテストで全部通り，本番データでだけ落ちる**．
本 run で実測して確認した:

```
List(UInt8)   由来 uint8   → writeable=False
List(Float32) 由来 float32 → writeable=False
Python list   由来        → writeable=True   (astype が挟まる)
```

**残る作業**は前の記述 (`KifDataset` 側のスライス経路の新設，
`FileManager.get_item`/`get_items` の縮退，hcpe 経路との分岐整理) に
加えて，writeability を確立する一手．Q2 の回答がそれを決める．

なお `streaming_dataset.py` の `_yield_*` は
`columnar_batch.slice(batch_indices)` を通す — index 配列による
fancy indexing は必ずコピーを作るので **writeable であり，同じ罠は
踏んでいない**．

### Deferred 2 — 決定の前提だった「4 本目の軸」が存在しない

`training_loop.py:1183` は `Stage1TrainingLoop = RawLogitsTrainingLoop`
の**別名**で，`git log -S` によれば **2026-08-09 (`568863f`) から**
そうである．`multi_stage_training.py:428` と `:577` は同じクラスを
構築している．詳細と下流の影響は § Corrections を参照．

行番号も更新した (run 関数 `:376-523`/`:525-672`，工場の完全一致
`:704-731` ≡ `:794-821` を `diff` で再確認，`streaming_dataset.py` は
**−14** で `:834-891`/`:894-948`)．footprint は **~585 行**．
**G3 は不変**．

### Deferred 5 / Deferred 7 / O9 — 行番号と根拠の更新のみ

- **Deferred 5**: 全体 +7 (`:1117` が同期)．**2 つ目の同期
  `int((~has_legal).sum().item())` が `:1122` にある**ことを新たに記録．
  休眠の根拠が 1 つ強くなった — `RawLogitsTrainingLoop._compute_policy_loss`
  (`:1170`) はマスク処理を丸ごと迂回するので，Stage1/Stage2 からは
  **データ以前にクラスとして**到達不能である．**G1 不変**．
- **Deferred 7**: `gradient_noise_scale.py` 側は**移動なし**，消費側
  (`training_loop.py`) のみ +7 で `:1031-1038`．**G1 不変**．
- **O9**: 一律 −1〜−4．加えて**テスト網羅の実態を確定** —
  `sample_ratio` / `TABLESAMPLE` に触れるテストは `tests/` に**ゼロ**で，
  "fake" と呼ばれる 2 本は BigQuery クライアントを fake しておらず
  `pm.get_page` を lambda に差し替えて**この所見の対象経路を丸ごと
  潰している**．修正時には回帰テストの土台自体を作る必要がある．
  **G1 不変**．

## Corrections to the source records

- [`2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md`](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)
  — 同 run が Deferred 2 に追記した訂正 (i)「4 本目の**挙動**の軸として
  `TrainingLoop` サブクラスが違う」の撤回．`Stage1TrainingLoop` は
  `RawLogitsTrainingLoop` の別名で，しかもその別名は **2026-08-09 の
  `568863f` から存在していた**ので，この訂正は**書かれた時点で既に
  誤り**だった．

  記録した教訓は「2 つの構築箇所の**名前が違うことだけを確認し，
  その名前の定義を読まなかった**」こと．別名は定義側にしか現れない
  ので，利用箇所の grep では区別がつかない．**「2 つの名前が違う」は
  「2 つのクラスが違う」の証拠にならない**．

  下流の影響も記録した: 2026-08-14 の設計判断「`TrainingLoop`
  サブクラスは戦略として注入する」は，この誤った訂正を根拠に提示
  された選択肢なので，**存在しない差異のための設計**になっている．

`2026-08-10-src-maou-infra-file-system.md` (D13 の元記録) には訂正を
**追記していない** — D13 の診断 (per-sample の `np.empty` + memcpy) は
正しく，誤っていたのは**その後の run が決めた処方の実現可能性**で
あって元記録の診断ではないため．read-only の発見は `coverage.md` の
行と本記録に書いた．

## Doc findings

**なし**．N10 の修正は `_columnar_batch_to_structured_array` の内部
実装で，公開 CLI にも文書化された出力形式にも触れない．
`docs/commands/` の記述と矛盾する箇所は生じていないので
`reviews/*.md` の提案は起票していない．

## Out of scope

**なし** (新規に気づいた所見 N10 は本 run 内で消化したので backlog
行を起票していない)．

O9 のテスト網羅ゼロは**新規所見ではなく O9 行の一部**として書いた —
独立した行にすると，O9 の修正時に必ず一緒に扱う作業が別の行として
二重に管理されることになるため．

## Environment notes

- **torch はコンテナ再生成で消えていた**ので `uv sync --extra cpu` で
  再導入した (前 run と同じ手順)．本 run の QA には
  `app/learning/dataset.py` を import する経路が含まれるため必要．
- **`uv lock` が 1 度失敗した** — `maou[tensorrt-infer]` の
  `tensorrt-cu12-libs` 解決でエラーになったが，再実行で成功した
  (`Resolved 241 packages`)．`--offline` は不可 (キャッシュに無い)．
  恒常的な障害ではなくプロキシ越しの一時的なものと判断した．
- **G1/G3 は本 run では発火していない** — 出荷した N10 の QA は
  すべてこの環境で実行できた．G1 (GPU / BigQuery) と G3 (~585 行の
  等価性) は，着手しなかった 4 行に付いたままである．

## QA

| 対象 | 実行 | 結果 |
|---|---|---|
| `ruff format src/ tests/` | 実行 | 2 files reformatted |
| `ruff check src/ tests/ --fix` | 実行 | All checks passed |
| `mypy src/` | 実行 | Success: no issues found in 135 source files |
| `pytest tests/maou/infra/file_system/` | 実行 | 31 passed |
| `pytest` (file_system + domain/data + app/learning/test_dataset + bigquery contract) | 実行 | **309 passed, 1 skipped** |
| 無効化テスト | 実行 | ゼロ埋めを外すと `test_missing_move_win_rate_is_zero_filled` が NaN 検出で落ちる |
| pre-commit | commit 時に実行 | pass (`--no-verify` は不使用) |

## Reconciliation (6d)

触れた項目 5 + 新規所見 1 = **6**．

```
6 = 1 resolved + 0 in flight + 0 decided + 5 re-triaged + 0 new rows + 0 not-a-finding
```

- **resolved (1)**: N10 — 本 run 内で起票・修正・PR 化．行を作って
  いないので削除する行も無い．6a の分離可能性の試験では「修正と
  同じ PR」に当たる (指定ブランチ 1 本なので必然)．
- **decided (0)**: Q2 (D13 の writeability) は**問うたが，本記録を
  書いている時点で回答が確定していない**ため decided に数えていない．
  回答が得られれば D13 行に決定を書き，実質的な G4 を retire する．
- **re-triaged (5)**: D13 / Deferred 2 / Deferred 5 / Deferred 7 / O9．
  うち **2 件 (D13, Deferred 2) は "changed shape"** で，文言を鋭くする
  だけでなく**前提そのものを訂正**している．

backlog 行数: **5 → 5** (削除なし，追加なし)．

数だけ見ると本 run は台帳を動かしていない．動いたのは行の**中身**で
あり，具体的には (i) D13 が「あとは作業」から「もう一度決めが要る」へ
**後退**したこと，(ii) Deferred 2 の決定が前提を失っていると判ったこと，
(iii) O9 の修正に回帰テストの土台作りが含まれると確定したこと，
の 3 点である．いずれも**着手してから判ったのでは遅い**種類の情報で，
再検証がその段で捕まえたことがこの run の産出物にあたる．

## Version bump

`pyproject.toml`: **0.92.0 → 0.92.1** (patch — `fix:`)．`uv.lock` の
`maou` エントリも同時に更新．

Rust crate には触れていないので `rust/*/Cargo.toml` の bump は無し．
