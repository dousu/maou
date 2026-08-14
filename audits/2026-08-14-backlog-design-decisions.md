---
kind: backlog
date: 2026-08-14
path:
  - tests/conftest.py
  - tests/test_conftest_optional_deps.py
  - docs/testing-guide.md
scope: python (tests) + docs
level: medium
last_sha: 2312f65
---

# `/audit-backlog` — 設計判断 4 件をユーザに問い，N4 を消化

`audits/coverage.md` の backlog 13 行 (Deferred 7 + Out-of-scope 6) を
全て HEAD (`2312f65`) に対して再検証した．**自動帯は空**で，13 行すべてが
P4 以上かつ全行にゲート付きだった．

この run は `.claude/commands/audit-backlog.md` の**改訂後 (`88a3425`,
PR #499) の最初の run** である．改訂で入った step 3d — 設計判断そのものを
ユーザに問う枠 — を初めて行使し，**4 問すべてを設計判断に充てた**．
4 件とも回答を得て，うち 1 件 (N4) はこの run 内で実装まで到達した．

前 7 run (2026-08-09 〜 2026-08-14) が「全 run 緑・自動帯は縮小・G4 行は
不変」を繰り返していた状態を，これが崩している．

## 再検証 (step 2)

13 行すべてを開いた．**stale 0 / changed shape 1 / confirmed 12**．

前 run (`001d16e`) 以降に変わったのは `pre_process.py` とテストだけ
(前 run 自身の出荷) なので，行番号のずれは小さい．

**changed shape 1 件**:

- **Deferred 3** — streaming アダプタ対は「redundant な `hasattr` ガード
  1 つの差」ではない．`Stage1StreamingAdapter.set_epoch` は
  `hasattr(self._dataset, "set_epoch")` で守る (`:756-758`) が
  `Stage2StreamingAdapter` は無防備に呼ぶ (`:718`)．**挙動差**なので，
  統合は「ガードを消す」か「両方に付ける」かの選択を含む．メソッド順序と
  docstring の句読点も違う (前 run が既に指摘済み)．

**確認できた主な行番号の更新** (いずれもコードが動いた結果であり，
記録の診断が誤っていたわけではない):

| 行 | 更新 |
|---|---|
| Deferred 2 | 工場の完全一致末尾は **30 行** (`:703-732`/`:793-822`)．行が言う 28 行は過少．違うログ文字列は **4 本** (2 本ではない)．`dataset.py:242-316`/`:319-391` は関数ではなく**クラス** `Stage1Dataset`/`Stage2Dataset` |
| Deferred 5 | コードが dormant であることを**自ら文書化**するようになった (`training_loop.py:497-503`)．マスク機構を将来経路のために残す旨が明記されている |
| Deferred 6 | `_record_stream` ヘルパは `:752-780` (行の `:750-780` は 2 行ずれ) |
| Deferred 7 | `.item()` は `:150` / `:189` / `:192` / `:247` |
| D5 | `_warn_if_oom_risk` は `_concatenate_numpy` では先頭文 (`:336`) だが `_concatenate_columnar` では内包表記の後 (`:379-381`)．「各 concatenate の先頭」は numpy にしか当たらない |
| D10+D11 | `total_pages()` は `:775-777`．`StreamingHcpeDataSource` の構築は `console/pre_process.py:567` (行の `:494` から移動)，その `total_pages()` は `:151-153` |
| D14(b) | `benchmark_polars_io.py` の正確な位置は **`src/maou/infra/utility/`** (出荷物)．`_use_columnar` 8 箇所のうち preprocess の役割は **`:539` だけ**，`:456` は learn 側，残る 6 箇所は `FileManager` 内部の共用 |
| O5 | S3/GCS の elif は `:492`/`:524`．`describe_missing_input_options()` は `:27-88`，呼び出し `:571` |
| O9 | `get_page` は `:486-533` |
| N6-2 | `ObjectStorageDataSource` の override は `:480` (行の `:475` から移動)．テストの範囲は `:77-100` |
| N4 | `_OPTIONAL_DEPS` は `:30-32`，`pytest_make_collect_report` は `:97-121`，サマリは `:193-204` |

## Classification (step 3a)

**自動帯 (P1-P3) は空**．13 行すべてが P4 以上で，全行にゲートが付いた．

| ID | 行 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|
| B1 | O5 (a)+(d)+ノブ削除 | P6 | CLI オプションの削除/再意味付け | G4 |
| B2 | D5 `cache_mode` | P6 | 公開ノブの削除 | G4 |
| B3 | D14(b) 二重 ABC | P6 | 公開の継承関係が消える | G2, G4 |
| B4 | N6-2 基底 `iter_batches_df` | P6 (改名) / P4 (分岐) | 決めがクラスを決める | G4 |
| B5 | D13 columnar スライス | P4/P6 | 決め次第で公開名が消える | G2, G4 |
| B6 | O9 BigQuery サンプリング | P4 | 同じ呼び出しで返る行が変わる | G4, G1 |
| B7 | N4 torch 依存テスト | P4 | どのテストが走るかが変わる | G4 |
| B8 | D10+D11 (1) `total_pages()` | P4 | 戻り値が変わる | G4 |
| B9 | Deferred 2 Stage1/2 統合 | P4 | ~730 行の挙動保存リファクタ | G3, G4 |
| B10 | Deferred 3 アダプタ 6→3 | P4 | 公開名 6 つ + テスト書き換え | G2 |
| B11 | Deferred 7 GNS 同期 | P4 | 数値等価性の確認が要る | G1 |
| B12 | Deferred 6 `stream.synchronize()` | P4 | GPU 意味論の変更 | G1 |
| B13 | Deferred 5 per-batch 同期 | P4 | dormant，実マスク経路とセット | G1 |

### B10 を P3 に落とさなかった理由

`_build_model` の `else: raise TypeError` (`stage_component_factory.py:880-882`)
は 2 run 続けて**到達不能**と確認されており，「実行され得ない分岐の削除」
として P3 に見える．しかし単独で消すと最後の枝が `else` になり，第 3 の
head 型に対する挙動が `TypeError` から「黙って Stage 2 扱い」に変わる
(消さないと mypy の possibly-unbound に触れるため，枝を残す選択肢が無い)．

fail-safe は上向きなので **P4** に置き，アダプタ統合の決めと一体で扱う．

## Decisions asked (step 3d)

`AskUserQuestion` を 1 回，**4 問すべてを設計判断 (設計判断) に充てた**．
受理エントリと向きエントリは無い — 自動帯が空で，かつ判断帯 13 件は
いずれも向きが決まるまで diff が書けなかったため．

### Q1 — `cache_mode` ノブの去就 (設計判断)

**settles: D5 全体 + O5(d)**

提示した選択肢:

1. **ノブごと削除 (推奨)** — 常に `"file"` 相当にする．2 倍ピークの
   footgun が消え，`learn-model` に `--input-cache-mode` が無い非対称も
   「公開するノブが無い」ので自動的に解消
2. 残して `learn-model` にも公開 — 層をまたいで揃えるが 2 倍ピークは残る
3. 残すが `memory` を遅延結合に — 2 倍ピークだけ消す
4. 現状維持

**ユーザの回答: (1) ノブごと削除．**

根拠として提示したのは「`memory` モードの唯一の利点である『1 つの配列と
して渡せる』を必要とする caller が現状ゼロ」という再検証結果．

**この run では実装していない** (予算外の規模: `file_data_source.py` +
`console/utility.py` 2 コマンド + `learn_model.py` + `infra/utility/
benchmark_polars_io.py` + `docs/commands/`)．D5 行と O5 行に決定を書き，
両方の G4 を retire した．

### Q2 — torch 依存テストの扱い (設計判断)

**settles: N4**

提示した選択肢:

1. **CPU extra を必須化 (推奨)** — torch を `_OPTIONAL_DEPS` から外す
2. torch 非依存の薄いテストへ切り出す
3. collect 失敗を skip でなく error にする (全依存について)
4. 現状維持

**ユーザの回答: (1) CPU extra を必須化．**

**この run で実装・出荷した** (下記 Applied)．N4 行は削除．

### Q3 — `preprocess.DataSource` ABC の位置づけ (設計判断)

**settles: N6-2 + D14(b)**

提示した選択肢:

1. **HCPE 専用と明示する (推奨)** — ABC と基底 `iter_batches_df` を
   HCPE 専用と明記 (改名を含む)，`FileDataSource` から継承を外す
2. 汎用のままにし `array_type` で分岐
3. 基底を abstract にする
4. 現状維持

**ユーザの回答: (1) HCPE 専用と明示する．**

再検証が積み上げてきた「override せず基底を着ているのは HCPE 専用の
`StreamingHcpeDataSource` だけ」という事実がこの向きを支持していた．

**この run では実装していない** (G2 が残る: テスト 3 ファイルが
`FileDataSource` を `PreProcess` に渡している)．N6-2 行と D14(b) 行に
決定を書き，両方の G4 を retire した．**G2 は retire していない** —
設計の回答は環境・結合の制約を動かさない．

### Q4 — `--input-local-cache` と `--input-local-cache-dir` (設計判断)

**settles: O5(a)**

提示した選択肢:

1. **dir に一本化 (推奨)** — bool flag を廃止，dir の有無だけで判定
2. flag に一本化 — S3/GCS にも flag を渡す
3. 両方を有効化条件にする
4. 現状維持

**ユーザの回答: (1) dir に一本化．**

**この run では実装していない** (予算外)．O5 行に決定を書き G4 を retire．

### 予算に入らなかった設計判断 (次 run の待ち行列，3d のランク順)

3d のランク付けは「答えが何を unblock するか」による．今回の 4 問は
「複数行を一度に解く」(Q1: 2 行, Q3: 2 行) と「最も多く再 triage された
行」(Q2: 8 run 連続) を優先した．落選したのは以下で，**この順序で次 run
が開くべきである**:

1. **B8 — D10+D11(1) `FileDataSource.total_pages()` の意味**．
   「ファイル数」と「yield 数」のどちらを意味させるか．contained で，
   決まれば後続 run が小さな diff で書ける．**ただし Q1 の回答
   (`cache_mode` 削除) がこの行を消滅させる可能性がある** —
   `cache_mode="memory"` が無くなれば食い違い自体が起きない．
   次 run は Q1 の実装後に再検証してから問うのが効率的．
2. **B5 — D13 columnar スライスの根本解決**．Q3 の回答 (ABC の整理) が
   `FileDataSource` の形を変えるので，その後の方が形が定まる．
3. **B6 — O9 BigQuery サンプリングの非決定性**．3 つの向き (一時テーブル
   実体化 / 決定的ハッシュ / 併用拒否)．G1 (BigQuery 不在) は**質問を
   妨げない**ので次 run で問える．
4. **B9 / B10 — Deferred 2 / Deferred 3**．答えが出ても 1 run では
   出荷できない規模 (G3 / G2)．

**B11 / B12 / B13 (Deferred 7 / 6 / 5) は待ち行列に入れていない** —
これらは設計の分岐ではなく「向きは判っているが，この環境では検証できない」
(G1: GPU 不在) 類なので，問うても答えが work を unblock しない．

## Applied

Q2 の回答の実装のみ．**P1** (出荷物に触れない: `tests/` と `docs/`) なので
version bump は無い．

| ファイル | 変更 |
|---|---|
| `tests/conftest.py:43-45` | `_OPTIONAL_DEPS` から `"torch"` を削除．除外理由をコメント (`:39-42`) とモジュール docstring (`:7-14`) に明記 |
| `tests/test_conftest_optional_deps.py` | membership に依存するテストの例示依存を torch から `onnx`/`onnxruntime`/`gradio` へ差し替え．**回帰テスト `TestTorchIsNotOptional` を追加** (3 本) |
| `docs/testing-guide.md:5-` | § 「前提: GPU extra が要る」を追加 |

### 回帰テスト

`TestTorchIsNotOptional` の 3 本:

- `test_torch_is_not_in_the_optional_set` — 集合そのものを固定
- `test_missing_torch_is_not_converted_to_a_skip` — 分類関数が `None` を返す
- `test_collect_hook_leaves_a_torch_failure_failed` — collect フックが
  `outcome` を書き換えず，accumulator にも記録しないことを固定

**非空虚性を確認済み**: `"torch"` を `_OPTIONAL_DEPS` に戻すと 3 本とも
失敗する (`assert 'skipped' == 'failed'` 他) ことを実測してから戻した．

## In flight

なし．判断帯の PR は開いていない — 4 件とも設計判断で，回答は同一
セッション内に得られ，実装できた 1 件はこの run の PR に載っている．

## Re-triaged

12 行．いずれも文言を鋭くして残した (下記 6a の内訳を参照)．

**うち 4 行 (D5 / O5 / D14(b) / N6-2) は「re-triaged」ではなく
「decided」**である — 設計判断が確定し G4 を retire したので，人間待ちで
なくなった．次 run は通常の作業として拾える．

**8 行は依然 re-triaged** (人間待ち or 環境待ち):
D13 / O9 / D10+D11 / Deferred 2 / Deferred 3 / Deferred 5 / Deferred 6 /
Deferred 7．

## Corrections to the source records

**なし．**

再検証で見つかった差分はすべて (i) コードが動いたことによる行番号のずれ，
または (ii) 記述の粒度不足であり，**記録の診断そのものが誤っていた例は
無い**．6b の訂正対象は診断/提案が誤っていた場合に限られる．

Deferred 3 の changed shape (streaming アダプタの `set_epoch` 差) は
記録の「redundant な hasattr ガード」という記述より実態が重いが，
これは前 run が既に「メソッド順序も違う」と指摘した線上の精緻化なので，
行の文言更新で足りると判断した．

## Doc findings

`reviews/2026-08-14-tests-require-a-gpu-extra.md` — **status: applied**．

**P2 の恒久承認では適用していない．** 訂正後の本文が現行コードから一意に
決まる drift correction ではなく，この run でユーザが選んだ**新しい方針**
だからである．適用の根拠は 3d でユーザが Q2 に回答したこと (judgment band
の実承認) であり，CLAUDE.md § "Standing approval — drift corrections only"
ではない．提案本文にもその旨を明記した．

## Out of scope

この run が新たに気づいた所見は**なし**．

## Environment notes

- **コンテナは再作成されていた** (8 run 連続)．venv は site-packages
  2 エントリの空から始まり，`uv sync --extra cpu` が必要だった．
  Rust 拡張 (`maou._rust`) は同期後に import 可能で，`maturin develop`
  は不要だった．
- **QA は全て実行できた** — G3 は無し．
  `uv run ruff format` / `ruff check --fix` / `mypy` はいずれも green．
  全スイート **1970 passed, 54 skipped** (71 秒)．
  変更後のサマリが `gradio` 1 モジュールのみを未収集として報告し，
  torch がそこに現れないことが，この run の変更が効いている証拠である
  (`visualize` extra を入れていないため gradio は残る)．
- **G1 は Deferred 5/6/7 と O9 に残る** — GPU も BigQuery もこの環境に
  無い．
