---
kind: backlog
date: 2026-08-14
path:
  - tests/conftest.py
  - src/maou/infra/console/pre_process.py
  - docs/rust-backend.md
  - scripts/
scope: python + docs
level: medium
last_sha: 001d16e
---

# `/audit-backlog` — 診断メッセージ 2 件と `.npy` 遺物スクリプト

`audits/coverage.md` の backlog 14 行 (Deferred 7 + Out-of-scope 7) を
全て HEAD (`001d16e`) に対して再検証し，自動帯 3 件と判断帯 1 件を出荷
した．**質問は上げていない．**

## 再検証 (step 2)

14 行すべてを開いた．**stale 0 / changed shape 2 / confirmed 12**．

行番号が記録どおりだった行が大半で，前 run までの再検証が効いている．
変わっていたのは 2 つ:

- **D13** — `__getitem__` はもう確保をしていない (`:692-696` の薄い委譲)．
  機構は `_columnar_batch_to_structured_array` (`:553-634`) へ移った．
  `np.empty` の `:584` は一致するが，フィールド書き込みは `:586-632` に
  広がっており記録の `:586-587` は狭すぎた．結論 (P4/P6 + G2 + G4) は不変．
- **O5 (d)** — `learn_model.py:847` は文字列リテラル `"file"` ではなく
  `cache_mode=_c` になっていた (`_s3_cache = "file"` が `:838`)．
  上書き手段が無い点は同じなので結論は不変．

どちらも「記録が間違っていた」のではなく「コードが動いた」ので，元記録への
訂正追記はしていない (6b の対象は診断そのものが誤っていた場合のみ)．

## Classification (step 3a)

| ID | 由来 | クラス | 判定した試験 | ゲート |
|---|---|---|---|---|
| N4-a | N4 | **P1** | 出荷物を触らない (`tests/` のみ)．bump 不要 | — |
| DOC-1 | *新規* | **P2** | 散文のみ + 訂正後の本文がコードから一意 | — |
| O5-a′ | O5 (a) | **P3** | 送出する例外の型も分岐条件も不変，diagnostics の文言のみ差分 | — |
| N9 | N9 | **P1** | 出荷物を触らない (`scripts/`)．bump 不要 | **G4** |
| D2 | Deferred 2 | P4 | 挙動が変わる | G3, G4 |
| D3 | Deferred 3 | P4 | 挙動が変わる | G2, G4 |
| D5 / D6 / D7 | Deferred 5/6/7 | P4 | 挙動が変わる | G1 |
| FS-D5 | infra D5 | P6 | ノブ削除は契約破壊 | G4 |
| FS-D10 | infra D10+D11 | P4 | 挙動が変わる | G4 |
| D13 | infra D13 | P4/P6 | 層跨ぎ | G2, G4 |
| D14b | infra D14 | P6 | 公開継承の除去 | G2, G4 |
| O5-rest | O5 (a)(d) | P6 | ノブ削除・新オプション | G4 |
| O9 | infra O9 | — | 向きが 3 案 | G1, G4 |
| N6-2 | N6-2 | P6 | 改名を含む | G4 |

### N9 の G4 を retire しなかった理由

再検証で「`.feather` 版へ書き直す」側は**修理ではなく書き下ろし**だと
判明した (下記 Consumed 参照) ので，2 案のうち 1 案は潰れている．
それでも G4 は残した — 残る選択は「削除する」対「今は残しておいて
後で別物を書く」であり，これは利用者自身の開発用ツールについての
決めだからである．クラスは P1 (出荷物を触らない) だが自動帯には
入れず，PR に判断を載せた．

## Consumed

| 由来の行 | 対象 | 出荷したもの | commit |
|---|---|---|---|
| N4 (一部) | `tests/conftest.py` | 依存欠如で丸ごと落ちたテストモジュールの明示 | `daa97b1` |
| *新規* | `docs/rust-backend.md` | `.npy` 併存記述の訂正 | `cae7445` + `cb395a4` |
| O5 (a) の一部 | `src/maou/infra/console/pre_process.py` | 入力エラーが原因のオプションを名指しする | `ae4a925` |
| **N9 (全部)** | `scripts/` | 遺物スクリプト 2 本の削除 | `9191a0b` |

## Applied

### `daa97b1` — collect 段の skip を集計後に明示する (P1)

`tests/conftest.py:16-25` に accumulator `_UNCOLLECTED_BY_DEP`，
`:119-121` で記録，`:160-202` に `format_uncollected_summary` と
`pytest_terminal_summary` を追加．

collect 段の skip はモジュール内の全テストを run から落とすのに，
pytest の末尾は skip 1 件としか報告しない．そのため
「57 passed, 3 skipped」が緑に見えたまま，変更したコードを 1 件も
実行していない QA が成立する — これは N4 行が 6 run 連続で実害を
記録してきた事象そのものである．

回帰テストは `tests/test_conftest_optional_deps.py:160-` に 4 本．
うち 1 本は**フックが accumulator に書くこと**を固定する
(それが無いと，まさに必要な場面でサマリが空になる)．

### `cae7445` + `cb395a4` — `.npy` 併存記述の訂正 (P2)

`docs/rust-backend.md:809` の
「Both formats are currently supported．Gradual migration recommended．」
は，同じファイルの `:726` (「受け付けるのは `.feather` だけ」) と
自己矛盾していた．`.npy` は 3 経路すべてで
`Only .feather files are supported` として拒否され，`src/` に読み書き
コードは残っていない (`benchmark_polars_io.py:5,8` の履歴コメントのみ)．

drift correction (訂正後の本文がコードから一意に決まる) なので
CLAUDE.md の常設承認で適用．提案は
`reviews/2026-08-14-rust-backend-npy-both-supported.md`
(`status: applied`, `applied_in: cae7445`)．

### `ae4a925` — 入力エラーが原因のオプションを指す (P3)

`src/maou/infra/console/pre_process.py:27-92` に
`describe_missing_input_options()` を追加し，`:570-580` の `else` から
呼ぶ．`--input-s3` / `--input-gcs` は companion 4 つが揃って初めて
分岐に入るため，1 つ欠けると全 elif を外れて
「Please specify an input source …」で停止していた — 入力ソースは
指定済みなのに，そこを疑わせる文言だった．`--input-dataset-id` と
`--input-table-name` の片方だけを渡した BigQuery 入力も同じ場所に落ちる．

P3 の根拠: 送出するのは同じ `ValueError`，分岐条件は 1 つも触っておらず，
差分は文言だけ．**characterization test** で「何も指定しない呼び出しは
従来の文言のまま」を固定した (`tests/maou/infra/console/test_pre_process.py:36`)．
非空虚性は fix を無効化して確認済み — wiring を戻すと
`test_s3_without_cache_dir_reports_the_cache_dir` が落ち，
characterization test は緑のままだった (期待どおり)．

版: `pyproject.toml` 0.89.11 → **0.89.12** (`fix:` → patch)，`uv.lock` 同期．

### `9191a0b` — `.npy` 遺物スクリプト 2 本の削除 (P1 + G4)

N9 行は「実行可能性は未確認 (glob が空振りする以外の欠陥があるかは
見ていない)」としていた．今回確かめた結果，**glob 以外に 3 つ**あった:

1. `Network(embed_dim=…, depth=…, num_heads=…, dropout_rate=…)` —
   現在の `Network.__init__` は keyword-only で
   `board_vocab_size` / `embedding_dim` / `architecture` / `block` /
   `layers` … を取り，渡している 4 引数はどれも存在しない．即 `TypeError`．
2. `LossOptimizerFactory.create_loss_functions(gce_parameter=0.7)` —
   現在の分類メソッドは引数を取らない (`setup.py:810-812`)．即 `TypeError`．
3. `inputs, (...) = batch` → `inputs.to(device)` / `model(inputs)` —
   `KifDataset.__getitem__` の入力側は `tuple[Tensor, Tensor]`
   (盤面 id と持ち駒) なので単一テンソルとして扱えない．

つまり「`.feather` 版へ書き直す」は 4 つの変わった API に対する
書き下ろしであり，backlog の消化作業ではない．参照は
`audits/` と `reviews/` の記述のみでゼロ (`grep` 済み)．

## In flight

**PR は 1 本 (指定ブランチ `claude/audit-backlog-7ndmh3` による collapse)．**
クラス毎の PR 分割ができないため，レビュー単位は commit が担う．
自動帯 3 commit と判断帯 1 commit が同居している．

- 判断帯: `9191a0b` (N9, `scripts/` 2 本の削除)．
  **N9 の行はこの PR 内で削除した** — 6a の separability test により，
  collapse した run では行削除と修正が同じ PR に載って一緒に受理/棄却
  されるので乖離しない．
- したがって**この PR はマージしていない** (5d: 判断帯を含む stack は
  `main` へマージしない)．自動帯 3 件もこの PR の受理を待つ．

`9191a0b` だけを落としたい場合は，その commit を落として再 push すれば
残り 3 件は独立に成立する (依存は無い)．

## Re-triaged

行を残したまま文言を鋭くしたもの:

- **N4** — 「黙って消える」部分を消化したので，残るのは決めそのもの
  (薄いテストへの切り出し vs CPU extra 必須化)．**P4 + G4** 継続．
- **O5** — (a) のメッセージ部分を消化．残るのは bool flag と dir の
  一致 (a の決め)，(d) の `--input-cache-mode` 不在，ノブ自体の削除．
  (d) の記述を `learn_model.py:838/847` の現状に合わせて訂正．
- **D13** — 機構の所在と書き込み範囲を現在の行に更新．結論は不変．

手を付けなかった 10 行 (D2 / D3 / D5 / D6 / D7 / FS-D5 / FS-D10 /
D14b / O9 / N6-2) は文言も結論も変更なし．全て前 run までの再検証が
現行 HEAD でそのまま成立していた．

## Corrections to the source records

**なし．** 今回の changed shape 2 件 (D13 / O5(d)) はいずれもコードが
動いたことによるもので，記録の診断や提案する修正が誤っていたわけでは
ない．6b の訂正追記は診断が誤っていた場合に限る．

## Doc findings

- `reviews/2026-08-14-rust-backend-npy-both-supported.md` —
  `status: applied`, `applied_in: cae7445`．**P2 の常設承認**で適用
  (訂正後の本文がコードから一意に決まる drift correction)．

O5-a′ (P3) は CLI のオプションを増減していないので `docs/commands/` の
更新は不要 (エラー文言のみの変更)．

## Out of scope

この run が新たに気づいたもの: **DOC-1 のみ**で，同じ run 内で消化した
ので backlog 行は起票していない．他に新規所見なし．

## Environment notes

- コンテナは**また**再作成されており，venv は site-packages 2 エントリの
  空から始まった．`uv sync --extra cpu` が必要で，入れる前は `pytest` すら
  無い．これは N4 行が記録し続けている事象の 7 回目である
  (この run では `maturin develop` は不要だった — 触った Python 経路が
  Rust 拡張を要求しなかったため)．
- **G1 (この環境で検証不能)**: GPU が無いので D5 / D6 / D7 の同期削減は
  測れない．BigQuery が無いので O9 の非決定性は fake client で再現できない．
- **G3**: D2 (~600 行の学習経路統合) の等価性をこの環境で示す手段が無い．
- QA は全て実行した (下記 step 7 参照)．未実行のチェックは無い．
