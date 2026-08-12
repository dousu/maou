---
kind: backlog
date: 2026-08-12
path:
  - src/maou/domain/data/arrow_format.py
  - src/maou/domain/data/dataframe_io.py
  - src/maou/infra/file_system/streaming_file_source.py
  - src/maou/interface/preprocess.py
  - rust/maou_io/src/arrow_io.rs
  - rust/maou_index/src/index.rs
  - rust/maou_shogi/src/dfpn/tt/mod.rs
  - rust/maou_usi/src/agent.rs
  - .pre-commit-config.yaml
  - docs/code-quality.md
  - docs/rust-backend.md
level: medium
last_sha: ab2a7aa
---

# `/audit-backlog` — Arrow の File/Stream 判定 (O10 / O7 / O8) と clippy hook

`audits/coverage.md` の backlog 25 行 (deferred 12 + out-of-scope 13) を
HEAD (`ab2a7aa`) に対して再検証し，判断コストで分類してから，
「Arrow の File/Stream 判定」で繋がっている 3 行を軸に消化した．
セレクタなしの通常 run．**ユーザには何も聞いていない** — 決めを要する
ものは PR がその決めを運んでいる．

開始時点で earlier run から引き継いだ open PR は無かった (`list_pull_requests` = 空)．

## 環境

このコンテナは venv が空の状態から始まったので，QA を回せる状態にするまでに
`uv sync` → `uv sync --extra cpu` → `maturin develop --release` (13 分) が要った．
**N4 行が言うとおり**，CPU extra を入れる前は `tests/maou/infra/file_system` が
まるごと skip される (torch 未導入でモジュール単位 skip)．入れてから
92 passed になった．今回の変更は同パッケージに触れるので，
extra 無しで回していれば無検証のまま緑に見えていた．

## Classification

P6 → P1 の順に降りて最初に当たったクラスを採り，そのあとゲートを見た．
**ゲートはクラスを変えず，自動帯から外すだけ**なので両方を記録する．

### 自動帯 (無停止で修正・PR・マージ)

| ID | 行 | 再検証 | クラス | 決め手 | ゲート |
|---|---|---|---|---|---|
| A-1 | O10 (out-of-scope) | confirmed | **P3** | 受け付ける全入力について成果物と戻り値が不変．判定式・判定幅 (8 バイト)・fallback 先が 2 実装で同一であることを確認し，characterization テストで固定した | なし |

**P3 の根拠として「置き場所が一意である」ことも要った．** O10 の行は
「O7 / O8残り / O10 は『判定をどこに置くか』という 1 つの決めを共有している」
と書いており，これは G4 (記録自身が決めを要ると言っている) に当たる．
再検証でこの G4 を**明示的に解除した**: `interface` は `infra` に依存できず
(CLAUDE.md の層規則)，`infra` と `interface` の双方が判定を要る以上，
両者が依存できる最下層は `domain` しかない．選択肢が 1 つに潰れているので，
これは決めではなく導出である．

### 判断帯 (PR にして未マージで残す)

| ID | 行 | 再検証 | クラス | 決め手 | ゲート | PR |
|---|---|---|---|---|---|---|
| B-1 | O7 | confirmed | **P4** | 今まで失敗していた入力が成功するようになる．既存データは読めたまま，既存の呼び出しも有効なまま | なし | #478 |
| B-2 | O8 残り | **changed shape** | **P4** | Stream 形式の入力が chunk 対象に入る (今までは黙って外れていた) | なし | #478 |
| C-1 | NEW-2 | confirmed + 計測 | **P3** (コードはテスト専用の lint 修正で挙動不変) | — | **G4 保持** | #479 |
| D-1 | NEW-1 | confirmed | **judgment (P4 下限)** | P2 の一意性テストに落ちる — 書き方が 3 通りあり得る | G4 | #480 |
| D-2 | NEW-3 | confirmed | **judgment (P4 下限)** | 同上 — `.feather` 行の訂正は一意だが `.npy` 行の削除/注記は著者判断 | G4 | #480 |

**C-1 の G4 の扱い**: 行は「入れると既存 warning の数だけ初回コストが出る．
まず計測してから可否を決めること」と書いていた．計測して G4 の**理由**は
消えた (workspace 全体で warning 3 件，すべてテストコード) が，
「入れるか」の決めそのものはユーザのものなので **G4 は保持**し，自動マージ
しなかった．fail-safe は上向き．

**D-1 / D-2 のクラス**: doc 変更はプログラムの振る舞いを変えないので
ラダーの P4-P6 の定義には直接当てはまらない．P2 の第 2 テスト
(「訂正後の本文が現行コードから一意に決まるか」) に落ちる以上，自動帯には
入れられないので，fail-safe 規則どおり **P4 を下限**として判断帯に置いた．

## Consumed

| ID | 由来 | 対象 | 出荷内容 | PR |
|---|---|---|---|---|
| A-1 (O10) | [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) | `domain/data`, `infra/file_system` | `domain/data/arrow_format.py` を新設し，二重化していた判定を集約．`infra` 側は再輸出 | **#477 — マージ済み** (`dc2231e`) |

## Applied

| 変更 | 場所 | commit |
|---|---|---|
| `arrow_format.py` 新設 (`ARROW_FILE_MAGIC` / `is_arrow_ipc_file_bytes` / `is_arrow_ipc_file_format` / `scan_row_count`) | `src/maou/domain/data/arrow_format.py` | `1b46fdc` |
| 自前の定数を削除して共有判定を使用 | `src/maou/domain/data/dataframe_io.py:15,50` | `1b46fdc` |
| 自前実装 (約 50 行) を削除して再輸出 | `src/maou/infra/file_system/streaming_file_source.py:20-27` | `1b46fdc` |
| characterization テスト 12 件 | `tests/maou/domain/data/test_arrow_format.py` | `1b46fdc` |

バージョン: `pyproject.toml` `0.86.4` → `0.86.5`．

## In flight

すべて backlog 行を残したまま (6a: 行を消すのはマージされたときだけ)．

| PR | クラス | base | 決めてもらう点 |
|---|---|---|---|
| **#478** | P4 | `claude/audit-backlog-cz2r2u` (#477) — `domain/data/arrow_format` に依存するため stack | Stream 形式の `.feather` を全経路で読めるようにしてよいか．今まで「索引構築が落ちる」「サイズ調整から黙って外れる」だった入力が処理されるようになる．既存データ・既存コマンドは壊れない |
| **#479** | P3+G4 | `main` (independent) | Rust コミットのたびに `cargo clippy --workspace --all-targets -- -D warnings` を走らせてよいか．初回コストは計測済みで実質ゼロ (warning 3 件を PR 内で解消) |
| **#480** | judgment (docs) | `main` (independent) | `reviews/` 提案 2 件の承認．doc 本体は未編集．NEW-1 は「追記か言語別表への再構成か」，NEW-3 は「`.npy` 行を削除するか legacy 注記か」 |

`#477` がマージされたので，`#478` の base は `main` へ retarget して
リフレッシュする必要がある (5b)．

## Re-triaged

| 行 | 再検証 | 今回動かさなかった理由 |
|---|---|---|
| app/learning Deferred 2 | confirmed (行番号一致) | ~400 行の multi-stage 学習経路の refactor．クラスは P3 だが，1 回の backlog run の粒度を超える．専用の変更として扱うべき |
| app/learning Deferred 3 | confirmed | 6 つの公開名が消えるので **P6**．テストからも参照される |
| app/learning Deferred 4 | confirmed (`_ensure_device` は現在 6 箇所) | ~250 行の基底クラス抽出．Deferred 2 と同じ理由 |
| app/learning Deferred 5 | confirmed dormant | mask を供給する経路が無いままなので着手できない．状況は前回から変わっていない |
| app/learning Deferred 6 / 7 | confirmed | **G1** — GPU が要る．この環境では正しさを確かめられない |
| infra D1 (`moveWinRate`) | confirmed | **P5 + G4**．CLI 既定が `--policy-target-mode win_rate` であることも確認した (`learn_model.py:569`) ので，`--no-streaming` は既定オプションで落ちる．**ask 対象として問い，「dtype に足す」の回答を得て実装した → PR #482** (下記 § Ask 参照) |
| infra D2 (seed) | confirmed (**shape 変化**) | `__train_test_split` は seed を受け取れるようになったが (`file_data_source.py:186`)，公開 `train_test_split(test_ratio)` と ABC (`app/learning/dl.py:74`) には seed が無く，呼び出し側 (`dl.py:244`, `stage_component_factory.py:99,196`) も渡さない．端から端まで無 seed という結論は変わらない．**P4 + G4** |
| infra D3+D4 | confirmed (**大幅に縮小**) | 下記 Corrections 参照．**残りは P3・ゲート無しで，次 run の自動帯の最有力候補** |
| infra D5 / D13 / D14 / D15 | confirmed | 今回の軸 (Arrow の形式判定) から外れる．D15 は「運用上のリスクとして実在するか」自体が未判断のまま |
| infra D8+D9 | confirmed (production caller ゼロを `src`/`tests`/`docs` で再確認) | **ask 対象として問い，「削除する」の回答を得て実装した → PR #483** (下記 § Ask 参照)．行の後半 (`train_test_split` の `list(range(N))`) は**未処理のまま残している** — seed 固定時の分割値が変わるので D2 と同じ決めに帰着する |
| infra D10+D11 (1) | confirmed (**dormant**) | 下記 Corrections 参照 |
| O5 | confirmed (**(a) の見立てを訂正**) | 下記 Corrections 参照 |
| O9 | confirmed | **P4 + G1** — BigQuery が無いと直したことを確かめられない．`TABLESAMPLE` をページごとに引き直す (`bq_data_source.py:405-420`) 一方で総数は別クエリ (`:236-243`) から採る構造は現存 |
| N3 | confirmed | `dataset.py:45` / `hcpe_transform.py:62` とも `abc.ABCMeta` 不使用のまま．`ABC` にすると現存の非準拠実装が構築時に落ちるので，何が壊れるかの洗い出しが先 |
| N4 | confirmed (実害を再確認) | この run 自身が踏んだ (上記 § 環境)．ただし「薄いテストへ切り出す」か「CPU extra を必須にする」かの決めは未解決のまま |

## Corrections to the source records

いずれも**診断そのものが現状と食い違っていた**ものに限る．
worklist としての状態 (解決済み等) は書かない — それは行の削除で表す．

1. **[2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D3+D4**:
   「ディスパッチ 4 箇所」のうち 2 箇所は既に解消されている．
   `file_data_source.py:559-564` の「呼び出しごとに dict 再構築」は
   module 級の `_DF_TO_NUMPY_CONVERTERS` / `_DF_TO_COLUMNAR_CONVERTERS`
   (`:41-56`) に整理済みで，`:905-919` の if/elif も現存しない
   (その行は今 `total_pages()`)．残るのは (a) 3 entry の変換テーブルが
   2 モジュールに重複，(b) structured 変換器 2 本，の 2 点のみ．

2. **同 D10+D11 (1)**: `FileDataSource.total_pages()` と
   `cache_mode="memory"` の `iter_batches` の食い違いは残っているが，
   **production caller はゼロ**．記録が唯一の caller として挙げる
   `hcpe_transform.py:679` が受け取るのは `StreamingHcpeDataSource`
   (`console/pre_process.py:494` が構築) で，そちらは
   `len(self._file_paths)` を返し 1 ファイル 1 batch を yield するので
   tqdm は正しい．したがって記録が言う「tqdm が 1/N で止まる」は
   現在の経路では起きない．

3. **同 O5(a)**: 「`maou pre-process --input-s3 --input-local-cache` は
   無言の no-op」は誤り．`--input-local-cache` (bool flag) と
   `--input-local-cache-dir` (str) は別のオプションで，S3/GCS の elif が
   見ているのは後者である (`pre_process.py:419`,`:451`)．dir を渡さないと
   elif を全て外れて最後の `else` に落ち，
   **"Please specify an input source (file path, BigQuery table, GCS
   bucket, or S3 bucket)" という誤誘導エラーで停止する** (`:497-501`)．
   黙って無視されるわけではないが，メッセージが原因を指していない．

4. **同 O8**: 「正しい向きは『Stream 形式でも行数を取れるようにする』」は
   正しい．ただしその先に，記録が想定していない別の欠陥がある — 下記 New．

## Doc findings

| 提案 | 対象 | status | 経路 |
|---|---|---|---|
| `reviews/2026-08-12-code-quality-rust-local-hooks.md` | `docs/code-quality.md` | `pending` | **P2 の standing approval は適用しない**．追記の書き方が一意に決まらないため (4d の "No" 分岐)．doc 本体は未編集 |
| `reviews/2026-08-12-rust-backend-performance-table.md` | `docs/rust-backend.md` | `pending` | 同上 |

この run では source fix が durable doc を新たに無効化してはいない
(`docs/commands/` の CLI オプションには触れていない)．上の 2 件は
NEW-1 / NEW-3 として元から backlog にあったもの．

## Out of scope (この run が新たに気付いたもの)

- **N-1: polars が書いた `.feather` と Rust writer が書いた `.feather` は
  結合できない．** `merge_hcpe_feather_files` が
  `It is not possible to concatenate arrays of different data types
  (BinaryView, LargeBinary)` で落ちる．polars 1.38 は `Binary` 列を
  BinaryView で書き，`maou_io::save_feather` (arrow-rs) は LargeBinary で
  書くため．**File/Stream 形式とは無関係で writer 依存**であることを実測で
  切り分けた (polars 書きの File 形式同士でも Rust 書きと混ぜれば落ちる)．
  `coverage.md` の Out-of-scope backlog に起票済み．

  **見つかった経緯が重要**: O8 の回帰テストを書いたとき，Stream 形式の
  ファイルを chunk させようとして落ちた．最初は「O8 の修正が壊した」と
  読めたが，File 形式でも同じことが起きるのを確かめて切り分いた．
  記録の見立てを試すときは，落ちた原因が本当にその変更かを別軸で
  確かめること — ここで止めていれば O8 を誤って re-triage していた．

## Reconciliation (6d)

触れた項目 + 新規発見 = 26 (backlog 25 行 + 新規 1)

- **resolved** (行削除・マージ済み): 1 — O10 (#477, `dc2231e`)
- **in flight** (行保持・PR リンク付与): 5 — O7 / O8残り (#478), NEW-2 (#479),
  NEW-1 / NEW-3 (#480)
- **re-triaged** (行保持・文面を鋭くした): 4 — D3+D4, D10+D11, O5, および
  上の Re-triaged 表に挙げた残り 15 行 (文面変更なし，理由は本記録に記載)
- **new row**: 1 — N-1 (BinaryView/LargeBinary)
- **not a finding**: 0

行数: 25 → 25 (O10 を削り，N-1 を足した)．

## Environment notes

- **G1 に当たったもの**: app/learning Deferred 6 / 7 (GPU 必須)，
  O9 (BigQuery 必須)．この環境では正しさを確立できない．
- **G3 は発生しなかった**: `uv sync --extra cpu` + `maturin develop --release`
  まで通したので，Python (pytest 全件 1855 passed / 54 skipped, mypy, ruff)
  も Rust (`cargo test -p maou_io -p maou_index -p maou_usi -p maou_shogi`)
  もこの環境で実行できた．
- **PR の CI について**: このリポジトリは PR でテストを回さない
  (`claude-code-review.yml` は `workflow_dispatch` のみ)．PR 上のチェックは
  `check-version-bump` 1 本で，#477 では success．**緑であることは
  テストが通った証拠ではない**ので，自動マージの根拠はローカルで回した
  上記 QA の方である．
- git hooks はコンテナに未インストールだったので
  `pre-commit install -t pre-commit -t commit-msg` を実行してから
  全コミットを打った (`--no-verify` は不使用)．

## Ask (1 回だけ・自動帯のマージ後)

`/audit-backlog` の分割テスト (「枝が実質的に異なる diff を生み，外すと
レビューして捨てる作業が無駄になるか」) を満たしたのは 2 件だけだったので，
**1 回の `AskUserQuestion` にまとめて**，自動帯 (#477 / #481) がマージされた
あとに聞いた．回答はこの run 内で実装し，PR にしてある — **回答は diff を
決めるだけで，マージを承認するものではない**ので，どちらも未マージのまま．

| 行 | 枝 | 回答 | 実装 |
|---|---|---|---|
| **D1** | (a) dtype に足す / (b) 変換直後に捨てる / (c) 明示エラーに留める | **(a) dtype に足す** | PR #482 |
| **D8+D9** | (a) 修復する / (b) 削除する / (c) 触らない | **(b) 削除する** | PR #483 |

**なぜこの 2 件だけか**: どちらも枝が共通行を持たない．D1 の (a) と (b) は
複数層にまたがる逆向きの設計で，D8+D9 の「削除」と「修復」は片方が公開名と
テストを消し，もう片方が構築子を書き換える．外した方を書いていたら，
レビューして捨てる分がそのまま無駄になる．

他の判断帯 (O7 / O8 / NEW-1 / NEW-2 / NEW-3) は**聞いていない** — いずれも
書くべきコードが 1 通りに決まり，判断は「これを入れるか」だけなので，
PR がその判断を運べる．

## In flight (追記: ask 後の 2 件)

| PR | クラス | base | 決めてもらう点 |
|---|---|---|---|
| **#482** | P5 | `main` (independent) | `moveWinRate` を dtype に載せる．**旧 dtype の structured array / `.npy` は新 reader と形が合わなくなる** (`.feather` は影響なし，古い列欠けはゼロ埋め)．dtype を `moveLabel` に揃えて float16 にした点だけ私の選択なので，float32 で揃えたければ 1 行 |
| **#483** | P6 | `main` (independent) | `file_level_split` の削除．**公開名とそのテストファイルが消える**．production caller はゼロ．データ互換性と CLI には影響なし |

**バージョンの衝突**: #482 と #483 はどちらも `0.87.0` を取っている
(`major_version_zero = true` なので breaking も minor)．後からマージする方は
再 bump が要る．両 PR の本文に明記した．

## Reconciliation (6d) — ask 後の最終値

触れた項目 + 新規発見 = 26 (backlog 25 行 + 新規 1)

- **resolved** (行削除・マージ済み): 1 — O10 (#477, `dc2231e`)
- **in flight** (行保持・PR リンク付与): 7 — O7 / O8残り (#478), NEW-2 (#479),
  NEW-1 / NEW-3 (#480), D1 (#482), D8+D9 前半 (#483)
- **re-triaged** (行保持): 17 — うち文面を鋭くしたのは D3+D4 / D10+D11 / O5 /
  D8+D9 後半
- **new row**: 1 — N-1 (BinaryView/LargeBinary)
- **not a finding**: 0

行数: 25 → 25 (O10 を削り，N-1 を足した)．D8+D9 は **#483 がマージされても
行を消さない** — 後半 (`list(range(N))` の索引メモリ) が未処理で残るため，
その部分だけに絞って残す．
