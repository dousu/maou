---
kind: backlog
date: 2026-08-12
path:
  - .pre-commit-config.yaml
  - src/maou/infra/file_system/file_data_source.py
  - src/maou/interface/preprocess.py
  - src/maou/interface/learn.py
  - src/maou/infra/console/learn_model.py
level: medium
last_sha: f4eb9a0
---

# `/audit-backlog` — 自動帯 (N6 / D6+D7 / O8前半) と判断帯 (N1)

`audits/coverage.md` の backlog 27 行 (deferred 13 + out-of-scope 14) を
全件 HEAD に対して再検証し，判断コストで分類したうえで自動帯を消化した．
セレクタなしの通常 run．

## Classification

判断コスト P6 → P1 の順に降りて最初に当たったクラスを採り，そのあと
ゲートを見た．**ゲートはクラスを変えず，自動帯から外すだけ**なので
両方を記録する．

### 自動帯 (無停止で修正・PR・マージ)

| ID | 行 | 再検証 | クラス | 決め手 | ゲート |
|---|---|---|---|---|---|
| N6 | out-of-scope | confirmed | **P1** | 出荷ファイルに触れない (`.pre-commit-config.yaml` のみ)．bump 対象なし | G4 を**解除** (下記) |
| D6+D7 | deferred | confirmed | **P3** | 受け付ける入力に対し成果物と戻り値が不変．差はメモリと診断のみ | なし (G3 は下記で解除) |
| O8 前半 | out-of-scope | confirmed | **P3** | 行数の値も例外も同一であることを実測で確認 | なし |

### 判断帯 (PR にして未マージで残す / 再トリアージ)

| ID | クラス | 決め手 | ゲート | 処遇 |
|---|---|---|---|---|
| N1 | **P4** | 観測可能な振る舞いが変わる (検証割合 10%→20%)．データは読めたまま，呼び出しも有効なまま | — | **PR #475 — ユーザ承認を得てマージ** |
| O8 後半 | P4 | 裸の `except` を締めると Stream 形式の扱いが変わる | — | 再トリアージ |
| D2 | P4 | 既定 seed を入れると分割そのものが変わる | — | 再トリアージ |
| D1 | **P5** | structured dtype が変わり，既存の配列/`.npy` が読み手と食い違う | G4 (記録が「dtype に足す」「変換直後に捨てる」の 2 方向を明示) | 再トリアージ |
| D8+D9 | **P6** (削除時) / P4 (修理時) | 削除なら公開名とテストが消える | G4 (削除か修理かの決めが先) | 再トリアージ |
| D3+D4 | P3 | ディスパッチ統合自体は振る舞いを変えない | **G2** (D1 の変換器 2 本と同じ場所に触るので単独で閉じない) | 再トリアージ |
| D5 | P4 | ノブの意味が変わる | G4 (O5 と一体という記録の判断が生きている) | 再トリアージ |
| D10+D11 | P4 | `total_pages()` の戻り値の意味が変わる | G4 (「ファイル数」か「yield 数」かの決め) | 再トリアージ |
| D13 残り | P4 | `__getitem__` の確保/コピー構造が変わる | G2 (`app/learning/dataset.py` と ABC に及ぶ) | 再トリアージ |
| D14 | P4 | 行数スキャンの共有化 + 二重 ABC の解消 | G2 (`benchmark_polars_io.py` の対応が要る) | 再トリアージ |
| D15 | P4 | 不完全 `.feather` の検出を入れると読める入力集合が変わる | G4 (記録自身が「運用上のリスクとして実在するかは要判断」と書く) | 再トリアージ |
| O5 | P4 | CLI オプションの効き方が変わる | G4 + **記録の誤診あり** (下記) | 再トリアージ |
| O7 | P4 | 今まで失敗していた Stream 形式 `.feather` が読めるようになる | G3 相当 (Rust の crate bump + ビルドが要る) | 再トリアージ |
| O9 | P4 | `TABLESAMPLE` の一貫性 | **G1** (BigQuery 実データなしでは正しさを確認できない) | 再トリアージ |
| O10 | P3 | 定数の重複排除自体は振る舞いを変えない | G4 (記録が「O7 で fallback 方針を触るなら同時に」と紐づけ)．共有先の置き場所 (domain か interface 再輸出か) が authored な選択になる | 再トリアージ |
| N3 | P4/P6 | `ABC` を継承させると現存の非準拠実装が構築時に落ちる | G4 (「何が壊れるかを洗う必要がある」) | 再トリアージ |
| N4 | P1 | `tests/` と環境設定にしか触れない | G4 (「薄いテストへ切り出す」か「CPU extra を必須にする」かの決め) | 再トリアージ (**実測を追記**) |
| app/learning Deferred 2/3/4 | P4 / **P6** / P4 | 3 は 6 つの公開名が消える | G4 (いずれも「独立した reviewed change として出すべき」規模) | 再トリアージ |
| app/learning Deferred 5/6/7 | P4 | GPU セマンティクスの変更 | **G1** (実 GPU での測定が要る) | 再トリアージ |

### ゲートの解除

- **N6 の G4 を解除した．** 行は判断点を 3 つ挙げていたが，再検証で
  (a) は行自身が「Python 側 (`ruff-format`) は自動整形なので揃えるなら
  後者」と書いており一意に決まる，(c) は `cargo fmt` が crate 単位で
  動くことによる技術的要請で判断ではない，と分かった．(b) の clippy は
  **N6 のやること (fmt hook) の外**なので切り離して新規行に落とした．
  残った選択肢が 1 つになったので G4 は消える．
- **D6+D7 の G3 を解除した．** `file_data_source.py` の唯一のテスト
  (`test_file_data_source.py`) は torch 未導入だとモジュールごと skip
  される．そのままでは「振る舞い不変」が結果ではなく主張になるため，
  `uv sync --extra cpu` を入れてから QA した (下記 Environment notes)．

## Consumed

| 項目 | 出所 | 対象 | 出したもの | PR |
|---|---|---|---|---|
| N6 | 2026-08-12 backlog auto-band-and-p4 | `.pre-commit-config.yaml` | `cargo fmt --all` の local hook (自動整形) | #474 (`195bafa`) |
| D6+D7 | 2026-08-10 infra/file_system | `file_data_source.py` | 未読の状態 5 件と到達不能分岐の削除 | #474 (`cb4a78e`) |
| O8 前半 | 2026-08-10 infra/file_system | `interface/preprocess.py` | 行数取得の全列実体化をやめる | #474 (`cb4a78e`) |
| N1 | 2026-08-12 backlog auto-band-and-p4 | `interface/learn.py` + `infra/console/learn_model.py` | `--test-ratio` 既定を `DEFAULT_TEST_RATIO` 1 つに寄せる (0.2) | #475 (`b9f0c06`) |

## Applied

- `.pre-commit-config.yaml:64-84` — `cargo-fmt` local hook (`195bafa`)
- `file_data_source.py:207-214` — `_FileEntry.dtype` / `.memmap` 削除 (`cb4a78e`)
- `file_data_source.py:243,259,269` — `bit_pack` / `memmap_arrays` /
  `_last_file_idx` の未読属性を削除 (`cb4a78e`)
- `file_data_source.py:865-` — `iter_batches_df` の到達不能分岐を削除 (`cb4a78e`)
- `interface/preprocess.py:178-189` — `select(pl.len())` 化 (`cb4a78e`)
- `interface/learn.py:47-54,308` — `DEFAULT_TEST_RATIO` の導入 (`b9f0c06`, PR #475)
- `infra/console/learn_model.py:887` — 直値 0.1 の除去 (`b9f0c06`, PR #475)

バージョン: `pyproject.toml` 0.86.2 → 0.86.3 (#474) → 0.86.4 (#475)．
N6 は出荷ファイルに触れないので bump なし．

## In flight

なし．判断帯として出した PR #475 は，**同一セッション内でユーザが
「確認したのでマージしてよい」と回答した**ため，5e の「ユーザがこの
セッションで答えた場合はその決定を適用してマージする」に従いマージした
(`4c27c18`)．したがって N1 は in flight ではなく resolved 扱いで，
backlog 行も削除した (6a: 行を消すのは**マージされたとき**)．

判断の中身は PR 本文に残っている: `--test-ratio` 未指定時の既定を
doc が書く 0.2 に寄せた．もう一方の選択肢 (streaming の実測既定である
0.1 に寄せ，doc を直す) は採らなかった．

## Re-triaged

- **D12(d)** (`total_pages`/`total_rows`/`file_paths` の重複状態) —
  **やらない判断**．再検証したところ，3 つとも `__init__` の末尾で
  `cum_lengths` から一度だけ設定され，以後どこからも書き換えられない．
  記録が心配していた desync ハザードは D12(a) の消化 (PR #456) で既に
  消えており，**残っているのは set-once の導出値**なので property 化は
  リスクを下げない．加えて `file_paths` は導出値ではなく**構築子の入力**
  であり，property 化するとアクセスごとに list を作り直す劣化になる
  (`total_rows` は `np.int64` から `int` へ型も動く)．D12 行は (c) と
  合わせて削除した．
- **O8 後半** (裸の `except Exception` が Stream 形式を握り潰す) — 前半の
  消化で行は残る．締めると，これまで `ok_files` へ素通ししていた
  Stream 形式ファイルが例外で落ちるようになるので **P4**．正しい向きは
  「例外を締める」ではなく「Stream 形式でも行数を取れるようにする」
  (`streaming_file_source.scan_row_count` と同じ判定) だが，interface は
  infra に依存できないので判定を **domain へ寄せる必要**があり，
  **O10 の「共有先をどこに置くか」と同じ決めに帰着する**．O7/O8後半/O10 は
  独立した 3 件ではなく，**Arrow File/Stream 判定の置き場所という 1 つの
  決め**を共有している — 次の run はこの 3 行をまとめて扱うべき．
- **N4** — 再検証で数字を更新した．torch なし: 57 passed + 3 skipped /
  `--extra cpu`: **90 passed** (記録の 52/83 は当時の値)．
  `test_file_data_source.py` は**モジュールごと** skip される
  (`SKIPPED [1] ...: optional dependency 'torch' is not installed`)．
  本 run はこれを踏んで CPU extra を入れてから QA した．**判断はまだ
  要る** (薄いテストへ切り出すか CPU extra を必須にするか) ので行は残す．

## Corrections to the source records

いずれも「解決済み」ではなく**診断そのものが誤り**だったもの．
`audits/README.md` の規則どおり，該当記録に追記した (worklist 状態は
書かない)．

1. **D12(c) の `cumulative_rows`** — 記録は「`sum(lengths)` の二重計算」
   と書くが，`cumulative_rows` は**ループ内で使う走行合計**で，
   `cum_lengths` (`np.cumsum`) が作られるのはループを抜けた後である．
   マイルストーンごとに `sum(lengths)` を取る形に「直す」と O(n²) の
   悪化になる．**欠陥ではない**ので棄却した．`milestone_interval` の
   7 行 3 分岐も，圧縮しても読みやすくならず欠陥もないので棄却．
2. **N2 のキャッシュ検証** — 記録は「一部だけ欠けている / 壊れている
   という最も検証が要るケースを通らない」と書くが，**逆である**．
   一部欠けている場合は `all_cache_exists` が False になるので早期
   return せず，末尾の検証を**通る**．スキップされるのは全ページ揃って
   いる場合だけで，そこは直前の `__check_local_cache_exists` が全ページ
   について確認済みである．`__check_local_cache_exists` は
   `cache_path.exists()` だけなので破損は検出しないが，末尾の検証も
   ファイル**数**を数えるだけで破損は検出しない．**述べられた欠陥は
   存在しない**ので棄却し，行を削除した．
3. **O5(a) の `--input-local-cache`** — 記録は「BigQuery にしか渡され
   ず，S3/GCS は無言の no-op」と書くが，**2 つの別オプションを混同して
   いる**．`--input-local-cache` (bool フラグ) と
   `--input-local-cache-dir` (パス) があり，S3/GCS 分岐は
   `local_cache_dir=input_local_cache_dir` を**ちゃんと渡している**
   (`pre_process.py:427`/`:459`)．実際の欠陥は別で，(i) bool フラグの方は
   BigQuery だけが読み (`use_local_cache=` `:399`)，S3/GCS では黙って
   無視される，(ii) S3/GCS では `--input-local-cache-dir` が事実上**必須**
   なのに任意に見え，省くと elif 連鎖を全て外れて
   `"Please specify an input source"` という**入力を指定しているのに
   出る誤ったメッセージ**で落ちる．(b)(c)(d) は記録どおり有効．

## Doc findings

- **今回の修正による durable-doc drift はなかった．** 確認した範囲:
  `docs/code-quality.md` (pre-commit 節は Python スコープの原則を述べる
  だけで，Rust hook の追加と矛盾しない)，`docs/rust-backend.md:672-684`
  (`iter_batches_df` の用例 — 契約は不変)，`docs/commands/learn_model.md`
  (N1 は doc の 0.2 に**寄せた**ので doc は両経路について正しくなる)．
  よって `reviews/*.md` の提案は起こしていない．
- **文書化したい内容はあるが，それは drift ではなく新規の指針**
  (`docs/code-quality.md` の local hook 原則に rustfmt の実例を足す)．
  P2 の一意性テストに落ちるので承認待ちの側に倒し，新規 backlog 行に
  した (下記 NEW-1)．

## Out of scope (この run が新しく気づいたもの)

`coverage.md` の out-of-scope backlog に 3 行として追加した．

- **NEW-1** `docs/code-quality.md` — § 「linter/formatter は local hook で
  回す」が Python ツールしか挙げていない．本 run で rustfmt が 2 例目に
  なったが，追記は drift 訂正ではなく**新しい指針**なので承認が要る．
- **NEW-2** `.pre-commit-config.yaml` — `cargo clippy` の hook が無い
  (N6 から意図的に切り離した)．入れると既存 warning の初回コストが出る．
  workspace 全体の warning 数は未計測．
- **NEW-3** `docs/rust-backend.md:724-728` — § Performance Comparison の表が
  古い．`.feather` 行が `iter_batches()` を「❌ Not supported」としているが
  実際は動く (`test_file_data_source_iter_batches` が通る)．`.npy` /
  `Cloud (cached)` の行は，`FileManager` が
  `"Only .feather files are supported"` で弾くようになった今は存在しない
  経路．`.feather` 行の訂正は一意だが `.npy` 行を消すか legacy と注記
  するかは選択なので，全体としては P2 に落ちない．

## Environment notes

- **torch がこのコンテナに入っていなかった**．`uv sync --extra cpu` で
  導入して QA した (`uv.lock` は不変)．これをしないと
  `tests/maou/infra/file_system/test_file_data_source.py` が丸ごと skip
  され，D6+D7 の変更が**無検証のまま緑に見える**．これは N4 そのもの．
- **GPU なし**: app/learning Deferred 5/6/7 は G1 で手を付けられない．
- **BigQuery / S3 / GCS の資格情報なし**: O9 は G1．N1 の対象分岐も
  実行到達できないため回帰テストはソース検査で固定した．
- **Rust**: `cargo fmt --all -- --check` は HEAD で clean (PR #464 の結果)．
  `cargo` / `rustfmt 1.8.0-stable` は利用可能なので N6 の hook は実機で
  検証できた．O7 は crate bump とビルドが要るため本 run では扱っていない．
- リポジトリは PR に対してテストを走らせない (`claude-code-review.yml` は
  `workflow_dispatch` のみ)．#474 で緑だったのは `check-version-bump` の
  1 件のみで，**マージの根拠は本 run がこのコンテナで走らせた QA** である．

## Reconciliation

単位は **backlog の行**．触れた 27 行 + 新規に気づいた 3 件 = **30**．
すべてに 1 つずつ処遇を割り当てる (割り当てられない項目があれば，それが
この検算で捕まえたかった欠陥である)．

| 処遇 | 行数 | 内訳 |
|---|---|---|
| **resolved** (修正がマージされ，行を削除) | 3 | N6 (#474), D6+D7 (#474), N1 (#475) |
| **in flight** (行維持・PR リンク) | 0 | #475 は同一セッションでユーザが承認したためマージ済み |
| **re-triaged** (行維持・文面を鋭くした) | 22 | O8, N4, D1, D2, D3+D4, D5, D8+D9, D10+D11, D13, D14, D15, O5, O7, O9, O10, N3 (16 行) + app/learning Deferred 2/3/4/5/6/7 (6 行) |
| **new row** | 3 | NEW-1, NEW-2, NEW-3 |
| **not a finding** (棄却し，行を削除) | 2 | D12 (下記 (c) と (d) の両方が消えたので行ごと), N2 |

検算: 3 + 0 + 22 + 3 + 2 = **30** ✓

**O8 の数え方**: 前半 (全列実体化) はマージされたが，後半 (裸の `except`)
が残るので**行は維持**される．行単位の会計なので **re-triaged に 1 回だけ**
数え，resolved には数えない — 行が消えていないのに resolved に数えると，
削除行数と合わなくなる．

backlog 行数: **27 → 25**
(削除 5 行: N6 / D6+D7 / N1 / D12 / N2，追加 3 行: NEW-1 / NEW-2 / NEW-3．
27 − 5 + 3 = 25．内訳は deferred 12 行 + out-of-scope 13 行．)

主表の `Open items`: `src/maou/infra/file_system` を **10 deferred → 8
deferred** に更新 (D6 と D7 の 2 findings が消えた)．`src/maou/app/learning`
は本 run で解決した行がないので 6 deferred のまま．
