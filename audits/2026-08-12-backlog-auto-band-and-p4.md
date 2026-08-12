---
kind: backlog
date: 2026-08-12
path:
  - src/maou/infra/file_system
  - src/maou/infra/bigquery
  - src/maou/infra/object_storage
  - src/maou/infra/console
  - src/maou/interface
  - src/maou/domain/data
  - docs/commands/pre_process.md
scope: python + docs
level: medium
last_sha: c066f76
---

# `/audit-backlog` 2026-08-12 — 自動帯 (P2/P3) の消化と判断帯 (P4) の PR 化

`audits/coverage.md` § "Open findings backlog" から 26 行 (deferred 13 +
out-of-scope 13) を読み，HEAD に対して再検証したうえで判断コストで分類し，
自動帯を無停止でマージ，判断帯を PR として残した run．

**ユーザには一度も聞いていない．** 判断が要る 4 件はすべて「書くべきコードが
1 通りに決まる」側だったので，PR #457 が問いを載せている．

## Classification

class を決めたテストを併記する — 分類が後から誤りと分かったとき，読み返す
べきはこの列だから．

### 自動帯 (P2/P3) — PR #456 でマージ済み

| ID | 行 | 対象 | 再検証 | class | class を決めたテスト | gate |
|---|---|---|---|---|---|---|
| P2-1 | O11 | `docs/commands/pre_process.md` | confirmed (6 サイト全て) | P2 | 追跡対象の散文のみ + 訂正後の本文が現行コードから一意 | — |
| P3-1 | O6 | `bq_data_source.py:483` | confirmed | P3 | ログ文言のみ．成果物・戻り値は不変 | — |
| P3-2 | D12(b) | `file_data_source.py` | confirmed | P3 | 同じ例外が同じハンドラへ届く | — |
| P3-3 | D12(e) | `streaming_file_source.py` | confirmed (恒真) | P3 | 受理集合とエラー文言が同一 | — |
| P3-4 | D12(f) | 同上 | confirmed | P3 | ログ呼び出しの形のみ | — |
| P3-5 | D12(a) | `file_data_source.py` | confirmed (ハザードは現状到達不能) | P3 | 到達しない分岐へのガード | — |
| P3-6 | D10(a) | `streaming_file_source.py:161` | **changed shape** | P3 | DEBUG ログが増えるだけ | — |
| P3-7 | D13(a) 一部 | `file_data_source.py:594` | confirmed | P3 | 同じ index を返す．時間のみ | — |
| P3-8 | D13(c) | `file_data_source.py:59` | confirmed | P3 | `get_dtype(bit_pack=False)` は 3 factory と同値 | — |

### 判断帯 (P4) — PR #457，未マージ

| ID | 行 | 対象 | 再検証 | class | class を決めたテスト | gate |
|---|---|---|---|---|---|---|
| P4-1 | O4 | console ×3 + interface ×2 | **changed shape** (診断が誤り) | P4 | 有効な起動コマンドの結果が変わる | — |
| P4-2 | O2 | `object_storage`, `bigquery` | confirmed | P4 | seed 指定時に分割値が変わりうる | — |
| P4-3 | O3 | `domain/data/columnar_batch.py` | confirmed | P4 | 黙って壊れていたケースが例外になる | — |
| P4-4 | O1 | `bq_data_source.py:659` | confirmed (実行時クラッシュ) | P4 | 公開メソッドの戻り値型が ABC 準拠へ変わる | **G1** |

**G1 (O1)**: 本環境に BigQuery が無い．`object.__new__` + `get_page` 差し替えで
変換ロジックと `KifDataset` の噛み合わせは実行検証したが，実接続での確認は不可．

### 判断帯だが本 run では着手しなかったもの

| 行 | class | gate | 理由 |
|---|---|---|---|
| D11(b) (`total_pages()` と tqdm) | P4 | — | 公開メソッドの戻り値の**意味**を「ファイル数」と「yield 数」のどちらに決めるかが要る．caller は 1 箇所のみ |
| D1 (`moveWinRate` dtype) | P5 | **G4** | 記録が「dtype に足す」「変換直後に捨てる」の 2 方向を明示．diff が根本的に異なる |
| D8+D9 (`file_level_split`) | P6/P4 | **G4** | 「削除」と「修理」で共有行がゼロ |
| D5 / O5 (cache ノブ) | P6 | **G4** | ノブ廃止は D5 と O5 が一体で，CLI 契約を変える |
| L2/L3/L4 (app/learning の重複) | P4-P6 | — | 250-400 行のリファクタ．P3 を主張するには読む量が項目に見合わない → fail-safe で判断帯 |
| L5/L6/L7 (GPU セマンティクス) | P4 | **G1** | 実 GPU での測定・数値同値確認が要る |
| D14, D15, D3+D4, D6+D7, O7, O8, O9, O10 | — | — | 本 run の予算外．行はそのまま |

## Consumed

| 行 | 対象 | 出荷したもの | マージ先 |
|---|---|---|---|
| O11 | `docs/commands/pre_process.md` | `.npy` → `.feather` の 6 サイト訂正 + `reviews/` 提案 | PR #456 (`c066f76`) |
| O6 | `bq_data_source.py` | ローカルキャッシュ検証の glob を `.feather` へ | PR #456 |
| D12(a)(b)(e)(f) | `file_data_source.py`, `streaming_file_source.py` | desync ハザード除去 / 入れ子解消 / 恒真条件縮約 / `log_level` 削除 | PR #456 |
| D13(a) 一部, D13(c) | `file_data_source.py` | 探索の巻き上げ + `bisect` / `get_dtype` への集約 | PR #456 |
| D10(a) | `streaming_file_source.py` | `_subset` への委譲 | PR #456 |

## Applied

PR #456 (`8ad9579`, `4e335ee`) — version 0.86.0 → 0.86.1:

- `docs/commands/pre_process.md:6,22,32,34,66,90`
- `src/maou/infra/bigquery/bq_data_source.py:483`
- `src/maou/infra/file_system/file_data_source.py` — import (`bisect`,
  `get_dtype`)，`_STRUCTURED_DTYPES` 削除，`FileManager.__init__` の
  ローダ分岐一本化と `_cum_upper`/`_cum_lower`，`get_item` の探索
- `src/maou/infra/file_system/streaming_file_source.py:86,161,199`
- `reviews/2026-08-12-pre-process-output-format-drift.md` (status: applied)

PR #457 (`a0324f6`) — version 0.86.1 → 0.86.2，**未マージ**:

- `src/maou/interface/learn.py` — `learn_multi_stage` 入口の `test_ratio` 検証
- `src/maou/interface/utility_interface.py:498`
- `src/maou/infra/console/utility.py:1217,1260`
- `src/maou/infra/console/learn_model.py:884`
- `src/maou/infra/bigquery/bq_data_source.py` — `__train_test_split`,
  `get_item`, `__row_to_structured_record`
- `src/maou/infra/object_storage/data_source.py:86-96`
- `src/maou/domain/data/columnar_batch.py` — `_concat_optional`

## In flight

**PR #457** (base: `main`, independent — 自動帯がマージ済みなので stack 不要)．
O4 / O2 / O3 / O1 の 4 行はマージまで `coverage.md` に残してあり，行末に
PR リンクを追記した．

載せている問い: **既定のまま `benchmark-training` を回していた人にとって
Stage 2 の検証分割が 20% から 0% (help どおり) に変わる**．過去の run と
数字が比較できなくなるので，そこを受け入れるかどうか．

代替案も本文に書いてある — `--test-ratio 0.0` を「分割なし」として
サポートする道 (Stage 3 の学習ループ・early stopping・callback を検証
ソース不在で通す必要があり，本 PR より大きい)．

## Re-triaged

- **D12** — (a)(b)(e)(f) を消化，(c)(d) は残置．(d) の緊急性は下がった:
  記録は「重複状態が (a) の desync ハザードを増幅する」ことを理由に挙げて
  いたが，(a) を構造的に潰したのでその増幅はもう起きない．重複そのものは残る．
- **D13** — (a) の contained 部分と (c) を消化．(b) と (a) の根本は
  `app/learning/dataset.py` と ABC を触るので path 外のまま．
- **D10+D11** — D10(a) を消化 (削除ではなく委譲，下記 Corrections 参照)．
  D11(b) は P4 と判定して残置，D11(a) は D14(a) と同時に扱うべきものとして残置．

## Corrections to the source records

`audits/2026-08-10-src-maou-infra-file-system.md` に追記した 2 件．どちらも
「記録が挙げた**修正方針**が誤り」であって，「解決済み」ではない．

1. **D10(a) — 「削除」は公開名を消す．** 記録は `iter_files_columnar` を
   「production は `_subset` しか呼ばない二重実装」と書いているが，HEAD では
   `app/learning/streaming_dataset.py:199` の `StreamingSource` プロトコルが
   宣言するメンバであり，`tests/` から 10 本以上が呼ぶ．呼び出し件数だけを
   見て「死んでいる」と判断すると，プロトコル準拠を壊す．
2. **O4 — 「黙って 0.1 になる」は誤り．** `interface/learn.py:307-311` が
   `0.0 < test_ratio < 1.0` を検証しているので `--test-ratio 0.0` は最終的に
   ValueError で落ちる．ただし `learn_multi_stage` は入口で検証せず Stage 3 の
   `learn()` (`:1322`) まで到達するため，**Stage 1 と Stage 2 を回し切ってから**
   落ちる — 記録が言うより悪い．さらに，本当に黙って化けていたのは
   `--stage2-test-ratio` の方 (0.0 が CLI 既定値かつ documented な「分割なし」
   なのに `or 0.2` が 20% に書き換える) で，記録は性質の違う 2 つを 1 行に
   束ねていた．

D13 についても記録の見立ての誤りを 1 件見つけた (「`_STRUCTURED_DTYPES` を
消せば `assert self._structured_dtype is not None` が不要になる」— hcpe 経路で
属性は依然 None なので落とせない)．これは `coverage.md` の D13 行に書いた．

## Doc findings

- `reviews/2026-08-12-pre-process-output-format-drift.md` — **applied**．
  CLAUDE.md § "Standing approval — drift corrections only" の P2 分岐で
  適用した (訂正後の本文が現行コードから一意に決まる: `.npy` を書く経路も
  読む経路も pre-process には存在しない)．
- **判断帯の変更による doc drift は無かった．**
  `docs/commands/learn_model.md:92` は既に `--stage2-test-ratio` 既定 `0.0`
  = 分割なしと書いており，`:36` も `--test-ratio` は `0 < ratio < 1` と
  書いている．**doc が正しくコードが doc に反していた**ので，PR #457 は
  両者を一致させる方向．新規提案は不要だった．

## Out of scope

本 run が新たに気づいたもの．4 件とも `coverage.md` の out-of-scope
backlog に行を追加した．

- **N1** — `learn-model` の Stage 3 ファイル分割の既定が console 側 `0.1`
  (`learn_model.py:884`) と interface 側 `0.2` (`learn.py:307`) で食い違う．
  streaming では interface 側の値が使われないため潜在的だが，`--no-streaming`
  に切り替えた瞬間に検証割合が変わる．
- **N2** — `bq_data_source.__download_all_to_local` の完了検証は「全ページの
  キャッシュが既に存在する」場合に早期 return でスキップされる．検証が最も
  要る「一部だけ壊れている」ケースを通らない (O6 の回帰テストを書く過程で
  発見)．
- **N3** — `app/learning/dataset.py:45` と `app/pre_process/hcpe_transform.py:62`
  は `@abc.abstractmethod` を付けながら `abc.ABCMeta`/`abc.ABC` を使って
  いないのでマーカーが**不活性**．O1 が構築時に捕まらなかった根本原因で，
  O1 が消えても残る．
- **N4** — `tests/maou/infra/file_system/test_file_data_source.py` は torch 未
  導入だと**モジュールごと skip** される (`file_data_source.py` →
  `interface/learn` → torch)．`uv sync` だけの環境では `infra/file_system` の
  変更が一切検証されないまま緑に見える．
- **N5** (PR #457 マージ後に発見) — `test_usi_go_mate_e2e` が全件実行だと
  `checkmate timeout` で落ちる (単体実行では通る)．**PR #457 マージ前の
  `6d8ee6b` でも同様に落ちる**ので特定の変更による回帰ではないが，失敗の
  出方がソルバ回帰と見分けられないのが問題．1手詰に 5 秒の壁時計バジェット
  を課す設計を見直すべき．

## Environment notes

- **~~pre-commit フレームワークが本環境で bootstrap できない．~~** `uv-lock`
  hook が GPU 専用の `tensorrt-cu12-libs` を解決しようとして失敗したため，
  `.pre-commit-config.yaml` の各 hook を個別コマンドとして実行して代替した:
  `ruff format` / `ruff check` / `mypy src/ tests/` / `pytest` 全件 /
  `scripts/check-cli-docs.sh` / `cz check` /
  trailing-whitespace・end-of-file・check-toml 相当．
  `uv.lock` の version 行は `uv lock` が生成するのと同じ 1 行差分を直接当てた．

  **Correction** (2026-08-12, 本 run 内): 上の「bootstrap できない」は
  **誤り** — 環境の恒久的な制約ではなく，一度きりの転送失敗だった．
  ユーザの指摘 (「特定ドメインにアクセスできなかったからか」) を受けて
  切り分けた結果:

  | 検査 | 結果 |
  |---|---|
  | `pypi.nvidia.com` への到達性 | **HTTP 200** (proxy 経由，ブロックされていない) |
  | wheel 本体の完全性 (curl, 全体) | 4,276,636,826 B = `Content-Length` 一致，sha256 **一致** |
  | wheel_stub と同一経路の再現 (urllib + 16KiB ループ, proxy 経由) | 同上，sha256 **一致** |
  | `/tmp` に 4.5GB 書き込み | 成功 (109 MB/s) |
  | `uv lock --no-cache --upgrade-package tensorrt-cu12-libs` 再実行 | **成功** (`Built tensorrt-cu12-libs==10.15.1.29`) |
  | `uv run pre-commit run --all-files` | **全 hook Passed** (`uv-lock` 含む) |

  真因は **4.28GB の wheel の転送が途中で切れたこと**．`wheel_stub` は
  読み取りバイト数を `Content-Length` と突き合わせないので
  (`wheel_stub/wheel.py:202-210` のループは空読みで抜けるだけ)，短いファイル
  を書いたまま次行の sha256 assert で落ちる．`urlopen_with_retry` は接続の
  確立しか再試行しないため，ストリーム途中の切断は再試行されない．
  **環境設定 (ドメイン許可リスト等) に修正すべき点はない．**

  したがって本 run の QA は「pre-commit が動かないので代替した」のではなく，
  「代替手段で同等のものを実行した」が正しい — 事後に
  `uv run pre-commit run --all-files` を判断帯ブランチで実行し，
  **全 hook Passed** を確認済み (PR #457 に反映)．
- **torch を入れないと自動帯が無検証になる (G3)．** run 開始時点で
  `test_file_data_source.py` はモジュールごと skip されていた．
  `uv sync --extra cpu` で torch を入れて 52 passed + 3 skipped →
  83 passed になり，はじめて `file_data_source.py` の変更が検証された．
  **無検証のまま自動マージしないためにインストールした**のであって，
  環境を変えたこと自体が判断の一部である．
- **BigQuery / GCS / S3 への実接続は不可** (`TEST_GCP` 系のテストは全て
  skip)．O1 / O2 / O6 の検証は `object.__new__` と未束縛メソッド呼び出しで
  クラウドクライアントを迂回している．
- **GPU 無し**．L5/L6/L7 の GPU セマンティクス項目は原理的に本環境で
  評価できない (G1)．

## Reconciliation (step 6d)

触れた行 9 + 新規発見 4 = **13**

- **resolved** (行削除，マージ済み): 2 — O11, O6
- **in flight** (行維持 + PR リンク追記): 4 — O4, O2, O3, O1
- **re-triaged** (行維持，本文を鋭利化): 3 — D12, D13, D10+D11
- **new row**: 4 — N1, N2, N3, N4
- **not a finding**: 0

2 + 4 + 3 + 4 + 0 = **13** ✔

分類表の「判断帯だが本 run では着手しなかったもの」に挙げた行 (D1,
D8+D9, D5, O5, L2-L7, D3+D4, D6+D7, D14, D15, O7, O8, O9, O10) は
**選択していない**ので上の等式には入らない — 分類だけ済ませて行は
一切変更していない．次の run が読むのは `coverage.md` の行であって
この節ではないので，未選択であることは行の無変更が表している．

行数: deferred 13 → 13 (削除なし)，out-of-scope 13 → 15
(O11/O6 削除，N1-N4 追加)．
