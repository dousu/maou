---
kind: backlog
date: 2026-08-14
level: medium
path:
  - src/maou/app/learning
  - src/maou/app/pre_process
  - src/maou/infra/file_system
  - docs/commands
  - audits
last_sha: 97dfb2a
---

# `/audit-backlog` 2026-08-14 (3 巡目) — `wait_stream` の出荷と，記録の処方が誤りだった 1 件

**backlog に G4 の行が 1 つも無い状態で始まった最初の run** である．過去
3 run の設計判断で 8 行すべての G4 が retire され，残るのは「決定済み，
あとは作業」か「G1/G2/G3 が環境・結合で塞いでいる」かのどちらかになった．

本 run はそのうち **Deferred 6 (P4，G1 retire 済み)** を実装・出荷し，
**N6-2 の記録された処方が誤りである**ことを再検証で突き止めて，向きを
問い直した．

## Classification

自動帯 (P1-P3) は **空** — 11 run 連続．G4 が付いた行は **ゼロ**．

| ID | backlog 行 | 対象 | 再検証 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|---|---|
| B-1 | Deferred 6 | `app/learning` | confirmed (行番号一致) | **P4** | 挙動は観測可能に変わる (ホストが止まらない) が，既存データも既存呼び出しも壊れない | G1 は 2026-08-14 に retire 済み |
| B-2 | N6-2 | `app/pre_process` | **changed shape** | **P6 または P3** (向き次第) | 分岐で異なる | 向き (新規) |
| B-3 | D14(b) | `infra/file_system` | confirmed，行移動 | **P6** | 公開 ABC の継承が消える | **G2** |
| B-4 | O9 | `infra/bigquery` | confirmed (行番号一致) | **P4** | 同じ `page_num` で返る行が変わる | **G1** + キー列未決 |
| B-5 | D13 | `infra/file_system` | confirmed，行移動 | **P4/P6** | — | **G2** |
| B-6 | Deferred 2 | `app/learning` | confirmed，行移動 | **P4** | — | **G3** |
| B-7 | Deferred 5 | `app/learning` | confirmed (行番号一致) | **P4** | — | **G1** |
| B-8 | Deferred 7 | `app/learning` | confirmed (行番号一致) | **P4** | — | **G1** |

再検証: **confirmed 7 / changed shape 1 / stale 0**．

## Consumed

| 行 | 由来 | 出荷したもの | PR |
|---|---|---|---|
| Deferred 6 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `training_loop.py:460` のホストブロックを device 側の順序保証へ置換 + 回帰テスト 4 本 | [#503](https://github.com/dousu/maou/pull/503) |
| N6-2 | [2026-08-13 scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md) | 基底 `iter_batches_df` を abstract 化し，HCPE 決め打ちの本体を `StreamingHcpeDataSource` へ移設 + 回帰テスト 2 本 | [#503](https://github.com/dousu/maou/pull/503) |
| D14(b) | [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) | `FileDataSource` から `preprocess.DataSource` の継承を削除 + テスト 11 箇所を `StreamingHcpeDataSource` へ移行 + 回帰テスト 1 本 | [#503](https://github.com/dousu/maou/pull/503) |

## Applied

- `src/maou/app/learning/training_loop.py:457-465` — `stream.synchronize()`
  を `compute_stream.wait_stream(stream)` に置換．`compute_stream` は
  `:422` で既に保持されており，allocator 再利用ハザードは `:454` の
  `_record_stream(next_ctx, compute_stream)` が別途カバーしている．
- `tests/maou/app/learning/test_training_loop.py` —
  `TestIterateCudaOverlapOrdering` を追加 (4 本)．`torch.cuda.Stream` /
  `current_stream` / `stream` を差し替えて **CPU 上で**順序保証を固定する．
- `src/maou/app/pre_process/hcpe_transform.py:102-115` — `DataSource.iter_batches_df`
  を `@abc.abstractmethod` にし，HCPE 決め打ちの既定実装 (54 行) を削除．
  クラス docstring の「抽象メソッドは 3 本」を 4 本へ訂正．
- `src/maou/infra/file_system/streaming_hcpe_source.py:155-207` — 削除した
  既定実装の本体を，唯一それを着ていた HCPE 専用クラスへ移設．
- `src/maou/infra/file_system/file_data_source.py:47-62` — `preprocess.DataSource`
  の継承を削除し，「学習経路のソースであって pre-process のそれではない」ことを
  クラス docstring に明記．`iter_batches` / `iter_batches_df` / `total_pages` は
  ABC 非経由の呼び出し側 (`benchmark_polars_io.py`) のため具象メソッドとして残す．
- `tests/maou/app/pre_process/test_datasource_abc.py` — 不変条件を**反転**
  (`iter_batches_df` は abstract である / 既定実装は無い)，`_MissingIterBatchesDf`
  と `test_file_data_source_is_not_a_preprocess_source` を追加．
- `tests/maou/app/pre_process/test_search_value.py` — `_NoDataSource` に
  `iter_batches_df` を追加．
- `tests/maou/{app/pre_process/test_hcpe_transform,integrations/test_app_hcpe_transform,integrations/test_convert_and_preprocess}.py`
  — `PreProcess(datasource=FileDataSource(..., array_type="hcpe"))` の
  **11 箇所**を `StreamingHcpeDataSource(file_paths=...)` へ移行．
  `array_type="preprocessing"` の `local_datasource` は出力の読み出しに
  `FileDataSource` の具象 `iter_batches()` を使うだけなので**据え置き**．
- `docs/commands/pre_process.md:57-62` — ローカル入力の実装クラス名を
  `FileDataSource` から `StreamingHcpeDataSource` へ訂正 (P2 drift correction)．
- `pyproject.toml` — `0.91.1` → **`0.92.0`**．`fix:` の patch 1 つと，
  ABC に abstract メソッドが増える **breaking change** (`feat!:`) を含むため．
  0.x では破壊的変更を minor に載せるのがこのリポジトリの慣行
  (`232358e` / `bdda7b5` / `d0c4984` / `fc6e968` が先例)．

## Decisions asked

`AskUserQuestion` を **1 回**，**4 問**上げた．全問回答を得た．

### Q1 — 受理 (PR #503)

> PR #503 (backlog Deferred 6) をマージしてよいですか？

選択肢: **「マージする (推奨)」** / 「現状維持」．
**ユーザは「マージする」を選択．** 副作用 (ホストが先行できるように
なるため GPU メモリのピークがわずかに増えうる，この環境では未測定) を
提示したうえでの受理である．

### Q2 — 向き (N6-2)

> 基底 `iter_batches_df` の HCPE 決め打ちをどう「HCPE 専用と明示」するか

選択肢: **(A) 基底を abstract 化 (推奨)** / (B) ヘルパへ切り出し +
docstring / (C) 記録どおり ABC を改名 / (D) 行を落とす．
**ユーザは (A) を選択．** 本 run で実装・出荷した．
(C) は再検証で誤りと判った案なので提示のうえ非推奨と明記した．

**この問いを立てた理由**: 2026-08-14 の設計判断「HCPE 専用と明示する」
は有効だが，行が書いていた実装の処方 (ABC の改名) が誤りだったため，
**同じ決定の下で実装が 2 案に割れた**．差分が材料的に異なり，外すと
レビューごと捨てることになるので，書く前に問うた．

### Q3 — 設計判断 (O9)

> 決定的ハッシュのキーを何にするか

選択肢: **行全体のフィンガープリント (推奨)** / `id` 列 /
clustering・partitioning キー優先 / `sample_ratio` とページングの併用拒否．
**ユーザは「行全体のフィンガープリント」を選択．**

- **settles**: O9 行の未決点 (i)「キー列の決め」．前 run (2026-08-14
  cache-knob-removal) が (b) 決定的ハッシュへの置換を決めた際に
  唯一残していた点である．
- **本 run では実装しない．** **G1** (BigQuery がこの環境に無い) は
  設計の回答では動かないため．行には決定と，残る (ii)(iii)(iv) を
  「通常作業」として書いた．

### Q4 — 設計判断 / scope (D14(b))

> G2 (テスト 3 ファイル 11 箇所の移行) を取り込んでよいか

選択肢: **取り込む — 1 本の PR で (推奨)** / 2 段階に分ける /
継承は残す / 行を落とす．
**ユーザは「取り込む」を選択．** 本 run で実装・出荷した
(step 4a の「隣を引き込む」解決を，ユーザの承認のうえで採った)．

### 予算に入らなかった設計判断 (次 run の待ち行列)

本 run の 4 問はすべて埋まったため，以下は問えていない．順序は
step 3d のランク付けどおり:

1. **D13** — 設計は 2026-08-14 に決定済み (`ColumnarBatch` を直接
   スライスする) なので**問う必要は無い**．残るのは G2 の作業量だけで，
   **次 run の先頭候補**．
2. **Deferred 2** — 設計は決定済み (統合する / `TrainingLoop` サブ
   クラスは戦略として注入)．G3 (~600 行の等価性をこの環境で示せない) の
   扱いをユーザに問う余地がある — 「測れないまま出荷してよいか」は
   Deferred 6 で一度回答を得ている論点なので，同じ答えが効く可能性がある．
3. **Deferred 5 / Deferred 7** — G1．どちらも「意味論的に自明な等価
   変換ではない」ことが確認済みで，GPU が要る点は動かない．

## In flight

**なし．** 判断帯 3 件 (Deferred 6 / N6-2 / D14(b)) はすべて同一
セッション内で回答を得たので，PR #503 は open のまま放置されず，
本 run 中にマージされた．

## Re-triaged

- **B-2 (N6-2)** — *(本 run で消化済み — 以下は再検証時点の記録)* 記録の処方「基底 `iter_batches_df` を HCPE 専用と明記
  して**改名する** (P6)」は**誤り**．再検証で，`iter_batches_df` の 4 実装
  のうち **3 つが汎用**であることが判った:
  - `FileDataSource.iter_batches_df` (`file_data_source.py:575`) は
    `array_type` で hcpe / preprocessing / stage1 / stage2 を分岐する．
  - `ObjectStorageDataSource.iter_batches_df` (`data_source.py:480`) は
    hcpe / preprocessing でスキーマを切り替える．
  - `BigQueryDataSource.iter_batches_df` (`bq_data_source.py:737`) は
    DataFrame をそのまま返す．
  - HCPE 決め打ちなのは**基底の既定実装**
    (`hcpe_transform.py:95-148`，スキーマ取得は `:114-118`) だけで，
    それを着ているのは `StreamingHcpeDataSource` 1 つ (HCPE 専用クラス
    なのでそこでは正しい)．

  ABC のメソッド名を HCPE 名へ改名すると，汎用の 3 実装と
  `docs/rust-backend.md:680,701,714,725,730` が誤った名前になる．
  改名すべきは**契約**ではなく**既定実装**だった．よって「HCPE 専用と
  明示する」という 2026-08-14 の設計判断は有効なまま，その実装の向きが
  2 案に割れる．

- **B-3 (D14(b))** — *(本 run で消化済み — 以下は再検証時点の記録)* G2 の規模を確定した．`FileDataSource` を
  `PreProcess(datasource=)` に渡している箇所は **3 ファイル 11 箇所**
  (`test_hcpe_transform.py:92,146,177,210,280,305,331` /
  `test_app_hcpe_transform.py:190,264` /
  `test_convert_and_preprocess.py:231,328`)．記録が挙げていた
  `test_convert_and_preprocess.py:430,534` は **BigQueryDataSource /
  S3DataSource** であって `FileDataSource` ではない．また各テストの
  `local_datasource.iter_batches()` は `FileDataSource` の具象メソッドを
  呼ぶだけなので，継承を外しても**実行時には壊れない** — 破れるのは
  `datasource=` の型 (mypy) の側だけ．

- **B-5 / B-6 / B-7 / B-8** — 行番号のみ更新．結論は不変．

## Corrections to the source records

- [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md)
  N6-2 — 提案された修正 (ABC メソッドの改名) が誤りであった旨を追記．

## Doc findings

- [`reviews/2026-08-14-pre-process-local-datasource-drift.md`](../reviews/2026-08-14-pre-process-local-datasource-drift.md)
  — `docs/commands/pre_process.md:58` が pre-process のローカル入力を
  「local `FileDataSource`」と書いていたが，CLI が構築するのは
  `StreamingHcpeDataSource` (`console/pre_process.py:533,536`)．
  **この drift は本 run の変更より前から存在した**が，D14(b) で
  `FileDataSource` が `preprocess.DataSource` を失ったことで「古い」だけ
  でなく型としても成り立たない記述になった．
  **P2 判定**: 訂正後の本文はコードから一意に決まる (ローカル経路が
  構築するクラスは 1 つだけ) ので **drift correction**．CLAUDE.md の
  standing approval により本 run で適用し `status: applied` にした．
- `docs/rust-backend.md:673-680,701,714,725,730` の
  `FileDataSource.iter_batches_df()` の例は **drift ではない** —
  当該メソッドは具象メソッドとして残っており，ABC を経由しない直接
  呼び出しは従来どおり動く．

（以下は Deferred 6 について）`docs/performance.md:34-39` は `_iterate_cuda_overlap` を高レベルに
述べるだけで，ホストブロックの有無には触れていないため drift しない．
`docs/stage2-speed-investigation.md` は当時の調査記録であり挙動の主張を
していない．

## Out of scope

なし (新規所見なし)．

## Reconciliation (6d)

触れた項目 + 新規発見 = **8** (backlog 8 行 + 新規 0)

- **resolved** (行を削除，fix は同一 PR に同居): **3** — Deferred 6 / N6-2 /
  D14(b)．単一 PR の run なので 6a の分離可能性テストにより削除は PR の中．
- **in flight**: **0**
- **decided** (行は残し，設計判断を書き込んだ．人間待ちではない): **1** — O9
  (キーは行全体のフィンガープリント．G1 のため未実装)
- **re-triaged** (行保持・文面を鋭くした): **4** — D2 / D13 / Deferred 5 /
  Deferred 7 (いずれも行番号の更新のみで結論は不変)
- **new row**: **0**
- **not a finding**: **0**

`3 + 0 + 1 + 4 + 0 + 0 = 8` ✓

backlog 行数: **8 → 5**．うち**人間待ちの行はゼロ** (G4 は 1 つも無い) で，
5 行すべてが「決定済みだが G1/G2/G3 が塞いでいる」状態にある．

## Environment notes

- `torch` はコンテナに未導入だったため `uv sync --extra cpu` で導入した
  (前 run と同じ手順)．
- **GPU は無い．** B-1 の回帰テストは `torch.cuda.Stream` /
  `current_stream` / `stream` を差し替えて CPU 上で*順序保証の呼び出し*を
  固定するもので，**実 GPU 上の性能改善は測っていない**．Deferred 6 の
  G1 は 2026-08-14 の設計判断で「意味論的に自明な等価変換なので測定は
  正しさの条件ではない」として retire 済みであり，本 run はその判断に
  従っている．
