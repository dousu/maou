---
kind: backlog
date: 2026-08-14
level: medium
path:
  - src/maou/app/learning
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
| Deferred 6 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `training_loop.py:460` のホストブロックを device 側の順序保証へ置換 + 回帰テスト 4 本 | (下記) |

## Applied

- `src/maou/app/learning/training_loop.py:457-465` — `stream.synchronize()`
  を `compute_stream.wait_stream(stream)` に置換．`compute_stream` は
  `:422` で既に保持されており，allocator 再利用ハザードは `:454` の
  `_record_stream(next_ctx, compute_stream)` が別途カバーしている．
- `tests/maou/app/learning/test_training_loop.py` —
  `TestIterateCudaOverlapOrdering` を追加 (4 本)．`torch.cuda.Stream` /
  `current_stream` / `stream` を差し替えて **CPU 上で**順序保証を固定する．
- `pyproject.toml` — `0.91.1` → `0.91.2` (`fix:` → patch)．

## Decisions asked

(step 3d の記録．本文は報告時に確定．)

## In flight

(本文は報告時に確定．)

## Re-triaged

- **B-2 (N6-2)** — 記録の処方「基底 `iter_batches_df` を HCPE 専用と明記
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

- **B-3 (D14(b))** — G2 の規模を確定した．`FileDataSource` を
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

なし．`docs/performance.md:34-39` は `_iterate_cuda_overlap` を高レベルに
述べるだけで，ホストブロックの有無には触れていないため drift しない．
`docs/stage2-speed-investigation.md` は当時の調査記録であり挙動の主張を
していない．

## Out of scope

なし (新規所見なし)．

## Environment notes

- `torch` はコンテナに未導入だったため `uv sync --extra cpu` で導入した
  (前 run と同じ手順)．
- **GPU は無い．** B-1 の回帰テストは `torch.cuda.Stream` /
  `current_stream` / `stream` を差し替えて CPU 上で*順序保証の呼び出し*を
  固定するもので，**実 GPU 上の性能改善は測っていない**．Deferred 6 の
  G1 は 2026-08-14 の設計判断で「意味論的に自明な等価変換なので測定は
  正しさの条件ではない」として retire 済みであり，本 run はその判断に
  従っている．
