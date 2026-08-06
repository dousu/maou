---
title: 自己対局の濃い教師を新レコード型でなく前処理出力スキーマへ載せる
date: 2026-08-06
status: applied
applied_in: TBD
target:
  - docs/design/training-quality/index.md
risk: low
reversibility: easy
---

# 提案: §3.3「新しいレコード型が要る」を撤回し，既存の前処理出力スキーマへ載せる

## 背景

`docs/design/training-quality/index.md` §3.3 は
「HCPE は `bestMove16` + `gameResult` + `eval` しか持てないため，
visit 分布と探索値を載せる**新しいレコード型**が要る」と書いていた．

user 指摘: **HCPE は共通規格なのでカラム構成は変えたくない．前処理の中で
実施するのはどうか．拡張しないと性能上の問題が生じるのか．**

調査した結果，**指摘のとおりで，かつ元の記述が誤っていた**．

## 事実 1: 前処理出力スキーマが既に必要な形をしている

| | 形式 | 中身 |
|---|---|---|
| HCPE (`.feather`) | `hcp / eval / bestMove16 / gameResult / id / partitioningKey / ratings / endgameStatus / moves` | 1 手 + 1 勝敗 |
| **前処理出力** | `id / boardIdPositions / piecesInHand / **moveLabel List(Float32) 1496** / moveWinRate List(Float32) 1496 / bestMoveWinRate / **resultValue Float32**` | **1496 次元の密な分布 + スカラー値** |

visit 分布は `moveLabel` に，探索値は `resultValue` にそのまま入る．
**HCPE を通す必要がそもそもない**ので共通規格は無傷である．

損失側も変更不要．policy は KLDiv で分布を相手にし，value は
`BCEWithLogitsLoss` (`src/maou/domain/loss/loss_fn.py`) で `[0,1]` の実数を
相手にするので，どちらもソフトターゲットを受け付ける．

## 事実 2: 格納コストは 2.83 倍だが性能問題にならない (実測)

held-out 878 局を `maou pre-process` で実際に変換し，20,000 行で測った (LZ4)．

現状 `moveLabel` の非ゼロは**平均 1.02 個** (最大 9) であり，
174 B/row という小ささはこの疎性を LZ4 が潰した結果である．
visit 分布 (非ゼロ ~83) では圧縮が効かない．

| | B/row | 倍率 | 38.5M 行での概算 |
|---|---|---|---|
| 現状 (ほぼ one-hot) | 176 | 1.00x | 6.8 GB |
| **visit 分布** | 497 | **2.83x** | **19.1 GB** |
| `moveLabel` も `moveWinRate` も密 | 819 | 4.66x | 31.5 GB |
| visit 分布 + `moveWinRate` 削除 | 462 | 2.63x | 17.8 GB |
| 同 zstd | 266 | 1.51x | 10.2 GB |

- 学習時の読み出しは 7.1 it/s × batch 2048 = 14,500 行/s ⇒ **7.2 MB/s**．
  どのディスクでも余裕がある
- ローダーは現状も `List(Float32)` を `(N, 1496)` の密テンソルへ展開するので
  **バッチのテンソル形状も GPU メモリも変わらない**
- 混合コーパスは相性が良い．自己対局行は `moveWinRate` が空，
  floodgate 行は `moveLabel` が one-hot なので互いの疎な側が圧縮を稼ぐ

## 変更内容 (`docs/design/training-quality/index.md`)

- §3.3 を「新レコード型の設計」から
  **「自己対局の出力を `pre-process` の第 2 の入力形式として受ける」**へ書き換え．
  上記 2 表と「HCPE も前処理出力もスキーマを変えない」を明記
- **floodgate と自己対局を別々に `pre-process` する**方針を明記．
  出所を示す列すら要らず，次が自動的に解ける:
  - `--position-count-threshold` のフォールバックを自己対局に適用しない
    (探索値のほうが良質なのでフォールバックは有害)
  - `resultValue` の意味が 2 種類になっても別ファイルなので学習側は区別不要
  - Zobrist 集約が自己対局レコードを floodgate の one-hot と混ぜない
- §3.4 に「`--stage3-data-path` は単一ディレクトリを glob するので
  混合比は行数で決まる」を追記

## 検討したが採用しなかった案

- **HCPE にカラムを追加する**: 共通規格を壊す (user 判断)．
  前処理出力で足りるので技術的にも不要
- **出所を示す `source` 列を前処理出力へ追加する**: 別ディレクトリで
  別々に前処理すれば同じ効果が得られるのでスキーマ変更は不要
- **疎表現で格納する**: 2.83 倍を 1.5 倍程度に減らせる余地はある
  (zstd で 1.51x) が，19 GB は問題にならないので**先に測ってから判断する**

## リスク

- **低**: ドキュメントのみの変更．コードは未着手
- 格納コストの実測は 878 局 (83,536 行) の 20,000 行標本による外挿である．
  実コーパスの合法手数分布が違えば倍率は動く
