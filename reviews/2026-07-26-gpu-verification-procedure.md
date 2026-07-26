---
title: GPU (Colab L4) 検証手順の文書化と A/B ハーネスの CLI 露出に伴う docs 更新
date: 2026-07-26
status: applied
applied_in: 27dacbc
target:
  - docs/design/usi-engine/verification.md (新設)
  - docs/design/usi-engine/index.md
  - docs/commands/selfplay.md
risk: low
reversibility: easy
---

# 提案: GPU 検証手順 (Colab L4 / 事前ビルド wheel) の文書化

## 背景

USI campaign の残件は 3 件で，いずれも DevContainer (CPU) では閉じられない:

| 残件 | 閉じられない理由 (worklog 2026-07-26) |
|---|---|
| 未決 1 TimeStrategy 定数 | CPU 23 playouts/秒では「時計が効くが枯渇しない」regime を作れない |
| 未決 5 バッチ aggregator | CPU では `Mutex<Session>` が上限で並列スケール 0．GPU でのバッチ効果は未測定 |
| GUI 実機検証 | GUI (将棋所/ShogiGUI/ShogiHome) を動かせる環境が手元にない |

user 決定 (2026-07-26):

- **GPU を使えるのは Colab (L4) のみ**．GPU 検証は **Release `latest` の
  事前ビルド wheel** を使う (Rust ビルド不要)．検証モデルは
  `model_20260725_044443_vit-19.8m_32_fp16.onnx` (ViT 19.8M / fp16 / IR 9)．
- **GUI 実機検証は将来の課題としてドキュメント化するにとどめる**．

問題は，A/B ハーネス (`--mode horizon` 等) が **Rust example
(`cargo run -p maou_usi --example selfplay_ab`) にしかなく wheel に含まれない**
ことだった．そこで A/B と持ち時間モードを `maou selfplay` に露出する
(`src/`/`rust/` 側は本レビュー対象外 — 同 PR に含む)．

## ドキュメント変更内容 (本レビューの承認対象)

### `docs/design/usi-engine/verification.md` (新設)

USI エンジンの検証手順を 1 枚にまとめる:

1. **GPU 検証 (Colab L4，事前ビルド wheel)** — インストールは
   `docs/design/position-search/benchmarking.md` §4 を参照 (重複記述を避け
   差分のみ書く: モデル配置と `--tensorrt --cuda --batch-size` 既定)．
2. **未決 1 (TimeStrategy horizon) の手順と判定基準** — `maou selfplay
   --ab-mode horizon` の実行セル，**regime ゲート** (投了/proven で早期終了
   しない・min_think に張り付かない・終局時残り時間が初期値の大半を残さない)
   を先に確認し，外れていたら時計設定を変えてやり直す，という運用を明記．
   判定は Wilson CI + ペア t 値 + 残り時間で行う．
3. **未決 5 (バッチ aggregator) の手順と判定基準** — `--parallel 1/2/4/8` の
   playouts/秒 (wall clock) を比較し，スケールすれば aggregator 不要，
   頭打ちなら採用検討，という決定規則を明記．
4. **TRT/GPU の健全性確認** — subtree 再利用 (`--ab-mode subtree`) と
   `go mate` / ponder / keep-alive の headless smoke (一括 pipe ではなく
   「応答待ち→quit」の Python driver 断片を載せる — compass invariant)．
5. **GUI 実機検証 (将来課題)** — 環境要件 (GUI + できれば GPU) と確認項目
   チェックリスト (keep-alive 空行の扱い = 未決 2 の既定値判断 / TRT 初回
   ビルド中に readyok を待てるか / OpeningScript が実サーバ経由で消化される
   か / ponder の実挙動)．**未実施であることを明示**する．

### `docs/design/usi-engine/index.md` (更新)

- §12 未決事項の表に「現状」列を足し，1-6 の決着状況 (決着 / GPU 実測待ち /
  実機待ち) と根拠 (PR・手順書) を書く．§12 から verification.md へリンク．

### `docs/commands/selfplay.md` (更新)

- 新オプション (`--ab-mode` / `--playouts-b` / `--horizon` / `--horizon-b` /
  `--clock-ms` / `--byoyomi-ms` / `--inc-ms` / `--min-think-ms` /
  `--alternate-colors`) を CLI options 表に追加．
- Output 節に A/B サマリ (W/D/L・Wilson CI・Elo・ペア t 値・引き継ぎ訪問数・
  残り持ち時間) と throughput 行 (wall clock playouts/秒) を追記．
- 「A/B は Rust example 経由」という記述を CLI 経由に差し替える (example は
  同じ `maou_usi::ab` を使う薄いラッパーとして残る)．

## 追記 (2026-07-26，同承認範囲)

- index.md 冒頭の到達状態メモ (7c221d3 で入れた 3 行) が §12 の現状列と
  食い違うため，「3/4/6 決着・1/5 GPU 待ち・2 GUI 待ち」へ更新した．
  実装状況の説明の正確性維持であり，新しい設計判断は含まない．
- user 指示 (2026-07-26) により，Colab 手順には棋力測定ハーネスの再較正
  (`--ab-mode budget` を GPU 予算域で回し 1 doubling あたり Elo を測り直す)
  も含めた — CPU で得た「1 doubling ≈ 208 Elo」は 16→64 playouts の極低
  予算域の値で，GPU の高予算域へ外挿できないため．

## リスクと理由

- **risk: low** — ドキュメントのみ．手順は既存 (benchmarking.md §4) の
  Colab 手順の再利用で，新しい設計判断は「GUI 実機検証を将来課題として
  据え置く」という user 決定の記録のみ．
- **reversibility: easy** — 新設ファイルの削除と 2 ファイルの巻き戻し．

## ロールバック

verification.md の削除，index.md §12 と selfplay.md の該当行の巻き戻し．
