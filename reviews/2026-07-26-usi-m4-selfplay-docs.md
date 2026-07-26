---
title: USI M4 (selfplay / OpeningScript / MaxMovesToDraw in-search) のドキュメント反映
date: 2026-07-26
status: pending
target:
  - docs/commands/selfplay.md
  - docs/commands/usi.md
  - docs/design/usi-engine/index.md
risk: low
reversibility: easy
---

# 提案: USI M4 実装のドキュメント反映 (commands + design 節ステータス)

## 背景

USI エンジン campaign の M4 (OpeningScript / 自己対局 driver / MaxMovesToDraw
in-search 引き分け終端化) を実装した (feat/usi-m4，`src/`/`rust/` 側は本
レビュー対象外)．CLAUDE.md の「CLI コマンド変更時は docs/commands/ を更新」
「新コマンドは docs/commands/<command>.md を新設」の MUST に従う変更と，
design doc の節ステータス更新 (起草時の「全節が設計方針」注記が M1-M4 実装
完了後は不正確) を提案する．

## ドキュメント変更内容 (本レビューの承認対象)

### `docs/commands/selfplay.md` (新設)

`maou selfplay` コマンドのドキュメント (Overview + CLI options + Output +
Example の既存形式)．要点:

- in-process 自己対局 (agent 2 個直接駆動・評価器プロセス内共有)
- 終局判定は USI 対局と同一実装 (宣言/千日手 4 回 + 連続王手分類/最大手数/
  投了/非合法手)
- JSONL 棋譜出力 + サマリ表示，並列度・多様化 (ランダム序盤/script) の説明
- HCPE 生成接続は次 campaign (スコープ外) の明記

### `docs/commands/usi.md` (更新)

- 冒頭の到達状態を「M3 まで」→「M4 まで」に更新: OpeningScript / in-search
  MaxMovesToDraw 終端化を追加し，「未実装: go mate のみ」へ．selfplay.md への
  相互リンクを追加．
- CLI options 表に `--opening-script` を追加，`--max-moves-to-draw` の説明に
  in-search 終端化を追記．
- USI options 表に `OpeningScript` (string) を追加．

### `docs/design/usi-engine/index.md` (節ステータス更新 — 承認時に適用)

冒頭の「本ドキュメント起草時点では全節が設計方針 (マイルストーン M1-M4 で
実装)」の直後に 1 行追記する:

```
> 2026-07-26: M1-M4 実装完了 (M1=#393/#394, M2=#395, M3=#397, M4=本 PR)．
> 未決事項 4 (in-search) は実装済み・未決 5 (バッチ aggregator) は計測待ち，
> 未決 1-3 は未決のまま．
```

節ごとの個別マークは行わない (M 対応は §12 の表が既に持つため，冒頭 1 行で
十分と判断．個別マークが必要なら指摘を受けて拡張する)．

## リスクと理由

- **risk: low** — ドキュメントのみ．コード側の挙動説明は実装済みテスト
  (Rust 83 + Python selfplay/usi CLI) に対応する記述で，新規の設計決定を
  含まない．
- **reversibility: easy** — ドキュメント編集の取り消しのみ．

## ロールバック

selfplay.md の削除と usi.md / index.md の該当節の巻き戻しのみ．
