---
title: fetch-floodgate コマンド追加に伴う docs/commands 新設
date: 2026-07-15
status: applied
applied_in: fddc4be
# user 承認 2026-07-15 (セッション限定のドキュメント編集包括承認)
target:
  - docs/commands/fetch_floodgate.md
---

# 提案: `maou fetch-floodgate` の CLI ドキュメント新設

## 背景

floodgate からの棋譜取得をユーティリティコマンド
`maou fetch-floodgate` として実装した (user 指示，本番学習パイプライン
でも利用)．CLAUDE.md の MUST「新 CLI コマンド追加時は
docs/commands/<command-name>.md を作成」に従い，ドキュメントを新設する．

floodgate 側の実地調査結果 (2026-07-15):

- 年次 7z の配布 URL は `x/wdoor2025.7z` (404) から
  `archive/wdoor2025.7z` へ移動していた (user 提供情報で確認)
- `archive/` には 2014-2026 の .7z と 2011/2012 の .tar.xz がある．
  当年 (2026) 分も日次更新されている
- 7z 内部は `YYYY/<ファイル名>.csa` のフラット構造 (日別ディレクトリ
  なし)．対局日はファイル名末尾タイムスタンプから導出する
- 日別リスティング `x/YYYY/MM/DD/` は従来どおり (HTTPS 化のみ)

## 提案内容

### docs/commands/fetch_floodgate.md (新規)

既存フォーマット (build_engine.md 型: Overview / Requirements /
CLI options / Execution flow / Validation and guardrails /
Example invocation / Implementation references) に従い作成:

- 2 戦略 (daily = 日別クロール / archive = 年次アーカイブ展開) と
  auto (32 日以上で archive + 未収録日の daily 補完) の説明
- **壊れやすさの明示**: floodgate の非公式公開仕様依存であること，
  空振り時は FloodgateStructureError で非ゼロ終了すること
- optional extra `fetch` (py7zr) の要件 (daily と .tar.xz は標準
  ライブラリのみ)
- resumable (既存スキップ + .part 書込) と hcpe-convert への接続例

## 根拠

- 実装: src/maou/{app/fetcher,interface,infra/http,infra/console} +
  tests 47 件 green (実ネットワーク e2e: 2025-01-05 の 289 局を取得し
  既存コーパスと bit-identical を確認)
- CLAUDE.md MUST (Documentation) が新 CLI コマンドのドキュメント作成を
  要求している
