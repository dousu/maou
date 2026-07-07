---
status: applied
applied_in:
date: 2026-07-08
target: [docs/design/position-search/index.md, CLAUDE.md]
risk: low
reversibility: trivial
---

# 1局面探索エンジンの設計草案を docs/design/position-search/ に新設

## Trigger
user 指示 (2026-07-08): 「このセッションで検討した1局面探索機能の設計草案を
作成してください。今後のセッションでは実装してうまくいったら、この設計草案を
修正しながら進めていくことを想定しています」— 明示指示のため approve 済み扱いで
起票する．設計内容の出典は worklog/2026-07-07-211609.md (charter) と
worklog/2026-07-08-065408.md (設計調査 + 第一マイルストーン実装)．

## Proposed change
- `docs/design/position-search/index.md` を新規作成．内容: 目的/スコープ，
  crate 配置と wheel 可搬性，PUCT MCTS コア (実装済み仕様: 親手番視点統計・
  virtual loss・バッチ収集・展開同期・固定プール)，Evaluator 境界と ONNX 契約，
  予算 API，最終手選択 (暫定 + 未決)，GC 方針，詰み探索統合方針 (ルート並行
  dfpn / 葉詰み見送り / AND-OR 伝播)，千日手方針，ベンチ計測規律とベースライン，
  マイルストーン + 未決事項一覧．各節に実装済み/設計方針/未決を明記する
  living document 形式．
- `CLAUDE.md` Documentation Links に
  `| 1局面探索エンジン設計 | docs/design/position-search/index.md |` の行を追加．

## Motivation
campaign の設計判断・未決事項が worklog (immutable) と会話にしか存在せず，
次セッション以降の実装で参照・更新できる committed な設計文書がなかった．
user が「実装しながら修正していく」運用を明示した．

## Alternatives considered
1. 設計が固まるまで起票を遅延 (前 checkpoint の判断) — user が living document
   運用を指示したため撤回．未決を未決と明記すれば churn は許容できる．
2. 単一ファイル docs/design/position-search.md — tsume-solver と同様に将来
   GC/evaluator 等の詳細 doc へ分割する見込みがあるためディレクトリ + index.md 形式．

## What this enables
- 次セッションが会話履歴なしで設計前提 (統計の意味論，Evaluator 契約，
  binding 制約) を参照できる．
- 「実装済み/方針/未決」のタグにより，草案更新の差分が実装の進捗記録になる．

## What this constrains
- maou_search の公開仕様を変える実装は本ドキュメントの同節更新を伴う
  (docs 同期義務の範囲内)．

## Rollback plan
docs/design/position-search/ を削除し CLAUDE.md の追加行を revert するだけ
(コード非依存)．
