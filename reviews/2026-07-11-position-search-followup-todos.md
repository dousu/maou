---
status: approved
applied_in:
title: proven の CLI 露出と確定値衝突規則の docs 反映 (search.md + 設計 §8.3/§11)
target: docs/commands/search.md, docs/design/position-search/index.md
---

# proven の CLI 露出と確定値衝突規則の docs 反映 (search.md + 設計 §8.3/§11)

## Trigger

main マージ前のコードレビュー (2026-07-11) で確認された 2 点について，
当初は設計ドキュメントへ TODO として残す案だったが，小規模で完了する
ため修正まで行うと user が判断 (2026-07-11)．修正に伴う docs 同期を
本 review で固定する．

## 実施した修正 (コード側)

1. **ルート子の確定値 (`RootChildStat.proven`) の PyO3/CLI 露出**
   (maou_rust 0.16.0 / maou 0.33.0): `SearchRootChild.proven`
   (ルート視点 0/0.5/1，未確定 None) を追加し，`maou search` の
   Candidates 表示に `proven=win|draw|loss` サフィックスを付与．
   確定前の Q が高いまま残る負け確定手が bestmove と食い違って見える
   問題を説明可能にする．
2. **確定値の衝突規則を「先勝ち」と確定** (maou_search 0.15.2):
   履歴非依存の詰み探索 (root-dfpn / leaf-mate) と千日手 1 回近似は，
   詰み筋が対局履歴との再出現を跨ぐ稀な局面で同一ノードに異なる確定値を
   出し得る．確定値は伝播後に覆せないため `try_mark_proven` の CAS
   先勝ちで確定し，一意性を仮定していた過強な debug_assert を撤去．

## Proposed change (docs — 承認後に適用)

1. **docs/commands/search.md**: Outputs の Candidates 例に
   `proven=win` を追加し，`proven=win|draw|loss` サフィックスの意味
   (ルート視点，winrate は確定前平均のまま，負け確定手は全滅時以外
   bestmove に選ばれない) を追記．
2. **docs/design/position-search/index.md §8.3**: 確定値衝突の
   先勝ち規則と，マーク/伝播スキャンの SeqCst 化 (store-buffering で
   親確定を取りこぼす race の解消，maou_search 0.15.1) を明文化．
3. **同 §11 実装状況**: proven 露出の行を追加．

## Rationale

- proven 露出は「負け確定除外」(§6，maou_search 0.15.0) の挙動を
  CLI から観察可能にする自然な補完で，出力 1 行の追記のみ (破壊なし)．
- 先勝ち規則は「近似 (§9) はエンジン自身の千日手意味論であり誤りでは
  ない」ため，どちらが先に確定しても健全．dfpn 優先 (上書き) は確定値
  の不変性を壊し伝播済みの値と矛盾するため採らない．

## Result

user 承認 (2026-07-11，会話): 「reviews/ は approve するので小規模な
変更で終わるなら todo じゃなくて修正まで完了させてください」→ 修正
まで完了する方針に切り替えて適用．
