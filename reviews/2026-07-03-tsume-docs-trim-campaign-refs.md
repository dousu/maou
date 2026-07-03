---
title: 設計ドキュメントから campaign 内部参照を除去 (user 直接編集)
status: applied
approved_by: user (2026-07-03 直接編集 + コミット指示)
applied_in: (docs コミット SHA を applied 化時に記入)
---

# 設計ドキュメントから campaign 内部参照を除去

## 経緯

user が docs/design/tsume-solver/ を直接編集し (「追加で修正が必要だと思ったところを
修正した」)，コミットを指示した．内容は，実装忠実な設計資料から **campaign 運用の内部
参照** (oracle-is-user 注記・worklog/compass への言及・版数の節目) を取り除く整理．

## 適用した変更 (user 編集)

| ファイル | 変更 |
|---|---|
| aigoma-optimization.md | §冒頭/§8.2 の「中合い」表現を「合駒」へ統一．「正解の oracle はユーザ」段落を削除 (KH MinLength の但し書き含む)．tsume_4 の具体例を一般化 |
| move-ordering-and-pv.md | §9-b.3 の「正解の oracle はユーザ」段落を削除・「現状」プレフィックスを外し本文へ統合 |
| index.md | 実装済み手法一覧の「(campaign 状態は compass + worklog が source)」注記を削除 |
| optimization-proposals.md | 「git 履歴と worklog/」→「git 履歴」 |
| references.md | 「開発の節目」節 (版数テーブル) を丸ごと削除 |
| search-architecture.md | 冒頭の「旧版にあった二エンジン構成…v3.0.0 で全廃」文を削除 (§2.6 に既述のため重複) |

## 不変条件・確認

- 内部リンク破損なし (「開発の節目」「oracle はユーザ」「worklog/compass」への
  dangling 参照が docs 内に残っていないことを grep で確認)．
- 実装記述・アルゴリズム説明・出典は無変更 (campaign 運用メタ情報のみ除去)．
- oracle-is-user / 無駄合い non-count のポリシー自体は `scratchpad/compass.md`
  (§🚫 VETOES #5) に binding として残るため，設計 docs から外しても運用は不変．
