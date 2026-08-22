---
status: applied
applied_in: e243614
date: 2026-08-22
target: [docs/design/game-analysis/gui.md, docs/commands/analyze_gui.md]
risk: low
reversibility: easy
---

# ワークベンチ実装と docs の残差 2 件を docs 側に寄せる

## Trigger

`reviews/2026-08-22-analyze-gui-workbench.md` (applied) の反映後に
`/code-review high` を回したところ，**ワークベンチの実装と gui.md の
記述がまだ 2 点食い違っている**ことが分かった．どちらも「実装で埋める /
docs を直す」の二択で，user が後者を選んだ (2026-08-22)．

| 残差 | docs の記述 | 実装 |
|---|---|---|
| グラフ縦軸 | §6「縦軸 4 択: 先手/後手 × 勝率/評価値」 | 先手視点固定の 2 択 (勝率/評価値) |
| 候補手の列 | §7「順位/日本語表記/訪問数/勝率/prior/詰み確定」 | 順位/指し手/訪問数/勝率 の 4 つ |

## 提案

### 1. §6 — 縦軸は先手視点固定の 2 択とする

後手視点は先手視点の鏡映であり，同じ 1 本の曲線を読み替えるだけである．
3 カラム化で右レールの操作子が増えたため，**ワークベンチではトグルを
増やさない**ことを優先する (user 決定 2026-08-22)．

- 変換関数 (`sente_winrate` / `sente_eval_cp`) と単体テストは残す —
  レポート JSON が手番視点である以上，先手視点化の変換自体は必要
- あわせて，x 軸が **ply ではなくスナップショット番号**であること
  (レポートの `positions[i].ply` は «その手を指す直前の局面» なので
  `ply - 1` に置く) を明記する．現在位置の縦線・● / ★・`data-scrub` の
  クリックがすべて同じ軸に乗る根拠になる
- `docs/commands/analyze_gui.md` の «selectable perspective (sente /
  gote × …)» も同じ内容に直す

### 2. §7 — prior / 詰み確定値は行のホバーで出す

右レール (約 300px) で 6 列は詰まるため，常時表示は 4 列に留め，
**prior と確定値は候補手行の `title` (ホバー)** で拾えるようにする
(user 決定 2026-08-22)．データは従来どおり
`analysis_gui.node_candidates_table` が全 6 値を返しており，捨てては
いない．

- 実装側は `_candidate_tooltip` を追加して `title` を付ける
  (`src/maou/interface/analysis_workbench.py`)
- §7 の «スライダー» は実装 (数値入力) に合わせて «数値入力» に直す

## 影響

- 記述のみ．§6 の「自動分類は延期」「悪手ジャンプリストは閾値フィルタ
  1 本」の決定には触れない．
- 後手視点トグルを将来入れたくなった場合は本提案を上書きする形で
  再提案する (変換関数が残っているので実装コストは小さい)．

## 適用

user 承認 (2026-08-22) を受けて `e243614` で docs と実装 (`title` の
追加) を同時に反映した．前段の
`reviews/2026-08-22-analyze-gui-workbench.md` は docs を別コミットに
分けたが，本提案は docs 修正が 4 箇所と小さく，実装 (ホバー) と
対で読めた方がよいので 1 コミットにまとめている．
