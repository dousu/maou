---
status: applied
applied_in: 40ad115
title: 設計ドキュメント §6 最終手選択を確定基準 (robust child + 負け確定除外) で更新
target: docs/design/position-search/index.md
---

# 設計ドキュメント §6 最終手選択を確定基準 (robust child + 負け確定除外) で更新

## Trigger

user 指示 (2026-07-11): 自己対局が不可能な現環境で，ウェブ上のベスト
プラクティス調査 + 現環境で可能な検討をもとに最終手決定基準を確定する．
自己対局が可能になったらさらなる検討を行う旨を設計ドキュメントに残す．
実装は ea8b807 (maou_search 0.15.0 / maou 0.32.0) で完了済みのため，
§6「暫定仕様 — 未決」が実装と乖離した．

## Proposed change (承認後に適用)

docs/design/position-search/index.md を以下のとおり更新する:

1. **§6 見出し**: 「最終手選択 (暫定仕様 — 未決)」→
   「最終手選択 (実装済み — maou_search v0.15.0)」．本文を確定基準で書き換え:
   - 優先順位: (1) root-dfpn が詰みを証明 → 詰み手順の初手 (§8.1，最優先)．
     (2) root の勝敗が AND-OR 伝播で確定 (勝ち/引き分け) → 確定値を達成する子
     (§8.3)．(3) それ以外 → **robust child**: 負け確定 (ルート視点 proven=0)
     の手を除外して訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で
     先頭．全手が負け確定なら除外なしで同基準 (どれも同値)．乱択 (温度) なしの
     決定論的選択 (本機能は match play 相当の 1 局面探索であり，自己対局学習の
     多様性付与は上位レイヤーの責務)．
   - 根拠 (2026-07-11 web 調査 + mock 検証):
     - 主要エンジンの評価時基準は「訪問回数 argmax (robust child)」が共通土台:
       AlphaZero (評価時 τ→0 = argmax visits)，dlshogi (proven 順序 →
       move_count 最大 → nnrate tiebreak)，lc0 (terminal rank → N → Q → P)．
     - 生の max-Q (max child) は文献上最悪の基準とされ，mock でも低訪問子の
       ノイズ Q を拾う挙動を機械的に再現した (41te 局面 5k playouts で訪問割合
       10% の子を選択．release / MockEvaluator / 決定論)．
     - **負け確定の手の除外は dlshogi と同じ健全性規則**で，自己対局に依存せず
       論理的に正当化できる (これが現環境で確定できる根拠)．「勝ち確定の子の
       優先」は maou の AND-OR 即時伝播 (1 子でも負け確定なら root 勝ち確定)
       により (2) に包含される．
   - **自己対局導入後の再検討** (未決として残す — user 指示の主眼):
     現環境では強さの比較検証ができないため，以下は自己対局フレームワーク
     導入後に対局比較で再評価する:
     - **LCB (secure child) 化**: leela-zero v0.17 が採用 ("Improved root move
       selection by using Lower Confidence Bounds"; Student-t 分位 ×
       二項分散，実験値 3200 visits で 61.75% 勝率)．KataGo も採用
       (lcbStdevs=5.0，minVisitPropForLCB=0.15 — 最多訪問の 15% 以上の子に
       限り LCB 最大を採用)．一方 lc0 は不採用 (robust child + Q tiebreak) で
       強豪間でも結論が割れる．導入には per-child の value 分散統計 (二乗和)
       追加とパラメータ (z，訪問ゲート) の対局チューニングが必要．
     - **最短詰みの選好**: root-dfpn は find_shortest=false (§8.1) のため
       非最短の詰み手順を返し得る (41te で 45 手 line)．lc0 は moves-left で
       短い勝ち/長い負けを選好する．
     - 温度乱択 (KataGo chosenMoveTemperature=0.10 相当) は対局多様性が必要に
       なった時点で検討．
2. **§11 実装状況**: 「最終手選択 (robust child + 負け確定除外)」を
   ✅ 実装済み (maou_search v0.15.0，§6) として追加し，未決事項 #1
   「最終手選択 (visit 最大 vs visit フィルタ + Q 最大)」を
   「最終手選択の LCB (secure child) 化の要否 | 自己対局フレームワーク導入後に
   対局比較」に差し替える．
3. **§12 参考**: lc0 / KataGo / leela-zero (move selection の参照実装) を追記．

## 調査ソース (audit trail)

- KataGo: `cpp/search/searchresults.cpp` (LCB オーバーライド)，
  `cpp/search/searchparams.cpp`，`cpp/configs/gtp_example.cfg`
  (lcbStdevs=5.0 / minVisitPropForLCB=0.15，default on)
  <https://github.com/lightvector/KataGo>
- leela-zero: `src/UCTNode.cpp` (`get_eval_lcb` = mean − t分位 × stddev，
  `cfg_lcb_min_visit_ratio` ゲート)，release v0.17 notes，issue #2282
  (max_lcb_root 実験 61.75% @ 3200 visits)
  <https://github.com/leela-zero/leela-zero>
- lc0: `src/search/classic/search.cc` `GetBestChildrenNoTemperature`
  (terminal rank → N → Q → P; changelog に LCB 採用記録なし)
  <https://github.com/LeelaChessZero/lc0>
- dlshogi: `usi/UctSearch.cpp` `select_max_child_node` /
  `compare_child_node_ptr_descending` (IsWin 回避 / IsLose 優先 →
  move_count → nnrate) <https://github.com/TadaoYamaoka/DeepLearningShogi>
- AlphaZero の評価時 argmax-visits / 学習時 N^(1/τ) (τ=1 を 30 手,以降 →0):
  公開解説 (Oracle Developers "Lessons from AlphaZero part 3" 等)
- MCTS 最終手選択の分類 (robust/max/secure child): Roelofs "MCTS in a Modern
  Board Game Framework" 等

## What this enables

- §6 が実装 (ea8b807) と一致し，「自己対局導入後に LCB 等を再検討する」という
  再開条件つきの決定が設計に固定される (会話コンテキストに依存しない)．

## What this constrains

- 最終手選択の変更 (LCB 化・温度導入等) は自己対局による対局比較を通すことが
  前提になる (mock では強さを検証できないため)．

## Rollback plan

docs/design/position-search/index.md §6/§11/§12 の該当編集を revert (git)．
コード側 (ea8b807) は本 review と独立に revert 可能．
