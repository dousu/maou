---
status: applied
applied_in: 552ea57
title: 設計ドキュメント §8 詰み探索統合を dfpn リソース戦略の成果で更新
target: docs/design/position-search/index.md
---

# 設計ドキュメント §8 詰み探索統合を dfpn リソース戦略の成果で更新

## Trigger

worklog/2026-07-11-021426.md — dfpn リソース戦略トラック完成 (leaf-mate async /
root-dfpn 予算 CLI / PV-mate 棄却 / 詰み探索 default-on)．設計ドキュメント
docs/design/position-search/index.md §8 は 8.2 を「第一版見送り」と記述して
おり実装と乖離した．user 指示 (/checkpoint-context arg「設計ドキュメントにも
今回の変更を反映してください」)．

## Proposed change (承認後 /checkpoint-context step 5 で適用)

docs/design/position-search/index.md を以下のとおり更新する:

1. **§8 見出し**: 「(8.1/8.3 は実装済み，8.2 は第一版見送り)」→
   「(8.1/8.2/8.3 実装済み，詰み探索はデフォルト有効)」．

2. **§8.1 root-dfpn 更新**:
   - `SearchOptions::root_dfpn` **既定 true** に (詰み探索はデフォルト有効化)．
   - `root_dfpn_nodes` **既定 2,000,000** に (旧 2^20)．**CLI 露出**
     (`--root-dfpn-nodes` / `--root-dfpn-depth`)．
   - **NN 非依存で NN 盲点の詰みを補正する**役割を明記: 41te (NN が winrate 0.28
     と誤判定するが黒に 41 手詰めがある局面) を予算 2M で 5.3s / G*7d /
     winrate 1.0 に反転させた実証を追記 (Colab A100)．
   - **find_shortest=false ゆえ探索自体は予算非依存**で最初の詰みを返す;
     予算↑で所要時間↑は **TT 確保コスト** (TT=(max_nodes*2).clamp(2^18,2^23)
     を探索前に memset; 予算に比例) である旨を注記．41te は ~1.2M ノードで
     解けるため既定 2^20 は僅かに届かず，既定を 2M にした根拠を記す．

3. **§8.2 葉ノードの詰み探索 — 全面書き換え (「第一版見送り」→「実装済み — 非同期」)**:
   - `SearchOptions::leaf_mate` (**既定 true**) で有効化．**専用 mate スレッド
     (`leaf_mate_threads`) の非同期設計**: 探索スレッドは王手手段を持つ葉 (root
     除外) を Arc 共有キューへ try-push するだけで solve せず (ブロックしない)，
     mate スレッドが `new_leaf_mate` (小 TT・find_shortest=false・`leaf_mate_nodes`
     既定 50) で df-pn を回し，worker が結果を `try_mark_proven(WIN)` で反映して
     AND-OR 伝播する (§8.3)．**探索 NPS に影響しない** (Colab: NPS 95-99% 保持)．
   - **健全性**: mate スレッドは NodePool 非参照 (compact が &mut を取るため Arc
     キューのみ)．proof 適用は worker が inner scope 内で行い compact と自然排他，
     **GC 世代 (generation) ガード**で compact 後の無効 index への誤マーク=偽証明を
     破棄．df-pn の `Checkmate`/`CheckmateNoPv` のときのみ勝ち確定 (root 並行 dfpn
     と同じ健全パス)．
   - **限界 (NN-coupled)**: leaf-mate は MCTS が詰み筋を降りることに依存する．
     NN 盲点 (41te winrate 0.28) では MCTS が詰み筋を避け leaf-mate が starve する
     → NN 盲点は root-dfpn (NN 非依存) が担当する，と役割分担を記す．
   - 旧「dfpn API は solve 毎 TT 新規確保で葉高頻度呼び出しは破綻」は
     `min_tt_entries` (小 TT) + async 専用スレッド化で解消した旨に更新．

4. **§8 に役割分担と棄却の注記を追加** (8.4 等):
   - **役割分担**: leaf-mate (MCTS が降りる narrow mate) / root-dfpn (root・NN
     盲点, NN 非依存) / 63 手級 (受けが広い長手 tsume は bounded mate search の
     範囲外 = NN の positional eval が実用解)．
   - **PV-mate は棄却 (REFUTED)**: PV 上の大予算 df-pn を実装 (a4ab7f3) したが
     Colab 実測で leaf-mate の broad coverage に dominated (29 手を leaf-mate が
     割り PV-mate 単独は未証明; NPS ~78% に低下) と判明し撤去 (2738717)．
     「深い詰みは leaf_mate_nodes を上げる方が PV-mate より常に良い」．

5. **§11 実装状況**: 詰み探索 (§8.1/8.2/8.3) を実装済み・default-on に更新，
   版数を maou_shogi 5.6.0 / maou_search 0.14.0 / maou_rust 0.15.0 / maou 0.31.0
   に更新．「主要な未決事項」から葉詰み探索 (§8.2) を外し，main マージ準備
   (feat-push トリガー削除) を残タスクに．

## What this enables

- 設計ドキュメントが実装 (詰み探索 default-on, leaf-mate async, root-dfpn 予算
  CLI) と一致し，「詰みで評価が一変するか」= YES (41te 実証) が設計に固定される．

## What this constrains

- §8 の記述は実装 (rust/maou_search) が正．今後 SearchOptions の詰み探索 default
  や leaf-mate/root-dfpn の設計を変える際は本節を同期する．

## Rollback plan

docs/design/position-search/index.md §8 の該当編集を revert (git)．コード側は
無関係 (doc のみ)．
