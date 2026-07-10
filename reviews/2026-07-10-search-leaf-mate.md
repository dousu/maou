---
status: pending
applied_in:
title: maou search に leaf-mate オプション追加 (docs/commands/search.md 同期)
target: docs/commands/search.md
---

# maou search に leaf-mate オプション追加 (docs/commands/search.md 同期)

## Trigger

campaign「1局面探索機能」dfpn リソース戦略トラックで，MCTS の葉の短手詰み
探索 (leaf-mate) を追加した (dlshogi の葉ノード短手数詰み探索相当)．**探索
スレッドをブロックしない非同期設計** — 探索スレッドは王手手段を持つ葉を
専用 mate スレッドへ依頼するだけで，mate スレッド (余剰 CPU) が df-pn を
回し詰みを証明したら葉を勝ち確定にして AND-OR 伝播する (探索 NPS 不変)．
`search_board.py` に `--leaf-mate/--no-leaf-mate`・`--leaf-mate-nodes`・
`--leaf-mate-threads` を追加し，Stats 出力に `leaf_mates` を追加したため，
durable-doc target (docs/commands/search.md) の同期義務が発生する
(CLAUDE.md Documentation ルール)．

## Proposed change (承認後 /checkpoint-context step 5 で適用)

docs/commands/search.md を以下のとおり更新する:

1. **CLI options 表**に 5 行追加 (`--root-dfpn` 行の直後):
   - `--root-dfpn-nodes INT` | default `1048576` | Node budget for the root
     dfpn mate search. Larger reaches deeper mates (NN-independent) at the cost
     of more CPU.
   - `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root
     dfpn mate search (max 2047).
   - `--leaf-mate/--no-leaf-mate` | default off | Enable short mate search
     at MCTS leaves. Search threads only enqueue mate requests (never block);
     dedicated mate threads run the df-pn on spare CPU and mark proven leaves,
     so search NPS is unaffected (dlshogi-style leaf mate search).
   - `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn
     call. Smaller = cheaper and restricts to shorter mates.
   - `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate
     threads (use spare CPU cores).

2. **Overview** の探索機能列挙に「optional asynchronous per-leaf short mate
   search on dedicated threads (`--leaf-mate`)」を root-parallel dfpn と
   並べて追記．

3. **Stats fields** の記述に `leaf_mates` (leaf-mate が葉で詰みを証明した
   回数) を `proven_nodes` の後に追加．

## What this enables

- 余剰 CPU (GPU 律速時) で MCTS 非到達の短手詰みを葉で捕捉する機能が
  CLI から使え，仕様が doc に固定される．
- `leaf_mates` 統計で leaf-mate の発火回数を計測でき，性能評価
  (63手 benchmark の bootstrap 効果) の解釈に使える．

## What this constrains

- `--leaf-mate-nodes` のデフォルト 50 (dlshogi の LeafDfpnNodesLimit 相当)．
  変更時は本 doc も同期する．
- leaf-mate は `Checkmate`/`CheckmateNoPv` (= root 並行 dfpn と同じ健全な
  証明) のときだけ葉を勝ち確定にする (偽陽性を出さない) — この健全性契約を
  維持する．非同期適用時は GC 世代 (`generation`) 一致を確認して compact 後の
  無効 index への誤マークを防ぐ．
- mate スレッドは `NodePool` に触れず Arc 共有キューだけで通信する (compact が
  `&mut NodePool` を取るため; root-dfpn と同じ方針)．proof 適用は探索スレッド
  (`apply_mate_results`) が inner scope 内で行い compact と自然排他される．

## Rollback plan

docs/commands/search.md の追加分を削除し，コード側は maou_search 0.13.0 /
maou_rust 0.15.0 / maou 0.29.0 (maou_shogi 5.6.0 は既存の leaf-mate ソルバ
のみで維持可) の leaf-mate 関連コミットを revert する (PV-mate は dominated
と実証され 0.29.0 で撤去済み)．
