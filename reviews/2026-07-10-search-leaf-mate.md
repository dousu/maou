---
status: pending
applied_in:
title: maou search に leaf-mate オプション追加 (docs/commands/search.md 同期)
target: docs/commands/search.md
---

# maou search に leaf-mate オプション追加 (docs/commands/search.md 同期)

## Trigger

campaign「1局面探索機能」dfpn リソース戦略トラックで，MCTS の各葉に短手詰み
探索 (leaf-mate) を追加した (dlshogi の葉ノード短手数詰み探索相当)．
`search_board.py` に CLI オプション `--leaf-mate/--no-leaf-mate` と
`--leaf-mate-nodes` を追加し，Stats 出力に `leaf_mates` を追加したため，
durable-doc target (docs/commands/search.md) の同期義務が発生する
(CLAUDE.md Documentation ルール)．

## Proposed change (承認後 /checkpoint-context step 5 で適用)

docs/commands/search.md を以下のとおり更新する:

1. **CLI options 表**に 2 行追加 (`--root-dfpn` 行の直後):
   - `--leaf-mate/--no-leaf-mate` | default off | Run a short df-pn mate
     search at each MCTS leaf. When a mate is proven the leaf is marked won
     and propagated through the tree (dlshogi-style leaf mate search).
   - `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn
     call. Smaller = cheaper and restricts to shorter mates.

2. **Overview** の探索機能列挙に「optional per-leaf short mate search
   (`--leaf-mate`)」を root-parallel dfpn と並べて追記．

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
  維持する．

## Rollback plan

docs/commands/search.md の追加分を削除し，コード側は maou_shogi 5.6.0 /
maou_search 0.10.0 / maou_rust 0.12.0 / maou 0.26.0 の leaf-mate 関連
コミットを revert する．
