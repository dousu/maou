---
status: applied
applied_in:
date: 2026-07-08
target: [docs/rust-backend.md]
risk: low
reversibility: trivial
---

# rust-backend.md のファイルツリーを実構成に一致させる

## Trigger
user 指示 (2026-07-08): 「docs/rust-backend.md の maou_shogi のファイルツリー
構成が古そうです．これも修正しておいてください」— 明示指示のため
approve 済み扱いで起票する．worklog/2026-07-08-065408.md の調査でも
doc の stale (dfpn.rs 単一ファイル表記 vs 実際は dfpn/ 13 ファイル) を確認済．

## Proposed change
docs/rust-backend.md § "Rust Project Structure" のツリーを `find rust/*/src`
の実出力に一致させる:
- maou_shogi: attack/bitboard/piece/position/sfen/zobrist を追加，
  `dfpn.rs` → `dfpn/` ディレクトリ (api/solver/search//movegen//tt/ 等) に展開
- maou_io: sparse_array.rs を追加，各ファイルに説明を付記
- maou_index: src 配下 (index.rs/path_scanner.rs/error.rs) を明記

## Motivation
ツリーが 4 crate 時代 + dfpn 単一ファイル時代のまま (少なくとも dfpn
ディレクトリ化以前から未更新)．新規読者・次セッションが誤った構成を
前提にする実害がある．

## Alternatives considered
1. ツリーを削除して「find で確認せよ」にする — 一覧性が失われ，
   モジュール役割の 1 行説明を置く場所も失うため棄却．
2. crate 名だけの浅いツリーに縮退 — dfpn/ の下位構造は詰将棋ソルバー
   の設計 doc から頻繁に参照されるため，深さを保つ価値があり棄却．

## What this enables
docs から実際のモジュール構成と役割を正しく把握できる．

## What this constrains
モジュール追加/改名時にこのツリーの更新が必要 (既存の docs 同期義務の範囲内)．

## Rollback plan
docs/rust-backend.md の該当セクションを revert するだけ．
