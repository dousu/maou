---
title: 二歩マスク修正に伴う dfpn 39te canonical 値の docs 同期
date: 2026-07-12
status: pending
target:
  - CLAUDE.md
---

# 提案: CLAUDE.md の 39te canonical ノード数を更新

## 背景

`Bitboard::occupied_files()` の col 7 (8筋) 分割誤りにより，8筋に自歩が
あっても先手の `P*8i` (マス 71) が二歩チェックを素通りして合法生成されて
いた (maou_shogi 5.7.1 で修正)．movegen は dfpn も共用するため，受け方の
非合法応手が消えて探索順序が変化し，39te の canonical ノード数が変わった．

- 29te: 396,516 nodes — **不変**
- 39te: 17,545,528 → **17,593,615** nodes (+0.27%)
- mate-39 / STRICT Some(39) / canonical PV は**不変** (soundness 無傷)

## 提案内容

CLAUDE.md「重いテスト (Rust dfpn)」表の `test_39te_measure` 行:

```
39te canonical (17,545,528 nodes / mate-39 / STRICT Some(39) / canonical PV)
```

を

```
39te canonical (17,593,615 nodes / mate-39 / STRICT Some(39) / canonical PV)
```

に更新する．

## 根拠

- rust/maou_shogi/src/dfpn/tests.rs の canonical assert は更新済み
  (release 実測で green を確認)
- 修正自体の回帰テスト: bitboard.rs `test_occupied_files_all_squares` /
  movegen.rs `test_drop_pawn_nifu_file8_rank9` ほか
