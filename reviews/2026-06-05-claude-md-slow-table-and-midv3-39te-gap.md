---
title: "CLAUDE.md [SLOW] テスト表の更新 (mid_v2 削除で 6 件消失) + mid_v3 39te ply24 gap の follow-up 記録"
date: 2026-06-05
status: applied
applied_in: af9cbcd
crate: maou_shogi
applies_to: CLAUDE.md
---

# CLAUDE.md [SLOW] テスト表更新 + mid_v3 39te ply24 gap

## Trigger
mid_v2 廃止（[[2026-06-05-deprecate-mid-midv2-pns]] Phase 1, v2.0.0）で，CLAUDE.md
「重いテスト (Rust dfpn) — release ビルド必須」表に載る **8 件中 6 件が削除**された
（`solve_via_v2` ベースの研究/診断/soundness guard）．CLAUDE.md は silently 編集禁止のため，
本 review で更新を提案する．

## 提案 1: CLAUDE.md の `[SLOW]` テスト表を更新

### 削除された行（tests.rs から消失済 — 表からも除去）
| テスト名 | 消失理由 |
|---|---|
| `test_tsume_39te_ply24_mate15_regression` | mid_v3 で Mate(27)≠Mate(15) → 削除（gap は下記） |
| `test_tsume_39te_ply24_mate15_soundness_depth25` | mid_v3 で PV mismatch → 削除 |
| `test_tsume_39te_ply2_no_false_nomate` | mid_v3 で 10M nodes NoCheckmate → 削除 |
| `test_tsume_39te_backward_1m` | mid_v2 逆順解析研究テスト → 削除 |
| `test_ply24_diagnostic` | mid_v2 診断テスト → 削除 |
| `test_ply24_tt_sharing_effectiveness` | mid_v2 TT 共有研究テスト → 削除 |

### 存続する行（2 件）
| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_counter_check_diagnostic` | - | 診断用ログ出力（v2 非依存） |
| `test_no_checkmate_counter_check_probe` | 10M nodes | ノード予算プローブ（v2 非依存） |

### 追加提案: mid_v3 の SLOW テストを表に載せる
| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_mid_v3` | - | mid_v3 1te/3te/29te canonical（18,539 nodes / mate-29 / STRICT Some(29)） |
| `test_mid_v3_counter_check_example` | - | 逆王手詰将棋 mate-7 健全性 |

→ CLAUDE.md の該当表（`## 重いテスト` 節）をこの内容に差し替え．`cargo test --release ... --ignored`
の運用記述はそのまま．

## 提案 2: compass invariant の更新
- 「mid/mid_v2 は触らない・修正不要」→ **「mid/mid_v2 は削除済（local_expansion.rs = 共有 LE 型のみ）」**．
- 「mid_v2 SLOW テスト退行は許容」invariant は **削除**（テスト自体が消えた）．

## 提案 3 (follow-up 課題記録): mid_v3 39te ply24 gap
mid_v3 は **29te を 18,539 nodes で完璧に解く**が，**39te ply24 subproblem は解けない**:
1. `find_shortest(false)` を mid_v3 が無視 → 非最短 mate（Mate(27)）を返す．
   - 真因候補: mid_v3 の IDS は threshold 成長のみで mate-distance 最小化 path を取らない．
2. ply2 hard 局面で 10M nodes 使い切り → **NoCheckmate を返す**（Unknown であるべき）．
   - budget 切れを NoCheckmate と報告する箇所がある疑い（soundness 観点で要確認）．
これは「mid_v3 を production / 汎用 39te に広げる」前に潰すべき gap．29te 特化の現状では露呈しない．

## 影響 / リスク
- ドキュメント更新のみ（コード影響なし）．
- 提案 3 は調査課題の記録であり，本 review では実装しない．

## 検証
- なし（ドキュメント）．適用後 CLAUDE.md の表が tests.rs の実態と一致することを目視確認．
