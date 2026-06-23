---
title: "dfpn 純粋技術名リファクタ (v4/v3/kh マーカー除去) に伴う CLAUDE.md [SLOW] テーブル更新"
status: applied
date: 2026-06-23
branch: feat/tsume-solver
applied_in: ea09131
---

## Trigger

ユーザ指示 (2026-06-23): 「ファイル名やコードコメントを他のプロダクトや作業過程に依存しない
純粋な技術的情報のみに修正する．関数名にバージョンを入れたり，KH や Phase，〇〇作詰将棋といった
言葉を完全に削除する．いまは mid_v4 しかなくなっているので，過去の状態関係なく，単なる mid を
表現するような関数名で書き換える」「旧 mid_v2 だけが呼んでいる部分はすべて削除」「現在の mid_v4 で
ある統一後の mid を公開 API に繋ぎかえる」．

## 実装済 (rust commit; maou_shogi crate)

`rust/maou_shogi/src/dfpn/` を純粋技術名へ整理 (behavior-preserving; canonical node 数不変で検証)．

### A. dead code 削除 (旧 mid_v2/旧 tt だけが使っていた部分)
- `local_expansion.rs` (mid_v2 `MidLocalExpansion`, 1,262 行) 削除．
- `tt.rs` (旧 Dual TT `TranspositionTable`, 2,770 行) 削除．
- `profile.rs` (`ProfileStats`) 削除．
- solver.rs の never-read フィールド ~95 個 (旧 mid_v2 のみが読んでいた `depth`/`draw_ply`/`max_ply`/
  診断フィールド群) + 関連 dead メソッド/構造体 (`PostCaptureSummary`, `VisitBreakdown`, `DfPnEntry`,
  `ProvenEntry` 等) を削除．entry.rs の旧 tt proof-tag 定数群も削除．
- 計 ~5,900 行削除．元のツリーは警告ゼロだったため，削除対象は compiler が完全に同定 (削除後も
  default + 全 feature で warning 0)．

### B. const ゲートのインライン化 (常時同値を返す版番/KH 名ゲート)
- `khorder_enabled`(=true) / `v4_smprop`(=true) / `kh_parity_order`(=false) /
  `kh_parent_enabled`(=true) / `v4_kh_moves`(=true) を呼び出し箇所で定数簡約し関数削除 (dead 分岐畳込)．

### C. 識別子リネーム (v4/v3/kh マーカー除去 → 機能名)
- ファイル: `mid_v4.rs`→`mid.rs`, `tt_v4.rs`→`tt.rs`, `kh_local_expansion.rs`→`local_expansion.rs`．
- 公開 API 繋ぎかえ: `solve_via_v4`→`solve_impl`，公開 `DfPnSolver::solve()` が統一 mid を直接呼ぶ．
  公開 entry (`solve_tsume` / `solve_tsume_with_timeout` / `solve_tsume_and_collect_pn_dn_dist`) の
  署名は不変 (PyO3 `maou_rust` 影響なし)．
- 関数/フィールド/setter/局所変数 ~60 個から v4/v3/kh を除去 (`search_impl_v4`→`search_impl`,
  `init_pn_dn_or_kh`→`init_pn_dn_or`, `mate1ply_kh`→`mate1ply`, `set_kh_dml`→`set_dml`,
  `v3_nodes`→`nodes` 等)．診断 env 変数も prefix 除去 (`V4_NODHMP`→`NODHMP` 等)．
- テスト改名: `test_v4_29te`→`test_29te`, `test_v4_39te_measure`→`test_39te_measure`,
  `test_v4_counter_check_example`→`test_counter_check_example`．

### D. コメント純粋技術化
- KH / KomoringHeights / YaneuraOu 帰属，`*.hpp:行` 参照，Phase N / version / plan codename /
  廃止 narration / memory-link を除去．アルゴリズム説明と学術論文引用 (df-pn, DFPN-E Kishimoto
  NeurIPS 2019, Deep df-pn Song Zhang 2017) は保持．

### 挙動検証 (canonical, release, --test-threads=1 --ignored)
- **29te = 9,288 nodes / mate-29 / STRICT Some(29)** — リファクタ前後で完全一致．
- **39te = 4,272,957 nodes / mate-59 / Some(55) / sound** — 完全一致．
- 標準 fast tests pass / default + 全 feature で error 0・warning 0．

## Proposal (CLAUDE.md durable-doc 編集 = 要承認)

CLAUDE.md「### 重いテスト (Rust dfpn)」の [SLOW] テーブル該当 3 行を以下へ更新する
(テスト名から `v4_` を除去，"mid_v4 (bundle default)" / "39te bundle default" を統一後の "mid" /
"39te" 表記へ)．

現状 (185-187 行):

| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_v4_29te` | - | mid_v4 (bundle default) 1te/3te/29te canonical (9,288 nodes / mate-29 / STRICT Some(29)) |
| `test_v4_39te_measure` | 30M nodes (default) | 39te bundle default canonical (4,272,957 nodes / Some(55) / sound) |
| `test_v4_counter_check_example` | - | 逆王手詰将棋 mate-7 健全性 |

更新後:

| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_29te` | - | mid 1te/3te/29te canonical (9,288 nodes / mate-29 / STRICT Some(29)) |
| `test_39te_measure` | 30M nodes (default) | 39te canonical (4,272,957 nodes / Some(55) / sound) |
| `test_counter_check_example` | - | 逆王手詰将棋 mate-7 健全性 |

(他 2 行 `test_counter_check_diagnostic` / `test_no_checkmate_counter_check_probe` は改名なし，据置．)

## Notes

- 本リファクタの code 部 (rust/) は naming 整理のため reviews 対象外 (worklog/compass 管轄)．本提案は
  CLAUDE.md durable-doc 同期のみを承認対象とする．
- docs/ (design/tsume-solver, plans) の KH/Phase narration 整理は別途スコープ確認の上で対応 (本提案外)．
