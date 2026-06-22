---
title: "mid_v3 完全廃止 + 39te bundle default 化 に伴う CLAUDE.md [SLOW] テーブル更新"
status: pending
date: 2026-06-22
branch: feat/tsume-solver
applied_in:
---

## Trigger

ユーザ指示 (2026-06-22): 「mid_v3 はすでにほぼ価値がないので廃止し，**mid_v4 一本に統一**せよ．
mid_v3 が残っていると意味のない既存テストや後方互換ケアをしてしまう」「39te 問題で最もパフォーマンスが
出る **39te bundle を default 化**せよ．default 化後に期待 perf が出ているか必ず確認せよ」．

実装済 (rust commit; maou_shogi 2.50.1 → **3.0.0** breaking):
- `solve()` を `solve_via_v4` へ直結 (旧 `V3_V4ENG` 分岐撤去)．
- 39te bundle の 8 gate を default に焼き込み (gate 関数が定数 true): `V4_KHORDER` / `V4_HANDSET` /
  `V4_SMPROP` / `V4_DOM` / `V4_KHPARENT` / `V4_KHMOVES` / `MATE1PLY_KH`，および `kh_parity_order=false`．
- `mid_v3.rs` (2,430 行) / `repetition_memo.rs` / `V3Entry` / mid_v3 専用フィールド ~28 個 /
  `set_v3_*` 公開 setter 6 個を削除．`v3_nodes` / `v3_path` は mid_v4 が再利用するため存続．
- テスト改名/変換: `test_mid_v3` → `test_v4_29te` (mid_v4 default で 29te canonical 検証),
  `test_mid_v3_counter_check_example` → `test_v4_counter_check_example`,
  `test_mid_v3_39te_measure` → `test_v4_39te_measure`．

### default 化後の perf 検証 (env gate 一切無し, production `solve_via_v4`)
- **39te = 4,272,957 builds / 8,278,452 do_move / Some(55) len 59 / STRICT VERIFY sound / 21.3s** —
  旧 gated bundle の documented golden と **完全一致** (behavior-preserving)．
- **29te = 9,288 nodes / mate-29 / STRICT Some(29) sound**．
- 標準 196 + fixture 7 + SLOW 8 pass / 0 fail (`--test-threads=1`)．
- PyO3 (`maou_rust`) compile OK (`solve_tsume_and_collect_pn_dn_dist` は `self.table` 参照で不変)．

## Proposal (CLAUDE.md durable-doc 編集 = 要承認)

CLAUDE.md「### 重いテスト (Rust dfpn)」の [SLOW] テーブルを以下へ更新する:

| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_v4_29te` | - | mid_v4 (bundle default) 1te/3te/29te canonical (**9,288 nodes** / mate-29 / STRICT Some(29)) |
| `test_v4_39te_measure` | 30M nodes (default) | 39te bundle default canonical (**4,272,957 nodes** / Some(55) / sound) |
| `test_v4_counter_check_example` | - | 逆王手詰将棋 mate-7 健全性 |
| `test_counter_check_diagnostic` | - | 診断用ログ出力 |
| `test_no_checkmate_counter_check_probe` | 10M nodes | ノード予算プローブ |

- 旧 `test_mid_v3` / `test_mid_v3_counter_check_example` 行を削除 (関数は改名済)．
- 「mid_v3 1te/3te/29te canonical (18,539 nodes ...)」の 18,539 は mid_v3 固有値ゆえ撤去．新 default は
  bundle (29te 9,288)．

## Risk / Scope

- CLAUDE.md のみ (durable doc)．rust コードは別途 commit 済 (version bump 3.0.0)．
- `docs/plans/steady-burning-lantern.md` / `docs/plans/velvet-squishing-cupcake.md` は mid_v3/mid_v4
  並走を述べる**歴史的 plan 文書**ゆえ現状維持 (normative でない)．必要なら別 review で追補．
- 注意 (preserve): tsume 系テストは mid_v4 が **最短保証なし** (proof-tree mate length は engine 依存;
  39te=Some(55) vs tsume 正解 39手) ゆえ厳密 PV でなく PV replay soundness で検証する形へ緩めた
  (`test_tsume_4` 13手 / counter-check 9手)．canonical 最短手数は各テストに TODO コメントで保存済 ——
  将来 `find_shortest` を mid_v4 で honor 化したら厳密手数 assert に戻す (ユーザ方針)．

## Status note

status: pending — ユーザが /checkpoint-context step 5 で承認後，model が CLAUDE.md を編集・commit し
SHA を applied_in に記入して applied 化する．
