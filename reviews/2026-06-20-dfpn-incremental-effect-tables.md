---
title: "YaneuraOu 級 incremental effect (利き) テーブルを Board に導入 — dfpn query 律速の構造的根治"
status: approved
date: 2026-06-20
branch: feat/tsume-solver
base-sha: 5ccc9b7  # v2.42.0
campaign: "39te を KH 相当の実行時間で解く (探索 wall ~15.5s vs KH info-time 14.9s; per-op 1.2-1.4× 遅)"
trigger: worklog/2026-06-20-103255.md
---

## 0. この doc の位置づけ

ユーザが「次セッションは do_move より query が多いという dfpn の特性に合わせ **YaneuraOu 級の
incremental effect 機構を実装** する」を選択 (2026-06-20)．本 doc は着手前の **design proposal**:
動機・設計方針・soundness 検証計画・リスク・既存 invariant との整合を確定する．**Board の内部
表現を変える architectural 変更．方針はユーザ承認済 (status: approved, 2026-06-20)．次セッションで
着手し，完了後に docs/architecture.md / docs/rust-backend.md を更新して status: applied + SHA を記入する
(docs 編集はユーザが行う; モデルは直接 CLAUDE.md/docs を編集しない)．**

## 1. 🎯 動機 (KH 時間差の構造的真因; v2.42 で確定)

KH との残り実行時間差は **探索効率でなく per-primitive 定数速度** に集約される
([[project_dfpn_primitive_gap]], worklog/2026-06-20-103255.md):

- **カウントは KH 以下**: 39te node 0.80× (3.11M vs 3.88M) / do_move 0.86× (6.38M vs 7.45M)．
  探索アルゴリズムは KH 以上に効率的 = 無駄探索が原因でない．
- **NPS per-op 1.2-1.4× 遅**: maou ~0.41M do_move/s vs KH ~0.50-0.58M．
- **真因 = incremental 利きテーブル不在**: KH(YaneuraOu) は do_move 毎に各マスの利き
  (effect) を incremental 更新するため `attackers_to` / mate1ply / 王手生成が **テーブル参照**．
  maou は利きを持たず **query 毎に magic/pext で計算**する．**dfpn は do_move より query
  (mate1ply / 王手生成 / attackers_to / TT 照合) が圧倒的多数** ゆえ，この設計差が NPS 差の主因．
- bench_primitives: movegen ~480ns / mate1ply ~300ns (根 951ns) / **do+undo ~26ns**．
  maou の do_move は利き更新が無く安い = 利き更新コストを足しても query 削減が上回る余地がある．

V4PROF 律速 (39te): child_loop ~63% のうち lookahead(mate1ply) ~22% / cl_other ~19% /
tt_lookup ~13.5%．mate1ply と王手生成 (movegen ~9%) が effect 化の最大受益先．

## 2. 設計方針 (案; 詳細は着手時に確定)

- **表現**: Board に「各マス sq に対する各色の利き数 (または利き元 bitboard)」を保持する．
  YaneuraOu LONG_EFFECT 相当 (board_effect[color][sq] = 利き数, byte_board) + 遠方利き
  (long_effect, 飛び駒の貫通) を別管理する案を第一候補とする．まず **利き数 board (近接+遠方
  合算)** から始め，attackers_to / mate1ply の参照に使えるか検証する (最小増分).
- **更新**: do_move / undo_move で，動いた駒・取られた駒・遮断/開放された遠方利きの差分のみ
  更新する (incremental)．undo は do の逆操作で完全復元．
- **利用先 (query を参照化)**: ①`attackers_to(sq, color)` ②mate1ply の escape/capture/
  interpose 判定 ③王手生成 (rev_check の代替) ④is_in_check．段階的に置換し各段で計測する．
- **soundness**: 全 do_move で「incremental 利き == フルスキャン再計算」を debug_assert で
  突合する検証モード (PDM_VERIFY / MATE1PLY_CAND 差分 harness と同じ流儀) を必須とする．
  release では assert を外す．**利きは詰み判定の土台ゆえ不整合 = 偽詰み = 即不健全**．

## 3. リスク / 留意点

- **大規模 cross-cutting**: Board の hot path (do_move/undo_move) に分岐が増える → do_move が
  ~26ns から増える．query 削減がこれを上回らないと逆効果 (要 min-of-N 計測; 早期に bench で gating)．
- **wheel 可搬性 = binding** ([[feedback_wheel_portability_binding]]): SIMD/POPCNT 等の HW 命令を
  使う場合は **runtime gate のみ** (native build 禁止)．スカラ fallback 必須．
- **メモリ**: 利き board は Board 1 個あたり数百 byte 増．dfpn は Board 1 個を mutate しながら
  do/undo するので増加は定数 (TT とは無関係)．
- **段階導入**: 一括置換でなく ①利き board 追加 + 検証 (探索不変) → ②attackers_to 置換 →
  ③mate1ply 置換 → ④王手生成置換，と段ごとに node 不変 + soundness + wall を計測する．

## 4. 既存 invariant との整合 (死守)

- **canonical mid_v3 18,539 / node 不変** (3.11M/9,288/17,720) / 標準 199 + ignored 8 pass．
  effect 化は query の**結果を変えてはならない** (純粋な高速化; 探索不変)．
- **soundness > efficiency**: STRICT VERIFY None=偽証明=即調査．利き不整合は debug_assert で全捕捉．
- **時間最適化は探索不変が基本**: node/do_move/mate-len 不変を確認しつつ wall を削る．

## 5. 受け入れ基準 (完了の定義)

- 39te search_wall が現 ~15.5s から有意 (min-of-N で noise 超え) に短縮し，KH info-time 14.9s に
  接近 or 下回る．node/do_move/Some(55)/canonical 18,539 全不変．199+8 pass．
- 利き検証モードで全 do_move 等価 (incremental == フルスキャン)．
- docs/architecture.md (Board 表現) / docs/rust-backend.md を更新．

## 6. 承認後のアクション

着手は次セッション．本 review は方針の合意記録．完了時に docs を更新し status: applied + SHA を記入する
(docs 編集はユーザが行う; モデルは直接 CLAUDE.md/docs を編集しない)．

---

## 7. Stage 1 実装結果 (2026-06-20, v2.43.0)

**実装した内容** (`rust/maou_shogi/src/board.rs`, cargo feature `effect_table` = default **無効**):
- `Board.effect: [[u8; 81]; 2]` = 色別利き数 byte board (YaneuraOu `board_effect` 相当)．
- `put_piece`/`remove_piece` を **self-consistent な単一窓口**化し，`effect_on_put`/`effect_on_remove`
  で incremental 更新する．遠方利きの遮断/開放は「`sq` を貫通する slider 集合」(`effect_sliders_through`,
  is_attacked_by と同じ逆スキャン) の利きを **occ を明示渡しした XOR-diff** (`update_slider_effects`,
  `before & !after` / `after & !before`) で `sq` より先の射線だけ更新する．bb 変更前に一括計算するため
  put/remove の順序に依存しない．
- soundness: `verify_effect` (フルスキャン再計算と比較) + `EFFECT_VERIFY=1` で全 do_move/undo_move 検証
  + unit test `test_effect_incremental_matches_fullscan` (平手/片玉/39te を DFS playout, 駒打ち・捕獲・
  成り・遮断/開放を網羅)．
- **feature OFF (production default) は完全ゼロコスト**: field/維持/helpers/検証/テストを全て
  `#[cfg(feature = "effect_table")]` で除外，`effect_on_put`/`_on_remove` は no-op インライン消去．
  → default build は挙動・コードともベースライン同一 (199 pass / canonical 18,539 / 39te 不変)．

**検証結果** (✅ 全 sound):
- feature ON: `test_effect_incremental_matches_fullscan` pass．canonical 29te を `EFFECT_VERIFY=1` で
  実探索 → 18,539 / mate-29 / desync panic 無し (全 do_move/undo_move で incremental == フルスキャン)．
  39te bundle: node 3,105,196 / do_moves 6,384,324 / Some(55) **完全不変** (探索を一切変えない)．

**⚠ gating measurement (最重要; この approach の本質的制約)**:
- 39te bundle search_wall: ベースライン ~15.5s → effect 維持 ON で **~31.4s (min-of-3, +約16s, ~2×)**．
  XOR-diff 化前 (naive full subtract/add) は 34.0s だったので，**XOR-diff の寄与は -8% のみ**．
- → **書き込みは律速でない．律速は「影響 slider ごとの PEXT 利き再計算」** (`effect_sliders_through` の
  3-4 PEXT + slider ごと 2 PEXT) × 約 32M put/remove．Stage 1 単体は query 削減ゼロゆえ純コスト増．
- **結論**: naive/XOR-diff な per-slider 再計算方式は本質的に ~2× 遅く，query 側 (Stage 2: is_attacked_by/
  is_in_check/checkers の参照化) の削減で取り戻せる規模ではない (query は wall の一部，維持は +16s)．
- **viability には direction-aware long-effect が必須**: YaneuraOu は事前計算した射線セグメント + 方向
  byte で更新を **PEXT 無し**の O(有効方向数) にする．これを入れて維持を ~10× 安く (~2-3s) した上で
  Stage 2/3 (mate1ply の after-move 補正含む) を載せて初めて net 改善が見込める．**これは別の大規模・
  soundness-critical な実装** (count table と Stage 2 query 参照化は再利用可; 更新内部のみ拡張)．

**現状の位置づけ**: 受け入れ基準 (§5: wall 短縮) は未達ゆえ **status は approved のまま** (applied にしない)．
Stage 1 は「正しく検証された count effect 基盤 + 維持コストの確定計測」までを default-off で land．
**次の大きな一手 = direction-aware long-effect** (採否はユーザ判断; 入れる場合は本 review を更新 or 新 review)．

**道具/コマンド**:
- feature build: `CARGO_TARGET_DIR=.tmp_diag/cargo-target cargo test --release -p maou_shogi --features effect_table --no-run`
- 検証テスト: `<binF> board::tests::test_effect_incremental_matches_fullscan --exact --nocapture`
- 実探索検証: `EFFECT_VERIFY=1 <binF> dfpn::tests::test_mid_v3 --ignored --exact --nocapture --test-threads=1`
- overhead 計測: feature build で 39te bundle (env stack は scratchpad/current.md 参照) → search_wall を min-of-N．
