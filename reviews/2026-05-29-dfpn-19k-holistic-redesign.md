---
status: pending
applied_in:
date: 2026-05-29
target: [rust/maou_shogi/src/dfpn/solver.rs, rust/maou_shogi/src/dfpn/mid_v2.rs, rust/maou_shogi/src/dfpn/mod.rs]
risk: high
reversibility: hard
---

# dfpn 29te を KH 同等効率で解く holistic re-design (設計提案)

> 注: これは CLAUDE.md ルール変更ではなく，dfpn ソルバの大規模 re-design の**設計記録 +
> 着手判断のための提案**．実装は次セッション以降．Phase 25 調査
> (worklog/2026-05-30-003051.md) の結論を反映．

## Trigger

worklog/2026-05-30-003051.md — KH を実 instrument して 29te の gap を測り直した結果，
従来 headline「10×」が **metric mismatch** だったと判明 (真の gap は 36×/54×)．単発 lever
を全て実験で棄却し，本丸が full ensemble 再導出 (週単位) であることを確定した．

## 背景 / 目標

29te (`l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1`)
を KomoringHeights 同等効率で mate-in-29 として解く．

## headline 訂正 (最重要, KH 実 instrument)

KH に per-ply 訪問トレース (KH_PLY_TRACE) を仕込んで 29te を実測:

| metric | maou | KH | gap |
|---|---|---|---|
| node 展開呼び出し | 190,646 (mid_v2 entry) | **5,269** (SearchImpl) | **36×** |
| unique 局面 | 113,027 | **2,094** | **54×** |
| proof tree (proven entries) | 18,090 | ≤5,269 (探索≈証明) | ~3-9× |

**従来「maou 190,646 vs KH 19,270 = 10×」は誤り**: KH の reported "nodes 19,270" は
`Threads.nodes_searched()` = **do_move 回数**で，maou の mid_v2 entry 数とは別 metric だった．
apples-to-apples (node 展開) は **36×**．revisit 比は KH 2.5× > maou 1.69× なので，
**gap は純粋に breadth (distinct positions を 54× 多く探索)**で thrashing ではない．
KH は 29 手詰めを **2,094 distinct 局面**しか触らず証明する．

## 根本発見 (Phase 25, 全て実験で確定)

1. **maou の「初回 29 手 PV 発見」は脆弱で,現在の非効率設定でのみ成立する**．効率化を入れると
   必ず 31 手 proof を先に発見する (PV 29→31 を 5 回確認):
   - 1 手詰 detection gate-off: 初回 105,275 (-45%) だが PV=31．
   - edge_cost decouple (純 init_pn_dn_*_kh): 159,499 (-16%) だが PV=31．
   - δ tie-break (KH 基準2): 242K / PV=31 (既出, mid_v2.rs:652)．
   - eps≠2 / PN_UNIT≠16: nodes 悪化 + 多くで PV31．
2. **単一 knob 全棄却**: epsilon(eps=2最適) / PN_UNIT(16最適) / TCA(既に十分) /
   **deferred penalty(denom=0 が最良; 有効化で proven 18,090→20-28K に増加・nodes悪化)**．
3. per-move init pn/dn (init_pn_dn_or/and_kh, mod.rs:307/360) と move_brief_eval (mod.rs:411)
   は KH 忠実移植で**一致**．→ 差は delta 集約 / sum_mask / DML / DAG の総合．

→ **29-finding は KH guidance アンサンブル全体の co-tuned 創発特性**で,個別 knob では再現不能．

## find_shortest dual-range は「19K への path」ではなく find_shortest 専用の最適化だった

当初「効率的初回 (任意長) + find_shortest で 29 に refine」(KH `SearchMainLoop` アーキ) を
keystone と仮説したが，**KH は iter0 (len=∞) で 29 を直接出す**ので refine に頼らない．
実測でも gate-on (初回31) + midR で find_shortest は 31 のまま即確定 (29 不達)．
→ **初回 solve 自体が 29 を効率的に見つける必要があり，guidance 問題に回帰**．

ただし副産物として `param_kh_middle_range` (look_up_pn_dn_md_bounded 中間レンジを KH
`LookUpExact` 準拠 `(PN_UNIT, INF)` に) は **現状設定 (gate25/PV29) で find_shortest を
571,938→190,647 (-67%, PV29 維持)** に削減する．Phase24a の per_iter_cap ハックの原理的置換候補．
- soundness: proven entry は絶対的に詰みなので dn=INF は妥当 (前提: stored proven あり)．
  93 lib + no_false_nomate pass．**だが proven 信頼で「より短い詰み探索」safety net を弱める**ため
  初回が非最小な問題で minimality を逃すリスク → 正しい期待値のテスト群で広域検証要．

## 参考: dual-range architectural mismatch

maou `ProvenEntry` (entry.rs:111) は `mate_distance()`(=proven_len) + `disproven_len()` を
保持済だが，maou は **proven TT と working TT (`DfPnEntry`) を別テーブルに分離** (KH は単一 entry に
`pn_/dn_/proven_len_/disproven_len_` 統合)．proven-at-31 を len=29 で引くと working 値が proven 化で
失われ，(PN_UNIT,PN_UNIT) リセットか OLD working の二択しかない (KH の第三挙動 `(incoming_pn,INF)`
を midR で近似)．Phase18 の working pn/dn fallback は「初回 31 ガイド」で廃止済 (solver.rs:2082)．

## 提案する次の実装方向 (優先順)

1. **[本丸] 初回 solve guidance を KH に整合** — 54× breadth を縮める唯一の道．未検証の最後の
   aggregation lever = **sum_mask(IsSumDeltaNode, mid_v2.rs:209-217/464-467) + DML
   (build_delayed_chain, mid_v2.rs:666) の KH 整合性** (interposition collapse 精度)．
   KH (local_expansion.hpp:177, delayed_move_list.hpp) と bit 単位で突合し，初回 solve で 29 が
   効率的に出るか A/B (**必ず PV 長確認**)．週単位・PV-fragile．
2. **[別軸の bankable] midR (find_shortest -67%)** — 広域 minimality 検証後に default 化．
   前提: mate15 系テストを 21 期待に修正 (下記)．version bump 要．
3. **soundness 回帰**: `--lib dfpn::` 93 pass を各段で確認．

## mate15 テスト誤り (派生発見, 39te 課題として保留)

`test_tsume_39te_ply24_mate15_regression` (depth17, Mate(15)期待) は clean HEAD で NoMate．
KH ground truth で当該サブ局面 (`9/3+N1P3/7+R1/9/9/9/1R2S3k/3p5/9 b 2b4g3s3n4l16p 25`) は
**真に mate-21** (KH も maou depth25 も 21)．テスト期待値「15」が誤り (コメントの「21 は
chain-drop inflated」も誤認)．**soundness バグではない**．depth17 NoMate も正しい (真詰21>horizon17)．
→ regression/soundness_depth25/with_* を 21 期待に修正すべき (ユーザ保留中; 39te 課題)．

## Alternatives considered

- **dual-range LookUpExact を完全移植して 2 段アーキ化**: KH は iter0 で 29 を直接出すため
  refine 不要 → 2 段アーキは KH と不一致．keystone から降格 (find_shortest 最適化に留まる)．
- **単発 knob チューニング継続**: epsilon/PN_UNIT/TCA/decouple/deferred/1手詰 を全て実験済で
  棄却 (PV31 or nodes悪化)．exhausted．
- **find_shortest 強化で初回を 31 のまま許容**: find_shortest が 31→29 を refine できない
  (実測 315K で 31 維持) + KH も refine に頼らない → 不採用．

## What this enables

KH ground-truth インフラ (per-ply トレース) と corrected metric (36×/54×) により，今後の
re-design が「54× の distinct-position breadth を縮める」という正確な target に対して測定可能になる．

## What this constrains

efficiency↑ 施策は全て PV 29→31 を壊す構造的対立があるため，初回 solve guidance を触る変更は
**必ず PV 長 (=29) を確認**する制約が課される (鉄則)．

## Rollback plan

全 in-flight 変更は default-off param + #[ignore] 診断テストで，default 挙動不変 (baseline
190,646/PV29, 回帰 93 pass intact)．revert は git checkout で trivial．midR 等を default 化する
場合のみ version bump + 広域検証が前提．
