---
status: pending
applied_in:
date: 2026-05-30
target: [rust/maou_shogi/src/dfpn/proof_hand.rs, rust/maou_shogi/src/dfpn/solver.rs, rust/maou_shogi/src/dfpn/mod.rs, rust/maou_shogi/src/dfpn/tests.rs]
risk: high
reversibility: medium
---

# dfpn Phase 28: KH 極小証明駒 (minimal proof hand) — sound だが 29te breadth +16% で逆効果．真因は proof-tree でなく失敗ライン探索

> ユーザ承認「終端極小証明駒を本格実装．maou の本格的な回収まで視野に」を受けた着手．
> 結果は**負の知見**だが，(1) 実装は健全で default-off 資産として保持，(2) 反証側 (候補A) の
> 鏡像基盤になる，(3) 19K ギャップの真因を**再特定**したため記録する．

## やったこと (Stage 1〜4, 全て default-off, dfpn lib 101 tests pass)

- **新規 `proof_hand.rs`**: KH `HandSet` (hands.hpp) 忠実移植．
  - `add_if_hand_gives_other_evasions` = AND 詰み局面で単一離れ王手のとき「攻め方が独占する
    合駒駒」を証明駒に記録 (soundness keystone)．接触王手は no-op．歩は二歩/合駒マス考慮．
  - `ProofHandSet` (要素 max 集約) / `proof_hand_terminal_and` / `hand_clip` / `hand_min`．
  - 単体テスト 8 本 (接触/離れ/二歩/複数王手/集約/clip) green．
- **`mid_v2` proof store 4 箇所**を `param_minimal_proof_hand` (default false) 後ろに配線:
  終端 AND (solver.rs:7188) / 1 手詰 child OR (CheckMate1Ply, :7257) / OR-proven BeforeHand /
  AND-proven 要素 max + AddIf (:7530)．KH 通り **AND 集約は children に BeforeHand を掛けない**．
- 診断テスト `test_tsume_6_29te_minimal_proof_hand` (gate) + `_proof_hand_diag` (OFF/ON 計測)．
- soundness: 29te は ON でも **Mate(29)** 維持，不正手順 (8i7g) なし．`hand_clip(ph, att_hand)` guard．

## 計測 (29te, depth=31, 50M)

| find_shortest | mph | nodes | unique 局面 | TT proof | TT confirmed-disproof |
|---|---|---|---|---|---|
| false | **off** | **162,550** | **90,029** | 3,133 | 1,217 |
| false | on  | 187,815 (**+15.5%**) | 103,835 | 2,997 (−4%) | 1,652 (+36%) |

KH 実測: **19,270 nodes / 2,094 unique / 5,269 total**．

## 真因の再特定 (これが本質)

**「proof-tree 9× bloat → 極小化 → 19K」仮説は棄却された．**

- maou の証明木 (proven entries) = **3,133 ≈ KH 2,094 (1.5×)．bloat していない．**
  (worklog の 18,090 は別計測経路の値で production `solve_via_v2` には当てはまらなかった．)
- 極小化しても proof entries は −4% しか縮まず，guidance を撹乱して nodes +16%・unique +15%・
  confirmed-disproof +36%．**完全に逆効果．**

**真のギャップ = breadth (訪問 distinct 局面数)．**

| | KH | maou |
|---|---|---|
| unique 訪問 | 2,094 | 90,029 (**43×**) |
| total/unique (再訪率) | **9.2×** | 1.8× |
| unique / 証明木 | ≈1× | **29×** |

> **KH は狭く深い** (少数局面を 9 回深掘り)．**maou は広く浅い** (43× 多い局面を 1.8 回薄く触る)．
> maou は証明木 3,133 を確定するのに **90,029 局面を訪問** = **約 86,000 が証明木に入らない失敗ライン**．
> 19K ギャップ = この失敗ライン探索の除去．証明駒でも guidance tuning (Phase 22-27 全敗) でもない．

## 失敗ライン爆発の機序 (仮説)

df-pn が失敗ライン (詰まない攻め手) を深追いしないには dn を速く大きくして切替える必要がある．
KH は**反証駒 (maximal disproof hand) + confirmed disproof** の hand-dominance で詰まない局面の
近傍を即 dn=∞ にして探索ごと刈る．maou は反証を **att_hand のまま，かつ `scope_disproof`
(default true) で scope 限定 (confirmed にしない)** で保存している．これは Phase 26 の
**false-NoMate (GHI 汚染)** を避ける soundness 由来の妥協であり，代償として反証が弱く再利用されず
失敗ラインが別 ply / 別経路で再展開され 86K の distinct 局面になっている．

## Phase 28b: 反証側 (候補A) も実装 → **同様に棄却** (v1.9.0)

ユーザ選択を受け `proof_hand.rs` の鏡像で反証駒も実装した:
- `drop_check_squares` (`compute_checkers_at` board.rs:154 逆利き忠実鏡像) + `DisproofHandSet`
  (要素 **min** 集約) + `remove_if_hand_gives_other_checks` (hands.hpp:137)．単体テスト 4 本 green．
- mid_v2 disproof store を `param_minimal_disproof_hand` (default off) 後ろに配線:
  終端 OR (王手なし) = `disproof_hand_terminal_or`，集約 (curr.dn==0, OR) = 子要素 min +
  BeforeHand + RemoveIf．AND は att_hand_self 踏襲．scope_disproof と併用で GHI-safe．

**4-way sweep (29te, find_shortest=false, 全て Mate(29) — false-NoMate なし):**

| proof | disproof | nodes | unique |
|---|---|---|---|
| off | off | **162,550** | **90,029** |
| on | off | 187,815 (+15.5%) | 103,835 |
| **off** | **on** | **199,739 (+22.9%)** | **111,294** |
| on | on | 185,331 (+14.0%) | 100,550 |

→ 反証駒も **+23%**．**証明駒・反証駒の両方が棄却された．**

## 統一結論: hand 値極小化は 19K の lever ではない

- **total/unique = 1.8×** → 失敗ライン 86K は**再訪されていない**．問題は「弱い反証で再探索」では
  なく，そもそも 86K の失敗局面を 1 回ずつ**展開してしまう selectivity**．
- 証明駒/反証駒は TT dominance を介して子 pn/dn を決める = **guidance そのもの**．guidance は
  Phase 22–27 で枯渇が実証済みの deep local optimum．故に hand 値をいじると proof でも disproof でも
  dominance ベース guidance が撹乱され**必ず悪化**する．
- **19K ギャップ = どの子を展開するかの selectivity = 探索制御の構造**．hand 値の極小性ではない．

## 残る道 (要ユーザ判断)

1. **KH SearchImpl 一体移植** (MovePicker + df-pn+ + TCA + GHI を統合体として)．piece-by-piece の
   re-derivation では届かない＝「本格的な回収」．数週間・高 soundness リスク．
2. **162K を operating point として受容** (sound・約1秒・production 十分)．

## 健全性メモ

- proof/disproof 共に完全 default-off (OFF 路 byte-identical)．**dfpn lib 105 tests pass**
  (単体テスト 12 本含む)．4-way sweep 全構成 Mate(29)．
- 既存 doctest `param_root_trace` (solver.rs バレ ``` fence) は **HEAD 由来の pre-existing 破損**で
  本変更と無関係 (pre-commit は cargo doctest を走らせない)．
