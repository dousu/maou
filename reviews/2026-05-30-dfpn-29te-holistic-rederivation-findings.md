---
status: applied
applied_in: 6176412
date: 2026-05-30
target: [rust/maou_shogi/src/dfpn/solver.rs, rust/maou_shogi/src/dfpn/mid_v2.rs, rust/maou_shogi/src/dfpn/tt.rs, rust/maou_shogi/src/dfpn/entry.rs]
risk: high
reversibility: hard
---

# dfpn 29te holistic re-derivation: guidance 空間は枯渇，真因は proof-hand 非極小化 (構造)

> これは CLAUDE.md ルール変更ではなく，「holistic re-derivation 着手」(ユーザ承認) の調査
> フェーズの結論 + 次の実装方向の提案 + 監査証跡．Phase 27 セッション (2026-05-30)．

## Trigger

ユーザ指示「29te を mid_v2 で KH 相当の 19K ノードで解く．init solve 162K → KH 5,269
SearchImpl 相当の効率改善」．AskUserQuestion で「holistic re-derivation 着手」を選択．

## 確定した ground truth (KH 実 instrument, `.tmp_diag/kh_29te_plytrace.log`)

- **KH PV (mate-29)**: `S*7i 8h9g 8f8g+ 7h8g G*8f 9g8f 5g6h+ 8i7g R*8e 8f9g 8e8g+ 9g8g
  P*8f 8g8f P*8e 8f8g G*8f 8g9h 7i8h+ 9h8h 8f7g 8h8i 6h6g 8i9h S*8g 9h9g 8g8h 9g9h N*8f`
- **maou PV (mate-29)**: 1〜7 手目は KH と完全一致．**8 手目 (ply 7, 防御 AND ノード) で分岐:
  maou `L*7g` vs KH `8i7g`** (両者 7g 合駒だが maou は香打・KH は桂移動)．
- **KH: 2,094 unique / 5,269 total visits** (per-ply 上限 ~230 unique)．
- **maou: 113,027 unique / 190,646 total, proven_entries=18,090** (`visit29.log`)．
  → 探索 breadth 54×，proof tree 自体が **9× bloat** (18,090 vs 2,094)．
- KH `kPnDnUnit=2`，maou `PN_UNIT=16` (8× 粗い量子)．

### breadth 爆発は ply 7 (PV 分岐点) から始まる

maou `proven_per_ply` を KH `unique`/ply と比較すると，ply 1〜6 (PV 一致区間) では maou ≤ KH
だが，**ply 7 以降 maou の proven 数単独が KH の total unique を超える** (ply7: maou 106 vs
KH 48; ply9: 143 vs 66)．→ 防御 AND ノードの分岐から下流の proof tree が KH の 9× に膨れる．

## 全 guidance lever を実験で棄却 (holistic に組合せても改善せず)

全て 29te depth=31, find_shortest=false, kh_dml+scope=ON (現 default)，baseline=162,550/Mate(29)．

| config | nodes | Δ | PV |
|---|---|---|---|
| **default (eps2)** | **162,550** | — | 29 |
| IsSumDeltaNode (sdn) | 225,891 | +39% | 29 |
| root_ids (KH SearchEntry IDS) | 174,298 | +7% | 29 |
| eps=4 / 8 / 16 / 32 | 269K/273K/348K/238K | +46〜+114% | 29 |
| deferred penalty (denom8+floor) | 183,711 | +13% | 29 |
| decouple (純 KH init_pn_dn) | 199,040 | +22% | 29 |
| decouple + sdn | 215,903 | +33% | **31** |
| decouple + sdn + eps4 | 265,416 | +63% | 29 |
| decouple + sdn + eps8 | 301,726 | +86% | **31** |
| decouple + eps8 | 308,994 | +90% | **31** |
| sdn + eps8 | 200,726 | +24% | 29 |

**全 config が default より悪化** (nodes 増 or PV 29→31 退行 or 両方)．epsilon scale-match 仮説
(PN_UNIT=16 なので KH `second_phi+1` ≈ maou `+16`) は反証 (eps↑ で単調悪化)．
KH-faithful OR ensemble (decouple+sdn+eps) も KH の narrow+29 を再現せず．
→ **gap は search guidance では一切動かない**ことが多重確認された．default は深い局所最適．

## 真因 = proof-hand 非極小化 (構造，guidance-invariant)

guidance で動かない = proof tree の **shape/表現** の問題．確認した事実:

1. **mid_v2 は proven を実 hand (`att_hand_self`) で store する** (solver.rs:7176, :7475 付近)．
   KH `GetWinResult`/`GetLoseResult` (local_expansion.hpp:548-642) は `HandSet` で
   **極小証明駒 (proof hand) / 極大反証駒 (disproof hand)** を再帰的に計算して store する
   (`BeforeHand` で使用駒を除去しつつ子から伝播)．
2. **TT look_up は hand-dominance を行う** (tt.rs:1228/1246/1281/1306, `hand_gte_forward_chain`):
   proof は stored hand ⊆ query (query が駒過多) でヒット，disproof は stored ⊇ query でヒット．
   → dominance 自体は mid_v2 でも有効．
3. **しかし stored hand が極小でない**ため，maou の dominance は KH より弱い: 実 hand の
   superset しか collapse できず，「より少ない駒でも詰む」変種を別エントリとして再探索・再 store
   する．これが proven_entries 9× bloat と 113K unique breadth の構造的原因 (guidance 非依存)．
4. legacy `mid` 経路は `record_proof` (要素 min) で弱い minimization をしていたが，**mid_v2 は
   record_proof/get_proof_hand を一切呼ばない** (mid_v2 body grep で確認済)．

## 提案する次の実装 (holistic re-derivation 本体)

**mid_v2 に KH `HandSet` 極小証明駒 / 極大反証駒 を移植する**:

1. `MidSearchResult` に proof/disproof `hand: [u8; HAND_KINDS]` を追加 (現在は pn/dn/md のみ)．
2. 終端 proven (1 手詰/0 手詰) で KH `CheckMate1Ply` の proof hand を初期化．
3. `current_result` の win/lose 集約で KH `GetWinResult`/`GetLoseResult` 相当:
   - OR win: best child の proof hand から `BeforeHand` で極小化して伝播．
   - OR lose (全子反証): 子 disproof hand の極大集合 (要素 max)．
   - AND win (全子詰): 子 proof hand の極小集合 (要素 min)．
   - AND lose: 子 disproof hand から伝播．
4. store 時に実 hand ではなく算出した極小/極大 hand を渡す．
5. soundness: 極小化が緩すぎる (駒を落としすぎ) と **false mate**，きつすぎると collapse 不足．
   KH の `BeforeHand`/`HandSet` を bit 単位で忠実移植すること．

### soundness リスク (high)

proof-hand 計算は詰みの正しさに直結する．誤ると false-mate/false-nomate を生む．
**必ず**: 93 dfpn lib (single-thread) + `test_tsume_6_29te` Mate(29) strict PV +
`test_tsume_39te_ply2_no_false_nomate` + `soundness_depth25` を各段で確認．段階導入
(default-off param → A/B → 広域検証 → default 化) を厳守．

### 想定効果

proof tree 18,090 → ~2,094 (KH 相当) に近づけば breadth も連動縮小し，162K → 19K 目標に
初めて構造的に接近できる見込み．ただし PV-fragile (鉄則: PV 長 29 を必ず確認)．

## Alternatives considered (棄却)

- guidance knob 単体/組合せ: 上表の通り全棄却 (holistic 組合せも悪化)．
- PN_UNIT=16→2 rescale: dynamic range/constant 連動が必要で高侵襲，かつ proof bloat は
  guidance-invariant なので scale 変更単独では proof 表現を直さない (副次的)．
- root_ids / find_shortest 強化: Phase 24 で棄却済 (root_ids PV31 退行, find_shortest は
  initial gap を縮めない)．

## 本セッションの in-flight code (default 不変，93 lib pass)

- `param_is_sum_delta_node` (default false) + `is_sum_delta_node` (mod.rs) + `apply_force_max`
  (mid_v2.rs): KH `IsSumDeltaNode` 忠実移植．**負の結果 (sdn 単体+39%)** だが holistic 検証用に保持．
- 診断テスト 3 本 (#[ignore], **[SLOW]**): `test_tsume_29te_is_sum_delta_ab`,
  `test_tsume_29te_breadth_sweep`, `test_tsume_29te_kh_ensemble_sweep`．上表を再現する．
- Cargo.toml 1.6.0 → 1.7.0 (feat: lever + 診断追加)．default 挙動は完全不変．

## 実装機構の深掘り (着手フェーズの調査結果)

### tightening が生じる機構
proof-hand 極小化の効果は **終端の極小 proof hand が上へ伝播** して初めて生じる:
- maou `mid_v2` は終端 mate で `att_hand_self` (実 hand 全部) を store (solver.rs:7176)．
- OR proven は `adjust_hand_for_move(or_move, child_ph)` で child proof hand を逆算するが
  (mod.rs:807; drop→+1, capture→-1)，child が実 hand を store していると逆算しても親の
  実 hand に戻るだけで **一切締まらない**．
- → 終端を極小化しないと OR/AND 中間の極小化は無意味．**部分実装は効果ゼロ**で，
  終端 + OR + AND を同時に入れる必要がある (holistic)．

### soundness 基盤 (確認済，決定的)
- `position_key(board) = board.board_hash` = **盤面のみ** (mod.rs:792)．TT entry は
  `(board_hash, attacker_hand)` でキー化．
- 詰将棋は **駒数保存**: board_hash が盤上駒を固定 → attacker_hand が決まれば
  **defender_hand = 全駒 − 盤上 − attacker_hand で一意確定**．
- ゆえに attacker proof hand dominance (`query ⊇ stored_proof`, tt.rs hand_gte_forward_chain)
  は sound: attacker が駒過多 = (保存則で) defender が駒過少 = より詰めやすい → proof 有効．
- 同様に disproof は `stored_disproof ⊇ query` で sound (defender 駒過多 = 詰みにくい)．

### 終端 proof hand は ∅ ではない (重要な落とし穴)
naive に「終端 mate は attacker hand 不要 → ∅」とすると **unsound**: attacker_hand=∅ は
保存則で defender が最大持ち駒 = 全く別の (詰まないかもしれない) 局面を主張する．正しい終端
proof hand = 「complementary defender hand でも逃れられない最小 attacker hand」= **詰将棋の
証明駒計算** (KH `HandSet{DisproofHandTag}.Get(pos)` / `CheckMate1Ply` の proof_hand)．

### keystone (次セッションの最初の実装対象)
maou には「mate の極小証明駒を返す」clean primitive が **無い** (`has_mate_in_1_with` は
bool のみ, solver.rs:6423)．legacy `mid` の proof-hand 計算は **maou 固有アーキに深く絡む**:
init_and_proof (solver.rs:3710 ∅ 初期化 → 6648 で proven 合駒の adjust を要素 max 蓄積 →
4302 で `min(att_hand)` clip して store) は prefilter/chain 合駒 path 専用で，GHI 伝播 /
path-dependent disproof / K-M cycle_root と絡む．→ **lift ではなく re-derivation が必要**．
**この終端極小証明駒の正しい構成が keystone** (contact check は hand 非依存で極小，distant
check は interposition 要件を子 proof hand から要素 max で積む)．これを OR (adjust) → AND
(要素 max) → 終端まで chain させる．**誤ると false mate** (最悪のバグ) なので拙速 rush は禁止．

### 実装順序 (提案)
1. 終端 mate の極小証明駒 primitive を確立 (legacy mid base case を特定 or KH DisproofHand 移植)．
2. `mid_v2` proof 側のみ default-off param で chain 実装 (終端 + OR adjust + AND max)．
3. 各段で 93 lib + tsume_6_29te Mate(29) PV + no_false_nomate + soundness_depth25 を gating．
4. proof 側で breadth/proof-tree が縮むことを diag (proven_entries, unique) で確認．
5. disproof 側 (極大反証駒) を追加．
6. A/B → 広域検証 → default 化判断 (version bump)．

## Rollback

全 in-flight は default-off param + #[ignore] テストで default 挙動不変．revert は git checkout で trivial．
proof-hand 移植 (着手フェーズ; code 未投入) を始める場合は上記順序で別途 default-off param で段階導入する．
