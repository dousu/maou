# 証明数・反証数の計算

### 4.1 WPN: Weak Proof Number スケールドサム式 (v0.29.0+)

**出典:** Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)

証明数/反証数の**二重計数問題**(double-counting problem)に対処する手法．
DAG 構造の探索木において，共有ノードが複数の親から重複してカウントされ
pn/dn が過大評価される問題を緩和する．

#### 式の変遷

| バージョン | AND pn | OR dn |
|-----------|--------|-------|
| 標準 df-pn | `sum(child_pn)` | `sum(child_dn)` |
| WPN 原論文 | `max(child_pn) + (count-1)` | (未適用) |
| v0.29.0 スケールドサム | `max + (sum-max) >> γ` | `sum` |
| **v0.31.0 対称適用** | `max + (sum-max) >> γ` | `max + (sum-max) >> γ` |

#### スケールドサム式 (v0.29.0〜)

```
AND pn = max(child_pn) + (sum(child_pn) - max(child_pn)) >> WPN_GAMMA_SHIFT
OR  dn = max(child_dn) + (sum(child_dn) - max(child_dn)) >> WPN_GAMMA_SHIFT
```

- `WPN_GAMMA_SHIFT = 6` → 非最大子の寄与を 1/64 に減衰
- crossover 点: `max = PN_UNIT × 2^6 = 1024`
  - max < crossover: 旧 `max + (count-1)*PN_UNIT` より保守的
  - max > crossover: 旧式より積極的

旧式 `max + (count-1)` は非最大子の実際の値を伝播しない問題があった．
スケールドサムは実際の子の値を使いつつ，SNDA と協調して DAG 重複を割り引く
中間的近似である．

```
例: 子が 3 つ (pn = 3, 5, 2)

  標準 sum:   3+5+2 = 10
  旧 WPN:     max(5) + (3-1)*PN_UNIT = 5+32 = 37
  スケールドサム (γ=6):
              max(5) + (3+2)>>6 = 5+0 = 5  (非最大子が小さい場合)
              max(100) + (80+50)>>6 = 100+2 = 102  (非最大子が大きい場合)
```

#### OR dn WPN の有効化条件 (v0.31.0)

以前の試験 (v0.29.0 以前) では OR dn WPN により `test_no_checkmate_counter_check`
が失敗した (2M ノード予算超過)．これは初期 dn が全ノードで `PN_UNIT` (bucket 4 固定)
だったため，WPN による削減が予算配分を過度に絞った結果である．

v0.30.0 で `heuristic_dn_from_pn` を導入し初期 dn を bucket 4〜10 に拡張した
ことで OR dn WPN が安定して動作するようになった (§5.1 参照)．

**実装:** solver.rs (OR ノード collect), pns.rs (OR ノード伝播)

### 4.2 CD-WPN: Chain-Drop Weak Proof Number

**出典:** maou 独自手法

チェーン合駒(§8)に特化した WPN の変種．
チェーン合駒では同一マスへの異なる駒種の drop が子ノードとなるが，
これらは同一マスへの合駒として意味的にグループ化できる．

#### 式の変遷

| バージョン | AND pn |
|-----------|--------|
| 初期 | `max(child_pn) + (grouped_count - 1) * PN_UNIT` |
| **v0.35.0 スケールドサム** | `max(rep) + (sum(rep) - max(rep)) >> γ` |

#### スケールドサム式 (v0.35.0〜)

CD-WPN はドロップを `to_sq` でグループ化し，グループ代表値を用いてスケールドサムを適用する:

```
group_rep[sq] = min(child_pn of drops to sq)   // cross-deduce で最初の証明がグループを代表
CD-WPN pn = max(group_rep) + (sum(group_rep) - max(group_rep)) >> WPN_GAMMA_SHIFT
```

- 旧式 `max + (grouped_count - 1) * PN_UNIT` は非最大グループの pn 変化を伝播しない問題があった
- スケールドサムは WPN (§4.1) と同じ原理で実際のグループ代表値を使いつつ DAG 二重カウントを割り引く
- グループ代表値 = `min(cpn)`: cross-deduce により同一マスのドロップは一括証明されるため，
  最も小さい cpn が「このマスを詰めるコスト」を代表する

**実装:** solver.rs (AND ノード collect)

- `chain_king_sq` が `Some` の場合(チェーン AND ノード)に CD-WPN を適用
- `chain_king_sq` が `None` の場合は標準 WPN スケールドサムを使用
- 実装: `cd_sq_min_pn[sq]` でマスごとの min(cpn) を追跡し，`drop_squares_seen` ビット列でグループを走査

### 4.3 VPN: Virtual Proof Number (Saito et al. 2006)

**出典:** Saito et al. 2006

AND ノードの pn 計算で証明済み子(cpn=0)を除外する．
証明済み子は pn=0 で sum に影響しないが，子選択ループからのスキップにより
SNDA ペア収集と子選択の効率化に寄与する．

**実装:** solver.rs (AND ノード collect)

AND ノードの子収集ループで `cpn == 0` の子を `continue` で除外．

### 4.4 SNDA: Source Node Detection Algorithm (Kishimoto 2010)

**出典:** Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation" (AAAI 2010)

DAG(転置)による pn/dn の**過大評価**を検出・修正する．
同一のリーフノードが複数の子を通じて重複カウントされる場合，
source ハッシュに基づくグループ化で重複分を控除する．

```
  Without SNDA (overcounting):     With SNDA (corrected):

      OR (dn = 3+5 = 8)               OR (dn = max(3,5) = 5)
     / \                              / \
   AND  AND                         AND  AND
   dn=3 dn=5                       dn=3 dn=5
     \  /                            \  /
      \/                              \/
     LEAF  <-- same source           LEAF  <-- grouped by source
     dn=?                           deduction = (3+5) - max(3,5) = 3
                                    corrected dn = 8 - 3 = 5
```

**実装:** mod.rs (`snda_dedup`), solver.rs (OR/AND collect)

TT エントリに `source` フィールドを追加
(v0.24.0 で u64→u32 に圧縮，上位 32 bit 切り捨て．衝突確率は 2⁻³² で実用上十分)．
`(source, value)` ペアをソートし，同一 source グループで:

```
deduction = sum(group) - max(group)
```

控除後: `value' = raw_sum - total_deduction` (最低値 PN_UNIT)

- OR ノード: `(source, dn)` ペアで dn を補正 (WPN 適用後に実施)
- AND ノード: `(source, pn)` ペアで pn を補正 (WPN 適用後に実施)

#### WPN との適用順序と floor

SNDA は WPN スケールドサムの**後**に適用する:

```
1. WPN:  current_dn = max(cdn) + (sum(cdn) - max(cdn)) >> γ
2. SNDA: current_dn = snda_dedup(pairs, current_dn)
3. floor: current_dn = max(current_dn, max_cdn)   // OR dn のみ
```

floor (下限 = max(child_dn)) を設定する理由:
- SNDA のハッシュ衝突により，無関係なノードが同一グループに入ることがある
- 過剰控除が発生しても，単一の最大子 dn を下回らないことを保証する

AND pn も同様に `max(child_pn)` が WPN+SNDA 後の下限 (v0.20.24〜)．

**SNDA の適用範囲の限界:**
SNDA は直接の兄弟ノードが同一 source を持つ場合のみ補正する．
孫以下の深い DAG 合流 (異なる source ハッシュを持つ子が共通の孫を持つ場合)
は補正できない．この深い合流に対しては WPN が第一の対策となる．

**出典との差異:**
- 論文は親ポインタ追跡による共通祖先検出を提案するが，
  maou_shogi では source ハッシュ(リーフ位置キー)によるグループ化で近似
- 積極的 max 集約方式を採用: グループ内で最大値のみを残す
  (保守的方式 v0.11.0 → 積極的方式 v0.15.0 に移行)

---
