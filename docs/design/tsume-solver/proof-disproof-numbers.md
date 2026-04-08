# 証明数・反証数の計算

### 4.1 WPN: Weak Proof Number (Ueda et al. 2008)

**出典:** Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)

証明数の**二重計数問題**(double-counting problem)に対処する手法．
DAG 構造の探索木において，共有ノードが複数の親から重複してカウントされ
証明数が過大評価される問題を，分岐係数を組み込んだ推定量で解決する．

```
  Standard AND node             WPN AND node
  pn = sum(child_pn)            pn = max(child_pn) + (count - 1)

      AND                           AND
     / | \                         / | \
   OR  OR  OR                    OR  OR  OR
  pn=3 pn=5 pn=2               pn=3 pn=5 pn=2

  pn = 3+5+2 = 10              pn = max(3,5,2) + (3-1) = 7
```

標準: `pn(AND) = sum(child_pn)`
WPN: `pn(AND) = max(child_pn) + (unproven_count - 1)`

**実装:** solver.rs (AND ノード collect)

AND ノードの pn 合計を `max(cpn) + (unproven_count - 1)` で計算．
VPN (§4.3) による証明済み子の除外，SNDA (§4.4) による DAG 合流補正と併用．

**出典との差異:**
- 論文は OR/AND 両ノードに WPN を適用するが，maou_shogi では AND ノードのみに適用
- SNDA との併用時に過剰補正が発生する問題を v0.20.24 で修正:
  SNDA 控除後の pn が `max(child_pn)` を下回らないようフロアを設定

### 4.2 CD-WPN: Chain-Drop Weak Proof Number

**出典:** maou 独自手法

チェーン合駒(§8)に特化した WPN の変種．
チェーン合駒では同一マスへの異なる駒種の drop が子ノードとなるが，
これらは同一マスへの合駒として意味的にグループ化できる．

CD-WPN はドロップを `to_sq` でグループ化し，グループ数を `unproven_count` とする:

```
grouped_count = チェーン合駒の到達マス数(駒種ではなくマス数)
pn(AND) = max(child_pn) + (grouped_count - 1)
```

**実装:** solver.rs (AND ノード collect)

- `chain_king_sq` が `Some` の場合(チェーン AND ノード)に CD-WPN を適用
- `chain_king_sq` が `None` の場合は標準 WPN を使用

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

控除後: `pn' = raw_sum - total_deduction` (最低値 1)

- OR ノード: `(source, dn)` ペアで dn を補正
- AND ノード: `(source, pn)` ペアで pn を補正

**出典との差異:**
- 論文は親ポインタ追跡による共通祖先検出を提案するが，
  maou_shogi では source ハッシュ(リーフ位置キー)によるグループ化で近似
- 積極的 max 集約方式を採用: グループ内で最大値のみを残す
  (保守的方式 v0.11.0 → 積極的方式 v0.15.0 に移行)
- AND ノードでの SNDA + WPN 併用時の過剰補正を v0.20.24 で修正:
  SNDA 控除後の pn が `max(child_pn)` を下回らないようにクランプ

---

