# 閾値制御

df-pn の効率は「どの部分木にどれだけの探索予算 (閾値) を配るか」で決まる．本節は子へ渡す
閾値の計算 (1+ε)，巡回グラフでの完全性保証 (TCA)，値のスケール (PN_UNIT) を扱う．

### 3.1 1+ε トリック (Pawlewicz & Lew 2007)

**出典:** Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)

標準 df-pn は最有望子 c1 に `min(parent_th, second_best + 1)` の閾値を与える
(second_best = 2 番目に良い子の φ)．c1 の φ が second_best を 1 超えた瞬間に別の子へ
切り替わり，seesaw effect (スラッシング) が起きる．1+ε トリックは切替の余裕を増やして
1 訪問あたりの探索を深くする．

**実装:** `front_pn_dn_thresholds` (`search/expansion.rs`)．φ/δ 統一で次を計算する:

```
(thφ, thδ) = (thpn, thdn) if OR else (thdn, thpn)

child_thφ   = min(thφ, second_φ + second_φ/8 + 1)   // 1+ε 子 φ 閾値 (ε=1/8)
child_thδ   = new_thdelta_for_best_move(thδ)          // 残り δ 予算を best child に配分
```

子 φ 予算はノード内の 1+ε (ε=1/8, ×1.125) を採用する．標準式 `second_φ + 1` より
子の切替が減り閾値反復の再展開が減る (採用時の測定: 39te full 最小化 nodes −29%・
wall −13%，29te −25%; 最短手数・canonical PV・全 tests 不変)．**閾値は df-pn の効率のみに
作用し健全性に影響しない** (健全性は STRICT verify が担保する;
[loop-ghi §7.5](loop-ghi.md))．なお ε をさらに緩める (`second+second/8+4` 等) と
nodes は減るが verify/do_move 増で wall が悪化する (過緩和)．

`new_thdelta_for_best_move` は親の δ 予算から兄弟の δ 寄与を差し引いて best child に与える:

```
delta_except_best = sum_delta_except_best                      // 兄弟の δ 総和 (sum_mask 内)
if 未展開手が多い: delta_except_best += ((moves - idx) / 8).max(1)   // 未展開ペナルティ
if best が sum_mask 対象: delta_except_best += max_delta_except_best  // max 集約分を戻す
child_thδ = max(0, thδ - delta_except_best)
```

- `sum_delta_except_best` / `max_delta_except_best` は §[4.1](proof-disproof-numbers.md) の δ 集約
  (sum_mask による sum / max の使い分け) を反映する．
- 乗算的な 1+ε は **ノード内 (ε=1/8, 上式)** と **root の閾値成長 (§3.4, 各反復 1.7×)** の
  2 箇所で働く．root 側が深い探索への指数的余裕を，ノード内側が子切替 (seesaw) の抑制を担う．

### 3.2 TCA: Threshold Controlling Algorithm (Kishimoto & Müller 2008; Kishimoto 2010)

**出典:** Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008);
Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation" (AAAI 2010)

巡回グラフ (DCG) 上で df-pn は pn/dn を**過小評価**し不完全になりうる．ループ検出により子が
`(INF, 0)` 等を返すと兄弟の値が過小評価され，本来必要な探索が早期に打ち切られる．TCA は
ループ子が存在する間，閾値を拡張して兄弟の深い探索を促し，完全性を回復する．

**実装:** `search_impl` + `extend_search_threshold` (`search_result.rs`)．`inc_flag` による
TCA の実装形式は KomoringHeights で実装されている方式に基づく．`inc_flag` で制御する:

```
// search_impl 入口:
if expansion.does_have_old_child():        // 子が path 上の先祖を参照 (ループ/転置)
    inc_flag += 1
// 入口と各反復:
if inc_flag > 0:
    extend_search_threshold(curr, &mut thpn, &mut thdn)
        // thpn = max(thpn, curr.pn + 1);  thdn = max(thdn, curr.dn + 1)  (INF は除外)

// step_best_child: first-visit 子を展開したら対称に減算
if is_first_visit && inc_flag > 0:
    inc_flag -= 1     // terminal/budget で push されない子でも DEC して会計を対称に保つ
```

- **inc_flag DEC の対称性が要点**: first-visit の子を展開するたびに inc_flag を 1 減らす．
  terminal/budget で expansion が push されない子でも DEC を欠かすと，inc_flag が過剰に積もり
  閾値を過剰拡張し，OR ノードで次手を過剰展開してしまう (探索量が膨らむ)．
- ループ子検出 (`does_have_old_child`) は TT 経由で子が現探索パス上の局面を指すかで判定する
  ([loop-ghi.md §7](loop-ghi.md))．

**出典との差異:** 論文の乗算的拡張 (2×) は再帰で指数膨張する問題があるため，本実装は
「現在値 +1 まで閾値を引き上げる」加算的拡張を inc_flag のスコープ内でのみ適用する．

### 3.3 PN_UNIT スケーリング

pn/dn の 1 単位を定数 `PN_UNIT` (`mod.rs`, 既定 16) で表現する．初期値・加算定数・フロアを
PN_UNIT 単位で表すことで，閾値配分に解像度の余裕を持たせる (PN_UNIT=1 が素の df-pn に相当)．

| 区分 | 例 |
|------|----|
| 初期値 | `init_pn_dn_or` / `init_pn_dn_and` の base = `PN_UNIT`，加算は `PN_UNIT` 単位 ([heuristics §5.1](initial-heuristics.md)) |
| 加算定数 | edge cost，子閾値の `+1` 等 |
| 終端値 | `K_INFINITE_PN_DN = u64::MAX / 2 - 1` (INF センチネル)，`0` (証明/反証) はスケール対象外 |

PN_UNIT を上げると中間的な初期値 (例 1.5 単位相当) を表現でき，heuristic の解像度が上がる．
pn/dn は `PnDn = u64` で保持し，加算は INF へ向けて飽和 clamp する (`clamp_pn_dn`)．

### 3.4 閾値成長 (反復深化)

root では `search_impl_root` を未解決の間繰り返し，毎反復で閾値を `th = max(th, ⌊val × 1.7⌋ + 1)`
に拡大する ([search-architecture.md §2.3](search-architecture.md))．これが探索範囲を段階的に
広げる反復深化であり，1+ε の乗算的拡大を root レベルで実現する．停滞 (val が増えない) は
理論上起きず，打ち切りは budget / timeout が担う．

### 3.5 旧アーキとの差異

旧二エンジン期にあった depth-adaptive epsilon (denom を IDS depth で切替),
depth-limited disproof 格納閾値,チェーン用 pn_floor 等は，深さ制限 IDS と Dual TT を前提と
した機構であり統一 mid では廃止された．現行の horizon (深さ上限) 起因の偽反証は
**反証の scope 化** ([loop-ghi.md §7.2](loop-ghi.md)) で健全に扱う．
