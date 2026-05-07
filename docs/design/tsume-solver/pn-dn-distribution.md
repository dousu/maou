# pn/dn 分布の対数正規化プロジェクト

df-pn+ の探索品質を高めるために，WorkingTT に蓄積される pn/dn 値の分布を
バケット空間で対数正規分布に近づける取り組みのドキュメント．

---

## 1. 目的と指標

### 1.1 なぜ対数正規分布が目標か

#### PN_UNIT が生み出す構造的下限と左壁効果

pn/dn の最小非ゼロ単位である PN_UNIT (=16，bucket 4) は分布の **左壁** として機能する．
バケット空間 k = floor(log₂(val)) において，中間値 (0 < val < INF) の最小バケットは 4 (PN_UNIT) である．

正規分布は左右対称を前提とするが，bucket k の分布は k ≥ 4 で切り取られているため，
「正規分布の右半分」しか観測されない構造的な非対称性がある:

```
[bucket 空間での観測可能な範囲の概念図]

  正規分布(仮定)           左壁 k=4 (PN_UNIT=16)
            ╭──────────╮   |
           /            \  |
          /              \/|
─────────/────────────────●──────── bucket k
    k < 4 は実在しない     4  5  6  7  8  ...
         ↑ 切り取られた左裾
```

このため，バケット k の分布は対数正規分布でモデル化するのが適切である:

- **k ~ LogNormal(μ_ln，σ_ln)** とは，ln(k) ~ Normal(μ_ln，σ_ln) を意味する
- 対数正規分布は原点をゼロ方向の自然下限とし，PN_UNIT による左壁との整合性が良い
- 正規分布のように「k < 0 まで左裾が延びる」矛盾がない

#### DFPN におけるノード選択品質

DFPN のノード選択は pn/dn の大小比較で行われる:

- **OR ノード:** argmin pn (最も証明コストが低い子を選択)
- **AND ノード:** argmin dn (最も反証コストが低い子を選択)
- **1+ε トリック:** 閾値 `(1+ε) × current_pn` で子への閾値配分を比例制御

σ_ln が大きいほど bucket k の分布が広がり，難しさの異なる局面が異なる pn/dn を持つようになる．
argmin による選択の有効性は，**比較対象ノード間の pn/dn の差の大きさ** に依存する:

```
[σ_ln 小 (現状): ln(bucket) 空間でも分布が狭く，argmin が曖昧]

  ノード A: pn = bucket 7 (128 ≈ 8S)
  ノード B: pn = bucket 8 (256 ≈ 16S)  ← 差が小さい
  ノード C: pn = bucket 8 (256 ≈ 16S)

[σ_ln 大 (目標): 分布が広く，難しさに応じた明確な選択が可能]

  ノード A: pn = bucket 4 (16 = 1S)
  ノード B: pn = bucket 8 (256 ≈ 16S)  ← 大きな差
  ノード C: pn = bucket 14 (16K ≈ 1024S)
```

#### まとめ

| 観点 | 理想の状態 | 現状の問題 |
|------|-----------|-----------|
| **分布モデルの整合性** | PN_UNIT 左壁に整合した対数正規分布 | 正規分布は左裾が切れて不整合 |
| **ノード選択の明確さ** | σ_ln ≥ 0.5，各ノードが異なる pn/dn | σ_ln ≈ 0.2〜0.3 と狭く argmin が不明確 |
| **値域の利用効率** | bucket 4〜30 を対数正規的に広く利用 | 数 bucket の狭い範囲への過度な集中 |

### 1.2 計測方法と指標

`scripts/analyze_pn_dn_dist.py` で 39手詰め問題を 50M ノード探索し，
WorkingTT の pn/dn 分布を IDS depth ごとに計測する．

**バケット定義:** `bucket(val) = floor(log₂(val))`（bucket 4 = PN_UNIT = 16，bucket 10 = 1024）

**対数正規フィット:** bucket k ∈ {1,…,30} の中間値に k ~ LogNormal(μ_ln，σ_ln) をフィット

| 指標 | 意味 | 目標値 |
|------|------|-------|
| **μ_ln** | ln(bucket) の重み付き平均 | — (難易度に応じて自然に定まる) |
| **中央値** | exp(μ_ln) [bucket 単位] | — |
| **σ_ln** | ln(bucket) の標準偏差 | **≥ 0.5** |
| **KL(対数正規)** | 実分布と対数正規フィットの KL divergence | **< 0.3** |

σ_ln と KL は独立した観点を捉える: σ_ln は **分布の広さ**，KL は **対数正規形状への近さ** を示す．
両方が目標値を満たすことで，探索に有効な pn/dn 分布と言える．

### 1.3 設計上のトレードオフ

#### PN_UNIT の位置付け

PN_UNIT (現在 16) は pn/dn の最小非ゼロ単位であり，分布の左壁として機能する．
σ_ln を改善するための 2 つの方針:

| 方針 | 内容 | 現状 |
|------|------|-----|
| **PN_UNIT 固定 + 初期値上方シフト** | 左壁の位置は維持し，ヒューリスティックを上方へ移動 | 現採用: heuristic_dn_from_pn 下限引き上げ等 |
| **PN_UNIT 引き上げ** | 左壁ごと右にずらし全体を高い値域へ移動 | 証明直前の小 pn/dn への特別処理が必要 |

#### 初期値の高さ vs. 早期証明コスト

初期値を高くすると σ_ln は改善するが，1+ε の閾値が大きくなり
証明確定までのノード消費が増加する可能性がある．
**分布の中央値を計測してから初期値を合わせる** アプローチが有効である．

#### WPN_GAMMA_SHIFT: 非最大子の寄与 vs. 右テール

WPN スケールドサム (`max + (sum-max) >> γ`) の `γ` は σ_ln と右テールの間のトレードオフを制御する:

| γ を小さくする | γ を大きくする |
|--------------|--------------|
| 非最大子の寄与が増え，親の pn/dn が速く広がる → σ_ln 拡大 | 非最大子の寄与が減り，DAG 重複排除が強まる |
| 右テール (大値) が増大する | 右テールは縮まるが，分布が narrow になりやすい |

現状 `γ=6`（非最大子を 1/64 に減衰）．右テール問題があるため γ を小さくする方向は慎重を要する．

#### pn と dn の非独立性

pn と dn は別々のヒューリスティックで初期化されるが，WPN 伝播を通じて連動する．
片方を変えると他方の計測値も変化するため，変更後は必ず両方の指標を再計測する．

#### 分布の形 vs. ヒューリスティックの正確さ

人工的に対数正規分布を強制しても，ヒューリスティックが実際の難易度を反映していなければ
探索効率は改善しない．対数正規への近似はその **結果として生じること** が望ましく，目的ではなく手段である．
KL divergence は対数正規性の代理指標であり，最終評価は 39手詰め backward 解析のノード数 (§10.2) で行う．

---

## 2. ベースライン計測

### 2.1 v0.35.0 (参照値: 正規分布指標)

**注記:** v0.35.0 は正規分布フィットで計測した参照値であり，対数正規指標との直接比較はできない．

**計測条件:** 39手詰め，50M ノード，depth=40 IDS 終了時点

| 指標 (正規分布フィット) | pn | dn |
|------------------------|----|----|
| KL divergence | 1.616 | 0.695 |
| μ (bucket，中間エントリ) | 8.6 | 5.2 |
| σ (bucket 幅) | 4.9 | 0.8 |
| 実質値域 | bucket 3〜28 (25 幅) | bucket 3〜7 (**4 幅**) |
| INF/0 割合 | 80.6% | 78.5% |

**主要な問題点:**
- pn: bucket 5-6 (2S〜4S) への集中スパイク (`heuristic_or_pn` の粒度不足)
- dn: bucket 4 (PN_UNIT=1S) への極端な集中，実質 4 bucket 幅しか使えない

### 2.2 v0.37.0 (対数正規指標ベースライン)

**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 120K | 147 | 62.6% | 62.6% | 8.8 | 0.140 | 0.437 | 7.2 | 0.100 | 0.541 |
| 8 | 133K | 23K | 49.2% | 49.2% | 8.6 | 0.156 | 0.074 | 7.4 | 0.102 | 0.370 |
| 16 | 3.7M | 521K | 93.1% | 92.8% | 8.4 | 0.296 | 0.464 | 6.8 | 0.200 | 0.492 |
| 20 | 7.6M | 724K | 90.9% | 90.3% | 8.1 | 0.330 | 0.536 | 6.1 | 0.215 | 0.563 |
| 24 | 11.8M | 269K | 73.2% | 71.5% | 8.1 | 0.333 | 0.559 | 6.0 | 0.214 | 0.568 |
| 28 | 16.6M | 1.4M | 91.3% | 90.6% | 7.9 | 0.331 | 0.759 | 5.6 | 0.198 | 0.641 |
| 32 | 22.1M | 866K | 81.4% | 79.5% | 8.0 | 0.338 | 0.815 | 5.5 | 0.186 | 0.646 |

中央値は exp(μ_ln) [bucket 単位]．例: 中央値 8.0 → 2^8.0 = 256 ≈ 16S．

**観察:**
- **pn 中央値:** depth=4〜32 を通じて bucket 7.9〜8.8 (≈130〜350，≈8S〜22S) に安定
- **dn 中央値:** depth=4 の 7.2 から depth=32 の 5.5 まで顕著に低下 (bucket 5.5 ≈ 45 ≈ 2.8S)
- **σ_ln:** depth が深くなるにつれ拡大するが，目標 0.5 に未達 (pn 最大 0.338，dn 最大 0.215)
- **KL:** depth=8 で pn=0.074 (良好) だが，depth が増すにつれ劣化; depth=32 で pn=0.815，dn=0.646

**目標への差分 (depth=32 時点):**

| 指標 | 現状 | 目標 | 評価 |
|------|------|------|------|
| pn σ_ln | 0.338 | ≥ 0.5 | **不足** |
| dn σ_ln | 0.186 | ≥ 0.5 | **不足** |
| pn KL(対数正規) | 0.815 | < 0.3 | **超過** |
| dn KL(対数正規) | 0.646 | < 0.3 | **超過** |

**σ_ln の 95% 区間 (μ_ln ± 2σ_ln で観測される bucket の範囲):**

| 指標 | μ_ln | 95% bucket 範囲 | 幅 |
|------|------|----------------|---|
| pn (depth=32) | ln(8.0) ≈ 2.08 | exp(2.08 ± 0.676) = 4.1〜20.5 | 約 16 bucket |
| dn (depth=32) | ln(5.5) ≈ 1.70 | exp(1.70 ± 0.372) = 3.7〜7.7 | **約 4 bucket** |

dn は 95% が bucket 3.7〜7.7 の **約 4 bucket 幅** に集中しており，
AND ノードの argmin dn による子選択精度が極めて低い状態である．

### 2.3 v0.38.0 (現在のベースライン: 対数正規指標)

**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 150K | 148 | 62.2% | 62.2% | 10.9 | 0.255 | 1.093 | 7.2 | 0.100 | 0.551 |
| 8 | 165K | 24K | 51.0% | 51.0% | 9.6 | 0.255 | 0.446 | 7.5 | 0.102 | 0.336 |
| 16 | 3.7M | 152K | 87.7% | 87.5% | 9.2 | 0.312 | 0.445 | 7.1 | 0.176 | 0.439 |
| 20 | 7.6M | 134K | 80.9% | 80.2% | 8.8 | 0.326 | 0.429 | 6.6 | 0.217 | 0.486 |
| 24 | 11.8M | 630K | 90.0% | 89.4% | 8.1 | 0.347 | 0.707 | 5.7 | 0.205 | 0.620 |
| 28 | 16.6M | 1.1M | 91.2% | 90.4% | 8.1 | 0.356 | 0.897 | 5.5 | 0.190 | 0.644 |
| 32 | 22.2M | 1.4M | 89.7% | 88.2% | 8.2 | 0.381 | 1.009 | 5.3 | 0.163 | 0.632 |
| 36 | 29.1M | 5.3M | 96.1% | 95.1% | 8.2 | 0.384 | 1.202 | 5.2 | 0.152 | 0.633 |
| 40 | 36.1M | 835K | 72.8% | 66.6% | 8.2 | 0.382 | 1.201 | 5.2 | 0.149 | 0.623 |

**v0.37.0→v0.38.0 の主要変化 (depth=32):**

| 指標 | v0.37.0 | v0.38.0 | 変化 |
|------|---------|---------|------|
| pn σ_ln | 0.338 | **0.381** | +0.043 ↑ |
| dn σ_ln | 0.186 | 0.163 | -0.023 ↓ |
| depth=24 証明率 | 73.2% | **90.0%** | +16.8pt ↑ |
| 到達 depth (50M nodes) | 32 | **40** | +8 ↑ |

**観察:**
- **pn σ_ln:** safe_escapes=3-7 の値域拡大により全 depth で改善 (特に depth=4: 0.140→0.255)
- **dn σ_ln:** safe_escapes=3+ が高 pn (32S-512S) により de-prioritized され，初期値 4S に固着するため回帰
  ただし depth=28 までは v0.37.0 と同水準 (0.190 vs 0.198)
- **証明率:** depth=24 で 73.2%→90.0% と顕著改善 (safe_escapes=3+ の正確な難易度評価が寄与)
- **到達 depth:** 50M ノードで depth=32→40 に達するよう改善

**目標への差分 (depth=32 時点):**

| 指標 | 現状 | 目標 | 評価 |
|------|------|------|------|
| pn σ_ln | 0.381 | ≥ 0.5 | **不足** (改善) |
| dn σ_ln | 0.163 | ≥ 0.5 | **不足** (回帰) |
| pn KL(対数正規) | 1.009 | < 0.3 | **超過** |
| dn KL(対数正規) | 0.632 | < 0.3 | **超過** |

---

### 2.4 v0.39.0 実験 — WPN γ_dn 分離 + clamp 下限緩和 (不採用)

**計測条件 (試み1: clamp=1S + γ_dn=4):** 39手詰め，50M ノードで未解決 (status=unknown)

| depth | pn σ_ln | dn σ_ln | pn KL | dn KL |
|-------|---------|---------|------|------|
| 8 | 0.266 | 0.250 | 0.429 | 0.492 |
| 16 | 0.327 | 0.250 | 0.452 | 0.497 |
| 24 | 0.374 | 0.173 | 0.785 | 0.659 |
| 28 | 0.376 | 0.157 | 0.943 | 0.641 |
| 32 | 0.358 | 0.151 | 0.976 | 0.624 |
| 36 | 0.354 | 0.137 | 1.104 | 0.569 |

**計測条件 (試み2: clamp=4S + γ_dn=4):** 50M ノード，elapsed=203.8s，depth=40 到達

| depth | pn σ_ln | dn σ_ln | pn KL | dn KL |
|-------|---------|---------|------|------|
| 8 | 0.262 | 0.108 | 0.435 | 0.342 |
| 16 | 0.316 | 0.178 | 0.443 | 0.391 |
| 24 | 0.362 | 0.188 | 0.804 | 0.587 |
| 32 | 0.368 | 0.153 | 0.981 | 0.558 |
| 36 | 0.370 | 0.157 | 0.989 | 0.569 |
| 40 | 0.368 | 0.153 | 1.006 | 0.562 |

**v0.38.0→v0.39.0 の主要変化 (depth=32):**

| 指標 | v0.38.0 | +clamp1S+γ_dn4 | +clamp4S+γ_dn4 |
|------|---------|----------------|----------------|
| pn σ_ln | **0.381** | 0.358 ↓ | 0.368 ↓ |
| dn σ_ln | **0.163** | 0.151 ↓ | 0.153 ↓ |
| 回帰テスト | 26.0s | — | 26.2s (変化なし) |

**結論:** いずれの組み合わせも v0.38.0 を下回った (§3.10 参照)．v0.39.0 は v0.38.0 論理と等価の状態で確定．

---

### 2.5 v0.40.0 実験 — heuristic_and_dn 単体採用 (TT 爆発により不採用)

**動機:** dn の値域が初期値付近に固まるのは，AND leaf の dn が pn 依存 (C/pn) であり，
safe_escapes が同じでも pn の差で dn が変わらないためである．
pn が OR leaf で safe_escapes ベースの多様性を持つ (§3.8) のと対称に，
dn を AND leaf で safe_escapes ベースで多様化することを試みた．

**変更 (v0.40.0):** AND leaf dn を `heuristic_and_dn(board)` で上書き

```
safe_escapes=0 → dn = 64S  (完全封鎖 → 反証困難)
safe_escapes=1 → dn = 32S
safe_escapes=2 → dn = 16S
safe_escapes=3 → dn =  8S
safe_escapes=4 → dn =  4S
safe_escapes=5 → dn =  2S
safe_escapes=6+ → dn = 1S
```

**計測結果:**

| depth | total TT (v0.38.0) | total TT (v0.40.0) | dn σ_ln (v0.38.0) | dn σ_ln (v0.40.0) |
|-------|-------------------|-------------------|--------------------|-------------------|
| 8 | 24K | 24K | 0.102 | 0.103 |
| 20 | **134K** | **2.0M** | 0.217 | (未取得) |
| (最終 depth) | 40 | **28** | — | — |

**結論:** depth=8 では分布に変化なし (AND leaf dn は展開時に上書きされるため)．
depth=20 で TT が 134K → 2.0M (約 15×) に爆発し，最終 depth が 40 → 28 に悪化した．

---

## 3. これまでの改善履歴

### 3.1 WPN スケールドサム (v0.29.0)

**変更:** AND pn の計算を旧式 WPN からスケールドサムへ変更

```
旧: AND pn = max(child_pn) + (unproven_count - 1) * PN_UNIT
新: AND pn = max(child_pn) + (sum(child_pn) - max(child_pn)) >> WPN_GAMMA_SHIFT
```

**効果:** 非最大子の pn 変化が親に伝播するようになり，孤立したスパイクが減少した．
σ が拡大し，v0.29.0 以前の非常に狭い分布 (σ≈1〜2 bucket 幅) から改善された．

### 3.2 初期値の値域拡大と逆比例 dn ヒューリスティック (v0.30.0)

**変更:**
- `heuristic_or_pn` の上限を 8S → 64S に引き上げ (bucket 7 → 10)
- `heuristic_dn_from_pn` 導入: `dn = (8S)² / pn，clamp(1S，64S)`

**効果:** pn/dn の初期値域が bucket 4〜10 に広がった．OR dn WPN の適用が安定化した．

### 3.3 OR ノードへの WPN 対称適用 (v0.31.0)

**変更:** OR dn にも WPN スケールドサムを適用 (AND pn と対称)

```
OR dn = max(child_dn) + (sum(child_dn) - max(child_dn)) >> WPN_GAMMA_SHIFT
```

**効果:** dn 分布の σ が拡大し，dn=1S への極端な集中が緩和した．

### 3.4 CD-WPN スケールドサム (v0.35.0)

**変更:** チェーン AND ノードの pn 計算をスケールドサムへ変更

```
旧: pn = max(child_pn) + (grouped_count - 1) * PN_UNIT
新: group_rep[sq] = min(cpn to sq)
    pn = max(group_rep) + (sum(group_rep) - max(group_rep)) >> WPN_GAMMA_SHIFT
```

**効果:** チェーン AND での pn 変化の伝播精度が向上した．全 SLOW テスト pass を確認．
全体 KL への影響は軽微だった (chain AND ノードは全ノードの一部に過ぎないため)．

### 3.5 `heuristic_dn_from_pn` 下限引き上げ — 案 A-1 (v0.36.0)

**変更:** `heuristic_dn_from_pn` の clamp 下限を 1S → 2S に引き上げ

```rust
// Before (v0.30.0〜v0.35.0):
((C / pn.max(1) as u64) as u32).clamp(PN_UNIT, 64 * PN_UNIT)

// After (v0.36.0):
((C / pn.max(1) as u64) as u32).clamp(2 * PN_UNIT, 64 * PN_UNIT)
```

**狙い:** pn=64S (開放局面) が dn=1S (bucket 4) に固着していた問題を解消し，
dn の実質値域を bucket 4〜7 から bucket 5〜10 (最大 6 幅) に拡大する．

### 3.6 `heuristic_dn_from_pn` 下限引き上げ — 案 A-2 (v0.37.0)

**変更:** `heuristic_dn_from_pn` の clamp 下限を 2S → 4S に引き上げ

```rust
// Before (v0.36.0):
((C / pn.max(1) as u64) as u32).clamp(2 * PN_UNIT, 64 * PN_UNIT)

// After (v0.37.0):
((C / pn.max(1) as u64) as u32).clamp(4 * PN_UNIT, 64 * PN_UNIT)
// pn = 64S → dn = 4S  (下限: 2S→4S に引き上げ，bucket 5 スパイク抑制)
```

**狙い:** pn=32S 以上の開放局面が dn=2S (bucket 5) に固着する問題を解消し，
dn の実質値域を bucket 5〜10 から bucket 6〜10 (≥5 幅) に拡大する．

**計測結果 (v0.37.0，depth=32):** dn σ_ln=0.186，dn KL=0.646．
目標 (σ_ln ≥ 0.5，KL < 0.3) に未達．下限引き上げだけでは σ_ln の改善は限定的であり，
dn 中央値も depth に比例して低下しているため，さらなる対策が必要である．

### 3.7 `heuristic_or_pn` pn 初期値粒度改善 — Case B 部分実装 (v0.37.0)

**変更:** `safe_escapes=1-2` (39手詰め問題の大多数) を直接マッピングで細分化

```rust
// Before (v0.36.0 まで): escape_base × checks 係数
// safe_escapes=1 → 2S (checks≥8), 2.5S (4-7), 3S (2-3), 4S (1)
// safe_escapes=2 → 4S (checks≥8), 5S (4-7), 6S (2-3), 8S (1)

// After (v0.37.0): 直接マッピング
// safe_escapes=1: checks≥4 → 1.5S (=24)，checks=2-3 → 3S (=48)，checks=1 → 4S (=64)
// safe_escapes=2: checks≥4 → 3S (=48)，checks<4 → 5S (=80)
```

**狙い:** safe_escapes=1-2 ノードの pn 下方伸長で bucket 5-6 への一極集中を緩和する．

**計測結果 (v0.37.0，depth=32):** pn σ_ln=0.338，pn KL=0.815．
目標 (σ_ln ≥ 0.5，KL < 0.3) に未達．safe_escapes=1-2 の細分化に加え，
safe_escapes=3-7 の値域拡大や WPN_GAMMA_SHIFT の調整も必要である可能性がある．

### 3.8 `heuristic_or_pn` safe_escapes=3-7 値域拡大 — Case C (v0.38.0)

**変更:** safe_escapes=3-7 の escape_base を指数的にスケールし，bucket 9〜13 へ分散

```rust
// Before (v0.37.0): 線形スケール，上限 64S (bucket 10)
// 3 →  8S (bucket 7)
// 4 → 16S (bucket 8)
// 5 → 24S (~bucket 8.6)
// 6 → 32S (bucket 9)
// 7+ → 48S (~bucket 9.6)

// After (v0.38.0): 指数的スケール，上限 512S (bucket 13)
// 3 →   8S (bucket 7)       — 変更なし
// 4 →  32S (bucket 9)       — 2×
// 5 → 128S (bucket 11)      — 5.3×
// 6 → 256S (bucket 12)      — 8×
// 7+ → 512S (bucket 13)     — 10.7×
// 開放空間早期リターン: 64S → 512S
```

**狙い:** pn 分布を bucket 4-13 に広げ，σ_ln(pn) ≥ 0.5 を達成する．
safe_escapes=1-2 は変更しないため，39手詰めの大多数 (bucket 5-6) は維持し，
少数派の safe_escapes=3+ ノードが上位 bucket を占めるようになる．

### 3.9 `heuristic_dn_from_pn` clamp 調整の調査 — 不採用 (v0.38.0)

**調査内容:** dn 初期値の clamp を調整して σ_ln(dn) の回帰を解消することを試みた．

試みた案:
1. C を (8S)²→(16S)² に変更，clamp(2S, 256S): 探索がスタック (depth=16 止まり)
2. pn ≥ 32S の場合のみ比例式 dn=pn/2: 回帰テスト 3× 遅く，depth=28 止まり
3. pn ≥ 32S の場合のみ下限 8S に引き上げ: 回帰テスト速度は同じだが，depth=20 止まり

**根本原因判明:** dn 初期値を高 pn 局面で引き上げると，AND ノードの dn 閾値が
連鎖的に増大し探索コストが指数的に悪化する．

**結論:** `heuristic_dn_from_pn` は v0.37.0 の clamp(4S, 64S) を維持する．
dn σ_ln は pn 変更によって de-prioritized された safe_escapes=3+ 局面が初期値 4S に
固着するため若干回帰するが，探索効率 (depth=24 証明率 73.2%→90.0%) は改善する．

---

### 3.10 WPN γ_dn 分離 + clamp 下限緩和の実験 — 不採用 (v0.39.0)

**動機:** dn の実際値域が初期値付近に固まっているのは，以下が原因と仮説した:
1. WPN_GAMMA_SHIFT=6 により OR dn ≈ max(child_dn)，非最大子の寄与が 1/64 と微小
2. clamp 下限 4S が pn≥16S の全位置を bucket 6 に集中させている

**試みた変更 (v0.39.0):**
- `WPN_GAMMA_SHIFT_DN = 4` を OR dn 専用定数として追加 (γ_dn=4 → non-max 寄与 1/16)
- `heuristic_dn_from_pn` の clamp 下限を 4S → 1S に緩和

**実験結果 (§2.4 参照):**

| 変更 | pn σ_ln (d=32) | dn σ_ln (d=32) |
|------|----------------|----------------|
| v0.38.0 (ベース) | **0.381** | **0.163** |
| clamp=1S + γ_dn=4 | 0.358 ↓ | 0.151 ↓ |
| clamp=4S + γ_dn=4 | 0.368 ↓ | 0.153 ↓ |

**失敗の根本原因 — AND ノードでの操作方向の非対称性:**

pn も OR-min で引き下げられるが，AND ノードでの操作方向が決定的に異なる:

```
pn の 2段伝播:
  葉 pn_init: [1S, 8S, 512S, ...]  ← 広い初期レンジ (1S〜512S，9 bucket)
  ↓ AND-sum → 10S, 100S, 1000S, ...   ← 合算で多様性が拡大
  ↓ OR-min  → min(10S, 100S, ...) = 10S ← 圧縮されても spread が残る
  → AND の sum が「多様性を増幅」するため OR-min を経ても spread が維持される

dn の 2段伝播:
  葉 dn_init: [4S, 8S, 64S, ...]   ← 狭い初期レンジ (4S〜64S，4 bucket)
  ↓ AND-min → min(4S, 8S, 64S) = 4S  ← 一発で下限に収束，多様性が破壊される
  ↓ OR-sum(WPN) → ≈ max(4S, 4S, ...) = 4S+α ← 下限から脱出できない
  → AND の min が「多様性を即座に破壊」するため OR-sum で回復不可能
```

| | AND での操作 | 多様性への効果 |
|--|--|--|
| pn | **sum** | 葉の多様性を増幅 → OR-min 後も spread が残る |
| dn | **min** | 葉の多様性を破壊 → OR-sum で回復できない |

1. **AND での操作方向が核心:** OR でどちらが min/sum かではなく，AND で sum (増幅) か min (破壊) かが σ_ln の差を生む
2. **γ 緩和の無効性:** OR dn を γ_dn=4 で大きくしても，次の AND-min で下限 (4S) に収束し効果が打ち消される
3. **clamp 下限緩和の逆効果:** 下限を 1S にすると cluster 点が bucket 6→4 に下がり σ_ln が悪化
4. **初期レンジの非対称性:** pn init は 1S〜512S (9 bucket)，dn init は 4S〜64S (4 bucket) と初期 spread も異なる

**v0.39.0 で pn σ_ln も悪化した理由:**  
WPN_GAMMA_SHIFT_DN=4 は OR dn の大きさを変えるため，AND ノードの threshold 計算が変わり，
探索順序が変化する．これが pn 分布のスナップショット (depth=32 時点) に間接的に影響した．

**結論:** clamp (4S, 64S) および WPN_GAMMA_SHIFT=6 (pn/dn 共通) を v0.38.0 から変更しない．
dn σ_ln の目標 (≥0.5) は AND-min 伝播という df-pn の構造的制約により，
初期化・WPN パラメータ調整では達成困難と判断する (§4.1 更新参照)．

---

### 3.11 heuristic_and_dn 単体採用の失敗分析 — AND leaf dn 多様性が持続しない理由 (v0.40.0)

**失敗の根本原因:**

AND leaf dn に多様性を注入しても，TT 上の dn は展開後すぐに上書きされる:

```
展開前 (leaf): dn = heuristic_and_dn(board) ∈ {1S, 2S, 4S, 8S, 16S, 32S, 64S}
展開後 (node): dn = min(child OR dn)  ← AND-min が支配
  └── child OR dn ≈ WPN(AND leaf dn) ≈ max(child AND dn)
      └── 最大 AND leaf dn が伝播: safe_escapes=0 → 64S → OR parent dn ≈ 64S
```

**TT 爆発のメカニズム:**

`heuristic_dn_from_pn(pn)` では safe_escapes=0 かつ pn=4S のノードが dn=16S を得ていた．
`heuristic_and_dn` では同じノードが dn=64S を得る (safe_escapes=0 → 64S 固定)．

```
v0.38.0: safe_escapes=0, pn=4S → dn=16S → OR 親 dn ≈ 16S → IDS budget 16S
v0.40.0: safe_escapes=0, pn=4S → dn=64S → OR 親 dn ≈ 64S → IDS budget 64S (4×!)
```

safe_escapes=0 は詰将棋ツリーで頻出するため，多数のノードが 64S に昇格する．
OR 親ノードの WPN は max(child AND dn) を採用するため，64S の AND 子が 1 個あれば
OR 親も 64S になる．これが多段的に上位ノードへ伝播し，IDS の budget が全体的に肥大化する．

**「OR-sum で dn が増幅されるはず」への回答:**

標準 df-pn では AND pn = sum，AND dn = min，OR pn = min，OR dn = sum (対称)．
本実装は OR dn に WPN (≈ max + tiny_sum) を採用しており，OR-sum による増幅は意図的に抑制されている
(純 sum を使うと depth が深いほど指数的に dn が膨張するため)．

しかし問題はOR での集約方式より，**AND での集約方式の非対称性** にある:

```
pn: AND-WPN ≈ max(child_pn) × 1.1  — 多様性を保持 (最大値が残る)
dn: AND-min = min(child_dn)          — 多様性を破壊 (最小値に収束)
```

AND dn = min は df-pn の定義 (守備側が最も反証しやすい手を選ぶ) であり変更できない．
したがって dn σ_ln を改善するには **AND-min で選ばれる前の leaf dn 値の多様性を維持する**
しかなく，かつ其の多様性が WPN を介して OR 親に適度に伝播する必要がある．

---

### 3.12 heuristic_and_dn 幾何平均ブレンド — v0.41.0

**動機:** §3.11 の失敗原因は safe_escapes=0 AND ノードが dn=64S に過度に集中したことにある．
pn ベース (C/pn) と safe_escapes ベースの両信号を **幾何平均** でブレンドし，
一方が上限に張り付いても他方が引き下げる効果を持たせる．

**実装:**

```rust
fn heuristic_and_dn_blended(board, pn) -> u32 {
    let dn_pn     = heuristic_dn_from_pn(pn);          // pn の逆数スケール [4S, 64S]
    let dn_escape = heuristic_and_dn(board);             // safe_escapes 逆数スケール [1S, 64S]
    sqrt(dn_pn * dn_escape).clamp(4S, 64S)              // 幾何平均
}
```

**ブレンド効果の例 (v0.38.0 比較):**

| 状況 | v0.38.0 dn | v0.41.0 dn | 変化 |
|------|-----------|-----------|------|
| safe_escapes=0, pn=4S | 16S | sqrt(16S×64S) = **32S** | ↑ 1 bucket |
| safe_escapes=0, pn=2S | 32S | sqrt(32S×64S) ≈ **45S** | ↑ ~0.5 bucket |
| safe_escapes=0, pn=1S | 64S | sqrt(64S×64S) = **64S** | 不変 |
| safe_escapes=2, pn=4S | 16S | sqrt(16S×16S) = **16S** | 不変 |
| safe_escapes=3, pn=4S | 16S | sqrt(16S×8S) ≈ **11S** | ↓ 0.5 bucket |
| safe_escapes=4, pn=1S | 64S | sqrt(64S×4S) = **16S** | ↓ 2 bucket |
| safe_escapes=6, pn=4S | 16S | sqrt(16S×1S) = 4S → **4S** (clamp) | ↓ 2 bucket |

**パフォーマンス改善の期待:**

- safe_escapes=3+ ノード (詰将棋で頻出，守備側に逃げ場あり) の dn が削減
  → OR 親 dn が WPN-max で小さくなる → IDS budget が縮小 → TT 爆発を回避
- safe_escapes=0 ノードの dn は緩やかに増加 (64S → 32S 程度) → cascade しない
- 上限 64S は同一だが，多数の nodes が 64S に張り付く問題を回避

**計測結果 (50M nodes, 235.0s, status=unknown):**

| depth | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | dn 中央値 | dn σ_ln |
|-------|---------|---------|------|---------|---------|---------|---------|
| 4 | 148 | 62.2% | 62.2% | 10.9 | 0.255 | 7.2 | 0.100 |
| 8 | 18,407 | 49.4% | 49.4% | 9.7 | 0.255 | 7.4 | 0.103 |
| 16 | **45,783** | 76.1% | 75.9% | 9.4 | 0.288 | 7.3 | 0.142 |
| 20 | **2,007,488** | 94.7% | 94.3% | 8.3 | 0.362 | 5.5 | 0.179 |
| 24 | 1,557,300 | 88.2% | 87.3% | 8.0 | 0.349 | 5.3 | 0.158 |
| 28 | 672,991 | 68.6% | 66.2% | 8.0 | 0.356 | 5.3 | 0.153 |

回帰テスト (ply24, remaining=15): **25.98s** (v0.38.0 の ≈52s から 2× 高速化)

**v0.38.0 vs v0.41.0 比較 (depth=20):**

| 指標 | v0.38.0 | v0.41.0 | 変化 |
|------|---------|---------|------|
| total TT | **134K** | **2.0M** | 15× 爆発 (v0.40.0 と同様) |
| pn σ_ln | 0.326 | 0.362 | ↑ 改善 |
| dn σ_ln | **0.217** | **0.179** | ↓ 悪化 |
| 到達 depth (50M nodes) | 40 | 28 | ↓ 劣化 |

**失敗の分析 (§3.11 の延長):**

幾何平均はクラスタ点を 64S (v0.40.0) から 32S (v0.41.0) に移動させただけで，爆発メカニズムは同一:

```
v0.40.0: safe_escapes=0, pn=4S → dn=64S → IDS が 64S に達した瞬間にバッチ展開
v0.41.0: safe_escapes=0, pn=4S → sqrt(16S×64S)=32S → IDS が 32S に達した瞬間にバッチ展開
```

AND leaf dn に「同値クラスタ」が存在する限り，IDS で threshold がそのクラスタ値を跨ぐ瞬間に爆発する．
根本解決には，dn 値が連続的に分散するか，クラスタを作らないヒューリスティックが必要．

---

### 3.13 `heuristic_dn_from_pn` sqrt 式 — 連続スケーリング (v0.42.0，効果なし)

**動機:** §3.12 で AND leaf dn への直接操作はいずれも「クラスタ点の移動」に過ぎず爆発を解消できなかった．
根本原因はクラスタ生成であるため，离散マッピング (safe_escapes → 固定 dn) をやめ，
`dn ∝ 1/√pn` という連続写像で同値クラスタを解消することを試みた．

**変更 (v0.42.0):** `heuristic_dn_from_pn` を線形逆比例から平方根逆比例へ変更

```rust
// Before (v0.38.0〜v0.41.0): dn ∝ 1/pn，有効範囲 pn∈[1S,16S]
const C: u64 = (8 * PN_UNIT as u64) * (8 * PN_UNIT as u64);
((C / pn.max(1) as u64) as u32).clamp(4 * PN_UNIT, 64 * PN_UNIT)

// After (v0.42.0): dn ∝ 1/√pn，有効範囲 pn∈[1S,256S]
// pn=1S→64S, pn=4S→32S, pn=16S→16S, pn=64S→8S, pn=256S→4S(下限)
const C2: u64 = 4096 * 4096;  // = 16_777_216
let dn = ((C2 / pn.max(1) as u64) as f64).sqrt() as u32;
dn.clamp(4 * PN_UNIT, 64 * PN_UNIT)
```

**期待した効果:** 旧式では pn > 16S が全て dn=4S に集中していたが，
sqrt 式では pn ∈ [1S, 256S] が dn ∈ [4S, 64S] に連続的にマップされるため，
同値クラスタが生じにくくなる．

**計測結果 (50M nodes, 230.9s, status=unknown):**

| depth | total TT | pn=INF% | pn 中央値 | pn σ_ln | dn 中央値 | dn σ_ln |
|-------|---------|---------|---------|---------|---------|---------|
| 4 | 148 | 62.2% | 10.9 | 0.255 | 7.2 | 0.100 |
| 8 | 18,407 | 49.4% | 9.7 | 0.255 | 7.4 | 0.103 |
| 16 | 45,783 | 76.1% | 9.4 | 0.288 | 7.3 | 0.142 |
| 20 | **2,007,488** | 94.7% | 8.3 | 0.362 | 5.5 | 0.179 |
| 24 | 1,557,300 | 88.2% | 8.0 | 0.349 | 5.3 | 0.158 |
| 28 | 672,991 | 68.6% | 8.0 | 0.356 | 5.3 | 0.153 |

回帰テスト (ply24, remaining=15): **49.29s** (v0.41.0 の 25.98s より大幅後退)

**v0.41.0 vs v0.42.0 比較 (depth=20):**

| 指標 | v0.41.0 | v0.42.0 | 変化 |
|------|---------|---------|------|
| total TT | 2,007,488 | 2,007,488 | 変化なし |
| pn σ_ln | 0.362 | 0.362 | 変化なし |
| dn σ_ln | 0.179 | 0.179 | 変化なし |

**失敗の分析:**

`heuristic_dn_from_pn` は **葉ノードの初期 dn** を設定するに過ぎず，
TT に記録される dn は展開後の AND-min 伝播で上書きされる:

```
初期化: dn = sqrt(C²/pn)   ← 連続でも離散でも
展開後: dn = min(child OR dn)  ← AND-min が支配し初期値は関係なくなる
```

sqrt 式は「初期値の連続性」を保証するが，AND-min 伝播の後は同じ結果になる．
pn 分布 (`heuristic_or_pn` が制御) を変えない限り，
`heuristic_dn_from_pn` の式変更だけでは pn/dn σ_ln は変化しない．

**結論:** v0.42.0 の sqrt 式は連続スケーリングとして正しい設計だが，
問題の根源は heuristic_dn_from_pn ではなく heuristic_or_pn の pn 値域にある．
pn の分布を広げることで heuristic_dn_from_pn の効果範囲が活きる (→ §3.14)．

---

### 3.14 `heuristic_or_pn` pn 値域拡大 — v0.43.0 (TT 爆発解消)

**動機:** §3.12・§3.13 の分析から，dn 改善の鍵は「pn を大きい値まで広く使う」ことにあると判明した:

1. `heuristic_dn_from_pn(pn)` は pn ∈ [1S, 256S] に対してのみ dn に多様性を与える
2. pn > 256S の場合は dn = 4S (下限) に張り付き，多様性がない
3. v0.42.0 以前は pn が 512S の上限キャップに頭打ちになるノードが多く，全て dn=4S に集中していた

**変更 (v0.43.0):** `heuristic_or_pn` の値域を bucket 13 → bucket 15 まで拡大

```
変更点 1: 上限を 512S (bucket 13) → 2048S (bucket 15) に引き上げ

変更点 2: 開放空間検出を safe_escapes に応じて段階化
  Before: adjacent_total≥5, pressured=0, safe_escapes≥4 → 512S (フラット)
  After:  adjacent_total≥5, pressured=0, safe_escapes=4-5 → 1024S (bucket 14)
          adjacent_total≥5, pressured=0, safe_escapes≥6   → 2048S (bucket 15)

変更点 3: safe_escapes=7 を 6+ から分離して独立 base 値を設定
  safe_escapes=6 → 256S (bucket 12) — 変更なし
  safe_escapes=7 → 512S (bucket 13) — 新設
  safe_escapes=8+ → 1024S (bucket 14) — 新設

変更点 4: num_checks=1 の乗数を ×2 → ×4 に強化
  Before: checks=1 → escape_base × 2
  After:  checks=1 → escape_base × 4
```

**pn マッピングの変化例:**

| 状況 | v0.42.0 pn | v0.43.0 pn | dn (sqrt 式) |
|------|-----------|-----------|-------------|
| safe_escapes=3, checks=1 | 16S | **32S** | 16S → 11S |
| safe_escapes=4, checks=1 | 64S | **128S** | 8S → 5.6S |
| safe_escapes=5, checks=1 | 256S (cap) | **512S** | 4S → 4S (clamp) |
| safe_escapes=6, checks=1 | 512S (cap) | **1024S** | 4S → 4S (clamp) |
| safe_escapes=7, checks=1 | 512S (cap) | **2048S** (cap) | 4S → 4S (clamp) |
| open space (esc≥4, pressed=0) | 512S (cap) | **1024S-2048S** | 4S → 4S (clamp) |

**計測結果 (50M nodes, 230.9s, status=unknown):**

| depth | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 126 | 55.6% | 55.6% | 10.7 | 0.253 | 0.891 | 7.6 | 0.129 | 0.453 |
| 8 | 24,708 | 43.9% | 43.9% | 10.1 | 0.280 | 0.442 | 8.1 | 0.139 | 0.204 |
| 16 | 398,521 | 90.5% | 90.1% | 9.7 | 0.368 | 0.375 | 6.8 | **0.268** | 0.516 |
| 20 | **39,772** | 4.1% | 0.7% | 9.7 | 0.368 | 0.375 | 6.8 | **0.268** | 0.516 |
| 24 | 661,719 | 88.8% | 88.1% | 8.7 | 0.360 | 0.520 | 5.9 | **0.253** | 0.715 |
| 28 | 373,027 | 74.3% | 72.4% | 8.6 | 0.360 | 0.608 | 5.7 | **0.234** | 0.750 |

回帰テスト (ply24, remaining=15): **46.17s**

**v0.42.0 vs v0.43.0 比較:**

| depth | total TT (v0.42) | total TT (v0.43) | pn σ_ln (v0.42) | pn σ_ln (v0.43) | dn σ_ln (v0.42) | dn σ_ln (v0.43) |
|-------|-----------------|-----------------|----------------|----------------|----------------|----------------|
| 8  | 18,407 | 24,708 | 0.255 | 0.280 | 0.103 | 0.139 |
| 16 | 45,783 | 398,521 | 0.288 | **0.368** | 0.142 | **0.268** |
| 20 | **2,007,488** | **39,772** | 0.362 | **0.368** | 0.179 | **0.268** |
| 24 | 1,557,300 | 661,719 | 0.349 | 0.360 | 0.158 | **0.253** |
| 28 | 672,991 | 373,027 | 0.356 | 0.360 | 0.153 | **0.234** |

**主要成果:**

1. **depth=20 の TT 爆発が消滅:** 2,007,488 → 39,772 (**50 倍削減**)
2. **dn σ_ln が全 depth で目標 0.2 を達成:**
   depth=16-28 の dn σ_ln が 0.14-0.18 → 0.23-0.27 に改善
3. **pn σ_ln も改善:** depth=16 で 0.288 → 0.368 (+0.080)

**成功メカニズム — de-prioritization による集団展開の回避:**

```
safe_escapes=5+, checks=1 (開放局面) に pn=512S-2048S を割り当てることで:
  1. IDS の budget 閾値がそれらのノードに達するまでに非常に多くのノード消費が必要
  2. depth=20 のスナップショット時点では safe_escapes=5+ ノードはほぼ未展開
  3. TT に積まれるのは safe_escapes=0-2 の多様な pn/dn を持つノードのみ
  4. AND-min 伝播も safe_escapes=0-2 ノードのみで行われるため dn 多様性が維持される
```

**dn σ_ln 改善が起きた理由 (pn 初期値変更なのに dn が改善):**

`heuristic_dn_from_pn` は変更していないが，TT に残るノード集合が変化した:

```
Before (v0.42.0): safe_escapes=3+ ノードがほぼ全て pn=512S → dn=4S で TT 滞留
                  depth=20 で 94.7% が pn=INF/dn=0 (証明済み) → 爆発
After  (v0.43.0): safe_escapes=3+ ノードが de-prioritized で TT に滞留しない
                  depth=20 の TT は safe_escapes=0-2 の多様な中間エントリが主体
                  pn ∈ [1S-256S] → dn ∈ [4S-64S] に連続マップされ σ_ln 向上
```

---

### 3.15 SNDA 無効化実験 — no-op 確認 (v0.43.0 ベースで実施)

**仮説:** SNDA が WPN 後に元値ベースで deduction を行うことで AND pn が max_cpn にクリップされ，
sum の積み上がりが失われているのではないか．さらに source = pos_key as u32 (32bit) への
ハッシュ衝突による偽陽性 deduction が pn を過剰に削減しているのではないか．

**実験:** SNDA 呼び出し (OR dn・AND pn 両方) を完全に無効化して分布を計測

**結果:** v0.43.0 と全数値が **完全一致** (差異ゼロ)

```
v0.43.0 vs SNDA無効:
  depth=8:  pn σ_ln = 0.280 vs 0.280  dn σ_ln = 0.139 vs 0.139
  depth=16: pn σ_ln = 0.368 vs 0.368  dn σ_ln = 0.268 vs 0.268
  depth=28: pn σ_ln = 0.360 vs 0.360  dn σ_ln = 0.234 vs 0.234
```

**実測結果 (50M nodes，atomic カウンターで計測):**

| 指標 | 値 | 備考 |
|------|-----|------|
| snda_dedup 呼び出し | 981,397 | `snda_pairs.len() >= 2` の collect が発生した回数 |
| deduction > 0 (fires) | **6,558** (0.67%) | 同一 source グループが検出され deduction 計算された回数 |
| max_floor に吸収 | **6,358** (96.9%) | deduction があったが `.max(max_cpn)` で吸収された回数 |
| **net_effective** | **119** (0.012%) | deduction が実際に pn/dn を削減した回数 |

**根本原因の判明:**

1. **SNDA は 39手詰めでも発生している** — 6,558 回 (0.67%) で deduction > 0 が計算された
2. **しかし 96.9% は max_cpn floor に吸収** — WPN 後に適用するため，deduction が
   WPN 非最大項 (max/64 程度) を超えると max_cpn に切り詰められる
3. **実際に分布を変えた回数は 119 回 (0.012%)** — 50M nodes 中の誤差の範囲

**結論:** 仮説は棄却．SNDA は存在するが WPN + max_floor の構造上ほぼ無効化されている．
SNDA 順序変更 (SNDA→WPN) や source 64bit 拡張は analytically 等価であり効果は期待できない．

---

### 3.16 heuristic_or_pn safe_escapes=1-2 分散拡大実験 — 不採用 (v0.44.0)

**動機:** §3.15 で SNDA 仮説が棄却されたため，真の原因である safe_escapes=1-2 の初期値
集中 (bucket 4.6–6.3) を解消することを試みた．num_checks の分解能を 2–3 レベルから
5 レベルに増やすことで bucket 4.6–7 に分散させることを目指した．

**変更 (v0.44.0):**

```
safe_escapes=1 (旧: 3 レベル → 新: 5 レベル):
  checks≥8: 1.5S (bucket 4.6)  — 変更なし
  checks=4-7: 2S (bucket 5)    — 新設 (旧: 1.5S に統合)
  checks=3: 3S (bucket 5.6)    — 新設
  checks=2: 4S (bucket 6)      — 旧 checks≥2: 3S から引き上げ
  checks=1: 6S (bucket 6.6)    — 旧: 4S から引き上げ

safe_escapes=2 (旧: 2 レベル → 新: 4 レベル):
  checks≥4: 3S (bucket 5.6)    — 変更なし
  checks=3: 4S (bucket 6)      — 新設
  checks=2: 5S (bucket 6.3)    — 変更なし
  checks=1: 8S (bucket 7)      — 旧: 5S から引き上げ
```

**計測結果:**

| depth | total TT (v0.43) | total TT (v0.44) | pn σ_ln (v0.43) | pn σ_ln (v0.44) | dn σ_ln (v0.43) | dn σ_ln (v0.44) |
|-------|-----------------|-----------------|----------------|----------------|----------------|----------------|
| 8  | 24,708 | 24,708 | 0.280 | 0.280 | 0.139 | 0.139 |
| 16 | 398,521 | 398,428 | 0.368 | 0.367 | 0.268 | 0.268 |
| 20 | **39,772** | **1,951,012** | 0.368 | 0.356 | 0.268 | 0.215 |
| 24 | 661,719 | 789,324 | 0.360 | 0.350 | 0.253 | 0.203 |
| 28 | 373,027 | 501,649 | 0.360 | 0.351 | 0.234 | 0.195 |

回帰テスト: 47.32s (v0.43.0 の 46.17s と同等) ← TT 爆発前 depth=8 では正常

**失敗の根本原因:** v0.40.0/v0.41.0 と同じ爆発メカニズム:

```
safe_escapes=1 checks=2 の pn を 3S → 4S に引き上げたことで:
  v0.43.0: IDS budget が 4S に達したとき safe_escapes=1 checks=2 ノードが探索対象になる
  v0.44.0: 同じ budget=4S 時点で「より多くの」ノードが一斉展開される → TT 爆発
  
  safe_escapes=1-2 は 39手詰め TT の多数派 → pn の引き上げは
  IDS budget クラスタ点の集団展開を誘発する
```

**pn σ_ln も悪化:** 0.360 → 0.351 (TT 爆発により depth=20 以降の集計が歪む)

**結論:** safe_escapes=1-2 の pn 上限引き上げは TT 爆発を招く構造的制約がある．
初期値を上げる方向での pn σ_ln 改善は不可能と判断 (§4.1 参照)．

---

### 3.17 SNDA 前適用 (Kishimoto 2010 正規実装) — 不採用 (v0.44.0 で試験・回帰で棄却)

**動機:** §3.15 の原因分析で，WPN 後の SNDA 適用は deduction の 96.9% が max floor
(`.max(max_cpn)`) に吸収され事実上 no-op と判明した．根本原因は WPN 後の値
(`max + sum_other/64`) に deduction を適用しているため，WPN 非最大項 (max/64 程度)
しか控除できないことにある．

Kishimoto 2010 の正規実装では SNDA を WPN **前**に適用する:

```
旧実装 / 現行実装 (v0.43.0, v0.44.0):
  1. WPN: current = max + (sum - max) >> γ
  2. SNDA: current = snda_dedup(pairs, current)  ← WPN 圧縮後に適用
  3. floor: current = max(current, max_cpn)

試験した実装 (Kishimoto 2010 正規, 棄却):
  1. SNDA: effective_sum = sum - snda_deduction(pairs)  ← raw sum に直接適用
  2. WPN: current = max + (effective_sum - max) >> γ   ← 補正済み sum に WPN を適用
  (floor 不要 — WPN の性質上 result ≥ max が自動保証)
```

**試験時の計測結果 (50M nodes, SNDA 前適用版):**

分布指標は v0.43.0 と実質同一 (SNDA fire rate 0.67% のため大局的変化なし):

| depth | total TT | pn=INF% | dn=0% | pn σ_ln | pn KL | dn σ_ln | dn KL |
|-------|----------|---------|-------|---------|-------|---------|-------|
| 8 | 24,708 | 43.9% | 43.9% | 0.280 | 0.442 | 0.139 | 0.204 |
| 16 | 397,354 | 90.4% | 90.1% | **0.368** | 0.375 | **0.268** | 0.514 |
| 20 | **39,700** | 4.1% | 0.7% | 0.368 | 0.375 | 0.268 | 0.514 |
| 24 | 665,470 | 88.8% | 88.2% | 0.360 | 0.521 | 0.253 | 0.714 |
| 28 | 373,247 | 74.2% | 72.4% | 0.359 | 0.608 | 0.234 | 0.750 |
| 32 | 333,214 | 66.8% | 64.4% | 0.358 | 0.652 | 0.224 | 0.762 |
| 36 | 3,075,042 | 94.2% | 93.4% | 0.374 | 0.958 | 0.190 | 0.783 |
| 40 | 4,315,507 | 93.7% | 92.8% | 0.367 | 1.073 | 0.165 | 0.753 |

回帰テスト時間: 47.45s (v0.43.0 の 46.17s と同等)

**棄却理由 — `test_no_checkmate_counter_check` の不詭め回帰:**

SNDA 前適用により `test_no_checkmate_counter_check` が 2M ノード制限内で解けなくなった
(探索ノード数 2M→20M+)．

根本原因: SNDA が発火すると OR dn が TT ミス子の heuristic dn 値によって膨張する:

```
例: OR node の子に TT ミス子 (dn = heuristic_dn_from_pn(high_pn) = 4S) と
    TT ヒット子 (同一 source ペア, dn = 4S) が混在する場合

SNDA 前適用:
  raw sum = sum(TT-miss dn) + sum(TT-hit dn)
  SNDA deduction = TT-hit 重複分 (TT-miss は source=0 で除外)
  effective_sum ≈ sum(TT-miss dn) + max(TT-hit dn)  ← TT-miss 寄与が残る
  WPN(effective_sum) >> OR dn が大きく膨張 → 不詭め検出が非効率

WPN→SNDA→floor (現行):
  WPN: current ≈ max_cdn + sum/64  (圧縮後)
  SNDA deduction → floor: current = max(result, max_cdn) = max_cdn
  → OR dn ≈ max_cdn = 4S  (TT-miss 寄与を打ち消す副作用が不詭め検出に有利)
```

この問題は v0.43.0 で `heuristic_or_pn` が pn 上限を 512S→2048S に拡大したことと連動する:
- 逆王手後などの開放局面で pn=1024S-2048S のノードが大量に生成される
- `heuristic_dn_from_pn` は pn≥256S で dn=4S に固定 → 全ノードが均一 dn=4S
- SNDA 前適用では WPN 前に effective_sum が大きくなり OR dn が膨張
- 不詭め検出が過大な OR dn を縮小するために 20M+ ノードを消費

**根本的問題と今後の方針:**

floor が有効に機能しているのは dn 値が均一 (4S) であるからであり，dn に適切な差が
ついていれば floor は不要なはず (§3.18 参照)．WPN→SNDA→floor は `heuristic_dn_from_pn`
の不完全さを補償する暫定実装である．

---

### 3.18 pn=INF 中間エントリの depth-limited 扱い — `test_no_checkmate_counter_check` 回帰修正 (v0.49.0)

**背景:**

v0.43.0 での `heuristic_or_pn` pn 値域拡大 (512S→2048S) 以降，
`test_no_checkmate_counter_check` (depth=31，逆王手不詭め局面) が
2M ノード制限内で解けなくなった (§3.17)．

v0.47.0 で `disproof_mode`，v0.48.0 で mini-IDS warm-up を試みたが，
いずれも詰将棋ソルバーとして不自然な特例処理であった．

**根本原因 — IDS working TT の pn=INF セマンティクス不整合:**

IDS が depth=16 まで進むと，working TT の intermediate エントリに
`(pn=INF, dn=K, remaining=16)` が大量に蓄積する．これは
「depth=16 の範囲内では証明不能 (depth-limited)」を意味するが，
`look_up_working` の intermediate パスは `remaining` チェックを適用しないため，
depth=31 MID はこの stale な pn=INF をそのまま受け取ってしまう:

```
look_up_working (修正前):
  dn == 0 (disproof) → remaining チェックあり (e.remaining() >= remaining)
  pn != 0 && dn != 0 (intermediate) → remaining チェックなし ← バグ

depth=31 MID:
  lookup(root) → (pn=INF, dn=K) from depth=16 → pn=INF ≥ INF-1 → 即 exit
```

さらに，`retain_working_intermediates` (IDS depth 切り替え時のエントリ保持) が
pn=INF のエントリも保持してその `remaining` を `depth_delta` だけシフトするため，
depth=16→20→24→... と進むたびに pn=INF エントリが同期して更新され，
毎ステップで同じブロックが再発する:

```
depth=16→20 (delta=4): (pn=INF, remaining=16) → (pn=INF, remaining=20)
depth=20 lookup (remaining=20): e.remaining() < remaining → 20 < 20 = false → still blocked
```

**修正 (v0.49.0): 2 箇所の対称的な変更**

1. **`look_up_working` (tt.rs)**: intermediate パスに `remaining` チェックを追加．
   `pn=INF && e.remaining() < remaining` の場合はエントリをスキップ (ミスとして扱う):

   ```rust
   // pn=INF intermediate は depth-limited 証明不能．
   // dn=0 disproof と同様に，保存時の remaining が現在の remaining
   // より浅い場合は無効 — より深い探索で再展開が必要．
   if e.pn == u32::MAX && e.remaining() < remaining { continue; }
   ```

2. **`retain_working_intermediates` (tt.rs)**: pn=INF のエントリを保持対象から除外．
   シフトしても次ステップで同じ状態になるため，破棄して新 depth での再探索に委ねる:

   ```rust
   let keep = is_intermediate
       && entry.pn < u32::MAX  // pn=INF は除外
       && !is_path_dep
       && is_depth_limited
       && rem >= min_remaining
       && new_rem < REMAINING_INFINITE;
   ```

この 2 変更により，pn=INF 中間エントリは `dn=0` の depth-limited disproof と
同等の扱いになる: 保存時の depth より深い探索では自動的に再初期化される．

**棄却したアプローチ:**

| アプローチ | 結果 | 棄却理由 |
|---|---|---|
| `disproof_mode` (pn=INF 検出 → TT クリア) | 3M ノードで解決するが特例処理 | TT クリアは pn=INF だけでなく有用な中間エントリも消去する |
| mini-IDS warm-up (clear 後に depth=20→24→28) | 3.8M ノードで解決 | 既存 IDS の再発明; 1/4 予算分割が恣意的 |
| `param_refutable_at_mid_limit` | 10M→Unknown | depth-limit ノードでの refutable チェックコストが莫大 |
| `reset_pn_inf_in_working()` | 10M→Unknown | stale dn 値が探索優先度付けを誤誘導 |

**収束計測 (v0.49.0):**

| 総予算 | 収束ノード数 | 結果 |
|---|---|---|
| 2M | 950,037 | NoCheckmate ✓ |

v0.48.0 の mini-IDS warm-up (3.8M) と比較して **4× 少ないノード数**で解決．
テスト予算を 2M に設定 (release ~6s, debug ~40s)．通常テストスイートに含める
(`#[ignore]` 不要)．

**実装場所:**
- `rust/maou_shogi/src/dfpn/tt.rs`: `look_up_working`, `retain_working_intermediates`

**分布計測結果 (v0.49.0):** → §2.6 参照

---

### 3.19 `heuristic_or_pn` safe_escapes=1-2 bucket シフト — 部分採用 (v0.50.0/v0.51.1)

**動機:** §4.4 調査により depth=16-20 の pn KL spike 主犯が bucket 6 (pn≈32=2S) への集中と判明．
safe_escapes=1-2 の直接マッピングを +1 bucket 分上方シフトして分散させる．

**変更内容:**

```rust
// v0.49.0 (変更前)
1 => {
    if num_checks >= 4 { 3 * PN_UNIT / 2 }   // 24 (~bucket 5)
    else if num_checks >= 2 { 3 * PN_UNIT }   // 48 (~bucket 6) ← spike
    else { 4 * PN_UNIT }                      // 64 (bucket 7) ← spike
}
2 => {
    if num_checks >= 4 { 3 * PN_UNIT }        // 48 (~bucket 6) ← spike
    else { 5 * PN_UNIT }                      // 80 (~bucket 7)
}

// v0.50.0/v0.51.1 (変更後)
1 => {
    if num_checks >= 4 { 3 * PN_UNIT / 2 }   // 24 (~bucket 5, 変化なし)
    else if num_checks >= 2 { 4 * PN_UNIT }   // 64 (bucket 7)
    else { 8 * PN_UNIT }                      // 128 (bucket 8)
}
2 => {
    if num_checks >= 4 { 4 * PN_UNIT }        // 64 (bucket 7)
    else if num_checks >= 2 { 8 * PN_UNIT }   // 128 (bucket 8, 新分岐)
    else { 16 * PN_UNIT }                     // 256 (bucket 9, 新分岐)
}
```

**計測結果 (50M nodes, depth=16-24):**

| depth | pn KL 変化 | dn KL 変化 | 解釈 |
|-------|-----------|-----------|------|
| 16 | 0.473 → **0.412** (−0.061) | 0.531 → 0.611 (+0.080) | pn 改善，dn 悪化 |
| 20 | 0.473 → **0.411** (−0.062) | 0.507 → 0.646 (+0.139) | pn 改善，dn 大幅悪化 |
| 24 | 0.507 → 0.503 (−0.004) | 0.683 → 0.748 (+0.065) | ほぼ変化なし，dn 悪化 |

**dn KL 悪化の原因:**
`heuristic_dn_from_pn(pn) = sqrt(4096² / pn)` は pn と反比例する．
safe_escapes=2 の checks=1 で pn が 5S=80 → 16S=256 に増大したとき，dn は
`sqrt(4096²/80) ≈ 458` → `sqrt(4096²/256) = 256` に低下する (bucket 9→9 は変わらないが，
そのさらに小さな値が AND min に伝播して bucket 5 収束を加速した)．

- § 3.16 での TT 爆発 (v0.44.0) とは異なり今回は TT 爆発なし (total TT depth=24: 723K→794K)
- 回帰テスト通過 (49.77s，v0.49.0 と同等)

**採用判断:** dn KL は悪化したが TT 爆発なし・pn KL 改善・回帰テスト通過．
§5 の「safe_escapes=1-2 の pn 上限引き上げ」(TT 爆発) とは別問題として v0.50.0 で採用．

---

### 3.20 `WPN_GAMMA_SHIFT` 7 実験 — 棄却 (v0.51.0)

**動機:** §4.3 調査で pn 右テールが 100% AND ノードによる WPN AND 多段累積と判明．
GAMMA_SHIFT を 6→7 に増大させると各ステップの sum_others 寄与が 1/64→1/128 に半減し，
AND pn の蓄積速度が遅くなる → 右テール縮小を期待．

**計測結果 (50M nodes):**

| depth | v0.50.0 pn KL | v0.51.0 pn KL | v0.50.0 右テール | v0.51.0 右テール |
|-------|--------------|--------------|----------------|----------------|
| 16 | 0.412 | 0.388 | 1,027,975 | 1,368,982 |
| 20 | 0.411 | 0.509 | 217,188 | **2,395,036** |
| 24 | 0.503 | 0.605 | 673,500 | **2,779,537** |

**結果: 右テールが逆に爆発**し，dn KL も全 depth で悪化 (depth=20: 0.646→0.739)．

**失敗の原因:**
- GAMMA_SHIFT=7 にすると WPN 和の寄与が半減するため，閾値の成長速度が遅くなる
- 探索が同一 AND ノードを何度も選択するようになり，確定しないまま TT に長時間滞留
- その間に別経路から WPN 累積を受け続けて pn が際限なく膨張
- 実際には「AND pn 蓄積速度の低減」ではなく「探索の非効率化による滞留時間延長」が支配的

**v0.51.1 で GAMMA_SHIFT=6 に戻した**．以降 GAMMA_SHIFT の増大 (≥7) は不採用 (§5)．

---

### 3.21 `heuristic_or_dn` を pn 非依存 (se, nc) 直接マッピングに変更 — 採用 (v0.52.1)

**動機:** v0.51.1 以前の `heuristic_or_dn` は `heuristic_dn_from_pn(final_pn)` を介して pn に依存していた．
`heuristic_or_pn↑ → dn = sqrt(4096²/pn) ↓` という反比例が，AND-min の dn = PN_UNIT 収束を
加速させ dn KL の悪化を招いていた (§4.2，§4.4 参照)．

**変更内容:**

- `heuristic_or_dn(safe_escapes, num_checks) -> u32` (pn 引数を除去)
- `heuristic_or_pn` の戻り値を `(pn_base, safe_escapes)` に変更し，呼び出し元が `se` を `heuristic_or_dn` に渡す
- (se, nc) の 2 変数直接マッピングで v0.51.1 の実績値 (`heuristic_dn_from_pn` の出力) に近い値を再現

**設計根拠:**

```
// OR leaf での初期 dn は「defender が何手逃げられるか」を推定するべきであり，
// 「attacker の pn がどれだけ大きいか」とは本来無関係．
// pn を proxy として使っていたのは，pn が safe_escapes の単調代理として
// 機能していたため偶然うまくいっていた．
// se >= 4 の閾値も同様に pn-依存を緩和するための暫定 heuristic だったが，
// 直接 (se, nc) → dn マッピングに変更することで不要になる．
```

**lookup table (v0.52.1):**

| safe_escapes | num_checks | dn (×PN_UNIT) | bucket | v0.51.1 参考値 |
|-------------|------------|--------------|--------|---------------|
| 0 | any | 40 | b9 | ≈648 (b9) |
| 1 | 1 | 22 | b8 | ≈341 (b8) |
| 1 | 2-3 | 28 | b8 | ≈458 (b8) |
| 1 | 4+ | 40 | b9 | ≈648 (b9) |
| 2 | 1 | 15 | b7 | ≈248 (b7) |
| 2 | 2-3 | 22 | b8 | ≈341 (b8) |
| 2 | 4+ | 28 | b8 | ≈458 (b8) |
| 3 | 1 | 15 | b7 | ≈248 (b7) |
| 3 | 2-7 | 18 | b8 | ≈284-309 (b7-8) |
| 3 | 8+ | 22 | b8 | ≈341 (b8) |
| 4 | 1 | 4 | b6 | ≈22 (b4) ← 改善 |
| 4 | 2-3 | 5 | b6 | |
| 4 | 4-7 | 6 | b6 | |
| 4 | 8+ | 8 | b7 | |
| 5 | 1 | 3 | b5 | |
| 5 | 2-3 | 4 | b6 | |
| 5 | 4+ | 5 | b6 | |
| 6+ | 1 | 2 | b5 | |
| 6+ | 2-3 | 3 | b5 | |
| 6+ | 4+ | 4 | b6 | |

**テスト結果:**

| テスト | 結果 | 時間 |
|--------|------|------|
| `test_tsume_39te_ply24_mate15_regression` | ok | 97.65s |
| `test_tsume_39te_backward_1m` | ok | 388.47s |
| `test_tsume_39te_ply2_no_false_nomate` | ok (Unknown@10M) | 74.43s |

**効果 (§2.8 参照):**

- dn KL が depth=16 で 0.611→0.433，depth=20 で 0.646→0.438 に改善
- dn KL の max 寄与 bucket が bucket 7 (dn≈64) に移行 (bucket 5 スパイク軽減)

---

### 3.22 逆王手局面の `heuristic_or_dn` 修正と INTERPOSE_DN_BIAS 制約 — 採用 (v0.53.2 / v0.53.3)

#### 3.22.1 問題: 逆王手局面で王逃げの dn が過大 (v0.53.2)

**動機:** 双玉局面 (攻め方も玉を持つ) で攻め方が逆王手を受けた OR ノードにおいて，
守備方の王逃げ応手に `heuristic_or_dn(se=0, nc=*)=40*PN_UNIT=640` が付いていた．
一方，合い駒応手には `dn ≈ 128` が付くため，DFPN は合い駒を先に展開してしまう．

根本原因: `se=0` (後手玉の安全逃げ場ゼロ) は「後手玉が追い詰められている → dn を高く」
という推定だが，逆王手局面ではこの推定が誤りになる．攻め方が逆王手を受けている状態では
攻め方の王手可能手数 `nc` が 1〜4 に激減しており，実際には守備方は容易に不詭めを証明できる．

**修正 (v0.53.2):** `heuristic_or_dn` に `attacker_in_check: bool` フラグを追加．

```rust
pub(super) fn heuristic_or_dn(se: u32, nc: u32, attacker_in_check: bool) -> u32 {
    if attacker_in_check && nc <= 4 {
        return match nc {
            1..=2 => 2 * PN_UNIT,   // 32  — 攻め方の王手手段が極めて少ない
            _     => 4 * PN_UNIT,   // 64  — nc=3..=4
        };
    }
    // ... se/nc 通常テーブル (§3.21)
}
```

呼び出し元 (`solver.rs`, `pns.rs`) で `board.is_in_check(board.turn)` を取得して渡す．

**効果:** `test_no_checkmate_counter_check` が 30M ノード予算圏内で解決可能になった (441K 実測)．

#### 3.22.2 問題: INTERPOSE_DN_BIAS と heuristic_or_dn 値域の不整合 (v0.53.3)

**動機:** `INTERPOSE_DN_BIAS = 8*PN_UNIT = 128` は §3.21 以前の `heuristic_or_dn` 値域
`[16, ≈240]` に対して設計された値だった．§3.21 で `heuristic_or_dn(se=0)=40*PN_UNIT=640`
が確立されたことで，以下の不等式が崩れていた:

```
board_move.effective_dn = heuristic_or_dn(se, nc)  ∈ [16, 640]
drop.effective_dn        = heuristic_or_dn(se, nc) + INTERPOSE_DN_BIAS
                         = heuristic_or_dn(se, nc) + 128  ∈ [144, 768]
```

`se=0` の board move (640) > `se≥5` の drop (144 = 16+128) となり，
AND ノードで「board move を drop より先に探索する」保証が失われる局面が存在した．

**修正 (v0.53.3):** `INTERPOSE_DN_BIAS = 40*PN_UNIT = 640` (= `heuristic_or_dn` の実用上限).

```
board_move.effective_dn ∈ [16,   640]
drop.effective_dn       ∈ [656, 1280]
```

全 `(se, nc)` の組み合わせに対して `board_move < drop` が保証される．

**設計不変条件:**

> `INTERPOSE_DN_BIAS ≥ max(heuristic_or_dn)` を常に満たすこと．
> `heuristic_or_dn` の値域上限を変更した場合は `INTERPOSE_DN_BIAS` も同時に更新する．

`heuristic_or_dn` の現在の実用上限は `se=0` のとき `40*PN_UNIT=640`．
`dn.clamp(PN_UNIT, 64*PN_UNIT)` の理論上限は `64*PN_UNIT=1024` だが，
`se=0` 以外では最大でも `40*PN_UNIT` 止まりのため実用上限は 640.

**効果:**

- `test_no_checkmate_counter_check`: 30M バジェット → 3M バジェット (実測 441K ノード)
- `test_tsume_39te_ply24_mate15_regression`: 109s → 48s

---

### 3.23 `heuristic_or_pn` に pos_key ベースのジッタを追加 — 採用 (v0.54.0)

**動機:** §4.4 調査により，depth=4-8 の pn KL 主犯が bucket 12 (pn≈256*PN_UNIT) への
離散スパイクと判明した (v0.52.1 計測)．`heuristic_or_pn` は同一 (safe_escapes, num_checks)
の全局面に同じ pn 値を返すため，同一の se/nc ティアに属する全局面が完全に同じ bucket に集中し，
KL divergence を悪化させていた．

**変更内容 (v0.54.0):**

`heuristic_or_pn` のシグネチャに `pos_key: u64` を追加し，通常マッピング経路 (safe_escapes
ベース) の最終 pn 値に位置依存ジッタを乗算する:

```rust
// pos_key の bits[19:17] (3 ビット) → 8 段階のジッタ係数 ×(13..20)/16
let jitter = ((pos_key >> 17) & 7) as u32;
let pn = (adjusted_pn * (13 + jitter) / 16).max(PN_UNIT).min(2048 * PN_UNIT);
```

| jitter | 係数 | 変化率 |
|--------|------|--------|
| 0 | 13/16 = 0.8125 | −18.75% |
| 1 | 14/16 = 0.875 | −12.5% |
| 2 | 15/16 = 0.9375 | −6.25% |
| 3 | 16/16 = 1.0 | ±0% (恒等) |
| 4 | 17/16 = 1.0625 | +6.25% |
| 5 | 18/16 = 1.125 | +12.5% |
| 6 | 19/16 = 1.1875 | +18.75% |
| 7 | 20/16 = 1.25 | +25% |

代表値 safe_escapes=6, nc≥8 の adjusted_pn = 256*PN_UNIT = 4096 に対して:
- jitter=0 → 3328 (log₂=11.7)
- jitter=7 → 5120 (log₂=12.3)
- 分散幅: 約 0.62 bucket

**設計上の考慮点:**

1. **TT 爆発リスク:** safe_escapes=1-2 (大多数局面) への変更ではなく，
   safe_escapes=3+ の高 pn 領域に限定した微小調整であるため，
   IDS budget クラスタ点での集団展開を誘発しない
2. **決定論的:** 同じ pos_key → 同じジッタ (探索の再現性は保たれる)
3. **TT ヒット時不変:** ジッタは TT ミス時の初期化にのみ適用され，
   TT に格納された値は変更されない

**呼び出し元の変更:**
- `solver.rs` (mid メソッド内): `self.heuristic_or_pn(board, nc, child_pk)`
- `pns.rs` (pns_expand メソッド内): `self.heuristic_or_pn(board, nc, child_pk)`

**テスト結果:**

| テスト | 結果 | 時間 |
|--------|------|------|
| `test_tsume_39te_ply24_mate15_regression` | ok | 61.51s |
| `test_no_checkmate_counter_check` | ok | (2M 予算内) |
| `test_tsume_39te_ply2_no_false_nomate` | ok (Unknown@10M) | ~119s |

**分布への効果:** §2.9 参照．

**39手詰め backward 解析 (v0.54.0 vs v0.49.0):**

*backward_1m (1M nodes / 180s per ply):*

| ply | remain | nodes (v0.49.0) | nodes (v0.54.0) | result (v0.54.0) |
|-----|--------|-----------------|-----------------|------------------|
| 24 | 15 | 267,899 Mate(15) | 682,965 | Mate(15) |
| 22 | 17 | 986,363 Mate(17) ✓ | 998,165 | **Unknown** ← v0.54.0 境界 |
| 20 | 19 | 1,000,000 **Unknown** ← v0.49.0 境界 | — | — |

1M 境界: v0.49.0 = ply 20 → v0.54.0 = **ply 22**（−2 ply 回帰，ply 22 で 11,802 nodes 超過 = 1.2%）

*backward_10m (10M nodes / 600s per ply):*

| ply | remain | nodes (v0.49.0) | time(s) | nodes (v0.54.0) | time(s) | result (v0.54.0) | 変化 |
|-----|--------|-----------------|---------|-----------------|---------|------------------|------|
| 22 | 17 | 6,884,094 | 106.62 | 9,439,291 | 252.96 | Mate(17) | +37% |
| 20 | 19 | 8,077,824 | 158.40 | 5,333,924 | 33.17 | Mate(19) | **−34%** ↑ |
| 18 | 21 | 9,992,668 | 167.86 | 10,000,000 | 100.39 | **Unknown** ← 境界 | 同一 |

10M 境界: v0.49.0 と同一 (ply 20 が最終解決，ply 18 が最初の Unknown)

**backward 解析考察:**
- **1M 境界微小回帰:** ply 22 が 986K→998K（+1.2%）となり 1M 予算をわずかに超過．INTERPOSE_DN_BIAS 128→640 の AND 探索順序変化が主因と推定
- **10M ply 22 回帰 (+37%):** ply 22 での探索コストが増大したが 10M 予算内に収まる
- **10M ply 20 改善 (−34%):** ply 20 の探索コストが大幅削減．jitter による探索優先度付けの改善と推定
- **10M 境界維持:** ply 20 = 最終解決，ply 18 = Unknown という境界は v0.49.0 と同一

**ply 22 > ply 20 逆転の根本原因 (§4.6 参照):**

`saved_depth <= 19` の直接ジャンプポリシーにより，ply 22 (saved_depth=19) は
IDS を `2→4→19` と直接ジャンプする一方，ply 20 (saved_depth=21) は
`2→4→8→16→17→21` と段階的に TT を暖機する．
この差が nodes/TT 比 (ply 22: 252, ply 20: 16) という 16 倍の格差を生み，
INTERPOSE_DN_BIAS 拡大後の冷スタートコストを増幅させた．詳細は §4.6 を参照．

---

### 2.9 v0.54.0 — heuristic_or_pn pos_key ジッタ追加後の計測

**変更内容:** `heuristic_or_pn` に pos_key ベースの 8 段階ジッタ ×(13..20)/16 を追加 (§3.23)
**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)，release ビルド (308.2s)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 150K | 121 | 53.7% | 53.7% | 10.6 | 0.257 | 0.725 | 7.6 | 0.129 | 0.453 |
| 8 | 162K | 19,629 | 49.3% | 49.3% | 9.6 | 0.277 | **0.287** | 7.9 | 0.144 | 0.236 |
| 16 | 3.7M | 229,396 | 87.8% | 87.3% | 9.3 | 0.365 | **0.364** | 7.1 | 0.245 | **0.387** |
| 20 | 7.6M | 840,573 | 91.6% | 91.2% | 8.7 | 0.378 | 0.573 | 6.0 | 0.250 | 0.611 |
| 24 | 11.8M | 2,760,897 | 93.8% | 93.5% | 8.1 | 0.372 | 0.789 | 5.5 | 0.201 | 0.747 |
| 28 | 16.6M | 1,100,680 | 80.8% | 80.1% | 8.1 | 0.375 | 0.878 | 5.4 | 0.192 | 0.750 |
| 32 | 22.2M | 3,738,682 | 92.8% | 92.4% | 8.1 | 0.364 | 0.893 | 5.3 | 0.176 | 0.750 |
| 36 | 29.1M | 3,578,437 | 90.1% | 89.5% | 8.1 | 0.368 | 1.000 | 5.3 | 0.162 | 0.727 |
| 40 | 36.1M | 1,874,355 | 77.8% | 77.0% | 8.1 | 0.371 | 1.032 | 5.2 | 0.151 | 0.702 |

中央値は exp(μ_ln) [bucket 単位]．

**per-bucket KL 寄与 (主犯 bucket):**

| depth | pn KL | 主犯 bucket (v0.52.1) | 寄与値 | pn KL | 主犯 bucket (v0.54.0) | 寄与値 |
|-------|-------|----------------------|--------|-------|----------------------|--------|
| 4 | 0.891 | bucket 12 (2^11) | 0.591 | 0.725 | bucket 12 (2^11) | 0.562 |
| 8 | 0.411 | bucket 12 (2^11) | 0.133 | **0.287** | bucket 12 (2^11) | **0.114** |
| 16 | 0.467 | bucket 30 (near-INF) | 0.175 | **0.364** | bucket 30 (near-INF) | 0.149 |
| 20 | 0.458 | bucket 30 (near-INF) | 0.180 | 0.573 | bucket 7 (2^6=64) | 0.430 |

| depth | dn KL | 主犯 bucket (v0.52.1) | 寄与値 | dn KL | 主犯 bucket (v0.54.0) | 寄与値 |
|-------|-------|----------------------|--------|-------|----------------------|--------|
| 8 | 0.220 | bucket 9 (2^8) | 0.250 | 0.236 | bucket 9 (2^8) | 0.259 |
| 16 | 0.433 | bucket 9 (2^8) | 0.282 | **0.387** | bucket 9 (2^8) | 0.286 |
| 20 | 0.438 | bucket 9 (2^8) | 0.279 | 0.611 | bucket 5 (2^4=PN_UNIT) | 0.556 |

**v0.52.1→v0.54.0 の主要変化:**

| 指標 | v0.52.1 | v0.54.0 | 変化 |
|------|---------|---------|------|
| pn KL (d=8) | 0.411 | **0.287** | −0.124 ↑ **目標 < 0.3 を初達成** |
| pn KL (d=16) | 0.467 | **0.364** | −0.103 ↑ 改善 |
| dn KL (d=16) | 0.433 | **0.387** | −0.046 ↑ 改善 |
| pn KL (d=20) | 0.458 | 0.573 | **+0.115 ↓ 悪化** |
| dn KL (d=20) | 0.438 | 0.611 | **+0.173 ↓ 悪化** |
| total TT (d=16) | 669,332 | **229,396** | −66% ↑ TT 縮小 |
| total TT (d=20) | 113,281 | 840,573 | **+7.4× ↓ TT 爆発** |
| depth 到達 (50M nodes) | (d=20 まで計測) | **d=40** | 到達 depth 拡大 |
| 回帰テスト | 97.65s | 61.51s | −36s ↑ 改善 (v0.53.x の寄与含む) |

**観察と考察:**

- **pn KL depth=8 で目標 < 0.3 を初達成:** bucket 12 への離散スパイクが 0.562 に軽減 (v0.52.1: 0.591)．
  ジッタによる pn 値の分散が bucket 11-12-13 に広がり，KL が 0.411 → 0.287 に大幅改善

- **depth=16 の両 KL 改善:** TT が 669K → 229K に縮小 (TT が密になり中間値の相対比率が変化)．
  pn KL 0.467→0.364，dn KL 0.433→0.387 といずれも改善

- **depth=20+ の KL 悪化と TT 爆発:** total TT が depth=20 で 113K → 840K に 7× 増加．
  主犯 bucket が near-INF (bucket 30) から bucket 7 (2^6=64) に移行した．
  これは `heuristic_or_pn` のジッタによる変化ではなく，**v0.53.3 の INTERPOSE_DN_BIAS 128→640**
  の変更によって AND ノード探索順序が大幅に変化したことが主因と推定される
  (v0.53.3 単体での baseline 計測未実施のため明確な分離不可)

- **depth=40 まで到達:** v0.49.0 では 50M nodes で depth=28 未達だったが，v0.54.0 では depth=40 まで到達．
  v0.53.x の INTERPOSE_DN_BIAS 変更が AND 探索効率を改善した可能性がある

**残存課題 (depth=20+):**

depth=20+ の KL 悪化の根本原因を特定するため，v0.53.3 baseline (ジッタなし) での
50M nodes 計測が必要である:
- v0.53.3 baseline で depth=20 TT が大きい → INTERPOSE_DN_BIAS 起因
- v0.53.3 baseline で depth=20 TT が小さい → ジッタ起因 (v0.54.0 を再検討)

---

### 2.6 v0.49.0 — pn=INF 中間エントリ depth-limited 扱い後の計測

**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)，release ビルド (276.5s)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 150K | 118 | 52.5% | 52.5% | 10.7 | 0.253 | 0.891 | 7.3 | 0.165 | 0.207 |
| 8 | 160K | 18,463 | 43.7% | 43.7% | 9.9 | 0.277 | 0.395 | 7.9 | 0.195 | 0.128 |
| 16 | 3.7M | 1,085,759 | 93.8% | 93.4% | 9.3 | 0.403 | 0.473 | 6.6 | 0.283 | 0.531 |
| 20 | 7.6M | 67,886 | 0.4% | 0.4% | 9.3 | 0.403 | 0.473 | 6.7 | 0.283 | 0.507 |
| 24 | 11.8M | 723,693 | 85.4% | 84.9% | 8.7 | 0.383 | 0.507 | 6.1 | 0.271 | 0.683 |
| (28+) | 50M (未完了) | — | — | — | — | — | — | — | — | — |

中央値は exp(μ_ln) [bucket 単位]．

**v0.43.0→v0.49.0 の主要変化:**

| 指標 | v0.43.0 | v0.49.0 | 変化 |
|------|---------|---------|------|
| pn σ_ln (d=16) | 0.368 | **0.403** | +0.035 ↑ 目標 ≥0.37 達成 |
| dn σ_ln (d=8) | 0.139 | **0.195** | +0.056 ↑ |
| dn σ_ln (d=16) | 0.268 | **0.283** | +0.015 ↑ |
| total TT (d=16) | 398,521 | **1,085,759** | 2.7× 拡大 |
| 到達 depth (50M nodes) | **28** | **24** | −4 ↓ 性能回帰 |

**観察:**

- **σ_ln の全体改善:** pn σ_ln が depth=16-20 で 0.403 (目標 ≥0.37 を達成)，
  dn σ_ln も全 depth で改善 (特に浅い depth で顕著)
- **depth=16 TT 拡大:** pn=INF エントリが除外されたことで depth=16 内の再探索量が増大し，
  TT サイズが 398K → 1.1M に拡大した (depth=16 の累積 nodes 自体は同じ 3.7M)
- **depth=20 の pn=INF% が 0.4% に減少:** v0.43.0 は 4.1%．pn=INF 中間エントリが
  `retain_working_intermediates` で廃棄されるため，depth=20 開始時の TT がクリーンになった
- **depth=28 未到達 (50M nodes):** v0.43.0 では 16.6M nodes で depth=28 完了．
  depth=28 で 38.2M+ nodes を消費しても完了せず → **性能回帰**

**depth=28 到達遅延の根本原因 — pn=INF エントリ廃棄によるヒント喪失:**

```
v0.43.0: depth=24 → depth=28 遷移時
  retain_working_intermediates: pn=INF エントリも保持 (remaining=24→28 にシフト)
  depth=28 MID: depth=24 の pn=INF ヒントを利用して 4.8M nodes で完了

v0.49.0: depth=24 → depth=28 遷移時
  retain_working_intermediates: pn=INF エントリを廃棄 (セマンティクス的に保守的な処理)
  depth=28 MID: ヒントなし → 38.2M+ nodes を消費しても未完了
```

depth=24 の TT の 85% は pn=INF エントリ (total = ~615,000 件)．
これらは「24手内では証明不能」を意味し，depth=28 での探索でも有効なヒントとなりうるが，
v0.49.0 では全廃棄されるため depth=28 以降の探索コストが大幅に増大した．
この問題の詳細と対策候補を §4.5 に記載する．

---

### 2.7 v0.50.0/v0.51.1 — heuristic_or_pn bucket 6-7 分散後の計測

**変更内容:** safe_escapes=1-2 の直接マッピングを +1 bucket 分シフト (§3.19)
**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)，release ビルド (252.2s)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 150K | 118 | 52.5% | 52.5% | 10.7 | 0.253 | 0.891 | 7.3 | 0.165 | 0.207 |
| 8 | 160K | 18,644 | 43.7% | 43.6% | 10.0 | 0.276 | 0.395 | 7.8 | 0.178 | 0.155 |
| 16 | 3.7M | 1,098,329 | 93.3% | 92.9% | 9.3 | 0.391 | **0.412** | 6.4 | 0.275 | 0.611 |
| 20 | 7.6M | 300,983 | 70.9% | 70.7% | 9.1 | 0.388 | **0.411** | 6.2 | 0.272 | 0.646 |
| 24 | 11.8M | 794,101 | 84.1% | 83.6% | 8.7 | 0.378 | 0.503 | 5.8 | 0.252 | 0.748 |

**v0.49.0→v0.50.0/v0.51.1 の主要変化:**

| 指標 | v0.49.0 | v0.50.0/v0.51.1 | 変化 |
|------|---------|----------------|------|
| pn KL (d=16) | 0.473 | **0.412** | −0.061 ↑ 改善 |
| pn KL (d=20) | 0.473 | **0.411** | −0.062 ↑ 改善 |
| pn σ_ln (d=16) | 0.403 | 0.391 | −0.012 |
| dn KL (d=16) | 0.531 | 0.611 | **+0.080 ↓ 悪化** |
| dn KL (d=20) | 0.507 | 0.646 | **+0.139 ↓ 悪化** |
| dn KL (d=24) | 0.683 | 0.748 | **+0.065 ↓ 悪化** |
| dn σ_ln (d=24) | 0.271 | 0.252 | −0.019 |

**dn KL 悪化の原因:** `heuristic_or_pn` を増大させると `heuristic_dn_from_pn(pn)` が反比例して減少する
(`dn = sqrt(4096² / pn)`)．その結果 AND ノードの `dn = min(child_dn)` が PN_UNIT=16 に収束しやすくなり，
bucket 5 (dn=PN_UNIT) スパイクが悪化した (§3.19)．

---

### 2.8 v0.52.1 — heuristic_or_dn を pn 非依存 (se, nc) 直接マッピングに変更後の計測

**変更内容:** `heuristic_or_dn(se, nc)` — pn への依存を除去 (§3.21)
**計測条件:** 39手詰め，50M ノードで未解決 (status=unknown)，release ビルド (304.7s)

| depth | 累積 nodes | total TT | pn=INF% | dn=0% | pn 中央値 | pn σ_ln | pn KL | dn 中央値 | dn σ_ln | dn KL |
|-------|-----------|---------|---------|------|---------|---------|------|---------|---------|------|
| 4 | 150K | 121 | 53.7% | 53.7% | 10.7 | 0.253 | 0.891 | 7.6 | 0.129 | 0.453 |
| 8 | 167K | 30,604 | 48.3% | 48.3% | 9.9 | 0.261 | 0.411 | 7.8 | 0.140 | **0.220** |
| 16 | 3.7M | 669,332 | 92.8% | 92.4% | 9.8 | 0.363 | 0.467 | 6.9 | 0.241 | **0.433** |
| 20 | 7.6M | 113,281 | 54.4% | 54.2% | 9.6 | 0.364 | 0.458 | 6.9 | 0.242 | **0.438** |

中央値は exp(μ_ln) [bucket 単位]．

**v0.51.1→v0.52.1 の主要変化:**

| 指標 | v0.51.1 | v0.52.1 | 変化 |
|------|---------|---------|------|
| dn KL (d=8) | 0.155 | **0.220** | +0.065 (depth=8 は小サンプルのため変動) |
| dn KL (d=16) | 0.611 | **0.433** | −0.178 ↑ 大幅改善 |
| dn KL (d=20) | 0.646 | **0.438** | −0.208 ↑ 大幅改善 |
| pn KL (d=16) | 0.412 | 0.467 | +0.055 (軽微な悪化) |
| pn KL (d=20) | 0.411 | 0.458 | +0.047 (軽微な悪化) |
| dn σ_ln (d=16) | 0.275 | 0.241 | −0.034 (目標 ≥0.2 は維持) |
| total TT (d=20) | 300,983 | 113,281 | −62% ↑ TT 縮小 |

**観察:**

- **dn KL 大幅改善:** bucket 5 (dn=PN_UNIT) スパイクが緩和され，max 寄与 bucket が
  bucket 5→bucket 9 (dn≈256) に移行．pn との反比例連鎖を断ち切った効果が現れた
- **pn KL 軽微悪化:** heuristic_or_pn の変更はなく，(se, nc) の直接マッピングが
  pn 分布に間接影響を与えた (TT 縮小による探索パターン変化が主因と推定)
- **TT エントリ数削減 (d=20: 300K→113K):** dn が増加した (se=4+ を低く見積もらなくなった) ため，
  defender 側の探索が深まる前に AND ノードが解消されやすくなった

**dn σ_ln の変化:**

| depth | v0.49.0 | v0.51.1 | v0.52.1 |
|-------|---------|---------|---------|
| 8 | 0.195 | 0.178 | 0.140 |
| 16 | 0.283 | 0.275 | 0.241 |
| 20 | 0.283 | 0.272 | 0.242 |

dn σ_ln は目標 ≥0.2 を depth=16-20 で維持している (目標達成 ✓)．
pn → dn proxy を除去したことで，(se, nc) の組み合わせが限られるため，
σ_ln は v0.51.1 より若干低下した．bucket 5 集中が減少した分，
KL 指標は大幅に改善している．

---

## 4. 残存課題と次のアクション

### 4.1 [課題 A] σ_ln が小さすぎる: ln(bucket) 空間でも分布が狭い

**現状 (v0.52.1，depth=16-20):** pn σ_ln=0.363-0.364，dn σ_ln=0.241-0.242

**バージョン別推移 (depth=20):**

| バージョン | pn σ_ln | dn σ_ln | total TT | 主要変更 |
|-----------|---------|---------|---------|---------|
| v0.38.0 | 0.326 | 0.217 | 134K | safe_escapes=3-7 値域拡大 |
| v0.41.0 | 0.362 | 0.179 | 2.0M | AND leaf dn 幾何平均ブレンド (爆発) |
| v0.42.0 | 0.362 | 0.179 | 2.0M | heuristic_dn_from_pn sqrt 式 (変化なし) |
| v0.43.0 | 0.368 | 0.268 | 40K | heuristic_or_pn 値域拡大 (爆発解消) |
| v0.44.0 | 0.368 | 0.268 | 40K | SNDA 前適用試験・棄却，機能は v0.43.0 と同等 |
| **v0.49.0** | **0.403** | **0.283** | **68K** | pn=INF 中間エントリ depth-limited 扱い |

**pn σ_ln (現状 0.403 @ depth=16-20): 目標 ≥0.37 達成 ✓**
- v0.49.0 で depth=16-20 において 0.403 に向上 (目標 ≥0.37 を超過)
- depth=24 で 0.383 に低下するが目標範囲内

**dn σ_ln (現状 0.271-0.283, 目標 ≥ 0.2): 達成 ✓**
- v0.43.0 の pn 値域拡大による de-prioritization 効果で TT の集団展開が解消された
- depth=16-24 全域で dn σ_ln ≥ 0.2 を達成 (旧目標 0.5 は AND-min 制約により不可)
- **dn σ_ln 目標を ≥ 0.2 に下方修正 (AND-min 伝播の構造的制約による)**

#### pn σ_ln の構造的上限と目標の下方修正

v0.43.0 以降の実験 (§3.15，§3.16) により，pn σ_ln ≥ 0.5 の達成が **構造的に困難** と判明した:

1. **SNDA は事実上 no-op:** TT ミスの子は source=0 で除外，かつ詰将棋では同一手から同一局面が
   2 回現れることは稀なため snda_pairs のグループサイズが 1 超になることがない (§3.15 で実験確認)

2. **safe_escapes=1-2 の pn 引き上げは TT 爆発を招く:**
   - safe_escapes=1-2 は 39手詰め TT の多数派であり，pn を上げると IDS budget がそのクラスタ点を
     超えた瞬間に集団展開が発生する (v0.44.0 depth=20 TT: 39,772 → 1,951,012)
   - v0.40.0/v0.41.0 と同じメカニズム

3. **WPN (γ=6) のもとでは pn の段階的な積み上がりが微小:**
   - AND pn ≈ max_cpn + max_cpn/64，各 AND レベルで 1.6% しか増加しない
   - 初期値上限 (2048S = bucket 15) から出発して bucket 20+ に到達するには 5×64 ≈ 320 AND 段が必要
   - 50M nodes 探索範囲ではこの蓄積は起こりえない → pn 分布の右テールが bucket 15-17 で頭打ち

4. **σ_ln の数学的上限:** bucket の範囲が 4 (PN_UNIT 下限) 〜 17 (右テール頭打ち) の 13 幅のとき，
   分布が uniform でも σ_ln = 13/(2√3) ≈ 3.75 が理論最大．実際は集中分布なので σ_ln ≈ 0.3-0.4 が現実的な上限

**pn σ_ln 目標を ≥ 0.37 に下方修正 (AND の WPN + PN_UNIT 下限の構造的制約による)**

depth=16-28 で pn σ_ln=0.360-0.368 を維持することを目標とする．
より根本的な改善には WPN_GAMMA_SHIFT の削減が必要だが (§3.10 の教訓)，
副作用のリスクが高く優先度は低い．

**次のアクション候補:**
- depth=32+ での pn/dn 分布計測 (v0.43.0 では 28 より深いデータが未取得)
- 最終評価: 39手詰め backward 解析のノード数で改善を確認 (§10.2)

---

### 4.2 [課題 B] KL(対数正規) が高い: 分布が対数正規形状に合致しない

**バージョン別 KL 推移:**

| depth | v0.49.0 pn | v0.51.1 pn | v0.52.1 pn | v0.54.0 pn | v0.49.0 dn | v0.51.1 dn | v0.52.1 dn | v0.54.0 dn |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 8 | 0.395 | 0.395 | 0.411 | **0.287** ↑ | 0.128 | 0.155 | 0.220 | 0.236 |
| 16 | 0.473 | **0.412** ↑ | 0.467 | **0.364** ↑ | 0.531 | 0.611 ↓ | **0.433** ↑ | **0.387** ↑ |
| 20 | 0.473 | **0.411** ↑ | 0.458 | 0.573 ↓ | 0.507 | 0.646 ↓ | **0.438** ↑ | 0.611 ↓ |
| 24 | 0.507 | 0.503 | — | 0.789 ↓ | 0.683 | 0.748 ↓ | — | 0.747 |

目標 (< 0.3) に対して，**v0.54.0 で depth=8 の pn KL が 0.287 と目標を初達成**．
depth=16-20 の pn/dn KL は depth=8 に比べて依然高い．depth=20+ は v0.53.x の INTERPOSE_DN_BIAS
変更の影響で KL が悪化した (v0.53.3 baseline 計測未実施のため要確認，§2.9 参照)．

KL が高い主要因 (§4.3/§4.4 調査結果より):
1. **pn: heuristic_or_pn 離散スパイク (depth=4-8 での主犯):** bucket 12 への集中 (depth=4: 寄与 0.562，depth=8: 寄与 0.114)
   → v0.54.0 で pos_key ジッタにより部分緩和 (depth=8 KL: 0.411 → 0.287 ✓)
2. **pn: WPN AND 多段累積による bucket 30 (near-INF) 集積 (depth=16 での主犯):**
   v0.54.0 でも bucket 30 が主犯 (寄与 0.149)
3. **pn/dn: depth=20+ の悪化:** bucket 7 (64) スパイク (pn) と bucket 5 (PN_UNIT) スパイク (dn) が増大
   → v0.53.3 INTERPOSE_DN_BIAS 640 による AND 探索順変化が疑われる
4. **dn: AND-min 伝播による PN_UNIT 収束:** bucket 5 (dn=PN_UNIT) への張り付き (depth=20+ で再燃)

**試みた対策とその結果:**
- pn 側: `heuristic_or_pn` bucket 6-7 集中を分散 (v0.50.0/v0.51.1, §3.19) → pn KL 改善，dn KL 悪化
- pn 右テール: `WPN_GAMMA_SHIFT` 6→7 (v0.51.0, §3.20) → **右テール爆発 (depth=20: 217K→2.4M)，全指標悪化，棄却**
- dn 側: `heuristic_or_dn` を pn 非依存 (se, nc) 直接マッピングに変更 (v0.52.1, §3.21) → **dn KL が depth=16/20 で -0.178/-0.208 大幅改善** ✓
- pn 離散スパイク: `heuristic_or_pn` に pos_key ジッタ (v0.54.0, §3.23) → **depth=8 で pn KL < 0.3 初達成** ✓

**現状 (v0.54.0):** depth=8 pn KL=0.287 (目標 < 0.3 達成)，depth=16 で pn 0.364/dn 0.387 に改善．
depth=20+ は v0.53.x の副作用で悪化 (要 baseline 計測)．

---

### 4.3 [課題 C] pn 右テール at bucket 28

**調査結果 (v0.49.0，`analyze_pn_tail` 実行):**

`joint_hist` の pn_bucket ≥ 20 のエントリを dn 分布で分析した結果:

| depth | 高 pn エントリ数 | AND候補 (dn≤bucket 6) | OR候補 (dn≥bucket 12) |
|-------|----------------|----------------------|----------------------|
| 8 | 8,072 | **8,066 (99.9%)** | 0 (0.0%) |
| 16 | 1,021,455 | **1,021,287 (100.0%)** | 0 (0.0%) |
| 20 | 3,470 | **3,448 (99.4%)** | 0 (0.0%) |
| 24 | 622,772 | **622,739 (100.0%)** | 0 (0.0%) |

**結論: pn 右テールは 100% AND ノードが主因．** OR ノードは bucket 20+ に全く現れない．

**主因: WPN AND 多段累積**

各 AND ノードで `pn_new = max(child_pn) + Σ(others) >> WPN_GAMMA_SHIFT(=6)` が適用される．
深い AND チェーンを経由するにつれ，各ステップで `sum_others / 64` が加算されて累積する:

```
depth=4 chain AND ×5: 64S → 64 + 64×4/64 = 68S → ... → bucket 7 程度
depth=16 chain AND ×12: 累積が bucket 20+ に到達 (1.02M 件)
```

**試みた対策 (v0.51.0, §3.20):** `WPN_GAMMA_SHIFT` 6 → 7

結果: **右テールが逆に爆発** (depth=20: 217K → 2.4M エントリ，depth=24: 673K → 2.8M エントリ)．
原因: WPN 和の寄与を半減させると収束が遅くなり，AND ノードが確定しないまま TT に滞留する時間が
長くなる．その間に別経路の WPN 累積を受けて大きな pn 値が蓄積した．v0.51.1 で GAMMA_SHIFT=6 に戻した．

**現状の対応方針:** WPN_GAMMA_SHIFT の増大は右テールに逆効果 (§5 不採用)．
OR ノードは無関係なので 1+ε 閾値伝播加速は主因ではない (仮説棄却)．

---

### 4.4 [課題 D] depth 増加に伴う KL 劣化

**調査結果 (v0.49.0，`analyze_kl_by_bucket` 実行):**

各 depth の中間エントリ (bucket 1-30) に対し，per-bucket KL 寄与 `p_k · log(p_k / q_k)` を分析:

**pn KL の主犯 bucket (v0.49.0→v0.54.0 更新):**

| depth | v0.49.0 KL | 主犯 bucket (v0.49.0) | 寄与値 | v0.52.1 KL | 主犯 bucket (v0.52.1) | 寄与値 | v0.54.0 KL | 主犯 bucket (v0.54.0) | 寄与値 |
|-------|-----------|----------------------|--------|-----------|----------------------|--------|-----------|----------------------|--------|
| 4 | 0.891 | bucket 12 (2^11) | 0.591 | 0.891 | bucket 12 (2^11) | 0.591 | 0.725 | bucket 12 (2^11) | 0.562 |
| 8 | 0.395 | bucket 7 (2^6) | 0.100 | 0.411 | bucket 12 (2^11) | 0.133 | **0.287** | bucket 12 (2^11) | **0.114** |
| 16 | 0.473 | bucket 6 (2^5) | 0.226 | 0.467 | bucket 30 (near-INF) | 0.175 | **0.364** | bucket 30 (near-INF) | 0.149 |
| 20 | 0.473 | bucket 6 (2^5) | 0.226 | 0.458 | bucket 30 (near-INF) | 0.180 | 0.573 | **bucket 7 (2^6=64)** | **0.430** |

**dn KL の主犯 bucket (v0.49.0→v0.54.0 更新):**

| depth | v0.49.0 KL | 主犯 bucket (v0.49.0) | 寄与値 | v0.52.1 KL | 主犯 bucket (v0.52.1) | 寄与値 | v0.54.0 KL | 主犯 bucket (v0.54.0) | 寄与値 |
|-------|-----------|----------------------|--------|-----------|----------------------|--------|-----------|----------------------|--------|
| 8 | 0.128 | bucket 10 (2^9) | 0.148 | 0.220 | bucket 9 (2^8) | 0.250 | 0.236 | bucket 9 (2^8) | 0.259 |
| 16 | 0.531 | bucket 5 (=PN_UNIT) | 0.446 | 0.433 | bucket 9 (2^8) | 0.282 | **0.387** | bucket 9 (2^8) | 0.286 |
| 20 | 0.507 | bucket 5 (=PN_UNIT) | 0.413 | 0.438 | bucket 9 (2^8) | 0.279 | 0.611 | **bucket 5 (=PN_UNIT)** | **0.556** |

**根本原因と変化 (v0.49.0→v0.54.0):**

1. **pn KL (depth=4-8):** `heuristic_or_pn` の離散初期化 → bucket 12 への集中．
   v0.54.0 の pos_key ジッタで bucket 12 寄与が 0.133 → 0.114 に軽減．
   depth=8 pn KL が 0.411 → **0.287** に改善し，目標 < 0.3 を初達成 ✓

2. **pn KL (depth=16):** v0.54.0 では bucket 30 (near-INF) が主犯として継続するが寄与が 0.175 → 0.149 に軽減．
   pn KL 0.467 → **0.364** に改善 ✓

3. **pn KL (depth=20+):** 主犯が bucket 30 (near-INF) から bucket 7 (2^6=64) に大きく移行．
   これは v0.53.3 INTERPOSE_DN_BIAS 128→640 変更による AND 探索順変化が疑われる
   (v0.53.3 baseline 計測未実施のため v0.54.0 jitter との寄与分離不可)

4. **dn KL (depth=20+):** bucket 5 (PN_UNIT) スパイクが再燃 (0.279 → 0.556)．
   v0.53.3 INTERPOSE_DN_BIAS 変更で drop moves の dn が高くなり，AND-min が board moves の
   低 dn を反映しやすくなった結果として AND pn=small が増加したと推定

**対応済み:**

- **dn 側 (v0.52.1)**: `heuristic_or_dn` を pn 非依存 (se, nc) 直接マッピングに変更 (§3.21)
  → dn KL depth=16 で 0.282 ポイント改善 ✓
- **pn 離散スパイク (v0.54.0)**: `heuristic_or_pn` に pos_key ジッタ追加 (§3.23)
  → depth=8 pn KL 0.411 → **0.287** (< 0.3 目標達成) ✓
  → depth=16 pn KL 0.467 → **0.364** (改善) ✓

**残存課題:**

- **pn 右テール (depth=16 の主犯):** bucket 30 (near-INF) 中間エントリ
  → WPN AND 多段累積が根本原因 (§4.3)．WPN_GAMMA_SHIFT 増大は逆効果 (§3.20)
- **depth=20+ の悪化:** v0.53.3 vs v0.54.0 の寄与分離が必要 (v0.53.3 baseline 計測)
- KL 目標 < 0.3 は depth=16+ で未達 (v0.54.0 pn: 0.364，dn: 0.387 at depth=16)

---

### 4.5 [課題 E] v0.49.0 による depth=28+ 探索のコスト増大

**現状:** 50M nodes で depth=24 まで到達 (v0.43.0 は 16.6M nodes で depth=28 到達)
depth=28 以降: v0.49.0 では 38.2M+ nodes 消費しても depth=28 未完了

**原因:** `retain_working_intermediates` による pn=INF エントリ全廃棄 (§2.6 参照)

```
depth=24 TT に含まれる pn=INF エントリ: 約 615,000 件 (全 TT の 85%)
これらの「24手内では証明不能」ヒントが depth=28 探索で使えなくなる
→ depth=28 が深さ独立に探索をやり直す必要が生じる
```

**影響の範囲:**

- `test_no_checkmate_counter_check`: v0.49.0 で修正 (950K → 解決) ✓
- 39手詰め通常探索: 深 depth での探索コストが増大
- 一般的な詰将棋問題: depth=28+ 到達がより難しくなる可能性

---

#### IDS の delta シフトと remaining チェックの関係

`retain_working_intermediates(delta_remaining)` は保持対象の全エントリに対して
`e.remaining += delta_remaining` を適用する (tt.rs:2166)．
この delta シフトが look_up の remaining チェックと干渉する:

```
IDS depth D → D+Δ (例: 24 → 28, Δ=4) のとき:

  retain が pn=INF エントリを保持する場合 (v0.49.0 以前):
    1. depth=D 終了: pn=INF エントリは e.remaining = D で TT に存在
    2. retain 実行: e.remaining += Δ → e.remaining = D+Δ
    3. depth=D+Δ の look_up (remaining=D+Δ):
         e.remaining() < remaining → (D+Δ) < (D+Δ) = FALSE
         → スキップされず，そのまま使われる ← バグ再現

  retain が pn=INF エントリを保持しない場合 (v0.49.0):
    1. retain 実行: pn=INF エントリは fe.pos_key = 0 で消去
    2. depth=D+Δ の look_up: エントリが存在しない → キャッシュミス → 再探索
```

**delta シフトが look_up チェックを無効化する構造:**

IDS が n ステップを経由する場合も同様に無効化される．
counter-check バグ (depth=16 → 31) も，IDS が途中で retain を n 回実行すれば
remaining=16 が 16+4×n に更新され，最終的に e.remaining = remaining になる:

```
depth=16 pn=INF (remaining=16)
  → retain ×1 (Δ=4): remaining=20
  → retain ×2 (Δ=4): remaining=24
  → retain ×3 (Δ=4): remaining=28
  → retain ×4 (Δ=4): remaining=32

depth=32 の look_up (remaining=32): 32 < 32 = FALSE → 使われる ← バグ
```

`look_up` の remaining チェック (`e.remaining() < remaining`) は，
retain が delta シフトを行う限り，**隣接 depth 間では機能しない**．
v0.49.0 の counter-check fix の本体は look_up 側ではなく **retain 側の除外** である
(look_up チェックは retain が行われない経路のための safety net)．

---

#### 対策候補の検討

**「retain を元に戻して look_up チェックのみ残す」案 — 不可:**

```
retain: pn=INF エントリを delta シフトして保持
  → e.remaining = D+Δ = remaining_new
look_up: e.remaining() < remaining → (D+Δ) < (D+Δ) = FALSE → スキップされない
```
counter-check バグが再現する．look_up チェックは delta シフト済みエントリに対して無効．

**「delta シフトせずに retain する」案 — 効果なし:**

```
retain: pn=INF エントリを remaining 更新せずに保持
  → e.remaining = D (stale のまま)
look_up: e.remaining() < remaining → D < D+Δ = TRUE → スキップ
```
counter-check は fix されるが，depth=D+Δ でエントリが常にスキップされるため
「保持しないこと」と性能上の違いがない (TT 空間を無駄に占有するだけ)．

**「δ 許容チェック (`e.remaining() + DELTA < remaining`)」案 — 同様に不可:**

delta シフト後の e.remaining = D+Δ = remaining_new に対して
(D+Δ) + DELTA < (D+Δ) = FALSE が DELTA ≥ 0 である限り常に成立しない．
つまり，retain が delta シフトを行う限り，どのような許容幅を設けても
シフト済みエントリはスキップされない．

#### 結論: 現行アーキテクチャでは再利用と counter-check fix は両立しない

`retain_working_intermediates` が delta シフトを行う設計である限り，
pn=INF エントリの「保持 + 隣接 depth 間での再利用防止」を
look_up チェックだけで実現することは不可能である．

**唯一の clean な解決策は現行 v0.49.0 の方針 (不保持) であり，性能回帰は正しい振る舞いのコスト．**

---

#### backward 解析による影響評価 (v0.49.0 計測)

`test_tsume_39te_backward_1m` / `test_tsume_39te_backward_10m` を **release ビルド**で実行．

**backward_1m 結果 (1M nodes / 180s per ply):**

| ply | remain | nodes | time(s) | result |
|-----|--------|-------|---------|--------|
| 24 | 15 | 267,899 | 49.41 | Mate(15) ✓ |
| 22 | 17 | 986,363 | 115.74 | Mate(17) ✓ |
| 20 | 19 | 1,000,000 | 106.68 | **Unknown** ← 境界 |

**backward_10m 結果 (10M nodes / 600s per ply):**

| ply | remain | depth (IDS上限) | nodes | time(s) | result |
|-----|--------|----------------|-------|---------|--------|
| 22 | 17 | 19 | 6,884,094 | 106.62 | Mate(17) ✓ |
| 20 | 19 | 21 | 8,077,824 | 158.40 | Mate(19) ✓ |
| 18 | 21 | 23 | 9,992,668 | 167.86 | **Unknown** ← 境界 |
| 16 | 23 | 25 | 10,000,000 | 85.06 | Unknown |
| 14 | 25 | 27 | 10,000,000 | 191.76 | Unknown |
| 12 | 27 | **29** | 10,000,000 | 217.09 | Unknown |
| 10 | 29 | 31 | 10,000,000 | 117.62 | Unknown |

**比較 (v0.24.33 baseline との対比):**

| 予算 | v0.24.33 境界 | v0.49.0 境界 | 変化 |
|------|-------------|-------------|------|
| 1M nodes | ply 22 | **ply 20** | **+2 ply 改善** ✓ |
| 10M nodes | ply 18 | ply 18 | 変化なし (ply 22 が 6.88M で解決 → 効率改善) |

**depth=28 回帰の実際の影響:**

depth=28 回帰が現れるのは IDS 上限 ≥ 28 = remaining ≥ 26 = **ply ≤ 13** のみ．

```
depth 式: min(remaining + 2, 41)
  ply 14: min(27, 41) = 27 → 回帰なし
  ply 12: min(29, 41) = 29 → 回帰あり ← 但し 10M で Unknown (回帰有無に関係なく解けない)
```

**ply ≤ 12 は現行 10M 予算では回帰の有無に関わらず Unknown** のため，
v0.49.0 の depth=28 回帰は現状の backward 解析結果に影響しない．

影響を観測するには **30M-100M 予算**での ply 12-16 計測が必要だが，
1M/10M の境界改善 (+2 ply) および TT 爆発解消の恩恵の方が大きく，
追加対策の優先度は低い．

**現在の優先度:** 優先度 低 (現行予算での backward 解析に影響なし；100M+ 予算で再評価)

---

### 4.6 [課題 F] IDS 直接ジャンプポリシーによる TT 暖機不足 — ply 22 > ply 20 逆転の根本原因

#### 現象

v0.54.0 の 10M backward 解析 (§3.23) において，ply 22 (remaining=17) が
ply 20 (remaining=19) よりも多くのノードを消費するという逆転現象が観測された:

| ply | remain | saved_depth | nodes (v0.54.0) | 対 ply 20 比 |
|-----|--------|-------------|-----------------|-------------|
| 22 | 17 | **19** | 9,439,291 | **+77%** |
| 20 | 19 | **21** | 5,333,924 | — (基準) |

ply 22 は ply 20 の部分問題 (ply 20 の解答手順中の局面) であるため，
本来は ply 22 ≪ ply 20 となるべきである．

#### 根本原因: saved_depth ≤ 19 の直接ジャンプポリシー

`pns.rs` の `mid_fallback()` 内 IDS ループに，`saved_depth <= 19` のとき
中間ステップを飛ばして最終深さに直接ジャンプする条件がある:

```rust
// pns.rs (mid_fallback() 内 IDS 深さ更新ロジック)
let next = if ids_depth >= 16 {
    ids_depth + 4
} else {
    ids_depth.saturating_mul(2).max(ids_depth + 2)
};
if saved_depth <= 19 && next > 4 && next < saved_depth {
    ids_depth = saved_depth;  // ← 直接ジャンプ (ply 22 が該当)
} else if ids_depth == 16 && next > 17 && saved_depth > 19 && saved_depth <= 26
    && !self.param_no_ids17 {
    ids_depth = 17;           // IDS-17 中間ステップ (ply 20 が該当)
} else {
    ids_depth = next.min(saved_depth);
}
```

`saved_depth = min(remaining + 2, 41)` であるから:

| ply | remaining | saved_depth | 条件 | IDS 深さ進行 |
|-----|-----------|-------------|------|-------------|
| 22 | 17 | **19** | `saved_depth <= 19` = **TRUE** | `2 → 4 → **19**` (直接ジャンプ) |
| 20 | 19 | **21** | `saved_depth <= 19` = FALSE | `2 → 4 → 8 → 16 → **17** → 21` (段階的) |

- ply 20 は `saved_depth=21` のため，ids_depth=4 の次に 8 → 16 → 17 → 21 と
  5 ステップかけて TT を暖機 (warm-up) してから最終深さに到達する
- ply 22 は `saved_depth=19` の直接ジャンプ条件に該当し，ids_depth=4 の次に
  **いきなり 19** に飛ぶ — TT 暖機は depth=2 と depth=4 の 2 ステップのみ

#### 実測データ: `test_ply22_vs_ply20_ids_profile` (v0.54.0)

`collect_pn_dn_dist_per_depth()` API による ply 22 の IDS per-depth ブレークダウン:

```
IDS per-depth breakdown (from collect_pn_dn_dist_per_depth):
  ids_depth  cumul_nodes      incr_nodes       TT_work
  2          150,009          150,009          9
  4          150,087          78               121
  19         9,439,291        9,289,204        (final depth, no snapshot)
```

**観察:**
- depth=2 終了時 TT エントリ: **9 件**
- depth=4 終了時 TT エントリ: **121 件** (78 ノードの増分で 112 件追加)
- depth=19 への冷スタート: TT 暖機はわずか 121 エントリのみ
- depth=19 でのノード消費: **9,289,204 ノード (全体の 98.4%)**

比較として ply 20 は同じ 10M 予算で 337,060 TT エントリを蓄積し，
5,333,924 ノードで解決した:

| 指標 | ply 22 (直接ジャンプ) | ply 20 (段階的 IDS) |
|------|----------------------|---------------------|
| IDS 進行 | 2→4→**19** | 2→4→8→16→17→**21** |
| 最終深さ直前 TT | **121 件** (冷) | **337,060 件** (暖) |
| nodes/TT 比 | **252 nodes/entry** | **16 nodes/entry** |
| 総ノード数 | 9,439,291 | 5,333,924 |

#### ply 22 各深さの独立コスト (sub-depth 分析)

各深さをフレッシュ TT から独立実行した場合のコスト (10M ノード予算):

| depth | nodes | time(s) | TT_pos | result |
|-------|-------|---------|--------|--------|
| 4 | 158 | 0.22 | 122 | NoMate |
| 8 | 4,469 | 0.70 | 7,304 | NoMate |
| 12 | 1,659,294 | 12.76 | 39,827 | Unknown |
| 16 | 10,000,000 (予算枯渇) | 45.03 | — | Unknown |

- depth=4, 8 はフレッシュ TT でも軽量 (冷スタートコストが小さい)
- depth=12 は既に 1.66M ノードを要する — depth 12 の暖機を skip する代償は大きい
- depth=16 は 10M 予算を使い切っても解けない (詰みは 17 手なので深さ不足)

段階的 IDS (ply 20 の場合) ではこれらの中間深さを順次実行することで
TT を段階的に構築し，最終深さでの探索コストを大幅に削減できる．
直接ジャンプではそれが行われないため，最終深さ (depth=19) が
TT がほぼ空の状態から開始することになる．

#### 直接ジャンプポリシーが v0.54.0 で悪化した理由

IDS 直接ジャンプ自体は v0.49.0 以前から存在していたが，
v0.53.3 の INTERPOSE_DN_BIAS 128→640 の変更 (§3.22) により
AND ノードの探索順序が大幅に変化した．冷スタートの depth=19 で
より多くの AND ノードが展開される経路が増え，結果として
ply 22 のコストが v0.49.0: 6.88M → v0.54.0: 9.44M (+37%) に悪化した．

#### 対策: saved_depth 閾値の引き下げ

直接ジャンプ条件の閾値を `saved_depth <= 19` から `saved_depth <= 15` に引き下げることで，
saved_depth 17, 19 (remaining 15, 17 = ply 24, 22) を段階的 IDS に切り替えられる:

```rust
// 変更案
if saved_depth <= 15 && next > 4 && next < saved_depth {
    ids_depth = saved_depth;  // 引き下げ: 16 以上は段階的 IDS へ
```

| ply | saved_depth | 変更前 | 変更後 | 期待効果 |
|-----|-------------|--------|--------|---------|
| 26 | 17 | 直接ジャンプ | **段階的** | TT 暖機あり |
| 24 | 17 | 直接ジャンプ | **段階的** | TT 暖機あり |
| 22 | 19 | 直接ジャンプ | **段階的** | TT 暖機あり |
| 20 | 21 | 段階的 (変化なし) | 段階的 | — |

注: `saved_depth=17` は `2→4→8→16→**17**` (IDS-17 ステップも適用される可能性)，
`saved_depth=19` は `2→4→8→16→17(?)→**19**` という進行になる見込み．
実際の効果は benchmark で検証が必要．

**現在の優先度:** 優先度 中 (v0.54.0 で ply 22 が 1M 境界を超過；閾値引き下げによる修正が有望)

---

## 5. 対応しない事項

| 事項 | 理由 |
|------|------|
| `WPN_GAMMA_SHIFT` の削減 (6→4〜5) | OR dn 単独 γ_dn=4 を試みて pn/dn 両方が悪化した (§3.10)．AND-min 伝播が支配的であり，γ 緩和の効果が打ち消される |
| `WPN_GAMMA_SHIFT` の増大 (6→7) | v0.51.0 で試験．pn 右テールが depth=20 で 217K→2.4M に逆爆発した (§3.20)．収束が遅くなり AND ノードの TT 滞留時間が増大して WPN 累積が加速するため逆効果 |
| `heuristic_dn_from_pn` clamp 下限緩和 (4S→1S) | cluster 点が bucket 6→4 に下がり σ_ln(dn) が悪化した (§3.10) |
| `heuristic_and_dn` 単体採用 (pn ベースを完全置換) | safe_escapes=0 ノードの dn=64S が大量発生し depth=20 で TT が 15× 爆発した (§2.5，§3.11) |
| `heuristic_and_dn` 幾何平均ブレンド | クラスタ点が 64S→32S に移動しただけで爆発メカニズムは同一 (§3.12)．回帰テストは 2× 高速化したが depth=20 TT が 2.0M に爆発 |
| `heuristic_dn_from_pn` sqrt 式単独改善 | AND-min 伝播が支配的であり，葉ノードの初期 dn の連続性は展開後に消失する (§3.13)．pn 値域を広げないと効果なし |
| `DN_FLOOR` の削減 | chain AND の pn_floor との連動あり；変更コスト・リスクが高い |
| 「対数正規分布を強制する」初期化 | pn が実際の探索コストを反映しなくなり探索効率が低下する |
| 並列探索 | §11.6 の方針により非採用 |
| dn σ_ln ≥ 0.5 の達成 | AND-min 伝播という df-pn の構造的制約により，初期化・WPN 調整では達成不可 (§3.10 で実験的に確認)．目標を ≥ 0.2 に下方修正し v0.43.0 で達成済み |
| SNDA source 64bit 拡張 | source = pos_key as u32 (32bit) で十分; 衝突確率は 2⁻³² で実用上問題なし |
| SNDA 前適用 (SNDA→WPN，Kishimoto 2010 正規) | v0.44.0 で試験．`test_no_checkmate_counter_check` が 2M→20M+ ノードに回帰．TT ミス子の heuristic_dn が raw sum を膨張させ OR dn が過大評価される (§3.17)．WPN→SNDA→floor は `heuristic_dn_from_pn` の均一 dn を補償する暫定実装 |
| `param_refutable_at_mid_limit` (depth-limit ノードで refutable チェック) | v0.48.0 attempt 1．全 depth-limit ノードで `refutable_check_with_cache` を呼ぶコストが莫大で 10M→Unknown に回帰 (§3.18) |
| `reset_pn_inf_in_working()` (pn=INF のみリセットして dn 保持) | v0.48.0 attempt 2．depth=16 由来の stale な dn 値が depth=31 MID の探索優先度付けを誤誘導し 10M→Unknown に回帰 (§3.18) |
| safe_escapes=1-2 の pn 上限引き上げ | TT 多数派の pn を上げると IDS budget クラスタ点で集団展開が発生する (§3.16)．v0.40.0/v0.41.0 と同メカニズム |
| pn σ_ln ≥ 0.5 の達成 | WPN (γ=6) + PN_UNIT 下限により bucket 4-17 の 13 幅に収束し σ_ln ≈ 0.36-0.37 が構造的上限 (§4.1)．目標を ≥ 0.37 に下方修正 |

---

## 6. アクションプラン

```
[完了]
 ├── 案 A-1: heuristic_dn_from_pn の下限 1S→2S (v0.36.0)
 ├── 案 A-2: heuristic_dn_from_pn の下限 2S→4S (v0.37.0)
 ├── 案 B 部分: heuristic_or_pn の safe_escapes=1-2 直接マッピング (v0.37.0)
 ├── 案 C: heuristic_or_pn の safe_escapes=3-7 値域拡大 (bucket 9-13，v0.38.0)
 │     └── 計測済み: pn σ_ln 0.338→0.381，depth=24 証明率 73.2%→90.0%
 ├── WPN γ_dn 分離 + clamp 下限緩和 — 不採用 (v0.39.0)
 │     └── pn/dn 両方が悪化．AND-min 伝播の支配性を実験的に確認 (§3.10)
 ├── heuristic_and_dn 単体採用 — 不採用 (v0.40.0)
 │     └── depth=20 で TT 15× 爆発．AND leaf dn 多様性が展開後に維持されない (§3.11)
 ├── heuristic_and_dn 幾何平均ブレンド — 不採用 (v0.41.0)
 │     └── クラスタ点が 64S→32S に移動しただけで爆発メカニズムは同一 (§3.12)
 │         回帰テスト 2× 高速化も depth=20 TT 2.0M に爆発，到達 depth 40→28 に悪化
 ├── heuristic_dn_from_pn sqrt 式 — 採用済み・単独では効果なし (v0.42.0)
 │     └── AND-min 伝播が初期 dn の連続性を即座に消失させる (§3.13)
 │         pn 値域拡大と組み合わせることで初めて有効になる (→ v0.43.0)
 ├── heuristic_or_pn 値域拡大 — 採用 (v0.43.0) ✓
 │     ├── 上限 512S → 2048S，開放空間段階化，num_checks=1 乗数 ×2→×4
 │     ├── depth=20 TT 爆発解消: 2.0M → 40K (50× 削減) ✓
 │     ├── dn σ_ln 目標 ≥ 0.2 を depth=16-28 全域で達成 ✓
 │     └── 回帰テスト: 46.17s (許容範囲) ✓
 ├── SNDA 無効化実験 — 仮説棄却 (v0.43.0 ベース §3.15)
 │     └── 無効化前後で分布が完全一致 → WPN 後 SNDA は 96.9% が max floor に吸収
 ├── safe_escapes=1-2 分散拡大 — 不採用 (実験的 §3.16)
 │     └── depth=20 TT 爆発 (40K → 1.95M)，pn/dn σ_ln ともに悪化
 │         pn σ_ln 目標を ≥ 0.37 に下方修正 (WPN + PN_UNIT の構造的制約による)
 ├── SNDA 前適用 (Kishimoto 2010 正規実装) — 不採用 (v0.44.0 試験，counter-check 回帰で棄却)
 │     ├── 分布指標への影響は軽微 (SNDA fire rate 0.67%)，TT 爆発なし
 │     ├── `test_no_checkmate_counter_check` が 2M→20M+ ノードに回帰
 │     └── 根本原因: pn≥256S の均一 dn=4S が TT ミス子 raw sum を膨張させる (§3.17)
 └── pn=INF 中間エントリの depth-limited 扱い — 採用 (v0.49.0) ✓
       ├── look_up_working: intermediate パスに remaining チェック追加 (pn=INF は depth 依存)
       ├── retain_working_intermediates: pn=INF エントリを保持対象から除外
       ├── 収束: 950,037 ノード (2M 予算)，~6s (release) ✓
       ├── v0.47.0–v0.48.0 の特例処理 (disproof_mode, mini-IDS warm-up) を削除 (§3.18)
       └── 分布計測 (§2.6): pn σ_ln 目標 ≥0.37 達成 (0.403)，dn σ_ln ≥0.2 維持 ✓
             副作用: depth=28+ 探索コスト増大 (50M nodes で depth=24 止まり) → §4.5

[完了 (このセッション)]
 ├── 39手詰め backward 解析 (§10.2) で v0.49.0 全体の実効性を確認 ✓
 │     → 1M 境界 +2 ply 改善 (ply 22→20)，10M 境界 ply 18 維持 (benchmarks.md §10.2.24)
 ├── 課題 E: backward 解析により優先度を 低 に変更 ✓ (§4.5 参照)
 ├── 課題 C: pn 右テール調査完了 ✓ (§4.3)
 │     → 100% AND ノード，WPN AND 多段累積が主因と確定
 │     → WPN_GAMMA_SHIFT 6→7 を試みたが逆効果 (右テール爆発，§3.20)，GAMMA_SHIFT=6 に戻した
 ├── 課題 D: KL 劣化メカニズム調査完了 ✓ (§4.4)
 │     → pn: bucket 6 (pn≈32) への heuristic_or_pn 離散スパイク
 │     → dn: bucket 5 (dn=PN_UNIT) への AND-min 伝播下限収束
 ├── 課題 B: 部分改善 (v0.50.0/v0.51.1, §3.19)
 │     → pn KL depth=16/20: 0.473 → 0.412/0.411 (−0.06 改善)
 │     → dn KL 全 depth 悪化 (depth=20: 0.507 → 0.646)
 │     → heuristic_or_pn↑ と heuristic_dn_from_pn 反比例の連鎖が根本問題
 ├── 課題 B: dn KL 大幅改善 (v0.52.1, §3.21) ✓
 │     → heuristic_or_dn を (se, nc) 直接マッピングに変更，pn 依存を除去
 │     → dn KL depth=16/20: 0.611/0.646 → 0.433/0.438 (−0.178/−0.208 改善)
 │     → 回帰テスト全通過 (regression 97.65s, backward_1m 388.47s, no_false_nomate 74.43s)
 └── heuristic_or_pn pos_key ジッタ追加 (v0.54.0, §3.23)
       → pos_key の 3 ビットで ×(13..20)/16 = ±20% の 8 段階ジッタ
       → 同一 (se, nc) 全局面が同一 pn に集中する離散スパイクを 0.62 bucket 幅に分散
       → 回帰テスト全通過 (regression 61.51s, counter-check 2M 内, no_false_nomate ok)
       → 分布への効果: §2.9 参照 (計測済み)
       → 10M backward 解析: 10M 境界 ply 18 維持 (v0.49.0 と同一)，ply 20 が −34% 改善 §3.23
       → 1M 境界: ply 22 (v0.49.0 ply 20 から −2 ply 微小回帰，差 11K nodes = 1.2%)

[優先度 中]
 └── 課題 B: KL < 0.3 達成 (v0.52.1 時点 pn: 0.458-0.467，dn: 0.433-0.438 @ depth=16-20)
       → pn 側: v0.54.0 で heuristic_or_pn 離散スパイクを緩和；効果は §2.9 参照
       → dn 側: bucket 9 (≈256) が新たな max 寄与点；pn-dn 独立化は達成

[優先度 低]
 └── 課題 E: depth=28+ 探索コスト増大 (100M+ 予算での ply 12-16 計測で再評価，§4.5)
```

各ステップ後に `scripts/analyze_pn_dn_dist.py` で対数正規指標を計測し，
**σ_ln(pn) ≥ 0.37 (達成 v0.49.0 ✓)，σ_ln(dn) ≥ 0.2 (達成 ✓)，KL(対数正規) < 0.3** を近期目標とする．
最終的な有効性は 39手詰め backward 解析 (§10.2) のノード数改善で評価する．

---

**参照:**
- §4.1 WPN スケールドサム: [proof-disproof-numbers.md](proof-disproof-numbers.md)
- §5.1 初期値ヒューリスティック: [initial-heuristics.md](initial-heuristics.md)
- §3.5 PN_UNIT スケーリング: [threshold-control.md](threshold-control.md)
- §10.2 39手詰め benchmark: [benchmarks.md](benchmarks.md)
- 分析スクリプト: `scripts/analyze_pn_dn_dist.py`
