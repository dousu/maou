# 閾値制御

### 3.1 1+ε トリック (Pawlewicz & Lew 2007)

**出典:** Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)

標準 df-pn では子 c1 の pn 閾値を `min(parent_th, pn2 + 1)` で設定する
(`pn2` は2番目に小さい pn)．c1 の pn が pn2 を1超えた瞬間に
他の子に切り替わり，seesaw effect(スラッシング)が発生する．

```
  Standard df-pn threshold (seesaw effect):

  OR node (pn_th=100)
   |
   +-- c1: pn=10  <-- selected (min pn)
   +-- c2: pn=11  (pn2 = 11)
   |
   child_pn_th = min(100, 11+1) = 12
   --> c1 explores until pn reaches 12, then switches to c2
   --> c2 explores until pn reaches 13, then back to c1
   --> rapid switching = thrashing

  1+epsilon trick:

  OR node (pn_th=100)
   |
   +-- c1: pn=10  <-- selected
   +-- c2: pn=11  (pn2 = 11)
   |
   epsilon = 11/4 + 1 = 3
   child_pn_th = min(100, 11+3) = 14
   --> c1 gets 4 more units of exploration before switching
   --> deeper search per visit, less thrashing
```

1+ε トリックは `+1` を乗算型に変更:

```
pn_threshold(c1) = min(parent_th, ceil(pn2 * (1 + ε)))
```

pn が小さい時は小さな増分(細かい制御)，pn が大きい時は大きな増分(深い探索を許容)．

**実装:** solver.rs (child threshold computation)

```
// v0.22.0: 自然精度 epsilon (§3.5 方針A + v0.21.1 で /3 に増加)
epsilon = second_best / 3 + PN_UNIT
sibling_based = second_best + epsilon ≈ second_best * 4/3 + PN_UNIT
```

OR ノード: `child_pn_th = min(eff_pn_th, second_best + epsilon)`
AND ノード: `child_dn_th = min(eff_dn_th, second_best + epsilon)`

PN\_UNIT=16 では `second_best = 3S = 48` のとき `epsilon = 16 + 16 = 32`，
`sibling_based = 80`(5.0S)となり，PN\_UNIT=1 の 4S に対し ~25% の閾値余裕が
各 OR/AND レベルで得られる．12 レベルの累積で `1.25^12 ≈ 15 倍` の余裕．

**出典との差異:**
- 論文は `ceil(pn2 * (1 + ε))` (純粋な乗算)だが，maou\_shogi では
  `second_best + second_best / 4 + PN_UNIT` で乗算を近似
- `min` キャップは論文どおり適用し，全域で乗算型の性質を維持

### 3.2 TCA (Kishimoto & Müller 2008; Kishimoto 2010)

**出典:** Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008);
Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation" (AAAI 2010)

巡回グラフ(DCG)上での pn/dn **過小評価**を修正するアルゴリズム．
ループ検出により子ノードが `(INF, 0)` を返すと，兄弟ノードの pn/dn が過小評価される．
TCA は OR ノードでループ子が存在する場合に閾値を拡張し，兄弟の深い探索を促す．

df-pn は有限 DCG 上で不完全だが，TCA を加えると完全になる．

**実装:** mod.rs (`TCA_EXTEND_DENOM`), solver.rs (MID ループ)

- **拡張量**: `threshold / TCA_EXTEND_DENOM + 1` (`TCA_EXTEND_DENOM = 4`，約25%の加算)
- **適用条件**: OR ノードでループ子(`path` 上の子)が存在する場合
- AND ノードではループ子が即時反証を引き起こすため拡張不要

**出典との差異:**
- 論文は乗算的拡張(2×)を提案するが，再帰で指数的に増大する問題がある
- maou_shogi では加算的拡張(約25%)を採用し，各レベルで独立に適用されるため膨張を抑制

### 3.3 閾値フロア

MID ループ内で閾値が過度に縮小するのを防ぐフロア値を設定する．

**実装:** solver.rs (child threshold computation)

- **PN フロア(通常)**: `pn_floor = (eff_pn_th as u64 * 2 / 3) as u32`
  (v0.21.1 で 1/2→2/3 に引き上げ，v0.23.0 で u64 昇格によりオーバーフロー修正)
- **PN フロア(チェーン AND)**: `pn_floor = max(DN_FLOOR, (eff_pn_th as u64 * 2 / 3) as u32)`
- **DN フロア(OR)**: `dn_floor_or = DN_FLOOR`
- **DN フロア(通常)**: `dn_floor = DN_FLOOR`

`DN_FLOOR = 100 * PN_UNIT`（§3.5 参照）．

チェーン合駒構造では閾値が深いネストで指数的に枯渇するため，
フロアにより最低限の探索予算を保証する．

チェーン AND ノードでは DN_FLOOR(=100) を PN フロアにも適用し，
OR 親の sibling_based(2〜5)に制約されず子 OR に十分な pn 予算を
伝播する(dn のチェーン用キャップ外しと同じ発想)．
backward 解析で ply 24 サブ問題が 1M→397K ノードに改善(§10.2)．

### 3.4 停滞検出

MID ループ内で pn/dn が改善しない場合に早期終了する．

**実装:** solver.rs (`ZERO_PROGRESS_LIMIT`, `STAGNATION_LIMIT`)

- `ZERO_PROGRESS_LIMIT = 16`: 子 `mid()` が消費するノード数が 0 の回数が連続16回で進展なしと判定
- `STAGNATION_LIMIT = 4`: best child の pn/dn と閾値が連続4回不変で MID ループを終了

### 3.5 PN\_UNIT 統一スケーリング

pn/dn の 1 単位を定数 `PN_UNIT`(mod.rs)で表現し，全てのスケーリング対象を
明示する仕組み．PN\_UNIT=16(v0.21.0)で閾値飢餓を緩和し，
1+ε 閾値の余裕を確保する（KomoringHeights の初期 pn=10-80 に相当）．
PN\_UNIT=1 で従来動作と等価であり，スケーリング漏れの検証に使用する．

**設計原理:**

全ての pn/dn を完全にスケーリングすればソルバーの挙動は一致する．
逆に言えば，PN\_UNIT を変更して挙動が変わるならスケーリング漏れがある．
この原理を用いて PN\_UNIT=1 と PN\_UNIT=64 の結果を比較し，
漏れを機械的に特定・修正した．

**スケーリング対象:**

| 区分 | 具体例 |
|------|--------|
| 初期値 | TT ミスの pn=1/dn=1，heuristic\_or\_pn/heuristic\_and\_pn 返り値 |
| 加算定数 | edge\_cost\_or/and，sacrifice\_check\_boost，epsilon の +1，progress\_floor の +1，TCA の +1 |
| フロア・バイアス | DN\_FLOOR，INTERPOSE\_DN\_BIAS，`.max(N)` のリテラル |
| WPN の加算分 | `(unproven_count - 1) * PN_UNIT`（盤面カウントを pn 単位に変換） |
| TT ミス判定 | `cpn == PN_UNIT && cdn == PN_UNIT`（heuristic 初期化の条件） |

**スケーリング不要:**

| 区分 | 理由 |
|------|------|
| 終端値 (INF, 0) | 証明/反証のセンチネル |
| 盤面状態の比較 (safe\_escapes >= 4 等) | 手数・マス数であり pn/dn 値ではない |
| ループカウンタ (ZERO\_PROGRESS\_LIMIT 等) | イテレーション回数 |

**除算の丸め等価性:**

除算を含む計算は「PN\_UNIT=1 相当に戻してから除算し再スケール」する
(divide-at-unit-scale パターン):

```
// 等価パターン: PN_UNIT=1 と同じ丸めを再現(スケーリング漏れ検証用)
let epsilon = second_best / PN_UNIT / 4 * PN_UNIT + PN_UNIT;
```

例: `second_best = 3 * PN_UNIT` のとき
- PN\_UNIT=1: `3 / 4 + 1 = 0 + 1 = 1`
- PN\_UNIT=64 (等価): `192 / 64 / 4 * 64 + 64 = 0 + 64 = 64` (= 1 × 64) ✓

適用箇所: epsilon (`/4`)，pn\_floor (`/2`)，TCA (`/TCA_EXTEND_DENOM`)，
Deep df-pn (`/DEEP_DFPN_R`)．

**自然精度パターン:**

divide-at-unit-scale はスケーリング漏れの検証には不可欠だが，
PN\_UNIT > 1 の本来の利点である**除算の解像度向上**を殺してしまう．
閾値飢餓の改善には，除算の自然精度をそのまま活かすパターンが有効:

```
// 自然精度パターン (v0.22.0): PN_UNIT > 1 で epsilon が増大する
// v0.21.1 で除数を /4 → /3 に変更し閾値余裕をさらに拡大
let epsilon = second_best / 3 + PN_UNIT;
```

例: `second_best = 3 * PN_UNIT` のとき

| | PN\_UNIT=1 | PN\_UNIT=16 (等価) | PN\_UNIT=16 (自然精度 /3) |
|--|----------|------------------|---------------------|
| second\_best | 3 | 48 | 48 |
| epsilon | 1 | 16 | 16 + 16 = 32 |
| sibling\_based | 4 | 64 | 80 |
| PN\_UNIT 単位 | 4.0 | 4.0 | **5.0** |

自然精度では PN\_UNIT=1 の整数切り捨て `3/4 = 0` が
`48/4 = 12` として正確に計算され，epsilon が 75% 増加する．
これにより子 AND に渡る pn 閾値が増大し，閾値飢餓が緩和される．

heuristic\_or\_pn が S〜3S の範囲で中間値(例: 1.5S)を返す場合，
second\_best の分布がさらに広がり，自然精度の恩恵が増す．

**使い分け:**

- **等価パターン**: スケーリング漏れの検証，回帰テスト
- **自然精度パターン**: 閾値飢餓の改善（本番運用）

**pn/dn 値の全体マップ (PN\_UNIT = S):**

全ての pn/dn 初期値・バイアス・フロアの相対関係を示す．
S = PN\_UNIT（v0.21.0: 16）．

*初期値(子ノード展開時に TT に格納される値):*

| 値 | S 倍率 | 適用対象 | 条件 | 節 |
|---|-------|---------|------|---|
| S | 1 | OR 子 pn | 標準局面，逃げ場 0〜1 | §5.1 |
| S + S/4 | 1.25 | OR 子 pn | 逃げ場=2 | §5.1 |
| S + S/4 〜 S + S/2 | 1.25〜1.5 | OR 子 pn | 逃げ場 4〜5 | §5.1 |
| 2S + S/4 〜 3S | 2.25〜3 | OR 子 pn | 王手少＋逃げ場多，開放空間 | §5.1 |
| n×S + S/4 〜 n×S + e×S/2 | 〜n+e/2 | AND 子 pn | 応手数 n，逃げ場 e(0.67n〜n+e/2 に調整) | §5.1 |
| S | 1 | AND/OR 子 dn | 全子共通 | — |

*加算コスト(初期値に上乗せ):*

| 値 | S 倍率 | 適用対象 | 条件 | 節 |
|---|-------|---------|------|---|
| 0 | 0 | pn 加算 | 成・取王手(edge\_cost\_or) / 合駒(edge\_cost\_and) | §5.2 |
| S | 1 | pn 加算 | 近い静か王手(距離≤2) / 玉逃げ | §5.2 |
| 2S | 2 | pn 加算 | 遠い静か王手(距離≥3) / 駒取り応手 / 全捨て駒 | §5.2, §9.3 |

*閾値制御パラメータ:*

| 値 | S 倍率 | 用途 | 節 |
|---|-------|------|---|
| ε ≈ second\_best/3 + S | ~1.33倍 | OR/AND の 1+ε 手切替 | §3.1 |
| pn\_floor = eff\_pn\_th\*2/3 | 親の67% | AND 子 pn 閾値の最低保証 | §3.3 |
| pn\_floor(チェーン AND) = 100S | 100 | チェーン AND 子 pn 閾値の最低保証 | §3.3 |
| DN\_FLOOR = 100S | 100 | AND 子 dn / OR 子 dn の最低保証 | §3.3 |
| progress\_floor = best\_pn + S | +1 | 子 pn 閾値のゼロ進捗防止 | §3.3 |
| TCA 拡張 ≈ threshold/4 + S | +25% | ループ検出時の閾値拡張 | §3.2 |

*dn バイアス(AND ノードの応手選択順序):*

| 値 | S 倍率 | 適用対象 | 節 |
|---|-------|---------|---|
| 0 | 0 | チェーン AND: 内側ドロップ(距離1) | §8.7 |
| (d−1)×S | 1〜5+ | チェーン AND: 外側ドロップ(距離 d) | §8.8 |
| 8S | 8 | 非チェーン AND: ドロップ(合駒後回し) | §8.6 |
| 8S | 8 | チェーン AND: 非ドロップ(玉逃げ後回し) | §8.6 |

*WPN/CD-WPN の加算分(AND の current\_pn 計算):*

| 値 | S 倍率 | 用途 | 節 |
|---|-------|------|---|
| (n−1)×S | n−1 | WPN: 未証明子 n 個の加算分 | §4.1 |
| (g−1)×S | g−1 | CD-WPN: グループ数 g の加算分 | §4.2 |

*Deep df-pn バイアス(TT ミス時の深い ply):*

| 値 | S 倍率 | 条件 | 節 |
|---|-------|------|---|
| S | 1 | ply ≤ depth/2 | §5.3 |
| S + ⌊(ply − depth/2)/4⌋×S | 1〜数倍 | ply > depth/2 | §5.3 |

**相対関係の読み方:**

OR 子 pn(1S〜3S)と AND 子 dn(1S)の比が探索の OR/AND バランスを決める．
DN\_FLOOR(100S)は OR 子 pn の 33〜100 倍であり，
dn 閾値が枯渇しにくいことを保証する一方，pn 側にはこの水準のフロアがない
（チェーン AND を除く）．これが閾値飢餓の構造的要因である(§10.2)．

INTERPOSE\_DN\_BIAS(8S)は OR 子 pn(1〜3S)の 3〜8 倍であり，
合駒を玉逃げ・駒取りの後に探索させる効果が十分に働いている．

PN\_UNIT を拡大すると，上記の全ての値が比例してスケールされる．
改善の余地は「S 倍率が整数に丸められている箇所」にあり，
PN\_UNIT > 1 で中間値(1.5S 等)を設定することで
heuristic の解像度を上げられる(§10.2 方針 A)．

**検証結果:**

PN\_UNIT=1 と PN\_UNIT=64 で 126 テスト全通過（pass/fail 完全一致）．
backward 解析では ply 24 まで完全一致(396,636 ノード，343,999 TT エントリ)．
ply 22 以降の微差(TT 7 エントリ = 0.005%)は予算上限到達後の
TT クラスタ衝突パターンのみ．

**スケーリング漏れの特定に至った経緯:**

| 発見した漏れ | 症状 | 特定方法 |
|------------|------|---------|
| WPN `(unproven_count - 1)` に PN\_UNIT 未適用 | PN\_UNIT=64 で 2 テスト FAIL | テスト結果の比較 |
| TT ミス判定 `cpn == 1` | 同上 | 同上 |
| depth 制限超過時の初期 pn=1u32 | ply 24 以降で探索パターン乖離 | backward 解析の diff |
| 除算の丸め精度差 | P\*4g で 1.5%のノード数差 | 4M クラスタ TT での比較 |

---

