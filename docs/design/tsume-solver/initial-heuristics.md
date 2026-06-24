# 初期値ヒューリスティック

標準 df-pn は全リーフを `(pn=1, dn=1)` で初期化するが，局面の特徴に基づく初期値を与えると
最有望子の選択精度が上がり探索量が減る (df-pn+)．本節は子の seed pn/dn の計算と，展開前に
1 手詰を即決するインライン検出を扱う．

### 5.1 df-pn+ 風の per-move 初期値 (GPW 2004; KomoringHeights)

**出典:** Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004)．
df-pn+ の局面特徴に基づく初期化は KomoringHeights v0.4.0 でも実装されている．

**実装:** `init_pn_dn_or` / `init_pn_dn_and` (`heuristics.rs`)．加算は全て `PN_UNIT` (= U) 単位
([threshold §3.3](threshold-control.md))．「支援」は対象マスへの攻め方/守備方の利き数で測る．

#### OR 子 (攻め方の王手 → 守備方局面): `init_pn_dn_or`

base `(pn, dn) = (U, U)` から手の性質で調整する:

| 条件 | 調整 | 意図 |
|------|------|------|
| 受け駒の支援 ≥ 2 (王手マスが厚く受けられる) | `pn += U` | 詰みにくい → 後回し |
| 攻め支援 + drop_bonus > 受け支援 | `dn += U` | 攻めが利く有望王手 → 優先 |
| 金/銀を取る王手 | `dn += U` | 守備の主力を除去 → 有望 |
| その他の駒を取る王手 | `pn += U` | — |
| 静かな (取らない) 王手 | `pn += U` | — |

#### AND 子 (守備方の応手 → 攻め方局面): `init_pn_dn_and`

| 応手の種類 | `(pn, dn)` | 意図 |
|-----------|-----------|------|
| 駒を取る応手 | `(2U, U)` | 攻め駒除去 → 攻め方不利寄り |
| 玉移動 | `(U, U)` | — |
| 良い逃げ (攻め支援 < 受け支援 + drop_bonus) | `(2U, U)` | 受けが成立しやすい |
| 悪い逃げ | `(U, 2U)` | 受けが崩れやすい → 反証しやすい |

これらの per-move 値が **エッジコスト** (親→子遷移の手にコストを付与する DFPN-E 的アイデア,
NeurIPS 2019) として機能し，最有望子の初期順序を決める．

### 5.2 エッジコストと difficulty / ordering の分離

既定では per-move 初期値 (§5.1) が seed pn/dn にそのまま折り込まれ，pn が「難易度推定」と
「手の良し悪し (ordering)」の二役を担う．`Params::decouple_edge_cost` (既定 off) を有効にすると:

- seed pn/dn は `init_pn_dn_*` の **純粋な難易度推定**のみとする (エッジコストを pn から外す)．
- 手の良し悪しは `move_brief_eval` の **tie-break** に分離する
  ([move-ordering-and-pv.md §9.1](move-ordering-and-pv.md))．

エッジコストを pn に折り込むと pn が二重信号となり δ-sum 算術を歪めうる，という観察に基づく
切り分けオプションである (既定 off = エッジコスト folded)．

### 5.3 インライン 1 手詰検出 (constructive)

子ノードを本格展開する前に，1 手詰を即決して `pn=0` (詰み proven) を確定すれば，展開と
再帰を丸ごと省ける．

**実装:** `mate1ply` (`movegen/mate1ply.rs`)．**constructive** (詰ます手 `Move` 自体を返す):

1. **玉の幾何による候補生成**: `generate_mate_candidates` が玉の周囲から詰ます手の候補
   (王手マス・打ち駒位置) を構築する (全合法手生成を避ける)．
2. `mate_move_in_1ply_maxdist(.., 2)` で距離 ≤ 2 の候補を検証する (健全な部分集合)．
3. 逆王手 (攻め方が王手されている) 局面は `mate1ply_cached_near2` の別経路で扱う．

王手手の列挙は **`CheckCache`** (`movegen/check_cache.rs`, direct-mapped 8192 エントリ・
1 局面あたり最大 32 手) で再利用し，同一局面の王手再生成を避ける．

**設計判断:** 3 手以上のインライン詰み検出は，閾値制御・TT を伴わない網羅探索となり mid 本体
より非効率になるため実装しない (mid の再帰に委ねる)．`Params::obvious_final_max_depth` は浅い
AND 親で子 OR の 1 手詰を即 proven 化する適用上限を制御する．
