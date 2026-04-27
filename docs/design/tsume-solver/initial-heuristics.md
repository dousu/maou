# 初期値ヒューリスティック

### 5.1 df-pn+ ヒューリスティック初期化 (GPW 2004; KomoringHeights v0.4.0)

**出典:** Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004);
KomoringHeights v0.4.0

標準 df-pn は全リーフを `(pn=1, dn=1)` で初期化するが，
df-pn+ では局面の特徴に基づいて初期 pn/dn を設定する．
玉の逃げ場が少ない局面ほど pn を小さく(詰みやすい)，
王手手段が多い局面ほど dn を大きく(反証しにくい)する．

**実装:**

#### `heuristic_or_pn` (solver.rs)

OR 子(攻め方局面)の初期 pn．王手数と玉の安全な逃げ場で 1S〜64S に調整:

**escape\_base (safe\_escapes に基づくベース値):**

| safe\_escapes | escape\_base (S = PN\_UNIT) | bucket |
|--------------|----------------------------|--------|
| 0 | 1S | 4 |
| 1 | 2S | 5 |
| 2 | 4S | 6 |
| 3 | 8S | 7 |
| 4 | 16S | 8 |
| 5 | 24S | ~8.6 |
| 6 | 32S | 9 |
| 7+ | 48S | ~9.6 |
| 開放空間 (隣接≥5，圧迫=0，逃げ場≥4) | **64S** (直接返却) | 10 |

**num\_checks によるスケーリング (escape\_base の 1.0〜2.0 倍):**

| num\_checks | 乗数 | 備考 |
|------------|------|------|
| ≥8 | ×1.0 | 多数の王手 → ベースのまま |
| 4〜7 | ×1.25 | |
| 2〜3 | ×1.5 | |
| 1 | ×2.0 | 王手が1つのみ → 詰みにくい |

上限: 64S (=1024, bucket 10)．開放空間のみ上限適用前に直接返却．

v0.30.0 で safe\_escapes の値域を 1S〜8S (bucket 4-7) から 1S〜48S (bucket 4-9.6) に
拡張し，上限も 8S → 64S に引き上げた (§5.1.1 `heuristic_dn_from_pn` 導入と対応)．

#### `heuristic_dn_from_pn` (mod.rs) — v0.30.0

OR 子の初期 dn を初期 pn から逆比例で算出する．

```
C = (8S)² = (8 × PN_UNIT)²
dn = C / pn,  clamp(PN_UNIT, 64 × PN_UNIT)
```

pn と dn の積が (8S)² ≈ 一定となる逆比例関係:

| 初期 pn | 初期 dn | bucket (dn) |
|---------|---------|-------------|
| 1S | 64S | 10 |
| 2S | 32S | 9 |
| 4S | 16S | 8 |
| 8S | 8S | 7 |
| 16S | 4S | 6 |
| 32S | 2S | 5 |
| 64S | 1S | 4 |

**導入の背景 (v0.30.0):**
v0.29.x 以前は初期 dn を全 OR 子で `PN_UNIT` (1S, bucket 4) に固定していた．
この状態で OR dn WPN を適用すると，多子ノードで WPN が dn を過度に削減し，
`test_no_checkmate_counter_check` が 2M ノード予算を超過して失敗した．

`heuristic_dn_from_pn` により初期 dn を bucket 4〜10 に展開することで，
OR dn WPN の削減が過度にならず安定動作するようになった (§4.1 参照)．

#### `heuristic_and_pn` (solver.rs)

AND 子(守備方局面)の初期 pn．応手数と玉の安全な逃げ場で調整:

| 条件 | 初期 pn (S = PN\_UNIT) | v0.20.x 互換 |
|------|----------------------|-------------|
| 逃げ場なし | `n * 2/3 * S` | `n * 2/3 * S` |
| 逃げ場=1 | `n * S + S/4` | `n * S` |
| 逃げ場=2 | `n * S + S/2` | `n * S` |
| 逃げ場=3 | `n * S + 3S/2` | `(n+1) * S` |
| 逃げ場≥4 | `n * S + e*S/2 + S/4` | `(n + e/2) * S` |

n = num\_defenses, e = safe\_escapes．v0.21.0 で逃げ場 1〜2 にも
中間値を返すことで閾値配分の精度を向上させた．

### 5.2 DFPN-E エッジコスト型 (NeurIPS 2019)

**出典:** "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)

リーフ(ノード)ではなくエッジ(親→子遷移の手)にヒューリスティックコストを付与する．
展開済みノードではエッジコストがゼロになるため，実質的には初期 pn への加算として機能する．

**実装:** mod.rs

#### `edge_cost_or` (OR ノードの王手): mod.rs

| 手の種類 | コスト |
|---------|--------|
| 成王手 / 取王手 | 0 (最有力) |
| 近い静か王手 (距離≤2) | 1 |
| 遠い静か王手 (距離≥3) | 2 |

#### `edge_cost_and` (AND ノードの応手): mod.rs

| 応手の種類 | コスト |
|-----------|--------|
| 合駒 (drop) | 0 (攻め方が取り進んで有利) |
| 玉の逃げ / 駒移動 | 1 |
| 駒取り | 2 (攻め駒除去で攻め方不利) |

**出典との差異:**
- 論文のコスト関数はドメイン非依存の汎用設計だが，
  maou_shogi では将棋の詰みに特化したドメイン知識(成/取/距離/合駒)を組み込み

### 5.3 Deep df-pn (Song Zhang et al. 2017)

**出典:** Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)

深い位置ほど初期 dn を高く設定し，浅い解を優先する．
論文推奨値: `dn_init = max(1, ceil(R * depth))` (R=0.4, Othello/Hex)．

**実装:** mod.rs (`DEEP_DFPN_R`), solver.rs (`look_up_pn_dn`)

TT ミス時(pn=1, dn=1, source=0)に深さバイアスを適用:

```
if ply > depth / 2:
    biased_pn = 1 + (ply - depth/2) / DEEP_DFPN_R    (DEEP_DFPN_R = 4)
```

浅い ply (depth の前半) は標準 df-pn と同じ pn=1 を維持．

**出典との差異:**
- 論文は dn にバイアスを適用するが，maou_shogi では **pn にバイアス**を適用
- 論文の R=0.4 (小さいほど積極的) に対し，maou_shogi では `R=4` (整数除算)
- 深い ply の未探索子の pn を上げることで，探索済みの浅い子を優先する効果
- バイアス適用は depth の後半のみ(前半は標準 pn=1 で不詰検出を維持)

### 5.4 インライン詰み検出

child_init フェーズ(子ノードの TT 初回参照時)で，
MID の再帰呼び出しなしに1手・3手の詰み/不詰を即座に判定する．

**実装:** solver.rs (child init)

#### AND 子ノード(OR 局面)の検出: solver.rs (child init, `or_node` ブランチ)

1. `generate_defense_moves(board)` で全応手を生成
2. 応手なし → 即詰み確定(pn=0, dn=INF)
3. `ply + 2 < depth` なら3手詰め判定:
   - 各応手を実行し `has_mate_in_1_with(board, checks)` で全応手に1手詰みがあるか確認
   - 全応手に対して1手詰みが存在 → 即詰み(pn=0)

#### OR 子ノード(AND 局面)の検出: solver.rs (child init, `!or_node` ブランチ)

1. `generate_check_moves(board)` で全王手を生成
2. 王手なし → 即不詰(pn=INF, dn=0)
3. `ply + 2 < depth` なら:
   - `has_mate_in_1_with(board, checks)` で1手詰み判定
   - `try_capture_tt_proof(board, checks, remaining)` で TT 参照の即証明

#### `has_mate_in_1_with` ヘルパー: solver.rs

`board.mate_move_in_1ply(checks, us)` で1手詰みを検出．
詰み発見時は詰み局面を TT に記録し，将来の探索で再利用可能にする．

**設計判断:** 5手以上のインライン検出は MID の枝刈り(閾値制御・TT 参照)なしの
網羅探索となり，MID 自体より非効率になるため実装しない．
過去に実装した budget 付き N 手詰め検出(static_mate)は TT 汚染と
探索効率の悪化を招いたため v0.20.24 で削除した．

---

