# 証明数・反証数の集約

本節は子の φ/δ を親へ集約する規則と，DAG (転置) による**二重計数**を抑える機構を扱う．
旧版にあった WPN / CD-WPN / VPN / SNDA は統一 mid のコードには存在しない (二重計数除去は
δ-sum の `sum_mask` + DAG 検出 + cross-hand 集約が担う)．論文手法との関係は §4.4 で述べる．

### 4.1 φ/δ 集約と δ-sum (sum_mask)

**出典:** φ/δ 集約と sum_mask による二重計数除去は KomoringHeights で実装されている手法に基づく．

親ノードの値は子の集約で決まる ([search-architecture.md §2.2](search-architecture.md)):

```
φ(n) = min over children of δ(child)        // 最有望子 (best child)
δ(n) = Σ over sum_mask children of δ(child)  +  max over non-mask children of δ(child)
```

**実装:** `current_result` / `recalc_delta` (`search/expansion.rs`)．δ の総和は `sum_mask`
ビット集合で「sum で足す子」と「max で 1 回だけ数える子」を区別する:

```rust
for child in children_except_best:
    if sum_mask.test(child):
        sum_delta_except_best += δ(child)      // 独立な子: 加算
    else:
        max_delta_except_best  = max(.., δ(child))  // DAG 合流する子: max のみ
```

- 標準 df-pn は AND ノードの pn (= δ) を子の単純 sum とするが，DAG では複数の親を持つ子が
  各親から重複加算され δ が過大評価される．`sum_mask` を落とした子は max 集約となり 1 回だけ
  数えられるため，過大評価が抑えられる．
- `sum_mask` は親→子へ伝播する (`front_sum_mask`)．子へ渡す δ 閾値の計算
  ([threshold §3.1](threshold-control.md)) もこの sum/max 区別を反映する．

### 4.2 EliminateDoubleCount (DAG 検出)

**出典:** DAG 合流による二重計数の除去 (EliminateDoubleCount) は KomoringHeights で実装されて
いる手法に基づく．

どの子の `sum_mask` を落とすかは **DAG 合流の検出**で決める．`search_impl` 入口で毎回
`eliminate_double_count` を実行する (`no_dag()` env でのみ無効化; 既定で有効):

**実装:** `eliminate_double_count` (`search/mod.rs`)．

1. 各子の TT エントリが持つ親局面キー (`parent_board_key`) を参照する．
2. その親が**現在の探索パス上の先祖**と一致する子 = 別経路から本部分木へ合流する転置 (DAG)．
3. 該当子の `sum_mask` ビットを落とす → δ 集約で sum でなく max 扱いになる (二重計数除去)．

発火回数は診断カウンタ `dag_fires` で計測する (root 終了時の `[dfpn] … dag=` 行)．

### 4.3 cross-hand 集約 (持ち駒越境)

**出典:** 親エントリの cross-hand 参照 (`look_up_parent`) と証明駒・反証駒の活用は
KomoringHeights で実装されている手法に基づく．持ち駒優越の基礎は Nagai 2002．

TT は局面キー (盤面のみ，持ち駒は除く) でクラスタ化され，同一盤面の異なる持ち駒エントリを
保持する ([transposition-table.md §6.1](transposition-table.md))．集約・参照では持ち駒の
支配関係 (`hand_gte` / `hand_gte_forward_chain`) を使って越境的に証明/反証を再利用する:

- **`look_up_parent`** (`tt/mod.rs`): 子の集約に必要な親エントリを，厳密一致 hand があれば
  それを，無ければ `apply_delta_hand` で上位/下位 hand から境界を合成して返す
  (cross-hand 推論)．
- **証明駒・反証駒** (`proof_hand.rs`): 詰み (proven) 時はその局面を詰ます**極小の攻め方持ち駒**
  (`ProofHandSet`: 子の証明駒の要素 max + 打ち駒補正)，不詰 (disproven) 時は詰まない**極大の
  攻め方持ち駒** (`DisproofHandSet`: 子の反証駒の要素 min + 補正) を計算して TT へ格納する．
  これにより hand-dominance の集約が強まり，異なる持ち駒の局面間で結果が広く再利用される．

### 4.4 論文手法との関係

旧版は二重計数・過大/過小評価に WPN (Ueda 2008) / SNDA (Kishimoto 2010) / VPN (Saito 2006) の
近似式を用いていた．統一 mid はこれらを採用せず，**δ-sum の sum_mask による sum/max 切替
(§4.1) + DAG 合流の直接検出 (§4.2) + cross-hand 集約 (§4.3)** で二重計数を扱う:

- WPN/SNDA は「兄弟が同一 source を共有する」「max + 減衰和」といった近似で重複を割り引くが，
  孫以下の深い合流や持ち駒違いの転置を正確には扱えない．
- 本実装は TT の `parent_board_key` で合流を直接判定し，持ち駒は支配関係で越境集約するため，
  近似パラメータ (γ shift 等) のチューニングを必要としない．

WPN/CD-WPN/SNDA を用いた分布チューニング (KL/σ 指標，γ スイープ) は旧版で試行されたが，
現行コードに対応機構は無い (記録は git 履歴)．

**出典 (prior art):** Ueda et al., "Weak Proof-Number Search" (CG 2008);
Kishimoto, "Dealing with Infinite Loops, …" (AAAI 2010, SNDA); Saito et al., "Virtual Proof
Number" (2006)．
