# 手順改善・PV 復元

## 9. 手順改善 (move ordering)

最有望子の選択は φ/δ の集約値で決まるが (§[search 2.4](search-architecture.md))，値が同点の
子の間では手の良し悪しで順序を決める．これにより詰みやすい/反証しやすい手を先に深掘りできる．

### 9.1 探索結果の比較順序

**実装:** `compare_results` (`search_result.rs`)．子は次の辞書式順序でソートされる (best = 先頭):

1. **φ 昇順** — 手番側の目標に近い子を優先．
2. **δ 昇順** — 同点なら相手の負けに近い子を優先．
3. **len** — proven (pn=0) の同点時: OR は**短い詰み**，AND は**長い詰み (最長抵抗)** を優先．
4. **repetition_start** — 千日手起点 ([loop-ghi §7.1](loop-ghi.md))．
5. **amount** — 探索量．
6. **move eval tie-break** — 上記が全て同点の最終 tie は手の評価で破る (`LocalExpansion` 側)．

### 9.2 手の静的評価 (`move_brief_eval`)

**実装:** `move_brief_eval` / `king_supports` (`heuristics.rs`)．tie-break に用いる i32 キー
(小さいほど優先) を，盤情報から安価に計算する:

- 成れるのに成らない手にペナルティ (成り王手・成り応手を優先)．
- 駒価値を反映 (価値の高い駒を動かす手を優先)．
- 玉との位置関係を反映 (玉に迫る手を優先)．

エッジコスト ([heuristics §5.2](initial-heuristics.md)) を pn から分離する
`decouple_edge_cost` モードでも，この tie-break は同じ順序を維持する (難易度推定と手順選好の
分離)．旧版にあった Killer move・捨て駒ブースト・TT Best Move 動的手順改善は統一 mid では
廃止された (記録は [legacy/](legacy/README.md))．

---

## 9-b. PV 復元 (Principal Variation Extraction)

df-pn は探索木を明示的に保持しないため，詰み証明後に **TT を辿って PV (最善手順) を復元**する．
統一 mid は 2 段構成: **`verify_proof` (健全性検証付き replay) → `build_pv` (手順構築)**
(`search/pv.rs`)．旧版の 3-phase 構成 (complete_or_proofs / extract_pv_recursive / 検証) は
この 2 段に整理された．

### 9-b.1 verify_proof (STRICT replay)

証明木を再帰的に辿り直し，**PV 上の全応手が実際に詰むこと**を検証する (偽の証明を弾く)．

- **OR ノード** (攻め方): まず `mate1ply` を試し ([heuristics §5.3](initial-heuristics.md))，
  無ければ王手を列挙して各子を再帰検証する．1 つでも詰む王手があれば proven．
- **AND ノード** (守備方): 全合法応手を列挙し，**全てが詰みに至る**ことを要求する (最長抵抗)．
- **memo** (`HashMap<u64, Option<u16>>`): 局面ごとに検証済み手数を記録し再検証を避ける．
- 予算 (既定 80M ノード) 内で詰み手数 `Option<u16>` を返す．None = 未完/不健全．

この STRICT 検証は，canonical テストの健全性ゲート (偽証明 = 即停止) の基盤である．

### 9-b.2 build_pv (手順構築)

`verify_proof` が残した手数 memo を辿り，手順を構築する:

- **OR ノード**: 詰みが証明された子のうち**最短**手数の手を選ぶ (最善の攻め)．
- **AND ノード**: 詰みに至る応手のうち**最長**手数の手を選ぶ (最善の受け = 最長抵抗)．
- これにより「最善応手に対する最短詰み手順」が得られる．

### 9-b.3 find_shortest と CheckmateNoPv

`find_shortest` (既定 true) は PV 上の OR ノードで未証明の王手を追加証明し，より短い手順が
あれば採用する (`complete_or_proofs` 相当)．これは 1 子あたり `pv_nodes_per_child` (既定 1024)
ノードの予算を使う．長手数 (17 手以上) の問題でこの予算が尽きると，詰みは証明済みでも PV を
返せず `TsumeResult::CheckmateNoPv` となる ([index §2](index.md))．`pv_nodes_per_child` を
増やすと改善する．`find_shortest=false` では最初に見つかった手順をそのまま返す
(最短保証なし; ノード数は削減)．
