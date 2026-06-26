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

### 9-b.3 find_shortest (余詰探索) と最短手数

`find_shortest` (既定 true) は **mate-length パラメータ化探索 (len-aware df-pn)** で最短手数の
詰みを返す (KH `SearchMainLoop` 相当の余詰探索):

1. `run_search_at_len(len=DEPTH_MAX)` で詰みを 1 つ見つけ手数 `d` を得る．
2. `len = d-2` で再探索する．より短い詰みがあれば置換，無ければ `d` が最短と確定する
   (詰将棋の手数は奇数ゆえ 2 ずつ短縮)．dual-range len-aware TT (`proven_len`/`disproven_len`)
   により前回 len=d で proven のノードも len<d では再評価され，無関係なノードは warm TT のまま
   再利用される．
3. loop guard: `shorter.pn()==0 && shorter.len() < d` のときのみ採用する (非厳密短縮・len 境界の
   偽結果による oscillation を防ぐ)．

**len 予算の強制 (3.2.0; 重要)**: len-bounded 探索は **len 予算を超える proof を返してはならない**．
これを欠くと余詰の `len=d-2` 探索が予算超過の偽 proof (例: len=53 探索が mate_len=55) を返し収束
しない．3 点で予算を強制する:
- `MateLen::sub` は下限 0 で saturate (旧 `wrapping_sub` は 0 未満で u32::MAX≒∞へ wrap し予算が
  消失していた)．
- look-ahead (`check_obvious_final_or_node`) は子の budget が mate-1 を許す場合のみ proven を seed．
- `build_expansion` は非終端ノードを `len < 1手` で **budget-limited disproven** にする (予算切れ
  cutoff)．DEPTH_MAX 探索では len が高位飽和し発火しないので first-mate 挙動は不変．

**正解の oracle はユーザ**: 詰将棋の最短手数/正解 PV は **ユーザが絶対的参照点**である．KH は
詰将棋において参照点ではない (MinLength 余詰は**無駄合いを手数に数える**ため現実の最短と乖離する;
[aigoma §8.2](aigoma-optimization.md))．maou が既知手順と異なる解を出したら，採用前に SFEN と現 PV
を提示してユーザに確認する．

**現状 (3.2.0)**: 29te は最短 29 手を confirm (len=27 が正しく disprove)．**39te は 57→45 手まで
前進するが len=43 で false-disproof する完全性バグが残る** (真の最短 39 手に未到達; cold TT でも
再現するため warm-TT/無駄合い由来ではなく len-bounded 探索の完全性の別バグ)．
`test_39te_divergence_probe` が分岐局面の残手数を再帰確認して局所化する診断方法論を提供する．
`find_shortest=false` では最初に見つかった手順 (最短保証なし; ノード数削減) を返す．
