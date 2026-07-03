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

旧版にあった Killer move・捨て駒ブースト・TT Best Move 動的手順改善は統一 mid では
廃止された (記録は git 履歴)．

---

## 9-b. PV 復元 (Principal Variation Extraction)

df-pn は探索木を明示的に保持しないため，詰み証明後に **TT を辿って PV (最善手順) を復元**する．
統一 mid は 2 段構成: **`verify_proof` (健全性検証付き replay) → `build_pv` (手順構築)**
(`search/pv.rs`)．旧版の 3-phase 構成 (complete_or_proofs / extract_pv_recursive / 検証) は
この 2 段に整理された．

### 9-b.1 verify_proof (STRICT replay; solve_impl の最終権威)

証明木を再帰的に辿り直し，**PV 上の全応手が実際に詰むこと**を検証する (偽の証明を弾く)．
`solve_impl` はこれを**最終権威**とし，STRICT が `Some(d)` のときのみ `Checkmate` を返す
([loop-ghi §7.5](loop-ghi.md))．

- **OR ノード** (攻め方): まず `mate1ply` を試し ([heuristics §5.3](initial-heuristics.md))，
  無ければ王手を列挙して TT-proven の子を候補化し再帰検証する．1 つでも詰む王手があれば
  proven．候補の子 key/hand は do_move せず incremental に算出し (`hashes_after`/`hand_after`)，
  query 一括構築 + prefetch で TT の DRAM latency を隠す (探索の child loop と同手法)．
- **AND ノード** (守備方): 全合法応手を列挙し，**全てが詰みに至る**ことを要求する (最長抵抗)．
  無駄合いの除外 (手数集計のみ) は [aigoma §8.2](aigoma-optimization.md)．
- **memo** (`FxHashMap<u64, Option<u16>>`): 局面ごとに検証結果を記録し再検証を避ける．
  **経路依存の None (千日手拒否・budget 枯渇由来) は memo しない** — dep 伝播で依存が自
  subtree 内に閉じた None のみ cache する (verify 内 GHI 対策;
  [loop-ghi §7.5](loop-ghi.md))．Some は構成的に経路非依存で常に memo する．
- 予算 (既定 80M call) 内で詰み手数 `Option<u16>` を返す．None = 未完/不健全 → `Unknown`．

**2-tier fast/full**: verify は「証明 DAG 閉包の全 replay」ゆえ全候補検証は高価
(導入前は 39te で全体 wall の ~31% を占めていた)．そこで 2 段構成にする:

1. **fast tier**: OR ノードで TT len 昇順の**最初に検証成功した候補**を採用して打ち切る．
   どの検証済 child でも詰みの証明になるため **soundness は全候補検証と同一**．非保証なのは
   PV の最短選択のみ．
2. **full fallback**: fast の PV 長が search の最短 claim (`mate_len`) を超えた場合のみ，
   全候補から最短を選ぶ動作で再検証し最短性を保全する (canonical 問題では fast で常に
   一致し fallback は発火しない)．

導入時の測定 (39te): verify wall 24.5s → 6.6s．ログは
`STRICT VERIFY Some(d) (..., wall=…, tier=fast|full-fallback)` で tier を報告する．

この STRICT 検証は，canonical テストの健全性ゲート (偽証明 = 即停止) の基盤である．

### 9-b.2 build_pv (手順構築)

`verify_proof` が局面ごとに記録した最適手 (`pv_choice: FxHashMap<u64, Move>`) を辿り，
手順を構築する:

- **OR ノード**: 検証済みの子のうち**最短**手数の手 (fast tier では TT len 昇順の最初の
  検証成功手; PV 長 guard は §9-b.1) を選ぶ (最善の攻め)．
- **AND ノード**: 詰みに至る応手のうち**最長**手数の手を選ぶ (最善の受け = 最長抵抗;
  無駄合いは集計から除外済)．
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

**len 予算の強制 (重要)**: len-bounded 探索は **len 予算を超える proof を返してはならない**．
これを欠くと余詰の `len=d-2` 探索が予算超過の偽 proof (例: len=53 探索が mate_len=55) を返し収束
しない．3 点で予算を強制する:
- `MateLen::sub` は下限 0 で saturate する (`wrapping_sub` だと 0 未満で u32::MAX≒∞へ wrap し
  予算が消失する)．
- look-ahead (`check_obvious_final_or_node`) は子の budget が mate-1 を許す場合のみ proven を seed．
- `build_expansion` は非終端ノードを `len < 1手` で **budget-limited disproven** にする (予算切れ
  cutoff)．DEPTH_MAX 探索では len が高位飽和し発火しないので first-mate 挙動は不変．

len 予算は**無駄合い-free len credit** ([aigoma §8.4](aigoma-optimization.md)) で
無駄合い抜き手数に一致させている (これを欠くと無駄合い込み raw ply の予算で短い詰みを偽
disproof する)．29te 最短 29 手 / **39te 最短 39 手** / post-2c3d 31 手を user oracle と一致で
confirm する (canonical anchor: 29te 396,516 / 39te 17,545,528 nodes; `tests.rs` が assert)．
`test_39te_divergence_probe` が分岐局面の残手数を再帰確認して局所化する診断方法論を提供する．
`find_shortest=false` では最初に見つかった手順 (最短保証なし; ノード数削減) を返す．
