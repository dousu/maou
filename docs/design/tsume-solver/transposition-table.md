# 転置表管理

### 6.1 持ち駒優越 (Nagai 2002)

**出典:** Nagai 2002

盤面が同一で持ち駒が異なる局面間の包含関係を利用した TT 再利用:

```
  Proof reuse (pn=0):              Disproof reuse (dn=0):

  TT: hand={P,G} -> pn=0          TT: hand={P,P,G} -> dn=0

  query: hand={P,P,G}             query: hand={P}
  {P,P,G} >= {P,G} ? YES          {P} <= {P,P,G} ? YES
  -> reuse proof                   -> reuse disproof
  (more pieces = easier to mate)  (fewer pieces = harder to mate)
```

- **証明(pn=0)**: 攻め方の持ち駒が TT エントリ以上 → 再利用可(持ち駒が多いほど詰ませやすい)
- **反証(dn=0)**: 攻め方の持ち駒が TT エントリ以下 → 再利用可(持ち駒が少ないほど詰ませにくい)

**実装:** mod.rs (`hand_gte`), tt.rs (`look_up`)

- TT キー: `position_key(board)` = 盤面ハッシュ(持ち駒を**含まない**)
- TT 値: 同一クラスタ内に同一 `pos_key` の複数エントリを保持(§6.6)
- Lookup 時: クラスタ内で証明エントリを先に走査(証明優先)，その後反証エントリを走査

### 6.2 前方チェーン補填 (maou 独自)

**出典:** maou 独自手法

持ち駒優越の拡張として，歩 ≤ 香 ≤ 飛のカスケード補填を実装する．
チェーン合駒の文脈で，攻め方が合駒を取った後の持ち駒構成が異なっても，
前方利き系の駒種間で代替関係を認める．

**実装:** mod.rs (`hand_gte_forward_chain`)

代替関係:
- 歩の不足 → 香で代替可能
- 香の不足 → 飛で代替可能
- 歩の不足 → 飛で代替可能(カスケード)

桂・銀・金・角は独立判定(利きの方向が異なるため代替不可)．

### 6.3 Pareto Frontier 管理

**出典:** Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)

同一盤面に対する複数の TT エントリを Pareto frontier で管理する．
持ち駒とエントリの支配関係に基づき，冗長なエントリを排除する．

**実装:** tt.rs (`store_impl`)

- **最大エントリ数**: `CLUSTER_SIZE = 6`(v0.20.34〜，旧 `MAX_TT_ENTRIES_PER_POSITION = 16`)
- **証明エントリ(pn=0)**: 最小持ち駒のエントリを保持(少ない持ち駒で証明できるほど汎用的)
- **反証エントリ(dn=0)**: 最大持ち駒のエントリを保持(多い持ち駒で反証できるほど汎用的)
- **支配判定**: `hand_gte_forward_chain` (§6.2) による拡張支配関係を使用
- **容量超過時**: 異なる `pos_key` のエントリを優先的に置換．
  証明/確定反証を保護しつつ，`|pn - dn|` が最小の中間エントリを犠牲にする

#### 反証挿入時の中間エントリ除去

フラットテーブルでは反証(dn=0)エントリ挿入時に，
同一 `pos_key` の中間エントリ(pn>0, dn>0)を積極的に除去してスロットを確保する．
旧 HashMap 版では Vec に追加するだけで中間エントリは保護されていた
(`remaining` の不一致で将来必要になりうるため)．

フラットテーブルではクラスタの 6 スロットが厳密な上限であるため，
反証の挿入を保証するには中間エントリの除去が不可避である．
中間エントリは探索の再訪時に再計算可能であり，
反証(確定結果)の保持を優先する設計判断である．

#### 置換スコア `|pn - dn|` の設計根拠

中間エントリの置換優先度にはスコア `|pn - dn|` を使用する．
この値が小さいエントリ(pn ≈ dn)は証明にも反証にも近くない「均衡状態」であり，
探索方向が未確定のため情報価値が相対的に低い．
一方，`|pn - dn|` が大きいエントリは証明(pn << dn)または反証(pn >> dn)に
偏っており，探索の進行方向を示す有用な情報を持つ．

**既知の限界:**

`pn = dn = 100,000`（深く探索済みの均衡局面）と `pn = dn = 1`（新規エントリ）は
同一スコア 0 となり，深い探索結果が新規エントリと同等に退避される．
探索量ベースのスコア(例: `pn + dn`)を組み合わせることで改善できる可能性があるが，
以下の理由で現時点では採用しない:

1. 1M クラスタでの TT 利用率は ~18%(10M ノード探索)であり，
   置換の影響は限定的
2. `pn + dn` スコアでは `pn=1, dn=1` の新規エントリが最優先で退避され，
   IDS の浅い反復で蓄積した初期値が過保護になるリスクがある
3. 置換ポリシーの変更は探索パターン全体に波及し，退行テストが必要

### 6.4 TT ガベージコレクション

**実装:** tt.rs (`gc_working_sampling`, `gc_proven`, `gc_working_overflow`), solver.rs (periodic GC)

v0.24.10 で KomoringHeights のサンプリング GC 方式を参考に全面刷新．

- **サンプリング GC** (`gc_working_sampling`):
  1. ストライドサンプリングで 10,000 エントリの amount 分布を収集
  2. `select_nth_unstable` で下位 20% の amount 閾値を決定
  3. 証明/反証済み局面の obsolete intermediate を ProvenTT 参照で除去
  4. 閾値以下の disproof を除去(intermediate は保護)
  5. CutAmount: `max_amount > 32` なら生存エントリの amount を半減
- **overflow GC** (`gc_working_overflow`): obsolete intermediate の除去のみ(disproof 保護)
- **充填率 GC**: WorkingTT 70% 超で `gc_working` を発動(1M ノード毎)
- **ProvenTT GC**: 充填率 70% 超で confirmed disproof → proof の順に段階除去
- **探索パス保護**: GC 前に `path` 配列のエントリの amount を 255 に引き上げ
- **IDS 間清掃**: `clear_working()` + `clear_proven_disproofs()` (従来通り)

**rem=0 仮反証の TT store 廃止** (v0.24.14):
depth-limit 到達時の仮反証(remaining=0)は TT に格納せず，
`look_up_pn_dn` で `remaining == 0` の場合に `has_proof` で動的判定する．
rem=0 disproof はクラスタの 64.7% を占めていた overflow の主因であり，
同じ IDS depth の同じ深さでしか参照されないため TT 格納は不要．

### 6.5 TT Best Move 保存 (KomoringHeights v0.4.0)

**出典:** KomoringHeights v0.4.0

TT エントリに最善手(`best_move: u16`)を保存し，動的手順改善(§9.1)に使用する．

**実装:** entry.rs (`DfPnEntry`), tt.rs (`store_with_best_move`, `look_up_best_move`)

- `store_with_best_move`: MID ループの中間結果保存時に最善子の Move16 を記録
- `look_up_best_move`: TT ヒット時に最善手を取得し，手順の先頭にスワップ

### 6.6 TT データ構造

**実装:** `entry.rs` (`DfPnEntry`), `tt.rs` (`TranspositionTable`)

```rust
#[repr(C)]
struct DfPnEntry {
    source: u64,               // SNDA ソースハッシュ (§4.4)
    pn: u32,                   // 証明数
    dn: u32,                   // 反証数
    hand: [u8; HAND_KINDS],   // 攻め方の持ち駒 (7 bytes)
    path_dependent: bool,      // GHI フラグ (§7.1)
    remaining: u16,            // 深さ制約 (0..depth or REMAINING_INFINITE)
    best_move: u16,            // 最善手 (Move16 エンコーディング)
    amount: u16,               // 探索投資量 (GC/置換の保護優先度)
}  // 32 bytes (#[repr(C)] + field order optimization)
```

**TT 全体構造:** フラットハッシュテーブル (v0.20.34 〜)

v0.20.34 で `FxHashMap<u64, Vec<DfPnEntry>>` から固定サイズのフラット配列に置換(§11.4)．
`CLUSTER_SIZE = 6` エントリ/クラスタ，デフォルト 2M クラスタ(≈ 480 MB)．
`pos_key & (num_clusters - 1)` によるダイレクトインデクシングで O(1) アクセス．

```
TTFlatEntry = { pos_key: u64, entry: DfPnEntry }  // 40 bytes
Cluster     = [TTFlatEntry; 6]                     // 240 bytes
Table       = Vec<Cluster>                         // 2M clusters ≈ 480 MB
```

**置換ポリシー:** クラスタ満杯時は異なる `pos_key` のエントリを優先的に置換し，
同一 `pos_key` の証明(pn=0)・確定反証(dn=0, REMAINING\_INFINITE)は保護する．
パレートフロンティア管理(§6.3)，前方チェーン比較(§6.2)，
経路依存フラグ(§7.1)のセマンティクスは完全に維持．

反証エントリの挿入時，クラスタが foreign protected エントリ(別 pos\_key の
proof/confirmed disproof)で埋まっている場合，`replace_weakest_for_disproof`
により foreign depth-limited disproof → foreign confirmed disproof → foreign proof
の優先順で1スロットを犠牲にして挿入する(v0.21.1)．

#### クラスタ衝突と暗黙的置換

フラットハッシュテーブルでは TT GC(§6.4)とは独立に，
**ハッシュ衝突による暗黙的置換**が発生する．

異なる `pos_key` が `pos_key % num_clusters` で同一クラスタにマッピングされると，
6 スロットを複数の局面が共有する．スロットが満杯になると `replace_weakest` が
異なる `pos_key` の中間エントリを上書きする．
これは TT GC のように明示的に発動するのではなく，
`store_impl` の通常動作として常に発生する．

#### クラスタサイズの決定根拠 (v0.21.1, Historical)

**注:** 以下は v0.21.1 時点の単一 TT (CLUSTER\_SIZE=6) の分析．
v0.24.0 以降は Dual TT (ProvenTT: CLUSTER\_SIZE=8, WorkingTT: CLUSTER\_SIZE=6) に
変更されている(§6.6.3, §6.6.4 参照)．

**CLUSTER\_SIZE = 6 の理由:**

将棋の持ち駒は 7 種(歩・香・桂・銀・金・角・飛)であり，
同一盤面に異なる持ち駒で到達する**転置**(transposition)が頻繁に発生する．
TT は pos\_key(盤面ハッシュ)でクラスタを索引し，hand(持ち駒)で
エントリを区別する．同一 pos\_key に必要な hand バリアント数は:

| 状況 | 必要バリアント数 | 根拠 |
|------|---------------|------|
| 典型的な中盤 | 2-3 | proof + intermediate(1-2 hand) |
| 合駒チェーン | 4-6 | 駒種ごとの合駒で異なる hand |
| depth boundary 付近 | 3-5 | depth-limited NM × 複数 hand |

CLUSTER\_SIZE=4 と CLUSTER\_SIZE=6 の比較(4M クラスタ):

| CLUSTER\_SIZE | TT max (29手詰め) | 結果 |
|--------------|------------------|------|
| 4 | 8.3M | NOT PROVED (120M nodes) |
| **6** | **12.5M** | **SOLVED (109M nodes)** |

CLUSTER\_SIZE=4 では hand バリアントがクラスタに収まらず TT が飽和する．
CLUSTER\_SIZE=6 は hand バリアント(典型3-5) + foreign 衝突分(1-2)で必要十分．

#### クラスタ数の決定根拠 (v0.21.1)

**DEFAULT\_NUM\_CLUSTERS = 2M (1<<21) の理由:**

Poisson 近似による overflow 確率分析:
`N` 個のエントリを `C` 個のクラスタに格納するとき，
各クラスタの期待エントリ数は `λ = N/C`．
クラスタが溢れる(CLUSTER\_SIZE 以上のエントリ)確率は
`P(X >= 6) for X ~ Poisson(λ)`．

| C(クラスタ) | スロット | メモリ | λ @5M | P(overflow) @5M | λ @10M | P(overflow) @10M |
|------------|---------|--------|-------|----------------|--------|-----------------|
| 1M | 6M | 240 MB | 5.0 | 38% | >容量 | — |
| **2M** | **12M** | **480 MB** | **2.5** | **3.5%** | **5.0** | **38%** |
| 4M | 24M | 960 MB | 1.25 | 0.2% | 2.5 | 3.5% |

**実測ベンチマーク(29手詰め no\_pns):**

| クラスタ | TT max | 結果 | ノード | 時間 | メモリ |
|---------|--------|------|--------|------|--------|
| 1M | 3.1M | NOT PROVED | >120M | — | 240 MB |
| **2M** | **6.3M** | **SOLVED** | **59M** | **201s** | **480 MB** |
| 4M | 12.5M | SOLVED | 109M | 556s | 960 MB |

2M クラスタが最速である理由:
- **キャッシュ効率**: 480 MB は L3 キャッシュに近い範囲．960 MB は完全にキャッシュ外
- **NPS 向上**: メモリアクセスの局所性が高く，実効 NPS が向上
- **TT 充足性**: 6.3M エントリは29手詰めの探索空間に十分

1M クラスタでは TT が 3.1M で飽和(ハッシュ衝突限界)し，問題が要求する
~6M の固有エントリを保持できないため解決不能．

**Periodic GC:** TT が capacity の 80% に達した場合に amount ベースの GC を実行:
Phase 1 で amount=0 の中間エントリを除去，Phase 2 で全中間エントリを除去．
ただしクラスタレベルの飽和(全体 52% でも特定クラスタが満杯)では
capacity 閾値に達しないため発動しない(§6.6.1)．

**amount フィールド (v0.22.0):** KomoringHeights の `amount_` に相当する
探索投資量メトリック．中間エントリは更新のたびに +1，proof に +100，
disproof に +25-50 のボーナスを加算．`replace_weakest` は最小 amount の
エントリを優先置換し，探索投資量の大きいエントリを保護する．
フィールド順序の最適化(`#[repr(C)]` + source u64 先頭配置)により
エントリサイズ 40 bytes を維持(amount 追加前と同サイズ)．

#### 6.6.1 クラスタ飽和問題

v0.21.1 の 29 手詰め診断で特定された TT の構造的課題．

**用語定義:**

| 用語 | 対象 | 症状 |
|------|------|------|
| **グローバル飽和** | TT 全体メモリ | 容量 80% で GC 発動 |
| **クラスタ飽和** | 1 クラスタ(6 スロット) | store 失敗で探索停滞 |
| **TT スラッシング** | TT エントリ組成 | intermediate=0，新規挿入ゼロ |
| **MID 停滞** | MID ループの進捗 | pn/dn 不変で脱出 |

**症状:** TT がグローバル容量の 52% しか使用していないにもかかわらず，
新規エントリの挿入が失敗し，探索が停滞する．

**原因:** 将棋の持ち駒転置により同一 `pos_key` に対して複数の hand バリアントが
必要だが，固定サイズクラスタ(6 エントリ)では以下が同時に発生する:

1. **hand バリアント飽和**: 同一盤面に 4-6 種の hand バリアント(proof + disproof +
   intermediate × 複数 hand)がクラスタを占有
2. **foreign protected 占有**: 別の `pos_key` の proof/confirmed disproof が
   クラスタの全スロットを占有し，新規エントリが挿入不能
3. **NM 非伝搬**: depth-limited NM を store しても同一 hand のエントリが
   クラスタに存在しないため look\_up でヒットしない

**影響:** MID ループが同じ子を繰り返し選択し pn/dn が不変のまま
ノードを浪費する(sc\_loop\_hang，MID ループ停滞)．

**現在の対策 (v0.22.0):**

| 対策 | 機構 | 効果 |
|------|------|------|
| `replace_weakest_for_disproof` | foreign protected の段階的犠牲 | NM 挿入可能に |
| amount ベース置換 | 低 amount エントリを優先淘汰 | 高価値エントリの保護 |
| single-child 停滞検出 | SC\_STAGNATION\_LIMIT=4 | 無限ループ防止 |
| MID ループ停滞検出 | pn/dn 不変で脱出 | depth boundary 空振り防止 |
| MID チャンク 1M 固定 | TT 成長チェック頻度向上 | Frontier 早期移行 |

**根本的な限界:** クラスタ方式では hand バリアント数がクラスタサイズを超える
問題に対して構造的に脆弱であり，対症療法(停滞検出)で補っている．

**TT 構造の変遷と設計トレードオフ:**

| 版 | 構造 | NPS | TT 容量 | 問題 |
|----|------|-----|---------|------|
| v0.20.32 | FxHashMap + Vec | ~227K | 14.3M | ヒープ確保 + ポインタ追跡 |
| v0.20.34 | フラットクラスタ (1M×6) | **~868K** | 1.3M | クラスタ飽和 |
| v0.22.0 | フラットクラスタ (2M×6) | ~253K | 6.3M | クラスタ飽和(緩和) |
| v0.22.1 | リニアプロービング (8M) | ~89K | 8.5K overflow | GC rebuild コスト |

HashMap→フラット化で NPS が 3.83× 向上した一方，TT の実効容量が
14.3M → 1.3M に激減した．v0.22.0 でクラスタ数を 2M に拡大し実効容量を
6.3M に改善したが，キャッシュ効率の低下で NPS は ~253K に低下．

KomoringHeights はリニアプロービング + amount ベース置換を採用しており，
クラスタサイズの制約がない．ただし maou\_shogi では HashMap/リニアプロービングの
NPS が低い(~227K)ため，クラスタ方式の NPS 優位(~253K-868K)を維持しつつ
クラスタ飽和を緩和する方向で改善を進めている(方針D §10.2)．

**実効容量と持ち駒バリアントの関係:**

クラスタ方式の実効容量が理論上限(全スロット数)を大幅に下回る原因は，
持ち駒バリアントによるクラスタ偏在である．以下にクラスタ構成ごとの
実効容量率(= 実測 TT max / 全スロット数)を示す:

| クラスタ構成 | 全スロット数 | 実測 TT max | 実効容量率 | 問題 |
|------------|-----------|-----------|----------|------|
| 1M × 6 | 6M | 3.1M | **52%** | ハッシュ衝突限界で解決不能 |
| 2M × 6 | 12M | 6.3M | **53%** | 29手詰め解決可能(最速) |
| 4M × 6 | 24M | 12.5M | **52%** | 解決可能だがキャッシュ効率悪化 |

全構成で実効容量率が約 **52-53%** に収束している．
これは Poisson 過程による確率的限界ではなく，
詰将棋固有の持ち駒バリアント構造に起因する:

1. **hand バリアントの偏在**: 合駒チェーンの探索では同一 `pos_key` に
   4-6 種の hand バリアント(proof + disproof + intermediate × 複数 hand)
   が必要となり，特定クラスタが早期に飽和する
2. **foreign 衝突との複合**: 持ち駒バリアントが多い局面のクラスタに，
   偶然同一インデックスの別 `pos_key` が衝突すると 6 スロットが即座に枯渇
3. **GC で回復不能**: クラスタ内の proof/confirmed disproof は GC で
   除去されないため，一度飽和したクラスタは空きスロットが回復しない

この「実効容量 ~52%」はクラスタ方式の構造的上限であり，
クラスタ数を増やしても比率は改善しない(メモリ増加分だけ絶対容量が増える)．
29 手詰めは 2M クラスタの 6.3M エントリで解決可能だが，
より大規模な問題(39 手詰め等)では TT 容量不足が律速要因となる．

**リニアプロービングによる構造的解決の試み(v0.22.1，不採用):**

クラスタ飽和を根本的に解決するため，v0.22.1 でリニアプロービング
(8M エントリ，tombstone 方式)を実装・評価した．
クラスタサイズの制約がないため overflow は 98% 削減(449K→8.5K)されたが，
GC の計算量差異により NPS が大幅に低下し，不採用となった．
詳細は方針D(§10.2)を参照．

#### 6.6.2 NPS 最適化の分析 (v0.22.0)

**注:** 以下のプロファイルは v0.22.0 ベース．v0.24.0 以降の Dual TT 導入，
v0.24.7 の Zobrist hand\_hash，v0.24.14 の rem=0 廃止により，
メモリアクセスパターンと TT 操作のコスト構成が大幅に変化している．

29 手詰め no\_pns (74.2M ノード) のプロファイル結果:

| 操作 | 時間割合 | 呼出回数 | 平均(ns) | 備考 |
|------|---------|---------|---------|------|
| **child\_init 合計** | **56.3%** | **73M** | **2604** | 子ノード初期化 |
| └ ci\_lookup | 18.2% | 342M | 424 | TT クラスタスキャン |
| └ ci\_fastpath | 11.4% | 342M | 266 | depth limit チェック + store |
| └ ci\_do/undo\_move | 7.6% | 683M | 89 | 盤面状態変更 |
| └ ci\_resolve | 2.5% | 233M | 86 | 初期化後の解決チェック |
| └ ci\_inline | 2.0% | 342M | 45 | TT ミス時 heuristic 計算 |
| **movegen\_check** | **16.1%** | **28M** | **1975** | 王手生成 |
| **movegen\_defense** | **12.6%** | **46M** | **919** | 応手生成 |
| **main\_loop\_collect** | **7.2%** | **93M** | **259** | MID ループの pn/dn 収集 |
| tt\_store | 2.3% | 93M | 82 | TT 書き込み |
| do\_move/undo\_move | 2.7% | 148M | 60 | MID ループの盤面操作 |
| tt\_lookup | 1.3% | 74M | 60 | mid() エントリ時の TT 参照 |

**ボトルネック:** `child_init` が全体の 56% を占める．内訳では `ci_lookup`
(TT クラスタスキャン，424ns/回)と `ci_fastpath`(266ns/回)が支配的．
各 mid() 呼び出しで平均 4.7 個の子ノードに対して do\_move → TT lookup →
heuristic → undo\_move を実行する．

**実施した最適化と結果:**

| 最適化 | 時間 | 改善 | 採否 |
|--------|------|------|------|
| fastpath/lookup 重複排除 + depth limit フラグ化 | 287s | **-3%** | **採用** |
| 1-pass lookup (proof/disproof/exact 統合) | 306s | +7% 悪化 | 不採用 |
| child\_cache 差分更新 (main\_loop\_collect 削減) | >1350s | >4x 悪化 | 不採用 |

1-pass lookup は proof の early return を喪失して悪化．
child\_cache はスタック上の ArrayVec<593> が分岐予測を破壊し NPS が壊滅．

**NPS 改善の限界:** `child_init` の主要コストは do\_move/undo\_move(盤面操作)と
TT クラスタスキャン(hand 比較)であり，これらはアルゴリズムレベルの最適化では
削減困難．大きな NPS 改善には手生成(movegen, 29%)やビットボード操作レベルの
最適化が必要であり，TT 構造や df-pn アルゴリズムの改善とは直交する．

**NPS 改善候補(v0.22.1 起案，v0.23.0 で E1/E2/E5 採用，E4/E6 は既実装):**

以下はプロファイルデータに基づく改善案であり，各推定値は
29 手詰め no\_pns (74.2M ノード) のプロファイルから算出した概算値．

| # | 改善案 | 対象 | 推定改善 | 難度 | 状態 |
|---|--------|------|---------|------|------|
| E1 | ci\_resolve の再 lookup 廃止 | child\_init (2.5%) | +1-2% | 低 | **採用** (v0.23.0) |
| E2 | 王手生成キャッシュ | movegen\_check (16.1%) | +2-3% | 中 | **採用** (v0.23.0) |
| E3 | main\_loop\_collect の遅延評価 | main\_loop\_collect (7.2%) | +1-1.5% | 中 | 見送り |
| E4 | hand 比較の SWAR パック化 | ci\_lookup (18.2%) | +1-2% | 中 | **既実装** |
| E5 | 経路スタック化(FxHashSet→配列) | path 操作 | +0.5% | 低 | **採用** (v0.23.0) |
| E6 | step attacks テーブル化 | movegen (29%) | +1-2% | 低 | **既実装** |

**E1: ci\_resolve の再 lookup 廃止 — 採用(v0.23.0)**

depth-limit ファストパスで store 直後に `look_up_pn_dn` を呼んでいた箇所を，
`table.has_proof()` で proof の有無を先にチェックする方式に変更．
proof があれば `(cpn, cdn) = (0, INF)`，なければ store 後は必ず
`(INF, 0)` になるため再 lookup が不要．`has_proof` は `look_up_pn_dn` の
Pass 1 と同一ロジックで Pass 2/3 を省略する軽量メソッド．

**E2: 王手生成キャッシュ — 採用(v0.23.0)**

`CheckCache`(8192 エントリ，direct-mapped)で `generate_check_moves` の
結果をキャッシュ．局面ハッシュ(`board.hash`)をキーとする．
`UnsafeCell` による内部可変性で `&self` アクセスを実現し，
`mid()` のスタックフレーム最適化を阻害しない．

**実装上の知見:** 当初 `generate_check_moves_cached` を `&mut self` で
定義したところ，`mid()` のスタックフレーム最適化が阻害され，
テストスレッドの 8MB スタックでオーバーフローが発生した．
`mid()` の `children: ArrayVec<..., 593>` が各フレーム約 19KB を消費するため，
`&mut self` の追加的なレジスタスピルがスタック限界を超える原因だった．
`UnsafeCell` で `&self` にすることで解決．

**E3: main\_loop\_collect の遅延評価 — 見送り**

DAG 転置により子の pn/dn が探索済み子以外でも変化する可能性があり，
子キャッシュ方式は正確性リスクが高い．安全な最小最適化(ループ子フラグ
キャッシュ)のベネフィットが限定的なため見送り．

**E4: hand 比較の SWAR パック化 — 既実装**

`hand_gte` は SWAR (SIMD Within A Register) で u64 パック比較を
既に実装済み(`mod.rs`)．`hand_gte_forward_chain` の高速パスとして機能する．

**E5: 経路スタック化 — 採用(v0.23.0)**

`self.path: FxHashSet<u64>` を `[u64; 48]` + `path_len` の固定長配列
スタックに置換．LIFO スタック規律(mid() 入口で push，全出口で pop)により
`insert`/`remove` は O(1)，`contains` は最大 41 要素の線形スキャン．
FxHashSet のハッシュ計算・ヒープ操作を完全に排除．

**E6: step attacks テーブル化 — 既実装**

`STEP_ATTACKS: LazyLock<[[[Bitboard; 81]; PIECE_BB_SIZE]; 2]>` として
全駒種 × 全マスの step attacks がプリコンピュート済み(`attack.rs`)．

**Movegen: 玉移動合法性チェック高速化 — 採用(v0.23.0)**

`generate_defense_moves_inner` の玉移動合法性チェックを
`do_move` / `is_in_check` / `undo_move` から
`board.is_attacked_by_excluding(to, attacker, false, Some(king_sq))` に置換．
`excluded_sq=Some(king_sq)` で占有ビットボードから玉の元マスを除外し，
飛び駒が玉の元マスを貫通して移動先を利く場合を正しく検出する．

**pn\_floor 乗算オーバーフロー修正 — 採用(v0.23.0)**

`eff_pn_th * 2 / 3` を `(eff_pn_th as u64 * 2 / 3) as u32` に変更．
`eff_pn_th` が INF 近傍(complete\_or\_proofs 由来)の場合に u32 乗算が
オーバーフローし，debug ビルドで panic，release ビルドで不正値となる
既存バグを修正．この修正により 23 テストの既存失敗が解消された．

**未実施の NPS 改善候補(v0.23.0 時点):**

| # | 改善案 | 対象 | 推定改善 | 難度 |
|---|--------|------|---------|------|
| E7 | TT クラスタ単一パススキャン | ci\_lookup (18.2%) | +2-3% | 低 |
| E8 | SNDA pairs の事前確保 | main\_loop\_collect (7.2%) | +1-2% | 低 |
| E9 | child\_init lookup キャッシュ | child\_init (ci\_resolve) | +3-5% | 中 |
| E10 | collect の即時 proof/disproof 脱出 | main\_loop\_collect | +2-5% | 中 |
| E11 | look\_up\_pn\_dn に best\_move 統合 | ci\_lookup | +1-2% | 中 |

**E7: TT クラスタ単一パススキャン**

`look_up_pn_dn` は proof(Pass 1)→ disproof+exact(Pass 2)の2パスで
クラスタを走査する．単一ループで proof / disproof / exact\_match を
同時チェックすればクラスタ走査回数が 1/2 になる．
v0.22.1 で1パス統合を試みたが proof の early return 喪失で 7% 悪化した
経緯がある(§6.6.2 実施済み最適化)．2パス→1パスの再設計が必要．

**E8: SNDA pairs の事前確保**

`snda_pairs: Vec<(u64, u32)>` が mid() 呼び出しごとにスタック上で
宣言される．`DfPnSolver` のフィールドに移動して事前確保(capacity=16)
すればヒープ再確保を回避できる．

**E9: child\_init lookup キャッシュ**

child\_init の do\_move → `look_up_pn_dn` → undo\_move の後に
ci\_resolve で再度 `look_up_pn_dn` を呼んでいる箇所がある．
do\_move/undo\_move は TT を変更しないため，最初の lookup 結果を
キャッシュすれば再 lookup を省略できる．

**E10: collect の即時 proof/disproof 脱出**

OR ノードの collect で cpn=0 の子が見つかった時点で即座に OR proof を
store して return できる(SNDA 計算を省略)．AND ノードでも cdn=0 で同様．

**E11: look\_up\_pn\_dn に best\_move 統合**

`look_up_best_move` が別途クラスタスキャンを行っている．
`look_up_pn_dn` の返り値に best\_move を含めれば1回のスキャンで済む．

#### 6.6.3 Dual TT + hand\_hash 混合 (v0.24.0)

v0.24.0 でクラスタ飽和問題(§6.6.1)の構造的解決を目指し，
単一の TranspositionTable を **ProvenTT + WorkingTT** に分離し，
ProvenTT に hand\_hash 混合インデクシングを導入した．

**構成:**

| テーブル | 内容 | クラスタ | エントリ/クラスタ | インデクシング | メモリ |
|---------|------|--------|-----------------|-------------|--------|
| ProvenTT | proof (pn=0) + confirmed disproof | 2M | 4 | pos\_key XOR hand\_hash | 256 MB |
| WorkingTT | intermediate + depth-limited disproof | 2M | 6 | pos\_key | 384 MB |
| 合計 | — | — | — | — | 640 MB |

**注:** v0.24.14 で構成が更新された．最新構成は §6.6.4 を参照．

ProvenTT は `pos_key ^ hand_hash(hand)` でクラスタを特定し，
hand バリアントを異なるクラスタに分散する:

```rust
fn hand_hash(hand: &[u8; 7]) -> u64 {
    u64::from_le_bytes([hand[0], ..., hand[6], 0])
        .wrapping_mul(0x9E3779B97F4A7C15)
}
fn proven_cluster_start(pos_key, hand) -> usize {
    (pos_key ^ hand_hash(hand)) & mask
}
```

**注:** v0.24.7 で hand\_hash は Zobrist ベースに変更された(§6.6.4)．

WorkingTT は pos\_key ベースインデクシングを維持(hand\_gte disproof 再利用のため)．

**注:** v0.24.6 で WorkingTT も hand\_hash 混合に変更された(§6.6.4)．

**エントリ圧縮:**

DfPnEntry を 32→24 bytes に圧縮し TTFlatEntry (pos\_key u64 + DfPnEntry) は 40→32 bytes:

| フィールド | 変更 | 節約 |
|-----------|------|------|
| source | u64→u32 (SNDA ハッシュ切り捨て) | -4B |
| amount | u16→u8 (0-255，PROOF\_BONUS=100 が収まる) | -1B |
| path\_dependent + remaining | remaining\_flags: u16 に pack (bit15 + bits0-14) | -1B |

REMAINING\_INFINITE: u16::MAX → 0x7FFF (15 ビットの最大値，depth 0-127 に十分)．
snda\_dedup の pairs: (u64, u32) → (u32, u32)．

**2段階検索:**

ProvenTT と WorkingTT を独立した検索メソッドに分離:
- `look_up_proven(pos_key, hand, remaining)`: ProvenTT のみ
- `look_up_working(pos_key, hand, remaining)`: WorkingTT のみ
- `look_up(...)`: 統合ラッパー(proven → working fallback)

**GC 戦略:**

| メソッド | ProvenTT | WorkingTT | 呼び出しタイミング |
|---------|---------|-----------|----------------|
| retain\_proofs() | そのまま | confirmed disproof 保持，他は除去 | Frontier サイクル間 |
| retain\_proofs\_only() | そのまま | 全クリア | PNS→MID fallback 境界 |
| clear\_working() | そのまま | 全クリア | IDS depth 切り替え |
| clear\_proven\_disproofs() | confirmed disproof 除去 | そのまま | IDS depth 切り替え |
| gc\_by\_amount() | そのまま | amount ベース除去 | Periodic GC (1M ノード毎) |

**注:** v0.24.10 以降，`gc_by_amount()` は KomoringHeights 方式の
`gc_working_sampling()` に置き換えられた(§6.6.4)．
v0.24.14 では rem=0 disproof が TT に格納されなくなったため，
GC 対象の TT 構成が根本的に変化している．

`retain_proofs()` は Frontier サイクルで呼ばれ，
WorkingTT の confirmed disproof (!path\_dep, REMAINING\_INFINITE) を保持しつつ
中間エントリを除去する(段階的クリア)．

IDS depth 切り替え時は `clear_working()` + `clear_proven_disproofs()` で
構造的不詰エントリの汚染を防止する(NoMate バグ対策)．

**ProvenTT 置換ポリシー:**

ply ベースの amount: `proof.amount = 255 - ply`，`disproof.amount = 128 - ply`．
ルートに近い proof ほど高い amount を持ち，eviction 耐性が高い．
replace\_weakest\_proven は lowest amount を evict し，
新エントリの amount が既存最弱以上の場合のみ置換する．

**祖先チェックによる ProvenTT 挿入スキップ:**

proof (pn=0) を store する際，`path` 配列を遡り祖先に proof が存在すれば
挿入をスキップする．祖先の証明が子の証明を包含するため，
ProvenTT が恒常的にスリムに保たれる．

**NoMate バグ対策:**

IDS の浅い depth で格納された confirmed disproof が深い depth で
偽の不詰判定を引き起こす問題(39手詰め ply 22 で depth=21→NoMate)を
`clear_proven_disproofs()` の IDS 切替時呼び出しで解決．

**ベンチマーク (39手詰め root, depth=41):**

| 構成 | 予算 | proven\_overflow | working\_overflow | NPS |
|------|------|----------------|-------------------|-----|
| v0.24.0 初期(pos\_key ProvenTT, 全クリア retain) | 50M | 12.0M | 5.1M | 183K |
| **最終構成(hand\_hash + 段階retain + 祖先チェック)** | **50M** | **44.5K (-99.6%)** | **4.3M (-16%)** | **244K (+33%)** |
| v0.24.0 初期 | 100M | 47.0M | 8.3M | 244K |
| **最終構成** | **100M** | **3.0M (-94%)** | **8.9M (+7%)** | **260K (+7%)** |

**不採用とした改善案:**

| 対象 | 案 | 結果 | 不採用理由 |
|------|---|------|-----------|
| ProvenTT | CLUSTER\_SIZE=6 | overflow -29〜54% | WorkingTT overflow +63〜133%, NPS -14% |
| ProvenTT | 4M clusters | overflow -32〜40% | WorkingTT overflow +61〜69%, NPS -16% |
| ProvenTT | full\_hash インデクシング | 多数テスト失敗 | look\_up 時に full\_hash が必要で API 変更が困難 |
| WorkingTT | CLUSTER\_SIZE=8 | overflow 悪化(+19%) | 改善なし |
| WorkingTT | hand\_hash 混合 | overflow 悪化(+33%) | hand\_gte disproof 再利用の喪失で NPS -17%．**v0.24.6 で Zobrist 差分近傍走査と組み合わせて実装(§6.6.4)** |
| WorkingTT | 2-way set associative | overflow -5〜15% | secondary スキャンで NPS -18〜21% |
| GC | 段階 retain(クラスタ飽和残存時) | NPS -13% | 保持 disproof がクラスタ圧迫 |

**注:** v0.24.6 以降，WorkingTT も hand\_hash 混合インデクシングに変更された．
Zobrist XOR 差分による近傍クラスタ走査(§6.6.4)により hand\_gte の機能を
回復させたため，v0.24.0 時点で不採用だった hand\_hash 混合が実用的になった．

#### 6.6.4 Zobrist hand\_hash + 近傍クラスタ走査 + GC 刷新 (v0.24.2〜v0.24.14)

v0.24.2〜v0.24.14 で TT のインデクシング，hand\_gte の正確性，GC 戦略を
包括的に改善した．

**課題:**

1. PV 復元時の proof 見逃し: ProvenTT の hand\_hash 混合クラスタリングにより，
   証明駒と検索時の持ち駒が異なるとクラスタが一致せず proof を発見できない
2. hand\_gte の機能低下: hand\_hash インデクシングの導入で，
   `hand_gte_forward_chain` が自クラスタ内でしか機能しなくなった
3. WorkingTT の disproof 飽和: depth-limited disproof(特に rem=0)が
   クラスタの 64.7% を占め，intermediate の空きがなくなる

**改善 1: Zobrist hand\_hash (v0.24.7)**

hand\_hash を golden ratio multiplication から Zobrist hash に変更:

```rust
fn hand_hash(hand: &[u8; HAND_KINDS]) -> u64 {
    let mut h = 0u64;
    for k in 0..HAND_KINDS {
        h ^= ZOBRIST.hand_hash(Color::Black, k, hand[k] as usize);
    }
    h
}
```

利点:
- ランダムキーによる良好なハッシュ分散(Max entries/pos: 6→1)
- XOR 差分で持ち駒 1 枚の増減のクラスタ位置を O(1) 計算可能:
  `new_hash = old_hash ^ hand_hash_diff(k, old_val, new_val)`

**改善 2: ProvenTT クラスタサイズ 8 (v0.24.3)**

`PROVEN_CLUSTER_SIZE` を 4→8 に拡大．
持ち駒の種類数(7)+1=8 とすることで，同一盤面の持ち駒バリアントの
proof/confirmed disproof がクラスタに収まりやすくなる．
ProvenTT overflow: 44,535→0 に完全解消．

**改善 3: WorkingTT hand\_hash 混合インデクシング (v0.24.6)**

WorkingTT のクラスタキーを `pos_key` → `pos_key ^ hand_hash(hand)` に変更．
合駒を取った後の OR ノードで持ち駒バリアント(最大 32+)が
同一クラスタに集中する問題を解消．Working overflow: 64% 削減．

**改善 4: Zobrist XOR 差分による近傍クラスタ走査 (v0.24.8)**

hand\_hash インデクシングで hand\_gte が自クラスタ内でしか機能しない問題を解決．
Zobrist hash の XOR 差分特性を利用し，持ち駒 ±1 のクラスタを効率的に走査:

- proof 検索 (hand ≥ e.hand): 持ち駒 -1 の近傍クラスタ走査
- disproof 検索 (e.hand ≥ hand): 持ち駒 +1 の近傍クラスタ走査

`neighbor_scan` フラグで制御:
- `false` (デフォルト): 自クラスタのみ(探索ホットパス)
- `true`: ±1 近傍クラスタ追加走査(PV 復元時，合駒チェーン時)

適用メソッド: `look_up_proven`, `look_up_working`, `has_proof`,
`get_proof_hand`, `has_path_dependent_disproof`,
`get_disproof_remaining`, `get_effective_disproof_info`

PV 復元用の `look_up_proven_subset` は Zobrist 差分で全部分集合を走査．

**改善 5: サンプリング GC (v0.24.10, KomoringHeights 参考)**

KomoringHeights (`github.com/komori-n/KomoringHeights`) の GC 方式を参考に
WorkingTT の GC をサンプリング+閾値+CutAmount に刷新:

1. ストライドサンプリング(10,000 エントリ)で amount 分布を収集
2. `select_nth_unstable` で下位 20% の除去閾値を決定
3. 証明/反証済み局面の obsolete intermediate を ProvenTT 参照で除去
4. 閾値以下の disproof を除去(intermediate は保護)
5. CutAmount: `max_amount > 32` なら生存エントリの amount を半減

intermediate 保護が重要: intermediate は探索の進捗(pn/dn 中間値)を保持しており，
除去すると再計算コストで探索が崩壊する(29手詰め no\_pns のリグレッションで確認)．

overflow GC は obsolete intermediate 除去のみ(disproof は保護):
disproof を除去すると再訪問で再生成→再overflow→再GC のサイクルに陥る．

**改善 6: rem=0 仮反証の TT store 廃止 (v0.24.14)**

depth-limit 到達時の仮反証(remaining=0)を TT に格納せず，
`look_up_pn_dn` で動的に判定する:

```rust
fn look_up_pn_dn_impl(..., remaining: u16, ...) -> (u32, u32, u32) {
    if remaining == 0 {
        if self.table.has_proof(pos_key, hand) { return (0, INF, 0); }
        return (INF, 0, 0);
    }
    // 通常の TT lookup
}
```

rem=0 disproof の特性:
- クラスタの 64.7% を占め overflow の主因
- 同じ IDS depth の同じ深さでしか参照されない(`remaining()>=remaining` は rem=0 のみマッチ)
- 他の探索パスでは利用されない

child init の depth-limit fast path も TT store をスキップし，
ローカル変数 `cpn`/`cdn` で解決チェックを行う(TT 再 lookup をバイパス)．

**構成 (v0.24.14):**

| テーブル | 内容 | クラスタ | エントリ/クラスタ | インデクシング | メモリ |
|---------|------|--------|-----------------|-------------|--------|
| ProvenTT | proof (pn=0) + confirmed disproof | 2M | 8 | pos\_key XOR Zobrist hand\_hash | 512 MB |
| WorkingTT | intermediate + depth-limited disproof (rem>0) | 2M | 6 | pos\_key XOR Zobrist hand\_hash | 384 MB |
| 合計 | — | — | — | — | 896 MB |

**ベンチマーク (39手詰め root, depth=41, 50M ノード):**

| 構成 | Working overflow | NPS | 備考 |
|------|-----------------|-----|------|
| v0.24.0 最終(pos\_key WorkingTT) | 4,300,000 | 244K | §6.6.3 ベースライン |
| + Zobrist hand\_hash 両 TT | 1,228,000 | 200K | -71% |
| + ±1 近傍走査 + neighbor\_scan | 1,097,000 | 150K | hand\_gte 正確性向上 |
| + サンプリング GC + intermediate 保護 | 993,000 | 176K | |
| + overflow GC で disproof 保護 | 4,332,000 | 107K | 29手詰め no\_pns 修復 |
| **+ rem=0 store 廃止** | **170,000** | **219K** | **-96%, NPS +10%** |

---

