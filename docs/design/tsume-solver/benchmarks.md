# 既知の課題とベンチマーク

### 10.1 29手詰め問題

```
SFEN: l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1
```

**テスト:** `test_tsume_6_29te` (tests.rs), `test_tsume_6_29te_no_pns` (tests.rs)

| 構成 | ノード数 | 結果 |
|------|---------|------|
| PNS + IDS | ~18.5M | 29手詰め (正解) |
| IDS のみ (PNS なし) | ~18.5M | 29手詰め (正解) |

29手詰めは **PNS なしに IDS のみで解ける**ことが確認されている．
これは IDS-MID 単体のロバストネスを示す重要なベンチマーク．

**v0.24.14 での状態:** PNS + IDS，IDS のみ共に通過確認済み．
TT 改善(サンプリング GC + intermediate 保護 + 探索パス保護)により，
overflow GC が有用エントリを破壊する問題を解消し安定的に解ける．

### 10.2 39手詰め問題

```
SFEN: 9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1
```

**テスト:** `test_tsume_39te_aigoma` (tests.rs, `#[ignore]`),
`test_tsume_39te_ply22_no_pns` (tests.rs, `#[ignore]`),
`test_tsume_39te_backward_{1m,10m,120m,500m}` (tests.rs, `#[ignore]`),
`test_tsume_39te_profile_{depth_scan,ply20_timeline}` (tests.rs, `#[ignore]`, v0.24.32+),
`test_tsume_39te_ply25_gap_diagnosis` (tests.rs, `#[ignore]`, v0.24.44+),
`test_tsume_39te_ply24_mate15_regression` (tests.rs, 非 ignore 回帰テスト)

---

**現状 (v0.24.44):** ルートからの完全解法は依然未達成．
backward 解析の境界は v0.24.33 で確立した **ply 14 (Mate(25))** (120M 予算) /
**ply 12 (残り 27 手)** (500M 予算) のまま．v0.24.44 では境界自体は前進しなかったが，
`test_tsume_39te_ply25_gap_diagnosis` による定量診断で **gap の真の原因が
4 層の構造** として特定された (詳細: §10.2 末尾「v0.24.44 ply25 gap 診断」参照):

1. **IDS depth 切替時の intermediate エントリ全消去**: `clear_working()` により
   浅い depth で積み上げた中間進捗 (実測 75K エントリ) が次 IDS step で失われる．
   合駒チェーンの不詰部分証明が毎 step 再構築される
2. **`cross_deduce` (§8.5 同一マス証明転用) が実質機能していない**: depth=17 でも
   `cross_deduce_hits = 0` を観測．§8 の合駒最適化群が期待通り発火していない
3. **depth inflation cliff の前進**: 同じ ply 24 根局面で depth=17 なら 367K で
   解けるが depth=21 では 6M budget で Unknown．v0.24.0 時点の「depth=23 が壁」
   という認識から，現状は depth 19〜21 で破綻する
4. **個別サブ問題の難度**: 23 応手中 **3 件 (L\*6g, B\*6g, B\*7g) が 500K budget で
   Unknown**．v0.22.0 時点の「全て 200K 以内で解ける」という記述は現状成立しない

これを受けて施策優先順位を組み替え，**最優先を「IDS depth 切替時の intermediate
選択的保持」 (新規)**，次点を「cross_deduce 修復 + 駒種横断 disproof 共有」
とした．詳細は §10.2 末尾の該当セクションを参照．

**§10.2 の読み方 (時系列ナビゲーション):**

| 節 | バージョン | 内容 |
|:---:|:---|:---|
| 問題の構造 | 共通 | チェーン合駒の分岐構造 |
| backward 解析 / 閾値飢餓 | v0.20.34〜v0.22.0 | 初期の律速要因 (閾値飢餓 → TT 飽和) |
| Dual TT 調査 | v0.24.0 | Dual TT の効果と残存課題 |
| TT 改善後調査 | v0.24.14〜v0.24.16 | 近傍走査と WorkingTT 改善 |
| 39手詰め最新調査 | v0.24.27〜v0.24.33 | Plan D + PNS NM 昇格判定修正 |
| **v0.24.44 ply25 gap 診断** | **v0.24.44** | **現行課題の 4 層構造と施策再優先化** |

---

**前状態 (v0.24.33):** PNS の NM 昇格判定を TT ベース + memoize のハイブリッド
(`refutable_check_with_cache`) に置き換え，従来 PNS 時間の 99.97% を消費していた
`depth_limit_all_checks_refutable` の律速要因を除去した．この 1 コミットで
backward_120m の境界が **ply 20 (Mate(19))** から **ply 14 (Mate(25))** へ
4 ply も前進し，ply 16 (Mate(23)) までが通常 `solve()` 経路で解けるようになった．
詳細は後述 §10.2 末尾「v0.24.33 PNS NM 昇格判定の非対称性解消」参照．

**前状態 (v0.24.27〜v0.24.29):** ルートからの統合探索は依然未解決だが，サブ問題の解決範囲は
大きく改善された．`test_tsume_39te_ply22_no_pns` (mid\_fallback 直接呼び出し，
120M ノード予算) で **ply 22 (Mate(17)) を 104〜107s で解ける**ようになった．
通常 `solve()` 経路でも 120M 予算で ply 22 まで解け，境界は ply 20 (Mate(19))
に移動していた．詳細は後述 §10.2 末尾「39手詰め最新調査 (v0.24.27)」参照．

**歴史的経緯 (v0.22.0):** `test_tsume_39te_aigoma` (10M ノード / 60s) では UNKNOWN．
backward 解析で ply 24 サブ問題は 333K で Mate(21) に解けるが，
ply 22 (残り 17 手) は 1M ノードで Unknown．

#### 問題の構造

39手詰め問題はチェーン合駒最適化(§8)のメインターゲットである．
ply 24 で銀5g→6fの開き王手(飛車8gの横利き開放)が発生し，
飛車(8g)と玉(1g)の間の5マス(7g, 6g, 5g, 4g, 3g)に
チェーン合駒構造が出現する．

ply 25 AND ノードの 23応手を個別に探索した結果 (**v0.22.0 時点**):

| カテゴリ | 応手 | ノード数 | 結果 |
|---------|------|---------|------|
| 即詰み | P/L/N/S/G/B\*2g | 各1 | Mate(1) |
| 3〜5筋合駒 | P\*3g 〜 N\*5g (9手) | 41〜80K | Mate(7〜15) |
| 6〜7筋合駒 | L\*6g, B\*6g, B\*7g, N\*6g | 71K〜83K | Mate(13〜15) |
| 玉逃げ | 1g1f | 140K | Mate(19) |
| PV(最長) | 1g1h | 103 | Mate(13) |
| 不詰 | P\*7g, N\*7g | 各~200K | NoMate |

v0.22.0 当時は「すべて 200K ノード以内で解け，23 応手の合計は約 **1.5M ノード**
に収まる」と評価されていた．

> **⚠️ 注記 (v0.24.44 再測定):** `test_tsume_39te_ply25_gap_diagnosis`
> Phase 0b で 500K budget / 15s per response の fresh solver で再測定した
> ところ，**3 応手 (L\*6g, B\*6g, B\*7g) が 500K budget で Unknown となった**．
> 個別合計は Unknown 3 件を含めて 405K ノード程度であり，「全て 200K 以内」
> という v0.22.0 の評価は現状必ずしも成立しない．個別サブ問題ウォームアップ
> を基礎とした施策は再評価が必要である．
> 詳細は §10.2 末尾「v0.24.44 ply25 gap 診断」の Phase 0b 実測表参照．

#### backward 解析

PV を逆順にたどり，各偶数 ply(攻め方手番 = OR ノード)を個別に解いた結果
(1M ノード / 180 秒)．バージョン間の推移を示す:

| Ply | 残り手数 | v0.20.35 ノード | v0.20.35 結果 | v0.22.0 ノード | v0.22.0 結果 |
|-----|---------|----------------|-------------|---------------|------------|
| 38 | 1 | 14 | Mate(1) | 14 | Mate(1) |
| 26 | 13 | 103 | Mate(13) | 100 | Mate(13) |
| **24** | **15** | **396,636** | **Mate(21)** | **332,630** | **Mate(21)** |
| **22** | **17** | **1,000,000** | **Unknown** | **1,000,000** | **Unknown** |

ply 24: v0.20.35 の 397K → v0.22.0 の 333K に **16% 改善**．
ply 22: TT エントリが 574K → 916K に成長(探索範囲拡大)するが 1M では未解決．

v0.21.0 では自然精度 epsilon の導入で ply 24 が一時退行(1M Unknown)したが，
v0.22.0 で heuristic 拡張 + epsilon /3 + pn\_floor 2/3 + TT 改善により回復・改善．

#### 閾値飢餓と TT クラスタ飽和

統合探索(ルート→ply 24)で MID が深い ply に到達できない問題．
v0.20.34〜v0.22.0 の調査で，課題の性質が「閾値飢餓」から
「TT クラスタ飽和」に変化したことが判明した．

**v0.20.34 での課題(閾値飢餓):**
`heuristic_or_pn` が S〜3S (= 1〜3) と小さく，MID only 50M→100M で
TT エントリが完全同一(完全停滞)．原因は PN\_UNIT 統一スケーリング(§3.5)の
未導入によるスケーリング漏れ(§3.5 参照)．

**v0.22.0 での課題(TT クラスタ飽和):**
heuristic\_or\_pn を S〜8S (= 16〜128，KomoringHeights の pn=10-80 相当)に拡張，
epsilon を /3 に増加，pn\_floor を 2/3 に引き上げた結果，
**閾値伝搬自体は大幅に改善された**:

- backward 解析 ply 24: 397K → 333K (16% 改善)
- 29 手詰め: **74.2M ノードで解決**(v0.20.36 では解決不能)
- ply 24 の個別応手: N\*7g NoMate 200K → 82K

しかし 39 手詰めの統合探索は v0.22.0 時点で依然 UNKNOWN であった
(v0.24.27 現在は ply 22 サブ問題までは解決可能 — 本節末尾参照)．
29 手詰めの探索停滞の診断(v0.21.1)で，停滞の原因が閾値飢餓ではなく
**TT クラスタ飽和**(§6.6.1)であることが特定された:

- 29 手詰めの ply 26 で pn\_th=373 (23S) — **閾値は十分**
- 停滞の原因: NM の store 失敗(foreign protected がクラスタ占有)→ pn/dn 不変
  → 1+ε の予算増加メカニズムが回らない

29 手詰めは TT クラスタ飽和対策(§6.6.1)により解決．
39 手詰めではクラスタ飽和に加え探索空間の深さも課題．

**39 手詰めが未解決の理由 (v0.24.44 再整理):**

v0.24.44 の `test_tsume_39te_ply25_gap_diagnosis` により原因が
**4 層の構造**として明確化された (実測データは §10.2 末尾参照):

1. **IDS depth 切替時の intermediate エントリ全消去** (最優先課題):
   `clear_working()` が WorkingTT を全消去するため，浅い IDS step で
   積み上げた中間進捗 (depth=17 で 75K エントリ実測) が次 step で失われ，
   合駒チェーンの不詰部分証明が毎 step ゼロから再構築される
2. **合駒最適化の機能停止**: `cross_deduce_hits = 0` (depth=17 でも)，
   `prefilter_hits = 17〜45`，`capture_tt_hits = 0 (depth≥21)` など，
   §8 の合駒最適化群が実運用で期待通り発火していない
3. **depth inflation cliff の前進**: 同じ ply 24 根局面で depth=17 なら
   367K で解けるが depth=21 では 6M budget で Unknown．v0.24.0 時点の
   「depth=23 が壁」という認識から，現状 depth 19〜21 で破綻する
4. **個別サブ問題自体の難度**: 23 応手中 3 件 (L\*6g, B\*6g, B\*7g) が
   500K budget で Unknown となる．サブ問題ウォームアップによる統合探索
   高速化は単体では十分な効果が見込めない

**旧認識からの変遷:**

| 旧認識 (〜v0.22.0) | 現状 (v0.24.44) | 状態 |
|-------------------|-----------------|:---:|
| TT クラスタ飽和 | §6.6.4 の改善で overflow 大幅削減 | **解消** |
| ノード予算不足 | 500M+ で ply 14 到達 | **緩和** |
| 探索空間の深さ | depth inflation として再理解 | **残存** (層 3) |
| — | intermediate 全消去 | **新規** (層 1，最優先) |
| — | 合駒最適化の機能停止 | **新規** (層 2) |
| — | サブ問題自体の難度 | **新規** (層 4) |

従来の「TT クラスタ飽和」「ノード予算」という二大律速要因は，
v0.22.0〜v0.24.16 の §6.6 系改善でいずれも緩和されており，
現状の律速要因は **intermediate 情報の IDS step 間非保持** と
**合駒最適化の実質非発火** に移っている．

**採用済み手法の一覧 (主要項目):**

下表は 39 手詰め問題に寄与した主要施策の抜粋．完全な時系列は
[index.md](index.md) §1 の実装済み手法一覧 (73 項目) を参照．

| 手法 | 版 | 効果 |
|------|-----|------|
| PN\_UNIT=16 統一スケーリング (§3.5) | v0.20.36 | スケーリング基盤整備 |
| 自然精度 epsilon /3 (§3.1) | v0.21.1 | 各 OR レベル ~33% マージン |
| heuristic\_or\_pn S-8S 2D (§5.1) | v0.21.1 | KomoringHeights 相当の初期 pn |
| pn\_floor 2/3 (§3.3) | v0.21.1 | AND カスケード減衰 67%/level |
| TT 2M クラスタ + amount 置換 (§6.6) | v0.22.0 | クラスタ飽和緩和 |
| single-child / MID 停滞検出 | v0.21.1 | 停滞の対症療法 |
| IDS 動的予算配分 (§2.6) | v0.21.0 | MID 停滞→Frontier 早期移行 |
| path スタック化 (§6.6.2 E5) | v0.23.0 | FxHashSet→配列，ハッシュ計算排除 |
| ci\_resolve 再 lookup 廃止 (§6.6.2 E1) | v0.23.0 | has\_proof で Pass 2/3 省略 |
| 王手生成キャッシュ (§6.6.2 E2) | v0.23.0 | 8192 エントリ direct-mapped |
| 玉移動合法性高速化 | v0.23.0 | do\_move/undo\_move→is\_attacked\_by\_excluding |
| pn\_floor オーバーフロー修正 (§3.3) | v0.23.0 | u64 昇格で 23 テスト失敗を修正 |
| Dual TT (ProvenTT + WorkingTT) (§6.6.3) | v0.24.0 | proof と intermediate の分離管理 |
| ProvenTT hand\_hash 混合 (§6.6.3) | v0.24.0 | overflow -99.6% |
| Zobrist hand\_hash indexing (§6.6.4) | v0.24.7 | クラスタ衝突緩和 |
| Zobrist XOR 差分近傍走査 (§6.6.4) | v0.24.8 | 合駒駒種変化の TT 再利用 |
| サンプリング GC (§6.6.4) | v0.24.10 | 探索パス保護 + obsolete 除去 |
| rem=0 仮反証 store 廃止 (§6.6.4) | v0.24.14 | WorkingTT overflow -96% |
| proof(-1) + 歩 disproof(+1) 近傍走査 (§6.6.4) | v0.24.16 | hand 多様性への対応 |
| Plan D ProvenEntry 分離 (§6.6.5) | v0.24.26 | TT エントリレイアウト再設計 |
| WorkingTT cluster 6→8 (§6.6.5) | v0.24.27 | eviction thrashing 解消，ply22 -34% time |
| **PNS NM 昇格判定ハイブリッド化** (§10.2 末尾) | **v0.24.33** | **`depth_limit_all_checks_refutable` 律速要因解消，境界 ply 20→14** |
| Frontier PNS 予算の proof rate ベース制御 (§10.2) | v0.24.37 | 生産的サイクルでの予算拡大 |
| IDS ProvenTT disproof 選択的保持 (§6.6.3) | v0.24.38 | `clear_proven_disproofs_below(min_depth)` |
| IDS depth 32+ で +4 刻み (§2.3) | v0.24.40 | 深い問題での段階的深化 |
| depth-adaptive epsilon (§3.1) | v0.24.41 | saved\_depth ≥ 19 で eps\_denom=2 |
| PV visit 予算の動的スケーリング (§10.2) | v0.24.42 | MateNoPV 回避 |
| GC Phase 2 no-op バグ修正 (§6.6) | v0.24.43 | `clear_proven_disproofs_below(u32::MAX)` |
| **ply25 gap 診断テストの追加** (§10.2 末尾) | **v0.24.44** | **gap の 4 層構造を特定，施策優先順位の組み替え** |

#### 今後の改善方針 (v0.22.0 時点，後続更新あり)

> **⚠️ 注記**: 本節は v0.22.0 時点の改善方針であり，その後 v0.24.0〜v0.24.43
> で多くが実装・評価されている．**現行の施策優先順位は §10.2 末尾
> 「v0.24.44 ply25 gap 診断と施策優先順位の組み替え」を参照**のこと．

**方針 A: 閾値伝搬の改善 — 採用(v0.21.0-v0.21.1)**

PN\_UNIT=16 統一スケーリング(§3.5)を基盤に，3 つの改善を実施:
1. **自然精度 epsilon /3 (§3.1)**: 各 OR/AND レベルで ~33% の閾値マージン
2. **heuristic\_or\_pn S-8S 2D (§5.1)**: safe\_escapes × num\_checks の二次元スケーリング
3. **pn\_floor 2/3 (§3.3)**: AND カスケード減衰を (1/2)^N → (2/3)^N に改善

これにより閾値伝搬は大幅改善し 29 手詰め(74.2M nodes)の解決に至った．

**方針 B: IDS フルデプスの動的予算配分 — 採用(v0.21.0)**

MID を 1M ノード固定チャンクに分割し TT 成長を監視．
停滞検出で Frontier に早期移行する(§2.6 参照)．

#### v0.22.0 のベンチマーク
8. **MID チャンク 1M 固定**: 停滞の早期検出のため 15M→1M に縮小．
9. **Periodic GC 容量ベース化**: 50M/60M のハードコード → capacity×80%/90% に自動適応．

**backward 解析 (v0.22.0, 1M ノード/180秒):**

| Ply | 残り手数 | ノード数 | TT エントリ | 結果 | v0.20.35 比 |
|-----|---------|---------|-----------|------|------------|
| 38 | 1 | 14 | 39 | Mate(1) | 同一 |
| 28 | 11 | 87 | 185 | Mate(11) | 89→87 |
| 26 | 13 | 100 | 189 | Mate(13) | 103→100 |
| **24** | **15** | **332,630** | **277,660** | **Mate(21)** | **397K→333K (改善)** |
| **22** | **17** | **1,000,000** | **915,557** | **Unknown** | 574K→916K (TT 成長) |

ply 24 サブ問題が v0.20.35 の 397K から 333K に改善．
TT エントリは 344K→278K に圧縮されつつ Mate(21) を発見．

**backward 解析 (v0.24.0 Dual TT, 1M ノード/180秒):**

| Ply | 残り手数 | ノード数 | TT total | proven | disproven | intermediate | 結果 |
|-----|---------|---------|----------|--------|-----------|-------------|------|
| 38 | 1 | 14 | 39 | 2 | 24 | 13 | Mate(1) |
| 36 | 3 | 2 | 8 | 4 | 0 | 4 | Mate(3) |
| 34 | 5 | 12 | 36 | 12 | 2 | 22 | Mate(5) |
| 32 | 7 | 17 | 46 | 14 | 2 | 30 | Mate(7) |
| 30 | 9 | 22 | 56 | 16 | 6 | 34 | Mate(9) |
| 28 | 11 | 87 | 185 | 51 | 16 | 118 | Mate(11) |
| 26 | 13 | 100 | 189 | 53 | 20 | 116 | Mate(13) |
| **24** | **15** | **317,327** | **321,703** | **5,842** | **266,712** | **49,149** | **Mate(21)** |
| **22** | **17** | **1,000,000** | **29,452** | **15,610** | **13,842** | **0** | **Unknown** |

**TT エントリ増加パターンの分析:**

ply 38→26: TT エントリは線形的に増加(39→189，残り手数に比例)．
**ply 24 で 189→321K に急増(約 1,700 倍)**し，指数的な増加が発生している．

ply 24 の TT 組成を見ると **disproven=267K が全体の 83%** を占める．
これは合駒チェーン(ply 24 は `P*1g` 後の局面)における
玉方の逃げ先の不詰証明が指数的に分岐していることを示す:

- 合駒の種類(歩・香・桂・銀・金・角・飛) × 逃げ先の分岐
  → 各バリアントの不詰を個別に証明する必要がある
- disproven エントリ 1 件は「この持ち駒・この残り深さで不詰」を表すため，
  持ち駒バリアント × 深さ制限のバリエーションで急増する

ply 22 では 1M ノード制限に到達し，**intermediate=0**(GC で全除去済み)．
ProvenTT に proven=15.6K が残っているが，探索を進めるための
intermediate エントリが WorkingTT から全て GC されており行き詰まっている．

**v0.22.0 との比較:**

| Ply | v0.22.0 TT | v0.24.0 TT total | v0.24.0 proven | 変化 |
|-----|-----------|-----------------|----------------|------|
| 24 | 277,660 | 321,703 | 5,842 | TT +16% (disproven 増加) |
| 22 | 915,557 | 29,452 | 15,610 | TT -97% (GC 後の残留) |

v0.24.0 では Dual TT の Frontier サイクル間 `retain_proofs()` が
WorkingTT の中間エントリを除去(confirmed disproof は保持)するため，
ply 22 で GC 後の残留エントリが大幅に減少(916K→29K)している．
v0.22.0 では GC 後も intermediate の一部が残っていたが，
v0.24.0 では ProvenTT のエントリのみが残り，WorkingTT は完全に空になる．

**指数的増加の境界:**

```
ply 26→24: ×1,700 (189→321K) — 合駒チェーン分岐の影響
ply 24→22: 1M ノード制限で打ち切り — 必要ノード数は数百万以上と推定
```

39 手詰め全体を解くには ply 22 以前の局面も証明する必要があり，
合駒分岐による指数的な TT エントリ増加が律速要因である．

**29 手詰めベンチマーク (v0.22.0, depth=31):**

| 構成 | モード | ノード | 時間 | TT max | 結果 |
|------|--------|--------|------|--------|------|
| v0.20.36 (1M clusters) | no\_pns | >120M | — | 3.1M | UNKNOWN |
| v0.22.0 (1M clusters) | no\_pns | 115M | 411s | 3.1M | Mate(29) |
| **v0.22.0 (2M clusters)** | **no\_pns** | **74.2M** | **287s** | **6.3M** | **Mate(29)** |
| v0.22.0 (4M clusters) | no\_pns | 109M | 556s | 12.5M | Mate(29) |
| v0.22.0 (2M clusters) | PNS あり | — | <310s | 6.3M | Mate(29) |

1M clusters は amount ベース置換(v0.22.0)で解決可能になったが 2M より遅い．
2M が最適: キャッシュ効率(480MB)と TT 容量(6.3M)のバランス．
PNS あり版は 50M ノード / 300s 以内で解決(test\_tsume\_6\_29te)．

**探索停滞の診断で特定された問題と修正:**

| 問題 | 症状 | 原因 | 修正 |
|------|------|------|------|
| sc\_loop\_hang | single-child ループ無限反復 | pn/dn 不変でも脱出条件なし | SC\_STAGNATION\_LIMIT=4 |
| ply 31 集中 | 93% のノードが depth boundary | nodes\_used>1 で停滞リセット | 常に pn/dn 変化を確認 |
| TT 凍結 | 新規 proof/disproof 挿入ゼロ | foreign protected がクラスタ占有 | replace\_weakest\_for\_disproof |
| TT ハッシュ衝突 | 全体 52% なのに飽和 | 1M クラスタでの衝突限界 | 2M クラスタ化 |

**方針 C: Frontier Variant の軽量化 — 採用(v0.22.1: アリーナ再利用)**

`frontier_variant()` が `pns_main()` を最大 50 回呼び出す際，毎回の
`Vec<PnsNode>` の allocate/deallocate を回避するため，アリーナを
`frontier_variant()` のスコープで 1 回確保し，`pns_main_with_arena()` で
再利用する方式に変更．`pns_main()` は既存の呼び出し元向けラッパーとして保持．

ベンチマーク(39 手詰め，同条件比較):
- baseline: ~74K NPS (4.5M nodes / 60s)
- v0.22.1: ~89K NPS (5.3M nodes / 60s)，**NPS +20% 改善**

deferred\_drops の ArrayVec 化は合駒数が容量 8 を超えるケースがあり不採用．

**方針 D: TT 構造の改善 — 部分採用(v0.22.0: amount ベース置換)，リニアプロービングは不採用**

§6.6.1 で分析した通り，クラスタ方式は NPS で優位(~253K-868K)だが
実効容量が全スロット数の約 52% に制限される構造的課題がある．
この課題の根本解決を目指し，v0.22.1 でリニアプロービングを実装・評価した．

**リニアプロービング実装の構成:**
- 8M エントリ(= 320 MB，クラスタ 2M×6 の 480 MB より省メモリ)
- tombstone 方式: 削除時にスロットを tombstone マーカーで埋め，lookup 時にスキップ
- MAX\_PROBE=128: 最大プローブ距離(これを超えると挿入を断念)
- amount ベース置換: クラスタ方式と同様

**39 手詰め(10M nodes / 60s，同条件比較):**

| 方式 | NPS | overflow | 結果 |
|------|-----|---------|------|
| baseline (クラスタ 2M×6) | ~89K | 449K | UNKNOWN |
| tombstone + MAX\_PROBE=128 | ~89K | 8.5K (**-98%**) | UNKNOWN |

overflow は 449K→8.5K と 98% 削減され，クラスタ飽和問題は本質的に解消した．
しかし 10M ノード規模では NPS 差は顕在化しない(両方 ~89K で同等)．

**29 手詰め(大規模探索での退行):**

| 方式 | テスト | 制限 | 結果 |
|------|--------|------|------|
| baseline (クラスタ) | test\_tsume\_6\_29te | 50M / 300s | **Mate(29)** |
| tombstone | test\_tsume\_6\_29te | 50M / 300s | **タイムアウト** |
| tombstone | test\_tsume\_6\_29te\_no\_pns | 120M / 1200s | **Mate(29)** (overflow 63M) |

50M ノード・300s の制限内でタイムアウト．
制限を 120M / 1200s に緩和すると解決するが overflow が 63M に達し，
GC rebuild の累積コストで実効 NPS が大幅に低下していることが示唆される．

**不採用の理由:**

不採用の根本原因は **GC の計算量のオーダーの違い** にある．
詰将棋ソルバーは Frontier サイクル毎の `retain_proofs()` と
Periodic GC(1M ノード毎)で高頻度に GC を実行するため，
1 回あたりの GC コストが全体の NPS に直結する:

| 操作 | クラスタ方式 | リニアプロービング |
|------|-----------|----------------|
| エントリ削除 | `pos_key = 0`(O(1)/エントリ) | tombstone マーク(O(1)/エントリ) |
| GC 全体 | O(N) 線形スキャン **のみ** | O(N) スキャン + O(N) クリア + **O(K×probe) 再挿入** |
| GC 後の状態 | 即座に使用可能 | tombstone 蓄積でプローブ距離が延伸 |
| GC 頻度 | Frontier 毎 + 1M ノード毎 | 同左(ただしコスト大) |

クラスタ方式の `retain_proofs()` は単純な O(N) スキャンで `pos_key=0` をセットするのみ:

```rust
// クラスタ方式の GC — O(N) の線形スキャンのみ
fn retain_proofs(&mut self) {
    for fe in self.table.iter_mut() {
        if fe.pos_key == 0 { continue; }
        let keep = fe.entry.pn == 0
            || (fe.entry.dn == 0 && !fe.entry.path_dependent);
        if !keep { fe.pos_key = 0; }
    }
}
```

リニアプロービングでは同等の GC に 3 パスが必要:

1. **スキャン**: 生存エントリを Vec に収集 — O(N)
2. **クリア**: 全 8M スロットをゼロ埋め — O(N)
3. **再挿入**: 収集したエントリをプローブして挿入 — O(K×probe)，
   かつ再挿入時にプローブ衝突が発生

この差が Frontier サイクル毎(最大 50 回/IDS イテレーション)と
Periodic GC(1M ノード毎)で累積し，29 手詰め(74M ノード)規模では
数十回以上の GC が実行されるため，実効 NPS を探索不能レベルまで低下させた．

**tombstone 方式の LOOKUP\_PROBE ジレンマ:**

tombstone 蓄積によりプローブ距離が延伸するため，
lookup 時のプローブ距離に制限(LOOKUP\_PROBE)を設ける必要がある:

| LOOKUP\_PROBE | 正確性 | NPS への影響 | 問題 |
|-------------|--------|-----------|------|
| 24 (短い) | **不正確** | 軽微 | proof エントリを発見できず PV が不完全(25手で切断) |
| 128 (長い) | 正確 | **NPS 低下** | 毎回 128 スロットまでスキャン |

短い LOOKUP\_PROBE では詰将棋の正確性に影響し(proof エントリの見落としで
PV が途中で切断される)，長い LOOKUP\_PROBE では lookup 1 回あたりの
コストが増大して NPS が改善しない．クラスタ方式では固定 6 スロットの
スキャンで済むため，このジレンマは存在しない．

**結論: NPS と容量のトレードオフは構造的**

| 指標 | クラスタ方式 | リニアプロービング |
|------|-----------|----------------|
| NPS | **~253K-868K** | ~89K(GC コストで低下) |
| 実効容量率 | ~52%(hand バリアント制約) | ~100%(プローブで吸収) |
| GC コスト | O(N) | O(N) + O(K×probe) |
| lookup コスト | O(6) 固定 | O(1)〜O(128) 可変 |
| 29手詰め(50M/300s) | **Mate(29)** | タイムアウト |

クラスタ方式は「NPS が高いが容量に制約がある」，
リニアプロービングは「容量制約がないが GC で NPS が低下する」
という根本的なトレードオフを持つ．
現状は NPS 優位のクラスタ方式を基盤とし，
クラスタ飽和は対症療法(停滞検出 + amount ベース置換)で緩和している．

v0.21.1 の調査で特定された TT の構造的問題:

1. **クラスタ飽和**: 固定 6 エントリ/クラスタでは同一盤面の hand バリアントが
   溢れて store 失敗が発生する．対症療法として停滞検出を実装済み．
2. **置換ポリシー**: amount ベース置換(v0.22.0)で改善済み．

今後の代替候補:
- **クラスタサイズ拡大**: CLUSTER\_SIZE=8〜12 でクラスタ飽和を緩和しつつ
  キャッシュ効率を維持(リニアプロービングの代替)
- **store 失敗時の fallback**: store 成否を呼び出し元に返し，
  look\_up に依存せず直接値を使う．

#### Dual TT での 39手詰め調査 (v0.24.0)

**大予算テスト (root, depth=41):**

| 予算 | ノード | TT entries | NPS | proven\_overflow | working\_overflow | 結果 |
|------|--------|-----------|-----|----------------|-----------------|------|
| 50M | 50M | 1.12M | 201K | 11.8M | 6.3M | Unknown |
| 100M | 100M | 1.45M | 285K | 37.3M | 8.9M | Unknown |
| 200M | 200M | — | 320K | — | — | Unknown |
| 500M | 500M | — | 425K | — | — | Unknown |

500M ノード(19.6分)でも解けない．TT 使用率 7.3% に対し overflow 37% (100M)．

**depth スケーリング (ply 24, 50M budget):**

| depth | ノード | 結果 |
|-------|--------|------|
| 17 | 317K | **Mate(21)** |
| 19 | 7.3M | **Mate(21)** |
| 21 | 4.1M | **Mate(21)** |
| 23 | 50M | Unknown |
| 25-41 | 50M | Unknown |

depth 23 が構造的な壁: depth 21 までは解けるが 23 以降は 50M でも解けない．
depth=23 で初めて非 PV 変化の深い分岐が探索対象になり，
合駒チェーンの不詰証明が指数的に拡大する．

クラスタ飽和の定量分析・改善案の評価・不採用案の詳細は §6.6.3 を参照．

#### 構造的課題の最終状態 (v0.24.0)

**課題 A: クラスタ飽和 — ProvenTT は解決，WorkingTT は残存**

hand\_hash 混合(§6.6.3)により ProvenTT overflow を 99.6% 削減．
WorkingTT overflow は hand\_gte 再利用維持のため残存(改善案は全て NPS 低下で不採用)．

**課題 B: IDS の depth 切り替えによる情報損失 — 部分解決**

**Frontier サイクル間:** `retain_proofs()` を段階的クリアに変更(§6.6.3)．
confirmed disproof (!path\_dep, REMAINING\_INFINITE) を Frontier サイクル間で保持する．
中間エントリと path-dep disproof は除去．

**IDS depth 切り替え時:** `clear_working()` で WorkingTT を全クリアし，
`clear_proven_disproofs()` で ProvenTT の confirmed disproof を除去する．
浅い depth の confirmed disproof が深い depth を汚染するのを防止(NoMate バグ §6.6.3)．

**段階的 retain の効果:**
ProvenTT のクラスタ飽和が解消された状態(§6.6.3)で段階的 retain を導入することで，
NPS が hand\_hash only の 226K → 244K に改善(confirmed disproof の Frontier サイクル間保持が有効)．
ProvenTT のクラスタ飽和が残存する状態では段階的 retain は逆に NPS を低下させる
(保持された disproof がクラスタを圧迫するため)．

残存する情報損失: IDS depth 切り替え時の全クリアは依然として行われる．
ただし ProvenTT の proof は全フェーズを通じて永続するため，
proof ベースの探索効率は IDS 切り替えの影響を受けない．

**課題 C: 合駒チェーンの指数的分岐**

backward 解析で ply 26→24 で TT が 189→321K(×1,700)に急増．
組成は disproven=267K(83%)が支配的で，合駒チェーンの不詰証明が指数的に分岐する．

ply 24 は depth=17 なら 317K ノードで解けるが，depth=41 では 50M でも解けない．
depth が大きいと depth-limited pruning の発動が遅れ，
合駒分岐の探索が深くまで続くためノード数が指数的に増加する．

解決の方向性:
- **合駒の事前フィルタ強化**: 合駒の種類を事前に絞り込み分岐を減らす
- **合駒の不詰証明の共有**: 異なる合駒種で同じ不詰パターンが繰り返される場合に
  hand\_gte を活用して証明を共有
- **Progressive deepening**: 浅い depth で証明した結果を deep の探索に漸進的に活用

**v0.24.0 最終構成での到達点:**

| 課題 | 状態 | 改善内容 | 残存影響 |
|------|------|---------|---------|
| A: ProvenTT クラスタ飽和 | **解決** | hand\_hash 混合(§6.6.3) | overflow -99.6% |
| A: WorkingTT クラスタ飽和 | **残存** | hand\_gte 維持のため pos\_key 必須 | overflow 4.3M (50M) |
| B: IDS 情報損失 | **部分解決** | 段階 retain(§6.6.3) | IDS 切替時の全クリアは残存 |
| C: 合駒分岐 | **残存** | §8.1-8.8 で最大限最適化済み | 指数的増加が律速要因 |

**総合改善:** ベースライン(v0.24.0 初期)対比で NPS +33% (50M: 183K → 244K)．
ProvenTT overflow 99.6% 削減，WorkingTT overflow 16% 削減．

**KomoringHeights との比較:**
KomoringHeights はこの 39手詰め問題を ~10分で解く．
リニアプロービング + amount ベース置換でクラスタ飽和問題がなく，
TT の実効容量が高い．maou\_shogi は課題 A (ProvenTT) を解決し NPS を改善したが，
WorkingTT のクラスタ飽和と合駒分岐の指数的増加(課題 C)が残存しており，
39手詰めの解決には更なる改善または大幅なノード予算(推定 1B+)が必要と考えられる．

#### TT 改善後の 39手詰め調査 (v0.24.14〜v0.24.16)

v0.24.2〜v0.24.16 の TT 改善(§6.6.4)により WorkingTT overflow を大幅削減:

**大予算テスト (root, depth=41):**

| 構成 | 予算 | NPS | proven\_overflow | working\_overflow | 結果 |
|------|------|-----|----------------|-----------------|------|
| v0.24.0 最終 | 50M | 244K | 44.5K | 4.3M | Unknown |
| v0.24.14 (rem=0 廃止) | 50M | 358K | 0 | 170K | Unknown |
| **v0.24.16 (proof近傍+歩disproof)** | **50M** | **107K** | **7** | **3.1M** | **Unknown** |

**改善点:**
- NPS: 244K → 358K (**+47%**)
- ProvenTT overflow: 44.5K → 0 (完全解消，CLUSTER\_SIZE=8 + Zobrist hand\_hash)
- WorkingTT overflow: 4.3M → 170K (**-96%**，rem=0 store 廃止 + サンプリング GC)

**v0.24.14 で 200M ノードでも解けない原因:**

**backward 解析 (v0.24.14 → v0.24.16, 1M ノード/180秒):**

v0.24.14 (近傍走査なし)では ply 24 が Unknown にリグレッションしたが，
v0.24.16 (proof(-1) + 歩disproof(+1) 近傍走査)で修復:

| Ply | 残り手数 | v0.24.0 | v0.24.14 | v0.24.16 |
|-----|---------|---------|---------|---------|
| 38 | 1 | 14 Mate(1) | 10 Mate(1) | 10 Mate(1) |
| 26 | 13 | 100 Mate(13) | 112 Mate(13) | 103 Mate(13) |
| **24** | **15** | **317K Mate(21)** | **1M Unknown** | **371K Mate(21)** |
| 22 | 17 | 1M Unknown | 1M Unknown | 1M Unknown |

v0.24.14 のリグレッション原因: hand\_hash インデクシング導入で
`hand_gte_forward_chain` が自クラスタ内でしか機能しなくなり，
合駒チェーン分岐の proof 再利用が阻害された．
v0.24.16 の proof(-1) 近傍走査でヒットの 77% をカバーし修復．

50M ノード時の詳細診断:
- **root pn/dn**: `rpn=88, rdn=128` で 50M ノード間全く変化なし(完全停滞)
- **ply 分布**: ply=39,40 に 60%+ のノードが集中(depth boundary 付近で回転)
- **IDS depth**: depth=41 の最終ステップに到達済みだが MID が進捗しない
- **TT エントリ**: IDS の `clear_working` で周期的にリセット(max 400K 程度)

**停滞の構造:**

探索は depth=41 で MID ループに入るが，root の pn/dn が更新されない．
ply=39,40 (depth boundary の 1〜2 手前)でノードが消費され続け，
deeper ply に到達する閾値が生成されない．

これは v0.24.0 の分析(§10.2 課題 C)と同じ構造で:
1. **合駒チェーン分岐**: depth=41 でもチェーン合駒の不詰証明が指数的に分岐
2. **depth boundary 集中**: ply=39,40 の depth-limited 仮反証が繰り返し生成
   (v0.24.14 では rem=0 は TT に store しないが，has\_proof チェックのコストは残る)
3. **Working TT 容量**: GC の改善で intermediate の保持率は改善(1.2→2.0/cluster)
   したが，根本的な探索空間の爆発には対処できない

**v0.24.0 からの変化:**

| 課題 | v0.24.0 | v0.24.16 | 状態 |
|------|---------|---------|------|
| A: ProvenTT overflow | 44.5K | **7** | **完全解決** |
| A: WorkingTT overflow | 4.3M | **3.1M** | **構造改善(近傍走査+GC)** |
| B: NPS | 244K | **107K** | **低下(近傍走査コスト)** |
| B-2: ply 24 backward | 317K Mate(21) | **371K Mate(21)** | **回復(proof 近傍走査)** |
| C: 合駒分岐 | 指数的増加 | 同左 | **残存(律速要因)** |
| D: depth boundary 集中 | ply 39,40 に集中 | 同左 | **残存** |

**NPS 低下の要因分析:**
v0.24.0 の 244K → v0.24.16 の 107K は proof(-1) 近傍走査(最大7クラスタ)と
歩 disproof(+1) 近傍走査(2クラスタ)のコスト．
近傍走査ヒット率は 0.27% と低いが，proof ヒットは合駒チェーンの
大きな部分木を一発で刈るため，探索正確性の改善に不可欠．
近傍走査なし(219K NPS)では ply 24 が Unknown にリグレッションするため，
NPS と正確性のトレードオフとして107K を許容する．

**今後の方向性 (v0.24.16 時点):**

> **⚠️ 注記**: 本節は v0.24.16 時点の方向性であり，現行の優先順位は
> §10.2 末尾「v0.24.44 ply25 gap 診断」参照．

TT の構造的課題(overflow，NPS)は大幅に改善されたため，
残る律速要因は**探索アルゴリズムレベルの改善**:

- **合駒の不詰証明の効率化**: 異なる合駒種での不詰パターンの共有(hand\_gte の活用強化)
- **IDS depth 戦略**: depth=41 一括ではなく，depth=32→41 の段階で
  Frontier Variant の MID 停滞検出をより早く発動させる
- **MID 閾値伝搬の改善**: depth boundary 付近でのノード集中を緩和する
  閾値制御(dn\_floor の adaptive 化等)

#### 39手詰め最新調査 (v0.24.27, Plan D + WorkingTT 拡張)

v0.24.23〜v0.24.27 で PV 抽出経路と TT エントリレイアウトを再設計し，
**WorkingTT を 6→8 entries/cluster に拡張** (§6.6.5)．
ply 22 サブ問題が初めて通常の `solve()` 経路で解けるようになった．

**ply22 回帰テスト (`test_tsume_39te_ply22_no_pns`):**

| 構成 | 時間 | Δ |
|---|---|---|
| Plan D baseline (v0.24.26) | 159.59s | — |
| **v0.24.27 (WorkingTT 6→8)** | **104〜107s** | **-34.6%** |

`mid_fallback` 直接呼び出し経路で 120M ノード予算 / 1200s timeout．
WorkingTT slot を 33% 増やすことで intermediate エントリの eviction
thrashing が解消された．

**注意 — 「10M で解ける」narrative の整理:**

v0.24.0〜v0.24.27 の改善過程で「10M で ply 22 が解ける」という記述が
複数箇所に出るが，これは **2 種類の予算が混同されていた**:

1. **PV 抽出予算 (`PV_VISIT_BUDGET = 10M`)**: 探索完了後の TT 歩行 (longest
   resistance reconstruction) における visit 回数上限．Plan B (v0.24.23) で
   `mate_distance` fast path を導入し O(B^D) → O(depth × B) に改善した結果，
   10M visit で十分．**ただし v0.24.29 でこの fast path は unsound (無駄合
   bypass バグ) として廃止された (§10.2 末尾「v0.24.29 PV 抽出 fast path
   廃止」参照)．**
2. **探索ノード予算 (`max_nodes`)**: DFPN 探索本体のノード展開上限．
   ply 22 を解くには **120M nodes** 必要 (`test_tsume_39te_ply22_no_pns` の値)．

「10M」は (1) であり (2) ではない．以下の backward 解析でこの違いを明確化した．

**バックワード解析 (v0.24.27, 通常 `solve()` 経路):**

PV を逆順にたどり，各偶数 ply (攻め方手番 = OR ノード) を個別に解いて
解ける境界 ply を特定する．予算別に 3 回実施．

| Ply | 残り | 1M nodes | 10M nodes | 120M nodes |
|----:|----:|---|---|---|
| 38 | 1 | Mate(1) ✓ | Mate(1) ✓ | Mate(1) ✓ |
| 36 | 3 | Mate(3) ✓ | Mate(3) ✓ | Mate(3) ✓ |
| 34 | 5 | Mate(5) ✓ | Mate(5) ✓ | Mate(5) ✓ |
| 32 | 7 | Mate(7) ✓ | Mate(7) ✓ | Mate(7) ✓ |
| 30 | 9 | Mate(9) ✓ | Mate(9) ✓ | Mate(9) ✓ |
| 28 | 11 | Mate(11) ✓ | Mate(11) ✓ | Mate(11) ✓ |
| 26 | 13 | Mate(13) ✓ | Mate(13) ✓ | Mate(13) ✓ |
| **24** | **15** | **Unknown** | **Mate(15) ✓** (367k) | **Mate(15) ✓** (367k) |
| **22** | **17** | Unknown | Unknown | **Mate(17) ✓** (38M, 419s) |
| **20** | **19** | Unknown | Unknown | **Unknown (timeout)** (10.9M, 1817s) |

**判明事項:**

1. **境界は ply 24/22 → ply 22/20 にシフト**: v0.24.27 で初めて通常 `solve()` 経路
   から ply 22 が解けるようになった (Mate(17) を 38M ノードで証明)
2. **ply 20 (Mate(19)) は 120M でもタイムアウト**: 10.9M ノード時点で 1817 秒
   経過 → 探索速度 6 kn/s と異常に遅い (ply 22 の 91 kn/s と比べて 1/15)
3. **ノード予算ではなく時間が律速**: 深い局面ほど 1 ノードあたりのコストが
   重く，枝の厚さが速度を落とす．ply 20 の真の必要ノード数は 30-100M と推定

**コスト増大率 (v0.24.27):**

| 区間 | ノード比 | 時間比 |
|---|---|---|
| ply 26 → 24 | 103 → 367k = **×3565** | 0.4 → 41s = **×100** |
| ply 24 → 22 | 367k → 38M = **×104** | 41 → 419s = **×10** |
| ply 22 → 20 | 38M → 10.9M+ (打ち切り) | 419 → 1817s+ |

ply 26→24 の跳躍 (×3565) が最大．以後 2 手深くなるごとに概ね **×10〜100** の
スケールでコストが増加する．完全解 (ply 0) までの距離はまだ大きく，
ply 20 で既に 5 時間相当の見積もり → **完全解には数十時間〜数日** と推定．

**測定スクリプト:** `test_tsume_39te_backward_1m`, `test_tsume_39te_backward_10m`,
`test_tsume_39te_backward_120m` (tests.rs, `#[ignore]`)．
ログは `/tmp/tsume_39te_backward_*.log`．

#### v0.24.29 PV 抽出 fast path 廃止と再測定

v0.24.23〜v0.24.28 で `extract_pv_recursive_inner` に存在した
TT `mate_distance` ベースの fast path が **無駄合 chain 下で unsound**
だったため廃止した (§6.6.5, commit `6686002`)．具体的には:

- Fast path は子の **raw** TT distance を比較していた
- slow path は `effective_len = total_len - 2 × useless_pairs` で比較
- 深い chain drop が proven 状態になると fast path が fire し，chain
  inflate された raw distance で Mate(21) を誤って返す可能性があった

実害は潜在的で，v0.24.27 時点のテストでは偶発的に fire しなかった
(chain drop 子が全て proven という条件が稀だった) ため ply 24 Mate(15)
を正しく返していた．v0.24.29 で fast path を完全削除し slow path のみを
使うようにした．slow path は visit 予算 10M で cost が bound されている．

**v0.24.29 再測定 (backward_120m，fast path 廃止後):**

| Ply | 残り | v0.24.27 (fast path) | v0.24.29 (slow のみ) |    Δ |
|----:|----:|---|---|---|
| 38-26 | 1-13 | < 0.5s | < 0.5s | noise |
| 24 | 15 | 41.18s (367k nodes) | **41.10s** (367k nodes) | ~0% |
| 22 | 17 | 418.91s (38.1M nodes) | **398.44s** (38.1M nodes) | **-4.9%** |
| 20 | 19 | Unknown timeout (1817s) | Unknown (search bottlenecked) | — |

**重要な発見**: 深い aigoma (ply 22 Mate(17)) で v0.24.29 の方が **4.9% 速い**．
これは直感に反する結果だが理由は明快:

- fast path は「全 child の TT lookup + distance 取得」を試み，いずれかが
  失敗したら slow path に fallback する 2 パス構造だった
- chain drop 子が proven になる確率が低い deep aigoma では fast path が
  ほぼ fire せず，**事前 TT lookup が純粋なオーバーヘッド**になっていた
- v0.24.29 では fast path が完全に削除されたため，このオーバーヘッドが消滅

浅い詰み (typical <15 手) では full dfpn suite で 140s → 156s (+11%)
の軽微な regression がある．これは fast path が実際に fire していた
浅いケースでの slow path への単純化コスト．深い詰み (律速要因) では
むしろ改善しているため，**v0.24.29 は soundness と性能の両面で v0.24.27
より優れている**．

**今後の方向性 (v0.24.29 時点):**

> **⚠️ 注記**: 本節は v0.24.29 時点の方向性．NPS 根本改善は v0.24.33 の
> PNS NM 昇格判定修正で解決済み．段階的 IDS depth は v0.24.40 で導入済み．
> **探索分割 (サブ問題ウォームアップ)** は v0.24.44 の診断で
> 単体では効果限定的と判明しており，現行の優先順位は §10.2 末尾
> 「v0.24.44 ply25 gap 診断」参照．

ply 20 以降を解くには次の改善が必要と推定していた:

- **NPS の根本的改善**: ply 20 で 6 kn/s は異常．move generation,
  TT lookup, check generation のいずれかが指数的に重くなっている
  → **v0.24.33 で解決**: `depth_limit_all_checks_refutable` が
  PNS 時間の 99.97% を消費していた律速要因を TT + memoize ハイブリッド化で除去
- **合駒チェーンの不詰証明共有**: 異なる駒種で同一不詰パターンが繰り返される
  ケースを `hand_gte_forward_chain` で吸収する深さ・幅両方向の拡張
  → **部分実装**: Zobrist XOR 差分近傍走査 (v0.24.8) + proof(-1) 近傍走査
  (v0.24.16) で改善．ただし v0.24.44 診断で `cross_deduce_hits = 0` が
  観測され，駒種横断 disproof 共有は未達成
- **段階的 IDS depth**: depth=41 一括ではなく depth=32→41 の段階解法で
  Frontier Variant の MID 停滞検出を早く発動させる
  → **v0.24.40 で部分導入**: IDS depth 32+ で +4 刻み
- **探索分割**: ply 24 サブ問題のような中間目標で TT ウォームアップ後に
  ルートから解き直すアプローチ
  → **v0.24.44 で再評価**: Phase 0b 測定で個別サブ問題の 3 件が 500K budget
  で Unknown となり，単純ウォームアップは効果限定的．施策 I (intermediate
  保持) と統合する必要がある

※ このうち 1 番目 (NPS の根本的改善) について，v0.24.32 の診断調査で
律速要因が特定され，v0.24.33 で解決された．次節参照．

#### 39手詰め最新調査 (v0.24.32 診断 + v0.24.33 fix)

##### v0.24.32: NPS 崩壊の診断

v0.24.31 までの観測で「ply 20 で NPS が時間と共に低下し，ノード予算を
増やしても探索が進まない完全な plateau が発生する」という異常を
`test_tsume_39te_profile_ply20_timeline` で再現した:

| budget | nodes       | 実時間 | NPS       | 備考 |
|--------|-------------|--------|-----------|---|
| 30s    | 569k        | 30.5s  | 18.7 kn/s | 通常 |
| 90s    | 7,036k      | 90.4s  | 77.8 kn/s | 通常 |
| 180s   | 10,856k     | 192.5s | 56.4 kn/s | **plateau 突入** |
| 360s   | 10,859k     | 367.5s | 29.5 kn/s | **+175s で +2,816 nodes のみ** |

180s → 360s で追加 175 秒を使ったにもかかわらずノード数が +0.03% しか
増えない完全停止．`verbose` 追加計測で `frontier_variant()` iter 1 の
PNS フェーズに 115 秒が消えていることを特定:

```
[fv] iter 1 total=114080ms pns=113857ms(1791nodes arena=281) mid=0ms retain=150ms
[pns_main_exit] arena=281 iters=1792 | sel=0ms exp=113984ms undo=1ms bk=0ms
  refut_invocations=1,723 refut_calls=17,240,174
  refut_ns=113,970,357,047 refut_limit_hits=1,721/1,723
```

**PNS 時間 114 秒の 99.97% が `depth_limit_all_checks_refutable` に消費され，
1,723 回の呼出しの 1,721 回 (99.9%) が REFUTABLE_CALL_LIMIT=10,000 に到達**．

原因は MID 側 (`solver.rs:1194`) と PNS 側 (`pns.rs:2271,2458`) で同じ NM 昇格
判定に異なる実装を使っていた非対称性:

| 関数 | 実装 | 1 呼出あたりコスト |
|---|---|---|
| `all_checks_refutable_by_tt` (MID が使う) | TT ルックアップのみ | ~16 µs |
| `depth_limit_all_checks_refutable` (PNS が使う) | 5 レベルの再帰 movegen, 上限 10,000 回 | ~66 ms (limit 到達時) |

差は **約 4,000 倍**．v0.24.31 までは PNS がすべてこの遅い方を使っていた．

##### v0.24.33: ハイブリッド判定への修正

新設した `refutable_check_with_cache` で次の 3 段階判定を行う:

1. **TT ベース高速判定** (`all_checks_refutable_by_tt`, ~2µs/王手) を先行
2. false の場合は既存の `refutable_check_failed` HashSet で memoize 確認
   (既出なら skip)
3. 未キャッシュなら再帰判定にフォールバックし，false の場合は pos_key を
   キャッシュに記録

既存の `refutable_check_failed` フィールドは宣言・clear() はされていたが
insert/contains が無い完全な死蔵コードだった．v0.24.33 で初めて実運用に
乗せる．再帰判定の結果は `all_checks_refutable_recursive` が position 依存
(探索深さ非依存) のため false の memoize は sound．

##### ply20 timeline: plateau の消滅

```
test_tsume_39te_profile_ply20_timeline (ply 20 独立 solve() 複数予算):
```

| budget | v0.24.31 nodes | **v0.24.33 nodes** | v0.24.31 NPS | **v0.24.33 NPS** |
|-------:|---------------|--------------------|--------------|------------------|
| 30s    | 569k          | **1,082k** (+90%)  | 18.7 kn/s    | **35.5 kn/s**    |
| 90s    | 7,036k        | 6,988k (同等)      | 77.8 kn/s    | 77.3 kn/s        |
| 180s   | 10,856k       | **14,973k** (+38%) | 56.4 kn/s    | **82.9 kn/s**    |
| 360s   | 10,859k       | **30,947k** (+185%)| 29.5 kn/s    | **85.8 kn/s**    |

**plateau が完全消滅**し，ノード数が予算に比例して単調増加するようになった．
360s で 30.9M ノードに到達 (v0.24.31 は 10.9M で停止していた)．

##### backward_120m: 境界が ply 14 に前進 (Mate(19)/(21)/(23) を初解決)

`test_tsume_39te_backward_120m` で v0.24.33 の新境界を測定 (120M nodes /
1800s per ply, 最初の Unknown で停止)．ログは `/tmp/tsume_39te_backward_120m.log`:

| Ply | 残り | v0.24.29 nodes/time | **v0.24.33 nodes/time** | Δ |
|----:|----:|---|---|---|
| 38-26 | 1-13 | <0.5s | <0.5s | — |
| 24 | 15 | 367k / 41.10s | **367k / 33.56s** | **-18% time** |
| 22 | 17 | 38.1M / 398.44s | **16.9M / 192.28s** | **-56% nodes, -52% time** |
| **20** | **19** | **Unknown (timeout 1817s)** | **76.7M / 638.13s** **Mate(19)** ✓ | **初解決** |
| **18** | **21** | (未到達) | **105.8M / 738.86s** **Mate(21)** ✓ | **初解決** |
| **16** | **23** | (未到達) | **119.4M / 976.26s** **Mate(23)** ✓ | **初解決** |
| **14** | **25** | (未到達) | 119,998,148 / 911.70s **Unknown** | **新境界** |

**境界が ply 20 → ply 14 へ 4 ply 前進**．ply 20 (Mate(19))，ply 18 (Mate(21))，
ply 16 (Mate(23)) が通常 `solve()` 経路で初めて解けた．

ply 16 は 119.4M / 120M 予算の **99.5%** まで使用しており，budget edge
ギリギリで解けている．ply 14 は 99.998M で Unknown に留まり新境界となった．

##### コスト増大率の再評価 (v0.24.33)

| 区間 | v0.24.29 時間比 | **v0.24.33 時間比** |
|---|---|---|
| ply 26 → 24 | ×100 | ×80 (0.42→33.6s) |
| ply 24 → 22 | ×10 | ×5.7 (33.6→192.3s) |
| ply 22 → 20 | timeout | ×3.3 (192.3→638.1s) |
| ply 20 → 18 | — | ×1.2 (638→739s) |
| ply 18 → 16 | — | ×1.3 (739→976s) |
| ply 16 → 14 | — | <×1 (既に budget 上限) |

2 ply 深くなるごとのコスト増大率が ×10〜100 から **×1.2〜5** 程度に
劇的に低下．当面は時間ではなく **ノード予算そのもの** が律速で，
純粋な探索効率 (合駒分岐の展開) が次の壁．

##### 回帰テスト (v0.24.33 で実施):

- `test_tsume_39te_ply24_mate15_regression`: PASS (45s, Mate(15) canonical)
- `test_tsume_39te_ply22_no_pns`: PASS (8.8M nodes / 93s; v0.24.29 baseline
  38M nodes / 419s → **4.5× 高速化**)
- `test_tsume_39te_backward_1m`: 境界は引き続き ply 22/Unknown (1M budget で
  ply 22 は解けないが，ply 24 まで従来通り解ける)
- `test_tsume_39te_backward_10m`: ply 20 (Mate(19)) が **10M budget で初解決**
  (8.84M nodes / 131s)．
- 非 ignored テスト全 127 件: PASS

##### 今後の方向性 (v0.24.33 時点)

> **⚠️ 注記**: 本節は v0.24.33 時点の方向性．**v0.24.44 の診断で
> 「探索分割 (サブ問題ウォームアップ)」は単体では効果限定的と判明**し，
> 「合駒チェーンの不詰証明共有」は `cross_deduce_hits = 0` (実運用で
> 未発火) という根本的問題が観測された．現行の優先順位は §10.2 末尾
> 「v0.24.44 ply25 gap 診断と施策優先順位の組み替え」参照．

ply 14 以降を解くには次の改善が必要と推定していた:

- **ノード予算の拡大**: ply 14 は 120M で Unknown．ply 20 → 18 → 16 の
  コスト増大率が ×1.2〜1.3 で緩やかなので，数百 M 程度で ply 12〜10 まで
  到達する見込み．ただしルート (ply 0) には更に 7 ply 分の深さがある
  → **部分確認**: 500M 予算で ply 14 (Mate(25)) 解決，ply 12 で Unknown
- **合駒チェーンの不詰証明共有**: 依然律速要因．`hand_gte_forward_chain`
  の更なる拡張または inter-ply proof reuse
  → **v0.24.44 で再整理**: 既存の `cross_deduce_children` が実運用で
  ほぼ発火していないことが判明し，**既存実装の修復が先決**となった
- **段階的 IDS depth**: depth=41 一括ではなく段階解法
  → **v0.24.40 で +4 刻み導入済み**．更なる細分化は v0.24.44 施策 III として検討
- **探索分割**: ply 24/22 のサブ問題を先行解決し TT ウォームアップ後に
  ルートから解き直すアプローチ
  → **v0.24.44 で再評価**: 個別サブ問題自体に難度の差があり，intermediate
  保持 (新規施策 I) と統合しない限り単体での効果は期待しにくい

NPS 崩壊 (元 v0.24.31 以前の "ply 20 で 6 kn/s" の異常) は解消されたため，
残る課題は**探索アルゴリズム空間での最適化**である．

##### backward_500m: 500M ノード予算での境界 (ply 14→ply 12 に前進)

`test_tsume_39te_backward_500m` (500M nodes / 3600s per ply, `verbose` feature)
で PNS 空回り統計と `refutable_check_with_cache` の経路別ヒット率を測定:

| Ply | 残り | Nodes | Time(s) | 結果 | PNS spin% | refut tt/memo/rec |
|----:|----:|---:|---:|---|---:|---|
| 38-26 | 1-13 | ≤103 | ≤0.5 | Mate(N) | 0-32% | 0/0/3 |
| 24 | 15 | 367k | 34.7 | Mate(15) | 75.7% | 0/259/3265 |
| 22 | 17 | 16.9M | 199.1 | Mate(17) | 86.5% | 4/1228/4843 |
| 20 | 19 | 201.7M | 1039.7 | Mate(19) | 84.9% | 0/53160/6732 |
| 18 | 21 | 243.4M | 1092.4 | Mate(21) | 86.0% | 0/1269/2875 |
| 16 | 23 | 454.1M | 2622.7 | **MateNoPV** | 87.1% | 2/3369/4550 |
| **14** | **25** | **498.3M** | **2727.7** | **Mate(25)** ✓ | 88.1% | 17/5507/5469 |
| **12** | **27** | 500.0M | 3152.8 | **Unknown** | **92.2%** | 1/378/455 |

500M 予算で **ply 14 (Mate(25)) が解決可能**となり，境界は **ply 12 (残り 27 手)** に移動した．
120M 予算 (境界 ply 14) からさらに **2 ply 前進**．

PV(Mate(25)) が取れたのは ply 14 のみ．ply 16 は MateNoPV (証明成功だが
PV\_VISIT\_BUDGET=10M 超過)．

**コスト増大率 (v0.24.33, 500M):**

| 区間 | ノード比 | 時間比 |
|---|---|---|
| ply 24 → 22 | 367k → 16.9M = ×46 | 35 → 199s = ×5.7 |
| ply 22 → 20 | 16.9M → 202M = ×12 | 199 → 1040s = ×5.2 |
| ply 20 → 18 | 202M → 243M = ×1.2 | 1040 → 1092s = ×1.1 |
| ply 18 → 16 | 243M → 454M = ×1.9 | 1092 → 2623s = ×2.4 |
| ply 16 → 14 | 454M → 498M = ×1.1 | 2623 → 2728s = ×1.0 |
| ply 14 → 12 | 498M → 500M (cap) | 2728 → 3153s |

ply 24→22 が ×46 の跳躍で最大．以後の増大は ×1.0〜2.4 と緩やか．
500M 予算では **ply 14 (498.3M) が budget edge ギリギリ** で解け，
ply 12 (500.0M) が exactly cap で Unknown．

##### 識別された新課題

**課題 A: PNS root pn/dn 不変率 (85〜92%) の解釈と実効性の検証**

PNS メインループの各イテレーションで arena root の pn/dn が変化したかを計測した:

```
ply 24: 75.7% 不変 → ply 22: 86.5% → ply 14: 88.1% → ply 12: 92.2%
```

初期の分析ではこれを「空回り (spin)」と表現したが，この解釈は
PNS の導入目的に照らして **不正確** である．

**PNS が導入された目的 (§11.7, §10.2 閾値飢餓):**

PNS は MID の閾値飢餓 (TT エントリ停滞) を解決するために導入された．
MID が 50M → 100M ノードで TT エントリが完全同一になる問題 (§10.2
「閾値飢餓と TT クラスタ飽和」) に対し，PNS は MID が到達できない
部分木をグローバルに選択・証明し，TT に proof エントリを蓄積する
ことで MID の閾値突破を補助する．

**PNS の MID への寄与メカニズム:**

1. PNS は `pns_expand` で子を展開し，即座に TT に store する
   (証明/反証/中間の全てを store)
2. PNS メインループ終了時に `pns_store_to_tt` で proven ノードの
   best move を TT に追加格納する
3. 後続の MID フェーズで child init 時に TT を参照し，
   PNS が蓄積した proof エントリにヒットして分岐を即座に解決する
4. `retain_proofs()` でサイクル間の中間エントリは除去されるが，
   **proof (pn=0) と confirmed disproof は保持** されるため，
   PNS が蓄積した証明はサイクルをまたいで永続する

したがって **root pn/dn が不変でも PNS が新しい部分木を証明していれば
生産的な探索**である．root 不変は「最善手が変わらない」ことを意味するが，
非最善手の部分木で proof が蓄積されていれば，MID がその部分木に到達した
際に TT ヒットで即座に解決される．

**真の効率指標の候補:**

root pn/dn 変化率ではなく，以下を計測すべき:

- PNS サイクルあたりの **新規 proof store 数** (TT に新しく追加された
  pn=0 エントリ数)
- PNS サイクル後の MID フェーズでの **TT hit による child init 即解決数**
  (PNS が蓄積した proof を MID が実際に利用した回数)
- PNS サイクルあたりの **arena growth** (新ノード数 — arena が成長して
  いなければ PNS は既知領域を走査しているだけ)

これらの指標が低い場合に初めて PNS サイクルの予算削減が正当化される．
逆に proof store 数が高ければ，root 不変率 92% でも PNS は有効に機能
している可能性がある．

**改善の方向性:**

- PNS 生産性指標 (proof store / arena growth) を `verbose` feature で追加計測し，
  Frontier Variant の PNS 予算比率 (現行 remaining/20 = 5%) を実データに基づいて調整する
- PNS arena stagnation detection (P4) が正しく機能しているかを深い ply で再検証する
  (growth\_stall\_count の発火頻度と打ち切りタイミング)

**課題 B: `all_checks_refutable_by_tt` TT 経路は実質不活性**

全 ply 合計で TT 経路のヒットは **24 回** (0.03%) に対し，
memoize 経路は **64,970 回** (68%)．TT 経路は IDS depth 切替で
`clear_working()` + `clear_proven_disproofs()` が実行された後の
PNS で呼ばれるため，REMAINING\_INFINITE disproof が TT に残存しない．
現状は memoize 経路が実質的な高速パスとして機能している．

改善の方向性:
- `clear_proven_disproofs()` の見直し (REMAINING\_INFINITE disproof の
  一部を IDS depth 切替で保持する)
- 確認的不要と判断された場合は TT 経路を除去し `refutable_check_failed`
  のみで判定する

**課題 C: 深い ply での PV 抽出不完全 (MateNoPV)**

ply 16 (454M ノード探索) で証明成功後に `PV_VISIT_BUDGET = 10M` 内で
PV を再構成できなかった．TT が巨大化 (数百万エントリ) した状態での
longest resistance 再構成コストが予算を超過する．

改善の方向性:
- `PV_VISIT_BUDGET` の引き上げ (10M → 50M 等)
- PV 抽出アルゴリズムの効率化 (chain drop 子のスキップ等)

**課題 D: PNS 生産性指標の計測基盤 (v0.24.35)**

v0.24.35 で `verbose` feature に PNS 生産性指標を追加した．
従来の root pn/dn 不変率(空振り率)に加え，PNS の実効的な生産性を測定する
3 指標を `pns_main_with_arena` および `frontier_variant` で計測する:

| 指標 | フィールド | 意味 |
|------|----------|------|
| proof store 数 | `dbg_pns_proof_stores` | `pns_store_to_tt` で TT に新規格納された pn=0 エントリ数 |
| arena growth | `dbg_pns_arena_growth` | PNS サイクルで展開された新規アリーナノード数 |
| PNS サイクル数 | `dbg_pns_cycles` | `pns_main_with_arena` の呼び出し回数 |

Frontier Variant の各サイクルで `[fv] iter N pns: proofs=X arena_growth=Y spin=Z%`
を出力し，サイクル単位の生産性を可視化する．

**計測の目的:**

1. root pn/dn 不変 ≠ 真の空振りであることを定量的に検証する
2. Frontier PNS 予算比率(現行 remaining/20 = 5%)の adaptive 化の判断材料とする
3. PNS が生産的なサイクル(proof store > 0)と非生産的なサイクル(arena growth = 0)を
   区別し，非生産的サイクルで MID に予算を回す最適化の基盤とする

**計測結果 (39手詰め 10M 予算 backward 解析):**

| Ply | FV iters | Total proofs | Arena growth | Avg spin | Zero-proof cycles | Proof rate (/1K) |
|-----|----------|-------------|-------------|----------|-------------------|-----------------|
| 22 | 14 | 3,820 | 1,474K | 84.2% | 0/14 | 8.7 |
| 18 | 5 | 3,249 | 1,485K | 83.6% | 0/5 | 13.0 |
| 16 | 14 | 3,385 | 2,216K | 85.0% | 2/14 | 7.7 |

**重要な所見:**

- spin 率 83-85% でも全 ply で proof store は正値(平均 100-300 proofs/cycle)
- arena growth は毎サイクル 10万〜20万ノード — PNS は新しい部分木を展開している
- ply 16 で zero-proof サイクルが 2/14 出現 — 深い ply ほど PNS 生産性が低下する傾向

**課題 E: Frontier PNS 予算の動的制御 (v0.24.36〜v0.24.37)**

v0.24.36 で zero-proof early skip，v0.24.37 で proof rate ベース予算拡大を導入し，
Frontier Variant の PNS 予算を前サイクルの proof store 数に基づいて動的制御する:

- **zero-proof early skip (v0.24.36):** `consecutive_zero_proofs >= 2` で PNS スキップ，
  MID 予算を `remaining/4` → `remaining/3` に増加
- **proof rate ベース予算拡大 (v0.24.37):** 前サイクルの proof store > 0 の場合，
  PNS 予算を `remaining/20 (5%)` → `remaining/10 (10%)` に倍増．
  proof store が正値のサイクルでは PNS がより多くの proof を TT に蓄積でき，
  後続 MID の TT ヒット率を向上させる
- **proof store の取得:** `pns_store_to_tt` の戻り値(u64)を `last_pns_proof_stores` に格納

**課題 F: IDS depth 切替時の ProvenTT disproof 選択的保持 (v0.24.38)**

v0.24.38 で `clear_proven_disproofs()` を `clear_proven_disproofs_below(min_depth)` に
変更し，ProvenTT の confirmed disproof を確認時の IDS depth に基づいて選択的に除去する．

**実装:**

- ProvenEntry の flags bits 1-6 (disproof では未使用) に確認時の IDS depth を格納
- `clear_proven_disproofs_below(min_depth)` は `disproof_depth() < min_depth` の
  エントリのみ除去
- IDS 切替時の閾値: `next_ids_depth / 2`
  - 例: IDS 2→4→8→16→32→41 で 8→16 に進行する場合，
    threshold=8 で depth<8 (depth=2,4) の disproof を除去し depth=8 を保持
- TT に `current_ids_depth` フィールドを追加し，`store_proven` で自動的に記録

**計測結果:**

完全除去(clear なし)では ply 24 で +26% ノード増加の退行が発生したが，
選択的除去(`ids_depth / 2` 閾値)では ply 24 ノード数が v0.24.37 と完全同一(367,331)
に回復．浅い disproof のみ除去し，深い disproof を保持することで退行を回避．

**安全性:** 全 127 テスト pass，ply 24 ノード数退行なし

**課題 H: Depth-adaptive epsilon による閾値余裕の最適化 (v0.24.41)**

パラメータグリッドサーチにより 1+ε の epsilon 除数が depth に依存する最適値を
持つことを発見した．16 構成の網羅的テストで以下が判明:

| Config | Ply24 Nodes (depth=17) | Ply22 10M (depth=19) |
|--------|----------------------|---------------------|
| baseline (eps_denom=3) | **367,331** (最適) | Unknown |
| eps_denom=2 | 1,000,000 (+172% 退行) | **Mate(17) 8.1M** ✓ |
| eps_denom=4 | 1,000,000 (+172% 退行) | Unknown |

eps_denom=3 は depth=17 に最適だが depth=19 では不足．
eps_denom=2 は depth=19 で ply 22 を解けるが depth=17 では退行．

**depth-adaptive epsilon:**

```
eps_denom = if saved_depth >= 19 { 2 } else { 3 }
```

IDS の全反復で最終 depth (saved_depth) に基づいて eps_denom を決定し，
浅い反復でも深い問題向けの epsilon 余裕を適用する．

**計測結果 (10M backward 解析):**

| Ply | baseline | depth-adaptive | 変化 |
|-----|---------|---------------|------|
| 24 (depth=17) | 367,331 | **367,331** | **同一(退行なし)** |
| 22 (depth=19) | 10M Unknown | **8,111,974 Mate(17)** | **10M 内で解決可能に** |

**計測結果 (120M backward 解析，v0.24.42):**

| Ply | 残り | v0.24.33 nodes | v0.24.42 nodes | 変化 | v0.24.33 結果 | v0.24.42 結果 |
|-----|-----|---------------|---------------|------|-------------|-------------|
| 24 | 15 | 367,331 | 367,331 | 同一 | Mate(15) | Mate(15) |
| 22 | 17 | 16,942,722 | 31,361,713 | +85% | Mate(17) | Mate(17) |
| 20 | 19 | 76,683,642 | **39,118,955** | **-49%** | Mate(19) | Mate(19) |
| 18 | 21 | 105,800,075 | **97,588,759** | **-7.8%** | Mate(21) | Mate(21) |
| 16 | 23 | 119,400,000 | **117,080,263** | **-1.9%** | Mate(23) | Mate(23) |
| 14 | 25 | 120M Unknown | (120M Unknown) | — | Unknown | Unknown |

ply 20 で **49% ノード削減**が最大の効果．ply 22 はノード数増加(+85%)だが，
10M 予算で baseline が Unknown のところ 8.1M で解ける(探索の質が向上)．
境界は ply 14 で変化なし．

**課題 I: PV visit 予算の動的スケーリング (v0.24.42)**

v0.24.41 の depth-adaptive epsilon により ply 18 の探索パターンが変化し，
PV\_VISIT\_BUDGET=10M (固定) では AND ノードの全 defender 評価で
予算超過して MateNoPV が発生していた．

PV visit 予算を探索ノード数に応じてスケーリング:

```
pv_visit_budget = min(max(nodes_searched / 4, 10M), 50M)
```

- ply 18: 97.6M 探索 → budget 24M → **MateNoPV → Mate(21) 解消**
- 上限 50M で過大な PV 抽出時間を防止

**課題 J: GC Phase 2 no-op バグ修正 (v0.24.43)**

v0.24.38 の `clear_proven_disproofs()` → `clear_proven_disproofs_below(min_depth)`
リファクタリング時に，GC Phase 2 の呼び出しが
`clear_proven_disproofs_below(0)` のまま残されていた．

**バグ:** `disproof_depth() < 0` は u32 で常に false → no-op．
コメント「全 confirmed disproof を除去」とは真逆の挙動になり，
元の `clear_proven_disproofs()` の無条件全除去機能が失われていた．

**修正:** `clear_proven_disproofs_below(u32::MAX)` で全 disproof を除去．

**120M backward 解析での検証 (v0.24.42 vs v0.24.43):**

| Ply | v0.24.42 nodes | v0.24.43 nodes | 変化 |
|-----|---------------|---------------|------|
| 24 | 367,331 | 367,331 | 同一 |
| 22 | 31,361,713 | 31,361,713 | 同一 |
| 20 | 39,118,955 | 39,118,955 | 同一 |
| 18 | 97,588,759 (Mate(21)) | 97,588,759 (Mate(21)) | 同一 |
| 16 | 117,080,263 | 117,080,263 | 同一 |

**リグレッションなし．** 全 ply でノード数完全同一．
GC Phase 2 は ProvenTT 充填率が 70% を超過した時にのみ発火する Phase であり，
120M backward 解析では到達しないため，no-op バグは挙動に影響していなかった．
ただし長期実行や大予算 (500M+) でのバグ顕在化を防ぐため修正は必須．

#### v0.24.44 ply25 gap 診断と施策優先順位の組み替え

v0.24.44 で `test_tsume_39te_ply25_gap_diagnosis` を追加し，「個別サブ問題は
200K 以内で解けるのに統合探索は指数爆発する」という §10.2 冒頭の観察に対し，
verbose+tt_diag+profile の全カウンタを使った定量診断を行った．
ログは `/tmp/tsume_39te_ply25_gap_diagnosis.log`．

**診断構成 (3 フェーズ):**

- **Phase 0a**: ply 25 AND node を直接 solve (depth=15, 5M budget)．
  DfPn が root を OR として扱うため attacker 反転して NoMate を返し不成立．
  (設計上の制約として記録)
- **Phase 0b**: ply 25 AND node の 23 守備応手を個別に fresh solver で solve
  (各 depth=14, 500K budget, 15s timeout)．個別総和を計測．
- **Phase 1**: ply 24 OR node (5g6f を打つ前) を 3 種の depth (17, 21, 25) で
  solve し，depth inflation による counter 変化を観測．

**Phase 0b 実測結果 (v0.24.44, 500K budget / 15s per defender):**

| カテゴリ | 応手 | ノード範囲 | 結果 |
|---------|------|-----------|------|
| 即詰み (P/L/N/S/G/B\*2g) | 6 手 | 各 1 | Mate(1) |
| 簡単な合駒 (1g1h, P\*3g, B\*3g, N\*3g) | 4 手 | 38〜586 | Mate(11-13) |
| 中程度 (P\*4g〜N\*5g, 1g1f) | 6 手 | 322〜32,751 | Mate(11-15) |
| **Unknown (解けず)** | **L\*6g, B\*6g, B\*7g** | **7,167〜8,703** | **timeout** |
| 不詰 | P\*7g, N\*7g | 52,298〜122,029 | NoMate |
| 重い | N\*6g | 133,575 | Mate(11) |

個別合計ノード数: **405,821** (Unknown 3 件含む)．

**重要**: v0.22.0 時点の「23 応手すべて 200K 以内で解ける，合計 1.5M」という
記述は v0.24.44 では成立しない．**3 応手 (L\*6g, B\*6g, B\*7g) が 500K/15s では
解けず Unknown** となる．benchmarks.md 冒頭 (§10.2 先頭) の個別サブ問題の難度
記述は現状の measurement に基づいて再評価が必要である．

**Phase 1 実測結果 (v0.24.44, ply 24 OR node depth inflation):**

| Depth | Nodes | Time | 結果 | proven | disproven | **intermediate** | pns\_cycles |
|:-----:|------:|-----:|:----:|-------:|----------:|----------------:|:-----------:|
| **17** | **367,331** | 54.5s | **Mate(15)** ✓ | 8,121 | 143,053 | **75,358** | 1 |
| 21 | 6,021,120 | 120.7s | Unknown | 95,311 | 84,300 | **0** | 2 |
| 25 | 10,303,488 | 150.7s | Unknown | 229,487 | 159,176 | **0** | 3 |

v0.24.0 の「depth 21 = 4.1M Mate(21), depth 23 = 50M Unknown」に対し，
v0.24.44 では **cliff が depth 19〜21 に前進**している．depth=17 は同等
(367K Mate(15)) だが，depth=21 以降は大幅に悪化．

**診断結果 — gap の真の原因 (4 層構造):**

| 層 | 現象 | 診断データ |
|:---:|------|-----------|
| I | **IDS depth 切替時に intermediate エントリが全消去され中間進捗がリセット** | depth=17 で 75K intermediate → depth=21 で 0 |
| II | **`cross_deduce` と `prefilter` が合駒チェーンでほぼ発火しない** | cross\_deduce=0 @depth=17, prefilter\_hits=17〜45 |
| III | 結果として合駒チェーンの不詰証明が毎 IDS step でゼロから再構築される | disproven=143K→84K→159K (不安定) |
| IV | 個別サブ問題自体に想定より難しいものがある | 3/23 が 500K budget で Unknown |

**合駒最適化カウンタ遷移 (depth 依存):**

| カウンタ | depth=17 | depth=21 | depth=25 | 評価 |
|:---|---:|---:|---:|:---|
| `prefilter_hits` (§8.4) | 45 | 45 | 17 | 高 depth で減少 |
| `cross_deduce_hits` (§8.5) | **0** | 38 | 132 | **depth=17 でも 0** |
| `deferred_already_proven` | 4 | 0 | 7 | ほぼゼロ |
| `capture_tt_hits` (§5.4) | 20 | 0 | 0 | **depth≥21 で完全停止** |
| `refut_tt_hits` | 0 | 45,889 | 15 | depth=21 で異常スパイク |

特に重要な発見:

- **`cross_deduce_hits = 0` が depth=17 でも観測**: 同一マス駒種横断の
  証明転用 (§8.5) が実質機能していない．優先度の高い修復対象．
- **`capture_tt_hits = 0` at depth≥21**: インライン詰み検出で TT 内の
  proof を使う経路が，intermediate=0 状態では完全不能．
- **`refut_tt_hits` が depth=21 で 45,889 → depth=25 で 15 に急落**:
  depth=21 だけ REMAINING\_INFINITE disproof が残って refut cache が
  機能するが，depth=25 では消失．非常に不安定な挙動．

**MID プロファイル (全 depth 共通の支配項):**

`child_init` が 70-71% の CPU 時間を占める．内訳 `ci_do/undo_move` 7.5%,
`ci_inline` 3-6%, `ci_resolve` 5-6%．残りは TT lookup + 初期化ロジック．

##### 施策優先順位の組み替え

診断結果を踏まえ，「今後の方向性」(v0.24.33 時点) の優先順位を組み替える．
従来の提案で「サブ問題ウォームアップ」が最優先だったが，Phase 0b の実測で
**個別サブ問題の 3 件が 500K budget で Unknown** となり，ウォームアップ
単体では不十分であることが判明した．また intermediate=0 問題が
全施策の効果を阻害していることが明らかになった．

**最優先 (施策 I): IDS depth 切替時の intermediate 選択的保持 (新規)**

現在の `clear_working()` は IDS depth 昇格時に WorkingTT を全消去するため，
depth=17 で積み上げた 75K intermediate エントリが depth=19 以降で全て失われる．
これにより合駒チェーンの不詰部分証明が深い IDS step で再構築を強いられる．

- 実装方針: 課題 F (v0.24.38) の `clear_proven_disproofs_below(min_depth)` の
  **intermediate 版** を導入し，「現在 IDS depth に近い intermediate」の
  一部を次の IDS step に持ち越す
- 保持基準の候補: `remaining >= (next_ids_depth - 4)` や `amount >= threshold`
- soundness 検証: intermediate は pn>0, dn>0 の暫定値なので「次の step で
  再計算される」前提の保持．path\_dependent フラグは必須チェック
- 期待効果: Phase 1 の depth=21 で 6M Unknown → 数倍の改善見込み
- リスク: 保持した intermediate が次 step の閾値伝搬を阻害する可能性
  (§6.6.3 の amount ベース置換と相互作用)

**高優先 (施策 II): cross\_deduce の修復と駒種横断 disproof 共有の実装**

Phase 1 で **`cross_deduce_hits = 0` が depth=17 で観測** された事実は，
§8.5「同一マス証明転用」が実質的に機能していないことを示す．
これは 従来プロポーザル §「駒種横断 disproof 共有」の前提を覆す発見であり，
まず **既存の cross\_deduce 実装がなぜ 0 ヒットなのか** の原因究明が先決．

- Step 1: `cross_deduce_children` の呼び出し経路を確認し，発火条件
  (deferred\_children の更新順序など) を検証
- Step 2: 修復後，proof 転用に加えて **disproof 転用版 (hand\_lte 方向)**
  を実装．Zobrist XOR 差分近傍走査 (§6.6.4) を駒種差分に拡張
- 期待効果: 合駒チェーンの disproven エントリ (Phase 1 で 84K〜159K) の
  大部分が共有可能になり，disproof 生成コストを 数分の一 に削減

**中優先 (施策 III): IDS 段階刻みの細分化と Frontier 早期発火**

元の施策 4 を維持．施策 I の intermediate 保持と組み合わせることで
相乗効果がある．v0.24.40 の「IDS depth 32+ で +4 刻み」を更に細分化し，
22 → 24 → 26 → 28 → 30 → 32 → 35 → 38 → 41 とする．

**再評価 (施策 IV): サブ問題ウォームアップは単体不可，施策 I と統合する**

当初の最優先案 (サブ問題を個別解決してから統合探索) は，Phase 0b で
**個別サブ問題の 3 件が 500K で Unknown** となる事実から，単体では
効果が期待できない．ただし施策 I (intermediate 保持) と組み合わせれば，
ウォームアップで生成した intermediate が IDS step 間で保持され，
統合探索で有効に再利用できる可能性がある．単独プロジェクトとしては
降格し，施策 I の検証フェーズで併せて評価する．

**据え置き (施策 V): Aigoma 局所 Mini-solver**

元の施策 3．depth 非依存の独立 solver で合駒チェーンを local 解決する
アプローチは魅力的だが，施策 I-II の効果を見極めてから判断する．

**不採用 (元施策 6): pattern disproof cache**

Phase 1 の disproven エントリ数 (84K〜229K) が proven (95K〜229K) と同程度
であることから，「pattern cache で多くを共有」の前提が成立しない可能性が
高い．TT ベースの施策 II (駒種横断 disproof 共有) の方が既存インフラを
活用できる．

##### 次のアクション項目

1. **`cross_deduce_hits = 0` の原因調査** (`solver.rs:3559` 付近)．
   最も少ないコストで大きな知見が得られる可能性が高い
2. `clear_working()` / `retain_proofs_only()` の挙動確認と intermediate
   選択的保持の soundness 分析 (施策 I の実装準備)
3. L\*6g / B\*6g / B\*7g の個別再計測 (depth=14 設定が不適切かの確認)．
   Phase 0b の設定バグの可能性を排除する

#### v0.24.45 施策 I 実装: IDS depth 切替時の intermediate 選択的保持

v0.24.45 で `retain_working_intermediates(min_remaining, delta_remaining)` を
追加し，IDS depth 遷移時 (pns.rs:~1720) の `clear_working()` を置き換えた．

**保持条件**: `pn > 0 && dn > 0 && !path_dependent && remaining < REMAINING_INFINITE`
**remaining シフト**: 保持時に `new_remaining = old_remaining + delta` に更新し，
旧 IDS depth で計算された pn/dn を新 depth での下限値として安全に再利用する．

**Frontier サイクル境界**: 当初 `retain_proofs_and_intermediates()` で非 path-dep
intermediate も保持する案を試みたが，**`test_no_checkmate_gold_interposition` で
soundness 違反**が発生した．Frontier サイクル境界は PNS arena が破棄される境界で
あり，stale intermediate pn/dn を次サイクルに持ち越すと PNS の frontier 選択
(most-proving node) が誤誘導される．NoMate 証明で stale propagation が発散して
false Checkmate を返す経路が存在するため，Frontier 境界は従来通り
`retain_proofs()` で intermediate を全除去する方針を維持した．

**v0.24.45 測定結果 (施策 I のみ):**

| 指標 | v0.24.44 | v0.24.45 | Δ |
|------|---------:|---------:|:-:|
| Phase 1 depth=17 time | 54.49s | 52.09s | **-4.4%** |
| Phase 1 depth=21 nodes | 6.02M | 6.20M | +3% |
| Phase 1 depth=25 nodes | 10.30M | 9.40M | **-9%** |
| depth=25 TT overflow | 42,531 | 25,137 | **-41%** |
| depth=25 pns_arena_growth | 985K | 693K | -30% |
| depth=25 deferred_act(pns) | 344 | 572 | +66% |
| depth=25 deferred_already_proven | 7 | 19 | +171% |
| **depth cliff (Unknown→Mate)** | depth 19〜21 | **同じ** | **不変** |

**結論**: TT overflow -41%，deferred 再活性化 +66% などの副次的改善は確認されたが，
**depth cliff 自体は移動しなかった**．intermediate=0 問題の表層対応では真の
cliff 要因 (後述) を解消できない．

#### v0.24.46 ply 分布詳細調査: **真の cliff 要因は「depth 境界 thrashing」**

v0.24.46 で `test_tsume_39te_ply25_gap_diagnosis` に以下の追加計測を実装した:

- **cross_deduce funnel カウンタ** (solver.rs): `diag_cd_guard_and_drop`,
  `diag_cd_guard_child_proven`, `diag_cd_no_siblings`, `diag_cd_entered_main`
- **ply 分布の完全取得**: stderr 経由で `[ids_diag] ply_visits: ...` の全 IDS
  iteration 分を記録 (pns.rs:1560 の既存 verbose 出力を活用)

##### 観察 1: cross_deduce は「壊れていない」ことが判明

| 段階 | depth=17 | depth=21 | depth=25 |
|:---|---:|---:|---:|
| `cd_guard_and_drop` (AND + drop best) | 1,690 | 47,944 | 77,092 |
| `cd_guard_child_proven` (関数呼出) | 273 | 5,662 | 11,580 |
| `cd_entered_main` (本体ループ) | 273 | 5,657 | 11,579 |
| `cross_deduce_hits` (転用成功) | **0** | 33 | 75 |
| 転用成功率 (hits/entered) | 0% | 0.58% | 0.65% |

`cross_deduce_children` は depth=17 で **273 回 本体ループまで到達**している．
しかし転用成功数はゼロ．内側の兄弟ループで `cpn_j == 0` (既に証明済み) が
成立し全スキップされるため，転用が不要な状態である．

**結論: 施策 II (cross_deduce 修復) は不要**．関数は設計通り動作しており，
depth=17 での 0 hits は「MID 本体の証明伝搬が十分に速く補助が不要」という
正常動作．depth=21/25 での ~0.6% 成功率も低く，cross_deduce の改善で depth
cliff を突破するシナリオは見込めない．

##### 観察 2: **93% のノードが境界 4 ply に集中** (最重要発見)

各 Phase 1 solve の最終 IDS iteration で観測された ply 分布:

| Ply | depth=17 final | depth=21 final | depth=25 final |
|---:|---:|---:|---:|
| 0〜12 (浅部) | 21,628 | 25,272 | 4,043 |
| 13 | 42,312 | 33,889 | 4,115 |
| 14 | 26,011 | 29,595 | 3,379 |
| **15** | **140,857** | 200,444 | 12,185 |
| 16 | 51,967 | 134,260 | 11,836 |
| 17 | — | 886,891 | 72,776 |
| 18 | — | 497,510 | 49,095 |
| **19** | — | **2,752,869** | 311,505 |
| 20 | — | 1,312,754 | 178,405 |
| 21 | — | — | 1,132,864 |
| 22 | — | — | 774,469 |
| **23** | — | — | **4,388,266** |
| 24 | — | — | 1,891,596 |
| **Total** | **282,811** | **5,873,234** | **8,834,642** |

**3 つの決定的パターン:**

1. **境界 4 ply に ~93% 集中**:
   - depth=17: ply 13〜16 = 261K (92.3%)
   - depth=21: ply 17〜20 = 5,450K (92.8%)
   - depth=25: ply 21〜24 = 8,187K (92.7%)

2. **`ply = depth - 2` に ~50% 単独集中**:
   - depth=17: ply 15 = 49.8%
   - depth=21: ply 19 = 46.9%
   - depth=25: ply 23 = 49.7%

3. **浅い ply 0〜12 は depth 間でほぼ同一** (比率 0.55x〜1.42x):
   depth=17 から depth=21 で追加される ~5.45M ノードは，
   **すべて新たに開かれた ply 17〜20 の境界層**で発生している．

##### 根本原因の特定: Depth 境界 Thrashing

`ply = depth - 2` は chain aigoma の 2 手サイクル (合駒→捕獲) と一致し，
ここで `remaining = 2` となる．§8.1-§8.8 の合駒最適化が効果を発揮するには
**十分な remaining が必要**だが，境界層では remaining が小さすぎて部分解決に
留まり，再訪問の度に同じ計算を繰り返す:

| 層 | 現象 | 診断データ |
|:---:|---|---|
| α | `remaining ∈ [1,3]` で chain aigoma 変種が爆発的に生成 | ply 15/19/23 に 50% 集中 |
| β | depth-limited disproof は `remaining` が小さいほど厳密化が必要で，浅い depth の disproof が深い depth で再利用不可 | depth=17 の境界情報が depth=21 に役立たない |
| γ | PN_UNIT + depth-adaptive epsilon は `remaining` に応じて変化せず，境界層の thrashing を抑止できない | ply 19 で 2.75M visits |
| δ | v0.24.14 の rem=0 store 廃止により境界層の仮反証は cache されない | ply 19 visits = 同じ局面を 100 回以上再訪 |

**これまでの施策 (intermediate 保持・cross_deduce) では境界層の 93% 集中に
対処できない**ことが実データから確定した．cliff 突破には境界層での探索動作
自体を変更する施策が必要である．

##### 新施策候補の検討 (v0.24.46)

**施策 α: 境界層での chain aigoma 早期終了** (最有力)

`remaining ≤ N_threshold` かつ chain aigoma 検出 (`chain_bb_cache` 非空) の
AND ノードで，chain 合駒の列挙を段階的に抑制する．

- 実装箇所: `pns.rs:506` の `generate_chain_drops` 呼出前に `remaining` 条件を追加
- soundness 要件: 以下のいずれかの条件で safe:
  1. 生成を skip した分の chain 合駒が mate 経路の主要応手に含まれていない
  2. 生成 skip 後の AND ノードが dn=0 (defender が防ぎきれる) を返しても，
     それが depth-limited disproof として扱われる (false NoMate ではない)
- リスク: chain drop を skip することで「守備方が chain で防げた」mate を見落とし，
  false Checkmate を返す可能性
- 期待効果: ply 19 の 2.75M visits を 1/10 以下に削減

**施策 β: 境界層における visit cap + memoization**

`(position_key, hand, remaining)` 組で visits が N 回を超えたら memoized 返答に
ショートカット．v0.24.33 の `refutable_check_with_cache` と同じ発想の適用．

- 実装箇所: `mid()` 入口で position+hand+remaining をキーにした軽量カウンタ
- リスク: memoize key 空間の管理コスト，同一 key でも pn/dn が変化する場合の誤差
- 期待効果: thrashing の絶対的上限設定

**施策 γ: 境界層 branching factor 削減**

`remaining ≤ 3` で chain aigoma 検出時，§8.2 の 3 カテゴリ制限を更に 1 カテゴリへ
絞り込む．歩合のみ生成し角・桂を除外．

- 実装箇所: `pns.rs:543` の `generate_chain_drops`
- リスク: 除外した駒種が mate 経路に必要な場合の soundness 違反
- 期待効果: branching factor を 3 → 1 に削減

**施策 δ: PNS に境界層探索を責任転嫁**

depth-limited MID の ply ≈ depth-2 を PNS arena に移譲し，PNS の best-first
選択で thrashing を回避．Frontier Variant の元々の目的の延長．

- 実装箇所: IDS ループ内で境界層到達時に PNS サイクルへ切替
- リスク: PNS arena のメモリ圧迫
- 期待効果: PNS の `amount` ベースで同一ノード再訪を抑制

##### 施策 α 採用の根拠と soundness 検討

以下の 4 つの基準で施策 α を最優先候補とする:

1. **効果のポテンシャル**: 4 施策中最大 (ply 19 の 2.75M → 数百 K)
2. **実装範囲**: MID 内の局所的な条件分岐のみ
3. **soundness 証明**: chain_length vs remaining の比較で論理的に安全条件を導出可能
4. **測定容易性**: 既存の ply 分布カウンタで効果が即座に測定可能

**soundness 検証の論理:**

`chain_length = between.count()` (飛び駒と玉の間のマス数)
とすると，chain aigoma の 1 ステップ (drop + capture) は `2` ply を消費する．
defender が chain aigoma で防ぎきるには `2 * chain_length` ply が必要．

`remaining < 2 * chain_length` の場合，chain は必ず途中で depth-limit に到達する．
- 到達時点で mate 判定が未確定なら depth-limited disproof を返す (既存の挙動)
- この時点で chain drop を生成して再帰する意味は **薄い** (ただしゼロではない)

**リスク評価:**

- chain aigoma の 1 ステップが短い場合 (chain_length = 2 等) は
  `remaining = 3` でも 1 ステップ完全実行可能 → 単純な閾値だと sound 違反の恐れ
- 施策 α は **全面 skip ではなく「弱い駒 1 種のみ生成」で開始**し，
  段階的に厳密化する実験アプローチを採用する

##### 施策 α 実装計画 (v0.24.46+)

1. **ステップ 1 (最小介入)**: `remaining ≤ 2 && chain_bb_cache 非空` の MID AND
   ノードで，`generate_chain_drops` の生成結果を **強い駒 (飛車・角) のみに制限**
   する．歩合・桂合・銀合などの弱い駒を除外．
   - soundness: 弱い駒は強い駒で支配されるため (§8.2 前方利き系) skip 可能
   - 測定: Phase 1 depth=21/25 の ply 19/23 visits 削減率

2. **ステップ 2**: ステップ 1 で回帰なし & 効果あり → `remaining ≤ 3` に閾値拡大

3. **ステップ 3**: ステップ 2 で退行あり → 特定の chain_length のみに条件限定

各ステップで `test_tsume_39te_ply25_gap_diagnosis` を再実行し counter 遷移を
検証する．

#### v0.24.47 施策 α 実装と soundness 違反 (不採用)

v0.24.47 で施策 α を **`remaining <= 1 && chain_bb_cache 非空` の AND ノードで
chain マスへの drop を `ArrayVec::retain` で除去** する最小介入で実装した．

**初期観測 (soundness 違反発覚前):**

| Depth | Result | Nodes | Time | vs v0.24.46 |
|:---:|:---:|---:|---:|:---|
| 17 | Mate(15) ✓ | 367,332 | 54.2s | 不変 |
| 21 | Unknown | 5,582,848 | 120.7s | -7.3% |
| 25 | **Mate(15) ✓** | **158,199** | **35.0s** | **-98.5%** (apparent cliff 突破) |

depth=25 で 10.3M budget で Unknown だったものが 158K nodes で "Mate(15)" を
返し，一見 cliff 突破に見えた．非 ignored 128 テスト全 pass．新 regression test
`test_tsume_39te_ply24_mate15_regression_depth25` を `moves.len() == 15` のみ
verify する緩和形式で追加した．

**soundness 違反の発覚:**

depth=25 での root 初手が canonical の `5g6f` ではなく **`5g4f` (silver → 4f)**
に変化していた．当初「どちらも valid な discovered check で残り 14 手同一」と
誤解したが，ユーザ指摘により **`5g4f` 後に defender が `P*4g` (銀の裏に歩合)
で逃げ切れる**ことが判明．

検証テスト `test_tsume_39te_5g4f_p4g_escape_verify` (v0.24.49) で programmatic
に確認:

- 5g4f → P*4g 後の局面 (SFEN: `9/3+N1P3/7+R1/9/9/5S3/1R3p2k/3p5/9 b 2b4g3s3n4l15p 27`)
- クリーン codebase (施策 α revert 済み) で 5M / 300s budget で solve
- **結果: NoCheckmate (nodes=2,782,633)** → defender 逃げ切りが確定

つまり `5g4f` は canonical Mate(15) の valid な初手ではなく，施策 α が **false
Checkmate** を生成していた．v0.24.47 は cbbe8f5 で revert 済み．

##### 施策 α soundness 違反の根本原因分析

**汚染メカニズム (step-by-step):**

1. 施策 α filter は `!or_node && remaining <= 1 && !chain_bb.is_empty()` で発火
2. IDS は 2→4→8→16→24→25 で進行．各 IDS step で異なる depth の MID を実行
3. **浅い IDS step (depth=2, 4, 8, 16, 24)** の AND ply 1/3/7/15/23 で filter 発火
   (ply=depth-1 で rem=1 の AND ノード)
4. filter は chain マス (4g/5g/6g/7g) への drop を list から除去．本来の defender
   最善手 `P*4g` が列挙されなくなる
5. MID は残った defense moves (king moves + non-chain drops) のみを評価
6. 仮に全て mate に至るなら AND ノードは **`pn=0` (proven)** と判定される
7. proven entry は **ProvenTT に `remaining = REMAINING_INFINITE` でストア**される
   (`entry.rs:177` の `ProvenEntry::new_proof` 実装．proof は常に depth 非依存
   として扱われる)
8. 後続の IDS step (depth=4, 8, ..., 25) で同一 `(pos_key, hand)` を lookup すると
   **filter の無い文脈でも pn=0 が返る**．contamination 完成
9. depth=25 最終 IDS step の root 探索で 5g4f の子 AND が "既に proven" として
   短絡し，5g4f が最善手として採用される
10. 最終的に 5g4f → canonical PV の残り 14 手を組み合わせた "Mate(15)" が返る

**設計上の矛盾点:**

ProvenTT の proof entry は「mate 存在は depth に依存しない真」として
`REMAINING_INFINITE` で保存される (v0.24.26 Plan D，§6.6.5 参照)．この不変条件
下では **filter 条件下でのみ sound な proof** を保存すると必然的に汚染する．

これは benchmarks.md §10.2 の **「false NoMate バグ」(v0.24.18 commit 6e3efc3)**
のパターンの逆版 (false Checkmate) であり，設計ドキュメントが

> 浅い depth の confirmed disproof が深い depth を汚染する (NoMate バグ §6.6.3)

と警告していたものの **proof 側での再発**である．v0.24.18 は disproof 側の
対策として `clear_proven_disproofs_below()` を導入したが，**proof 側には同等の
対策が無い** (proof は depth 非依存として扱われるため)．

##### 施策 α の部分導入可能性の再検討

単純な「move filter で列挙を絞る」アプローチは ProvenTT の depth 非依存性と
本質的に矛盾する．しかし **filter の作用範囲を狭める / 異なる形で適用する**
ことで部分的な効果を得られる可能性がある．以下，6 つの variant を検討した:

**A-1: filter-tagged proof entries**

ProvenEntry に「filter 文脈下で生成された」フラグを追加し，IDS 深度切替時に
除去する．

- soundness: ✅ (汚染源を探知・除去できる)
- 実装コスト: **高** — ProvenEntry は 12 byte に密に pack されており (v0.24.26
  Plan D)，追加 bit を確保できない．構造体拡張でクラスタサイズ再計算が必要
- 判定: **不採用** (コスト/リスクが高すぎる)

**A-2: 最終 IDS 深度のみ filter 発火**

`self.depth == saved_depth_for_epsilon` の時のみ filter を有効化し，浅い
IDS step での汚染を回避する．

- soundness: ❌ — 最終 step で生成された false proof が PV 抽出 (`extract_pv_limited`)
  経由で誤った PV を返す．解決の時点で false Checkmate が確定する
- 判定: **不採用**

**A-3: move ordering のみへの適用 (chain drops を末尾へ移動)**

filter で除去せず，chain drops を move 配列の**末尾**に reorder する．MID の
threshold cutoff が先に non-chain moves で発火すれば chain drops は評価
されない．

- soundness: ✅ — AND ノードの `pn = sum(children.pn)` は全 child 評価が
  必要．threshold cutoff で途中 return する場合は `pn > 0` が返り，
  **false proven は発生しない** (MID は sum が threshold を超えたら早期 return
  するため，残り children 未評価でも "proven (pn=0)" にはならない)
- 効果: 不明．thrashing の本質的原因 (各 child 自体の rem=0/1 での
  繰り返し評価) は解消されない．**ply 分布改善は限定的**と予測
- 判定: **試験価値あり** (コスト低，リスクゼロ)

**A-4: DN heuristic inflation (初期値の水増し)**

chain drops の初期 dn を inflate して「disprove しにくい」child と見せかけ，
AND node の child 選択 (min dn) から外す．

- soundness: ✅ — 初期 dn は探索で上書きされる．ordering 効果のみで
  pn aggregation は影響なし
- 効果: A-3 より強い move ordering 効果．ただし A-3 同様 thrashing 自体は解消
  しない可能性
- 判定: **試験価値あり** (コスト低，A-3 と併用可能)

**A-5: 局所 specialized resolver (one-shot 評価)**

AND ノードで境界層 (`remaining <= 2` + chain aigoma 検出) の場合，通常の MID
再帰ではなく **専用の一括評価ルーチン**を呼ぶ．結果は ProvenTT に保存**しない**
(一時的な値として使う)．

- soundness: ✅ — 局所計算を TT に持ち越さなければ汚染しない
- 効果: 境界層の thrashing を根本的に解消．各 chain drop の評価を 1 回だけ行う
- 実装コスト: **中** — 専用ルーチンと TT store skip 経路が必要．既存 MID
  インターフェースとの整合性検証が必要
- 判定: **最有力候補** (施策 α の本来の意図を soundness を保ったまま実現できる)

**A-6: 施策 δ (PNS 責任転嫁) への統合**

境界層探索を PNS に委ね，PNS の `amount` ベース eviction で thrashing を
自然に抑制する．

- soundness: ✅ — PNS は df-pn とは別アルゴリズムで arena 管理．汚染ルートが無い
- 実装コスト: **中〜高** — IDS ループで境界層到達を検出し PNS 経路へ切替
- 判定: **独立施策として並行検討**．施策 δ 単体で評価

##### 施策 α 部分導入の結論

| Variant | soundness | 効果見込み | コスト | 推奨 |
|:---:|:---:|:---:|:---:|:---:|
| A-1 tagged proof | ✅ | 高 | 高 | — |
| A-2 最終 IDS のみ | ❌ | — | — | 不採用 |
| A-3 move reorder | ✅ | 低〜中 | 低 | **試験** |
| A-4 DN inflation | ✅ | 中 | 低 | **試験** |
| A-5 局所 resolver | ✅ | 高 | 中 | **最有力** |
| A-6 施策 δ 統合 | ✅ | 高 | 中〜高 | **並行検討** |

**推奨順**:

1. まず **A-3 + A-4** を組み合わせた「境界層 chain drop 優先度降下」を
   試験する．soundness-safe で実装コスト最小．効果が限定的でも後続施策の
   評価基準として有用
2. 次に **A-5 局所 resolver** を実装．施策 α の本来の意図 (境界層 chain
   aigoma 早期終了) を soundness を保ったまま実現する本命施策
3. **A-6 施策 δ** は独立方針として並行検討．境界層対策の代替案として
   位置付け

**他手法との調整による改善可能性:**

施策 α の失敗は「単独 filter で列挙を絞る」アプローチの限界を示した．他の
既存手法との組合せで境界層 thrashing を間接的に緩和する方向も検討する:

- **施策 I (v0.24.45 intermediate 保持)** は IDS 境界の情報損失を緩和したが
  cliff 不変だった．A-5 の局所 resolver 結果を「一時キャッシュ」として
  施策 I の retention スキーム下で保持すれば相乗効果の可能性
- **§8.3 遅延展開** の `deferred_children` に境界層専用の activation order
  を追加 (例: rem<=2 では chain drop の activation を保留し，non-chain
  children が全て解決するまで評価を遅らせる)．これは §8.5 cross_deduce の
  補完としても機能
- **§3.1 depth-adaptive epsilon** を境界層限定で更に絞り込む (例: rem<=2
  の AND で `eps_denom = 1`)．閾値伝搬の特性変化で thrashing を緩和

##### 次のアクション項目 (v0.24.49)

1. **benchmarks.md §10.2 の本セクション記述済み** (v0.24.47 失敗の正式記録
   および施策 α 部分導入の再評価)
2. **A-3 + A-4 (move ordering + DN inflation)** の最小試験実装を検討
3. **A-5 局所 resolver** の設計検討 (TT store skip 経路 + interface 定義)
4. **A-6 施策 δ** の独立設計検討 (§11.7 Frontier Variant の拡張として
   定式化できるか検討)

### 10.3 ミクロコスモス(1525手詰)の解法比較

| ソルバー | 解答時間 | 主要手法 |
|---------|---------|---------|
| 脊尾詰 (1997) | ~20時間 | PN*，~188M 局面 |
| KomoringHeights | ~10分 | df-pn+, SNDA, 証明駒/反証駒, GHI 対策, 合駒遅延展開 |
| shtsume | ~1分 | 不明(最速ソルバー) |
| やねうら王 | 解けない | TT GC 未実装(~3.5TB 必要) |
| maou_shogi | 未挑戦 | — |

---

