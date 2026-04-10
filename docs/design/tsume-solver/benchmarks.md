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
`test_tsume_39te_backward_{1m,10m,120m}` (tests.rs, `#[ignore]`),
`test_tsume_39te_profile_{depth_scan,ply20_timeline}` (tests.rs, `#[ignore]`, v0.24.32+)

**現状 (v0.24.33):** PNS の NM 昇格判定を TT ベース + memoize のハイブリッド
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

ply 25 AND ノードの 23応手を個別に探索すると**すべて 200K ノード以内で解ける**:

| カテゴリ | 応手 | ノード数 | 結果 |
|---------|------|---------|------|
| 即詰み | P/L/N/S/G/B\*2g | 各1 | Mate(1) |
| 3〜5筋合駒 | P\*3g 〜 N\*5g (9手) | 41〜80K | Mate(7〜15) |
| 6〜7筋合駒 | L\*6g, B\*6g, B\*7g, N\*6g | 71K〜83K | Mate(13〜15) |
| 玉逃げ | 1g1f | 140K | Mate(19) |
| PV(最長) | 1g1h | 103 | Mate(13) |
| 不詰 | P\*7g, N\*7g | 各~200K | NoMate |

最も重い応手は P\*7g / N\*7g の不詰証明(各 ~200K ノード)であり，
23応手の合計は約 **1.5M ノード**に収まる．

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

**39 手詰めが未解決の理由:**

1. **探索空間の深さ**: ply 24 のチェーン合駒で指数的分岐．
   サブ問題は 333K で解けるがルートからの 24 レベルで予算が不足
2. **TT クラスタ飽和**: チェーン合駒の駒種バリアントが hand 多様性を増大
3. **ノード予算**: テスト(test\_tsume\_39te\_aigoma)が 10M / 60s と小さい．
   29 手詰めが 74M を要したことから 39 手詰めは数百 M が必要と推定

**採用済み手法の一覧:**

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

#### 今後の改善方針

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

**今後の方向性:**

TT の構造的課題(overflow，NPS)は大幅に改善されたため，
残る律速要因は**探索アルゴリズムレベルの改善**:

- **合駒の不詰���明の効率化**: 異なる合駒種での不詰パターンの共有(hand\_gte の活用強化)
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

ply 20 以降を解くには次の改善が必要と推定していた:

- **NPS の根本的改善**: ply 20 で 6 kn/s は異常．move generation,
  TT lookup, check generation のいずれかが指数的に重くなっている
- **合駒チェーンの不詰証明共有**: 異なる駒種で同一不詰パターンが繰り返される
  ケースを `hand_gte_forward_chain` で吸収する深さ・幅両方向の拡張
- **段階的 IDS depth**: depth=41 一括ではなく depth=32→41 の段階解法で
  Frontier Variant の MID 停滞検出を早く発動させる
- **探索分割**: ply 24 サブ問題のような中間目標で TT ウォームアップ後に
  ルートから解き直すアプローチ

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

##### 今後の方向性 (v0.24.33 以降)

ply 14 以降を解くには次の改善が必要:

- **ノード予算の拡大**: ply 14 は 120M で Unknown．ply 20 → 18 → 16 の
  コスト増大率が ×1.2〜1.3 で緩やかなので，数百 M 程度で ply 12〜10 まで
  到達する見込み．ただしルート (ply 0) には更に 7 ply 分の深さがある
- **合駒チェーンの不詰証明共有**: 依然律速要因．`hand_gte_forward_chain`
  の更なる拡張または inter-ply proof reuse
- **段階的 IDS depth**: depth=41 一括ではなく段階解法
- **探索分割**: ply 24/22 のサブ問題を先行解決し TT ウォームアップ後に
  ルートから解き直すアプローチ

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

**課題 F: IDS depth 切替時の ProvenTT disproof 保持 (v0.24.38)**

v0.24.38 で IDS depth 切替時の `clear_proven_disproofs()` 呼び出しを除去した．

**変更前:** IDS 反復間で ProvenTT の confirmed disproof を全除去（NoMate バグ対策）．
`all_checks_refutable_by_tt` の TT 経路ヒット率が 0.03% と事実上死蔵していた．

**変更後:** ProvenTT の confirmed disproof を IDS depth 切替間で保持する．

**安全性の根拠:**

- ProvenTT に格納される disproof は `is_proven_entry()` の条件
  (`dn=0`, `!path_dependent`, `remaining=REMAINING_INFINITE`) を満たすもののみ
- これらは `mid_fallback` の NM 昇格時に `depth_limit_all_checks_refutable()` で
  position 依存(depth 非依存)の完全検証を経ている
- `depth_limit_all_checks_refutable()` は depth=5 の再帰で全王手が反証可能かを
  盤面状態のみから判定し，IDS depth に依存しない
- v0.24.33 で `refutable_check_with_cache` のハイブリッド判定が導入され，
  NM 昇格の精度が大幅に向上している

**期待効果:**

- `all_checks_refutable_by_tt` TT 経路の活性化 (0.03% → 有意なヒット率)
- PNS NM 昇格判定の高速化（TT ヒットで memoize/再帰フォールバック回避）
- IDS の深い反復で浅い反復の NM 証明を再利用

### 10.3 ミクロコスモス(1525手詰)の解法比較

| ソルバー | 解答時間 | 主要手法 |
|---------|---------|---------|
| 脊尾詰 (1997) | ~20時間 | PN*，~188M 局面 |
| KomoringHeights | ~10分 | df-pn+, SNDA, 証明駒/反証駒, GHI 対策, 合駒遅延展開 |
| shtsume | ~1分 | 不明(最速ソルバー) |
| やねうら王 | 解けない | TT GC 未実装(~3.5TB 必要) |
| maou_shogi | 未挑戦 | — |

---

