# Df-Pn 最適化手法リサーチ

## 概要

maou_shogiの詰将棋ソルバー(Df-Pn)の最適化に向けて，
主要論文とミクロコスモス(1525手詰)解法の手法群を調査した結果をまとめる．

## 調査した論文・ソルバー

| 文献/ソルバー | 年 | 概要 |
|-------------|------|------|
| Nagai 2002 | 2002 | Df-Pn原論文(東大博士論文) |
| Kishimoto & Müller 2004/2005 | 2004-2005 | GHI対策，閾値制御，完全性証明 |
| Pawlewicz & Lew 2007 | 2007 | 1+εトリック(乗算型閾値) |
| Kishimoto 2010 | 2010 | SNDA(過大評価対策) + TCA(過小評価対策) |
| Kishimoto et al. 2012 | 2012 | Proof Number Search 20年のサーベイ |
| Song Zhang et al. 2017 | 2017 | Deep df-pn(深さ依存初期値) |
| NeurIPS 2019 | 2019 | DFPN-E(エッジコスト型ヒューリスティック) |
| KomoringHeights | 2020- | ミクロコスモス10分で解くソルバー |
| shtsume | 2020- | ミクロコスモス1分で解くソルバー |

## 手法の分類と詳細

### 1. 閾値制御(Threshold Management)

#### 1.1 標準 df-pn (Nagai 2002)

ORノードで子c1を選択する際の閾値:

```
pn_threshold(c1) = min(pn_threshold(parent), pn2 + 1)
dn_threshold(c1) = dn_threshold(parent) - sum_dn + dn(c1)
```

`pn2`は2番目に小さいpn．`+1`により，c1のpnがpn2を超えた瞬間に
他の子に切り替わる → **seesaw effect(スラッシング)**の原因．

#### 1.2 df-pn+ (Nagai 2002)

`+1`を定数δ(>1)に変更:
```
pn_threshold(c1) = min(pn_threshold(parent), pn2 + delta)
```
加算型で一定のスラッシング抑制効果があるが，pnが大きい領域では不十分．

#### 1.3 1+εトリック (Pawlewicz & Lew 2007)

**乗算型**に変更:
```
pn_threshold(c1) = min(pn_threshold(parent), ceil(pn2 * (1 + epsilon)))
```

- pnが小さい時: 小さな増分(細かい制御)
- pnが大きい時: 大きな増分(深い探索を許容)
- TTオーバーフロー時に効果が劇的(再展開頻度を大幅削減)

**maou_shogi:** 実装済み(v0.11.0，v0.18.4 で論文式に修正)．
`ceil(second_best * (1 + ε))` (ε = 1/2) を使用．
整数演算では `(second_best * 3 + 1) / 2` で ceiling division を実現．
初期 pn が 1-5 と小さいため，論文推奨値(ε = 1/4)より大きい値を採用．

#### 1.4 TCA (Kishimoto & Müller 2008)

循環グラフでの**過小評価**を修正するアルゴリズム．
df-pnは有限DCG(有向循環グラフ)上で不完全だが，TCAを加えると完全になる．

**maou_shogi:** 実装済み(v0.14.0)．
OR ノードでループ子(path 上の子)が存在する場合，MID ループ閾値と
子ノード閾値を加算的に拡張する(約25%: `threshold / 4 + 1`)．
乗算的拡張(2×)は再帰で指数膨張するため，加算的拡張を採用．
AND ノードではループ子が即時反証を引き起こすため拡張不要．

### 2. 初期 pn/dn の設定

#### 2.1 標準初期化

全リーフを`(pn=1, dn=1)`で初期化．

#### 2.2 df-pn+ ヒューリスティック初期化

位置の特徴に基づいて初期pn/dnを設定:
- 玉の逃げ場が少ない → pn小(詰みやすい)
- 王手手段が多い → dn大(反証しにくい)

KomoringHeights v0.4.0で導入し大幅な性能改善を報告．

**maou_shogi:** 実装済み(v0.12.0)．AND子(OR局面)は`heuristic_or_pn`で
玉の逃げ場に基づく初期pnを設定，OR子(AND局面)は`heuristic_and_pn`で
応手数・逃げ場に基づく初期pnを設定．dnは`depth_biased_dn`(深さ依存)．

#### 2.3 DFPN-E エッジコスト型 (NeurIPS 2019)

リーフではなく**エッジ(親→子遷移)**にヒューリスティックコストを付与．
df-pn+より効果が高いと報告(子ノードの質ではなく，手の質で評価)．

**maou_shogi:** 実装済み(v0.13.0)．
手の質(成+取，取，成，静か手)と玉との距離に基づくエッジコストを
初期 pn に加算する．OR ノードの王手には `edge_cost_or`，
AND ノードの応手には `edge_cost_and` を適用．

#### 2.4 Deep df-pn (Song Zhang et al. 2017)

深さに応じて初期値を調整:
```
pn_init = 1
dn_init = max(1, ceil(R * depth))    # R=0.4が最適
```

深い位置ほど反証コストを高くし，浅い解を優先．
~50%のイテレーション削減，~35%のノード削減(Othello/Hexで実験)．

**maou_shogi:** 実装済み(v0.12.0)．
`DEEP_DFPN_R = 0.5` で `dn_init = max(base, ceil(R * ply))` を使用．
論文推奨値(R=0.4)より積極的に深い位置の反証コストを引き上げる．

### 3. 転置表(TT)管理

#### 3.1 持ち駒優越(Hand Dominance)

盤面が同一で持ち駒が異なる局面間の包含関係を利用:
- 証明(pn=0): 攻め方の持ち駒が記録より多い → 再利用可
- 反証(dn=0): 攻め方の持ち駒が記録より少ない → 再利用可

Nagai 2002で導入．300手超の問題を解く鍵となった．

**maou_shogi:** 実装済み(Pareto frontier管理)．

#### 3.2 証明駒/反証駒 (Proof/Disproof Pieces)

詰み/不詰みに必要な**最小限の持ち駒セット**を追跡．
証明駒が少ないほど，より多くの局面でTTヒットが発生する．

**maou_shogi:** 実装済み．

#### 3.3 TTガベージコレクション

TTが満杯になった際に不要エントリを回収する．
YaneuraOuはGC未実装のためミクロコスモスに~3.5TB必要(解けない)．
GPS Shogiが実装したが「読むのがほぼ不可能」とされる複雑さ．

**maou_shogi:** 実装済み(v0.12.0)．
`tt_gc_threshold` 設定時に100Kノードごとにサイズチェックし，
閾値超過時に容量の75%までGCを実行する．
デフォルトは0(無効)．`retain_proofs()`による反復深化間の清掃も併用．

#### 3.4 TT Best Move保存

TTエントリに最善手と詰み手数を保存し，PV復元を高速化．
KomoringHeights v0.4.0で導入．

**maou_shogi:** ✅ 実装済み(v0.16.0)．`DfPnEntry` に `best_move: u16` フィールドを追加．
MID ループの中間結果保存時に `store_with_best_move` で最善子の Move16 を記録．
`look_up_best_move` で TT ヒット時に取得し，動的手順改善(§7.1)に使用する．

### 4. DAG/ループ対策

#### 4.1 SNDA (Kishimoto 2010)

DAG(転置)によるpn/dnの**過大評価**を検出・修正:
- TTの親ポインタを追跡し，共通祖先を検出
- 過大評価検出時，sumの代わりにmaxを使用

df-pn + TCA + SNDAは他のソルバーで解けなかった問題を解いた．

**maou_shogi:** 積極的 SNDA 実装済み(v0.15.0，v0.11.0 で保守的方式を導入)．
TT エントリに `source` フィールドを追加し，同一 source グループの
重複分を控除する方式．v0.15.0 で積極的 max 集約に移行:
`deduction = sum(group) - max(group)` により，グループ内で最大値のみを残す．
TCA(過小評価対策)との併用により過小評価リスクを緩和する．

#### 4.2 GHI問題対策 (Kishimoto & Müller 2004/2005)

同一局面が異なる経路で異なる結果を持つ問題．
KomoringHeightsはdual TT(base/twin)で経路依存/非依存の不詰を区別．

**maou_shogi:** 経路依存フラグ付き反証で GHI を緩和(v0.15.0)．
`path`(FxHashSet)によるループ検出に加え，ループ検出に由来する反証を
`path_dependent = true` かつ有限 `remaining` で TT に保存する．
これにより経路依存の反証は異なる深さの探索で自動的に再評価される．
KomoringHeights の dual TT(base/twin)方式ほど完全ではないが，
経路依存の反証が TT を永続的に汚染する問題を軽減する．

#### 4.3 Virtual Proof Number (Saito et al. 2006)

AND ノードの pn 計算で `sum(pn_children)` の代わりに，
証明済み子を除外した「仮想証明数」を使用する．
これにより AND ノードの pn がより正確になり，手選択が改善される．

**maou_shogi:** 実装済み(v0.15.0)．
AND ノードの子収集ループで証明済み子(cpn=0)を証明駒蓄積後に
`continue` で除外し，pn 合計および子選択から排除する．
証明済み子は pn=0 で sum に影響しないが，SNDA ペア収集と
子選択ループのスキップにより効率化される．

#### 4.4 Weak Proof Number (Ueda et al. 2008)

証明数の**二重計数問題**(double-counting problem)に対処する手法．
DAG 構造の探索木において，共有ノードが複数の親から重複してカウントされ
証明数が過大評価される問題を，証明数と分岐係数(branching factor)を
組み合わせた探索推定量で解決する．
SNDA が同一 source の重複を控除するのに対し，WPN は分岐構造自体を
推定に組み込む点で異なるアプローチを取る．

将棋と Othello での実験で，標準の証明数探索やその深さ優先変種より
高い性能を示した．

**maou_shogi:** v0.17.0 で実装済み．AND ノードの証明数計算を
`sum(pn_children)` から `max(pn_children) + (unproven_count - 1)` に変更．
VPN(証明済み子除外)および SNDA(DAG 合流補正)と併用．

### 5. 合駒(中合い)最適化

#### 5.1 無駄合い枝刈り

紐のない合駒(futile interposition)をスキップ．

**maou_shogi:** 実装済み(futile/chain分類)．

#### 5.2 合駒遅延展開 (KomoringHeights v0.5.0)

合駒の展開自体を遅延させ，不要な分岐を削減．

**maou_shogi:** 実装済み(v0.12.0)．
AND ノードの合駒(drop)を `deferred_children` に分離し，
非合駒応手の探索後に活性化する完全な遅延展開を実装．
`LAZY_INTERPOSE_THRESHOLD = 8` 未満の場合は即座に合流．
合駒事前証明(interpose pre-solve)も併用．

### 6. 静的詰め判定

リーフノードで短手数の詰みを検出し，Df-Pnオーバーヘッドを回避．

**maou_shogi:** 実装済み(1手/3手インライン + budget制N手詰め)．

### 7. 手順改善(Move Ordering)

#### 7.1 動的手順改善

TT の Best Move ヒントや killer move 等を利用し，OR ノードの王手順序を
動的に改善する．初期の静的な手順(DFPN-E のエッジコスト)に加え，
探索中に得られた情報で手順を更新する．

alpha-beta 探索では killer move heuristic と history heuristic が
手順改善の標準手法として広く使われている．
証明数探索での利用は研究例が少ないが，OR ノードで有望な王手を
先に試すことで探索木の早期枝刈りが期待できる．
TT Best Move 保存(§3.4)との併用が前提となる．

**maou_shogi:** ✅ 実装済み(v0.16.0, v0.18.3 で Killer Move 追加)．
TT Best Move(§3.4)を利用した動的手順改善に加え，
Killer Move Heuristic を OR ノードに導入(v0.18.3)．
手順の優先度: TT Best Move > Killer Move(2スロット/ply) > 静的手順．
Killer Move は OR ノードの証明達成時および閾値超過時に記録し，
同一 ply の別の局面でも優先的に探索する．

### 8. 反復深化(Iterative Deepening)

#### 8.1 IDS-dfpn

探索深さ制限を段階的に増加させる反復深化方式．
浅い解を確実に先に発見し，TT に浅い証明を蓄積させる．
PN*(Seo et al. 2001)が反復深化型証明数探索の先駆で，
深さ優先の PN* から PDS(Nagai 1999)，df-pn(Nagai 2002)へと発展した．

反復深化のオーバーヘッド(ノード再展開)は詰将棋では約20%と報告されており，
十分実用的な水準にある．浅い証明を TT に蓄積することで
深い探索での TT ヒット率が向上し，総探索量を削減できる．

**maou_shogi:** ✅ 実装済み(v0.16.0)．`solve()` 内で自動的な深さ段階化を実装．
偶数ステップ(2, 4, 8, 16, ..., depth)で倍増しながら反復深化する．
各浅い反復にはノード予算(total の 1/16)を割り当て，最終反復にノードを温存する．
反復間で `retain_proofs()` により証明エントリのみを引き継ぎ，
浅い深さの仮反証・中間値の汚染を防止する．
`find_shortest` モードでの PV 長制限による2パスの反復深化も併用している．

### 9. 探索フレームワーク変種

#### 9.1 Best-First PN Search / PN*

df-pn の深さ優先制約を緩和し，best-first 的に探索する変種．
PN*(Seo, Iida & Uiterwijk 2001)は Korf の RBFS アルゴリズムに基づく
反復深化型の証明数探索で，詰将棋 295 問のテストで最高性能を示した．

メモリ消費が増大するが，同一ノードの再展開(thrashing)を完全に回避できる．
空間計算量は O(d)(d = 最大探索深さ)で，探索パス上のノードのみを保持する．

**maou_shogi:** v0.18.0 で実装済み．`solve()` の Phase 1 として Best-First PNS を実行し，
アリーナ上限(2M ノード)到達時は Phase 2 で IDS-dfpn (MID) にフォールバックする．
明示的な探索木上で most-proving node を選択・展開し，グローバルに最適なノード選択を行う．
TT を子ノード初期化と転置検出に活用し，既存の全最適化(WPN, VPN, SNDA, DFPN-E,
Deep df-pn, ヒューリスティック初期化, 静的詰め判定)と併用する．

### 10. 並列化(Parallelization)

#### 10.1 並列 df-pn (Kaneko 2010)

複数スレッドで異なるサブツリーを同時に探索する．
Kaneko (2010, AAAI) は共有メモリ環境での df-pn 並列化を提案し，
共有転置表を用いた複数エージェントの自律的探索を実装した．

各ノードに**仮想証明数/反証数**(virtual proof/disproof numbers)を導入し，
探索中のエージェント数を将来の証明数増加の推定に利用することで
エージェント間の効率的な協調を実現する．
詰将棋の大規模問題で 8 スレッドで 3.58 倍の高速化を達成(並列効率 ~0.5)．

Pawlewicz & Hayward (2014) の SPDFPN は Hex で並列効率 ~0.7 を達成し，
Hoki, Kaneko, Kishimoto & Ito (2013) は分散メモリ環境への拡張を提案した．

**maou_shogi:** 未実装．TT のスレッドセーフ化と負荷分散が必要．

## ミクロコスモス(1525手詰)の解法比較

| ソルバー | 解答時間 | 主要手法 |
|---------|---------|---------|
| 脊尾詰 (1997) | ~20時間 | PN*，~188M局面 |
| KomoringHeights | ~10分 | df-pn+, SNDA, 証明駒/反証駒, GHI対策, 合駒遅延展開 |
| shtsume | ~1分 | 不明(最速ソルバー) |
| やねうら王 | 解けない | TT GC未実装(~3.5TB必要) |

## maou_shogi 実装状況と改善ロードマップ

### 実装済み手法

| 手法 | 状態 | 期待効果 | 備考 |
|------|------|---------|------|
| 1+εトリック(乗算型) | ✅ 実装済み | 大 | `epsilon = second_best / 4 + 1` |
| df-pn+(ヒューリスティック初期pn/dn) | ✅ 実装済み | 中〜大 | 玉の逃げ場ベース |
| Deep df-pn(深さ依存初期値) | ✅ 実装済み | 中 | R=0.5 |
| SNDA(過大評価対策) | ✅ 実装済み | 大(長手数) | 積極的 max 集約(v0.15.0，v0.11.0 で導入) |
| TT GC | ✅ 実装済み | 必須(超長手数) | 100Kノード周期(v0.12.0) |
| 合駒遅延展開 | ✅ 実装済み | 中 | deferred_children + Pre-Solve |
| DFPN-E(エッジコスト型) | ✅ 実装済み | 中〜大 | 手の質ベース(v0.13.0) |
| TCA(過小評価対策) | ✅ 実装済み | 中〜大 | 加算的閾値拡張(v0.14.0) |
| 静的詰め判定 | ✅ 実装済み | 中 | 1手/3手インライン + N手予算付き |
| 持ち駒優越(Hand Dominance) | ✅ 実装済み | 大 | Pareto frontier管理 |
| 証明駒/反証駒 | ✅ 実装済み | 大 | Nagai 2002準拠 |
| VPN(Virtual Proof Number) | ✅ 実装済み | 中 | AND 証明済み子除外(v0.15.0) |
| GHI対策(経路依存フラグ) | ✅ 実装済み | 正確性+効率 | path\_dependent + 有限 remaining(v0.15.0) |
| IDS-dfpn(自動深さ段階) | ✅ 実装済み | 中 | 倍増ステップ + retain\_proofs(v0.16.0) |
| TT Best Move保存 | ✅ 実装済み | 中 | DfPnEntry に best\_move: u16(v0.16.0) |
| 動的手順改善 | ✅ 実装済み | 中〜大 | TT Best Move を OR ノードで先頭にスワップ(v0.16.0) |
| WPN(Weak Proof Number) | ✅ 実装済み | 中 | max(pn) + (unproven\_count - 1) で二重計数緩和(v0.17.0) |
| Best-First PNS | ✅ 実装済み | 大 | Phase 1 PNS + Phase 2 MID フォールバック(v0.18.0) |

### 未実装手法(効果/コスト比の高い順)

| 優先度 | 手法 | 節 | 期待効果 | 実装コスト | 備考 |
|--------|------|---|---------|-----------|------|
| 1 | 並列 df-pn | §10.1 | 大 | 高 | TT スレッドセーフ化 + 負荷分散 |

## 参考文献

### 論文

- Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002)
- Nagai, "Df-pn Algorithm for Searching AND/OR Trees and Its Applications" (Ph.D. Dissertation, UTokyo, 2002)
- Seo, Iida & Uiterwijk, "The PN*-search algorithm: Application to tsume-shogi" (AI 129, 2001)
- Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)
- Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008)
- Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation of Depth-First Proof-Number Search" (AAAI 2010)
- Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)
- Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)
- DFPN-E: "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)
- Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)
- Kishimoto, Winands, Müller & Saito, "Game-tree search using proof numbers: The first twenty years" (ICGA Journal 35, 2012)
- Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010)
- Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014)
- Hoki, Kaneko, Kishimoto & Ito, "Parallel Dovetailing and its Application to Depth-First Proof-Number Search" (ICGA Journal 36, 2013)
- Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004)
- Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)

### 日本語リソース

- [やねうら王 - 詰将棋アルゴリズムdf-pnのすべて](https://yaneuraou.yaneu.com/2024/05/08/all-about-df-pn/)
- [やねうら王 - ミクロコスモスは解けますか？](https://yaneuraou.yaneu.com/2020/12/30/yaneuraou-matesolver-microcosmos/)
- [コウモリのちょーおんぱ - df-pnアルゴリズムの解説](https://komorinfo.com/blog/df-pn-basics/)
- [コウモリのちょーおんぱ - KomoringHeightsを作った](https://komorinfo.com/blog/komoring-heights/)
- [コウモリのちょーおんぱ - GHI問題対策](https://komorinfo.com/blog/and-or-tree-ghi-problem/)
- [コウモリのちょーおんぱ - 証明駒／反証駒の活用方法](https://komorinfo.com/blog/proof-piece-and-disproof-piece/)
- [コウモリのちょーおんぱ - KomoringHeights v0.4.0](https://komorinfo.com/blog/komoring-heights-v040/)
- [コウモリのちょーおんぱ - KomoringHeights v0.5.0](https://komorinfo.com/blog/komoring-heights-v050/)
- [すぎゃーんメモ - Rustでつくる詰将棋Solver](https://memo.sugyan.com/entry/2021/11/11/005132)
- [Qhapaq - 高速な詰将棋アルゴリズムを完全に理解したい](https://qhapaq.hatenablog.com/entry/2020/07/19/233054)
- [人工知能学会誌 - 詰将棋探索技術(PDF)](https://www.jstage.jst.go.jp/article/jjsai/26/4/26_392/_pdf)
