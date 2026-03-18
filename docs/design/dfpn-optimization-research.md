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

**maou_shogiとの差分:**
maou_shogiは `epsilon = (second_best / 16).min(256) + 1` を使用．
pn≥4096で事実上+257の固定加算になり，乗算型の利点が失われる．

#### 1.4 TCA (Kishimoto & Müller 2008)

循環グラフでの**過小評価**を修正するアルゴリズム．
df-pnは有限DCG(有向循環グラフ)上で不完全だが，TCAを加えると完全になる．

**maou_shogiとの差分:** 未実装．

### 2. 初期 pn/dn の設定

#### 2.1 標準初期化

全リーフを`(pn=1, dn=1)`で初期化．

#### 2.2 df-pn+ ヒューリスティック初期化

位置の特徴に基づいて初期pn/dnを設定:
- 玉の逃げ場が少ない → pn小(詰みやすい)
- 王手手段が多い → dn大(反証しにくい)

KomoringHeights v0.4.0で導入し大幅な性能改善を報告．

**maou_shogiとの差分:**
現在は応手数/王手数で初期化(`(n, n)`)しているが，
位置の質に基づく差別化はしていない．

#### 2.3 DFPN-E エッジコスト型 (NeurIPS 2019)

リーフではなく**エッジ(親→子遷移)**にヒューリスティックコストを付与．
df-pn+より効果が高いと報告(子ノードの質ではなく，手の質で評価)．

**maou_shogiとの差分:** 未実装．

#### 2.4 Deep df-pn (Song Zhang et al. 2017)

深さに応じて初期値を調整:
```
pn_init = 1
dn_init = max(1, ceil(R * depth))    # R=0.4が最適
```

深い位置ほど反証コストを高くし，浅い解を優先．
~50%のイテレーション削減，~35%のノード削減(Othello/Hexで実験)．

**maou_shogiとの差分:** 未実装．

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

**maou_shogiとの差分:** 探索中のGCは未実装．
`retain_proofs()`で反復深化間の清掃のみ実施．

#### 3.4 TT Best Move保存

TTエントリに最善手と詰み手数を保存し，PV復元を高速化．
KomoringHeights v0.4.0で導入．

**maou_shogiとの差分:** 未実装(PV復元は再探索で実施)．

### 4. DAG/ループ対策

#### 4.1 SNDA (Kishimoto 2010)

DAG(転置)によるpn/dnの**過大評価**を検出・修正:
- TTの親ポインタを追跡し，共通祖先を検出
- 過大評価検出時，sumの代わりにmaxを使用

df-pn + TCA + SNDAは他のソルバーで解けなかった問題を解いた．

**maou_shogiとの差分:** 未実装．

#### 4.2 GHI問題対策 (Kishimoto & Müller 2004/2005)

同一局面が異なる経路で異なる結果を持つ問題．
KomoringHeightsはdual TT(base/twin)で経路依存/非依存の不詰を区別．

**maou_shogiとの差分:** `path`(FxHashSet)によるループ検出のみ．
経路依存/非依存の区別は未実装．

### 5. 合駒(中合い)最適化

#### 5.1 無駄合い枝刈り

紐のない合駒(futile interposition)をスキップ．

**maou_shogi:** 実装済み(futile/chain分類)．

#### 5.2 合駒遅延展開 (KomoringHeights v0.5.0)

合駒の展開自体を遅延させ，不要な分岐を削減．

**maou_shogiとの差分:** `INTERPOSE_DN_BIAS`で**選択を後回し**にしているが，
**展開自体の遅延**はしていない．

### 6. 静的詰め判定

リーフノードで短手数の詰みを検出し，Df-Pnオーバーヘッドを回避．

**maou_shogi:** 実装済み(1手/3手インライン + budget制N手詰め)．

## ミクロコスモス(1525手詰)の解法比較

| ソルバー | 解答時間 | 主要手法 |
|---------|---------|---------|
| 脊尾詰 (1997) | ~20時間 | PN*，~188M局面 |
| KomoringHeights | ~10分 | df-pn+, SNDA, 証明駒/反証駒, GHI対策, 合駒遅延展開 |
| shtsume | ~1分 | 不明(最速ソルバー) |
| やねうら王 | 解けない | TT GC未実装(~3.5TB必要) |

## maou_shogi 改善ロードマップ

効果/コスト比の高い順に整理:

| 優先度 | 手法 | 期待効果 | 実装コスト | 備考 |
|--------|------|---------|-----------|------|
| 1 | 1+εトリック(乗算型) | 大 | 低 | 閾値計算の変更のみ |
| 2 | df-pn+(ヒューリスティック初期pn/dn) | 中〜大 | 中 | KomoringHeightsで実証済み |
| 3 | Deep df-pn(深さ依存初期値) | 中 | 低 | ~35%ノード削減(論文値) |
| 4 | SNDA(過大評価対策) | 大(長手数) | 高 | DAG-heavyな問題で効果大 |
| 5 | TT GC | 必須(超長手数) | 高 | ミクロコスモス解法に必要 |
| 6 | TCA(過小評価対策) | 中〜大 | 中 | 循環グラフでの完全性保証 |
| 7 | GHI対策強化 | 正確性+効率 | 高 | dual TT方式 |
| 8 | TT Best Move保存 | 中 | 中 | PV復元の高速化 |

## 参考文献

### 論文

- Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002)
- Seo, Iida & Uiterwijk, "The PN*-search algorithm: Application to tsume-shogi" (AI 129, 2001)
- Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)
- Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008)
- Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation of Depth-First Proof-Number Search" (AAAI 2010)
- Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)
- Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)
- DFPN-E: "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)
- Kishimoto, Winands, Müller & Saito, "Game-tree search using proof numbers: The first twenty years" (ICGA Journal 35, 2012)
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
