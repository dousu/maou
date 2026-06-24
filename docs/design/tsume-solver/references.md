# 参考文献

### 論文

- Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002)
- Nagai, "Df-pn Algorithm for Searching AND/OR Trees and Its Applications" (Ph.D. Dissertation, UTokyo, 2002)
- Seo, Iida & Uiterwijk, "The PN*-search algorithm: Application to tsume-shogi" (AI 129, 2001)
- Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)
- Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008)
- Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation of Depth-First Proof-Number Search" (AAAI 2010)
- Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)
- "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)
- Kishimoto, Winands, Müller & Saito, "Game-tree search using proof numbers: The first twenty years" (ICGA Journal 35, 2012)
- Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004)
- Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)
- Allis, "Searching for Solutions in Games and Artificial Intelligence" (1994)
- Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010) — 並列方針の検討用 ([opt-proposals §11.1](optimization-proposals.md))
- Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014) — 同上

> 旧版で参照していた WPN (Ueda 2008) / VPN (Saito 2006) / SNDA (Kishimoto 2010) / Deep df-pn
> (Song Zhang 2017) は，統一 mid のコードには対応機構が無い (二重計数は δ-sum + DAG + cross-hand
> で扱う; [proof-disproof-numbers.md §4.4](proof-disproof-numbers.md))．prior art として
> [legacy/](legacy/README.md) の記述で参照される．

### 既存ソルバー

| ソルバー | 本実装が参考にした手法 |
|---------|----------------------|
| KomoringHeights | 中合いの遅延展開 ([aigoma §8.1](aigoma-optimization.md))，証明駒・反証駒の活用 ([TT §6.4](transposition-table.md))，mate length を保持する TT ([TT §6.3](transposition-table.md))，二重計数除去・cross-hand 親参照 ([proof-numbers §4](proof-disproof-numbers.md))，サンプリング GC ([TT §6.5](transposition-table.md)) |
| shtsume | ミクロコスモスを高速に解く詰将棋ソルバー |
| やねうら王 | df-pn の詳細解説 (下記リソース) |

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

---

## 開発の節目

詳細な版数ごとの変更履歴・campaign の計測記録は **git 履歴**・`scratchpad/compass.md`・
`worklog/` が source of truth である (本 docs には maou 内部の版数を記さない方針)．主要な節目:

| 節目 | 内容 |
|------|------|
| 初期 (v0.x) | df-pn + 持ち駒優越を基盤に，二エンジン構成 (Best-First PNS + IDS-dfpn) + Dual TT + Frontier Variant + warmup を発展させた (記録は [legacy/](legacy/README.md)) |
| 統一 (v3.0.0) | 二エンジンを廃し**統一 `mid` 一本**へ再構築 (φ/δ 統一・len-aware 単一 TT・cross-hand・二重計数除去・DML・反復深化 root loop) |
| 整理 (v3.1.x) | 純粋技術名へのリネーム + モジュール file-split (search/ movegen/ tt/ ネスト，god-object 分解) |
