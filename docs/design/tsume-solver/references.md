# 参考文献

### 論文

- Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002)
- Nagai, "Df-pn Algorithm for Searching AND/OR Trees and Its Applications" (Ph.D. Dissertation, UTokyo, 2002)
- Seo, Iida & Uiterwijk, "The PN*-search algorithm: Application to tsume-shogi" (AI 129, 2001)
- Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)
- Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008)
- Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation of Depth-First Proof-Number Search" (AAAI 2010)
- Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)
- Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)
- "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)
- Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)
- Saito et al., "Virtual Proof Number" (2006)
- Kishimoto, Winands, Müller & Saito, "Game-tree search using proof numbers: The first twenty years" (ICGA Journal 35, 2012)
- Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010)
- Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014)
- Hoki, Kaneko, Kishimoto & Ito, "Parallel Dovetailing and its Application to Depth-First Proof-Number Search" (ICGA Journal 36, 2013)
- Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004)
- Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)
- Schaeffer, "The History Heuristic and Alpha-Beta Search Enhancements in Practice" (TCDE 14(4), 1989)
- Allis, "Searching for Solutions in Games and Artificial Intelligence" (1994)

### 既存ソルバー

| ソルバー | 特徴 |
|---------|------|
| KomoringHeights | df-pn+, SNDA, 証明駒/反証駒, GHI 対策, 合駒遅延展開 |
| shtsume | ミクロコスモス ~1分で解く最速ソルバー |
| やねうら王 | TT GC 未実装(~3.5TB 必要で解けない) |

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

## 変更履歴

| 日付 | 版 | 内容 |
|------|-----|------|
| 2026-04-14 | v0.24.78 | §10.2.5 施策 C: `skip_warmup` デフォルト true で warmup 無効化．refutable disproof が NM 蓄積を代替するため warmup は冗長．backward_10m_warmup で ply 22 が Mate(17) → Unknown に退行する問題を解消 |
| 2026-04-14 | v0.24.77 | §10.2.5 施策 B: refutable check 再帰深さの log-adaptive 化．`outer_solve_depth.ilog2()+1` で IDS target depth に基づく d を選択．IDS 中間 step 全体で d が一貫し TT 状態不整合を回避．backward_10m ply 24/22/20 が 33-38% 高速化．refutable check パラメータを `set_refutable_params(depth, call_limit)` で設定可能に |
| 2026-04-14 | v0.24.76 | §10.2.5 H3 完成: refutable disproof の `hand_gte` 支配チェック追加．`store_refutable_disproof` で既存エントリ支配で 83% の冗長挿入をスキップ．全レベル格納 × hand_gte 圧縮で depth=21 Unknown → Mate(15), depth=25 は Mate(15) 維持．disproof 挿入内訳診断カウンタ追加 |
| 2026-04-14 | v0.24.75 | §10.2.5 H3: **refutable disproof** entry 種別を導入．ProvenEntry flags bit 7 で refutable disproof をマーク．通常の look_up_proven では confirmed と同様に可視 (MID 活用可能)，`skip_refutable_disproof` フラグで PNS 探索中のみ不可視化 (arena-limited false NM 防止)．depth_limit_all_checks_refutable の NM 結果を TT に蓄積して再計算を回避 |
| 2026-04-14 | v0.24.74 | §10.2.5 H3 最初の試み: refutable check の NM を confirmed disproof として全レベルで ProvenTT に格納．gap diagnosis depth=17 で Mate(15) → Unknown に退行．PNS の `[pns_false_nm]` 診断ログ追加で false NM 発生メカニズムを特定 (PNS の arena-limited exploration が escape path を先に発見) |
| 2026-04-10 | v0.24.73 | §10.2.4 v0.24.68 の warmup 段間 intermediate 保持を revert．fc-normalized hash との相互作用で false proof (Mate(7) at ply 22) が発生 |
| 2026-04-10 | v0.24.72 | §10.2.4 施策 α 無効化．Frontier variant PNS が filter context を認識せず ABSOLUTE tag で store するため false proof 発生．tag infrastructure は保持 |
| 2026-04-10 | v0.24.71 | §10.2.4 proof_tag propagation infrastructure: tag-aware look_up_proven + IDS break guard + look_up_proven_tag |
| 2026-04-10 | v0.24.70 | §6.6.4, §10.2 fc-normalized hand hash: Pawn/Lance/Rook 総和ベースの WorkingTT クラスタで forward-chain 等価な intermediate を共有．ply 22 warmup -7.4%，no-warmup で初めて Mate(17) 到達 |
| 2026-04-10 | v0.24.69 | §8.4, §10.2 PostCaptureSummary 容量 4x 拡大 (64K → 256K)．hash collision 削減で ply 24 -14.6% (no-warmup) |
| 2026-04-10 | v0.24.66 | §10.2.2 Warmup NM false NoMate 修正: outer_solve_depth ガード + clear_working_entry．edge_cost_or 駒打ちペナルティ (+PN_UNIT/2) |
| 2026-04-10 | v0.24.65 | §10.2.2 Adaptive warmup depths: solve() 内で段階的 depth の warmup solve を実行．ply 22 が 10M + warmup で初解決 |
| 2026-04-16 | v0.25.5 | §10.2.13, §6.6.6 F3 (or_success_cache) default ON + full_hash keying．ply 18 nodes **-75%** (387M→96M)，ply 22 nodes **-59%**．100M 予算で Mate(21) 到達 |
| 2026-04-16 | v0.25.4 | §10.2.12 M-1 refutable fast path 4 strategy (F1-F4) 比較．F1+F3 勝者 (refut_tt_hits 0→2032)．F3 単独で default 候補に |
| 2026-04-16 | v0.25.3 | §10.2.11, §3.6 M-D: adaptive disproof threshold policy 精緻化 (depth 20-22→1, ≥23→3)．S-2 ply 24 退行境界診断 (remaining=1 致命性の実証) |
| 2026-04-16 | v0.25.2 | §10.2.10, §10.2.9 M-A: refutable depth フロア (target≥20→8) で ply 20 false-NoMate 根絶．warmup + refutable depth 組合せ検証で仮説確定 |
| 2026-04-16 | v0.25.1 | §10.2.8 S-1: depth-adaptive disproof threshold (opt-in)．no-mate test で default 化 NG と判定 |
| 2026-04-16 | v0.25.0 | §10.2.7, §3.6, §6.6.6 A-1: PNS arena 動的容量化 (効果なし，基盤のみ)．B-2: depth-limited disproof 選択的格納 (ply 18 NPS +54%，opt-in) |
| 2026-04-10 | v0.24.64 | §8.4 Post-Capture Proof Summary Cache: pos_key ベース O(1) proof/disproof lookup |
| 2026-04-10 | v0.24.63 | §10.2 IDS NM 昇格判定の `ids_depth >= saved_depth` ガードで false NoMate 防止 |
| 2026-04-10 | v0.24.62 | §8.5 Multi-step 逆方向不詰共有: 異マスの兄弟ドロップにも disproof 伝搬 |
| 2026-04-10 | v0.24.61 | §8.5 逆方向不詰共有: disproven 合駒の post-capture disproof を兄弟に伝搬 |
| 2026-04-10 | v0.24.60 | §2.3 IDS warmup mid_fallback: depth > 19 で full-depth 前に saved-4 で nested IDS + forced denom=3．IDS 直接ジャンプを depth ≤ 19 に制限 |
| 2026-04-10 | v0.24.35〜43 | §10.2, §3.1, §6.6 Frontier PNS 予算の動的制御 (v0.24.35-37)．ProvenTT disproof 選択的保持 (v0.24.38)．IDS depth 32+ 段階的深化 (v0.24.40)．depth-adaptive epsilon パラメータチューニング (v0.24.41)．PV visit budget 動的スケーリング (v0.24.42)．GC Phase 2 no-op バグ修正と refactoring (v0.24.43)． |
| 2026-04-04 | v0.24.0 | §6.6.3 Dual TT(ProvenTT hand\_hash 混合 + WorkingTT pos\_key) + エントリ圧縮(40→32B) + 2段階検索 + 段階的 retain + NoMate バグ修正 + ply ベース amount + 祖先チェック．NPS +33%(50M: 183K→244K)，ProvenTT overflow -99.6%．WorkingTT 改善3案は NPS 低下で不採用．§9-b PV 復元．§10.2 構造的課題の最終状態(A: ProvenTT 解決/WorkingTT 残存，B: 部分解決，C: 残存) |
| 2026-03-31 | v0.22.1 | §10.2 方針C 採用(PNS アリーナ再利用)，方針D 不採用(リニアプロービング: NPS 低下)．NPS +20% 改善(同条件比較) |
| 2026-03-30 | v0.22.0 | §6.6 amount フィールド導入 + §6.6.1 クラスタ飽和問題の文書化．amount ベース replace\_weakest + GC．min\_depth は有用性なしで削除 |
| 2026-03-30 | v0.21.1 | §6.6 TT 2M クラスタ化 + クラスタ飽和対策 + 停滞検出改善．29手詰め 59M nodes / 201s で解決．§3.1 epsilon 1/3，§5.1 heuristic S-8S，§3.3 pn\_floor 2/3 |
| 2026-03-29 | v0.21.0 | §10.2 方針A+B 実装．PN\_UNIT=16 + 自然精度 epsilon(§3.1, §3.5) + heuristic 高解像度化(§5.1) + IDS 動的予算配分(§2.6)．N\*7g NoMate: 200K→82K，TT 成長 344K→628K |
| 2026-03-29 | v0.20.36 | §3.5 PN\_UNIT 統一スケーリング + divide-at-unit-scale パターン．スケーリング漏れ4件特定・修正，PN\_UNIT=64 で完全等価性確認 |
| 2026-03-29 | v0.20.35 | §3.3 チェーン AND pn\_floor boost．ply 24 サブ問題 1M Unknown → 397K Mate に改善 |
| 2026-03-29 | v0.20.34 | §6.6 フラットハッシュテーブル化，§11.4 採用．NPS 3.83× |
| 2026-03-29 | v0.20.33 | §11 最適化案の評価(§11.1-11.7)，§10.2 ベンチマーク追加，§11.7 Frontier Variant 採用 |
| — | v0.20.0 | §4.2 CD-WPN，§6.2 前方チェーン補填，§8.7-8.8 チェーン順序最適化 |
| — | v0.18.0 | §2.2 Best-First PNS，§8.5 同一マス証明転用，§9.2 Killer Move |
| — | v0.16.0 | §2.3 IDS-dfpn，§6.5 TT Best Move，§7.2 NM Remaining 伝播 |
| — | v0.1.0 | 初版: §2.1 Df-Pn，§6.1 持ち駒優越 |
