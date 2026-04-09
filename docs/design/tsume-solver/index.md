# 詰将棋ソルバー設計ドキュメント

## 目次

| ドキュメント | 内容 |
|------------|------|
| [探索アーキテクチャ](search-architecture.md) | Df-Pn, Best-First PNS, IDS-dfpn, 全体フロー |
| [閾値制御](threshold-control.md) | 1+ε トリック, TCA, 閾値フロア, PN_UNIT 統一スケーリング |
| [証明数・反証数の計算](proof-disproof-numbers.md) | WPN, CD-WPN, VPN, SNDA |
| [初期値ヒューリスティック](initial-heuristics.md) | df-pn+, DFPN-E, Deep df-pn, インライン詰み検出 |
| [転置表管理](transposition-table.md) | 持ち駒優越, Pareto frontier, TT GC, Dual TT, Zobrist hand_hash |
| [ループ・GHI 対策と手順改善](loop-ghi-and-move-ordering.md) | 経路依存フラグ, NM Remaining, TT Best Move, Killer Move, PV 復元 |
| [合駒最適化](aigoma-optimization.md) | Futile/Chain 分類, 遅延展開, プレフィルタ, DN バイアス |
| [既知の課題とベンチマーク](benchmarks.md) | 29手詰め, 39手詰め, ミクロコスモス |
| [最適化案の評価](optimization-proposals.md) | History Heuristic, フラットハッシュテーブル, Frontier Variant 等 |
| [参考文献](references.md) | 論文, 既存ソルバー, 日本語リソース, 変更履歴 |

---

## 1. 概要

maou_shogi の詰将棋ソルバーは Df-Pn (Depth-First Proof-Number Search, Nagai 2002) を
基盤とし，Best-First PNS と IDS-dfpn(Frontier Variant 統合）の
2フェーズ探索を採用する．

### 設計目標

1. **cshogi が解けない問題をカバーする**: 片玉局面，合駒(中合い)の正確な探索，最短手順保証
2. **どんな詰将棋問題も解ける**: リソースパラメータ(`depth`, `nodes`, `timeout`)の増加で対応
3. **高度な枝刈りによる効率化**: cshogi が取り入れていない手法を積極的に導入

### 実装ファイル

`rust/maou_shogi/src/dfpn/` モジュールに全ての探索ロジックを実装(合計約 8,600 行，テスト除く)．

| ファイル | 行数 | 内容 |
|---------|------|------|
| `solver.rs` | ~3,230 | `DfPnSolver` 構造体，`mid()` 関数，child init，MID ループ |
| `pns.rs` | ~2,680 | 手生成，PNS メインループ，IDS-dfpn，Frontier Variant，PV 復元 |
| `tt.rs` | ~1,850 | Dual フラットハッシュテーブル型転置表(ProvenTT + WorkingTT) |
| `mod.rs` | ~490 | 定数，ユーティリティ関数(SNDA, hand\_gte, DFPN-E 等) |
| `entry.rs` | ~165 | `DfPnEntry`, `PnsNode` データ構造 |
| `profile.rs` | ~210 | プロファイリングマクロ・統計 |

### 実装済み手法一覧

| # | 手法 | 出典 | 節 | 導入版 |
|---|------|------|-----|--------|
| 1 | Df-Pn | Nagai 2002 | §2.1 | v0.1.0 |
| 2 | Best-First PNS | Seo, Iida & Uiterwijk 2001 (PN*) | §2.2 | v0.18.0 |
| 3 | IDS-dfpn | Seo et al. 2001; Nagai 2002 | §2.3 | v0.16.0 |
| 4 | 1+ε トリック | Pawlewicz & Lew 2007 | §3.1 | v0.11.0 |
| 5 | TCA | Kishimoto & Müller 2008; Kishimoto 2010 | §3.2 | v0.14.0 |
| 6 | WPN | Ueda et al. 2008 | §4.1 | v0.17.0 |
| 7 | CD-WPN | maou 独自 | §4.2 | v0.20.0 |
| 8 | VPN | Saito et al. 2006 | §4.3 | v0.15.0 |
| 9 | SNDA | Kishimoto 2010 | §4.4 | v0.11.0 |
| 10 | df-pn+ ヒューリスティック初期化 | KomoringHeights v0.4.0; GPW 2004 | §5.1 | v0.12.0 |
| 11 | DFPN-E | Kishimoto et al., NeurIPS 2019 | §5.2 | v0.13.0 |
| 12 | Deep df-pn | Song Zhang et al. 2017 | §5.3 | v0.12.0 |
| 13 | インライン詰み検出 | — | §5.4 | v0.12.0 |
| 14 | 持ち駒優越 | Nagai 2002 | §6.1 | v0.1.0 |
| 15 | 前方チェーン補填 | maou 独自 | §6.2 | v0.20.0 |
| 16 | Pareto frontier 管理 | Breuker et al. 1994 | §6.3 | v0.12.0 |
| 17 | TT GC | — | §6.4 | v0.12.0 |
| 18 | TT Best Move | KomoringHeights v0.4.0 | §6.5 | v0.16.0 |
| 19 | 経路依存フラグ付き GHI 対策 | Kishimoto & Müller 2004/2005 | §7.1 | v0.15.0 |
| 20 | NM Remaining 伝播 | — | §7.2 | v0.16.0 |
| 21 | Futile/Chain 合駒分類 | — | §8.1 | v0.12.0 |
| 22 | チェーンドロップ3カテゴリ制限 | — | §8.2 | v0.12.0 |
| 23 | 合駒遅延展開 | KomoringHeights v0.5.0 | §8.3 | v0.12.0 |
| 24 | TT ベース合駒プレフィルタ | — | §8.4 | v0.16.0 |
| 25 | 同一マス証明転用 | — | §8.5 | v0.18.0 |
| 26 | 合駒 DN バイアス | — | §8.6 | v0.12.0 |
| 27 | チェーンマス内→外順序 | — | §8.7 | v0.20.0 |
| 28 | チェーン深さ DN スケーリング | — | §8.8 | v0.20.0 |
| 29 | TT Best Move 動的手順改善 | — | §9.1 | v0.16.0 |
| 30 | Killer Move | — | §9.2 | v0.18.3 |
| 31 | 捨て駒ブースト | — | §9.3 | v0.13.0 |
| 32 | Frontier Variant (PNS→局所MID) | maou 独自 | §11.7 | v0.20.33 |
| 33 | TT フラットハッシュテーブル | — | §6.6, §11.4 | v0.20.34 |
| 34 | チェーン AND pn\_floor boost | maou 独自 | §3.3 | v0.20.35 |
| 35 | PN\_UNIT 統一スケーリング | maou 独自 | §3.5 | v0.20.36 |
| 36 | path スタック化 (FxHashSet→配列) | — | §6.6.2 | v0.23.0 |
| 37 | ci\_resolve 再 lookup 廃止 (has\_proof) | — | §6.6.2 | v0.23.0 |
| 38 | 王手生成キャッシュ (CheckCache) | — | §6.6.2 | v0.23.0 |
| 39 | 玉移動合法性チェック高速化 | — | §6.6.2 | v0.23.0 |
| 40 | pn\_floor 乗算オーバーフロー修正 | — | §3.3 | v0.23.0 |
| 41 | Dual TT (ProvenTT + WorkingTT) | KomoringHeights 参考 | §6.6.3 | v0.24.0 |
| 42 | TT エントリ圧縮 (40→32 bytes) | — | §6.6.3 | v0.24.0 |
| 43 | IDS depth 切替時 confirmed disproof クリア | — | §6.6.3 | v0.24.0 |
| 44 | ply ベース ProvenTT amount | — | §6.6.3 | v0.24.0 |
| 45 | 祖先チェックによる ProvenTT 挿入スキップ | — | §6.6.3 | v0.24.0 |
| 46 | ProvenTT hand\_hash 混合インデクシング | — | §6.6.3 | v0.24.0 |
| 47 | 2段階検索 (look\_up\_proven / look\_up\_working) | — | §6.6.3 | v0.24.0 |
| 48 | 段階的 retain\_proofs (confirmed disproof 保持) | — | §6.6.3 | v0.24.0 |
| 49 | Zobrist hand\_hash インデクシング | — | §6.6.4 | v0.24.7 |
| 50 | ProvenTT クラスタサイズ 8 (HAND\_KINDS+1) | — | §6.6.4 | v0.24.3 |
| 51 | WorkingTT hand\_hash 混合インデクシング | — | §6.6.4 | v0.24.6 |
| 52 | Zobrist XOR 差分による近傍クラスタ走査 | — | §6.6.4 | v0.24.8 |
| 53 | PV 復元用部分集合クラスタ走査 | — | §6.6.4, §9-b | v0.24.2 |
| 54 | サンプリング GC (KomoringHeights 参考) | KomoringHeights | §6.6.4 | v0.24.10 |
| 55 | rem=0 仮反証の TT store 廃止 | — | §6.6.4 | v0.24.14 |
| 56 | GC 時の obsolete intermediate 除去 | — | §6.6.4 | v0.24.12 |
| 57 | 探索パス保護 (GC 前 amount 引き上げ) | — | §6.6.4 | v0.24.11 |
| 58 | proof(-1) + 歩disproof(+1) 近傍走査 | — | §6.6.4 | v0.24.16 |
| 59 | ~~PV 抽出 fast path (TT mate\_distance)~~ **v0.24.29 で廃止 (無駄合 unsound)** | — | §6.6.5, §10.2 | v0.24.23〜v0.24.28 |
| 60 | TT entry 24 bytes 維持 (amount 再利用) Plan B | maou 独自 | §6.6.5 | v0.24.24 |
| 61 | ProvenEntry 分離 (Plan D) | maou 独自 | §6.6.5 | v0.24.26 |
| 62 | WorkingTT cluster size 6→8 (cache-aligned, +33% slots) | maou 独自 | §6.6.5 | v0.24.27 |
| 63 | PV 抽出で effective\_len 比較を常用 (fast path 廃止) | maou 独自 | §10.2 | v0.24.29 |
