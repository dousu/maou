# legacy/ — 旧二エンジン期 (v0.x〜v0.55) の記録

> [!WARNING]
> **このディレクトリは現行 (統一 `mid`, maou_shogi 3.1.x) の設計ではない．**
> ここに含まれる文書は，v3.0.0 で廃止された **旧二エンジン構成** —
> Best-First PNS (arena) + IDS-dfpn + Frontier Variant + Dual TT (ProvenTT / WorkingTT) +
> warmup + refutable disproof — を前提とした記録である．
> 現行アーキの設計は親ディレクトリ [`../index.md`](../index.md) を参照すること．

## なぜ残すか

旧アーキ向けに開発・計測された手法の中には，**今後 mid を改善する際に再検討する価値のある
方法論**が含まれる (user 指示 2026-06-24)．そのため campaign ログを削除せず verbatim で保全する．

## 含まれる文書

| ファイル | 内容 | 新 mid で再検討しうる方法論の例 |
|---|---|---|
| [benchmarks.md](benchmarks.md) | 39 手詰め campaign の詳細計測ログ (v0.22〜v0.55) | 閾値飢餓 / cliff 診断の方法論，backward 解析 (ply 別 sub-problem 切り出し) |
| [pn-dn-distribution.md](pn-dn-distribution.md) | pn/dn 分布の対数正規化プロジェクト | 分布計測指標 (KL/σ)，初期値ヒューリスティックの値域設計，WPN γ スイープ |
| [optimization-proposals.md](optimization-proposals.md) | 最適化案の採否評価 (原本) | History Heuristic / lazy movegen / grid search 等の採否根拠 |

## 読むときの注意

- **節番号 (§10.2 等) は旧 docs 基準**であり，親ディレクトリの新 docs とは対応しない．
- 文書中の機構名 (PNS, Frontier, ProvenTT/WorkingTT, refutable disproof, WPN/CD-WPN/VPN/SNDA,
  Killer move 等) は **現行コードには存在しない**．
- 版数 (v0.24.xx 等) は旧 campaign のもの．現行 campaign 状態は `scratchpad/compass.md` +
  `worklog/` が source of truth．
- ここの手法を新 mid へ適用する場合は **未検証の候補**として扱い，compass §🚦 TRIPWIRES
  (PRE-LEVER / METRIC-PROVENANCE 等) の discharge を経ること．
