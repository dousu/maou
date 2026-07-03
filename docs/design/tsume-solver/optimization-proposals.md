# 最適化案の評価

本節は，採用しないことを**方針として確定**した最適化案を記録する．個々の探索パラメータ
(lever) の採否計測の経緯は git 履歴に残る．

## 11.1 並列 df-pn — 採用しない (binding)

**出典:** Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010);
Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014)

複数スレッドで異なるサブツリーを同時探索する手法 (共有 TT + 仮想証明数)．8 スレッドで ~3.5×
(並列効率 ~0.5) という報告がある．**maou_shogi では並列探索を採用しない (再提案も行わない)．**

**理由:**

1. **wheel 可搬性 (binding)**: 配布する wheel はビルド環境差に依存しない可搬バイナリである
   ことを binding 制約とする．並列実装や `target-cpu=native` ビルドはこれを壊すため採らない．
   HW 命令の利用は **runtime gate** (実行時 CPU 機能判定) でのみ許す．
2. **アルゴリズム改良を優先**: 性能改善は単一スレッドでの探索アルゴリズム改良 (TT 構造・
   二重計数除去・合駒の遅延展開・証明駒/反証駒の活用等) で達成する方針とする．
3. **実装・保守コスト**: TT のスレッドセーフ化 + 仮想証明数 + 負荷分散は大規模で，soundness
   検証コストが極めて高い．並列効率 ~0.5 の利得はこのコストに見合わない．
4. **プロジェクト全体の指針**: 主な開発対象は評価関数学習・データパイプラインであり，詰将棋
   ソルバーは探索基盤の一要素．単一スレッドで自己完結する実装を維持し，学習パイプラインへの
   統合を単純に保つ．

## 11.2 native ビルド (target-cpu=native) — 採用しない (binding)

ビルドホストの CPU 命令に最適化する `target-cpu=native` は wheel 可搬性を壊すため採用しない
(§11.1-1 と同じ binding 制約)．SIMD 等の HW 命令は runtime gate 経由でのみ用いる．

## 11.3 旧二エンジン期の採否記録

History Heuristic・lazy move generation・PNS 予算適応配分・Frontier Variant・各種 TT 圧縮など，
旧アーキで評価された最適化案の採否とベンチマークの記録は git 履歴に残る．これらは旧二エンジン
構成 (PNS + IDS-dfpn + Dual TT) を前提とするため，統一 mid への適用は**未検証の候補**として
扱い，採用検討時は `scratchpad/compass.md` の §🚦 TRIPWIRES (PRE-LEVER / METRIC-PROVENANCE 等)
を discharge すること．
