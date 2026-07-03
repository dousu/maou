---
title: 詰将棋ソルバー Python バインディング整理 + legacy ドキュメント削除
status: approved
approved_by: user (2026-07-03 指示: python 呼び出し周りの調査整理 + legacy docs 削除)
applied_in: (適用コミット SHA を applied 化時に記入)
---

# 詰将棋ソルバー Python バインディング整理 + legacy ドキュメント削除

## 承認の経緯

user が 2026-07-03 に 2 点を明示指示した:

1. > pythonからの詰将棋ソルバー呼び出し周りで調査クエリを整理してください。
   > 特に scripts/analyze_pn_dn_dist.py といったpythonから呼び出して調査する
   > コードが不要になっているのも気になります。
2. > legacy/pn-dn-distribution.md をはじめとしたlegacyドキュメントは
   > すでに不要になっているので削除してください

## 評価で判明したバインディング境界の不整合

Python→`maou._rust.maou_shogi.solve_tsume`→`dfpn::solve_tsume_with_timeout`→
`DfPnSolver::solve()` の経路を精査した結果:

1. **`draw_ply` は受け取るが無視**されていた — `with_timeout(_, _, _draw_ply, _)` の
   第 3 引数はアンダースコア付きで完全に未使用 (千日手処理は `path_depths` へ移行済)．
   Python API が公開する `draw_ply=32767` は no-op だった．
2. **`depth >= 48` で PanicException** — `with_timeout` は `depth < PATH_CAPACITY(48)`
   を assert するため，Python から `depth=50` を渡すと `ValueError` でなく
   `pyo3_runtime.PanicException` が飛んでいた．docstring は「範囲: 1〜数百程度」と誤記．
3. **`scripts/analyze_pn_dn_dist.py`** は既に削除済の `solve_tsume_pn_dn_dist` を import
   しており import 段階で壊れていた (mid 統一以降ずっと空データを描画する化石)．

## 適用した変更

### コード (rust/)

| 対象 | 変更 |
|---|---|
| `dfpn/solver.rs`, `dfpn/api.rs`, `maou_rust/maou_shogi.rs` | 無視されていた `draw_ply` 引数を全経路から削除 (`new`/`with_timeout`/`solve_tsume*`/pyfunction signature)．dfpn/tests.rs の全呼出も追随 |
| `maou_rust/maou_shogi.rs` | pyfunction を全 `Option` 引数化し Rust 側デフォルトへ一元委譲．`depth` が 1..=47 外なら PanicException でなく `ValueError` を返すガードを追加．docstring を実態 (上限 47・メモリ ≈512MB・Ctrl-C 非対応) へ修正 |
| `dfpn/solver.rs`, `dfpn/search/mod.rs` | **回帰修正**: cleanup 第 1 波で `path_dominance` フィールドを実質常時 true から false 化してしまい dominance 枝刈りが無効化 (29te 396,516→534,219 nodes) していた．全読取サイトを無条件化し `Params` 構造体ごと削除して常時有効を確定 |

バージョン: maou_shogi 4.0.0→**5.0.0**，maou_rust 0.4.0→**0.5.0** (公開シグネチャの
breaking change)．

### テスト

- `tests/maou/domain/board/test_solve_tsume.py` を新設 (Python バインディング境界):
  正常系 (詰み/不詰/デフォルト省略)・引数バリデーション (不正 SFEN・depth 範囲は
  ValueError)・GIL 解放下の並行呼び出し安全性．

### スクリプト

- `scripts/analyze_pn_dn_dist.py` を削除 (壊れた化石)．
- `scripts/benchmark_tsume.py` / `benchmark_vs_cshogi.py`: 39te の depth を 63→47
  (上限修正) + dev プロファイル計測の注意書きを追加．

### ドキュメント

- `docs/design/tsume-solver/legacy/` を削除 (README/benchmarks/pn-dn-distribution/
  optimization-proposals の 4 ファイル)．旧二エンジン期の記録は git 履歴が source of truth．
- 上記を参照していた 9 doc + index の legacy リンク 18 箇所を「記録は git 履歴」へ書換．
- `docs/rust-backend.md`: `solve_tsume` の Python/Rust API 例を新シグネチャへ更新
  (draw_ply 削除・depth 上限・メモリ/Ctrl-C 注意・dev プロファイル注意)．
- `docs/design/tsume-solver/index.md`: 公開 API 節の signature 更新．
- `docs/plans/fancy-spinning-sonnet.md` (gitignore 対象のローカル作業ファイル) を削除．

## 検証

- lib 160 passed / 0 failed / 5 ignored + doctest 1 passed (`--test-threads=1`)．
- release canonical anchor **回復確認**: 29te 396,516 nodes / Some(29)，
  39te 17,545,528 nodes / Some(39) (path_dominance 修正後)．
- Python: `test_solve_tsume.py` 全 pass (maturin develop 後)．

## 不変条件

- 探索アルゴリズムは第 1 波 (refactor d662e70) 時点と同一へ回復 (path_dominance 回帰を是正)．
- 削除した legacy docs の内容は git 履歴で追える．
