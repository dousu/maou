---
title: find_shortest 未確定時 unknown 化 + ベンチ/テスト命名を実手数へ
status: approved
approved_by: user (2026-07-03 指示)
applied_in: (適用コミット SHA を applied 化時に記入)
---

# find_shortest 未確定時 unknown 化 + ベンチ/テスト命名を実手数へ

## 承認の経緯

user が 2026-07-03 に指示:

1. > 最短で返すオプション find_shortest がついている場合は最短が求められなかった
   > 場合は unknown を返すようにしてください。そうでない場合は，詰みを見つけた
   > 時点で PV を返す形で問題ありません。
2. > find_shortest をつけない場合は呼び出し側で早いレスポンスを求めているので
   > 予算を使い切らずに最初の詰みを見つけた時点で返してほしいです。
3. > ベンチの命名や Rust でのテストを実際の詰み手数に合わせて修正してください。

## 変更 1: find_shortest のセマンティクス厳格化 (指示 1・2)

**旧挙動**: find_shortest=True で budget/timeout が短縮ループ途中で切れると，「その予算内で
見つかった最短 (最小性未証明)」を `Checkmate` で返していた (非最小の可能性があった)．

**新挙動** (`search/mod.rs` `solve_impl`):
- 短縮ループが **len=d-2 の不詰 (`dn()==0`) を確認** できた場合，または d<=1 の場合のみ
  `shortest_confirmed=true`．この時だけ `Checkmate(d)` を返す．
- budget/timeout で最小性を証明しきれなかった (d-2 探索が inconclusive) 場合は
  **`Unknown` を返す** (非最小の詰みを返さない)．
- find_shortest=False は短縮ループを実行せず，初回探索が最初の詰み (pn==0) を見つけた
  時点で `run_search_at_len` が返る = **予算を使い切らず first-mate を即返却** (指示 2 は
  既存挙動で充足; 実測 29te で 7,217 node / 予算 500K 未消費で返ることを確認)．

健全性: STRICT verify は従来どおり authoritative (偽詰みは出さない)．本変更は完全性側の
締め込み (非最小 Checkmate を Unknown へ) であり，canonical anchor (十分予算 → 最小性確定)
は不変．

## 変更 2: ベンチ/テスト命名を実手数へ (指示 3)

benchmark の te-ラベルに実手数との不一致があった (`tsume4_5te` の SFEN は実際 11 手，
`tsume3_7te` は 9 手)．maou の無駄合い抜き最短手数を release ビルドで実測し (ユーザ確認済
Rust test の assert 値と一致)，以下へ統一:

| SFEN | 実手数 | 新キー (bench) | 新テスト名 (Rust) |
|---|---|---|---|
| 6s2/…6BBk | 9 | tsume_9te_a | test_tsume_9te_a |
| 7nl/…7kp | 9 | tsume_9te_b | test_tsume_9te_b |
| 4+P2kl/… | 11 | tsume_11te_a | test_tsume_11te_a |
| 7nk/…5R3 | 11 | tsume_11te_b | test_tsume_11te_b |
| …5Pk2 | 17 | tsume_17te | test_tsume_17te |

- `scripts/benchmark_tsume.py` / `benchmark_vs_cshogi.py`: PROBLEMS/TSUME_PROBLEMS の
  キーを実手数順・実手数名へ．
- `rust/…/dfpn/tests.rs`: index 名の tsume テスト (test_tsume_2/3/4/5) を length 名へ
  リネーム (test_tsume_9te は _a に; 派生 test_tsume_3_ryu... と内部コメント参照も追随)．
  length 名の既存テスト (1te/3te/9te_with_silver 等) は元々正しく無変更．

## バージョン

find_shortest はライブラリ挙動変更 (feat) ゆえ minor bump: maou_shogi 5.0.0→5.1.0，
maou_rust 0.5.0→0.6.0．

## 検証

- lib 160 passed / 0 failed / 5 ignored + doctest 1 passed (--test-threads=1)．
- release canonical anchor 不変: 29te 396,516/Some(29)，39te 17,545,528/Some(39)．
- Python `test_solve_tsume.py` 13 passed (新規 find_shortest セマンティクス 3 件含む)．
- 実測 (release): find_shortest=True + nodes=50K → unknown / +nodes=500K → checkmate(29);
  find_shortest=False → checkmate(31), 7,217 node (予算未消費)．

## ドキュメント

- `api.rs` / `solver.rs` / `maou_rust/maou_shogi.rs` の find_shortest docstring 更新．
- `docs/rust-backend.md` パラメータガイド更新．
