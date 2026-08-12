---
kind: backlog
date: 2026-08-12
path:
  - rust/maou_shogi
  - rust/maou_usi
scope: rust
level: low
last_sha: ae9c659
---

# `/audit-backlog` 2026-08-12 — N5 残件 (dfpn TT の貸し出し時 fill) を許容判断で締める

`audits/coverage.md` § "Out-of-scope backlog" の **N5** を消化して閉じた run．
コードは 1 行も変えていない — **残っていた 1 件を「許容する」というユーザ判断で
閉じた**ので，記録がこの run の成果物そのものである．

## なぜコード修正でなく記録なのか

backlog 行の削除条件は「finding が resolved であること」であって「コードが
変わったこと」ではない．一方，ledger 行を消すと finding の本文ごと消えるため，
**許容判断は必ずどこかに残さないと次の監査で同じものが新規 finding として
再浮上する** (「9.5GB/s で帯域律速」「MCTS と別スレッド」という測定と結論ごと
失われた状態で再起票される)．record は不変なので既存の
`2026-08-12-backlog-auto-band-and-p4.md` に追記もできない．よってこの新しい
record が判断の置き場になる．

## Consumed

- **N5** (`rust/maou_shogi` + `rust/maou_usi`) — dfpn の TT 確保が探索の実費を
  桁違いに上回っていた問題．

## 判断 — 貸し出し時の `fill` は許容する

**対象**: TT を貸し出すたびに走る `fill` が **O(size)** である点 (既定 352MB で
**0.037s/探索**)．

**ユーザ判断 (2026-08-12)**: **許容する．最適化しない．**

**根拠** (いずれも本 finding の調査中に実測済み — 再測定は不要):

1. **最適化余地が無い．** 352MB ÷ 0.037s = **9.5GB/s** で既にメモリ帯域に
   張り付いている．PR #470 の「空 slot 全ゼロ表現」で消えたのは**確保コスト
   だけ**で，fill 自体は 0.043s → 0.037s の 14% 改善どまりだった．
2. **残る手段は対価が大きい．** fill を消すには世代スタンプで fill 自体を
   不要にするしかない (TT サイズを縮める案は，却下済みの「既定
   `RootDfpnNodes` 引き下げ」と同じ reach 縮小の対価を払う)．
3. **実戦では持ち時間に出ない．** dfpn は MCTS と別スレッドで並走するので
   wall clock に現れない (`go#1 = 0.502s` = 予算ぴったりで確認済み)．
   消費するのは CPU 時間であって持ち時間ではない．
4. **`go mate` では critical path に乗るが誤差．** 詰み探索の予算は通常秒
   オーダーなので 37ms は無視できる．

**再開条件**: プロファイルで実害が見えたとき．その時点で世代スタンプ化を
検討する．それまで backlog には戻さない．

## N5 の閉じ方の内訳

閉じ方が 2 種類混ざっているので明示しておく (ledger 上はどちらも「行を消す」で
同じだが，後から読んだときに区別できないと困る)．

| 項目 | 閉じ方 |
|---|---|
| `go mate` が `max_nodes = u64::MAX` を TT サイズへ流していた | コード修正 (#464) |
| 初手で TT 確保が時間予算を食う | コード修正 (#469 — `isready` で前払い) |
| `Entry::null()` が全ゼロでなく `alloc_zeroed` に落ちない | コード修正 (#470) |
| 「詰み証明済だが PV 復元不能」が時間切れと区別できない | コード修正 (#471) |
| 既定 `RootDfpnNodes` を下げる | **判断で却下** (2026-08-12) — 動機だった確保コストが #469/#470 で消え，残る対価は reach 縮小のみになったため |
| 貸し出し時の `fill` が O(size) | **判断で許容** (2026-08-12) — 本 record の上節 |

## 起票時の原症状の再確認 (本 run で実施)

N5 は元々「`tests/maou/infra/console/test_usi_cli.py::test_usi_go_mate_e2e` が
**全件実行だと** `checkmate timeout` で落ちる (単体実行では通る)」として起票
された．行に記録されていた実測 (`go mate` 初回 4.856s → **0.052s**) は単発
計測だったため，**原症状そのもの (全件実行) を本 run で回して確認した**．

```
$ uv run pytest -q          # 全件．pre-commit の test hook と同じ条件
1068 passed, 78 skipped in 39.80s

$ uv run pytest tests/maou/infra/console/test_usi_cli.py -q
9 passed in 5.97s
```

全件実行でも `checkmate timeout` は再現せず，`test_usi_go_mate_e2e` を含む
usi CLI テスト 9 件が通る．環境は `uv sync` (base extra のみ) で，torch 依存の
78 件は skip されているが，usi 経路は Rust 拡張だけで動くので影響しない
(この skip の件は別行 **N4** として backlog に残っている)．

原記録が問題視していたもう一方の点「失敗の出方がソルバ回帰と見分けられない」
は #471 (`info string` で「詰み証明済だが PV 復元不能」を時間切れと区別) が
手当て済み．

**もし将来 `test_usi_go_mate_e2e` が全件実行でだけ落ちたら**，これは N5 の
再燃であって新規 finding ではない — この record と `#464`/`#469`/`#470`/`#471`
から読み直すこと．

## Ledger changes

- `audits/coverage.md` § "Out-of-scope backlog" から **N5 行を削除**．
- 同 § "Records of runs that resolved rows deleted from here" に本 record を追加．
- 主表 (path 単位) には行を追加しない — `kind: backlog` の run は path 全体を
  監査していないため．
