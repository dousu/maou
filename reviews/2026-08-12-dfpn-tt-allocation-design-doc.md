---
status: applied
applied_in: a970043
date: 2026-08-12
target:
  - docs/design/tsume-solver/tt-allocation.md
  - docs/design/tsume-solver/index.md
  - docs/design/tsume-solver/transposition-table.md
  - docs/design/tsume-solver/optimization-proposals.md
risk: low
reversibility: trivial
---

# dfpn TT の確保まわりの設計判断を設計ドキュメントに起こす

## Trigger

ユーザ指示 (2026-08-12):

> 設計判断がある分には設計ドキュメントを作成してそこに記載してください

直前に `audits/coverage.md` の **N5** を「貸し出し時の `fill` コストは許容する」
というユーザ判断で閉じ，`audits/2026-08-12-backlog-n5-fill-accepted.md` に
記録した (`5b6f40c`)．ただし `audits/` は**監査の走査記録**であって設計の
参照先ではない — N5 で確定した判断は 1 件 (fill 許容) ではなく，
**TT をいつ・どれだけ・どのように確保するかという一連の設計判断**である．

現状これらは PR の commit message (#464 / #469 / #470 / #471) と
audits の record にしか無く，`docs/design/tsume-solver/` を読んでも辿れない．

## 承認について

CLAUDE.md § MUST rules は durable doc の編集に承認済み `reviews/*.md` を要求する．
本件は**ユーザが設計ドキュメントへの記載そのものを明示的に指示している**ため，
その指示を承認として扱い本 run 内で適用した．先例は
`reviews/2026-08-12-root-dfpn-nodes-usi-option.md` (`c0fa2c4`) で同じ形をとっている．

P2 の standing approval (drift correction) **ではない** — 既存記述の訂正ではなく
新しい節の追加であり，訂正後の本文が現行コードから一意に決まる類ではないため，
本来は判断帯に落ちる．明示指示があるからこそ適用できる，という位置づけである．

## 記載する設計判断

| # | 判断 | 種別 | 出所 |
|---|---|---|---|
| 1 | TT サイズは**ノード予算の見込み**から決め，**停止条件とは切り離す** (`set_tt_nodes_hint`) | 実装済 | #464 |
| 2 | 空 slot は持ち駒のビット反転で**全ゼロ表現**にし `alloc_zeroed` に載せる | 実装済 | #470 |
| 3 | 初回確保は `isready` で**前払い**する (`warm_tt_pool`)．式は `tt_entries_for` を共有する | 実装済 | #469 |
| 4 | 既定 `RootDfpnNodes = 2M` を下げる案は**却下** | 判断 | 2026-08-12 |
| 5 | 貸し出し時の `fill` (O(size)) は**許容**する | 判断 | 2026-08-12 |
| 6 | `checkmate timeout` の 2 事象を `info string` で区別する | 実装済 | #471 |

4 と 5 が今回のユーザ判断，1〜3 と 6 は実装済みだが設計としての根拠が
どこにも書かれていなかったもの．

## 置き場所の判断

**新規ファイル `tt-allocation.md` を作り，`transposition-table.md` には入れない．**

既存 `transposition-table.md` は TT の**探索意味論** (持ち駒優越，len-aware
エントリ，cross-hand 参照，GC) を扱う．今回の判断群は**探索が返す値を一切変えない**
— 動くのは応答時間とメモリ常駐だけで，検証方法も「探索が 1 ノードもずれないこと」
という別物になる．同じファイルに混ぜると「TT を触ると探索結果が変わりうる」という
読み手の警戒が両方に掛かってしまうため分けた．

節番号は既存の連番 (§6 = 転置表管理，§11 = 最適化案の評価) に続けて **§12** とした．

## 差分の内訳

1. **新規 `docs/design/tsume-solver/tt-allocation.md`** — §12.1〜12.7．
   - §12.1 経緯 (テストの flaky として起票したのは誤りだった，という教訓を含む)
   - §12.2 サイズ決定と `RootDfpnNodes`．既定 2M 引き下げ却下の根拠
   - §12.3 全ゼロ空 slot．**効果は確保コストのみ**で fill には効かない点を明記
   - §12.4 `isready` 前払い．position-search §5.1 と同じ原則である旨と，
     式を共有しないと warm が静かに無駄になる点
   - §12.5 fill 許容の判断と再開条件
   - §12.6 実測表 + 健全性検証 (`test_29te` / `test_false_proof_hunt`)
   - §12.7 `checkmate timeout` の区別 (関連仕様変更)
2. **`index.md`** — 目次に 1 行追加．
3. **`transposition-table.md`** — 冒頭に棲み分けの 1 文と相互リンク．
4. **`optimization-proposals.md`** — §11.3 として却下・許容の 2 件を要約 (本体は
   `tt-allocation.md` 側)．既存 §11.3 は §11.4 へ繰り下げ (他ファイルからの参照は
   `§11.1` のみで，繰り下げの影響が無いことを確認済み)．

## 数値の裏取り

記載した値はすべて本 run で実コードに当てて確認した (記憶や record からの
コピーではない):

| 記載 | 出所 |
|---|---|
| `clamp(budget_nodes*2, min_tt_entries, 1<<23)` | `rust/maou_shogi/src/dfpn/search/mod.rs` `tt_entries_for` |
| floor 既定 `1<<18` | `rust/maou_shogi/src/dfpn/solver.rs` `min_tt_entries` |
| 上限 `1<<23` = 704MB | `solver.rs` の doc comment |
| 既定 2M → 352MB (主 TT 256MB + 千日手表 96MB) | `rust/maou_shogi/src/dfpn/tt/mod.rs` `BufferPool` doc |
| `RootDfpnNodes` 既定 2,000,000 | `rust/maou_usi/src/agent.rs` |
| 「~41 手級 / 41te は ~1.2M ノード」 | `rust/maou_search/src/search.rs:285` |
| `DFPN_TT_POOL_BYTES` 既定 1GiB / `0` で無効化 | `tt/mod.rs` `max_pooled_bytes` |
| `warm_tt_pool` を `isready` で呼ぶ | `rust/maou_usi/src/backend.rs` |
| ビット反転で空 slot が全ゼロ | `rust/maou_shogi/src/dfpn/tt/entry.rs` |

実測値 (4.856s → 0.052s，peak RSS 745MB → 462MB 等) は #464〜#471 の調査時の
計測をそのまま引いている (本 run で再測定はしていない — 環境が違えば絶対値は
動くが，判断を支えているのは比と桁である)．

## 影響

- コード変更なし．`src/` にも `rust/` にも触れないのでバージョン bump は不要．
- `docs/commands/` は変更しない (CLI オプションの説明は既に
  `2026-08-12-root-dfpn-nodes-usi-option.md` で適用済み)．
