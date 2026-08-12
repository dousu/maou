---
status: applied
applied_in: c0fa2c4
date: 2026-08-12
target: [docs/commands/usi.md]
risk: low
reversibility: trivial
---

# `RootDfpnNodes` を USI オプション表に追加する

## Trigger

ユーザ指示 (2026-08-12):

> root_dfpn_nodes は setoption に追加してください。停止条件に関しては時間の
> ままでヒントとして扱ってください。TT サイズが満杯になっても GC があるので
> 全く無意味ではないので USI 経由は時間で止まる仕様にしましょう。

背景は `audits/coverage.md` の N5 — `go mate` が `max_nodes = u64::MAX` を
渡すため TT が常に上限 (1<<23 = 704MB) になり，1 手詰でも初回応答が数秒
かかっていた (実測 4.856s)．ノード予算を **TT サイズのヒント**として
切り離し，ノブを USI に公開した．

## 承認について

CLAUDE.md § MUST rules は durable doc の編集に承認済み `reviews/*.md` を
要求する．本件は**ユーザが機能追加そのものを明示的に指示している**ため，
その指示を承認として扱い本 run 内で適用した．P2 の standing approval
(drift correction) ではない — 新規オプションの記述なのでドリフトではなく，
承認の根拠は上記のユーザ指示である．

記述内容は実装から一意に決まる: オプション名・型・既定値は
`rust/maou_usi/src/agent.rs` の `OptionDecl`，TT サイズ式は
`rust/maou_shogi/src/dfpn/search/mod.rs:560`，`go mate` が時間でのみ止まる
ことは `rust/maou_usi/src/backend.rs` の `solve_mate` による．

## Before / After

`docs/commands/usi.md` の USI options 表に 1 行追加する
(`RootDfpn` / `LeafMate` の行の直後)．

```diff
 | `RootDfpn` / `LeafMate` | check | Mate search toggles. |
+| `RootDfpnNodes` | spin | Node budget for the root dfpn mate search (default `2000000`). For `go` it is the search cutoff; for `go mate` it only **sizes the transposition table** — that search stops on time/`stop` as the USI spec requires. The table holds `clamp(nodes * 2, 2^18, 2^23)` entries and is written in full on allocation (≈7 ms/MB; the default is ≈352 MB), so this is a fixed cost paid per search, not just at startup. Raise it only to chase long mates; an undersized table is collected by GC rather than losing the mate, so it costs time, not correctness. |
```

## 記述の根拠 (実測)

| 主張 | 根拠 |
|---|---|
| 既定 2000000 | `agent.rs` の `OptionDecl` (`c.root_dfpn_nodes.unwrap_or(2_000_000)`) |
| `clamp(nodes*2, 2^18, 2^23)` | `dfpn/search/mod.rs:560` |
| 確保時に全バイト書き込む | `dfpn/tt/mod.rs:65` が明記 (`Entry::null()` が全ゼロでないため) |
| 約 7 ms/MB | 実測: 22MB:0.205s / 88MB:0.741s / 352MB:2.485s / 704MB:4.856s |
| 既定は約 352MB | 2,000,000 × 2 = 4,194,304 entries × (64+24)B |
| **毎探索**の固定費 | プールは貸し出し時に初期値で `fill` するため 2 回目以降も O(size) (2^23 で 0.091s / 2^18 で 0.003s) |
| 小さくても解けなくならない | 実測: `RootDfpnNodes=1000` でも mate-29 (約 40 万ノード) を 29 手で解く．Rust 側は `test_tt_nodes_hint_does_not_limit_search` が固定 |
| `go mate` は時間で止まる | `backend.rs` の `solve_mate` は `max_nodes = u64::MAX` のまま，ヒントのみ設定 |
