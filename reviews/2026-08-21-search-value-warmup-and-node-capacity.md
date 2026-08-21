---
status: applied
applied_in: 73319cd
---

# search-values に `warmupMs` と `--node-capacity` を足す

## 背景 (Colab L4 実測，2026-08-20)

`DFPN_TT_POOL_BYTES` の同一セッション A/B (warmup 後，50 局面):

| | median | mean |
|---|---|---|
| プール無効 (OFF) | 184.0 ms | 184.3 ms |
| プール有効 (ON) | 75.5 ms | 75.8 ms |

**2.44x** (DevContainer のモックモデル + `--playouts 1` では 9.0x)．差が縮むのは
NN コストが露出したためで，予測どおりである．ON の 75.5 ms は 800 playouts なので
**10,596 playouts/s** — `docs/performance.md` の L4 batch 64 = 10,257 playouts/s と
一致する．**184 ms がこれを覆い隠していた**．

ここから所要時間を出すと ply>=120 (5.3M) で 111.2 時間，ply>=60 (18.7M) で
392.2 時間になる．**しかしこれは下限であって見積りではない．**

## 問題 1: `elapsedMs` は 1 局面の総コストではない

`elapsedMs` は Rust 側 `elapsed_ms` で，計測区間は `search.rs` の
`let start = Instant::now()` からである．その手前にある

- `NodePool::new(opts.node_capacity)` — 既定 `1 << 20` = 1,048,576 ノード．
  `Node` は約 48 B なので**約 50 MB を 1M 回のループで初期化する**
- root の同期 NN 評価 1 回

は `warmup_ms` として別掲され，**`elapsedMs` には入らない**．そして
`search_with_history` は `run_search(..., reused: None, ...)` を呼ぶので，
**この経路は保持木を引き継がず，確保は局面ごとに払われる**．

`warmup_ms` は Python へ露出済み (`maou_search.rs` の `#[pyo3(get)]`) だが，
`search_value.py` が拾っていないため**出力のどこにも記録が無い**．したがって
現状のデータでは 111.2 時間に何を足せばよいかが分からない．

真の所要時間は `111.2 時間 × (1 + warmup_ms / 75.5)` である:

| `warmup_ms` | ply>=120 (5.3M) | ply>=60 (18.7M) |
|---|---|---|
| 0 (現在の報告値) | 111 時間 | 392 時間 |
| 25 ms | 148 時間 | 522 時間 |
| 75 ms | 222 時間 | 784 時間 |
| 150 ms | 331 時間 | 1,169 時間 |

投与量 (どの `--min-ply` まで下ろすか) はこの値で決まるので，**記録しないまま
本番蓄積に入ると，GPU 予算を外した状態で数百時間を走らせることになる**．

## 問題 2: ノードプールが 3 桁過剰である

ノードは「未展開の子へ降りた playout」1 回につき 1 個しか確保されない
(`search.rs` の `pool.alloc()`)．したがって必要数は playout 予算で上から
押さえられ，**800 playouts が要るのは約 800 ノード**である．既定の 1,048,576 は
約 1,300 倍にあたる．

容量を絞っても**探索は変わらない** — ノードプールの GC (`pool.compact`) は
プールが**枯渇したときにしか**走らないので，必要数を上回っている限り木は同一に
なる．ただし `node_capacity` は `SearchOptions` にはあるがモジュール関数
`search()` にしか露出しておらず，`SearchEngine.search` からは触れないので
search-values からは指定できない．

## 提案する変更

### コード (`src/` / `rust/`)

1. `SearchEngine.search` に `node_capacity` を通す (モジュール関数側には既にある)
2. `search_value.py` が `result.warmup_ms` を記録する — 出力へ `warmupMs` 列を
   足し，要約に `mean_warmup_ms` を出す
3. 容量の既定を playout 予算から決める (`2 * max_playouts + 4096`)．
   `--node-capacity` で上書きできる
4. **観測を同梱する**: 要約へ `node_capacity` / `max_nodes_used` / `gc_runs` を
   出し，`gc_runs > 0` なら警告する．GC が走った実行は木が刈られており，
   刈られなかった実行と探索値を並べられない (教師データとして混ぜられない)

### ドキュメント (`docs/commands/utility_search_values.md`) ← **この提案の承認対象**

1. CLI options 表に `--node-capacity` の行を足す
2. Output 表に `warmupMs` の行を足し，`--output-path` の説明の列挙にも足す
3. **「Cost (実測)」節に L4 の A/B と「`elapsedMs` だけで外挿しない」を書く** —
   この節は既に「公称 playouts/s からの割り算を根拠にしない」と書いているが，
   今回踏んだのは**その次の罠**である．実測値を使ってなお，計測区間の外に
   ある固定費を数え落とすと下限を見積もる
4. 「律速の切り分け」節の集計スニペットへ `warmupMs` の中央値を足す

## 併せて記録すべきこと (この提案の対象外)

- **L4 の A/B は `identical: False` だった**．DevContainer で確認した
  「探索はビット単位で不変」は**モックモデル + `--playouts 1`** の条件下の
  主張であり，dfpn に並走相手がいない．playouts 800 では dfpn が MCTS と
  並走するので，確保が速くなったぶん同じ壁時計の中でより深く詰みを証明し，
  木が変わる (worklog が予測していた「詰み探索が実質的に深くなる」)．
  **意図した挙動だが，不変性の主張は本番へ外挿できない**．
  `scratchpad/compass.md` § 環境リファレンスの「208→23ms / 9.0x，探索不変」を
  スコープ付きに直す (campaign 層なので `reviews/` の対象外)
- 既存の 1.25M 行は**プール導入前 (= OFF 相当) の出力**である．今後 ON で
  積むと教師データが異種混合になる (値としては ON のほうが真値に近いので
  品質勾配であって誤りではないが，反証テストとしては交絡)
- 50 局面の wall clock 3.43 秒/局面から外挿してはいけない．探索は OFF で
  9.2 秒 / 171 秒 = **5.4%** しかなく，残りは `_iter_targets` が
  `_read_hashes` で HCPE を読み直すぶん (109 ファイル × 約 1.74 秒 ≒ 190 秒)．
  **ファイル数に比例する実行あたり固定費**なので局面数には比例せず，
  5.3M では消える
