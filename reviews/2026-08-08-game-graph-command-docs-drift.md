---
status: pending
applied_in:
date: 2026-08-08
target: [docs/commands/build_game_graph.md, docs/commands/visualize.md]
risk: low
reversibility: trivial
---

# game_graph のコマンドドキュメントを実装に追従させる

## Trigger

`/audit-and-fix src/maou/domain/game_graph high` の step 4 (ドキュメント整合
検査)．`audits/2026-08-08-src-maou-domain-game-graph.md` を参照．

`src/maou/domain/game_graph/` を記述しているドキュメントを探索した結果，
実質的な記述を持つのは `docs/commands/build_game_graph.md` (生成側) と
`docs/commands/visualize.md` の 147-194 行 (表示側) の 2 本だった．
この 2 本に対し，コード側で真偽が判定できる主張を全件照合した．

なお `CLAUDE.md` / `AGENTS.md` / `README.md` / `docs/architecture.md` /
`docs/adr-*.md` には game_graph への言及が一切ない．ドリフトは生じ得ないが，
このモジュールはアーキテクチャ文書上の居場所を持っていないということでもある．

## Proposed change

### 1. `visualize.md:172` — ノード色の参照フィールドが誤り (WRONG)

ノードの塗り色は `sente_best_move_win_rate` から計算される
(`static/game_graph_canvas.js:241` → `color: winRateToColor(n.sente_best_move_win_rate)`)．
`result_value` は `sente_result_value` としてクライアントに渡るが
(`game_graph_visualization.py:236`)，ツールチップ/詳細表示にしか使われず
塗りには一切関与しない．`result_value` の影響範囲を追う読者を実際に誤導する．

Before:
```
- ノードの**色**: 勝率(`result_value`)に応じたグラデーション
```
After:
```
- ノードの**色**: 最善手勝率(`best_move_win_rate`，先手視点に変換した値)に応じたグラデーション
```
(173 行の >55% / <45% の閾値は `game_graph_canvas.js:45,51` と一致しており正しい．)

### 2. `visualize.md:174` — ノードサイズの関数形が誤り (WRONG，軽微)

`game_graph_canvas.js:64-66` は
`probabilityToRadius(p) = (NODE_MIN_SIZE + (NODE_MAX_SIZE - NODE_MIN_SIZE) * Math.sqrt(p)) / 2`．
比例ではなく √p のアフィン関数．

Before:
```
- ノードの**サイズ**: 親エッジの確率(`probability`)に比例
```
After:
```
- ノードの**サイズ**: 親エッジの確率(`probability`)の平方根に応じて変化
```

### 3. `visualize.md:182` — 局面統計の項目数が不足 (STALE)

`GameGraphVisualizationInterface.get_node_stats`
(`game_graph_visualization.py:490-504`) は 6 項目を返す．うち `定跡` は
`domain/game_graph/openings.py` の唯一のユーザー可視の出力である．

Before:
```
- **局面統計**: 勝率，最善手勝率，深さ，分岐数
```
After:
```
- **局面統計**: Zobrist Hash，勝率(手番視点)，最善手勝率，深さ，分岐数，
  定跡(一致する定跡がある場合のみ「定跡名(カテゴリ)」形式で表示)
```

### 4. `visualize.md:186-190` — コントロールの列挙が実装に追いついていない (STALE)

`game_graph_server.py:764-834` は以下も構築するが，いずれも未記載:
「更新」ボタン (:783)，「ルートに設定」ボタン (:789)，パンくずリスト
(:794-799)，「エクスポート」アコーディオン内の USI position 文字列テキスト
ボックス (:822) と CSV出力/CSVダウンロード (:827-835)．

Before:
```
- **表示深さ**: サブグラフの表示階層数(1-10，デフォルト3)
- **最小確率**: 表示するエッジの最小確率閾値(0.001-0.3，デフォルト0.01)
- **ルートに戻る**: グラフのルートノードに戻る
```
After:
```
- **表示深さ**: サブグラフの表示階層数(1-20，デフォルト3)
- **最小確率**: 表示するエッジの最小確率閾値(0.001-0.3，デフォルト0.01)
- **更新**: 現在の表示深さ・最小確率でサブグラフを再描画する
- **ルートに戻る**: グラフのルートノードに戻る
- **ルートに設定**: 選択中のノードを新しいルートに設定する
- **パンくずリスト**: ルート変更の履歴．クリックで任意の階層に戻れる
- **エクスポート**: 選択局面の USI position 文字列の表示と，
  指し手一覧の CSV 出力/ダウンロード
```
(表示深さの上限 1-10 → 1-20 は `game_graph_server.py:765-770` の
`gr.Slider(minimum=1, maximum=20, value=3, step=1)` に基づく．最小確率の
0.001-0.3/デフォルト 0.01 は :774-779 と一致しており変更しない．)

### 5. `build_game_graph.md:6` および `## Output format` — `metadata.json` が未記載 (STALE)

このコマンドは第 3 の成果物 `metadata.json` を常に出力する
(`build_game_graph.py:190-195` の `io.save_metadata(output_dir, {"initial_sfen": resolved_sfen})`，
`game_graph_io.py:23` の `METADATA_FILENAME`)．これは装飾的な差分ではない．
`game_graph_server.py:287,297` がこれを読み戻して `initial_sfen` からルートの
手番を決定し，`_to_sente_perspective` の先手視点変換に使う．ドキュメントだけを
見て書かれた第三者ツールの出力は，全勝率が逆側から描画されても気づけない．

Before (line 5-6):
```
- preprocessデータ(局面単位・集約済み `.feather`)からBFSでゲームグラフ(有向グラフ)を構築し，
  `nodes.feather` + `edges.feather` として出力する．
```
After:
```
- preprocessデータ(局面単位・集約済み `.feather`)からBFSでゲームグラフ(有向グラフ)を構築し，
  `nodes.feather` + `edges.feather` + `metadata.json` として出力する．
```

`## Output format` の `edges.feather` 表 (71 行) の直後に以下を追加:
```
### `metadata.json`

| Key | Type | Description |
| --- | --- | --- |
| `initial_sfen` | string | BFS開始局面のSFEN．`--initial-sfen` 未指定時は平手初期局面 |

`maou visualize --array-type game-graph` はこのファイルからルート局面の手番を
判定し，勝率を先手視点に変換する．欠落するとこの変換が行えないため，
グラフディレクトリを外部ツールで生成する場合も必ず出力すること．
```

### 6. `build_game_graph.md:20` — 宙に浮いた「Epic 2」参照 (STALE)

「Epic 2」はリポジトリ全体 (`docs/`，`reviews/`，`audits/`，全ソース) で
この 1 箇所にしか出現せず，参照先が存在しない．助言の内容自体は正しい
(構築時 0.001 に対し表示時デフォルト 0.01) ので，解決しない符牒だけを外す．

Before (`--min-probability` 行の末尾):
```
表示時のフィルタリング(Epic 2)より小さい値を設定すべき．
```
After:
```
`maou visualize` の表示時フィルタ(デフォルト 0.01)より小さい値を設定すべき．
```

### 7. `build_game_graph.md:19` — `--max-depth` の上限が未記載 (ACCURATE-BUT-FRAGILE)

`GameGraphBuilder.build` (`builder.py:74-78`) は `max_depth > 65535` で
`ValueError` を送出する．`depth` が `UInt16` で永続化されるため
(`schema.py:19`)．`--min-probability` が `click.FloatRange` で入口検証される
のに対し，こちらは `type=int` のみ (`build_game_graph.py:28-34`) なので，
利用者は preprocess の全ロードを終えた後になって初めて失敗に気づく．

Before:
```
| `--max-depth INT` | No | `30` | BFSの最大探索深さ．初期局面からの手数上限． |
```
After:
```
| `--max-depth INT` | No | `30` | BFSの最大探索深さ．初期局面からの手数上限．`depth` が UInt16 で保存されるため 65535 以下である必要があり，超過時は preprocess ロード後に `ValueError` となる． |
```

## Motivation

いずれも `/audit-and-fix` の step 4 で 1 件ずつコードと照合して確認した．
特に 5 (`metadata.json`) は静かに壊れる種類の欠落で，ドキュメント準拠の
出力が「読めるが全勝率が逆」という形で失敗する．1 と 3 は
`src/maou/domain/game_graph/` の 2 つの成果物 (`result_value` フィールドと
定跡機能) が，ドキュメント上でそれぞれ誤った場所に置かれている・
存在しないことになっている，という audit 対象そのものに関わる誤りである．

## Alternatives considered

- **`visualize.md` の該当箇所をまとめて書き直す．** 却下: 検証した主張と
  検証していない主張が混ざり，どこがコード照合済みかが失われる．今回照合
  したのは 147-194 行のうち具体的に真偽判定できるものだけで，それ以外に
  手を入れると証跡の意味が薄れる．
- **7, 10, 11 のような「今は正しいが陳腐化しやすい列挙」に対し，テストや
  pre-commit フックで守る仕組みを入れる．** 今回は却下 (別作業として分離):
  リポジトリには既に `check-cli-docs` フックと `benchmark-training-sync`
  skill という同種の仕組みがあり，どう統合するかはドキュメント修正より
  大きな設計判断になる．本提案は事実誤りの訂正に限る．
- **`openings.py` のドキュメントを本提案に含める.** 却下: 未文書化機能に
  新規ドキュメントを書き起こす作業は，既存記述のドリフト訂正とは性質も
  分量も異なる．一括承認の単位として不適切なので分離する (下記参照)．

## What this enables

- `result_value` / `best_move_win_rate` のどちらが描画に効くかを，コードを
  読まずにドキュメントから正しく辿れる．
- グラフディレクトリを外部で生成する際に `metadata.json` の必要性が分かり，
  先手視点変換が静かに壊れるのを防げる．
- `--max-depth` の実上限を実行前に知れる．

## What this constrains

- `visualize.md` のコントロール一覧と局面統計一覧は，UI に要素が増えるたび
  更新が必要な手動列挙のままである．本提案はこれを解消しない (上記
  alternatives の 2 番目を参照)．
- ノード色の記述を `best_move_win_rate` に直すことで，`result_value` が
  どこで使われるのかがドキュメント上どこにも書かれていない状態になる．
  ツールチップ用途である旨は別途補う余地がある．

## Rollback plan

`docs/commands/build_game_graph.md` と `docs/commands/visualize.md` の 2 ファイル
のみの変更なので，このコミットを revert すれば元に戻る．コードは一切変更
しないため，ビルド・テストへの影響はない．バージョン bump も伴わない．

## Follow-up (本提案に含めない)

- **`openings.py` が完全に未文書化** (`OpeningEntry` / `OpeningInfo` /
  `OpeningDatabase.find_opening` / 9 件の `_DEFAULT_OPENINGS`)．
  `docs/`，`CLAUDE.md`，`AGENTS.md`，`README.md` のいずれにも記述がなく，
  定跡名 (矢倉/相掛かり/角換わり/横歩取り模様/ゴキゲン中飛車/先手中飛車/
  振り飛車模様/相振り飛車模様) の grep も 0 件．最長一致の照合規則は
  非自明であり，かつユーザー可視 (上記 3) なので，意図的な省略ではなく
  純粋な欠落と判断する．新規ドキュメントの起草として別途扱う．
- `build_game_graph.md` の CLI オプション表 (7 件) と列定義表 (nodes 6 列 /
  edges 7 列) は現時点で完全に正確だが，手動列挙で守るものがない．特に列
  定義は `GameGraphIO._validate_schema` (`game_graph_io.py:193-207`) により
  ロード時のハード契約になっており，dtype を 1 つ誤ったドキュメント準拠の
  writer は「劣化」ではなく「ロード失敗」を引き起こす．
