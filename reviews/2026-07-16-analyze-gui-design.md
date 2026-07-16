---
title: analyze-gui コマンド新設 (棋譜解析 GUI, Gradio) の設計
date: 2026-07-16
status: approved
applied_in:
# user 承認 2026-07-16 (修正付き)．修正内容と適用状況は本文末尾
# 「承認記録」を参照．docs/commands/analyze_gui.md は実装 PR で適用予定．
target:
  - docs/design/game-analysis/gui.md
  - docs/design/game-analysis/index.md
  - docs/commands/analyze_gui.md
risk: low
reversibility: moderate
---

# 提案: `maou analyze-gui` — 棋譜解析インターフェース (Gradio GUI)

## 背景

user 要望 (2026-07-16):

> 前のセッションで棋譜解析コマンドを作ったので，これをもっと実用的に利用する
> ための棋譜解析インターフェース (GUI) を作成してください．
> - 将棋の盤面が表示されていて，棋譜全体の評価値やグラフも確認できる，
>   どの手が良かったり悪かったのかを確認しやすいインターフェース
> - 候補手を上位 N 個表示できたり，候補手を盤面上で矢印等で示せる
> - 提示された棋譜の手順から予測手順へと分岐する継盤機能
> - 任意の局面からインターフェースで手を指すことができて，その局面での
>   1 局面解析を実行する機能
> - 基本方針: 既存の Gradio を使った UI にする．棋譜解析ツールにどのような
>   機能があるとより便利かをウェブから調べたうえで設計する

## 既存ツールの機能調査 (2026-07-16, ウェブ)

| ツール | 本設計に取り込む機能 |
|---|---|
| ShogiGUI | 棋譜解析 (1 手ごと評価値+読み筋+一致率)，検討 (MultiPV 候補手数設定)，**盤上矢印 (最善手は PV 3 手分の連鎖矢印，次善以下は 1 手矢印)**，評価値グラフ，**継ぎ盤 (別盤で送り戻し・自由に指す・変化手順を本譜に分岐として追加)** |
| ShogiHome | 評価値/期待勝率の表示切替，候補手の読み筋一覧，読み筋の「再現」(別盤再生) と「棋譜に挿入」(分岐追加)，1 ウィンドウ完結 UI |
| lishogi / lichess | **勝率低下閾値による手の分類 (inaccuracy / mistake / blunder)**，クリック可能な評価値グラフ (クリックで局面ジャンプ)，悪手のリトライ学習 |
| ぴよ将棋 | グラフ上の悪手/疑問手マーカー (色 + 三角の向きで手番)，グラフ操作での局面ジャンプ |
| KENTO | Web ベース (サーバー解析)，局面+手順の文字列エクスポート/共有 |

出典: [ShogiGUI 棋譜解析](https://sites.google.com/site/shogigui/%E3%83%9E%E3%83%8B%E3%83%A5%E3%82%A2%E3%83%AB/%E6%A3%8B%E8%AD%9C%E8%A7%A3%E6%9E%90) /
[ShogiGUI 検討](https://sites.google.com/site/shogigui/%E3%83%9E%E3%83%8B%E3%83%A5%E3%82%A2%E3%83%AB/%E6%A4%9C%E8%A8%8E) /
[ShogiGUI 継ぎ盤](https://sites.google.com/site/shogigui/%E3%83%9E%E3%83%8B%E3%83%A5%E3%82%A2%E3%83%AB/%E3%81%9D%E3%81%AE%E4%BB%96%E3%81%AE%E6%A9%9F%E8%83%BD/%E7%B6%99%E3%81%8E%E7%9B%A4) /
[ShogiHome での棋譜の検討](https://imonote.hatenablog.jp/entry/2024/10/06/153336) /
[lishogi analysis](https://lishogi.org/analysis) /
[ぴよ将棋 解析の見方](https://you-kyan.work/piyo-shogi-analysis) /
[KENTO](https://www.kento-shogi.com/)

## 調査で確定した既存実装の前提 (2026-07-16)

- **盤面描画 (domain)**: `SVGBoardRenderer.render(position, highlight_squares,
  turn, record_id, move_arrow) -> str (SVG)`
  (src/maou/domain/visualization/board_renderer.py:113)．矢印 (`MoveArrow`,
  駒打ち対応)・マスのハイライト・持ち駒・手番バッジ・座標描画をサポート済み．
  **ただし矢印は 1 本のみ** — 候補手 N 本には拡張が必要
- **盤操作 (domain)**: `Board.set_sfen / get_sfen / get_turn /
  get_legal_moves / move_from_usi / push_move / pop_move /
  get_board_id_positions / get_pieces_in_hand`
  (src/maou/domain/board/shogi.py) — 「任意局面で合法手を列挙して指す」の
  フルセットが既にある
- **日本語表記**: `GameGraphVisualizationInterface.usi_to_japanese` ほか
  (src/maou/interface/game_graph_visualization.py:1053) が「7六歩」形式を生成
- **解析エンジン**: `SearchEngine` (モデル 1 回ロード + 局面ごと search，
  GIL 解放) と `GameAnalyzer` (1 局一括解析 → JSON dict)．per-position JSON は
  `ply / side_to_move / sfen / played_move / best_move / match / winrate /
  eval_cp / played_move_winrate / winrate_loss / pv / candidates(usi, visits,
  winrate, prior, proven) / mate_found / playouts / elapsed_ms / stop /
  record_*` (src/maou/app/analysis/game_analyzer.py:494)．
  **winrate / eval_cp は手番視点** (docs/design/game-analysis/index.md §7)
- **PV は最善手系列のみ** (`SearchResult.pv`)．候補手ごとの PV は
  `SearchRootChild` に無い
- **Gradio 前例**: `maou visualize` が gr.Blocks + gr.HTML (SVG 盤面) +
  gr.Plot (Plotly) + gr.Dataframe + gr.State の構成
  (src/maou/infra/visualization/gradio_server.py)．gradio は
  `visualize` extra (遅延 import + 導入案内)，**plotly はコア依存**
- ply 軸の評価値折れ線グラフは既存に無い (新規．Plotly `go.Scatter` で
  追加依存なし)

## 設計

### CLI

```
maou analyze-gui
  [--input-path FILE]            # CSA/KIF．起動時ロード (GUI からもアップロード可)
  [--report PATH]                # analyze-game の JSON レポート読込 (閲覧モード)
  [--model-path PATH]            # 省略時 mock 評価器 (開発検証専用と GUI に明示)
  [--time-ms N | --playouts P]   # GUI 内解析のデフォルト予算 (default 1000ms)
  [--num-candidates K]           # 候補手表示数の初期値 (default 5)
  [探索 passthrough: --threads --batch-size
   --root-dfpn/--no-root-dfpn --root-dfpn-nodes --root-dfpn-depth
   --leaf-mate/--no-leaf-mate --leaf-mate-nodes --leaf-mate-threads
   --cuda/--no-cuda --tensorrt/--no-tensorrt --trt-cache-dir]
  [--port N] [--share] [--server-name HOST]   # visualize と同名
```

- 探索 passthrough・実行プロバイダの選択肢は `maou search` / `maou analyze-game`
  と同一に維持する (analyze-game 設計の user 指示に従う)．
- gradio は visualize と同じ遅延 import (`uv sync --extra visualize` を案内)．
- 利用モード 3 態:
  1. **棋譜 + モデル**: GUI 内で一括解析も 1 局面解析も可能 (フル機能)
  2. **棋譜 + `--report`**: 解析済み JSON の閲覧 (モデル不要．CPU 環境や
     Colab で解析した結果をローカルで見る用途)．1 局面解析ボタンは無効化
  3. **棋譜のみ**: 盤面再生と分岐 (継盤) のみ．解析系は無効化

### レイヤー構成 (既存パターン踏襲)

| 層 | ファイル | 責務 |
|---|---|---|
| infra/console | `src/maou/infra/console/analyze_gui.py` | click コマンド + `LAZY_COMMANDS` 登録 |
| infra/visualization | `src/maou/infra/visualization/analysis_gui_server.py` | gr.Blocks 構築・イベント配線・サーバー起動 |
| interface | `src/maou/interface/analysis_gui.py` | 表示整形 (SVG/テーブル/グラフ用データ/日本語表記)．`usi_to_japanese` 等を再利用 |
| app | `src/maou/app/analysis/analysis_session.py` | `AnalysisSession` usecase = 分岐木 + 解析キャッシュ + エンジン呼び出し |
| domain | `src/maou/domain/visualization/board_renderer.py` | 複数矢印拡張 (下記) |

- `GradioVisualizationServer` (3801 行) には統合しない．責務が異なる
  (データセット可視化 vs 棋譜解析) ため別サーバーモジュールとし，
  `game_graph_shared.py` の共有部品 (空プロット等) は再利用する．
- app 層が `maou._rust` (SearchEngine / parse_csa_str / parse_kif_str) を
  直接呼ぶのは `app/analysis/game_analyzer.py` と同じ扱い．

### 画面構成 (gr.Blocks)

```
┌────────────────────────┬──────────────────────────────┐
│ 盤面 (gr.HTML, SVG)     │ タブ: [グラフ] [棋譜] [候補手]  │
│  - 最終手ハイライト      │  グラフ: 評価値/勝率推移        │
│  - 候補手矢印 (トグル)   │   + 悪手マーカー + 現在位置線   │
│ ナビ: |◀ ◀ ▶ ▶| スライダー│  棋譜: 指し手リスト            │
│ 分岐パンくず + 本譜へ戻る │   (日本語表記/評価値/損失/分類) │
│ 手入力状態 (成り確認等)   │  候補手: 上位 N テーブル        │
├────────────────────────┤   (指し手/訪問数/勝率/prior/詰み)│
│ 局面情報: SFEN・position │ 解析: [この局面を解析][全局面解析]│
│ 文字列コピー・棋譜コメント │  予算設定・進捗・キャンセル      │
└────────────────────────┴──────────────────────────────┘
サマリ帯: 対局者・結果・一致率 (先手/後手)・平均勝率損失・worst moves
```

### 評価値グラフと手の分類

- x = ply，y = **先手視点の勝率** (デフォルト) / 評価値 cp (切替)．
  JSON の `winrate` / `eval_cp` は手番視点なので，表示時に
  `side_to_move == "w"` の行を `1 - winrate` / `-eval_cp` に変換する
  (この変換は interface 層の純関数とし単体テストを書く)．
- 手の分類は lishogi 方式で `winrate_loss` 閾値により
  **疑問手 (?!) / 悪手 (?) / 大悪手 (??)** を判定．デフォルト閾値は
  0.05 / 0.10 / 0.20 (設定 UI で変更可)．`played_move_winrate` が null
  (実戦手が未訪問) の場合は分類しない．`match == true` は ✓ 表示．
- グラフにはぴよ将棋方式で分類マーカーを重畳 (色で深刻度，`mate_found` は
  ★)．現在表示中の ply に縦線．
- グラフクリックでの局面ジャンプは gr.Plot (Plotly) がイベント非対応の
  可能性があるため，実装時に gr.LinePlot (`.select`) で可否を検証する．
  不可の場合はナビスライダーと棋譜リスト行クリックで代替 (機能等価)．

### 候補手の表示と矢印 (SVGBoardRenderer 拡張)

- 候補手テーブル: `candidates` 上位 N (N は 1〜num-candidates のスライダー)
  を順位/日本語表記/訪問数/勝率 (手番視点のまま明示)/prior/詰み確定で表示．
- **盤上矢印**: `render` の矢印引数を複数化する．
  `move_arrows: list[ArrowSpec]` (`ArrowSpec` = 既存 `MoveArrow` +
  `color / width / opacity / label`)．既存の `move_arrow` 引数は当面残して
  後方互換 (内部で 1 要素リストに変換)．
  - 最善手 = 濃色・太線．2 位以下は訪問数比で透明度を下げる (ShogiGUI の
    「最善は目立ち，次善以下は控えめ」の慣行)．
  - 最善手のみ PV 先頭 3 手を連鎖矢印表示するオプション (ShogiGUI 方式．
    データは `pv` にあり)．
  - 直前手ハイライト (既存 highlight_squares) と併用．
- 候補手テーブルの行クリック = その手で分岐 (下記継盤へ)．

### 継盤 (分岐) モデル — app 層 `AnalysisSession`

```python
@dataclass
class VariationNode:
    move_usi: str | None            # None = 初期局面 (root)
    parent: "VariationNode | None"
    children: list["VariationNode"]
    is_mainline: bool               # 本譜の手か
    analysis: dict | None           # この手の直前局面の解析結果キャッシュ
```

- ロード時に `GameRecord.moves` から本譜チェーンを構築．analyze-game JSON
  (`positions[i]`) は対応する本譜ノードのキャッシュに取り込む．
- **盤面で手を指す / 候補手行をクリック / PV 再生** はすべて「現在ノードの
  子に手を追加して移動」に統一 (同じ手の子が既にあれば再利用)．本譜から
  外れた時点で自動的に分岐が生まれる — ShogiHome の「棋譜に挿入」が暗黙に
  行われる形．
- パンくず表示: `本譜 42手目 ▶ △8四飛 ▶ ▲2四歩 …`．「本譜へ戻る」で
  分岐点の本譜側へ復帰．分岐は複数保持でき，セッション中は消えない．
- 現局面の `position` 文字列 (`position sfen ... moves ...`) と SFEN を
  常時エクスポート表示 (KENTO の共有性の代替．他ツールへの持ち出し口)．
- 分岐手順の KIF エクスポートはスコープ外 (未決事項へ)．

### 任意局面での 1 局面解析

- 「この局面を解析」ボタン: 現在ノードの局面を `SearchEngine.search(
  sfen=初期SFEN, moves=root からの USI 経路, ...)` で解析．経路を渡すのは
  千日手履歴を正しく効かせるため (GameAnalyzer と同じ規約)．
- 予算は UI の time-ms / playouts (デフォルトは CLI 指定値)．
- 結果はノードにキャッシュし，候補手テーブル・矢印・局面情報を更新．
  再訪時は再解析しない (明示的な「再解析」ボタンで上書き)．
- 「全局面解析」ボタン: 本譜全体を GameAnalyzer 相当のループで解析
  (進捗バー + キャンセル)．結果はグラフ・棋譜リストに反映され，JSON として
  ダウンロード可能 (analyze-game の出力スキーマと同一 — CLI と GUI で
  レポートの相互運用を保つ)．

### 盤面クリック入力

- SVG の各マス + 持ち駒に透明 rect (`data-square` 属性) を重ね，クリックで
  hidden `gr.Textbox` に値を書いて Python コールバックを発火させる
  (JS は `gr.Blocks(js=...)` で注入)．
- 2 クリック方式: 1 回目 = 自駒/持ち駒選択 → 合法手の行き先を
  `highlight_squares` で表示．2 回目 = 行き先確定．成/不成が両方合法なら
  確認ボタンを表示．非合法クリックは選択解除．
- 合法手判定は `Board.get_legal_moves()` の列挙をフィルタするだけで済む
  (新規ロジック不要)．
- フォールバックとして合法手 Dropdown (日本語表記) も併設 (Gradio の
  HTML 内 JS サニタイズで clickable SVG が不成立だった場合も機能を失わない．
  テストもこちら経由で書ける)．

### 状態管理と並行性

- `SearchEngine` はサーバープロセスで 1 個 (モデル 1 回ロード)．
  探索系イベントは Gradio queue の `concurrency_limit=1` で直列化する
  (同時探索によるメモリ/EP 資源の競合を避ける．8GB DevContainer 制約)．
- セッション状態 (棋譜・分岐木・現在ノード・解析キャッシュ) は `gr.State`
  (ブラウザセッション独立，エンジンのみ共有)．
- 一括解析は長時間になり得るためキャンセルフラグを最初から実装する．

### テスト計画

- 視点変換 (手番視点 → 先手視点)・分類閾値の純関数単体
- `VariationNode` 木: 追加/同一手の子再利用/本譜復帰/root からの USI 経路生成
- `SVGBoardRenderer` 複数矢印: N 本・色/透明度・駒打ち矢印・後方互換
  (`move_arrow` 単数引数)
- `AnalysisSession`: mock 評価器で 1 局面解析 → キャッシュヒット → 再解析上書き
- interface 整形: analyze-game JSON fixture → グラフ用データ/棋譜テーブルの
  golden
- CLI: CliRunner でオプション検証 (サーバー起動はモック)．gr.Blocks の
  demo 構築が例外なく通るスモーク．ブラウザ e2e は対象外 (手動確認)

### 実装マイルストーン

1. **閲覧機能** — 棋譜ロード + 盤面再生 + `--report` JSON 読込 + 評価値
   グラフ/分類/棋譜リスト + 候補手テーブル/複数矢印 (renderer 拡張含む)
2. **対話解析機能** — 分岐木 + 盤面クリック入力 + 1 局面解析 + 全局面解析
   (エンジン常駐 + 直列化 + キャンセル)

PR は 1 と 2 で分割可能な構造にする (1 だけでもモデル無し環境で価値がある)．

### 版数 (実装時)

- maou 0.43.0 → 0.44.0 (feat: 新コマンド + renderer 拡張)
- Rust 変更なしの見込み (既存 `SearchEngine` / parse API で完結)．
  候補手ごとの PV (下記未決 4) を採る場合のみ maou_search / maou_rust minor

## 提案する docs 変更 (本 review の approve 対象)

1. **docs/design/game-analysis/gui.md (新規)**: 本設計を設計ドキュメントとして
   常設 (章立て: 目的とスコープ / 既存ツール調査 / CLI / レイヤー構成 /
   画面構成 / グラフと手の分類 / 候補手と矢印 / 継盤モデル / 1 局面解析 /
   盤面入力 / 状態管理 / テスト / マイルストーンと未決事項)．
   position-search の複数ファイル慣行 (index.md + benchmarking.md) に倣い
   game-analysis 配下の 2 枚目とする
2. **docs/design/game-analysis/index.md**: §9 未決事項の「解析結果の可視化
   (勝率グラフ等) との接続」を gui.md への参照に更新
3. **docs/commands/analyze_gui.md (新規)**: CLAUDE.md MUST に従い実装 PR 内で
   既存フォーマット (Overview / CLI options / Example) で作成

## 未決事項 (user 確認ポイント)

1. コマンド名 `analyze-gui` でよいか (代案: `review-game`, `kifu-gui`)
2. 悪手分類のデフォルト閾値 0.05 / 0.10 / 0.20 (勝率損失) でよいか
3. マイルストーン 1 (閲覧) と 2 (対話解析) を別 PR に分割してよいか
4. 候補手ごとの PV (Rust `SearchRootChild` 拡張が必要)．MVP は
   「最善手のみ PV 連鎖矢印 + 候補手は 1 手矢印」で開始し，候補手 PV は
   分岐機能 (候補手クリック → その局面を解析) で代替する方針でよいか
5. 分岐手順の KIF/KI2 エクスポートと「悪手リトライ学習」(lishogi の
   learn from your mistakes 相当) は将来拡張として据え置きでよいか

## 承認記録 (2026-07-16)

user は本提案を **修正付きで承認**した:

1. **手の良し悪しの自動分類 (疑問手/悪手/大悪手のバッジ・グラフマーカー) は
   延期** — 勝率損失の閾値がモデル出力・探索アルゴリズムに依存して変わり
   得るため，実モデルでの解析実績を見てから設計する (未決事項 2 は閾値の
   採否ごと延期)．JSON 生値 (`winrate_loss`) の列表示と事実情報
   (一致 ✓ / 詰み ★) の表示は維持する
2. **実装の出来は Gradio サーバーを起動し playwright のスクリーンショットで
   確認する** (検証手順の指示)
3. 未決事項 1 / 3 / 4 / 5 は提案どおり (コマンド名 `analyze-gui`，
   2 PR 分割，候補手 PV は分岐で代替，KIF エクスポート/リトライ学習は将来)

上記修正を反映した設計本文は docs/design/game-analysis/gui.md (適用済み)
が正．本 review の設計節は起草時点の内容として保存する．

## 根拠

- user 要望 5 点 (盤面/グラフ/良悪手可視化/候補手矢印/継盤+任意局面解析)
  をすべて既存資産 (SVGBoardRenderer, Board, SearchEngine, analyze-game
  JSON, Gradio 前例) の上で実現でき，新規依存はゼロ (plotly コア依存済み，
  gradio visualize extra 済み)
- 既存ツール調査により，一致率/分類バッジ/矢印の強弱/継ぎ盤の「本譜に
  分岐追加」/グラフマーカーという定番 UX を確認し設計に反映した
- VETO 整合: 時間配分は GUI でも予算指定のみ (探索側 API 不変更)．wheel
  可搬性に影響なし (Python 層のみ．Rust 変更なしの見込み)．main 直コミット
  なし (feat/analyze-gui ブランチ)
