# 棋譜解析 GUI (analyze-gui) 設計ドキュメント

> **状態: 実装済み (マイルストーン 1・2 とも，living document)**．実装の進行に
> 合わせて本ドキュメントを更新する．各節に「実装済み」「設計方針 (未実装)」
> 「未決」のいずれかを明記する．
> 実装済み記述の正は常にコードであり，乖離を見つけたら本ドキュメント側を直す．
> 提案・承認の経緯: reviews/2026-07-16-analyze-gui-design.md
> 前提となる解析コマンドの設計: [index.md](index.md)

## 1. 目的とスコープ (設計方針)

`maou analyze-game` の解析結果を実用的に活用するための **Gradio ベースの
棋譜解析 GUI** `maou analyze-gui`．

- 盤面表示 + 棋譜全体の評価値グラフで，どの手が良かった/悪かったかを
  確認しやすくする
- 候補手を上位 N 個表示し，盤上に矢印で示す
- 継盤 (分岐): 本譜の手順から予測手順へ分岐して検討できる
- 任意の局面で GUI から手を指し，その局面の 1 局面解析を実行できる
- **スコープ外**: 対局機能，複数エンジン比較，分岐手順の KIF/KI2
  エクスポート (将来拡張)，悪手リトライ学習 (将来拡張)，
  手の良し悪しの自動分類 (延期 — §6)

## 2. 既存ツール調査の反映 (設計方針)

2026-07-16 のウェブ調査 (詳細と出典は reviews/2026-07-16-analyze-gui-design.md):

| ツール | 本設計に取り込んだ機能 |
|---|---|
| ShogiGUI | 盤上矢印 (最善手は PV 連鎖矢印，次善以下は 1 手矢印)，評価値グラフ，継ぎ盤 (別盤で自由に指す・変化手順を本譜に分岐として追加) |
| ShogiHome | 評価値/期待勝率の表示切替，候補手一覧，読み筋の分岐追加 |
| lishogi | 勝率低下閾値による手の分類 (疑問手/悪手/大悪手 — 採用は延期，§6)，グラフからの局面ジャンプ |
| ぴよ将棋 | グラフ上の悪手マーカー (色で深刻度) |
| KENTO | 局面+手順の文字列エクスポート |

## 3. CLI (実装済み)

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

- 探索 passthrough・実行プロバイダの選択肢は `maou search` /
  `maou analyze-game` と同一に維持する．
- gradio は visualize と同じ遅延 import (`uv sync --extra visualize` を案内)．
- 利用モード 3 態:
  1. **棋譜 + モデル**: GUI 内で一括解析も 1 局面解析も可能 (フル機能)
  2. **棋譜 + `--report`**: 解析済み JSON の閲覧 (モデル不要)
  3. **棋譜のみ**: 盤面再生と分岐 (継盤)
- モデル未指定 (モード 2/3) でも解析ボタンは使えるが，決定論的な
  **mock 評価器**で動作し UI に「開発検証専用」と明示する
  (`analyze-game` の `--model-path` 省略時と同じ扱い)．当初案の
  「モデル未指定時は解析ボタン無効化」は本節の mock 記述と矛盾して
  いたため，実装時に mock + 明示へ一本化した (2026-07-17)．

## 4. レイヤー構成 (実装済み)

| 層 | ファイル | 責務 |
|---|---|---|
| infra/console | `src/maou/infra/console/analyze_gui.py` | click コマンド + `LAZY_COMMANDS` 登録 |
| infra/visualization | `src/maou/infra/visualization/analysis_gui_server.py` | gr.Blocks 構築・イベント配線・サーバー起動 |
| interface | `src/maou/interface/analysis_gui.py` | 表示整形 (SVG/テーブル/日本語表記/クリック状態機械/パンくず)．`usi_to_japanese` 等を再利用 |
| interface | `src/maou/interface/analysis_workbench.py` | ワークベンチ HTML の組み立て (3 カラム 1 枚・評価値グラフ SVG・悪手リスト) |
| static | `src/maou/static/analysis_workbench.{css,js}` | Modernist テーマのスコープ付き CSS・委譲リスナー・ショートカット配線 |
| app | `src/maou/app/analysis/analysis_session.py` | 棋譜ロード・分岐木 (`VariationTree`)・合法手列挙 |
| app | `src/maou/app/analysis/interactive_analyzer.py` | 常駐エンジン (`InteractiveAnalyzer`)・1 局面/全局面解析・レポート組み立て |
| domain | `src/maou/domain/visualization/board_renderer.py` | 複数矢印拡張・クリック標的 rect・選択/行き先の塗り |

- `GradioVisualizationServer` (データセット可視化) には統合しない．
  `game_graph_shared.py` の共有部品は再利用する．
- app 層が `maou._rust` (SearchEngine / parse_csa_str / parse_kif_str) を
  直接呼ぶのは `app/analysis/game_analyzer.py` と同じ扱い．

## 5. 画面構成 (実装済み)

**3 カラムのワークベンチ**を `gr.HTML` **1 枚**で描画する (user 決定
2026-08-22 / Claude Design 案 1a)．タブは全廃し，悪手・棋譜・盤面・
グラフ・候補手・解析パネルを**常時同時表示**する．

```
ヘッダー: 対局者・結果・エンジン種別・レポート有無
サマリ帯: 一致率 (先手/後手)・平均勝率損失・悪手数・解析済み手数
┌───────────────┬─────────────────────┬──────────────────┐
│ 悪手 (損失≥閾値) │ 分岐タグ + パンくず   │ 評価値グラフ (SVG) │
│  手数/表記/損失  │  + 本譜へ戻る        │  勝率帯・悪手 ●    │
│  クリックでジャンプ│                    │  詰み ★・現在位置線 │
├───────────────┤ 盤面 (SVG, 52px セル) │  クリックでジャンプ │
│ 棋譜            │  最終手ハイライト     ├──────────────────┤
│  日本語表記/✓/   │  候補手矢印 (トグル)  │ 候補手 上位 N      │
│  勝率/損失/★    │  選択・行き先の塗り   │  順位/表記/訪問数/  │
│  閾値以上は赤字  │                     │  勝率/prior/詰み   │
│  残り高さいっぱい │ ナビ: |◀ ◀ ▶ ▶|     │  クリックで分岐     │
│  でスクロール    │ 手入力状態 (成り確認) ├──────────────────┤
│                │ 局面情報: SFEN/       │ 解析パネル         │
│                │  position 文字列      │  予算・進捗バー     │
│                │  棋譜コメント         │  ステータス行       │
│                │                     │  [この局面][全局面] │
│                │                     │  [キャンセル][保存] │
└───────────────┴─────────────────────┴──────────────────┘
ショートカット帯: ←→ / ⇧←⇧→ / Home End / B / Space / 1-5 / L / Esc
```

- 想定画面幅は**ノート PC 1280〜1440px** (user 指定)．左右レールを
  284/304px に固定し，中央カラムに 52px セル (676px) の盤面が 1280px で
  収まるようにする．ワークベンチ全体は `min-width: 1280px`．
- 見た目は Modernist (Archivo・赤単色・角丸ゼロ・2px 罫)．
  Google Fonts を `demo.launch(head=...)` から読み，オフラインでは
  system-ui / Hiragino Mincho にフォールバックする (レイアウトは不変)．
- 盤面の配色・寸法・書体は `BoardTheme` に外出しし，ワークベンチは
  `MODERNIST_BOARD_THEME` を渡す．**引数なしの `SVGBoardRenderer()` の
  出力は従来と同一** (`visualize` / game-graph は無変更)．

## 6. 評価値グラフ (実装済み)

- x = ply，y = 縦軸 4 択: **先手/後手 × 勝率/評価値** (デフォルトは
  先手勝率)．analyze-game JSON の `winrate` / `eval_cp` は手番視点
  なので，表示時に `side_to_move == "w"` の行を `1 - winrate` /
  `-eval_cp` に変換する (interface 層の純関数とし単体テストを書く)．
  後手視点は先手視点の鏡映 (勝率 1 - x / 評価値符号反転) — 同じ 1 本の
  曲線を後手の目線で読むための表示切替であり，解析自体は手番対称．
- 局面情報の注記に**現局面のエンジン評価行** (勝率/評価値を手番視点 +
  先手視点で併記，playouts，詰み発見) を表示する — 分岐局面の
  「この局面を解析」の結果もここで確認できる (ノードの解析キャッシュが
  ソースなので本譜/分岐とも同じ経路)．
- `mate_found` (詰み発見 = 事実情報) は ★ マーカーで重畳表示．
  現在表示中の ply に縦線．
- 棋譜リストには JSON の生値 (`winrate_loss` 等) をそのまま列表示する．
  `match == true` は ✓ 表示 (エンジン最善との一致 = 事実情報)．
- **手の良し悪しの自動分類 (疑問手/悪手/大悪手のバッジ・グラフマーカー) は
  延期** (user 決定 2026-07-16)．勝率損失の閾値がモデル出力・探索
  アルゴリズムに依存して変わり得るため，実モデルでの解析実績を見てから
  設計する．
- **悪手ジャンプリストは「分類」ではなく閾値フィルタ 1 本**として実装する
  (user 承認 2026-08-22)．上の延期決定の理由 (閾値がモデル依存) を次の
  かたちで温存する:
  - 3 段階分類も固定閾値も導入しない — 勝率損失が閾値以上の手を列挙する
    フィルタのみ
  - **閾値は UI 上で可変** (既定 0.10，見出しの数値を直接編集)．左の
    悪手リストと棋譜リストの赤字が連動する
  - 「疑問手」等のバッジ文言は付けず**数値のみ**表示する
- グラフ本体は**インライン SVG** を interface 層で組み立てる
  (`analysis_workbench.py`)．`gr.Plot` は `gr.HTML` 1 枚の中に置けない
  ため Plotly から移行した (2026-08-22)．
- **グラフクリックでの局面ジャンプは実装済み** — プロット領域のクリックで
  その手数へ，悪手の ● と詰みの ★ は個別にジャンプする．gr.Plot 時代の
  「非対応なら棋譜リストで代替」という未決事項はこれで解消した．

## 7. 候補手の表示と矢印 (設計方針)

- 候補手テーブル: `candidates` 上位 N (N は 1〜num-candidates の
  スライダー) を順位/日本語表記/訪問数/勝率 (手番視点のまま明示)/prior/
  詰み確定で表示．
- **盤上矢印**: `SVGBoardRenderer.render` の矢印引数を複数化する．
  `move_arrows: list[ArrowSpec]` (`ArrowSpec` = 既存 `MoveArrow` +
  `color / width / opacity / label`)．既存の `move_arrow` 引数は残して
  後方互換 (内部で 1 要素リストに変換)．
  - 最善手 = 濃色・太線．2 位以下は訪問数比で透明度を下げる
  - 最善手のみ PV 先頭 3 手を連鎖矢印表示するオプション (データは `pv`)
  - 直前手ハイライト (既存 highlight_squares) と併用
- 候補手ごとの PV は Rust 拡張が必要なため採らない (user 承認 2026-07-16)．
  候補手クリック → その手で分岐して解析，で代替する．

## 8. 継盤 (分岐) モデル (実装済み)

app 層 `analysis_session.py` が分岐木を保持する．ノードは親子を
**ID で参照する配列格納** (arena) とし，`gr.State` の deepcopy に耐える
plain data にする (実装が正):

```python
@dataclass
class VariationNode:
    node_id: int                    # VariationTree.nodes の索引
    parent_id: int | None           # root は None
    move_usi: str | None            # None = 初期局面 (root)
    snapshot: PositionSnapshot      # この手を指した後の局面
    children: list[int]
    is_mainline: bool               # 本譜のノードか
    analysis: dict | None           # このノードの局面の解析結果キャッシュ

@dataclass
class VariationTree:
    nodes: list[VariationNode]
    mainline_ids: list[int]         # 索引 = ply
    current_id: int
```

- 解析キャッシュは「**そのノードの局面を探索した結果**」を持つ
  (analyze-game の `positions[i]` は「i+1 手目を指す直前の局面 =
  本譜ノード i」に対応)．当初案の「この手の直前局面」表現と同じ対応を
  ノード自身に付け替えたもの．
- ロード時に本譜チェーンを構築．analyze-game JSON (`positions[i]`) は
  対応する本譜ノードのキャッシュに取り込む (`apply_report_to_tree`)．
- **盤面で手を指す / 候補手行をクリック / PV 再生** はすべて
  `advance_move` (「現在ノードの子に手を追加して移動」，同じ手の子が
  既にあれば再利用) に統一．本譜から外れた時点で自動的に分岐が生まれる．
- パンくず表示: `本譜 42手目 ▶ △8四飛 ▶ ▲2四歩 …`．「本譜へ戻る」で
  分岐点の本譜側へ復帰．分岐は複数保持でき，セッション中は消えない．
- 現局面の `position` 文字列 (`position sfen ... moves ...`) と SFEN を
  常時エクスポート表示する．

## 9. 任意局面での 1 局面解析 (実装済み)

- 「この局面を解析」ボタン: 現在ノードの局面を `SearchEngine.search(
  sfen=初期SFEN, moves=root からの USI 経路, ...)` で解析．経路を渡すのは
  千日手履歴を正しく効かせるため (GameAnalyzer と同じ規約)．
- 予算は UI の time-ms / playouts (デフォルトは CLI 指定値，未指定は
  1000ms)．
- 結果はノードにキャッシュし，候補手テーブル・矢印・局面情報を更新．
  再訪時は再解析しない (明示的な「再解析 (上書き)」ボタンで上書き)．
  本譜ノードは実戦手比較 (`played_move` / `match` / `winrate_loss`)
  付き，分岐ノードは実戦手なし (`played_move: null`) の記録になる．
- 「PV を分岐で再生」ボタン: 現在ノードの解析 PV を分岐として一括適用する
  (§8 の `advance_move` 経路)．
- 「全局面解析」ボタン: 本譜全体を `InteractiveAnalyzer.analyze_mainline`
  (ジェネレータ) で解析し，進捗をテキスト表示する．キャンセルは協調
  フラグ (実行中の 1 局面は完了を待って停止．解析済み局面はキャッシュに
  残る)．完走時は analyze-game の出力スキーマと同一の JSON として
  ダウンロード可能 (CLI と GUI でレポートの相互運用を保つ)．グラフ・
  棋譜リスト・サマリも生成レポートで更新される．

## 10. 盤面クリック入力 (実装済み)

- SVG の各マス + 持ち駒 (枚数 > 0 の表示行) に透明 rect
  (`data-click="sq:{square}"` / `data-click="hand:{b|w}:{piece_type}"`,
  square は column-major) を重ねる (`SVGBoardRenderer.render` の
  `interactive=True`)．
- JS → Python は hidden Textbox では**なく** `gr.HTML` の
  `server_functions` + `js_on_load` ブリッジを使う (Gradio 6 では JS で
  Textbox の値を変えても `.change()` が発火しない — game_graph_server.py
  と同じパターン)．クリック委譲リスナーは `demo.launch(head=...)` で
  注入し，永続する `#board-display` コンテナに 1 回だけ付ける．
- **ブリッジ .change の制約 (実測 2026-07-17)**: server_functions の
  `trigger("change")` 経由イベントは value 更新のみ反映され，
  `gr.update` の prop 更新 (visible / choices) が適用されない．
  - ワークベンチ化 (2026-08-22) で**この制約はむしろ利点になった** —
    出力が HTML 文字列 1 個だけになり，すべて value 更新で済むため
    `.then()` の連鎖が不要になった．成/不成ボタンと合法手一覧も
    HTML の一部として描き直される．
- **ブリッジのレーン分割 (2026-08-22)**: 全 UI 操作を `data-action`
  文字列に符号化し，用途ごとにブリッジを 5 本に分ける．

  | レーン | 用途 | 直列化 |
  |---|---|---|
  | nav | ナビ・盤面クリック・分岐・表示オプション | なし |
  | engine | 1 局面解析 / 再解析 | `concurrency_id="engine"` |
  | engineAll | 全局面解析 (ジェネレータ) | 同上 |
  | cancel | キャンセル | なし (実行中でも通す) |
  | download | レポート JSON 保存 | なし |

  盤面クリック・棋譜行・候補手行・悪手行・グラフはすべて委譲リスナー
  1 つで拾い，`data-action` で分岐する (再描画のたびに付け直さない)．
- 2 クリック方式 (interface 層 `handle_board_click` 純関数): 1 回目 =
  自駒/持ち駒選択 → 合法手の行き先を専用色で塗る (選択 = アンバー，
  行き先 = グリーン)．2 回目 = 行き先確定．成/不成が両方合法なら
  「成る/成らず」確認ボタンを表示．非合法クリックは選択解除，選択中の
  別自駒クリックは選択切替．
- 合法手判定は `Board.get_legal_moves()` の列挙 (`legal_move_infos`) を
  フィルタするだけ．
- フォールバックとして合法手 Dropdown (日本語表記) + 「指す」ボタンも
  併設 (clickable SVG が不成立でも機能を失わない．ユニットテストも
  こちら経由)．

## 11. 状態管理と並行性 (実装済み)

- `SearchEngine` はサーバープロセスで 1 個 (`InteractiveAnalyzer` が遅延
  構築しモデルは 1 回ロード)．探索系イベント (1 局面解析/再解析/全局面
  解析) は `concurrency_id="engine"` + `concurrency_limit=1` で直列化する
  (メモリ/EP 資源の競合を避ける．8GB DevContainer 制約)．キャンセル
  ボタンは別レーンで即時実行される．
- セッション状態 (棋譜 `SessionView`・分岐木 `VariationTree`・クリック
  状態 `ClickState`) は `gr.State` (ブラウザセッション独立，エンジンのみ
  共有)．`gr.State` の初期値はセッションごとに deepcopy されるため，
  状態には PyO3 オブジェクト (Board 等) を持たせず plain data のみとする．
- 一括解析は長時間になり得るためキャンセルフラグ (threading.Event) を
  実装済み．
- 制約: 盤面クリックブリッジの受け渡しバッファ (`pending_click`) は
  クロージャとして全ブラウザセッションに共有される (game_graph_server の
  `_pending` と同じ)．ローカル解析ツールとして単一利用者を前提とする．
- 局面スライダーは `.release` イベントで配線する (ナビゲーションボタン等
  がスライダー値をプログラム更新しても再描画が二重に走らない．`.change`
  はプログラム更新でも発火するため使わない)．

## 12. テスト (実装済み)

- 視点変換 (手番視点 → 先手視点) の純関数単体
- `VariationTree`: 追加/同一手の子再利用/本譜復帰/root からの USI 経路
  生成/レポート取込/deepcopy 可能性
- 盤面クリック状態機械: 選択/解除/切替/行き先確定/成・不成保留/駒打ち/
  相手持ち駒の無視
- `SVGBoardRenderer` 複数矢印: N 本・色/透明度・駒打ち矢印 (手番側始点)・
  後方互換．クリック標的 rect・選択/行き先の塗り
- `InteractiveAnalyzer`: mock 評価器で 1 局面解析 (本譜/分岐)・全局面
  解析 + キャンセル・レポート組み立て．サーバー側でキャッシュヒット →
  再解析上書き
- interface 整形: analyze-game JSON fixture → グラフ用データ/棋譜テーブルの golden
- ワークベンチ HTML (`analysis_workbench.py`): 3 カラムの各ブロックが
  描画されること・悪手フィルタの閾値連動・評価値グラフ SVG の座標
  (勝率帯/悪手 ●/詰み ★/現在位置線)・`data-action` 属性の符号化
- `data-action` 文字列の解釈 (`analysis_gui_server.py`): 各レーンが
  対応するアクションだけを受け付けること
- `BoardTheme`: 既定テーマの盤面 SVG が変更前と**バイト単位で同一**で
  あること (`visualize` / game-graph への非回帰)
- CLI: CliRunner でオプション検証 (サーバー起動はモック)．gr.Blocks の
  demo 構築が例外なく通るスモーク
- **実装の出来は Gradio サーバーを起動し playwright のスクリーンショットで
  確認する** (user 指示 2026-07-16)．自動 e2e テストとしては保守しない
  (検証手順として実施)．ワークベンチ化では棋譜行クリック・←→・Home/End・
  B・L・Space・盤面 2 クリックでの着手・本譜外の手による自動分岐・
  表示オプションの往復を実機で確認した (2026-08-22)

## 13. マイルストーンと決定事項

実装は 2 段階に分割する (user 承認 2026-07-16)．**両マイルストーンとも
実装済み** (1: maou 0.44.0 / 2: maou 0.45.0):

1. **閲覧機能** — 棋譜ロード + 盤面再生 + `--report` JSON 読込 + 評価値
   グラフ/棋譜リスト + 候補手テーブル/複数矢印 (renderer 拡張含む)
2. **対話解析機能** — 分岐木 + 盤面クリック入力 + 1 局面解析 + 全局面解析
   (エンジン常駐 + 直列化 + キャンセル)

決定事項 (user 承認 2026-07-16):

- コマンド名は `analyze-gui`
- **手の良し悪しの自動分類は延期** — 閾値がモデル出力・探索アルゴリズムに
  依存するため，実モデルでの解析実績を見てから設計する

3 段階目として **UI のワークベンチ化**を実施した (user 決定 2026-08-22)．
Claude Design で 3 案 (3カラム / 盤面主役2カラム / 全画面盤面+タイムライン)
を検討し **1a 3カラム・ワークベンチ**を採用．タブ 3 枚 + 縦長レイアウトを
`gr.HTML` 1 枚の 3 カラムに置き換え，Modernist テーマと
キーボードショートカットを導入した (§5 / §6 / §10 参照)．
- 候補手ごとの PV (Rust 拡張) は採らず，候補手クリック → 分岐 → 解析で代替
- 分岐手順の KIF/KI2 エクスポートと悪手リトライ学習は将来拡張として据え置き

未決事項:

- グラフクリックによる局面ジャンプの実現方式 (gr.Plot のイベント対応次第)
- 傾斜配分など解析予算 UI の高度化 (index.md §9 と共通)
