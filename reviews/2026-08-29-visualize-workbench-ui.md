---
status: applied
applied_in: a24a0ef
date: 2026-08-29
target: [docs/commands/visualize.md]
risk: low
reversibility: easy
---

# `maou visualize` を 3 カラムのワークベンチに作り替えたことによる doc drift の訂正

## Trigger

user から Claude Design の handoff バンドル
`maou_visualize_workbench.html` が添付され，「maou visualize の UI を
添付ファイルのような形にアップデートしてください」との依頼．

実装は済んでいる (本提案と同じブランチ
`claude/maou-visualize-ui-update-87frn7`)．コードだけが先に変わって
`docs/commands/visualize.md` § UI機能 / § ゲームグラフの可視化 が
現行 UI と食い違うため，その訂正を提案する．

## 実装した UI (参考・確定済み)

analyze-gui (`analysis_workbench.css`) と同じ Modernist トークン
(Archivo / 赤単色 `#ec3013` / 角丸ゼロ / 2px 罫) を
`static/visualize_workbench.css` に `#viz-workbench` スコープで定義し，
3 カラムのワークベンチに組み替えた．実装は **analyze-gui と同じ gr.HTML 1 枚**に書き直した．
user から 2 度の指摘 (「gr.HTML でもっと綺麗にできないか」→「見た目の
品質が水準まで届いていない，gr.HTML 1 枚に書き直してイベント配線も
やり直してほしい」) を受けての判断で，Gradio コンポーネントを CSS で
寄せる方針は捨てている．

### 構成

| 層 | 追加/変更 | 役割 |
|---|---|---|
| interface | `visualize_workbench.py` (新規) | 状態 + 表示データ → ワークベンチ全体の HTML |
| infra | `static/visualize_workbench.css` (全面書き直し) | モックのトークンと寸法をそのまま持つ唯一の定義元 |
| infra | `static/visualize_workbench.js` (全面書き直し) | `data-action` の委譲・キー操作・再描画後の追従 |
| infra | `gradio_server.py` | gr.HTML 1 枚 + `gr.State` + 2 レーンのブリッジ |
| infra | `game_graph_server.py` | 同じワークベンチを描画 (スタンドアロン) |
| app | `record_renderer.py` | `Distribution` を追加 (Plotly Figure を廃止) |

イベント配線は analyze-gui と同じ方式に統一した．UI 操作は
`data-action` 文字列として JS から届き，`server_functions` +
`trigger("change")` のブリッジで `_on_action` に入り，状態を更新して
HTML を返す．レーンは `nav` / `load` の 2 本に分け，読み込みが行送りを
詰まらせないようにしてある．

### ヒストグラム

モックの SVG (幅 16px・間隔 20px・ベースライン y=100・20 本) を
数値で駆動する．現在レコードが属する階級だけ accent 色にし，残りは
neutral-500 に落とす — モックの配色そのままである．
Plotly はデータ型ごとに別の色 (青/緑/橙/紫) を使っていたので廃止し，
app 層は描画非依存の `Distribution` (数値列 + 軸ラベル) だけを返す．

### 削除したもの

gr.Dataframe / gr.JSON / gr.Plot / gr.Accordion / gr.Dropdown /
gr.Slider などのコンポーネントと，それらに紐づくタプル形状の
ハンドラ (`_search_and_cache` / `_navigate_next_record` /
`_check_indexing_status_with_transition` など計 22 メソッド)．
`gradio_server.py` は 4,254 行から 1,668 行になった．
Plotly 経路 (`generate_analytics`) と，その遅延 import 対策として
前段で入れた `pandas` 先読み・`visualize` extra への `pandas` 追加も
不要になったので戻した．

## Drift (訂正したい記述)

いずれも「現行コードから訂正後の本文が一意に決まる」種類のずれ．

1. § 検索結果タブ / § データ分析タブ — **タブは無くなった**．
   結果一覧は左レールの「結果」セクション，ヒストグラムは右レールの
   「分布」セクションになった．見出しを
   「§ 結果一覧」「§ 分布・統計」に改め，「タブ」の語を落とす．
2. § Data Source Management の手順 3 「Array Type: データ型を選択」 —
   データ型の切替は**トップバーのセグメンテッドコントロール**に移った．
   手順から外し，トップバーの説明として書き直す．
3. 同 手順 4 「Load Data Source」 — ボタン名は「読み込み」．
4. § グラフビュー(左パネル) / § 詳細パネル(右パネル) —
   グラフは**中央**，詳細は**右レール**，コントロール (表示深さ・
   最小確率・更新・ルートに戻る/設定・パンくず・エクスポート) は
   **左レール**．見出しを
   「§ グラフビュー(中央)」「§ 詳細パネル(右レール)」
   「§ コントロール(左レール)」に改める．
5. § 凡例 (新規) — 左レールに凡例を常設した．色 (青>55% / グレー /
   赤<45%)・選択リング・深さ打ち切りの破線リング・ノード径と線幅の
   意味は `static/game_graph_canvas.js` と一致させてある．
   ドキュメント側は既に色の説明を持っているので，
   「凡例は左レールに常設」の 1 行を足すだけでよい．

## 変えない箇所

- CLI オプションは一切変えていないので § CLI オプションはそのまま．
- SFEN 検索の array_type ごとの実装差，定跡テーブル，Stage1 表示仕様は
  UI 変更の影響を受けない．
- § 既知の問題 のモードバッジの件も未修整のまま (今回の対象外)．

## 判断が要る点 (承認時に決めてほしい)

なし．1-5 はいずれもコードから一意に決まる訂正で，新しい方針や
節の再構成を伴わない．

## Rejected alternatives

- **analyze-gui と同じ gr.HTML 1 枚に作り替える**: 検索・ページ送り・
  バックグラウンドインデックス・ブリッジなど 2400 行のイベント配線を
  全部書き直すことになり，UI の見た目を揃える目的に対して割に合わない．
- **Gradio コンポーネントを CSS で寄せる**: 2 度試したが，
  gr.JSON の行番号・gr.Dataframe のセル箱・各所の角丸と枠は
  スコープ付き CSS で潰しても水準に届かなかった (user 指摘)．
  1 枚 HTML なら見た目の定義元が 1 つになり，モックと 1 対 1 で
  対応が取れる．
