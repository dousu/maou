---
status: pending
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
Gradio コンポーネントを 3 カラムに組み替えた．
analyze-gui と違い gr.HTML 1 枚には**していない** — 検索・
ページネーション・インデックス構築など既存のイベント配線を温存する
ためレイアウトと CSS を変える方針とし，そのうえで **表示専用の
パネルだけ** gr.HTML に置き換えた (user から「gr.HTML でもっと綺麗に
できないか」との指摘を受けた 2 段目の作業)．

置き換えたのは Gradio 固有の枠・行番号・セル箱がデザインに寄せられない
4 つ:

| 旧 | 新 | 見た目 |
|---|---|---|
| 結果一覧 `gr.Dataframe` | `gr.HTML` | 2px 罫の見出し + 1px 罫の行，選択行に accent |
| レコード詳細 `gr.JSON` | `gr.HTML` | vz-kv のキー/値 2 段組 |
| データセット統計 `gr.JSON` | `gr.HTML` | vz-kv の 2 列グリッド |
| 局面統計 / 指し手一覧 (グラフ) | `gr.HTML` | 同上 |

HTML は `game_graph_shared.build_kv_html` /
`build_stats_grid_html` / `build_row_table_html` が組む (値は
`html.escape` 済み)．行クリックは `.select()` が使えないので，
ゲームグラフと同じ `server_functions` + `js_on_load` ブリッジを
新設した (`static/visualize_workbench.js`)．結果一覧の選択行は
サーバーが返す HTML に載るので，レコード送りでも強調が追随する．

- トップバー: `MAOU VISUALIZE` / データ型セグメンテッドコントロール /
  モードバッジ / ステータス (件数・パス) / 再構築・更新
- レコードモード: 左レール 340px (データソース・検索・結果一覧) |
  中央ステージ (盤面・レコード送り) | 右レール 420px
  (レコード詳細・分布・データセット統計)
- グラフモード: 左レール (パンくず・表示コントロール・凡例・
  エクスポート) | 中央ステージ (グラフ) | 右レール (選択局面・
  局面統計・指し手一覧・分岐分析)

`maou visualize --array-type game-graph` はスタンドアロンの
`game_graph_server.py` に分岐するため，そちらも同じワークベンチに
揃えた (揃えないと同じコマンドで 2 つの見た目が出る)．

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
- **ヒストグラムの配色も Modernist に寄せる**: 配色は app 層
  (`record_renderer.py`) にあり，データ型ごとに色を変えて意味を持たせて
  いる．シェルの restyle の範囲を超えるので手を付けていない．
- **analyze-gui と同じ gr.HTML 1 枚に寄せ切る**: 上表の 4 つを置き換えた
  時点で Gradio の chrome が残るのは入力系 (Dropdown / Slider / Number)
  だけで，そこは CSS で足りている．入力を HTML 化すると値の往復まで
  自前になり，得るものに対して壊す面が大きい．

## 付随して直したもの (drift ではない)

`gr.Dataframe` を全廃した副作用で **起動時に pandas を読むものが
無くなり**，plotly の配列判定 (`sys.modules` の pandas を import せず
参照する) がワーカースレッドの初回 import と競合して
`partially initialized module 'pandas' has no attribute 'Series'`
で描画が落ちた．`game_graph_shared._preimport_pandas()` で
メインスレッドから先読みし，`visualize` extra に `pandas` を明示した
(従来は gradio 経由の推移的依存だった)．
