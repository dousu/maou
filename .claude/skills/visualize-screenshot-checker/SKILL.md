---
name: visualize-screenshot-checker
description: maou visualize の可視化機能をスクリーンショットで網羅的にチェックする．Gradio サーバーを起動し，各画面状態のスクリーンショットを撮影・目視確認する．
user-invocable: true
disable-model-invocation: true
argument-hint: [--array-type <hcpe|stage1|stage2|preprocessing>]
---

# Visualize Screenshot Checker

`maou visualize` の可視化機能をスクリーンショットベースで網羅的にチェックするスキル．
Playwright で各画面状態を撮影し，Claude の画像認識でビジュアル検証を行う．

## 前提条件

```bash
uv sync --extra cpu --extra visualize
uv run playwright install --with-deps chromium
uv run maturin develop
```

## チェック手順

以下のフェーズを順番に実行する．各フェーズでサーバー起動→撮影→停止のサイクルを繰り返す．
`$ARGUMENTS` に `--array-type` の指定がある場合，該当する array-type のみチェックする．
指定がなければ全 array-type をチェックする．

### Phase 0: サーバー起動・撮影・停止のヘルパー

各フェーズで以下のパターンを繰り返す:

```bash
# 起動（array-type は各フェーズで指定）
uv run maou visualize --use-mock-data --array-type $ARRAY_TYPE --port 7860 &
SERVER_PID=$!
sleep 8
```

```bash
# 撮影（各チェック項目ごとにオプションを変える）
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-XXXX.png \
  --settle-time 3000
```

```bash
# 停止
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
sleep 2
```

**重要: 実行フローと報告の原則**

1. 全フェーズの撮影を先に一括実行する（各撮影でエラーが発生した場合のみ即時対処）
2. 全撮影完了後に `Read` ツールで各画像をまとめて読み取り，チェック項目に沿って確認する
3. 最後に「結果報告のフォーマット」に従い，スクリーンショットパス一覧表と結果サマリーをユーザーに報告する

途中経過の報告は行わない．ユーザーへの報告は最終レポート1回のみとする．

---

### Phase 1: 初期状態チェック（全 array-type 共通）

**対象**: `--array-type hcpe`（代表として）

サーバー起動後，操作なしの初期画面を撮影する．

#### Check 1.1: フルページ初期状態

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0101-initial-full.png \
  --settle-time 5000
```

**確認項目**:
- [ ] Gradio UI が正常にレンダリングされている（Loading 画面でない）
- [ ] 左サイドバー（検索・ナビゲーション）と右メインパネル（盤面・詳細）の2カラムレイアウト
- [ ] モードバッジに「MOCK」と表示されている
- [ ] タブが3つ表示されている（概要，検索結果，データ分析）
- [ ] レコードナビゲーションボタン（前/次）が存在する
- [ ] ページナビゲーションボタン（前/次）が存在する

#### Check 1.2: モードバッジ

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --selector "#mode-badge" \
  --output /tmp/check-0102-mode-badge.png
```

**確認項目**:
- [ ] 「MOCK」テキストが明瞭に表示されている
- [ ] バッジの背景色・テキスト色が視認可能

---

### Phase 2: 盤面レンダリングチェック（array-type 別）

各 array-type でサーバーを起動し，盤面の描画を確認する．

#### Check 2.1: HCPE 盤面表示

`--array-type hcpe` で起動．ID 検索で任意のレコードを表示する．

```bash
# ID 検索でレコードを表示してから盤面を撮影
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "fill:#id-search-input input:mock_id_0" \
  --action "click:#id-search-btn" \
  --action "wait:#board-display svg" \
  --selector "#board-display" \
  --output /tmp/check-0201-hcpe-board.png
```

**確認項目**:
- [ ] 9x9 の将棋盤が表示されている
- [ ] 格子線が均等に描画されている
- [ ] 座標ラベルが表示されている（上部: 9 8 7 6 5 4 3 2 1，右側: 一〜九 or 1〜9）
- [ ] 盤面の背景色が温かみのあるニュートラル系（`#f9f6f0` 付近）
- [ ] 駒が漢字で表示されている（「?」マークがない）
- [ ] 先手駒（黒）と後手駒（赤系）の色分けが明確
- [ ] 後手駒が180度回転している（上下逆さ）
- [ ] 持ち駒エリアが盤面の左右に表示されている
- [ ] 着手の矢印が表示されている（bestMove16 が存在する場合）

#### Check 2.2: Stage1 盤面表示

`--array-type stage1` で起動．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "wait:#board-display svg" \
  --selector "#board-display" \
  --output /tmp/check-0202-stage1-board.png
```

**確認項目**:
- [ ] 9x9 の将棋盤が表示されている
- [ ] reachable squares がハイライト（青系の半透明）で表示されている
- [ ] ハイライトされたマスの位置が盤面上で妥当（ランダムに散らばっている）
- [ ] 駒が正しく表示されている

#### Check 2.3: Stage2 盤面表示

`--array-type stage2` で起動．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "wait:#board-display svg" \
  --selector "#board-display" \
  --output /tmp/check-0203-stage2-board.png
```

**確認項目**:
- [ ] 9x9 の将棋盤が表示されている
- [ ] 合法手の矢印が表示されている（存在する場合）
- [ ] 駒が正しく表示されている

#### Check 2.4: Preprocessing 盤面表示

`--array-type preprocessing` で起動．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "wait:#board-display svg" \
  --selector "#board-display" \
  --output /tmp/check-0204-preprocessing-board.png
```

**確認項目**:
- [ ] 9x9 の将棋盤が表示されている
- [ ] 着手の矢印が表示されている（moveLabel が存在する場合）
- [ ] 駒が正しく表示されている

---

### Phase 3: 将棋規則の正確性チェック

HCPE の盤面で将棋のルール・表示規則を重点的に確認する．
`--array-type hcpe` で起動．

#### Check 3.1: 座標系の正確性

フルページスクリーンショットで座標を確認．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "wait:#board-display svg" \
  --selector "#board-display" \
  --output /tmp/check-0301-coordinates.png \
  --width 1920 --height 1080
```

**確認項目**:
- [ ] 列番号（筋）が右から左へ 1→9 の順で表示されている（将棋の慣例）
- [ ] 行番号（段）が上から下へ 一→九（または 1→9）の順で表示されている
- [ ] 盤面の向き: 先手（下手）が画面下側，後手（上手）が画面上側

#### Check 3.2: 駒の表示

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0302-pieces.png
```

**確認項目**:
- [ ] 先手の駒（黒色 `#2c2c2c`）が正立で表示されている
- [ ] 後手の駒（赤系 `#c41e3a`）が倒立（180度回転）で表示されている
- [ ] 全ての駒が漢字表記されている（「?」や空白でない）
- [ ] 成駒がある場合，正しい漢字が使われている（例: と，成香，成桂，成銀，馬，龍）

#### Check 3.3: 持ち駒の表示

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0303-pieces-in-hand.png
```

**確認項目**:
- [ ] 先手の持ち駒が盤面の右側（または下側）に表示されている
- [ ] 後手の持ち駒が盤面の左側（または上側）に表示されている
- [ ] 持ち駒の枚数が数字で表示されている（2枚以上の場合）
- [ ] 持ち駒がない場合，「なし」等の表示または空欄

---

### Phase 4: 概要タブチェック

`--array-type hcpe` で起動．レコード表示時の概要タブ内容を確認する．

#### Check 4.1: 概要タブの詳細表示

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0401-overview-tab.png
```

**確認項目**:
- [ ] 概要（📋 概要）タブが選択状態で表示されている
- [ ] レコードの詳細情報が JSON 形式で表示されている
- [ ] ID フィールドが表示されている
- [ ] array-type 固有のフィールドが表示されている（HCPE: eval, bestMove16 等）

---

### Phase 5: 検索結果タブチェック

`--array-type hcpe` で起動．

#### Check 5.1: 検索結果テーブル

検索結果タブをクリックしてテーブルを表示し，撮影する．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:button[role='tab']:nth-of-type(2)" \
  --output /tmp/check-0501-search-results.png
```

**確認項目**:
- [ ] 検索結果タブ（📊 検索結果）の内容が表示されている
- [ ] テーブルにカラムヘッダーが表示されている
- [ ] テーブルにデータ行が表示されている（mock データ）
- [ ] ページ情報（Page X / Y）が表示されている

#### Check 5.2: ページナビゲーション

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --selector "#page-info" \
  --output /tmp/check-0502-page-info.png
```

**確認項目**:
- [ ] ページ番号が表示されている
- [ ] 前ページ・次ページボタンが表示されている

---

### Phase 6: データ分析タブチェック

各 array-type で分析チャートを確認する．

#### Check 6.1: HCPE データ分析

`--array-type hcpe` で起動．データ分析タブに切り替えて確認．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:button[role='tab']:nth-of-type(3)" \
  --output /tmp/check-0601-analytics-hcpe.png
```

**確認項目**:
- [ ] データ分析タブ（📈 データ分析）が表示可能
- [ ] 評価値分布のヒストグラムが表示されている
- [ ] チャートの軸ラベルが日本語で表示されている
- [ ] チャートのデータが妥当（mock データの範囲内）

#### Check 6.2: Stage1 データ分析

`--array-type stage1` で起動．データ分析タブに切り替えて確認．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:button[role='tab']:nth-of-type(3)" \
  --output /tmp/check-0602-analytics-stage1.png
```

**確認項目**:
- [ ] 到達可能マス数の分布チャートが表示されている

#### Check 6.3: Stage2 データ分析

`--array-type stage2` で起動．データ分析タブに切り替えて確認．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:button[role='tab']:nth-of-type(3)" \
  --output /tmp/check-0603-analytics-stage2.png
```

**確認項目**:
- [ ] 合法手数の分布チャートが表示されている

---

### Phase 7: レコードナビゲーションチェック

`--array-type hcpe` で起動．レコード間の移動を確認する．

#### Check 7.1: レコードインジケータ

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --selector "#record-indicator" \
  --output /tmp/check-0701-record-indicator.png
```

**確認項目**:
- [ ] 「Record X / Y」形式の表示がある
- [ ] 数値が妥当（0 でない，総数と一致）

---

### Phase 8: エラー状態チェック

#### Check 8.1: ID 検索の空状態

サーバー起動直後，検索未実行の状態を確認．

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0801-no-search.png
```

**確認項目**:
- [ ] エラーメッセージが表示されていない
- [ ] 盤面エリアに初期状態（空盤面またはプレースホルダ）が表示されている
- [ ] UI がクラッシュしていない

---

### Phase 9: レスポンシブ・レイアウトチェック

#### Check 9.1: ワイド画面（1920x1080）

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0901-wide.png \
  --width 1920 --height 1080 \
  --no-full-page
```

**確認項目**:
- [ ] 2カラムレイアウトが維持されている
- [ ] 盤面が適切なサイズで表示されている
- [ ] 余白が極端に大きくない

#### Check 9.2: 標準画面（1280x720）

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0902-standard.png \
  --width 1280 --height 720 \
  --no-full-page
```

**確認項目**:
- [ ] レイアウトが崩れていない
- [ ] 全ての主要要素が表示されている

#### Check 9.3: ナロー画面（768x1024）

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --output /tmp/check-0903-narrow.png \
  --width 768 --height 1024 \
  --no-full-page
```

**確認項目**:
- [ ] レイアウトが適応している（1カラムまたは縮小表示）
- [ ] 主要要素が切れていない
- [ ] スクロールで全ての要素にアクセスできる

---

### Phase 10: テーブル行クリックによる盤面更新チェック

`--array-type hcpe` で起動．検索結果テーブルの行をクリックして盤面が変わることを確認する．

#### Check 10.1: テーブル行クリックで盤面が更新される

```bash
# 検索結果タブに切り替えてテーブル行をクリック
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:button[role='tab']:nth-of-type(2)" \
  --action "click:#search-results-table [data-testid='cell-1-0']" \
  --action "wait:#board-display svg" \
  --output /tmp/check-1001-table-row-click.png
```

**確認項目**:
- [ ] 盤面が表示されている（テーブル行クリック後に更新された）
- [ ] 駒が正しく表示されている
- [ ] エラーメッセージが表示されていない

---

### Phase 11: データソース変更の状態遷移チェック

`--array-type hcpe` で起動．データソースを変更した際のインデクシング状態遷移を確認する．

#### Check 11.1: データソース変更後のインデクシング表示

```bash
# Array Type ドロップダウンをクリックしてデータソース変更UIを撮影
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:#array-type-dropdown" \
  --settle-time 1000 \
  --output /tmp/check-1101-indexing-start.png
```

**確認項目**:
- [ ] インデクシング中の表示が確認できる（またはデータソース切り替えUIが表示されている）
- [ ] UI がクラッシュしていない

#### Check 11.2: インデクシング完了後の状態

```bash
uv run maou utility screenshot \
  --url http://localhost:7860 \
  --action "click:#array-type-dropdown" \
  --action "wait-hidden:.loading-spinner" \
  --settle-time 5000 \
  --output /tmp/check-1102-indexing-complete.png
```

**確認項目**:
- [ ] インデクシングが完了し，通常状態に戻っている
- [ ] 盤面やテーブルが正常に表示されている

---

## 結果報告のフォーマット

全フェーズの撮影・確認を完了した後，以下の形式で **1回だけ** ユーザーに報告する．
途中経過の報告は行わない．

```
## Visualize Screenshot Check Report

### 実行環境
- array-type: (チェックした array-type)
- date: (実行日)

### スクリーンショット一覧

撮影した全スクリーンショットのパスを以下に示す．
`Read` ツールで各ファイルを直接確認できる．

| Check | 内容 | ファイルパス | 撮影結果 |
|-------|------|------------|---------|
| 1.1 | フルページ初期状態 | `/tmp/check-0101-initial-full.png` | OK / ERROR |
| 1.2 | モードバッジ | `/tmp/check-0102-mode-badge.png` | OK / ERROR |
| 2.1 | HCPE 盤面 | `/tmp/check-0201-hcpe-board.png` | OK / ERROR / SKIP |
| 2.2 | Stage1 盤面 | `/tmp/check-0202-stage1-board.png` | OK / ERROR / SKIP |
| 2.3 | Stage2 盤面 | `/tmp/check-0203-stage2-board.png` | OK / ERROR / SKIP |
| 2.4 | Preprocessing 盤面 | `/tmp/check-0204-preprocessing-board.png` | OK / ERROR / SKIP |
| 3.1 | 座標系の正確性 | `/tmp/check-0301-coordinates.png` | OK / ERROR |
| 3.2 | 駒の表示 | `/tmp/check-0302-pieces.png` | OK / ERROR |
| 3.3 | 持ち駒の表示 | `/tmp/check-0303-pieces-in-hand.png` | OK / ERROR |
| 4.1 | 概要タブ | `/tmp/check-0401-overview-tab.png` | OK / ERROR |
| 5.1 | 検索結果テーブル | `/tmp/check-0501-search-results.png` | OK / ERROR |
| 5.2 | ページナビゲーション | `/tmp/check-0502-page-info.png` | OK / ERROR |
| 6.1 | HCPE データ分析 | `/tmp/check-0601-analytics-hcpe.png` | OK / ERROR / SKIP |
| 6.2 | Stage1 データ分析 | `/tmp/check-0602-analytics-stage1.png` | OK / ERROR / SKIP |
| 6.3 | Stage2 データ分析 | `/tmp/check-0603-analytics-stage2.png` | OK / ERROR / SKIP |
| 7.1 | レコードインジケータ | `/tmp/check-0701-record-indicator.png` | OK / ERROR |
| 8.1 | ID 検索の空状態 | `/tmp/check-0801-no-search.png` | OK / ERROR |
| 9.1 | ワイド画面 | `/tmp/check-0901-wide.png` | OK / ERROR |
| 9.2 | 標準画面 | `/tmp/check-0902-standard.png` | OK / ERROR |
| 9.3 | ナロー画面 | `/tmp/check-0903-narrow.png` | OK / ERROR |
| 10.1 | テーブル行クリック | `/tmp/check-1001-table-row-click.png` | OK / ERROR |
| 11.1 | インデクシング開始 | `/tmp/check-1101-indexing-start.png` | OK / ERROR |
| 11.2 | インデクシング完了 | `/tmp/check-1102-indexing-complete.png` | OK / ERROR |

- **OK**: スクリーンショット撮影成功
- **ERROR**: 撮影時にエラー発生（セレクタ未検出，タイムアウト等）
- **SKIP**: `--array-type` 指定により対象外

### 結果サマリー

| Phase | チェック項目数 | Pass | Fail | 備考 |
|-------|------------|------|------|------|
| 1. 初期状態 | X | X | X | |
| 2. 盤面レンダリング | X | X | X | |
| 3. 将棋規則 | X | X | X | |
| 4. 概要タブ | X | X | X | |
| 5. 検索結果タブ | X | X | X | |
| 6. データ分析タブ | X | X | X | |
| 7. ナビゲーション | X | X | X | |
| 8. エラー状態 | X | X | X | |
| 9. レスポンシブ | X | X | X | |
| 10. テーブル行クリック | X | X | X | |
| 11. データソース変更 | X | X | X | |

### Fail 項目の詳細
（Fail があった場合，スクリーンショットのパスと問題点を記述）

| Check | ファイルパス | 問題点 |
|-------|------------|--------|
| X.X | `/tmp/check-XXXX.png` | 問題の説明 |
```

## 注意事項

- サーバーの起動には 8-10 秒程度かかる．`sleep 8` を確保すること
- Gradio は SPA のため `--settle-time 3000` 以上を推奨
- **`--selector "#board-display"` を使用する場合，必ず `--action "wait:#board-display svg"` を併用すること**．盤面 SVG は非同期レンダリングされるため，`wait` なしではデータ読み込み前の空盤面がキャプチャされる
- mock データはランダム生成のため，盤面の駒配置は毎回異なる
- `--action` オプションでUI操作（ID検索，タブ切り替え，行クリック等）を撮影前に実行可能
- アクションフォーマット: `TYPE:SELECTOR[:VALUE]`（click, fill, wait, wait-text, wait-hidden）
- 複数アクションは `--action` を繰り返し指定し，指定順に実行される
- パッケージ管理には `uv` を使用すること（`poetry` は使用しない）

## 関連ドキュメント

- [Shogi Conventions](../../../docs/visualization/shogi-conventions.md) - 座標系・駒配置の規約
- [Browser Automation Guide](../../../docs/browser-automation.md) - Playwright の設定・トラブルシューティング
- [UI/UX Design](../../../docs/visualization/UI_UX_REDESIGN.md) - デザインシステム
