# `maou screenshot`

## Overview

Playwrightを使用してGradio UIのスクリーンショットを取得するコマンド．ビジュアルリグレッションテストやGradioベースの可視化インターフェースのドキュメント作成に使用する．

### 前提条件

`playwright` extraが必要：

```bash
uv sync --extra visualize && uv run playwright install chromium
```

## CLI オプション

### 接続設定

| フラグ | デフォルト | 説明 |
|--------|------------|------|
| `--url TEXT` | `http://localhost:7860` | 対象URL．【F:src/maou/infra/console/screenshot.py†L23】 |
| `--wait-for TEXT` | `.gradio-container` | キャプチャ前に待機するCSSセレクタ．【F:src/maou/infra/console/screenshot.py†L25】 |
| `--timeout INT` | `30000` | ナビゲーションタイムアウト（ミリ秒）．【F:src/maou/infra/console/screenshot.py†L26】 |

### 出力設定

| フラグ | デフォルト | 説明 |
|--------|------------|------|
| `--output, -o PATH` | `/tmp/gradio-screenshot.png` | 出力ファイルパス．【F:src/maou/infra/console/screenshot.py†L24】 |
| `--base64` | `false` | ファイル出力の代わりにbase64をstdoutに出力する．Claude Vision API等での利用を想定．【F:src/maou/infra/console/screenshot.py†L304-309】 |

### キャプチャ設定

| フラグ | デフォルト | 説明 |
|--------|------------|------|
| `--selector, -s TEXT` | （なし） | 要素キャプチャ用のCSSセレクタ．指定した場合，その要素のみをキャプチャする．【F:src/maou/infra/console/screenshot.py†L311-316】 |
| `--full-page/--no-full-page` | `true` | スクロール可能なフルページをキャプチャするか．【F:src/maou/infra/console/screenshot.py†L318-321】 |
| `--width INT` | `1280` | ビューポート幅．【F:src/maou/infra/console/screenshot.py†L27】 |
| `--height INT` | `720` | ビューポート高さ．【F:src/maou/infra/console/screenshot.py†L28】 |
| `--settle-time INT` | `3000` | 動的コンテンツが安定するまでの待機時間（ミリ秒）．【F:src/maou/infra/console/screenshot.py†L29】 |

### アクション設定

| フラグ | デフォルト | 説明 |
|--------|------------|------|
| `--action TEXT` | （なし） | キャプチャ前に実行するUIアクション．フォーマット: `TYPE:SELECTOR[:VALUE]`．複数回指定可能．【F:src/maou/infra/console/screenshot.py†L353-364】 |
| `--action-settle-time INT` | `500` | 各アクション後の待機時間（ミリ秒）．【F:src/maou/infra/console/screenshot.py†L30】 |

## Action types

スクリーンショット撮影前にUIを操作するための5種類のアクションタイプ．【F:src/maou/infra/console/screenshot.py†L32-39】

| アクションタイプ | フォーマット | 説明 |
|------------------|-------------|------|
| `click` | `click:SELECTOR` | CSSセレクタで指定した要素をクリックする．|
| `fill` | `fill:SELECTOR:VALUE` | 入力要素にテキストを入力する．SELECTORとVALUEの区切りは右端の`:`で判定されるため，CSSセレクタ内に`:`を含めることができる．|
| `wait` | `wait:SELECTOR` | 要素が表示状態（visible）になるまで待機する．|
| `wait-text` | `wait-text:SELECTOR:VALUE` | 指定テキストを含む要素が表示されるまで待機する．|
| `wait-hidden` | `wait-hidden:SELECTOR` | 要素が非表示状態（hidden）になるまで待機する．|

**パースルール**: 最初の`:`でアクション種別を分離し，`fill`/`wait-text`は右端の`:`でSELECTORとVALUEを分離する．これにより，CSS疑似セレクタ（例: `button:nth-of-type(3)`）をSELECTOR内に含めることができる．【F:src/maou/infra/console/screenshot.py†L57-123】

## 使用例

### 基本的なスクリーンショット

```bash
maou screenshot --url http://localhost:7860 --output /tmp/test.png
```

### Base64出力（Claude Vision API向け）

```bash
maou screenshot --url http://localhost:7860 --base64
```

### 特定要素のキャプチャ

```bash
maou screenshot --url http://localhost:7860 --selector "#mode-badge"
```

### フルページスクリーンショット

```bash
maou screenshot --url http://localhost:7860 --full-page
```

### ID検索後に盤面をキャプチャ

```bash
maou screenshot \
  --action "fill:#id-search-input input:mock_id_0" \
  --action "click:#id-search-btn" \
  --action "wait:#board-display svg" \
  --output /tmp/id-search.png
```

### データ分析タブに切り替えてキャプチャ

```bash
maou screenshot \
  --action "click:button[role='tab']:nth-of-type(3)" \
  --output /tmp/analytics.png
```

### Gradio UIセレクタ参考

| セレクタ | 説明 |
|----------|------|
| `.gradio-container` | メインコンテナ（デフォルトの待機対象） |
| `#mode-badge` | データモード表示（MOCK/REAL） |
| `#id-search-input` | レコードID検索入力欄 |
| `#prev-page` | 前ページボタン |
| `#next-page` | 次ページボタン |

## 実装リファレンス

- ソースファイル: [`src/maou/infra/console/screenshot.py`](../../src/maou/infra/console/screenshot.py)

## 関連コマンド

- [`maou visualize`](./visualize.md) - Gradioベースの将棋データ可視化ツール
