# Screenshot Settle Time Design

## Problem

Gradio UIのスクリーンショット取得時，サーバーがロード中の状態（「Loading...」スピナー）がキャプチャされてしまう問題．

## Root Cause

`screenshot.py`の実装では，`.gradio-container`が表示された後に500msの固定待機を行っていたが，Gradioの初期化が完了するまでに500ms以上かかることがある．

## Solution

`--settle-time`オプションを追加し，動的コンテンツの安定待機時間をユーザーが設定可能にする．

## Changes

### 1. screenshot.py

- `DEFAULT_SETTLE_TIME = 3000`定数を追加
- `--settle-time`CLIオプションを追加（デフォルト: 3000ms）
- `page.wait_for_timeout(500)`を`page.wait_for_timeout(settle_time)`に変更

### 2. SKILL.md

- Command Optionsテーブルに`--settle-time`を追加
- Troubleshootingセクションに「Loading Screen Captured」の解決方法を追記

## Testing

手動テストで動作確認:
- デフォルト（3000ms）でUIが正しくキャプチャされることを確認
- 「Loading...」画面ではなく実際のUIコンテンツが表示される
