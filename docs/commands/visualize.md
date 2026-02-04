# `maou visualize`

## Overview

Gradioベースの将棋データ可視化ツール．HCPE，preprocessing，stage1，stage2形式のデータファイルを読み込み，盤面表示，レコード検索，データ分析機能を提供する．

### サポートするデータ型

| データ型 | 説明 | 用途 |
|----------|------|------|
| `hcpe` | 棋譜データ（評価値付き） | 従来の学習データ確認 |
| `preprocessing` | 前処理済みデータ | 学習前のデータ検証 |
| `stage1` | 到達可能マス学習用 | 基礎学習データ確認 |
| `stage2` | 合法手学習用 | 中間学習データ確認 |

## CLI オプション

### データソース

| フラグ | 必須 | 説明 |
|--------|------|------|
| `--input-dir PATH` | いずれか1つ | データファイルを含むディレクトリパス．再帰的に`.feather`ファイルを検索する． |
| `--input-files TEXT` | いずれか1つ | カンマ区切りのデータファイルパスリスト．例：`file1.feather,file2.feather` |
| `--use-mock-data` | いずれか1つ | モックデータを使用（UIテスト用）．実際のファイルは読み込まない． |

**注意**: データソースを指定せずに起動した場合，UIからデータを読み込むことができる．

### データ型

| フラグ | 必須 | 説明 |
|--------|------|------|
| `--array-type {hcpe,preprocessing,stage1,stage2}` | Yes | 読み込むデータの形式を指定する． |

### サーバー設定

| フラグ | デフォルト | 説明 |
|--------|------------|------|
| `--port INT` | 7860 | Gradioサーバーのポート番号． |
| `--server-name TEXT` | 127.0.0.1 | サーバーバインドアドレス．外部アクセスを許可する場合は`0.0.0.0`を指定． |
| `--share` | false | Gradio公開リンクを作成する．Google Colab環境では自動的に有効化される． |
| `--debug-mode` | false | 詳細ログを有効化する． |

### モデル評価（オプション）

| フラグ | 必須 | 説明 |
|--------|------|------|
| `--model-path PATH` | No | ONNXモデルパス．指定すると各局面の評価値を表示できる． |

## 使用例

### Stage1データの可視化

```bash
# ファイル指定
maou visualize --input-files ./data/stage1/stage1_data.feather --array-type stage1

# ディレクトリ指定
maou visualize --input-dir ./data/stage1/ --array-type stage1
```

### HCPEデータの可視化

```bash
maou visualize --input-dir ./data/hcpe/ --array-type hcpe
```

### データソースなしで起動（UI操作）

```bash
maou visualize --array-type stage1
# ブラウザでUIからデータを読み込む
```

### モックデータでUIテスト

```bash
maou visualize --use-mock-data --array-type hcpe
```

### 公開リンク作成（Google Colab等）

```bash
maou visualize --input-files data.feather --array-type stage1 --share
```

## UI機能

### Data Source Management

データソースの動的な変更が可能：

1. **Source Type**: DirectoryまたはFile Listを選択
2. **Directory Path / File Paths**: パスを入力（2文字以上で候補表示）
3. **Array Type**: データ型を選択
4. **Load Data Source**: データを読み込み

### 盤面表示

- 駒の配置を将棋盤として表示
- Stage1/Stage2では到達可能マスをハイライト表示
- 持ち駒を左右に表示

### レコードナビゲーション

- **前のレコード / 次のレコード**: 1件ずつ移動
- **レコードインジケーター**: 現在位置を表示（例：Record 5 / 20）

### ページネーション

- **ページサイズ**: 10〜100件で調整可能
- **ページ移動**: 前へ / 次へ ボタン

### 検索機能

- **ID検索**: レコードIDで直接検索
- **評価値範囲検索**: HCPEデータのみ対応

### 検索結果タブ

テーブル形式でレコード一覧を表示．行クリックで該当レコードの盤面を表示．

| カラム | 説明 |
|--------|------|
| Index | 通し番号 |
| ID | レコードID |
| Reachable Squares（Stage1） | 到達可能マス数 |
| Eval（HCPE） | 評価値 |

### データ分析タブ

- データ分布のヒストグラム表示
- 統計情報の表示

## Stage1データの表示仕様

Stage1データ（`--array-type stage1`）では：

1. **盤面**: 単一の駒のみが配置された局面
2. **ハイライト**: 到達可能マスが水色でハイライト表示
3. **レコード詳細**:
   - `id`: レコードID
   - `reachable_count`: 到達可能マス数
   - `array_type`: "stage1"

## 既知の問題

- UIからデータを読み込んだ場合，モードバッジ（「REAL MODE」/「NO DATA」）が正しく更新されないことがある（機能には影響なし）

## 関連コマンド

- [`maou utility generate-stage1-data`](./generate-stage1-data.md) - Stage1データ生成
- [`maou learn-model`](./learn_model.md) - モデル学習
- [`maou pre-process`](./pre_process.md) - データ前処理
