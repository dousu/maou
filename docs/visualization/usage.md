# Maou将棋データ可視化ツール - 使用例

このドキュメントでは，可視化ツールの具体的な使用方法とワークフローを説明します．

## 目次

1. [インストール](#インストール)
2. [基本的な使い方](#基本的な使い方)
3. [VSCodeでの使用](#vscodeでの使用)
4. [Google Colabでの使用](#google-colabでの使用)
5. [検索例](#検索例)
6. [トラブルシューティング](#トラブルシューティング)

---

## インストール

### 依存関係のインストール

```bash
# 可視化ツール込みでインストール
poetry install -E visualize

# 他のエクストラと組み合わせ
poetry install -E cpu -E visualize  # CPU学習 + 可視化
poetry install -E cuda -E visualize  # CUDA学習 + 可視化
```

### インストール確認

```bash
# ヘルプ表示
poetry run maou visualize --help

# 期待される出力:
# Usage: maou visualize [OPTIONS]
# ...
```

---

## 基本的な使い方

### ディレクトリからデータを可視化

```bash
# HCPEデータを可視化
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe

# Preprocessingデータを可視化
poetry run maou visualize \
  --input-path ./data/preprocessing \
  --array-type preprocessing
```

### 特定ファイルを指定

```bash
# 複数ファイルを指定
poetry run maou visualize \
  --input-path data1.feather --input-path data2.feather --input-path data3.feather \
  --array-type hcpe
```

### カスタムポート

```bash
# ポート8080で起動
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --port 8080
```

---

## VSCodeでの使用

VSCodeのポートフォワーディング機能を使用して，ブラウザなしで可視化ツールを使用できます．

### ステップ1: サーバー起動

VSCodeターミナルで以下を実行：

```bash
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe
```

### ステップ2: ポートフォワーディング

1. VSCodeが自動的にポート7860を検出
2. 画面右下に通知が表示される: 「Port 7860 is being used by a process」
3. 「Open in Browser」をクリック

### ステップ3: ブラウザでアクセス

転送されたURL（例: `http://localhost:7860`）が自動的に開きます．

### デバッグモード

詳細ログを確認する場合：

```bash
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --debug-mode
```

ログ出力例：

```
INFO:maou.infra.visualization.gradio_server:Initializing visualization server: 10 files, type=hcpe, 1000 records indexed
INFO:maou.interface.visualization:VisualizationInterface initialized
Running on local URL:  http://127.0.0.1:7860
```

---

## Google Colabでの使用

Google Colabでは，`--share`フラグを使用して公開リンクを作成します．

### ステップ1: Colabノートブック準備

```python
# セル1: インストール
!pip install maou[visualize]
```

### ステップ2: データアップロード

```python
# セル2: Google Driveマウント（推奨）
from google.colab import drive
drive.mount('/content/drive')

# または，ファイル直接アップロード
from google.colab import files
uploaded = files.upload()
```

### ステップ3: サーバー起動

```python
# セル3: 可視化サーバー起動
!maou visualize \
  --input-path /content/drive/MyDrive/maou_data/hcpe \
  --array-type hcpe \
  --share
```

### 出力例

```
INFO:maou.infra.visualization.gradio_server:Initializing visualization server...
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Running on public URL: https://1234abcd5678efgh.gradio.live

This share link expires in 72 hours.
```

**公開URL**（例: `https://1234abcd.gradio.live`）をブラウザで開くと，Gradio UIにアクセスできます．

---

## 検索例

### 例1: 特定レコードの確認

**目的**: IDが`mock_id_42`のレコードを表示

**手順**:
1. サーバー起動後，Gradio UIを開く
2. 左パネルの「ID検索」セクションに移動
3. 「レコードID」フィールドに `mock_id_42` を入力
4. 「ID検索」ボタンをクリック

**結果**:
- 右パネルに将棋盤が表示される
- 「レコード詳細」に評価値，手数などが表示される

### 例2: 高評価局面の探索

**目的**: 評価値が500〜1000の局面を探索

**手順**:
1. 左パネルの「評価値範囲検索」セクションに移動
2. 「最小評価値」に `500` を入力
3. 「最大評価値」に `1000` を入力
4. 「範囲検索」ボタンをクリック

**結果**:
- 「検索結果」テーブルに該当レコード一覧が表示
- 最初のレコードの盤面が右パネルに表示
- ページネーション機能で次のページに移動可能

### 例3: 均衡局面のブラウジング

**目的**: 評価値が-50〜50の均衡局面を確認

**手順**:
1. 「最小評価値」に `-50`
2. 「最大評価値」に `50`
3. 「1ページあたりの件数」を `50` に設定
4. 「範囲検索」をクリック
5. 「次へ →」ボタンで次ページを表示

**ページ情報例**:
```
ページ 1 / 10 （500 件中 50 件表示）
```

---

## ワークフロー例

### ワークフロー1: 学習データ品質確認

**目的**: 学習データに異常な局面がないか確認

```bash
# 1. Preprocessingデータを可視化
poetry run maou visualize \
  --input-path ./data/preprocessing \
  --array-type preprocessing
```

**確認手順**:
1. 評価値範囲検索で極端な値（-1000〜-800）を検索
2. ボード表示で局面が正常か目視確認
3. 異常があればデータ生成プロセスを見直し

### ワークフロー2: 学習進捗モニタリング

**目的**: エポックごとの学習データの偏りを確認

```bash
# epoch10のデータを可視化
poetry run maou visualize \
  --input-path ./output/preprocessing/epoch10 \
  --array-type preprocessing
```

**確認内容**:
- データセット情報の「総レコード数」
- 評価値分布（複数範囲で検索）
- 特定の戦型の局面数

### ワークフロー3: デバッグ

**目的**: 特定のエラーが発生した局面を確認

```bash
# デバッグモードで起動
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --debug-mode
```

**手順**:
1. ログからエラーが発生したレコードIDを特定
2. ID検索でそのレコードを表示
3. ボード状態を確認し，エラー原因を特定

---

## 高度な使用例

### カスタムサーバーバインド

外部からアクセスする場合：

```bash
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --server-name 0.0.0.0 \
  --port 7860
```

**注意**: セキュリティリスクがあるため，信頼できるネットワークのみで使用してください．

### モデル評価値オーバーレイ（将来機能）

```bash
# ONNXモデルを指定（現在は未実装）
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --model-path ./models/model_epoch10.onnx
```

**期待される機能**:
- レコードの評価値とモデルの予測値を比較
- 差分が大きい局面をハイライト

---

## トラブルシューティング

### 問題1: 「Command 'visualize' requires visualization dependencies」

**原因**: Gradioがインストールされていない

**解決策**:

```bash
poetry install -E visualize
```

### 問題2: ポートがすでに使用されている

**エラー**:
```
OSError: [Errno 48] Address already in use
```

**解決策1**: 別のポートを指定

```bash
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --port 8080
```

**解決策2**: 既存のプロセスを終了

```bash
# ポート7860を使用しているプロセスを確認
lsof -i :7860

# プロセスIDを確認してkill
kill -9 <PID>
```

### 問題3: 「No input files found」

**原因**: 指定されたディレクトリに`.feather`ファイルがない

**解決策**:

```bash
# ディレクトリ内容を確認
ls -la ./data/hcpe/

# .featherファイルがあることを確認
# なければ，正しいディレクトリパスを指定
```

### 問題4: VSCodeでポートフォワーディングが動作しない

**解決策**:

1. VSCodeの「ポート」パネルを開く（下部パネル）
2. 手動でポート7860を追加
3. 「ブラウザで開く」アイコンをクリック

### 問題5: データが表示されない

**原因**: モックデータのみ使用している（Phase 3実装時点）

**現在の状態**:
- Phase 1-3完了: モックデータで動作
- 実データ統合は今後の実装予定

**確認方法**:

```bash
# デバッグモードで起動
poetry run maou visualize \
  --input-path ./data/hcpe \
  --array-type hcpe \
  --debug-mode

# ログで「Mock」が表示される場合はモックモード
```

---

## パフォーマンスチューニング

### 大規模データセット（1000万件以上）

**推奨設定**:

```bash
# ページサイズを小さく設定
# Gradio UI内で「1ページあたりの件数」を10-20に設定
```

**理由**:
- ページサイズが大きいと，初回検索が遅くなる
- 10-20件で十分な閲覧性を確保

### メモリ不足の場合

**症状**: サーバーがクラッシュする，応答が遅い

**解決策**:

1. **ファイル数を減らす**:
   ```bash
   # 一部のファイルのみ指定
   poetry run maou visualize \
     --input-path data1.feather --input-path data2.feather \
     --array-type hcpe
   ```

2. **環境変数でログレベルを下げる**:
   ```bash
   export MAOU_LOG_LEVEL=WARNING
   poetry run maou visualize --input-path ./data/hcpe --array-type hcpe
   ```

---

## まとめ

本ツールは，大規模な将棋学習データを効率的に探索するための強力なツールです．VSCodeとGoogle Colabの両方で動作し，直感的なUIにより，データ確認作業を大幅に効率化します．

### 次のステップ

1. **実データ統合**: Phase 3完了後，実際の.featherファイルからデータを読み込めるようになります
2. **高度な機能**: モデル評価値オーバーレイ，棋譜再生機能など
3. **パフォーマンス最適化**: Rustによる完全I/O実装，並列インデックス構築

詳細は[design.md](./design.md)および[api.md](./api.md)を参照してください．
