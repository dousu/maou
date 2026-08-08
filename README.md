# maou
将棋AIを作ってみる

## 開発環境

基本的にはdevcontainerを前提にする．
以下のスクリプトを実行する．

```bash
# devcontainerの場合は以下をインストールしておく
# bash scripts/devcontainer.sh
bash scripts/dev-init.sh
# pre-commit系の設定
uv run bash scripts/pre-commit.sh
```

uvはプロジェクト直下に`.venv`を作成するため，VScodeのインタプリタには`.venv/bin/python`を設定する．

ここでシェルスクリプトを実行するような構成になっているのは，
devcontainerのfeaturesになるべくインストールを任せたいため．

featuresはインストール順序としては最後になるためPython等に依存しているとDockerfileにインストール処理を書けない．

### Serena MCP Server（Claude Code連携）

Claude Codeでのトークン消費を削減するため，Serena MCPサーバーを使用している．
DevContainerでは自動設定されるが，ローカル環境では手動セットアップが必要．

```bash
# Serenaのインストール
bash scripts/setup-serena.sh

# rust-analyzerのインストール（Rustシンボル解析用）
rustup component add rust-analyzer

# 環境変数の設定（シェル起動時に実行）
bash scripts/start-serena.sh
```

Claude Codeで使用する場合は，`.mcp.json`がプロジェクトルートに存在することを確認する．
Serenaはpyright（Python）とrust-analyzer（Rust）を使用してシンボルレベルのコード解析を提供する．

### Rustバックエンドのビルド

**重要**: このプロジェクトはPython拡張モジュールとしてRustコードを使用しています．
`uv sync`を実行した後，Rust拡張モジュールを明示的にビルドする必要があります．

#### 通常の環境（開発マシン・devcontainer）

```bash
# Rustツールチェーンのインストール（未インストールの場合）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Rust拡張モジュールのビルド（必須）
uv run maturin develop

# ビルド確認
uv run python -c "from maou._rust.maou_io import hello; print(hello())"
# 期待される出力: "Maou I/O Rust backend initialized"
```

#### 非対話的環境（Google Colab・Jupyter Notebook）

Google ColabやJupyter Notebookなどの非対話的環境では，`-y`フラグを使用して自動承認モードでインストールします．

**重要**: Colabでは各セル（`!`コマンド）が独立したシェルセッションで実行されるため，
環境変数の設定が次のセルに引き継がれません．以下のいずれかの方法で実行してください．

##### 方法1: すべてのコマンドを1つのセルで実行（推奨）

```bash
# Rustインストール + ビルドを1つのセルで実行
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  source "$HOME/.cargo/env" && \
  uv run maturin develop
```

##### 方法2: 各コマンドでPATHを明示的に設定

```bash
# セル1: Rustツールチェーンのインストール
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# セル2: PATHを設定してビルド（source不要）
!export PATH="$HOME/.cargo/bin:$PATH" && uv run maturin develop

# セル3: ビルド確認
!export PATH="$HOME/.cargo/bin:$PATH" && uv run python -c "from maou._rust.maou_io import hello; print(hello())"
```

##### 方法3: 環境変数を永続的に設定

```python
# セル1: Rustインストール
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# セル2: PATHをPython経由で設定（以降のセルでも有効）
import os
os.environ['PATH'] = f"{os.path.expanduser('~')}/.cargo/bin:{os.environ['PATH']}"

# セル3: ビルド実行
!uv run maturin develop

# セル4: ビルド確認
!uv run python -c "from maou._rust.maou_io import hello; print(hello())"
```

**注意**: `uv sync --extra cpu --extra gcp`などを実行した場合でも，
上記の`uv run maturin develop`を実行しないとRust拡張モジュールが利用できず，
`maou hcpe-convert`などのコマンドが動作しません．

依存関係を更新した後も，必ず`uv run maturin develop`を実行してください．

### Pythonアップデート方法

```bash
# pythonのバージョンが新しくなっていることを確認する
python --version
# 既存のvenvを削除して再作成する
rm -rf .venv
bash scripts/dev-init.sh

# 注意: dev-init.shはRust拡張モジュールのビルドも実行します
# 手動で依存関係をインストールする場合は以下のように実行してください
# uv sync --extra cpu --extra gcp --no-dev
# uv run maturin develop
```

### uvキャッシュ削除

uvのキャッシュがたまってストレージ容量を圧迫している場合は以下のコマンドでuvのキャッシュを消せる．
GitHub Codespacesを使っている場合等，ストレージ容量をなるべく削減したいときに利用する．

```bash
uv cache clean
```

### GCPを使う場合

以下のコマンドを実行してGCPへの認証をしておくと，テストやプログラム中でGCPを利用できる．

```bash
gcloud auth application-default login
# gcloud projects listで設定可能なプロジェクトを確認できる
gcloud config set project "your-project-id"
gcloud auth application-default set-quota-project "your-project-id"
```

なお，GCPを使ったテストをするときは以下のように行う．

```bash
TEST_GCP=true uv run pytest
```

### AWSを使う場合

```bash
aws configure sso --use-device-code --profile default
# アクセストークンが切れたら以下のように再認証する
# aws sso login --use-device-code --profile default
```

なお，AWSを使ったテストをするときは以下のように行う．

```bash
TEST_AWS=true uv run pytest
```


## torch.autogradのAnomaly Detectionを使ったデバッグ

学習中に不安定な勾配を追跡したい場合は，`maou learn-model` や
`maou utility benchmark-training` に `--detect-anomaly` フラグを付けると
`torch.autograd.set_detect_anomaly` が有効化される．デフォルトは無効のため，
推論やベンチマーク時に余計なオーバーヘッドを負うことはない．
例えば以下のように実行する．

```bash
uv run maou learn-model --detect-anomaly [...他の引数]

uv run maou utility benchmark-training --detect-anomaly [...他の引数]
```

## TensorBoardへの出力

学習ループはTensorBoardにスカラー指標のみを書き出します。パラメータ／勾配の
ヒストグラム出力を制御していた `--tensorboard-histogram-frequency` と
`--tensorboard-histogram-module` は2026-08-04に削除されました。

## Preprocessingデータの読み込み方式

前処理済みデータは Arrow IPC (`.feather`) が既定で、読み込み方式は
`--input-cache-mode {file,memory}` で選びます (`mmap` は deprecated で、
内部的に `file` に変換されます)。`KifDataset` は `torch.from_numpy()` で
ゼロコピー変換するため、read-only 配列を渡すと `ValueError` になります
(`src/maou/app/learning/dataset.py:186-198`)。
