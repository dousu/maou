# maou
将棋AIを作ってみる

## 開発環境

基本的にはdevcontainerを前提にする．
以下のスクリプトを実行する．

```bash
# devcontainerの場合は以下をインストールしておく
# bash scripts/devcontainer.sh
bash scripts/dev-init.sh
# ここで確認できるpythonのパスをVScodeのインタプリタとして設定する
poetry env info --path
# pre-commit系の設定
poetry run bash scripts/pre-commit.sh
```

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
`poetry sync`または`poetry install`を実行した後，Rust拡張モジュールを明示的にビルドする必要があります．

#### 通常の環境（開発マシン・devcontainer）

```bash
# Rustツールチェーンのインストール（未インストールの場合）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Rust拡張モジュールのビルド（必須）
poetry run maturin develop

# ビルド確認
poetry run python -c "from maou._rust.maou_io import hello; print(hello())"
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
  poetry run maturin develop
```

##### 方法2: 各コマンドでPATHを明示的に設定

```bash
# セル1: Rustツールチェーンのインストール
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# セル2: PATHを設定してビルド（source不要）
!export PATH="$HOME/.cargo/bin:$PATH" && poetry run maturin develop

# セル3: ビルド確認
!export PATH="$HOME/.cargo/bin:$PATH" && poetry run python -c "from maou._rust.maou_io import hello; print(hello())"
```

##### 方法3: 環境変数を永続的に設定

```python
# セル1: Rustインストール
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# セル2: PATHをPython経由で設定（以降のセルでも有効）
import os
os.environ['PATH'] = f"{os.path.expanduser('~')}/.cargo/bin:{os.environ['PATH']}"

# セル3: ビルド実行
!poetry run maturin develop

# セル4: ビルド確認
!poetry run python -c "from maou._rust.maou_io import hello; print(hello())"
```

**注意**: `poetry sync -E cpu -E gcp`などを実行した場合でも，
上記の`poetry run maturin develop`を実行しないとRust拡張モジュールが利用できず，
`maou hcpe-convert`などのコマンドが動作しません．

依存関係を更新した後も，必ず`poetry run maturin develop`を実行してください．

### Pythonアップデート方法

```bash
# pythonのバージョンが新しくなっていることを確認する
python --version
# 今使っている古いpython環境を確認する
poetry env list
poetry env remove ${古いpython環境}
poetry env use python
bash scripts/dev-init.sh

# 注意: dev-init.shはRust拡張モジュールのビルドも実行します
# 手動で依存関係をインストールする場合は以下のように実行してください
# poetry sync -E cpu -E gcp --without dev
# poetry run maturin develop
```

### poetry cache削除

poetryのcacheがたまってストレージ容量を圧迫している場合は以下のコマンドでpoetryのcacheを消せる．
GitHub Codespacesを使っている場合等，ストレージ容量をなるべく削減したいときに利用する．

```bash
poetry cache clear --all .
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
TEST_GCP=true poetry run pytest
```

### AWSを使う場合

```bash
aws configure sso --use-device-code --profile default
# アクセストークンが切れたら以下のように再認証する
# aws sso login --use-device-code --profile default
```

なお，AWSを使ったテストをするときは以下のように行う．

```bash
TEST_AWS=true poetry run pytest
```

## Masked Autoencoderの事前学習コマンド例

`maou pretrain`コマンドでMasked Autoencoderの事前学習を行う．
事前学習で保存したstate_dictは`maou learn-model --resume-from`オプションで読み込める．

```bash
poetry run maou pretrain \
  --input-dir data/preprocessed \
  --input-format preprocess \
  --output-path artifacts/masked-autoencoder.pt \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --mask-ratio 0.75 \
  --hidden-dim 512
```

## torch.autogradのAnomaly Detectionを使ったデバッグ

学習中に不安定な勾配を追跡したい場合は，`maou learn-model` や
`maou utility benchmark-training` に `--detect-anomaly` フラグを付けると
`torch.autograd.set_detect_anomaly` が有効化される．デフォルトは無効のため，
推論やベンチマーク時に余計なオーバーヘッドを負うことはない．
例えば以下のように実行する．

```bash
poetry run maou learn-model --detect-anomaly [...他の引数]

poetry run maou utility benchmark-training --detect-anomaly [...他の引数]
```

## TensorBoardヒストグラムの制御

学習ループではデフォルトで損失や正解率などのスカラー値のみをTensorBoardに
書き出すため，ログファイルは比較的軽量で扱えます。パラメータ／勾配の分布を
追跡したい場合は `--tensorboard-histogram-frequency` を設定することで，指定した
エポック間隔ごとにヒストグラムを追加できます。0を指定するとヒストグラム記録は
無効化されたままです（従来のスカラー指標は常に出力されます）。

特定モジュールだけを記録したい場合は `--tensorboard-histogram-module` を複数回
指定してグロブパターン（例: `backbone.*`, `head.value*`）を登録してください。

```bash
poetry run maou learn-model \
  --tensorboard-histogram-frequency 5 \
  --tensorboard-histogram-module "backbone.*" \
  --tensorboard-histogram-module "head.*" \
  [...他の引数]
```

フィルタを指定しない場合は全パラメータ／勾配を記録します。ヒストグラム無効時も
学習率や検証指標などのスカラー値は引き続き記録されるため，既存のTensorBoard
ダッシュボードはそのまま利用できます。

## Preprocessingデータのメモリマップ方式

前処理済みの`.npy`ファイルはデフォルトでコピーオンライト(`mmap_mode="c"`)
としてメモリマップされます。これにより`numpy.memmap`が
`writeable=True`のまま保持され、`torch.from_numpy()`によるゼロコピー変換が
可能になります。ファイルを汚染することなくテンソルから値を更新できるため、
訓練用データセットをGPUへ転送する際のコピーを削減できます。

ファイルシステム／オブジェクトストレージ／BigQueryの各データソースには
`preprocessing_mmap_mode`引数を追加しており、必要に応じてコピーオンライト以外の
モードへ切り替えることもできます。
