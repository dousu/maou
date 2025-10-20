# 依存関係管理

このドキュメントでは，maouプロジェクトにおける依存関係管理の方法について説明します．特に，Poetryの「extras」機能を使用して，異なるGPU環境やクラウドプロバイダごとに必要なライブラリを分けて管理する方法を紹介します．

## Poetryのextrasとは

Poetryの「extras」機能は，オプショナルな依存関係をグループ化し，必要に応じてインストールできるようにする機能です．これにより，以下のようなメリットがあります：

- **ストレージ容量の節約**: 必要なライブラリのみをインストールすることで，ストレージ容量を節約できます
- **クラウドコストの削減**: クラウド環境では，不要なライブラリをインストールしないことでコストを削減できます
- **環境ごとの最適化**: 異なる環境（開発，テスト，本番など）に適した依存関係を管理できます

## maouプロジェクトでのextrasの使用方法

maouプロジェクトでは，以下の2種類のextrasを定義しています：

### 1. GPUタイプごとのextra

異なるGPU環境に応じた依存関係を管理します：

- **cpu**: CPU環境用
  - 含まれるライブラリ: torch (PyPI版), torchinfo, torch-tb-profiler, onnxruntime, onnxruntime-tools, onnxsim
  - 用途: GPUを使用しない環境での学習や推論

- **cuda**: NVIDIA GPU環境用
  - 含まれるライブラリ: torch (PyPI版), pytorch-cuda (CUDAメタパッケージ), torchinfo, torch-tb-profiler, pynvml, onnxruntime-gpu, onnxruntime-tools, onnxsim
  - 用途: NVIDIA GPUを使用した高速な学習や推論
  - 特記事項: CUDAメタパッケージはPyTorch公式リポジトリから取得し，PyPI版のtorchと組み合わせて利用します

- **mpu**: Apple Silicon環境用
  - 含まれるライブラリ: torch, torchinfo, torch-tb-profiler, onnxruntime-gpu, onnxruntime-tools, onnxsim
  - 用途: Apple Silicon（M1/M2/M3チップ）のMPSを使用した学習や推論
  - 特記事項: 標準のPyTorchパッケージにMPSサポートが含まれています

- **tpu**: Google TPU環境用
  - 含まれるライブラリ: torch-xla, torch, torchinfo, torch-tb-profiler, onnxruntime-gpu, onnxruntime-tools, onnxsim
  - 用途: Google Cloud TPUを使用した高速な学習

### 2. クラウドプロバイダごとのextra

異なるクラウドプロバイダに応じた依存関係を管理します：

- **gcp**: Google Cloud Platform用
  - 含まれるライブラリ: google-cloud-storage, google-cloud-bigquery, google-crc32c, google-cloud-bigquery-storage
  - 用途: GCSやBigQueryを使用したデータの保存や分析

- **aws**: Amazon Web Services用
  - 含まれるライブラリ: boto3
  - 用途: S3などのAWSサービスを使用したデータの保存や分析

## インストール方法

### 基本的なインストール

```bash
# 基本インストール（最小構成）
poetry install

# CPU + GCP環境用
poetry install -E cpu -E gcp

# CUDA + GCP環境用（CUDAメタパッケージ + pynvml を含む）
poetry install -E cuda -E gcp

# CPU + AWS環境用
poetry install -E cpu -E aws

# CUDA + AWS環境用
poetry install -E cuda -E aws

# TPU + GCP環境用（Google Cloud TPU環境）
poetry install -E tpu -E gcp

# MPU + AWS環境用（Apple Silicon + AWS）
poetry install -E mpu -E aws
```

### 環境変数を使用した自動インストール

以下のようなスクリプトを作成することで，環境変数に基づいて適切な依存関係を自動的にインストールできます：

```bash
#!/bin/bash
# install-deps.sh

# GPUタイプとクラウドプロバイダを環境変数から取得
GPU_TYPE=${GPU_TYPE:-"cpu"}  # デフォルトはCPU
CLOUD_PROVIDER=${CLOUD_PROVIDER:-"gcp"}  # デフォルトはGCP

# インストールコマンドを構築
INSTALL_CMD="poetry install -E $GPU_TYPE -E $CLOUD_PROVIDER"

echo "Installing dependencies for GPU type: $GPU_TYPE and cloud provider: $CLOUD_PROVIDER"
echo "Running: $INSTALL_CMD"

# 実行
eval $INSTALL_CMD
```

使用例：

```bash
# CPU + GCP環境用（デフォルト）
./install-deps.sh

# CUDA + GCP環境用
GPU_TYPE=cuda ./install-deps.sh

# CPU + AWS環境用
CLOUD_PROVIDER=aws ./install-deps.sh

# CUDA + AWS環境用
GPU_TYPE=cuda CLOUD_PROVIDER=aws ./install-deps.sh
```

## PyTorchのCPU/CUDA依存関係

`pyproject.toml`では，`torch`と`pytorch-cuda`をそれぞれオプショナル依存関係として定義しています．`torch`はデフォルトのPyPIソースから取得し，`pytorch-cuda`はCUDAドライバとライブラリを束ねたメタパッケージとしてPyTorch公式インデックスから取得します．この構成により，CPU環境のインストールではPyPIのみを参照し，CUDA環境のインストール時のみPyTorch公式リポジトリにアクセスします．

PyTorch公式リポジトリは次のように設定されています：

```toml
# PyTorch CUDA版のソース設定
[[tool.poetry.source]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
priority = "explicit"
```

`cuda` extraでは`pytorch-cuda`と`pynvml`が追加で解決されるため，CUDAメタパッケージに付随する`nvidia-*`ファミリの依存関係が必要なときだけインストールされます。一方，`cpu` extraはPyPI上の`torch`のみを参照するため，CUDA用インデックスにはアクセスしません。

### ローカルでの確認手順

以下のコマンドを実行すると，CPUインストールではPyTorch公式インデックスにアクセスしないこと，CUDAインストールでは`pytorch-cuda`の取得に同インデックスが使用されることを確認できます。

```bash
poetry install -E cpu
poetry install -E cuda
```

前者では`https://download.pytorch.org/whl/cu128`へのアクセスが行われない一方で，後者では`pytorch-cuda`とCUDA関連の依存関係が解決されるログが出力されます。

## 使用例

### 異なるGPU環境でのPyTorchの使用例

```python
import torch

# GPU利用可能かどうかを確認
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# モデルとテンソルをデバイスに移動
model = MyModel().to(device)
inputs = torch.randn(1, 3, 224, 224).to(device)

# 推論
outputs = model(inputs)
```

### GCPとAWSでのストレージアクセスの違い

以下は，GCPとAWSでのストレージアクセスの違いを示すコード例です：

```python
# GCP (Google Cloud Storage) を使用する場合
def upload_to_gcp(local_file_path, bucket_name, blob_name):
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{blob_name}")

# AWS (S3) を使用する場合
def upload_to_aws(local_file_path, bucket_name, object_name):
    import boto3

    s3_client = boto3.client('s3')

    s3_client.upload_file(local_file_path, bucket_name, object_name)
    print(f"File {local_file_path} uploaded to s3://{bucket_name}/{object_name}")
```

## 新しいextraの追加方法

新しいGPUタイプやクラウドプロバイダを追加する場合は，以下の手順で行います：

1. `pyproject.toml`の`[tool.poetry.dependencies]`セクションに，新しいライブラリを`optional = true`として追加
2. `[tool.poetry.extras]`セクションに，新しいextraを定義し，必要なライブラリを指定
3. 必要に応じて，特殊なソース設定を追加

例えば，新しいクラウドプロバイダ「azure」を追加する場合：

```toml
[tool.poetry.dependencies]
# 既存の依存関係
# ...

# Azure関連
azure-storage-blob = {version = "^12.13.0", optional = true}
azure-cosmos = {version = "^4.3.0", optional = true}

[tool.poetry.extras]
# 既存のextra
# ...

# 新しいextra
azure = ["azure-storage-blob", "azure-cosmos"]
```

## 依存関係の更新

依存関係を更新する場合は，以下のコマンドを使用します：

```bash
# すべての依存関係を更新
poetry update

# 特定のパッケージを更新
poetry update パッケージ名
```

更新後は，各環境でテストを行い，問題がないことを確認してください．
