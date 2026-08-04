# 依存関係管理

このドキュメントでは，maouプロジェクトにおける依存関係管理の方法について説明します．特に，uvと標準の「extras」機能を使用して，異なるGPU環境やクラウドプロバイダごとに必要なライブラリを分けて管理する方法を紹介します．

パッケージ管理には **uv のみ** を使用します．pip や Poetry は使用しません．

## extrasとは

extras (`[project.optional-dependencies]`) は，オプショナルな依存関係をグループ化し，必要に応じてインストールできるようにするPEP 621標準の機能です．これにより，以下のようなメリットがあります：

- **ストレージ容量の節約**: 必要なライブラリのみをインストールすることで，ストレージ容量を節約できます
- **クラウドコストの削減**: クラウド環境では，不要なライブラリをインストールしないことでコストを削減できます
- **環境ごとの最適化**: 異なる環境 (開発，テスト，本番など) に適した依存関係を管理できます

## maouプロジェクトでのextrasの使用方法

maouプロジェクトでは，以下の4種類のextrasを定義しています：

### 1. GPUタイプごとのextra

異なるGPU環境に応じた依存関係を管理します．**これら3つは排他** で，`pyproject.toml` の `[tool.uv] conflicts` で同時指定できないよう宣言されています：

- **cpu**: CPU環境用
  - 含まれるライブラリ: torch (CPU版), torchinfo, torch-tb-profiler, onnxruntime, onnxruntime-tools, onnxslim
  - 用途: GPUを使用しない環境での学習や推論

- **cuda**: NVIDIA GPU環境用
  - 含まれるライブラリ: torch (CUDA版), torchinfo, torch-tb-profiler, nvidia-ml-py, onnxruntime-gpu, onnxruntime-tools, onnxslim
  - 用途: NVIDIA GPUを使用した高速な学習や推論
  - 特記事項: PyTorchのCUDA版は特殊なindex設定が必要

- **mpu**: Apple Silicon環境用
  - 含まれるライブラリ: torch, torchinfo, torch-tb-profiler, onnxruntime-gpu, onnxruntime-tools, onnxslim
  - 用途: Apple Silicon (M1/M2/M3チップ) のMPSを使用した学習や推論
  - 特記事項: 標準のPyTorchパッケージにMPSサポートが含まれています

### 2. 推論専用のextra

学習を伴わない推論のみの環境向けに，torchを含まない軽量なextraを用意しています：

- **cpu-infer**: CPU推論用 (onnxruntime のみ)
- **onnx-gpu-infer**: GPU推論用 (onnxruntime-gpu==1.22.*)
- **tensorrt-infer**: TensorRT推論用 (onnxruntime-gpu==1.22.*, tensorrt-cu12==10.* ほか)

`onnx-gpu-infer` / `tensorrt-infer` のバージョンが固定されているのは意図的です．Rust wheel (`maou_search`) が ort crate 経由で onnxruntime を静的リンクしており，実行時に dlopen される provider の `.so` が**静的コアと同一版でないとABI不一致でロードに失敗する**ためです．

静的リンクされる版は `rust/maou_search/Cargo.toml` の `ort` のバージョンで決まります：

| ort | onnxruntime | TensorRT |
|---|---|---|
| `=2.0.0-rc.10` (現在) | 1.22 | 10系 |
| `=2.0.0-rc.13` | 1.28 | 11系 |

`ort` が `=` で厳密固定されているのはこのためです．**`ort` を上げるときは `pyproject.toml` の
`onnxruntime-gpu` / `tensorrt-cu12` と `uv.lock` の解決版も必ず同時に合わせる必要があります**．

### 依存更新時の落とし穴 (2026-08-04 に実際に踏んだ)

Dependabot がこの結合を知らずに片側だけ更新し，以下が同時に起きた：

1. **範囲を広げただけでは版は揃わない**．`onnxruntime-gpu>=1.22,<1.29` に緩めても
   `uv lock` は最小変更で再解決するため 1.22.0 が据え置かれる．
   一方 Rust 側は ort rc.13 で 1.28 を静的リンクしており，食い違ったまま lock される．
   揃えるには明示的な upgrade が要る：

   ```bash
   uv lock --upgrade-package onnxruntime-gpu --upgrade-package tensorrt-cu12
   ```

2. **`pyproject.toml` だけ更新して `uv.lock` を再生成し忘れる**と，
   pre-commit の `uv-lock` フックが落ちる．`uv lock --check` で事前に検出できる．

3. **polars 0.54.4 はこのリポジトリの feature 構成でコンパイルが通らない**
   (`polars-stream` 内で `error[E0433]: cannot find type IRStringFunction`)．
   PR には Rust をビルドする CI が無いため，`cargo build` / `maturin develop` を
   手元で通すまで誰も気付かない．そのため polars は 0.53 に据え置いている．

詳細は `pyproject.toml` のコメントを参照してください．

### 3. クラウドプロバイダごとのextra

異なるクラウドプロバイダに応じた依存関係を管理します：

- **gcp**: Google Cloud Platform用
  - 含まれるライブラリ: google-cloud-storage, google-cloud-bigquery[pandas], google-cloud-bigquery-storage
  - 用途: GCSやBigQueryを使用したデータの保存や分析

- **aws**: Amazon Web Services用
  - 含まれるライブラリ: boto3
  - 用途: S3などのAWSサービスを使用したデータの保存や分析

### 4. その他のextra

- **visualize**: 可視化ツール用 (gradio, matplotlib)
- **fetch**: floodgate棋譜取得 (`fetch-floodgate`) の年次 `.7z` アーカイブ展開用 (py7zr)

### 開発用依存関係 (extraではない)

pytest や mypy などの開発ツールは extra ではなく `[dependency-groups]` の `dev` グループにまとめています．`uv sync` はデフォルトで dev グループを含めるため，明示的な指定は不要です．本番環境などで除外したい場合は `--no-dev` を使用します．

## インストール方法

### 基本的なインストール

```bash
# 基本インストール (最小構成 + dev グループ)
uv sync

# CPU + GCP環境用
uv sync --extra cpu --extra gcp

# CUDA + GCP環境用
uv sync --extra cuda --extra gcp

# CPU + AWS環境用
uv sync --extra cpu --extra aws

# CUDA + AWS環境用
uv sync --extra cuda --extra aws

# MPU + AWS環境用 (Apple Silicon + AWS)
uv sync --extra mpu --extra aws

# 本番環境用 (dev グループを除外)
uv sync --extra cpu --extra gcp --no-dev
```

`scripts/dev-init.sh` は開発環境向けに `uv sync --extra cpu --extra visualize --group dev` を実行します．

なお `uv sync` は `.venv` を `pyproject.toml` / `uv.lock` の内容に**同期**します．ロックファイルに無いパッケージは削除されるため，手動で入れたものは残りません．

### 環境変数を使用した自動インストール

以下のようなスクリプトを作成することで，環境変数に基づいて適切な依存関係を自動的にインストールできます：

```bash
#!/bin/bash
# install-deps.sh
set -eu

# GPUタイプとクラウドプロバイダを環境変数から取得
GPU_TYPE=${GPU_TYPE:-"cpu"}  # デフォルトはCPU
CLOUD_PROVIDER=${CLOUD_PROVIDER:-"gcp"}  # デフォルトはGCP

echo "Installing dependencies for GPU type: $GPU_TYPE and cloud provider: $CLOUD_PROVIDER"

uv sync --extra "$GPU_TYPE" --extra "$CLOUD_PROVIDER"
```

使用例：

```bash
# CPU + GCP環境用 (デフォルト)
./install-deps.sh

# CUDA + GCP環境用
GPU_TYPE=cuda ./install-deps.sh

# CPU + AWS環境用
CLOUD_PROVIDER=aws ./install-deps.sh

# CUDA + AWS環境用
GPU_TYPE=cuda CLOUD_PROVIDER=aws ./install-deps.sh
```

## PyTorchのCUDA版の特殊な設定

PyTorchのCUDA版は，通常のPyPIリポジトリではなく，PyTorch公式のリポジトリからインストールする必要があります．maouプロジェクトでは，以下のように設定しています：

```toml
# extra ごとに参照するindexを切り替える
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cuda", extra = "cuda" },
  { index = "pytorch-cuda", extra = "mpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

`explicit = true` は「`[tool.uv.sources]` で明示的に指定されたパッケージ以外はこのindexから取得しない」という指定です．これがないと，他のパッケージまでPyTorchのindexから解決されてしまいます．

この設定により，`cuda` extraを指定してインストールする際に，PyTorch公式のリポジトリからCUDA 12.8対応のバージョンがインストールされます．

また，`cpu` / `cuda` / `mpu` は同じ `torch` を別indexから引くため共存できません．uvにはこれを宣言する仕組みがあり，以下のように排他を明示しています：

```toml
[tool.uv]
package = true
conflicts = [
  [
    { extra = "cpu" },
    { extra = "cuda" },
    { extra = "mpu" },
  ],
]
```

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

1. `pyproject.toml`の`[project.optional-dependencies]`セクションに，新しいextra名でライブラリを列挙する
2. 必要に応じて，`[tool.uv.sources]` / `[[tool.uv.index]]` で特殊なindex設定を追加する
3. 既存extraと共存できない場合は `[tool.uv] conflicts` に排他を宣言する
4. `uv lock` を実行して `uv.lock` を更新し，コミットに含める

例えば，新しいクラウドプロバイダ「azure」を追加する場合：

```toml
[project.optional-dependencies]
# 既存のextra
# ...

# 新しいextra
azure = [
    "azure-storage-blob>=12.13.0",
    "azure-cosmos>=4.3.0",
]
```

Poetryと異なり，個々のライブラリを `optional = true` として別途宣言する必要はありません．extraのリストに書くだけでオプショナル扱いになります．

### 個別パッケージの追加

`pyproject.toml` を手で編集せずにパッケージを追加する場合は `uv add` を使用します：

```bash
# 通常の依存関係に追加
uv add パッケージ名

# extra に追加
uv add --optional visualize パッケージ名

# dev グループに追加
uv add --group dev パッケージ名

# 削除
uv remove パッケージ名
```

`uv add` / `uv remove` は `pyproject.toml` と `uv.lock` の両方を更新します．

## 依存関係の更新

依存関係を更新する場合は，以下のコマンドを使用します：

```bash
# すべての依存関係を更新 (ロックファイルを再解決してから同期)
uv lock --upgrade && uv sync

# 特定のパッケージを更新
uv lock --upgrade-package パッケージ名 && uv sync
```

`uv.lock` はコミット対象です．pre-commitの `uv-lock` フック (`astral-sh/uv-pre-commit`) が `pyproject.toml` と `uv.lock` の整合性を検査するため，`pyproject.toml` を変更したまま `uv lock` を忘れるとコミット時に検出されます．

更新後は，各環境でテストを行い，問題がないことを確認してください．

## Rust拡張モジュールの再ビルド

このプロジェクトはPython拡張モジュールとしてRustコードを使用しています．依存関係を更新した後は，Rust拡張モジュールの再ビルドが必要です：

```bash
uv run maturin develop
```

詳細は [README.md](../README.md) の「Rustバックエンドのビルド」を参照してください．
