# Polars プリビルド調査レポート

## 背景

最小の Codespaces 環境(2-core / 4GB RAM)では，Rust の polars クレートを含む `maturin develop` ビルドが重く，時間がかかるかメモリ不足で失敗する．GitHub Actions でプリビルドした成果物を持ち込むことで，`uv sync` 時の maturin 実行を回避できないか調査した．

## 結論: 実現可能

GitHub Actions でプリビルドした wheel を Codespaces にインストールする構成は**実現可能**である．以下に3つのアプローチを実現性・複雑さの観点で整理する．

---

## アプローチ比較

| アプローチ | 複雑さ | 変更範囲 | 推奨度 |
|---|---|---|---|
| A. wheel 事前インストール方式 | 低 | dev-init.sh + CI workflow | **推奨** |
| B. 別パッケージ分離方式 | 高 | パッケージ構造全体 | 将来的に検討 |
| C. カスタムビルドバックエンド方式 | 中 | pyproject.toml + wrapper | 非推奨 |

---

## アプローチ A: wheel 事前インストール方式（推奨）

### 概要

1. GitHub Actions で maturin wheel をビルドし，GitHub Releases にアップロード
2. `dev-init.sh` で `gh release download` → `uv pip install` で wheel をインストール
3. `uv sync --no-install-project` で Python 依存のみインストール(maturin ビルドをスキップ)

### メリット

- 既存のプロジェクト構造を変更しない
- `pyproject.toml` の `build-system` はそのまま(ローカル開発者は従来通り `maturin develop` 可能)
- 実装が最もシンプル

### デメリット

- `uv sync` ではなく `uv sync --no-install-project` + `uv pip install` の2段階になる
- wheel のバージョン管理が手動(タグ push 時に自動化は可能)

### 実装詳細

#### 1. GitHub Actions ワークフロー

```yaml
# .github/workflows/build-rust-wheel.yml
name: Build Rust Wheel

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:  # 手動トリガーも可能

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Build wheel
        uses: PyO3/maturin-action@v1
        with:
          target: x86_64
          manylinux: 2_28
          args: >-
            --release
            --out dist
            --interpreter python3.12
            --strip
          sccache: 'true'

      - name: Upload wheel as artifact
        uses: actions/upload-artifact@v4
        with:
          name: wheel-linux-x86_64
          path: dist/*.whl

      - name: Upload to GitHub Release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*.whl
```

**ポイント:**
- `manylinux: 2_28` で Codespaces の Ubuntu 環境と互換
- `--strip` でデバッグシンボルを除去し，wheel サイズを削減
- `sccache: 'true'` で CI ビルドの再実行を高速化
- `workflow_dispatch` で手動トリガーも可能(Rust コード変更時)

#### 2. dev-init.sh の変更

```bash
#!/bin/bash
set -eux

# --- Rust toolchain は不要(プリビルド wheel を使用) ---

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv generate-shell-completion bash >> ~/.bash_completion

REPO="owner/maou"  # GitHub リポジトリ
WHEEL_PATTERN="maou-*-cp312-cp312-manylinux_2_28_x86_64.whl"

# プリビルド wheel のダウンロードとインストール
if gh release download --repo "$REPO" --pattern "$WHEEL_PATTERN" \
     --dir /tmp/wheels --clobber 2>/dev/null; then
    echo "Pre-built wheel found. Installing..."

    # Python 依存のみインストール(maturin ビルドをスキップ)
    uv sync --no-install-project --extra cpu --extra visualize --group dev

    # プリビルド wheel をインストール
    uv pip install /tmp/wheels/$WHEEL_PATTERN

    rm -rf /tmp/wheels
else
    echo "Pre-built wheel not found. Building from source..."

    # Rust toolchain が必要なフォールバック
    if ! command -v rustup &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # メモリ最適化設定(既存のロジック)
    # ... (現在の dev-init.sh の cargo config 設定)

    uv sync --extra cpu --extra visualize --group dev
fi

# Rust モジュールの検証
uv run python -c "from maou._rust.maou_io import hello; print(hello())"

source .venv/bin/activate
uv cache clean
```

#### 3. wheel ファイル名について

maturin が生成する wheel ファイル名は以下の形式:

```
maou-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

- `cp312-cp312`: CPython 3.12 用(ABI タグ)
- `manylinux_2_28_x86_64`: Linux glibc 2.28 以上の x86_64

#### 4. `uv sync --no-install-project` の挙動

```bash
# 依存パッケージのみインストール(maou 自体はビルド/インストールしない)
uv sync --no-install-project --extra cpu --extra visualize --group dev
```

これにより:
- `polars`, `torch`, `click` 等の依存はすべてインストールされる
- `maou` パッケージ自体の maturin ビルドはスキップされる
- その後 `uv pip install /tmp/wheels/maou-*.whl` でプリビルド wheel をインストール

---

## アプローチ B: 別パッケージ分離方式

### 概要

Rust 拡張を `maou-rust` として独立パッケージに分離し，`maou` 本体は純粋な Python パッケージにする．

### 構成イメージ

```
maou/
├── pyproject.toml          # build-backend = "hatchling" (Python only)
├── crates/
│   └── maou-rust/
│       ├── pyproject.toml  # build-backend = "maturin"
│       └── Cargo.toml
└── src/
    └── maou/
```

```toml
# Root pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "maou"
dependencies = ["maou-rust>=0.1.0"]

[tool.uv.sources]
# 開発時: workspace メンバーからビルド
maou-rust = { workspace = true }
# または Codespaces 用: プリビルド wheel の URL
# maou-rust = { url = "https://github.com/.../maou_rust-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl" }
```

```toml
# crates/maou-rust/pyproject.toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "maou-rust"
version = "0.1.0"
```

### メリット

- `maou` 本体の `uv sync` で Rust ビルドが不要になる
- `maou-rust` は PyPI にも公開可能
- `[tool.uv.sources]` で開発環境ごとにソースを切り替え可能

### デメリット

- プロジェクト構造の大幅な変更が必要
- import パスの変更(`maou._rust` → `maou_rust` など)
- uv workspace の設定が複雑化する
- 2つのパッケージのバージョン管理が必要

---

## アプローチ C: カスタムビルドバックエンド方式（非推奨）

### 概要

PEP 517 の `backend-path` を使い，環境変数で maturin ビルドをスキップするカスタムバックエンドを作成する．

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "backend"
backend-path = ["_custom_build"]
```

```python
# _custom_build/backend.py
import os
from maturin import *

if os.environ.get("SKIP_RUST_BUILD"):
    def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
        # プリビルド wheel をコピーするだけ
        ...
```

### 非推奨理由

- maturin の内部 API に依存するため脆弱
- デバッグが困難
- アプローチ A で十分に目的を達成できる

---

## 補足: uv の関連設定

### cache-keys（ローカル開発向け最適化）

Rust ソースが変更されていない場合に `uv sync` のリビルドをスキップ:

```toml
[tool.uv]
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "rust/**/*.rs" },
]
```

### flat index（wheel ホスティング）

GitHub Releases の wheel を index として参照:

```toml
[[tool.uv.index]]
name = "maou-wheels"
url = "https://github.com/owner/maou/releases/expanded_assets/v0.2.0"
format = "flat"
explicit = true
```

### --no-build-package（ビルド抑制）

特定パッケージのソースビルドを明示的に禁止:

```bash
uv sync --no-build-package maou
```

---

## 推奨実装手順

1. **GitHub Actions ワークフローを追加** (`.github/workflows/build-rust-wheel.yml`)
2. **手動でワークフローを実行**し，wheel が正常にビルドされることを確認
3. **GitHub Release を作成**し，wheel がアップロードされることを確認
4. **dev-init.sh を修正**し，プリビルド wheel のダウンロード → インストールに変更
5. **Codespaces で動作確認** (新規 Codespace を作成してテスト)
6. **cache-keys を pyproject.toml に追加**(ローカル開発者向けの最適化)

## 参考資料

- [PyO3/maturin-action](https://github.com/PyO3/maturin-action) - GitHub Action for maturin
- [Maturin Distribution Guide](https://www.maturin.rs/distribution.html)
- [uv: Managing Dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv: Package Indexes](https://docs.astral.sh/uv/concepts/indexes/)
- [uv: Settings Reference](https://docs.astral.sh/uv/reference/settings/)
- [GitHub Releases API](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [softprops/action-gh-release](https://github.com/softprops/action-gh-release)
