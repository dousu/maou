# Testing Guide

## Testing Requirements

**Framework**: Use `uv run pytest`

### 前提: GPU extra が要る

テストスイートの実行には `torch` が要る．`uv sync` の base install だけ
では `tests/maou/app/learning/` をはじめ torch を import するモジュール
が collect 段で失敗する．**先に GPU extra を入れること．**

```bash
uv sync --extra cpu                     # または --extra cuda
```

`torch` は `tests/conftest.py` の `_OPTIONAL_DEPS` に**敢えて入れて
いない**．collect 段の skip はモジュールを丸ごと落とすため，torch を
そこに入れると「環境が整っていない」実行が緑として報告されてしまう．
`onnxruntime` / `onnx` / `gradio` / `matplotlib` は該当モジュールが
局所的なので，従来どおり skip に書き換えられる．

```bash
uv run pytest                           # Run all tests
uv run pytest --cov=src/maou            # Run with coverage
TEST_GCP=true uv run pytest             # Test GCP features
TEST_AWS=true uv run pytest             # Test AWS features
```

### Test Requirements
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests
- **Edge cases**: Test error scenarios and boundary conditions
- **Integration tests**: Test cloud provider integrations when applicable

## Test Organization and Conventions

**CRITICAL:** Tests must strictly mirror the source directory structure following Clean Architecture layers.

### Directory Structure Rules

**Pattern:**
```
src/maou/{layer}/{module}/file.py
  → tests/maou/{layer}/{module}/test_file.py
```

**Layer Mapping:**
- `src/maou/domain/` → `tests/maou/domain/`
- `src/maou/app/` → `tests/maou/app/`
- `src/maou/interface/` → `tests/maou/interface/`
- `src/maou/infra/` → `tests/maou/infra/`

**Examples:**
```
src/maou/domain/board/shogi.py
  → tests/maou/domain/board/test_shogi.py

src/maou/app/learning/training_loop.py
  → tests/maou/app/learning/test_training_loop.py

src/maou/infra/s3/s3_data_source.py
  → tests/maou/infra/s3/test_s3_data_source.py
```

### Test File Naming Conventions

**Required format:** `test_{module_name}.py`

**Rules:**
1. **Prefix:** Always start with `test_` for pytest discovery
2. **Module name:** Match the source file name exactly
3. **Descriptive suffixes:** Add clarifying suffixes when testing specific aspects

**Examples:**
- ✅ `test_s3_data_source.py` - Primary S3 DataSource tests
- ✅ `test_file_data_source_stage_support.py` - Stage-specific feature tests
- ❌ `test_validation.py` - Too generic
- ❌ `tests/maou/app/test_training_loop.py` - Wrong location (should be in app/learning/)

### Special Test Directories

**Integration Tests:** `tests/maou/integrations/`
- Purpose: End-to-end tests spanning multiple layers
- Naming: `test_{workflow}_{scenario}.py`
- Examples: `test_app_hcpe_converter.py`, `test_convert_and_preprocess.py`

### Test Resource Files

**Rule:** Co-locate test resources with the test files that use them.

**Structure:**
```
tests/maou/{layer}/{module}/
├── test_feature.py
└── resources/
    ├── sample_input.csa
    └── expected_output.feather
```

### Creating New Tests

**Workflow:**
1. Identify source file: `src/maou/{layer}/{module}/feature.py`
2. Create test file: `tests/maou/{layer}/{module}/test_feature.py`
3. Add test class: `class TestFeatureName:` (optional but recommended)
4. Add test functions: `def test_{specific_behavior}() -> None:`
5. Add resources: Create `resources/` directory if needed

**Template:**
```python
"""Tests for {layer}.{module}.{feature} module."""

from pathlib import Path

import pytest

from maou.{layer}.{module}.{feature} import FeatureClass


class TestFeatureClass:
    """Test suite for FeatureClass."""

    def test_{specific_behavior}(self) -> None:
        """Test that {specific behavior} works correctly."""
        # Arrange
        instance = FeatureClass()

        # Act
        result = instance.method()

        # Assert
        assert result == expected_value
```

### Running Tests by Layer

```bash
# All tests
uv run pytest

# Specific layer
uv run pytest tests/maou/domain/
uv run pytest tests/maou/app/
uv run pytest tests/maou/infra/

# Specific module
uv run pytest tests/maou/app/learning/
uv run pytest tests/maou/domain/board/

# Integration tests only
uv run pytest tests/maou/integrations/

# With coverage
uv run pytest --cov=src/maou --cov-report=html
```

### 実行時間と Rust 拡張のビルドコスト

Python テストは `maou._rust` 拡張に依存するので，所要時間は「拡張のビルド」
と「テスト本体」に分かれる．どちらもプロファイルと並列度で大きく変わる．

`[tool.maturin] profile = "py-ext"` が Python 拡張の既定 (定義は root
`Cargo.toml`)．明示ビルド (`maturin develop`) と `uv run` の暗黙リビルドが
同じプロファイルを使うので，最適化拡張が debug 拡張に差し替わることはない．

#### 実測値 (2026-08-09, 4 CPU / 16GB, crate registry 温)

コールドビルド:

| プロファイル | `jobs=1` | `jobs=4` | ピーク RSS |
|---|---|---|---|
| py-ext (既定) | 37分47秒 | **10分09秒** | 3.8GB / **7.2GB** |
| release | 30分41秒 | — | 3.7GB (jobs=1) |

反復ビルド (`rust/maou_shogi` を実際に内容変更):

| プロファイル | 再ビルド | 変更なし | `uv run` (変更なし) |
|---|---|---|---|
| py-ext (既定) | **6秒** | 3秒 | 0〜1秒 |
| release | 129〜133秒 | 3秒 | — |

テスト本体 (`pytest` 報告値, 1725 passed / 54 skipped):

| 拡張 | 時間 |
|---|---|
| py-ext (既定) | 92〜100秒 |
| release | 95.5秒 |
| dev (2026-08-09 以前の既定) | 191秒 |

`.so` サイズ: py-ext 56MB / release 50MB / dev 234MB．

#### 読み方

- **並列度が最大のレバー**．コールドビルドが `jobs=1` と `jobs=4` で
  **3.7倍**違い，プロファイル選択の影響を桁で上回る．
  `.cargo/config.toml` は `jobs` を設定していないので cargo 既定 (CPU 数)
  が効く．ただし `jobs=4` はピーク 7.2GB を要求するので，メモリ制約環境は
  `scripts/dev-init.sh` が user cargo config に `jobs = 1` を書いて絞る．
- **`--release` を付けると反復ビルドが 20倍以上遅くなる**．
  `lto = "thin"` は cdylib を再リンクするたびに依存グラフ全体の thin LTO を
  やり直し，**この工程はキャッシュできない**ため，再ビルドごとに約 127秒 の
  固定費が乗る．py-ext は `lto = false` なのでこれが無い．
  release 相当の数値が必要な性能計測を除き，`--release` は付けない．
- **テスト本体の測定は ±25% ばらつく**．他のビルドと並行させると 92秒の
  スイートが 135秒に見える．性能比較をするなら他の作業を止め，複数回測る．

#### pre-commit の `test` フック

`uv run pytest -v -s` を `always_run: true` で回すので
(`.pre-commit-config.yaml`)，**doc だけの変更でも**テストが走る．
拡張がコールドな新しいコンテナでは最初のコミットがビルド時間ごと待たされる
ので，先に

```bash
uv run python -c pass     # 拡張ビルドを温めるだけ
```

を一度流しておくと以後のコミットが速い．

#### 遅いのか壊れたのかの切り分け

`uv run` が返ってこないとき，この2つは見分けられる:

- **遅いだけ** — `Building maou @ file:///...` を出したあと，cargo の
  コンパイル中は**無出力で数分〜数十分黙る**．`pgrep -f rustc` で進行中か
  確認できる．
- **本当に失敗** — 1分程度で `hint:` / `help:` ブロックを出して終了する．
  例: `tensorrt-cu12-libs` の sdist ビルドは `nvidia-smi` を要求するため
  CPU only 環境では失敗しうる (2026-08-09 に一度観測，同日の再試行では
  再現しなかった)．

**無出力で長い = ビルド中，メッセージが出て止まる = 失敗**と読む．
待つ前に `pgrep -f rustc` を見るのが最短の切り分けになる．
