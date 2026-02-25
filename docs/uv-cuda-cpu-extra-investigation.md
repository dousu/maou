# uv sync --extra cpu で CUDA パッケージがインストールされる問題の調査

## 概要

`uv sync --extra cpu` を実行すると，`nvidia-cusolver-cu12`，`nvidia-nccl-cu12` 等の CUDA 関連パッケージがインストールされる問題を調査した．

## コードベースからわかったこと

### pyproject.toml の設定は正しい

`pyproject.toml` では，GPU 種類ごとの extra が正しく分離されている:

```toml
[project.optional-dependencies]
cpu = [
    "torch>=2.6.0,<=2.7.0",
    "torchinfo>=1.8.0",
    "torch-tb-profiler>=0.4.3",
    # ... (CUDA 依存なし)
]
cuda = [
    "torch>=2.6.0,<=2.7.0",
    "torchinfo>=1.8.0",
    "torch-tb-profiler>=0.4.3",
    "nvidia-ml-py>=11.4.1",
    # ...
]
```

conflicts 定義で cpu / cuda / mpu / tpu を排他指定:

```toml
[tool.uv]
conflicts = [
  [
    { extra = "cpu" },
    { extra = "cuda" },
    { extra = "mpu" },
    { extra = "tpu" },
  ],
]
```

sources 定義で extra ごとに PyTorch インデックスを分離:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cuda", extra = "cuda" },
  { index = "pytorch-cuda", extra = "mpu" },
  { index = "pytorch-cuda", extra = "tpu" },
]
```

### uv.lock の resolution-markers に問題がある

`uv.lock` ヘッダーのグローバル resolution-markers は extra 条件を正しく含んでいる:

```toml
resolution-markers = [
    # tpu fork
    "python_full_version >= '3.12' and extra != 'extra-4-maou-cpu' and ... and extra == 'extra-4-maou-tpu'",
    # cuda fork
    "python_full_version >= '3.12' and ... and extra == 'extra-4-maou-cuda' and ...",
    # cpu fork (non-darwin)
    "python_full_version >= '3.12' and sys_platform != 'darwin' and extra == 'extra-4-maou-cpu' and ...",
    # cpu fork (darwin)
    "python_full_version >= '3.12' and sys_platform == 'darwin' and extra == 'extra-4-maou-cpu' and ...",
    # extra なし fork
    "python_full_version >= '3.12' and extra != 'extra-4-maou-cpu' and extra != 'extra-4-maou-cuda' and ...",
    # ... (各 python_full_version < '3.12' バリアント)
]
```

しかし，**個別パッケージの resolution-markers に extra 条件が欠落**している:

| パッケージ | resolution-markers | 問題 |
|---|---|---|
| `torch 2.7.0` (macOS CPU) | `sys_platform == 'darwin'` | darwin のみ(正常) |
| `torch 2.7.0+cpu` | `sys_platform != 'darwin'` | non-darwin のみ(正常) |
| `torch 2.7.0+cu128` | `python_full_version >= '3.12'` + `python_full_version < '3.12'` | **全 Python バージョン・全プラットフォームに一致(問題)** |

`torch 2.7.0+cu128` の resolution-markers は本来 `extra == 'extra-4-maou-cuda'` 等の条件で cuda / mpu / tpu fork のみに制限されるべきだが，無条件マーカーになっている．これにより，cpu fork でも `torch+cu128` が解決グラフに含まれ，その依存関係である `nvidia-*` パッケージ群がインストールされる．

### torchinfo / torch-tb-profiler は無関係

当初，推移的依存(torchinfo や torch-tb-profiler が torch に依存し，その解決で CUDA 版が引かれる)を疑ったが，`uv.lock` を確認した結果:

- **torchinfo**: 依存関係なし(pure Python パッケージ)
- **torch-tb-profiler**: `pandas` と `tensorboard` のみに依存(`torch` を依存として宣言していない)

したがって，これらのパッケージを `[tool.uv.sources]` に追加しても効果はない．

### NVIDIA パッケージの依存元は torch+cu128 のみ

`nvidia-cusolver-cu12` と `nvidia-nccl-cu12` を依存関係として宣言しているのは `torch 2.7.0+cu128` のみであることを確認した:

```toml
# uv.lock: torch 2.7.0+cu128 の依存
dependencies = [
    { name = "nvidia-cublas-cu12", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
    { name = "nvidia-cusolver-cu12", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
    { name = "nvidia-nccl-cu12", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
    # ... (合計 14 の nvidia-* パッケージ)
]
```

## Web 調査からわかったこと

### uv の既知の問題群

PyTorch のカスタムインデックス + conflicts + extras の組み合わせは uv で多数の問題が報告されている:

| Issue | 内容 | 状態 |
|---|---|---|
| [#9640](https://github.com/astral-sh/uv/issues/9640) | `uv sync --extra cpu` で CUDA 版 torch も解決される | **Closed (Fixed)** PR [#9370](https://github.com/astral-sh/uv/pull/9370) で conflict markers をロックファイルに追加 |
| [#9296](https://github.com/astral-sh/uv/issues/9296) | resolution-markers が重複する | **Closed (Fixed)** PR #9780 で重複排除 |
| [#9734](https://github.com/astral-sh/uv/issues/9734) | `accelerate` 等の torch 依存パッケージ追加で CPU/GPU 両方インストールされる | Open |
| [#12290](https://github.com/astral-sh/uv/issues/12290) | `uv sync --extra cpu` 後の `uv run` で NVIDIA パッケージがインストールされる | Closed (Not Planned) |
| [#16522](https://github.com/astral-sh/uv/issues/16522) | torch 以外のエコシステムパッケージ(xformers 等)にも自動バリアント選択を適用してほしい | Open (Feature Request) |
| [#16368](https://github.com/astral-sh/uv/issues/16368) | PyPI / torch backend 間の extra 切り替えが正しく動作しない | Open |
| [#11498](https://github.com/astral-sh/uv/issues/11498) | `uv sync` で environment markers が無視される | Open |

### 修正の経緯と現状

最も関連する [#9640](https://github.com/astral-sh/uv/issues/9640) は 2024年12月に PR [#9370](https://github.com/astral-sh/uv/pull/9370) で修正済みとされている．この修正では，ロックファイルに「conflict markers」を導入し，extra に基づく依存関係の分離を実現した．

実際に，本プロジェクトの `uv.lock` ヘッダー(グローバル resolution-markers)には extra 条件が含まれており，修正の効果が確認できる．

しかし，**個別パッケージの resolution-markers には extra 条件が含まれていない**(#9296 の修正で「conflict markers は別の場所で管理されるため重複を排除」としたため)．この設計が正しく機能しているかどうかが，今回の問題の核心である．

### 根本的な制約

uv は PEP 508 の environment markers をベースにパッケージ解決を行うが，「extra」は PEP 508 の仕様上 environment marker ではなく dependency marker であるため，パッケージレベルの resolution-markers に extra 条件を自然に含めることが難しい．これが PyTorch の CPU/CUDA バリアント問題の根本的な原因となっている．

## ワークアラウンドの比較

### 1. uv.lock の再生成

```bash
rm uv.lock
uv lock
```

| 項目 | 評価 |
|---|---|
| 手軽さ | 高 |
| 効果の確実性 | 中(uv のバージョン次第) |
| 副作用 | 他パッケージのバージョンが変わる可能性あり |
| 推奨度 | まず最初に試すべき |

uv のバージョンアップにより修正が取り込まれている可能性がある．現在 uv 0.8.17 を使用中だが，最新版に更新してからの再生成が望ましい．

### 2. uv run 時に --extra を常に明示

```bash
# uv run 時に extra を指定しないと再 sync が走り CUDA が入る
uv run --extra cpu python train.py

# または --no-sync で再 sync を防止
uv run --no-sync python train.py
```

| 項目 | 評価 |
|---|---|
| 手軽さ | 中(毎回指定が必要) |
| 効果の確実性 | 高 |
| 副作用 | なし |
| 推奨度 | #12290 で報告された `uv run` 起因の問題への対策として有効 |

### 3. 環境変数 UV_NO_SYNC=true の設定

```bash
export UV_NO_SYNC=true
uv run python train.py
```

| 項目 | 評価 |
|---|---|
| 手軽さ | 高(一度設定すれば済む) |
| 効果の確実性 | 高 |
| 副作用 | 依存関係の変更が自動反映されなくなる |
| 推奨度 | CI/CD 環境や固定環境では有効 |

### 4. torch 依存パッケージを [tool.uv.sources] に追加

```toml
torchinfo = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cuda", extra = "cuda" },
]
```

| 項目 | 評価 |
|---|---|
| 手軽さ | 中 |
| 効果の確実性 | **低**(本調査で torchinfo / torch-tb-profiler は torch を依存として宣言していないことが判明) |
| 副作用 | これらのパッケージが PyTorch インデックスに存在しない場合 `uv lock` が失敗する |
| 推奨度 | **非推奨**(根本原因が推移的依存ではないため効果なし) |

### 5. uv の新バージョンを待つ

| 項目 | 評価 |
|---|---|
| 手軽さ | 高(待つだけ) |
| 効果の確実性 | 不明 |
| 副作用 | なし |
| 推奨度 | 長期的には [WheelNext](https://github.com/astral-sh/uv/issues/16522) による根本解決が期待される |

## 推奨アクション

1. **まず uv を最新版に更新し `uv.lock` を再生成する**(ワークアラウンド 1)
2. **再生成後も問題が再現する場合，`uv run --no-sync` または `UV_NO_SYNC=true` で回避する**(ワークアラウンド 2, 3)
3. **問題が再現する場合は [astral-sh/uv](https://github.com/astral-sh/uv/issues) に issue を報告する**(#9640 は Fixed とされているが，同じ症状がリグレッションしている可能性がある)

## 参考資料

- [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) - uv 公式の PyTorch 統合ガイド
- [#9640 - Inconsistent or incorrect package resolution for PyTorch when using environment markers](https://github.com/astral-sh/uv/issues/9640) - 本問題に最も近い報告(Fixed)
- [#9370 - Add conflict markers to the lock file](https://github.com/astral-sh/uv/pull/9370) - #9640 の修正 PR
- [#9296 - Repeated markers in resolution-markers](https://github.com/astral-sh/uv/issues/9296) - resolution-markers 重複問題(Fixed)
- [#9734 - Support PyTorch CPU and GPU dependency](https://github.com/astral-sh/uv/issues/9734) - 推移的依存で CPU/GPU 両方入る問題(Open)
- [#12290 - Multi-Accelerator torch setup and unexpected uv run behaviour](https://github.com/astral-sh/uv/issues/12290) - `uv run` 時の再 sync 問題(Closed, Not Planned)
- [#16522 - Automatic CPU/GPU wheel variant selection for all packages](https://github.com/astral-sh/uv/issues/16522) - WheelNext 提案(Open)
- [#16368 - Switching torch extra between pypi & torch backend](https://github.com/astral-sh/uv/issues/16368) - PyPI/torch backend 切り替え問題(Open)
- [#11498 - uv sync: environment markers ignored when extra is given](https://github.com/astral-sh/uv/issues/11498) - environment markers 無視問題(Open)

## 解決

### 原因

旧 `uv.lock` (uv 0.8.17 で生成)では，個別パッケージの依存 marker に以下のような条件が含まれていた:

```toml
{ name = "nvidia-cuda-runtime-cu12", marker = "sys_platform == 'linux' or sys_platform == 'win32' or (extra == 'extra-4-maou-cpu' and extra == 'extra-4-maou-cuda') or ..." }
```

`sys_platform == 'linux'` が `or` で結合されているため，Linux 環境では extra の値に関係なく常に true と評価され，cpu extra でも CUDA パッケージが解決対象に含まれていた．

### 修正方法

uv 0.10.4 で `uv.lock` を再生成(`rm uv.lock && uv lock`)することで解決した．新しい lock ファイルでは marker が正しく生成され，`uv sync --extra cpu` で CUDA パッケージがインストールされなくなった．

### 検証結果

- `uv sync --extra cpu`: torch 2.7.0+cpu のみインストール．nvidia-* パッケージなし
- `uv sync --extra cuda --dry-run`: nvidia-* パッケージ 16 個が正しくインストール対象に含まれる

## 調査環境

- uv (修正前): 0.8.17
- uv (修正後): 0.10.4
- Python: >=3.11, <3.13
- torch: 2.7.0 (cpu) / 2.7.0+cu128 (cuda)
- 初回調査日: 2025-02-25
- 解決日: 2026-02-25
