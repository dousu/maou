# Serena MCP Server導入によるトークン削減設計

## 概要

Claude Code利用時のトークン消費量を50%以上削減するため，oraias/serena MCPサーバーを導入する．シンボルレベルのコード解析により，ファイル全体の読み込みを回避し，必要な部分のみをコンテキストに含める．

## 背景・課題

- Clean Architecture（domain/app/interface/infra）の層を跨いだコード追跡でトークン消費が多い
- 特にinterfaceで定義されたプロトコルの実装（infra層）を探す際に，Grep + 複数ファイルReadが必要
- 1回の探索で500-2000トークンを消費

## 目標

- トークン消費量50%以上削減
- コード探索と編集の両方で効率改善
- Python/Rust両言語でシンボルレベル解析を実現

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    DevContainer                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Claude Code (CLI)                    │  │
│  │  - MCPクライアントとしてSerenaに接続              │  │
│  │  - シンボル検索をGrep/Globより優先使用            │  │
│  └──────────────┬───────────────────────────────────┘  │
│                 │ MCP Protocol (stdio)                  │
│  ┌──────────────▼───────────────────────────────────┐  │
│  │              Serena MCP Server                    │  │
│  │  - find_symbol, find_referencing_symbols         │  │
│  │  - insert_after_symbol, replace_symbol           │  │
│  │  - コードインデックス管理                         │  │
│  └──────┬─────────────────────────┬─────────────────┘  │
│         │ LSP Protocol            │ LSP Protocol        │
│  ┌──────▼──────┐           ┌──────▼──────┐             │
│  │   pylsp     │           │rust-analyzer│             │
│  │ (Python)    │           │   (Rust)    │             │
│  └──────┬──────┘           └──────┬──────┘             │
│         │                         │                     │
│  ┌──────▼─────────────────────────▼─────────────────┐  │
│  │          /workspaces/maou                         │  │
│  │  src/maou/ (Python)    rust/maou_io/ (Rust)      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 実装詳細

### 1. DevContainer設定

**変更ファイル: `.devcontainer/devcontainer.json`**

```jsonc
{
  "features": {
    // 既存のfeatureに追加
    "ghcr.io/devcontainers/features/rust:1": {}  // rust-analyzer用
  },
  "postCreateCommand": "bash scripts/dev-init.sh && bash scripts/setup-serena.sh",
  "postStartCommand": "bash scripts/start-serena.sh"
}
```

**新規ファイル: `scripts/setup-serena.sh`**

```bash
#!/bin/bash
# Serenaとpylspのインストール
uv tool install serena
uv tool install python-lsp-server[all]
# rust-analyzerはRust featureで自動インストール
```

**新規ファイル: `scripts/start-serena.sh`**

```bash
#!/bin/bash
# バックグラウンドでLSPサーバー起動（Serenaから呼び出される）
# Serena自体はClaude CodeからMCP経由で起動されるため
# ここでは事前準備のみ

# pylspのパスを環境変数に設定
export PYLSP_PATH=$(which pylsp)
export RUST_ANALYZER_PATH=$(which rust-analyzer)
```

### 2. Serena設定

**変更ファイル: `.claude/settings.local.json`**

```jsonc
{
  "permissions": {
    // 既存の設定を維持
  },
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": ["serena", "--project-root", "/workspaces/maou"],
      "env": {
        "SERENA_LSP_PYTHON": "pylsp",
        "SERENA_LSP_RUST": "rust-analyzer"
      }
    }
  }
}
```

**新規ファイル: `serena.toml`**（プロジェクトルート）

```toml
[project]
name = "maou"
root = "/workspaces/maou"

[languages.python]
enabled = true
lsp_command = "pylsp"
include = ["src/**/*.py", "tests/**/*.py"]
exclude = ["**/__pycache__/**", "**/.venv/**"]

[languages.rust]
enabled = true
lsp_command = "rust-analyzer"
include = ["rust/**/*.rs"]
exclude = ["rust/**/target/**"]

[indexing]
# 起動時に全ファイルをインデックス化
on_startup = true
# 変更検知で自動再インデックス
watch = true
```

### 3. LSP設定

**`pyproject.toml` への追記**

```toml
[tool.pylsp-mypy]
enabled = true
live_mode = false  # パフォーマンス優先

[tool.pylsp]
plugins.pyflakes.enabled = false  # ruffと重複回避
plugins.mccabe.enabled = false
plugins.pycodestyle.enabled = false
plugins.rope_autoimport.enabled = true  # インポート補完
```

**新規ファイル: `rust/maou_io/rust-analyzer.toml`**

```toml
[cargo]
buildScripts.enable = true

[check]
command = "clippy"

[imports]
granularity.group = "module"
prefix = "self"
```

### 4. Claude Code設定

**`CLAUDE.md` への追記**

```markdown
## MCP Server: Serena（トークン効率化）

コード探索時は以下のツール優先順位に従う:

### 探索時の優先順位
1. **Serena `find_symbol`** - クラス・関数・変数の定義を探す時
2. **Serena `find_referencing_symbols`** - シンボルの使用箇所を探す時
3. **Serena `get_symbol_definition`** - 特定シンボルの定義内容のみ取得
4. **Glob/Grep** - ファイルパターン検索やテキスト検索（シンボルでない場合）
5. **Read** - ファイル全体の理解が必要な場合のみ

### 編集時の優先順位
1. **Serena `replace_symbol`** - 関数・クラス単位の置換
2. **Serena `insert_after_symbol`** - シンボル直後への挿入
3. **Edit** - 行単位の細かい修正

### 使用例
- ❌ `StorageProtocol`の実装を探す → Grep + Read（トークン大）
- ✅ `StorageProtocol`の実装を探す → `find_referencing_symbols("StorageProtocol")`（トークン小）
```

## メモリ要件

| コンポーネント | メモリ使用量 |
|---------------|-------------|
| pylsp | 200-400MB |
| rust-analyzer | 300-500MB |
| Serena | 100-200MB |
| **合計** | **最大1.1GB追加** |

DevContainerのメモリ上限を4GB以上に設定すること．

## 期待効果

| 指標 | 従来 | 導入後 | 改善率 |
|------|------|--------|--------|
| 探索1回あたりトークン | 500-2000 | 50-200 | 70-90%削減 |
| プロトコル実装検索 | Grep + 複数Read | find_referencing_symbols | 大幅削減 |

## 運用

### 効果測定

```bash
# Claude Codeのトークン使用量を確認
# セッション終了時に表示される統計を記録

# 導入前後で同じ作業を実施して比較:
# 例: 「StorageProtocolの実装をすべて列挙して」
```

### トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| Serenaが起動しない | uvxパスが通っていない | `which uvx`確認，`uv tool install serena`再実行 |
| シンボルが見つからない | インデックス未完了 | 初回起動後30秒待つ，`serena.toml`のincludeパス確認 |
| pylspエラー | 仮想環境の認識失敗 | `VIRTUAL_ENV`環境変数を設定 |
| rust-analyzerエラー | Cargo.tomlが見つからない | `serena.toml`のrootパスを確認 |
| メモリ不足 | LSP + Serenaのメモリ使用 | DevContainerのメモリ上限を4GB以上に |

### メンテナンス

```bash
# Serenaアップデート
uv tool upgrade serena

# インデックス再構築（シンボルが古い場合）
# Claude Codeセッションを再起動
```

### ロールバック

問題が解決しない場合，`.claude/settings.local.json`から`mcpServers`セクションを削除すれば従来動作に戻る．

## 実装タスク

1. [x] `scripts/setup-serena.sh` 作成
2. [x] `scripts/start-serena.sh` 作成
3. [x] `.devcontainer/devcontainer.json` 更新
4. [x] `serena.toml` 作成
5. [x] `.claude/settings.local.json` にMCPサーバー追加
6. [x] `pyproject.toml` にpylsp設定追加
7. [x] `rust/maou_rust/rust-analyzer.toml` 作成
8. [x] `CLAUDE.md` にツール優先順位追記
9. [ ] DevContainer再構築・動作確認
10. [ ] 効果測定（トークン削減率の検証）

## 実装時の変更点

- **Serenaインストール方法**: PyPIではなくGitHubから直接インストール
  - `uv tool install git+https://github.com/oraios/serena`
  - MCP設定: `uvx --from git+https://github.com/oraios/serena serena start-mcp-server`

## 次のステップ

DevContainer再構築後，以下で効果を測定:

1. DevContainerを再構築
2. Claude Codeセッションを開始
3. 「StorageProtocolの実装をすべて列挙して」と依頼
4. Serenaツールが使用されることを確認
5. トークン使用量を記録
