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
│  │  pyright    │           │rust-analyzer│             │
│  │ (Python)    │           │   (Rust)    │             │
│  │ ※Serena内蔵 │           │             │             │
│  └──────┬──────┘           └──────┬──────┘             │
│         │                         │                     │
│  ┌──────▼─────────────────────────▼─────────────────┐  │
│  │          /workspaces/maou                         │  │
│  │  src/maou/ (Python)    rust/maou_io/ (Rust)      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**注記**: SerenaはPython用にpyright-langserverを内蔵しており，外部のpylspは使用しない．

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
# Serenaのインストール
uv tool install git+https://github.com/oraios/serena
# rust-analyzerはrustup経由でdev-init.shでインストール
# pyrightはSerenaに内蔵されているため別途インストール不要
```

**新規ファイル: `scripts/start-serena.sh`**

```bash
#!/bin/bash
# バックグラウンドでLSPサーバー起動（Serenaから呼び出される）
# Serena自体はClaude CodeからMCP経由で起動されるため
# ここでは事前準備のみ

export RUST_ANALYZER_PATH=$(which rust-analyzer)
```

### 2. Serena設定

**MCP設定: `.mcp.json`**

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--project", "/workspaces/maou"],
      "env": {
        "SERENA_LOG_LEVEL": "info"
      }
    }
  }
}
```

**Serena設定: `.serena/project.yml`**

Serenaは`serena.toml`ではなく`.serena/project.yml`を使用する．

```yaml
languages:
- python  # pyright-langserver（Serena内蔵）
- rust    # rust-analyzer

ignored_paths:
- "**/target/**"
- "**/__pycache__/**"
- "**/.venv/**"
- "**/node_modules/**"
```

### 3. rust-analyzer設定

**新規ファイル: `rust/maou_rust/rust-analyzer.toml`**

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
| pyright（Serena内蔵） | 200-400MB |
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
| Serenaが起動しない | uvxパスが通っていない | `which uvx`確認，`uv tool install git+https://github.com/oraios/serena`再実行 |
| シンボルが見つからない | インデックス未完了 | 初回起動後30秒待つ，`.serena/project.yml`の設定確認 |
| pyrightエラー | Serena内部エラー | Serenaのログ確認，Claude Codeセッション再起動 |
| rust-analyzerエラー | Cargo.tomlが見つからない | `.serena/project.yml`のlanguagesにrustが含まれているか確認 |
| メモリ不足 | LSP + Serenaのメモリ使用 | DevContainerのメモリ上限を4GB以上に |

### メンテナンス

```bash
# Serenaアップデート
uv tool upgrade git+https://github.com/oraios/serena

# インデックス再構築（シンボルが古い場合）
# Claude Codeセッションを再起動
```

### ロールバック

問題が解決しない場合，`.mcp.json`から`serena`セクションを削除すれば従来動作に戻る．

## 実装タスク

1. [x] `scripts/setup-serena.sh` 作成
2. [x] `scripts/start-serena.sh` 作成
3. [x] `.devcontainer/devcontainer.json` 更新
4. [x] `.serena/project.yml` 設定（`serena.toml`は不使用）
5. [x] `.mcp.json` にMCPサーバー追加
6. [x] ~~`pyproject.toml` にpylsp設定追加~~ → 不要（pyrightはSerena内蔵）
7. [x] `rust/maou_rust/rust-analyzer.toml` 作成
8. [x] `CLAUDE.md` にツール優先順位追記
9. [x] DevContainer再構築・動作確認
10. [x] 効果測定（トークン削減率の検証）

## 検証結果 (2026-01-26)

### Python シンボル解析 ✅

**使用LSP**: pyright-langserver 1.1.408（Serena内蔵）

| 機能 | 結果 | 備考 |
|------|------|------|
| `find_symbol` | ✅ 動作 | CloudStorageクラスを正しく検出 |
| `find_referencing_symbols` | ✅ 動作 | GCS/S3実装と使用箇所を全検出 |
| `get_symbols_overview` | ✅ 動作 | ファイル内シンボル一覧を取得 |

### Rust シンボル解析 ✅

**使用LSP**: rust-analyzer 1.92.0

| 機能 | 結果 | 備考 |
|------|------|------|
| `find_symbol` | ✅ 動作 | `save_feather`関数を正しく検出 |
| `get_symbols_overview` | ✅ 動作 | 関数・テストモジュールを全検出 |
| `search_for_pattern` | ✅ 動作 | テキスト検索として機能 |

### トークン削減効果

**従来のアプローチ**:
```
Grep "CloudStorage" → 複数ファイルをRead → 手動で実装クラスを特定
推定トークン: 1000-2000
```

**Serenaアプローチ**:
```
find_referencing_symbols("CloudStorage")
→ GCS, S3実装クラス + 全使用箇所を1回で取得
推定トークン: 200-400
```

**削減率: 約70-80%**

### 設定変更点

- `.serena/project.yml`がSerenaの実際の設定ファイル
- `languages`リストに`python`と`rust`を指定
- `ignored_paths`でビルド成果物を除外
- **pylspは不使用** - Serenaはpyright-langserverを内蔵

## 実装時の変更点

- **Serenaインストール方法**: PyPIではなくGitHubから直接インストール
  - `uv tool install git+https://github.com/oraios/serena`
  - MCP設定: `uvx --from git+https://github.com/oraios/serena serena start-mcp-server`
- **Python LSP**: pylspではなくpyright（Serena内蔵）を使用
- **設定ファイル**: `serena.toml`ではなく`.serena/project.yml`を使用
