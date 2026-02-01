# Code Exploration Policy Design

## Overview

Claude Codeがコードベースを探索する際に，必ずExploreサブエージェント（`Task` tool with `subagent_type=Explore`）を使用することを強制するポリシーを追加する．

## Problem

プラン実行中に予期せぬトラブルが発生した場合，Claude Codeが直接Grep/Glob/Readを連続実行してコード調査を行うことがある．これにより:

- コンテキストが肥大化する
- 非効率な探索が行われる
- Exploreサブエージェントの利点（トークン効率，並列検索）が活かされない

## Solution

CLAUDE.mdに新規セクション「Code Exploration Policy (MUST)」を追加し，コード探索時のExploreサブエージェント使用を強制する．

## Design

### Section Location

`## Critical Rules (MUST)` セクションの直後に配置

### Section Content

```markdown
## Code Exploration Policy (MUST)

コードベースの調査・探索には，MUST use `Task` tool with `subagent_type=Explore`.

### Covered Operations
- ファイル検索（Glob/Grep）を複数回行う調査
- 複数ファイルを読んでコードを理解する作業
- エラー原因の調査
- 実装方法を決めるための既存コード調査

### MUST NOT: Direct Multi-file Exploration
以下の操作を直接行うことを禁止:
- 調査目的での連続的なGrep/Glob実行
- 複数ファイルを順次Readして回る探索
- トラブルシューティング時のコード調査

### Exceptions (Direct Access Allowed)
1. **ユーザーが明示的にファイルパスを指定** - 「src/foo.pyを読んで」
2. **単一ファイルの特定行を確認** - エラーメッセージの「file:line」参照
3. **Exploreで特定済みファイルへのアクセス** - 既知の場所への編集

### Decision Criteria
- 「どこにあるか分からない」→ Explore必須
- 「このファイルのこの部分」→ 直接アクセス可
```

## Scope

- **適用範囲**: 会話全体を通じて常時適用
- **対象操作**: Grep/Glob検索 + 複数ファイル読み込み

## Exceptions

1. ユーザーが明示的にファイルパスを指定した場合
2. 単一ファイルの特定行を確認する場合
3. Exploreで既に特定済みのファイルへのアクセス

## Implementation

CLAUDE.mdの `## Critical Rules (MUST)` セクションの直後に上記セクションを追加する．
