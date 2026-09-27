---
status: applied          # pending | approved | applied | rejected
applied_in: 97d4c64
date: 2026-09-27
target: [CLAUDE.md, docs/git-workflow.md]
risk: low
reversibility: trivial
---

# CLAUDE.md の探索ツール名を `Agent` に直し，コミット帰属ルールの矛盾を解消する

## Trigger

2026-09-27，`/claude-api prompt-audit` による設定ファイル監査で，durable doc に
属する 2 件が見つかった．`.claude/` と `AGENTS.md` の修正は同じ PR で適用済みだが，
この 2 件は `CLAUDE.md` / `docs/` が対象のため，承認待ちとしてここに提案する．

## 変更 1: `CLAUDE.md:172` のツール名 (drift correction)

```diff
-コードベースの調査・探索には，MUST use `Task` tool with `subagent_type=Explore`.
+コードベースの調査・探索には，MUST use the `Agent` tool with `subagent_type=Explore`.
```

Claude Code のサブエージェント起動ツールは現在 `Agent` という名前で
(`subagent_type` 引数は同じ)，`Task` という名前のツールは存在しない．
訂正後の本文は現行ツール一覧から一意に決まる．

## 変更 2: コミット／PR 帰属ルールの矛盾 (要判断)

次の 2 組が同じ点について逆のことを定めている:

- 禁止: `docs/git-workflow.md:29-30` (`❌ Co-authored-by trailers` / `❌ AI tool references`)，
  および `.claude/skills/pr-preparation-checks/SKILL.md:145-146`
- 必須: `.claude/skills/gh-pr/SKILL.md` § Attribution (required)，`AGENTS.md` § Attribution (required)

最近のコミットはトレーラーを持っている．**どちらを正とするかの判断が必要**:

- (a) 帰属を必須とする → `docs/git-workflow.md:29-30` と `pr-preparation-checks` の
  該当 2 行を削除する (推奨: 実際の履歴と一致する)
- (b) 帰属を禁止する → `gh-pr` と `AGENTS.md` の Attribution 節を書き換える

## 影響

どちらも指示ファイルだけの変更で，コードやテストには影響しない．
変更 2 が未解決のままだと，どちらのスキルを読み込んだかでエージェントの挙動が変わる．

## 決定

2026-09-27 にユーザが変更 1 を承認し，変更 2 は (b) 帰属を記載しない方向に決定した．
理由: コミット等のコード変更の本質に帰属は関係ない．`gh-pr` と `AGENTS.md` の
Attribution 節，`gh-pr` のチェックリストと例，`checkpoint-context` のトレーラー指示を
書き換えた．`docs/git-workflow.md` と `pr-preparation-checks` はすでに禁止しているため
変更していない．
