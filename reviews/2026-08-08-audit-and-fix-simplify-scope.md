---
status: applied
applied_in: ""
date: 2026-08-08
target:
  - .claude/commands/audit-and-fix.md
risk: low
reversibility: trivial (single-section revert)
---

# `/audit-and-fix` step 2 は path 全体ではなく diff を simplify していた

## Trigger

`/audit-and-fix src/maou/app/learning high` の実行中，ユーザーから
「audit-and-fix では code-review で変更があった部分だけ simplify して
いるのではないか．実装意図としては path 範囲すべての simplify である」
と指摘を受けた．調査の結果，指摘は正しかった．

## Motivation

`/simplify` はこのリポジトリの資産ではない (`.claude/commands/` には
`audit-and-fix.md` / `checkpoint-context.md` / `resume-context.md` の
3 つしかなく，`/simplify` は組み込みコマンド)．したがってその既定
挙動をリポジトリ側から変えることはできず，`audit-and-fix.md` の側で
明示的に上書きするしかない．

`/simplify` の既定スコープは**現在の diff** である:

- Phase 0: 「Run `git diff @{upstream}...HEAD` ... **Treat this diff as
  the review scope.**」
- 4 つのレビュー観点のうち 3 つが diff 前提の文言:
  - Reuse: 「Flag **new code** that re-implements ...」
  - Simplification: 「Flag unnecessary complexity **the diff adds**」
  - Efficiency: 「Flag wasted work **the diff introduces**」

Phase 0 には「If a PR number, branch name, or file path was passed as an
argument, review that target instead」という節もあるが，観点側の文言が
diff 前提のままなので，path を渡してもレビューは変更行に引き戻される．

そして `/audit-and-fix` の step 2 は step 1 の**直後**に走る．step 1 は
バグ修正を適用済みなので，作業ツリーには必ず diff がある．結果として
step 2 は「step 1 が今さっき直した行」だけを見ることになる — path 中で
最も小さく，かつ数分前にレビューされたばかりの部分である．

audit の目的は逆で，**誰も見直していない蓄積した複雑さ**を見つけること
にある．それは定義上この run が触っていないファイルにある．

## Proposed change

`.claude/commands/audit-and-fix.md` の step 2 を差し替える．

### Before

```markdown
### 2. Simplification cleanup

Run `/simplify <path>` for the quality-only cleanups (reuse,
simplification, efficiency, altitude) that step 1 did not already fix.
Quality-only means lower risk, so this one auto-applies.
```

### After

```markdown
### 2. Simplification cleanup — the whole path, never the diff

Run `/simplify <path>` for the quality-only cleanups (reuse,
simplification, efficiency, altitude) that step 1 did not already fix.
Quality-only means lower risk, so this one auto-applies.

**Scope is `<path>` in full — every file under it, changed or not.**

This needs stating because `/simplify` is a general-purpose command whose
*default* scope is the current diff: its own instructions say "treat this
diff as the review scope", and three of its four angles are worded around
what "the diff adds", "the diff introduces", or what is "new code".
Passing a path is meant to override that, but the angle wording pulls the
review back toward changed lines anyway.

Left at its default here it reviews **whatever step 1 just committed** —
the smallest slice of the path, and the one slice already reviewed
minutes earlier. That inverts the point of the step. A path is audited to
find accumulated complexity nobody has revisited, which by definition
lives in the files this run did *not* touch.

So when handing the work to review agents, override the framing
explicitly: name `<path>`, tell each agent to read the files under it,
and phrase its angle as "existing code in this module" rather than "new
code" or "the diff". If a review returns citing only lines this run
changed, its scope was wrong — re-run it with the path restated.

The same caution applies to `/code-review <path> <level>` in step 1: it
also falls back to a branch/working-tree diff when one exists, which on a
**resumed** audit is this run's own earlier commits. Check its scope note
names `<path>`, not a commit range, before trusting its findings as a
path-wide bug hunt.

Record in step 9 which files the pass actually covered. "Ran /simplify"
is not evidence the path was covered; a run that reviewed only the step 1
diff must be reported as **not done**, not as a clean pass.
```

## Alternatives considered

1. **`.claude/commands/simplify.md` としてリポジトリ内に `/simplify` を
   フォークする．** スコープ問題を根本から断てるが，組み込み版の改良を
   一切受け取れなくなる保守負債を負う．step 2 が 4 行の上書きで済む
   うちは割に合わない．
2. **step 2 から `/simplify` を外し，4 観点を `audit-and-fix.md` に
   直書きする．** 同じ保守負債をより悪い形で負う (観点の文面が 2 か所に
   分岐する)．
3. **何もせず，実行のたびに口頭で上書きする．** 実際に起きたのがこれで，
   ユーザーの指摘がなければ気付かれなかった．コマンドは「前の run の
   記憶がない」前提で書かれているのだから，run 側の裁量に頼るのは
   この文書の設計と矛盾する．

## What this enables

- step 2 が本来の意図どおり path 全体を対象にする．
- step 1 の `/code-review` にも同じ罠 (resumed audit で自分の過去
   コミットを diff として拾う) があることを明文化した．
- step 9 に「実際に covered したファイルを記録せよ」という検証可能な
   義務を追加したので，スコープ事故が記録に残る．「/simplify を実行した」
   ことと「path を covered した」ことを区別する．

## What this constrains

- step 2 が高くつく．path 全体 (本 run では 11,544 行) を読むレビューは
  diff レビューより明確に重い．level が `low` の場合でも path 全体が
  対象である点は変わらない — 軽くするのは観点の深さであって範囲ではない．
- 組み込み `/simplify` の Phase 0 の文言が将来変わると，この節の引用が
  陳腐化する．引用ではなく「既定は diff スコープなので path を明示せよ」
  という規則として書いてあるので，文言が変わっても規則は生き残る．

## Rollback plan

step 2 を Before の 4 行に戻す．他のどの節にも依存していない単一
セクションの差し替えなので，revert は自明．
