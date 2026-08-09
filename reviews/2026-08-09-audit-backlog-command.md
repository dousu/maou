---
status: pending
applied_in:
date: 2026-08-09
target:
  - CLAUDE.md
  - audits/README.md
risk: medium
reversibility: moderate
---

# /audit-backlog: deferred / out-of-scope 消化を正常ルートにする

## Trigger

`.claude/commands/audit-backlog.md` を新設した (コマンドファイル自体は
`CLAUDE.md` / `docs/` 配下でないため本提案の対象外)．これに伴い
2つの committed durable doc が実態と合わなくなる．

きっかけは今回の Tier A 消化作業で，`audits/` の運用に3つのギャップが
見つかったこと．

1. **deferred 項目に到達経路がない**．`coverage.md` は「11 deferred」と
   **件数だけ**を持ち，項目本体は record ファイルの中にしかない．
   `audits/README.md` は out-of-scope backlog について
   「a per-run record is read only when someone opens that specific path —
   so a finding filed there is visible exactly to the audit least able to
   act on it」と正しく論じているが，**deferred 項目はまさにその状態に
   置かれたまま**である．out-of-scope を救出した論理が deferred には
   適用されていない．

2. **record の immutability が実運用と矛盾している**．
   `audits/README.md` は record を「Immutable once the run's status is
   `done`」とするが，実際には `916e874`
   (「非有限損失ガード統合を Applied へ移動 (deferred 12→11)」) で
   `done` record の Deferred 項目が Applied に移されている．
   ルールと実務のどちらかを直す必要がある．

   **これが 1 と同根である**ことが，本提案の初版をユーザレビューに
   かけて判明した．初版は deferred 項目を record 内に残したまま
   「RESOLVED 注釈を追記する」方式を提案していたが，これは誤検出を
   構造的に防げない: record は消せないので，消化済み項目が毎回 worklist
   として再浮上する．注釈をフィルタとして使う手はあるが，
   フィルタの有無に正しさが依存する設計になる．
   out-of-scope が誤検出しないのは **coverage.md から行を削除できる**
   からであり，deferred も同じ構造にすれば注釈機構ごと不要になる．
   本提案はその方針に改訂済み (§4, §5)．

3. **backlog 消化の記録先が未定義**．個別 finding を消化した run は
   `audits/` に何を書くべきか決まっていない．メインテーブルに `done` 行を
   書くと未監査パスを監査済みと主張してしまう (今回この判断が必要になった)．

## Proposed change

### 1. CLAUDE.md — Files テーブル (現 84-88行付近)

`audits/YYYY-MM-DD-<path-slug>.md` の行を，consumption record も
表せるよう改める．

**Before**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. | yes |
```

**After**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. | yes |
| `audits/YYYY-MM-DD-backlog-<slug>.md` | One record per `/audit-backlog` run (`kind: backlog`). Consumes individual deferred / out-of-scope findings; does **not** earn a main-table row. | yes |
```

### 2. CLAUDE.md — Files テーブル (commands)

**Before**

```markdown
| `.claude/commands/audit-and-fix.md` | Writer of `audits/`. | yes |
```

**After**

```markdown
| `.claude/commands/audit-and-fix.md` | Writer of `audits/` (path 単位の監査). | yes |
| `.claude/commands/audit-backlog.md` | Writer of `audits/` (deferred / out-of-scope の個別消化). | yes |
```

### 3. CLAUDE.md — MUST rules (現 129行付近)

**Before**

```markdown
- MUST record every `/audit-and-fix` run in `audits/` (ledger row +
  record file) and commit it — ...
```

**After**

```markdown
- MUST record every `/audit-and-fix` **and `/audit-backlog`** run in
  `audits/` and commit it — `/audit-and-fix` writes a ledger row + record
  file; `/audit-backlog` writes a `kind: backlog` record and MUST NOT
  write a main-table row (a row there claims the whole path was audited).
  `audits/` is the only cross-session record of repo-wide audit coverage,
  and unlike `scratchpad/` it survives container reclamation. An
  interrupted run MUST still write its resume point before the session
  ends.
- MUST treat `audits/coverage.md` § "Open findings backlog" as the **only**
  live worklist of open findings, deferred and out-of-scope alike:
  `/audit-and-fix` MUST append a row for every finding it leaves open, and
  a run that resolves one MUST delete its row. MUST NOT read a record's
  `## Deferred` / `## Out of scope` section to decide what work remains —
  a record is an immutable account whose Deferred section stays true after
  the finding ships, so treating it as a worklist re-surfaces resolved
  findings forever. MUST NOT amend a record to carry state (no `RESOLVED`
  markers, no Deferred→Applied moves, no renumbering); a **correction** of
  a wrong diagnosis is the only permitted amendment.
```

### 4. audits/README.md — deferred 項目の到達性 (新規小節)

§ "The out-of-scope backlog" の直後に追加:

```markdown
## Deferred findings

A record's `## Deferred` section holds findings the audit **confirmed but
deliberately did not fix** — ambiguous, cross-layer, or needing a
decision. A deferred finding is a diagnosis with the fix withheld pending
a decision, **not** a decision never to fix it.

Deferred findings therefore get a row in `coverage.md`'s **Deferred
backlog**, exactly as out-of-scope findings get one in the Out-of-scope
backlog. The retrieval argument above applies to both classes in full:
what is written only into a record is visible only to whoever opens that
record.

`coverage.md` is the authority on **what is open**; records are the
authority on **what happened**. The row is the condensed, deletable index
entry; the record's Deferred section is the durable reasoning behind it.
Both are written, and only the row is ever deleted.
```

### 5. audits/README.md — records are worklist-free

**Before**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. Immutable once the run's status is `done`. |
```

**After**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. Immutable once the run's status is `done` — it is an account, never a worklist (see below). |
| `audits/YYYY-MM-DD-backlog-<slug>.md` | One record per `/audit-backlog` run (`kind: backlog`). Consumes individual findings; gets no main-table row. |
```

そして § "Status vocabulary" の直前に:

```markdown
## Records are accounts, not worklists

A `done` record is the account of one run at one time. Its Deferred
section says "as of that run, this was deferred" — and that stays true
forever, **including after the finding has shipped**.

That is why no command reads a record to decide what work remains. Doing
so would re-surface every resolved finding on every run, with no way to
remove it: a record cannot be "cleared" without destroying the account,
so the list would only ever grow. Deleting a row from `coverage.md` is
what marks a finding consumed, and it is the only mechanism that does.

So a record is **never amended to carry state**: no `RESOLVED` markers,
no moving an item from Deferred into Applied, no renumbering. Commit
`916e874` did move a deferred item into Applied — that predates the
Deferred backlog, when the record was the only place to record it, and it
is not the pattern to follow now.

The one narrow exception is a **correction**: when a later run proves a
record's diagnosis or proposed fix *wrong*, append a short note saying
so, because an uncorrected record actively misleads the next reader. A
correction states what the record got wrong, never whether the work is
done:

```markdown
   **Correction** (YYYY-MM-DD, `<sha>`): the fix suggested above would
   have <consequence>, because <what the record missed>.
```
```

## Motivation

`audits/` は container reclamation を越える唯一の監査記録なので，
「記録されたが誰も消化しない finding」が溜まると，ledger が
**やったことの記録** としては正しいまま **やるべきことの所在** としては
無価値になる．今回 22件 (deferred 13 + out-of-scope 9) が溜まっており，
うち deferred 13件は record を開かない限り誰の視界にも入らなかった．

immutability の矛盾を放置すると，次のセッションは `916e874` の前例と
README の文面のどちらに従うかを毎回判断し直すことになる．

そして「消化済みを表現する手段」を record 側に持たせようとすると必ず
誤検出が残る．record は削除できない (削除は account の破壊) ので，
worklist としての状態は**削除できる場所**に置くしかない．
それが `coverage.md` の行である．

## Alternatives considered

- **`/audit-and-fix` に消化モードを足す**．却下: あのコマンドは既に505行
  あり，スコープが「1パス」であることが resumability の根拠になっている．
  個別 finding 消化はスコープの定義が違う (パスを横断する) ので，
  同じコマンドに入れると step 0 の path 解決が二重化する．
- **record を glob して Deferred を読み，RESOLVED 注釈で消化済みを
  マークする** (本提案の初版)．**却下**: record は削除できないので，
  消化済み項目が毎回 worklist として再浮上する．注釈をフィルタに使えば
  回避できるが，「フィルタが正しく効いているか」に設計の正しさが依存し，
  フィルタを書き忘れた初版は実際に誤検出する仕様になっていた．
  行削除なら誤検出が**構造的に**起こりえない．
  さらに注釈方式は `/audit-and-fix` step 0c (「re-audit 前に record の
  Deferred を読む」) にも同じ問題を残す — 誤検出が別コマンドへ移るだけ．
- **deferred は record にだけ置き，転記しない** (現状)．却下: 13件が
  record を開かない限り不可視で，実際に誰も消化していなかった．
- **消化 run にもメインテーブル行を書く**．却下: 未監査パスを
  監査済みと主張することになる (本提案 3 の理由)．

## What this enables

- deferred / out-of-scope が「溜まる一方」でなくなる — 消化に正常ルートが
  でき，優先度提示 → ユーザ選択 → 修正 → 行削除 + 記録 が1コマンドで回る．
- **誤検出が構造的に起こりえない** — 消化 = 行削除なので，消化済み項目が
  worklist に再浮上する経路が存在しない．`coverage.md` = 何が開いているか，
  record = 何が起きたか，と権威が1つずつに分かれる．
- record が真に immutable になる (correction のみ例外)．
  `916e874` のような state 書き換えが不要になる．

## What this constrains

- `/audit-and-fix` は open finding を **必ず** `coverage.md` に append
  しなければならない (record だけに書くと不可視)．コマンド側 step 9 を
  改訂済み．これは追加の義務だが，deferred が失われる唯一の経路を閉じる．
- `coverage.md` が長くなる (現在 deferred 13行 + out-of-scope 6行)．
  ただし消化すれば縮む — それが設計意図であり，record と違って縮められる．
- 行と record の二重記述になる．行は条約された index，record は durable な
  理由づけ，という役割分担で許容する (out-of-scope で既に確立した形)．
- consumption record が増えるので `audits/` のファイル数は増える．
  ledger のメインテーブルは増えない．

## Rollback plan

いずれも文面のみ．`CLAUDE.md` は 3箇所，`audits/README.md` は 2箇所の
差分なので revert は容易．コマンドファイルを消せば参照も消える
(コマンド不使用でも既存の `/audit-and-fix` 運用は影響を受けない)．
