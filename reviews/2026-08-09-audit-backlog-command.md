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
  file; `/audit-backlog` writes a `kind: backlog` record, deletes the
  backlog rows it resolved, and annotates the source record's Deferred
  item, but MUST NOT write a main-table row (a row there claims the whole
  path was audited). `audits/` is the only cross-session record of
  repo-wide audit coverage, and unlike `scratchpad/` it survives container
  reclamation. An interrupted run MUST still write its resume point before
  the session ends.
```

### 4. audits/README.md — deferred 項目の到達性 (新規小節)

§ "The out-of-scope backlog" の直後に追加:

```markdown
## Deferred findings

A record's `## Deferred` section holds findings the audit **confirmed but
deliberately did not fix** — ambiguous, cross-layer, or needing a
decision. `coverage.md` carries only their *count* per row, so unlike
out-of-scope findings they are reachable only by opening the specific
record.

That makes them the more neglected class, not the less important one: the
retrieval argument above ("a per-run record is read only when someone
opens that specific path") applies to them in full. `/audit-backlog`
exists to close that gap — it globs `audits/*.md` and reads every
Deferred section, rather than waiting for someone to audit that path
again.

A deferred item is therefore **not** a decision to never fix it. It is a
diagnosis with the fix withheld pending a decision.
```

### 5. audits/README.md — immutability の明確化

**Before**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. Immutable once the run's status is `done`. |
```

**After**

```markdown
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. The account of the run is immutable once `done` — but a resolved Deferred item is **annotated in place** (see below). |
| `audits/YYYY-MM-DD-backlog-<slug>.md` | One record per `/audit-backlog` run (`kind: backlog`). Consumes individual findings; gets no main-table row. |
```

そして § "Status vocabulary" の直前に:

```markdown
## Amending a done record

A `done` record is the account of one run at one time, and that account
is immutable — do not rewrite its findings, reasoning, or numbering.

But a `## Deferred` section is also a live worklist. When a later run
resolves a deferred item, it is **annotated additively**, leaving the
original text and number intact:

```markdown
3. **`foo.py:120` — <original finding, unchanged>**
   **Not applied because** <original reason, unchanged>
   **RESOLVED** YYYY-MM-DD in `<sha>` — see
   [YYYY-MM-DD backlog](YYYY-MM-DD-backlog-<slug>.md).
```

Never renumber surviving items — the numbers are cited from
`coverage.md` and from other records. Update the row's open-item count in
`coverage.md` to match.

This resolves the contradiction between the immutability rule and commit
`916e874`, which moved a deferred item into Applied on a `done` record.
Annotation keeps both properties: the account stays intact, and the
worklist stays accurate.
```

## Motivation

`audits/` は container reclamation を越える唯一の監査記録なので，
「記録されたが誰も消化しない finding」が溜まると，ledger が
**やったことの記録** としては正しいまま **やるべきことの所在** としては
無価値になる．今回 22件 (deferred 13 + out-of-scope 9) が溜まっており，
うち deferred 13件は record を開かない限り誰の視界にも入らなかった．

immutability の矛盾を放置すると，次のセッションは `916e874` の前例と
README の文面のどちらに従うかを毎回判断し直すことになる．

## Alternatives considered

- **`/audit-and-fix` に消化モードを足す**．却下: あのコマンドは既に505行
  あり，スコープが「1パス」であることが resumability の根拠になっている．
  個別 finding 消化はスコープの定義が違う (パスを横断する) ので，
  同じコマンドに入れると step 0 の path 解決が二重化する．
- **deferred 項目を coverage.md に全部転記する**．却下: 13件の本文を
  ledger に展開すると ledger が読めなくなり，record との二重管理になる．
  glob で読む方が単一の真実を保てる．
- **消化 run にもメインテーブル行を書く**．却下: 未監査パスを
  監査済みと主張することになる (本提案 3 の理由)．

## What this enables

- deferred / out-of-scope が「溜まる一方」でなくなる — 消化に正常ルートが
  でき，優先度提示 → ユーザ選択 → 修正 → 行削除 + 記録 が1コマンドで回る．
- record を後から安全に注釈できる — immutability を壊さずに worklist を
  最新に保てる．

## What this constrains

- `/audit-backlog` は今後 record の Deferred 番号を **renumber できない**
  (他ファイルから参照されるため)．注釈は追記のみ．
- consumption record が増えるので `audits/` のファイル数は増える．
  ledger のメインテーブルは増えない．

## Rollback plan

いずれも文面のみ．`CLAUDE.md` は 3箇所，`audits/README.md` は 2箇所の
差分なので revert は容易．コマンドファイルを消せば参照も消える
(コマンド不使用でも既存の `/audit-and-fix` 運用は影響を受けない)．
