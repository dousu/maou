# reviews/

Single audit trail for changes to durable project knowledge:

- new or changed rules in `CLAUDE.md`
- architectural changes (layers, dependency direction, new conventions)
- promotions from `scratchpad/` or `worklog/` into project-level docs
- retirement of previously-applied rules

Flat folder — no subdirectories. Lifecycle is tracked by the `status:`
frontmatter field on each file.

## Filename

`reviews/YYYY-MM-DD-<kebab-title>.md` (date is JST).

## Frontmatter

```yaml
---
status: pending          # pending | approved | applied | rejected
applied_in:              # commit SHA, filled when status becomes applied
date: YYYY-MM-DD
target: [CLAUDE.md, docs/architecture.md]
risk: low | medium | high
reversibility: trivial | moderate | hard
---
```

## Status lifecycle

```
pending  → applied   (user approves; the MODEL applies the edit +
                      commits, then writes status: applied,
                      applied_in: <sha>)
pending  → rejected  (user rejects; file RETAINED as do-not-redo
                      provenance, with the reason in the body)
pending  → deleted   (rejected AND never substantive)
```

Approval is **not bound to one command**. `/checkpoint-context` step 5 is
the routine reconciliation point; `/audit-and-fix` step 8 reconciles the
proposals its own run filed. What the rule requires is explicit user
approval before the edit — not the command it happens in.

`status: applied` and `status: rejected` are terminal — never modified.
`approved` is permitted as an intermediate when approval and apply
must be split in time; normal flow skips it.

## See also

`docs/memory-architecture.md` § "Review proposal shape" — full template
and field meanings.
