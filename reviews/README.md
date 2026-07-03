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
status: pending          # pending | approved | applied
applied_in:              # commit SHA, filled when status becomes applied
date: YYYY-MM-DD
target: [CLAUDE.md, docs/architecture.md]
risk: low | medium | high
reversibility: trivial | moderate | hard
---
```

## Status lifecycle

```
pending  → applied   (user approves during /checkpoint-context;
                      user applies the edit + commits;
                      model writes status: applied, applied_in: <sha>)
pending  → deleted   (user rejects during /checkpoint-context)
```

`status: applied` is terminal — applied proposals are never modified.
`approved` is permitted as an intermediate when approval and apply
must be split in time; normal flow skips it.

## See also

`docs/memory-architecture.md` § "Review proposal shape" — full template
and field meanings.
