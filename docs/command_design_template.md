# Command design document template

Use this template whenever you add or update a command reference under
`docs/commands`. The goal is to keep every CLI design document consistent, easy
to skim, and traceable back to the source code.

## Required sections

1. **Title** – ``# `maou <subcommand>` ``.
2. **Overview** – Two or three bullet points summarizing what the command does
   and which modules own the implementation. Include citations to the CLI and
   primary interface/app modules.
3. **CLI options** – Break flags into logical groups (e.g., input selection,
   caching, performance knobs). Each group should use a Markdown table with the
   columns `Flag`, `Required`, and `Description` (or similar for grouped flags).
   Reference the CLI or interface file where each option is defined.
4. **Execution flow** – Ordered list describing how the CLI hands work to the
   interface and app layers. Cite the relevant functions so readers can jump to
   the code.
5. **Validation and guardrails** – Bullet list of important invariants and
   failure modes (mutual exclusivity, range checks, dependency requirements).
6. **Outputs and usage** – Explain what the command prints or writes, any JSON
   structure, and how to apply the results. Include citations to formatter or
   writer code. Provide a concrete **Example invocation** block that starts with
   `poetry run maou ...`.
7. **Implementation references** – Bullet list linking to the CLI, interface,
   and app modules that power the command.

## Style guidelines

- Keep paragraphs short and prefer bullet lists for step-by-step explanations.
- Cite source files using the repository-relative format (e.g.,
  `【F:src/maou/infra/console/utility.py†L520-L620】`).
- When describing tables of flags, group related options rather than listing
  dozens of single rows.
- Mention mutually exclusive options, default values, and any auto-detected
  behavior in the tables so operators have a one-stop reference.
- Use present tense and imperative voice for recommendations (“Use the
  recommendations to tune `maou learn-model`...”).
- Always include at least one example command that readers can copy/paste.

Following this structure keeps every command document predictable, which makes it
easier for operators and contributors to find the information they need without
reading multiple files.
