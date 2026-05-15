# Planning Lane Contract

This file applies only inside `.planning/`.

## Scope

- Keep planning artifacts coherent.
- Prefer atomic CLI writes over manual edits when a command exists.
- Preserve naming canon and timeline integrity.

## Default lane

Planning hygiene is a single-agent lane by default. Do not fork parallel
planning edits unless `gad team` explicitly assigns separate ownership.

## Working rules

1. Read the project-root `AGENTS.md` first.
2. Treat `STATE.xml`, `ROADMAP.xml`, `TASK-REGISTRY.xml`, `DECISIONS.xml`,
   `REQUIREMENTS.xml`, and `ERRORS-AND-ATTEMPTS.xml` as the planning source.
3. Use `gad state log`, `gad tasks ...`, `gad decisions ...`, and
   `gad handoffs ...` when those commands cover the change.
4. Report deltas only and always close with gaps.
