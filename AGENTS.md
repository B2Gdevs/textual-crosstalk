# conduit_proj Agent Contract

Project name: `conduit_proj`
Project id: `conduit-proj`

## Soul pointer

Read `SOUL.md` first, then read the soul body it points to under
`narrative/souls/` when present. Gilgamesh of Uruk is the default setup soul.
Projects can override that by updating `SOUL.md` with a stable project-specific
soul pointer.

## Loop

Use this loop every session:

1. `gad snapshot --projectid conduit-proj`
2. Pick one task
3. Implement the task
4. `gad tasks stamp <task-id> --projectid conduit-proj --status done --agent <agent> --runtime <runtime>`
5. `gad state log "<delta>" --projectid conduit-proj`
6. Commit

## Planning IDs

Use canonical IDs exactly:

| Entity | Format |
|---|---|
| decisions | `CONDUIT-PROJ-D-<n>` |
| tasks | `CONDUIT-PROJ-T-<phase>-<n>` |
| handoffs | `h-<ISO>-conduit-proj-<phase>` |
| requirements | `CONDUIT-PROJ-R-<n>` |
| errors | `CONDUIT-PROJ-E-<n>` |

## Communication style

- SITREP format.
- Tables when structure helps.
- Report deltas only.
- Always close with gaps.
- Call entities by registered name, not shorthand.

## GAD CLI quick reference

| Command | Purpose |
|---|---|
| `gad snapshot --projectid conduit-proj` | Hydrate planning context before work |
| `gad tasks list --projectid conduit-proj` | Inspect available work |
| `gad tasks stamp <task-id> --projectid conduit-proj --status done --agent <agent> --runtime <runtime>` | Stamp task completion |
| `gad state log "<delta>" --projectid conduit-proj` | Append a state-log entry |
| `gad decisions add CONDUIT-PROJ-D-<n> --projectid conduit-proj --summary "..."` | Record a decision |
| `gad handoffs list --projectid conduit-proj` | Inspect open handoffs |
| `gad handoffs claim <handoff-id>` | Claim a handoff |
| `gad handoffs complete <handoff-id> --by <runtime>` | Close a handoff |

## Lane discipline

Single-agent by default. Use multi-agent execution only when `gad team`
is active and the work has been explicitly split into lanes.

## Files

- `AGENTS.md` (this file) is the **source contract** for every runtime.
- `CLAUDE.md` is a thin entrypoint for Claude Code that points back here
  and adds Claude-only addenda (skills, subagents, Claude harness tools).
- Other runtime entrypoints (`.cursorrules`, codex `AGENTS.md`,
  `GEMINI.md`, etc.) should follow the same pattern: read this file,
  then add only runtime-specific addenda. Do not duplicate the contract
  into every entrypoint — drift is inevitable.
- `.planning/AGENTS.md` narrows the rules for planning-only edits.
