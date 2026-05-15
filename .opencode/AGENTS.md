# conduit_proj — opencode entrypoint

The source contract is `AGENTS.md` at the repo root. Read it first.

```
Read: ../AGENTS.md
```

## opencode-only notes

- This project uses the GAD framework. Run `gad snapshot --projectid
  conduit-proj` at session start.
- Stamp completed work with `gad tasks stamp <id> --projectid
  conduit-proj --runtime opencode --agent opencode --status done`.
- Planning XML under `.planning/` is mutated only via `gad` CLI.
