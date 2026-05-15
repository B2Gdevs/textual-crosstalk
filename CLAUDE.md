# conduit_proj

Claude Code entrypoint. **The source contract is `AGENTS.md` at the repo
root.** Read that file first — it is the authoritative agent contract
for `conduit_proj` regardless of which runtime you are.

```
Read: ./AGENTS.md
```

Everything below is a Claude-only addendum. Anything that should apply
project-wide (loop, planning IDs, CLI reference) belongs in `AGENTS.md`.

## Claude-only notes

- Use `Skill <name>` to invoke installed project skills before the
  first edit when they match the task.
- The Claude harness exposes `Agent` (subagent dispatch), `TaskCreate`,
  and parallel tool calls — prefer them over sequential bash when work
  is independent.
