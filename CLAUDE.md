# conduit_proj

Claude Code entrypoint. **The source contract is `AGENTS.md` at the repo
root.** Read that file first — it is the authoritative agent contract
for `conduit_proj` regardless of which runtime you are.

```
Read: ./AGENTS.md
```

Everything below is a Claude-only addendum. Anything that should apply
to every runtime (loop, planning IDs, CLI reference, lane discipline)
belongs in `AGENTS.md`, not here.

## Claude-only notes

- Use `Skill <name>` to invoke installed project skills before the
  first edit when they match the task (e.g. `frontend-design`,
  `web-design-guidelines`, `gad-visual-context-system` for UI work).
- The Claude harness exposes `Agent` (subagent dispatch), `TaskCreate`,
  and parallel tool calls — prefer them over sequential bash when work
  is independent.
- Other runtime entrypoints (`.cursorrules`, codex `AGENTS.md`,
  `GEMINI.md`, etc.) are also references to the same `AGENTS.md` source;
  if you edit project-wide rules, edit `AGENTS.md` and let the runtime
  files stay thin.
