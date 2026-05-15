# conduit_proj

Gemini CLI entrypoint. **The source contract is `AGENTS.md` at the
repo root.** Read that file first — it is the authoritative agent
contract for `conduit_proj` regardless of which runtime you are.

```
Read: ./AGENTS.md
```

## Gemini-only notes

- This project uses the GAD framework. Run `gad snapshot --projectid
  conduit-proj` at session start to hydrate planning context.
- All planning entities (decisions, tasks, requirements, errors) live
  under `.planning/` and are mutated via the `gad` CLI — never
  hand-edit the XML files.
- Per memory `feedback_no_ai_sdk_use_openai_compatible.md`: when
  adding new LLM calls to this project, use the `openai` npm package
  or raw fetch — do NOT add `ai` / `@ai-sdk/*` packages.
