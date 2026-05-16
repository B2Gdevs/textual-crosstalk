# Tier 2 distilled int8 speaker classifier — deferred (Tier 1 already at 100%)

**Source:** session 2026-05-16 readme gaps

When Tier 1 was planned, Tier 2 (distilled int8 CNN, ~5-15 MB, ~5-20ms inference) was the next escalation if Tier 1 left gaps. Tier 1 measured 100% on the 5-way ElevenLabs benchmark — there's no accuracy headroom left to chase.

Tier 2 trigger conditions (when to revisit):
- Operator deploys to a device where ~70 MB install delta is prohibitive (mobile, embedded).
- Tier 1's 60ms per-inference latency starts limiting use case (e.g. very long meetings where classifier runs continuously).
- Generalization test (tier1-generalization-test todo) reveals Tier 1 falls below 95% on unseen voices and a smaller specialized model could win on closed-set.

Without one of those triggers, the right move is to spend cycles elsewhere — Tier 1 is sufficient for the current voice-conversation use case.

Source provenance: README 'Known gaps' #10 (classifier latency), D-019 (original tiering), D-024 (Tier 1 result), session 2026-05-16.
