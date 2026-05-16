# By-design tradeoffs that look like gaps but aren't

**Source:** session 2026-05-16 readme gaps

Some items in the README's 'Known gaps' section are deliberate design choices, not undone work. Captured here so a future agent doesn't try to 'fix' them.

## Per-machine voiceprint cache (~/.conduit/voiceprint_user_onnx.npy)

The operator's voice embedding is stored on the local machine only — never synced, never uploaded, never embedded in the repo. Re-enrollment is needed on a fresh machine (first session captures + caches automatically).

**Why kept:** voice biometric is sensitive personal data. Cloud sync would create a privacy hazard and a single-point-of-compromise. The 3-second auto-enroll on first session is fast enough that re-enrollment friction is negligible.

**When to revisit:** if a user explicitly opts in to cross-device sync via an authenticated personal store (encrypted backup, hardware key). Not before.

## Voice dataset gitignored (data/dataset/)

The 150-clip ElevenLabs corpus + operator session wavs are excluded from the repo via .gitignore.

**Why kept:** (a) ElevenLabs clips are billable artifacts — committing them would mean re-billing on every fresh clone. (b) Operator session wavs contain the operator's voice — same privacy reasoning as above. (c) Dataset is reproducible: scripts/conduit_tui/dataset_gen.py is idempotent and rebuilds from the manifest.

**When to revisit:** if we need a frozen golden-set for CI gating (see ci-regression-harness note) — then a small subset goes in repo OR as an encrypted CI artifact.

## Tier 0 numpy classifier kept as fallback

Tier 0 was measured at 24% on real voices (Tier 1 hit 100%). We kept Tier 0 anyway as a CONDUIT_SPEAKER_TIER=numpy fallback.

**Why kept:** zero-dependency mode. Operator may install conduit on a stripped-down environment without onnxruntime; Tier 0 still works (just badly for multi-bot). Removing it would mean a hard onnxruntime dep, which contradicts the 'minimal install' principle.

**When to revisit:** when Tier 2 ships (smaller learned model + onnxruntime as the absolute floor). At that point Tier 0 has no role.

Source provenance: README 'Known gaps' #8, #9, session 2026-05-16.
