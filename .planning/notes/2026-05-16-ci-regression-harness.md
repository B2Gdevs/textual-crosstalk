# CI / automated regression — gate accuracy + latency on PRs

**Source:** session 2026-05-16 readme gaps

Current benchmark harness is operator-run (python -m scripts.conduit_tui.benchmark). No CI gating means accuracy or latency regressions could slip into main between operator-driven measurements.

What we'd need:
1. GitHub Action that runs benchmark on every PR.
2. Threshold gates: classifier accuracy >= 95% (Tier 1 baseline), AEC ERLE >= 8 dB, classifier latency <= 100ms, AEC latency <= 5ms per chunk.
3. Cache the data/dataset/ corpus in CI somehow — currently gitignored as billable artifact. Options: generate on-demand with cached ElevenLabs response, store as encrypted artifact, or use a smaller frozen dev corpus.
4. Report numbers in PR comment.

Risk of NOT doing this: the same bug class we hit at e9f5be8 (claimed 85-90% accuracy without measuring, actual 24%). The harness exists; CI just enforces 'measure before merging'.

Effort: 1-2h to wire GitHub Action + threshold script, plus the corpus-in-CI strategy decision. Not blocking current work but increasingly valuable as the repo gains more contributors / commits per week.

Source provenance: README 'Known gaps' #7, session 2026-05-16.
