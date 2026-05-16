---
owner: agent
snoozed_until: null
---
# Measure AEC ERLE on a real speaker→air→mic loop (not just synthetic)

**Source:** session 2026-05-16 readme gaps

Current AEC numbers (-3.2 dB on pure tones, 11.5 dB on broadband ad-hoc) are all from synthetic signals. Need real measurement to know whether the speex AEC is doing useful work on the operator's actual hardware.

Plan:
1. Use the data/dataset/user/*.wav recordings (already captured by operator_capture.py). Each is the AEC-cleaned mic stream during a session.
2. Cross-reference against the bot TTS that played during the same session — the orchestrator already feeds bot PCM to the AEC reference path; we have full provenance.
3. Compute residual_rms / bot_rms over time windows during TTS playback. ERLE = 20*log10(echo_input_rms / residual_rms).
4. Report mean ERLE + ERLE-over-time curve to see convergence behavior on real hardware.

Bonus: compare ERLE across operator sessions to see if AEC adapts cleanly across acoustic environments (different rooms, different mics).

Refs: D-017 (AEC architecture), benchmark.py benchmark_aec().
