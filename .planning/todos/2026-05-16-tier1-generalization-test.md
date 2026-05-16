---
owner: agent
snoozed_until: null
---
# Test Tier 1 ONNX classifier on unseen voices (not just the 5 ElevenLabs we trained the harness against)

**Source:** session 2026-05-16 readme gaps

Current 100% accuracy is on a closed-set of the 5 ElevenLabs voices we generated (Sarah, Roger, Laura, Charlie, George). Need to verify the embedding actually generalizes. Plan:

1. Generate 10-clip corpora for 5-10 additional ElevenLabs voices NOT in the current pool. Add to data/dataset/voices/ with manifest entries.
2. Re-run the benchmark with the new voices in the test set, enrolment from 1 clip per voice as before. 
3. Also test cross-voice contamination — does Sarah's template ever score higher than the actual speaker?
4. Report top-1, EER, and confusion matrix.

Expected outcome: still 90%+ since the model was trained on VoxCeleb2 (5994 real speakers, much harder than 10-15 ElevenLabs voices). If it drops, surface the failure cases for analysis.

Refs: error E8 in ERRORS-AND-ATTEMPTS.xml, decision D-024 (Tier 1 ONNX rationale), task 07-01.
