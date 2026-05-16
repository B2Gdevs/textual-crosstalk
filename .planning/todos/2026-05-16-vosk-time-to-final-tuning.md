---
owner: agent
snoozed_until: null
---
# Adapt Crosstalk silence thresholds for Vosk's slower time-to-final (no utterance_end_ms equivalent)

**Source:** session 2026-05-16 readme gaps

Vosk path has 4138 ms latency-to-final vs Deepgram's 2352 ms because Vosk has no utterance_end_ms — turn-finalization relies entirely on Crosstalk.SETTLED_THRESHOLD_MS wall-clock silence. With Deepgram tuned for utterance_end_ms=1500, Crosstalk fires LLM after 600ms silence — but with Vosk, Vosk itself buffers more before emitting is_final.

Action:
1. Profile where the 4138ms goes — is_final emission lag vs Crosstalk wait time. Print timestamps in vosk_client.py to see.
2. If Vosk's AcceptWaveform → Result() pipeline is the bottleneck, tune Vosk's recognizer for shorter accept windows OR finalize on partial-stable detection (same partial text for N ms).
3. Lower Crosstalk SETTLED_THRESHOLD_MS for the Vosk path (env-conditional: CROSSTALK_SETTLED_THRESHOLD_MS_VOSK=300?).
4. Re-measure WER + latency-to-final.

Goal: bring Vosk time-to-final under 3s while keeping WER 3.5%.

Refs: bc707d6 (live STT numbers), task 08-01.
