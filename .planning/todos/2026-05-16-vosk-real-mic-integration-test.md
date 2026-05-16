---
owner: operator
snoozed_until: null
---
# Run TUI with CONDUIT_STT=vosk and verify real-mic end-to-end (not just wav replay)

**Source:** session 2026-05-16 readme gaps

Vosk path has only been smoke-tested with wav file replay through VoskStream.send(). First real-microphone validation hasn't happened yet.

Steps:
1. setup.ps1 (or python main.py with venv active)
2. Set CONDUIT_STT=vosk in .env.local
3. Have a normal conversation. Watch for:
   - First-partial latency (compare to Deepgram baseline of ~877ms)
   - Word-level transcription quality on real mic audio
   - Whether Crosstalk fires turns correctly given Vosk's 4138ms time-to-final
   - Any contract-mismatch issues (Vosk has no utterance_end_ms — does anything else break?)
4. Capture the session wav (operator_capture.py persists automatically).
5. Compare to a CONDUIT_STT=deepgram session for the same phrases.

If turn-detection feels sluggish, the vosk-time-to-final-tuning todo is the follow-up.

Refs: 2794adc (Vosk shipped), bc707d6 (WER numbers).
