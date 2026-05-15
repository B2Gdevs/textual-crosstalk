# Conduit — Realtime Voice Conversation + Char Transcription

## Verbatim Test Brief

> Build a Python conversation + transcription program. 2.5 hours. Share with GitHub user **RioPopper** when done. Up to $100 of credits reimbursable. Use as much AI as needed.
>
> **Part 1** (do first, completely):
> 1. User speaks → realtime transcription via **Deepgram** (API key `07101d289c209b4af367aef6c517e640c5493079`). Latency target: <300ms from word-spoken to word-saved.
> 2. Data format: `char, start-time, end-time, notes/debug`. One entry per char. "hello world" → 11 or 12 entries.
> 3. Word start/end times come from Deepgram; char-level timestamps estimated by interpolation across word duration.
> 4. Transcription → LLM response (Groq / Fireworks / Cerebras / OpenRouter — operator's choice; Groq is fastest for conversation).
> 5. LLM response → human-sounding TTS via **ElevenLabs**.
> 6. Char-level timestamps of TTS output saved in the same format.
>
> End state: program where user converses with LLM (human voice), all spoken+heard chars saved in unified format.
>
> **Part 2** (start AFTER Part 1 complete):
> - Implement **crosstalk** per `https://github.com/tarzain/crosstalk` — intelligent interruption detection so the LLM doesn't cut off the user.
> - Familiarize with that repo's logic + impl.
> - Integrate into Part 1.
>
> **Evaluation criteria**: data format, accuracy, latency.

---

## Deadline

Timer started: 2026-05-15 ~12:35 local  
Hard stop: ~15:05 local (2.5 hours)  
Priority: Part 1 complete before touching Part 2.

## Deliverable

GitHub repo shared with user **RioPopper** (collaborator OR public URL).  
README required: usage + API key setup.

## Constraints

- Python only
- $100 credit cap (Deepgram + Groq + ElevenLabs costs)
- API keys: Deepgram `07101d289c209b4af367aef6c517e640c5493079`; Groq + ElevenLabs keys from operator env

## Success Criteria (Eval)

1. **Data format** — `(char, start_time, end_time, notes)` per char, "hello world" = 11-12 entries
2. **Accuracy** — Deepgram transcription fidelity; ElevenLabs char timestamps correct
3. **Latency** — <300ms word-spoken to word-saved; fast LLM first-token

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Test mandated |
| Audio I/O | sounddevice | Windows-friendly, numpy buffers |
| STT | Deepgram SDK v3+ websocket | Mandated; word-level timestamps |
| LLM | Groq (llama-3.1-8b-instant) | Fastest first-token; Cerebras fallback |
| TTS | ElevenLabs text_to_speech.with_timestamps | Native char timestamps |
| Data store | Python dataclass list + JSONL flush | Simple, matches schema |
| Crosstalk (P2) | tarzain/crosstalk logic | Study repo first — D-007 |

## Data Format Schema

```python
@dataclass
class CharEntry:
    char: str           # single character
    start_time: float   # seconds since session start
    end_time: float     # seconds since session start
    notes: str          # e.g. "user,interpolated" or "bot,native"
```

Output file: `conversation.jsonl` — one JSON object per line, appended after each turn.

## Phases

| Phase | Title | Goal |
|---|---|---|
| 00 | Setup | Venv + env vars + API smoke tests |
| 01 | Part 1 loop | STT + LLM + TTS + unified store |
| 02 | Part 2 crosstalk | tarzain/crosstalk integration |
| 03 | Ship | GitHub + README + share with RioPopper |

## Key Decisions

- D-001: Python 3.11+
- D-002: Deepgram SDK v3+ websocket
- D-003: Groq llama-3.1-8b-instant (Cerebras fallback)
- D-004: ElevenLabs with_timestamps endpoint
- D-005: sounddevice for audio
- D-006: JSONL data store
- D-007: Crosstalk approach TBD after repo review
- D-008: Repo sharing strategy decided at session end
