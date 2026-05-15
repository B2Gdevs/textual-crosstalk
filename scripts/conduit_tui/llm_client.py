"""
llm_client.py — OpenRouter primary client + OpenAI/Groq fallback chain.

Decision D-003: Groq llama-3.1-8b-instant primary; OpenRouter or OpenAI fallback.
Implementation uses openai SDK with base_url override for OpenRouter/Groq
(both expose OpenAI-compatible endpoints).

Memory: feedback_no_ai_sdk_use_openai_compatible — use `openai` SDK only,
never @ai-sdk/* in this repo context.
"""
from __future__ import annotations

import os
import time
from typing import Iterator


from openai import AsyncOpenAI


_SYSTEM_PROMPT = (
    "You are a helpful, conversational AI assistant. "
    "Keep responses concise and natural-sounding for voice conversation — "
    "2-4 sentences maximum. No markdown, no lists. Speak naturally."
)


class LLMClient:
    """Try OpenRouter first, then Groq, then OpenAI direct."""

    def __init__(
        self,
        openrouter_api_key: str | None = None,
        openrouter_model: str = "meta-llama/llama-3.3-70b-instruct",
        groq_api_key: str | None = None,
        groq_model: str = "llama-3.1-8b-instant",
        openai_api_key: str | None = None,
    ) -> None:
        self._clients: list[tuple[AsyncOpenAI, str, str]] = []  # (client, model, label)

        if openrouter_api_key:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/conduit-proj",
                    "X-Title": "conduit",
                },
            )
            self._clients.append((client, openrouter_model, "openrouter"))

        if groq_api_key:
            client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_api_key,
            )
            self._clients.append((client, groq_model, "groq"))

        if openai_api_key:
            client = AsyncOpenAI(api_key=openai_api_key)
            self._clients.append((client, "gpt-4o-mini", "openai"))

        if not self._clients:
            raise ValueError("No LLM API keys configured — set OPENROUTER_API_KEY or GROQ_API_KEY")

    async def complete(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, float, str]:
        """Returns (response_text, latency_seconds, provider_label)."""
        messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        last_error: Exception | None = None
        for client, model, label in self._clients:
            try:
                t0 = time.monotonic()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=256,
                    temperature=0.7,
                )
                latency = time.monotonic() - t0
                text = resp.choices[0].message.content or ""
                return text.strip(), latency, label
            except Exception as exc:
                last_error = exc
                print(f"[llm] {label} failed: {exc}, trying next...")
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
