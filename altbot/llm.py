"""Thin async wrapper around the Anthropic SDK used by the orchestrator.

Kept behind a small interface so tests (and other backends) can swap it out.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

import anthropic

# Models that accept the server-side `fallbacks: "default"` parameter.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    """The model (and any fallback) declined the request."""


class LLM(Protocol):
    def stream_text(
        self, *, model: str, system: str, prompt: str, effort: str, max_tokens: int
    ) -> AsyncIterator[str]: ...

    async def decide(
        self, *, model: str, system: str, prompt: str, schema: dict[str, Any], effort: str
    ) -> dict[str, Any]: ...


class ClaudeLLM:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, fallbacks: bool = True):
        self.client = client or anthropic.AsyncAnthropic()
        self.fallbacks = fallbacks

    def _extra(self, model: str) -> dict[str, Any]:
        if self.fallbacks and model in FALLBACK_MODELS:
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    async def stream_text(
        self, *, model: str, system: str, prompt: str, effort: str, max_tokens: int
    ) -> AsyncIterator[str]:
        async with self.client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
            **self._extra(model),
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            raise RefusalError("the model declined to answer this turn")

    async def decide(
        self, *, model: str, system: str, prompt: str, schema: dict[str, Any], effort: str
    ) -> dict[str, Any]:
        response = await self.client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            **self._extra(model),
        )
        if response.stop_reason == "refusal":
            raise RefusalError("the leader declined to make a decision")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)
