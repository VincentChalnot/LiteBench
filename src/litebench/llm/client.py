from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import litellm

# litellm prints a lot at import time; quiet it down for CLI UX.
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw: Any = None
    # Populated on agent turns when the model decides to call tools. Each dict
    # follows OpenAI tool_call shape: {"id", "type", "function": {"name", "arguments"}}.
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        extra_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = extra_params or {}

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult:
        started = time.perf_counter()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        params.update(self.extra_params)
        params.update(kwargs)

        resp = await litellm.acompletion(**params)
        latency_ms = int((time.perf_counter() - started) * 1000)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        tool_calls: list[dict[str, Any]] | None = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                # litellm normalizes across providers but objects can be pydantic
                # models or dicts depending on the backend; coerce through dict.
                if hasattr(tc, "model_dump"):
                    tool_calls.append(tc.model_dump())
                elif hasattr(tc, "to_dict"):
                    tool_calls.append(tc.to_dict())
                else:
                    tool_calls.append(dict(tc))

        return ChatResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw=resp,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    async def complete(self, prompt: str, **kwargs: Any) -> ChatResult:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)
