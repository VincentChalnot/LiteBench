from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from litebench.core.models import Sample


@dataclass
class Tool:
    """One callable function exposed to the model.

    ``schema`` follows OpenAI / Anthropic JSON Schema for parameters. The handler
    runs synchronously (LLM tool calls are coarse-grained; we don't need async
    handlers for the kinds of deterministic tools — calc, search stubs — that
    ship with LiteBench).
    """

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., str]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


@dataclass
class ToolCall:
    """One tool invocation by the agent during a rollout."""

    name: str
    arguments: dict[str, Any]
    result: str
    error: str | None = None


@dataclass
class AgentTrace:
    """Per-sample rollout record."""

    final_answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    stop_reason: str = "complete"  # complete | max_steps | error


class AgentTask(ABC):
    """A benchmark whose samples are run as multi-step tool-using rollouts.

    Deliberately not a subclass of Task — the two classes share the *shape* of a
    benchmark (samples, prompts, scoring) but dispatch through different Runner
    paths. Keeping them independent avoids an import cycle and lets the scoring
    signature honestly reflect the agent trace.
    """

    name: str = ""
    description: str = ""
    max_steps: int = 10

    @abstractmethod
    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        ...

    @abstractmethod
    def tools(self) -> list[Tool]:
        ...

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        ...

    @abstractmethod
    def score_trace(self, sample: Sample, trace: AgentTrace) -> tuple[float, bool]:
        ...

    def system_prompt(self) -> str | None:
        return None
