from __future__ import annotations

from typing import Any

from litebench.agent.base import AgentTask
from litebench.tasks.arc import ARCTask
from litebench.tasks.base import Task
from litebench.tasks.gsm8k import GSM8KTask
from litebench.tasks.gsm8k_agent import GSM8KAgentTask
from litebench.tasks.humaneval import HumanEvalTask
from litebench.tasks.math_task import MATHTask
from litebench.tasks.mmlu import MMLUTask
from litebench.tasks.truthfulqa import TruthfulQATask

# AgentTask and Task don't share a base, but the registry values are all no-arg
# constructible so the caller treats them uniformly.
_REGISTRY: dict[str, type[Any]] = {
    "humaneval": HumanEvalTask,
    "gsm8k": GSM8KTask,
    "gsm8k-agent": GSM8KAgentTask,
    "mmlu": MMLUTask,
    "truthfulqa": TruthfulQATask,
    "math": MATHTask,
    "arc": ARCTask,
}


def get_task(name: str) -> Task | AgentTask:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown task: {name}. Try: {', '.join(_REGISTRY)}")
    return _REGISTRY[key]()


def list_tasks() -> list[str]:
    return list(_REGISTRY)


__all__ = ["AgentTask", "Task", "get_task", "list_tasks"]
