from __future__ import annotations

from pathlib import Path
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


def get_task(name: str, **kwargs) -> Task | AgentTask:
    """Resolve a task by name or path.

    Accepts:
      - Registered task name (e.g. ``"gsm8k"``, ``"mmlu"``)
      - Path to a ``.yaml`` / ``.yml`` file → :class:`CustomTask`
      - Path to a ``.py`` file → dynamically imported Task subclass

    Special kwargs:
      - ``subject`` (str) — MMLU only: restrict to one subject
      - ``arc_easy`` (bool) — ARC only: use ARC-Easy config
    """
    path = Path(name)

    # Custom YAML task
    if path.exists() and path.suffix.lower() in {".yaml", ".yml"}:
        from litebench.tasks.custom import CustomTask

        return CustomTask(path)

    # Custom Python task
    if path.exists() and path.suffix.lower() == ".py":
        return _load_custom_python_task(path)

    # Registry lookup
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown task: {name}. Try: {', '.join(_REGISTRY)}")

    cls = _REGISTRY[key]

    if key == "mmlu" and kwargs.get("subject"):
        return cls(subject=kwargs["subject"])
    if key == "arc" and kwargs.get("arc_easy"):
        return cls(config="ARC-Easy")

    return cls()


def _load_custom_python_task(path: Path) -> Task | AgentTask:
    """Import a ``.py`` file and return the first Task/AgentTask subclass."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, (Task, AgentTask))
            and attr is not Task
            and attr is not AgentTask
        ):
            return attr()

    raise ValueError(
        f"No Task or AgentTask subclass found in {path}. "
        f"Define a class inheriting from Task or AgentTask."
    )


def list_tasks() -> list[str]:
    return list(_REGISTRY)


__all__ = ["AgentTask", "Task", "get_task", "list_tasks"]
