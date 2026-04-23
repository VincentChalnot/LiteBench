"""Built-in tools ready for AgentTask subclasses to consume."""

from __future__ import annotations

import ast
import operator

from litebench.agent.base import Tool

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float | int:
    """Evaluate a pure arithmetic expression. Rejects anything beyond the AST nodes above.

    Using `eval` even with a restricted globals dict is a security footgun — this
    AST walk is the cheapest way to guarantee the model can't import os or call
    ``__import__``.
    """
    tree = ast.parse(expr, mode="eval")

    def walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
            return op(walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
            return op(walk(node.operand))
        raise ValueError(f"unsupported node: {type(node).__name__}")

    return walk(tree)


def _calculator_handler(expression: str) -> str:
    try:
        result = _safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error: {e}"
    # Keep integers as ints so the model sees "42" not "42.0".
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


CALCULATOR = Tool(
    name="calculator",
    description=(
        "Evaluate a math expression (+ - * / % ** //). "
        "Use for arithmetic instead of reasoning in your head. "
        'Example: calculator(expression="(17 + 9) * 3")'
    ),
    schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression, e.g. '(17 + 9) * 3' or '2**10'.",
            }
        },
        "required": ["expression"],
    },
    handler=_calculator_handler,
)


def _final_answer_handler(answer: str) -> str:
    # Returning the answer string verbatim means the scorer sees exactly what
    # the model committed to — no prose around it.
    return answer


FINAL_ANSWER = Tool(
    name="final_answer",
    description=(
        "Submit your final answer and end the task. "
        "Call this exactly once when you are done. "
        'Example: final_answer(answer="42")'
    ),
    schema={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to submit. Be concise.",
            }
        },
        "required": ["answer"],
    },
    handler=_final_answer_handler,
)
