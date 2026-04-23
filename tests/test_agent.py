import pytest

from litebench.agent.tools import CALCULATOR, FINAL_ANSWER, _safe_eval


class TestSafeEval:
    def test_arithmetic(self):
        assert _safe_eval("1 + 2 + 3") == 6
        assert _safe_eval("(17 + 9) * 3") == 78
        assert _safe_eval("2**10") == 1024
        assert _safe_eval("-7 + 5") == -2

    def test_rejects_imports(self):
        with pytest.raises(ValueError):
            _safe_eval("__import__('os').system('ls')")

    def test_rejects_names(self):
        with pytest.raises(ValueError):
            _safe_eval("x + 1")

    def test_rejects_calls(self):
        with pytest.raises(ValueError):
            _safe_eval("abs(-5)")

    def test_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            _safe_eval("1/0")


class TestCalculatorHandler:
    def test_success_returns_int_when_whole(self):
        # 3.0 should collapse to "3", not "3.0" — the model shouldn't see spurious floats.
        assert CALCULATOR.handler(expression="6 / 2") == "3"

    def test_success_keeps_fractional(self):
        assert CALCULATOR.handler(expression="7 / 2") == "3.5"

    def test_error_returns_readable_string(self):
        out = CALCULATOR.handler(expression="x + 1")
        assert out.startswith("Error:")


class TestFinalAnswer:
    def test_echoes(self):
        assert FINAL_ANSWER.handler(answer="42") == "42"


class TestToolSchema:
    def test_openai_shape(self):
        spec = CALCULATOR.to_openai_tool()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "calculator"
        assert "expression" in spec["function"]["parameters"]["properties"]
