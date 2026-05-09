"""
Tests for the run_context module.
"""

import pytest

from cyberai import RunContext, Usage


class TestRunContext:
    """Tests for the RunContext dataclass."""

    def test_wraps_context_object(self):
        """RunContext wraps the provided context object."""
        my_context = {"key": "value"}
        run_ctx = RunContext(context=my_context)
        assert run_ctx.context == my_context
        assert run_ctx.context is my_context

    def test_wraps_none_context(self):
        """RunContext can wrap None as context."""
        run_ctx = RunContext(context=None)
        assert run_ctx.context is None

    def test_wraps_custom_class_context(self):
        """RunContext can wrap a custom class instance."""
        class MyContext:
            def __init__(self, value: int):
                self.value = value

        ctx = MyContext(42)
        run_ctx = RunContext(context=ctx)
        assert run_ctx.context.value == 42

    def test_default_usage_is_empty(self):
        """RunContext has a default empty Usage instance."""
        run_ctx = RunContext(context=None)
        assert isinstance(run_ctx.usage, Usage)
        assert run_ctx.usage.requests == 0
        assert run_ctx.usage.input_tokens == 0
        assert run_ctx.usage.output_tokens == 0
        assert run_ctx.usage.total_tokens == 0

    def test_usage_can_be_provided(self):
        """RunContext can be created with a custom Usage."""
        custom_usage = Usage(requests=5, input_tokens=100, output_tokens=50, total_tokens=150)
        run_ctx = RunContext(context=None, usage=custom_usage)
        assert run_ctx.usage is custom_usage
        assert run_ctx.usage.requests == 5

    def test_usage_accumulates(self):
        """Usage can be accumulated through the RunContext."""
        run_ctx = RunContext(context=None)

        new_usage = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)
        run_ctx.usage.add(new_usage)

        assert run_ctx.usage.requests == 1
        assert run_ctx.usage.input_tokens == 50
        assert run_ctx.usage.output_tokens == 25
        assert run_ctx.usage.total_tokens == 75

        # Add more usage
        run_ctx.usage.add(new_usage)

        assert run_ctx.usage.requests == 2
        assert run_ctx.usage.input_tokens == 100
        assert run_ctx.usage.output_tokens == 50
        assert run_ctx.usage.total_tokens == 150

    def test_generic_typing_with_dict(self):
        """RunContext is generic and works with dict context."""
        run_ctx: RunContext[dict] = RunContext(context={"api_key": "secret"})
        assert run_ctx.context["api_key"] == "secret"

    def test_generic_typing_with_string(self):
        """RunContext is generic and works with string context."""
        run_ctx: RunContext[str] = RunContext(context="my session")
        assert run_ctx.context == "my session"

    def test_context_is_mutable(self):
        """The wrapped context can be mutated."""
        ctx = {"counter": 0}
        run_ctx = RunContext(context=ctx)

        run_ctx.context["counter"] += 1
        assert run_ctx.context["counter"] == 1

        # Original dict is also mutated (same reference)
        assert ctx["counter"] == 1
