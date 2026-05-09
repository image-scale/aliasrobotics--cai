"""
Tests for the guardrail system.
"""

import pytest

from cyberai import Agent, RunContext
from cyberai.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
    input_guardrail,
    output_guardrail,
)
from cyberai.exceptions import UserError


class TestGuardrailFunctionOutput:
    """Tests for GuardrailFunctionOutput dataclass."""

    def test_guardrail_output_fields(self):
        """GuardrailFunctionOutput has output_info and tripwire_triggered."""
        output = GuardrailFunctionOutput(
            output_info={"check": "passed"},
            tripwire_triggered=False,
        )

        assert output.output_info == {"check": "passed"}
        assert output.tripwire_triggered is False

    def test_guardrail_output_tripwire_true(self):
        """GuardrailFunctionOutput can have tripwire_triggered=True."""
        output = GuardrailFunctionOutput(
            output_info="Malicious content detected",
            tripwire_triggered=True,
        )

        assert output.tripwire_triggered is True

    def test_guardrail_output_none_info(self):
        """GuardrailFunctionOutput can have None output_info."""
        output = GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=False,
        )

        assert output.output_info is None


class TestInputGuardrailDataclass:
    """Tests for InputGuardrail dataclass."""

    def test_input_guardrail_fields(self):
        """InputGuardrail has guardrail_function and optional name."""

        def check_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(
            guardrail_function=check_func,
            name="content_filter",
        )

        assert guardrail.guardrail_function is check_func
        assert guardrail.name == "content_filter"

    def test_input_guardrail_name_defaults_none(self):
        """InputGuardrail name defaults to None."""

        def check_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=check_func)

        assert guardrail.name is None

    def test_get_name_with_explicit_name(self):
        """get_name() returns the explicit name when provided."""

        def my_checker(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(
            guardrail_function=my_checker,
            name="custom_name",
        )

        assert guardrail.get_name() == "custom_name"

    def test_get_name_falls_back_to_function_name(self):
        """get_name() falls back to function name when no explicit name."""

        def security_checker(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=security_checker)

        assert guardrail.get_name() == "security_checker"


class TestInputGuardrailResult:
    """Tests for InputGuardrailResult dataclass."""

    def test_input_guardrail_result_fields(self):
        """InputGuardrailResult contains guardrail and output."""

        def check_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=check_func)
        output = GuardrailFunctionOutput(output_info="checked", tripwire_triggered=False)

        result = InputGuardrailResult(guardrail=guardrail, output=output)

        assert result.guardrail is guardrail
        assert result.output is output


class TestOutputGuardrailDataclass:
    """Tests for OutputGuardrail dataclass."""

    def test_output_guardrail_fields(self):
        """OutputGuardrail has guardrail_function and optional name."""

        def check_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(
            guardrail_function=check_func,
            name="output_validator",
        )

        assert guardrail.guardrail_function is check_func
        assert guardrail.name == "output_validator"

    def test_output_guardrail_name_defaults_none(self):
        """OutputGuardrail name defaults to None."""

        def check_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=check_func)

        assert guardrail.name is None

    def test_get_name_with_explicit_name(self):
        """get_name() returns the explicit name when provided."""

        def my_validator(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(
            guardrail_function=my_validator,
            name="custom_validator",
        )

        assert guardrail.get_name() == "custom_validator"

    def test_get_name_falls_back_to_function_name(self):
        """get_name() falls back to function name when no explicit name."""

        def quality_checker(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=quality_checker)

        assert guardrail.get_name() == "quality_checker"


class TestOutputGuardrailResult:
    """Tests for OutputGuardrailResult dataclass."""

    def test_output_guardrail_result_fields(self):
        """OutputGuardrailResult contains guardrail, agent_output, agent, and output."""

        def check_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test_agent")
        guard_output = GuardrailFunctionOutput(output_info="valid", tripwire_triggered=False)

        result = OutputGuardrailResult(
            guardrail=guardrail,
            agent_output="agent response",
            agent=agent,
            output=guard_output,
        )

        assert result.guardrail is guardrail
        assert result.agent_output == "agent response"
        assert result.agent is agent
        assert result.output is guard_output


class TestInputGuardrailRun:
    """Tests for InputGuardrail.run() method."""

    @pytest.mark.asyncio
    async def test_run_sync_guardrail(self):
        """InputGuardrail.run() executes sync guardrail function."""
        calls = []

        def check_func(ctx, agent, input):
            calls.append((ctx, agent, input))
            return GuardrailFunctionOutput(output_info="checked", tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(agent, "hello", ctx)

        assert isinstance(result, InputGuardrailResult)
        assert result.output.tripwire_triggered is False
        assert result.output.output_info == "checked"
        assert len(calls) == 1
        assert calls[0][1] is agent
        assert calls[0][2] == "hello"

    @pytest.mark.asyncio
    async def test_run_async_guardrail(self):
        """InputGuardrail.run() executes async guardrail function."""
        calls = []

        async def check_func(ctx, agent, input):
            calls.append(input)
            return GuardrailFunctionOutput(output_info="async_checked", tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(agent, "async input", ctx)

        assert result.output.output_info == "async_checked"
        assert calls == ["async input"]

    @pytest.mark.asyncio
    async def test_run_with_list_input(self):
        """InputGuardrail.run() works with list input."""

        def check_func(ctx, agent, input):
            return GuardrailFunctionOutput(
                output_info=f"items: {len(input)}",
                tripwire_triggered=False,
            )

        guardrail = InputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(agent, [{"role": "user", "content": "hi"}], ctx)

        assert result.output.output_info == "items: 1"

    @pytest.mark.asyncio
    async def test_run_returns_guardrail_reference(self):
        """InputGuardrail.run() returns result with guardrail reference."""

        def check_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=check_func, name="my_guard")
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(agent, "test", ctx)

        assert result.guardrail is guardrail
        assert result.guardrail.get_name() == "my_guard"

    @pytest.mark.asyncio
    async def test_run_non_callable_raises_error(self):
        """InputGuardrail.run() raises UserError if function not callable."""
        guardrail = InputGuardrail(guardrail_function="not a function")  # type: ignore
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        with pytest.raises(UserError):
            await guardrail.run(agent, "test", ctx)


class TestOutputGuardrailRun:
    """Tests for OutputGuardrail.run() method."""

    @pytest.mark.asyncio
    async def test_run_sync_guardrail(self):
        """OutputGuardrail.run() executes sync guardrail function."""
        calls = []

        def check_func(ctx, agent, output):
            calls.append((ctx, agent, output))
            return GuardrailFunctionOutput(output_info="validated", tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(ctx, agent, "agent output")

        assert isinstance(result, OutputGuardrailResult)
        assert result.output.output_info == "validated"
        assert result.agent_output == "agent output"
        assert result.agent is agent
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_run_async_guardrail(self):
        """OutputGuardrail.run() executes async guardrail function."""
        calls = []

        async def check_func(ctx, agent, output):
            calls.append(output)
            return GuardrailFunctionOutput(output_info="async_validated", tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=check_func)
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(ctx, agent, "async output")

        assert result.output.output_info == "async_validated"
        assert calls == ["async output"]

    @pytest.mark.asyncio
    async def test_run_returns_guardrail_reference(self):
        """OutputGuardrail.run() returns result with guardrail reference."""

        def check_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=check_func, name="out_guard")
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await guardrail.run(ctx, agent, "test")

        assert result.guardrail is guardrail
        assert result.guardrail.get_name() == "out_guard"

    @pytest.mark.asyncio
    async def test_run_non_callable_raises_error(self):
        """OutputGuardrail.run() raises UserError if function not callable."""
        guardrail = OutputGuardrail(guardrail_function=123)  # type: ignore
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        with pytest.raises(UserError):
            await guardrail.run(ctx, agent, "test")


class TestInputGuardrailDecorator:
    """Tests for input_guardrail decorator."""

    def test_decorator_without_args(self):
        """input_guardrail decorator works without parentheses."""

        @input_guardrail
        def my_guardrail(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(my_guardrail, InputGuardrail)
        assert my_guardrail.get_name() == "my_guardrail"

    def test_decorator_with_name(self):
        """input_guardrail decorator accepts name argument."""

        @input_guardrail(name="custom_checker")
        def my_guardrail(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(my_guardrail, InputGuardrail)
        assert my_guardrail.get_name() == "custom_checker"

    def test_decorator_with_async_function(self):
        """input_guardrail decorator works with async functions."""

        @input_guardrail
        async def async_guardrail(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(async_guardrail, InputGuardrail)

    @pytest.mark.asyncio
    async def test_decorated_guardrail_runs(self):
        """Decorated input guardrail can be run."""

        @input_guardrail(name="test_guard")
        def check_input(ctx, agent, input):
            return GuardrailFunctionOutput(
                output_info=f"checked: {input}",
                tripwire_triggered=False,
            )

        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await check_input.run(agent, "hello", ctx)

        assert result.output.output_info == "checked: hello"


class TestOutputGuardrailDecorator:
    """Tests for output_guardrail decorator."""

    def test_decorator_without_args(self):
        """output_guardrail decorator works without parentheses."""

        @output_guardrail
        def my_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(my_guardrail, OutputGuardrail)
        assert my_guardrail.get_name() == "my_guardrail"

    def test_decorator_with_name(self):
        """output_guardrail decorator accepts name argument."""

        @output_guardrail(name="quality_check")
        def my_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(my_guardrail, OutputGuardrail)
        assert my_guardrail.get_name() == "quality_check"

    def test_decorator_with_async_function(self):
        """output_guardrail decorator works with async functions."""

        @output_guardrail
        async def async_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        assert isinstance(async_guardrail, OutputGuardrail)

    @pytest.mark.asyncio
    async def test_decorated_guardrail_runs(self):
        """Decorated output guardrail can be run."""

        @output_guardrail(name="test_guard")
        def check_output(ctx, agent, output):
            return GuardrailFunctionOutput(
                output_info=f"validated: {output}",
                tripwire_triggered=False,
            )

        agent = Agent(name="test")
        ctx = RunContext(context=None)

        result = await check_output.run(ctx, agent, "response")

        assert result.output.output_info == "validated: response"


class TestGuardrailTripwire:
    """Tests for guardrail tripwire functionality."""

    @pytest.mark.asyncio
    async def test_input_guardrail_tripwire(self):
        """Input guardrail can trigger tripwire."""

        @input_guardrail
        def malicious_check(ctx, agent, input):
            if "attack" in str(input):
                return GuardrailFunctionOutput(
                    output_info="Malicious content detected",
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = Agent(name="test")
        ctx = RunContext(context=None)

        # Safe input
        result1 = await malicious_check.run(agent, "hello world", ctx)
        assert result1.output.tripwire_triggered is False

        # Malicious input
        result2 = await malicious_check.run(agent, "sql injection attack", ctx)
        assert result2.output.tripwire_triggered is True
        assert "Malicious" in result2.output.output_info

    @pytest.mark.asyncio
    async def test_output_guardrail_tripwire(self):
        """Output guardrail can trigger tripwire."""

        @output_guardrail
        def pii_check(ctx, agent, output):
            if "ssn" in str(output).lower():
                return GuardrailFunctionOutput(
                    output_info="PII detected",
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = Agent(name="test")
        ctx = RunContext(context=None)

        # Safe output
        result1 = await pii_check.run(ctx, agent, "Your order is confirmed")
        assert result1.output.tripwire_triggered is False

        # PII in output
        result2 = await pii_check.run(ctx, agent, "Your SSN is 123-45-6789")
        assert result2.output.tripwire_triggered is True


class TestGuardrailWithContext:
    """Tests for guardrails using context."""

    @pytest.mark.asyncio
    async def test_input_guardrail_uses_context(self):
        """Input guardrail can access context."""

        @input_guardrail
        def context_guard(ctx: RunContext[dict], agent, input):
            allowed = ctx.context.get("allowed_topics", [])
            if any(topic in str(input) for topic in allowed):
                return GuardrailFunctionOutput(output_info="allowed", tripwire_triggered=False)
            return GuardrailFunctionOutput(output_info="blocked", tripwire_triggered=True)

        agent = Agent(name="test")
        ctx = RunContext(context={"allowed_topics": ["security", "networking"]})

        result1 = await context_guard.run(agent, "security question", ctx)
        assert result1.output.tripwire_triggered is False

        result2 = await context_guard.run(agent, "random topic", ctx)
        assert result2.output.tripwire_triggered is True

    @pytest.mark.asyncio
    async def test_output_guardrail_uses_context(self):
        """Output guardrail can access context."""

        @output_guardrail
        def context_guard(ctx: RunContext[dict], agent, output):
            max_length = ctx.context.get("max_output_length", 100)
            if len(str(output)) > max_length:
                return GuardrailFunctionOutput(
                    output_info=f"Too long: {len(str(output))}",
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(output_info="ok", tripwire_triggered=False)

        agent = Agent(name="test")
        ctx = RunContext(context={"max_output_length": 20})

        result1 = await context_guard.run(ctx, agent, "short")
        assert result1.output.tripwire_triggered is False

        result2 = await context_guard.run(ctx, agent, "this is a very long response that exceeds the limit")
        assert result2.output.tripwire_triggered is True
