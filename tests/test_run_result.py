"""
Tests for RunResult and RunConfig.
"""

import pytest

from cyberai import Agent, ModelSettings
from cyberai.guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)
from cyberai.handoff import HandoffInputData
from cyberai.items import MessageItem, ModelResponse, ToolCallItem, ToolCallOutputItem
from cyberai.model import Model, ModelProvider
from cyberai.run_result import RunConfig, RunResult
from cyberai.usage import Usage


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_run_config_default_values(self):
        """RunConfig has sensible defaults."""
        config = RunConfig()

        assert config.model is None
        assert config.model_provider is None
        assert config.model_settings is None
        assert config.handoff_input_filter is None
        assert config.input_guardrails is None
        assert config.output_guardrails is None
        assert config.max_turns == float("inf")
        assert config.tracing_disabled is False

    def test_run_config_with_model_string(self):
        """RunConfig can have a model as string."""
        config = RunConfig(model="gpt-4")

        assert config.model == "gpt-4"

    def test_run_config_with_model_instance(self):
        """RunConfig can have a Model instance."""

        class MockModel(Model):
            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        model = MockModel()
        config = RunConfig(model=model)

        assert config.model is model
        assert isinstance(config.model, Model)

    def test_run_config_with_model_provider(self):
        """RunConfig can have a ModelProvider."""

        class MockModel(Model):
            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        class MockProvider(ModelProvider):
            def get_model(self, model_name):
                return MockModel()

        provider = MockProvider()
        config = RunConfig(model_provider=provider)

        assert config.model_provider is provider

    def test_run_config_with_model_settings(self):
        """RunConfig can have ModelSettings."""
        settings = ModelSettings(temperature=0.5, max_tokens=100)
        config = RunConfig(model_settings=settings)

        assert config.model_settings is settings
        assert config.model_settings.temperature == 0.5

    def test_run_config_with_handoff_input_filter(self):
        """RunConfig can have handoff_input_filter."""

        def my_filter(data: HandoffInputData) -> HandoffInputData:
            return data

        config = RunConfig(handoff_input_filter=my_filter)

        assert config.handoff_input_filter is my_filter

    def test_run_config_with_input_guardrails(self):
        """RunConfig can have input_guardrails list."""

        def guard_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=guard_func)
        config = RunConfig(input_guardrails=[guardrail])

        assert config.input_guardrails is not None
        assert len(config.input_guardrails) == 1
        assert config.input_guardrails[0] is guardrail

    def test_run_config_with_output_guardrails(self):
        """RunConfig can have output_guardrails list."""

        def guard_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=guard_func)
        config = RunConfig(output_guardrails=[guardrail])

        assert config.output_guardrails is not None
        assert len(config.output_guardrails) == 1
        assert config.output_guardrails[0] is guardrail

    def test_run_config_with_max_turns(self):
        """RunConfig can have max_turns."""
        config = RunConfig(max_turns=10)

        assert config.max_turns == 10

    def test_run_config_tracing_disabled(self):
        """RunConfig can disable tracing."""
        config = RunConfig(tracing_disabled=True)

        assert config.tracing_disabled is True

    def test_run_config_trace_include_sensitive_data(self):
        """RunConfig has trace_include_sensitive_data flag."""
        config = RunConfig(trace_include_sensitive_data=False)

        assert config.trace_include_sensitive_data is False

    def test_run_config_workflow_name(self):
        """RunConfig has workflow_name."""
        config = RunConfig(workflow_name="Security scan workflow")

        assert config.workflow_name == "Security scan workflow"

    def test_run_config_trace_id(self):
        """RunConfig can have custom trace_id."""
        config = RunConfig(trace_id="custom-trace-123")

        assert config.trace_id == "custom-trace-123"

    def test_run_config_group_id(self):
        """RunConfig can have group_id."""
        config = RunConfig(group_id="session-456")

        assert config.group_id == "session-456"

    def test_run_config_trace_metadata(self):
        """RunConfig can have trace_metadata."""
        metadata = {"user": "test", "env": "dev"}
        config = RunConfig(trace_metadata=metadata)

        assert config.trace_metadata == metadata


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_run_result_fields(self):
        """RunResult has required fields."""
        agent = Agent(name="test_agent")

        result = RunResult(
            input="Hello",
            new_items=[],
            raw_responses=[],
            final_output="Response",
            _last_agent=agent,
        )

        assert result.input == "Hello"
        assert result.new_items == []
        assert result.raw_responses == []
        assert result.final_output == "Response"

    def test_run_result_last_agent_property(self):
        """RunResult.last_agent returns the last agent."""
        agent = Agent(name="security_scanner")

        result = RunResult(
            input="scan",
            new_items=[],
            raw_responses=[],
            final_output="done",
            _last_agent=agent,
        )

        assert result.last_agent is agent
        assert result.last_agent.name == "security_scanner"

    def test_run_result_with_items(self):
        """RunResult can have new_items."""
        agent = Agent(name="test")
        items = [
            MessageItem(agent=agent, raw_item={}, role="assistant", content="Hello"),
            ToolCallItem(agent=agent, raw_item={}, call_id="1", name="scan", arguments="{}"),
            ToolCallOutputItem(agent=agent, raw_item={}, call_id="1", output="done"),
        ]

        result = RunResult(
            input="test",
            new_items=items,
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
        )

        assert len(result.new_items) == 3
        assert isinstance(result.new_items[0], MessageItem)
        assert isinstance(result.new_items[1], ToolCallItem)

    def test_run_result_with_raw_responses(self):
        """RunResult can have raw_responses."""
        agent = Agent(name="test")
        responses = [
            ModelResponse(output=[], usage=Usage(input_tokens=10)),
            ModelResponse(output=[], usage=Usage(input_tokens=20)),
        ]

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=responses,
            final_output="output",
            _last_agent=agent,
        )

        assert len(result.raw_responses) == 2
        assert result.raw_responses[0].usage.input_tokens == 10

    def test_run_result_input_guardrail_results(self):
        """RunResult has input_guardrail_results."""
        agent = Agent(name="test")

        def guard_func(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=guard_func)
        guard_output = GuardrailFunctionOutput(output_info="ok", tripwire_triggered=False)
        guard_result = InputGuardrailResult(guardrail=guardrail, output=guard_output)

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
            input_guardrail_results=[guard_result],
        )

        assert len(result.input_guardrail_results) == 1
        assert result.input_guardrail_results[0].output.tripwire_triggered is False

    def test_run_result_output_guardrail_results(self):
        """RunResult has output_guardrail_results."""
        agent = Agent(name="test")

        def guard_func(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=guard_func)
        guard_output = GuardrailFunctionOutput(output_info="valid", tripwire_triggered=False)
        guard_result = OutputGuardrailResult(
            guardrail=guardrail,
            agent=agent,
            agent_output="output",
            output=guard_output,
        )

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
            output_guardrail_results=[guard_result],
        )

        assert len(result.output_guardrail_results) == 1
        assert result.output_guardrail_results[0].agent is agent


class TestRunResultFinalOutputAs:
    """Tests for RunResult.final_output_as() method."""

    def test_final_output_as_type(self):
        """final_output_as() casts output to specified type."""
        agent = Agent(name="test")

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output="string output",
            _last_agent=agent,
        )

        output = result.final_output_as(str)
        assert output == "string output"

    def test_final_output_as_with_dict(self):
        """final_output_as() works with dict."""
        agent = Agent(name="test")

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output={"key": "value"},
            _last_agent=agent,
        )

        output = result.final_output_as(dict)
        assert output == {"key": "value"}

    def test_final_output_as_raise_on_incorrect_type(self):
        """final_output_as() raises TypeError when type doesn't match."""
        agent = Agent(name="test")

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output="string",
            _last_agent=agent,
        )

        with pytest.raises(TypeError):
            result.final_output_as(int, raise_if_incorrect_type=True)

    def test_final_output_as_no_raise_by_default(self):
        """final_output_as() doesn't raise by default."""
        agent = Agent(name="test")

        result = RunResult(
            input="test",
            new_items=[],
            raw_responses=[],
            final_output="string",
            _last_agent=agent,
        )

        # Should not raise, just returns the output
        output = result.final_output_as(int)
        assert output == "string"


class TestRunResultToInputList:
    """Tests for RunResult.to_input_list() method."""

    def test_to_input_list_empty(self):
        """to_input_list() with no new items returns original input."""
        agent = Agent(name="test")

        result = RunResult(
            input="Hello",
            new_items=[],
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
        )

        input_list = result.to_input_list()

        assert len(input_list) == 1
        assert input_list[0]["role"] == "user"
        assert input_list[0]["content"] == "Hello"

    def test_to_input_list_with_items(self):
        """to_input_list() includes new items."""
        agent = Agent(name="test")
        items = [
            MessageItem(agent=agent, raw_item={}, role="assistant", content="Hi"),
        ]

        result = RunResult(
            input="Hello",
            new_items=items,
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
        )

        input_list = result.to_input_list()

        assert len(input_list) == 2
        assert input_list[0]["role"] == "user"
        assert input_list[1]["role"] == "assistant"

    def test_to_input_list_with_list_input(self):
        """to_input_list() works with list input."""
        agent = Agent(name="test")
        original_input = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
        ]

        result = RunResult(
            input=original_input,
            new_items=[MessageItem(agent=agent, raw_item={}, role="assistant", content="Final")],
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
        )

        input_list = result.to_input_list()

        assert len(input_list) == 3
        assert input_list[0]["content"] == "First"
        assert input_list[2]["content"] == "Final"

    def test_to_input_list_preserves_order(self):
        """to_input_list() preserves item order."""
        agent = Agent(name="test")
        items = [
            MessageItem(agent=agent, raw_item={}, role="assistant", content="Response 1"),
            ToolCallItem(agent=agent, raw_item={}, call_id="1", name="tool", arguments="{}"),
            ToolCallOutputItem(agent=agent, raw_item={}, call_id="1", output="result"),
            MessageItem(agent=agent, raw_item={}, role="assistant", content="Response 2"),
        ]

        result = RunResult(
            input="Start",
            new_items=items,
            raw_responses=[],
            final_output="done",
            _last_agent=agent,
        )

        input_list = result.to_input_list()

        # Original input + 4 new items
        assert len(input_list) == 5
        assert input_list[0]["role"] == "user"
        assert input_list[1]["role"] == "assistant"
        assert input_list[1]["content"] == "Response 1"


class TestRunResultListInput:
    """Tests for RunResult with list inputs."""

    def test_run_result_with_list_input(self):
        """RunResult works with list input."""
        agent = Agent(name="test")
        list_input = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        result = RunResult(
            input=list_input,
            new_items=[],
            raw_responses=[],
            final_output="output",
            _last_agent=agent,
        )

        assert result.input == list_input
        assert len(result.input) == 2
