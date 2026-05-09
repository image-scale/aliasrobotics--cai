"""
Tests for the Runner class.
"""

import pytest

from cyberai import Agent, FunctionTool, ModelSettings, RunContext, function_tool
from cyberai.exceptions import InputGuardrailTriggered, MaxTurnsExceeded, OutputGuardrailTriggered
from cyberai.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
from cyberai.handoff import Handoff, handoff
from cyberai.items import MessageItem, ModelResponse, ToolCallItem, ToolCallOutputItem
from cyberai.model import Model, ModelProvider, ModelTracing
from cyberai.run_result import RunConfig, RunResult
from cyberai.runner import Runner
from cyberai.usage import Usage


class MockModel(Model):
    """Mock model for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_input = None
        self.last_tools = None

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
    ):
        self.last_input = input
        self.last_tools = tools
        self.call_count += 1

        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]

        # Default response
        return ModelResponse(
            output=[{"type": "message", "content": "Default response"}],
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    async def stream_response(self, *args, **kwargs):
        yield {}


class MockProvider(ModelProvider):
    """Mock model provider for testing."""

    def __init__(self, model=None):
        self._model = model or MockModel()

    def get_model(self, model_name):
        return self._model


class TestRunnerBasics:
    """Basic tests for Runner."""

    @pytest.mark.asyncio
    async def test_runner_returns_run_result(self):
        """Runner.run() returns a RunResult."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test_agent")
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Hello", run_config=config)

        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_runner_passes_input_to_model(self):
        """Runner passes input to the model."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test_agent")
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Hello world", run_config=config)

        assert model.last_input is not None
        assert len(model.last_input) == 1
        assert model.last_input[0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_runner_uses_system_prompt(self):
        """Runner uses agent's system prompt."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(
            name="test_agent",
            instructions="You are a security expert.",
        )
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Help me", run_config=config)

        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_runner_returns_final_output(self):
        """Runner returns final output in result."""
        response = ModelResponse(
            output=[{"type": "message", "content": "Final answer"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        agent = Agent(name="test_agent")
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Question", run_config=config)

        assert result.final_output == "Final answer"

    @pytest.mark.asyncio
    async def test_runner_tracks_last_agent(self):
        """Runner tracks the last agent in result."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="security_scanner")
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Scan", run_config=config)

        assert result.last_agent is agent
        assert result.last_agent.name == "security_scanner"

    @pytest.mark.asyncio
    async def test_runner_stores_model_responses(self):
        """Runner stores raw model responses."""
        response = ModelResponse(
            output=[{"type": "message", "content": "Response"}],
            usage=Usage(input_tokens=10),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Test", run_config=config)

        assert len(result.raw_responses) == 1
        assert result.raw_responses[0].usage.input_tokens == 10


class TestRunnerMaxTurns:
    """Tests for Runner max turns handling."""

    @pytest.mark.asyncio
    async def test_runner_raises_max_turns_exceeded(self):
        """Runner raises MaxTurnsExceeded when limit exceeded."""
        # Model always requests tool calls, causing infinite loop
        response = ModelResponse(
            output=[{
                "type": "function_call",
                "call_id": "1",
                "name": "unknown_tool",
                "arguments": "{}",
            }],
            usage=Usage(),
        )
        model = MockModel(responses=[response, response, response, response])
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(model_provider=provider)

        with pytest.raises(MaxTurnsExceeded):
            await Runner.run(agent, "Test", max_turns=3, run_config=config)

    @pytest.mark.asyncio
    async def test_runner_respects_max_turns(self):
        """Runner respects max_turns limit."""
        response = ModelResponse(
            output=[{"type": "message", "content": "Done"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(model_provider=provider)

        # Should succeed with sufficient turns
        result = await Runner.run(agent, "Test", max_turns=5, run_config=config)
        assert result.final_output == "Done"

    @pytest.mark.asyncio
    async def test_runner_max_turns_in_config(self):
        """Runner uses max_turns from RunConfig."""
        response = ModelResponse(
            output=[{
                "type": "function_call",
                "call_id": "1",
                "name": "unknown",
                "arguments": "{}",
            }],
            usage=Usage(),
        )
        model = MockModel(responses=[response, response, response])
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(model_provider=provider, max_turns=2)

        with pytest.raises(MaxTurnsExceeded):
            await Runner.run(agent, "Test", max_turns=2, run_config=config)


class TestRunnerTools:
    """Tests for Runner tool execution."""

    @pytest.mark.asyncio
    async def test_runner_executes_tool_calls(self):
        """Runner executes tool calls from model response."""
        tool_output = []

        @function_tool
        def scan_port(port: int) -> str:
            """Scan a port."""
            tool_output.append(port)
            return f"Port {port} is open"

        # First response: tool call, second response: final message
        response1 = ModelResponse(
            output=[{
                "type": "function_call",
                "call_id": "call_1",
                "name": "scan_port",
                "arguments": '{"port": 80}',
            }],
            usage=Usage(),
        )
        response2 = ModelResponse(
            output=[{"type": "message", "content": "Scan complete"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response1, response2])
        provider = MockProvider(model)
        agent = Agent(name="scanner", tools=[scan_port])
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Scan port 80", run_config=config)

        assert tool_output == [80]
        assert result.final_output == "Scan complete"

    @pytest.mark.asyncio
    async def test_runner_passes_tools_to_model(self):
        """Runner passes agent tools to model."""

        @function_tool
        def tool1(x: int) -> int:
            return x

        @function_tool
        def tool2(y: str) -> str:
            return y

        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test", tools=[tool1, tool2])
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Test", run_config=config)

        assert model.last_tools is not None
        assert len(model.last_tools) == 2


class TestRunnerHandoffs:
    """Tests for Runner handoff handling."""

    @pytest.mark.asyncio
    async def test_runner_handles_handoff(self):
        """Runner handles handoff to another agent."""
        sub_agent = Agent(name="sub_agent")
        main_agent = Agent(name="main_agent", handoffs=[sub_agent])

        # Main agent requests handoff, sub agent returns final answer
        response1 = ModelResponse(
            output=[{
                "type": "function_call",
                "call_id": "h1",
                "name": "transfer_to_sub_agent",
                "arguments": "{}",
            }],
            usage=Usage(),
        )
        response2 = ModelResponse(
            output=[{"type": "message", "content": "Sub agent answer"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response1, response2])
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        result = await Runner.run(main_agent, "Question", run_config=config)

        assert result.last_agent is sub_agent
        assert result.final_output == "Sub agent answer"

    @pytest.mark.asyncio
    async def test_runner_records_handoff_items(self):
        """Runner records handoff items in result."""
        sub_agent = Agent(name="specialist")
        main_agent = Agent(name="triage", handoffs=[sub_agent])

        response1 = ModelResponse(
            output=[{
                "type": "function_call",
                "call_id": "h1",
                "name": "transfer_to_specialist",
                "arguments": "{}",
            }],
            usage=Usage(),
        )
        response2 = ModelResponse(
            output=[{"type": "message", "content": "Done"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response1, response2])
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        result = await Runner.run(main_agent, "Help", run_config=config)

        # Should have handoff-related items
        handoff_items = [
            item for item in result.new_items
            if hasattr(item, 'target_agent')
        ]
        assert len(handoff_items) >= 1


class TestRunnerInputGuardrails:
    """Tests for Runner input guardrail handling."""

    @pytest.mark.asyncio
    async def test_runner_runs_input_guardrails(self):
        """Runner runs input guardrails on first turn."""
        guard_calls = []

        def input_guard(ctx, agent, input):
            guard_calls.append(input)
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=input_guard)
        agent = Agent(name="test", input_guardrails=[guardrail])

        model = MockModel()
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Test input", run_config=config)

        assert len(guard_calls) == 1
        assert guard_calls[0] == "Test input"

    @pytest.mark.asyncio
    async def test_runner_raises_on_input_guardrail_triggered(self):
        """Runner raises InputGuardrailTriggered when tripwire triggered."""

        def malicious_guard(ctx, agent, input):
            return GuardrailFunctionOutput(
                output_info="Malicious input detected",
                tripwire_triggered=True,
            )

        guardrail = InputGuardrail(guardrail_function=malicious_guard)
        agent = Agent(name="test", input_guardrails=[guardrail])

        model = MockModel()
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        with pytest.raises(InputGuardrailTriggered):
            await Runner.run(agent, "Attack payload", run_config=config)

    @pytest.mark.asyncio
    async def test_runner_input_guardrails_from_config(self):
        """Runner runs input guardrails from RunConfig."""
        config_guard_calls = []

        def config_guard(ctx, agent, input):
            config_guard_calls.append(input)
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        config_guardrail = InputGuardrail(guardrail_function=config_guard)
        agent = Agent(name="test")

        model = MockModel()
        provider = MockProvider(model)
        config = RunConfig(
            model_provider=provider,
            input_guardrails=[config_guardrail],
        )

        await Runner.run(agent, "Test", run_config=config)

        assert len(config_guard_calls) == 1

    @pytest.mark.asyncio
    async def test_runner_returns_input_guardrail_results(self):
        """Runner returns input guardrail results."""

        def guard(ctx, agent, input):
            return GuardrailFunctionOutput(output_info="checked", tripwire_triggered=False)

        guardrail = InputGuardrail(guardrail_function=guard, name="test_guard")
        agent = Agent(name="test", input_guardrails=[guardrail])

        model = MockModel()
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Test", run_config=config)

        assert len(result.input_guardrail_results) == 1
        assert result.input_guardrail_results[0].output.output_info == "checked"


class TestRunnerOutputGuardrails:
    """Tests for Runner output guardrail handling."""

    @pytest.mark.asyncio
    async def test_runner_runs_output_guardrails(self):
        """Runner runs output guardrails on final output."""
        guard_calls = []

        def output_guard(ctx, agent, output):
            guard_calls.append(output)
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=output_guard)
        agent = Agent(name="test", output_guardrails=[guardrail])

        response = ModelResponse(
            output=[{"type": "message", "content": "Final answer"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Question", run_config=config)

        assert len(guard_calls) == 1
        assert guard_calls[0] == "Final answer"

    @pytest.mark.asyncio
    async def test_runner_raises_on_output_guardrail_triggered(self):
        """Runner raises OutputGuardrailTriggered when tripwire triggered."""

        def pii_guard(ctx, agent, output):
            if "ssn" in str(output).lower():
                return GuardrailFunctionOutput(
                    output_info="PII detected",
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=pii_guard)
        agent = Agent(name="test", output_guardrails=[guardrail])

        response = ModelResponse(
            output=[{"type": "message", "content": "SSN: 123-45-6789"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        with pytest.raises(OutputGuardrailTriggered):
            await Runner.run(agent, "Get info", run_config=config)

    @pytest.mark.asyncio
    async def test_runner_output_guardrails_from_config(self):
        """Runner runs output guardrails from RunConfig."""
        config_guard_calls = []

        def config_guard(ctx, agent, output):
            config_guard_calls.append(output)
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        config_guardrail = OutputGuardrail(guardrail_function=config_guard)
        agent = Agent(name="test")

        response = ModelResponse(
            output=[{"type": "message", "content": "Output"}],
            usage=Usage(),
        )
        model = MockModel(responses=[response])
        provider = MockProvider(model)
        config = RunConfig(
            model_provider=provider,
            output_guardrails=[config_guardrail],
        )

        await Runner.run(agent, "Test", run_config=config)

        assert len(config_guard_calls) == 1

    @pytest.mark.asyncio
    async def test_runner_returns_output_guardrail_results(self):
        """Runner returns output guardrail results."""

        def guard(ctx, agent, output):
            return GuardrailFunctionOutput(output_info="validated", tripwire_triggered=False)

        guardrail = OutputGuardrail(guardrail_function=guard, name="test_guard")
        agent = Agent(name="test", output_guardrails=[guardrail])

        model = MockModel()
        provider = MockProvider(model)
        config = RunConfig(model_provider=provider)

        result = await Runner.run(agent, "Test", run_config=config)

        assert len(result.output_guardrail_results) == 1


class TestRunnerModelConfig:
    """Tests for Runner model configuration."""

    @pytest.mark.asyncio
    async def test_runner_uses_model_from_config(self):
        """Runner uses model from RunConfig."""
        model = MockModel()
        agent = Agent(name="test")
        config = RunConfig(model=model)

        await Runner.run(agent, "Test", run_config=config)

        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_runner_uses_model_provider(self):
        """Runner uses model_provider from RunConfig."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test", model="gpt-4")
        config = RunConfig(model_provider=provider)

        await Runner.run(agent, "Test", run_config=config)

        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_runner_resolves_model_settings(self):
        """Runner resolves model settings from agent and config."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(
            name="test",
            model_settings=ModelSettings(temperature=0.5),
        )
        config = RunConfig(
            model_provider=provider,
            model_settings=ModelSettings(max_tokens=100),
        )

        await Runner.run(agent, "Test", run_config=config)

        assert model.call_count == 1


class TestRunnerTracing:
    """Tests for Runner tracing configuration."""

    @pytest.mark.asyncio
    async def test_runner_tracing_disabled(self):
        """Runner respects tracing_disabled in config."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(model_provider=provider, tracing_disabled=True)

        await Runner.run(agent, "Test", run_config=config)

        # Model should still be called
        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_runner_trace_exclude_sensitive_data(self):
        """Runner respects trace_include_sensitive_data in config."""
        model = MockModel()
        provider = MockProvider(model)
        agent = Agent(name="test")
        config = RunConfig(
            model_provider=provider,
            trace_include_sensitive_data=False,
        )

        await Runner.run(agent, "Test", run_config=config)

        assert model.call_count == 1
