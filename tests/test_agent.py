"""
Tests for the agent module.
"""

import pytest

from cyberai import Agent, FunctionTool, ModelSettings, RunContext, function_tool


class TestAgent:
    """Tests for the Agent class."""

    def test_create_agent_with_name(self):
        """Agent can be created with just a name."""
        agent = Agent(name="test_agent")

        assert agent.name == "test_agent"
        assert agent.instructions is None
        assert agent.tools == []
        assert agent.handoffs == []

    def test_create_agent_with_instructions(self):
        """Agent can be created with string instructions."""
        agent = Agent(
            name="security_agent",
            instructions="You are a cybersecurity expert.",
        )

        assert agent.name == "security_agent"
        assert agent.instructions == "You are a cybersecurity expert."

    def test_agent_with_description(self):
        """Agent can have a description."""
        agent = Agent(
            name="recon_agent",
            description="Agent for reconnaissance tasks",
        )

        assert agent.description == "Agent for reconnaissance tasks"

    def test_agent_with_handoff_description(self):
        """Agent can have a handoff description."""
        agent = Agent(
            name="exploit_agent",
            handoff_description="Handles exploitation phase",
        )

        assert agent.handoff_description == "Handles exploitation phase"

    def test_agent_with_model_settings(self):
        """Agent can have custom model settings."""
        settings = ModelSettings(temperature=0.7, max_tokens=1000)
        agent = Agent(
            name="creative_agent",
            model_settings=settings,
        )

        assert agent.model_settings.temperature == 0.7
        assert agent.model_settings.max_tokens == 1000

    def test_agent_with_model_string(self):
        """Agent can have a model specified as string."""
        agent = Agent(
            name="gpt_agent",
            model="gpt-4",
        )

        assert agent.model == "gpt-4"

    def test_agent_default_model_settings(self):
        """Agent has default empty ModelSettings."""
        agent = Agent(name="test")

        assert isinstance(agent.model_settings, ModelSettings)
        assert agent.model_settings.temperature is None

    def test_agent_reset_tool_choice_default(self):
        """Agent has reset_tool_choice defaulting to True."""
        agent = Agent(name="test")

        assert agent.reset_tool_choice is True

    def test_agent_output_type(self):
        """Agent can have a custom output type."""

        class CustomOutput:
            pass

        agent = Agent(name="test", output_type=CustomOutput)

        assert agent.output_type is CustomOutput


class TestAgentWithTools:
    """Tests for Agent tool functionality."""

    def test_agent_with_tools_list(self):
        """Agent can be created with a list of tools."""

        @function_tool
        def scan_port(port: int) -> str:
            """Scan a port."""
            return f"Port {port} is open"

        agent = Agent(
            name="scanner",
            tools=[scan_port],
        )

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "scan_port"

    def test_get_all_tools(self):
        """get_all_tools returns all agent tools."""

        @function_tool
        def tool1(x: int) -> int:
            return x

        @function_tool
        def tool2(y: str) -> str:
            return y

        agent = Agent(name="test", tools=[tool1, tool2])

        all_tools = agent.get_all_tools()

        assert len(all_tools) == 2
        assert all(isinstance(t, FunctionTool) for t in all_tools)

    def test_add_tool(self):
        """add_tool adds a tool to the agent."""

        @function_tool
        def new_tool(value: int) -> int:
            return value * 2

        agent = Agent(name="test")
        assert len(agent.tools) == 0

        agent.add_tool(new_tool)

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "new_tool"


class TestAgentHandoffs:
    """Tests for Agent handoff functionality."""

    def test_agent_with_handoffs(self):
        """Agent can be created with handoffs."""
        sub_agent = Agent(name="sub_agent")
        main_agent = Agent(
            name="main_agent",
            handoffs=[sub_agent],
        )

        assert len(main_agent.handoffs) == 1
        assert main_agent.handoffs[0].name == "sub_agent"

    def test_add_handoff(self):
        """add_handoff adds a handoff to the agent."""
        main_agent = Agent(name="main")
        sub_agent = Agent(name="sub")

        main_agent.add_handoff(sub_agent)

        assert len(main_agent.handoffs) == 1
        assert main_agent.handoffs[0] is sub_agent


class TestAgentGuardrails:
    """Tests for Agent guardrail lists."""

    def test_agent_with_input_guardrails(self):
        """Agent can have input guardrails."""

        def check_input(ctx, agent, input_data):
            return {"tripwire_triggered": False}

        agent = Agent(
            name="protected",
            input_guardrails=[check_input],
        )

        assert len(agent.input_guardrails) == 1

    def test_agent_with_output_guardrails(self):
        """Agent can have output guardrails."""

        def check_output(ctx, agent, output):
            return {"tripwire_triggered": False}

        agent = Agent(
            name="protected",
            output_guardrails=[check_output],
        )

        assert len(agent.output_guardrails) == 1


class TestAgentClone:
    """Tests for Agent.clone() method."""

    def test_clone_creates_copy(self):
        """clone() creates a copy of the agent."""
        original = Agent(
            name="original",
            instructions="Original instructions",
        )

        cloned = original.clone()

        assert cloned is not original
        assert cloned.name == "original"
        assert cloned.instructions == "Original instructions"

    def test_clone_with_overrides(self):
        """clone() can override specific fields."""
        original = Agent(
            name="original",
            instructions="Original",
            model_settings=ModelSettings(temperature=0.5),
        )

        cloned = original.clone(
            name="cloned",
            instructions="New instructions",
        )

        assert cloned.name == "cloned"
        assert cloned.instructions == "New instructions"
        # Non-overridden fields are preserved
        assert cloned.model_settings.temperature == 0.5

    def test_clone_does_not_modify_original(self):
        """clone() does not modify the original agent."""
        original = Agent(name="original", instructions="Original")

        cloned = original.clone(instructions="Modified")

        assert original.instructions == "Original"
        assert cloned.instructions == "Modified"


class TestAgentGetSystemPrompt:
    """Tests for Agent.get_system_prompt() method."""

    @pytest.mark.asyncio
    async def test_get_system_prompt_string(self):
        """get_system_prompt returns string instructions."""
        agent = Agent(
            name="test",
            instructions="You are a helpful assistant.",
        )
        ctx = RunContext(context=None)

        prompt = await agent.get_system_prompt(ctx)

        assert prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_get_system_prompt_none(self):
        """get_system_prompt returns None when no instructions."""
        agent = Agent(name="test")
        ctx = RunContext(context=None)

        prompt = await agent.get_system_prompt(ctx)

        assert prompt is None

    @pytest.mark.asyncio
    async def test_get_system_prompt_sync_callable(self):
        """get_system_prompt works with sync callable."""

        def dynamic_instructions(ctx: RunContext, agent: Agent) -> str:
            return f"Instructions for {agent.name}"

        agent = Agent(
            name="dynamic_agent",
            instructions=dynamic_instructions,
        )
        ctx = RunContext(context=None)

        prompt = await agent.get_system_prompt(ctx)

        assert prompt == "Instructions for dynamic_agent"

    @pytest.mark.asyncio
    async def test_get_system_prompt_async_callable(self):
        """get_system_prompt works with async callable."""

        async def async_instructions(ctx: RunContext, agent: Agent) -> str:
            return f"Async instructions for {agent.name}"

        agent = Agent(
            name="async_agent",
            instructions=async_instructions,
        )
        ctx = RunContext(context=None)

        prompt = await agent.get_system_prompt(ctx)

        assert prompt == "Async instructions for async_agent"

    @pytest.mark.asyncio
    async def test_get_system_prompt_with_context(self):
        """get_system_prompt callable can use context."""

        def context_aware(ctx: RunContext[dict], agent: Agent) -> str:
            target = ctx.context.get("target", "unknown")
            return f"Target is {target}"

        agent = Agent(
            name="context_agent",
            instructions=context_aware,
        )
        ctx = RunContext(context={"target": "192.168.1.1"})

        prompt = await agent.get_system_prompt(ctx)

        assert prompt == "Target is 192.168.1.1"
