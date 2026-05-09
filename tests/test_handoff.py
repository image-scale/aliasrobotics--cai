"""
Tests for the handoff system.
"""

import pytest

from cyberai import Agent, RunContext
from cyberai.handoff import (
    Handoff,
    HandoffInputData,
    HandoffInputFilter,
    handoff,
    _transform_string_function_style,
)
from cyberai.exceptions import ModelBehaviorError, UserError


class TestHandoffDataclass:
    """Tests for the Handoff dataclass."""

    def test_handoff_has_required_fields(self):
        """Handoff has tool_name, tool_description, agent_name, on_invoke_handoff."""

        async def invoke(ctx, args):
            return Agent(name="target")

        h = Handoff(
            tool_name="transfer_to_agent",
            tool_description="Transfer to another agent",
            input_json_schema={},
            on_invoke_handoff=invoke,
            agent_name="target_agent",
        )

        assert h.tool_name == "transfer_to_agent"
        assert h.tool_description == "Transfer to another agent"
        assert h.agent_name == "target_agent"
        assert h.on_invoke_handoff is invoke

    def test_handoff_has_input_json_schema(self):
        """Handoff has input_json_schema for passing data to target agent."""

        async def invoke(ctx, args):
            return Agent(name="target")

        schema = {"type": "object", "properties": {"reason": {"type": "string"}}}
        h = Handoff(
            tool_name="transfer",
            tool_description="Transfer",
            input_json_schema=schema,
            on_invoke_handoff=invoke,
            agent_name="target",
        )

        assert h.input_json_schema == schema

    def test_handoff_has_optional_input_filter(self):
        """Handoff has optional input_filter to modify inputs."""

        async def invoke(ctx, args):
            return Agent(name="target")

        def my_filter(data: HandoffInputData) -> HandoffInputData:
            return data

        h = Handoff(
            tool_name="transfer",
            tool_description="Transfer",
            input_json_schema={},
            on_invoke_handoff=invoke,
            agent_name="target",
            input_filter=my_filter,
        )

        assert h.input_filter is my_filter

    def test_handoff_input_filter_defaults_to_none(self):
        """Handoff input_filter defaults to None."""

        async def invoke(ctx, args):
            return Agent(name="target")

        h = Handoff(
            tool_name="transfer",
            tool_description="Transfer",
            input_json_schema={},
            on_invoke_handoff=invoke,
            agent_name="target",
        )

        assert h.input_filter is None

    def test_handoff_strict_json_schema_default(self):
        """Handoff strict_json_schema defaults to True."""

        async def invoke(ctx, args):
            return Agent(name="target")

        h = Handoff(
            tool_name="transfer",
            tool_description="Transfer",
            input_json_schema={},
            on_invoke_handoff=invoke,
            agent_name="target",
        )

        assert h.strict_json_schema is True


class TestHandoffDefaultToolName:
    """Tests for Handoff.default_tool_name()."""

    def test_default_tool_name_simple(self):
        """default_tool_name generates transfer_to_{agent_name} style names."""
        agent = Agent(name="security_agent")

        name = Handoff.default_tool_name(agent)

        assert name == "transfer_to_security_agent"

    def test_default_tool_name_with_spaces(self):
        """default_tool_name handles agent names with spaces."""
        agent = Agent(name="Security Agent")

        name = Handoff.default_tool_name(agent)

        assert name == "transfer_to_security_agent"

    def test_default_tool_name_uppercase(self):
        """default_tool_name converts to lowercase."""
        agent = Agent(name="SecurityAgent")

        name = Handoff.default_tool_name(agent)

        assert name == "transfer_to_securityagent"

    def test_default_tool_name_special_characters(self):
        """default_tool_name handles special characters."""
        agent = Agent(name="recon-agent#1")

        name = Handoff.default_tool_name(agent)

        assert name == "transfer_to_recon_agent_1"


class TestHandoffDefaultToolDescription:
    """Tests for Handoff.default_tool_description()."""

    def test_default_tool_description_basic(self):
        """default_tool_description generates description from agent name."""
        agent = Agent(name="billing_agent")

        desc = Handoff.default_tool_description(agent)

        assert "Handoff to the billing_agent agent" in desc

    def test_default_tool_description_with_handoff_description(self):
        """default_tool_description includes agent's handoff_description."""
        agent = Agent(
            name="exploit_agent",
            handoff_description="Handles exploitation phase of penetration testing",
        )

        desc = Handoff.default_tool_description(agent)

        assert "exploit_agent" in desc
        assert "Handles exploitation phase" in desc


class TestHandoffGetTransferMessage:
    """Tests for Handoff.get_transfer_message()."""

    def test_get_transfer_message(self):
        """get_transfer_message returns formatted message."""

        async def invoke(ctx, args):
            return Agent(name="target")

        h = Handoff(
            tool_name="transfer",
            tool_description="Transfer",
            input_json_schema={},
            on_invoke_handoff=invoke,
            agent_name="target",
        )

        agent = Agent(name="security_scanner")
        msg = h.get_transfer_message(agent)

        assert "security_scanner" in msg
        assert "assistant" in msg


class TestHandoffInputData:
    """Tests for HandoffInputData dataclass."""

    def test_handoff_input_data_fields(self):
        """HandoffInputData has required fields."""
        data = HandoffInputData(
            input_history="User: Hello",
            pre_handoff_items=("item1", "item2"),
            new_items=("item3",),
        )

        assert data.input_history == "User: Hello"
        assert data.pre_handoff_items == ("item1", "item2")
        assert data.new_items == ("item3",)

    def test_handoff_input_data_is_frozen(self):
        """HandoffInputData is immutable (frozen)."""
        data = HandoffInputData(
            input_history="test",
            pre_handoff_items=(),
            new_items=(),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            data.input_history = "modified"


class TestHandoffFunction:
    """Tests for the handoff() function."""

    def test_handoff_creates_from_agent(self):
        """handoff() creates a Handoff from an Agent."""
        target = Agent(name="target_agent")

        h = handoff(target)

        assert isinstance(h, Handoff)
        assert h.agent_name == "target_agent"

    def test_handoff_default_naming(self):
        """handoff() uses default naming from agent."""
        target = Agent(name="security_agent")

        h = handoff(target)

        assert h.tool_name == "transfer_to_security_agent"
        assert "security_agent" in h.tool_description

    def test_handoff_custom_tool_name(self):
        """handoff() supports custom tool name."""
        target = Agent(name="scanner")

        h = handoff(target, tool_name_override="call_scanner")

        assert h.tool_name == "call_scanner"

    def test_handoff_custom_tool_description(self):
        """handoff() supports custom tool description."""
        target = Agent(name="scanner")

        h = handoff(target, tool_description_override="Invoke the scanner agent")

        assert h.tool_description == "Invoke the scanner agent"

    def test_handoff_with_input_filter(self):
        """handoff() supports input_filter argument."""
        target = Agent(name="target")

        def my_filter(data: HandoffInputData) -> HandoffInputData:
            return data

        h = handoff(target, input_filter=my_filter)

        assert h.input_filter is my_filter

    def test_handoff_empty_input_schema_by_default(self):
        """handoff() creates empty input schema when no input_type."""
        target = Agent(name="target")

        h = handoff(target)

        assert h.input_json_schema == {}


class TestHandoffOnHandoff:
    """Tests for handoff() on_handoff callback."""

    def test_handoff_with_on_handoff_no_input(self):
        """handoff() supports on_handoff callback without input."""
        target = Agent(name="target")
        callback_called = []

        def on_handoff_callback(ctx: RunContext):
            callback_called.append(True)

        h = handoff(target, on_handoff=on_handoff_callback)

        assert h is not None
        assert h.agent_name == "target"

    def test_handoff_on_handoff_wrong_args_without_input(self):
        """on_handoff without input_type must take exactly 1 argument."""
        target = Agent(name="target")

        def bad_callback(ctx, extra):
            pass

        with pytest.raises(UserError):
            handoff(target, on_handoff=bad_callback)

    def test_handoff_with_input_type(self):
        """handoff() supports on_handoff with input_type."""
        target = Agent(name="target")

        def on_handoff_callback(ctx: RunContext, data: dict):
            pass

        h = handoff(target, on_handoff=on_handoff_callback, input_type=dict)

        assert h is not None
        assert h.input_json_schema != {}

    def test_handoff_input_type_without_on_handoff_raises(self):
        """input_type without on_handoff raises UserError."""
        target = Agent(name="target")

        with pytest.raises(UserError):
            handoff(target, input_type=str)

    def test_handoff_on_handoff_with_input_wrong_args(self):
        """on_handoff with input_type must take exactly 2 arguments."""
        target = Agent(name="target")

        def bad_callback(ctx):
            pass

        with pytest.raises(UserError):
            handoff(target, on_handoff=bad_callback, input_type=str)


class TestHandoffInvoke:
    """Tests for invoking handoffs."""

    @pytest.mark.asyncio
    async def test_invoke_handoff_returns_target_agent(self):
        """Invoking a handoff returns the target agent."""
        target = Agent(name="target_agent")
        h = handoff(target)

        ctx = RunContext(context=None)
        result = await h.on_invoke_handoff(ctx, "")

        assert result is target
        assert result.name == "target_agent"

    @pytest.mark.asyncio
    async def test_invoke_handoff_calls_on_handoff(self):
        """Invoking a handoff calls the on_handoff callback."""
        target = Agent(name="target")
        callback_values = []

        def on_handoff_callback(ctx: RunContext):
            callback_values.append("called")

        h = handoff(target, on_handoff=on_handoff_callback)

        ctx = RunContext(context=None)
        result = await h.on_invoke_handoff(ctx, "")

        assert result is target
        assert callback_values == ["called"]

    @pytest.mark.asyncio
    async def test_invoke_handoff_async_on_handoff(self):
        """Invoking a handoff works with async on_handoff callback."""
        target = Agent(name="target")
        callback_values = []

        async def async_callback(ctx: RunContext):
            callback_values.append("async_called")

        h = handoff(target, on_handoff=async_callback)

        ctx = RunContext(context=None)
        result = await h.on_invoke_handoff(ctx, "")

        assert result is target
        assert callback_values == ["async_called"]

    @pytest.mark.asyncio
    async def test_invoke_handoff_with_input(self):
        """Invoking a handoff with input parses JSON and calls callback."""
        target = Agent(name="target")
        received_data = []

        def on_handoff_callback(ctx: RunContext, data: dict):
            received_data.append(data)

        h = handoff(target, on_handoff=on_handoff_callback, input_type=dict)

        ctx = RunContext(context=None)
        result = await h.on_invoke_handoff(ctx, '{"reason": "needs specialist"}')

        assert result is target
        assert received_data == [{"reason": "needs specialist"}]

    @pytest.mark.asyncio
    async def test_invoke_handoff_with_input_invalid_json(self):
        """Invoking a handoff with invalid JSON raises ModelBehaviorError."""
        target = Agent(name="target")

        def on_handoff_callback(ctx: RunContext, data: dict):
            pass

        h = handoff(target, on_handoff=on_handoff_callback, input_type=dict)

        ctx = RunContext(context=None)
        with pytest.raises(ModelBehaviorError):
            await h.on_invoke_handoff(ctx, "not valid json")

    @pytest.mark.asyncio
    async def test_invoke_handoff_with_input_empty_string_raises(self):
        """Invoking a handoff expecting input with empty string raises error."""
        target = Agent(name="target")

        def on_handoff_callback(ctx: RunContext, data: dict):
            pass

        h = handoff(target, on_handoff=on_handoff_callback, input_type=dict)

        ctx = RunContext(context=None)
        with pytest.raises(ModelBehaviorError):
            await h.on_invoke_handoff(ctx, "")


class TestTransformStringFunctionStyle:
    """Tests for _transform_string_function_style helper."""

    def test_simple_string(self):
        """Simple string is lowercased."""
        result = _transform_string_function_style("transfer_to_agent")
        assert result == "transfer_to_agent"

    def test_spaces_to_underscores(self):
        """Spaces are converted to underscores."""
        result = _transform_string_function_style("transfer to agent")
        assert result == "transfer_to_agent"

    def test_special_chars_to_underscores(self):
        """Special characters are converted to underscores."""
        result = _transform_string_function_style("agent-1#test")
        assert result == "agent_1_test"

    def test_uppercase_to_lowercase(self):
        """Uppercase is converted to lowercase."""
        result = _transform_string_function_style("TransferToAgent")
        assert result == "transfertoagent"


class TestInputJsonSchemaGeneration:
    """Tests for input JSON schema generation."""

    def test_simple_type_string(self):
        """Simple string type generates string schema."""
        target = Agent(name="target")

        def callback(ctx, data: str):
            pass

        h = handoff(target, on_handoff=callback, input_type=str)

        assert h.input_json_schema == {"type": "string"}

    def test_simple_type_int(self):
        """Simple int type generates integer schema."""
        target = Agent(name="target")

        def callback(ctx, data: int):
            pass

        h = handoff(target, on_handoff=callback, input_type=int)

        assert h.input_json_schema == {"type": "integer"}

    def test_simple_type_bool(self):
        """Simple bool type generates boolean schema."""
        target = Agent(name="target")

        def callback(ctx, data: bool):
            pass

        h = handoff(target, on_handoff=callback, input_type=bool)

        assert h.input_json_schema == {"type": "boolean"}
