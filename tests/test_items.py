"""
Tests for the items module.
"""

import pytest

from cyberai import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageItem,
    ModelResponse,
    ToolCallItem,
    ToolCallOutputItem,
    Usage,
)


class MockAgent:
    """Mock agent for testing."""
    def __init__(self, name: str = "test_agent"):
        self.name = name


class TestMessageItem:
    """Tests for MessageItem."""

    def test_create_message_item(self):
        """MessageItem can be created with content."""
        agent = MockAgent()
        item = MessageItem(
            agent=agent,
            raw_item={"role": "assistant", "content": "Hello"},
            content="Hello",
        )

        assert item.content == "Hello"
        assert item.role == "assistant"
        assert item.type == "message_item"
        assert item.agent is agent

    def test_message_item_to_input(self):
        """MessageItem can be converted to input format."""
        agent = MockAgent()
        item = MessageItem(
            agent=agent,
            raw_item={},
            content="Test message",
            role="assistant",
        )

        input_item = item.to_input_item()

        assert input_item["role"] == "assistant"
        assert input_item["content"] == "Test message"

    def test_message_item_user_role(self):
        """MessageItem can have user role."""
        agent = MockAgent()
        item = MessageItem(
            agent=agent,
            raw_item={},
            content="User input",
            role="user",
        )

        assert item.role == "user"
        assert item.to_input_item()["role"] == "user"


class TestToolCallItem:
    """Tests for ToolCallItem."""

    def test_create_tool_call_item(self):
        """ToolCallItem can be created with tool call data."""
        agent = MockAgent()
        item = ToolCallItem(
            agent=agent,
            raw_item={},
            call_id="call_123",
            name="scan_network",
            arguments='{"target": "192.168.1.1"}',
        )

        assert item.call_id == "call_123"
        assert item.name == "scan_network"
        assert item.arguments == '{"target": "192.168.1.1"}'
        assert item.type == "tool_call_item"

    def test_tool_call_to_input(self):
        """ToolCallItem can be converted to input format."""
        agent = MockAgent()
        item = ToolCallItem(
            agent=agent,
            raw_item={},
            call_id="call_456",
            name="run_exploit",
            arguments='{"exploit": "CVE-2021-1234"}',
        )

        input_item = item.to_input_item()

        assert input_item["role"] == "assistant"
        assert len(input_item["tool_calls"]) == 1
        assert input_item["tool_calls"][0]["call_id"] == "call_456"
        assert input_item["tool_calls"][0]["name"] == "run_exploit"


class TestToolCallOutputItem:
    """Tests for ToolCallOutputItem."""

    def test_create_tool_output_item(self):
        """ToolCallOutputItem can be created with output."""
        agent = MockAgent()
        item = ToolCallOutputItem(
            agent=agent,
            raw_item={},
            call_id="call_123",
            output="Port 22 is open",
        )

        assert item.call_id == "call_123"
        assert item.output == "Port 22 is open"
        assert item.type == "tool_call_output_item"

    def test_tool_output_to_input(self):
        """ToolCallOutputItem can be converted to input format."""
        agent = MockAgent()
        item = ToolCallOutputItem(
            agent=agent,
            raw_item={},
            call_id="call_789",
            output={"status": "success", "ports": [22, 80]},
        )

        input_item = item.to_input_item()

        assert input_item["role"] == "tool"
        assert input_item["tool_call_id"] == "call_789"
        assert "status" in input_item["content"]


class TestHandoffCallItem:
    """Tests for HandoffCallItem."""

    def test_create_handoff_call_item(self):
        """HandoffCallItem can be created."""
        agent = MockAgent()
        item = HandoffCallItem(
            agent=agent,
            raw_item={},
            call_id="handoff_123",
            target_agent_name="exploit_agent",
            arguments="{}",
        )

        assert item.call_id == "handoff_123"
        assert item.target_agent_name == "exploit_agent"
        assert item.type == "handoff_call_item"

    def test_handoff_call_to_input(self):
        """HandoffCallItem can be converted to input format."""
        agent = MockAgent()
        item = HandoffCallItem(
            agent=agent,
            raw_item={},
            call_id="handoff_456",
            target_agent_name="recon_agent",
            arguments='{"reason": "need more info"}',
        )

        input_item = item.to_input_item()

        assert input_item["role"] == "assistant"
        assert "transfer_to_recon_agent" in input_item["tool_calls"][0]["name"]


class TestHandoffOutputItem:
    """Tests for HandoffOutputItem."""

    def test_create_handoff_output_item(self):
        """HandoffOutputItem can be created."""
        source = MockAgent("source_agent")
        target = MockAgent("target_agent")
        item = HandoffOutputItem(
            agent=source,
            raw_item={},
            source_agent=source,
            target_agent=target,
        )

        assert item.source_agent is source
        assert item.target_agent is target
        assert item.type == "handoff_output_item"

    def test_handoff_output_to_input(self):
        """HandoffOutputItem can be converted to input format."""
        source = MockAgent("source")
        target = MockAgent("target")
        item = HandoffOutputItem(
            agent=source,
            raw_item={},
            source_agent=source,
            target_agent=target,
        )

        input_item = item.to_input_item()

        assert input_item["role"] == "tool"
        assert "target" in input_item["content"]


class TestModelResponse:
    """Tests for ModelResponse."""

    def test_create_model_response(self):
        """ModelResponse can be created."""
        usage = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)
        response = ModelResponse(
            output=[{"role": "assistant", "content": "Response"}],
            usage=usage,
        )

        assert len(response.output) == 1
        assert response.usage.total_tokens == 75
        assert response.referenceable_id is None

    def test_model_response_with_id(self):
        """ModelResponse can have a referenceable_id."""
        response = ModelResponse(
            output=[],
            usage=Usage(),
            referenceable_id="resp_12345",
        )

        assert response.referenceable_id == "resp_12345"

    def test_to_input_items_with_dicts(self):
        """ModelResponse can convert dict outputs to input items."""
        response = ModelResponse(
            output=[
                {"role": "assistant", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
            usage=Usage(),
        )

        input_items = response.to_input_items()

        assert len(input_items) == 2
        assert input_items[0]["content"] == "First"

    def test_to_input_items_with_run_items(self):
        """ModelResponse can convert RunItem outputs to input items."""
        agent = MockAgent()
        response = ModelResponse(
            output=[
                MessageItem(agent=agent, raw_item={}, content="Hello"),
                ToolCallItem(
                    agent=agent,
                    raw_item={},
                    call_id="c1",
                    name="test",
                    arguments="{}",
                ),
            ],
            usage=Usage(),
        )

        input_items = response.to_input_items()

        assert len(input_items) == 2


class TestItemHelpers:
    """Tests for ItemHelpers class."""

    def test_extract_text_from_message(self):
        """extract_text returns text from MessageItem."""
        agent = MockAgent()
        item = MessageItem(agent=agent, raw_item={}, content="Test content")

        text = ItemHelpers.extract_text(item)

        assert text == "Test content"

    def test_extract_text_from_non_message(self):
        """extract_text returns None for non-message items."""
        agent = MockAgent()
        item = ToolCallItem(
            agent=agent,
            raw_item={},
            call_id="c1",
            name="test",
            arguments="{}",
        )

        text = ItemHelpers.extract_text(item)

        assert text is None

    def test_extract_all_text(self):
        """extract_all_text concatenates all message texts."""
        agent = MockAgent()
        items = [
            MessageItem(agent=agent, raw_item={}, content="Hello "),
            ToolCallItem(agent=agent, raw_item={}, call_id="c1", name="t", arguments="{}"),
            MessageItem(agent=agent, raw_item={}, content="World"),
        ]

        text = ItemHelpers.extract_all_text(items)

        assert text == "Hello World"

    def test_input_to_new_input_list_string(self):
        """input_to_new_input_list converts string to user message."""
        result = ItemHelpers.input_to_new_input_list("Hello")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_input_to_new_input_list_list(self):
        """input_to_new_input_list returns copy of list."""
        original = [{"role": "user", "content": "Hi"}]
        result = ItemHelpers.input_to_new_input_list(original)

        assert result == original
        assert result is not original  # Should be a copy

    def test_get_tool_calls(self):
        """get_tool_calls extracts tool call items."""
        agent = MockAgent()
        items = [
            MessageItem(agent=agent, raw_item={}, content="Hi"),
            ToolCallItem(agent=agent, raw_item={}, call_id="c1", name="t1", arguments="{}"),
            ToolCallItem(agent=agent, raw_item={}, call_id="c2", name="t2", arguments="{}"),
        ]

        tool_calls = ItemHelpers.get_tool_calls(items)

        assert len(tool_calls) == 2
        assert all(isinstance(tc, ToolCallItem) for tc in tool_calls)

    def test_get_messages(self):
        """get_messages extracts message items."""
        agent = MockAgent()
        items = [
            MessageItem(agent=agent, raw_item={}, content="First"),
            ToolCallItem(agent=agent, raw_item={}, call_id="c1", name="t", arguments="{}"),
            MessageItem(agent=agent, raw_item={}, content="Second"),
        ]

        messages = ItemHelpers.get_messages(items)

        assert len(messages) == 2
        assert all(isinstance(m, MessageItem) for m in messages)

    def test_create_tool_output_item(self):
        """create_tool_output_item creates a ToolCallOutputItem."""
        agent = MockAgent()

        item = ItemHelpers.create_tool_output_item(
            agent=agent,
            call_id="call_123",
            output="Success",
        )

        assert isinstance(item, ToolCallOutputItem)
        assert item.call_id == "call_123"
        assert item.output == "Success"
        assert item.agent is agent
