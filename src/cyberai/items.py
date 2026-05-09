"""
Run items and model response types for agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, TypedDict, Union

from .usage import Usage

if TYPE_CHECKING:
    from .agent import Agent


class MessageContent(TypedDict, total=False):
    """Content of a message."""
    type: Literal["text", "refusal"]
    text: str
    refusal: str


class ToolCallData(TypedDict):
    """Data for a tool call."""
    call_id: str
    name: str
    arguments: str


class InputItem(TypedDict, total=False):
    """An input item for the model."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str | List[MessageContent]
    tool_calls: List[ToolCallData]
    tool_call_id: str
    name: str


@dataclass
class RunItemBase:
    """Base class for run items generated during agent execution."""

    agent: Any
    """The agent whose run caused this item to be generated."""

    raw_item: Any
    """The raw item data."""

    def to_input_item(self) -> InputItem:
        """Converts this item into an input item for the model."""
        if isinstance(self.raw_item, dict):
            return self.raw_item
        raise NotImplementedError("Subclasses must implement to_input_item")


@dataclass
class MessageItem(RunItemBase):
    """Represents a text message from the model."""

    content: str
    """The text content of the message."""

    role: Literal["assistant", "user"] = "assistant"
    """The role of the message sender."""

    type: Literal["message_item"] = "message_item"

    def to_input_item(self) -> InputItem:
        """Convert to input format."""
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class ToolCallItem(RunItemBase):
    """Represents a tool call request from the model."""

    call_id: str
    """Unique identifier for this tool call."""

    name: str
    """The name of the tool to call."""

    arguments: str
    """The JSON-encoded arguments for the tool."""

    type: Literal["tool_call_item"] = "tool_call_item"

    def to_input_item(self) -> InputItem:
        """Convert to input format."""
        return {
            "role": "assistant",
            "tool_calls": [{
                "call_id": self.call_id,
                "name": self.name,
                "arguments": self.arguments,
            }],
        }


@dataclass
class ToolCallOutputItem(RunItemBase):
    """Represents the output of a tool call."""

    call_id: str
    """The call_id of the tool call this is a response to."""

    output: Any
    """The output of the tool call."""

    type: Literal["tool_call_output_item"] = "tool_call_output_item"

    def to_input_item(self) -> InputItem:
        """Convert to input format."""
        output_str = str(self.output) if not isinstance(self.output, str) else self.output
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": output_str,
        }


@dataclass
class HandoffCallItem(RunItemBase):
    """Represents a handoff request from one agent to another."""

    call_id: str
    """Unique identifier for this handoff call."""

    target_agent_name: str
    """The name of the agent to hand off to."""

    arguments: str
    """The JSON-encoded arguments for the handoff."""

    type: Literal["handoff_call_item"] = "handoff_call_item"

    def to_input_item(self) -> InputItem:
        """Convert to input format."""
        return {
            "role": "assistant",
            "tool_calls": [{
                "call_id": self.call_id,
                "name": f"transfer_to_{self.target_agent_name}",
                "arguments": self.arguments,
            }],
        }


@dataclass
class HandoffOutputItem(RunItemBase):
    """Represents a completed handoff between agents."""

    source_agent: Any
    """The agent that initiated the handoff."""

    target_agent: Any
    """The agent that received the handoff."""

    type: Literal["handoff_output_item"] = "handoff_output_item"

    def to_input_item(self) -> InputItem:
        """Convert to input format."""
        return {
            "role": "tool",
            "content": f"Handoff to {self.target_agent.name if hasattr(self.target_agent, 'name') else str(self.target_agent)} completed",
        }


# Type alias for all run item types
RunItem = Union[
    MessageItem,
    ToolCallItem,
    ToolCallOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
]


@dataclass
class ModelResponse:
    """Response from a model invocation."""

    output: List[Any]
    """A list of output items (messages, tool calls, etc.) from the model."""

    usage: Usage
    """Token usage information for this response."""

    referenceable_id: str | None = None
    """Optional ID for referencing this response in subsequent calls."""

    def to_input_items(self) -> List[InputItem]:
        """Convert all output items to input format."""
        items = []
        for item in self.output:
            if hasattr(item, "to_input_item"):
                items.append(item.to_input_item())
            elif isinstance(item, dict):
                items.append(item)
        return items


class ItemHelpers:
    """Helper methods for working with run items."""

    @classmethod
    def extract_text(cls, item: RunItem) -> str | None:
        """Extract text content from an item if it's a message."""
        if isinstance(item, MessageItem):
            return item.content
        return None

    @classmethod
    def extract_all_text(cls, items: List[RunItem]) -> str:
        """Extract and concatenate all text from message items."""
        texts = []
        for item in items:
            text = cls.extract_text(item)
            if text:
                texts.append(text)
        return "".join(texts)

    @classmethod
    def input_to_new_input_list(
        cls,
        input_data: str | List[InputItem]
    ) -> List[InputItem]:
        """Convert input (string or list) to a list of input items."""
        if isinstance(input_data, str):
            return [{
                "role": "user",
                "content": input_data,
            }]
        return list(input_data)

    @classmethod
    def get_tool_calls(cls, items: List[RunItem]) -> List[ToolCallItem]:
        """Extract all tool call items from a list."""
        return [item for item in items if isinstance(item, ToolCallItem)]

    @classmethod
    def get_messages(cls, items: List[RunItem]) -> List[MessageItem]:
        """Extract all message items from a list."""
        return [item for item in items if isinstance(item, MessageItem)]

    @classmethod
    def create_tool_output_item(
        cls,
        agent: Any,
        call_id: str,
        output: Any
    ) -> ToolCallOutputItem:
        """Create a tool call output item."""
        return ToolCallOutputItem(
            agent=agent,
            raw_item={"call_id": call_id, "output": str(output)},
            call_id=call_id,
            output=output,
        )
