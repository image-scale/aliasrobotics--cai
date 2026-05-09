"""
Handoff system for agent-to-agent delegation.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

from typing_extensions import TypeAlias, TypeVar

from .run_context import RunContext, TContext

if TYPE_CHECKING:
    from .agent import Agent


THandoffInput = TypeVar("THandoffInput", default=Any)

OnHandoffWithInput = Callable[[RunContext[Any], THandoffInput], Any]
OnHandoffWithoutInput = Callable[[RunContext[Any]], Any]


@dataclass(frozen=True)
class HandoffInputData:
    """Data passed to input_filter functions during handoff."""

    input_history: str | tuple[Any, ...]
    """The input history before Runner.run() was called."""

    pre_handoff_items: tuple[Any, ...]
    """The items generated before the agent turn where the handoff was invoked."""

    new_items: tuple[Any, ...]
    """The new items generated during the current agent turn, including the
    handoff trigger item and tool output."""


HandoffInputFilter: TypeAlias = Callable[[HandoffInputData], HandoffInputData]
"""A function that filters the input data passed to the next agent."""


@dataclass
class Handoff(Generic[TContext]):
    """A handoff is when an agent delegates a task to another agent.

    For example, in a customer support scenario you might have a "triage agent"
    that determines which agent should handle the user's request, and sub-agents
    that specialize in different areas like billing, account management, etc.
    """

    tool_name: str
    """The name of the tool that represents the handoff."""

    tool_description: str
    """The description of the tool that represents the handoff."""

    input_json_schema: dict[str, Any]
    """The JSON schema for the handoff input. Can be empty if no input required."""

    on_invoke_handoff: Callable[[RunContext[Any], str], Awaitable["Agent[TContext]"]]
    """The function that invokes the handoff. Parameters:
    1. The run context
    2. The arguments from the LLM as a JSON string (empty string if no schema)

    Must return an agent.
    """

    agent_name: str
    """The name of the agent that is being handed off to."""

    input_filter: HandoffInputFilter | None = None
    """A function that filters the inputs passed to the next agent.

    By default, the new agent sees the entire conversation history. Use this
    to filter inputs, e.g., to remove older inputs or remove tools from
    existing inputs.
    """

    strict_json_schema: bool = True
    """Whether the input JSON schema is in strict mode."""

    def get_transfer_message(self, agent: "Agent[Any]") -> str:
        """Get the transfer message for this handoff."""
        return f"{{'assistant': '{agent.name}'}}"

    @classmethod
    def default_tool_name(cls, agent: "Agent[Any]") -> str:
        """Generate default tool name for a handoff.

        Converts agent name to snake_case format with 'transfer_to_' prefix.
        """
        return _transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_tool_description(cls, agent: "Agent[Any]") -> str:
        """Generate default tool description for a handoff."""
        desc = f"Handoff to the {agent.name} agent to handle the request."
        if agent.handoff_description:
            desc = f"{desc} {agent.handoff_description}"
        return desc


def _transform_string_function_style(name: str) -> str:
    """Transform a string to function-style naming (snake_case, lowercase).

    Replaces spaces and non-alphanumeric characters with underscores.
    """
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return name.lower()


def handoff(
    agent: "Agent[TContext]",
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    on_handoff: OnHandoffWithInput[THandoffInput] | OnHandoffWithoutInput | None = None,
    input_type: type[THandoffInput] | None = None,
    input_filter: HandoffInputFilter | None = None,
) -> Handoff[TContext]:
    """Create a handoff from an agent.

    Args:
        agent: The agent to handoff to.
        tool_name_override: Optional override for the handoff tool name.
        tool_description_override: Optional override for the tool description.
        on_handoff: A function that runs when the handoff is invoked.
            Can take just context, or context and input (if input_type provided).
        input_type: The type of input to the handoff. If provided, the input
            will be validated against this type.
        input_filter: A function that filters inputs passed to the next agent.

    Returns:
        A Handoff object that can be used with an Agent.
    """
    from .exceptions import ModelBehaviorError, UserError

    input_json_schema: dict[str, Any] = {}

    if input_type is not None:
        if on_handoff is None:
            raise UserError("on_handoff must be provided when input_type is specified")

        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 2:
            raise UserError("on_handoff must take two arguments: context and input")

        input_json_schema = _generate_simple_schema(input_type)
    elif on_handoff is not None:
        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 1:
            raise UserError("on_handoff must take one argument: context")

    async def _invoke_handoff(
        ctx: RunContext[Any], input_json: str
    ) -> "Agent[Any]":
        if input_type is not None:
            if not input_json:
                raise ModelBehaviorError(
                    "Handoff function expected non-null input, but got empty string"
                )

            import json
            try:
                validated_input = json.loads(input_json)
            except json.JSONDecodeError as e:
                raise ModelBehaviorError(f"Invalid JSON input for handoff: {e}")

            input_func = cast(OnHandoffWithInput[THandoffInput], on_handoff)
            result = input_func(ctx, validated_input)
            if inspect.isawaitable(result):
                await result
        elif on_handoff is not None:
            no_input_func = cast(OnHandoffWithoutInput, on_handoff)
            result = no_input_func(ctx)
            if inspect.isawaitable(result):
                await result

        return agent

    tool_name = tool_name_override or Handoff.default_tool_name(agent)
    tool_description = tool_description_override or Handoff.default_tool_description(agent)

    return Handoff(
        tool_name=tool_name,
        tool_description=tool_description,
        input_json_schema=input_json_schema,
        on_invoke_handoff=_invoke_handoff,
        input_filter=input_filter,
        agent_name=agent.name,
    )


def _generate_simple_schema(input_type: type) -> dict[str, Any]:
    """Generate a simple JSON schema for a given type."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }

    if input_type in type_map:
        return type_map[input_type]

    if hasattr(input_type, "__annotations__"):
        properties = {}
        required = []
        for field_name, field_type in input_type.__annotations__.items():
            if field_type in type_map:
                properties[field_name] = type_map[field_type]
            else:
                properties[field_name] = {"type": "string"}
            required.append(field_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    return {"type": "object"}
