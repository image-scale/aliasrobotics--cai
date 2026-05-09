"""
Agent class for the CyberAI framework.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Union, cast

from typing_extensions import Awaitable

from .model_settings import ModelSettings
from .run_context import RunContext, TContext
from .tool import FunctionTool

if TYPE_CHECKING:
    pass


# Type alias for instructions that can be string or async/sync callable
InstructionsType = Union[
    str,
    Callable[[RunContext[Any], "Agent[Any]"], str],
    Callable[[RunContext[Any], "Agent[Any]"], Awaitable[str]],
    None,
]


@dataclass
class Agent(Generic[TContext]):
    """An agent is an AI model configured with instructions, tools, guardrails,
    handoffs, and more.

    We strongly recommend passing `instructions`, which is the "system prompt"
    for the agent. In addition, you can pass `handoff_description`, which is a
    human-readable description of the agent, used when the agent is used inside
    tools/handoffs.

    Agents are generic on the context type. The context is a (mutable) object
    you create. It is passed to tool functions, handoffs, guardrails, etc.
    """

    name: str
    """The name of the agent."""

    instructions: InstructionsType = None
    """The instructions for the agent. Will be used as the "system prompt"
    when this agent is invoked. Describes what the agent should do and how
    it responds.

    Can either be a string, or a function that dynamically generates
    instructions for the agent. If you provide a function, it will be called
    with the context and the agent instance. It must return a string.
    """

    description: str | None = None
    """A description of the agent. This is used in the CLI to show the agent's
    description."""

    handoff_description: str | None = None
    """A description of the agent. This is used when the agent is used as a
    handoff, so that an LLM knows what it does and when to invoke it."""

    handoffs: List[Any] = field(default_factory=list)
    """Handoffs are sub-agents that the agent can delegate to. You can provide
    a list of handoffs, and the agent can choose to delegate to them if
    relevant. Allows for separation of concerns and modularity."""

    model: str | Any | None = None
    """The model implementation to use when invoking the LLM.

    By default, if not set, the agent will use the default model configured
    in model_settings.
    """

    model_settings: ModelSettings = field(default_factory=ModelSettings)
    """Configures model-specific tuning parameters (e.g., temperature, top_p)."""

    tools: List[FunctionTool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    input_guardrails: List[Any] = field(default_factory=list)
    """A list of checks that run in parallel to the agent's execution, before
    generating a response. Runs only if the agent is the first agent in the
    chain."""

    output_guardrails: List[Any] = field(default_factory=list)
    """A list of checks that run on the final output of the agent, after
    generating a response. Runs only if the agent produces a final output."""

    output_type: type[Any] | None = None
    """The type of the output object. If not provided, the output will be str."""

    reset_tool_choice: bool = True
    """Whether to reset the tool choice to the default value after a tool has
    been called. Defaults to True. This ensures that the agent doesn't enter
    an infinite loop of tool usage."""

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Make a copy of the agent, with the given arguments changed.

        Example:
            new_agent = agent.clone(instructions="New instructions")
        """
        return dataclasses.replace(self, **kwargs)

    async def get_system_prompt(
        self,
        run_context: RunContext[TContext]
    ) -> str | None:
        """Get the system prompt for the agent.

        Args:
            run_context: The current run context.

        Returns:
            The system prompt string, or None if not set.
        """
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            result = self.instructions(run_context, self)
            if inspect.isawaitable(result):
                return await cast(Awaitable[str], result)
            return cast(str, result)
        return None

    def get_all_tools(self) -> List[FunctionTool]:
        """Get all tools available to this agent.

        Returns:
            A list of all function tools.
        """
        return list(self.tools)

    def add_tool(self, tool: FunctionTool) -> None:
        """Add a tool to this agent.

        Args:
            tool: The tool to add.
        """
        self.tools.append(tool)

    def add_handoff(self, handoff: Any) -> None:
        """Add a handoff to this agent.

        Args:
            handoff: The handoff (another agent or Handoff object) to add.
        """
        self.handoffs.append(handoff)
