"""
Run result and configuration for agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Union

from typing_extensions import TypeVar

from .guardrail import InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult
from .handoff import HandoffInputFilter
from .items import ItemHelpers, ModelResponse, RunItem
from .model import Model, ModelProvider
from .model_settings import ModelSettings

if TYPE_CHECKING:
    from .agent import Agent

T = TypeVar("T")


@dataclass
class RunConfig:
    """Configures settings for the entire agent run."""

    model: Union[str, "Model", None] = None
    """The model to use for the entire agent run. If set, will override
    the model set on every agent. The model_provider must be able to
    resolve this model name."""

    model_provider: "ModelProvider | None" = None
    """The model provider to use when looking up string model names."""

    model_settings: ModelSettings | None = None
    """Configure global model settings. Any non-null values will override
    the agent-specific model settings."""

    handoff_input_filter: HandoffInputFilter | None = None
    """A global input filter to apply to all handoffs. If Handoff.input_filter
    is set, that will take precedence."""

    input_guardrails: List["InputGuardrail[Any]"] | None = None
    """A list of input guardrails to run on the initial run input."""

    output_guardrails: List["OutputGuardrail[Any]"] | None = None
    """A list of output guardrails to run on the final output of the run."""

    max_turns: int | float = float("inf")
    """The maximum number of turns to run the agent for. A turn is one AI
    invocation including any tool calls."""

    tracing_disabled: bool = False
    """Whether tracing is disabled for the agent run."""

    trace_include_sensitive_data: bool = True
    """Whether to include potentially sensitive data in traces."""

    workflow_name: str = "Agent workflow"
    """The name of the run, used for tracing."""

    trace_id: str | None = None
    """A custom trace ID to use for tracing."""

    group_id: str | None = None
    """A grouping identifier for linking multiple traces."""

    trace_metadata: dict[str, Any] | None = None
    """Additional metadata to include with the trace."""


@dataclass
class RunResult:
    """The result of an agent run."""

    input: Union[str, List[Any]]
    """The original input items before run() was called."""

    new_items: List[RunItem]
    """The new items generated during the agent run (messages, tool calls,
    tool outputs, handoffs, etc.)."""

    raw_responses: List[ModelResponse]
    """The raw LLM responses generated during the agent run."""

    final_output: Any
    """The output of the last agent."""

    _last_agent: "Agent[Any]"
    """The last agent that was run (internal)."""

    input_guardrail_results: List[InputGuardrailResult] = field(default_factory=list)
    """Guardrail results for the input messages."""

    output_guardrail_results: List[OutputGuardrailResult] = field(default_factory=list)
    """Guardrail results for the final output of the agent."""

    @property
    def last_agent(self) -> "Agent[Any]":
        """The last agent that was run."""
        return self._last_agent

    def final_output_as(self, cls: type[T], raise_if_incorrect_type: bool = False) -> T:
        """Cast the final output to a specific type.

        Args:
            cls: The type to cast the final output to.
            raise_if_incorrect_type: If True, raise TypeError if the
                final output is not of the given type.

        Returns:
            The final output cast to the given type.
        """
        if raise_if_incorrect_type and not isinstance(self.final_output, cls):
            raise TypeError(f"Final output is not of type {cls.__name__}")
        return self.final_output  # type: ignore

    def to_input_list(self) -> List[Any]:
        """Creates a new input list, merging the original input with all
        the new items generated during the run.

        Returns:
            A list of input items suitable for a subsequent run.
        """
        original_items = ItemHelpers.input_to_new_input_list(self.input)
        new_input_items = [item.to_input_item() for item in self.new_items]
        return original_items + new_input_items
