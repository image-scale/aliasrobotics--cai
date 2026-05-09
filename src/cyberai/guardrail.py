"""
Input and output guardrails for agent safety validation.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Union, overload

from typing_extensions import TypeVar

from .exceptions import UserError
from .run_context import RunContext, TContext

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function."""

    output_info: Any
    """Optional information about the guardrail's output. Can include details
    about the checks performed and granular results."""

    tripwire_triggered: bool
    """Whether the tripwire was triggered. If triggered, agent execution
    will be halted."""


@dataclass
class InputGuardrailResult:
    """The result of running an input guardrail."""

    guardrail: InputGuardrail[Any]
    """The guardrail that was run."""

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class OutputGuardrailResult:
    """The result of running an output guardrail."""

    guardrail: OutputGuardrail[Any]
    """The guardrail that was run."""

    agent_output: Any
    """The output of the agent that was checked by the guardrail."""

    agent: "Agent[Any]"
    """The agent that was checked by the guardrail."""

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class InputGuardrail(Generic[TContext]):
    """Input guardrails are checks that run in parallel to agent execution.

    They can be used to:
    - Check if input messages are off-topic
    - Take over control if unexpected input is detected
    - Validate input before processing

    Use the @input_guardrail decorator to turn a function into an InputGuardrail,
    or create an InputGuardrail manually.

    Guardrails return a GuardrailFunctionOutput. If tripwire_triggered is True,
    agent execution stops and InputGuardrailTriggered is raised.
    """

    guardrail_function: Callable[
        [RunContext[TContext], "Agent[Any]", Union[str, list[Any]]],
        Union[GuardrailFunctionOutput, Awaitable[GuardrailFunctionOutput]],
    ]
    """A function that receives the context, agent, and input, and returns
    a GuardrailFunctionOutput."""

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, the
    guardrail function's name will be used."""

    def get_name(self) -> str:
        """Get the name of this guardrail."""
        if self.name:
            return self.name
        return self.guardrail_function.__name__

    async def run(
        self,
        agent: "Agent[Any]",
        input: Union[str, list[Any]],
        context: RunContext[TContext],
    ) -> InputGuardrailResult:
        """Run this guardrail on the given input.

        Args:
            agent: The agent being guarded.
            input: The input to check (string or list of input items).
            context: The current run context.

        Returns:
            InputGuardrailResult containing the guardrail and its output.

        Raises:
            UserError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        output = self.guardrail_function(context, agent, input)
        if inspect.isawaitable(output):
            return InputGuardrailResult(
                guardrail=self,
                output=await output,
            )

        return InputGuardrailResult(
            guardrail=self,
            output=output,
        )


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Output guardrails are checks that run on the final output of an agent.

    They can be used to:
    - Check if output passes validation criteria
    - Ensure output meets quality standards
    - Detect potentially harmful content

    Use the @output_guardrail decorator to turn a function into an OutputGuardrail,
    or create an OutputGuardrail manually.

    Guardrails return a GuardrailFunctionOutput. If tripwire_triggered is True,
    OutputGuardrailTriggered is raised.
    """

    guardrail_function: Callable[
        [RunContext[TContext], "Agent[Any]", Any],
        Union[GuardrailFunctionOutput, Awaitable[GuardrailFunctionOutput]],
    ]
    """A function that receives the context, agent, and output, and returns
    a GuardrailFunctionOutput."""

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, the
    guardrail function's name will be used."""

    def get_name(self) -> str:
        """Get the name of this guardrail."""
        if self.name:
            return self.name
        return self.guardrail_function.__name__

    async def run(
        self,
        context: RunContext[TContext],
        agent: "Agent[Any]",
        agent_output: Any,
    ) -> OutputGuardrailResult:
        """Run this guardrail on the given output.

        Args:
            context: The current run context.
            agent: The agent that produced the output.
            agent_output: The output to check.

        Returns:
            OutputGuardrailResult containing guardrail, agent, output, and result.

        Raises:
            UserError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        output = self.guardrail_function(context, agent, agent_output)
        if inspect.isawaitable(output):
            return OutputGuardrailResult(
                guardrail=self,
                agent=agent,
                agent_output=agent_output,
                output=await output,
            )

        return OutputGuardrailResult(
            guardrail=self,
            agent=agent,
            agent_output=agent_output,
            output=output,
        )


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)

# Type aliases for input guardrail functions
_InputGuardrailFuncSync = Callable[
    [RunContext[TContext_co], "Agent[Any]", Union[str, list[Any]]],
    GuardrailFunctionOutput,
]
_InputGuardrailFuncAsync = Callable[
    [RunContext[TContext_co], "Agent[Any]", Union[str, list[Any]]],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    func: _InputGuardrailFuncAsync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
    InputGuardrail[TContext_co],
]: ...


def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co]
    | _InputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    InputGuardrail[TContext_co]
    | Callable[
        [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
        InputGuardrail[TContext_co],
    ]
):
    """Decorator that transforms a function into an InputGuardrail.

    Can be used directly (no parentheses) or with keyword args:

        @input_guardrail
        def my_guardrail(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        @input_guardrail(name="custom_name")
        async def my_async_guardrail(ctx, agent, input):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    Args:
        func: The guardrail function (optional if using with parentheses).
        name: Optional custom name for the guardrail.

    Returns:
        An InputGuardrail wrapping the function.
    """

    def decorator(
        f: _InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co],
    ) -> InputGuardrail[TContext_co]:
        return InputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator


# Type aliases for output guardrail functions
_OutputGuardrailFuncSync = Callable[
    [RunContext[TContext_co], "Agent[Any]", Any],
    GuardrailFunctionOutput,
]
_OutputGuardrailFuncAsync = Callable[
    [RunContext[TContext_co], "Agent[Any]", Any],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    func: _OutputGuardrailFuncAsync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
    OutputGuardrail[TContext_co],
]: ...


def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co]
    | _OutputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    OutputGuardrail[TContext_co]
    | Callable[
        [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
        OutputGuardrail[TContext_co],
    ]
):
    """Decorator that transforms a function into an OutputGuardrail.

    Can be used directly (no parentheses) or with keyword args:

        @output_guardrail
        def my_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        @output_guardrail(name="custom_name")
        async def my_async_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    Args:
        func: The guardrail function (optional if using with parentheses).
        name: Optional custom name for the guardrail.

    Returns:
        An OutputGuardrail wrapping the function.
    """

    def decorator(
        f: _OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co],
    ) -> OutputGuardrail[TContext_co]:
        return OutputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator
