"""
Runner orchestrates agent execution with turn tracking, tool execution,
handoffs, and guardrail checks.
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, List, Union

from .agent import Agent
from .exceptions import (
    InputGuardrailTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTriggered,
)
from .guardrail import InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult
from .handoff import Handoff, handoff as create_handoff
from .items import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageItem,
    ModelResponse,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from .model import Model, ModelTracing
from .model_settings import ModelSettings
from .run_context import RunContext
from .run_result import RunConfig, RunResult
from .tool import FunctionTool
from .usage import Usage

if TYPE_CHECKING:
    pass

DEFAULT_MAX_TURNS = float("inf")


class Runner:
    """Orchestrates agent execution.

    The Runner handles the agent execution loop:
    1. Invoke the agent with input
    2. If there's a final output, return it
    3. If there's a handoff, switch to the new agent
    4. If there are tool calls, execute them and continue

    The Runner also handles input/output guardrails and enforces
    the maximum number of turns.
    """

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[Any],
        input: Union[str, List[Any]],
        *,
        context: Any | None = None,
        max_turns: int | float = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Run a workflow starting at the given agent.

        The agent runs in a loop until a final output is generated:
        1. The agent is invoked with the input
        2. If there's a final output, the loop terminates
        3. If there's a handoff, run the loop with the new agent
        4. Otherwise, execute tool calls and re-run

        Args:
            starting_agent: The starting agent to run.
            input: The initial input (string or list of input items).
            context: Optional context object passed to tools and guardrails.
            max_turns: Maximum number of turns. Raises MaxTurnsExceeded if exceeded.
            run_config: Global settings for the entire agent run.

        Returns:
            RunResult containing all inputs, items, and the final output.

        Raises:
            MaxTurnsExceeded: If max_turns is exceeded.
            InputGuardrailTriggered: If an input guardrail trips.
            OutputGuardrailTriggered: If an output guardrail trips.
        """
        if run_config is None:
            run_config = RunConfig()

        current_turn = 0
        original_input: Union[str, List[Any]] = copy.deepcopy(input)
        generated_items: List[RunItem] = []
        model_responses: List[ModelResponse] = []

        run_context: RunContext[Any] = RunContext(context=context)

        input_guardrail_results: List[InputGuardrailResult] = []
        current_agent = starting_agent
        is_first_turn = True

        while True:
            current_turn += 1

            if current_turn > max_turns:
                raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")

            # Run input guardrails on first turn
            if is_first_turn:
                all_input_guardrails = (
                    list(starting_agent.input_guardrails)
                    + (run_config.input_guardrails or [])
                )
                input_guardrail_results = await cls._run_input_guardrails(
                    starting_agent,
                    all_input_guardrails,
                    copy.deepcopy(input),
                    run_context,
                )
                is_first_turn = False

            # Get the model for this agent
            model = cls._get_model(current_agent, run_config)
            if model is None:
                # No model available - return what we have
                return RunResult(
                    input=original_input,
                    new_items=generated_items,
                    raw_responses=model_responses,
                    final_output=None,
                    _last_agent=current_agent,
                    input_guardrail_results=input_guardrail_results,
                    output_guardrail_results=[],
                )

            # Build the input for the model
            input_items = ItemHelpers.input_to_new_input_list(original_input)
            input_items.extend([item.to_input_item() for item in generated_items])

            # Get system prompt and tools
            system_prompt = await current_agent.get_system_prompt(run_context)
            all_tools = current_agent.get_all_tools()
            handoffs = cls._get_handoffs(current_agent)

            # Resolve model settings
            model_settings = current_agent.model_settings.resolve(run_config.model_settings)

            # Determine tracing mode
            if run_config.tracing_disabled:
                tracing = ModelTracing.DISABLED
            elif not run_config.trace_include_sensitive_data:
                tracing = ModelTracing.ENABLED_WITHOUT_DATA
            else:
                tracing = ModelTracing.ENABLED

            # Get response from model
            response = await model.get_response(
                system_instructions=system_prompt,
                input=input_items,
                model_settings=model_settings,
                tools=all_tools,
                output_schema=None,  # Could be agent.output_type
                handoffs=handoffs,
                tracing=tracing,
            )

            model_responses.append(response)
            run_context.usage.add(response.usage)

            # Process the response
            new_items, next_step = await cls._process_response(
                current_agent, response, all_tools, handoffs, run_context
            )
            generated_items.extend(new_items)

            if next_step == "final":
                # Extract final output from response
                final_output = cls._extract_final_output(response)

                # Run output guardrails
                all_output_guardrails = (
                    list(current_agent.output_guardrails)
                    + (run_config.output_guardrails or [])
                )
                output_guardrail_results = await cls._run_output_guardrails(
                    all_output_guardrails,
                    current_agent,
                    final_output,
                    run_context,
                )

                return RunResult(
                    input=original_input,
                    new_items=generated_items,
                    raw_responses=model_responses,
                    final_output=final_output,
                    _last_agent=current_agent,
                    input_guardrail_results=input_guardrail_results,
                    output_guardrail_results=output_guardrail_results,
                )
            elif isinstance(next_step, Agent):
                # Handoff to new agent
                current_agent = next_step
            # Otherwise continue the loop (tool calls were processed)

    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model | None:
        """Get the model to use for this agent."""
        # Check run_config.model first
        if isinstance(run_config.model, Model):
            return run_config.model
        elif isinstance(run_config.model, str) and run_config.model_provider:
            return run_config.model_provider.get_model(run_config.model)

        # Check agent.model
        if isinstance(agent.model, Model):
            return agent.model
        elif isinstance(agent.model, str) and run_config.model_provider:
            return run_config.model_provider.get_model(agent.model)
        elif run_config.model_provider:
            return run_config.model_provider.get_model(None)

        return None

    @classmethod
    def _get_handoffs(cls, agent: Agent[Any]) -> List[Handoff[Any]]:
        """Get handoffs from agent, converting Agents to Handoffs."""
        handoffs = []
        for item in agent.handoffs:
            if isinstance(item, Handoff):
                handoffs.append(item)
            elif isinstance(item, Agent):
                handoffs.append(create_handoff(item))
        return handoffs

    @classmethod
    async def _process_response(
        cls,
        agent: Agent[Any],
        response: ModelResponse,
        tools: List[FunctionTool],
        handoffs: List[Handoff[Any]],
        run_context: RunContext[Any],
    ) -> tuple[List[RunItem], Union[str, Agent[Any]]]:
        """Process a model response and return new items and next step.

        Returns:
            Tuple of (new items, next step).
            next step is "final" for final output, "continue" for more processing,
            or an Agent for handoff.
        """
        new_items: List[RunItem] = []
        has_tool_calls = False
        handoff_agent: Agent[Any] | None = None

        for output_item in response.output:
            if isinstance(output_item, dict):
                item_type = output_item.get("type")

                if item_type == "message":
                    # Text message from model
                    content = output_item.get("content", "")
                    new_items.append(MessageItem(
                        agent=agent,
                        raw_item=output_item,
                        role="assistant",
                        content=content,
                    ))

                elif item_type == "function_call":
                    # Tool call
                    has_tool_calls = True
                    call_id = output_item.get("call_id", "")
                    name = output_item.get("name", "")
                    arguments = output_item.get("arguments", "{}")

                    new_items.append(ToolCallItem(
                        agent=agent,
                        raw_item=output_item,
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                    ))

                    # Check if this is a handoff
                    handoff = cls._find_handoff(name, handoffs)
                    if handoff:
                        # This is a handoff - invoke it
                        handoff_agent = await handoff.on_invoke_handoff(run_context, arguments)
                        new_items.append(HandoffCallItem(
                            agent=agent,
                            raw_item=output_item,
                            call_id=call_id,
                            target_agent_name=handoff_agent.name,
                            arguments=arguments,
                        ))
                        new_items.append(HandoffOutputItem(
                            agent=agent,
                            raw_item={},
                            source_agent=agent,
                            target_agent=handoff_agent,
                        ))
                    else:
                        # Execute the tool
                        tool = cls._find_tool(name, tools)
                        if tool:
                            output = await tool.on_invoke_tool(run_context, arguments)
                            new_items.append(ToolCallOutputItem(
                                agent=agent,
                                raw_item={},
                                call_id=call_id,
                                output=str(output),
                            ))

        # Determine next step
        if handoff_agent:
            return new_items, handoff_agent
        elif has_tool_calls:
            return new_items, "continue"
        else:
            return new_items, "final"

    @classmethod
    def _find_tool(cls, name: str, tools: List[FunctionTool]) -> FunctionTool | None:
        """Find a tool by name."""
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    @classmethod
    def _find_handoff(cls, name: str, handoffs: List[Handoff[Any]]) -> Handoff[Any] | None:
        """Find a handoff by tool name."""
        for handoff in handoffs:
            if handoff.tool_name == name:
                return handoff
        return None

    @classmethod
    def _extract_final_output(cls, response: ModelResponse) -> Any:
        """Extract final output from model response."""
        for output_item in response.output:
            if isinstance(output_item, dict):
                if output_item.get("type") == "message":
                    return output_item.get("content", "")
        return None

    @classmethod
    async def _run_input_guardrails(
        cls,
        agent: Agent[Any],
        guardrails: List[InputGuardrail[Any]],
        input: Union[str, List[Any]],
        context: RunContext[Any],
    ) -> List[InputGuardrailResult]:
        """Run input guardrails and return results.

        Raises InputGuardrailTriggered if any guardrail trips.
        """
        if not guardrails:
            return []

        results: List[InputGuardrailResult] = []
        tasks = [
            guardrail.run(agent, input, context)
            for guardrail in guardrails
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result.output.tripwire_triggered:
                raise InputGuardrailTriggered(result)
            results.append(result)

        return results

    @classmethod
    async def _run_output_guardrails(
        cls,
        guardrails: List[OutputGuardrail[Any]],
        agent: Agent[Any],
        agent_output: Any,
        context: RunContext[Any],
    ) -> List[OutputGuardrailResult]:
        """Run output guardrails and return results.

        Raises OutputGuardrailTriggered if any guardrail trips.
        """
        if not guardrails:
            return []

        results: List[OutputGuardrailResult] = []
        tasks = [
            guardrail.run(context, agent, agent_output)
            for guardrail in guardrails
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result.output.tripwire_triggered:
                raise OutputGuardrailTriggered(result)
            results.append(result)

        return results

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[Any],
        input: Union[str, List[Any]],
        *,
        context: Any | None = None,
        max_turns: int | float = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Run a workflow synchronously.

        This wraps the async run() method. Note: This won't work if
        there's already an event loop running (e.g., in async context).

        Args:
            starting_agent: The starting agent to run.
            input: The initial input.
            context: Optional context object.
            max_turns: Maximum number of turns.
            run_config: Global settings for the run.

        Returns:
            RunResult containing all inputs, items, and final output.
        """
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                run_config=run_config,
            )
        )
