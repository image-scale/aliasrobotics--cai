# Todo

## Plan
Build the framework bottom-up starting with foundational types (exceptions, usage, settings) then core abstractions (items, context, tool), then high-level components (agent, handoff, guardrail), and finally the runner. Each task delivers a complete, testable feature.

## Tasks
- [>] Task 1: Implement the core exceptions and usage tracking (exceptions for max turns exceeded, model behavior errors, guardrail violations + usage dataclass for token tracking)
- [ ] Task 2: Implement model settings and run context (model configuration parameters like temperature, tool_choice + context wrapper that carries state through agent runs)
- [ ] Task 3: Implement the function tool system (function_tool decorator that converts Python functions to LLM tools with JSON schema generation from type hints and docstrings)
- [ ] Task 4: Implement run items and model response types (message items, tool call items, handoff items, model response wrapper with usage)
- [ ] Task 5: Implement the Agent class (agent with name, instructions, tools, handoffs, guardrails, model settings, and system prompt generation)
- [ ] Task 6: Implement the handoff system (handoff dataclass, handoff decorator for agent-to-agent delegation with input filtering)
- [ ] Task 7: Implement input and output guardrails (guardrail decorators and classes for validating agent inputs/outputs with tripwire support)
- [ ] Task 8: Implement the model interface (abstract Model and ModelProvider base classes for LLM integration)
- [ ] Task 9: Implement RunResult and RunConfig (run result containing items, outputs, guardrail results + run configuration with model/provider settings)
- [ ] Task 10: Implement the Runner (orchestrates agent execution loop with turn tracking, tool execution, handoffs, and guardrail checks)
