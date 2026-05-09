# Acceptance Criteria

## Task 1: Core exceptions and usage tracking

### Acceptance Criteria
- [x] AgentsError is the base exception class for all framework exceptions
- [x] MaxTurnsExceeded stores a message and can be raised when turn limit is exceeded
- [x] ModelBehaviorError stores a message for unexpected model behavior (e.g., invalid JSON)
- [x] UserError stores a message for user mistakes when using the framework
- [x] InputGuardrailTriggered stores a guardrail_result and can be raised when input guardrail trips
- [x] OutputGuardrailTriggered stores a guardrail_result and can be raised when output guardrail trips
- [x] Usage dataclass tracks requests, input_tokens, output_tokens, total_tokens (all default to 0)
- [x] Usage.add() method adds another Usage instance's values to this one

## Task 2: Model settings and run context

### Acceptance Criteria
- [x] ModelSettings dataclass with temperature, top_p, frequency_penalty, presence_penalty (all optional floats)
- [x] ModelSettings has tool_choice (optional: "auto", "required", "none", or custom string)
- [x] ModelSettings has parallel_tool_calls (optional bool), truncation (optional), max_tokens (optional int)
- [x] ModelSettings.resolve() merges another ModelSettings, overriding non-None values
- [x] RunContext generic dataclass wraps user-provided context object
- [x] RunContext carries a Usage instance that accumulates across the run

## Task 3: Function tool system

### Acceptance Criteria
- [x] FunctionTool dataclass has name, description, params_json_schema, on_invoke_tool callback
- [x] function_tool decorator converts a Python function to a FunctionTool
- [x] JSON schema is generated from function type hints (int, str, float, bool, etc.)
- [x] Function description is extracted from docstring
- [x] Parameter descriptions are extracted from docstring (Google/Sphinx style)
- [x] Functions can take optional RunContext as first parameter
- [x] Tool invocation parses JSON arguments and calls the function
- [x] Invalid JSON raises ModelBehaviorError
- [x] Missing required arguments raises ModelBehaviorError
- [x] Both sync and async functions are supported

## Task 4: Run items and model response types

### Acceptance Criteria
- [x] MessageItem dataclass represents a text message from the model
- [x] ToolCallItem dataclass represents a tool call request with name, call_id, arguments
- [x] ToolCallOutputItem dataclass represents the output of a tool call
- [x] HandoffCallItem dataclass represents a handoff request
- [x] HandoffOutputItem dataclass represents a completed handoff with source/target agent
- [x] ModelResponse dataclass contains output items, usage, and optional referenceable_id
- [x] ItemHelpers class with helper methods for extracting text from items
- [x] Items can be converted to input format for subsequent model calls

## Task 5: Agent class

### Acceptance Criteria
- [x] Agent dataclass with name (required) and optional instructions (string or callable)
- [x] Agent has description and handoff_description fields
- [x] Agent has tools list, handoffs list, model, and model_settings fields
- [x] Agent has input_guardrails and output_guardrails lists
- [x] Agent.get_system_prompt() returns instructions (supports both str and async callable)
- [x] Agent.clone() creates a copy with optional field overrides
- [x] Agent.get_all_tools() returns all tools including function tools

## Task 6: Handoff system

### Acceptance Criteria
- [x] Handoff dataclass with tool_name, tool_description, agent_name, on_invoke_handoff callback
- [x] Handoff has input_json_schema for passing data to the target agent
- [x] Handoff has optional input_filter to modify inputs before passing to target
- [x] handoff() function creates a Handoff from an Agent with default naming
- [x] handoff() supports on_handoff callback for pre-handoff logic
- [x] Handoff.default_tool_name() generates "transfer_to_{agent_name}" style names
- [x] Invoking a handoff returns the target agent

## Task 7: Input and output guardrails

### Acceptance Criteria
- [x] GuardrailFunctionOutput dataclass with output_info and tripwire_triggered
- [x] InputGuardrail dataclass with guardrail_function and optional name
- [x] InputGuardrail.run() executes the guardrail and returns InputGuardrailResult
- [x] InputGuardrailResult contains guardrail reference and output
- [x] OutputGuardrail dataclass with guardrail_function and optional name
- [x] OutputGuardrail.run() executes the guardrail and returns OutputGuardrailResult
- [x] OutputGuardrailResult contains guardrail, agent_output, agent, and output
- [x] input_guardrail decorator converts a function to InputGuardrail
- [x] output_guardrail decorator converts a function to OutputGuardrail
- [x] Both sync and async guardrail functions are supported
- [x] Guardrail.get_name() returns the name or falls back to function name

## Task 8: Model interface

### Acceptance Criteria
- [x] ModelTracing enum with DISABLED, ENABLED, ENABLED_WITHOUT_DATA values
- [x] ModelTracing.is_disabled() and include_data() helper methods
- [x] Model abstract base class with get_response() abstract method
- [x] Model has stream_response() abstract method returning AsyncIterator
- [x] ModelProvider abstract base class with get_model() abstract method
- [x] get_response() takes system_instructions, input, model_settings, tools, output_schema, handoffs, tracing
- [x] stream_response() has the same signature as get_response()

## Task 9: RunResult and RunConfig

### Acceptance Criteria
- [x] RunConfig dataclass with model, model_provider, model_settings, handoff_input_filter fields
- [x] RunConfig has input_guardrails, output_guardrails, max_turns fields
- [x] RunConfig has tracing_disabled flag
- [x] RunResult dataclass with items list, final_output, and last_agent
- [x] RunResult has input_guardrail_results and output_guardrail_results lists
- [x] RunResult.to_input_list() converts result items to input format

## Task 10: Runner

### Acceptance Criteria
- [x] Runner class with run() class method that orchestrates agent execution
- [x] Runner.run() returns a RunResult with all generated items
- [x] Runner tracks turn count and raises MaxTurnsExceeded when limit is exceeded
- [x] Runner executes tools when model requests tool calls
- [x] Runner handles handoffs by switching to the target agent
- [x] Runner runs input guardrails on the initial input
- [x] Runner runs output guardrails on the final output
- [x] Runner raises InputGuardrailTriggered/OutputGuardrailTriggered when tripwire is triggered
