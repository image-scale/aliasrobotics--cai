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
- [ ] Agent dataclass with name (required) and optional instructions (string or callable)
- [ ] Agent has description and handoff_description fields
- [ ] Agent has tools list, handoffs list, model, and model_settings fields
- [ ] Agent has input_guardrails and output_guardrails lists
- [ ] Agent.get_system_prompt() returns instructions (supports both str and async callable)
- [ ] Agent.clone() creates a copy with optional field overrides
- [ ] Agent.get_all_tools() returns all tools including function tools
