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
- [ ] FunctionTool dataclass has name, description, params_json_schema, on_invoke_tool callback
- [ ] function_tool decorator converts a Python function to a FunctionTool
- [ ] JSON schema is generated from function type hints (int, str, float, bool, etc.)
- [ ] Function description is extracted from docstring
- [ ] Parameter descriptions are extracted from docstring (Google/Sphinx style)
- [ ] Functions can take optional RunContext as first parameter
- [ ] Tool invocation parses JSON arguments and calls the function
- [ ] Invalid JSON raises ModelBehaviorError
- [ ] Missing required arguments raises ModelBehaviorError
- [ ] Both sync and async functions are supported
