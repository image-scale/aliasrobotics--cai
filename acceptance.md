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
- [ ] ModelSettings dataclass with temperature, top_p, frequency_penalty, presence_penalty (all optional floats)
- [ ] ModelSettings has tool_choice (optional: "auto", "required", "none", or custom string)
- [ ] ModelSettings has parallel_tool_calls (optional bool), truncation (optional), max_tokens (optional int)
- [ ] ModelSettings.resolve() merges another ModelSettings, overriding non-None values
- [ ] RunContext generic dataclass wraps user-provided context object
- [ ] RunContext carries a Usage instance that accumulates across the run
