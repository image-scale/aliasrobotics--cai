# Acceptance Criteria

## Task 1: Core exceptions and usage tracking

### Acceptance Criteria
- [ ] AgentsError is the base exception class for all framework exceptions
- [ ] MaxTurnsExceeded stores a message and can be raised when turn limit is exceeded
- [ ] ModelBehaviorError stores a message for unexpected model behavior (e.g., invalid JSON)
- [ ] UserError stores a message for user mistakes when using the framework
- [ ] InputGuardrailTriggered stores a guardrail_result and can be raised when input guardrail trips
- [ ] OutputGuardrailTriggered stores a guardrail_result and can be raised when output guardrail trips
- [ ] Usage dataclass tracks requests, input_tokens, output_tokens, total_tokens (all default to 0)
- [ ] Usage.add() method adds another Usage instance's values to this one
