"""
Exception classes for the CyberAI agents framework.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class AgentsError(Exception):
    """Base class for all exceptions in the CyberAI agents framework."""

    pass


class MaxTurnsExceeded(AgentsError):
    """Raised when the maximum number of turns is exceeded during agent execution."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelBehaviorError(AgentsError):
    """Raised when the model does something unexpected, e.g., calling a non-existent tool
    or providing malformed JSON.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class UserError(AgentsError):
    """Raised when the user makes an error using the CyberAI framework."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InputGuardrailTriggered(AgentsError):
    """Raised when an input guardrail tripwire is triggered."""

    def __init__(self, guardrail_result: Any) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(f"Input guardrail triggered tripwire")


class OutputGuardrailTriggered(AgentsError):
    """Raised when an output guardrail tripwire is triggered."""

    def __init__(self, guardrail_result: Any) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(f"Output guardrail triggered tripwire")
