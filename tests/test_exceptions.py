"""
Tests for the exceptions module.
"""

import pytest

from cyberai import (
    AgentsError,
    InputGuardrailTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTriggered,
    UserError,
)


class TestAgentsError:
    """Tests for the base AgentsError class."""

    def test_is_exception(self):
        """AgentsError should be an Exception subclass."""
        assert issubclass(AgentsError, Exception)

    def test_can_be_raised(self):
        """AgentsError can be raised and caught."""
        with pytest.raises(AgentsError):
            raise AgentsError("test error")


class TestMaxTurnsExceeded:
    """Tests for MaxTurnsExceeded exception."""

    def test_inherits_from_agents_error(self):
        """MaxTurnsExceeded should inherit from AgentsError."""
        assert issubclass(MaxTurnsExceeded, AgentsError)

    def test_stores_message(self):
        """MaxTurnsExceeded should store the message."""
        exc = MaxTurnsExceeded("Max turns (10) exceeded")
        assert exc.message == "Max turns (10) exceeded"

    def test_can_be_raised_and_caught(self):
        """MaxTurnsExceeded can be raised and caught."""
        with pytest.raises(MaxTurnsExceeded) as exc_info:
            raise MaxTurnsExceeded("Turn limit reached")
        assert exc_info.value.message == "Turn limit reached"


class TestModelBehaviorError:
    """Tests for ModelBehaviorError exception."""

    def test_inherits_from_agents_error(self):
        """ModelBehaviorError should inherit from AgentsError."""
        assert issubclass(ModelBehaviorError, AgentsError)

    def test_stores_message(self):
        """ModelBehaviorError should store the message."""
        exc = ModelBehaviorError("Invalid JSON from model")
        assert exc.message == "Invalid JSON from model"

    def test_can_be_raised_for_invalid_json(self):
        """ModelBehaviorError can be raised for malformed JSON."""
        with pytest.raises(ModelBehaviorError) as exc_info:
            raise ModelBehaviorError("Invalid JSON input for tool test_tool: {invalid}")
        assert "Invalid JSON" in exc_info.value.message


class TestUserError:
    """Tests for UserError exception."""

    def test_inherits_from_agents_error(self):
        """UserError should inherit from AgentsError."""
        assert issubclass(UserError, AgentsError)

    def test_stores_message(self):
        """UserError should store the message."""
        exc = UserError("Invalid configuration")
        assert exc.message == "Invalid configuration"

    def test_can_be_raised_for_user_mistake(self):
        """UserError can be raised for user mistakes."""
        with pytest.raises(UserError) as exc_info:
            raise UserError("RunContextWrapper must be the first parameter")
        assert "RunContextWrapper" in exc_info.value.message


class TestInputGuardrailTriggered:
    """Tests for InputGuardrailTriggered exception."""

    def test_inherits_from_agents_error(self):
        """InputGuardrailTriggered should inherit from AgentsError."""
        assert issubclass(InputGuardrailTriggered, AgentsError)

    def test_stores_guardrail_result(self):
        """InputGuardrailTriggered should store the guardrail result."""
        mock_result = {"tripwire_triggered": True, "info": "Malicious input detected"}
        exc = InputGuardrailTriggered(mock_result)
        assert exc.guardrail_result == mock_result

    def test_can_be_raised_with_result(self):
        """InputGuardrailTriggered can be raised with a guardrail result."""
        mock_result = {"output_info": "blocked"}
        with pytest.raises(InputGuardrailTriggered) as exc_info:
            raise InputGuardrailTriggered(mock_result)
        assert exc_info.value.guardrail_result == mock_result


class TestOutputGuardrailTriggered:
    """Tests for OutputGuardrailTriggered exception."""

    def test_inherits_from_agents_error(self):
        """OutputGuardrailTriggered should inherit from AgentsError."""
        assert issubclass(OutputGuardrailTriggered, AgentsError)

    def test_stores_guardrail_result(self):
        """OutputGuardrailTriggered should store the guardrail result."""
        mock_result = {"tripwire_triggered": True, "info": "Dangerous output detected"}
        exc = OutputGuardrailTriggered(mock_result)
        assert exc.guardrail_result == mock_result

    def test_can_be_raised_with_result(self):
        """OutputGuardrailTriggered can be raised with a guardrail result."""
        mock_result = {"output_info": "unsafe command blocked"}
        with pytest.raises(OutputGuardrailTriggered) as exc_info:
            raise OutputGuardrailTriggered(mock_result)
        assert exc_info.value.guardrail_result == mock_result
