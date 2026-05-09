"""
Tests for the model_settings module.
"""

import pytest

from cyberai import ModelSettings


class TestModelSettings:
    """Tests for the ModelSettings dataclass."""

    def test_default_values_are_none(self):
        """All fields should default to None."""
        settings = ModelSettings()
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.frequency_penalty is None
        assert settings.presence_penalty is None
        assert settings.tool_choice is None
        assert settings.parallel_tool_calls is None
        assert settings.truncation is None
        assert settings.max_tokens is None

    def test_temperature_can_be_set(self):
        """Temperature can be set to a float value."""
        settings = ModelSettings(temperature=0.7)
        assert settings.temperature == 0.7

    def test_top_p_can_be_set(self):
        """Top_p can be set to a float value."""
        settings = ModelSettings(top_p=0.9)
        assert settings.top_p == 0.9

    def test_frequency_penalty_can_be_set(self):
        """Frequency penalty can be set."""
        settings = ModelSettings(frequency_penalty=0.5)
        assert settings.frequency_penalty == 0.5

    def test_presence_penalty_can_be_set(self):
        """Presence penalty can be set."""
        settings = ModelSettings(presence_penalty=0.3)
        assert settings.presence_penalty == 0.3

    def test_tool_choice_accepts_auto(self):
        """Tool choice can be 'auto'."""
        settings = ModelSettings(tool_choice="auto")
        assert settings.tool_choice == "auto"

    def test_tool_choice_accepts_required(self):
        """Tool choice can be 'required'."""
        settings = ModelSettings(tool_choice="required")
        assert settings.tool_choice == "required"

    def test_tool_choice_accepts_none_string(self):
        """Tool choice can be 'none'."""
        settings = ModelSettings(tool_choice="none")
        assert settings.tool_choice == "none"

    def test_tool_choice_accepts_custom_string(self):
        """Tool choice can be a custom tool name."""
        settings = ModelSettings(tool_choice="my_custom_tool")
        assert settings.tool_choice == "my_custom_tool"

    def test_parallel_tool_calls_can_be_true(self):
        """Parallel tool calls can be True."""
        settings = ModelSettings(parallel_tool_calls=True)
        assert settings.parallel_tool_calls is True

    def test_parallel_tool_calls_can_be_false(self):
        """Parallel tool calls can be False."""
        settings = ModelSettings(parallel_tool_calls=False)
        assert settings.parallel_tool_calls is False

    def test_truncation_accepts_auto(self):
        """Truncation can be 'auto'."""
        settings = ModelSettings(truncation="auto")
        assert settings.truncation == "auto"

    def test_truncation_accepts_disabled(self):
        """Truncation can be 'disabled'."""
        settings = ModelSettings(truncation="disabled")
        assert settings.truncation == "disabled"

    def test_max_tokens_can_be_set(self):
        """Max tokens can be set to an integer."""
        settings = ModelSettings(max_tokens=1024)
        assert settings.max_tokens == 1024


class TestModelSettingsResolve:
    """Tests for ModelSettings.resolve() method."""

    def test_resolve_with_none_returns_self(self):
        """Resolving with None should return self."""
        settings = ModelSettings(temperature=0.5)
        resolved = settings.resolve(None)
        assert resolved is settings

    def test_resolve_overrides_temperature(self):
        """Override should replace temperature."""
        base = ModelSettings(temperature=0.5, top_p=0.9)
        override = ModelSettings(temperature=0.8)
        resolved = base.resolve(override)

        assert resolved.temperature == 0.8
        assert resolved.top_p == 0.9

    def test_resolve_does_not_override_with_none(self):
        """None values in override should not replace base values."""
        base = ModelSettings(temperature=0.5, max_tokens=100)
        override = ModelSettings(top_p=0.95)
        resolved = base.resolve(override)

        assert resolved.temperature == 0.5
        assert resolved.top_p == 0.95
        assert resolved.max_tokens == 100

    def test_resolve_multiple_fields(self):
        """Multiple fields can be overridden at once."""
        base = ModelSettings(temperature=0.5, top_p=0.9, max_tokens=50)
        override = ModelSettings(temperature=0.7, max_tokens=100, tool_choice="auto")
        resolved = base.resolve(override)

        assert resolved.temperature == 0.7
        assert resolved.top_p == 0.9
        assert resolved.max_tokens == 100
        assert resolved.tool_choice == "auto"

    def test_resolve_returns_new_instance(self):
        """Resolve should return a new instance, not mutate original."""
        base = ModelSettings(temperature=0.5)
        override = ModelSettings(temperature=0.8)
        resolved = base.resolve(override)

        assert resolved is not base
        assert base.temperature == 0.5
        assert resolved.temperature == 0.8

    def test_resolve_all_fields(self):
        """All fields can be overridden."""
        base = ModelSettings()
        override = ModelSettings(
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            tool_choice="required",
            parallel_tool_calls=True,
            truncation="auto",
            max_tokens=2048,
        )
        resolved = base.resolve(override)

        assert resolved.temperature == 0.7
        assert resolved.top_p == 0.95
        assert resolved.frequency_penalty == 0.1
        assert resolved.presence_penalty == 0.2
        assert resolved.tool_choice == "required"
        assert resolved.parallel_tool_calls is True
        assert resolved.truncation == "auto"
        assert resolved.max_tokens == 2048
