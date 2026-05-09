"""
Model settings for configuring LLM behavior.
"""

from dataclasses import dataclass, fields, replace
from typing import Literal, Optional


@dataclass
class ModelSettings:
    """Settings to use when calling an LLM.

    This class holds optional model configuration parameters (e.g., temperature,
    top_p, penalties, truncation, etc.).

    Not all models/providers support all of these parameters, so please check
    the API documentation for the specific model and provider you are using.
    """

    temperature: Optional[float] = None
    """The temperature to use when calling the model."""

    top_p: Optional[float] = None
    """The top_p to use when calling the model."""

    frequency_penalty: Optional[float] = None
    """The frequency penalty to use when calling the model."""

    presence_penalty: Optional[float] = None
    """The presence penalty to use when calling the model."""

    tool_choice: Optional[Literal["auto", "required", "none"] | str] = None
    """The tool choice to use when calling the model."""

    parallel_tool_calls: Optional[bool] = None
    """Whether to use parallel tool calls when calling the model.
    Defaults to False if not provided."""

    truncation: Optional[Literal["auto", "disabled"]] = None
    """The truncation strategy to use when calling the model."""

    max_tokens: Optional[int] = None
    """The maximum number of output tokens to generate."""

    def resolve(self, override: Optional["ModelSettings"]) -> "ModelSettings":
        """Produce a new ModelSettings by overlaying any non-None values from
        the override on top of this instance.

        Args:
            override: The ModelSettings to overlay. If None, returns self.

        Returns:
            A new ModelSettings with merged values.
        """
        if override is None:
            return self

        changes = {
            field.name: getattr(override, field.name)
            for field in fields(self)
            if getattr(override, field.name) is not None
        }
        return replace(self, **changes)
