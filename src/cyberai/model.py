"""
Abstract model interface for LLM integration.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:
    from .handoff import Handoff
    from .items import ModelResponse
    from .model_settings import ModelSettings
    from .tool import FunctionTool


class ModelTracing(enum.Enum):
    """Configuration for model tracing."""

    DISABLED = 0
    """Tracing is disabled entirely."""

    ENABLED = 1
    """Tracing is enabled, and all data is included."""

    ENABLED_WITHOUT_DATA = 2
    """Tracing is enabled, but inputs/outputs are not included."""

    def is_disabled(self) -> bool:
        """Check if tracing is disabled."""
        return self == ModelTracing.DISABLED

    def include_data(self) -> bool:
        """Check if tracing should include data."""
        return self == ModelTracing.ENABLED


class Model(abc.ABC):
    """The base interface for calling an LLM.

    Implementations of this class handle the actual communication with
    language model APIs (e.g., OpenAI, Anthropic, local models).
    """

    @abc.abstractmethod
    async def get_response(
        self,
        system_instructions: str | None,
        input: Union[str, List[Any]],
        model_settings: "ModelSettings",
        tools: List["FunctionTool"],
        output_schema: Any | None,
        handoffs: List["Handoff[Any]"],
        tracing: ModelTracing,
    ) -> "ModelResponse":
        """Get a response from the model.

        Args:
            system_instructions: The system prompt/instructions for the model.
            input: The input to the model - either a string or list of
                input items in the expected format.
            model_settings: Configuration for the model (temperature, etc.).
            tools: The tools available for the model to call.
            output_schema: Optional schema for structured output.
            handoffs: The handoffs available to the model.
            tracing: Tracing configuration.

        Returns:
            The full model response containing output items and usage.
        """
        pass

    @abc.abstractmethod
    def stream_response(
        self,
        system_instructions: str | None,
        input: Union[str, List[Any]],
        model_settings: "ModelSettings",
        tools: List["FunctionTool"],
        output_schema: Any | None,
        handoffs: List["Handoff[Any]"],
        tracing: ModelTracing,
    ) -> AsyncIterator[Any]:
        """Stream a response from the model.

        Args:
            system_instructions: The system prompt/instructions for the model.
            input: The input to the model - either a string or list of
                input items in the expected format.
            model_settings: Configuration for the model (temperature, etc.).
            tools: The tools available for the model to call.
            output_schema: Optional schema for structured output.
            handoffs: The handoffs available to the model.
            tracing: Tracing configuration.

        Returns:
            An async iterator of response stream events.
        """
        pass


class ModelProvider(abc.ABC):
    """The base interface for a model provider.

    Model provider is responsible for looking up Models by name.
    Different providers can support different model backends.
    """

    @abc.abstractmethod
    def get_model(self, model_name: str | None) -> Model:
        """Get a model by name.

        Args:
            model_name: The name of the model to get. If None, returns
                the default model for this provider.

        Returns:
            The model instance.
        """
        pass
