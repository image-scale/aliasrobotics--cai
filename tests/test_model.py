"""
Tests for the model interface.
"""

from collections.abc import AsyncIterator
from typing import Any, List

import pytest

from cyberai import FunctionTool, ModelSettings
from cyberai.handoff import Handoff
from cyberai.items import ModelResponse
from cyberai.model import Model, ModelProvider, ModelTracing
from cyberai.usage import Usage


class TestModelTracing:
    """Tests for ModelTracing enum."""

    def test_tracing_disabled_value(self):
        """ModelTracing.DISABLED has value 0."""
        assert ModelTracing.DISABLED.value == 0

    def test_tracing_enabled_value(self):
        """ModelTracing.ENABLED has value 1."""
        assert ModelTracing.ENABLED.value == 1

    def test_tracing_enabled_without_data_value(self):
        """ModelTracing.ENABLED_WITHOUT_DATA has value 2."""
        assert ModelTracing.ENABLED_WITHOUT_DATA.value == 2

    def test_is_disabled_true(self):
        """is_disabled() returns True for DISABLED."""
        assert ModelTracing.DISABLED.is_disabled() is True

    def test_is_disabled_false_for_enabled(self):
        """is_disabled() returns False for ENABLED."""
        assert ModelTracing.ENABLED.is_disabled() is False

    def test_is_disabled_false_for_enabled_without_data(self):
        """is_disabled() returns False for ENABLED_WITHOUT_DATA."""
        assert ModelTracing.ENABLED_WITHOUT_DATA.is_disabled() is False

    def test_include_data_true_for_enabled(self):
        """include_data() returns True for ENABLED."""
        assert ModelTracing.ENABLED.include_data() is True

    def test_include_data_false_for_disabled(self):
        """include_data() returns False for DISABLED."""
        assert ModelTracing.DISABLED.include_data() is False

    def test_include_data_false_for_enabled_without_data(self):
        """include_data() returns False for ENABLED_WITHOUT_DATA."""
        assert ModelTracing.ENABLED_WITHOUT_DATA.include_data() is False


class TestModelAbstractClass:
    """Tests for Model abstract base class."""

    def test_model_is_abstract(self):
        """Model cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Model()  # type: ignore

    def test_model_requires_get_response(self):
        """Model subclass must implement get_response."""

        class IncompleteModel(Model):
            def stream_response(self, *args, **kwargs):
                yield None

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore

    def test_model_requires_stream_response(self):
        """Model subclass must implement stream_response."""

        class IncompleteModel(Model):
            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore


class TestConcreteModel:
    """Tests for concrete Model implementations."""

    def test_concrete_model_can_be_instantiated(self):
        """A complete Model subclass can be instantiated."""

        class ConcreteModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                yield {"type": "chunk"}

        model = ConcreteModel()
        assert isinstance(model, Model)

    @pytest.mark.asyncio
    async def test_get_response_can_be_called(self):
        """get_response() can be called on concrete implementation."""

        class TestModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                return ModelResponse(
                    output=[],
                    usage=Usage(input_tokens=10, output_tokens=20),
                )

            async def stream_response(self, *args, **kwargs):
                yield {}

        model = TestModel()
        response = await model.get_response(
            system_instructions="You are helpful",
            input="Hello",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.ENABLED,
        )

        assert isinstance(response, ModelResponse)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_stream_response_returns_iterator(self):
        """stream_response() returns an async iterator."""

        class TestModel(Model):
            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                for i in range(3):
                    yield {"chunk": i}

        model = TestModel()
        chunks = []

        async for chunk in model.stream_response(
            system_instructions=None,
            input="test",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == {"chunk": 0}
        assert chunks[2] == {"chunk": 2}


class TestModelProviderAbstractClass:
    """Tests for ModelProvider abstract base class."""

    def test_provider_is_abstract(self):
        """ModelProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelProvider()  # type: ignore

    def test_provider_requires_get_model(self):
        """ModelProvider subclass must implement get_model."""

        class IncompleteProvider(ModelProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore


class TestConcreteModelProvider:
    """Tests for concrete ModelProvider implementations."""

    def test_concrete_provider_can_be_instantiated(self):
        """A complete ModelProvider subclass can be instantiated."""

        class ConcreteModel(Model):
            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        class ConcreteProvider(ModelProvider):
            def get_model(self, model_name):
                return ConcreteModel()

        provider = ConcreteProvider()
        assert isinstance(provider, ModelProvider)

    def test_get_model_returns_model(self):
        """get_model() returns a Model instance."""

        class ConcreteModel(Model):
            def __init__(self, name):
                self.name = name

            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        class ConcreteProvider(ModelProvider):
            def get_model(self, model_name):
                return ConcreteModel(model_name or "default")

        provider = ConcreteProvider()

        model = provider.get_model("gpt-4")
        assert isinstance(model, Model)
        assert model.name == "gpt-4"

    def test_get_model_with_none_name(self):
        """get_model() can be called with None for default model."""

        class ConcreteModel(Model):
            def __init__(self, name):
                self.name = name

            async def get_response(self, *args, **kwargs):
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        class DefaultProvider(ModelProvider):
            def get_model(self, model_name):
                return ConcreteModel(model_name or "default-model")

        provider = DefaultProvider()
        model = provider.get_model(None)

        assert model.name == "default-model"


class TestModelWithTools:
    """Tests for Model with tools parameter."""

    @pytest.mark.asyncio
    async def test_get_response_receives_tools(self):
        """get_response() receives tools list."""
        received_tools = []

        class ToolTrackingModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                received_tools.extend(tools)
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        from cyberai import function_tool

        @function_tool
        def scan_port(port: int) -> str:
            """Scan a port."""
            return f"Port {port}"

        model = ToolTrackingModel()
        await model.get_response(
            system_instructions=None,
            input="scan",
            model_settings=ModelSettings(),
            tools=[scan_port],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.ENABLED,
        )

        assert len(received_tools) == 1
        assert isinstance(received_tools[0], FunctionTool)
        assert received_tools[0].name == "scan_port"


class TestModelWithHandoffs:
    """Tests for Model with handoffs parameter."""

    @pytest.mark.asyncio
    async def test_get_response_receives_handoffs(self):
        """get_response() receives handoffs list."""
        received_handoffs = []

        class HandoffTrackingModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                received_handoffs.extend(handoffs)
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        from cyberai import Agent
        from cyberai.handoff import handoff

        target = Agent(name="specialist")
        h = handoff(target)

        model = HandoffTrackingModel()
        await model.get_response(
            system_instructions=None,
            input="delegate",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[h],
            tracing=ModelTracing.ENABLED,
        )

        assert len(received_handoffs) == 1
        assert isinstance(received_handoffs[0], Handoff)
        assert received_handoffs[0].agent_name == "specialist"


class TestModelSettings:
    """Tests for Model with ModelSettings."""

    @pytest.mark.asyncio
    async def test_get_response_receives_settings(self):
        """get_response() receives model settings."""
        received_settings = []

        class SettingsTrackingModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
            ):
                received_settings.append(model_settings)
                return ModelResponse(output=[], usage=Usage())

            async def stream_response(self, *args, **kwargs):
                yield {}

        model = SettingsTrackingModel()
        settings = ModelSettings(temperature=0.7, max_tokens=500)

        await model.get_response(
            system_instructions="Be creative",
            input="Generate a story",
            model_settings=settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.ENABLED,
        )

        assert len(received_settings) == 1
        assert received_settings[0].temperature == 0.7
        assert received_settings[0].max_tokens == 500
