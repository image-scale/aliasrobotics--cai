"""
Tests for the tool module.
"""

import json
from typing import Optional

import pytest
from pydantic import BaseModel

from cyberai import FunctionTool, ModelBehaviorError, RunContext, function_tool


class TestFunctionTool:
    """Tests for the FunctionTool dataclass."""

    def test_function_tool_attributes(self):
        """FunctionTool should have required attributes."""
        async def mock_invoke(ctx, args):
            return "ok"

        tool = FunctionTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=mock_invoke,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.params_json_schema == {"type": "object", "properties": {}}
        assert tool.strict_json_schema is True

    def test_function_tool_strict_schema_default(self):
        """FunctionTool strict_json_schema defaults to True."""
        async def mock_invoke(ctx, args):
            return "ok"

        tool = FunctionTool(
            name="test",
            description="test",
            params_json_schema={},
            on_invoke_tool=mock_invoke,
        )
        assert tool.strict_json_schema is True


class TestFunctionToolDecorator:
    """Tests for the function_tool decorator."""

    def test_simple_function(self):
        """Simple function can be converted to a tool."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = function_tool(add)

        assert tool.name == "add"
        assert isinstance(tool, FunctionTool)
        assert "a" in tool.params_json_schema.get("properties", {})
        assert "b" in tool.params_json_schema.get("properties", {})

    def test_function_with_defaults(self):
        """Function with default values works correctly."""

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        tool = function_tool(greet)

        assert tool.name == "greet"
        # name should be required, greeting optional
        required = tool.params_json_schema.get("required", [])
        assert "name" in required

    def test_name_override(self):
        """name_override changes the tool name."""

        def my_function(x: int) -> int:
            return x

        tool = function_tool(my_function, name_override="custom_name")

        assert tool.name == "custom_name"

    def test_description_override(self):
        """description_override changes the tool description."""

        def my_function(x: int) -> int:
            """Original description."""
            return x

        tool = function_tool(my_function, description_override="Custom description")

        assert tool.description == "Custom description"

    def test_docstring_description_extraction(self):
        """Description is extracted from docstring."""

        def documented_function(x: int) -> int:
            """This function does something useful."""
            return x

        tool = function_tool(documented_function)

        assert "something useful" in tool.description

    def test_decorator_syntax_without_args(self):
        """Decorator can be used without parentheses."""

        @function_tool
        def simple_tool(value: str) -> str:
            return value.upper()

        assert isinstance(simple_tool, FunctionTool)
        assert simple_tool.name == "simple_tool"

    def test_decorator_syntax_with_args(self):
        """Decorator can be used with arguments."""

        @function_tool(name_override="renamed_tool")
        def original_name(value: str) -> str:
            return value

        assert isinstance(original_name, FunctionTool)
        assert original_name.name == "renamed_tool"


class TestToolInvocation:
    """Tests for tool invocation."""

    @pytest.mark.asyncio
    async def test_invoke_simple_function(self):
        """Simple function can be invoked through tool."""

        def multiply(a: int, b: int) -> int:
            return a * b

        tool = function_tool(multiply)
        ctx = RunContext(context=None)

        result = await tool.on_invoke_tool(ctx, '{"a": 3, "b": 4}')

        assert result == 12

    @pytest.mark.asyncio
    async def test_invoke_with_defaults(self):
        """Function with defaults invokes correctly."""

        def power(base: int, exponent: int = 2) -> int:
            return base ** exponent

        tool = function_tool(power)
        ctx = RunContext(context=None)

        # With default
        result = await tool.on_invoke_tool(ctx, '{"base": 3}')
        assert result == 9

        # Overriding default
        result = await tool.on_invoke_tool(ctx, '{"base": 2, "exponent": 3}')
        assert result == 8

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        """Function taking RunContext receives it."""

        def get_context_value(ctx: RunContext[dict], key: str) -> str:
            return ctx.context.get(key, "not found")

        tool = function_tool(get_context_value)
        ctx = RunContext(context={"secret": "password123"})

        result = await tool.on_invoke_tool(ctx, '{"key": "secret"}')

        assert result == "password123"

    @pytest.mark.asyncio
    async def test_invoke_async_function(self):
        """Async function can be invoked."""

        async def async_double(value: int) -> int:
            return value * 2

        tool = function_tool(async_double)
        ctx = RunContext(context=None)

        result = await tool.on_invoke_tool(ctx, '{"value": 21}')

        assert result == 42

    @pytest.mark.asyncio
    async def test_invoke_async_with_context(self):
        """Async function with context works."""

        async def async_lookup(ctx: RunContext[dict], key: str) -> str:
            return ctx.context.get(key, "missing")

        tool = function_tool(async_lookup)
        ctx = RunContext(context={"data": "found"})

        result = await tool.on_invoke_tool(ctx, '{"key": "data"}')

        assert result == "found"

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(self):
        """Invalid JSON raises ModelBehaviorError."""

        def dummy(x: int) -> int:
            return x

        tool = function_tool(dummy)
        ctx = RunContext(context=None)

        with pytest.raises(ModelBehaviorError) as exc_info:
            await tool.on_invoke_tool(ctx, "{invalid json}")

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_required_arg_raises_error(self):
        """Missing required argument raises ModelBehaviorError."""

        def requires_args(a: int, b: int) -> int:
            return a + b

        tool = function_tool(requires_args)
        ctx = RunContext(context=None)

        with pytest.raises(ModelBehaviorError):
            await tool.on_invoke_tool(ctx, '{"a": 1}')

    @pytest.mark.asyncio
    async def test_empty_json_for_no_args(self):
        """Empty JSON works for functions with no required args."""

        def no_args() -> str:
            return "ok"

        tool = function_tool(no_args)
        ctx = RunContext(context=None)

        result = await tool.on_invoke_tool(ctx, "")

        assert result == "ok"


class TestDocstringParsing:
    """Tests for docstring parsing."""

    def test_google_style_docstring(self):
        """Google-style docstrings are parsed correctly."""

        def google_style(name: str, count: int) -> str:
            """Process a name multiple times.

            Args:
                name: The name to process.
                count: How many times to repeat.

            Returns:
                The processed result.
            """
            return name * count

        tool = function_tool(google_style)

        assert "Process a name" in tool.description
        # Parameter descriptions should be in schema
        props = tool.params_json_schema.get("properties", {})
        assert props.get("name", {}).get("description") == "The name to process."
        assert props.get("count", {}).get("description") == "How many times to repeat."

    def test_sphinx_style_docstring(self):
        """Sphinx-style docstrings are parsed correctly."""

        def sphinx_style(value: int, factor: int) -> int:
            """Multiply a value by a factor.

            :param value: The input value.
            :param factor: The multiplication factor.
            :return: The multiplied result.
            """
            return value * factor

        tool = function_tool(sphinx_style)

        assert "Multiply a value" in tool.description
        props = tool.params_json_schema.get("properties", {})
        assert props.get("value", {}).get("description") == "The input value."
        assert props.get("factor", {}).get("description") == "The multiplication factor."


class TestJsonSchemaGeneration:
    """Tests for JSON schema generation from type hints."""

    def test_int_type(self):
        """int type generates correct schema."""

        def int_func(x: int) -> int:
            return x

        tool = function_tool(int_func)
        props = tool.params_json_schema.get("properties", {})

        assert props.get("x", {}).get("type") == "integer"

    def test_str_type(self):
        """str type generates correct schema."""

        def str_func(text: str) -> str:
            return text

        tool = function_tool(str_func)
        props = tool.params_json_schema.get("properties", {})

        assert props.get("text", {}).get("type") == "string"

    def test_float_type(self):
        """float type generates correct schema."""

        def float_func(value: float) -> float:
            return value

        tool = function_tool(float_func)
        props = tool.params_json_schema.get("properties", {})

        assert props.get("value", {}).get("type") == "number"

    def test_bool_type(self):
        """bool type generates correct schema."""

        def bool_func(flag: bool) -> bool:
            return flag

        tool = function_tool(bool_func)
        props = tool.params_json_schema.get("properties", {})

        assert props.get("flag", {}).get("type") == "boolean"

    def test_optional_type(self):
        """Optional type allows null."""

        def optional_func(value: Optional[str] = None) -> str:
            return value or "default"

        tool = function_tool(optional_func)
        # Should work without error

    def test_pydantic_model_type(self):
        """Pydantic model type generates nested schema."""

        class InputModel(BaseModel):
            name: str
            value: int

        def model_func(data: InputModel) -> str:
            return f"{data.name}: {data.value}"

        tool = function_tool(model_func)
        props = tool.params_json_schema.get("properties", {})

        # Should have reference to the model schema
        assert "data" in props

    def test_strict_schema_no_additional_properties(self):
        """Strict schema sets additionalProperties to false."""

        def simple(x: int) -> int:
            return x

        tool = function_tool(simple)

        assert tool.params_json_schema.get("additionalProperties") is False
