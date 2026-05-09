"""
Function tool system for converting Python functions to LLM-callable tools.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Union, get_type_hints, get_origin, get_args, overload

from pydantic import BaseModel, Field, ValidationError, create_model

from .exceptions import ModelBehaviorError
from .run_context import RunContext

# Type aliases
ToolFunction = Callable[..., Any]


@dataclass
class FunctionTool:
    """A tool that wraps a function for LLM invocation.

    In most cases, you should use the `function_tool` decorator to create
    a FunctionTool, as it lets you easily wrap a Python function.
    """

    name: str
    """The name of the tool, as shown to the LLM."""

    description: str
    """A description of the tool, as shown to the LLM."""

    params_json_schema: dict[str, Any]
    """The JSON schema for the tool's parameters."""

    on_invoke_tool: Callable[[RunContext[Any], str], Any]
    """A function that invokes the tool with the given context and parameters.
    The params passed are:
    1. The tool run context.
    2. The arguments from the LLM, as a JSON string.

    You must return a string representation of the tool output, or something
    we can call str() on.
    """

    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode."""


@dataclass
class FunctionSchema:
    """Captures the schema for a Python function for LLM tool use."""

    name: str
    """The name of the function."""

    description: str | None
    """The description of the function."""

    params_pydantic_model: type[BaseModel]
    """A Pydantic model representing the function's parameters."""

    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters."""

    signature: inspect.Signature
    """The signature of the function."""

    takes_context: bool = False
    """Whether the function takes a RunContext as the first argument."""


def _detect_docstring_style(doc: str) -> str:
    """Detect the docstring style (google, sphinx, numpy)."""
    if ":param" in doc or ":type" in doc:
        return "sphinx"
    if "Args:" in doc or "Arguments:" in doc:
        return "google"
    if "Parameters\n" in doc and "----------" in doc:
        return "numpy"
    return "google"


def _parse_docstring(doc: str | None) -> tuple[str | None, dict[str, str]]:
    """Parse a docstring to extract description and parameter descriptions.

    Returns:
        A tuple of (description, param_descriptions dict).
    """
    if not doc:
        return None, {}

    lines = doc.strip().split("\n")
    description_lines = []
    param_descriptions: dict[str, str] = {}

    style = _detect_docstring_style(doc)

    if style == "google":
        in_args_section = False
        current_param = None
        current_desc = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith(("Args:", "Arguments:")):
                in_args_section = True
                continue
            elif stripped.startswith(("Returns:", "Raises:", "Yields:", "Examples:")):
                if current_param and current_desc:
                    param_descriptions[current_param] = " ".join(current_desc).strip()
                in_args_section = False
                continue

            if not in_args_section:
                if not stripped.startswith(("Args:", "Returns:", "Raises:")):
                    description_lines.append(stripped)
            else:
                # Check if this is a new parameter (has colon after name)
                if ":" in stripped and not stripped.startswith(" "):
                    if current_param and current_desc:
                        param_descriptions[current_param] = " ".join(current_desc).strip()

                    parts = stripped.split(":", 1)
                    # Handle "param_name (type): description" format
                    param_part = parts[0].strip()
                    if "(" in param_part:
                        param_part = param_part.split("(")[0].strip()
                    current_param = param_part
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []
                elif current_param:
                    current_desc.append(stripped)

        if current_param and current_desc:
            param_descriptions[current_param] = " ".join(current_desc).strip()

    elif style == "sphinx":
        current_param = None
        current_desc = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith(":param"):
                if current_param and current_desc:
                    param_descriptions[current_param] = " ".join(current_desc).strip()

                # Parse ":param name: description" or ":param type name: description"
                rest = stripped[6:].strip()  # Remove ":param"
                if ":" in rest:
                    parts = rest.split(":", 1)
                    param_name = parts[0].strip().split()[-1]  # Get last word (the name)
                    current_param = param_name
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif stripped.startswith((":type", ":return", ":rtype", ":raises")):
                if current_param and current_desc:
                    param_descriptions[current_param] = " ".join(current_desc).strip()
                current_param = None
                current_desc = []
            elif not stripped.startswith(":") and current_param is None:
                description_lines.append(stripped)
            elif current_param:
                current_desc.append(stripped)

        if current_param and current_desc:
            param_descriptions[current_param] = " ".join(current_desc).strip()

    description = " ".join(description_lines).strip()
    description = description if description else None

    return description, param_descriptions


def _generate_function_schema(
    func: ToolFunction,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FunctionSchema:
    """Generate a FunctionSchema from a Python function."""
    # Get function info
    func_name = name_override or func.__name__
    doc = inspect.getdoc(func)
    parsed_desc, param_descs = _parse_docstring(doc)
    description = description_override or parsed_desc

    # Get signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    # Check if first param is RunContext
    if params:
        first_name, first_param = params[0]
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect.Parameter.empty:
            origin = get_origin(ann) or ann
            if origin is RunContext:
                takes_context = True
            else:
                filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))

    # Add remaining params
    filtered_params.extend(params[1:])

    # Build Pydantic model fields
    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default

        if ann == inspect.Parameter.empty:
            ann = Any

        field_desc = param_descs.get(name)

        if default == inspect.Parameter.empty:
            # Required field
            fields[name] = (ann, Field(..., description=field_desc))
        else:
            # Optional field with default
            fields[name] = (ann, Field(default=default, description=field_desc))

    # Create dynamic Pydantic model
    model = create_model(f"{func_name}_args", **fields)
    json_schema = model.model_json_schema()

    # Ensure strict mode schema
    if "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # Handle $defs for nested schemas
    if "$defs" in json_schema:
        for def_name, def_schema in json_schema["$defs"].items():
            if "additionalProperties" not in def_schema:
                def_schema["additionalProperties"] = False

    return FunctionSchema(
        name=func_name,
        description=description,
        params_pydantic_model=model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
    )


@overload
def function_tool(
    func: ToolFunction,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FunctionTool: ...


@overload
def function_tool(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> Callable[[ToolFunction], FunctionTool]: ...


def function_tool(
    func: ToolFunction | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FunctionTool | Callable[[ToolFunction], FunctionTool]:
    """
    Decorator to create a FunctionTool from a function.

    By default, we will:
    1. Parse the function signature to create a JSON schema for parameters.
    2. Use the function's docstring to populate the description.
    3. Use the function's docstring to populate argument descriptions.

    If the function takes a RunContext as the first argument, it will be
    passed the context from the agent run.

    Args:
        func: The function to wrap.
        name_override: If provided, use this name instead of the function's name.
        description_override: If provided, use this description instead of the
            function's docstring.

    Returns:
        A FunctionTool wrapping the function.
    """

    def _create_function_tool(the_func: ToolFunction) -> FunctionTool:
        schema = _generate_function_schema(
            the_func,
            name_override=name_override,
            description_override=description_override,
        )

        async def _on_invoke_tool(ctx: RunContext[Any], input_json: str) -> Any:
            # Parse JSON input
            try:
                json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
            except json.JSONDecodeError as e:
                raise ModelBehaviorError(
                    f"Invalid JSON input for tool {schema.name}: {input_json}"
                ) from e

            # Validate with Pydantic
            try:
                parsed = (
                    schema.params_pydantic_model(**json_data)
                    if json_data
                    else schema.params_pydantic_model()
                )
            except ValidationError as e:
                raise ModelBehaviorError(
                    f"Invalid JSON input for tool {schema.name}: {e}"
                ) from e

            # Build args and kwargs from parsed model
            args = []
            kwargs = {}

            for idx, (name, param) in enumerate(schema.signature.parameters.items()):
                if schema.takes_context and idx == 0:
                    continue

                value = getattr(parsed, name, None)

                if param.kind in (
                    param.POSITIONAL_ONLY,
                    param.POSITIONAL_OR_KEYWORD,
                ):
                    args.append(value)
                else:
                    kwargs[name] = value

            # Call the function
            if inspect.iscoroutinefunction(the_func):
                if schema.takes_context:
                    result = await the_func(ctx, *args, **kwargs)
                else:
                    result = await the_func(*args, **kwargs)
            else:
                # Run sync function
                if schema.takes_context:
                    result = the_func(ctx, *args, **kwargs)
                else:
                    result = the_func(*args, **kwargs)

            return result

        return FunctionTool(
            name=schema.name,
            description=schema.description or "",
            params_json_schema=schema.params_json_schema,
            on_invoke_tool=_on_invoke_tool,
            strict_json_schema=True,
        )

    if callable(func):
        return _create_function_tool(func)

    def decorator(real_func: ToolFunction) -> FunctionTool:
        return _create_function_tool(real_func)

    return decorator
