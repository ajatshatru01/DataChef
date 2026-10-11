"""
ToolRegistry and ToolValidator — the interface between the AI reasoning layer
and the Analytical Execution Layer.

The model can only invoke tools present in this registry. All tool calls are
validated before any analytical function is executed.
"""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from datachef.tools.definitions import TOOL_DEFINITIONS


class ValidationError(Exception):
    """Raised when a tool call fails validation."""


class ToolRegistry:
    """
    Central registry that loads tool definitions at startup and routes
    validated tool calls to the appropriate handler function.
    """

    def __init__(self) -> None:
        self._tools = TOOL_DEFINITIONS

    # ------------------------------------------------------------------
    # Schema access (sent to the model)
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas in OpenAI/HF function-calling format."""
        return [defn["schema"] for defn in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        df: pl.DataFrame,
    ) -> tuple[dict[str, Any], int]:
        """
        Execute the tool identified by *tool_name* with *arguments* on *df*.

        Returns (result_dict, execution_time_ms).
        Raises ValidationError on invalid input (should be caught before calling).
        """
        defn = self._tools.get(tool_name)
        if defn is None:
            raise ValidationError(f"Unknown tool: '{tool_name}'")

        t0 = time.perf_counter()
        try:
            result = defn["handler"](df, **arguments)
        except Exception as exc:
            # Catch analytical errors and surface them as structured results
            result = {"error": f"Tool execution error: {exc}"}
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return result, elapsed_ms


class ToolValidator:
    """
    Validates a model-generated tool call against the registry schemas
    and the loaded dataset's column list.
    """

    def __init__(self, registry: ToolRegistry, column_names: list[str]) -> None:
        self._registry = registry
        self._column_names = set(column_names)

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        """
        Validate *tool_name* and *arguments*.

        Returns a list of error strings. An empty list means the call is valid.
        """
        errors: list[str] = []

        # 1. Tool must exist
        if not self._registry.has_tool(tool_name):
            errors.append(
                f"Tool '{tool_name}' does not exist. "
                f"Available tools: {self._registry.tool_names()}"
            )
            return errors  # Stop here — no schema to check against

        # 2. Required parameters must be present
        schema = self._registry._tools[tool_name]["schema"]["function"]
        params_schema = schema.get("parameters", {})
        required = params_schema.get("required", [])
        for req in required:
            if req not in arguments:
                errors.append(f"Required parameter '{req}' is missing.")

        properties = params_schema.get("properties", {})

        # Reject arguments that are not part of the public schema. This keeps
        # model-generated data from reaching handlers through **kwargs.
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            errors.append(f"Unexpected parameter(s): {unknown}.")

        # Validate JSON-schema types and numeric/list bounds before doing any
        # semantic validation such as checking column names.
        for param_name, value in arguments.items():
            prop = properties.get(param_name)
            if prop is not None:
                errors.extend(_validate_schema_value(param_name, value, prop))

        if errors:
            return errors

        # 3. Column name validation for known column-type parameters
        column_params = {
            "column", "numeric_col", "group_col", "value_col", "time_col",
            "x_col", "y_col", "col_a", "col_b", "color_col",
        }
        for param_name, value in arguments.items():
            if param_name in column_params and isinstance(value, str):
                if value and value not in self._column_names:
                    errors.append(
                        f"Column '{value}' (parameter '{param_name}') does not exist in the dataset. "
                        f"Valid column names are: {sorted(self._column_names)}"
                    )
            elif param_name == "columns" and isinstance(value, list):
                invalid = [c for c in value if c not in self._column_names]
                if invalid:
                    errors.append(
                        f"The following column names are invalid: {invalid}. "
                        f"Valid column names are: {sorted(self._column_names)}"
                    )

        # 4. Enum validation
        for param_name, value in arguments.items():
            prop = properties.get(param_name, {})
            allowed = prop.get("enum")
            if allowed and value not in allowed:
                errors.append(
                    f"Parameter '{param_name}' must be one of {allowed}, got '{value}'."
                )

        return errors

    def validate_and_format_error(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """
        Convenience wrapper. Returns a formatted error string if invalid,
        or None if the call is valid.
        """
        errors = self.validate(tool_name, arguments)
        if not errors:
            return None
        error_body = "\n".join(f"  • {e}" for e in errors)
        return (
            f"Tool call rejected. The following validation errors were found:\n"
            f"{error_body}\n\n"
            f"Please correct your tool call and try again."
        )


def _validate_schema_value(name: str, value: Any, schema: dict[str, Any]) -> list[str]:
    """Validate the JSON-schema subset used by DataChef tool definitions."""
    errors: list[str] = []
    expected = schema.get("type")
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }
    if expected in type_checks and not type_checks[expected](value):
        return [f"Parameter '{name}' must be of type {expected}."]

    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"Parameter '{name}' must be >= {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"Parameter '{name}' must be <= {schema['maximum']}.")

    if expected == "array":
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"Parameter '{name}' requires at least {min_items} item(s).")
        if max_items is not None and len(value) > max_items:
            errors.append(f"Parameter '{name}' accepts at most {max_items} item(s).")
        if schema.get("uniqueItems") and len({repr(item) for item in value}) != len(value):
            errors.append(f"Parameter '{name}' must not contain duplicate items.")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(
                    _validate_schema_value(f"{name}[{index}]", item, item_schema)
                )
    return errors
