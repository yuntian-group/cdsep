"""Control schema engine: scaffolding, parsing, and validation.

A *control schema* is a Python ``dataclass`` (or Pydantic ``BaseModel``) whose
fields describe the structured part of an agent's output -- the part that
Python routing logic uses to decide what happens next. The functions in this
module:

* generate prompt scaffolding from a schema, telling the LLM the exact JSON
  shape it must produce (:func:`generate_scaffolding`),
* parse an LLM response into a ``(control_dict, free_text)`` pair
  (:func:`parse_response`), and
* validate a parsed control dict against the schema, returning either an
  instance of the schema class or a list of human-readable errors
  (:func:`validate_control`).

The library never touches the schema slot of an agent's prompt; only the
data-flow slot is exposed to optimization. This is what gives the framework
its by-construction protocol stability guarantee.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Type, get_args, get_origin, get_type_hints

from pydantic import BaseModel


def _is_dataclass(cls: Type) -> bool:
    return dataclasses.is_dataclass(cls) and isinstance(cls, type)


def _is_pydantic(cls: Type) -> bool:
    try:
        return issubclass(cls, BaseModel)
    except TypeError:
        return False


def _type_description(tp: Any) -> str:
    """Human-readable description of a type for prompt scaffolding."""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is type(None):
        return "null"

    from typing import Literal, Optional, Union
    if origin is Literal:
        options = ", ".join(repr(a) for a in args)
        return f"one of [{options}]"

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return f"{_type_description(non_none[0])} or null"
        return " | ".join(_type_description(a) for a in args)

    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is bool:
        return "boolean (true/false)"
    if tp is str:
        return "string"

    return str(tp)


def get_schema_fields(schema_cls: Type) -> dict[str, Any]:
    """Return ``{field_name: type_annotation}`` for a dataclass or Pydantic model.

    Args:
        schema_cls: A Python ``dataclass`` type or a Pydantic ``BaseModel``
            subclass.

    Returns:
        A mapping from field name to type annotation, in declaration order.

    Raises:
        TypeError: If ``schema_cls`` is neither a dataclass nor a Pydantic
            model.

    Examples:
        >>> from dataclasses import dataclass
        >>> from typing import Literal
        >>> @dataclass
        ... class Ctrl:
        ...     action: Literal["go", "stop"]
        ...     n: int
        >>> get_schema_fields(Ctrl)["action"]
        typing.Literal['go', 'stop']
    """
    if _is_pydantic(schema_cls):
        return {name: info.annotation for name, info in schema_cls.model_fields.items()}
    if _is_dataclass(schema_cls):
        hints = get_type_hints(schema_cls)
        return {f.name: hints[f.name] for f in dataclasses.fields(schema_cls)}
    raise TypeError(f"Unsupported schema type: {schema_cls}. Use a dataclass or Pydantic BaseModel.")


def generate_scaffolding(schema_cls: Type, json_position: str = "begin") -> str:
    """Generate frozen prompt scaffolding from a control schema.

    The returned text is appended to the agent's editable system prompt at
    every LLM call. The optimizer never sees or edits this text -- this is
    what gives the framework its protocol-stability guarantee.

    Args:
        schema_cls: Control schema class (dataclass or Pydantic ``BaseModel``).
        json_position: ``"begin"`` (default; emit JSON then optional message)
            or ``"end"`` (write reasoning/message first, end with JSON). The
            parser handles either; this only affects the prompt instruction.

    Returns:
        A multi-line instruction string telling the model to emit a JSON
        object with the required fields, optionally followed by a free-form
        message.
    """
    fields = get_schema_fields(schema_cls)
    if json_position == "end":
        opening = (
            "You MUST end your response with a JSON control block (after "
            "any step-by-step reasoning or free-form message)."
        )
    else:
        opening = (
            "You MUST begin your response with a JSON control block on its "
            "own, then follow with your free-form message."
        )
    lines = [
        opening,
        "",
        "The JSON control block must be a single JSON object with exactly these fields:",
    ]
    for name, tp in fields.items():
        lines.append(f'  - "{name}": {_type_description(tp)}')
    lines.append("")
    lines.append("Output format (the JSON must be valid and parseable):")
    lines.append('```json')

    example = {}
    for name, tp in fields.items():
        example[name] = _example_value(tp)
    lines.append(json.dumps(example, indent=2))
    lines.append('```')
    lines.append("")
    lines.append("After the JSON block, write your free-form message/explanation.")
    return "\n".join(lines)


def _example_value(tp: Any) -> Any:
    """Generate an example value for a type."""
    origin = get_origin(tp)
    args = get_args(tp)

    from typing import Literal, Union
    if origin is Literal:
        return args[0]

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _example_value(non_none[0])
        return None

    if tp is int:
        return 0
    if tp is float:
        return 0.0
    if tp is bool:
        return False
    if tp is str:
        return "..."
    return "..."


def parse_response(raw: str) -> tuple[dict | None, str]:
    """Parse an LLM response into a ``(control_dict, message)`` pair.

    Tries fenced ```` ```json ... ``` ```` blocks first, then falls back to
    finding the first balanced ``{...}`` object. Anything after the JSON
    block is returned as the free-form message.

    Args:
        raw: The raw response string from the LLM.

    Returns:
        ``(control_dict, message)`` on success; ``(None, raw)`` if no
        valid JSON object can be extracted.

    Examples:
        >>> ctrl, msg = parse_response('{"answer": 42}\\nbecause physics')
        >>> ctrl
        {'answer': 42}
        >>> msg
        'because physics'
    """
    raw = raw.strip()

    # Try to find JSON within ```json ... ``` fences first
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if fence_match:
        try:
            control = json.loads(fence_match.group(1))
            after = raw[fence_match.end():].strip()
            return control, after
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON object (first { to matching })
    brace_start = raw.find('{')
    if brace_start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(brace_start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = raw[brace_start:i + 1]
                        try:
                            control = json.loads(json_str)
                            after = raw[i + 1:].strip()
                            return control, after
                        except json.JSONDecodeError:
                            break

    return None, raw


def validate_control(control_dict: dict, schema_cls: Type) -> tuple[Any | None, list[str]]:
    """Validate a parsed control dict against a schema class.

    Args:
        control_dict: Dict produced by :func:`parse_response`.
        schema_cls: Control schema (dataclass or Pydantic model).

    Returns:
        ``(instance, [])`` if every required field is present with a
        type-correct value; otherwise ``(None, [error_messages])`` listing
        every problem found (missing fields, wrong types, invalid Literal
        values, etc.).
    """
    if control_dict is None:
        return None, ["No control block found in response"]

    fields = get_schema_fields(schema_cls)
    errors = []

    for name, tp in fields.items():
        if name not in control_dict:
            errors.append(f"Missing required field: '{name}'")
            continue
        val = control_dict[name]
        field_errors = _validate_field(name, val, tp)
        errors.extend(field_errors)

    if errors:
        return None, errors

    if _is_pydantic(schema_cls):
        try:
            instance = schema_cls(**control_dict)
            return instance, []
        except Exception as e:
            return None, [str(e)]

    if _is_dataclass(schema_cls):
        try:
            filtered = {k: v for k, v in control_dict.items() if k in fields}
            instance = schema_cls(**filtered)
            return instance, []
        except Exception as e:
            return None, [str(e)]

    return None, [f"Unsupported schema type: {schema_cls}"]


def _validate_field(name: str, value: Any, tp: Any) -> list[str]:
    """Validate a single field value against its type."""
    origin = get_origin(tp)
    args = get_args(tp)

    from typing import Literal, Union

    if origin is Literal:
        if value not in args:
            return [f"Field '{name}': got {value!r}, expected one of {list(args)}"]
        return []

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if type(None) in args:
                return []
            return [f"Field '{name}': got None, but field is not optional"]
        for sub_tp in non_none:
            if not _validate_field(name, value, sub_tp):
                return []
        return [f"Field '{name}': got {type(value).__name__}, expected {_type_description(tp)}"]

    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"Field '{name}': expected integer, got {type(value).__name__}"]
    elif tp is float:
        if not isinstance(value, (int, float)):
            return [f"Field '{name}': expected number, got {type(value).__name__}"]
    elif tp is bool:
        if not isinstance(value, bool):
            return [f"Field '{name}': expected boolean, got {type(value).__name__}"]
    elif tp is str:
        if not isinstance(value, str):
            return [f"Field '{name}': expected string, got {type(value).__name__}"]

    return []
