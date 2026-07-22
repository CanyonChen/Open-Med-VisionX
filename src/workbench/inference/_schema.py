"""Small strict-schema helpers used by the dependency-free manifest model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, TypeVar

from .enums import ControlledStringEnum
from .errors import ManifestValidationError, ValidationIssue

_EnumT = TypeVar("_EnumT", bound=ControlledStringEnum)
_MISSING = object()


def fail(path: str, message: str, value: Any = None) -> NoReturn:
    raise ManifestValidationError(ValidationIssue(path, message, value))


def expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(path, "expected a mapping", value)
    for key in value:
        if not isinstance(key, str):
            fail(path, "mapping keys must be strings", key)
    return value


def expect_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        fail(path, "expected a sequence", value)
    return value


def check_keys(
    mapping: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        fail(path, f"unknown field(s): {', '.join(unknown)}")
    missing = sorted((required or set()) - set(mapping))
    if missing:
        fail(path, f"missing required field(s): {', '.join(missing)}")


def get_required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        fail(f"{path}.{key}", "field is required")
    return value


def get_optional(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return mapping.get(key, default)


def parse_string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        fail(path, "expected a string", value)
    result = value.strip()
    if not result and not allow_empty:
        fail(path, "must not be empty")
    if "\x00" in result:
        fail(path, "must not contain NUL characters")
    return result


def parse_optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return parse_string(value, path)


def parse_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, "expected a boolean", value)
    return value


def parse_int(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "expected an integer", value)
    if minimum is not None and value < minimum:
        fail(path, f"must be at least {minimum}", value)
    if maximum is not None and value > maximum:
        fail(path, f"must be at most {maximum}", value)
    return value


def parse_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, "expected a finite number", value)
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        fail(path, "expected a finite number", value)
    if minimum is not None and result < minimum:
        fail(path, f"must be at least {minimum}", value)
    if maximum is not None and result > maximum:
        fail(path, f"must be at most {maximum}", value)
    return result


def parse_enum(enum_type: type[_EnumT], value: Any, path: str) -> _EnumT:
    try:
        return enum_type.coerce(value)
    except ValueError as exc:
        fail(path, str(exc), value)


def parse_string_tuple(
    value: Any,
    path: str,
    *,
    minimum_length: int = 0,
    unique: bool = False,
) -> tuple[str, ...]:
    items = expect_sequence(value, path)
    result = tuple(parse_string(item, f"{path}[{index}]") for index, item in enumerate(items))
    if len(result) < minimum_length:
        fail(path, f"must contain at least {minimum_length} item(s)")
    if unique and len(set(result)) != len(result):
        fail(path, "items must be unique", result)
    return result


def parse_number_tuple(
    value: Any,
    path: str,
    *,
    minimum_length: int = 0,
    exact_length: int | None = None,
    positive: bool = False,
) -> tuple[float, ...]:
    items = expect_sequence(value, path)
    result = tuple(
        parse_number(item, f"{path}[{index}]", minimum=0.0 if positive else None)
        for index, item in enumerate(items)
    )
    if len(result) < minimum_length:
        fail(path, f"must contain at least {minimum_length} item(s)")
    if exact_length is not None and len(result) != exact_length:
        fail(path, f"must contain exactly {exact_length} items", result)
    if positive and any(item <= 0 for item in result):
        fail(path, "all items must be greater than zero", result)
    return result


def parse_size_2d(value: Any, path: str) -> tuple[int, int]:
    items = expect_sequence(value, path)
    if len(items) != 2:
        fail(path, "expected [height, width]", value)
    return (
        parse_int(items[0], f"{path}[0]", minimum=1),
        parse_int(items[1], f"{path}[1]", minimum=1),
    )


def parse_shape(value: Any, path: str) -> tuple[int | str | None, ...]:
    items = expect_sequence(value, path)
    if not items:
        fail(path, "shape must not be empty")
    result: list[int | str | None] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if item is None:
            result.append(None)
        elif isinstance(item, bool):
            fail(item_path, "boolean is not a valid dimension", item)
        elif isinstance(item, int):
            if item == 0 or item < -1:
                fail(item_path, "dimension must be positive, -1, null, or symbolic", item)
            result.append(None if item == -1 else item)
        elif isinstance(item, str):
            result.append(parse_string(item, item_path))
        else:
            fail(item_path, "dimension must be an integer, null, or symbolic string", item)
    return tuple(result)


def parse_json_value(value: Any, path: str) -> Any:
    """Validate and detach a JSON-compatible extension/metadata value."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            fail(path, "non-finite numbers are not allowed", value)
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                fail(path, "metadata keys must be strings", key)
            result[key] = parse_json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [parse_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    fail(path, "expected a JSON-compatible value", value)
