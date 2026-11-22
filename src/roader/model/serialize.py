"""Canonical JSON for the native model (D-011, AC-H5).

Sorted keys, no whitespace, floats rounded to 1e-6 and NaN written as null, so
the same model always produces the same bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from enum import Enum
from typing import Any

from roader.model.schema import RoadModel

FLOAT_DIGITS = 6


def stable_id(prefix: str, *parts: object) -> str:
    """Id from content, never from input order (D-011, NFR-2)."""
    text = "|".join(_id_part(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _id_part(part: object) -> str:
    if isinstance(part, float):
        return repr(round(part, 3))
    if isinstance(part, (list, tuple)):
        return "[" + ",".join(_id_part(p) for p in part) + "]"
    return str(part)


def to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        value = round(obj, FLOAT_DIGITS)
        return 0.0 if value == 0.0 else value
    if isinstance(obj, (list, tuple)):
        return [to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_plain(value) for key, value in obj.items()}
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return to_plain(obj.item())
    return obj


def dumps(model: RoadModel) -> bytes:
    text = json.dumps(to_plain(model), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def model_hash(model: RoadModel) -> str:
    return hashlib.sha256(dumps(model)).hexdigest()


def loads(data: bytes | str) -> RoadModel:
    return _build(RoadModel, json.loads(data))


def _build(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        options = [arg for arg in typing.get_args(tp) if arg is not type(None)]
        return _build(options[0], value)
    if origin in (list, tuple):
        args = typing.get_args(tp)
        if origin is tuple:
            return tuple(_build(args[0] if args else Any, item) for item in value)
        return [_build(args[0] if args else Any, item) for item in value]
    if origin is dict:
        _key, val_tp = typing.get_args(tp)
        return {key: _build(val_tp, item) for key, item in value.items()}
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        kwargs = {f.name: _build(hints[f.name], value[f.name]) for f in dataclasses.fields(tp) if f.name in value}
        return tp(**kwargs)
    if tp is float:
        return float(value)
    return value
