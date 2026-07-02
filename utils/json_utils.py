"""Helpers for writing strict JSON files."""

import json
import math
from pathlib import Path
from typing import Any


def sanitize_json(value: Any) -> Any:
    """Convert non-finite numbers and path-like values to strict JSON values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except Exception:
            pass
    return value


def dump_json(data: Any, fp, **kwargs) -> None:
    """Write strict JSON that browser JSON parsers can read."""
    json.dump(sanitize_json(data), fp, allow_nan=False, **kwargs)
