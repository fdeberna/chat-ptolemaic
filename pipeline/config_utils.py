from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must use key=value syntax: {raw}")
    key, value_raw = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key cannot be empty: {raw}")
    try:
        value = json.loads(value_raw)
    except json.JSONDecodeError:
        value = value_raw
    return key, value


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node: Dict[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for raw in overrides:
        key, value = parse_override(raw)
        set_nested(config, key, value)
    return config


def resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path)
