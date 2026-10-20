from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def _parse_value(value):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def apply_overrides(config, overrides):
    """Apply command-line overrides written as ``--section.key value``."""
    output = deepcopy(config)
    index = 0
    while index < len(overrides):
        key = overrides[index]
        if not key.startswith("--"):
            raise ValueError(f"Expected an override key, got {key!r}")
        key = key[2:]
        if index + 1 >= len(overrides) or overrides[index + 1].startswith("--"):
            value = True
            index += 1
        else:
            value = _parse_value(overrides[index + 1])
            index += 2

        target = output
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return output
