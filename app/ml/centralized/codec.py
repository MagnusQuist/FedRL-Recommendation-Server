"""gzip/base64 JSON helpers for centralized training state."""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

import torch.nn as nn


def encode_json_blob(value: Any) -> str:
    compressed = gzip.compress(json.dumps(value).encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


def decode_json_blob(blob: str) -> Any:
    try:
        compressed_bytes = base64.b64decode(blob, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 encoding in centralized payload") from e

    try:
        json_bytes = gzip.decompress(compressed_bytes)
    except Exception as e:
        raise ValueError("Invalid gzip payload in centralized payload") from e

    try:
        return json.loads(json_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("Invalid JSON in centralized payload") from e


def decode_tuples(data: str) -> list[dict]:
    decoded = decode_json_blob(data)
    if not isinstance(decoded, list):
        raise ValueError("Decoded interaction data must be a JSON list")
    return decoded


def module_state_to_json(module: nn.Module) -> dict:
    return {key: value.tolist() for key, value in module.state_dict().items()}
