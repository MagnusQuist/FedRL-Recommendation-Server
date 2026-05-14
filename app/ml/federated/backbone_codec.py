"""Encoding helpers for federated backbone weight blobs."""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

import numpy as np

from app.logging import logger


def encode_backbone_blob(weights: dict[str, np.ndarray]) -> str:
    """Encode NumPy backbone weights as gzip-compressed base64 JSON."""
    weights_json = {key: value.tolist() for key, value in weights.items()}
    compressed = gzip.compress(json.dumps(weights_json).encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


def decode_backbone_blob(blob: str) -> dict[str, list]:
    logger.info("Decoding backbone weights")
    try:
        compressed_bytes = base64.b64decode(blob, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 encoding in backbone_weights") from e

    try:
        json_bytes = gzip.decompress(compressed_bytes)
    except Exception as e:
        raise ValueError("Invalid gzip payload in backbone_weights") from e

    try:
        decoded: Any = json.loads(json_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("Invalid JSON in decompressed backbone_weights") from e

    if not isinstance(decoded, dict):
        raise ValueError("Decoded backbone_weights must be a JSON object")

    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ValueError(
                "Decoded backbone_weights contains a non-string parameter name"
            )
        if not isinstance(value, list):
            raise ValueError(
                f"Decoded backbone_weights parameter '{key}' must map to a list"
            )

    return decoded


def decode_backbone_arrays(blob: str) -> dict[str, np.ndarray]:
    decoded = decode_backbone_blob(blob)
    return {key: np.array(value, dtype=np.float32) for key, value in decoded.items()}
