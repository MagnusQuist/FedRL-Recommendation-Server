"""StrEnum of ``*.webp`` stems under ``app/static/product_label_images``. Restart the server after file changes."""

from __future__ import annotations

import re
import warnings
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parents[2] / "static" / "product_label_images"
PRODUCT_LABEL_EMPTY_VALUE = "__no_product_label_image__"


@lru_cache(maxsize=1)
def product_label_image_stems() -> tuple[str, ...]:
    if not _DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in _DIR.glob("*.webp")))


def _member_name(stem: str) -> str:
    if stem.isidentifier() and not stem[0].isdigit():
        return stem.upper()
    s = re.sub(r"[^0-9a-zA-Z_]", "_", stem)
    if not s or not s.isidentifier() or s[0].isdigit():
        return f"ID_{s}"
    return s.upper()


def _build_enum() -> type[StrEnum]:
    used: set[str] = set()
    members: dict[str, str] = {}
    for stem in product_label_image_stems():
        base = _member_name(stem)
        name, n = base, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        members[name] = stem
    if not members:
        warnings.warn(f"No .webp files under {_DIR}; enum uses sentinel only.", stacklevel=2)
        members = {"_NO_ASSETS": PRODUCT_LABEL_EMPTY_VALUE}
    return StrEnum("ProductLabelImage", members)


ProductLabelImage = _build_enum()