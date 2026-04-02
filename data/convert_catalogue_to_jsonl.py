"""Convert large JSON arrays to JSON Lines for faster, lower-memory catalogue seeding.

Usage (from repo root):
    python data/convert_catalogue_to_jsonl.py

Reads  data/product_items.json
Writes data/product_items.jsonl  (one JSON object per line)

Seed prefers product_items.jsonl when present; otherwise loads the .json file.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "product_items.json"
DST = REPO_ROOT / "data" / "product_items.jsonl"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing {SRC}")
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("product_items.json must be a JSON array")
    with DST.open("w", encoding="utf-8") as out:
        for i, obj in enumerate(raw):
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")
            if (i + 1) % 2000 == 0:
                print(f"  wrote {i + 1} lines...")
    print(f"Wrote {len(raw)} lines to {DST}")


if __name__ == "__main__":
    main()
