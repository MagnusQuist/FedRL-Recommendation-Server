"""Load catalogue CSV/JSON from ``data/``, wipe related tables, bulk insert, fix sequences."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import MetaData, delete, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


TABLE_NAMES = [
    "catalogue_versions",
    "categories",
    "food_item_categories",
    "substitution_groups",
    "substitution_group_items",
    "food_items",
]


DATA_DIR = Path(__file__).resolve().parent / "data"


def _get_database_url() -> str:
    """``DATABASE_URL`` (async SQLAlchemy URL)."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Expected an async SQLAlchemy URL such as "
            "'postgresql+asyncpg://user:password@host:5432/dbname'."
        )
    return database_url


def _clean_value(value: Any) -> Any:
    """Strip strings; map empty string to None."""
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return value


def _to_int(value: Any) -> int | None:
    value = _clean_value(value)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value).replace(".", "").strip()) if str(value).isdigit() else int(str(value).strip())


def _to_float(value: Any) -> float | None:
    value = _clean_value(value)
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, (int, float)):
        return float(str(value))
    return float(str(value).replace(",", ".").strip())


def _to_bool(value: Any) -> bool:
    value = _clean_value(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "yes", "y", "t"}


def _to_str(value: Any) -> str | None:
    value = _clean_value(value)
    if value is None:
        return None
    return str(value)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Parse CSV with sniffer; strip header keys and cell values."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, Any]] = []

        for row in reader:
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                cleaned[key.strip()] = _clean_value(value)
            rows.append(cleaned)

        return rows


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Top-level list or dict with ``items``/``products``/… key holding a list."""
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("items", "products", "food_items", "food_products", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        f"Unsupported JSON structure in {path.name}. "
        "Expected either a top-level list or a dict containing one of: "
        "'items', 'products', 'food_items', 'food_products', 'data'."
    )


def _load_categories(data_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(data_dir / "categories.csv")
    return [
        {
            "category_id": _to_int(row["category_id"]),
            "name": _to_str(row["name"]),
            "slug": _to_str(row["slug"]),
        }
        for row in rows
    ]


def _load_food_item_categories(data_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(data_dir / "food_item_categories.csv")
    return [
        {
            "product_id": _to_str(row["product_id"]),
            "category_id": _to_int(row["category_id"]),
        }
        for row in rows
    ]


def _load_substitution_groups(data_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(data_dir / "substitution_groups.csv")
    return [
        {
            "substitution_group_id": _to_int(row["substitution_group_id"]),
            "name": _to_str(row["name"]),
        }
        for row in rows
    ]


def _load_substitution_group_items(data_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(data_dir / "substitution_group_items.csv")
    return [
        {
            "substitution_group_id": _to_int(row["substitution_group_id"]),
            "product_id": _to_str(row["product_id"]),
        }
        for row in rows
    ]


def _load_food_items(data_dir: Path) -> list[dict[str, Any]]:
    rows = _read_json_rows(data_dir / "food_products.json")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "id": _to_str(row.get("id")),
                "name": _to_str(row.get("name")),
                "brand": _to_str(row.get("brand")),
                "product_weight_in_g": _to_int(row.get("product_weight_in_g")),
                "co2e_kg_pr_item_kg": _to_float(row.get("co2e_kg_pr_item_kg")),
                "estimated_co2e_kg_pr_item_weight_in_g": _to_float(row.get("estimated_co2e_kg_pr_item_weight_in_g")),
                "calories_per_100g": _to_int(row.get("calories_per_100g")),
                "protein_g_per_100g": _to_float(row.get("protein_g_per_100g")),
                "fat_g_per_100g": _to_float(row.get("fat_g_per_100g")),
                "carbs_g_per_100g": _to_float(row.get("carbs_g_per_100g")),
                "fiber_g_per_100g": _to_float(row.get("fiber_g_per_100g")),
                "salt_g_per_100g": _to_float(row.get("salt_g_per_100g")),
                "is_liquid": _to_bool(row.get("is_liquid")),
                "is_gluten_free": _to_bool(row.get("is_gluten_free")),
                "is_sugar_free": _to_bool(row.get("is_sugar_free")),
                "is_oekomærket_eu": _to_bool(row.get("is_oekomærket_eu")),
                "is_oekomærket_dk": _to_bool(row.get("is_oekomærket_dk")),
                "is_noeglehulsmaerket": _to_bool(row.get("is_noeglehulsmaerket")),
                "is_fuldkornsmaerket": _to_bool(row.get("is_fuldkornsmaerket")),
                "is_frozen": _to_bool(row.get("is_frozen")),
                "is_msc_maerket": _to_bool(row.get("is_msc_maerket")),
                "is_fairtrade": _to_bool(row.get("is_fairtrade")),
                "is_rainforest_alliance": _to_bool(row.get("is_rainforest_alliance")),
                "is_danish": _to_bool(row.get("is_danish")),
                "price_dkk": _to_float(row.get("price_dkk")),
            }
        )

    return normalized


async def _reflect_tables(engine: AsyncEngine) -> MetaData:
    """Reflect ``TABLE_NAMES`` for raw insert/delete."""
    metadata = MetaData()

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: metadata.reflect(bind=sync_conn, only=TABLE_NAMES)
        )

    return metadata


async def _reset_sequence(conn, table_name: str, pk_column: str) -> None:
    """Align PostgreSQL serial after explicit PK inserts."""
    await conn.execute(
        text(
            """
            SELECT setval(
                pg_get_serial_sequence(:table_name, :pk_column),
                COALESCE((SELECT MAX(""" + pk_column + """) FROM """ + table_name + """), 1),
                true
            )
            """
        ),
        {
            "table_name": table_name,
            "pk_column": pk_column,
        },
    )


async def seed_catalogue() -> None:
    """Replace catalogue tables from disk files (FK-safe order, then ``setval``)."""
    data_dir = DATA_DIR
    database_url = _get_database_url()

    categories = _load_categories(data_dir)
    food_item_categories = _load_food_item_categories(data_dir)
    substitution_groups = _load_substitution_groups(data_dir)
    substitution_group_items = _load_substitution_group_items(data_dir)
    food_items = _load_food_items(data_dir)

    engine = create_async_engine(database_url, future=True)
    metadata = await _reflect_tables(engine)

    categories_table = metadata.tables["categories"]
    food_item_categories_table = metadata.tables["food_item_categories"]
    substitution_groups_table = metadata.tables["substitution_groups"]
    substitution_group_items_table = metadata.tables["substitution_group_items"]
    food_items_table = metadata.tables["food_items"]
    catalogue_versions_table = metadata.tables["catalogue_versions"]

    async with engine.begin() as conn:
        # Clear in FK-safe order
        await conn.execute(delete(food_item_categories_table))
        await conn.execute(delete(substitution_group_items_table))
        await conn.execute(delete(categories_table))
        await conn.execute(delete(substitution_groups_table))
        await conn.execute(delete(food_items_table))

        # Insert parent tables
        if categories:
            await conn.execute(categories_table.insert(), categories)

        if substitution_groups:
            await conn.execute(substitution_groups_table.insert(), substitution_groups)

        if food_items:
            await conn.execute(food_items_table.insert(), food_items)

        # Insert join tables
        if food_item_categories:
            await conn.execute(food_item_categories_table.insert(), food_item_categories)

        if substitution_group_items:
            await conn.execute(
                substitution_group_items_table.insert(), substitution_group_items
            )

        # Register a new catalogue snapshot version for this seed run.
        seeded_at = datetime.now(timezone.utc)
        seeded_version = str(uuid4())
        await conn.execute(
            catalogue_versions_table.insert(),
            [{"version": seeded_version, "generated_at": seeded_at}],
        )

        # Reset sequences for explicit PK inserts
        await _reset_sequence(conn, "categories", "category_id")
        await _reset_sequence(conn, "substitution_groups", "substitution_group_id")
        await _reset_sequence(conn, "catalogue_versions", "id")

    await engine.dispose()

    print(
        "Catalogue seeded successfully: "
        f"catalogue_version={seeded_version}, "
        f"{len(food_items)} food_items, "
        f"{len(categories)} categories, "
        f"{len(food_item_categories)} food_item_categories, "
        f"{len(substitution_groups)} substitution_groups, "
        f"{len(substitution_group_items)} substitution_group_items."
    )


if __name__ == "__main__":
    asyncio.run(seed_catalogue())

