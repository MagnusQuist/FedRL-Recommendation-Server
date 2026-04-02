from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.logger import logger
from app.api.schemas.category import Category
from app.api.schemas.food_item import FoodItem
from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup
from app.api.schemas.substitution_group import SubstitutionGroup

# `data/` lives at the repository root (not inside `app/`), so resolve from
# the project root regardless of module location.
REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES_FILE = REPO_ROOT / "data" / "categories.json"
SUBSTITUTION_GROUPS_FILE = REPO_ROOT / "data" / "substitution_groups.json"
PRODUCT_ITEMS_FILE = REPO_ROOT / "data" / "product_items.json"
PRODUCT_ITEMS_JSONL = REPO_ROOT / "data" / "product_items.jsonl"

BATCH_SIZE = 200
# Log progress every N rows scanned from the product file (jsonl or json array).
SEED_PROGRESS_SCAN_INTERVAL = 2000
# Log every N food_items successfully flushed to the DB.
SEED_PROGRESS_INSERT_INTERVAL = 2000


async def _sync_postgres_sequences(session: AsyncSession) -> None:
    """After bulk INSERT with explicit ids, advance SERIAL sequences to MAX(id)."""
    conn = await session.connection()
    dialect = conn.dialect.name
    if dialect != "postgresql":
        return
    await session.execute(
        text(
            "SELECT setval(pg_get_serial_sequence('categories', 'id'), "
            "COALESCE((SELECT MAX(id) FROM categories), 1))"
        )
    )
    await session.execute(
        text(
            "SELECT setval(pg_get_serial_sequence('substitution_groups', 'id'), "
            "COALESCE((SELECT MAX(id) FROM substitution_groups), 1))"
        )
    )


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _iter_products_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _iter_products() -> tuple[Iterator[dict[str, Any]], int, str]:
    """Return (row iterator, row count for progress, label for logs)."""
    if PRODUCT_ITEMS_JSONL.exists():
        total = _count_nonempty_lines(PRODUCT_ITEMS_JSONL)
        return _iter_products_jsonl(PRODUCT_ITEMS_JSONL), total, str(PRODUCT_ITEMS_JSONL.name)
    raw = _load_json(PRODUCT_ITEMS_FILE)
    if not isinstance(raw, list):
        raise ValueError("product_items.json must be a JSON array")
    return iter(raw), len(raw), str(PRODUCT_ITEMS_FILE.name)


def _category_ids_from_payload(mains: list[dict[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for main in mains:
        ids.add(int(main["id"]))
        for sub in main.get("sub_categories", []):
            ids.add(int(sub["id"]))
    return ids


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def _build_food_item(row: dict[str, Any], category_id: int) -> FoodItem:
    weight_raw = row.get("product_weight_in_g") or "0"
    try:
        product_weight_in_g = float(weight_raw)
    except (TypeError, ValueError):
        product_weight_in_g = 0.0
    if product_weight_in_g <= 0:
        product_weight_in_g = 100.0

    subs_raw = row.get("sub_category") or []
    sub_category = [int(x) for x in subs_raw]

    return FoodItem(
        external_code=str(row["id"]),
        name=str(row["name"]),
        category_id=category_id,
        brand=(str(row.get("brand") or "").strip() or None),
        price_dkk=float(row.get("price_dkk") or 0.0),
        product_weight_in_g=product_weight_in_g,
        co2_kg_per_kg=float(row.get("co2_kg_per_kg") or 0.0),
        calories_per_100g=_opt_float(row.get("calories_per_100g")),
        protein_g_per_100g=_opt_float(row.get("protein_g_per_100g")),
        fat_g_per_100g=_opt_float(row.get("fat_g_per_100g")),
        carbs_g_per_100g=_opt_float(row.get("carbs_g_per_100g")),
        fiber_g_per_100g=_opt_float(row.get("fiber_g_per_100g")),
        salt_g_per_100g=_opt_float(row.get("salt_g_per_100g")),
        main_category=int(row.get("main_category", 0)),
        sub_category=sub_category,
        is_liquid=_bool(row, "is_liquid"),
        is_gluten_free=_bool(row, "is_gluten_free"),
        is_sugar_free=_bool(row, "is_sugar_free"),
        is_oekomærket_eu=_bool(row, "is_oekomærket_eu"),
        is_oekomærket_dk=_bool(row, "is_oekomærket_dk"),
        is_noeglehulsmaerket=_bool(row, "is_noeglehulsmaerket"),
        is_fuldkornsmaerket=_bool(row, "is_fuldkornsmaerket"),
        is_frozen=_bool(row, "is_frozen"),
        is_msc_maerket=_bool(row, "is_msc_maerket"),
        is_fairtrade=_bool(row, "is_fairtrade"),
        is_rainforest_alliance=_bool(row, "is_rainforest_alliance"),
        is_danish=_bool(row, "is_danish"),
    )


async def seed_catalogue() -> None:
    mains_raw = _load_json(CATEGORIES_FILE)
    groups_raw = _load_json(SUBSTITUTION_GROUPS_FILE)
    products_iter, product_total, product_label = _iter_products()

    if not isinstance(mains_raw, list) or not mains_raw:
        logger.warning("No categories in categories.json; nothing to seed.")
        return
    if product_total == 0:
        logger.warning("No product rows in %s; nothing to seed.", product_label)
        return
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("substitution_groups.json must be a non-empty array.")

    logger.info(
        "Catalogue seed: products from %s (%d rows). "
        "Tip: run `python data/convert_catalogue_to_jsonl.py` to stream without loading full JSON.",
        product_label,
        product_total,
    )

    valid_category_ids = _category_ids_from_payload(mains_raw)
    group_by_id = {int(g["id"]): g for g in groups_raw if "id" in g}

    async with AsyncSessionLocal() as session:
        existing_count = await session.scalar(select(func.count()).select_from(FoodItem))
        if existing_count and existing_count > 0:
            logger.info("Catalogue already seeded with %d food items; skipping.", existing_count)
            return

        logger.info("Catalogue seed: inserting categories…")
        for main in mains_raw:
            mid = int(main["id"])
            session.add(
                Category(
                    id=mid,
                    code=f"cat_{mid}",
                    name=str(main["name"]),
                    parent_id=None,
                )
            )
        await session.flush()

        for main in mains_raw:
            mid = int(main["id"])
            for sub in main.get("sub_categories", []):
                sid = int(sub["id"])
                session.add(
                    Category(
                        id=sid,
                        code=f"cat_{sid}",
                        name=str(sub["name"]),
                        parent_id=mid,
                    )
                )
        await session.flush()

        logger.info("Catalogue seed: inserting %d substitution groups…", len(group_by_id))
        for gid, g in sorted(group_by_id.items()):
            session.add(
                SubstitutionGroup(
                    id=gid,
                    code=str(gid),
                    name=str(g["name"]),
                    description=None,
                )
            )
        await session.flush()

        await _sync_postgres_sequences(session)

        skipped = 0
        seeded = 0
        scanned = 0
        pending_links: list[tuple[FoodItem, int]] = []
        last_insert_log = 0

        async def flush_batch() -> None:
            nonlocal seeded, pending_links, last_insert_log
            if not pending_links:
                return
            for fi, _ in pending_links:
                session.add(fi)
            await session.flush()
            for fi, sub_gid in pending_links:
                session.add(
                    FoodItemSubstitutionGroup(
                        food_item_id=fi.id,
                        substitution_group_id=sub_gid,
                        group_priority=1,
                    )
                )
            await session.flush()
            seeded += len(pending_links)
            pending_links = []
            if seeded - last_insert_log >= SEED_PROGRESS_INSERT_INTERVAL:
                last_insert_log = seeded
                pct = 100.0 * seeded / product_total if product_total else 0.0
                logger.info(
                    "Catalogue seed: inserted %d / %d products (%.1f%% by row count, some rows may skip).",
                    seeded,
                    product_total,
                    pct,
                )

        logger.info("Catalogue seed: scanning products…")
        for row in products_iter:
            scanned += 1
            if scanned % SEED_PROGRESS_SCAN_INTERVAL == 0:
                pct = 100.0 * scanned / product_total
                logger.info(
                    "Catalogue seed: scanned %d / %d lines (%.1f%%).",
                    scanned,
                    product_total,
                    pct,
                )

            if not isinstance(row, dict):
                skipped += 1
                continue
            subs = row.get("sub_category") or []
            if not subs:
                skipped += 1
                continue
            category_id = int(subs[0])
            if category_id not in valid_category_ids:
                skipped += 1
                continue
            sub_gid = int(row.get("substitution_group", -1))
            if sub_gid not in group_by_id:
                skipped += 1
                continue

            fi = _build_food_item(row, category_id)
            pending_links.append((fi, sub_gid))

            if len(pending_links) >= BATCH_SIZE:
                await flush_batch()

        await flush_batch()

        await session.commit()
        logger.info(
            "Catalogue seed done: %d food items from %s (%d substitution groups, %d category ids). "
            "Skipped %d invalid rows.",
            seeded,
            product_label,
            len(group_by_id),
            len(valid_category_ids),
            skipped,
        )


if __name__ == "__main__":
    asyncio.run(seed_catalogue())
