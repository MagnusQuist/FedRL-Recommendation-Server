from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.logger import logger

# `data/` lives at the repository root (not inside `app/`), so resolve from
# the project root regardless of module location.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_CANDIDATES = (
    REPO_ROOT / "data",
    REPO_ROOT / "data" / "food_product_enrichment",
)

BATCH_SIZE = int(os.getenv("CATALOGUE_SEED_BATCH_SIZE", "250"))
RESEED = os.getenv("CATALOGUE_RESEED", "0").strip().lower() in {"1", "true", "yes", "y"}
SEED_PROGRESS_SCAN_INTERVAL = 250
SEED_PROGRESS_INSERT_INTERVAL = 250

PRODUCTS_FILE_NAME = "final_product_list_enriched.csv"
PRODUCTS_JSONL_NAME = "final_product_list_enriched.jsonl"
CATEGORY_TAXONOMY_FILE_NAME = "category_taxonomy.csv"
SUBSTITUTION_GROUPS_FILE_NAME = "substitution_groups.csv"
PRODUCT_GROUP_MAP_FILE_NAME = "product_substitution_group_map.csv"
SCHEMA_FILE_NAME = "database_schema.sql"

PRODUCT_COLUMNS = [
    "id",
    "name",
    "brand",
    "product_weight_in_g",
    "co2_kg_per_kg",
    "calories_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "salt_g_per_100g",
    "is_liquid",
    "is_gluten_free",
    "is_sugar_free",
    "is_oekomærket_eu",
    "is_oekomærket_dk",
    "is_noeglehulsmaerket",
    "is_fuldkornsmaerket",
    "is_frozen",
    "is_msc_maerket",
    "is_fairtrade",
    "is_rainforest_alliance",
    "is_danish",
    "price_dkk",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "canonical_product_type",
    "prep_form",
    "primary_use",
    "flavor_variant",
    "sub_group_core",
    "sub_group_extended",
    "size_band",
    "sub_tier_default",
    "label_confidence",
    "needs_review",
]

SUBSTITUTION_GROUP_COLUMNS = [
    "sub_group_id",
    "group_level",
    "group_name",
    "member_count",
]

PRODUCT_GROUP_MAP_COLUMNS = [
    "product_id",
    "sub_group_id",
    "group_level",
]

CATEGORY_TAXONOMY_COLUMNS = [
    "category_id",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "product_count",
]

EXTRA_INDEX_DDLS = [
    "CREATE INDEX IF NOT EXISTS idx_products_primary_use ON products_enriched(primary_use)",
    "CREATE INDEX IF NOT EXISTS idx_products_prep_form ON products_enriched(prep_form)",
    "CREATE INDEX IF NOT EXISTS idx_products_frozen ON products_enriched(is_frozen)",
    "CREATE INDEX IF NOT EXISTS idx_products_liquid ON products_enriched(is_liquid)",
    "CREATE INDEX IF NOT EXISTS idx_map_product ON product_substitution_group_map(product_id, group_level)",
]


def _resolve_data_path(file_name: str) -> Path:
    for base in DATA_DIR_CANDIDATES:
        candidate = base / file_name
        if candidate.exists():
            return candidate
    expected = " or ".join(str(base / file_name) for base in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(f"Dataset file not found. Expected one of: {expected}")


SCHEMA_FILE = _resolve_data_path(SCHEMA_FILE_NAME)
CATEGORY_TAXONOMY_FILE = _resolve_data_path(CATEGORY_TAXONOMY_FILE_NAME)
SUBSTITUTION_GROUPS_FILE = _resolve_data_path(SUBSTITUTION_GROUPS_FILE_NAME)
PRODUCT_GROUP_MAP_FILE = _resolve_data_path(PRODUCT_GROUP_MAP_FILE_NAME)


def _resolve_products_path() -> tuple[Path, str]:
    for file_name in (PRODUCTS_FILE_NAME, PRODUCTS_JSONL_NAME):
        try:
            return _resolve_data_path(file_name), file_name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Could not find {PRODUCTS_FILE_NAME} or {PRODUCTS_JSONL_NAME} in any supported data directory."
    )


PRODUCTS_FILE, PRODUCTS_LABEL = _resolve_products_path()


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False

    for char in sql:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double

        if char == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)

    return statements


def _normalize_schema_sql(sql: str) -> str:
    normalized = re.sub(r"CREATE TABLE\s+(?!IF NOT EXISTS)", "CREATE TABLE IF NOT EXISTS ", sql)
    normalized = re.sub(r"CREATE UNIQUE INDEX\s+(?!IF NOT EXISTS)", "CREATE UNIQUE INDEX IF NOT EXISTS ", normalized)
    normalized = re.sub(r"CREATE INDEX\s+(?!IF NOT EXISTS)", "CREATE INDEX IF NOT EXISTS ", normalized)
    return normalized


def _load_schema_sql(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    return _split_sql_statements(_normalize_schema_sql(raw))


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row:
                yield row


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _parse_text(value: Any, *, allow_blank: bool = False) -> str | None:
    if value is None:
        return "" if allow_blank else None
    text_value = str(value).strip()
    if not text_value and not allow_blank:
        return None
    return text_value


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    text_value = str(value).strip()
    if text_value == "":
        return None
    return float(text_value)


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text_value = str(value).strip()
    if text_value == "":
        return None
    return int(text_value)


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text_value = str(value).strip().lower()
    if text_value == "":
        return None
    if text_value in {"1", "true", "t", "yes", "y"}:
        return True
    if text_value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _build_product_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _parse_text(row.get("id")),
        "name": _parse_text(row.get("name")),
        "brand": _parse_text(row.get("brand")),
        "product_weight_in_g": _parse_text(row.get("product_weight_in_g")),
        "co2_kg_per_kg": _parse_float(row.get("co2_kg_per_kg")),
        "calories_per_100g": _parse_float(row.get("calories_per_100g")),
        "protein_g_per_100g": _parse_float(row.get("protein_g_per_100g")),
        "fat_g_per_100g": _parse_float(row.get("fat_g_per_100g")),
        "carbs_g_per_100g": _parse_float(row.get("carbs_g_per_100g")),
        "fiber_g_per_100g": _parse_float(row.get("fiber_g_per_100g")),
        "salt_g_per_100g": _parse_float(row.get("salt_g_per_100g")),
        "is_liquid": _parse_bool(row.get("is_liquid")),
        "is_gluten_free": _parse_bool(row.get("is_gluten_free")),
        "is_sugar_free": _parse_bool(row.get("is_sugar_free")),
        "is_oekomærket_eu": _parse_bool(row.get("is_oekomærket_eu")),
        "is_oekomærket_dk": _parse_bool(row.get("is_oekomærket_dk")),
        "is_noeglehulsmaerket": _parse_bool(row.get("is_noeglehulsmaerket")),
        "is_fuldkornsmaerket": _parse_bool(row.get("is_fuldkornsmaerket")),
        "is_frozen": _parse_bool(row.get("is_frozen")),
        "is_msc_maerket": _parse_bool(row.get("is_msc_maerket")),
        "is_fairtrade": _parse_bool(row.get("is_fairtrade")),
        "is_rainforest_alliance": _parse_bool(row.get("is_rainforest_alliance")),
        "is_danish": _parse_bool(row.get("is_danish")),
        "price_dkk": _parse_float(row.get("price_dkk")),
        "category_l1": _parse_text(row.get("category_l1")),
        "category_l2": _parse_text(row.get("category_l2")),
        "category_l3": _parse_text(row.get("category_l3")),
        "category_l4": _parse_text(row.get("category_l4")),
        "canonical_product_type": _parse_text(row.get("canonical_product_type")),
        "prep_form": _parse_text(row.get("prep_form")),
        "primary_use": _parse_text(row.get("primary_use")),
        "flavor_variant": _parse_text(row.get("flavor_variant")),
        "sub_group_core": _parse_text(row.get("sub_group_core")),
        "sub_group_extended": _parse_text(row.get("sub_group_extended")),
        "size_band": _parse_text(row.get("size_band")),
        "sub_tier_default": _parse_text(row.get("sub_tier_default")),
        "label_confidence": _parse_text(row.get("label_confidence")),
        "needs_review": _parse_bool(row.get("needs_review")),
    }


def _iter_product_records() -> tuple[Iterator[dict[str, Any]], int, str]:
    if PRODUCTS_FILE.suffix.lower() == ".jsonl":
        total = _count_nonempty_lines(PRODUCTS_FILE)
        return (_build_product_record(row) for row in _iter_jsonl_rows(PRODUCTS_FILE)), total, PRODUCTS_LABEL
    total = _count_csv_rows(PRODUCTS_FILE)
    return (_build_product_record(row) for row in _iter_csv_rows(PRODUCTS_FILE)), total, PRODUCTS_LABEL


def _build_substitution_group_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sub_group_id": _parse_text(row.get("sub_group_id")),
        "group_level": _parse_text(row.get("group_level")),
        "group_name": _parse_text(row.get("group_name")),
        "member_count": _parse_int(row.get("member_count")),
    }


def _build_product_group_map_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "product_id": _parse_text(row.get("product_id")),
        "sub_group_id": _parse_text(row.get("sub_group_id")),
        "group_level": _parse_text(row.get("group_level")),
    }


def _iter_category_taxonomy_records() -> tuple[Iterator[dict[str, Any]], int]:
    total = _count_csv_rows(CATEGORY_TAXONOMY_FILE)

    def generator() -> Iterator[dict[str, Any]]:
        for idx, row in enumerate(_iter_csv_rows(CATEGORY_TAXONOMY_FILE), start=1):
            yield {
                "category_id": idx,
                "category_l1": _parse_text(row.get("category_l1")),
                "category_l2": _parse_text(row.get("category_l2")),
                "category_l3": _parse_text(row.get("category_l3")),
                "category_l4": _parse_text(row.get("category_l4")),
                "product_count": _parse_int(row.get("product_count")) or 0,
            }

    return generator(), total


async def _table_exists(session: AsyncSession, table_name: str) -> bool:
    conn = await session.connection()

    def _inner(sync_conn: Any) -> bool:
        return inspect(sync_conn).has_table(table_name)

    return await conn.run_sync(_inner)


async def _get_column_names(session: AsyncSession, table_name: str) -> set[str]:
    conn = await session.connection()

    def _inner(sync_conn: Any) -> set[str]:
        return {col["name"] for col in inspect(sync_conn).get_columns(table_name)}

    return await conn.run_sync(_inner)


async def _get_pk_columns(session: AsyncSession, table_name: str) -> list[str]:
    conn = await session.connection()

    def _inner(sync_conn: Any) -> list[str]:
        pk = inspect(sync_conn).get_pk_constraint(table_name) or {}
        return list(pk.get("constrained_columns") or [])

    return await conn.run_sync(_inner)


async def _get_check_constraints(session: AsyncSession, table_name: str) -> list[dict[str, Any]]:
    conn = await session.connection()

    def _inner(sync_conn: Any) -> list[dict[str, Any]]:
        return list(inspect(sync_conn).get_check_constraints(table_name) or [])

    return await conn.run_sync(_inner)


async def _table_count(session: AsyncSession, table_name: str) -> int:
    result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
    return int(result.scalar_one())


async def _drop_table_if_exists(session: AsyncSession, table_name: str) -> None:
    await session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))


async def _drop_incompatible_legacy_tables(session: AsyncSession) -> None:
    """Drop old tables that collide with the new catalogue schema.

    The legacy app schema used names like `substitution_groups` but with different
    columns (for example `id/code/name/...` instead of `sub_group_id/...`).
    We also drop partially-created new tables if their primary keys do not match
    the current schema, because CREATE TABLE IF NOT EXISTS will not repair them.
    """
    compatibility_rules: dict[str, set[str]] = {
        "products_enriched": {"id", "sub_group_core", "sub_group_extended", "canonical_product_type"},
        "category_taxonomy": {"category_id", "category_l1", "category_l2", "category_l3", "category_l4"},
        "substitution_groups": {"sub_group_id", "group_level", "group_name"},
        "product_substitution_group_map": {"product_id", "sub_group_id", "group_level"},
        "substitution_candidates": {"source_product_id", "candidate_product_id", "candidate_tier"},
    }
    expected_pk_columns: dict[str, list[str]] = {
        "products_enriched": ["id"],
        "category_taxonomy": ["category_id"],
        "substitution_groups": ["sub_group_id", "group_level"],
        "product_substitution_group_map": ["product_id", "sub_group_id", "group_level"],
        "substitution_candidates": ["source_product_id", "candidate_product_id"],
    }

    for table_name, required_columns in compatibility_rules.items():
        if not await _table_exists(session, table_name):
            continue
        existing_columns = await _get_column_names(session, table_name)
        if not required_columns.issubset(existing_columns):
            logger.warning(
                "Catalogue seed: dropping incompatible legacy table %s. Expected columns %s but found %s.",
                table_name,
                sorted(required_columns),
                sorted(existing_columns),
            )
            await _drop_table_if_exists(session, table_name)
            continue

        actual_pk_columns = await _get_pk_columns(session, table_name)
        expected = expected_pk_columns.get(table_name, [])
        if expected and actual_pk_columns != expected:
            logger.warning(
                "Catalogue seed: dropping %s due to primary key mismatch. Expected %s but found %s.",
                table_name,
                expected,
                actual_pk_columns,
            )
            await _drop_table_if_exists(session, table_name)
            continue

        if table_name == "products_enriched":
            checks = await _get_check_constraints(session, table_name)
            sub_tier_ok = False
            confidence_ok = False
            for chk in checks:
                sqltext = str(chk.get("sqltext") or "").lower()
                name = str(chk.get("name") or "")
                if "sub_tier_default" in sqltext or "sub_tier_default" in name:
                    if all(token in sqltext for token in ("core", "extended")):
                        sub_tier_ok = True
                if "label_confidence" in sqltext or "label_confidence" in name:
                    if all(token in sqltext for token in ("high", "medium", "low")):
                        confidence_ok = True
            if not sub_tier_ok or not confidence_ok:
                logger.warning(
                    "Catalogue seed: dropping %s due to incompatible check constraints.",
                    table_name,
                )
                await _drop_table_if_exists(session, table_name)


async def _ensure_schema(session: AsyncSession) -> None:
    logger.info("Catalogue seed: ensuring schema from %s…", SCHEMA_FILE.name)
    await _drop_incompatible_legacy_tables(session)
    for statement in _load_schema_sql(SCHEMA_FILE):
        await session.execute(text(statement))

    category_columns = await _get_column_names(session, "category_taxonomy")
    if "product_count" not in category_columns:
        logger.info("Catalogue seed: adding missing category_taxonomy.product_count column…")
        await session.execute(
            text(
                "ALTER TABLE category_taxonomy "
                "ADD COLUMN product_count INTEGER NOT NULL DEFAULT 0"
            )
        )

    for ddl in EXTRA_INDEX_DDLS:
        await session.execute(text(ddl))

    required_tables = {
        "products_enriched",
        "category_taxonomy",
        "substitution_groups",
        "product_substitution_group_map",
        "substitution_candidates",
    }
    missing_tables = [table for table in required_tables if not await _table_exists(session, table)]
    if missing_tables:
        raise RuntimeError(f"Missing required tables after schema bootstrap: {', '.join(missing_tables)}")


async def _clear_catalogue_tables(session: AsyncSession) -> None:
    logger.info("Catalogue seed: clearing existing rows from new catalogue tables…")
    for table_name in (
        "substitution_candidates",
        "product_substitution_group_map",
        "substitution_groups",
        "category_taxonomy",
        "products_enriched",
    ):
        if await _table_exists(session, table_name):
            await session.execute(text(f"DELETE FROM {table_name}"))


def _validate_non_null_fields(record: dict[str, Any], *, required_keys: Iterable[str], context: str) -> None:
    missing = [key for key in required_keys if record.get(key) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context}: missing required values for {joined}")


async def _insert_batches(
    session: AsyncSession,
    *,
    table_name: str,
    columns: list[str],
    row_iter: Iterable[dict[str, Any]],
    total_rows: int,
) -> int:
    placeholders = ", ".join(f":{col}" for col in columns)
    insert_sql = text(
        f"INSERT INTO {table_name} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )

    inserted = 0
    batch: list[dict[str, Any]] = []

    for row in row_iter:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            await session.execute(insert_sql, batch)
            inserted += len(batch)
            batch = []
            if inserted % SEED_PROGRESS_INSERT_INTERVAL == 0:
                pct = 100.0 * inserted / total_rows if total_rows else 0.0
                logger.info(
                    "Catalogue seed: inserted %d / %d rows into %s (%.1f%%).",
                    inserted,
                    total_rows,
                    table_name,
                    pct,
                )

    if batch:
        await session.execute(insert_sql, batch)
        inserted += len(batch)

    return inserted


async def _seed_products(session: AsyncSession) -> int:
    product_iter, total_rows, label = _iter_product_records()
    logger.info("Catalogue seed: loading products from %s (%d rows)…", label, total_rows)

    scanned = 0

    def generator() -> Iterator[dict[str, Any]]:
        nonlocal scanned
        for record in product_iter:
            scanned += 1
            if scanned % SEED_PROGRESS_SCAN_INTERVAL == 0:
                pct = 100.0 * scanned / total_rows if total_rows else 0.0
                logger.info(
                    "Catalogue seed: scanned %d / %d product rows (%.1f%%).",
                    scanned,
                    total_rows,
                    pct,
                )

            _validate_non_null_fields(
                record,
                required_keys=(
                    "id",
                    "name",
                    "category_l1",
                    "category_l2",
                    "category_l3",
                    "category_l4",
                    "canonical_product_type",
                    "prep_form",
                    "primary_use",
                    "flavor_variant",
                    "sub_group_core",
                    "sub_group_extended",
                    "size_band",
                    "sub_tier_default",
                    "label_confidence",
                    "needs_review",
                ),
                context=f"products_enriched row {record.get('id')!r}",
            )
            yield record

    inserted = await _insert_batches(
        session,
        table_name="products_enriched",
        columns=PRODUCT_COLUMNS,
        row_iter=generator(),
        total_rows=total_rows,
    )
    logger.info("Catalogue seed: inserted %d products.", inserted)
    return inserted


async def _seed_category_taxonomy(session: AsyncSession) -> int:
    row_iter, total_rows = _iter_category_taxonomy_records()
    logger.info("Catalogue seed: loading category taxonomy (%d rows)…", total_rows)

    def generator() -> Iterator[dict[str, Any]]:
        for record in row_iter:
            _validate_non_null_fields(
                record,
                required_keys=("category_id", "category_l1", "category_l2", "category_l3", "category_l4"),
                context=f"category_taxonomy row {record.get('category_id')!r}",
            )
            yield record

    inserted = await _insert_batches(
        session,
        table_name="category_taxonomy",
        columns=CATEGORY_TAXONOMY_COLUMNS,
        row_iter=generator(),
        total_rows=total_rows,
    )
    logger.info("Catalogue seed: inserted %d category taxonomy rows.", inserted)
    return inserted


async def _seed_substitution_groups(session: AsyncSession) -> int:
    total_rows = _count_csv_rows(SUBSTITUTION_GROUPS_FILE)
    logger.info("Catalogue seed: loading substitution groups from %s (%d rows)…", SUBSTITUTION_GROUPS_FILE.name, total_rows)

    def generator() -> Iterator[dict[str, Any]]:
        for row in _iter_csv_rows(SUBSTITUTION_GROUPS_FILE):
            record = _build_substitution_group_record(row)
            _validate_non_null_fields(
                record,
                required_keys=("sub_group_id", "group_level", "group_name"),
                context=f"substitution_groups row {record.get('sub_group_id')!r}",
            )
            yield record

    inserted = await _insert_batches(
        session,
        table_name="substitution_groups",
        columns=SUBSTITUTION_GROUP_COLUMNS,
        row_iter=generator(),
        total_rows=total_rows,
    )
    logger.info("Catalogue seed: inserted %d substitution groups.", inserted)
    return inserted


async def _seed_product_group_map(session: AsyncSession) -> int:
    total_rows = _count_csv_rows(PRODUCT_GROUP_MAP_FILE)
    logger.info("Catalogue seed: loading product/group map from %s (%d rows)…", PRODUCT_GROUP_MAP_FILE.name, total_rows)

    def generator() -> Iterator[dict[str, Any]]:
        for row in _iter_csv_rows(PRODUCT_GROUP_MAP_FILE):
            record = _build_product_group_map_record(row)
            _validate_non_null_fields(
                record,
                required_keys=("product_id", "sub_group_id", "group_level"),
                context=f"product_substitution_group_map row {record.get('product_id')!r}",
            )
            yield record

    inserted = await _insert_batches(
        session,
        table_name="product_substitution_group_map",
        columns=PRODUCT_GROUP_MAP_COLUMNS,
        row_iter=generator(),
        total_rows=total_rows,
    )
    logger.info("Catalogue seed: inserted %d product/substitution-group rows.", inserted)
    return inserted


async def seed_catalogue() -> None:
    async with AsyncSessionLocal() as session:
        await _ensure_schema(session)

        existing_products = await _table_count(session, "products_enriched")
        if existing_products > 0 and not RESEED:
            logger.info(
                "Catalogue already seeded with %d products in products_enriched; skipping. "
                "Set CATALOGUE_RESEED=1 to clear and seed again.",
                existing_products,
            )
            return

        await _clear_catalogue_tables(session)

        inserted_categories = await _seed_category_taxonomy(session)
        inserted_products = await _seed_products(session)
        inserted_groups = await _seed_substitution_groups(session)
        inserted_map_rows = await _seed_product_group_map(session)

        await session.commit()

        logger.info(
            "Catalogue seed done: %d products, %d category taxonomy rows, %d substitution groups, "
            "%d product/substitution-group links.",
            inserted_products,
            inserted_categories,
            inserted_groups,
            inserted_map_rows,
        )


if __name__ == "__main__":
    asyncio.run(seed_catalogue())
