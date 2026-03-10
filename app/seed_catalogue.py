import asyncio
import json
from pathlib import Path
from typing import Dict

from app.database import AsyncSessionLocal
from app.schemas.food_item import FoodItem


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "seed_catalogue.json"


def co2e_to_tonnes(total_co2e: float) -> float:
    """
    Convert dataset's total_co2e value to tonnes.

    Adjust this if your dataset uses different units. Right now we assume the
    value is already in tonnes and just cast to float.
    """
    return float(total_co2e)


async def seed_catalogue() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}")

    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    items = raw.get("items", [])
    if not items:
        print("No items found in dataset; nothing to seed.")
        return

    async with AsyncSessionLocal() as session:
        # Map from dataset item_id (e.g. "Ra00001-DK") to DB row
        id_map: Dict[str, FoodItem] = {}

        # First pass: insert all items with empty alternative_ids
        for row in items:
            item_id = row["id"]
            item = FoodItem(
                item_id=item_id,
                name=row["name"],
                category=row["category"],
                co2e_emission_tonnes=co2e_to_tonnes(row["total_co2e"]),
                price=row["price"],
                alternative_ids=[],
            )
            session.add(item)
            await session.flush()  # assign UUID primary key
            id_map[item_id] = item

        # Second pass: wire up alternative_ids using the mapping
        for row in items:
            item_id = row["id"]
            greener_ids = row.get("greener_alternative_ids", [])

            db_item = id_map[item_id]
            alt_uuids = []
            for alt_id in greener_ids:
                alt_item = id_map.get(alt_id)
                if alt_item is not None:
                    alt_uuids.append(alt_item.id)

            db_item.alternative_ids = alt_uuids

        await session.commit()
        print(f"Seeded {len(items)} food items.")


if __name__ == "__main__":
    asyncio.run(seed_catalogue())

