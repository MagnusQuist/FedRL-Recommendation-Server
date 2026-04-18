"""CLI for DB seeding (same entry points as server startup).

    python -m app.db.seeding                   # bootstrap if empty (default)
    python -m app.db.seeding --force           # create tables + full re-seed
    python -m app.db.seeding --catalogue-only  # catalogue only
    python -m app.db.seeding --backbones-only  # federated + centralized backbones only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv

load_dotenv()

from app.db.seeding.runner import bootstrap_if_empty, ensure_models, seed_all  # noqa: E402
from app.db.seeding.seed_backbone import (  # noqa: E402
    seed_centralized_backbone,
    seed_federated_backbone,
)
from app.db.seeding.seed_catalogue import seed_catalogue  # noqa: E402


async def _run(args: argparse.Namespace) -> None:
    if args.catalogue_only:
        await ensure_models()
        await seed_catalogue()
        return

    if args.backbones_only:
        await ensure_models()
        await seed_federated_backbone()
        await seed_centralized_backbone()
        return

    if args.force:
        await ensure_models()
        await seed_all()
        return

    await bootstrap_if_empty()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tables and seed the database.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create tables and run all seeders (catalogue is fully replaced).",
    )
    parser.add_argument(
        "--catalogue-only",
        action="store_true",
        help="Seed food catalogue only.",
    )
    parser.add_argument(
        "--backbones-only",
        action="store_true",
        help="Seed federated and centralized backbone rows only.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
