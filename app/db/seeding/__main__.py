"""Seeding CLI.

Usage:
    python -m app.db.seeding                   # bootstrap if the DB is empty
    python -m app.db.seeding --force           # create tables + re-run all seeders
    python -m app.db.seeding --catalogue-only  # re-seed only the catalogue
    python -m app.db.seeding --backbones-only  # re-seed only the backbones

The no-argument invocation mirrors what the server does on startup and is the
recommended way to provision a fresh database.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv

load_dotenv()

from app.db.seeding.runner import bootstrap_if_empty, ensure_models, seed_all
from app.db.seeding.seed_backbone import (
    seed_centralized_backbone,
    seed_federated_backbone,
)
from app.db.seeding.seed_catalogue import seed_catalogue


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
    parser = argparse.ArgumentParser(
        description="Create tables and seed the FedRL database.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create tables and re-run every seeder (full replace for the catalogue).",
    )
    parser.add_argument(
        "--catalogue-only",
        action="store_true",
        help="Re-seed only the food catalogue.",
    )
    parser.add_argument(
        "--backbones-only",
        action="store_true",
        help="Re-seed only the federated + centralized backbones.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
