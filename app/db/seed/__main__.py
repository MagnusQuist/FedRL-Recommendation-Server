"""CLI for DB seeding.

    python -m app.db.seed                   # bootstrap if empty (default)
    python -m app.db.seed --force           # create tables + full re-seed
    python -m app.db.seed --catalogue-only  # catalogue only
    python -m app.db.seed --backbones-only  # federated + centralized backbones only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv

load_dotenv()

from app.db.seed.run import bootstrap_if_empty, ensure_models, seed_all  # noqa: E402
from app.db.seed.backbone import seed_centralized_backbone, seed_federated_backbone  # noqa: E402
from app.db.seed.catalogue import seed_catalogue  # noqa: E402


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
    parser.add_argument("--force", action="store_true", help="Create tables and run all seeders.")
    parser.add_argument("--catalogue-only", action="store_true", help="Seed food catalogue only.")
    parser.add_argument("--backbones-only", action="store_true", help="Seed backbone rows only.")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
