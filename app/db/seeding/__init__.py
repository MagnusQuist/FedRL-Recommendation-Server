"""Database seeding package.

Public API for creating tables and loading seed data. The server calls
``bootstrap_if_empty()`` on startup; the same helpers are exposed through the
``python -m app.db.seeding`` CLI.
"""

from app.db.seeding.runner import (
    bootstrap_if_empty,
    ensure_models,
    seed_all,
)
from app.db.seeding.seed_backbone import (
    HIDDEN_DIM,
    INITIAL_VERSION,
    INPUT_DIM,
    OUTPUT_DIM,
    seed_centralized_backbone,
    seed_federated_backbone,
    serialise_weights,
)
from app.db.seeding.seed_catalogue import seed_catalogue
from app.db.seeding.seed_status import has_tables, is_database_seeded

__all__ = [
    "HIDDEN_DIM",
    "INITIAL_VERSION",
    "INPUT_DIM",
    "OUTPUT_DIM",
    "bootstrap_if_empty",
    "ensure_models",
    "has_tables",
    "is_database_seeded",
    "seed_all",
    "seed_catalogue",
    "seed_centralized_backbone",
    "seed_federated_backbone",
    "serialise_weights",
]
