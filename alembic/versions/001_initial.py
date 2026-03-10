"""Initial schema — food_items and global_backbone_versions

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "food_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("co2e_score", sa.Float, nullable=False,
                  comment="Normalised sustainability score 0–1. Higher = more sustainable."),
        sa.Column("co2e_kg_per_kg", sa.Float, nullable=False,
                  comment="Raw CO2e in kg per kg of product (Poore & Nemecek 2018)."),
        sa.Column("price", sa.Float, nullable=False, comment="Price in DKK."),
        sa.Column("unit", sa.String(50), nullable=False),
        sa.Column("alternative_ids", ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_food_items_category", "food_items", ["category"])

    op.create_table(
        "global_backbone_versions",
        sa.Column("version", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("weights_blob", sa.Text, nullable=False,
                  comment="gzip-compressed, base64-encoded JSON of backbone weight arrays."),
        sa.Column("algorithm", sa.String(10), nullable=False,
                  comment="'ts' or 'dqn'."),
        sa.Column("client_count", sa.Integer, nullable=False,
                  comment="Number of clients that contributed to this round."),
        sa.Column("total_interactions", sa.Integer, nullable=False,
                  comment="Sum of n_k across contributing clients."),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("global_backbone_versions")
    op.drop_index("ix_food_items_category", table_name="food_items")
    op.drop_table("food_items")