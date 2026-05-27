from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class CatalogueVersion(Base):
    __tablename__ = "catalogue_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class FoodItem(Base):
    __tablename__ = "food_items"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    product_weight_in_g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    co2e_kg_pr_item_kg: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    estimated_co2e_kg_pr_item_weight_in_g: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    calories_per_100g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protein_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    fat_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    carbs_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    fiber_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    salt_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    is_liquid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_gluten_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_sugar_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_oekomærket_eu: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_oekomærket_dk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_noeglehulsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_fuldkornsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_msc_maerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_fairtrade: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_rainforest_alliance: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_danish: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    price_dkk: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)

    food_item_categories: Mapped[list[FoodItemCategory]] = relationship(
        back_populates="food_item",
        cascade="all, delete-orphan",
    )
    substitution_group_items: Mapped[list[SubstitutionGroupItem]] = relationship(
        back_populates="food_item",
        cascade="all, delete-orphan",
    )
    categories: Mapped[list[Category]] = relationship(
        secondary="food_item_categories",
        back_populates="food_items",
        viewonly=True,
    )
    substitution_groups: Mapped[list[SubstitutionGroup]] = relationship(
        secondary="substitution_group_items",
        back_populates="food_items",
        viewonly=True,
    )

    __table_args__ = (
        Index("ix_food_items_name", "name"),
        Index("ix_food_items_brand", "brand"),
    )


class Category(Base):
    __tablename__ = "categories"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(140), nullable=False, unique=True)

    food_item_categories: Mapped[list[FoodItemCategory]] = relationship(
        back_populates="category",
        cascade="all, delete-orphan",
    )
    food_items: Mapped[list[FoodItem]] = relationship(
        secondary="food_item_categories",
        back_populates="categories",
        viewonly=True,
    )


class FoodItemCategory(Base):
    __tablename__ = "food_item_categories"

    product_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("food_items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("categories.category_id", ondelete="CASCADE"),
        primary_key=True,
    )

    food_item: Mapped[FoodItem] = relationship(back_populates="food_item_categories")
    category: Mapped[Category] = relationship(back_populates="food_item_categories")

    __table_args__ = (
        Index("ix_food_item_categories_product_id", "product_id"),
        Index("ix_food_item_categories_category_id", "category_id"),
    )


class SubstitutionGroup(Base):
    __tablename__ = "substitution_groups"

    substitution_group_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)

    substitution_group_items: Mapped[list[SubstitutionGroupItem]] = relationship(
        back_populates="substitution_group",
        cascade="all, delete-orphan",
    )
    food_items: Mapped[list[FoodItem]] = relationship(
        secondary="substitution_group_items",
        back_populates="substitution_groups",
        viewonly=True,
    )


class SubstitutionGroupItem(Base):
    __tablename__ = "substitution_group_items"

    substitution_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("substitution_groups.substitution_group_id", ondelete="CASCADE"),
        primary_key=True,
    )
    product_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("food_items.id", ondelete="CASCADE"),
        primary_key=True,
    )

    substitution_group: Mapped[SubstitutionGroup] = relationship(back_populates="substitution_group_items")
    food_item: Mapped[FoodItem] = relationship(back_populates="substitution_group_items")

    __table_args__ = (
        Index("ix_substitution_group_items_group_id", "substitution_group_id"),
        Index("ix_substitution_group_items_product_id", "product_id"),
    )


class FederatedModel(Base):
    __tablename__ = "federated_model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_blob: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="gzip-compressed, base64-encoded JSON of backbone weight arrays.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (UniqueConstraint("version"),)

    def __repr__(self) -> str:
        return f"<FederatedModel version={self.version}>"


class CentralizedModel(Base):
    __tablename__ = "centralized_model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    backbone_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of backbone weight arrays.",
    )
    reward_predictor_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of reward predictor weight arrays.",
    )
    item_head_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of TSItemHead state.",
    )
    price_head_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of TSPriceHead state.",
    )
    nudge_head_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of TSNudgeHead state.",
    )
    tuple_pool_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of the interaction tuple pool.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (UniqueConstraint("version"),)

    def __repr__(self) -> str:
        return f"<CentralizedModel version={self.version}>"


class AggregationEvent(Base):
    __tablename__ = "aggregation_events"

    aggregation_event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    aggregation_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    participating_clients_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    num_clients_in_round: Mapped[int] = mapped_column(Integer, nullable=False)
    total_interactions: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version_before: Mapped[str] = mapped_column(Text, nullable=True)
    model_version_after: Mapped[str] = mapped_column(Text, nullable=False)
    model_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    aggregation_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_global_model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    backbone_update_norm_l2: Mapped[float | None] = mapped_column(Float, nullable=True)
    backbone_relative_update_norm_l2: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_client_update_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_client_update_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    aggregation_threshold_k: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (Index("idx_aggregation_events_timestamp", "timestamp"),)


class CentralizedTrainingEvent(Base):
    __tablename__ = "centralized_training_events"

    centralized_training_event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    training_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    num_interactions: Mapped[int] = mapped_column(Integer, nullable=False)
    num_clients_contributing: Mapped[int] = mapped_column(Integer, nullable=False)
    contributing_client_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    cpu_usage_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    memory_usage_mb: Mapped[float] = mapped_column(Float, nullable=True)
    loss_before: Mapped[float] = mapped_column(Float, nullable=True)
    loss_after: Mapped[float] = mapped_column(Float, nullable=True)
    loss_delta: Mapped[float] = mapped_column(Float, nullable=True)
    model_version_before: Mapped[str] = mapped_column(Text, nullable=True)
    model_version_after: Mapped[str] = mapped_column(Text, nullable=False)
    model_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    training_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_training_interactions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bce_loss_improvement: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_update_norm_l2: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_relative_update_norm_l2: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("idx_centralized_training_events_timestamp", "timestamp"),)
