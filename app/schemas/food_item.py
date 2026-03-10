import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FoodItem(Base):
    __tablename__ = "food_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    item_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, comment="Item ID, e.g. 'Ra00001-DK'.")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Sustainability
    co2e_emission_tonnes: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="CO2e emission in tonnes."
    )

    # Commerce
    price: Mapped[float] = mapped_column(Float, nullable=False, comment="Price in DKK.")

    # Pre-computed greener alternatives in same category
    alternative_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<FoodItem id={self.id} name={self.name!r} category={self.category!r}>"


class GlobalBackboneVersion(Base):
    __tablename__ = "global_backbone_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weights_blob: Mapped[bytes] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of backbone weight arrays."
    )
    algorithm: Mapped[str] = mapped_column(
        String(10), nullable=False,
        comment="Algorithm this backbone belongs to: 'ts' or 'dqn'."
    )
    client_count: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Number of clients whose uploads contributed to this round."
    )
    total_interactions: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Sum of n_k across all contributing clients for this round."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<GlobalBackboneVersion version={self.version} "
            f"algorithm={self.algorithm!r} clients={self.client_count}>"
        )