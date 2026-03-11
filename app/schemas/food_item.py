import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String
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