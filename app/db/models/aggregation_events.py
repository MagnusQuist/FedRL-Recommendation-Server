from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AggregationEvent(Base):
    __tablename__ = "aggregation_events"

    aggregation_event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="Unique aggregation event identifier.")

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the aggregation event.",
    )

    aggregation_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, comment="Duration of the aggregation in milliseconds.")

    participating_clients_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, comment="List of client IDs participating in the aggregation.")

    num_clients_in_round: Mapped[int] = mapped_column(Integer, nullable=False, comment="Number of clients participating in the aggregation.")
    total_interactions: Mapped[int] = mapped_column(Integer, nullable=False, comment="Total interactions across participating clients in the aggregation round.")

    model_version_before: Mapped[str] = mapped_column(Text, nullable=True, comment="Model version before the aggregation.")

    model_version_after: Mapped[str] = mapped_column(Text, nullable=False, comment="Model version after the aggregation.")

    model_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True, comment="Resulting model size in bytes.")

    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the aggregation event logging.",
    )

    __table_args__ = (
        Index("idx_aggregation_events_timestamp", "timestamp"),
    )