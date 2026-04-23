from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class CentralizedTrainingEvent(Base):
    __tablename__ = "centralized_training_events"

    centralized_training_event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="Unique centralized training event identifier.")

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the centralized training event.",
    )

    training_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, comment="Duration of the centralized training in milliseconds.")

    num_interactions: Mapped[int] = mapped_column(Integer, nullable=False, comment="Number of interactions in the centralized training.")
    num_clients_contributing: Mapped[int] = mapped_column(Integer, nullable=False, comment="Number of clients contributing to the centralized training.")
    contributing_client_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, comment="List of client IDs contributing to the centralized training.")

    cpu_usage_percentage: Mapped[float] = mapped_column(Float, nullable=True, comment="CPU usage percentage during the centralized training.")
    memory_usage_mb: Mapped[float] = mapped_column(Float, nullable=True, comment="Memory usage in megabytes during the centralized training.")

    loss_before: Mapped[float] = mapped_column(Float, nullable=True, comment="Loss before the centralized training.")
    loss_after: Mapped[float] = mapped_column(Float, nullable=True, comment="Loss after the centralized training.")
    loss_delta: Mapped[float] = mapped_column(Float, nullable=True, comment="Loss delta between the loss before and after the centralized training.")

    model_version_before: Mapped[str] = mapped_column(Text, nullable=True, comment="Model version before the centralized training.")
    model_version_after: Mapped[str] = mapped_column(Text, nullable=False, comment="Model version after the centralized training.")

    model_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True, comment="Resulting model size in bytes.")

    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the centralized training event logging.",
    )

    __table_args__ = (
        Index("idx_centralized_training_events_timestamp", "timestamp"),
    )