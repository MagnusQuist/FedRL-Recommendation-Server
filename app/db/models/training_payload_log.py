from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class TrainingPayloadLog(Base):
    __tablename__ = "training_payload_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    client_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Client identifier that submitted this upload.",
    )
    payload_blob: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw payload blob received from the client (base64+gzip encoded).",
    )
    payload_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="UTF-8 payload size in bytes.",
    )
    payload_size_kb: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Payload size in kilobytes (bytes / 1024).",
    )
    payload_size_mb: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Payload size in megabytes (bytes / 1024 / 1024).",
    )
    full_request_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Full HTTP request body size in bytes.",
    )

    federated_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("federated_model_versions.id", ondelete="CASCADE"),
        nullable=True,
        comment="FK to a completed federated round.",
    )
    centralized_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("centralized_model_versions.id", ondelete="CASCADE"),
        nullable=True,
        comment="FK to a completed centralized round.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint(
            "("
            "(federated_model_version_id IS NOT NULL AND centralized_model_version_id IS NULL)"
            " OR "
            "(federated_model_version_id IS NULL AND centralized_model_version_id IS NOT NULL)"
            ")",
            name="ck_training_payload_logs_exactly_one_round_fk",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingPayloadLog id={self.id} client_id={self.client_id} "
            f"fed_round_id={self.federated_model_version_id} "
            f"cen_round_id={self.centralized_model_version_id}>"
        )
