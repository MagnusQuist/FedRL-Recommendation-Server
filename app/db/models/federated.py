from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class FederatedBackboneVersion(Base):
    __tablename__ = "federated_model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of backbone weight arrays."
    )
    client_count: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Number of clients whose uploads contributed to this round."
    )
    total_interactions: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Sum of n_k across all contributing clients for this round."
    )
    training_time_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Wall-clock seconds spent on FedAvg aggregation for this round. NULL for the seeded v1 row."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("version"),
    )

    def __repr__(self) -> str:
        return (
            f"<FederatedBackboneVersion version={self.version} "
            f"clients={self.client_count}>"
        )
