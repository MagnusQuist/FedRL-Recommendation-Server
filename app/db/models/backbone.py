from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class GlobalBackboneVersion(Base):
    __tablename__ = "global_backbone_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_blob: Mapped[str] = mapped_column(
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

    __table_args__ = (
        UniqueConstraint("algorithm", "version"),
    )

    def __repr__(self) -> str:
        return (
            f"<GlobalBackboneVersion version={self.version} "
            f"algorithm={self.algorithm!r} clients={self.client_count}>"
        )