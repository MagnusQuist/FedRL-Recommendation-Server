from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class CentralizedModelVersion(Base):
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

    __table_args__ = (
        UniqueConstraint("version"),
    )

    def __repr__(self) -> str:
        return f"<CentralizedModelVersion version={self.version}>"
