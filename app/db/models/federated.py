from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class FederatedModel(Base):
    __tablename__ = "federated_model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_blob: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="gzip-compressed, base64-encoded JSON of backbone weight arrays."
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
        return f"<FederatedModel version={self.version}>"
