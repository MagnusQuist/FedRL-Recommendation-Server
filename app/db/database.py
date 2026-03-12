import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Load .env file (if present) so env vars are available when this module is imported.
load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_database_url() -> str:
    """Build the SQLAlchemy database URL from environment variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("POSTGRES_USER", "fedrl")
    password = os.getenv("POSTGRES_PASSWORD", "fedrl")
    db = os.getenv("POSTGRES_DB", "fedrl")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


DATABASE_URL = _get_database_url()

engine = create_async_engine(
    DATABASE_URL,
    echo=_env_bool("SQL_ECHO", False),
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise