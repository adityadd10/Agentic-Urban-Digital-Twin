"""Process-wide settings (Phase 0, dev doc §12.1).

All configuration comes from environment variables / a `.env` file (dev doc
§12.2: "secrets only via `.env` (gitignored) + `pydantic-settings`"), never
hardcoded elsewhere. Every setting is prefixed `UDT_` to avoid collisions
with unrelated env vars on the host.

Usage:
    from udt.common.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/udt/common/config.py -> common -> udt -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Single source of process configuration.

    Fields grow as later modules need them (e.g. M9b adds LLM API keys,
    M10b adds API host/port) — add new fields here, never read `os.environ`
    directly elsewhere in `src/`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="UDT_",
        extra="ignore",
    )

    environment: str = "dev"
    log_level: str = "INFO"

    # Postgres/PostGIS (docker-compose service `postgres`; tables per §11.1)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "udt"
    postgres_user: str = "udt"
    postgres_password: str = "udt"

    # MLflow (dev doc §12.2: every training/eval run is an MLflow run)
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: f"file://{REPO_ROOT / 'runs' / 'mlflow'}"
    )

    # Filesystem layout (dev doc §12.1)
    data_dir: Path = REPO_ROOT / "data"
    runs_dir: Path = REPO_ROOT / "runs"

    # Determinism (dev doc §0.4 — every seeded run starts from this)
    default_seed: int = 0

    # LLM planning layer (dev doc §7.4, module M9b). `llm_api_key=None`
    # (the `.env.example` default, "leave unset until then") means
    # `llm/anthropic_client.py`'s `AnthropicLLMClient` can't be
    # constructed — every M9a graph test uses a scripted `LLMClient`
    # instead and never needs this at all.
    llm_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"  # dev doc §7.4: "Prototyping: Claude Sonnet"
    llm_max_tokens_per_decision: int = 20_000  # dev doc §7.4's exact budget-guard cap

    @property
    def postgres_dsn(self) -> str:
        """SQLAlchemy/psycopg-style DSN for the docker-compose Postgres service."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached Settings instance.

    Cached (not a module-level singleton) so tests can bypass it with
    `get_settings.cache_clear()` after monkeypatching environment variables.
    """
    return Settings()
