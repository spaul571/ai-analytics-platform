"""Central configuration loaded from environment / .env file / Streamlit secrets.

Every tunable that differs between machines lives here so no module hardcodes
a host, port, or file path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _setting(key: str, default: str) -> str:
    """Resolve one setting from the environment, then Streamlit secrets.

    Locally the value comes from .env via load_dotenv. On Streamlit Cloud there
    is no .env: secrets are supplied through st.secrets, and while Streamlit
    also mirrors them into os.environ, that mirroring is lazy and is not
    guaranteed to have run before this module is imported. Reading os.environ
    alone therefore silently falls back to the localhost default and the
    deployed app tries to reach an LM Studio that only exists on a laptop.

    st.secrets is consulted second rather than first so that an explicit
    environment variable still wins, which is what the acceptance scripts and
    the proxy tests rely on.
    """
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st

        secret = st.secrets.get(key)
    except Exception:
        # Not running under Streamlit, or no secrets file exists. Both are
        # normal: scripts/check_task_*.py import this module directly.
        return default

    return str(secret) if secret else default


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for the LM Studio OpenAI-compatible endpoint."""

    # default_factory, not a plain default: a bare default is evaluated once at
    # class-definition time, which on Cloud is too early for st.secrets to exist.
    base_url: str = field(
        default_factory=lambda: _setting("LLM_BASE_URL", "http://localhost:1234/v1")
    )
    model: str = field(
        default_factory=lambda: _setting("LLM_MODEL", "google/gemma-4-e4b")
    )
    api_key: str = field(default_factory=lambda: _setting("LLM_API_KEY", "lm-studio"))

    # Code generation must be near-deterministic. A 4B model at temperature 1.0
    # invents column names that do not exist in the schema.
    codegen_temperature: float = 0.1
    narrative_temperature: float = 0.4
    timeout_seconds: float = 120.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class DataConfig:
    """Dataset location and cleaning thresholds."""

    csv_path: Path = PROJECT_ROOT / os.getenv("DATA_PATH", "data/superstore.csv")
    date_columns: tuple[str, ...] = ("Order Date", "Ship Date")
    # IQR multiplier used by the outlier profiler (Task A3).
    iqr_multiplier: float = 1.5


LLM = LLMConfig()
DATA = DataConfig()
