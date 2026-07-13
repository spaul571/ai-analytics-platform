"""Central configuration loaded from environment / .env file / Streamlit secrets.

Every tunable that differs between machines lives here so no module hardcodes
a host, port, or file path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Store strings with Python objects rather than Arrow.
#
# pandas 3.0 defaults `mode.string_storage` to "auto", which means Arrow-backed
# strings whenever pyarrow is installed — and Streamlit always installs pyarrow.
# Every text column here is then an ArrowStringArray, so the `category` dtypes
# built on top of them are Arrow-backed too, and merely *materialising* one
# (iterating it, or calling .unique()) dispatches into pyarrow.compute.take.
#
# On the Linux wheels that Streamlit Cloud installs (pyarrow 25.0.0, pandas
# 3.0.1) that call segfaults: the container died with SIGSEGV inside
# pyarrow/compute.py:508 take, reached from Categorical.__iter__, the moment the
# sidebar built its filter options. The Windows wheels survive the same call,
# which is why this never reproduced locally.
#
# Setting "python" keeps strings in object arrays, so the crashing path is never
# entered. It costs some memory — the 62.1% downcast figure in the report is
# measured with this setting in place — and it is set here, in the module every
# other module imports, so that it is applied before any DataFrame is built.
pd.options.mode.string_storage = "python"


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
