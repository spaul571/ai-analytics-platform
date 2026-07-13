"""Central configuration loaded from environment / .env file.

Every tunable that differs between machines lives here so no module hardcodes
a host, port, or file path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for the LM Studio OpenAI-compatible endpoint."""

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    model: str = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
    api_key: str = os.getenv("LLM_API_KEY", "lm-studio")

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
