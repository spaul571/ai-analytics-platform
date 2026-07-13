"""Data quality profiling (Task A3).

Produces the numbers behind the data quality summary card in the UI and the
quality section of the written report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import DATA


@dataclass
class ColumnProfile:
    """Per-column quality and distribution statistics."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    outlier_count: int = 0  # numeric columns only, via IQR
    mean: float | None = None
    std: float | None = None
    min_value: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max_value: float | None = None


@dataclass
class DataQualityReport:
    """Whole-dataset quality snapshot."""

    row_count: int
    column_count: int
    duplicate_rows: int
    total_cells: int
    missing_cells: int
    memory_mb: float
    columns: list[ColumnProfile] = field(default_factory=list)
    cleaning_steps: list[str] = field(default_factory=list)

    @property
    def completeness_pct(self) -> float:
        """Share of non-null cells. The headline number on the quality card."""
        if not self.total_cells:
            return 0.0
        return 100.0 * (1 - self.missing_cells / self.total_cells)

    @property
    def total_outliers(self) -> int:
        return sum(c.outlier_count for c in self.columns)

    def to_frame(self) -> pd.DataFrame:
        """Tabular view for display in the UI and the report."""
        return pd.DataFrame(
            [
                {
                    "Column": c.name,
                    "Type": c.dtype,
                    "Nulls": c.null_count,
                    "Null %": round(c.null_pct, 2),
                    "Unique": c.unique_count,
                    "Outliers": c.outlier_count,
                    "Mean": None if c.mean is None else round(c.mean, 2),
                    "Median": None if c.median is None else round(c.median, 2),
                    "Std": None if c.std is None else round(c.std, 2),
                }
                for c in self.columns
            ]
        )


def iqr_outlier_count(series: pd.Series, multiplier: float = DATA.iqr_multiplier) -> int:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR].

    Used for reporting only; these rows are kept, not dropped. In this dataset
    the extreme Profit and Sales values are genuine large orders, and removing
    them would erase exactly the anomalies Task D3 is meant to surface.
    """
    clean = series.dropna()
    if clean.empty:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    return int(((clean < low) | (clean > high)).sum())


def profile_dataset(
    df: pd.DataFrame, cleaning_steps: list[str] | None = None
) -> DataQualityReport:
    """Build the full quality report for a loaded DataFrame."""
    columns: list[ColumnProfile] = []
    row_count = len(df)

    for name in df.columns:
        series = df[name]
        null_count = int(series.isna().sum())
        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            null_count=null_count,
            null_pct=100.0 * null_count / row_count if row_count else 0.0,
            unique_count=int(series.nunique(dropna=True)),
        )

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna().astype(float)
            if not clean.empty:
                profile.outlier_count = iqr_outlier_count(clean)
                profile.mean = float(clean.mean())
                profile.std = float(clean.std())
                profile.min_value = float(clean.min())
                profile.q1 = float(clean.quantile(0.25))
                profile.median = float(clean.median())
                profile.q3 = float(clean.quantile(0.75))
                profile.max_value = float(clean.max())

        columns.append(profile)

    return DataQualityReport(
        row_count=row_count,
        column_count=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        total_cells=int(np.prod(df.shape)),
        missing_cells=int(df.isna().sum().sum()),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        columns=columns,
        cleaning_steps=cleaning_steps or [],
    )
