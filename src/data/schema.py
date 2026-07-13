"""Schema extraction (Task A1).

The schema produced here is the single artefact handed to the LLM as context.
Its job is to give the model enough grounding to pick real column names and
real category values, while staying small enough to leave room for the
conversation history and few-shot examples inside the context window.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

# Categorical columns with more distinct values than this fall back to sample
# values, to keep the prompt from ballooning. The threshold is set to 20 so that
# Sub-Category (17 distinct values) is listed in full: its values are
# dataset-specific, so the model cannot guess them, and it is one of the most
# common grouping columns in the benchmark questions.
MAX_LISTED_CATEGORIES = 20
SAMPLE_VALUES = 3


@dataclass
class ColumnSchema:
    """Everything the LLM needs to know about one column."""

    name: str
    dtype: str
    semantic_type: str  # numeric | categorical | datetime | identifier
    null_count: int
    unique_count: int
    sample_values: list[str]
    categories: list[str] | None = None  # populated for low-cardinality columns
    min_value: str | None = None
    max_value: str | None = None
    description: str = ""


@dataclass
class DatasetSchema:
    """Full dataset description passed into the system prompt."""

    name: str
    row_count: int
    columns: list[ColumnSchema] = field(default_factory=list)
    business_context: str = ""

    def to_prompt_block(self) -> str:
        """Render the schema as compact text for injection into a system prompt.

        A table is used rather than JSON because it costs roughly half the
        tokens for the same information.
        """
        lines = [
            f"DATASET: {self.name} ({self.row_count:,} rows)",
            f"CONTEXT: {self.business_context}",
            "",
            "COLUMNS:",
        ]
        for col in self.columns:
            parts = [f"- {col.name} ({col.semantic_type}, {col.dtype})"]
            if col.description:
                parts.append(f"  desc: {col.description}")
            if col.categories:
                parts.append(f"  values: {', '.join(col.categories)}")
            elif col.semantic_type == "numeric":
                parts.append(f"  range: {col.min_value} to {col.max_value}")
            elif col.semantic_type == "datetime":
                parts.append(f"  range: {col.min_value} to {col.max_value}")
            else:
                parts.append(f"  examples: {', '.join(col.sample_values)}")
            if col.null_count:
                parts.append(f"  nulls: {col.null_count}")
            lines.append("\n".join(parts))
        return "\n".join(lines)

    def to_json(self) -> str:
        """Structured export for the written report (Section 5)."""
        return json.dumps(asdict(self), indent=2, default=str)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # A near-unique object column is an ID, not a category worth grouping on.
    if series.nunique(dropna=True) > 0.9 * len(series):
        return "identifier"
    return "categorical"


def extract_schema(
    df: pd.DataFrame,
    name: str,
    business_context: str = "",
    descriptions: dict[str, str] | None = None,
) -> DatasetSchema:
    """Build a DatasetSchema by inspecting a DataFrame.

    Args:
        df: The loaded, cleaned dataset.
        name: Human-readable dataset name shown to the LLM.
        business_context: One or two sentences telling the LLM what the data is
            about. This measurably improves column selection on small models.
        descriptions: Optional per-column notes merged into the schema.

    Returns:
        A DatasetSchema ready to be rendered into a prompt.
    """
    descriptions = descriptions or {}
    columns: list[ColumnSchema] = []

    for col_name in df.columns:
        series = df[col_name]
        sem = _semantic_type(series)
        unique_count = int(series.nunique(dropna=True))

        categories: list[str] | None = None
        if sem == "categorical" and unique_count <= MAX_LISTED_CATEGORIES:
            categories = [str(v) for v in sorted(series.dropna().unique())]

        min_value = max_value = None
        if sem in ("numeric", "datetime") and not series.dropna().empty:
            min_value = str(series.min())
            max_value = str(series.max())

        columns.append(
            ColumnSchema(
                name=col_name,
                dtype=str(series.dtype),
                semantic_type=sem,
                null_count=int(series.isna().sum()),
                unique_count=unique_count,
                sample_values=[str(v) for v in series.dropna().head(SAMPLE_VALUES)],
                categories=categories,
                min_value=min_value,
                max_value=max_value,
                description=descriptions.get(col_name, ""),
            )
        )

    return DatasetSchema(
        name=name,
        row_count=len(df),
        columns=columns,
        business_context=business_context,
    )
