"""Query engine (Task A2).

Two query modes over the in-memory DataFrame, both returning a DataFrame and a
logged execution time. No database server is involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Aggregation = Literal["count", "sum", "mean", "min", "max", "median", "std"]


@dataclass
class QueryResult:
    """A query outcome plus the timing evidence Task A4 asks for."""

    data: pd.DataFrame
    execution_ms: float
    rows_scanned: int
    rows_returned: int
    description: str

    def __repr__(self) -> str:
        return (
            f"QueryResult({self.rows_returned} rows in {self.execution_ms:.1f}ms, "
            f"scanned {self.rows_scanned:,})"
        )


@dataclass
class Filter:
    """One predicate applied before aggregation.

    Attributes:
        column: Column to filter on.
        op: One of 'in', 'eq', 'between', 'gte', 'lte'.
        value: A list for 'in', a 2-tuple for 'between', a scalar otherwise.
    """

    column: str
    op: Literal["in", "eq", "between", "gte", "lte"]
    value: Any


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    """Apply predicates sequentially. Unknown columns raise rather than pass
    silently, so a bad LLM-generated filter surfaces as an error instead of a
    wrong answer."""
    result = df
    for f in filters:
        if f.column not in result.columns:
            raise KeyError(f"Unknown filter column: {f.column!r}")
        series = result[f.column]
        if f.op == "in":
            result = result[series.isin(f.value)]
        elif f.op == "eq":
            result = result[series == f.value]
        elif f.op == "between":
            low, high = f.value
            result = result[series.between(low, high)]
        elif f.op == "gte":
            result = result[series >= f.value]
        elif f.op == "lte":
            result = result[series <= f.value]
        else:
            raise ValueError(f"Unsupported filter op: {f.op!r}")
    return result


def aggregate(
    df: pd.DataFrame,
    group_by: list[str],
    metrics: dict[str, list[Aggregation]],
    filters: list[Filter] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int | None = None,
) -> QueryResult:
    """Group and aggregate, optionally filtering first.

    Covers both required query modes: pass no filters for a direct aggregation,
    pass filters for a filtered query.

    Args:
        df: Source table.
        group_by: Columns to group on. Empty list aggregates the whole table.
        metrics: Mapping of column -> list of aggregations, e.g.
            {"Sales": ["sum", "mean"], "Order ID": ["count"]}.
        filters: Predicates applied before grouping.
        sort_by: Flattened output column to sort on, e.g. "Sales_sum".
        ascending: Sort direction.
        limit: Keep only the first N rows after sorting.

    Returns:
        QueryResult with a flat-column DataFrame and the elapsed time.
    """
    start = time.perf_counter()
    rows_scanned = len(df)

    working = apply_filters(df, filters) if filters else df

    for col in [*group_by, *metrics]:
        if col not in working.columns:
            raise KeyError(f"Unknown column: {col!r}")

    if group_by:
        grouped = working.groupby(group_by, observed=True, dropna=False).agg(metrics)
        # agg() with a dict produces a MultiIndex on the columns; flatten it to
        # 'Sales_sum' style names so downstream charting and the LLM see plain
        # column names.
        grouped.columns = ["_".join(c) for c in grouped.columns]
        out = grouped.reset_index()
    else:
        row = {
            f"{col}_{agg}": working[col].agg(agg)
            for col, aggs in metrics.items()
            for agg in aggs
        }
        out = pd.DataFrame([row])

    if sort_by:
        if sort_by not in out.columns:
            raise KeyError(f"Cannot sort by {sort_by!r}; available: {list(out.columns)}")
        out = out.sort_values(sort_by, ascending=ascending)

    if limit:
        out = out.head(limit)

    out = out.reset_index(drop=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    desc = f"aggregate(group_by={group_by}, metrics={metrics}"
    if filters:
        desc += f", filters={[(f.column, f.op, f.value) for f in filters]}"
    desc += ")"

    return QueryResult(
        data=out,
        execution_ms=elapsed_ms,
        rows_scanned=rows_scanned,
        rows_returned=len(out),
        description=desc,
    )
