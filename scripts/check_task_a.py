"""Task A acceptance check.

Loads the dataset, prints the schema exactly as the LLM will see it, runs the
five sample queries required at milestone M2, and verifies the sub-500ms
performance target from Task A4.

Run:  python -m scripts.check_task_a
"""

from __future__ import annotations

import sys

from src.data.loader import load_dataset
from src.data.profile import profile_dataset
from src.data.query import Filter, aggregate

PERFORMANCE_BUDGET_MS = 500.0


def main() -> int:
    print("=" * 70)
    print("TASK A ACCEPTANCE CHECK")
    print("=" * 70)

    df, schema, meta = load_dataset()

    print(f"\n[A1] Loaded {meta['rows']:,} rows x {meta['columns']} columns")
    print(f"     read_csv:  {meta['load_seconds'] * 1000:.1f} ms")
    print(f"     cleaning:  {meta['clean_seconds'] * 1000:.1f} ms")
    print(
        f"     memory:    {meta['memory_raw_mb']} MB raw -> {meta['memory_mb']} MB "
        f"after dtype downcast ({meta['memory_saved_pct']}% saved)"
    )
    print("\n[A1] Cleaning steps applied:")
    for step in meta["cleaning_steps"]:
        print(f"     - {step}")

    print("\n[A1] Schema as the LLM will receive it:")
    print("-" * 70)
    print(schema.to_prompt_block())
    print("-" * 70)

    report = profile_dataset(df, meta["cleaning_steps"])
    print(f"\n[A3] Completeness:   {report.completeness_pct:.2f}%")
    print(f"[A3] Duplicate rows: {report.duplicate_rows}")
    print(f"[A3] IQR outliers:   {report.total_outliers:,}")

    queries = [
        (
            "Q1 direct aggregation - total sales and profit per region",
            lambda: aggregate(
                df,
                group_by=["Region"],
                metrics={"Sales": ["sum"], "Profit": ["sum"], "Order ID": ["count"]},
                sort_by="Sales_sum",
            ),
        ),
        (
            "Q2 direct aggregation - mean discount by sub-category",
            lambda: aggregate(
                df,
                group_by=["Sub-Category"],
                metrics={"Discount": ["mean"], "Profit": ["mean"]},
                sort_by="Discount_mean",
                limit=10,
            ),
        ),
        (
            "Q3 filtered query - 2017 furniture sales by state",
            lambda: aggregate(
                df,
                group_by=["State"],
                metrics={"Sales": ["sum", "mean"], "Quantity": ["sum"]},
                filters=[
                    Filter("Category", "eq", "Furniture"),
                    Filter("Order Year", "eq", 2017),
                ],
                sort_by="Sales_sum",
                limit=10,
            ),
        ),
        (
            "Q4 filtered query - heavily discounted loss-making lines",
            lambda: aggregate(
                df,
                group_by=["Category", "Sub-Category"],
                metrics={"Profit": ["sum", "mean"], "Order ID": ["count"]},
                filters=[
                    Filter("Discount", "gte", 0.3),
                    Filter("Profit", "lte", 0),
                ],
                sort_by="Profit_sum",
                ascending=True,
            ),
        ),
        (
            "Q5 whole-table aggregation - no grouping",
            lambda: aggregate(
                df,
                group_by=[],
                metrics={"Sales": ["sum", "mean"], "Profit": ["sum"], "Discount": ["mean"]},
            ),
        ),
    ]

    print("\n[A2/A4] Sample queries")
    print("-" * 70)
    slowest = 0.0
    for label, fn in queries:
        result = fn()
        slowest = max(slowest, result.execution_ms)
        flag = "OK " if result.execution_ms < PERFORMANCE_BUDGET_MS else "SLOW"
        print(f"{flag} {label}")
        print(f"     {result.execution_ms:6.1f} ms | {result.rows_returned} rows returned")
        print(result.data.head(5).to_string(index=False, max_colwidth=24))
        print()

    print("-" * 70)
    passed = slowest < PERFORMANCE_BUDGET_MS
    verdict = "PASS" if passed else "FAIL"
    print(f"[A4] Slowest query: {slowest:.1f} ms (budget {PERFORMANCE_BUDGET_MS:.0f} ms) -> {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
