"""Task C acceptance check.

Renders every chart, exercises the auto-chart selector against the shapes the NL
pipeline actually produces, and writes a real PDF, DOCX, PNG and SVG to
`exports/`. Needs no LLM: the caption falls back to a mechanical description when
the model is unreachable, so the whole visual and export path can be verified
offline.

Run:  python -m scripts.check_task_c
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset
from src.viz import autochart, charts
from src.viz.export import (
    ReportPayload,
    figure_to_png,
    figure_to_svg,
    to_docx,
    to_pdf,
)
from src.viz.theme import Theme

OUT = Path("exports")


def main() -> int:
    print("=" * 74)
    print("TASK C ACCEPTANCE CHECK")
    print("=" * 74)

    df, schema, meta = load_dataset()
    OUT.mkdir(exist_ok=True)
    failures = 0

    # ---------------------------------------------------------------- C2
    print("\n[C2] Chart suite (need 6+, all on one palette)")
    print("-" * 74)
    for key, (label, builder) in charts.CHART_REGISTRY.items():
        for mode in ("light", "dark"):
            theme = Theme(mode)
            try:
                figure = builder(df, theme)
                traces = len(figure.data)
                title = figure.layout.title.text or "(no title)"
                if mode == "light":
                    print(f"  OK   {label:32s} {traces:2d} traces | {title}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {label:32s} [{mode}] {type(exc).__name__}: {exc}")
                failures += 1
    print(f"\n  {len(charts.CHART_REGISTRY)} chart types, both themes.")

    # ---------------------------------------------------------------- C3
    print("\n[C3] Auto chart selection (result shape -> chart type)")
    print("-" * 74)
    theme = Theme("light")

    # The exact result shapes the 10 benchmark questions produce.
    cases: list[tuple[str, pd.DataFrame, str]] = [
        (
            "categorical + metric",
            df.groupby("Region", observed=True)["Sales"].sum().reset_index(),
            "bar",
        ),
        (
            "time axis (YYYY-MM)",
            df.groupby("Order Month", observed=True)["Sales"].sum().reset_index(),
            "line",
        ),
        (
            "time axis (year)",
            df.groupby("Order Year", observed=True)["Sales"].sum().reset_index(),
            "line",
        ),
        (
            "US states",
            df.groupby("State", observed=True)["Sales"].sum().reset_index(),
            "map",
        ),
        (
            "single scalar",
            pd.DataFrame([{"Value": 18.72}]),
            "table",
        ),
        (
            "many categories",
            df.groupby("Customer Name", observed=True)["Sales"].sum().reset_index(),
            "table",
        ),
        (
            "two numerics",
            df[["Discount", "Profit"]].head(200),
            "scatter",
        ),
    ]

    for name, frame, expected in cases:
        chosen, alternatives, reason = autochart.select_chart_type(frame)
        ok = chosen == expected
        if not ok:
            failures += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name:22s} -> {chosen:8s} "
              f"(expected {expected}) | {reason}")

    # Rendering must succeed for every type it can choose.
    print("\n  Rendering each selected type:")
    for name, frame, _ in cases:
        chosen, _, _ = autochart.select_chart_type(frame)
        try:
            figure = autochart.render(frame, chosen, theme, title=name)
            state = "rendered" if figure is not None else "table (no figure)"
            print(f"    OK   {name:22s} {chosen:8s} {state}")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {name:22s} {chosen:8s} {type(exc).__name__}: {exc}")
            failures += 1

    # ---------------------------------------------------------------- C4
    print("\n[C4] Export")
    print("-" * 74)

    result = (
        df.groupby("Sub-Category", observed=True)["Profit"]
        .sum()
        .sort_values()
        .head(8)
        .reset_index()
    )
    figure = autochart.render(result, "bar", theme, title="Least profitable sub-categories")

    payload = ReportPayload(
        title="AI Analytics Report",
        question="Which sub-categories are losing us money?",
        narrative=(
            "**Tables, Bookcases and Supplies are structurally unprofitable.**\n\n"
            "- Tables lost $17,725 across the period.\n"
            "- Bookcases lost $3,473.\n"
            "- The losses track discount depth, not sales volume.\n\n"
            "Recommend capping discounts on Tables at 20%."
        ),
        data=result,
        figure=figure,
        caption="Tables lose more money than every other sub-category combined.",
        code="totals = df.groupby('Sub-Category', observed=True)['Profit'].sum()\n"
        "result = totals[totals < 0].sort_values()",
        filters={"Year range": "2014–2017", "Region": ["West", "East"]},
        dataset_name=schema.name,
        row_count=len(df),
        model="google/gemma-4-e4b",
    )

    exports = [
        ("PDF", "report.pdf", lambda: to_pdf(payload)),
        ("DOCX", "report.docx", lambda: to_docx(payload)),
        ("PNG", "chart.png", lambda: figure_to_png(figure)),
        ("SVG", "chart.svg", lambda: figure_to_svg(figure)),
    ]

    for label, filename, build in exports:
        try:
            data = build()
            path = OUT / filename
            path.write_bytes(data)
            print(f"  OK   {label:5s} {len(data):>8,} bytes -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {label:5s} {type(exc).__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILURES: {failures}")
    else:
        print("ALL CHECKS PASS — open exports/report.pdf and exports/report.docx")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
