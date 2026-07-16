"""Render every chart to PNG for the written report (report section 7).

Saves each chart in the suite at print resolution, in both light and dark mode,
plus a captions file. Beats screenshotting the browser: the images are crisp,
reproducible, and free of surrounding UI chrome.

Run:  python -m scripts.export_report_assets
Output: report/figures/

Writes straight into the report folder, which is the copy REPORT.tex compiles
against. It used to write to the gitignored exports/ and the figures were copied
across by hand, which meant the committed images could silently drift from the
code that draws them.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.data.loader import load_dataset
from src.viz import charts
from src.viz.export import figure_to_png
from src.viz.theme import Theme

OUT = Path("report/figures")

# The design decision behind each chart, for the report's caption.
CAPTIONS: dict[str, str] = {
    "trend": (
        "Revenue and profit over time, as shared-x small multiples rather than a "
        "dual-axis chart. A dual y-axis would put the crossover point wherever we "
        "chose the scales; separate panels keep both measures truthful."
    ),
    "map": (
        "Sales by state. Sequential single-hue blue, because sales is a magnitude "
        "and more is simply more. Switching the metric to Profit switches the "
        "scale to diverging blue/red with a neutral grey midpoint, so a state "
        "losing money cannot look like a state making a little."
    ),
    "correlation": (
        "Correlation matrix on a diverging scale anchored at zero: -1 must look as "
        "strong as +1 while 0 looks like nothing. Cells carry a 2px surface gap."
    ),
    "distribution": (
        "Profit distribution by category as a box plot rather than a histogram. "
        "The question is about spread and the long negative tail; a histogram of a "
        "heavily skewed variable hides exactly that."
    ),
    "sunburst": (
        "Sales composition, Category to Sub-Category. The hierarchy is genuine - "
        "each sub-category belongs to exactly one category - which is the "
        "precondition for a sunburst being honest rather than decorative."
    ),
    "animated": (
        "Sales by region animated across years. The y-axis is fixed across all "
        "frames; letting it rescale per frame would make every year look identical "
        "and destroy the comparison the animation exists to make."
    ),
    "scatter": (
        "Discount against profit, with a least-squares fit and a 95% confidence "
        "interval on the mean response. This is the project's central finding: "
        "margin collapses as discount deepens."
    ),
    "stacked": (
        "Sales by customer segment, stacked by category, with a 2px surface gap "
        "between segments so adjacent fills stay distinguishable."
    ),
}


def main() -> int:
    df, _, _ = load_dataset()
    OUT.mkdir(parents=True, exist_ok=True)

    lines = ["# Figure captions\n"]
    failures = 0

    for mode in ("light", "dark"):
        theme = Theme(mode)
        for key, (label, builder) in charts.CHART_REGISTRY.items():
            try:
                figure = builder(df, theme)
                png = figure_to_png(figure, width=1400, height=800, scale=2)
                path = OUT / f"{key}_{mode}.png"
                path.write_bytes(png)
                print(f"  OK   {path}  ({len(png):,} bytes)")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {key} [{mode}]: {type(exc).__name__}: {exc}")
                failures += 1

    for key, (label, _) in charts.CHART_REGISTRY.items():
        lines.append(f"\n## {label}\n")
        lines.append(f"`figures/{key}_light.png`\n")
        lines.append(CAPTIONS.get(key, "") + "\n")

    captions = OUT / "captions.md"
    captions.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  OK   {captions}")

    print(
        f"\n{len(charts.CHART_REGISTRY)} charts x 2 themes -> {OUT}/"
        + ("" if not failures else f"  ({failures} FAILURES)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
