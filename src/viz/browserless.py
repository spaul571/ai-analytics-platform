"""Chart rasterising without a browser (Task C4 fallback).

kaleido draws a Plotly figure by driving a headless Chrome. Streamlit Community
Cloud has no Chrome and cannot be given one: apt's `chromium` pulls Debian's own
libpython into the container and the interpreter then segfaults (see the git
history for packages.txt). Without a rasteriser the deployed reports lose their
chart, which is one of the four things the brief requires in an export.

So the deployed app redraws the figure with matplotlib instead - Agg is a pure
software rasteriser, no browser, no system libraries. This module reads the
traces back off the Plotly figure and redraws them. It is deliberately narrow:
it covers exactly the traces `autochart.render` emits (bar, line, scatter) and
refuses anything else rather than inventing an approximation. Choropleths are
refused - the map is drawn from Plotly's own geometry, which matplotlib does not
have.

Fidelity is close but not identical: same data, same palette, same titles, in
matplotlib's typography rather than Plotly's. That is the trade for having a
chart at all. Where Chrome exists (any local run) kaleido is used and this
module is never reached.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # no display, no GUI toolkit - must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Print, not screen: the PNG lands in a PDF or a Word page, so the figure is
# drawn on white with dark ink regardless of the app's light/dark theme.
_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e1e0d9"
_DEFAULT_BLUE = "#2a78d6"


class UnsupportedFigure(RuntimeError):
    """The figure holds a trace this fallback will not redraw."""


def _to_mpl_color(color) -> str:
    """Normalise one Plotly colour to something matplotlib accepts."""
    if isinstance(color, str) and color.startswith("rgb"):
        parts = color[color.index("(") + 1 : color.index(")")].split(",")
        return tuple(float(p) / 255 for p in parts[:3])
    return color


def _marker_colors(trace, count: int):
    """Resolve a trace's marker colour, whether flat or mapped through a scale.

    autochart colours a signed bar chart by running the values through the
    diverging scale, so `marker.color` is an array of numbers rather than a
    colour. That mapping has to be redone here or losses stop reading as losses.
    """
    marker = getattr(trace, "marker", None)
    color = getattr(marker, "color", None) if marker else None

    if color is None:
        return _DEFAULT_BLUE
    if isinstance(color, str):
        return _to_mpl_color(color)

    values = np.asarray(color, dtype=float)
    scale = getattr(marker, "colorscale", None)
    if scale is None:
        return _DEFAULT_BLUE

    low = marker.cmin if marker.cmin is not None else float(values.min())
    high = marker.cmax if marker.cmax is not None else float(values.max())
    span = high - low
    positions = (values - low) / span if span else np.full(count, 0.5)
    return _colormap(scale)(np.clip(positions, 0, 1))


def _colormap(scale) -> LinearSegmentedColormap:
    """Rebuild a Plotly colorscale as a matplotlib colormap.

    theme.py states its scales as [position, colour] stops, which is exactly
    what from_list takes, so the two renderers read the same palette rather
    than a matplotlib lookalike of it.
    """
    stops = [(float(position), _to_mpl_color(color)) for position, color in scale]
    return LinearSegmentedColormap.from_list("theme", stops)


def _axis_title(layout_axis) -> str:
    title = getattr(layout_axis, "title", None)
    return getattr(title, "text", None) or ""


def _draw_bar(ax, trace) -> None:
    x = [str(v) for v in trace.x]
    y = np.asarray(trace.y, dtype=float)
    ax.bar(x, y, color=_marker_colors(trace, len(y)), edgecolor="white", linewidth=0.8)
    if (y < 0).any():
        ax.axhline(0, color=_MUTED, linewidth=1)
    # Category labels are sub-category and state names - long enough to collide
    # horizontally past a handful of bars.
    if len(x) > 6 or max((len(v) for v in x), default=0) > 8:
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")


def _draw_scatter(ax, trace) -> None:
    x, y = trace.x, trace.y
    mode = trace.mode or "markers"
    line_color = _to_mpl_color(getattr(trace.line, "color", None) or _DEFAULT_BLUE)

    if "lines" in mode:
        ax.plot(
            x,
            y,
            color=line_color,
            linewidth=2,
            marker="o" if "markers" in mode else None,
            markersize=5,
        )
    else:
        ax.scatter(
            x,
            y,
            s=36,
            color=_marker_colors(trace, len(y)),
            edgecolor="white",
            linewidth=0.5,
        )


def supports(figure: go.Figure) -> bool:
    """Whether every trace in the figure can be redrawn here."""
    return bool(figure.data) and all(
        isinstance(trace, (go.Bar, go.Scatter)) for trace in figure.data
    )


def figure_to_png_fallback(
    figure: go.Figure, width: int = 1000, height: int = 560, scale: int = 2
) -> bytes:
    """Redraw a Plotly figure with matplotlib and return PNG bytes."""
    return _render(figure, "png", width, height, scale)


def figure_to_svg_fallback(figure: go.Figure, width: int = 1000, height: int = 560) -> bytes:
    """Redraw a Plotly figure with matplotlib and return SVG bytes."""
    return _render(figure, "svg", width, height, 1)


def _render(figure: go.Figure, fmt: str, width: int, height: int, scale: int) -> bytes:
    if not figure.data:
        raise UnsupportedFigure("The figure has no traces to draw.")

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi * scale)
    try:
        for trace in figure.data:
            if isinstance(trace, go.Bar):
                _draw_bar(ax, trace)
            elif isinstance(trace, go.Scatter):
                _draw_scatter(ax, trace)
            else:
                raise UnsupportedFigure(
                    f"{type(trace).__name__} charts need a browser to render; "
                    "this environment has none."
                )

        layout = figure.layout
        title = getattr(layout.title, "text", None)
        if title:
            ax.set_title(title, color=_INK, fontsize=13, loc="left", pad=12)
        ax.set_xlabel(_axis_title(layout.xaxis), color=_MUTED, fontsize=10)
        ax.set_ylabel(_axis_title(layout.yaxis), color=_MUTED, fontsize=10)

        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_MUTED, labelsize=9)

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, facecolor="white", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)  # Agg figures are not garbage collected on their own
