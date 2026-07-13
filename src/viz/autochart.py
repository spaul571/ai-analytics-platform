"""AI-driven visualisation (Task C3).

When the NL pipeline returns a result DataFrame, this module picks an
appropriate chart type for its shape, renders it, and asks the LLM for a
one-sentence caption.

The selection is rule-based, not model-generated. The shape of a DataFrame is a
fact - how many rows, which columns are numeric, whether the label column is a
date - and facts do not need an LLM. Asking a 4B model to choose the chart type
would add a second-and-a-half of latency and a new failure mode to a decision
that four `if` statements make correctly every time. The LLM is used for the one
part of this that genuinely needs language: the caption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from src.config import LLM
from src.llm.client import LLMClient, LLMError
from src.viz.theme import STATE_CODES, Theme

ChartType = Literal["bar", "line", "scatter", "map", "table"]

CHART_LABELS: dict[str, str] = {
    "bar": "Bar chart",
    "line": "Line chart",
    "scatter": "Scatter plot",
    "map": "Map",
    "table": "Table",
}

# Above this many categories a bar chart becomes an unreadable comb, so the
# result is shown as a table instead.
MAX_BARS = 30

CAPTION_SYSTEM_PROMPT = """You caption charts for a business dashboard.

Given a question and the data behind a chart, write ONE sentence describing what
the chart shows.

RULES:
- One sentence. No preamble, no "This chart shows".
- Name the single most important fact: the winner, the outlier, or the trend.
- Cite at most two numbers, and only numbers present in the data.
- Plain business language.
"""


@dataclass
class AutoChart:
    """A chart chosen for a result frame, plus its caption."""

    figure: go.Figure | None
    chart_type: ChartType
    caption: str = ""
    available_types: list[ChartType] = None  # type: ignore[assignment]
    reason: str = ""

    def __post_init__(self) -> None:
        if self.available_types is None:
            self.available_types = ["bar", "line", "scatter", "table"]


def _is_temporal(series: pd.Series) -> bool:
    """Is this column a time axis?

    Covers real datetimes plus the string forms the derived columns produce
    ('2016-11' from Order Month) and plain four-digit years from Order Year.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    if sample.str.fullmatch(r"\d{4}-\d{2}").all():  # YYYY-MM
        return True
    if sample.str.fullmatch(r"(19|20)\d{2}").all():  # YYYY
        return True
    return False


def _is_geographic(series: pd.Series) -> bool:
    """Does this column hold US state names we can map?"""
    values = set(series.dropna().astype(str).unique())
    if not values:
        return False
    return len(values & set(STATE_CODES)) >= max(3, 0.8 * len(values))


def select_chart_type(frame: pd.DataFrame) -> tuple[ChartType, list[ChartType], str]:
    """Choose a chart type from the shape of the result.

    Returns:
        (chosen, alternatives_the_user_may_switch_to, why_this_was_chosen)
    """
    if frame.empty:
        return "table", ["table"], "The query returned no rows."

    numeric = frame.select_dtypes("number").columns.tolist()
    non_numeric = [c for c in frame.columns if c not in numeric]

    # A single number is not a chart. Show it as a value.
    if frame.shape == (1, 1) or (len(frame) == 1 and len(numeric) == 1):
        return "table", ["table"], "A single value; a chart would add nothing."

    # No label column to plot against.
    if not non_numeric and len(numeric) < 2:
        return "table", ["table"], "No dimension to plot against."

    label_col = non_numeric[0] if non_numeric else frame.columns[0]

    # Geographic beats everything: a map of states is strictly more informative
    # than a 40-bar chart of state names.
    if _is_geographic(frame[label_col]) and numeric:
        return (
            "map",
            ["map", "bar", "table"],
            f"'{label_col}' holds US states, so the result is mapped.",
        )

    # Time on the x-axis means a line: the reader should see the shape of the
    # change, and bars imply discrete unrelated categories.
    if _is_temporal(frame[label_col]) and numeric:
        return (
            "line",
            ["line", "bar", "table"],
            f"'{label_col}' is a time axis, so the trend is drawn as a line.",
        )

    # Two numeric columns and no meaningful label: the question is about the
    # relationship between them.
    if len(numeric) >= 2 and not non_numeric:
        return (
            "scatter",
            ["scatter", "line", "table"],
            "Two numeric columns; the relationship between them is plotted.",
        )

    # A categorical label with a metric: compare magnitudes with bars.
    if non_numeric and numeric:
        if len(frame) > MAX_BARS:
            return (
                "table",
                ["table", "bar"],
                f"{len(frame)} categories is too many to read as bars.",
            )
        return (
            "bar",
            ["bar", "table"],
            f"Comparing a metric across {len(frame)} values of '{label_col}'.",
        )

    return "table", ["table"], "The result shape does not suit a chart."


def render(frame: pd.DataFrame, chart_type: ChartType, theme: Theme, title: str = "") -> go.Figure | None:
    """Render a result frame as the requested chart type."""
    if chart_type == "table" or frame.empty:
        return None

    numeric = frame.select_dtypes("number").columns.tolist()
    non_numeric = [c for c in frame.columns if c not in numeric]
    if not numeric:
        return None

    label_col = non_numeric[0] if non_numeric else frame.columns[0]
    value_col = numeric[-1] if numeric[-1] != label_col else numeric[0]

    signed = bool((frame[value_col] < 0).any())
    fig = go.Figure()

    if chart_type == "map":
        codes = frame[label_col].astype(str).map(STATE_CODES)
        limit = float(frame[value_col].abs().max())
        fig.add_trace(
            go.Choropleth(
                locations=codes,
                z=frame[value_col],
                locationmode="USA-states",
                colorscale=theme.diverging if signed else theme.sequential,
                zmid=0 if signed else None,
                zmin=-limit if signed else None,
                zmax=limit if signed else None,
                marker_line_color=theme.ink["surface"],
                marker_line_width=0.8,
                colorbar=dict(title=dict(text=value_col, side="right"), thickness=14),
                text=frame[label_col],
                hovertemplate="<b>%{text}</b><br>%{z:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            geo=dict(scope="usa", bgcolor=theme.ink["surface"], landcolor=theme.ink["grid"]),
            margin=dict(l=8, r=8, t=56, b=8),
        )

    elif chart_type == "line":
        fig.add_trace(
            go.Scatter(
                x=frame[label_col],
                y=frame[value_col],
                mode="lines+markers",
                line=dict(width=2, color=theme.categorical[0]),
                marker=dict(size=8),
                name=value_col,
                hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title=label_col, yaxis_title=value_col)

    elif chart_type == "scatter":
        x_col, y_col = numeric[0], numeric[-1]
        fig.add_trace(
            go.Scatter(
                x=frame[x_col],
                y=frame[y_col],
                mode="markers",
                marker=dict(
                    size=9,
                    color=theme.categorical[0],
                    line=dict(width=1, color=theme.ink["surface"]),
                ),
                hovertemplate=f"{x_col}: %{{x:,.2f}}<br>{y_col}: %{{y:,.2f}}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)

    else:  # bar
        # A signed metric gets the diverging scale: losses must not look like
        # small gains. An unsigned one keeps a single categorical hue, because
        # colouring bars by their own value double-encodes the height and adds
        # nothing.
        if signed:
            limit = float(frame[value_col].abs().max())
            marker = dict(
                color=frame[value_col],
                colorscale=theme.diverging,
                cmid=0,
                cmin=-limit,
                cmax=limit,
                line=dict(color=theme.ink["surface"], width=2),
            )
        else:
            marker = dict(
                color=theme.categorical[0],
                line=dict(color=theme.ink["surface"], width=2),
            )

        fig.add_trace(
            go.Bar(
                x=frame[label_col].astype(str),
                y=frame[value_col],
                marker=marker,
                hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title=label_col, yaxis_title=value_col)
        if signed:
            fig.add_hline(y=0, line=dict(color=theme.ink["axis"], width=1))

    fig.update_layout(
        template=theme.plotly,
        title=title or value_col,
        showlegend=False,
        height=420,
    )
    return fig


def caption(
    question: str, frame: pd.DataFrame, chart_type: ChartType, client: LLMClient | None = None
) -> str:
    """Ask the LLM for a one-sentence description of the chart.

    Falls back to a mechanical description if the model is unavailable, so a
    chart is never shipped without a caption.
    """
    if frame.empty:
        return "The query returned no rows."

    preview = frame.head(15).to_string(index=False)
    client = client or LLMClient()
    try:
        response = client.complete(
            [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {question}\n"
                        f"CHART TYPE: {CHART_LABELS[chart_type]}\n"
                        f"DATA:\n{preview}\n\nWrite the caption."
                    ),
                },
            ],
            temperature=LLM.narrative_temperature,
            max_tokens=120,
        )
        # The model occasionally returns two sentences despite the instruction;
        # keep the first, which is always the one carrying the finding.
        text = " ".join(response.text.split())
        first = re.split(r"(?<=[.!?])\s+", text)[0]
        return first
    except LLMError:
        return (
            f"{CHART_LABELS[chart_type]} of {len(frame)} rows "
            f"({', '.join(frame.columns[:3])})."
        )


def build(
    question: str,
    frame: pd.DataFrame,
    theme: Theme,
    client: LLMClient | None = None,
    override: ChartType | None = None,
) -> AutoChart:
    """Select, render, and caption a chart for an NL query result.

    Args:
        override: A chart type chosen by the user, which wins over the automatic
            selection. Task C3 requires the user be able to override.
    """
    chosen, alternatives, reason = select_chart_type(frame)
    chart_type = override or chosen

    figure = render(frame, chart_type, theme, title=question)
    text = caption(question, frame, chart_type, client) if figure is not None else ""

    return AutoChart(
        figure=figure,
        chart_type=chart_type,
        caption=text,
        available_types=alternatives,
        reason=reason if override is None else f"Overridden by user (auto: {chosen}).",
    )
