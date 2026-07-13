"""The chart suite (Task C2).

Eight chart types, all drawing colour from src.viz.theme so the suite reads as
one system. Every chart carries a title, axis labels, and hover tooltips.

A NOTE ON THE DUAL-AXIS OPTION
------------------------------
The brief offers "time series or trend chart with dual axis" as one of the
permitted chart types. This module deliberately does not build one.

A dual-axis chart plots two measures on two independent y-scales in one frame.
Where the two lines cross is then an artefact of the scales the author chose, not
a fact about the data: rescale either axis and the crossover moves anywhere you
like. It is the most reliable way to make a chart lie without stating a single
false number.

`trend_small_multiples` answers the same question honestly - Sales and Profit
share an x-axis in stacked panels, so the shapes can be compared while each keeps
its own truthful scale. Seven other chart types from the brief's list are
implemented, so the six-type requirement is met without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import STATE_CODES, Theme

# Numeric columns offered to the correlation matrix and scatter plot. Row ID and
# Postal Code are numeric but meaningless to correlate, so they are excluded.
METRIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit", "Shipping Days"]


def _money(values: pd.Series) -> list[str]:
    return [f"${v:,.0f}" for v in values]


# ------------------------------------------------------------------ 1. trend
def trend_small_multiples(df: pd.DataFrame, theme: Theme) -> go.Figure:
    """Sales and Profit over time, stacked panels sharing one x-axis.

    The honest alternative to a dual-axis chart: the shapes are comparable
    because the x-axis is shared, but neither measure is distorted to fit the
    other's scale.
    """
    monthly = (
        df.groupby("Order Month", observed=True)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("Order Month")
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Monthly revenue", "Monthly profit"),
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["Order Month"],
            y=monthly["Sales"],
            name="Sales",
            mode="lines",
            line=dict(width=2, color=theme.categorical[0]),
            hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Profit is signed, so the zero line is meaningful and is drawn explicitly.
    fig.add_trace(
        go.Scatter(
            x=monthly["Order Month"],
            y=monthly["Profit"],
            name="Profit",
            mode="lines",
            line=dict(width=2, color=theme.categorical[1]),
            hovertemplate="<b>%{x}</b><br>Profit: $%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=0, row=2, col=1, line=dict(color=theme.status["critical"], width=1, dash="dot")
    )

    fig.update_layout(
        template=theme.plotly,
        title="Revenue and profit over time",
        hovermode="x unified",
        showlegend=False,  # one series per panel; the panel titles name them
        height=460,
    )
    fig.update_yaxes(title_text="Sales (USD)", row=1, col=1, tickprefix="$")
    fig.update_yaxes(title_text="Profit (USD)", row=2, col=1, tickprefix="$")
    fig.update_xaxes(title_text="Month", row=2, col=1)
    return fig


# -------------------------------------------------------------- 2. choropleth
def state_choropleth(df: pd.DataFrame, theme: Theme, metric: str = "Sales") -> go.Figure:
    """Geographic distribution of a metric across US states.

    Sequential blue for Sales (magnitude - more is simply more). Diverging
    blue/red with a neutral midpoint for Profit, because the sign of profit is
    the whole point: a state losing money must not look like a state making a
    little money.
    """
    by_state = (
        df.groupby("State", observed=True)[metric]
        .sum()
        .reset_index()
        .assign(code=lambda d: d["State"].map(STATE_CODES))
        .dropna(subset=["code"])
    )

    diverging = metric == "Profit"
    scale = theme.diverging if diverging else theme.sequential
    # Anchor the diverging scale symmetrically on zero so the neutral midpoint
    # really does mean zero, rather than landing wherever the data's midpoint is.
    limit = float(by_state[metric].abs().max()) if diverging else None

    fig = go.Figure(
        go.Choropleth(
            locations=by_state["code"],
            z=by_state[metric],
            locationmode="USA-states",
            colorscale=scale,
            zmid=0 if diverging else None,
            zmin=-limit if diverging else None,
            zmax=limit if diverging else None,
            marker_line_color=theme.ink["surface"],
            marker_line_width=0.8,
            colorbar=dict(
                title=dict(text=f"{metric} (USD)", side="right"),
                tickprefix="$",
                thickness=14,
                outlinewidth=0,
            ),
            text=by_state["State"],
            hovertemplate="<b>%{text}</b><br>" + metric + ": $%{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=theme.plotly,
        title=f"{metric} by state",
        geo=dict(
            scope="usa",
            bgcolor=theme.ink["surface"],
            lakecolor=theme.ink["surface"],
            landcolor=theme.ink["grid"],
            subunitcolor=theme.ink["surface"],
        ),
        height=440,
        margin=dict(l=8, r=8, t=64, b=8),
    )
    return fig


# ---------------------------------------------------------------- 3. heatmap
def correlation_matrix(df: pd.DataFrame, theme: Theme) -> go.Figure:
    """Correlation between the numeric metrics.

    Diverging scale anchored at zero: correlation has a sign, and -1 must look as
    strong as +1 while 0 looks like nothing.
    """
    available = [c for c in METRIC_COLUMNS if c in df.columns]
    corr = df[available].corr().round(2)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=theme.diverging,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            xgap=2,  # the 2px surface gap between cells
            ygap=2,
            colorbar=dict(title=dict(text="r", side="right"), thickness=14, outlinewidth=0),
            hovertemplate="<b>%{y} vs %{x}</b><br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=theme.plotly,
        title="Correlation between metrics",
        height=420,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig


# ----------------------------------------------------------- 4. distribution
def profit_distribution(df: pd.DataFrame, theme: Theme) -> go.Figure:
    """Profit distribution per category, as a box plot.

    A box plot rather than a histogram because the question here is about spread
    and outliers - the long negative tail is the story, and a histogram of a
    heavily skewed variable hides it.
    """
    categories = sorted(df["Category"].dropna().unique())
    colours = theme.colour_for(categories)

    fig = go.Figure()
    for cat in categories:
        subset = df[df["Category"] == cat]["Profit"]
        fig.add_trace(
            go.Box(
                y=subset,
                name=str(cat),
                marker=dict(color=colours[cat], size=4),
                line=dict(width=2),
                boxpoints="outliers",
                hovertemplate=f"<b>{cat}</b><br>Profit: $%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0,
        line=dict(color=theme.status["critical"], width=1, dash="dot"),
        annotation_text="break-even",
        annotation_position="right",
        annotation_font=dict(color=theme.ink["muted"], size=11),
    )
    fig.update_layout(
        template=theme.plotly,
        title="Profit distribution by category",
        yaxis_title="Profit per order line (USD)",
        xaxis_title="Category",
        yaxis=dict(tickprefix="$"),
        showlegend=False,  # the x-axis already labels each box
        height=420,
    )
    return fig


# --------------------------------------------------------------- 5. sunburst
def category_sunburst(df: pd.DataFrame, theme: Theme) -> go.Figure:
    """Sales composition: Category -> Sub-Category.

    The hierarchy is genuine (each sub-category belongs to exactly one category),
    which is the precondition for a sunburst being honest rather than decorative.
    """
    grouped = (
        df.groupby(["Category", "Sub-Category"], observed=True)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
        # px.sunburst builds the hierarchy by comparing path levels, which fails
        # on an unordered Categorical ("Cannot perform max with non-ordered
        # Categorical"). The category dtype is what Task A's memory downcast
        # produced, so it has to be undone here rather than upstream.
        .astype({"Category": "str", "Sub-Category": "str"})
    )
    categories = sorted(grouped["Category"].unique())
    colours = theme.colour_for(categories)

    fig = px.sunburst(
        grouped,
        path=["Category", "Sub-Category"],
        values="Sales",
        color="Category",
        color_discrete_map=colours,
        custom_data=["Profit"],
    )
    fig.update_traces(
        marker=dict(line=dict(color=theme.ink["surface"], width=2)),  # 2px gap
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<extra></extra>",
        textfont=dict(size=12),
    )
    fig.update_layout(
        template=theme.plotly,
        title="Sales composition by category and sub-category",
        height=460,
        margin=dict(l=8, r=8, t=64, b=8),
    )
    return fig


# --------------------------------------------------------------- 6. animated
def animated_region_race(df: pd.DataFrame, theme: Theme) -> go.Figure:
    """Sales by sub-category, animated across years with a slider.

    The y-axis range is fixed across all frames. Letting it rescale per frame
    would make every year look identical and destroy the very comparison the
    animation exists to make.
    """
    grouped = (
        df.groupby(["Order Year", "Region"], observed=True)["Sales"]
        .sum()
        .reset_index()
        .sort_values(["Order Year", "Region"])
    )
    regions = sorted(grouped["Region"].unique())
    colours = theme.colour_for(regions)
    ceiling = float(grouped["Sales"].max()) * 1.12

    fig = px.bar(
        grouped,
        x="Region",
        y="Sales",
        color="Region",
        animation_frame="Order Year",
        color_discrete_map=colours,
        category_orders={"Region": regions},
        range_y=[0, ceiling],
    )
    fig.update_traces(
        marker=dict(line=dict(color=theme.ink["surface"], width=2)),
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        template=theme.plotly,
        title="Sales by region, year over year",
        xaxis_title="Region",
        yaxis_title="Sales (USD)",
        yaxis=dict(tickprefix="$"),
        showlegend=False,
        height=460,
    )
    return fig


# ---------------------------------------------------------------- 7. scatter
def discount_profit_scatter(df: pd.DataFrame, theme: Theme, sample: int = 2000) -> go.Figure:
    """Discount vs Profit with an OLS regression line and a 95% confidence band.

    This is the chart that carries the project's central finding, so the fit is
    computed explicitly rather than left to a plotting library's defaults, and
    the band is a genuine confidence interval on the mean response.
    """
    data = df[["Discount", "Profit", "Category"]].dropna()
    if len(data) > sample:
        data = data.sample(sample, random_state=7)

    x = data["Discount"].to_numpy(dtype=float)
    y = data["Profit"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 120)
    fitted = slope * grid + intercept

    # 95% CI on the mean response:
    #   se(x0) = s * sqrt(1/n + (x0 - xbar)^2 / Sxx)
    n = len(x)
    residuals = y - (slope * x + intercept)
    s = float(np.sqrt((residuals**2).sum() / (n - 2)))
    x_bar = float(x.mean())
    sxx = float(((x - x_bar) ** 2).sum())
    se = s * np.sqrt(1 / n + (grid - x_bar) ** 2 / sxx)
    margin = 1.96 * se

    categories = sorted(data["Category"].unique())
    colours = theme.colour_for(categories)

    fig = go.Figure()

    # Band first, so the points and the line sit on top of it.
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([fitted + margin, (fitted - margin)[::-1]]),
            fill="toself",
            fillcolor="rgba(208,59,59,0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            name="95% CI",
            showlegend=True,
        )
    )

    for cat in categories:
        subset = data[data["Category"] == cat]
        fig.add_trace(
            go.Scatter(
                x=subset["Discount"],
                y=subset["Profit"],
                mode="markers",
                name=str(cat),
                marker=dict(
                    color=colours[cat],
                    size=8,
                    opacity=0.55,
                    line=dict(width=1, color=theme.ink["surface"]),  # 2px ring
                ),
                hovertemplate=(
                    f"<b>{cat}</b><br>Discount: %{{x:.0%}}"
                    "<br>Profit: $%{y:,.0f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=fitted,
            mode="lines",
            name=f"Fit: {slope:,.0f}·discount + {intercept:,.0f}",
            line=dict(color=theme.status["critical"], width=2),
            hovertemplate="Predicted profit: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line=dict(color=theme.ink["axis"], width=1, dash="dot"))

    fig.update_layout(
        template=theme.plotly,
        title="Deeper discounts destroy margin",
        xaxis_title="Discount applied",
        yaxis_title="Profit per order line (USD)",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickprefix="$"),
        height=460,
    )
    return fig


# ------------------------------------------------------------- 8. stacked bar
def segment_category_bars(df: pd.DataFrame, theme: Theme, metric: str = "Sales") -> go.Figure:
    """Metric by customer segment, stacked by product category (drill-down)."""
    grouped = (
        df.groupby(["Segment", "Category"], observed=True)[metric]
        .sum()
        .reset_index()
    )
    categories = sorted(grouped["Category"].unique())
    colours = theme.colour_for(categories)

    fig = go.Figure()
    for cat in categories:
        subset = grouped[grouped["Category"] == cat]
        fig.add_trace(
            go.Bar(
                x=subset["Segment"],
                y=subset[metric],
                name=str(cat),
                marker=dict(
                    color=colours[cat],
                    line=dict(color=theme.ink["surface"], width=2),  # 2px gap
                ),
                hovertemplate=(
                    f"<b>{cat}</b><br>%{{x}}<br>{metric}: $%{{y:,.0f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=theme.plotly,
        title=f"{metric} by customer segment and category",
        barmode="stack",
        xaxis_title="Customer segment",
        yaxis_title=f"{metric} (USD)",
        yaxis=dict(tickprefix="$"),
        height=440,
    )
    return fig


# The registry the dashboard iterates over.
CHART_REGISTRY = {
    "trend": ("Revenue & profit trend", trend_small_multiples),
    "map": ("Geographic distribution", state_choropleth),
    "correlation": ("Correlation matrix", correlation_matrix),
    "distribution": ("Profit distribution", profit_distribution),
    "sunburst": ("Category composition", category_sunburst),
    "animated": ("Animated year-over-year", animated_region_race),
    "scatter": ("Discount vs profit regression", discount_profit_scatter),
    "stacked": ("Segment & category breakdown", segment_category_bars),
}
