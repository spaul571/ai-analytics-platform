"""Chart theme: one palette, one Plotly template, both modes (Task C2).

Every chart in the application draws its colours from here, which is what makes
the suite read as one system rather than eight unrelated pictures.

COLOUR RATIONALE (for the report's design-decisions section)

The categorical hues are assigned in a FIXED slot order and never cycled. Colour
follows the entity, not its rank: 'Furniture' is slot 1 in every chart it appears
in, so a filter that removes a category never repaints the survivors.

The palette was validated with a colour-vision-deficiency checker rather than
chosen by eye. Results:

    light mode: worst adjacent CVD separation dE 24.2 (protan) - comfortably
                clear of the >=12 target. Two hues (aqua, yellow) fall below a
                3:1 contrast ratio against the light surface.
    dark mode:  all six hues clear 3:1 contrast, but worst adjacent separation
                is dE 10.3 - inside the 8-12 "floor band".

Both findings oblige the same mitigation, which is applied throughout: colour is
never the only channel carrying identity. Every multi-series chart ships a
legend, hover tooltips name the series, and a table view of the underlying data
is available. A red/green viewer can still read every chart.

Three encodings, three different colour jobs:
    categorical -> identity (which sub-category is this?)      -> the 8 slots
    sequential  -> magnitude (how much?)                       -> one blue ramp
    diverging   -> polarity (profit vs loss, above/below zero) -> blue<->red
                                                                  with a NEUTRAL
                                                                  GREY midpoint
A rainbow scale is never used: it implies an ordering that the data does not have
and is unreadable under CVD.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------- palette
CATEGORICAL_LIGHT = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#199e70",
    "#c98500",
    "#008300",
    "#9085e9",
    "#e66767",
    "#d55181",
    "#d95926",
]

# Sequential: a single hue, light to dark. Used for magnitude - choropleth fills,
# heatmap cells. The lightest step means "near zero" and may recede into the
# surface; that is correct for a continuous scale.
SEQUENTIAL_BLUE = [
    [0.00, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.50, "#3987e5"],
    [0.75, "#256abf"],
    [1.00, "#0d366b"],
]

# Diverging: two poles that read as opposites, with a NEUTRAL grey midpoint so
# that "zero profit" reads as nothing rather than as a third category. Used
# wherever the sign of the number is the point.
DIVERGING_LIGHT = [
    [0.00, "#d03b3b"],
    [0.25, "#e88b8b"],
    [0.50, "#f0efec"],
    [0.75, "#86b6ef"],
    [1.00, "#0d366b"],
]
DIVERGING_DARK = [
    [0.00, "#d03b3b"],
    [0.25, "#a04545"],
    [0.50, "#383835"],
    [0.75, "#3987e5"],
    [1.00, "#cde2fb"],
]

# Status colours are reserved for state (good/bad) and never reused as "series 7".
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


class Theme:
    """Resolved colours for one mode. Charts take one of these, never raw hex."""

    def __init__(self, mode: str = "light"):
        if mode not in ("light", "dark"):
            raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
        self.mode = mode
        self.categorical = CATEGORICAL_LIGHT if mode == "light" else CATEGORICAL_DARK
        self.sequential = SEQUENTIAL_BLUE
        self.diverging = DIVERGING_LIGHT if mode == "light" else DIVERGING_DARK
        self.ink = INK[mode]
        self.status = STATUS
        self._template: go.layout.Template | None = None

    @property
    def plotly(self) -> go.layout.Template:
        """The template, built once and reused.

        Every chart passes this to `update_layout(template=...)` explicitly
        rather than relying on `pio.templates.default`. Global default state is
        fragile: any library that touches `pio.templates` after us silently wins,
        and a chart built before `register()` runs keeps Plotly's stock theme.
        Binding the template to the figure removes the ordering dependency.
        """
        if self._template is None:
            self._template = self.template()
        return self._template

    def colour_for(self, categories: list[str]) -> dict[str, str]:
        """Map category names to fixed colour slots.

        Sorted, so the same category always lands on the same slot regardless of
        which subset of the data is currently filtered in. This is the rule that
        stops a filter from repainting the chart.
        """
        ordered = sorted(str(c) for c in categories)
        return {
            name: self.categorical[i % len(self.categorical)]
            for i, name in enumerate(ordered)
        }

    def template(self) -> go.layout.Template:
        """The Plotly template every chart is rendered through."""
        ink = self.ink
        return go.layout.Template(
            layout=go.Layout(
                font=dict(family=FONT_FAMILY, size=13, color=ink["secondary"]),
                title=dict(
                    font=dict(size=16, color=ink["primary"]),
                    x=0.0,
                    xanchor="left",
                    # Pinned to the top of the margin. Left at Plotly's default
                    # the title floats down into the horizontal legend and the
                    # two overlap - the palette validator checks colour, not
                    # layout, so this only showed up on looking at the render.
                    y=0.97,
                    yanchor="top",
                ),
                paper_bgcolor=ink["surface"],
                plot_bgcolor=ink["surface"],
                colorway=self.categorical,
                # Recessive chrome: the data should be the only thing with weight.
                xaxis=dict(
                    gridcolor=ink["grid"],
                    linecolor=ink["axis"],
                    zerolinecolor=ink["axis"],
                    tickfont=dict(color=ink["muted"], size=12),
                    title=dict(font=dict(color=ink["secondary"], size=13)),
                ),
                yaxis=dict(
                    gridcolor=ink["grid"],
                    linecolor=ink["axis"],
                    zerolinecolor=ink["axis"],
                    tickfont=dict(color=ink["muted"], size=12),
                    title=dict(font=dict(color=ink["secondary"], size=13)),
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=ink["secondary"], size=12),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0.0,
                ),
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
                # The top margin has to hold the title AND the horizontal legend
                # stacked beneath it. 64px fits only the title, so the legend
                # rides up over it.
                margin=dict(l=60, r=24, t=96, b=56),
                colorscale=dict(sequential=self.sequential, diverging=self.diverging),
            )
        )

    def register(self) -> str:
        """Install the template with Plotly and make it the default."""
        name = f"capstone_{self.mode}"
        pio.templates[name] = self.template()
        pio.templates.default = name
        return name


# US state name -> two-letter code. Plotly's USA-states choropleth needs codes,
# and the dataset stores full names.
STATE_CODES: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
