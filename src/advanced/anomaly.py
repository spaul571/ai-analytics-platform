"""Anomaly detection (Task D3).

Three detectors over the same data, kept separate rather than merged into one
score, so that they can be compared:

    Isolation Forest  - multivariate, ranked. Scores every row on how easy it is
                        to isolate, then the worst `contamination` share is cut.
    IQR               - univariate, per column. The classic 1.5x rule.
    Z-score           - univariate, per column. Assumes normality, which Sales
                        and Profit badly violate; reported for comparison.

WHAT THE COMPARISON ACTUALLY SHOWED (and it is not what we expected)
-------------------------------------------------------------------
The intuition for reaching for a forest is that it catches rows whose COMBINATION
of values is odd while no single column is extreme. On this dataset that turns
out to be false. Measured across a contamination sweep:

    contamination   flagged   caught by forest but NOT by IQR
            1%          100                 0
            2%          200                 0
            5%          500                 1
           10%        1,000                22

At any operationally sensible threshold the forest finds nothing the univariate
rules miss. The honest conclusion is that its value here is not novelty.

Its value is SELECTIVITY. The union of the per-column IQR rules flags 2,851 rows
- 28.5% of the dataset. "Roughly a third of your orders are unusual" is not an
alert list anyone can act on; it is noise with a threshold attached. The forest
returns a bounded, RANKED set (200 rows at 2%) with a continuous score, so the
worst row can be put first and a human can start at the top. That ranking is the
thing IQR cannot give at all, and it is why the forest is the detector driving
the UI while IQR and z-score are reported alongside it as context.

This is worth stating plainly rather than quietly picking the algorithm that
sounds most advanced.

The LLM never decides what is anomalous. It narrates rows the detectors have
already flagged, in business terms. A model that could invent anomalies would be
a model that could invent a crisis that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import DATA, LLM
from src.llm.client import LLMClient, LLMError

# Columns the forest sees. Row ID and Postal Code are numeric but carry no
# behavioural signal, so including them would only add noise.
FEATURES = ["Sales", "Quantity", "Discount", "Profit", "Shipping Days"]

# Share of rows the forest is told to treat as anomalous. 2% of ~10k rows gives
# roughly 200 flags - few enough for a human to review, and it is a stated
# assumption rather than a discovered truth. This is the single most important
# knob and it is worth being honest about that in the report: the forest does
# not find "the anomalies", it ranks rows and we cut the top 2%.
CONTAMINATION = 0.02

Z_THRESHOLD = 3.0

NARRATIVE_SYSTEM_PROMPT = """You are a retail operations analyst.

You are given order line items that an anomaly detector has flagged, with their
figures. For EACH one, write a single sentence explaining in business terms why
it is unusual and what it likely means commercially.

RULES:
- One bullet per flagged row, in the order given.
- Start each bullet with the Sub-Category and the Order ID.
- Cite only numbers present in the row.
- Say what went wrong commercially (e.g. a discount deeper than the margin can
  absorb), not what the statistics say.
- No preamble, no summary, no restating these instructions.
"""


@dataclass
class AnomalyReport:
    """Flagged rows plus the agreement between the three detectors."""

    flagged: pd.DataFrame                    # rows the forest flagged, worst first
    scored: pd.DataFrame                     # every row, with its forest score
    iqr_counts: dict[str, int] = field(default_factory=dict)
    zscore_counts: dict[str, int] = field(default_factory=dict)
    forest_only: int = 0                     # caught by the forest, missed by IQR
    iqr_only: int = 0                        # caught by IQR, missed by the forest
    both: int = 0
    iqr_flagged_total: int = 0               # the union of the per-column IQR rules
    baseline_loss_rate: float = 0.0          # share of ALL rows that lose money
    narrative: str = ""
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def total_loss(self) -> float:
        """Money lost across the flagged rows. The number an executive wants."""
        if self.flagged.empty:
            return 0.0
        losses = self.flagged.loc[self.flagged["Profit"] < 0, "Profit"]
        return float(losses.sum())

    @property
    def flagged_loss_rate(self) -> float:
        """Share of flagged rows that actually lose money."""
        if self.flagged.empty:
            return 0.0
        return float((self.flagged["Profit"] < 0).mean())

    @property
    def enrichment(self) -> float:
        """How much likelier a flagged row is to be loss-making than a random one.

        This is the number that answers "is the detector any good?". A value of
        1.0 would mean it is picking rows at random.
        """
        if not self.baseline_loss_rate:
            return 0.0
        return self.flagged_loss_rate / self.baseline_loss_rate

    @property
    def selectivity(self) -> str:
        """One line contrasting the forest's alert list with IQR's."""
        total = len(self.scored)
        if not total:
            return ""
        return (
            f"Isolation Forest flags {self.flagged_count} rows "
            f"({100 * self.flagged_count / total:.1f}% of the data), ranked worst "
            f"first. The union of the per-column IQR rules flags "
            f"{self.iqr_flagged_total:,} ({100 * self.iqr_flagged_total / total:.1f}%), "
            "unranked — too many to action."
        )


def _iqr_mask(series: pd.Series, multiplier: float = DATA.iqr_multiplier) -> pd.Series:
    clean = series.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (series < low) | (series > high)


def _zscore_mask(series: pd.Series, threshold: float = Z_THRESHOLD) -> pd.Series:
    std = series.std()
    if not std or np.isnan(std):
        return pd.Series(False, index=series.index)
    return ((series - series.mean()).abs() / std) > threshold


def detect(df: pd.DataFrame, contamination: float = CONTAMINATION) -> AnomalyReport:
    """Run all three detectors and compare them.

    Args:
        df: The (optionally filtered) dataset.
        contamination: Share of rows the Isolation Forest treats as anomalous.

    Returns:
        An AnomalyReport. The narrative is left empty; call `narrate` to fill it.
    """
    features = [c for c in FEATURES if c in df.columns]
    working = df.dropna(subset=features).copy()

    # Standardise first. The forest partitions on raw values, and Sales spans
    # 0.4 to 22,638 while Discount spans 0 to 0.8 - without scaling, Sales would
    # dominate every split and Discount would be invisible to the model.
    scaled = StandardScaler().fit_transform(working[features])

    forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=7,   # reproducible: the demo must flag the same rows twice
        n_jobs=-1,
    )
    working["anomaly_score"] = forest.fit(scaled).score_samples(scaled)
    working["is_anomaly"] = forest.predict(scaled) == -1

    # Lower score = more anomalous, so ascending sort puts the worst first.
    flagged = (
        working[working["is_anomaly"]]
        .sort_values("anomaly_score")
        .loc[
            :,
            [
                "Order ID", "Order Date", "Region", "State", "Category",
                "Sub-Category", "Sales", "Quantity", "Discount", "Profit",
                "anomaly_score",
            ],
        ]
        .reset_index(drop=True)
    )

    iqr_any = pd.Series(False, index=working.index)
    iqr_counts: dict[str, int] = {}
    zscore_counts: dict[str, int] = {}
    for col in features:
        col_iqr = _iqr_mask(working[col])
        iqr_counts[col] = int(col_iqr.sum())
        zscore_counts[col] = int(_zscore_mask(working[col]).sum())
        iqr_any |= col_iqr

    forest_any = working["is_anomaly"]

    return AnomalyReport(
        flagged=flagged,
        scored=working,
        iqr_counts=iqr_counts,
        zscore_counts=zscore_counts,
        forest_only=int((forest_any & ~iqr_any).sum()),
        iqr_only=int((iqr_any & ~forest_any).sum()),
        both=int((forest_any & iqr_any).sum()),
        iqr_flagged_total=int(iqr_any.sum()),
        baseline_loss_rate=float((working["Profit"] < 0).mean()),
    )


def narrate(
    report: AnomalyReport, top_n: int = 5, client: LLMClient | None = None
) -> AnomalyReport:
    """Ask the LLM to explain the worst flagged rows in business terms.

    Mutates and returns the report. On LLM failure the flagged rows survive with
    an error note: the detection is a fact, the narration is a convenience.
    """
    if report.flagged.empty:
        report.narrative = "No anomalies were flagged in the current selection."
        return report

    worst = report.flagged.head(top_n)
    rows = worst[
        ["Order ID", "Sub-Category", "Sales", "Quantity", "Discount", "Profit"]
    ].to_string(index=False)

    client = client or LLMClient()
    try:
        response = client.complete(
            [
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{len(report.flagged)} order lines were flagged as "
                        f"anomalous out of {len(report.scored):,}. "
                        f"The {len(worst)} most extreme:\n\n{rows}\n\n"
                        "Explain each one."
                    ),
                },
            ],
            temperature=LLM.narrative_temperature,
            max_tokens=700,
        )
        report.narrative = response.text
        report.elapsed_seconds = response.elapsed_seconds
    except LLMError as exc:
        report.error = str(exc)
        report.narrative = (
            f"*(Explanations unavailable: {exc}. The flagged rows below are still "
            "valid — detection does not depend on the model.)*"
        )

    return report
