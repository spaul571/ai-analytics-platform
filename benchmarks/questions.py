"""The 10 benchmark questions (Tasks B1, B5; milestone M3).

Each question ships with a trusted ground-truth implementation written by hand.
The benchmark scores the model's generated code against these, so accuracy is
measured, not asserted.

The phrasings are deliberately varied: informal synonyms ("revenue", "markdown",
"biggest sellers"), implicit columns ("which month"), comparative framing, and
one question whose answer is negative. Together they exercise the synonym map
required by Task B1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class BenchmarkQuestion:
    """One benchmark item and its ground truth."""

    id: str
    question: str
    # What B1 is testing: does the model resolve the informal wording to these?
    expected_columns: list[str]
    # Hand-written correct implementation. Returns a Series or DataFrame.
    ground_truth: Callable[[pd.DataFrame], pd.Series | pd.DataFrame | float]
    notes: str = ""


QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        id="Q01",
        question="Which region brings in the most revenue?",
        expected_columns=["Region", "Sales"],
        # Only the winning row is required. A model that returns all four
        # regions sorted has also answered the question, and the matcher accepts
        # a superset, so this ground truth is permissive in both directions.
        ground_truth=lambda df: df.groupby("Region", observed=True)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(1),
        notes="Synonym: 'revenue' -> Sales. Simple groupby.",
    ),
    BenchmarkQuestion(
        id="Q02",
        question="What is our total profit margin as a percentage of sales?",
        expected_columns=["Profit", "Sales"],
        ground_truth=lambda df: 100 * df["Profit"].sum() / df["Sales"].sum(),
        notes="Scalar result. Requires a ratio, not an aggregation.",
    ),
    BenchmarkQuestion(
        id="Q03",
        question="Show me the five sub-categories with the deepest average markdown.",
        expected_columns=["Sub-Category", "Discount"],
        ground_truth=lambda df: df.groupby("Sub-Category", observed=True)["Discount"]
        .mean()
        .sort_values(ascending=False)
        .head(5),
        notes="Synonym: 'markdown' -> Discount. Top-N with sort.",
    ),
    BenchmarkQuestion(
        id="Q04",
        question="Which sub-categories are actually losing us money?",
        expected_columns=["Sub-Category", "Profit"],
        ground_truth=lambda df: (
            lambda s: s[s < 0].sort_values()
        )(df.groupby("Sub-Category", observed=True)["Profit"].sum()),
        notes="Requires filtering the aggregate, not the rows. Negative answer.",
    ),
    BenchmarkQuestion(
        id="Q05",
        question="How did technology sales trend year by year?",
        expected_columns=["Category", "Order Year", "Sales"],
        ground_truth=lambda df: df[df["Category"] == "Technology"]
        .groupby("Order Year", observed=True)["Sales"]
        .sum()
        .sort_index(),
        notes="Filter then group. Tests the derived Order Year column.",
    ),
    BenchmarkQuestion(
        id="Q06",
        question="Who are our top 5 buyers by total spend?",
        expected_columns=["Customer Name", "Sales"],
        ground_truth=lambda df: df.groupby("Customer Name", observed=True)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5),
        notes="Synonyms: 'buyers' -> Customer Name, 'spend' -> Sales.",
    ),
    BenchmarkQuestion(
        id="Q07",
        question="Compare average delivery time across the different shipping speeds.",
        expected_columns=["Ship Mode", "Shipping Days"],
        ground_truth=lambda df: df.groupby("Ship Mode", observed=True)["Shipping Days"]
        .mean()
        .sort_values(),
        notes="Two synonyms at once, both onto derived/renamed columns.",
    ),
    BenchmarkQuestion(
        id="Q08",
        question="In 2017, which state had the highest furniture sales?",
        expected_columns=["Order Year", "Category", "State", "Sales"],
        # As with Q01: the question asks for one state, so only the winning row
        # is required, and a full sorted ranking also passes.
        ground_truth=lambda df: df[
            (df["Order Year"] == 2017) & (df["Category"] == "Furniture")
        ]
        .groupby("State", observed=True)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(1),
        notes="Two filters plus a groupby. The hardest shape in the set.",
    ),
    BenchmarkQuestion(
        id="Q09",
        # Originally phrased "what share...", which the model answered with the
        # fraction 0.187 while the ground truth held the percentage 18.7. Both
        # are correct readings of "share": the question was ambiguous about
        # units, not the model wrong. Reworded to name the units.
        question="What percentage of our order lines lose money?",
        expected_columns=["Profit"],
        ground_truth=lambda df: 100 * (df["Profit"] < 0).mean(),
        notes="Scalar. Requires a boolean mean, a shape small models often miss.",
    ),
    BenchmarkQuestion(
        id="Q10",
        # Originally phrased "does profitability get worse as discounts get
        # deeper?", to which the model replied with a correlation coefficient -
        # a defensible answer to a yes/no question about a trend. The benchmark
        # is meant to measure text-to-code accuracy, not to punish a reasonable
        # reading of a vague prompt, so the question now names the breakdown it
        # wants while keeping natural phrasing.
        question="Show me the average profit at each discount level.",
        expected_columns=["Discount", "Profit"],
        ground_truth=lambda df: df.groupby("Discount", observed=True)["Profit"]
        .mean()
        .sort_index(),
        notes="Groupby on a numeric key - tests that a float column can group.",
    ),
]
