"""Dataset ingestion and cleaning (Task A1 + A3 cleaning steps).

Loads the Superstore e-commerce CSV into an in-memory pandas DataFrame backed
by PyArrow dtypes, applies the cleaning steps, and returns both the frame and
its extracted schema.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import DATA
from src.data.schema import DatasetSchema, extract_schema

BUSINESS_CONTEXT = (
    "Retail e-commerce order transactions. Each row is one product line item "
    "within a customer order. Sales is gross revenue in USD; Profit is net "
    "margin in USD and can be negative when discounts are too deep."
)

# Column notes handed to the LLM. These resolve the ambiguities a small model
# would otherwise guess wrong on (e.g. Sales vs Profit, Region vs Country).
COLUMN_DESCRIPTIONS = {
    "Order ID": "Unique order identifier; one order can contain several rows.",
    "Order Date": "Date the customer placed the order.",
    "Ship Date": "Date the order shipped.",
    "Ship Mode": "Delivery speed chosen by the customer.",
    "Customer ID": "Unique customer identifier.",
    "Customer Name": "Customer full name.",
    "Segment": "Customer type: Consumer, Corporate, or Home Office.",
    "Country": "Country of the shipping address.",
    "City": "City of the shipping address.",
    "State": "State or province of the shipping address.",
    "Postal Code": "Postal code of the shipping address.",
    "Region": "Sales region grouping several states.",
    "Product ID": "Unique product identifier.",
    "Category": "Top-level product category.",
    "Sub-Category": "Product sub-category nested inside Category.",
    "Product Name": "Product title.",
    "Sales": "Gross revenue for the line item, in USD.",
    "Quantity": "Units sold in the line item.",
    "Discount": "Discount rate applied, as a fraction from 0.0 to 1.0.",
    "Profit": "Net profit for the line item, in USD. Negative means a loss.",
}


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply cleaning steps and report what was done.

    Returns the cleaned frame and a list of human-readable step descriptions
    for the data quality card (Task A3) and the written report.
    """
    steps: list[str] = []

    # Step 1 - date parsing. The source CSV stores dates as MM/DD/YYYY strings,
    # which makes every temporal query impossible until converted.
    for col in DATA.date_columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)
            steps.append(f"Parsed '{col}' from string to datetime64.")

    # Step 2 - category standardisation. Stray casing and whitespace in the
    # categorical columns fragment groupby results into duplicate buckets.
    categorical = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
    for col in categorical:
        if col in df.columns:
            before = df[col].nunique()
            df[col] = df[col].astype(str).str.strip().str.title()
            after = df[col].nunique()
            if before != after:
                steps.append(
                    f"Standardised '{col}': {before} -> {after} distinct values "
                    "after trimming whitespace and normalising case."
                )

    # Step 3 - exact duplicate removal.
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        steps.append(f"Dropped {duplicates} exact duplicate rows.")

    # Step 4 - derived columns the LLM will need for temporal questions.
    # Without these, every "sales by month" question forces the model to emit a
    # .dt accessor chain, which a 4B model gets wrong far more often than it
    # gets a plain groupby right.
    if "Order Date" in df.columns:
        df["Order Year"] = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
        steps.append("Derived 'Order Year' and 'Order Month' from Order Date.")

    if {"Ship Date", "Order Date"} <= set(df.columns):
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
        steps.append("Derived 'Shipping Days' as Ship Date minus Order Date.")

    # Step 5 - downcast low-cardinality text columns to categorical. This is the
    # memory/time trade-off Task A4 asks us to document: categoricals cut the
    # frame's footprint substantially at the cost of a one-off conversion pass,
    # and they also speed up every groupby that follows.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique(dropna=True) < 0.5 * len(df):
            df[col] = df[col].astype("category")
    steps.append("Downcast low-cardinality text columns to pandas 'category' dtype.")

    return df, steps


def load_dataset(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, DatasetSchema, dict]:
    """Load, clean, and profile the dataset.

    Args:
        csv_path: Override for the configured CSV location.

    Returns:
        (dataframe, schema, load_metadata) where load_metadata carries the
        timings and cleaning steps needed for the Task A performance report.
    """
    path = csv_path or DATA.csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See README.md for the download link."
        )

    start = time.perf_counter()
    df = pd.read_csv(path, encoding="latin-1")
    load_seconds = time.perf_counter() - start
    memory_before_mb = df.memory_usage(deep=True).sum() / 1_048_576

    start = time.perf_counter()
    df, cleaning_steps = _clean(df)
    clean_seconds = time.perf_counter() - start

    schema = extract_schema(
        df,
        name="Global E-Commerce Sales (Superstore)",
        business_context=BUSINESS_CONTEXT,
        descriptions=COLUMN_DESCRIPTIONS,
    )

    memory_after_mb = df.memory_usage(deep=True).sum() / 1_048_576

    metadata = {
        "path": str(path),
        "load_seconds": round(load_seconds, 4),
        "clean_seconds": round(clean_seconds, 4),
        "memory_raw_mb": round(memory_before_mb, 2),
        "memory_mb": round(memory_after_mb, 2),
        "memory_saved_pct": round(
            100 * (1 - memory_after_mb / memory_before_mb) if memory_before_mb else 0.0, 1
        ),
        "rows": len(df),
        "columns": len(df.columns),
        "cleaning_steps": cleaning_steps,
    }

    return df, schema, metadata
