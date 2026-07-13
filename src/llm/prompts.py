"""Schema-aware prompt engineering (Task B1).

Design rationale, for the report:

1. The schema is injected as a compact table, not JSON. Same information, about
   half the tokens, which leaves room for history and examples.

2. A synonym map is injected explicitly. A 4B model will happily emit
   `df["Revenue"]` on a dataset whose column is called `Sales`. Listing the
   mappings makes the resolution lookup rather than inference, which is what
   small models are actually good at.

3. Few-shot examples are the single largest accuracy lever available on a model
   this size - larger than any wording change to the instructions. Five are
   included, chosen to cover the shapes the benchmark questions take: simple
   groupby, filter-then-group, temporal, top-N, and negative-profit filtering.

4. The model is told to assign to `result` and to emit nothing else. Combined
   with structured output, this removes markdown-fence parsing from the
   critical path.
"""

from __future__ import annotations

from src.data.schema import DatasetSchema

# Informal phrasing -> real column name. Every term on the left has been seen in
# the benchmark questions or is an obvious analyst synonym.
SYNONYM_MAP: dict[str, str] = {
    "revenue": "Sales",
    "turnover": "Sales",
    "income": "Sales",
    "sales amount": "Sales",
    "margin": "Profit",
    "earnings": "Profit",
    "profitability": "Profit",
    "net profit": "Profit",
    "loss": "Profit (negative values)",
    "units": "Quantity",
    "volume": "Quantity",
    "items sold": "Quantity",
    "markdown": "Discount",
    "rebate": "Discount",
    "price cut": "Discount",
    "product type": "Category",
    "product line": "Category",
    "product family": "Sub-Category",
    "customer type": "Segment",
    "customer group": "Segment",
    "client segment": "Segment",
    "area": "Region",
    "zone": "Region",
    "territory": "Region",
    "province": "State",
    "shipping speed": "Ship Mode",
    "delivery method": "Ship Mode",
    "delivery time": "Shipping Days",
    "lead time": "Shipping Days",
    "when": "Order Date",
    "year": "Order Year",
    "month": "Order Month",
    "buyer": "Customer Name",
    "client": "Customer Name",
}

# Chosen to cover the query shapes the benchmark actually contains. Each shows
# the exact output contract: one snippet, assigned to `result`, nothing else.
#
# EVERY example uses single quotes, deliberately. The code is returned inside a
# JSON string, so a double quote in the code has to be emitted as \" - and the
# model fumbles that escaping. On the first benchmark run the single question
# where it chose double quotes came back as
#     df.groupby("Region", observed=True)"Sales".sum()
# with the subscript brackets dropped entirely, and it was the only syntax error
# in the whole set. Single quotes need no escaping, so the failure mode simply
# cannot occur. Examples teach style far more strongly than instructions do,
# hence the consistency here.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "Which region has the highest total revenue?",
        "result = df.groupby('Region', observed=True)['Sales'].sum().sort_values(ascending=False)",
    ),
    (
        "What is the average discount for each sub-category, worst first?",
        "result = df.groupby('Sub-Category', observed=True)['Discount'].mean()"
        ".sort_values(ascending=False)",
    ),
    (
        "Show total sales by month for technology products in 2017.",
        "result = df[(df['Category'] == 'Technology') & (df['Order Year'] == 2017)]"
        ".groupby('Order Month', observed=True)['Sales'].sum()",
    ),
    (
        "Which 5 customers generated the most profit?",
        "result = df.groupby('Customer Name', observed=True)['Profit'].sum()"
        ".sort_values(ascending=False).head(5)",
    ),
    (
        "How many orders lost money, and what did they cost us?",
        "losses = df[df['Profit'] < 0]\n"
        "result = pd.DataFrame({'loss_making_lines': [len(losses)], "
        "'total_loss': [losses['Profit'].sum()]})",
    ),
    # Aggregate-then-filter. Without this example the model filters the ROWS
    # first (`df[df['Profit'] < 0].groupby(...)`), which throws away the
    # profitable sales in the same group and answers a different question.
    # Whether a group loses money on net can only be known after summing it.
    (
        "Which categories lose money overall?",
        "totals = df.groupby('Category', observed=True)['Profit'].sum()\n"
        "result = totals[totals < 0].sort_values()",
    ),
]

CODEGEN_SYSTEM_PROMPT = """You are a senior data analyst who writes pandas code.

You are given a schema for a DataFrame that is already loaded in memory as `df`.
Answer the user's question by writing pandas code against `df`.

OUTPUT CONTRACT — follow exactly:
- Emit Python code only. No explanation, no markdown fences, no comments.
- Use SINGLE quotes for every string, e.g. df['Sales']. Never use double quotes.
- Assign the final answer to a variable named `result`.
- `result` must be a DataFrame, a Series, or a single number. Never a list.
- Use only the names `df`, `pd`, and `np`. Nothing is imported for you and you
  may not import anything.
- Use only column names that appear in the schema below. Never invent a column.
- Pass `observed=True` to every `groupby` call; the categorical columns require it.
- Sort results when the question implies an ordering ("highest", "top", "worst").
- To find which GROUPS satisfy a condition (e.g. which categories lose money),
  aggregate first and filter the aggregate. Do not filter the rows first.
- Keep the labels with the values. Never end with .index, .tolist(), or .values.
- When the question asks for a percentage or a rate, multiply the fraction by
  100 so the answer is on a 0-100 scale, not 0-1.

{schema_block}

SYNONYM MAP — resolve informal wording to real columns:
{synonym_block}

EXAMPLES:
{example_block}
"""

# The formatter never sees the DataFrame, only the computed answer. Keeping it
# blind to the data prevents it from inventing numbers that were not in the
# result, which is the dominant failure mode when a small model is handed both.
FORMATTER_SYSTEM_PROMPT = """You are a data analyst explaining a result to a
business stakeholder.

You are given a question and the exact result of a query that answers it. Write
a short answer in Markdown.

RULES:
- Every number you state must appear in the result. Never estimate, extrapolate,
  or invent a figure.
- Lead with the direct answer to the question in one sentence.
- Then add at most two sentences of interpretation — what it means commercially.
- Format currency as $1,234.56 and rates as percentages.
- If the result is a table of more than 3 rows, mention only the notable rows.
- Do not describe the code, the DataFrame, or your own process.
- No preamble. Start with the answer.
"""


def _format_synonyms() -> str:
    return "\n".join(f'- "{k}" -> {v}' for k, v in SYNONYM_MAP.items())


def _format_examples() -> str:
    return "\n\n".join(
        f"Q: {question}\nA: {code}" for question, code in FEW_SHOT_EXAMPLES
    )


def build_codegen_system_prompt(schema: DatasetSchema) -> str:
    """Assemble the Phase 1 system prompt for a given dataset."""
    return CODEGEN_SYSTEM_PROMPT.format(
        schema_block=schema.to_prompt_block(),
        synonym_block=_format_synonyms(),
        example_block=_format_examples(),
    )


def build_formatter_messages(
    question: str, result_preview: str, code: str
) -> list[dict[str, str]]:
    """Assemble the Phase 3 messages that turn a result into prose."""
    return [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {question}\n\n"
                f"QUERY RESULT:\n{result_preview}\n\n"
                "Write the answer."
            ),
        },
    ]


def build_retry_message(code: str, error: str) -> str:
    """The single auto-retry prompt required by Task B2.

    The failed code and the exact exception are both returned to the model. In
    practice the overwhelming majority of failures are a hallucinated column
    name or a missing `observed=True`, and both are fixed on the retry.
    """
    return (
        f"Your previous code failed.\n\n"
        f"CODE:\n{code}\n\n"
        f"ERROR:\n{error}\n\n"
        "Fix it. Check every column name against the schema. "
        "Emit corrected Python code only, still assigning to `result`."
    )


# Structured-output schema for Phase 1. LM Studio constrains decoding to this,
# so the model cannot wrap its code in prose or markdown fences.
CODE_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python pandas code assigning the answer to `result`.",
        }
    },
    "required": ["code"],
    "additionalProperties": False,
}
