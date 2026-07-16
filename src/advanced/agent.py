"""ReAct-style tool-calling agent (Task D4).

The agent decides, per turn, whether to query the data, detect anomalies, draw a
chart, or answer. It observes the result of each tool call and decides again.
The full reasoning chain is recorded and surfaced in a debug panel.

WHY AN AGENT RATHER THAN THE TASK B PIPELINE
--------------------------------------------
The Task B pipeline is a fixed three-phase sequence: generate code, run it,
narrate. That is exactly right for "which region sold most", and it cannot answer
"find the anomalies and tell me if they are concentrated in one region", because
answering that needs two dependent steps where the second depends on the output
of the first. The agent trades determinism for the ability to compose.

WHY THE TOOLS ARE STRUCTURED, NOT CODE
--------------------------------------
Task B's tool is "write arbitrary pandas", which is powerful and needs a sandbox
to be safe. The agent's tools take typed arguments (a column name, an
aggregation, a filter) and the executor - not the model - builds the pandas call.
The model therefore cannot express anything outside the tool schema, which means
this surface needs no sandbox at all: it is safe by construction rather than by
containment. That is a real design distinction and the two approaches sit
side by side in this project on purpose.

THE ONE TOOL THAT LEAVES THE MACHINE
------------------------------------
`get_holidays` calls a public API, because the dataset cannot say which of its
1,237 order days were public holidays and "do sales move around holidays?" is a
fair question to ask of retail data. Network access is where agents usually
acquire their worst failure mode - a tool returns attacker-controlled prose, the
prose enters the context, and the model follows it - so the same principle is
applied rather than abandoned: the model supplies a year, the executor builds the
URL from constants, and only an ISO date and a truncated name come back. There is
no free-text field in the response for anyone to write instructions into. See
src/data/external.py for the full guard list. A web-search tool would have been
more impressive and would have handed the model a paragraph of someone else's
text; that is the trade being refused here.

`compare_dates` is the other half. An external fact is worth nothing until it
touches the data, so it takes dates from the model and does the join in the
executor: per-day metric on those dates against every other day. Fetch, then
analyse, then answer - a genuinely dependent chain the Task B pipeline cannot
express.

KNOWN LIMITATION (state it in the report)
-----------------------------------------
Gemma 4 E4B is a 4B model. It plans two or three steps competently and then
starts to loop, re-calling the same tool with the same arguments. MAX_STEPS caps
the loop, and repeated identical calls are detected and short-circuited. A larger
model would not need either guard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.config import LLM
from src.data.external import MAX_YEAR, MIN_YEAR, ExternalDataError, fetch_holidays
from src.data.query import Aggregation, Filter, aggregate
from src.data.schema import DatasetSchema
from src.llm.client import LLMClient, LLMError

# The model gets this many turns to reach an answer before we stop it. Three
# tool calls plus a final answer covers every question the tools can express;
# beyond that the model is looping, not thinking.
MAX_STEPS = 6

# compare_dates takes its dates from the model, so both are capped: a year of
# holidays is ~16 dates, and a window wide enough to swallow the whole calendar
# would make the comparison meaningless rather than wrong.
_MAX_COMPARE_DATES = 40
_MAX_COMPARE_WINDOW = 14

StepKind = Literal["thought", "action", "observation", "answer", "error"]


@dataclass
class Step:
    """One entry in the reasoning chain, for the debug panel."""

    kind: StepKind
    content: str
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class AgentResult:
    """What the agent concluded, and how it got there."""

    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart_type: str | None = None
    total_seconds: float = 0.0
    tool_calls: int = 0
    success: bool = True
    stopped_early: bool = False


# ------------------------------------------------------------------- schemas
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Aggregate the dataset. Use this to compute totals, averages or "
                "counts of a metric, optionally grouped by a column and "
                "optionally filtered first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["Sales", "Profit", "Quantity", "Discount", "Shipping Days"],
                        "description": "The numeric column to aggregate.",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max"],
                    },
                    "group_by": {
                        "type": "string",
                        "enum": [
                            "Region", "State", "Category", "Sub-Category",
                            "Segment", "Ship Mode", "Order Year", "Order Month",
                            "Customer Name", "none",
                        ],
                        "description": "Column to group by, or 'none' for a whole-table total.",
                    },
                    "filter_column": {
                        "type": "string",
                        "enum": [
                            "Region", "Category", "Sub-Category", "Segment",
                            "Order Year", "none",
                        ],
                        "description": "Optional column to filter on before aggregating.",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Value the filter column must equal.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Keep only the top N rows after sorting. 0 for all.",
                    },
                },
                "required": ["metric", "aggregation", "group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": (
                "Run Isolation Forest anomaly detection over the order lines and "
                "return the most anomalous ones. Use this when the user asks "
                "about unusual, suspicious, or problem orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many of the worst anomalies to return (1-20).",
                    }
                },
                "required": ["top_n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_holidays",
            "description": (
                "Look up the US public holidays for one year from an external "
                "calendar API. The dataset does not know which days were "
                "holidays. Use this before comparing sales around holidays, then "
                "pass the dates it returns to compare_dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": f"Year between {MIN_YEAR} and {MAX_YEAR}.",
                    }
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_dates",
            "description": (
                "Compare a metric on specific calendar dates against every other "
                "day in the data. Use this with the dates returned by "
                "get_holidays to measure whether those days differ from normal. "
                "A window widens each date into a range around it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ISO dates, e.g. ['2016-11-24', '2016-12-25'].",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["Sales", "Profit", "Quantity", "Discount"],
                    },
                    "window_days": {
                        "type": "integer",
                        "description": (
                            "Days either side of each date to include. 0 is the "
                            "day itself; 3 covers the surrounding week."
                        ),
                    },
                },
                "required": ["dates", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Draw the result of the previous query_data call as a chart. Call "
                "this only after query_data has returned rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "map", "scatter"],
                    }
                },
                "required": ["chart_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Give the final answer to the user. Call this when you have "
                "enough information. This ends the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The answer in Markdown. Cite only numbers you "
                            "actually observed from a tool."
                        ),
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data analyst agent with tools.

Answer the user's question by calling tools. Work in small steps: call one tool,
look at what it returns, then decide the next step.

RULES:
- Never state a number you have not seen in a tool result. If you need a figure,
  call a tool to get it.
- Call `final_answer` as soon as you can answer. Do not keep querying.
- Do not call the same tool twice with the same arguments.
- `create_chart` only works after `query_data` has returned rows.
- Write money as $1,234.56 — a single dollar sign BEFORE the number. Never wrap a
  number in dollar signs on both sides ($200$), and never use LaTeX math
  delimiters: the interface renders them as equations.

{schema_block}
"""

# The model emits LaTeX-style $...$ around figures regardless of the instruction
# above - it is a strong habit from its training data. Streamlit's Markdown
# renders $...$ as math, so a dollar amount silently becomes an italic equation.
# The prompt reduces the frequency; this regex removes what survives.
_MATH_WRAPPED_NUMBER = re.compile(r"\$(-?[\d,]+(?:\.\d+)?)\$")


def _strip_math_delimiters(text: str) -> str:
    """Turn `$200$` and `$-74,142$` into `$200` and `-$74,142`."""

    def replace(match: re.Match) -> str:
        number = match.group(1)
        if number.startswith("-"):
            return f"-${number[1:]}"
        return f"${number}"

    return _MATH_WRAPPED_NUMBER.sub(replace, text)


# ------------------------------------------------------------------ executor
class ToolExecutor:
    """Runs the tools. The model chooses; this class decides how it happens.

    Because the executor builds every pandas call from typed arguments, the model
    cannot express anything the tool schema does not allow. No sandbox is needed
    on this path - unlike Task B, where the model writes the code itself.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last_result: pd.DataFrame = pd.DataFrame()
        self.last_chart: str | None = None

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one tool call and return an observation string for the model."""
        try:
            if name == "query_data":
                return self._query(arguments)
            if name == "detect_anomalies":
                return self._anomalies(arguments)
            if name == "get_holidays":
                return self._holidays(arguments)
            if name == "compare_dates":
                return self._compare_dates(arguments)
            if name == "create_chart":
                return self._chart(arguments)
            return f"ERROR: unknown tool {name!r}."
        except Exception as exc:  # noqa: BLE001 - observations are fed back, not raised
            return f"ERROR: {type(exc).__name__}: {exc}"

    def _query(self, args: dict[str, Any]) -> str:
        metric: str = args.get("metric", "Sales")
        agg: Aggregation = args.get("aggregation", "sum")
        group_by: str = args.get("group_by", "none")
        top_n = int(args.get("top_n") or 0)

        filters: list[Filter] = []
        filter_column = args.get("filter_column", "none")
        filter_value = args.get("filter_value")
        if filter_column and filter_column != "none" and filter_value not in (None, ""):
            value: Any = filter_value
            if filter_column == "Order Year":
                value = int(filter_value)
            filters.append(Filter(filter_column, "eq", value))

        groups = [] if group_by in ("none", "", None) else [group_by]
        sort_column = f"{metric}_{agg}"

        result = aggregate(
            self.df,
            group_by=groups,
            metrics={metric: [agg]},
            filters=filters or None,
            sort_by=sort_column if groups else None,
            ascending=False,
            limit=top_n or None,
        )
        self.last_result = result.data

        if result.data.empty:
            return "The query returned no rows. The filter may be too narrow."

        preview = result.data.head(15).to_string(index=False)
        note = ""
        if len(result.data) > 15:
            note = f"\n... ({len(result.data) - 15} more rows)"
        return f"{len(result.data)} rows in {result.execution_ms:.0f}ms:\n{preview}{note}"

    def _anomalies(self, args: dict[str, Any]) -> str:
        # Imported here rather than at module scope: sklearn takes ~1s to import
        # and the agent is constructed on every Streamlit rerun.
        from src.advanced.anomaly import detect

        top_n = max(1, min(int(args.get("top_n") or 5), 20))
        report = detect(self.df)

        worst = report.flagged.head(top_n)[
            ["Order ID", "Sub-Category", "Sales", "Discount", "Profit"]
        ]
        self.last_result = worst

        return (
            f"Isolation Forest flagged {report.flagged_count} of "
            f"{len(report.scored):,} order lines "
            f"(total loss on flagged rows: ${report.total_loss:,.0f}).\n"
            f"The {len(worst)} worst:\n{worst.to_string(index=False)}"
        )

    def _holidays(self, args: dict[str, Any]) -> str:
        """The one tool that leaves the machine. Failure is an observation, not a crash."""
        try:
            holidays = fetch_holidays(int(args.get("year") or 0))
        except ExternalDataError as exc:
            # Handed back as an observation so the model can carry on with the
            # local tools. A dead API must not end the run.
            return f"ERROR: {exc}"

        listing = "\n".join(f"{h.day.isoformat()}  {h.name}" for h in holidays)
        return (
            f"{len(holidays)} US public holidays in {args.get('year')} "
            f"(from date.nager.at):\n{listing}\n"
            "Pass these dates to compare_dates to see how sales behaved around them."
        )

    def _compare_dates(self, args: dict[str, Any]) -> str:
        """Compare named days against every other day. The model supplies dates only."""
        metric: str = args.get("metric", "Sales")
        if metric not in ("Sales", "Profit", "Quantity", "Discount"):
            return f"ERROR: {metric!r} is not a metric this tool can compare."

        raw_dates = args.get("dates") or []
        if isinstance(raw_dates, str):  # small models sometimes send one string
            raw_dates = [raw_dates]

        days: list[pd.Timestamp] = []
        for value in raw_dates[:_MAX_COMPARE_DATES]:
            try:
                days.append(pd.Timestamp(str(value).strip()).normalize())
            except ValueError:
                continue  # skip what we cannot parse rather than fail the call
        if not days:
            return (
                "ERROR: no usable dates. Pass ISO dates like ['2016-11-24'], "
                "for example the ones get_holidays returned."
            )

        window = max(0, min(int(args.get("window_days") or 0), _MAX_COMPARE_WINDOW))
        order_days = self.df["Order Date"].dt.normalize()

        selected = pd.Series(False, index=self.df.index)
        for day in days:
            span = pd.Timedelta(days=window)
            selected |= order_days.between(day - span, day + span)

        inside = self.df.loc[selected, metric]
        outside = self.df.loc[~selected, metric]
        if inside.empty:
            return (
                f"No orders fall on those dates (within {window} days). The dataset runs "
                "2014-01-03 to 2017-12-30."
            )

        # Per-day means, not totals: the two groups differ hugely in size, so
        # totals would only restate that there are more ordinary days than
        # holidays. The per-day rate is the comparison that carries meaning.
        inside_days = order_days[selected].nunique()
        outside_days = order_days[~selected].nunique()
        inside_rate = inside.sum() / inside_days if inside_days else 0.0
        outside_rate = outside.sum() / outside_days if outside_days else 0.0
        delta = ((inside_rate / outside_rate) - 1) * 100 if outside_rate else 0.0

        self.last_result = pd.DataFrame(
            {
                "Period": [f"On the {len(days)} dates (within {window}d)", "All other days"],
                f"{metric} per day": [round(inside_rate, 2), round(outside_rate, 2)],
                "Days": [inside_days, outside_days],
                "Orders": [len(inside), len(outside)],
            }
        )

        return (
            f"{metric} on the {len(days)} given dates (within {window} days) vs every other day:\n"
            f"  those dates : ${inside_rate:,.2f} per day across {inside_days} days "
            f"({len(inside):,} order lines)\n"
            f"  other days  : ${outside_rate:,.2f} per day across {outside_days} days "
            f"({len(outside):,} order lines)\n"
            f"  difference  : {delta:+.1f}% per day"
        )

    def _chart(self, args: dict[str, Any]) -> str:
        if self.last_result.empty:
            return "ERROR: there is no result to chart yet. Call query_data first."
        chart_type = args.get("chart_type", "bar")
        self.last_chart = chart_type
        return (
            f"Rendered a {chart_type} chart of the previous result "
            f"({len(self.last_result)} rows). It is shown to the user."
        )


# --------------------------------------------------------------------- agent
class ReActAgent:
    """The observe-decide-act loop."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
        client: LLMClient | None = None,
        max_steps: int = MAX_STEPS,
    ):
        self.df = df
        self.schema = schema
        self.client = client or LLMClient()
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        """Answer a question by planning and calling tools.

        Never raises: a failed run comes back as an AgentResult with the trace
        intact, because the trace is the deliverable here as much as the answer.
        """
        executor = ToolExecutor(self.df)
        steps: list[Step] = []
        total = 0.0
        calls = 0
        seen: set[str] = set()

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    schema_block=self.schema.to_prompt_block()
                ),
            },
            {"role": "user", "content": question},
        ]

        for step_number in range(self.max_steps):
            try:
                response = self.client.complete(
                    messages,
                    temperature=LLM.codegen_temperature,
                    tools=TOOLS,
                    max_tokens=700,
                )
            except LLMError as exc:
                steps.append(Step("error", f"Model call failed: {exc}"))
                return AgentResult(
                    question=question,
                    answer=f"**The agent could not run.** {exc}",
                    steps=steps,
                    total_seconds=total,
                    tool_calls=calls,
                    success=False,
                )

            total += response.elapsed_seconds

            if not response.wants_tool:
                # The model answered in prose instead of calling final_answer.
                # That is a valid answer; take it rather than punishing it for
                # not following the protocol. Record it ONLY as the answer - it
                # was previously also logged as a "thought", which duplicated the
                # whole text in the reasoning chain.
                answer = _strip_math_delimiters(
                    response.text or "The agent produced no answer."
                )
                steps.append(
                    Step("answer", answer, elapsed_seconds=response.elapsed_seconds)
                )
                return AgentResult(
                    question=question,
                    answer=answer,
                    steps=steps,
                    data=executor.last_result,
                    chart_type=executor.last_chart,
                    total_seconds=total,
                    tool_calls=calls,
                )

            # Prose accompanying a tool call is genuine reasoning: keep it.
            if response.text:
                steps.append(
                    Step(
                        "thought",
                        _strip_math_delimiters(response.text),
                        elapsed_seconds=response.elapsed_seconds,
                    )
                )

            # Record the assistant turn verbatim so the model sees its own call
            # in the next round; omitting it makes small models re-issue it.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                if call.name == "final_answer":
                    answer = _strip_math_delimiters(
                        str(call.arguments.get("answer", "")).strip()
                    ) or "The agent returned an empty answer."
                    steps.append(Step("answer", answer))
                    return AgentResult(
                        question=question,
                        answer=answer,
                        steps=steps,
                        data=executor.last_result,
                        chart_type=executor.last_chart,
                        total_seconds=total,
                        tool_calls=calls,
                    )

                signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                steps.append(
                    Step(
                        "action",
                        f"{call.name}({json.dumps(call.arguments)})",
                        tool=call.name,
                        arguments=call.arguments,
                    )
                )

                if signature in seen:
                    # The 4B failure mode: re-issuing an identical call forever.
                    # Tell it plainly rather than running the tool again.
                    observation = (
                        "ERROR: you already made this exact call. Use the result "
                        "you were given, or call final_answer."
                    )
                else:
                    seen.add(signature)
                    observation = executor.run(call.name, call.arguments)
                    calls += 1

                steps.append(Step("observation", observation, tool=call.name))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )

        # Loop exhausted. Salvage whatever was gathered rather than returning
        # nothing - a partial answer with a visible trace beats a blank.
        steps.append(
            Step("error", f"Stopped after {self.max_steps} steps without a final answer.")
        )
        return AgentResult(
            question=question,
            answer=(
                f"**The agent ran out of steps** ({self.max_steps}) before "
                "reaching a conclusion. The reasoning chain and any data it "
                "gathered are below."
            ),
            steps=steps,
            data=executor.last_result,
            chart_type=executor.last_chart,
            total_seconds=total,
            tool_calls=calls,
            success=False,
            stopped_early=True,
        )
