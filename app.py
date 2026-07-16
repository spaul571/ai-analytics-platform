"""AI-Powered Data Analytics & Visualization Platform (Task C1).

Streamlit entry point. Three tabs - Overview, Exploration, AI Assistant - sharing
one global filter panel and one session state.

Run:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.advanced.agent import ReActAgent
from src.advanced.anomaly import CONTAMINATION, detect, narrate

# Imported for a side effect as well as for LLM: src.config sets pandas' string
# storage to "python". On the pandas 3 default of "auto" the strings are
# Arrow-backed, and materialising an Arrow-backed category segfaults the Linux
# container this deploys to. See the comment in src/config.py.
from src.config import LLM
from src.data.loader import load_dataset
from src.data.profile import profile_dataset
from src.llm.client import LLMClient
from src.llm.insights import PRESETS, generate_insight
from src.llm.memory import ConversationMemory
from src.llm.pipeline import NLQueryPipeline
from src.viz import autochart, charts
from src.viz.export import (
    ReportPayload,
    figure_renderable,
    figure_to_png,
    figure_to_svg,
    to_docx,
    to_pdf,
)
from src.viz.theme import Theme

st.set_page_config(
    page_title="AI Analytics Platform",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------- cache
@st.cache_resource(show_spinner="Loading dataset...")
def _bootstrap():
    """Load the dataset once per server process, not once per rerun.

    Streamlit re-executes this script top to bottom on every interaction, so
    without this cache the 10k-row CSV would be parsed and cleaned on every
    click. cache_resource rather than cache_data because the DataFrame is
    treated as a shared, read-only resource.
    """
    df, schema, meta = load_dataset()
    report = profile_dataset(df, meta["cleaning_steps"])
    return df, schema, meta, report


@st.cache_resource
def _client() -> LLMClient:
    return LLMClient()


# Streamlit re-executes this whole script on every widget interaction. Anything
# expensive that sits in the history loop therefore runs again for every past
# answer, on every click. Two things in that loop are expensive:
#
#   1. autochart.build() makes an LLM call to write the chart caption.
#   2. to_pdf()/to_docx() each render the figure through kaleido, and
#      st.download_button evaluates its `data` argument eagerly - the user does
#      not have to click it for the work to happen.
#
# Left alone, a session with five answers fires five LLM calls and builds ten
# documents every time a filter moves. The cache below keys on the question, the
# chart override, and the filter scope, so the work happens once per distinct
# state and is free thereafter.
@st.cache_data(show_spinner=False, max_entries=64)
def _cached_chart(question: str, override: str | None, scope_key: str, _frame: pd.DataFrame):
    """Build (and caption) the chart for one answer. Cached across reruns.

    `_frame` is underscore-prefixed so Streamlit does not try to hash the
    DataFrame; the explicit scope_key already identifies it.
    """
    return autochart.build(question, _frame, theme, client=client, override=override)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_pdf(_payload: ReportPayload, cache_key: str) -> bytes:
    return to_pdf(_payload)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_docx(_payload: ReportPayload, cache_key: str) -> bytes:
    return to_docx(_payload)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_png(_figure, cache_key: str) -> bytes:
    return figure_to_png(_figure)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_svg(_figure, cache_key: str) -> bytes:
    return figure_to_svg(_figure)


# The Exploration tab's eight figures are rebuilt on every rerun for the same
# reason - Streamlit re-executes the script, and Plotly figure construction over
# 10k rows is not free. They depend only on the filter scope, so they cache
# cleanly against it.
@st.cache_data(show_spinner=False, max_entries=16)
def _cached_figure(chart: str, scope_key: str, _frame: pd.DataFrame, metric: str = "Sales"):
    builders = {
        "trend": lambda: charts.trend_small_multiples(_frame, theme),
        "map": lambda: charts.state_choropleth(_frame, theme, metric),
        "correlation": lambda: charts.correlation_matrix(_frame, theme),
        "distribution": lambda: charts.profit_distribution(_frame, theme),
        "sunburst": lambda: charts.category_sunburst(_frame, theme),
        "animated": lambda: charts.animated_region_race(_frame, theme),
        "scatter": lambda: charts.discount_profit_scatter(_frame, theme),
        "stacked": lambda: charts.segment_category_bars(_frame, theme),
    }
    return builders[chart]()


df_all, schema, meta, quality = _bootstrap()
client = _client()

# ------------------------------------------------------------- session state
# Task C1 requires UI state to persist across interactions within the session.
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "history" not in st.session_state:
    st.session_state.history = []  # list[PipelineResult]
if "insights" not in st.session_state:
    st.session_state.insights = {}
if "chart_override" not in st.session_state:
    st.session_state.chart_override = {}

theme = Theme("light")


def _options(frame: pd.DataFrame, column: str) -> list[str]:
    """Sorted filter options for a column, without materialising the values.

    A `category` column already stores its distinct values in `.cat.categories`,
    so reading them costs nothing and scans nothing. The obvious spelling —
    sorted(frame[column].unique()) — is both a full column scan and, on pandas 3
    with an Arrow string backend, a call into pyarrow's take(), which is what
    segfaulted the deployed container. Non-categorical columns fall back to the
    scan because they have no categories to read.
    """
    values = frame[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return sorted(str(v) for v in values.cat.categories)
    return sorted(str(v) for v in values.dropna().unique())


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("AI Analytics")
    st.caption(schema.name)

    ok, message = client.health_check()
    if ok:
        st.success(f"Model ready: `{LLM.model}`", icon=":material/check_circle:")
    else:
        st.error(message, icon=":material/power_off:")

    st.divider()
    st.subheader("Filters")
    st.caption("These apply to every chart and every AI answer.")

    years = sorted(df_all["Order Year"].unique())
    year_range = st.select_slider(
        "Year range",
        options=years,
        value=(years[0], years[-1]),
    )

    regions = st.multiselect(
        "Region",
        options=_options(df_all, "Region"),
        default=[],
        placeholder="All regions",
    )
    categories = st.multiselect(
        "Category",
        options=_options(df_all, "Category"),
        default=[],
        placeholder="All categories",
    )
    segments = st.multiselect(
        "Customer segment",
        options=_options(df_all, "Segment"),
        default=[],
        placeholder="All segments",
    )

    st.divider()
    if st.button("Reset conversation", width="stretch"):
        st.session_state.memory.reset()
        st.session_state.history = []
        st.session_state.chart_override = {}
        st.rerun()


# --------------------------------------------------------------- apply filters
def _filtered(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["Order Year"].between(year_range[0], year_range[1])
    if regions:
        mask &= frame["Region"].isin(regions)
    if categories:
        mask &= frame["Category"].isin(categories)
    if segments:
        mask &= frame["Segment"].isin(segments)
    return frame[mask]


df = _filtered(df_all)

ACTIVE_FILTERS: dict[str, object] = {
    "Year range": f"{year_range[0]}-{year_range[1]}",
}
if regions:
    ACTIVE_FILTERS["Region"] = regions
if categories:
    ACTIVE_FILTERS["Category"] = categories
if segments:
    ACTIVE_FILTERS["Segment"] = segments

if df.empty:
    st.warning("No rows match the current filters. Widen them in the sidebar.")
    st.stop()

# Identifies the current filter scope. Every cache above is keyed on it, so
# changing a filter invalidates the cached charts and reports rather than
# serving a chart drawn from data the user is no longer looking at. Getting this
# wrong would be worse than having no cache: a stale chart is a wrong chart.
SCOPE_KEY = "|".join(
    [
        f"{year_range[0]}-{year_range[1]}",
        ",".join(sorted(regions)),
        ",".join(sorted(categories)),
        ",".join(sorted(segments)),
    ]
)


# ---------------------------------------------------------------------- header
scope = len(df)
st.markdown(
    f"### {scope:,} order lines in scope "
    f"<span style='color:#898781;font-size:0.7em'>of {len(df_all):,}</span>",
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_ai, tab_anomaly, tab_agent = st.tabs(
    ["Overview", "Exploration", "AI Assistant", "Anomalies", "Agent"]
)


# ================================================================== OVERVIEW
with tab_overview:
    sales = df["Sales"].sum()
    profit = df["Profit"].sum()
    margin = 100 * profit / sales if sales else 0
    loss_lines = int((df["Profit"] < 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${sales:,.0f}")
    c2.metric("Profit", f"${profit:,.0f}")
    c3.metric("Margin", f"{margin:.1f}%")
    c4.metric(
        "Loss-making lines",
        f"{loss_lines:,}",
        delta=f"{100 * loss_lines / scope:.1f}% of scope",
        delta_color="inverse",
    )

    st.divider()

    # -------------------------------------------------- data quality (Task A3)
    with st.expander("Data quality", expanded=False):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Completeness", f"{quality.completeness_pct:.2f}%")
        q2.metric("Duplicate rows", f"{quality.duplicate_rows:,}")
        q3.metric("IQR outliers", f"{quality.total_outliers:,}")
        q4.metric("Memory", f"{meta['memory_mb']} MB")

        st.caption(
            f"Loaded in {meta['load_seconds'] * 1000:.0f} ms and cleaned in "
            f"{meta['clean_seconds'] * 1000:.0f} ms. Categorical downcasting cut "
            f"the in-memory footprint from {meta['memory_raw_mb']} MB to "
            f"{meta['memory_mb']} MB ({meta['memory_saved_pct']}% saved)."
        )
        st.markdown("**Cleaning steps applied**")
        for step in meta["cleaning_steps"]:
            st.markdown(f"- {step}")

        st.markdown("**Column profile**")
        st.dataframe(quality.to_frame(), width="stretch", hide_index=True)

        st.info(
            f"The {quality.total_outliers:,} IQR outliers are reported but "
            "**not removed**. In this dataset they are genuine large orders and "
            "genuine large losses - dropping them would erase exactly the "
            "anomalies the platform exists to surface.",
            icon=":material/info:",
        )

    st.divider()

    # ------------------------------------------------- preset insights (B3)
    st.subheader("AI-generated insights")
    st.caption(
        "The figures below are computed in pandas, not written by the model. "
        "The LLM narrates numbers it is handed - it never invents them."
    )

    cols = st.columns(len(PRESETS))
    for col, (key, (title, description, _)) in zip(cols, PRESETS.items()):
        with col:
            if st.button(title, width="stretch", help=description):
                with st.spinner(f"Generating: {title}..."):
                    st.session_state.insights[key] = generate_insight(key, df, client)

    for key, insight in st.session_state.insights.items():
        with st.container(border=True):
            st.markdown(f"#### {insight.title}")
            if insight.error:
                st.warning(insight.error, icon=":material/warning:")
            st.markdown(insight.narrative)
            st.caption(f"Generated in {insight.elapsed_seconds:.1f}s by {LLM.model}")

            with st.expander("Underlying figures"):
                for name, frame in insight.tables.items():
                    st.markdown(f"**{name}**")
                    st.dataframe(frame, width="stretch", hide_index=True)


# =============================================================== EXPLORATION
with tab_explore:
    st.subheader("Visualisation suite")
    st.caption(
        "Eight chart types, one palette. Colour is assigned to entities in a "
        "fixed order, so filtering never repaints the survivors."
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_cached_figure("trend", SCOPE_KEY, df), width="stretch")
    with right:
        metric = st.radio(
            "Map metric",
            ["Sales", "Profit"],
            horizontal=True,
            key="map_metric",
            help="Profit uses a diverging scale so losses cannot look like small gains.",
        )
        st.plotly_chart(
            _cached_figure("map", SCOPE_KEY, df, metric), width="stretch"
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_cached_figure("correlation", SCOPE_KEY, df), width="stretch")
    with right:
        st.plotly_chart(_cached_figure("distribution", SCOPE_KEY, df), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_cached_figure("sunburst", SCOPE_KEY, df), width="stretch")
    with right:
        st.plotly_chart(_cached_figure("animated", SCOPE_KEY, df), width="stretch")

    st.plotly_chart(_cached_figure("scatter", SCOPE_KEY, df), width="stretch")
    st.caption(
        "The regression line is fitted with least squares and the band is a 95% "
        "confidence interval on the mean response."
    )

    st.plotly_chart(_cached_figure("stacked", SCOPE_KEY, df), width="stretch")

    with st.expander("Table view of the filtered data"):
        # The palette's light mode has two hues below 3:1 contrast, which obliges
        # a non-colour route to the same information. This is that route.
        st.dataframe(df.head(500), width="stretch", hide_index=True)
        st.caption(f"Showing 500 of {scope:,} rows in scope.")


# =============================================================== AI ASSISTANT
with tab_ai:
    st.subheader("Ask a question")
    st.caption(
        "Questions are answered against the **filtered** data shown in the "
        "sidebar. The generated code runs in a sandbox - no imports, no file "
        "access, no network."
    )

    pipeline = NLQueryPipeline(df, schema, client=client, memory=st.session_state.memory)

    examples = [
        "Which sub-categories are losing us money?",
        "Show me the top 5 customers by profit",
        "How did sales trend by month?",
        "Which state has the highest average discount?",
    ]
    example_cols = st.columns(len(examples))
    clicked = None
    for col, example in zip(example_cols, examples):
        if col.button(example, width="stretch"):
            clicked = example

    question = st.chat_input("Ask about the data...") or clicked

    if st.session_state.memory.turns:
        with st.expander(f"Conversation context ({len(st.session_state.memory)} turns)"):
            st.caption(
                "Follow-up questions are resolved against these. "
                "Use *Reset conversation* in the sidebar to clear them."
            )
            st.code(st.session_state.memory.summary(), language=None)

    if question:
        with st.spinner("Thinking..."):
            result = pipeline.ask(question)
        st.session_state.history.insert(0, result)

    for index, result in enumerate(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"**{result.question}**")

            if not result.success:
                st.error(result.answer)
                with st.expander("Debug trace"):
                    st.code("\n\n".join(result.trace), language="text")
                continue

            st.markdown(result.answer)

            key = f"chart_{index}_{abs(hash(result.question)) % 0xFFFF}"
            auto = None

            if not result.data.empty:
                override = st.session_state.chart_override.get(key)
                auto = _cached_chart(result.question, override, SCOPE_KEY, result.data)

                if auto.figure is not None:
                    st.plotly_chart(auto.figure, width="stretch", key=key)
                    st.caption(auto.caption)

                    control, _ = st.columns([1, 2])
                    with control:
                        choice = st.selectbox(
                            "Chart type",
                            options=auto.available_types,
                            index=(
                                auto.available_types.index(auto.chart_type)
                                if auto.chart_type in auto.available_types
                                else 0
                            ),
                            key=f"select_{key}",
                            help=auto.reason,
                        )
                        if choice != auto.chart_type:
                            st.session_state.chart_override[key] = choice
                            st.rerun()

                with st.expander("Result data"):
                    st.dataframe(result.data, width="stretch", hide_index=True)

            # ------------------------------------------------- export (C4)
            figure = auto.figure if auto else None
            payload = ReportPayload(
                title="AI Analytics Report",
                question=result.question,
                narrative=result.answer,
                data=result.data,
                figure=figure,
                caption=auto.caption if auto else "",
                code=result.code,
                filters=ACTIVE_FILTERS,
                dataset_name=schema.name,
                row_count=scope,
                model=LLM.model,
            )

            export_key = f"{key}:{SCOPE_KEY}:{result.code}"

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.download_button(
                    "PDF report",
                    data=_cached_pdf(payload, export_key),
                    file_name="ai_analytics_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key=f"pdf_{key}",
                )
            with e2:
                st.download_button(
                    "Word report",
                    data=_cached_docx(payload, export_key),
                    file_name="ai_analytics_report.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document"
                    ),
                    width="stretch",
                    key=f"docx_{key}",
                )
            # Without a browser the map cannot be rasterised at all, and
            # `data=` is evaluated eagerly - so ask first rather than let the
            # download button take the whole page down on every rerun.
            if figure_renderable(figure):
                with e3:
                    st.download_button(
                        "Chart PNG",
                        data=_cached_png(figure, export_key),
                        file_name="chart.png",
                        mime="image/png",
                        width="stretch",
                        key=f"png_{key}",
                    )
                with e4:
                    st.download_button(
                        "Chart SVG",
                        data=_cached_svg(figure, export_key),
                        file_name="chart.svg",
                        mime="image/svg+xml",
                        width="stretch",
                        key=f"svg_{key}",
                    )
            elif figure is not None:
                with e3:
                    st.caption(
                        "Map images need a browser, which this hosted "
                        "environment does not have. The PDF and Word reports "
                        "still download, without the map."
                    )

            with st.expander("How this was answered"):
                st.code(result.code, language="python")
                t1, t2, t3 = st.columns(3)
                t1.metric("Code generation", f"{result.codegen_seconds:.1f}s")
                t2.metric("Sandbox execution", f"{result.execution_ms:.0f}ms")
                t3.metric("Narrative", f"{result.format_seconds:.1f}s")
                if result.retried:
                    st.warning(
                        "The first attempt failed and was auto-corrected.",
                        icon=":material/refresh:",
                    )
                st.code("\n\n".join(result.trace), language="text")


# ============================================================ D3 ANOMALIES
@st.cache_data(show_spinner=False, max_entries=8)
def _cached_anomalies(scope_key: str, _frame: pd.DataFrame):
    """Isolation Forest is ~1s over 10k rows; cache it against the filter scope."""
    return detect(_frame)


with tab_anomaly:
    st.subheader("Anomaly detection")
    st.caption(
        "Isolation Forest over Sales, Quantity, Discount, Profit and Shipping "
        "Days. The model does not decide what is anomalous - it explains rows "
        "the detector has already flagged."
    )

    report = _cached_anomalies(SCOPE_KEY, df)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Flagged", f"{report.flagged_count:,}", delta=f"{CONTAMINATION:.0%} of scope")
    a2.metric("Loss on flagged rows", f"${report.total_loss:,.0f}")
    a3.metric(
        "Loss-making",
        f"{report.flagged_loss_rate:.1%}",
        delta=f"vs {report.baseline_loss_rate:.1%} baseline",
        delta_color="inverse",
    )
    a4.metric(
        "Enrichment",
        f"{report.enrichment:.2f}x",
        help="How much likelier a flagged row is to lose money than a random row. "
        "1.0x would mean the detector is picking at random.",
    )

    st.info(report.selectivity, icon=":material/filter_alt:")

    with st.expander("Why Isolation Forest and not just IQR?"):
        st.markdown(
            f"""
The usual argument for a forest is that it catches rows whose *combination* of
values is odd while no single column is extreme. **On this dataset that turned
out to be false.** Across a contamination sweep, the forest found essentially
nothing the per-column IQR rules missed:

| contamination | flagged | caught by forest but **not** IQR |
|---|---|---|
| 1% | 100 | 0 |
| 2% | 200 | 0 |
| 5% | 500 | 1 |
| 10% | 1,000 | 22 |

Its real value is **selectivity**. The union of the IQR rules flags
{report.iqr_flagged_total:,} rows — {100 * report.iqr_flagged_total / len(report.scored):.1f}%
of the data. "A third of your orders are unusual" is not an alert list anyone can
act on. The forest returns a bounded, **ranked** set with a continuous score, so
the worst row comes first and a human can start at the top. IQR cannot rank at
all.

Flagged rows are **{report.enrichment:.2f}x** likelier to be loss-making than a
random row, which is the test that actually matters.
"""
        )
        st.markdown("**Univariate detector counts, for comparison**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": list(report.iqr_counts),
                    "IQR (1.5x)": list(report.iqr_counts.values()),
                    "Z-score (>3)": [report.zscore_counts[c] for c in report.iqr_counts],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if st.button("Explain the worst anomalies", type="primary"):
        with st.spinner("Analysing flagged orders..."):
            st.session_state.anomaly_narrative = narrate(report, top_n=5, client=client)

    narrated = st.session_state.get("anomaly_narrative")
    if narrated is not None:
        with st.container(border=True):
            st.markdown("#### What went wrong")
            if narrated.error:
                st.warning(narrated.error, icon=":material/warning:")
            st.markdown(narrated.narrative)
            if narrated.elapsed_seconds:
                st.caption(f"Generated in {narrated.elapsed_seconds:.1f}s by {LLM.model}")

    st.markdown("#### Flagged order lines, worst first")
    st.dataframe(
        report.flagged.head(50),
        width="stretch",
        hide_index=True,
        column_config={
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly score",
                help="Lower is more anomalous.",
                min_value=float(report.flagged["anomaly_score"].min()),
                max_value=float(report.flagged["anomaly_score"].max()),
                format="%.3f",
            ),
            "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
            "Profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
            "Discount": st.column_config.NumberColumn("Discount", format="%.0f%%"),
        },
    )
    st.caption(f"Showing 50 of {report.flagged_count:,} flagged rows.")


# ================================================================ D4 AGENT
with tab_agent:
    st.subheader("Multi-turn reasoning agent")
    st.caption(
        "A ReAct loop: the model decides whether to query, detect anomalies, "
        "look up holidays, compare dates, chart, or answer — then observes the "
        "result and decides again. Its tools take typed arguments, so unlike the "
        "AI Assistant it never writes code and needs no sandbox."
    )

    agent_examples = [
        "Which region has the highest sales? Chart it.",
        "Find the most unusual orders and tell me what went wrong.",
        "Do sales rise around US public holidays in 2016?",
    ]
    agent_cols = st.columns(len(agent_examples))
    agent_clicked = None
    for col, example in zip(agent_cols, agent_examples):
        if col.button(example, width="stretch", key=f"agent_{example[:12]}"):
            agent_clicked = example

    agent_question = st.text_input(
        "Ask the agent", placeholder="Ask something that needs more than one step..."
    )
    ask = st.button("Run agent", type="primary") and agent_question
    question_for_agent = agent_clicked or (agent_question if ask else None)

    if question_for_agent:
        agent = ReActAgent(df, schema, client=client)
        with st.spinner("The agent is reasoning..."):
            st.session_state.agent_result = agent.run(question_for_agent)

    agent_result = st.session_state.get("agent_result")
    if agent_result is not None:
        with st.container(border=True):
            st.markdown(f"**{agent_result.question}**")

            if agent_result.stopped_early:
                st.warning(agent_result.answer, icon=":material/timer_off:")
            elif not agent_result.success:
                st.error(agent_result.answer)
            else:
                st.markdown(agent_result.answer)

            m1, m2, m3 = st.columns(3)
            m1.metric("Tool calls", agent_result.tool_calls)
            m2.metric("Reasoning steps", len(agent_result.steps))
            m3.metric("Time", f"{agent_result.total_seconds:.1f}s")

            if not agent_result.data.empty and agent_result.chart_type:
                figure = autochart.render(
                    agent_result.data,
                    agent_result.chart_type,  # type: ignore[arg-type]
                    theme,
                    title=agent_result.question,
                )
                if figure is not None:
                    st.plotly_chart(figure, width="stretch", key="agent_chart")

            if not agent_result.data.empty:
                with st.expander("Data the agent gathered"):
                    st.dataframe(agent_result.data, width="stretch", hide_index=True)

            # The collapsible reasoning chain the brief asks for.
            with st.expander("Reasoning chain", expanded=False):
                icons = {
                    "thought": ":material/psychology:",
                    "action": ":material/build:",
                    "observation": ":material/visibility:",
                    "answer": ":material/check_circle:",
                    "error": ":material/error:",
                }
                for number, step in enumerate(agent_result.steps, start=1):
                    st.markdown(
                        f"{icons[step.kind]} **{number}. {step.kind.upper()}**"
                        + (f" — `{step.tool}`" if step.tool else "")
                    )
                    if step.kind in ("observation", "action"):
                        st.code(step.content, language="text")
                    else:
                        st.markdown(step.content)
                    st.divider()
