# AI-Powered Data Analytics & Visualization Platform

Capstone project — Deep Learning, MSc CSE.

Natural-language analytics over the Global E-Commerce Sales (Superstore) dataset,
powered by **Gemma 4 E4B** running locally in **LM Studio**.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Data | pandas + PyArrow dtypes | ~10k rows fits in memory; PyArrow cuts string memory and read time |
| LLM | Gemma 4 E4B via LM Studio | Local, free, no API keys, no rate limits; OpenAI-compatible endpoint |
| UI | Streamlit | Native session state and tabs map directly onto the Task C requirements |
| Charts | Plotly | Interactive, exports to PNG/SVG via kaleido |

## Setup

Requires **Python 3.13** (verified on 3.13.12, Windows).

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

> **Use `python -m pip`, not `pip`.** On a typical Windows box with several
> Pythons installed, `where python` and `where pip` resolve to *different*
> interpreters — commonly python.org's 3.13 for `python` and the Microsoft Store
> build for `pip`. Packages then install into an interpreter that never runs your
> code, and every import fails with `ModuleNotFoundError` for a package `pip list`
> swears is installed. `python -m pip` binds pip to the interpreter that is
> actually executing, which makes the problem impossible. The virtualenv above
> also sidesteps it entirely.

### Dataset

Download the Superstore dataset and save it as `data/superstore.csv`:

- Kaggle: <https://www.kaggle.com/datasets/vivek468/superstore-dataset-final>

Expected: ~9,994 rows, 21 columns (`Order ID`, `Order Date`, `Region`, `Category`,
`Sub-Category`, `Sales`, `Quantity`, `Discount`, `Profit`, ...).

### LM Studio

1. Load `google/gemma-4-e4b`.
2. Developer tab → **Status: Running**.
3. Confirm the model id matches `LLM_MODEL` in `.env`:
   ```bash
   curl http://localhost:1234/v1/models
   ```

Recommended load settings: **Context Length 16384** (not the 131k maximum — the
KV cache at 131k wastes several GB of VRAM for prompts that never exceed ~2k
tokens). Temperature is set per-request in code, so the GUI value is ignored.

## Run

```bash
streamlit run app.py
```

## Verify

```bash
python -m scripts.check_task_a    # data layer: schema, 5 sample queries, <500ms budget
python -m scripts.check_task_b    # sandbox escape tests + 10-question LLM benchmark
python -m scripts.check_task_c    # 8 charts, auto-chart selection, PDF/DOCX/PNG/SVG
python -m scripts.check_task_d    # anomaly detection + agent tool schemas
```

`check_task_b` writes `benchmarks/results.csv` — the accuracy table required by
Task B5. Checks A, C, and the D3 half of D run without LM Studio; the benchmark
and the live agent need it up.

## Advanced features (Task D)

**D3 — Anomaly detection.** Isolation Forest over Sales, Quantity, Discount,
Profit and Shipping Days, with IQR and z-score reported alongside for comparison.

The comparison produced a result worth stating plainly: **the forest finds
nothing the univariate rules miss** at any usable threshold (0 forest-only
catches at 1% and 2% contamination). Its value is *selectivity*, not novelty —
the union of the per-column IQR rules flags 28.5% of the dataset, which is not an
actionable alert list, while the forest returns a bounded, **ranked** 200 rows.
Flagged rows are **2.22x** likelier to be loss-making than a random row.

**D4 — ReAct agent.** A tool-calling loop with four tools (`query_data`,
`detect_anomalies`, `create_chart`, `final_answer`). The model decides what to
call, observes the result, and decides again; the full reasoning chain is shown
in a collapsible panel.

The agent's tools take **typed arguments** — the executor builds the pandas call,
not the model — so this surface is safe by construction and needs no sandbox,
unlike the Task B pipeline where the model writes the code itself. The two
approaches sit side by side on purpose.

## The sandbox

Generated code is executed, so the sandbox is the security boundary. It has
three independent layers, documented in full in `src/llm/sandbox.py`:

1. **AST allowlist** (static, pre-execution). Only expressions and simple
   assignments over `df`/`pd`/`np` are permitted. `Import`, `ImportFrom`,
   function/class definitions, and loops are rejected outright. Any name or
   attribute starting with `_` is rejected, which kills the whole
   `__class__ → __subclasses__` traversal family in one rule. Introspection
   builtins (`eval`, `exec`, `getattr`, `globals`, `open`, …) are rejected by
   name.
2. **Restricted namespace.** `__builtins__` is replaced by an explicit dict of
   ~25 safe functions. Even a surviving reference has no `open` or `__import__`
   to reach.
3. **Timeout.** Execution runs on a worker thread with a 10s budget.

A substring blocklist (`if "import" in code`) would be defeated by
`__import__("os")` in one line. The escape-attempt suite in `check_task_b.py`
demonstrates eight such bypasses, all blocked.

## Layout

```
src/
  config.py            environment + tunables
  data/
    loader.py          A1  ingestion, cleaning, derived columns
    schema.py          A1  schema extraction -> LLM prompt block
    query.py           A2  aggregation + filtered query engine
    profile.py         A3  quality report, IQR outliers
  llm/
    client.py          B5  LM Studio client, timeout/truncation/empty handling
    prompts.py         B1  schema-aware prompt, synonym map, few-shot examples
    sandbox.py         B2  AST validation + restricted namespace + timeout
    pipeline.py        B2  3-phase pipeline with single auto-retry
    memory.py          B4  last-5-turn conversation context
    insights.py        B3  three preset analyses with LLM narration
  viz/
    theme.py           C2  validated palette, light + dark
    charts.py          C2  8 chart types
    autochart.py       C3  rule-based chart selection + LLM caption
    export.py          C4  PDF / DOCX / PNG / SVG export
  advanced/
    anomaly.py         D3  Isolation Forest + IQR + z-score
    agent.py           D4  ReAct loop, 4 typed tools
benchmarks/
  questions.py         the 10 benchmark questions + hand-written ground truth
scripts/
  check_task_a.py      Task A acceptance check
  check_task_b.py      Task B acceptance check + benchmark
  check_task_c.py      Task C acceptance check
  check_task_d.py      Task D acceptance check
deploy/
  llm_proxy.py         bearer-token gate in front of LM Studio
  start-demo.ps1       starts the proxy + Cloudflare Tunnel, prints the secrets
  stop-demo.ps1        stops both
DEPLOY.md              how the deployed app reaches a locally hosted model
```
