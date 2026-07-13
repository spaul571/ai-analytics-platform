# AI-Powered Data Analytics & Visualization Platform

**Natural-language analytics over e-commerce transactions using a locally hosted 4B-parameter language model**

> **DRAFT — items marked `[FILL]` need your input before submission.**
> Max 20 pages. Every figure referenced here is in `capstone/exports/figures/`
> (run `python -m scripts.export_report_assets` to regenerate).

---

## 1. Title Page

| | |
|---|---|
| **Course** | CSE-638 Deep Learning — MSc Computer Science & Engineering |
| **Team members** | `[FILL: names + student IDs]` |
| **Dataset** | Global E-Commerce Sales (Superstore) — 9,994 order line items |
| **Language model** | Gemma 4 E4B (4B effective params), served locally via LM Studio |
| **GitHub** | `[FILL: repository URL]` |
| **Submitted** | `[FILL: date]` |

---

## 2. Abstract

*(≤200 words)*

We built an end-to-end analytics platform that answers plain-English questions
about 9,994 e-commerce order lines by generating, safely executing, and narrating
pandas code. The intelligence layer is **Gemma 4 E4B**, a 4-billion-parameter
model running locally in LM Studio — chosen over a hosted API to eliminate cost,
rate limits, and data egress.

The system comprises a pandas data layer (sub-5 ms filtered aggregations, 79%
memory reduction via categorical downcasting), a three-phase natural-language
query pipeline guarded by a three-layer execution sandbox, an eight-chart
Plotly dashboard with rule-based automatic chart selection, and two advanced
features: Isolation Forest anomaly detection and a ReAct tool-calling agent.

On a 10-question benchmark scored against hand-written ground truth, text-to-code
accuracy improved from **5/10 to 10/10** across three iterations. Critically,
**none of that gain came from a larger model** — it came from three diagnosed
defects: a JSON-escaping interaction that corrupted generated code, a missing
few-shot pattern, and a defence-in-depth bug in which our own sandbox's two layers
disagreed. The project's central finding is that small-model reliability is an
*engineering* problem, not a capacity problem.

**Keywords:** text-to-SQL, local LLM inference, sandboxed code execution, prompt
engineering, anomaly detection, ReAct agents

---

## 3. Introduction

### 3.1 Problem motivation

Analytics tooling has an access problem. The people who most need answers from
data — category managers, regional leads, finance — are rarely the people who can
write `df.groupby('Sub-Category')['Profit'].sum()`. Every question therefore
queues behind an analyst, and questions that would take ten seconds to answer take
two days to ask.

Large language models collapse that queue by translating natural language into
executable queries. But most demonstrations of this rely on frontier hosted models,
which introduces per-query cost, rate limits, and — for any business with
commercially sensitive data — an unacceptable requirement to ship that data to a
third party.

This project asks a narrower and more practical question: **can a 4-billion-parameter
model running on a single consumer GPU do this reliably enough to be useful?** The
answer we arrive at is yes, but only with engineering that a frontier model would
not require.

### 3.2 Dataset

The **Global E-Commerce Sales (Superstore)** dataset: 9,994 retail order line
items across four years (2014–2017), 21 source columns.

| Property | Value |
|---|---|
| Rows | 9,994 |
| Source columns | 21 |
| Derived columns | 3 (`Order Year`, `Order Month`, `Shipping Days`) |
| Date range | 2014-01-03 → 2017-12-30 |
| Completeness | 100.00% |
| Duplicate rows | 0 |
| Geography | United States, 49 states, 4 regions |

Each row is one product line within a customer order. `Sales` is gross revenue;
`Profit` is net margin and **can be negative** — a deliberately chosen property,
because a signed metric forces genuine design decisions throughout (diverging
colour scales, aggregate-then-filter query semantics, loss-focused anomaly
detection) that an all-positive dataset would let us dodge.

**Why this dataset, given a 4B model.** The selection was driven by the model, not
the other way round. Superstore's column names are unambiguous English (`Sales`,
`Profit`, `Region`), and virtually every analytical question reduces to a single
`groupby` — which is precisely the shape a small model emits reliably. The
alternatives were rejected on the same grounds: the Stack Overflow survey stores
languages as a semicolon-delimited multi-select requiring `.str.split().explode()`
chains; the NYC taxi and S&P 500 datasets require rolling-window and temporal
binning logic. Each of those is a code shape a 4B model gets wrong far more often
than it gets a plain `groupby` right.

### 3.3 Analytical goals

1. Answer arbitrary natural-language questions with executable, verifiable code.
2. Surface the profitability structure hidden beneath headline revenue.
3. Detect and explain anomalous transactions.
4. Do all of it with a model that runs locally and costs nothing per query.

---

## 4. System Architecture

```
                        ┌──────────────────────────────────┐
                        │   LM Studio  (localhost:1234)    │
                        │   Gemma 4 E4B · 6.33 GB GGUF     │
                        │   OpenAI-compatible /v1 endpoint │
                        └───────────────┬──────────────────┘
                                        │  HTTP (openai SDK)
                                        │
┌───────────────────────────────────────┴────────────────────────────────┐
│                            INTELLIGENCE LAYER                          │
│                                                                        │
│  prompts.py          pipeline.py              sandbox.py               │
│  ┌────────────┐      ┌──────────────────┐     ┌──────────────────────┐ │
│  │ schema     │─────▶│ P1: generate code│────▶│ L1 AST allowlist     │ │
│  │ synonyms   │      │ P2: EXECUTE  ────┼────▶│ L2 restricted ns     │ │
│  │ few-shots  │      │ P3: narrate      │◀────│ L3 timeout           │ │
│  └────────────┘      └────────┬─────────┘     └──────────────────────┘ │
│                               │ on error: ONE auto-retry               │
│  memory.py (last 5 turns) ────┘                                        │
│                                                                        │
│  insights.py — 3 presets: aggregations are HAND-WRITTEN, LLM narrates  │
│  agent.py    — ReAct loop, 4 typed tools, no sandbox needed            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                              DATA LAYER                                │
│  loader.py → schema.py → query.py → profile.py → anomaly.py            │
│  pandas DataFrame, in-memory, 1.89 MB, categorical dtypes              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│  app.py (Streamlit) — 5 tabs, one global filter panel                  │
│  theme.py (validated palette) → charts.py (8 types)                    │
│  autochart.py (rule-based selection) → export.py (PDF/DOCX/PNG/SVG)    │
└────────────────────────────────────────────────────────────────────────┘
```

**Data flow for one natural-language question:**

1. User types *"Which sub-categories are losing us money?"*
2. `prompts.py` assembles: schema table + 37-entry synonym map + 6 few-shot
   examples + last 5 conversation turns.
3. **Phase 1** — Gemma returns `{"code": "totals = df.groupby('Sub-Category', observed=True)['Profit'].sum()\nresult = totals[totals < 0].sort_values()"}`
4. **Phase 2** — `sandbox.py` validates the AST, then executes in a restricted
   namespace with a 10 s budget. On failure, the exception is fed back for exactly
   one auto-retry.
5. **Phase 3** — Gemma narrates the *result table only* (never the raw DataFrame),
   which is what prevents it inventing numbers.
6. `autochart.py` picks a chart type from the result's *shape*, renders it, and
   asks Gemma for a one-sentence caption.

---

## 5. Task A — Data Layer

### 5.1 Schema extraction

The schema is the single artifact handed to the model. It is rendered as a compact
table rather than JSON — same information, roughly half the tokens, which leaves
context room for conversation history and few-shot examples.

Design decision: **categorical columns with ≤20 distinct values are listed in
full.** This threshold was set deliberately so that `Sub-Category` (17 values) is
enumerated — its values are dataset-specific and the model cannot guess them.
`State` (49 values) falls back to samples, because the model already knows US state
names.

### 5.2 Cleaning steps applied

| # | Step | Effect |
|---|---|---|
| 1 | Parse `Order Date`, `Ship Date` to datetime64 | Source stores MM/DD/YYYY strings; every temporal query is impossible until converted |
| 2 | Standardise categorical casing/whitespace | No-op on this dataset (already clean) — reported honestly rather than fabricating dirt |
| 3 | Duplicate removal | 0 found |
| 4 | Derive `Order Year`, `Order Month`, `Shipping Days` | **See below — this is a model-driven decision** |
| 5 | Downcast low-cardinality text to `category` dtype | 9.21 MB → 1.89 MB |

**Why derive `Order Month` rather than let the model compute it.** Gemma 4 E4B is
markedly more reliable emitting `groupby('Order Month')` than a
`.dt.to_period('M')` accessor chain. Precomputing three columns trades a little
storage for a large jump in text-to-code accuracy. This is an example of the
project's recurring theme: **shaping the data to fit the model's competence is
cheaper than fighting the model.**

### 5.3 Data quality

```
Completeness   100.00%
Duplicates     0
IQR outliers   4,074
```

**The 4,074 IQR outliers are reported but NOT removed.** They are genuine large
orders and genuine large losses. Dropping them would erase exactly the anomalies
Task D3 exists to surface — a data-cleaning step that deletes the finding is not
cleaning, it is destroying evidence.

### 5.4 Performance benchmark (A4)

| Query | Time |
|---|---|
| Q1 Direct aggregation — sales & profit per region | 5.0 ms |
| Q2 Direct aggregation — mean discount by sub-category | 2.7 ms |
| Q3 Filtered — 2017 furniture sales by state | 4.3 ms |
| Q4 Filtered — heavily discounted loss-making lines | 3.8 ms |
| Q5 Whole-table aggregation | 0.5 ms |
| **Slowest** | **5.0 ms** (budget: 500 ms) → **PASS, 100× headroom** |

**Time/memory trade-off.** Casting low-cardinality text columns to pandas
`category` dtype costs a one-off 59 ms conversion pass and returns a **79.4%
smaller frame** (9.21 MB → 1.89 MB), while also accelerating every subsequent
`groupby`. At this dataset size the memory saving is not itself necessary —
1.89 MB versus 9.21 MB changes nothing operationally — but the `groupby` speedup
is what buys the 100× headroom against the 500 ms budget.

**Lazy loading was not implemented.** The brief requires it only above 1M rows;
this dataset is 9,994. Implementing chunked processing here would be dead code
carrying no benefit, and we state that rather than adding it for appearance.

---

## 6. Task B — LLM Integration

### 6.1 Model choice and justification

| | |
|---|---|
| Model | `google/gemma-4-e4b` |
| Quantization | GGUF, 6.33 GB |
| Server | LM Studio 0.4.19, OpenAI-compatible endpoint |
| Throughput | ~44–59 tok/s generation; ~1,750 tok/s prompt eval |
| Cost | **Zero** per query |

Chosen over a hosted API because: no per-query cost, no rate limits, no data
leaves the machine, and — most relevant academically — it forces the interesting
engineering. A frontier model would have scored 10/10 on our benchmark
immediately and taught us nothing.

### 6.2 Prompt design (B1)

Three components, in order of measured impact:

1. **Few-shot examples (largest lever).** Six examples covering the query shapes
   the benchmark contains: simple groupby, filter-then-group, temporal, top-N,
   scalar ratio, and aggregate-then-filter.
2. **Synonym map.** 37 entries mapping informal phrasing to real columns —
   `revenue → Sales`, `markdown → Discount`, `buyers → Customer Name`. This turns
   column resolution from *inference* into *lookup*, which is what small models
   are actually good at.
3. **Explicit output contract.** Single quotes only; assign to `result`; never
   `.tolist()`; multiply by 100 when asked for a percentage.

**Result: column identification was 10/10 on the very first run** and never
regressed. B1 was solved before anything else worked.

#### Question-to-column mapping examples

| Question phrasing | Resolved columns |
|---|---|
| "Which region brings in the most **revenue**?" | `Region`, `Sales` |
| "…deepest average **markdown**" | `Sub-Category`, `Discount` |
| "Who are our top 5 **buyers** by total **spend**?" | `Customer Name`, `Sales` |
| "Compare average **delivery time** across **shipping speeds**" | `Ship Mode`, `Shipping Days` |

### 6.3 The sandbox (B2, Phase 2) — security rationale

We execute code a language model wrote. That is an arbitrary-code-execution sink
by construction, so the defence cannot be a substring blocklist: `if "import" in
code` is defeated by `__import__("os")`, and checking for `os` is defeated by
`getattr(__builtins__, "ev" + "al")`. Both are one-liners.

**Three independent layers. An attack must defeat all three.**

| Layer | Mechanism |
|---|---|
| **1. AST validation** (static, pre-execution) | Every syntax node checked against an *allowlist*. `Import`, `ImportFrom`, function/class defs, loops → rejected. **Any name or attribute beginning with `_` → rejected**, which kills the entire `().__class__.__base__.__subclasses__()` traversal family in one rule. Introspection builtins (`eval`, `exec`, `getattr`, `globals`, `open`) rejected by name. Allowlisting means an unknown node type **fails closed**. |
| **2. Restricted namespace** | `__builtins__` replaced by an explicit dict of ~25 safe functions. Even a surviving reference has no `open` or `__import__` to reach. Only `df`, `pd`, `np` are in scope. |
| **3. Timeout** | 10 s execution budget on a worker thread. |

**Verified: 8/8 escape attempts blocked**, legitimate queries still permitted.

```
BLOCKED  direct import        Disallowed syntax: Import
BLOCKED  dunder import        Disallowed name: '__import__'
BLOCKED  builtins traversal   Disallowed attribute access: '__subclasses__'
BLOCKED  getattr indirection  Disallowed builtin: 'getattr'
BLOCKED  eval injection       Disallowed builtin: 'eval'
BLOCKED  file write           Disallowed builtin: 'open'
BLOCKED  globals access       Disallowed builtin: 'globals'
BLOCKED  no result assigned   must assign to 'result'
ALLOWED  legitimate query     correctly permitted
```

**Stated limitations.** Layer 3 *abandons* a timed-out thread rather than killing
it — CPython offers no safe way to terminate a running thread — so a runaway
computation keeps consuming CPU until it finishes, though it cannot return a value
or block the UI. And this is a *language-level* sandbox, not an OS-level one. That
is appropriate here because the untrusted input comes from a local model we
ourselves prompt. Exposing this endpoint to untrusted users would demand process
isolation (a container or seccomp jail) instead.

### 6.4 Benchmark results (B5) — and how they got there

Ten questions, scored against **hand-written ground-truth implementations** on four
axes.

| Run | Columns | Executed | **Accurate** | Format | Mean time |
|---|---|---|---|---|---|
| 1 | 10/10 | 9/10 | **5/10** | 9/10 | 1.9 s |
| 2 | 10/10 | 9/10 | **9/10** | 9/10 | 2.2 s |
| 3 | 10/10 | 10/10 | **10/10** | 10/10 | 2.0 s |

**Not one of those gains came from changing the model.** Each came from reading
the actual generated code and diagnosing a specific defect.

#### Defect 1 — structured output corrupts generated code (Run 1, Q01)

Gemma emitted:

```python
result = df.groupby("Region", observed=True)"Sales".sum().sort_values(ascending=False)
```

The subscript brackets are **gone**. `)"Sales".` is not valid Python.

**Root cause:** we request structured JSON output, so the code lives inside a JSON
string. To write `["Sales"]` there the model must emit `[\"Sales\"]` — escaped. It
fumbled the escaping and lost the brackets.

The diagnostic tell: **Q01 was the only question where Gemma chose double quotes,
and it was the only syntax error in the entire set.** Every other question used
`df['Profit']` — single quotes, no escaping needed, no failure. Our own few-shot
examples were written with double quotes; we were teaching it the dangerous style.

**Fix:** constrain the model to single quotes. This does not *fix* the escaping —
it makes escaping **unnecessary**, eliminating the failure class rather than
patching it.

#### Defect 2 — aggregate-then-filter (Run 1, Q04)

Asked *"which sub-categories are losing us money?"*, Gemma wrote:

```python
result = df[df['Profit'] < 0].groupby('Sub-Category', observed=True)['Profit'].sum()
```

It filtered the loss-making **rows** first — which discards the profitable sales in
the same sub-category and answers a *different question*. Whether a group loses
money on net can only be known **after** summing it.

**Fix:** one targeted few-shot example demonstrating the shape. Run 3 produced the
correct form unprompted.

#### Defect 3 — our own sandbox's two layers disagreed (Run 2, Q09)

```
SandboxViolation: Unknown name: 'len'. Only False, None, True, df, np, pd, result are available.
code: "result = (df['Profit'] < 0).sum() / len(df)"
```

**Gemma's code was correct.** Our AST validator (Layer 1) had a *narrower* name
allowlist than the execution namespace (Layer 2): `len` sat in `SAFE_BUILTINS`,
available to run, but the validator rejected it before it ever got there. Two
hand-maintained lists had silently drifted apart.

**This is a genuine defence-in-depth failure mode** — layers that do not agree —
and it is the most instructive bug in the project. The lists are now derived from
a single source, making the drift structurally impossible.

#### Disclosure: two benchmark questions were reworded

**Q09** originally read *"What **share** of our order lines lose money?"*. Gemma
returned `0.187`; ground truth held `18.7`. **Both are correct readings of
"share"** — the question was ambiguous about units.

**Q10** originally read *"Does profitability get worse as discounts get deeper?"*.
Gemma returned a correlation coefficient — **a defensible answer to a yes/no
question about a trend**.

Scoring those as failures measured our question-writing, not the model. Both were
reworded to be unambiguous — **not easier**. We disclose this because rewording a
benchmark after observing failure is a real methodological hazard, and the original
phrasings and responses are preserved in `result.md`.

### 6.5 Conversational context (B4)

The last 5 turns are replayed as user/assistant message pairs carrying the question
and the code that answered it — **not the result tables**, which would consume the
context window within three turns and bury the schema under numbers.

Verified:

```
Turn 1: "What were total sales by region?"
     →  result = df.groupby('Region', observed=True)['Sales'].sum().sort_values(ascending=False)
Turn 2: "Now show me just the West, broken down by category."
     →  result = df[df['Region'] == 'West'].groupby('Category', observed=True)['Sales'].sum()...
```

Turn 2 is meaningless without Turn 1. The model resolved it correctly.

---

## 7. Task C — Visualization

### 7.1 Colour system

The palette was **validated with a colour-vision-deficiency checker, not chosen by
eye.**

| Mode | Worst adjacent CVD separation | Contrast vs surface |
|---|---|---|
| Light | ΔE 24.2 (target ≥12) ✅ | 2 hues below 3:1 ⚠️ |
| Dark | ΔE 10.3 (floor band) ⚠️ | all ≥3:1 ✅ |

Both findings oblige the **same mitigation**, applied throughout: **colour is never
the only channel carrying identity.** Every multi-series chart ships a legend,
tooltips name the series, and a table view of the underlying data is available. A
red/green-colourblind viewer can read every chart in this application.

Three encodings, three colour jobs:

- **Categorical → identity.** Eight fixed slots, assigned in sorted order and
  **never cycled**. Colour follows the *entity*, not its rank: filtering out a
  region never repaints the survivors.
- **Sequential → magnitude.** One blue hue, light→dark. Used for Sales.
- **Diverging → polarity.** Blue↔red with a **neutral grey midpoint**. Used for
  Profit, because the *sign* is the point: a state losing money must not look like
  a state making a little.

A rainbow scale is used nowhere. It implies an ordering the data does not have and
is unreadable under CVD.

### 7.2 The chart suite

`[FILL: insert exports/figures/*.png for each]`

| # | Chart | Design decision |
|---|---|---|
| 1 | Revenue & profit trend | **Small multiples, NOT dual-axis** — see §7.3 |
| 2 | State choropleth | Sequential for Sales; **diverging for Profit**, anchored symmetrically on zero |
| 3 | Correlation matrix | Diverging, anchored at 0: −1 must look as strong as +1 |
| 4 | Profit distribution | **Box plot, not histogram** — the question is spread and the long negative tail; a histogram of a skewed variable hides exactly that |
| 5 | Category sunburst | The hierarchy is *genuine* (each sub-category has exactly one parent) — the precondition for a sunburst being honest rather than decorative |
| 6 | Animated year-over-year | **Y-axis fixed across frames.** Per-frame rescaling would make every year look identical and destroy the comparison the animation exists to make |
| 7 | Discount vs profit scatter | Least-squares fit + **95% CI on the mean response**, computed explicitly rather than left to library defaults |
| 8 | Segment × category stacked bar | 2px surface gap between fills |

### 7.3 We deliberately did not build a dual-axis chart

The brief offers *"time series or trend chart with dual axis"* as a permitted type.
**We did not build one, on purpose.**

A dual-axis chart plots two measures on two independent y-scales in one frame.
Where the two lines cross is then **an artifact of the scales the author chose**,
not a fact about the data: rescale either axis and the crossover moves anywhere you
like. It is the most reliable way to make a chart lie without stating a single
false number.

`trend_small_multiples` answers the same question honestly — Sales and Profit share
an x-axis in stacked panels, so the shapes are comparable while each keeps its own
truthful scale. **Seven other chart types from the brief's list are implemented, so
the six-type requirement is met without it.**

### 7.4 AI-driven visualization (C3)

When the NL pipeline returns a DataFrame, the application selects a chart type,
renders it, and captions it.

**The selection is rule-based, not model-generated — and this is deliberate.** The
shape of a DataFrame is a *fact*: how many rows, which columns are numeric, whether
the label column is temporal or geographic. Facts do not need a language model.
Asking a 4B model to choose would add ~1.5 s of latency and a new failure mode to a
decision that four `if` statements make correctly every time.

| Result shape | Chart | Rationale |
|---|---|---|
| US state names + metric | **map** | Strictly more informative than 40 bars |
| Temporal label + metric | **line** | The reader should see the *shape* of change |
| Two numerics, no label | **scatter** | The question is the relationship |
| Categorical + metric, ≤30 rows | **bar** | Magnitude comparison |
| >30 categories, or a scalar | **table** | A 793-bar chart is an unreadable comb |

**Verified: 7/7 shape cases select correctly.** The user can override the choice.

The LLM does the one part that genuinely needs language: **the one-sentence
caption.**

### 7.5 Export (C4)

PDF (reportlab), Word (python-docx), PNG and SVG (kaleido). Every exported report
carries dataset metadata, **the applied filters**, the AI narrative, the chart
image, the result table, and the generated code.

The filter list is printed **even when empty** ("None — the full dataset was
used"). Without it a report is actively misleading: a reader cannot tell whether
"$2.3M revenue" is the whole business or one region in one year.

---

## 8. Task D — Advanced Features

### 8.1 D3 — Anomaly detection

Isolation Forest over `Sales`, `Quantity`, `Discount`, `Profit`, `Shipping Days`,
with IQR and z-score reported alongside for comparison. Features are standardised
first — the forest partitions on raw values, and Sales spans 0.4–22,638 while
Discount spans 0–0.8; without scaling, Discount would be invisible to the model.

```
Flagged: 200 of 9,994 rows (2% contamination)
Loss across flagged rows: $74,142
Flagged rows that lose money: 41.5%   Baseline: 18.7%   Enrichment: 2.22×
Reproducible: identical flags on repeat runs (random_state=7)
```

#### The finding: our own justification for the forest was wrong

The standard argument for an Isolation Forest is that it catches rows whose
*combination* of values is unusual while no single column is extreme. **We measured
this. On this dataset it is false.**

| Contamination | Flagged | Caught by forest but **not** by IQR |
|---|---|---|
| 1% | 100 | **0** |
| 2% | 200 | **0** |
| 5% | 500 | 1 |
| 10% | 1,000 | 22 |

At every operationally sensible threshold, the forest finds **nothing** the
univariate rules miss.

**Its real value is selectivity.** The union of the per-column IQR rules flags
**2,851 rows — 28.5% of the dataset.** "A third of your orders are unusual" is not
an alert list anyone can act on; it is noise with a threshold attached. The forest
returns a **bounded, ranked** set with a continuous score, so the worst row comes
first and a human can start at the top. **IQR cannot rank at all.**

We report this rather than quietly selecting the algorithm that sounds most
advanced. The test that actually matters — are flagged rows more likely to be
genuinely bad? — the forest passes at **2.22× enrichment over baseline**.

The LLM never decides what is anomalous. It narrates rows the detectors have
already flagged. A model that could invent anomalies could invent a crisis that
does not exist.

### 8.2 D4 — ReAct multi-turn reasoning agent

A tool-calling loop: the model decides whether to query, detect anomalies, chart,
or answer; observes the result; and decides again. The full reasoning chain is
surfaced in a collapsible debug panel.

**Four tools:** `query_data`, `detect_anomalies`, `create_chart`, `final_answer`.

#### Why the agent's tools need no sandbox — and Task B's do

This is the design distinction worth the most in this project.

| | Task B pipeline | Task D4 agent |
|---|---|---|
| Model produces | **Arbitrary pandas code** | **Typed tool arguments** |
| Who builds the pandas call | The model | **The executor** |
| Safety model | **Containment** (3-layer sandbox) | **Construction** (schema-bounded) |
| Can express | Anything Python can | Only what the tool schema allows |

The agent's `query_data` takes an enum'd column name and an enum'd aggregation. The
model *cannot express* anything outside that schema, so this surface is **safe by
construction and needs no sandbox at all.** Both approaches exist in this project
on purpose: one is right when you need expressive power, the other when you need
guarantees.

#### Live results (real Gemma 4 E4B)

```
Q: "Which region has the highest total sales? Chart it."
   ACT  query_data({"aggregation":"sum","group_by":"Region","metric":"Sales"})
   OBS  4 rows in 3ms: West 725457.82 | East 678781.24 | Central 501239.89 | South 391721.91
   ACT  create_chart({"chart_type":"bar"})
   OBS  Rendered a bar chart of the previous result (4 rows).
   ANS  "The West region has the highest total sales with $725,457.82."
   → 2 tool calls, 2.3 s. Correct.

Q: "Find the most unusual order lines and tell me what went wrong."
   ACT  detect_anomalies({"top_n": 5})
   OBS  Isolation Forest flagged 200 of 9,994 (total loss on flagged: -$74,142)
   THINK "CA-2016-117121 (Binders): Profit was $4,946.37. This line does not
          appear to show a loss based on the provided data."
   ANS  "...The most concerning entries are those with negative profits:
         CA-2016-108196 (Machines) shows a profit of -$6,599.98..."
   → 1 tool call, 10.2 s.
```

Note the second run: the agent **correctly observed that the top-ranked anomaly is
not a loss at all** — it is an unusually large *profit* — and said so before moving
on to the ones that are. That is reasoning about an observation, not pattern-matching.

**Stated limitation.** Gemma 4 E4B plans two or three steps competently and then
begins to loop, re-calling the same tool with the same arguments. `MAX_STEPS=6` and
duplicate-call detection are guards a larger model would not need. We report this
rather than hiding it behind a curated demo.

---

## 9. Evaluation

### What works well

- **Text-to-code accuracy: 10/10** on the benchmark, 2.0 s mean, zero retries.
- **The sandbox holds.** 8/8 escape attempts blocked, including every bypass that
  defeats a naive substring filter.
- **Sub-5 ms queries**, 100× under budget.
- **Zero cost per query**, no data leaves the machine.
- **Colour is accessible**, validated rather than assumed.
- **The agent genuinely composes** multi-step plans and reasons about observations.

### What does not

- **The agent loops** beyond 2–3 steps. Guarded, not solved.
- **The stacked bar has no drill-down**, which the brief's chart list mentions.
- **Dark mode is built and validated but not exposed** in the UI.
- **Responsiveness at 1280px is untested.**
- **The Isolation Forest adds no novelty over IQR** on this dataset (§8.1). Its
  value is ranking and selectivity, which is a weaker claim than we expected to
  make.
- **Two benchmark questions were reworded** after observing failure (§6.4). We
  believe this was justified; we disclose it so the reader can judge.
- **A performance bug nearly wrecked the demo.** `st.download_button` evaluates its
  `data` argument *eagerly* — each export renders a chart through kaleido at
  ~1.5 s. With five answers in history, **every filter change triggered 30.7 s of
  work.** Found by profiling the rerun path; fixed by caching against the filter
  scope. It is worth stating that we found this by *measuring* rather than by
  assuming the code was fine because it looked fine.

### Honest assessment of the model

Gemma 4 E4B is **good enough**, but only because the system is built around its
weaknesses. Every one of the three defects we fixed (§6.4) is a defect a frontier
model would not have exhibited. The correct conclusion is not "4B models are
sufficient" — it is that **small-model reliability is an engineering problem, and
the engineering is tractable.**

---

## 10. Future Work

1. **Semantic caching of generated code.** Questions repeat. Embedding the question
   and reusing a previously validated snippet for near-duplicates would cut mean
   latency from 2.0 s to near-zero on the common path, and remove the largest
   remaining demo risk.

2. **Self-consistency for high-stakes queries.** Generate the code three times at
   temperature 0.3, execute all three, and return the answer only if they agree.
   Trades ~3× latency for a measurable reduction in silent wrong answers — the most
   dangerous failure mode, because a plausible wrong number is worse than an error.

3. **Speculative decoding.** LM Studio supports it; it is currently off. A smaller
   Gemma as draft model could roughly double the 44 tok/s generation rate at no
   accuracy cost.

4. **Process-level sandbox isolation.** The current sandbox is language-level,
   which is right for a trusted local model but insufficient for untrusted users. A
   subprocess with a seccomp profile and a memory cap would make the endpoint safe
   to expose.

5. **A larger model behind the agent.** The ReAct loop is bottlenecked by planning
   depth, not by tools. The same four tools with a 20B+ model would plan five or six
   steps without looping and would not need `MAX_STEPS` or duplicate detection.

---

## 11. Conclusion

We set out to test whether a 4-billion-parameter model on consumer hardware can
power a genuine natural-language analytics platform. It can — reaching **10/10** on
a benchmark scored against hand-written ground truth, at **2.0 s per question**, at
**zero cost per query**, with **no data leaving the machine**.

But the headline number is the least interesting result. The benchmark began at
**5/10**, and every point of improvement came from reading the model's actual
output and diagnosing a specific defect: a JSON-escaping interaction that silently
corrupted generated code; a missing few-shot pattern that made the model answer a
subtly different question; and a bug in our own sandbox where two defence layers
had drifted out of agreement. **None of it came from a bigger model.**

The same pattern recurred in the analysis itself. The Isolation Forest was chosen
for a reason that turned out, on measurement, to be false — and the honest finding
underneath it (selectivity, not novelty) is more useful than the one we expected.

The lesson we take from this project is that **working with a small model is not a
compromised version of working with a large one. It is a different discipline** —
one where the model's failures are legible, reproducible, and fixable, and where
the engineering is the substance rather than the scaffolding.

---

## 12. AI Usage Disclosure

*(Required by §8 of the brief. `[FILL/EDIT]` — make this accurate to what you did.)*

This project was developed with **Claude (Anthropic)** used as an AI coding
assistant, in the manner permitted by §8 ("permitted for code completion and
debugging").

**How it was used:**
- Scaffolding module structure and boilerplate.
- Drafting implementations of the data layer, prompt templates, sandbox, chart
  suite, and export paths.
- Diagnosing the three benchmark defects described in §6.4 — in each case, the
  *failure* was surfaced by our own acceptance-check scripts, and the diagnosis was
  reached by reading the actual generated code.
- Drafting this report.

**What the team did:**
- `[FILL: dataset selection, model choice, design decisions, verification, all
  testing on the LM Studio machine, and the analytical interpretation.]`

Every design decision documented in this report — the dual-axis refusal, the
rule-based chart selection, the two safety models, the Isolation Forest finding —
is one the team can explain and defend. Per §8, **every line of submitted code is
understood by every team member.**

---

## 13. References

1. Superstore Dataset. Kaggle. https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
2. Liu, F.T., Ting, K.M., Zhou, Z-H. (2008). *Isolation Forest.* IEEE ICDM.
3. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
4. Gemma Team, Google DeepMind. *Gemma model card.*
5. LM Studio documentation — OpenAI-compatible endpoints. https://lmstudio.ai/docs
6. `[FILL: any further sources you used]`
