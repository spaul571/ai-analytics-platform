# Presentation — AI-Powered Data Analytics Platform

**15 minutes + 5 minutes Q&A. Worth 5 of 13 marks.**

> Marking: Live Demo 1.5 · Technical Explanation 1.5 · Insight Quality 1.0 ·
> Written Report 0.5 · Q&A 0.5
>
> **The two 1.5-mark criteria are Demo and Technical Explanation. Budget your time
> accordingly: demo it working, then explain how it works. Do not spend eight
> minutes on the dataset.**

---

## Timing plan

| Slide | Content | Time |
|---|---|---|
| 1 | Title | 0:30 |
| 2 | The problem | 1:00 |
| 3 | Architecture | 1:30 |
| 4–7 | **LIVE DEMO** | **5:00** |
| 8 | The sandbox (deep) | 2:00 |
| 9 | 5/10 → 10/10 | 2:00 |
| 10 | The anomaly finding | 1:30 |
| 11 | Limitations | 1:00 |
| 12 | Conclusion | 0:30 |
| | **Total** | **15:00** |

---

## Slide 1 — Title

# Natural-language analytics on a 4B model
### Can a model that runs on one consumer GPU replace the analyst queue?

**Team No. 5** — Shrikanta Paul · Md. Nurol Amin · Animesh Dey ·
Kazi Meherunnesa Eva · Fardin Ahmed Alvi · Anika Tahsin Prova
CSE-638 Deep Learning
Dataset: Global E-Commerce Sales (9,994 order lines)
Model: **Gemma 4 E4B**, local via LM Studio · **$0 per query**

---

## Slide 2 — The problem

**The people who need answers cannot write pandas.**

A category manager wants to know which products lose money. Answering takes ten
seconds of an analyst's time — but the question waits two days in a queue.

LLMs collapse that queue. But every demo you have seen uses a frontier hosted
model, which means:

- per-query cost
- rate limits
- **your commercial data leaves the building**

> **Our question: can a 4-billion-parameter model on one GPU do this reliably
> enough to be useful?**

**Answer: yes — but only with engineering a frontier model would not need. That
engineering is the substance of this project.**

---

## Slide 3 — Architecture

```
     Gemma 4 E4B ── LM Studio ── localhost:1234 (OpenAI-compatible)
                          │
    ┌─────────────────────┴──────────────────────┐
    │  INTELLIGENCE                              │
    │  schema + synonyms + few-shots             │
    │  ┌──────────┐  ┌─────────┐  ┌───────────┐  │
    │  │ P1 code  │─▶│ P2 EXEC │─▶│ P3 narrate│  │
    │  │   gen    │  │ SANDBOX │  │           │  │
    │  └──────────┘  └────┬────┘  └───────────┘  │
    │                     └── error → 1 retry    │
    └─────────────────────┬──────────────────────┘
                          │
        DATA (pandas, 1.89 MB, sub-5 ms queries)
                          │
        UI (Streamlit, 5 tabs, 8 charts, PDF/DOCX export)
```

**Talk track (30 s):** "Three phases. The model writes pandas, we execute it in a
sandbox, then the model narrates the *result* — never the raw data, which is what
stops it inventing numbers. If execution fails, the exception goes back to the
model for exactly one auto-retry."

---

## Slides 4–7 — LIVE DEMO (5 minutes) ⭐ 1.5 MARKS

> **Rehearse this until it is boring. Nothing else in the presentation is worth as
> much per minute.**
>
> **Pre-flight checklist:**
> - [ ] LM Studio: Status **Running**, model loaded, **Context 16384**, **Enable Thinking OFF**
> - [ ] `streamlit run app.py` already started, browser open, **sidebar shows "Model ready"**
> - [ ] Conversation history cleared
> - [ ] Zoom to 110% — assessors are at the back of the room
> - [ ] **Have `result3.md` open in a second tab** in case the model has a bad day

### Demo 1 — Overview tab (45 s)

Show headline metrics: **$2,297,201 revenue · 12.5% margin · 1,871 loss-making
lines.**

Click **Anomaly & Outlier Report**. The narrative generates live.

> "These figures are computed in pandas. The model only *narrates* numbers we hand
> it — it never writes the code for the front page. A 4B model writing the headline
> numbers would make our Overview tab a coin flip."

### Demo 2 — Ask a question (90 s)

Type: **"Which sub-categories are losing us money?"**

Expected: `totals = df.groupby('Sub-Category', observed=True)['Profit'].sum()` then
`result = totals[totals < 0].sort_values()`

Open **"How this was answered"**. Show the generated code and the timings.

> "Note what it did *not* do. It did not filter the loss-making rows first — that
> would throw away the profitable sales in the same sub-category and answer a
> different question. It aggregates, *then* filters. It got that wrong on our first
> benchmark run. Slide 9."

### Demo 3 — Follow-up (45 s) — proves B4

Type: **"Now just the West, by category."**

> "That question is meaningless on its own. It resolves only against the previous
> turn. We replay the last five exchanges as the model's own prior working."

### Demo 4 — The agent (90 s) ⭐

Agent tab → **"Which region has the highest total sales? Chart it."**

Open the **Reasoning chain** panel.

> "Two tool calls. It queried, *looked at the result*, decided a chart was needed,
> then answered. Two point three seconds.
>
> And here is the part that matters: **this agent needs no sandbox.** Its tools take
> typed arguments — an enum'd column, an enum'd aggregation — and *our executor*
> builds the pandas call, not the model. It cannot express anything outside the
> schema. That is safety by **construction**. The AI Assistant tab you just saw is
> safety by **containment**, because there the model writes arbitrary code. Both are
> in this project on purpose."

### Demo 5 — Export (30 s)

Click **PDF report**. Open it.

> "Dataset metadata, the applied filters, the AI narrative, the chart, the result
> table, and the generated code. The filters are printed even when empty — without
> them, a reader cannot tell whether '$2.3M revenue' is the whole business or one
> region in one year."

---

## Slide 8 — The sandbox ⭐ 1.5 MARKS (technical explanation)

**We execute code a language model wrote. That is an arbitrary-code-execution sink.**

### Why a blocklist fails

```python
if "import" in code:  reject     # defeated by:  __import__("os")
if "os" in code:      reject     # defeated by:  getattr(__builtins__, "ev"+"al")
```

Both are one-liners. **A substring filter is not a security boundary.**

### Three independent layers

| Layer | Mechanism |
|---|---|
| **1. AST allowlist** | Every syntax node checked against a permitted set. **Any identifier starting with `_` is rejected** — one rule that kills the entire `().__class__.__base__.__subclasses__()` traversal family. Unknown node types **fail closed**. |
| **2. Restricted namespace** | `__builtins__` replaced with ~25 safe functions. No `open`, no `__import__` to reach. |
| **3. Timeout** | 10 s budget on a worker thread. |

### Verified

```
BLOCKED  direct import       BLOCKED  eval injection
BLOCKED  dunder import       BLOCKED  file write
BLOCKED  builtins traversal  BLOCKED  globals access
BLOCKED  getattr indirection BLOCKED  no result assigned
ALLOWED  legitimate query    ← still works
```

**8/8.**

**Limitation, stated:** this is a *language-level* sandbox, not OS-level. Right for
a local model we prompt ourselves; **insufficient for untrusted users** — that needs
a container or seccomp jail.

---

## Slide 9 — 5/10 → 10/10 ⭐ THE CORE STORY

| Run | Accurate | What we fixed |
|---|---|---|
| 1 | **5/10** | — |
| 2 | **9/10** | Single quotes; aggregate-then-filter few-shot |
| 3 | **10/10** | Our own sandbox bug |

> ### **We never changed the model.**

### Defect 1 — structured output *corrupts* generated code

```python
result = df.groupby("Region", observed=True)"Sales".sum()
                                            ^^^^^^^^
                                    brackets are GONE
```

We request JSON output → the code lives in a JSON string → `["Sales"]` must be
emitted as `[\"Sales\"]` → **the model fumbled the escaping and dropped the
brackets.**

**The tell:** this was the **only** question where it chose double quotes, and the
**only** syntax error in the set.

**Fix: constrain it to single quotes.** We did not fix the escaping — we made
escaping *unnecessary*, eliminating the failure class.

### Defect 2 — it answered a different question

```python
df[df['Profit'] < 0].groupby('Sub-Category')['Profit'].sum()   # WRONG
```
Filters loss-making *rows* first → discards profitable sales in the same
sub-category. Whether a group loses money **on net** can only be known *after*
summing. **Fix: one targeted few-shot example.**

### Defect 3 — our own sandbox's two layers disagreed

```
SandboxViolation: Unknown name: 'len'
code: result = (df['Profit'] < 0).sum() / len(df)      ← Gemma was RIGHT
```

`len` was in the *execution namespace* but not in the *validator's allowlist*. Two
hand-maintained lists drifted apart.

> **This is a real defence-in-depth failure mode: layers that do not agree.** Now
> derived from one source.

### Disclosure

Two benchmark questions were **reworded** after we saw them fail. Q09 asked for a
"share" and got `0.187` while we expected `18.7` — **both correct readings.** We
disclose this because rewording a benchmark after observing failure is a real
methodological hazard.

---

## Slide 10 — The anomaly finding ⭐ 1.0 MARK (insight quality)

### The business finding

**Discounting is destroying margin — and it is structural, not incidental.**

| Sub-category | Total profit | Mean discount | Loss-making lines |
|---|---|---|---|
| **Binders** | **−$38,510** | **37%** | 613 |
| **Tables** | **−$30,761** | 26% | 174 |
| **Machines** | **−$30,118** | 31% | 44 |

**Tables lose money on average across *all* sales — mean profit −$55/line.** That is
not a run of unlucky orders; it is a pricing policy that does not work.

### The methodological finding — and it contradicts us

We chose an Isolation Forest because it should catch rows whose *combination* is
odd while no column is extreme. **We measured it. That is false here.**

| Contamination | Flagged | Caught by forest but NOT by IQR |
|---|---|---|
| 1% | 100 | **0** |
| 2% | 200 | **0** |
| 5% | 500 | 1 |

**It finds nothing IQR misses.**

**But IQR flags 28.5% of the dataset.** "A third of your orders are unusual" is not
an alert list — it is noise with a threshold.

> ### The forest's real value is **selectivity and ranking**:
> **200 rows, scored, worst-first. IQR cannot rank at all.**
> Flagged rows are **2.22× likelier to lose money** than random.

**We report this instead of quietly picking the algorithm that sounds most
advanced.**

---

## Slide 11 — Limitations (say these before they ask)

| | |
|---|---|
| **The agent loops** | Gemma plans 2–3 steps then re-calls the same tool. `MAX_STEPS` + duplicate detection are guards a bigger model would not need. |
| **No dual-axis chart** | **Deliberate.** A dual y-axis puts the crossover wherever you choose the scales — the most reliable way to make a chart lie without stating a false number. We built shared-x small multiples instead. Seven other listed chart types are implemented. |
| **Forest adds no novelty over IQR** | Slide 10. Ranking, not detection. |
| **A perf bug nearly killed this demo** | `st.download_button` evaluates `data` **eagerly** — 6.1 s of kaleido rendering *per answer, per rerun*. With 5 answers, every filter change cost **30.7 s**. Found by profiling, not by reading. Now cached. |
| **Dark mode built, not exposed. 1280px untested.** | Honest gaps. |

---

## Slide 12 — Conclusion

**10/10 · 2.0 s per question · $0 per query · no data leaves the machine.**

But the number is the least interesting result.

> ### It started at 5/10.
> ### Every point came from a diagnosed defect, not a bigger model.

- A JSON-escaping interaction that silently corrupted code
- A missing few-shot that made the model answer a *different question*
- A bug in **our own sandbox** where two defence layers disagreed

And when we measured our own algorithm choice, **the justification turned out to be
false** — and the honest finding underneath was more useful.

> **Working with a small model is not a compromised version of working with a large
> one. It is a different discipline — one where the failures are legible,
> reproducible, and fixable.**

---

# Q&A — likely questions, prepared answers ⭐ 0.5 MARKS

> **§7.3: "Using LLM-generated code without understanding it — Q&A will probe this
> directly."** Every team member must answer these without notes.

**Q: How does your sandbox stop `__import__("os")`?**
> The AST validator rejects any identifier beginning with an underscore, so
> `__import__` never parses through. And `Import` nodes are not in the allowlist at
> all. A substring check for "import" would be defeated by exactly that payload —
> which is why we don't use one.

**Q: What if the model writes `().__class__.__base__.__subclasses__()`?**
> Blocked by the same single rule — `__class__` starts with an underscore. That one
> rule kills the whole traversal family. We test it; it's in our escape suite.

**Q: Why not just use GPT-4 / Claude?**
> It would have scored 10/10 immediately and taught us nothing. The constraint is
> the point: local, free, private, and it forced us to actually diagnose failures
> rather than paper over them with capacity.

**Q: Your benchmark went 5→10. Did you tune to the test?**
> Partly, and we disclose exactly how. Two of the ten questions were reworded
> because they were genuinely ambiguous and the model's answers were defensible —
> that's in the report. The other three fixes were real defects: a JSON-escaping
> bug, a missing few-shot, and a bug in our own sandbox. The original run and the
> failing code are preserved in `result.md`.

**Q: Why is there no dual-axis chart? The brief lists one.**
> Because it's the most misleading chart form in common use. The crossover point is
> an artifact of the scales you pick — rescale either axis and it moves anywhere you
> want. We built shared-x small multiples, which answers the same question honestly.
> We still implement seven of the eight listed types.

**Q: Why Isolation Forest if it finds nothing IQR misses?**
> Because IQR flags 28.5% of the dataset and can't rank. The forest gives a bounded,
> scored, worst-first list of 200. We measured both — the enrichment over baseline
> is 2.22×. We'd rather tell you the justification we expected turned out false than
> pretend otherwise.

**Q: How do you know the model isn't making numbers up?**
> Three ways. The narration phase only ever sees the *computed result table*, never
> the raw DataFrame. The preset insights use hand-written aggregations — the model
> only narrates. And the benchmark scores against hand-written ground truth, so
> "accurate" means it matched an implementation we wrote ourselves.

**Q: What happens when the generated code fails?**
> The exception is fed back to the model with the failing code for exactly one
> auto-retry. One, not two — if it can't fix a hallucinated column name given the
> literal `KeyError`, a second attempt doesn't help and the user waits twice as long
> for the same failure.

**Q: Why does the agent not need a sandbox?**
> Its tools take typed arguments — an enum'd column, an enum'd aggregation — and our
> executor builds the pandas call. The model can't express anything the schema
> doesn't allow. That's safety by construction. Task B is safety by containment,
> because there the model writes the code. Different problems, different tools.

**Q: What would you do next?**
> Semantic caching of generated code — questions repeat, and it would take mean
> latency from 2 seconds to near zero. And self-consistency for high-stakes queries:
> generate three times, execute all three, only answer if they agree. A plausible
> wrong number is more dangerous than an error message.

---

## Final pre-flight

- [ ] LM Studio running, context 16384, Thinking OFF, **tested on venue wifi**
- [ ] App started, model-ready indicator green
- [ ] Demo rehearsed end-to-end **at least three times**
- [ ] `result.md` / `result3.md` open in a spare tab (fallback evidence)
- [ ] Every member can answer the sandbox question cold
- [ ] Report PDF submitted
