D:\2.EDUCATION\1.2 SUMMER-2026\CSE-638_Deep Learning\Final Assignment + Presentation + 1 Quiz\capstone>python -m scripts.check_task_b
==============================================================================
TASK B ACCEPTANCE CHECK
==============================================================================

[B2] Sandbox - escape attempts (all must be BLOCKED)
------------------------------------------------------------------------------
  BLOCKED  direct import          | Disallowed syntax: Import
  BLOCKED  dunder import          | Disallowed name: '__import__'
  BLOCKED  builtins traversal     | Disallowed attribute access: '__subclasses__'
  BLOCKED  getattr indirection    | Disallowed builtin: 'getattr'
  BLOCKED  eval injection         | Disallowed builtin: 'eval'
  BLOCKED  file write             | Disallowed builtin: 'open'
  BLOCKED  globals access         | Disallowed builtin: 'globals'
  BLOCKED  no result assigned     | Generated code must assign its answer to a variable named 'result'
  ALLOWED  legitimate query       | correctly permitted

[B5] LM Studio health check: Connected to google/gemma-4-e4b at http://localhost:1234/v1

[B1/B2/B5] Benchmark - 10 natural language questions
------------------------------------------------------------------------------
  PASS Q01  Which region brings in the most revenue?
          2.2s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q02  What is our total profit margin as a percentage of sales?
          1.5s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q03  Show me the five sub-categories with the deepest average markdown.
          1.8s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q04  Which sub-categories are actually losing us money?
          2.4s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q05  How did technology sales trend year by year?
          2.4s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q06  Who are our top 5 buyers by total spend?
          1.9s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q07  Compare average delivery time across the different shipping speeds.
          2.2s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q08  In 2017, which state had the highest furniture sales?
          2.2s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q09  What percentage of our order lines lose money?
          1.8s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q10  Show me the average profit at each discount level.
          2.0s | cols=Y | exec=Y | acc=Y | fmt=Y

[B4] Conversational context
------------------------------------------------------------------------------
  Turn 1: What were total sales by region?
          -> result = df.groupby('Region', observed=True)['Sales'].sum().sort_values(ascending=False)
  Turn 2: Now show me just the West, broken down by category.
          -> result = df[df['Region'] == 'West'].groupby('Category', observed=True)['Sales'].sum().sort_values(ascending=False)

  Follow-up resolved against history: YES
  Memory holds 2 turns (max 5)
  After reset: 0 turns

==============================================================================
BENCHMARK SUMMARY
==============================================================================
  Columns   10/10  (100%)
  Executed  10/10  (100%)
  Accurate  10/10  (100%)
  Format    10/10  (100%)
  Retried   0/10  (auto-correction was needed)
  Mean time 2.0s per question

Wrote benchmarks/results.csv - paste this table into the report (Task B5).