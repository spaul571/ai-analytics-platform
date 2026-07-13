"""Task B acceptance check and benchmark (milestone M3, report table for B5).

Runs the 10 benchmark questions through the full 3-phase pipeline and scores the
model against hand-written ground truth on four axes:

    columns   - did it identify the right columns?          (Task B1)
    executed  - did the generated code run in the sandbox?  (Task B2)
    accuracy  - does the answer match ground truth?         (Task B5)
    format    - is the narrative non-empty Markdown?        (Task B5)

Also exercises the sandbox against a set of escape attempts, and demonstrates
conversational context with a follow-up question (Task B4).

Run:  python -m scripts.check_task_b
"""

from __future__ import annotations

import sys

import pandas as pd

from benchmarks.questions import QUESTIONS, BenchmarkQuestion
from src.data.loader import load_dataset
from src.llm.client import LLMClient
from src.llm.pipeline import NLQueryPipeline, PipelineResult
from src.llm.sandbox import SandboxViolation, validate

# Relative tolerance when comparing floats. Generated code may sum in a
# different order than the ground truth, so exact equality is too strict.
TOLERANCE = 0.01

# Escape attempts the sandbox must reject. Each one defeats a naive substring
# blocklist, which is the point of demonstrating them.
ATTACKS: list[tuple[str, str]] = [
    ("direct import", "import os\nresult = os.listdir('.')"),
    ("dunder import", "result = __import__('os').listdir('.')"),
    ("builtins traversal", "result = ().__class__.__base__.__subclasses__()"),
    ("getattr indirection", "result = getattr(df, 'to_csv')('/tmp/leak.csv')"),
    ("eval injection", "result = eval('__import__(\"os\").system(\"whoami\")')"),
    ("file write", "result = open('leak.txt', 'w').write('data')"),
    ("globals access", "result = globals()"),
    ("no result assigned", "df.groupby('Region')['Sales'].sum()"),
]


def _candidate_series(frame: pd.DataFrame) -> list[pd.Series]:
    """Every plausible reading of a result frame as a labelled Series.

    A result like `Order Year | Sales` is ambiguous to a naive reader: both
    columns are numeric, so the label column cannot be identified by dtype.
    Rather than guess, produce one candidate per numeric column (keyed on the
    first column) and let the caller accept a match against any of them. This
    also tolerates the model returning extra metric columns alongside the one
    that was asked for.
    """
    candidates: list[pd.Series] = []
    if frame.shape[1] < 2:
        return candidates

    index = frame.iloc[:, 0].astype(str).values
    for col in frame.columns[1:]:
        if pd.api.types.is_numeric_dtype(frame[col]):
            candidates.append(pd.Series(frame[col].astype(float).values, index=index))
    return candidates


def _as_scalar(value) -> float | None:
    """Read a scalar answer out of whatever shape the model returned."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Series) and len(value) == 1:
        return float(value.iloc[0])
    if isinstance(value, pd.DataFrame):
        numeric = value.select_dtypes("number")
        if numeric.shape == (1, 1):
            return float(numeric.iloc[0, 0])
    return None


def _series_matches(actual: pd.Series, expected: pd.Series) -> bool:
    """Order-insensitive comparison keyed on the group labels."""
    a = actual.copy()
    e = expected.copy()
    a.index = a.index.astype(str)
    e.index = e.index.astype(str)
    common = e.index.intersection(a.index)
    # The model may legitimately return more rows than asked for (all
    # sub-categories when 5 were requested, say); require only that every
    # ground-truth row is present and correct.
    if len(common) < len(e):
        return False
    tolerance = TOLERANCE * e[common].abs().clip(lower=1.0)
    return bool(((a[common] - e[common]).abs() <= tolerance).all())


def _matches(actual, expected) -> bool:
    """Compare a pipeline result against ground truth."""
    if actual is None:
        return False

    # Scalar ground truth (a ratio or a percentage).
    if isinstance(expected, (int, float)):
        got = _as_scalar(actual)
        return got is not None and abs(got - float(expected)) <= TOLERANCE * max(
            abs(float(expected)), 1.0
        )

    if isinstance(expected, pd.DataFrame):
        expected = expected.iloc[:, -1]

    if not isinstance(expected, pd.Series):
        return False

    expected = expected.astype(float)

    if isinstance(actual, pd.Series):
        return _series_matches(actual.astype(float), expected)

    if isinstance(actual, pd.DataFrame):
        return any(
            _series_matches(candidate, expected)
            for candidate in _candidate_series(actual)
        )

    return False


def _columns_hit(code: str, question: BenchmarkQuestion) -> bool:
    """Did the generated code reference every column the question needs?"""
    return all(col in code for col in question.expected_columns)


def run_sandbox_checks() -> int:
    """Verify every escape attempt is rejected before execution. Returns failures."""
    print("\n[B2] Sandbox - escape attempts (all must be BLOCKED)")
    print("-" * 78)
    failures = 0
    for name, code in ATTACKS:
        try:
            validate(code)
        except SandboxViolation as exc:
            reason = str(exc).split(".")[0]
            print(f"  BLOCKED  {name:22s} | {reason}")
        else:
            print(f"  LEAKED   {name:22s} | *** VALIDATION PASSED - THIS IS A BUG ***")
            failures += 1

    # A legitimate query must still be allowed, or the sandbox is useless.
    legit = 'result = df.groupby("Region", observed=True)["Sales"].sum()'
    try:
        validate(legit)
        print(f"  ALLOWED  {'legitimate query':22s} | correctly permitted")
    except SandboxViolation as exc:
        print(f"  BROKEN   {'legitimate query':22s} | *** {exc} ***")
        failures += 1

    return failures


def run_benchmark(pipeline: NLQueryPipeline, df: pd.DataFrame) -> list[dict]:
    """Run all 10 questions and score them."""
    print("\n[B1/B2/B5] Benchmark - 10 natural language questions")
    print("-" * 78)

    rows: list[dict] = []
    for q in QUESTIONS:
        # Each question is scored independently, so history is not replayed here.
        result: PipelineResult = pipeline.ask(q.question, use_history=False)

        try:
            expected = q.ground_truth(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  {q.id}  ground truth itself failed: {exc}")
            continue

        actual = result.data if result.success else None
        accurate = bool(result.success and _matches(actual, expected))
        columns_ok = _columns_hit(result.code, q)
        format_ok = bool(result.success and result.answer.strip())

        status = "PASS" if accurate else ("RAN " if result.success else "FAIL")
        retry = " (retried)" if result.retried else ""
        print(f"  {status} {q.id}  {q.question}{retry}")
        print(f"        {result.total_seconds:5.1f}s | cols={'Y' if columns_ok else 'N'} "
              f"| exec={'Y' if result.success else 'N'} "
              f"| acc={'Y' if accurate else 'N'} "
              f"| fmt={'Y' if format_ok else 'N'}")
        if not result.success:
            print(f"        error: {result.error}")
            print(f"        code:  {result.code!r}")  # repr, to expose stray fences
        elif not accurate:
            print(f"        code: {result.code.replace(chr(10), ' ; ')[:110]}")

        rows.append(
            {
                "ID": q.id,
                "Question": q.question,
                "Columns": "Y" if columns_ok else "N",
                "Executed": "Y" if result.success else "N",
                "Accurate": "Y" if accurate else "N",
                "Format": "Y" if format_ok else "N",
                "Retried": "Y" if result.retried else "N",
                "Seconds": round(result.total_seconds, 1),
                "Code": result.code.replace("\n", " ; "),
            }
        )

    return rows


def demo_context(pipeline: NLQueryPipeline) -> None:
    """Demonstrate Task B4: a follow-up that only resolves against history."""
    print("\n[B4] Conversational context")
    print("-" * 78)
    pipeline.memory.reset()

    first = "What were total sales by region?"
    print(f"  Turn 1: {first}")
    r1 = pipeline.ask(first)
    print(f"          -> {r1.code}")

    # "that" and "just the West" are meaningless without the prior turn.
    follow_up = "Now show me just the West, broken down by category."
    print(f"  Turn 2: {follow_up}")
    r2 = pipeline.ask(follow_up)
    print(f"          -> {r2.code}")

    resolved = "West" in r2.code and "Category" in r2.code
    print(f"\n  Follow-up resolved against history: {'YES' if resolved else 'NO'}")
    print(f"  Memory holds {len(pipeline.memory)} turns (max {pipeline.memory.max_turns})")

    pipeline.memory.reset()
    print(f"  After reset: {len(pipeline.memory)} turns")


def main() -> int:
    print("=" * 78)
    print("TASK B ACCEPTANCE CHECK")
    print("=" * 78)

    # The sandbox needs no LLM, so check it first - it is the one part that can
    # be verified even when LM Studio is unreachable.
    sandbox_failures = run_sandbox_checks()

    client = LLMClient()
    ok, message = client.health_check()
    print(f"\n[B5] LM Studio health check: {message}")
    if not ok:
        print("\nCannot run the benchmark without the model. Sandbox results above "
              "are still valid.")
        return 1

    df, schema, _ = load_dataset()
    pipeline = NLQueryPipeline(df, schema, client=client)

    rows = run_benchmark(pipeline, df)
    demo_context(pipeline)

    table = pd.DataFrame(rows)
    total = len(table)
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)
    for axis in ("Columns", "Executed", "Accurate", "Format"):
        hits = int((table[axis] == "Y").sum())
        print(f"  {axis:9s} {hits}/{total}  ({100 * hits / total:.0f}%)")
    retried = int((table["Retried"] == "Y").sum())
    print(f"  {'Retried':9s} {retried}/{total}  (auto-correction was needed)")
    print(f"  {'Mean time':9s} {table['Seconds'].mean():.1f}s per question")

    out = "benchmarks/results.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out} - paste this table into the report (Task B5).")

    accurate = int((table["Accurate"] == "Y").sum())
    return 0 if sandbox_failures == 0 and accurate >= 7 else 1


if __name__ == "__main__":
    sys.exit(main())
