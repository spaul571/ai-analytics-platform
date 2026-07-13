"""Task D acceptance check.

D3 (anomaly detection) runs entirely offline: the detectors are scikit-learn and
pandas, and only the narration needs the model. D4 (the ReAct agent) needs LM
Studio, and is skipped with a clear message when it is unreachable.

Run:  python -m scripts.check_task_d
"""

from __future__ import annotations

import sys

from src.advanced.agent import TOOLS, ReActAgent
from src.advanced.anomaly import CONTAMINATION, detect, narrate
from src.data.loader import load_dataset
from src.llm.client import LLMClient


def check_anomalies(df) -> int:
    print("\n[D3] Anomaly detection")
    print("-" * 76)

    report = detect(df)

    print(f"  Isolation Forest flagged {report.flagged_count} of {len(report.scored):,} "
          f"rows (contamination={CONTAMINATION:.0%})")
    print(f"  Total loss across flagged rows: ${report.total_loss:,.0f}")

    print("\n  Is the detector any good?")
    print(f"    flagged rows that lose money : {report.flagged_loss_rate:6.1%}")
    print(f"    baseline (any row)           : {report.baseline_loss_rate:6.1%}")
    print(f"    enrichment                   : {report.enrichment:6.2f}x "
          "(1.0 would mean it is picking at random)")

    print("\n  Selectivity — the reason the forest is used over IQR:")
    print(f"    {report.selectivity}")

    print("\n  Detector overlap:")
    print(f"    both forest and IQR : {report.both:5d}")
    print(f"    forest only         : {report.forest_only:5d}")
    print(f"    IQR only            : {report.iqr_only:5d}")

    print("\n  Univariate counts per column:")
    print(f"    {'column':<16}{'IQR':>8}{'z>3':>8}")
    for col in report.iqr_counts:
        print(f"    {col:<16}{report.iqr_counts[col]:>8}{report.zscore_counts[col]:>8}")

    print("\n  Five most anomalous order lines:")
    worst = report.flagged.head(5)[
        ["Order ID", "Sub-Category", "Sales", "Quantity", "Discount", "Profit", "anomaly_score"]
    ]
    print(worst.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    failures = 0
    if report.flagged.empty:
        print("\n  FAIL: nothing was flagged.")
        failures += 1
    # The detector has to beat chance. Novelty over IQR is NOT the bar - the
    # sweep showed the forest finds nothing IQR misses at usable thresholds, and
    # that is documented rather than papered over. Enrichment is the real test.
    if report.enrichment < 1.5:
        print(f"\n  FAIL: enrichment {report.enrichment:.2f}x — barely better than random.")
        failures += 1
    else:
        print(f"\n  OK: flagged rows are {report.enrichment:.2f}x likelier to be "
              "loss-making than average.")

    # Determinism: the demo must flag the same rows twice. random_state is set,
    # so a second run must be identical.
    again = detect(df)
    if not report.flagged["Order ID"].equals(again.flagged["Order ID"]):
        print("\n  FAIL: detection is not reproducible across runs.")
        failures += 1
    else:
        print("\n  OK: detection is reproducible (identical flags on a second run).")

    return failures


def check_agent_schema() -> int:
    print("\n[D4] Agent tool schemas")
    print("-" * 76)
    names = [t["function"]["name"] for t in TOOLS]
    print(f"  {len(TOOLS)} tools registered: {', '.join(names)}")

    failures = 0
    if len(TOOLS) < 3:
        print("  FAIL: the brief requires at least 3 tools.")
        failures += 1
    for tool in TOOLS:
        fn = tool["function"]
        params = fn["parameters"]
        if not fn.get("description"):
            print(f"  FAIL: {fn['name']} has no description.")
            failures += 1
        if "properties" not in params:
            print(f"  FAIL: {fn['name']} has no parameters schema.")
            failures += 1
    if not failures:
        print("  OK: every tool has a description and a typed parameter schema.")
    return failures


def check_agent_live(df, schema, client) -> int:
    print("\n[D4] ReAct agent (live)")
    print("-" * 76)

    agent = ReActAgent(df, schema, client=client)

    questions = [
        "Which region has the highest total sales? Chart it.",
        "Find the most unusual order lines and tell me what went wrong.",
    ]

    failures = 0
    for question in questions:
        print(f"\n  Q: {question}")
        result = agent.run(question)

        for step in result.steps:
            marker = {
                "thought": "  think ",
                "action": "  ACT   ",
                "observation": "  obs   ",
                "answer": "  ANSWER",
                "error": "  ERROR ",
            }[step.kind]
            body = step.content.replace("\n", "\n           ")
            print(f"{marker} {body[:400]}")

        print(f"\n     {result.tool_calls} tool calls, {result.total_seconds:.1f}s, "
              f"{'ok' if result.success else 'STOPPED EARLY'}")

        if not result.success:
            failures += 1
        elif result.tool_calls == 0:
            print("     FAIL: the agent answered without calling any tool.")
            failures += 1

    return failures


def main() -> int:
    print("=" * 76)
    print("TASK D ACCEPTANCE CHECK")
    print("=" * 76)

    df, schema, _ = load_dataset()
    failures = check_anomalies(df)
    failures += check_agent_schema()

    client = LLMClient()
    ok, message = client.health_check()
    print(f"\n[D4] LM Studio: {message}")
    if ok:
        failures += check_agent_live(df, schema, client)
    else:
        print("\n  Skipping the live agent run. D3 results above are unaffected —")
        print("  the detectors are scikit-learn, not the model.")

    print("\n" + "=" * 76)
    print(f"FAILURES: {failures}" if failures else "ALL CHECKS PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
