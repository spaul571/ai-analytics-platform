(.venv) D:\2.EDUCATION\1.2 SUMMER-2026\CSE-638_Deep Learning\Final Assignment + Presentation + 1 Quiz\capstone>python -m scripts.check_task_d
============================================================================
TASK D ACCEPTANCE CHECK
============================================================================

[D3] Anomaly detection
----------------------------------------------------------------------------
  Isolation Forest flagged 200 of 9,994 rows (contamination=2%)
  Total loss across flagged rows: $-74,142

  Is the detector any good?
    flagged rows that lose money :  41.5%
    baseline (any row)           :  18.7%
    enrichment                   :   2.22x (1.0 would mean it is picking at random)

  Selectivity — the reason the forest is used over IQR:
    Isolation Forest flags 200 rows (2.0% of the data), ranked worst first. The union of the per-column IQR rules flags 2,851 (28.5%), unranked — too many to action.

  Detector overlap:
    both forest and IQR :   200
    forest only         :     0
    IQR only            :  2651

  Univariate counts per column:
    column               IQR     z>3
    Sales               1167     127
    Quantity             170     113
    Discount             856     300
    Profit              1881     107
    Shipping Days          0       0

  Five most anomalous order lines:
      Order ID Sub-Category     Sales  Quantity  Discount    Profit  anomaly_score
CA-2016-117121      Binders  9,892.74        13      0.00  4,946.37          -0.77
CA-2015-116638       Tables  4,297.64        13      0.40 -1,862.31          -0.76
CA-2016-108196     Machines  4,499.98         5      0.70 -6,599.98          -0.75
US-2017-168116     Machines  7,999.98         4      0.50 -3,839.99          -0.75
CA-2016-118689      Copiers 17,499.95         5      0.00  8,399.98          -0.75

  OK: flagged rows are 2.22x likelier to be loss-making than average.

  OK: detection is reproducible (identical flags on a second run).

[D4] Agent tool schemas
----------------------------------------------------------------------------
  4 tools registered: query_data, detect_anomalies, create_chart, final_answer
  OK: every tool has a description and a typed parameter schema.

[D4] LM Studio: Connected to google/gemma-4-e4b at http://localhost:1234/v1

[D4] ReAct agent (live)
----------------------------------------------------------------------------

  Q: Which region has the highest total sales? Chart it.
  ACT    query_data({"aggregation": "sum", "group_by": "Region", "metric": "Sales"})
  obs    4 rows in 3ms:
            Region   Sales_sum
              West 725457.8245
              East 678781.2400
           Central 501239.8908
             South 391721.9050
  ACT    create_chart({"chart_type": "bar"})
  obs    Rendered a bar chart of the previous result (4 rows). It is shown to the user.
  ANSWER The West region has the highest total sales with $725,457.82. This information is presented in the bar chart above.

     2 tool calls, 2.3s, ok

  Q: Find the most unusual order lines and tell me what went wrong.
  ACT    detect_anomalies({"top_n": 5})
  obs    Isolation Forest flagged 200 of 9,994 order lines (total loss on flagged rows: $-74,142).
           The 5 worst:
                 Order ID Sub-Category     Sales  Discount     Profit
           CA-2016-117121      Binders  9892.740       0.0  4946.3700
           CA-2015-116638       Tables  4297.644       0.4 -1862.3124
           CA-2016-108196     Machines  4499.985       0.7 -6599.9780

  think  The anomaly detection flagged a total loss of -$74,142 across 200 order lines.

           Looking at the top 5 most anomalous order lines:
           *   **CA-2016-117121 (Binders):** Sales were $9,892.74, Discount was $0.0, and Profit was $4,946.37. This line does not appear to show a loss based on the provided data snippet.
           *   **CA-2015-116638 (Tables):** Sales were $4,29
  ANSWER The anomaly detection flagged a total loss of -$74,142 across 200 order lines. The most unusual and concerning entries are those with negative profits, which indicate a financial loss on the sale. Specifically:
           *   Order **CA-2016-108196** (Machines) shows a profit of -$6,599.98.
           *   Order **US-2017-168116** (Machines) shows a profit of -$3,839.99.

           Thes

     1 tool calls, 10.2s, ok

============================================================================
ALL CHECKS PASS