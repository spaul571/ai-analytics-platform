D:\2.EDUCATION\1.2 SUMMER-2026\CSE-638_Deep Learning\Final Assignment + Presentation + 1 Quiz\capstone>python -m scripts.check_task_a
======================================================================
TASK A ACCEPTANCE CHECK
======================================================================

[A1] Loaded 9,994 rows x 24 columns
     read_csv:  33.7 ms
     cleaning:  59.0 ms
     memory:    9.21 MB raw -> 1.89 MB after dtype downcast (79.4% saved)

[A1] Cleaning steps applied:
     - Parsed 'Order Date' from string to datetime64.
     - Parsed 'Ship Date' from string to datetime64.
     - Derived 'Order Year' and 'Order Month' from Order Date.
     - Derived 'Shipping Days' as Ship Date minus Order Date.
     - Downcast low-cardinality text columns to pandas 'category' dtype.

[A1] Schema as the LLM will receive it:
----------------------------------------------------------------------
DATASET: Global E-Commerce Sales (Superstore) (9,994 rows)
CONTEXT: Retail e-commerce order transactions. Each row is one product line item within a customer order. Sales is gross revenue in USD; Profit is net margin in USD and can be negative when discounts are too deep.

COLUMNS:
- Row ID (numeric, int64)
  range: 1 to 9994
- Order ID (categorical, str)
  desc: Unique order identifier; one order can contain several rows.
  examples: CA-2016-152156, CA-2016-152156, CA-2016-138688
- Order Date (datetime, datetime64[us])
  desc: Date the customer placed the order.
  range: 2014-01-03 00:00:00 to 2017-12-30 00:00:00
- Ship Date (datetime, datetime64[us])
  desc: Date the order shipped.
  range: 2014-01-07 00:00:00 to 2018-01-05 00:00:00
- Ship Mode (categorical, category)
  desc: Delivery speed chosen by the customer.
  values: First Class, Same Day, Second Class, Standard Class
- Customer ID (categorical, category)
  desc: Unique customer identifier.
  examples: CG-12520, CG-12520, DV-13045
- Customer Name (categorical, category)
  desc: Customer full name.
  examples: Claire Gute, Claire Gute, Darrin Van Huff
- Segment (categorical, category)
  desc: Customer type: Consumer, Corporate, or Home Office.
  values: Consumer, Corporate, Home Office
- Country (categorical, category)
  desc: Country of the shipping address.
  values: United States
- City (categorical, category)
  desc: City of the shipping address.
  examples: Henderson, Henderson, Los Angeles
- State (categorical, category)
  desc: State or province of the shipping address.
  examples: Kentucky, Kentucky, California
- Postal Code (numeric, int64)
  desc: Postal code of the shipping address.
  range: 1040 to 99301
- Region (categorical, category)
  desc: Sales region grouping several states.
  values: Central, East, South, West
- Product ID (categorical, category)
  desc: Unique product identifier.
  examples: FUR-BO-10001798, FUR-CH-10000454, OFF-LA-10000240
- Category (categorical, category)
  desc: Top-level product category.
  values: Furniture, Office Supplies, Technology
- Sub-Category (categorical, category)
  desc: Product sub-category nested inside Category.
  values: Accessories, Appliances, Art, Binders, Bookcases, Chairs, Copiers, Envelopes, Fasteners, Furnishings, Labels, Machines, Paper, Phones, Storage, Supplies, Tables
- Product Name (categorical, category)
  desc: Product title.
  examples: Bush Somerset Collection Bookcase, Hon Deluxe Fabric Upholstered Stacking Chairs, Rounded Back, Self-Adhesive Address Labels for Typewriters by Universal
- Sales (numeric, float64)
  desc: Gross revenue for the line item, in USD.
  range: 0.444 to 22638.48
- Quantity (numeric, int64)
  desc: Units sold in the line item.
  range: 1 to 14
- Discount (numeric, float64)
  desc: Discount rate applied, as a fraction from 0.0 to 1.0.
  range: 0.0 to 0.8
- Profit (numeric, float64)
  desc: Net profit for the line item, in USD. Negative means a loss.
  range: -6599.978 to 8399.976
- Order Year (numeric, int32)
  range: 2014 to 2017
- Order Month (categorical, category)
  examples: 2016-11, 2016-11, 2016-06
- Shipping Days (numeric, int64)
  range: 0 to 7
----------------------------------------------------------------------

[A3] Completeness:   100.00%
[A3] Duplicate rows: 0
[A3] IQR outliers:   4,074

[A2/A4] Sample queries
----------------------------------------------------------------------
OK  Q1 direct aggregation - total sales and profit per region
        5.0 ms | 4 rows returned
 Region   Sales_sum  Profit_sum  Order ID_count
   West 725457.8245 108418.4489            3203
   East 678781.2400  91522.7800            2848
Central 501239.8908  39706.3625            2323
  South 391721.9050  46749.4303            1620

OK  Q2 direct aggregation - mean discount by sub-category
        2.7 ms | 10 rows returned
Sub-Category  Discount_mean  Profit_mean
     Binders       0.372292    19.843574
    Machines       0.306087    29.432669
      Tables       0.261285   -55.565771
   Bookcases       0.211140   -15.230509
      Chairs       0.170178    43.095894

OK  Q3 filtered query - 2017 furniture sales by state
        4.3 ms | 10 rows returned
       State  Sales_sum  Sales_mean  Quantity_sum
  California 40674.3705  290.531218           524
    New York 23624.1790  306.807519           259
  Washington 18536.3420  386.173792           169
       Texas 16010.8342  285.907754           199
Pennsylvania 15325.4510  312.764306           189

OK  Q4 filtered query - heavily discounted loss-making lines
        3.8 ms | 8 rows returned
       Category Sub-Category  Profit_sum  Profit_mean  Order ID_count
Office Supplies      Binders -38510.4964   -62.822996             613
      Furniture       Tables -30761.1238  -176.788068             174
     Technology     Machines -30118.6682  -684.515186              44
      Furniture    Bookcases -11097.7614  -158.539449              70
Office Supplies   Appliances  -8629.6412  -128.800615              67

OK  Q5 whole-table aggregation - no grouping
        0.5 ms | 1 rows returned
   Sales_sum  Sales_mean  Profit_sum  Discount_mean
2297200.8603  229.858001 286397.0217       0.156203

----------------------------------------------------------------------
[A4] Slowest query: 5.0 ms (budget 500 ms) -> PASS



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
  FAIL Q01  Which region brings in the most revenue? (retried)
          2.6s | cols=Y | exec=N | acc=N | fmt=N
        error: SandboxViolation: Generated code is not valid Python: invalid syntax
  PASS Q02  What is our total profit margin as a percentage of sales?
          1.5s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q03  Show me the five sub-categories with the deepest average markdown.
          2.2s | cols=Y | exec=Y | acc=Y | fmt=Y
  RAN  Q04  Which sub-categories are actually losing us money?
          1.7s | cols=Y | exec=Y | acc=N | fmt=Y
        code: result = df[df['Profit'] < 0].groupby('Sub-Category', observed=True)['Profit'].sum().sort_
  PASS Q05  How did technology sales trend year by year?
          1.9s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q06  Who are our top 5 buyers by total spend?
          1.5s | cols=Y | exec=Y | acc=Y | fmt=Y
  PASS Q07  Compare average delivery time across the different shipping speeds.
          1.6s | cols=Y | exec=Y | acc=Y | fmt=Y
  RAN  Q08  In 2017, which state had the highest furniture sales?
          1.9s | cols=Y | exec=Y | acc=N | fmt=Y
        code: result = df[(df['Order Year'] == 2017) & (df['Category'] == 'Furniture')].groupby('State',
  RAN  Q09  What share of our order lines lose money?
          1.6s | cols=Y | exec=Y | acc=N | fmt=Y
        code: result = df[df['Profit'] < 0].shape[0] / df.shape[0]
  RAN  Q10  Does profitability get worse as discounts get deeper?
          2.1s | cols=Y | exec=Y | acc=N | fmt=Y
        code: correlation = df['Discount'].corr(df['Profit']) * -1

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
  Executed  9/10  (90%)
  Accurate  5/10  (50%)
  Format    9/10  (90%)
  Retried   1/10  (auto-correction was needed)
  Mean time 1.9s per question

Wrote benchmarks/results.csv - paste this table into the report (Task B5).
