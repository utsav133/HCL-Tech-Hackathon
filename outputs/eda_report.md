# Exploratory data analysis and quality report

Dataset: synthetic, deliberately balanced; descriptive associations are **not causal**.

- Records: **1,800**; columns: **32**; distinct IDs: **1800**.
- Missing cells: **0**; duplicate IDs: **0**.
- Labels: **900 churned**, **900 retained**; churn rate: **50.0%** (artificial).
- Monthly prediction snapshots: **2025-01-01** through **2026-08-01**.

## Segmented descriptive churn

| Field | Segment | Customers | Churn rate |
|---|---|---:|---:|
| industry_segment | Retail | 600 | 50.0% |
| industry_segment | Subscription | 600 | 50.0% |
| industry_segment | Telecom | 600 | 50.0% |
| plan_tier | Basic | 600 | 50.0% |
| plan_tier | Premium | 600 | 50.0% |
| plan_tier | Standard | 600 | 50.0% |
| contract_type | Annual | 502 | 38.4% |
| contract_type | Month-to-month | 1014 | 57.0% |
| contract_type | Quarterly | 284 | 45.4% |
| customer_segment | Established | 974 | 52.8% |
| customer_segment | Loyal | 671 | 45.3% |
| customer_segment | New | 155 | 52.9% |
| payment_method | Bank Transfer | 507 | 44.4% |
| payment_method | Cash/Manual | 157 | 58.0% |
| payment_method | Credit Card | 698 | 51.7% |
| payment_method | Digital Wallet | 438 | 50.9% |
| auto_pay_enabled | 0 | 623 | 52.0% |
| auto_pay_enabled | 1 | 1177 | 48.9% |

## Reproducible EDA outputs

- `eda_overview.png`: label counts, monthly churn, usage distribution, billing mix.
- `eda_monthly.csv`: monthly cohort counts.
- `eda_segments.csv`: segment churn rates.
- `eda_numeric_by_churn.csv`: pre-cutoff numeric feature summaries by outcome.
- `eda_quality_report.json`: cleaning, uniqueness and explicit-label checks.

## Limitations

- Exact 50/50 target balance is imposed by the synthetic generator; population churn prevalence is unknown.
- Each customer has one prediction snapshot, not a time series of the same customer.
- No protected characteristics are generated. Do not infer causal impact from the segments.
- Avoid using outcome, snapshot date or split identity as model features.
