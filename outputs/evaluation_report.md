# Chronological test evaluation — synthetic customer churn

**IMPORTANT:** This report evaluates 1,800 synthetic, exactly balanced samples; it does not validate real-world retention outcomes.

## Label and temporal safeguards

Churn = cancellation / failed renewal during days +1 to +30 after prediction date. Pre-cutoff historical observation = previous 90 days.

| Partition | Dates | Rows | Churned |
|---|---|---:|---:|
| train_pre_cutoff | 2025-01-01 to 2026-01-01 | 1170 | 585 |
| purged_gap | 2026-02-01 to 2026-02-01 | 90 | — |
| calibration | 2026-03-01 to 2026-03-01 | 90 | 45 |
| threshold_validation | 2026-04-01 to 2026-05-01 | 180 | 90 |
| heldout_test | 2026-06-01 to 2026-08-01 | 270 | 135 |

February 2026 is **purged** from model fitting because its +30-day label is not available at March 1. March labels calibrate probabilities; April–May select risk threshold; June–August is untouched until final evaluation.

## Test results (independent chronological test, n=270)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Test TP | FP | FN | Selected threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.618 | 0.646 | 0.649 | 0.356 | 0.459 | 0.236 | 48 | 26 | 87 | 0.599 |
| LogisticRegression | 0.701 | 0.679 | 0.716 | 0.430 | 0.537 | 0.225 | 58 | 23 | 77 | 0.584 |
| DecisionTree | 0.428 | 0.393 | 0.333 | 0.148 | 0.205 | 0.253 | 20 | 40 | 115 | 0.509 |

**PR-AUC is average precision (AP).** A 50% prevalence gives an AP no-skill baseline of 0.50; this baseline will change for a realistic class distribution.

## Capacity and lift (XGBoost)

| Capacity | Top k | Churn captured | Precision@k | Recall@k | Lift@k |
|---|---:|---:|---:|---:|---:|
| 10% | 27 | 19 | 0.704 | 0.141 | 1.407 |
| 20% | 54 | 36 | 0.667 | 0.267 | 1.333 |
| 30% | 81 | 52 | 0.642 | 0.385 | 1.284 |
| 50% | 135 | 81 | 0.600 | 0.600 | 1.200 |

## Top global risk signals (mean absolute tree contribution to raw log odds)

1. `usage_change_pct` — 0.368
2. `renewal_due_in_days` — 0.213
3. `engagement_change_pct` — 0.180
4. `contract_type` — 0.130
5. `tenure_months` — 0.124
6. `usage_index_prev_30d` — 0.098
7. `billing_issues_90d` — 0.094
8. `total_billed_to_date_usd` — 0.088
9. `usage_index_recent_30d` — 0.081
10. `late_payments_90d` — 0.076

## Caveats

- Balanced synthetic set: 50% event prevalence, which distorts apparent lift/precision and probability calibration relative to natural churn prevalence.
- Review capacity and flagged true churners are based on observed synthetic test outcomes; none are proven to be saved by outreach.
- Revenue exposure is a monthly charge proxy, not observed prevented revenue or causal treatment effect.
- No future event fields in model; separate explicit outcome labels CSV must never be joined to predictors.
- Customer-level SHAP values are XGBoost uncalibrated margin (log-odds) contributions, not direct contributions in percent risk.
- Single synthetic cohort per customer: do not interpret time trends as longitudinal individual change.
- Comparative metrics are measured on this dataset, not the paper data; paper performance values are not reproduced.
- Small test (270) and calibration (90) groups: uncertainty is substantial and simulated performance may not generalize.
