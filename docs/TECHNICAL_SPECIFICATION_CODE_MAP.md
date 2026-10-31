# Technical implementation specification: process and file map

## Context and architecture

This application processes **synthetic, historical customer snapshots**, not production customer event streams. The architecture is a research-driven model comparison and human review dashboard. One prediction is issued per eligible customer at the first of a month. The previous 90 days are observed; the target is customer cancellation or failed renewal over the next 30 days. `churn_label=1` means churned; `0` means retained. The model sees only 28 historical features.

### Data layer

- Main source: `data/customer_churn_1800_dataset.csv` (1,800 rows; 32 fields).
- Target audit: `data/customer_churn_1800_explicit_labels.csv` (DO NOT use to build predictors).
- Dictionary: `data/customer_churn_data_dictionary.csv`.
- Generator: `src/generate_dataset_reference.py` (seed fixed; regenerates those files).
- EDA: `src/eda_report.py`; outputs `eda_quality_report.json`, `eda_report.md`, `eda_overview.png`, `eda_segments.csv`, `eda_monthly.csv`.

### Prediction model

- `src/train_and_evaluate.py`: eligibility and data quality assertions, chronological partitions, training-only preprocessing, Logistic Regression / Decision Tree / XGBoost, March sigmoid probability calibration, April–May threshold selection, June–August test; saved `joblib` bundles.
- `src/tune_xgboost.py`: optional 2-fold pre-test historical PR-AUC search; no final test data used for parameter search. Does not silently modify the production demonstration model.
- Model input fields: 28 predictors (all CSV columns except `customer_id`, `prediction_date`, `data_split`, `churn_label`).
- Score output: `risk_probability∈[0,1]`, `risk_percentage`, `risk_band`, `flag_at_validation_threshold` and `monthly_revenue_exposure_proxy_usd` (proxy only).
- Batch inference: `src/score_new_customers.py:score_frame()`; rejects outcome-labelled input and missing required columns, checks >=3 month tenure.
- Explanations: native TreeSHAP from XGBoost `pred_contribs=True` grouped to source fields. Numerical additivity to raw margins is checked.

### Prediction / evaluation chronology

| Part | Dates | n | Allowed work |
|---|---|---:|---|
| Train | Jan 2025–Jan 2026 | 1,170 | Fit preprocessing/classifiers |
| Purge | Feb 2026 | 90 | Exclude from fitting |
| Calibration | Mar 2026 | 90 | Fit probability calibrators |
| Validation | Apr–May 2026 | 180 | Choose risk threshold under 30% validation capacity |
| Held-out test | Jun–Aug 2026 | 270 | Evaluation only |

The code asserts temporal cutoffs; the label for date `t` cannot be used earlier than `t+30 days`. Each customer ID occurs exactly once in the synthetic demo.

### Service / application layer

- `app.py`: Streamlit premium interface; filter by industry/plan, select threshold or top-k, view metrics, inspect local/global explanations, segments and model evaluation; upload label-free CSV and export ranked review queue.
- `src/build_offline_dashboard.py`: static HTML file with embedded **retrospective test** data, JavaScript threshold/top-k behavior, queue and charts. No production scoring in browser; new record scoring requires Python model bundle.
- `outputs/retention_priority_queue.csv`: ranks customers by XGBoost score, contains no `churn_label`.
- `outputs/test_scored_with_ground_truth.csv`: evaluation-only CSV includes true labels; must not be shared with human retention reviewers as predictions.

## Acceptance rules by functional requirement

| ID | Input | Computation | Expected output / acceptance check |
|---|---|---|---|
| FR-01 | Historic snapshot and label definition | Pre-cutoff eligibility, target horizon | 0/1 labels, ≥3 tenure, unique IDs, no nulls |
| FR-02 | 28 past features | Model pipeline + calibrator | Probabilities in [0,1]; no post-cutoff fields |
| FR-03 | Dated cohorts | 5-part purge/calibration/validation/test | Train end +30d precedes calibration; validation end +30d precedes test |
| FR-04 | User threshold or top-k budget | `p≥t` or exactly first k in descending risk order | Report flagged, TP/FP/FN if ground truth exists |
| FR-05 | Fitted XGBoost + input features | TreeSHAP grouped to originals | Global `.csv`, customer positive/protective signals; additivity check |
| FR-06 | Risk predictions and historical evidence | Stable sorting by probability then ID | Queue with customer, segment, usage change, billing, explanations; **no label** |
| FR-07 | Sorted test probabilities and labels | precision@k, recall@k, lift@k | 10/20/30/50% capacity table and graph |
| FR-08 | Test predictions and segments | TP/FP/FN and segment precision/recall | `segmented_test_performance.csv` |
| FR-09 | Code + data + dependencies | `python run_all.py`, `pytest` | Deterministic artifacts and 6 smoke tests passing |

## HCLTech expected deliverable artifacts

| Deliverable | Source | Generated files |
|---|---|---|
| D-01 Working scoring prototype | `src/train_and_evaluate.py`, `src/score_new_customers.py`, `app.py` | Model `.joblib` bundles, scored CSV |
| D-02 EDA and drivers | `src/eda_report.py`, native TreeSHAP | `eda_report.md`, `.png`, `eda_*.csv`, `global_risk_drivers.csv` |
| D-03 Chronological evaluation | Train/evaluate + `src/make_figures.py` | `evaluation_report.md/json`, PR/ROC, reliability, lift, threshold and segment tables |
| D-04 Retention review dashboard | `app.py`, `src/build_offline_dashboard.py` | Label-free queue, interactive HTML, filters and top-k/threshold controls |
| D-05 Technical documentation | README, dictionary, code, tests | This specification, reproducibility scripts, warnings and evidence |

## Mathematical definitions

For any threshold `t` and test sample: `flag_i = 1[p_i >= t]`.

- `Precision = TP / (TP + FP)`.
- `Recall (churn capture) = TP / (TP + FN)`.
- `F1 = 2PR/(P+R)`.
- `PR-AUC`: implemented as `average_precision_score` (stepwise average precision), not trapezoidal interpolation.
- `ROC-AUC`: rank-discrimination metric.
- `Brier = mean((p_i - y_i)^2)`, lower is better; evaluate calibration bins too.
- `Top-k`: rank descending by `p_i`, stable tie-break by `customer_id`, select `k = ceil(capacity_pct*n)`.
- `Precision@k = churners_in_top_k / k`.
- `Capture@k = churners_in_top_k / all_churners`.
- `Lift@k = Precision@k / observed_positive_rate`.
- `Risk-weighted monthly charge exposure proxy = sum(p_i × monthly_charges_i)` for selected review rows; NOT retained revenue.

## Limitations / safeguards

The dataset has a deliberately balanced synthetic 50/50 outcome. Natural churn prevalence is unknown; test and calibration metrics are **not deployment-grade**. The dataset contains one snapshot per unique ID (not repeat longitudinal observations), so business drift or causal intervention effects cannot be quantified. SHAP explains model association, not cause. No protected attributes should be added for discriminatory prioritization. Retention actions are human decisions; this system does not trigger communications or account changes.
