# ATLAS: HCLTech Customer Churn Prediction & Retention Prioritization

**End-to-end runnable Python implementation + editable Excalidraw technical design.**

Based on your supplied **HCLTech Future Ready Data Science Hackathon** brief and the supplied research paper, *AI-based Customer Churn Prediction System using XGBoost for Early Retention and Revenue Optimization* (ICSCSS, 2026). The paper inspires the XGBoost-vs-baseline comparison; the HCLTech functional requirements determine the additional time-aware evaluation, calibrated risk, top-k prioritization, explainability and dashboard functionality.

**Important:** This is a synthetic, intentionally 50%-balanced dataset, **not actual customer data**. No observed retention effectiveness or real revenue saved is available. This prototype supports a human review queue, not automatic intervention. The research paper's published accuracy is NOT a result of this code.

## Quick start

From the extracted project root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt


# Complete rebuild: EDA + models + explainability + all evaluations + offline UI + validation
python run_all.py

# Verify acceptance checks
python -m pytest -q

# Launch the premium dashboard in browser (requires installed Streamlit)
streamlit run app.py
```

The **offline dashboard** also works without a Python web server: open `outputs/offline_retention_dashboard.html` in a browser. Its sliders and filters use the saved test scores and labels for retrospective *synthetic* review metrics.

For new customers, use the **Score New Customers** page of the Streamlit app or the CLI:

```bash
python src/score_new_customers.py --input samples/new_customers.csv --output outputs/my_scores.csv
```

The input has 28 pre-prediction features plus optional `customer_id` and `prediction_date`. **Do not include labels, outcomes or future-event fields.** New predictions have freshly computed individual XGBoost TreeSHAP explanations. The output contains no churn target.

### Optional research reproduction

```bash
# Search 8 XGBoost hyperparameter settings on two historical pre-test folds
python src/tune_xgboost.py

# Optional: deterministically regenerate the main synthetic data CSVs
# WARNING: overwrites data/*.csv; run run_all.py again afterwards
python src/generate_dataset_reference.py
```

The optional search saves recommended settings but does not silently change the already specified model. To adopt those settings, edit `build_models()` in `src/train_and_evaluate.py` and rerun `run_all.py`. **Never choose settings by repeatedly evaluating the held-out test set.**

## Architecture and process

```text
Synthetic generator / supplied customer snapshots
         |
         v
Validate eligibility, binary target, nulls, duplicates, pre-cutoff features
         |
         +---> EDA and data quality reports (D-02)
         |
         v
Time partitions: train -> purged gap -> calibrate -> threshold validation -> test
         |
         v
Feature preprocessing (train-only encoders / scaler)
         |
         v
Logistic regression | Decision tree | XGBoost (paper-inspired)
         |
         v
Calibrated 0..1 probabilities (March only) -> chosen threshold (Apr-May only)
         |
         v
Frozen June-Aug held-out test: AP, ROC-AUC, F1, calibration, segment errors
         |
         +---> Local and global XGBoost TreeSHAP (uncalibrated log-odds)
         |
         v
Risk-sorted label-free review queue | threshold & top-k capacity analysis
         |
         v
Premium Streamlit dashboard | portable interactive offline HTML dashboard
```

**Chronology:** Prediction point is 00:00 on the first of each month; the prior **90 days** provide features; churn label is cancellation/failure to renew on **days +1 through +30**. Eligible customers have `tenure_months >= 3`, are active at prediction, and are represented by one snapshot per customer. The synthetic generation assumptions are documented in `data/customer_churn_data_dictionary.csv`.

| Purpose | Months / records | Training role |
|---|---|---|
| Train | Jan 2025–Jan 2026 / **1,170** | Preprocessing and classifier fit |
| Purge | Feb 2026 / **90** | Excluded to prevent immature-label use |
| Calibrate | March 2026 / **90** | Platt-style probability calibration |
| Threshold validation | Apr–May 2026 / **180** | Capacity-constrained cutoff selection |
| Final held-out test | Jun–Aug 2026 / **270** | Evaluate only after configuration is frozen |

`data_split` in the source CSV groups Feb as original TRAIN and Mar–May as VALIDATION; `sets_for_time()` applies the stricter leakage-safe subdivisions shown above. `customer_id`, `prediction_date`, `data_split` and `churn_label` are excluded as model predictors. The explicit outcome CSV is **audit-only** and must never be joined into the input feature matrix.

## Files and what each one does

| File | Purpose |
|---|---|
| `data/customer_churn_1800_dataset.csv` | 1,800-row, 32-column historical synthetic dataset |
| `data/customer_churn_1800_explicit_labels.csv` | Outcome label and window evidence, **never model input** |
| `data/customer_churn_data_dictionary.csv` | Fields, meaning and timestamp availability |
| `src/generate_dataset_reference.py` | Optional deterministic synthetic-data generator |
| `src/eda_report.py` | Completeness, balance, EDA group summaries, charts and report |
| `src/train_and_evaluate.py` | Chronological splitting, pipelines, calibration, metrics, SHAP, queue, CSV reports, saved models |
| `src/tune_xgboost.py` | Optional time-safe historical parameter search |
| `src/score_new_customers.py` | CLI and reusable batch scoring with **fresh** customer SHAP |
| `src/make_figures.py` | ROC/PR, calibration, lift, model comparison, risk drivers figures |
| `src/build_offline_dashboard.py` | Standalone interactive HTML, recalculating selection metrics |
| `src/validate_outputs.py` | Saved-output validation, target-isolation and capacity checks |
| `app.py` | Premium working Streamlit app: cohort filters, threshold/top-k slider, queue, model reports, customer signals, upload, data quality |
| `run_all.py` | One-command orchestration of the complete workflow |
| `tests/test_pipeline.py` | Six automated acceptance smoke tests |
| `design/*.excalidraw` | Four editable concept and UI design files (not application source) |
| `samples/new_customers.csv` | 12 label-free customer examples for batch scoring |
| `outputs/` | Reproduced model artifacts, full reports, figures, HTML, scores and exports |

## Nine functional requirements — verification / source map

| Requirement | Code and output |
|---|---|
| **FR-01** Churn definition & eligible population | `src/generate_dataset_reference.py`; `data/customer_churn_data_dictionary.csv`; `src/eda_report.py`; `src/train_and_evaluate.py:validate()` |
| **FR-02** Pre-cutoff churn scoring | `src/train_and_evaluate.py:build_models()`; `src/score_new_customers.py:score_frame()`; `outputs/model_XGBoost.joblib` |
| **FR-03** Chronological validation | `src/train_and_evaluate.py:sets_for_time()`; Feb purge; `outputs/evaluation_report.json` |
| **FR-04** Threshold & error trade-off | `src/train_and_evaluate.py:select_threshold()`; `outputs/threshold_tradeoffs.csv`; Streamlit sliders |
| **FR-05** Global/local explanations | `src/train_and_evaluate.py:explain_xgb()`; fresh explanations from `src/score_new_customers.py`; `outputs/global_risk_drivers.csv` |
| **FR-06** Priority review queue | `outputs/retention_priority_queue.csv`; `app.py:ranked_frame()` and label-free CSV export |
| **FR-07** Capture, top-k and lift | `src/train_and_evaluate.py:topk_metrics()`; `outputs/capacity_lift.csv`; dashboard |
| **FR-08** Segment error analysis | `src/train_and_evaluate.py:segmented()`; `outputs/segmented_test_performance.csv` |
| **FR-09** Reproducible process | `run_all.py`, `requirements.txt`, `tests/test_pipeline.py`, this README, data dictionary, `outputs/evaluation_report.md` |

## Five expected HCLTech deliverables

- **D-01 — Churn scoring prototype:** `app.py`, trained model `outputs/model_XGBoost.joblib`, `src/score_new_customers.py` and sample CSV.
- **D-02 — Exploratory and driver analysis:** `outputs/eda_report.md`, `outputs/eda_overview.png`, `outputs/eda_*.csv`, `outputs/global_risk_drivers.csv`.
- **D-03 — Chronological evaluation report:** `outputs/evaluation_report.md/json`, PR/ROC, calibration, capacity/lift, threshold trade-offs and segment reports.
- **D-04 — Retention view:** `app.py`, `outputs/offline_retention_dashboard.html` and label-free `outputs/retention_priority_queue.csv`; editable designs in `design/`.
- **D-05 — Technical documentation:** this README, `docs/TECHNICAL_SPECIFICATION_CODE_MAP.md`, data dictionary, exact split assumptions, code and tests.

## Models and real execution on this synthetic dataset

The pre-specified primary XGBoost model uses 240 trees, depth 3, learning rate 0.045, subsample/column sample 0.85, min child weight 4 and L2 regularization 4. Two baselines are Logistic Regression (standardized features) and Decision Tree (depth 5, min leaf 22). All categorical fields are one-hot encoded **using training data only**.

| Model | Held-out AP/PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.645 | 0.666 | 0.643 | 0.400 | 0.493 |
| Logistic Regression | **0.701** | **0.679** | **0.716** | **0.430** | **0.537** |
| Decision Tree | 0.428 | 0.393 | 0.333 | 0.148 | 0.205 |

These metrics came from executing the included pipeline on **June–August 2026** synthetic test rows, after cutoff selection on validation. Logistic Regression performs better here. The paper's published **93.2% accuracy** belongs to a different data analysis and must not be claimed as our reproduced result.

For the XGBoost model, top **20%** capacity selects **54/270** test customers and captures **38/135** actual churners (precision@k 70.4%, recall@k 28.1%, lift 1.41×). This is a sample-specific result. A threshold tuned under a 30% validation capacity budget is *not* guaranteed to select ≤30% on the test cohort; use **top-k** to enforce strict capacity.

### Explainability, calibration and business guardrails

- TreeSHAP grouped across one-hot categories explains the uncalibrated XGBoost **log-odds margin**, not an additive percentage point change in final calibrated probability.
- Risk drivers are correlations/associations only. They do not establish churn causes.
- Revenue figures are *risk-weighted monthly-charge exposure proxies*. No proven retained customers or real avoided loss is measured.
- A what-if retention success slider is labeled as an **assumption**, not measured uplift. True uplift requires a randomized or otherwise causal study.
- Due to the artificial 50% positive sample, **risk probabilities are not production-calibrated** and must not be used for operational decisions without natural-prevalence validation/recalibration.
- No protected attributes are in this dataset. The prototype never automates outreach, offer/contract changes, nor price changes.
- Model file loading uses `joblib` and should be restricted to **trusted artifacts** only.

## Acceptance tests and project status

The data/model pipeline, EDA generation, offline dashboard generation, saved-output checks, new scoring and optional tuning were **executed successfully** on the provided data. `pytest` passed **6/6** tests. `app.py` was syntax-checked; you can launch the Streamlit application after installing the requirements. The Excalidraw files are **editable design artifacts**, not functional web pages by themselves.

## Source documents (supplied by participant, not bundled)

1. HCLTech, *Customer Churn Prediction and Retention Prioritization — Problem Statement and Participant Brief*.
2. *AI-based Customer Churn Prediction System using XGBoost for Early Retention and Revenue Optimization*, ICSCSS 2026, DOI `10.1109/ICSCSS69635.2026.11645828`.
