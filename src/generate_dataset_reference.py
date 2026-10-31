#!/usr/bin/env python3
"""Deterministic, entirely synthetic churn dataset for the HCLTech hackathon.

Run: python src/generate_dataset_reference.py
Requires: numpy. Data is synthetic, NOT empirical/real customer data.
"""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data"
SEED = 20261031
RNG = np.random.default_rng(SEED)
INDUSTRIES = ("Retail", "Telecom", "Subscription")
PLANS = ("Basic", "Standard", "Premium")
MONTHS = [(2025 + m // 12, m % 12 + 1) for m in range(20)]

FIELDS = [
 "customer_id", "prediction_date", "data_split", "industry_segment",
 "customer_segment", "plan_tier", "contract_type", "tenure_months",
 "monthly_charges_usd", "total_billed_to_date_usd", "payment_method",
 "auto_pay_enabled", "billing_issues_90d", "late_payments_90d",
 "failed_payments_90d", "support_tickets_90d",
 "unresolved_tickets_at_cutoff", "avg_ticket_resolution_hours_90d",
 "csat_score_last90d", "plan_suitability_score", "usage_index_prev_30d",
 "usage_index_recent_30d", "usage_change_pct", "engagement_sessions_prev_30d",
 "engagement_sessions_recent_30d", "engagement_change_pct",
 "days_since_last_activity", "renewal_due_in_days", "recent_plan_change_90d",
 "active_discount_pct", "prior_retention_contact_90d", "churn_label"
]
LABEL_FIELDS = [
 "customer_id", "prediction_date", "observation_start_date",
 "observation_end_date", "label_window_start_date", "label_window_end_date",
 "churn_label", "churn_label_text", "churn_event_type"
]

DESCRIPTIONS = {
 "customer_id": ("Identifier", "Unique synthetic customer key; one row per customer, no repeats.", "Identifier only; exclude from model."),
 "prediction_date": ("Date", "First day of a month when a churn forecast is made.", "Prediction-time metadata; use for chronological split, not a shortcut to outcome."),
 "data_split": ("Category", "TRAIN Jan 2025-Feb 2026; VALIDATION Mar-May 2026; TEST Jun-Aug 2026.", "Partition metadata; exclude as model feature."),
 "industry_segment": ("Category", "Business industry: Retail, Telecom, or Subscription.", "Observed by cutoff."),
 "customer_segment": ("Category", "Tenure-based customer segment: New (3-11m), Established (12-35m), Loyal (36m+).", "Observed by cutoff."),
 "plan_tier": ("Category", "Customer plan/service tier: Basic, Standard, or Premium.", "Observed by cutoff."),
 "contract_type": ("Category", "Contract or renewal cadence: Month-to-month, Quarterly, Annual.", "Observed by cutoff."),
 "tenure_months": ("Integer", "Completed active months as of prediction date; >=3.", "Observed by cutoff; eligibility signal."),
 "monthly_charges_usd": ("Decimal USD", "Current monthly-equivalent nominal fee before active discount.", "Observed by cutoff."),
 "total_billed_to_date_usd": ("Decimal USD", "Synthetic cumulative invoiced amount up to cutoff, including discounts.", "Observed by cutoff."),
 "payment_method": ("Category", "Credit Card, Bank Transfer, Digital Wallet, or Cash/Manual.", "Observed by cutoff."),
 "auto_pay_enabled": ("Binary", "1 if autopay enabled at cutoff, otherwise 0.", "Observed by cutoff."),
 "billing_issues_90d": ("Integer", "Count of billing issues within the 90-day lookback.", "Pre-cutoff 90 days."),
 "late_payments_90d": ("Integer", "Count of late payments within the 90-day lookback.", "Pre-cutoff 90 days."),
 "failed_payments_90d": ("Integer", "Count of failed payment attempts within the 90-day lookback.", "Pre-cutoff 90 days."),
 "support_tickets_90d": ("Integer", "Number of service/support contacts opened in the 90-day lookback.", "Pre-cutoff 90 days."),
 "unresolved_tickets_at_cutoff": ("Integer", "Lookback tickets unresolved on the prediction date; never exceeds ticket count.", "Known by cutoff."),
 "avg_ticket_resolution_hours_90d": ("Decimal hours", "Average resolution hours for resolved lookback tickets; 0 when none resolved.", "Pre-cutoff 90 days."),
 "csat_score_last90d": ("Decimal 1-5", "Latest synthetic customer-satisfaction survey score in lookback.", "Known by cutoff; assumed measured for each customer."),
 "plan_suitability_score": ("Decimal 1-5", "Synthetic plan-fit score from pre-cutoff feedback and suitability assessment.", "Known by cutoff; subjective synthetic measure."),
 "usage_index_prev_30d": ("Integer 0-100", "Normalized usage index for days -60 through -31 (industry-comparable index).", "Pre-cutoff 30-day baseline."),
 "usage_index_recent_30d": ("Integer 0-100", "Normalized usage index for days -30 through -1.", "Pre-cutoff most recent 30 days."),
 "usage_change_pct": ("Decimal %", "100*(recent usage index - previous usage index)/previous usage index.", "Derived exclusively from pre-cutoff values."),
 "engagement_sessions_prev_30d": ("Integer", "Pre-cutoff engagement sessions on days -60 through -31.", "Pre-cutoff 30-day baseline."),
 "engagement_sessions_recent_30d": ("Integer", "Pre-cutoff engagement sessions on days -30 through -1.", "Pre-cutoff most recent 30 days."),
 "engagement_change_pct": ("Decimal %", "100*(recent - previous engagement sessions)/previous sessions.", "Derived exclusively from pre-cutoff values; previous sessions >=1."),
 "days_since_last_activity": ("Integer", "Whole days since last engagement as of prediction; >=30 if no recent sessions.", "Known by cutoff."),
 "renewal_due_in_days": ("Integer", "Days until next planned contract renewal at prediction date.", "Known by cutoff (schedule), not a future outcome."),
 "recent_plan_change_90d": ("Binary", "1 if plan tier/cadence changed in the 90-day lookback.", "Pre-cutoff 90 days."),
 "active_discount_pct": ("Integer %", "Current active percentage discount: 0, 5, 10, 15, or 20.", "Known by cutoff."),
 "prior_retention_contact_90d": ("Binary", "1 if human retention contact occurred in pre-cutoff 90 days.", "Pre-cutoff historical interaction, not future intervention."),
 "churn_label": ("TARGET binary", "1=Churned; 0=Retained in days +1 through +30 after cutoff.", "POST-cutoff target. Never used as a predictor."),
 "observation_start_date": ("Date", "First day of 90-day lookback: prediction_date-90 days.", "Metadata; label audit only."),
 "observation_end_date": ("Date", "Last observed day: prediction_date-1 day.", "Metadata; label audit only."),
 "label_window_start_date": ("Date", "First eligible outcome day: prediction_date+1 day.", "Post-cutoff label metadata only; exclude from model."),
 "label_window_end_date": ("Date", "Last eligible outcome day: prediction_date+30 days.", "Post-cutoff label metadata only; exclude from model."),
 "churn_label_text": ("TARGET category", "Churned if churn_label=1; Retained if churn_label=0.", "POST-cutoff target alias. Exclude from predictors."),
 "churn_event_type": ("TARGET category", "Cancellation or Non-renewal; No churn event for retained customers.", "POST-cutoff event label. Exclude from predictors."),
}


def clip_int(x, lo, hi):
 return int(np.clip(round(x), lo, hi))


def create_candidate(industry, plan, prediction_date):
 # Latent synthetic customer experience class; deliberately NOT exported.
 condition = str(RNG.choice(["strained", "steady", "improving"], p=[0.34, 0.40, 0.26]))
 tenure = clip_int(RNG.gamma(2.3, 14) + 3, 3, 108)
 cust_segment = "New" if tenure < 12 else ("Established" if tenure < 36 else "Loyal")
 contract = str(RNG.choice(["Month-to-month", "Quarterly", "Annual"], p=[0.54,0.16,0.30]))
 days_max = {"Month-to-month":30,"Quarterly":90,"Annual":365}[contract]
 renewal_due = int(RNG.integers(1, days_max+1))
 industry_fee = {"Retail":12,"Telecom":24,"Subscription":18}[industry]
 plan_fee = {"Basic":19,"Standard":48,"Premium":91}[plan]
 monthly_charges = round(float(np.clip(industry_fee+plan_fee+RNG.normal(0,11), 9, 210)), 2)
 discount = int(RNG.choice([0,5,10,15,20], p=[0.61,0.13,0.13,0.09,0.04]))
 total_billed = round(monthly_charges * tenure * (1-discount/100) * float(RNG.uniform(0.91,1.06)),2)
 payment = str(RNG.choice(["Credit Card","Bank Transfer","Digital Wallet","Cash/Manual"],p=[.38,.29,.25,.08]))
 auto_pay = 0 if payment == "Cash/Manual" else int(RNG.random() < 0.73)

 adj = {"strained":0.90,"steady":0.26,"improving":0.09}[condition]
 billing_issues = min(5,int(RNG.poisson(adj+(0.16 if auto_pay==0 else 0))))
 late_payments = min(4,int(RNG.poisson(0.14+(0.53 if condition=="strained" else 0.16)+(0.23 if not auto_pay else 0))))
 failed_payments = min(3,int(RNG.poisson(0.07+0.15*billing_issues+0.12*late_payments)))
 tickets = min(7,int(RNG.poisson({"strained":1.60,"steady":0.71,"improving":0.40}[condition])))
 unresolved = int(RNG.binomial(tickets, {"strained":0.28,"steady":0.14,"improving":0.09}[condition]))
 resolved = tickets - unresolved
 resolution = 0.0 if resolved==0 else round(float(np.clip(RNG.gamma(2.4,5.5)+4*(condition=="strained"), 1.0,120.0)),1)
 csat = round(float(np.clip(4.35-0.36*billing_issues-0.24*tickets-0.45*unresolved
                                   +(0.18 if condition=="improving" else 0)+RNG.normal(0,.78), 1,5)),1)
 suitability = round(float(np.clip(3.80+{"strained":-0.52,"steady":0.02,"improving":0.33}[condition]
                                          +RNG.normal(0,.77),1,5)),1)

 usage_previous = clip_int(RNG.normal(63+(7 if plan=="Premium" else (-7 if plan=="Basic" else 0)),15),12,96)
 usage_pct = float(np.clip(RNG.normal({"strained":-0.35,"steady":-0.04,"improving":0.17}[condition],.20),-.83,.63))
 usage_recent = clip_int(usage_previous*(1+usage_pct),0,100)
 usage_change = round(100.0*(usage_recent-usage_previous)/usage_previous,1)
 engagement_previous = max(1,int(RNG.poisson({"Basic":10,"Standard":15,"Premium":20}[plan])))
 engagement_pct_noise = float(np.clip(RNG.normal({"strained":-.41,"steady":-.06,"improving":.21}[condition],.26),-.98,.92))
 engagement_recent = max(0,min(80,round(engagement_previous*(1+engagement_pct_noise)+RNG.normal(0,1.8))))
 engagement_recent = int(engagement_recent)
 engagement_change = round(100.0*(engagement_recent-engagement_previous)/engagement_previous,1)
 if engagement_recent == 0:
  days_last = int(RNG.integers(30,60))
 else:
  upper = max(2,min(30,int(36/(1+engagement_recent/7))))
  days_last = int(RNG.integers(0,upper))
 plan_change = int(RNG.random() < {"strained":.21,"steady":.11,"improving":.10}[condition])
 prior_retention = int(RNG.random() < {"strained":.22,"steady":.06,"improving":.03}[condition])

 # Churn is stochastic, not a trivial rule or copy of any feature.
 # Rate is *subsequently* balanced by stratified sampling; don't use these
 # generator-side probabilities as supervised training labels or as scores.
 log_odds = (-1.87 + .56*(contract=="Month-to-month") - .38*(contract=="Annual")
             - .30*(tenure>=36) - .20*auto_pay
             + .68*max(0,-usage_change/100)/.35
             + .55*max(0,-engagement_change/100)/.45
             + .17*billing_issues + .14*late_payments + .18*failed_payments
             + .27*unresolved + .18*max(0,3.5-csat)
             + .24*max(0,3.4-suitability)
             + .27*(days_last>=21) + .24*(renewal_due<=14)
             + .08*(industry=="Subscription")
             + .09*math.sin((prediction_date.month-1)/12*math.tau)
             + RNG.normal(0,.62))
 churn_probability = 1.0/(1.0+math.exp(-float(np.clip(log_odds,-12,12))))
 churn_label = int(RNG.random() < churn_probability)
 event = "No churn event"
 if churn_label:
  event = ("Non-renewal" if renewal_due<=30 and RNG.random()<.36 else "Cancellation")
 return {
 "industry_segment":industry, "customer_segment":cust_segment, "plan_tier":plan,
 "contract_type":contract, "tenure_months":tenure, "monthly_charges_usd":monthly_charges,
 "total_billed_to_date_usd":total_billed, "payment_method":payment,
 "auto_pay_enabled":auto_pay, "billing_issues_90d":billing_issues,
 "late_payments_90d":late_payments, "failed_payments_90d":failed_payments,
 "support_tickets_90d":tickets, "unresolved_tickets_at_cutoff":unresolved,
 "avg_ticket_resolution_hours_90d":resolution, "csat_score_last90d":csat,
 "plan_suitability_score":suitability, "usage_index_prev_30d":usage_previous,
 "usage_index_recent_30d":usage_recent, "usage_change_pct":usage_change,
 "engagement_sessions_prev_30d":engagement_previous,
 "engagement_sessions_recent_30d":engagement_recent,
 "engagement_change_pct":engagement_change,"days_since_last_activity":days_last,
 "renewal_due_in_days":renewal_due, "recent_plan_change_90d":plan_change,
 "active_discount_pct":discount, "prior_retention_contact_90d":prior_retention,
 "churn_label":churn_label, "_event_type":event,
 }


def build():
 records = []
 serial = 1
 for month_i, (yy,mm) in enumerate(MONTHS):
  pd = date(yy,mm,1)
  split = "TRAIN" if month_i < 14 else ("VALIDATION" if month_i < 17 else "TEST")
  for industry in INDUSTRIES:
   for plan in PLANS:
    groups = {0:[],1:[]}
    for _ in range(200):
     cand = create_candidate(industry,plan,pd)
     groups[cand["churn_label"]].append(cand)
     if len(groups[0])>=14 and len(groups[1])>=14:
      break
    if min(len(groups[0]),len(groups[1]))<5:
     raise RuntimeError(f"Not enough positives/negatives: {pd}/{industry}/{plan}")
    selected = [groups[label][int(idx)] for label in (0,1) for idx in RNG.choice(len(groups[label]),size=5,replace=False)]
    for r in selected:
     r["customer_id"] = f"SYN-{serial:06d}"
     r["prediction_date"] = pd.isoformat()
     r["data_split"] = split
     serial+=1
     records.append(r)
 # Sort chronologically with deterministic shuffle inside date; no customer repeats.
 records.sort(key=lambda r:(r["prediction_date"],float(RNG.random())))
 for r in records:
  assert set(r.keys()) == set(FIELDS)|{"_event_type"}
 return records


def write_csv(path, columns, records):
 with path.open("w", encoding="utf-8", newline="") as f:
  writer=csv.DictWriter(f, fieldnames=columns,lineterminator="\n")
  writer.writeheader()
  for r in records:
   writer.writerow({k:r[k] for k in columns})


def prepare_labels(records):
 output=[]
 for r in records:
  pd=date.fromisoformat(r["prediction_date"])
  output.append({
   "customer_id":r["customer_id"],"prediction_date":r["prediction_date"],
   "observation_start_date":(pd-timedelta(days=90)).isoformat(),
   "observation_end_date":(pd-timedelta(days=1)).isoformat(),
   "label_window_start_date":(pd+timedelta(days=1)).isoformat(),
   "label_window_end_date":(pd+timedelta(days=30)).isoformat(),
   "churn_label":r["churn_label"],
   "churn_label_text":"Churned" if r["churn_label"] else "Retained",
   "churn_event_type":r["_event_type"]
  })
 return output


def validate(records, labels):
 assert len(records) == len(labels) == 1800
 assert len({r["customer_id"] for r in records}) ==1800
 assert set(r["churn_label"] for r in records)=={0,1}
 assert not any(r[k] is None or r[k]=="" for r in records for k in FIELDS)
 assert not any(r[k] is None or r[k]=="" for r in labels for k in LABEL_FIELDS)
 assert Counter(r["churn_label"] for r in records)=={0:900,1:900}
 assert len(set(r["prediction_date"] for r in records)) == 20
 for r,l in zip(records,labels):
  assert r["churn_label"]==l["churn_label"] and r["customer_id"] == l["customer_id"]
  pd=date.fromisoformat(r["prediction_date"])
  assert (date.fromisoformat(l["observation_end_date"])-date.fromisoformat(l["observation_start_date"])).days==89
  assert date.fromisoformat(l["observation_end_date"]) < pd < date.fromisoformat(l["label_window_start_date"])
  assert (date.fromisoformat(l["label_window_end_date"])-pd).days==30
  assert r["tenure_months"]>=3
  assert r["customer_segment"] == ("New" if r["tenure_months"]<12 else ("Established" if r["tenure_months"]<36 else "Loyal"))
  assert r["unresolved_tickets_at_cutoff"]<=r["support_tickets_90d"]
  assert (r["avg_ticket_resolution_hours_90d"] == 0) == (r["support_tickets_90d"] == r["unresolved_tickets_at_cutoff"])
  assert 1 <=r["csat_score_last90d"]<=5 and 1<=r["plan_suitability_score"]<=5
  assert 0<=r["usage_index_recent_30d"]<=100 and 12<=r["usage_index_prev_30d"]<=96
  assert abs(r["usage_change_pct"]-round(100*(r["usage_index_recent_30d"]-r["usage_index_prev_30d"])/r["usage_index_prev_30d"],1))<.001
  assert abs(r["engagement_change_pct"]-round(100*(r["engagement_sessions_recent_30d"]-r["engagement_sessions_prev_30d"])/r["engagement_sessions_prev_30d"],1))<.001
  assert (r["days_since_last_activity"]<30) == (r["engagement_sessions_recent_30d"]>0)
  assert r["payment_method"]!="Cash/Manual" or r["auto_pay_enabled"]==0
  assert 1<=r["renewal_due_in_days"]<={"Month-to-month":30,"Quarterly":90,"Annual":365}[r["contract_type"]]
  assert r["active_discount_pct"] in {0,5,10,15,20}
  assert r["churn_label"] == (l["churn_label_text"]=="Churned")
  assert (r["churn_label"] == 0) == (l["churn_event_type"] == "No churn event")
  assert (r["data_split"]=="TRAIN")==(pd<date(2026,3,1))
  if pd>=date(2026,3,1):
   assert r["data_split"] == ("VALIDATION" if pd<date(2026,6,1) else "TEST")
 # Perfect 50/50 in each prediction month x industry x tier cohort.
 groups=defaultdict(Counter)
 for r in records:
  groups[(r["prediction_date"],r["industry_segment"],r["plan_tier"])][r["churn_label"]]+=1
 assert len(groups)==180
 assert all(cnt=={0:5,1:5} for cnt in groups.values())
 assert all(Counter(r["churn_label"] for r in records if r["data_split"]==s)[0]
            ==Counter(r["churn_label"] for r in records if r["data_split"]==s)[1]
            for s in ["TRAIN","VALIDATION","TEST"])
 return {
  "n_rows":len(records),"n_columns":len(FIELDS),"n_customers":1800,
  "label_distribution":{"0_Retained":900,"1_Churned":900},
  "split_label_counts":{s:{"0_Retained":sum(1 for r in records if r["data_split"]==s and r["churn_label"]==0),
                          "1_Churned":sum(1 for r in records if r["data_split"]==s and r["churn_label"]==1)}
                       for s in ["TRAIN","VALIDATION","TEST"]},
  "industry_label_counts":{s:{"0_Retained":sum(1 for r in records if r["industry_segment"]==s and r["churn_label"]==0),
                            "1_Churned":sum(1 for r in records if r["industry_segment"]==s and r["churn_label"]==1)}
                          for s in INDUSTRIES},
  "plan_label_counts":{s:{"0_Retained":sum(1 for r in records if r["plan_tier"]==s and r["churn_label"]==0),
                        "1_Churned":sum(1 for r in records if r["plan_tier"]==s and r["churn_label"]==1)}
                       for s in PLANS},
  "monthly_counts":{f"{yy:04d}-{mm:02d}":{"0_Retained":45,"1_Churned":45} for yy,mm in MONTHS},
  "distinct_month_industry_plan_groups":180,
  "each_group_counts":{"0_Retained":5,"1_Churned":5},
  "missing_values":0,"duplicate_customer_ids":0,"failed_data_checks":0,
  "observation_window_days":90,"prediction_horizon_days":30,
  "first_prediction_date":"2025-01-01","last_prediction_date":"2026-08-01",
  "random_seed":SEED,"synthetic_data":True
 }


def write_docs(report):
 dictionary_rows=[]
 for f in FIELDS + [x for x in LABEL_FIELDS if x not in FIELDS]:
  typ,meaning,availability=DESCRIPTIONS[f]
  dictionary_rows.append({"field_name":f,"data_type":typ,"definition":meaning,"feature_availability_or_warning":availability,
                           "file":"Main dataset" if f in FIELDS else "Explicit labels only"})
 write_csv(ROOT/"customer_churn_data_dictionary.csv",list(dictionary_rows[0].keys()),dictionary_rows)
 (ROOT/"customer_churn_quality_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 readme='''# HCLTech hackathon — synthetic customer churn dataset

**Use case:** Customer Churn Prediction and Retention Prioritization (supplied participant brief).
**Data status:** 100% SYNTHETIC, generated for demonstration. No real customer records or protected attributes.
**Size:** 1,800 unique customers; 32 main-data columns; no nulls, no duplicate IDs, no malformed dates.

## Files
- `customer_churn_1800_dataset.csv`: principal dataset for EDA, model training and chronological evaluation; **target `churn_label`** is its final column.
- `customer_churn_1800_explicit_labels.csv`: human-readable outcome labels and window definitions; **NEVER join its post-cutoff outcome columns into model features**.
- `customer_churn_data_dictionary.csv`: feature meanings, data types, and availability/leakage notes.
- `customer_churn_quality_report.json`: automated validation, class balance and split distribution.
- `customer_churn_1800_workbook.xlsx`: Excel view with data, outcome labels, field definitions and a balance summary.
- `generate_customer_churn_dataset.py`: deterministic source generator/validator (requires NumPy).

## Exact definitions and eligibility (FR-01)
- **Eligible population:** active synthetic customers at the prediction point, with >=3 completed months of tenure, no prior churn event; one row/customer/point. All generated records satisfy this eligibility rule. Enrollment/activity and prior-churn exclusions are assumptions rather than extra dataset columns.
- **Prediction point:** 00:00 on the first calendar day of each month (20 monthly cohorts, 2025-01-01 through 2026-08-01).
- **Observation window:** the 90 complete calendar days immediately before the prediction point (`prediction_date-90` through `prediction_date-1`).
- **Recent/previous windows:** `recent_30d` days -30..-1; `prev_30d` days -60..-31. The broader `90d` fields use days -90..-1.
- **Prediction horizon:** days +1 through +30 after the prediction point. A next-day churn event is included, and no information after the prediction point is used as a predictor.
- **Churn definition:** `churn_label = 1` if the customer cancels the ongoing account/service/membership **or fails a scheduled renewal** in the 30-day horizon. Otherwise `churn_label = 0`. The associated label text is `Churned` vs `Retained`. Customer relationship reduction without account termination is *not* separately labeled. This is an explicit modeling assumption, because the brief does not prescribe an operational churn event.
- **Outcome labels are synthetic:** generated probabilistically from signals such as declining usage, billing friction, support tickets, plan-fit and engagement, with random variation. These are modeled associations, not confirmed causal effects.

## Balance and chronology (FR-03)
For each month x industry segment x plan tier, precisely 10 records are sampled: 5 `Retained` and 5 `Churned`. Thus every month, industry, plan tier, and split is exactly 50/50 by target. This is deliberate stratified case-control sampling and **not representative of a real customer churn rate**.

| Set | Prediction months | Retained (0) | Churned (1) | Total |
|---|---|---:|---:|---:|
| TRAIN | 2025-01 to 2026-02 | 630 | 630 | 1,260 |
| VALIDATION | 2026-03 to 2026-05 | 135 | 135 | 270 |
| TEST (later held-out) | 2026-06 to 2026-08 | 135 | 135 | 270 |
| ALL | 2025-01 to 2026-08 | **900** | **900** | **1,800** |

**Leakage controls:** Sort/split by `prediction_date` and preserve `data_split`; no customer is repeated across dates/splits. Train model features on pre-cutoff data only. EXCLUDE `customer_id`, `prediction_date` (except for splitting/time-aware validation), `data_split`, `churn_label` (input target only), and every non-identifier post-cutoff column in `explicit_labels.csv` from `X`. No engineered "future retention", future contacts, post-churn usage, event time, observed churn types, or model risk estimates appear in the main features. Fit encoders, imputers, scalers, and calibration ONLY on training data (or validation for model selection) — not on test.

## Recommended hackathon workflow (FR-02, 04-09)
1. Load dataset CSV; chronological TRAIN/VALIDATION/TEST as supplied. Separate `y=churn_label`; drop identifiers, split indicator and prediction timestamp from `X`.
2. On TRAIN, build a preprocessing + classification pipeline (logistic regression, tree boosting, etc.) with category encoding. Retain VALIDATION for threshold choice and optional probability calibration. Never tune on TEST.
3. Predict TEST churn probabilities; report **PR-AUC**, **ROC-AUC**, precision, recall, F1, calibration (Brier score/reliability curve), and a confusion matrix.
4. Build a priority queue by sorting prediction probabilities descending; include customer_id, industry, tier, usage and engagement changes, billing issues, unresolved tickets, and model explanations. Use configurable score threshold or top-k review capacity. Report TP/FP/FN, precision/recall, top-k capture and lift.
5. Compare outcomes/error metrics by industry_segment, plan_tier, customer_segment, contract_type, tenure band, usage trends, and billing issues. Prefer model-derived drivers (e.g. SHAP or standardized coefficients), and do not describe them as proven causes.

## Important limitations
- Because the data is purposefully **50% churn**, measured prevalence, PR-AUC baseline, precision at a threshold, lift, and probability calibration are for the BALANCED synthetic sample only. Real-world deployment requires an independent natural-prevalence sample and recalibration/validation; **do not claim field readiness**.
- Three unrelated sectors are represented with shared numeric scales; usage is a standardized *index* rather than raw minutes, purchases or streaming hours. Charges are illustrative USD equivalents.
- CSAT and plan-suitability are assumed observed pre-cutoff for every customer; actual missingness and data coverage may differ. There are no protected characteristics.
- 1,800 customers are independent one-time snapshots, not a longitudinal panel; temporal patterns are cohorts at successive prediction dates, not within-customer histories. All labels, event types and signal relationships are fabricated.
- A customer's `churn_label` is a post-cutoff target, even though it appears in the CSV as ground truth for supervised learning. `explicit_labels.csv` is audit-only for all columns other than ID/prediction_date.

## Source mapping
FR-01: prediction windows, label dictionary. FR-02: target and historical predictors. FR-03: dated TRAIN/VALIDATION/TEST. FR-04, 06-07: capacity, ranking, threshold inputs (score must be learned). FR-05: interpretable risk signal fields. FR-08: industry, plan, tenure, billing and usage segments. FR-09: generator, dictionary, and validation report.

Regenerate: `python generate_customer_churn_dataset.py` (requires `numpy`). **Note**: generation rewrites CSV, JSON and README; the Excel file is an additional formatted view produced separately.
'''
 (ROOT/"customer_churn_README.md").write_text(readme,encoding="utf-8")


def main():
 records=build()
 labels=prepare_labels(records)
 report=validate(records,labels)
 write_csv(ROOT/"customer_churn_1800_dataset.csv",FIELDS,records)
 write_csv(ROOT/"customer_churn_1800_explicit_labels.csv",LABEL_FIELDS,labels)
 write_docs(report)
 print(json.dumps({k:v for k,v in report.items() if k not in ["monthly_counts"]},indent=2))

if __name__=="__main__":main()
