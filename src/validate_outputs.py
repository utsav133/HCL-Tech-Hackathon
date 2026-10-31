"""Quick quality gate for synthetic demo; all assertions must pass before submission."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/customer_churn_1800_dataset.csv',parse_dates=['prediction_date'])
out=ROOT/'outputs'
assert df.shape==(1800,32)
assert df.customer_id.is_unique and df.isna().sum().sum()==0
assert df.churn_label.value_counts().to_dict()=={0:900,1:900}
assert df.groupby(['prediction_date','industry_segment','plan_tier','churn_label']).size().eq(5).all()
assert len(df.groupby(['prediction_date','industry_segment','plan_tier','churn_label']))==20*3*3*2
assert df[df.data_split=='TRAIN'].prediction_date.max()<df[df.data_split=='VALIDATION'].prediction_date.min()
assert df[df.data_split=='VALIDATION'].prediction_date.max()<df[df.data_split=='TEST'].prediction_date.min()
report=json.loads((out/'evaluation_report.json').read_text())
sets=report['temporal_sets']
assert [sets[z]['n'] for z in ['train_pre_cutoff','purged_gap','calibration','threshold_validation','heldout_test']]==[1170,90,90,180,270]
review=pd.read_csv(out/'retention_priority_queue.csv')
eval_df=pd.read_csv(out/'test_scored_with_ground_truth.csv')
assert len(review)==len(eval_df)==270
assert 'churn_label' not in review.columns and 'churn_label' in eval_df.columns
assert review.risk_probability.between(0,1).all()
assert review.risk_probability.is_monotonic_decreasing
assert review.customer_id.is_unique
assert all(c in review.columns for c in ['top_risk_drivers','top_protective_signals','usage_change_pct','billing_issues_90d'])
assert review.top_risk_drivers.notna().all()
assert 0 < report['selected_xgboost_threshold'] < 1
for c in report['capacity_lift']:
    k=int(c['top_k']);tp=int(eval_df.head(k).churn_label.sum());p=tp/k
    assert tp==c['captured_churners']
    assert abs(p-c['precision_at_k'])<1e-5
    assert abs(p/0.5-c['lift_at_k'])<1e-5
for c in ['evaluation_report.json','evaluation_report.md','calibration_table.csv','threshold_tradeoffs.csv','capacity_lift.csv','segmented_test_performance.csv','global_risk_drivers.csv','offline_retention_dashboard.html']:
    assert (out/c).stat().st_size>10,c
assert (out/'offline_retention_dashboard.html').read_text().count('__PAYLOAD__')==0
print('PASS: 1,800 dataset, exact month/industry/plan labels, chronological cohorts, model outputs, no ground truth in review queue, top-k counts, all artifacts')
