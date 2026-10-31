"""FR-02, FR-05, FR-06: label-free batch inference with fresh local TreeSHAP.
Run: python src/score_new_customers.py --input samples/new_customers.csv
No churn labels/outcome fields are accepted in input.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
try:
    from src.train_and_evaluate import explain_xgb
except ImportError:
    from train_and_evaluate import explain_xgb

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_FIELDS = {'churn_label','churn_date','cancellation_date','renewal_outcome',
                  'retained_after_outreach','retention_success','label_end_date','observed_churn',
                  'churn_label_text','churn_event_type','label_window_start_date',
                  'label_window_end_date','observation_start_date','observation_end_date'}

def score_frame(source: pd.DataFrame, bundle_path: Path | str | None = None) -> pd.DataFrame:
    bundle_path = Path(bundle_path or ROOT/'outputs'/'model_XGBoost.joblib')
    bundle = joblib.load(bundle_path)  # only load locally trained, trusted joblib files
    features = bundle['features']
    frame=source.copy().reset_index(drop=True)
    forbidden = sorted(BLOCKED_FIELDS.intersection(frame.columns))
    if forbidden:
        raise ValueError('Post-prediction/target fields are not permitted in scoring input: '+', '.join(forbidden))
    missing = sorted(set(features)-set(frame.columns))
    if missing:
        raise ValueError('Missing required pre-cutoff model features: '+', '.join(missing))
    if frame.empty:
        raise ValueError('Scoring input is empty')
    if not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('Input has non-finite numeric values')
    if frame[features].isna().any().any():
        raise ValueError('Missing feature values; impute from training data before scoring')
    if not (pd.to_numeric(frame.tenure_months,errors='coerce')>=3).all():
        raise ValueError('Eligibility failed: tenure_months must be at least 3')
    if 'customer_id' not in frame.columns:
        frame['customer_id'] = ['NEW-'+str(i+1).zfill(5) for i in range(len(frame))]
    if frame.customer_id.isna().any() or frame.customer_id.duplicated().any():
        raise ValueError('Duplicate or missing customer_id')
    if 'prediction_date' in frame.columns:
        date = pd.to_datetime(frame.prediction_date, errors='coerce')
        if date.isna().any(): raise ValueError('Invalid prediction_date')
    # Saved artifacts were fitted on training data, and calibrator was fitted separately on March.
    raw=bundle['model'].predict_proba(frame[features])[:,1]
    p=bundle['calibrator'].predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1]
    frame['risk_probability']=p
    frame['risk_percentage']=(p*100).round(2)
    frame['risk_band']=pd.cut(p,[-0.01,0.4,0.7,1.01],labels=['Low','Medium','High']).astype(str)
    frame['flag_at_validation_threshold']=(p>=bundle['threshold']).astype(int)
    frame['monthly_revenue_exposure_proxy_usd']=(frame.monthly_charges_usd*p).round(2)
    # Re-compute explanations for uploaded customers, rather than reusing held-out SHAP.
    numeric=[c for c in features if c not in ['industry_segment','customer_segment','plan_tier','contract_type','payment_method']]
    _, local=explain_xgb(bundle['model'],frame,features,numeric)
    frame=pd.concat([frame,local],axis=1)
    frame=frame.sort_values(['risk_probability','customer_id'],ascending=[False,True]).reset_index(drop=True)
    frame.insert(0,'retention_priority_rank',np.arange(1,len(frame)+1))
    return frame


def main():
    p=argparse.ArgumentParser(description='Score eligible historical customer features with a calibrated XGBoost model')
    p.add_argument('--input',required=True,help='CSV containing 28 model fields, no outcome labels')
    p.add_argument('--output',default=str(ROOT/'outputs'/'batch_scored_customers.csv'))
    args=p.parse_args()
    frame=score_frame(pd.read_csv(args.input))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False)
    print(f'Wrote {len(frame)} labeled-free churn-risk predictions to {args.output}')

if __name__=='__main__': main()
