"""Optional research extension: leak-safe rolling historical XGBoost search.
Only Jan 2025-Jan 2026 model development records participate; final June-Aug test untouched.
Run: python src/tune_xgboost.py
"""
from __future__ import annotations
from pathlib import Path
import itertools
import json
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parents[1]
CATS=['industry_segment','customer_segment','plan_tier','contract_type','payment_method']
EXCLUDED=['customer_id','prediction_date','data_split','churn_label']

def main():
    df=pd.read_csv(ROOT/'data'/'customer_churn_1800_dataset.csv',parse_dates=['prediction_date'])
    development=df[(df.data_split=='TRAIN')&(df.prediction_date<='2026-01-01')].copy()
    features=[c for c in df if c not in EXCLUDED]
    num=[c for c in features if c not in CATS]
    folds=[('2025-06-01','2025-08-01','2025-09-01'),('2025-09-01','2025-11-01','2025-12-01')]
    params=list(itertools.product([160,240],[2,3],[0.05,0.09]))
    scores=[]
    for n_estimators, max_depth, learning_rate in params:
        fold_scores=[]
        for fit_end,val_start,val_end in folds:
            fit=development[development.prediction_date<=fit_end]
            val=development[development.prediction_date.between(val_start,val_end)]
            assert fit.prediction_date.max()+pd.Timedelta(days=30) < val.prediction_date.min()
            prep=ColumnTransformer([('num','passthrough',num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS)])
            model=Pipeline([('prep',prep),('classifier',XGBClassifier(
                n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,
                subsample=0.85,colsample_bytree=0.85,reg_lambda=4,min_child_weight=4,
                random_state=42,n_jobs=2,tree_method='hist',eval_metric='logloss'))])
            model.fit(fit[features],fit.churn_label)
            fold_scores.append(float(average_precision_score(val.churn_label,model.predict_proba(val[features])[:,1])))
        scores.append({'n_estimators':n_estimators,'max_depth':max_depth,'learning_rate':learning_rate,
                       'mean_historical_AP':round(sum(fold_scores)/len(fold_scores),6), 'fold_AP':fold_scores})
    scores.sort(key=lambda r:r['mean_historical_AP'],reverse=True)
    path=ROOT/'outputs'/'optional_xgboost_historical_search.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'selection_rule':'Highest historical rolling average precision; not test data',
                               'suggested_params':{k:scores[0][k] for k in ['n_estimators','max_depth','learning_rate']},
                               'candidates':scores},indent=2))
    print('Historical tuning complete. Best:',scores[0], '\nSaved',path)
    print('To use tuned parameters, consciously edit build_models() and rerun; do NOT use final test metrics to select.')

if __name__=='__main__':main()
