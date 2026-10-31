"""HCLTech research-guided churn modeling and leak-safe chronological evaluation.

Run from any directory: python src/train_and_evaluate.py --project-dir PATH
All customer records are synthetic; reported measurements are demonstration only.
"""
from __future__ import annotations
import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, DMatrix

SEED = 42
LABEL = 'churn_label'
META = ['customer_id', 'prediction_date', 'data_split', LABEL]
CATEGORICAL = ['industry_segment', 'customer_segment', 'plan_tier', 'contract_type', 'payment_method']
TRAIN_LAST = '2026-01-01'
PURGE_MONTH = '2026-02-01'
CALIBRATION_DATE = '2026-03-01'
THRESHOLD_MONTHS = ['2026-04-01', '2026-05-01']
TEST_FIRST = '2026-06-01'
TEST_LAST = '2026-08-01'
LABEL_HORIZON_DAYS = 30


def ensure(condition, message):
    if not condition:
        raise ValueError('DATA/LEAKAGE CHECK FAILED: '+message)


def validate(df):
    ensure(len(df) >= 500, 'Too few records')
    ensure(not df.isna().any().any(), 'Found missing values')
    ensure(df.customer_id.is_unique, 'Duplicate customer IDs')
    ensure(set(df[LABEL].unique()) == {0,1}, 'Target must contain explicit 0/1 labels')
    ensure(set(df.data_split.unique()) == {'TRAIN','VALIDATION','TEST'}, 'Expected split labels missing')
    ensure((df.tenure_months >= 3).all(), 'Customer tenure eligibility violated')
    ensure(df.prediction_date.notna().all(), 'Invalid prediction dates')
    ensure(df.prediction_date.dt.is_month_start.all(), 'Snapshot not first day of month')
    ensure((df[df.data_split=='TRAIN'].prediction_date.max() < df[df.data_split=='VALIDATION'].prediction_date.min()), 'Train/validation chronology')
    ensure((df[df.data_split=='VALIDATION'].prediction_date.max() < df[df.data_split=='TEST'].prediction_date.min()), 'Validation/test chronology')
    ensure(len(df.columns) == len(set(df.columns)), 'Duplicated columns')
    for c in [col for col in df.columns if col.endswith('_pct')]:
        ensure(np.isfinite(df[c]).all(), f'Nonfinite {c}')
    return {'total_records':int(len(df)), 'unique_customers':int(df.customer_id.nunique()),
            'missing_values':int(df.isna().sum().sum()), 'positive_labels':int(df[LABEL].sum()),
            'negative_labels':int((df[LABEL]==0).sum())}


def sets_for_time(df):
    train = df[(df.data_split=='TRAIN') & (df.prediction_date <= pd.Timestamp(TRAIN_LAST))].copy()
    purge = df[df.prediction_date == pd.Timestamp(PURGE_MONTH)].copy()
    cal = df[df.prediction_date == pd.Timestamp(CALIBRATION_DATE)].copy()
    tune = df[df.prediction_date.isin(pd.to_datetime(THRESHOLD_MONTHS))].copy()
    test = df[(df.data_split == 'TEST') & (df.prediction_date >= pd.Timestamp(TEST_FIRST)) & (df.prediction_date <= pd.Timestamp(TEST_LAST))].copy()
    ensure(len(train)>0 and len(purge)>0 and len(cal)>0 and len(tune)>0 and len(test)>0, 'Empty chronological set')
    # A label for a snapshot t is known only at end of t+30 days; no future model labels allowed.
    ensure(train.prediction_date.max() + timedelta(days=LABEL_HORIZON_DAYS) < cal.prediction_date.min(), 'Train labels overlap first scoring point')
    ensure(cal.prediction_date.max() + timedelta(days=LABEL_HORIZON_DAYS) < tune.prediction_date.min(), 'Calibration labels not available at threshold validation date')
    ensure(tune.prediction_date.max() + timedelta(days=LABEL_HORIZON_DAYS) < test.prediction_date.min(), 'Threshold-tuning labels not ready for test prediction')
    ensure(len(df) == sum(map(len, [train,purge,cal,tune,test])), 'Rows missing from chronological partition')
    ensure(not df.customer_id.duplicated().any(), 'Same customer appears in two temporal partitions')
    return train,purge,cal,tune,test


def build_models(numeric):
    # Fit transform layers only to TRAIN via sklearn pipeline.
    prep_tree = ColumnTransformer([('num','passthrough', numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],
        verbose_feature_names_out=True)
    prep_lr = ColumnTransformer([('num',StandardScaler(), numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],
        verbose_feature_names_out=True)
    return {
        'XGBoost': Pipeline([('prep',prep_tree),('classifier',XGBClassifier(
            n_estimators=240,max_depth=3,learning_rate=0.045,subsample=0.85,
            colsample_bytree=0.85,min_child_weight=4,reg_lambda=4,
            objective='binary:logistic',eval_metric='logloss',tree_method='hist',
            random_state=SEED,n_jobs=2))]),
        'LogisticRegression':Pipeline([('prep',prep_lr),('classifier',LogisticRegression(
            max_iter=3000,C=0.5,random_state=SEED))]),
        'DecisionTree':Pipeline([('prep',prep_tree),('classifier',DecisionTreeClassifier(
            max_depth=5,min_samples_leaf=22,random_state=SEED))])
    }


def raw_predict(model,df,features):
    return np.asarray(model.predict_proba(df[features])[:,1],dtype=float)


def fit_sigmoid_calibration(raw,y):
    values=logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1)
    calibrator = LogisticRegression(C=3.0,max_iter=1000,random_state=SEED)
    calibrator.fit(values,y)
    return calibrator


def calibrated(raw,calibrator):
    return calibrator.predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1]


def score_metrics(y,p,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    yp=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return {'n':int(len(y)), 'observed_churn_rate':round(float(y.mean()),6),
      'accuracy':round(float(accuracy_score(y,yp)),6),
      'pr_auc_average_precision':round(float(average_precision_score(y,p)),6),
      'roc_auc':round(float(roc_auc_score(y,p)),6),
      'precision':round(float(precision_score(y,yp,zero_division=0)),6),
      'recall':round(float(recall_score(y,yp,zero_division=0)),6),
      'f1':round(float(f1_score(y,yp,zero_division=0)),6),
      'brier_score':round(float(brier_score_loss(y,p)),6),
      'log_loss':round(float(log_loss(y,p,labels=[0,1])),6),
      'flagged':int(yp.sum()), 'true_positives':int(tp),'false_positives':int(fp),
      'false_negatives':int(fn),'true_negatives':int(tn), 'threshold':float(threshold)}


def select_threshold(y,p):
    # Business rule: fixed 30% outreach capacity and optimize recall / precision trade-off via F1.
    # threshold must be fitted on VALIDATION only; TEST cannot affect it.
    candidates=sorted(set([float(x) for x in np.round(p,7)] + [0.30,0.40,0.50,0.60,0.70,0.80,0.90]))
    n=len(y); options=[]
    for t in candidates:
        flagged=(p>=t).sum()
        if flagged <= int(n * 0.30) and flagged>0:
            f1=f1_score(y, p>=t, zero_division=0)
            options.append((f1,recall_score(y,p>=t,zero_division=0),-flagged,t))
    if not options:
        return 0.5
    return float(max(options)[3])


def topk_metrics(df,score_col='risk_probability',capacities=(0.10,0.20,0.30,0.50)):
    ranked=df.sort_values([score_col,'customer_id'],ascending=[False,True])
    n=len(df); p=float(df[LABEL].mean()); total=int(df[LABEL].sum())
    out=[]
    for share in capacities:
        k=max(1,int(np.ceil(n*share))); top=ranked.head(k); tp=int(top[LABEL].sum())
        precision=tp/k; recall=tp/total if total else 0
        out.append({'capacity_pct':int(100*share),'top_k':k,'captured_churners':tp,
          'false_positives':k-tp,'precision_at_k':round(precision,6),
          'recall_at_k':round(recall,6),'lift_at_k':round(precision/p,6) if p else None,
          'monthly_charge_exposure_proxy_usd':round(float(top.monthly_charges_usd.sum()),2),
          'risk_weighted_monthly_charge_proxy_usd':round(float((top.monthly_charges_usd*top[score_col]).sum()),2)})
    return out


def threshold_table(test,thresholds):
    return [score_metrics(test[LABEL],test.risk_probability,float(t)) for t in thresholds]


def bin_calibration(y,p,bins=8):
    # Equal-width reliability groups; small synthetic 270-row heldout
    edges=np.linspace(0,1,bins+1)
    ids=np.minimum(np.digitize(p,edges,right=False)-1,bins-1)
    rows=[]
    for b in range(bins):
        mask=ids==b
        if mask.sum():
            rows.append({'bin':b+1,'score_interval':f'{edges[b]:.3f}–{edges[b+1]:.3f}',
             'count':int(mask.sum()),'mean_predicted_risk':round(float(p[mask].mean()),5),
             'observed_churn_rate':round(float(np.asarray(y)[mask].mean()),5)})
    return rows


def segmented(test,selected_threshold):
    df=test.copy()
    df['tenure_band']=pd.cut(df.tenure_months,bins=[0,12,36,72,200],labels=['3–12','13–36','37–72','73+'],include_lowest=True)
    df['usage_trend']=pd.cut(df.usage_change_pct,bins=[-100000,-20,0,100000],labels=['Decline >20%','Mild decline','Flat/rising'])
    df['billing_segment']=np.where(df.billing_issues_90d>0,'Has billing issues','No billing issues')
    output=[]
    for col in ['industry_segment','plan_tier','contract_type','customer_segment','tenure_band','usage_trend','billing_segment']:
        for g,sub in df.groupby(col,dropna=False,observed=True):
            if len(sub)==0: continue
            y=sub.churn_label.to_numpy(); predicted=(sub.risk_probability>=selected_threshold).astype(int)
            tp=int(((predicted==1)&(y==1)).sum()); fp=int(((predicted==1)&(y==0)).sum())
            fn=int(((predicted==0)&(y==1)).sum())
            output.append({'segment_dimension':col,'segment':str(g),'count':len(sub),'churn_rate':round(float(y.mean()),4),
                'precision':round(tp/(tp+fp),4) if tp+fp else None,
                'recall':round(tp/(tp+fn),4) if tp+fn else None,
                'false_positives':fp,'false_negatives':fn,'flagged':int(predicted.sum())})
    return pd.DataFrame(output)


def explain_xgb(pipe,df,features,numeric):
    X=pipe.named_steps['prep'].transform(df[features])
    names=list(pipe.named_steps['prep'].get_feature_names_out())
    contrib=pipe.named_steps['classifier'].get_booster().predict(DMatrix(X),pred_contribs=True)
    ensure(contrib.shape[1]==len(names)+1, 'SHAP dimensions mismatch')
    raw=pipe.named_steps['classifier'].get_booster().predict(DMatrix(X),output_margin=True)
    ensure(np.allclose(contrib.sum(axis=1),raw,rtol=1e-3,atol=1e-3), 'SHAP log-odds additivity failed')
    original=[c for c in features]
    grouped=np.zeros((len(df),len(original)))
    for j,name in enumerate(names):
        if name.startswith('num__'):
            source=name.removeprefix('num__')
        elif name.startswith('cat__'):
            target=name.removeprefix('cat__')
            matches=[col for col in CATEGORICAL if target.startswith(col+'_')]
            ensure(len(matches)==1, f'Unknown dummy column {name}')
            source=matches[0]
        else: raise ValueError(f'Unrecognized transformed feature {name}')
        grouped[:,original.index(source)]+=contrib[:,j]
    ensure(np.allclose(grouped.sum(axis=1)+contrib[:,-1],raw,atol=1e-3), 'Grouped SHAP mismatch')
    global_rows=[{'feature':key,'mean_abs_shap_log_odds':round(float(v),6)} for key,v in zip(original,np.abs(grouped).mean(axis=0))]
    global_rows.sort(key=lambda r:r['mean_abs_shap_log_odds'],reverse=True)
    explanation=[]
    for i,(_,row) in enumerate(df.reset_index(drop=True).iterrows()):
        pos=np.argsort(-grouped[i]); neg=np.argsort(grouped[i]); positive=[j for j in pos if grouped[i,j]>0]; negative=[j for j in neg if grouped[i,j]<0]
        def fmt(j):
            key=original[j]; val=row[key]; val=f'{val:.1f}' if isinstance(val,(float,np.floating)) else str(val)
            return f'{key}={val}'
        explanation.append({'top_risk_drivers':' | '.join(fmt(j) for j in positive[:3]),
            'top_protective_signals':' | '.join(fmt(j) for j in negative[:2]),
            'positive_shap_log_odds_sum':round(float(sum(max(v,0) for v in grouped[i])),5),
            'explanations_note':'XGBoost contribution in raw log-odds; correlational, not causal'})
    return global_rows,pd.DataFrame(explanation)


def main(project_dir):
    project_dir=Path(project_dir).resolve(); out=project_dir/'outputs';out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(project_dir/'data'/'customer_churn_1800_dataset.csv',parse_dates=['prediction_date'])
    qa=validate(df)
    train,purge,cal,tune,test=sets_for_time(df)
    features=[c for c in df.columns if c not in META]
    numeric=[c for c in features if c not in CATEGORICAL]
    ensure(len(features)==28,'Expected 28 pre-cutoff features, found '+str(len(features)))
    models=build_models(numeric);results={};preds={}
    for name,pipe in models.items():
        pipe.fit(train[features],train[LABEL]);calibrator=fit_sigmoid_calibration(raw_predict(pipe,cal,features),cal[LABEL])
        p_tune=calibrated(raw_predict(pipe,tune,features),calibrator)
        p_test=calibrated(raw_predict(pipe,test,features),calibrator)
        t=select_threshold(tune[LABEL],p_tune)
        results[name]={'threshold_selected_on_april_may_validation':round(t,7),
            'validation_threshold_metrics':score_metrics(tune[LABEL],p_tune,t),
            'test_metrics':score_metrics(test[LABEL],p_test,t),
            'test_at_fixed_0_5':score_metrics(test[LABEL],p_test,0.5),
            'calibration_march_n':int(len(cal))}
        preds[name]={'model':pipe,'calibrator':calibrator,'prob':p_test,'threshold':t}
        joblib.dump({'model':pipe,'calibrator':calibrator,'features':features,'threshold':t},out/('model_'+name+'.joblib'))
    chosen=preds['XGBoost']; p=chosen['prob'];t=chosen['threshold']
    test=test.copy().reset_index(drop=True)
    test['risk_probability']=np.round(p,6)
    test['risk_percentage']=np.round(p*100,2)
    test['predicted_churn']=(p>=t).astype(int)
    test['risk_band']=pd.cut(p,bins=[-0.001,0.40,0.70,1.01],labels=['Low','Medium','High'])
    test['monthly_revenue_at_risk_proxy_usd']=np.round(test.monthly_charges_usd*p,2)
    global_shap, local=explain_xgb(chosen['model'],test,features,numeric)
    test=pd.concat([test,local],axis=1)
    test.sort_values(['risk_probability','customer_id'],ascending=[False,True],inplace=True)
    test.insert(0,'retention_priority_rank',np.arange(1,len(test)+1))
    queue_cols=['retention_priority_rank','customer_id','prediction_date','industry_segment','customer_segment',
      'plan_tier','contract_type','tenure_months','monthly_charges_usd','risk_probability','risk_percentage',
      'predicted_churn','risk_band','usage_change_pct','engagement_change_pct','billing_issues_90d',
      'failed_payments_90d','unresolved_tickets_at_cutoff','plan_suitability_score',
      'days_since_last_activity','top_risk_drivers','top_protective_signals',
      'monthly_revenue_at_risk_proxy_usd','churn_label','explanations_note']
    test[queue_cols].to_csv(out/'test_scored_with_ground_truth.csv',index=False)
    test[[c for c in queue_cols if c!='churn_label']].to_csv(out/'retention_priority_queue.csv',index=False)
    pd.DataFrame(global_shap).to_csv(out/'global_risk_drivers.csv',index=False)
    test.attrs['selected_threshold']=t
    seg=segmented(test,t);seg.to_csv(out/'segmented_test_performance.csv',index=False)
    pd.DataFrame(threshold_table(test,[0.2,0.3,0.4,0.5,0.6,0.7,0.8,t])).drop_duplicates('threshold').sort_values('threshold').to_csv(out/'threshold_tradeoffs.csv',index=False)
    caps=topk_metrics(test)
    pd.DataFrame(caps).to_csv(out/'capacity_lift.csv',index=False)
    reliability=bin_calibration(test.churn_label.to_numpy(),p,8)
    pd.DataFrame(reliability).to_csv(out/'calibration_table.csv',index=False)
    risk_summary={'research_paper':'AI-based Customer Churn Prediction System using XGBoost for Early Retention and Revenue Optimization (ICSCSS 2026)',
      'data_kind':'Synthetic balanced case-control; not real prevalence',
      'qa':qa,'labels':{'0':'Retained','1':'Churned within +1 to +30 days'},
      'observations':'Prior 90 days; earlier and latest 30 days for change features',
      'temporal_sets':{
        'train_pre_cutoff':{'dates':[str(train.prediction_date.min().date()),str(train.prediction_date.max().date())],'n':len(train),'churned':int(train.churn_label.sum())},
        'purged_gap':{'dates':[PURGE_MONTH,PURGE_MONTH],'n':len(purge),'reason':'February 1 labels become known after March 1 calibration cohort is scored'},
        'calibration':{'dates':[CALIBRATION_DATE,CALIBRATION_DATE],'n':len(cal),'churned':int(cal.churn_label.sum())},
        'threshold_validation':{'dates':THRESHOLD_MONTHS,'n':len(tune),'churned':int(tune.churn_label.sum())},
        'heldout_test':{'dates':[TEST_FIRST,TEST_LAST],'n':len(test),'churned':int(test.churn_label.sum())}},
      'selection':'XGBoost is paper-inspired PRIMARY MODEL (not selected by test). Threshold: max validation F1 with <=30% outreach capacity.',
      'selected_xgboost_threshold':round(float(t),7),
      'model_results':results,'capacity_lift':caps,'calibration_bins':reliability,
      'global_risk_drivers':global_shap[:12],
      'warnings':[
       'Balanced synthetic set: 50% event prevalence, which distorts apparent lift/precision and probability calibration relative to natural churn prevalence.',
       'Review capacity and flagged true churners are based on observed synthetic test outcomes; none are proven to be saved by outreach.',
       'Revenue exposure is a monthly charge proxy, not observed prevented revenue or causal treatment effect.',
       'No future event fields in model; separate explicit outcome labels CSV must never be joined to predictors.',
       'Customer-level SHAP values are XGBoost uncalibrated margin (log-odds) contributions, not direct contributions in percent risk.',
       'Single synthetic cohort per customer: do not interpret time trends as longitudinal individual change.',
       'Comparative metrics are measured on this dataset, not the paper data; paper performance values are not reproduced.',
       'Small test (270) and calibration (90) groups: uncertainty is substantial and simulated performance may not generalize.'
      ]}
    (out/'evaluation_report.json').write_text(json.dumps(risk_summary,indent=2,default=str),encoding='utf8')
    txt=['# Chronological test evaluation — synthetic customer churn','',
       '**IMPORTANT:** This report evaluates 1,800 synthetic, exactly balanced samples; it does not validate real-world retention outcomes.','',
       '## Label and temporal safeguards','',
       'Churn = cancellation / failed renewal during days +1 to +30 after prediction date. Pre-cutoff historical observation = previous 90 days.','',
       '| Partition | Dates | Rows | Churned |','|---|---|---:|---:|']
    for key,detail in risk_summary['temporal_sets'].items():
        txt.append(f"| {key} | {detail['dates'][0]} to {detail['dates'][-1]} | {detail['n']} | {detail.get('churned','—')} |")
    txt+=['','February 2026 is **purged** from model fitting because its +30-day label is not available at March 1. March labels calibrate probabilities; April–May select risk threshold; June–August is untouched until final evaluation.','',
          '## Test results (independent chronological test, n=270)','',
          '| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Test TP | FP | FN | Selected threshold |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,res in results.items():
        m=res['test_metrics'];txt.append(f"| {name} | {m['pr_auc_average_precision']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['brier_score']:.3f} | {m['true_positives']} | {m['false_positives']} | {m['false_negatives']} | {m['threshold']:.3f} |")
    txt+=['','**PR-AUC is average precision (AP).** A 50% prevalence gives an AP no-skill baseline of 0.50; this baseline will change for a realistic class distribution.','',
          '## Capacity and lift (XGBoost)','',
          '| Capacity | Top k | Churn captured | Precision@k | Recall@k | Lift@k |','|---|---:|---:|---:|---:|---:|']
    for c in caps:
        txt.append(f"| {c['capacity_pct']}% | {c['top_k']} | {c['captured_churners']} | {c['precision_at_k']:.3f} | {c['recall_at_k']:.3f} | {c['lift_at_k']:.3f} |")
    txt+=['','## Top global risk signals (mean absolute tree contribution to raw log odds)','']
    for i,row in enumerate(global_shap[:10],1): txt.append(f"{i}. `{row['feature']}` — {row['mean_abs_shap_log_odds']:.3f}")
    txt+=['','## Caveats','']+[f'- {w}' for w in risk_summary['warnings']]
    (out/'evaluation_report.md').write_text('\n'.join(txt)+'\n',encoding='utf8')
    print('PASS 1,800-dataset validation; strict time eligibility and outcome purge')
    print('PARTITIONS:',{k: v['n'] for k,v in risk_summary['temporal_sets'].items()})
    print('PRIMARY XGBOOST TEST:',results['XGBoost']['test_metrics'])
    print('RESULTS WRITTEN:',out)
    return risk_summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-dir',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    main(a.project_dir)
