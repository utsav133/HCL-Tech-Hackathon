"""Create presentation-friendly figures from completed empirical results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from scipy.special import logit
import joblib
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'outputs'
report=json.load(open(out/'evaluation_report.json'))
df=pd.read_csv(out/'test_scored_with_ground_truth.csv')
plt.rcParams.update({'font.size':10,'axes.titlesize':14,'figure.dpi':130})

fig,ax=plt.subplots(figsize=(8.2,4.7))
models=list(report['model_results'])
ap=[report['model_results'][k]['test_metrics']['pr_auc_average_precision'] for k in models]
ax.barh(models,ap)
ax.axvline(0.50,linestyle='--',color='gray',label='50% synthetic prevalence baseline')
ax.set(xlim=(0,1),xlabel='Average precision / PR-AUC',title='Held-out chronological test: discrimination')
for i,v in enumerate(ap):ax.text(min(v+.015,.95),i,f'{v:.3f}',va='center')
ax.invert_yaxis();ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,4.4))
y=df.churn_label.to_numpy();p=df.risk_probability.to_numpy()
fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
axs[0].plot(fpr,tpr,label=f"XGBoost AUC = {report['model_results']['XGBoost']['test_metrics']['roc_auc']:.3f}")
axs[0].plot([0,1],[0,1],linestyle='--',color='gray',linewidth=1)
axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve');axs[0].legend(frameon=False,fontsize=9)
axs[1].plot(recall,precision,label=f"Average precision = {report['model_results']['XGBoost']['test_metrics']['pr_auc_average_precision']:.3f}")
axs[1].axhline(y.mean(),linestyle='--',color='gray',linewidth=1)
axs[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall curve');axs[1].legend(frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(out/'roc_and_precision_recall.png',dpi=150);plt.close(fig)

cal=pd.read_csv(out/'calibration_table.csv')
fig,ax=plt.subplots(figsize=(6.5,4.8))
ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Perfect calibration')
ax.plot(cal.mean_predicted_risk,cal.observed_churn_rate,marker='o',linewidth=1.5,label='XGBoost (synthetic)')
for row in cal.itertuples():ax.annotate(str(row.count), (row.mean_predicted_risk,row.observed_churn_rate), xytext=(4,5),textcoords='offset points',fontsize=7)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed churn rate',title='Reliability diagram (bin labels = n)');ax.legend(frameon=False)
fig.tight_layout();fig.savefig(out/'calibration_curve.png',dpi=150);plt.close(fig)

drivers=pd.read_csv(out/'global_risk_drivers.csv').head(12).iloc[::-1]
fig,ax=plt.subplots(figsize=(8.5,5.5))
ax.barh(drivers.feature,drivers.mean_abs_shap_log_odds)
ax.set(xlabel='Mean |SHAP| contribution to XGBoost raw log-odds',title='Portfolio-level model risk signals')
fig.tight_layout();fig.savefig(out/'risk_drivers.png',dpi=150);plt.close(fig)

c=pd.read_csv(out/'capacity_lift.csv')
fig,ax=plt.subplots(figsize=(7.5,4.7))
ax.plot(c.capacity_pct,c.recall_at_k,marker='o',label='Churn capture (recall@k)')
ax.plot(c.capacity_pct,c.precision_at_k,marker='s',label='Precision@k')
ax.set(xlabel='Review capacity (%)',ylabel='Proportion',ylim=(0,1),title='Retention review trade-off (synthetic test)')
ax.legend(frameon=False)
fig.tight_layout();fig.savefig(out/'capacity_tradeoff.png',dpi=150);plt.close(fig)
print('Created 5 figures')
