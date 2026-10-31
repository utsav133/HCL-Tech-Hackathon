"""FR-01, FR-08 / D-02: reproducible exploratory + data quality report.
Run: python src/eda_report.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'customer_churn_1800_dataset.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA, parse_dates=['prediction_date'])
    target = 'churn_label'
    assert df[target].dropna().isin([0,1]).all(), 'Target is not binary.'
    qa = {
        'records': len(df), 'columns': len(df.columns),
        'distinct_customers': int(df.customer_id.nunique()),
        'duplicate_customer_ids': int(df.customer_id.duplicated().sum()),
        'missing_cells': int(df.isna().sum().sum()),
        'churned': int(df[target].sum()),
        'retained': int((df[target] == 0).sum()),
        'churn_rate': float(df[target].mean()),
        'prediction_date_min': str(df.prediction_date.min().date()),
        'prediction_date_max': str(df.prediction_date.max().date()),
        'industry_segments': df.industry_segment.value_counts().to_dict(),
        'plan_tiers': df.plan_tier.value_counts().to_dict(),
    }
    assert qa['duplicate_customer_ids'] == 0 and qa['missing_cells'] == 0
    assert qa['churned'] == qa['retained'] == 900
    (OUT / 'eda_quality_report.json').write_text(json.dumps(qa, indent=2), encoding='utf8')
    monthly = df.groupby('prediction_date', as_index=False).agg(customers=(target, 'size'), churned=(target,'sum'), churn_rate=(target,'mean'))
    monthly['prediction_date'] = monthly.prediction_date.dt.strftime('%Y-%m-%d')
    monthly.to_csv(OUT / 'eda_monthly.csv', index=False)
    group_rows = []
    for col in ['industry_segment','plan_tier','contract_type','customer_segment','payment_method', 'auto_pay_enabled']:
        for group, s in df.groupby(col, dropna=False):
            group_rows.append({'segment_type':col,'segment':str(group),'customers':len(s),
                               'churned':int(s.churn_label.sum()), 'churn_rate':float(s.churn_label.mean())})
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUT / 'eda_segments.csv', index=False)
    numeric_cols = ['tenure_months','monthly_charges_usd','usage_change_pct','engagement_change_pct',
                    'billing_issues_90d','support_tickets_90d','csat_score_last90d','plan_suitability_score',
                    'renewal_due_in_days','days_since_last_activity']
    descriptives = df.groupby(target)[numeric_cols].agg(['mean','median','std'])
    descriptives.to_csv(OUT / 'eda_numeric_by_churn.csv')
    # Descriptive visualization only. No scaling / resampling performed here.
    fig, axs = plt.subplots(2,2,figsize=(13,9))
    labels = ['Retained (0)','Churned (1)']
    axs[0,0].bar(labels,[(df[target]==0).sum(),(df[target]==1).sum()]);axs[0,0].set_title('Explicit churn labels (artificial 50/50)')
    monthly_x = pd.to_datetime(monthly['prediction_date'])
    axs[0,1].plot(monthly_x, monthly.churn_rate, marker='o', markersize=3);axs[0,1].set_ylim(0,1)
    axs[0,1].set_title('Monthly synthetic churn rate');axs[0,1].tick_params(axis='x', rotation=45)
    df.boxplot(column='usage_change_pct',by=target,ax=axs[1,0],showfliers=False)
    axs[1,0].set_title('Pre-cutoff usage change by outcome');axs[1,0].set_xlabel('churn_label')
    pd.crosstab(df.billing_issues_90d.gt(0),df.churn_label,normalize='index').plot(kind='bar',stacked=True,ax=axs[1,1]);
    axs[1,1].set_title('Churn mix by billing-issue status');axs[1,1].set_xlabel('Billing issue in lookback')
    fig.suptitle('HCLTech synthetic customer churn — exploratory analysis',fontsize=15,y=1.01)
    fig.tight_layout();fig.savefig(OUT/'eda_overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    md = [
        '# Exploratory data analysis and quality report', '',
        'Dataset: synthetic, deliberately balanced; descriptive associations are **not causal**.', '',
        f"- Records: **{qa['records']:,}**; columns: **{qa['columns']}**; distinct IDs: **{qa['distinct_customers']}**.",
        f"- Missing cells: **{qa['missing_cells']}**; duplicate IDs: **{qa['duplicate_customer_ids']}**.",
        f"- Labels: **{qa['churned']} churned**, **{qa['retained']} retained**; churn rate: **{qa['churn_rate']:.1%}** (artificial).",
        f"- Monthly prediction snapshots: **{qa['prediction_date_min']}** through **{qa['prediction_date_max']}**.",
        '', '## Segmented descriptive churn', '',
        '| Field | Segment | Customers | Churn rate |','|---|---|---:|---:|',
    ]
    for row in group_rows:
        md.append(f"| {row['segment_type']} | {row['segment']} | {row['customers']} | {row['churn_rate']:.1%} |")
    md.extend(['','## Reproducible EDA outputs','',
               '- `eda_overview.png`: label counts, monthly churn, usage distribution, billing mix.',
               '- `eda_monthly.csv`: monthly cohort counts.',
               '- `eda_segments.csv`: segment churn rates.',
               '- `eda_numeric_by_churn.csv`: pre-cutoff numeric feature summaries by outcome.',
               '- `eda_quality_report.json`: cleaning, uniqueness and explicit-label checks.',
               '', '## Limitations','',
               '- Exact 50/50 target balance is imposed by the synthetic generator; population churn prevalence is unknown.',
               '- Each customer has one prediction snapshot, not a time series of the same customer.',
               '- No protected characteristics are generated. Do not infer causal impact from the segments.',
               '- Avoid using outcome, snapshot date or split identity as model features.', ''])
    (OUT/'eda_report.md').write_text('\n'.join(md),encoding='utf8')
    print('EDA OK:',qa,'\nReports in:',OUT)
    return qa

if __name__ == '__main__': main()
