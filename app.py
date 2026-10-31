"""HCLTech / ATLAS polished Streamlit retention dashboard.
Run: streamlit run app.py
Fully functional controls, real saved synthetic-model outputs; no automated outreach.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.score_new_customers import score_frame

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
NAVY='#11213B'; BLUE='#3468F0'; TEAL='#099E91'; VIOLET='#7659D2'; GOLD='#E7A23A'

st.set_page_config(page_title='ATLAS | Churn Intelligence',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"] {font-family:Inter,system-ui,sans-serif;}
.stApp {background: #f5f8ff; color: #17283f;}
[data-testid="stSidebar"] {background: linear-gradient(160deg,#111e36 0%,#172d51 100%);}
[data-testid="stSidebar"] * {color: #eff5ff !important;}
[data-testid="stSidebar"] [data-baseweb="select"] div {color:#1e2e48 !important;}
h1,h2,h3 {color:#10213e;letter-spacing:-.025em;}
.block-container {padding-top:1.5rem;max-width:1530px;}
.atlas-header {border-radius:22px; padding:27px 34px; color:white;
 background:linear-gradient(110deg,#122a55,#3151a3 68%,#3774df);box-shadow:0 14px 28px rgba(32,70,145,.13); margin-bottom:14px;}
.atlas-header small{color:#d4e4ff;letter-spacing:.12em;font-weight:700;}
.atlas-header h1{color:white;margin:.2rem 0;font-size:2.1rem;line-height:1.17;}
.atlas-header p{color:#d5e2fc;margin-bottom:0;}
.atlas-kpi {background:white;border:1px solid #e7edfa;border-radius:16px;padding:18px 20px;
 box-shadow:0 5px 16px rgba(13,32,78,.035);margin-bottom:10px;}
.atlas-kpi .label {font-size:.83rem;color:#6d7e9b;font-weight:600;}
.atlas-kpi .value {font-weight:800;font-size:1.75rem;color:#152746;margin:7px 0 0;}
.atlas-kpi .subtitle {font-size:.73rem;color:#8a9ab0;}
.atlas-note {padding:14px 18px;background:#fff7e9;border:1px solid #f0ddae;
 border-radius:12px;color:#76521d;font-size:.87rem;margin:12px 0 20px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:15px;}
[data-testid="stDataFrame"] {border:1px solid #e8eefa;border-radius:12px;}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo():
    return pd.read_csv(OUT/'test_scored_with_ground_truth.csv',parse_dates=['prediction_date'])
@st.cache_data(show_spinner=False)
def load_json(filename):
    return json.loads((OUT/filename).read_text(encoding='utf8'))
@st.cache_data(show_spinner=False)
def load_csv(filename):
    return pd.read_csv(OUT/filename)

def card(container,title,value,sub=''):
    container.markdown('<div class="atlas-kpi"><div class="label">'+title+'</div><div class="value">'+str(value)+'</div><div class="subtitle">'+sub+'</div></div>',unsafe_allow_html=True)

def badge_risk(p):
    if p >= 0.7: return 'High'
    if p >= 0.4: return 'Medium'
    return 'Low'

def ranked_frame(df,mode,cap,threshold):
    ordered=df.sort_values(['risk_probability','customer_id'],ascending=[False,True]).reset_index(drop=True).copy()
    n=len(ordered)
    if mode=='Fixed outreach capacity':
        k=max(1,int(np.ceil(n*cap/100)))
        flagged=np.arange(n)<k
    else:
        flagged=ordered.risk_probability.to_numpy()>=threshold
    ordered['flag_for_review']=flagged
    ordered['review_priority_rank']=np.arange(1,n+1)
    return ordered,flagged

st.sidebar.markdown('## ◈ ATLAS')
st.sidebar.caption('RETENTION INTELLIGENCE  /  HCLTech Hackathon')
st.sidebar.divider()
section=st.sidebar.radio('Workspace',['Overview','Retention Queue','Model Evaluation','Risk Explanations','Data Quality','Score New Customers'],index=0)
st.sidebar.divider()
st.sidebar.caption('The decision-support system does not perform automated outreach, discounts or account changes.')
st.markdown('''<div class="atlas-header"><small>HCLTECH  /  FUTURE READY DATA SCIENCE HACKATHON</small>
<h1>Customer Churn & Retention Intelligence</h1>
<p>Risk forecasting · customer-level explainability · precision-at-capacity · human review prioritization</p></div>''',unsafe_allow_html=True)
st.markdown('<div class="atlas-note"><b>Demo data:</b> All results are from 1,800 deliberately balanced synthetic customers. Risk and retention impact are illustrative, not validated real-world probabilities or measured customers saved.</div>',unsafe_allow_html=True)

if section=='Score New Customers':
    st.subheader('Upload historical customer data for scoring')
    st.markdown('Upload a CSV with the **28 pre-prediction features**; exclude target labels, churn events and future retention outcomes.')
    uploaded=st.file_uploader('Customer features CSV',type=['csv'])
    st.download_button('Download valid scoring CSV example',(ROOT/'samples/new_customers.csv').read_bytes(),file_name='new_customers_sample.csv',mime='text/csv')
    if uploaded is None:st.stop()
    try:
        base=score_frame(pd.read_csv(uploaded))
    except (ValueError,KeyError,TypeError) as error:
        st.error(f'Cannot score uploaded data: {error}');st.stop()
    label_available=False
    st.success(f'Scored {len(base):,} eligible customer rows with calibrated XGBoost and fresh local TreeSHAP explanations.')
else:
    try: base=load_demo()
    except FileNotFoundError:
        st.error('Model outputs are missing. Run `python run_all.py` first.');st.stop()
    label_available=True

if section not in ('Model Evaluation','Risk Explanations','Data Quality'):
    f1,f2,f3=st.columns([1.3,1.3,1.5])
    with f1:
        industry=st.selectbox('Industry segment',['All']+sorted(base.industry_segment.astype(str).unique()))
    with f2:
        plan=st.selectbox('Plan tier',['All']+sorted(base.plan_tier.astype(str).unique()))
    with f3:
        mode=st.radio('Selection policy',['Fixed outreach capacity','Risk score threshold'],horizontal=True)
    df=base.copy()
    if industry!='All':df=df[df.industry_segment==industry]
    if plan!='All':df=df[df.plan_tier==plan]
    if len(df)==0:st.warning('No customers match the filters.');st.stop()
    if mode=='Fixed outreach capacity':
        cap=st.slider('Review capacity (% of selected customers)',5,100,20,5);threshold=0.5
    else:
        threshold=st.slider('Churn probability threshold',0.00,1.00,0.55,0.01);cap=20
    ranked,flag=ranked_frame(df,mode,cap,threshold)
    selected=ranked[flag].copy()
    total=len(ranked);reviewed=len(selected)
    weighted_exposure=float((selected.monthly_charges_usd*selected.risk_probability).sum())
    c1,c2,c3,c4=st.columns(4)
    card(c1,'Eligible customers',f'{total:,}','Filtered customer cohort')
    card(c2,'Selected for review',f'{reviewed:,}',f'{reviewed/total:.1%} of cohort')
    card(c3,'Mean selected risk',f'{selected.risk_probability.mean():.1%}' if reviewed else '—','Estimated probability on synthetic data')
    card(c4,'Monthly exposure proxy',f'${weighted_exposure:,.0f}','Not realized savings')
    if label_available:
        y=ranked.churn_label.to_numpy().astype(int)
        tp=int(np.sum(flag&(y==1)));fp=int(np.sum(flag&(y==0)));fn=int(np.sum((~flag)&(y==1)))
        prec=tp/reviewed if reviewed else 0.0
        recall=tp/int(y.sum()) if y.sum() else 0.0
        lift=prec/float(y.mean()) if y.mean() else 0.0
        c1,c2,c3,c4=st.columns(4)
        card(c1,'Precision at review',f'{prec:.1%}',f'{tp} true positives / {reviewed} reviewed')
        card(c2,'Churn captured',f'{recall:.1%}',f'{tp} of {int(y.sum())} actual churners')
        card(c3,'Lift at capacity',f'{lift:.2f}×','Compared to this cohort prevalence')
        card(c4,'TP / FP / FN',f'{tp} / {fp} / {fn}','Retrospective synthetic test only')
    else:
        st.info('No ground-truth future outcomes exist for uploaded customers. Precision, recall, TP/FP/FN and saved customers cannot be computed.')

if section=='Overview':
    left,right=st.columns([1.35,1])
    with left:
        st.subheader('Risk distribution')
        fig=px.histogram(ranked,x='risk_percentage',nbins=22,color_discrete_sequence=[BLUE],
                         labels={'risk_percentage':'Churn risk (%)'},opacity=.88)
        fig.update_layout(template='plotly_white',height=355,margin=dict(l=5,r=8,t=18,b=12),
                          yaxis_title='Customers',bargap=.14,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader('Flagged customers by industry')
        bars=selected.groupby('industry_segment').size().reset_index(name='customers')
        fig=px.bar(bars,x='customers',y='industry_segment',orientation='h',color_discrete_sequence=[TEAL])
        fig.update_layout(template='plotly_white',height=355,margin=dict(l=5,r=8,t=18,b=12),
                          yaxis_title=None,xaxis_title='Customers')
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Highest-risk customers in review queue')
    show=['review_priority_rank','customer_id','industry_segment','plan_tier','risk_percentage',
          'usage_change_pct','billing_issues_90d','top_risk_drivers']
    st.dataframe(selected[show].head(15),hide_index=True,use_container_width=True)

elif section=='Retention Queue' or section=='Score New Customers':
    st.subheader('Prioritized retention review queue')
    st.caption('Scores are ordered from highest to lowest. Signals are associations, not confirmed causes. Customer decisions remain human-led.')
    show=['review_priority_rank','customer_id','industry_segment','customer_segment','plan_tier',
          'risk_percentage','risk_band','monthly_charges_usd','usage_change_pct','engagement_change_pct',
          'billing_issues_90d','failed_payments_90d','unresolved_tickets_at_cutoff',
          'top_risk_drivers','top_protective_signals']
    show=[c for c in show if c in selected]
    st.dataframe(selected[show],hide_index=True,use_container_width=True,height=480)
    export=selected.drop(columns=['churn_label','predicted_churn','data_split'],errors='ignore')
    st.download_button('↓ Export label-free retention queue',export.to_csv(index=False).encode(),
                       file_name='retention_queue.csv',mime='text/csv')
    if label_available:
        with st.expander('Assumption-based retention scenario — NOT observed impact'):
            effect=st.slider('Assumed percentage of captured churners successfully retained',0,100,20,5)
            contact_cost=st.number_input('Review cost per customer ($)',0.0,1000.0,5.0,1.0)
            st.write(f'Illustrative prevented churners: **{tp*effect/100:.1f}** (assuming {effect}% success among true positives).')
            st.write(f'Hypothetical contact/review expense: **${reviewed*contact_cost:,.0f}**.')
            st.caption('Not causal uplift, realized intervention success, or proven revenue savings. A randomized intervention experiment is required.')

elif section=='Model Evaluation':
    st.subheader('Chronological held-out comparison')
    report=load_json('evaluation_report.json')
    metrics=[]
    for name,info in report['model_results'].items():
        m=info['test_metrics'];metrics.append({'Model':name,'PR-AUC / AP':m['pr_auc_average_precision'],
         'ROC-AUC':m['roc_auc'],'Precision':m['precision'],'Recall':m['recall'],
         'F1':m['f1'],'Brier (lower better)':m['brier_score'],'TP':m['true_positives'],
         'FP':m['false_positives'],'FN':m['false_negatives'],'Selected threshold':m['threshold']})
    st.dataframe(pd.DataFrame(metrics),use_container_width=True,hide_index=True)
    st.info('Research-inspired XGBoost is the primary demo model, but Logistic Regression performs better on this synthetic test set. Report this rather than assuming XGBoost always wins.')
    col1,col2=st.columns(2)
    with col1:
        st.subheader('Capture and lift at review capacity')
        cap=load_csv('capacity_lift.csv')
        chart=go.Figure();chart.add_trace(go.Scatter(x=cap.capacity_pct,y=cap.recall_at_k,mode='lines+markers',name='Recall @ k',line_color=BLUE))
        chart.add_trace(go.Scatter(x=cap.capacity_pct,y=cap.precision_at_k,mode='lines+markers',name='Precision @ k',line_color=TEAL))
        chart.update_layout(template='plotly_white',height=330,margin=dict(l=3,r=4,t=8,b=10),xaxis_title='Review capacity (%)',yaxis_tickformat='.0%')
        st.plotly_chart(chart,use_container_width=True)
    with col2:
        st.subheader('Probability reliability (synthetic sample)')
        cal=load_csv('calibration_table.csv')
        fig=go.Figure();fig.add_trace(go.Scatter(x=cal.mean_predicted_risk,y=cal.observed_churn_rate,mode='markers+lines',marker=dict(size=10,color=VIOLET),name='Observed'))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Ideal',line=dict(color='#8c99a8',dash='dash')))
        fig.update_layout(template='plotly_white',height=330,margin=dict(l=3,r=4,t=8,b=10),xaxis_title='Predicted probability',yaxis_title='Observed churn rate')
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Threshold trade-off counts')
    st.dataframe(load_csv('threshold_tradeoffs.csv'),hide_index=True,use_container_width=True)
    st.subheader('Segment performance')
    segments=load_csv('segmented_test_performance.csv')
    chosen=st.selectbox('Segment dimension',segments.segment_dimension.unique())
    st.dataframe(segments[segments.segment_dimension==chosen],hide_index=True,use_container_width=True)
    st.caption('Threshold was selected on April–May 2026 validation; June–August 2026 is held out. February is purged and March is calibration-only. Small segment samples are noisy.')

elif section=='Risk Explanations':
    st.subheader('Explainability: portfolio and individual customers')
    imp=load_csv('global_risk_drivers.csv').head(15).iloc[::-1]
    fig=px.bar(imp,x='mean_abs_shap_log_odds',y='feature',orientation='h',color_discrete_sequence=[VIOLET],
               labels={'mean_abs_shap_log_odds':'Mean absolute XGBoost SHAP (log-odds)','feature':'Historical feature'})
    fig.update_layout(template='plotly_white',height=530,margin=dict(l=5,r=8,t=8,b=8))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Inspect a customer explanation')
    customer=st.selectbox('Customer ID',base.sort_values('risk_probability',ascending=False).customer_id.tolist())
    row=base.loc[base.customer_id==customer].iloc[0]
    st.markdown(f'**Predicted churn risk:** {row.risk_probability:.1%}  |  **Segment:** {row.industry_segment} / {row.plan_tier}')
    st.write('**Risk-increasing signals:**',row.top_risk_drivers)
    st.write('**Protective signals:**',row.top_protective_signals)
    st.caption('Native XGBoost SHAP contributions explain UNCALIBRATED model log-odds, not changes in calibrated probability. Correlational only.')

elif section=='Data Quality':
    st.subheader('Dataset audit and leakage safeguards')
    qa=load_json('eda_quality_report.json')
    a,b,c,d=st.columns(4)
    card(a,'Rows',f"{qa['records']:,}")
    card(b,'Columns',qa['columns'])
    card(c,'Missing cells',qa['missing_cells'])
    card(d,'Duplicate IDs',qa['duplicate_customer_ids'])
    st.markdown('**Timeline: 90-day observation → prediction date → next 30-day churn target**')
    timeline=load_json('evaluation_report.json')['temporal_sets']
    tbl=[]
    for name, value in timeline.items():tbl.append({'Phase':name,'Start':value['dates'][0], 'End':value['dates'][-1], 'Rows':value['n']})
    st.dataframe(pd.DataFrame(tbl),hide_index=True,use_container_width=True)
    st.subheader('Monthly counts and synthetic churn')
    monthly=load_csv('eda_monthly.csv')
    st.dataframe(monthly,hide_index=True,use_container_width=True)
    st.caption('No target, customer identifier, split metadata or future outcomes are used as model predictors. Train-only preprocessing and time-separated calibration are enforced.')

st.divider()
st.caption('ATLAS · HCLTech hackathon prototype · Synthetic data only · Research-inspired XGBoost and baselines · Human-in-the-loop retention review')
