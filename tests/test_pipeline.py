"""Unit and acceptance smoke tests: pytest -q"""
import json
from pathlib import Path
import pandas as pd
import pytest
from src.train_and_evaluate import validate, sets_for_time, topk_metrics
from src.score_new_customers import score_frame
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def data():
    return pd.read_csv(ROOT/'data'/'customer_churn_1800_dataset.csv',parse_dates=['prediction_date'])

def test_fr01_explicit_labels_eligibility_quality(data):
    qa=validate(data)
    assert qa['total_records']==1800
    assert qa['positive_labels']==qa['negative_labels']==900
    assert qa['missing_values']==0

def test_fr03_temporal_purge(data):
    train,purge,cal,tune,test=sets_for_time(data)
    assert list(map(len,[train,purge,cal,tune,test]))==[1170,90,90,180,270]
    assert train.prediction_date.max()+pd.Timedelta(days=30)<cal.prediction_date.min()
    assert tune.prediction_date.max()+pd.Timedelta(days=30)<test.prediction_date.min()

def test_fr07_topk_math(data):
    test=data[data.data_split=='TEST'].copy()
    test['risk_probability']=test.tenure_months.astype(float)/200
    rows=topk_metrics(test,capacities=(0.2,))
    assert rows[0]['top_k']==54
    assert rows[0]['captured_churners']+rows[0]['false_positives']==54

def test_fr05_new_customers_get_fresh_explanations():
    if not (ROOT/'outputs'/'model_XGBoost.joblib').exists(): pytest.skip('First run python run_all.py')
    sample=pd.read_csv(ROOT/'samples/new_customers.csv')
    scored=score_frame(sample)
    assert len(scored)==len(sample)
    assert 'churn_label' not in scored
    assert scored.top_risk_drivers.notna().all()
    assert scored.risk_probability.between(0,1).all()
    assert scored.risk_probability.is_monotonic_decreasing

def test_fr02_reject_leakage():
    if not (ROOT/'outputs'/'model_XGBoost.joblib').exists(): pytest.skip('First run python run_all.py')
    sample=pd.read_csv(ROOT/'samples/new_customers.csv')
    sample['churn_label']=0
    with pytest.raises(ValueError,match='Post-prediction'):
        score_frame(sample)

def test_d03_label_free_queue_and_reports():
    if not (ROOT/'outputs'/'retention_priority_queue.csv').exists(): pytest.skip('First run python run_all.py')
    q=pd.read_csv(ROOT/'outputs'/'retention_priority_queue.csv')
    assert 'churn_label' not in q.columns
    assert 'top_risk_drivers' in q.columns
    r=json.loads((ROOT/'outputs'/'evaluation_report.json').read_text())
    assert 'pr_auc_average_precision' in r['model_results']['XGBoost']['test_metrics']
    assert (ROOT/'outputs'/'eda_report.md').exists()
