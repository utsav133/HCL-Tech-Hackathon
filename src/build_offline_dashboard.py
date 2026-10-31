"""Build offline interactive HTML viewer. Does not need a server or external assets."""
from pathlib import Path
import pandas as pd
import json
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs'
keep=['customer_id','industry_segment','plan_tier','contract_type','risk_probability','risk_percentage',
      'monthly_charges_usd','monthly_revenue_at_risk_proxy_usd','usage_change_pct','engagement_change_pct',
      'billing_issues_90d','top_risk_drivers','top_protective_signals','churn_label']
q=pd.read_csv(out/'test_scored_with_ground_truth.csv')[keep]
g=pd.read_csv(out/'global_risk_drivers.csv').head(10)
met=json.load(open(out/'evaluation_report.json'))
payload={'rows':q.to_dict('records'),'global_drivers':g.to_dict('records'),
    'models':{k:v['test_metrics'] for k,v in met['model_results'].items()},'selected':met['selected_xgboost_threshold']}
html=r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HCLTech | Customer Churn Intelligence</title>
<style>
:root{--bg:#f5f7fb;--ink:#142236;--muted:#56657b;--primary:#0c6aa8;--dark:#0f2946;--edge:#dfe6ed;--positive:#137c68;--red:#b64747}
*{box-sizing:border-box}body{background:var(--bg);font:14px/1.52 Inter,system-ui,-apple-system,Segoe UI,sans-serif;margin:0;color:var(--ink)}
header{background:var(--dark);color:white;padding:26px 32px}header .sub{color:#bad0e3;font-size:13px;max-width:850px}
main{max-width:1380px;margin:22px auto;padding:0 19px 35px}.ribbon{background:#fff4db;border:1px solid #ecd7a0;padding:13px 16px;border-radius:11px;font-size:13px;margin-bottom:16px}
section{background:#fff;border:1px solid var(--edge);border-radius:14px;box-shadow:0 2px 5px #102d4520;margin:0 0 17px;padding:20px 22px}h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:0 0 14px}h3{font-size:15px;margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{border:1px solid var(--edge);border-radius:11px;padding:14px;background:#f9fbfe}.metric strong{display:block;font-size:28px;margin-top:6px}.metric small{color:var(--muted)}.controls{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:15px}.controls label{display:block;font-weight:600;font-size:13px;margin-bottom:7px}select,input[type=range]{width:100%}select{padding:9px;border:1px solid #b5c8d8;background:white;border-radius:8px}.scope{display:flex;gap:25px;flex-wrap:wrap;margin-bottom:12px}.scope label{cursor:pointer}.subline{color:var(--muted);font-size:12px;margin:8px 0 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:18px}.riskcontrols{margin-top:17px;border-top:1px solid var(--edge);padding-top:15px}input[type=range]{accent-color:var(--primary)}.tabwrap{overflow:auto;max-height:400px;border:1px solid var(--edge);border-radius:9px}table{border-collapse:collapse;width:100%;font-size:12.5px}th{position:sticky;top:0;background:#edf4fa;text-align:left;z-index:1}td,th{padding:10px 11px;border-bottom:1px solid #eaf0f5;white-space:nowrap}td.explain{white-space:normal;min-width:250px;max-width:340px}tbody tr:hover{background:#f7fbff}.tag{border-radius:999px;padding:3px 8px;background:#d9ecfc;color:#175c8d;font-weight:600}.bar{height:12px;border-radius:4px;background:var(--primary)}.barrow{display:grid;grid-template-columns:190px 1fr 60px;align-items:center;gap:13px;margin:9px 0}.barbg{background:#edf4f8;border-radius:4px}.mini{font-size:12px;color:var(--muted)}.modeltable{margin-top:8px}button{border:0;background:var(--primary);color:white;padding:10px 14px;border-radius:8px;font-weight:600;cursor:pointer}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr}.barrow{grid-template-columns:130px 1fr 45px}header{padding:20px}}
</style></head><body><header><h1>Customer Churn Intelligence</h1><div class="sub">HCLTech | Future Ready Data Science Hackathon · XGBoost &amp; Traditional ML · Research-guided retention prioritization</div></header><main>
<div class="ribbon"><b>SYNTHETIC DEMO DATA</b> · Exactly 50% churn labels by design. This is a retrospective demonstration on the chronological test cohort, not live customer data. Risk percentages, precision and lift may not transfer to natural-prevalence populations. No contact/retention action is performed.</div>
<section><h2>1. Configure the retention review</h2><div class="controls"><div><label for="industry">Industry</label><select id="industry"><option>All</option><option>Retail</option><option>Telecom</option><option>Subscription</option></select></div><div><label for="plan">Plan tier</label><select id="plan"><option>All</option><option>Basic</option><option>Standard</option><option>Premium</option></select></div><div><label>Selection mode</label><div class="scope"><label><input type="radio" name="mode" value="capacity" checked> Top-k capacity</label><label><input type="radio" name="mode" value="threshold"> Risk threshold</label></div></div></div>
<div class="riskcontrols"><div class="row"><label for="cap" id="sliderLabel">Review top <b id="capLabel">20%</b> of customers</label><b id="riskHint"></b></div><input id="cap" type="range" min="5" max="100" step="5" value="20"><div class="subline">Top-k enforces a hard limit. A probability threshold can exceed outreach capacity. All results recalculate on the selected segment.</div></div>
</section><section><h2>2. Review capacity and outcomes</h2><div class="grid" id="metrics"></div><p class="subline">TP = flagged customer who subsequently churned · FP = flagged customer who did not churn · FN = missed churn customer. Synthetic ground truth is used only for retrospective evaluation.</p>
<div class="riskcontrols"><label for="effect">Hypothetical retention success among captured churners: <b id="effectLabel">20%</b></label><input id="effect" type="range" min="0" max="100" step="5" value="20"><p id="scenario" class="mini"></p><p class="mini">Assumed saved customers are <b>not measured retention uplift</b>. Actual intervention effectiveness requires an experiment or causal study.</p></div></section>
<section><div class="row"><h2>3. Prioritized retention queue</h2><button onclick="downloadCSV()">Export flagged review list</button></div><p class="mini">Ordered by descending XGBoost probability. CHURN LABEL is intentionally excluded from the review table and exported list.</p><div class="tabwrap"><table><thead><tr><th>Rank</th><th>Customer</th><th>Industry</th><th>Plan</th><th>Risk</th><th>Monthly charge</th><th>Usage Δ</th><th>Engagement Δ</th><th>Billing issues</th><th>Top risk signals</th></tr></thead><tbody id="queue"></tbody></table></div></section>
<section><h2>4. Global model explanation</h2><p class="mini">Mean absolute XGBoost feature contributions to uncalibrated log-odds, grouped across one-hot encoded categories. Importance is association, not confirmed cause.</p><div id="importance"></div></section>
<section><h2>5. Chronological held-out model comparison</h2><p class="mini">270 test customers (June–August 2026). Thresholds calibrated/selected using only earlier periods. A 30% validation capacity constraint was used for threshold choice. Scores are from this synthetic dataset, not the research publication.</p><div class="tabwrap"><table class="modeltable"><thead><tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier</th></tr></thead><tbody id="modeltable"></tbody></table></div><p class="mini">The logistic regression baseline outperforms XGBoost here. We show XGBoost as the research-inspired prototype rather than asserting superiority.</p></section>
</main><script>
const DATA=__PAYLOAD__;
let currentFlagged=[];
const byId=id=>document.getElementById(id);const fmt=x=>Number(x).toFixed(3);
const fPct=x=>`${(100*x).toFixed(1)}%`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function update(){
 const selectedIndustry=byId('industry').value,selectedPlan=byId('plan').value;
 const rows=DATA.rows.filter(r=>(selectedIndustry==='All'||r.industry_segment===selectedIndustry)&&(selectedPlan==='All'||r.plan_tier===selectedPlan)).sort((a,b)=>b.risk_probability-a.risk_probability||a.customer_id.localeCompare(b.customer_id));
 const mode=document.querySelector('input[name="mode"]:checked').value, slider=byId('cap');
 slider.min=mode==='capacity'?5:0;slider.max=mode==='capacity'?100:100;slider.step=mode==='capacity'?5:1;
 let flagged=[];
 if(mode==='capacity'){
  const n=Math.max(1,Math.ceil(rows.length*Number(slider.value)/100));flagged=rows.slice(0,n);
  byId('capLabel').textContent=`${slider.value}%`;byId('sliderLabel').innerHTML=`Review top <b id="capLabel">${slider.value}%</b> of customers`;
 } else {
  let cutoff=Number(slider.value)/100;flagged=rows.filter(r=>r.risk_probability>=cutoff);
  byId('sliderLabel').innerHTML=`Flag risk ≥ <b id="capLabel">${fPct(cutoff)}</b>`;
 }
 currentFlagged=flagged;
 const positives=rows.filter(r=>r.churn_label===1).length,TP=flagged.filter(r=>r.churn_label===1).length;
 const FP=flagged.length-TP,FN=positives-TP;
 const prec=flagged.length?TP/flagged.length:0, recall=positives?TP/positives:0,prev=rows.length?positives/rows.length:0;
 const lift=prev?prec/prev:0, exposure=flagged.reduce((a,r)=>a+r.monthly_charges_usd*r.risk_probability,0);
 byId('riskHint').textContent=`${flagged.length} / ${rows.length} customers`;
 const metrics=[['Eligible customers',rows.length],['Flagged for review',flagged.length],['Precision',fPct(prec)],['Churn capture (recall)',fPct(recall)],['Lift over segment baseline',lift.toFixed(2)+'×'],['TP / FP / FN',`${TP} / ${FP} / ${FN}`],['False negatives',FN],['Monthly charge exposure proxy','$'+exposure.toFixed(0)]];
 byId('metrics').innerHTML=metrics.map(([k,v])=>`<div class="metric"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`).join('');
 byId('effectLabel').textContent=byId('effect').value+'%';
 byId('scenario').textContent=`With assumed ${(Number(byId('effect').value))}% outreach effectiveness, scenario expected saved ≈ ${(TP*Number(byId('effect').value)/100).toFixed(1)} customers (not a proven outcome).`;
 byId('queue').innerHTML=flagged.slice(0,200).map((r,i)=>`<tr><td>${i+1}</td><td>${esc(r.customer_id)}</td><td>${esc(r.industry_segment)}</td><td>${esc(r.plan_tier)}</td><td><span class="tag">${r.risk_percentage.toFixed(1)}%</span></td><td>$${r.monthly_charges_usd.toFixed(2)}</td><td>${r.usage_change_pct.toFixed(1)}%</td><td>${r.engagement_change_pct.toFixed(1)}%</td><td>${r.billing_issues_90d}</td><td class="explain">${esc(r.top_risk_drivers)}</td></tr>`).join('');
}
function downloadCSV(){
 const cols=['customer_id','industry_segment','plan_tier','risk_probability','monthly_charges_usd','usage_change_pct','engagement_change_pct','billing_issues_90d','top_risk_drivers'];
 const escCSV=x=>'"'+String(x??'').replaceAll('"','""')+'"';
 let csv=[cols.join(','),...currentFlagged.map(r=>cols.map(c=>escCSV(r[c])).join(','))].join('\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='flagged_retention_queue.csv';a.click();URL.revokeObjectURL(a.href);
}
const best=DATA.global_drivers[0].mean_abs_shap_log_odds;
byId('importance').innerHTML=DATA.global_drivers.map(r=>`<div class="barrow"><span>${esc(r.feature)}</span><div class="barbg"><div class="bar" style="width:${Math.round(100*r.mean_abs_shap_log_odds/best)}%"></div></div><strong>${r.mean_abs_shap_log_odds.toFixed(3)}</strong></div>`).join('');
byId('modeltable').innerHTML=Object.entries(DATA.models).map(([name,v])=>`<tr><td><b>${esc(name)}</b></td><td>${fmt(v.pr_auc_average_precision)}</td><td>${fmt(v.roc_auc)}</td><td>${fmt(v.precision)}</td><td>${fmt(v.recall)}</td><td>${fmt(v.f1)}</td><td>${fmt(v.brier_score)}</td></tr>`).join('');
['industry','plan','cap','effect'].forEach(id=>byId(id).addEventListener('input',update));
document.querySelectorAll('input[name="mode"]').forEach(i=>i.addEventListener('change',()=>{byId('cap').value=i.value==='capacity'?20:Math.round(DATA.selected*100);update();}));
update();
</script></body></html>
'''
(out/'offline_retention_dashboard.html').write_text(html.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))),encoding='utf8')
print(out/'offline_retention_dashboard.html')
