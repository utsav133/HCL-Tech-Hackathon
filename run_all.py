"""One command to rebuild every D-01 through D-05 output.
Run from project root: python run_all.py
"""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=['eda_report.py','train_and_evaluate.py','make_figures.py','build_offline_dashboard.py','validate_outputs.py']
for name in SCRIPTS:
    print('\n==== RUNNING', name,'====',flush=True)
    subprocess.run([sys.executable,str(ROOT/'src'/name)],cwd=ROOT,check=True)
print('\nSUCCESS: all five delivery groups reproduced. See outputs/ and launch streamlit run app.py',flush=True)
