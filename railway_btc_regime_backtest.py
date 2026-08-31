#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path

SOURCE=os.getenv('BTC_FILTER_SOURCE','/data/v10_9_six_month/candidate_trades.csv')
OUT=os.getenv('BTC_FILTER_OUT','/data/v10_9_six_month/btc_regime_analysis')
cmd=[sys.executable,'analyze_btc_regime_filter.py','--source',SOURCE,'--out',OUT]
print('='*90,flush=True)
print('BTC REGIME FILTER BACKTEST',flush=True)
print('Command:',' '.join(cmd),flush=True)
print('='*90,flush=True)
rc=subprocess.call(cmd)
if rc!=0:raise SystemExit(rc)
print(f'RESULT_DIR: {OUT}',flush=True)
print('Backtest finished successfully.',flush=True)
