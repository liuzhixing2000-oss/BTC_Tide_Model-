#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path

SOURCE=Path(os.getenv('BTC_FILTER_SOURCE','/data/v10_9_six_month/candidate_trades.csv'))
OUT=os.getenv('BTC_FILTER_OUT','/data/v10_9_six_month/btc_regime_analysis')
DAYS=os.getenv('BACKTEST_DAYS','100')
MAX_SYMBOLS=os.getenv('BACKTEST_MAX_SYMBOLS','0')
REPLAY_OUT=SOURCE.parent

print('='*90,flush=True)
print('BTC REGIME FILTER BACKTEST',flush=True)
print('='*90,flush=True)

if not SOURCE.exists():
    print(f'Missing {SOURCE}. This Railway service has a separate /data volume.',flush=True)
    print('Auto-generating the V10.9 replay in THIS service before BTC-filter analysis...',flush=True)
    replay=[sys.executable,'backtest_v10_9_six_month.py','--days',str(DAYS),'--max-symbols',str(MAX_SYMBOLS),'--out',str(REPLAY_OUT)]
    print('Replay command:',' '.join(replay),flush=True)
    rc=subprocess.call(replay)
    if rc!=0:
        raise SystemExit(f'V10.9 replay failed with exit code {rc}')
    if not SOURCE.exists():
        raise SystemExit(f'Replay completed but {SOURCE} was not created.')
else:
    print(f'Using existing replay source: {SOURCE}',flush=True)

cmd=[sys.executable,'analyze_btc_regime_filter.py','--source',str(SOURCE),'--out',OUT]
print('Analysis command:',' '.join(cmd),flush=True)
rc=subprocess.call(cmd)
if rc!=0:raise SystemExit(rc)
print(f'RESULT_DIR: {OUT}',flush=True)
print('Backtest finished successfully.',flush=True)
