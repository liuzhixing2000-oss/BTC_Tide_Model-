#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

BASE=Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
PORTFOLIO=BASE/'portfolio_trades.csv'

print('='*90, flush=True)
print('V10.9 GRADE / SCORE DECOMPOSITION', flush=True)
print('='*90, flush=True)

if not PORTFOLIO.exists():
    print(f'Missing {PORTFOLIO}. This Railway service does not currently have the replay output.', flush=True)
    print('Auto-generating the V10.9 replay in THIS service before grade analysis...', flush=True)
    replay=[sys.executable,'backtest_v10_9_six_month.py','--days','100','--max-symbols','0','--out',str(BASE)]
    print('Replay command:',' '.join(replay),flush=True)
    rc=subprocess.call(replay)
    if rc!=0:
        raise SystemExit(rc)

if not PORTFOLIO.exists():
    raise SystemExit(f'Replay finished but {PORTFOLIO} is still missing.')

cmd=[sys.executable,'analyze_v10_9_backtest.py']
print('Analysis command:',' '.join(cmd),flush=True)
rc=subprocess.call(cmd)
if rc!=0:
    raise SystemExit(rc)
print('GRADE_ANALYSIS_DIR:',BASE/'analysis',flush=True)
print('Grade analysis finished successfully.',flush=True)
