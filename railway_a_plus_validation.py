#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

BASE=Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
PORTFOLIO=BASE/'portfolio_trades.csv'

print('='*90, flush=True)
print('V10.9 A+ EDGE VALIDATION', flush=True)
print('='*90, flush=True)

if not PORTFOLIO.exists():
    print(f'Missing {PORTFOLIO}. This Railway service does not currently have the replay output.', flush=True)
    print('Auto-generating the V10.9 100-day replay in THIS service before A+ validation...', flush=True)
    replay=[sys.executable,'backtest_v10_9_six_month.py','--days','100','--max-symbols','0','--out',str(BASE)]
    print('Replay command:',' '.join(replay),flush=True)
    rc=subprocess.call(replay)
    if rc!=0:
        raise SystemExit(rc)

if not PORTFOLIO.exists():
    raise SystemExit(f'Replay finished but {PORTFOLIO} is still missing.')

cmd=[sys.executable,'validate_a_plus_edge.py']
print('Validation command:',' '.join(cmd),flush=True)
rc=subprocess.call(cmd)
if rc!=0:
    raise SystemExit(rc)

print('A_PLUS_VALIDATION_DIR:',BASE/'a_plus_validation',flush=True)
print('A+ edge validation finished successfully.',flush=True)
