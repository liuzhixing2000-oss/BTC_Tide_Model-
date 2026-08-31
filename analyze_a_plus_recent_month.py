#!/usr/bin/env python3
from __future__ import annotations
import math, os, subprocess, sys
from pathlib import Path
import pandas as pd

BASE = Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
INFILE = BASE/'portfolio_trades.csv'
OUT = BASE/'a_plus_recent_month'
OUT.mkdir(parents=True, exist_ok=True)
START_EQUITY = 1000.0


def pf(x):
    x = pd.Series(x, dtype=float)
    gp=x[x>0].sum(); gl=-x[x<0].sum()
    return float(gp/gl) if gl>0 else math.inf


def metrics(df):
    if df.empty: return {'n':0}
    p=pd.to_numeric(df.pnl_usdt,errors='coerce').fillna(0.0)
    risk=pd.to_numeric(df.actual_risk_usdt,errors='coerce').replace(0,pd.NA)
    r=p/risk
    eq=START_EQUITY+p.cumsum()
    peak=eq.cummax()
    dd=(eq/peak-1).min() if len(eq) else 0
    return {'n':len(df),'pnl':p.sum(),'ret':p.sum()/START_EQUITY,'win':(p>0).mean(),'pf':pf(p),'meanR':r.mean(),'medianR':r.median(),'maxdd':dd,'stop':(df.exit_reason.astype(str)=='hard_structure_stop').mean()}


def ensure_replay():
    if INFILE.exists():
        print(f'Using existing replay: {INFILE}', flush=True)
        return
    print('='*90, flush=True)
    print('TIDE V10.9 — RECENT 30-DAY A+ ANALYSIS', flush=True)
    print('='*90, flush=True)
    print(f'Missing {INFILE}. Auto-generating the V10.9 100-day replay in THIS Railway service...', flush=True)
    cmd=[sys.executable,'backtest_v10_9_six_month.py','--days','100','--max-symbols','0','--out',str(BASE)]
    print('Replay command:',' '.join(cmd),flush=True)
    rc=subprocess.call(cmd)
    if rc!=0:
        raise SystemExit(rc)
    if not INFILE.exists():
        raise SystemExit(f'Replay finished but {INFILE} is still missing.')


def main():
    ensure_replay()
    df=pd.read_csv(INFILE)
    df['entry_time']=pd.to_datetime(df.entry_time,utc=True,errors='coerce')
    df=df.dropna(subset=['entry_time']).sort_values('entry_time')
    a=df[df.grade.astype(str)=='A+'].copy()
    if a.empty: raise SystemExit('No A+ trades')

    # Exact latest 30 calendar days ending at the replay's latest admitted-trade timestamp.
    end=df.entry_time.max()
    start=end-pd.Timedelta(days=30)
    recent=a[(a.entry_time>=start)&(a.entry_time<=end)].copy()
    recent['R']=pd.to_numeric(recent.pnl_usdt,errors='coerce')/pd.to_numeric(recent.actual_risk_usdt,errors='coerce').replace(0,pd.NA)
    recent.to_csv(OUT/'a_plus_last_30d_trades.csv',index=False)
    m=metrics(recent)

    lines=['# Tide V10.9 — A+ exact latest 30-day performance','',f'Window: {start.isoformat()} -> {end.isoformat()}','',f"- A+ trades: **{m.get('n',0)}**"]
    if m.get('n',0):
        lines += [f"- Net P/L: **{m['pnl']:+.2f} USDT**",f"- Return on 1000U: **{m['ret']:+.2%}**",f"- Win rate: **{m['win']:.1%}**",f"- Profit factor: **{m['pf']:.2f}**",f"- Mean R: **{m['meanR']:+.3f}R**",f"- Median R: **{m['medianR']:+.3f}R**",f"- Max drawdown: **{m['maxdd']:.2%}**",f"- Stop rate: **{m['stop']:.1%}**",'', '## Trades']
        for _,r in recent.iterrows():
            lines.append(f"- {r.entry_time.isoformat()} | {r.get('symbol','?')} | score={r.get('signal_score','?')} | risk={r.get('actual_risk_usdt','?')}U | P/L={float(r.get('pnl_usdt',0)):+.2f}U | R={float(r.get('R',0)):+.3f}")
    summary='\n'.join(lines)
    (OUT/'SUMMARY.md').write_text(summary,encoding='utf-8')
    print(summary, flush=True)
    print('RECENT_A_PLUS_DIR:',OUT,flush=True)

if __name__=='__main__': main()
