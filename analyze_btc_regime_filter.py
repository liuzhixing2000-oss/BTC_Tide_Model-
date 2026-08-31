#!/usr/bin/env python3
"""Compare V10.9 baseline vs BTC-regime-filtered altcoin portfolios.

Uses the existing candidate_trades.csv from the V10.9 replay, so it does NOT
redownload 234 altcoin histories. It fetches BTCUSDT 15m + 1h once, annotates
each candidate at its entry time, and re-runs portfolio admission under several
BTC gates.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

import backtest_v10_9_six_month as bt

VARIANTS = {
    'baseline': lambda r: False,
    'btc_below_ema20': lambda r: bool(r['btc_15m_below_ema20']),
    'btc_1h_down': lambda r: bool(r['btc_1h_down']),
    'combined': lambda r: bool(r['btc_15m_below_ema20'] and r['btc_1h_down']),
    'acute_drop': lambda r: bool((r['btc_ret_15m'] <= -0.004) or (r['btc_ret_1h'] <= -0.008)),
    'combined_or_acute': lambda r: bool((r['btc_15m_below_ema20'] and r['btc_1h_down']) or (r['btc_ret_15m'] <= -0.004) or (r['btc_ret_1h'] <= -0.008)),
}


def btc_context(d15: pd.DataFrame, d1h: pd.DataFrame, t: pd.Timestamp) -> dict:
    t = pd.Timestamp(t).tz_convert('UTC') if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz='UTC')
    x15 = d15.copy(); x15['open_time']=pd.to_datetime(x15.open_time,utc=True)
    x1 = d1h.copy(); x1['open_time']=pd.to_datetime(x1.open_time,utc=True)
    # Only completed candles available at signal time.
    a15 = x15[x15.open_time + pd.Timedelta(minutes=15) <= t].copy()
    a1 = x1[x1.open_time + pd.Timedelta(hours=1) <= t].copy()
    if len(a15) < 25 or len(a1) < 205:
        return {'btc_15m_below_ema20':False,'btc_1h_down':False,'btc_ret_15m':0.0,'btc_ret_1h':0.0,'btc_15m_close':np.nan,'btc_15m_ema20':np.nan}
    c15=a15.close.astype(float); ema20=c15.ewm(span=20,adjust=False).mean()
    c1=a1.close.astype(float); e50=c1.ewm(span=50,adjust=False).mean(); e200=c1.ewm(span=200,adjust=False).mean(); slope=e50.pct_change(3)
    cc=float(c1.iloc[-1]); aa=float(e50.iloc[-1]); bb=float(e200.iloc[-1]); ss=float(slope.iloc[-1])
    down=bool(cc<aa<bb and ss<0)
    ret15=float(c15.iloc[-1]/c15.iloc[-2]-1) if len(c15)>=2 else 0.0
    ret1=float(c1.iloc[-1]/c1.iloc[-2]-1) if len(c1)>=2 else 0.0
    return {'btc_15m_below_ema20':bool(c15.iloc[-1] < ema20.iloc[-1]),'btc_1h_down':down,
            'btc_ret_15m':ret15,'btc_ret_1h':ret1,'btc_15m_close':float(c15.iloc[-1]),'btc_15m_ema20':float(ema20.iloc[-1])}


def stats(tr: pd.DataFrame, start, end, candidates, rejected):
    s=bt.summary(tr,start,end,candidates,rejected)
    if tr.empty:
        s.update({'profit_factor':0.0,'max_drawdown':0.0,'win_rate':0.0,'stop_rate':0.0})
    return s


def by_grade(tr):
    out=[]
    if tr.empty:return pd.DataFrame(columns=['grade','trades','win_rate','net_pnl','avg_net_return','stop_rate'])
    for g,x in tr.groupby('grade'):
        out.append({'grade':g,'trades':len(x),'win_rate':float((x.pnl_usdt>0).mean()),'net_pnl':float(x.pnl_usdt.sum()),
                    'avg_net_return':float(x.net_return.mean()),'stop_rate':float((x.exit_reason=='hard_structure_stop').mean())})
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='/data/v10_9_six_month/candidate_trades.csv');ap.add_argument('--out',default='/data/v10_9_six_month/btc_regime_analysis');a=ap.parse_args()
    src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not src.exists():raise SystemExit(f'Missing {src}; run the V10.9 replay first.')
    cand=pd.read_csv(src);cand['entry_time']=pd.to_datetime(cand.entry_time,utc=True);cand['exit_time']=pd.to_datetime(cand.exit_time,utc=True)
    start=cand.entry_time.min().to_pydatetime();end=cand.entry_time.max().to_pydatetime()+pd.Timedelta(hours=6)
    fetch_start=start-pd.Timedelta(days=15)
    print(f'Candidates loaded: {len(cand)} | {start.isoformat()} -> {end.isoformat()}',flush=True)
    d15=bt.fetch_range('BTCUSDT','15',fetch_start,end);d1=bt.fetch_range('BTCUSDT','60',start-pd.Timedelta(days=25),end)
    print(f'BTC data: 15m={len(d15)} 1h={len(d1)}',flush=True)
    ctx=[]
    for n,t in enumerate(cand.entry_time,1):
        ctx.append(btc_context(d15,d1,t))
        if n%50==0:print('annotated',n,'/',len(cand),flush=True)
    ann=pd.concat([cand.reset_index(drop=True),pd.DataFrame(ctx)],axis=1)
    results=[];grade_rows=[];blocked_rows=[]
    for name,gate in VARIANTS.items():
        mask=[]
        for r in ann.to_dict('records'):
            blocked=False if r['symbol']=='BTCUSDT' else gate(r)
            mask.append(blocked)
        ann[f'blocked_{name}']=mask
        kept=ann[~pd.Series(mask,index=ann.index)].copy();blocked=ann[pd.Series(mask,index=ann.index)].copy()
        accepted,rejected=bt.portfolio(kept)
        s=stats(accepted,start,end,kept,rejected);s['variant']=name;s['blocked_signals']=int(len(blocked));s['kept_candidates']=int(len(kept));results.append(s)
        bg=by_grade(accepted)
        if not bg.empty:
            bg['variant']=name;grade_rows.append(bg)
        if not blocked.empty:
            b=blocked.copy();b['variant']=name;blocked_rows.append(b)
        print(name, 'blocked',len(blocked),'admitted',len(accepted),'pnl',s.get('net_pnl'),'pf',s.get('profit_factor'),'mdd',s.get('max_drawdown'),flush=True)
    res=pd.DataFrame(results);res.to_csv(out/'variant_summary.csv',index=False);ann.to_csv(out/'annotated_candidates.csv',index=False)
    if grade_rows:pd.concat(grade_rows,ignore_index=True).to_csv(out/'by_grade.csv',index=False)
    if blocked_rows:pd.concat(blocked_rows,ignore_index=True).to_csv(out/'blocked_signals.csv',index=False)
    base=res[res.variant=='baseline'].iloc[0]
    lines=['# BTC Regime Filter Backtest','',f'Candidates: **{len(cand)}**',f'Window: {start.isoformat()} → {end.isoformat()}','',
           '## Variant comparison','',
           '| Variant | Blocked | Admitted | Net P/L | Return | Win rate | PF | Max DD | Stop rate |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in res.iterrows():
        lines.append(f"| {r['variant']} | {int(r['blocked_signals'])} | {int(r.get('admitted',r.get('trades',0)))} | {r.get('net_pnl',0):+.2f}U | {100*r.get('return_pct',0):+.2f}% | {100*r.get('win_rate',0):.1f}% | {r.get('profit_factor',0):.2f} | {100*r.get('max_drawdown',0):.2f}% | {100*r.get('stop_rate',0):.1f}% |")
    lines += ['', '## Filter definitions','',
              '- `btc_below_ema20`: block alt longs when completed BTC 15m close < BTC 15m EMA20.',
              '- `btc_1h_down`: block alt longs when completed BTC 1h is below EMA50 < EMA200 with negative EMA50 slope.',
              '- `combined`: block only when both conditions are true.',
              '- `acute_drop`: block if last completed BTC 15m <= -0.4% OR last completed BTC 1h <= -0.8%.',
              '- `combined_or_acute`: combined filter plus acute-drop protection.',
              '', '## Interpretation','',
              'Prefer a filter only if PF and/or drawdown improve materially without eliminating most A+ trades. This remains an in-sample policy replay using the current eligible-symbol bundle; forward ledger remains the clean OOS test.']
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
