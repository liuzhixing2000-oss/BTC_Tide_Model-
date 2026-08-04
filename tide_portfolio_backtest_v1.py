#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class Pos:
    trade_id:int; symbol:str; entry_time:pd.Timestamp; exit_time:pd.Timestamp
    margin:float; leverage:float; net_return:float; priority:float

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--trades',type=Path,default=Path('/data/ab_results_all/universe_ab_trades.csv'))
    p.add_argument('--symbols',type=Path,default=Path('/data/ab_results_all/universe_ab_symbol_details.csv'))
    p.add_argument('--config',default='all_eligible')
    p.add_argument('--initial-capital',type=float,default=3000)
    p.add_argument('--margin-per-trade',type=float,default=100)
    p.add_argument('--leverage',type=float,default=10)
    p.add_argument('--max-positions',type=int,default=12)
    p.add_argument('--max-margin-utilisation',type=float,default=0.80)
    p.add_argument('--same-symbol-limit',type=int,default=1)
    p.add_argument('--output-dir',type=Path,default=Path('/data/portfolio_results'))
    return p.parse_args()

def rank(s):
    x=pd.to_numeric(s,errors='coerce')
    return x.rank(pct=True).fillna(0.5) if x.notna().sum()>1 else pd.Series(0.5,index=s.index)

def load(a):
    t=pd.read_csv(a.trades); s=pd.read_csv(a.symbols)
    if 'config' in t: t=t[t.config==a.config].copy()
    if 'config' in s: s=s[s.config==a.config].copy()
    for c in ['entry_time','exit_time']: t[c]=pd.to_datetime(t[c],utc=True)
    t['net_return']=pd.to_numeric(t['net_return'],errors='coerce')
    t=t.dropna(subset=['entry_time','exit_time','net_return'])
    comps=[]
    for c,w in [('score',.30),('validation_expectancy',.25),('validation_profit_factor',.20),('validation_return',.15),('max_drawdown',.10)]:
        if c in s: comps.append((w,rank(s[c].replace(np.inf,10))))
    s['priority_score']=sum(w*v for w,v in comps)/sum(w for w,_ in comps) if comps else .5
    t=t.merge(s[['symbol','priority_score']].drop_duplicates('symbol'),on='symbol',how='left')
    t['priority_score']=t['priority_score'].fillna(.5); t['trade_id']=range(len(t))
    return t.sort_values(['entry_time','priority_score'],ascending=[True,False]).reset_index(drop=True)

def simulate(t,a):
    cash=a.initial_capital; openp=[]; ledger=[]; eq=[]
    def close_due(now):
        nonlocal cash,openp
        keep=[]
        for p in sorted(openp,key=lambda z:z.exit_time):
            if p.exit_time<=now:
                pnl=p.margin*p.leverage*p.net_return; cash+=p.margin+pnl
                ledger.append({'trade_id':p.trade_id,'symbol':p.symbol,'entry_time':p.entry_time,'exit_time':p.exit_time,'margin':p.margin,'leverage':p.leverage,'notional':p.margin*p.leverage,'net_return_unlevered':p.net_return,'pnl':pnl,'priority_score':p.priority,'status':'executed'})
                eq.append({'time':p.exit_time,'equity':cash+sum(x.margin for x in openp if x.exit_time>p.exit_time)})
            else: keep.append(p)
        openp=keep
    eq.append({'time':t.entry_time.min(),'equity':cash})
    for et,b in t.groupby('entry_time',sort=True):
        close_due(et)
        for r in b.sort_values('priority_score',ascending=False).itertuples():
            equity=cash+sum(p.margin for p in openp); locked=sum(p.margin for p in openp)
            reason=None
            if len(openp)>=a.max_positions: reason='max_positions'
            elif sum(p.symbol==r.symbol for p in openp)>=a.same_symbol_limit: reason='same_symbol_limit'
            elif locked+a.margin_per_trade>equity*a.max_margin_utilisation+1e-9: reason='max_margin_utilisation'
            elif cash<a.margin_per_trade: reason='insufficient_cash'
            if reason:
                ledger.append({'trade_id':r.trade_id,'symbol':r.symbol,'entry_time':r.entry_time,'exit_time':r.exit_time,'net_return_unlevered':r.net_return,'priority_score':r.priority_score,'status':'rejected','rejection_reason':reason}); continue
            cash-=a.margin_per_trade
            openp.append(Pos(r.trade_id,r.symbol,r.entry_time,r.exit_time,a.margin_per_trade,a.leverage,float(r.net_return),float(r.priority_score)))
            eq.append({'time':et,'equity':cash+sum(p.margin for p in openp)})
    if openp: close_due(max(p.exit_time for p in openp))
    L=pd.DataFrame(ledger); E=pd.DataFrame(eq).sort_values('time')
    E['peak']=E.equity.cummax(); E['drawdown']=E.equity/E.peak-1
    X=L[L.status=='executed'].copy(); wins=X.loc[X.pnl>0,'pnl'].sum(); losses=-X.loc[X.pnl<0,'pnl'].sum()
    pf=math.inf if losses==0 and wins>0 else (wins/losses if losses>0 else math.nan)
    S={'initial_capital':a.initial_capital,'final_equity':float(cash),'total_return':float(cash/a.initial_capital-1),'executed_trades':int(len(X)),'rejected_trades':int((L.status=='rejected').sum()),'execution_rate':float(len(X)/len(t)),'win_rate':float((X.pnl>0).mean()),'profit_factor':float(pf),'max_drawdown':float(E.drawdown.min()),'margin_per_trade':a.margin_per_trade,'leverage':a.leverage,'max_positions':a.max_positions,'max_margin_utilisation':a.max_margin_utilisation}
    return L,E,S

def main():
    a=args(); t=load(a); L,E,S=simulate(t,a); a.output_dir.mkdir(parents=True,exist_ok=True)
    L.to_csv(a.output_dir/'portfolio_trade_ledger.csv',index=False); E.to_csv(a.output_dir/'portfolio_equity_curve.csv',index=False)
    E['month']=E.time.dt.to_period('M').astype(str); M=E.groupby('month',as_index=False).agg(start_equity=('equity','first'),end_equity=('equity','last'),max_drawdown=('drawdown','min')); M['return']=M.end_equity/M.start_equity-1; M.to_csv(a.output_dir/'portfolio_monthly_returns.csv',index=False)
    (a.output_dir/'portfolio_summary.json').write_text(json.dumps(S,indent=2),encoding='utf-8')
    report='\n'.join(['TIDE PORTFOLIO BACKTEST V1',f"Initial capital: {S['initial_capital']:.2f}",f"Final equity: {S['final_equity']:.2f}",f"Total return: {S['total_return']:.2%}",f"Max drawdown: {S['max_drawdown']:.2%}",f"Executed trades: {S['executed_trades']}",f"Rejected trades: {S['rejected_trades']}",f"Execution rate: {S['execution_rate']:.2%}",f"Win rate: {S['win_rate']:.2%}",f"Profit factor: {S['profit_factor']:.3f}"])+"\n"
    (a.output_dir/'portfolio_report.txt').write_text(report,encoding='utf-8'); print(report)
if __name__=='__main__': main()
