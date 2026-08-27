#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
INFILE=BASE/'portfolio_trades.csv'
OUT=BASE/'analysis'
OUT.mkdir(parents=True,exist_ok=True)
ONE_R=20.0
START_EQUITY=1000.0
RNG=np.random.default_rng(42)


def pf(x):
    p=x[x>0].sum(); n=-x[x<0].sum(); return float(p/n) if n>0 else math.inf

def maxdd(pnls):
    if len(pnls)==0:return 0.0
    eq=START_EQUITY+np.cumsum(np.asarray(pnls,float)); peak=np.maximum.accumulate(eq); return float(np.min(eq/peak-1))

def metrics(df,label):
    if df.empty:return {'label':label,'n':0}
    p=df.pnl_usdt.astype(float)
    return {'label':label,'n':len(df),'win_rate':float((p>0).mean()),'pnl_usdt':float(p.sum()),'return_on_1000':float(p.sum()/START_EQUITY),'avg_pnl':float(p.mean()),'median_pnl':float(p.median()),'profit_factor':pf(p),'max_drawdown':maxdd(p.to_numpy()),'avg_net_return':float(df.net_return.astype(float).mean()),'stop_rate':float((df.exit_reason=='hard_structure_stop').mean())}

def normalized_r(df):
    x=df.copy(); denom=pd.to_numeric(x.actual_risk_usdt,errors='coerce').replace(0,np.nan); x['r_multiple']=pd.to_numeric(x.pnl_usdt,errors='coerce')/denom
    return x

def bootstrap_mean_r(vals,nboot=20000):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)==0:return [None,None,None]
    idx=RNG.integers(0,len(vals),size=(nboot,len(vals))); means=vals[idx].mean(axis=1)
    return [float(vals.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def mc_drawdown(vals,n=10000):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if not len(vals):return {}
    d=[]
    for _ in range(n): d.append(maxdd(RNG.permutation(vals)))
    return {'median':float(np.median(d)),'p90_worse':float(np.quantile(d,.10)),'p95_worse':float(np.quantile(d,.05)),'worst':float(np.min(d))}

def main():
    if not INFILE.exists():raise SystemExit(f'Missing {INFILE}')
    df=pd.read_csv(INFILE)
    df['entry_time']=pd.to_datetime(df.entry_time,utc=True,errors='coerce'); df['month']=df.entry_time.dt.strftime('%Y-%m')
    df=normalized_r(df)

    scenarios={
      'A+ only':{'A+'},'A+ + B+':{'A+','B+'},'A+ + A':{'A+','A'},'All current':{'A+','A','A-','B+'}
    }
    scen=[]
    for name,grades in scenarios.items():scen.append(metrics(df[df.grade.isin(grades)],name))
    pd.DataFrame(scen).to_csv(OUT/'scenario_comparison.csv',index=False)

    score=[]
    for th in [70,75,80,85,90]:score.append(metrics(df[pd.to_numeric(df.signal_score,errors='coerce')>=th],f'score>={th}'))
    pd.DataFrame(score).to_csv(OUT/'score_thresholds.csv',index=False)

    monthly=[]
    for (g,m),x in df.groupby(['grade','month']):
        z=metrics(x,f'{g}-{m}');z.update({'grade':g,'month':m});monthly.append(z)
    pd.DataFrame(monthly).to_csv(OUT/'grade_monthly.csv',index=False)

    sym=[]
    for (g,s),x in df.groupby(['grade','symbol']):
        z=metrics(x,f'{g}-{s}');z.update({'grade':g,'symbol':s});sym.append(z)
    sm=pd.DataFrame(sym).sort_values('pnl_usdt',ascending=False);sm.to_csv(OUT/'grade_symbol.csv',index=False)

    robust=[]
    for g,x in df.groupby('grade'):
        base=float(x.pnl_usdt.sum()); ordered=x.sort_values('pnl_usdt',ascending=False)
        row={'grade':g,'n':len(x),'base_pnl':base}
        for k in [1,3,5]:row[f'pnl_remove_top_{k}']=float(ordered.iloc[k:].pnl_usdt.sum()) if len(ordered)>k else np.nan
        vals=x.r_multiple.dropna().to_numpy();mean,lo,hi=bootstrap_mean_r(vals);row.update({'mean_R':mean,'bootstrap_R_2.5%':lo,'bootstrap_R_97.5%':hi,**{f'mc_dd_{k}':v for k,v in mc_drawdown(x.pnl_usdt.to_numpy()).items()}});robust.append(row)
    pd.DataFrame(robust).to_csv(OUT/'robustness_by_grade.csv',index=False)

    norm=[]
    for g,x in df.groupby('grade'):
        r=x.r_multiple.replace([np.inf,-np.inf],np.nan).dropna(); norm.append({'grade':g,'n':len(r),'mean_R':float(r.mean()),'median_R':float(r.median()),'win_rate':float((r>0).mean()),'PF_R':pf(r),'sum_R':float(r.sum())})
    pd.DataFrame(norm).sort_values('mean_R',ascending=False).to_csv(OUT/'normalized_R_by_grade.csv',index=False)

    a=df[df.grade=='A+'].sort_values('pnl_usdt',ascending=False)
    conc={'a_plus_n':len(a),'a_plus_total_pnl':float(a.pnl_usdt.sum())}
    for k in [1,3,5,10]:
        conc[f'top_{k}_pnl']=float(a.head(k).pnl_usdt.sum()); conc[f'top_{k}_share']=float(a.head(k).pnl_usdt.sum()/a.pnl_usdt.sum()) if a.pnl_usdt.sum()!=0 else None
    (OUT/'a_plus_concentration.json').write_text(json.dumps(conc,indent=2))

    lines=['# V10.9 decomposition','']
    lines+=['## Scenario comparison','']+[f"- {r['label']}: n={r.get('n',0)}, P/L={r.get('pnl_usdt',0):+.2f}U, return={r.get('return_on_1000',0):+.2%}, win={r.get('win_rate',0):.1%}, PF={r.get('profit_factor',0):.2f}, maxDD={r.get('max_drawdown',0):.2%}" for r in scen]
    lines+=['','## Normalized R by grade','']
    for r in norm:lines.append(f"- {r['grade']}: n={r['n']}, mean={r['mean_R']:+.3f}R, median={r['median_R']:+.3f}R, sum={r['sum_R']:+.2f}R, PF(R)={r['PF_R']:.2f}")
    lines+=['','## Robustness by grade','']
    for r in robust:lines.append(f"- {r['grade']}: meanR={r['mean_R']:+.3f} [{r['bootstrap_R_2.5%']:+.3f},{r['bootstrap_R_97.5%']:+.3f}], base={r['base_pnl']:+.2f}U, remove best1={r['pnl_remove_top_1']:+.2f}U, best3={r['pnl_remove_top_3']:+.2f}U, best5={r['pnl_remove_top_5']:+.2f}U")
    lines+=['','## A+ concentration','',json.dumps(conc,ensure_ascii=False)]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    zipbase=str(BASE/'v10_9_decomposition');z=shutil.make_archive(zipbase,'zip',OUT);print((OUT/'SUMMARY.md').read_text());print('ANALYSIS_ZIP:',z)

if __name__=='__main__':main()
