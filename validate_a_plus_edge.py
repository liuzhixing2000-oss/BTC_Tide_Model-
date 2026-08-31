#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v10_9_six_month import fetch_range

BASE = Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
INFILE = BASE/'portfolio_trades.csv'
OUT = BASE/'a_plus_validation'
OUT.mkdir(parents=True, exist_ok=True)
START_EQUITY = 1000.0
RNG = np.random.default_rng(20260831)


def pf(x):
    x = pd.Series(x, dtype=float)
    gp = x[x>0].sum(); gl = -x[x<0].sum()
    return float(gp/gl) if gl > 0 else math.inf


def maxdd(pnls):
    a = np.asarray(pnls, float)
    if len(a)==0: return 0.0
    eq = START_EQUITY + np.cumsum(a)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq/peak - 1))


def metrics(df, label):
    if df.empty:
        return {'label':label,'n':0}
    p = pd.to_numeric(df.pnl_usdt, errors='coerce').fillna(0.0)
    risk = pd.to_numeric(df.actual_risk_usdt, errors='coerce').replace(0,np.nan)
    r = p/risk
    return {
        'label':label,'n':len(df),'pnl_usdt':float(p.sum()),'return_on_1000':float(p.sum()/START_EQUITY),
        'win_rate':float((p>0).mean()),'profit_factor':pf(p),'max_drawdown':maxdd(p.to_numpy()),
        'mean_R':float(r.mean()),'median_R':float(r.median()),'sum_R':float(r.sum()),'PF_R':pf(r.dropna()),
        'stop_rate':float((df.exit_reason.astype(str)=='hard_structure_stop').mean())
    }


def bootstrap_mean(vals, nboot=30000):
    v=np.asarray(vals,float);v=v[np.isfinite(v)]
    if not len(v): return (np.nan,np.nan,np.nan)
    idx=RNG.integers(0,len(v),size=(nboot,len(v)))
    m=v[idx].mean(axis=1)
    return float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))


def add_btc_regime(a):
    start = a.entry_time.min() - pd.Timedelta(days=5)
    end = a.entry_time.max() + pd.Timedelta(hours=2)
    d15 = fetch_range('BTCUSDT','15',start.to_pydatetime(),end.to_pydatetime())
    d1 = fetch_range('BTCUSDT','60',(start-pd.Timedelta(days=10)).to_pydatetime(),end.to_pydatetime())
    for d in (d15,d1):
        d['open_time']=pd.to_datetime(d.open_time,utc=True)
    d15['close_time']=d15.open_time+pd.Timedelta(minutes=15)
    d15['ema20']=d15.close.ewm(span=20,adjust=False).mean()
    d15['ret15']=d15.close.pct_change()
    d1['close_time']=d1.open_time+pd.Timedelta(hours=1)
    d1['ema20']=d1.close.ewm(span=20,adjust=False).mean()
    d1['ema50']=d1.close.ewm(span=50,adjust=False).mean()
    d1['ret1h']=d1.close.pct_change()
    d1['slope20']=d1.ema20.pct_change(3)

    left=a.sort_values('entry_time').copy()
    x=pd.merge_asof(left,d15[['close_time','close','ema20','ret15']].sort_values('close_time'),left_on='entry_time',right_on='close_time',direction='backward')
    x=x.rename(columns={'close':'btc15_close','ema20':'btc15_ema20'})
    x=pd.merge_asof(x.sort_values('entry_time'),d1[['close_time','close','ema20','ema50','ret1h','slope20']].sort_values('close_time'),left_on='entry_time',right_on='close_time',direction='backward',suffixes=('','_1h'))
    x=x.rename(columns={'close':'btc1h_close','ema20':'btc1h_ema20','ema50':'btc1h_ema50'})
    bull=(x.btc15_close>=x.btc15_ema20)&(x.btc1h_close>=x.btc1h_ema20)&(x.btc1h_ema20>=x.btc1h_ema50)&(x.slope20>=0)
    bear=(x.btc15_close<x.btc15_ema20)&(x.btc1h_close<x.btc1h_ema20)&(x.btc1h_ema20<x.btc1h_ema50)&(x.slope20<0)
    x['btc_regime']=np.select([bull,bear],['BULL','BEAR'],default='NEUTRAL')
    x['btc_acute_drop']=(x.ret15<=-0.004)|(x.ret1h<=-0.008)
    return x


def main():
    if not INFILE.exists(): raise SystemExit(f'Missing {INFILE}')
    df=pd.read_csv(INFILE)
    df['entry_time']=pd.to_datetime(df.entry_time,utc=True,errors='coerce')
    a=df[df.grade.astype(str)=='A+'].sort_values('entry_time').copy()
    if a.empty: raise SystemExit('No A+ trades in portfolio_trades.csv')
    a['R']=pd.to_numeric(a.pnl_usdt,errors='coerce')/pd.to_numeric(a.actual_risk_usdt,errors='coerce').replace(0,np.nan)

    base=metrics(a,'A+ baseline')
    mean,lo,hi=bootstrap_mean(a.R)
    base.update({'bootstrap_mean_R':mean,'bootstrap_R_lo':lo,'bootstrap_R_hi':hi})

    # Score slices. Use explicit score bins and threshold views to avoid cherry-picking one cut.
    score=pd.to_numeric(a.signal_score,errors='coerce')
    a['score_bin']=pd.cut(score,[-np.inf,79.999,84.999,89.999,94.999,np.inf],labels=['<80','80-84','85-89','90-94','95+'])
    score_rows=[]
    for k,x in a.groupby('score_bin',observed=True): score_rows.append(metrics(x,str(k)))
    for th in [70,75,80,85,90,95]: score_rows.append(metrics(a[score>=th],f'score>={th}'))
    pd.DataFrame(score_rows).to_csv(OUT/'score_slices.csv',index=False)

    # Risk-tier slices.
    pr=pd.to_numeric(a.planned_risk_usdt,errors='coerce')
    risk_rows=[]
    for k,x in a.groupby(pr): risk_rows.append(metrics(x,f'planned_risk={k:g}U'))
    pd.DataFrame(risk_rows).to_csv(OUT/'risk_tiers.csv',index=False)

    # Chronological thirds and rolling halves.
    n=len(a); thirds=np.array_split(np.arange(n),3)
    time_rows=[]
    for i,idx in enumerate(thirds,1): time_rows.append(metrics(a.iloc[idx],f'third_{i}'))
    time_rows.append(metrics(a.iloc[:n//2],'first_half'))
    time_rows.append(metrics(a.iloc[n//2:],'second_half'))
    pd.DataFrame(time_rows).to_csv(OUT/'time_stability.csv',index=False)

    # Winner concentration / robustness.
    ordered=a.sort_values('pnl_usdt',ascending=False)
    robust=[]
    for k in [0,1,3,5,10]:
        x=ordered.iloc[k:] if k else ordered
        row=metrics(x,f'remove_top_{k}')
        m,l,h=bootstrap_mean(x.R);row.update({'bootstrap_mean_R':m,'bootstrap_R_lo':l,'bootstrap_R_hi':h})
        robust.append(row)
    pd.DataFrame(robust).to_csv(OUT/'robustness.csv',index=False)

    # BTC regime / acute-drop validation.
    z=add_btc_regime(a)
    btc_rows=[]
    for reg,x in z.groupby('btc_regime'): btc_rows.append(metrics(x,f'btc_{reg.lower()}'))
    btc_rows.append(metrics(z[~z.btc_acute_drop],'exclude_acute_drop'))
    btc_rows.append(metrics(z[z.btc_acute_drop],'acute_drop_only'))
    pd.DataFrame(btc_rows).to_csv(OUT/'btc_regime.csv',index=False)
    z.to_csv(OUT/'a_plus_annotated.csv',index=False)

    # Simple pass/fail flags: not a proof, only a guard against obvious concentration/instability.
    second=metrics(a.iloc[n//2:],'second_half')
    remove3=metrics(ordered.iloc[3:],'remove_top_3') if n>3 else {'mean_R':np.nan,'profit_factor':np.nan,'n':0}
    flags={
        'baseline_mean_R_positive': bool(base['mean_R']>0),
        'baseline_PF_R_above_1_5': bool(base['PF_R']>1.5),
        'bootstrap_lower_bound_positive': bool(lo>0),
        'second_half_mean_R_positive': bool(second.get('mean_R',-999)>0),
        'remove_top3_mean_R_positive': bool(remove3.get('mean_R',-999)>0),
        'sample_at_least_30': bool(n>=30),
    }
    (OUT/'validation_flags.json').write_text(json.dumps(flags,indent=2),encoding='utf-8')

    lines=['# A+ EDGE VALIDATION','',
           f"Baseline: n={base['n']}, P/L={base['pnl_usdt']:+.2f}U, win={base['win_rate']:.1%}, PF={base['profit_factor']:.2f}, maxDD={base['max_drawdown']:.2%}, meanR={base['mean_R']:+.3f}, bootstrap95=[{lo:+.3f},{hi:+.3f}]",'',
           '## Score slices']
    for r in score_rows: lines.append(f"- {r['label']}: n={r['n']}, P/L={r.get('pnl_usdt',0):+.2f}U, meanR={r.get('mean_R',0):+.3f}, PF(R)={r.get('PF_R',0):.2f}, win={r.get('win_rate',0):.1%}")
    lines+=['','## Risk tiers']
    for r in risk_rows: lines.append(f"- {r['label']}: n={r['n']}, P/L={r.get('pnl_usdt',0):+.2f}U, meanR={r.get('mean_R',0):+.3f}, PF(R)={r.get('PF_R',0):.2f}")
    lines+=['','## Time stability']
    for r in time_rows: lines.append(f"- {r['label']}: n={r['n']}, P/L={r.get('pnl_usdt',0):+.2f}U, meanR={r.get('mean_R',0):+.3f}, PF(R)={r.get('PF_R',0):.2f}")
    lines+=['','## BTC regime']
    for r in btc_rows: lines.append(f"- {r['label']}: n={r['n']}, P/L={r.get('pnl_usdt',0):+.2f}U, meanR={r.get('mean_R',0):+.3f}, PF(R)={r.get('PF_R',0):.2f}, win={r.get('win_rate',0):.1%}")
    lines+=['','## Robustness']
    for r in robust: lines.append(f"- {r['label']}: n={r['n']}, P/L={r.get('pnl_usdt',0):+.2f}U, meanR={r.get('mean_R',0):+.3f}, bootstrap95=[{r.get('bootstrap_R_lo',np.nan):+.3f},{r.get('bootstrap_R_hi',np.nan):+.3f}]")
    lines+=['','## Validation flags','']+[f"- {k}: {v}" for k,v in flags.items()]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    zipbase=str(BASE/'a_plus_validation');zpath=shutil.make_archive(zipbase,'zip',OUT)
    print((OUT/'SUMMARY.md').read_text())
    print('A_PLUS_VALIDATION_ZIP:',zpath)

if __name__=='__main__': main()
