#!/usr/bin/env python3
from __future__ import annotations
import math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from backtest_v10_9_six_month import fetch_range

BASE=Path(os.getenv('BACKTEST_RESULT_DIR','/data/v10_9_six_month'))
INFILE=BASE/'portfolio_trades.csv'
OUT=BASE/'a_plus_regime'
OUT.mkdir(parents=True,exist_ok=True)
START_EQUITY=1000.0


def pf(x):
    x=pd.Series(x,dtype=float);gp=x[x>0].sum();gl=-x[x<0].sum()
    return float(gp/gl) if gl>0 else math.inf


def metrics(df,label):
    if df.empty:return {'label':label,'n':0}
    p=pd.to_numeric(df.pnl_usdt,errors='coerce').fillna(0.0)
    risk=pd.to_numeric(df.actual_risk_usdt,errors='coerce').replace(0,np.nan)
    r=p/risk
    return {'label':label,'n':len(df),'pnl_usdt':float(p.sum()),'win_rate':float((p>0).mean()),'PF':pf(p),'mean_R':float(r.mean()),'median_R':float(r.median()),'avg_mfe':float(pd.to_numeric(df.mfe,errors='coerce').mean()),'avg_mae':float(pd.to_numeric(df.mae,errors='coerce').mean())}


def ensure_replay():
    if INFILE.exists():return
    print(f'Missing {INFILE}. Auto-generating V10.9 100-day replay...',flush=True)
    cmd=[sys.executable,'backtest_v10_9_six_month.py','--days','100','--max-symbols','0','--out',str(BASE)]
    rc=subprocess.call(cmd)
    if rc!=0:raise SystemExit(rc)
    if not INFILE.exists():raise SystemExit(f'Replay finished but {INFILE} missing')


def btc_context(a):
    start=a.entry_time.min()-pd.Timedelta(days=12);end=a.entry_time.max()+pd.Timedelta(hours=6)
    d15=fetch_range('BTCUSDT','15',start.to_pydatetime(),end.to_pydatetime())
    d1=fetch_range('BTCUSDT','60',start.to_pydatetime(),end.to_pydatetime())
    d4=fetch_range('BTCUSDT','240',start.to_pydatetime(),end.to_pydatetime())
    for d,mins in [(d15,15),(d1,60),(d4,240)]:
        d['open_time']=pd.to_datetime(d.open_time,utc=True);d['close_time']=d.open_time+pd.Timedelta(minutes=mins)
    d15['ema20']=d15.close.ewm(span=20,adjust=False).mean();d15['ret15']=d15.close.pct_change();d15['ret1h_15']=d15.close.pct_change(4)
    d15['atr20_pct']=((pd.concat([(d15.high-d15.low).abs(),(d15.high-d15.close.shift()).abs(),(d15.low-d15.close.shift()).abs()],axis=1).max(axis=1)).rolling(20).mean()/d15.close)
    d15['rv24h']=d15.close.pct_change().rolling(96).std()*np.sqrt(96)
    d1['ema20']=d1.close.ewm(span=20,adjust=False).mean();d1['ema50']=d1.close.ewm(span=50,adjust=False).mean();d1['ret1h']=d1.close.pct_change();d1['ret4h']=d1.close.pct_change(4);d1['ret24h']=d1.close.pct_change(24);d1['slope20']=d1.ema20.pct_change(3)
    d4['ema20']=d4.close.ewm(span=20,adjust=False).mean();d4['ema50']=d4.close.ewm(span=50,adjust=False).mean();d4['ret4h_native']=d4.close.pct_change();d4['ret24h_4h']=d4.close.pct_change(6)
    x=a.sort_values('entry_time').copy()
    x=pd.merge_asof(x,d15[['close_time','close','ema20','ret15','ret1h_15','atr20_pct','rv24h']].sort_values('close_time'),left_on='entry_time',right_on='close_time',direction='backward')
    x=x.rename(columns={'close':'btc15_close','ema20':'btc15_ema20'})
    x=pd.merge_asof(x.sort_values('entry_time'),d1[['close_time','close','ema20','ema50','ret1h','ret4h','ret24h','slope20']].sort_values('close_time'),left_on='entry_time',right_on='close_time',direction='backward',suffixes=('','_1h'))
    x=x.rename(columns={'close':'btc1h_close','ema20':'btc1h_ema20','ema50':'btc1h_ema50'})
    x=pd.merge_asof(x.sort_values('entry_time'),d4[['close_time','close','ema20','ema50','ret4h_native','ret24h_4h']].sort_values('close_time'),left_on='entry_time',right_on='close_time',direction='backward',suffixes=('','_4h'))
    x=x.rename(columns={'close':'btc4h_close','ema20':'btc4h_ema20','ema50':'btc4h_ema50'})
    bull=(x.btc15_close>=x.btc15_ema20)&(x.btc1h_close>=x.btc1h_ema20)&(x.btc1h_ema20>=x.btc1h_ema50)&(x.slope20>=0)
    bear=(x.btc15_close<x.btc15_ema20)&(x.btc1h_close<x.btc1h_ema20)&(x.btc1h_ema20<x.btc1h_ema50)&(x.slope20<0)
    x['btc_regime']=np.select([bull,bear],['BULL','BEAR'],default='NEUTRAL')
    x['btc_acute_drop']=(x.ret15<=-0.004)|(x.ret1h<=-0.008)
    x['btc_above_4h_ema20']=x.btc4h_close>=x.btc4h_ema20
    return x


def effect_table(prior,recent):
    cols=['signal_score','raw_quality','next_quality','combined','confirmation_tests','volume_multiple','h1_strength','stop_pct','mfe','mae','bars_held','btc15_close','ret15','ret1h','ret4h','ret24h','atr20_pct','rv24h','ret4h_native','ret24h_4h']
    rows=[]
    for c in cols:
        if c not in prior.columns or c not in recent.columns:continue
        p=pd.to_numeric(prior[c],errors='coerce').dropna();r=pd.to_numeric(recent[c],errors='coerce').dropna()
        if len(p)<3 or len(r)<2:continue
        pm,rm=float(p.mean()),float(r.mean());ps=float(p.std(ddof=1));pooled=ps if ps>1e-12 else np.nan
        z=(rm-pm)/pooled if np.isfinite(pooled) else np.nan
        rows.append({'feature':c,'prior_n':len(p),'prior_mean':pm,'recent_n':len(r),'recent_mean':rm,'difference':rm-pm,'difference_in_prior_sd':z})
    return pd.DataFrame(rows).sort_values('difference_in_prior_sd',key=lambda s:s.abs(),ascending=False)


def categorical_rows(z,cut):
    rows=[]
    for c in ['btc_regime','btc_acute_drop','btc_above_4h_ema20','h1_trend','h4_trend','route']:
        if c not in z.columns:continue
        for v,x in z.groupby(c,dropna=False):
            old=x[x.entry_time<cut];new=x[x.entry_time>=cut]
            allm=metrics(x,f'{c}={v}');om=metrics(old,'prior');nm=metrics(new,'recent')
            rows.append({'feature':c,'value':str(v),'all_n':allm['n'],'all_meanR':allm.get('mean_R',np.nan),'all_PF':allm.get('PF',np.nan),'prior_n':om['n'],'prior_meanR':om.get('mean_R',np.nan),'recent_n':nm['n'],'recent_meanR':nm.get('mean_R',np.nan)})
    return pd.DataFrame(rows)


def main():
    ensure_replay()
    df=pd.read_csv(INFILE);df['entry_time']=pd.to_datetime(df.entry_time,utc=True,errors='coerce');df=df.dropna(subset=['entry_time']).sort_values('entry_time')
    a=df[df.grade.astype(str)=='A+'].copy()
    if a.empty:raise SystemExit('No A+ trades')
    z=btc_context(a)
    z['R']=pd.to_numeric(z.pnl_usdt,errors='coerce')/pd.to_numeric(z.actual_risk_usdt,errors='coerce').replace(0,np.nan)
    end=df.entry_time.max();cut=end-pd.Timedelta(days=30)
    recent=z[z.entry_time>=cut].copy();prior=z[z.entry_time<cut].copy()
    z['period']=np.where(z.entry_time>=cut,'RECENT_30D','PRIOR')
    z.to_csv(OUT/'a_plus_regime_annotated.csv',index=False)
    effects=effect_table(prior,recent);effects.to_csv(OUT/'numeric_regime_shift.csv',index=False)
    cats=categorical_rows(z,cut);cats.to_csv(OUT/'categorical_regime.csv',index=False)

    # Compare winners/losers to expose what normally distinguishes successful A+ trades.
    win=z[z.pnl_usdt>0];loss=z[z.pnl_usdt<=0]
    wl=effect_table(loss,win);wl.to_csv(OUT/'winner_vs_loser_features.csv',index=False)

    # Conservative pre-specified filter diagnostics; descriptive only, not optimization.
    filters={
      'all_A+':pd.Series(True,index=z.index),
      'BTC_not_bull':z.btc_regime.ne('BULL'),
      'BTC_bear_only':z.btc_regime.eq('BEAR'),
      'BTC_neutral_or_bear':z.btc_regime.isin(['NEUTRAL','BEAR']),
      'BTC_4h_below_EMA20':~z.btc_above_4h_ema20,
      'no_acute_drop':~z.btc_acute_drop,
      'h1_DOWN':z.h1_trend.astype(str).eq('DOWN'),
      'h4_DOWN':z.h4_trend.astype(str).eq('DOWN'),
      'h1_h4_DOWN':z.h1_trend.astype(str).eq('DOWN')&z.h4_trend.astype(str).eq('DOWN'),
    }
    frows=[]
    for name,mask in filters.items():
        x=z[mask];m=metrics(x,name);mr=metrics(x[x.entry_time>=cut],name+' recent');mp=metrics(x[x.entry_time<cut],name+' prior')
        frows.append({'filter':name,'n':m['n'],'pnl':m.get('pnl_usdt',0),'meanR':m.get('mean_R',np.nan),'PF':m.get('PF',np.nan),'win':m.get('win_rate',np.nan),'prior_n':mp['n'],'prior_meanR':mp.get('mean_R',np.nan),'recent_n':mr['n'],'recent_meanR':mr.get('mean_R',np.nan),'recent_pnl':mr.get('pnl_usdt',0)})
    pd.DataFrame(frows).to_csv(OUT/'filter_diagnostics.csv',index=False)

    base=metrics(z,'all A+');pm=metrics(prior,'prior');rm=metrics(recent,'recent30d')
    lines=['# A+ REGIME DECAY INVESTIGATION','',f'Comparison window: recent = {cut.isoformat()} -> {end.isoformat()}','',
           f"All A+: n={base['n']}, P/L={base['pnl_usdt']:+.2f}U, meanR={base['mean_R']:+.3f}, PF={base['PF']:.2f}, win={base['win_rate']:.1%}",
           f"Prior: n={pm['n']}, P/L={pm['pnl_usdt']:+.2f}U, meanR={pm['mean_R']:+.3f}, PF={pm['PF']:.2f}, win={pm['win_rate']:.1%}",
           f"Recent30d: n={rm['n']}, P/L={rm['pnl_usdt']:+.2f}U, meanR={rm['mean_R']:+.3f}, PF={rm['PF']:.2f}, win={rm['win_rate']:.1%}",'',
           '## Largest numeric environment shifts (recent vs prior)']
    for _,r in effects.head(10).iterrows():
        lines.append(f"- {r.feature}: prior={r.prior_mean:.5g}, recent={r.recent_mean:.5g}, shift={r.difference_in_prior_sd:+.2f} prior-SD")
    lines+=['','## Categorical regimes']
    for _,r in cats.iterrows():
        if r.all_n>=2:
            lines.append(f"- {r.feature}={r.value}: all n={int(r.all_n)} meanR={r.all_meanR:+.3f}; prior n={int(r.prior_n)} meanR={r.prior_meanR:+.3f}; recent n={int(r.recent_n)} meanR={r.recent_meanR:+.3f}")
    lines+=['','## Pre-specified filter diagnostics']
    for r in frows:
        lines.append(f"- {r['filter']}: n={r['n']}, meanR={r['meanR']:+.3f}, PF={r['PF']:.2f}; recent n={r['recent_n']}, recent meanR={r['recent_meanR']:+.3f}, recent P/L={r['recent_pnl']:+.2f}U")
    lines+=['','## Recent A+ trades']
    for _,r in recent.iterrows():
        lines.append(f"- {r.entry_time.isoformat()} {r.symbol}: score={r.signal_score:.1f}, h1={r.h1_trend}, h4={r.h4_trend}, BTC={r.btc_regime}, BTC1h={r.ret1h:+.2%}, BTC24h={r.ret24h:+.2%}, vol24={r.rv24h:.2%}, MFE={r.mfe:+.2%}, MAE={r.mae:+.2%}, P/L={r.pnl_usdt:+.2f}U, R={r.R:+.3f}")
    lines+=['','## Important limitation','- This is descriptive regime decomposition on only 32 historical A+ trades and 4 recent trades. Any apparent filter can be unstable. Do not promote a filter to production unless it also survives forward/OOS data.']
    summary='\n'.join(lines);(OUT/'SUMMARY.md').write_text(summary,encoding='utf-8');print(summary,flush=True);print('A_PLUS_REGIME_DIR:',OUT,flush=True)

if __name__=='__main__':main()
