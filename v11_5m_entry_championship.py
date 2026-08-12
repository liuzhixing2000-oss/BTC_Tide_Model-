#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 5m Entry Championship

Strict no-look-ahead benchmark:
- 15m Tide setup remains the setup engine.
- 5m is used only AFTER the 15m setup is fully available.
- Compares current 15m entry with several 5m execution triggers.
- Measures hard-stop risk, expectancy, PF, MDD, MFE/MAE, and rescue of
  trades where baseline hard-stop risk >2%.

Quick:
  python v11_5m_entry_championship.py --max-symbols 10

Full:
  python v11_5m_entry_championship.py
"""

from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--engine-file",type=Path,default=Path("crypto_tide_engine_v10_6.py"))
    p.add_argument("--bundle-dir",type=Path,default=Path("v10_bundle"))
    p.add_argument("--output-dir",type=Path,default=Path("v11_5m_entry_research"))
    p.add_argument("--days-5m",type=int,default=90)
    p.add_argument("--days-15m",type=int,default=120)
    p.add_argument("--days-1h",type=int,default=180)
    p.add_argument("--max-symbols",type=int,default=0)
    p.add_argument("--symbols",default="")
    p.add_argument("--entry-window-minutes",type=int,default=30)
    p.add_argument("--fixed-hold-hours",type=float,default=4.0)
    p.add_argument("--structure-buffer-atr",type=float,default=0.50)
    p.add_argument("--validation-days",type=int,default=30)
    return p.parse_args()

def sf(v,d=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d

def nt(s):
    return pd.to_datetime(s,utc=True,errors="coerce").astype("datetime64[ns, UTC]")

def load_engine(path,data_dir):
    data_dir.mkdir(parents=True,exist_ok=True)
    os.environ["TIDE_DATA_DIR"]=str(data_dir.resolve())
    spec=importlib.util.spec_from_file_location("v11engine",path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def prepare5(df):
    x=df.copy(); x["open_time"]=nt(x["open_time"]); x=x.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float); v=x.volume.astype(float)
    x["ema20_5m"]=c.ewm(span=20,adjust=False).mean()
    x["ema20_slope3_5m"]=x["ema20_5m"].pct_change(3)
    x["volume_mult_5m"]=v/v.rolling(20).median()
    x["prior_high_3_5m"]=h.shift(1).rolling(3).max()
    rng=(h-l).replace(0,np.nan); x["close_position_5m"]=(c-l)/rng
    return x

def signal_score(engine,row,hist):
    c=engine.signal_components(row,hist)
    raw=100*(engine.ACTIVE_SIGNAL_WEIGHTS["historical"]*c["historical"]+
             engine.ACTIVE_SIGNAL_WEIGHTS["wick"]*c["wick"]+
             engine.ACTIVE_SIGNAL_WEIGHTS["volume"]*c["volume"]+
             engine.ACTIVE_SIGNAL_WEIGHTS["close_position"]*c["close_position"])
    conf=20*(c["confirmation"]-.5)
    comb=sf(row.get("combined_setup_score"))
    adj=0 if not np.isfinite(comb) else .10*(comb-62)
    return round(float(np.clip(raw+conf+adj,0,100)),1)

def completed_setup(row):
    return bool(row.get("confirmed",row.get("signal",False))) and np.isfinite(sf(row.get("combined_setup_score"))) and np.isfinite(sf(row.get("confirmation_quality_score")))

def trig(row,mode):
    close=sf(row.close); op=sf(row.open); ph=sf(row.prior_high_3_5m); ema=sf(row.ema20_5m); slope=sf(row.ema20_slope3_5m)
    vm=sf(row.volume_mult_5m); cp=sf(row.close_position_5m)
    bull=close>op
    micro=bull and np.isfinite(ph) and close>ph and cp>=.55
    vol=micro and np.isfinite(vm) and vm>=1.5
    emaok=bull and np.isfinite(ema) and close>ema and slope>0 and cp>=.5
    if mode=="micro": return micro
    if mode=="volume": return vol
    if mode=="ema": return emaok
    if mode=="composite": return sum(map(bool,[micro,vol,emaok]))>=2
    if mode=="aggressive": return bull and cp>=.5
    return False

def find5(df5,available,mode,window):
    closes=df5.open_time+pd.Timedelta(minutes=5)
    ids=np.flatnonzero(((closes>available)&(closes<=available+pd.Timedelta(minutes=window))).to_numpy())
    for i in ids:
        if trig(df5.iloc[int(i)],mode): return int(i)
    return None

def stop_for(setup,entry,buf):
    atr=sf(setup.get("confirmation_signal_atr",setup.get("atr14")))
    low=sf(setup.get("confirmation_signal_low",setup.get("low")))
    if not np.isfinite(atr) or atr<=0 or not np.isfinite(low): return np.nan,np.nan
    stop=low-buf*atr
    risk=(entry-stop)/entry if entry>0 else np.nan
    return stop,risk

def sim(engine,df5,idx,entry,stop,hours):
    if not np.isfinite(stop) or stop>=entry:return None
    end=min(idx+int(round(hours*12)),len(df5)-1)
    ex=end; px=float(df5.iloc[end].close); reason=f"fixed_{hours:g}h"
    for j in range(idx+1,end+1):
        if float(df5.iloc[j].low)<=stop:
            ex=j; px=stop; reason="initial_stop"; break
    path=df5.iloc[idx:ex+1]
    return dict(
        exit_time=pd.Timestamp(df5.iloc[ex].open_time)+pd.Timedelta(minutes=5),
        exit_price=px,
        net_return=px/entry-1-float(engine.FEE_SLIPPAGE),
        mfe_pct=float(path.high.max())/entry-1,
        mae_pct=float(path.low.min())/entry-1,
        exit_reason=reason,
    )

def pf(r):
    r=pd.Series(r).dropna().astype(float); gp=r[r>0].sum(); gl=-r[r<0].sum()
    return np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)

def mdd(r):
    r=pd.Series(r).dropna().astype(float)
    if r.empty:return np.nan
    eq=(1+r.clip(lower=-.999)).cumprod(); return float((eq/eq.cummax()-1).min())

def stats(g):
    if g.empty:return dict(trades=0,expectancy=np.nan,win_rate=np.nan,profit_factor=np.nan,max_drawdown=np.nan,avg_stop_risk=np.nan,pct_stop_le2=np.nan,avg_mfe=np.nan,avg_mae=np.nan,rescued=0)
    r=g.net_return.astype(float); risk=g.hard_stop_risk_pct.astype(float)
    return dict(trades=len(g),expectancy=r.mean(),win_rate=(r>0).mean(),profit_factor=pf(r),max_drawdown=mdd(r),
                avg_stop_risk=risk.mean(),pct_stop_le2=(risk<=.02).mean(),avg_mfe=g.mfe_pct.mean(),avg_mae=g.mae_pct.mean(),
                rescued=int(g.rescued_from_gt2pct.sum()))

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    engine=load_engine(a.engine_file,a.output_dir/"engine_research_data")
    u=pd.read_csv(a.bundle_dir/"stage2_full_results.csv")
    if "eligible" in u.columns:u=u[u.eligible.astype(str).str.lower().eq("true")]
    if "score" in u.columns:u=u.sort_values("score",ascending=False)
    if a.symbols.strip():
        wanted={z.strip().upper() for z in a.symbols.split(",") if z.strip()}; u=u[u.symbol.isin(wanted)]
    if a.max_symbols>0:u=u.head(a.max_symbols)

    modes=[("BASE_15M","base"),("EARLY_5M_MICRO_BREAK","micro"),("EARLY_5M_VOLUME_BREAK","volume"),
           ("EARLY_5M_EMA_RECLAIM","ema"),("EARLY_5M_COMPOSITE","composite"),("EARLY_5M_AGGRESSIVE","aggressive")]
    rows=[]
    print("="*100); print("V11 5m ENTRY CHAMPIONSHIP"); print("Symbols:",len(u))

    for n,rowu in enumerate(u.itertuples(index=False),1):
        symbol=str(rowu.symbol); hist=sf(getattr(rowu,"score",np.nan)); hist=hist if np.isfinite(hist) else None
        print(f"[{n}/{len(u)}] {symbol}")
        try:
            df5=prepare5(engine.fetch_klines(symbol,"5",a.days_5m))
            df15=engine.model_frame(engine.fetch_klines(symbol,"15",a.days_15m),engine.fetch_klines(symbol,"60",a.days_1h))
            df15["open_time"]=nt(df15["open_time"]); df15=df15.sort_values("open_time").reset_index(drop=True)
        except Exception as e:
            print("    FAILED:",type(e).__name__,e); continue

        setups=0
        for i in range(len(df15)):
            setup=df15.iloc[i]
            if not completed_setup(setup):continue
            setups+=1
            avail=pd.Timestamp(setup.open_time)+pd.Timedelta(minutes=15)
            base_entry=float(setup.close)
            base_stop,base_risk=stop_for(setup,base_entry,a.structure_buffer_atr)
            close5=df5.open_time+pd.Timedelta(minutes=5)
            ids=np.flatnonzero((close5>=avail).to_numpy())
            if not len(ids):continue
            base_idx=int(ids[0])
            br=sim(engine,df5,base_idx,base_entry,base_stop,a.fixed_hold_hours)
            if br is None:continue

            common=dict(symbol=symbol,setup_time=setup.open_time,setup_available_time=avail,
                        raw_quality=sf(setup.get("raw_quality_score")),
                        next_quality=sf(setup.get("confirmation_quality_score")),
                        combined_quality=sf(setup.get("combined_setup_score")),
                        confirmation=int(setup.get("secondary_confirmation_tests",0)),
                        volume_multiple_15m=sf(setup.get("volume_multiple")),
                        signal_score=signal_score(engine,setup,hist),
                        baseline_entry_price=base_entry,baseline_hard_stop_risk_pct=base_risk)

            rows.append({**common,"method":"BASE_15M","entry_time":avail,"entry_price":base_entry,
                         "hard_stop":base_stop,"hard_stop_risk_pct":base_risk,"entry_delay_minutes_vs_15m":0.0,
                         "rescued_from_gt2pct":False,**br})

            for name,mode in modes[1:]:
                idx=find5(df5,avail,mode,a.entry_window_minutes)
                if idx is None:continue
                et=pd.Timestamp(df5.iloc[idx].open_time)+pd.Timedelta(minutes=5)
                ep=float(df5.iloc[idx].close); st,risk=stop_for(setup,ep,a.structure_buffer_atr)
                rr=sim(engine,df5,idx,ep,st,a.fixed_hold_hours)
                if rr is None:continue
                rescued=np.isfinite(base_risk) and base_risk>.02 and np.isfinite(risk) and risk<=.02
                rows.append({**common,"method":name,"entry_time":et,"entry_price":ep,"hard_stop":st,
                             "hard_stop_risk_pct":risk,"entry_delay_minutes_vs_15m":(et-avail).total_seconds()/60,
                             "rescued_from_gt2pct":rescued,
                             "trigger_volume_mult_5m":sf(df5.iloc[idx].get("volume_mult_5m")),**rr})
        print("    setups:",setups)

    if not rows:raise RuntimeError("No trades generated")
    t=pd.DataFrame(rows); t["setup_time"]=nt(t.setup_time); t["entry_time"]=nt(t.entry_time); t["exit_time"]=nt(t.exit_time)
    t.to_csv(a.output_dir/"v11_5m_entry_trades.csv",index=False)

    split=t.setup_time.max()-pd.Timedelta(days=a.validation_days)
    rr=[]
    for method,g in t.groupby("method"):
        tr=g[g.setup_time<split]; va=g[g.setup_time>=split]
        A,T,V=stats(g),stats(tr),stats(va)
        rr.append({"method":method,**A,"train_trades":T["trades"],"train_expectancy":T["expectancy"],
                   "validation_trades":V["trades"],"validation_expectancy":V["expectancy"],
                   "validation_win_rate":V["win_rate"],"validation_profit_factor":V["profit_factor"],
                   "validation_max_drawdown":V["max_drawdown"],"validation_avg_stop_risk":V["avg_stop_risk"],
                   "validation_pct_stop_le2":V["pct_stop_le2"],"validation_rescued":V["rescued"]})
    rank=pd.DataFrame(rr)
    rank["research_score"]=(.35*rank.validation_expectancy.rank(pct=True)+
                            .20*rank.validation_profit_factor.replace(np.inf,20).rank(pct=True)+
                            .20*rank.validation_pct_stop_le2.rank(pct=True)+
                            .15*(-rank.validation_avg_stop_risk).rank(pct=True)+
                            .10*rank.train_expectancy.rank(pct=True))
    rank=rank.sort_values(["research_score","validation_expectancy"],ascending=False)
    rank.to_csv(a.output_dir/"v11_5m_entry_rankings.csv",index=False)

    alt=t[~t.method.eq("BASE_15M") & (t.next_quality>=95) & (t.baseline_hard_stop_risk_pct>.02)]
    rescue=[]
    for method,g in alt.groupby("method"):
        rescue.append({"method":method,"trades":len(g),"rescued_to_le2pct":int(g.rescued_from_gt2pct.sum()),
                       "expectancy":g.net_return.mean(),"win_rate":(g.net_return>0).mean(),
                       "avg_new_risk":g.hard_stop_risk_pct.mean()})
    rescue=pd.DataFrame(rescue)
    rescue.to_csv(a.output_dir/"high_next_high_risk_rescue_results.csv",index=False)

    report="V11 5m ENTRY CHAMPIONSHIP\n"+"="*100+"\n"
    report+=f"Validation starts: {split}\n\n"
    report+="IMPORTANT: 5m entries occur only AFTER the scored 15m setup is available. No look-ahead.\n\n"
    report+="RANKINGS\n"+rank.to_string(index=False)+"\n\n"
    report+="HIGH NEXT / HIGH BASELINE-RISK RESCUE\n"+(rescue.to_string(index=False) if not rescue.empty else "None")
    (a.output_dir/"v11_5m_entry_report.txt").write_text(report,encoding="utf-8")
    print("\nTOP METHODS\n",rank[["method","trades","validation_trades","validation_expectancy","validation_profit_factor","validation_avg_stop_risk","validation_pct_stop_le2","validation_rescued","research_score"]].to_string(index=False))
    print("\nOutput:",a.output_dir)

if __name__=="__main__":
    main()
