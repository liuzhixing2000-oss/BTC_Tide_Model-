#!/usr/bin/env python3
import argparse, importlib.util, itertools, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def sf(v,d=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except:return d

def nt(s): return pd.to_datetime(s,utc=True,errors="coerce").astype("datetime64[ns, UTC]")

def load_engine(path,out):
    c=[path,Path("crypto_tide_engine_v10_8_2_2.py"),Path("crypto_tide_engine_v10_8_2_1.py"),
       Path("archive/crypto_tide_engine_v10_8_2_2.py")]
    sel=next((x for x in c if x.exists()),None)
    if sel is None: raise FileNotFoundError("No compatible Tide engine found")
    print("Using Tide engine:",sel)
    d=out/"engine_data"; d.mkdir(parents=True,exist_ok=True); os.environ["TIDE_DATA_DIR"]=str(d.resolve())
    sp=importlib.util.spec_from_file_location("v114engine",sel); m=importlib.util.module_from_spec(sp)
    sys.modules[sp.name]=m; sp.loader.exec_module(m); return m

def prep5(df):
    x=df.copy(); x["open_time"]=nt(x.open_time); x=x.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    o=x.open.astype(float); h=x.high.astype(float); l=x.low.astype(float); c=x.close.astype(float); v=x.volume.astype(float)
    rng=(h-l).replace(0,np.nan); x["body"]=((c-o)/rng).clip(-1,1); x["close_pos"]=((c-l)/rng).clip(0,1)
    prev=c.shift(1); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x["atr14"]=tr.rolling(14,min_periods=8).mean(); x["vol_mult"]=v/v.rolling(20,min_periods=10).median()
    x["micro"]=(c>h.shift(1).rolling(3,min_periods=2).max()).astype(int); x["close_time"]=x.open_time+pd.Timedelta(minutes=5)
    return x

def next_score(n,s):
    o,h,l,c,v=[sf(n.get(k)) for k in ["open","high","low","close","volume"]]
    if not all(np.isfinite(z) for z in [o,h,l,c,v]) or h<=l:return np.nan
    rng=h-l; body=np.clip((c-o)/rng,0,1); cp=np.clip((c-l)/rng,0,1)
    atr=sf(s.get("atr14")); imp=np.clip((c-o)/atr,0,1.5)/1.5 if np.isfinite(atr) and atr>0 else 0
    vr=sf(s.get("volume")); vol=np.clip(v/vr,0,3)/3 if np.isfinite(vr) and vr>0 else 0
    return float(np.clip(100*(.35*imp+.25*body+.25*cp+.15*vol),0,100))

def feat5(r,s):
    atr=sf(s.get("atr14")); p=(sf(r.close)-sf(r.open))/atr if np.isfinite(atr) and atr>0 else np.nan
    return p,max(0,sf(r.body,0)),sf(r.close_pos,0),sf(r.vol_mult,0),int(sf(r.micro,0)>0)

def feat10(r1,r2,s):
    atr=sf(s.get("atr14")); p=(sf(r2.close)-sf(r1.open))/atr if np.isfinite(atr) and atr>0 else np.nan
    return p,np.mean([max(0,sf(r1.body,0)),max(0,sf(r2.body,0))]),np.mean([sf(r1.close_pos,0),sf(r2.close_pos,0)]),np.mean([sf(r1.vol_mult,0),sf(r2.vol_mult,0)]),int(max(sf(r1.micro,0),sf(r2.micro,0))>0)

def stop(s,entry,buf):
    atr=sf(s.get("confirmation_signal_atr",s.get("atr14"))); low=sf(s.get("confirmation_signal_low",s.get("low")))
    if not np.isfinite(atr) or atr<=0 or not np.isfinite(low):return np.nan,np.nan
    st=low-buf*atr; return st,(entry-st)/entry

def sim(m,x,i,e,st,h=4):
    if not np.isfinite(st) or st>=e:return None
    end=min(i+int(h*12),len(x)-1); ex=end; px=float(x.iloc[end].close)
    for j in range(i+1,end+1):
        if float(x.iloc[j].low)<=st: ex=j; px=st; break
    r=px/e-1-float(m.FEE_SLIPPAGE); return r

def pf(r):
    r=pd.Series(r).dropna(); gp=r[r>0].sum(); gl=-r[r<0].sum()
    return np.inf if gl==0 and gp>0 else gp/gl if gl>0 else np.nan

def main():
    p=argparse.ArgumentParser(); p.add_argument("--engine-file",type=Path,default=Path("crypto_tide_engine_v10_8_2_2.py"))
    p.add_argument("--bundle-dir",type=Path,default=Path("v10_bundle")); p.add_argument("--output-dir",type=Path,default=Path("v11_4_next90_early_rescue"))
    p.add_argument("--max-symbols",type=int,default=80); p.add_argument("--full",action="store_true")
    p.add_argument("--days-5m",type=int,default=60); p.add_argument("--days-15m",type=int,default=90); p.add_argument("--days-1h",type=int,default=120); p.add_argument("--validation-days",type=int,default=20)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); m=load_engine(a.engine_file,a.output_dir)
    u=pd.read_csv(a.bundle_dir/"stage2_full_results.csv")
    if "eligible" in u:u=u[u.eligible.astype(str).str.lower().eq("true")]
    if "score" in u:u=u.sort_values("score",ascending=False)
    if not a.full:u=u.head(a.max_symbols)
    ev=[]
    for n,sym in enumerate(u.symbol.astype(str),1):
        print(f"[{n}/{len(u)}] {sym}",flush=True)
        try:
            x5=prep5(m.fetch_klines(sym,"5",a.days_5m)); x15=m.model_frame(m.fetch_klines(sym,"15",a.days_15m),m.fetch_klines(sym,"60",a.days_1h))
            x15["open_time"]=nt(x15.open_time); x15=x15.sort_values("open_time").reset_index(drop=True)
        except Exception as e: print(" FAIL",e); continue
        for i in range(len(x15)-1):
            s=x15.iloc[i]
            if not bool(s.get("signal",False)):continue
            t0=pd.Timestamp(s.open_time)+pd.Timedelta(minutes=15); ids=np.flatnonzero((x5.close_time>t0).to_numpy())
            if len(ids)<3:continue
            i1,i2=map(int,ids[:2]); r1,r2=x5.iloc[i1],x5.iloc[i2]; final=next_score(x15.iloc[i+1],s)
            if not np.isfinite(final):continue
            f5=feat5(r1,s); f10=feat10(r1,r2,s)
            fc=pd.Timestamp(x15.iloc[i+1].open_time)+pd.Timedelta(minutes=15); bi=np.flatnonzero((x5.close_time>=fc).to_numpy())
            if not len(bi):continue
            ib=int(bi[0]); be=float(x5.iloc[ib].close); bst,br=stop(s,be,.5); bret=sim(m,x5,ib,be,bst)
            if bret is None:continue
            for stage,idx,f in [("5M",i1,f5),("10M",i2,f10)]:
                ep=float(x5.iloc[idx].close); st,risk=stop(s,ep,.5); rr=sim(m,x5,idx,ep,st)
                if rr is None:continue
                ev.append([sym,t0,stage,final,final>=90,*f,ep,risk,br,bret,rr])
    cols=["symbol","setup_time","stage","final_next","is_next90","progress_atr","body","close_pos","volume_mult","micro_break","entry","risk","baseline_risk","baseline_ret","ret"]
    ev=pd.DataFrame(ev,columns=cols); ev["setup_time"]=nt(ev.setup_time); ev.to_csv(a.output_dir/"events.csv",index=False)
    if ev.empty:raise RuntimeError("No events")
    split=ev.setup_time.max()-pd.Timedelta(days=a.validation_days)
    rows=[]
    for stage in ["5M","10M"]:
        s=ev[ev.stage.eq(stage)]
        for pg,bg,cg,vg,mg in itertools.product([.1,.2,.3,.4],[.4,.55,.7],[.6,.75,.85],[.8,1.2,1.6],[0,1]):
            g=s[(s.progress_atr>=pg)&(s.body>=bg)&(s.close_pos>=cg)&(s.volume_mult>=vg)]
            if mg:g=g[g.micro_break.eq(1)]
            val=g[g.setup_time>=split]
            if len(val)<4:continue
            rr=val.ret.astype(float); basegt=val.baseline_risk>.02; rescued=basegt&(val.risk<=.02)
            rows.append(dict(stage=stage,progress=pg,body=bg,close_pos=cg,volume=vg,micro=mg,trades=len(val),
                             precision_next90=float(val.is_next90.mean()),expectancy=float(rr.mean()),profit_factor=pf(rr),
                             avg_risk=float(val.risk.mean()),pct_risk_le2=float((val.risk<=.02).mean()),
                             baseline_gt2=int(basegt.sum()),rescued_to_le2=int(rescued.sum()),rescue_rate=float(rescued.sum()/max(1,basegt.sum()))))
    res=pd.DataFrame(rows); res["sample_factor"]=np.minimum(res.trades/20,1)
    res["score"]=res.sample_factor*(.35*res.expectancy.rank(pct=True)+.2*res.profit_factor.replace(np.inf,20).rank(pct=True)+.2*res.precision_next90.rank(pct=True)+.15*res.rescue_rate.rank(pct=True)+.1*(-res.avg_risk).rank(pct=True))
    res=res.sort_values(["score","expectancy"],ascending=False); res.to_csv(a.output_dir/"validation_ranking.csv",index=False)
    base=ev[(ev.stage=="10M")&ev.is_next90].drop_duplicates(["symbol","setup_time"])
    print("\nBASELINE NEXT>=90",dict(trades=len(base),expectancy=float(base.baseline_ret.mean()),pf=pf(base.baseline_ret),avg_risk=float(base.baseline_risk.mean()),pct_le2=float((base.baseline_risk<=.02).mean())))
    print("\nTOP EARLY RULES\n",res.head(20).to_string(index=False))
    (a.output_dir/"report.txt").write_text(res.head(50).to_string(index=False),encoding="utf-8")
    print("\nOutput:",a.output_dir)

if __name__=="__main__":main()
