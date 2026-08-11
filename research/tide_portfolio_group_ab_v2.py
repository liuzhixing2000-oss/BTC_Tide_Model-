#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = {
"BTC":"MAJOR","ETH":"MAJOR","SOL":"MAJOR","XRP":"MAJOR","BNB":"MAJOR","ADA":"MAJOR",
"DOGE":"MEME","PEPE":"MEME","1000PEPE":"MEME","BONK":"MEME","1000BONK":"MEME",
"WIF":"MEME","FLOKI":"MEME","1000FLOKI":"MEME","BOME":"MEME","BRETT":"MEME",
"MEME":"MEME","1000SHIB":"MEME","DOGS":"MEME","POPCAT":"MEME","MOG":"MEME",
"TURBO":"MEME","NEIRO":"MEME","1000RATS":"MEME","BANANAS31":"MEME",
"FET":"AI","RENDER":"AI","RNDR":"AI","TAO":"AI","WLD":"AI","ARKM":"AI","AI16Z":"AI",
"VIRTUAL":"AI","GRASS":"AI","NEAR":"AI","IO":"AI","ATH":"AI","AIXBT":"AI",
"SUI":"L1","APT":"L1","AVAX":"L1","SEI":"L1","TIA":"L1","INJ":"L1","ATOM":"L1",
"DOT":"L1","KAS":"L1","TON":"L1","ALGO":"L1","HBAR":"L1","ICP":"L1","ETC":"L1","ENA":"L1",
"ARB":"L2","OP":"L2","STRK":"L2","MANTA":"L2","ZK":"L2","METIS":"L2","BLAST":"L2",
"IMX":"L2","MATIC":"L2","POL":"L2",
"UNI":"DEFI","AAVE":"DEFI","CRV":"DEFI","LDO":"DEFI","MKR":"DEFI","SNX":"DEFI",
"COMP":"DEFI","SUSHI":"DEFI","PENDLE":"DEFI","DYDX":"DEFI","JUP":"DEFI","GMX":"DEFI",
"ONDO":"RWA","MPL":"RWA","CFG":"RWA","POLYX":"RWA",
"LINK":"INFRA","PYTH":"INFRA","BAND":"INFRA","GRT":"INFRA","FIL":"INFRA","AR":"INFRA","STORJ":"INFRA",
"XLM":"PAYMENT","LTC":"PAYMENT","BCH":"PAYMENT","XMR":"PAYMENT","ZEC":"PAYMENT","DASH":"PAYMENT",
"AXS":"GAMING","SAND":"GAMING","MANA":"GAMING","GALA":"GAMING","PIXEL":"GAMING",
"PORTAL":"GAMING","ILV":"GAMING","RON":"GAMING","MAGIC":"GAMING","YGG":"GAMING"
}

def group_of(symbol):
    b=str(symbol).upper()
    if b.endswith("USDT"): b=b[:-4]
    if b in GROUPS: return GROUPS[b]
    if any(x in b for x in ("PEPE","DOGE","BONK","FLOKI","SHIB","WIF")): return "MEME"
    return "OTHER"

def pr(s):
    x=pd.to_numeric(s,errors="coerce")
    return x.rank(pct=True).fillna(.5) if x.notna().sum()>1 else pd.Series(.5,index=s.index)

def load(trades,symbols,config):
    t=pd.read_csv(trades); s=pd.read_csv(symbols)
    if "config" in t: t=t[t.config==config].copy()
    if "config" in s: s=s[s.config==config].copy()
    t["entry_time"]=pd.to_datetime(t.entry_time,utc=True)
    t["exit_time"]=pd.to_datetime(t.exit_time,utc=True)
    t["net_return"]=pd.to_numeric(t.net_return,errors="coerce")
    t=t.dropna(subset=["entry_time","exit_time","net_return"])
    comps=[]
    for col,w in [("score",.30),("validation_expectancy",.25),("validation_profit_factor",.20),
                  ("validation_return",.15),("max_drawdown",.10)]:
        if col in s: comps.append((w,pr(s[col].replace(np.inf,10) if col=="validation_profit_factor" else s[col])))
    if comps:
        z=sum(w for w,_ in comps); s["priority_score"]=sum(w*v for w,v in comps)/z
    else: s["priority_score"]=.5
    t=t.merge(s[["symbol","priority_score"]].drop_duplicates("symbol"),on="symbol",how="left")
    t["priority_score"]=t.priority_score.fillna(.5); t["group"]=t.symbol.map(group_of)
    t["trade_id"]=np.arange(len(t))
    return t.sort_values(["entry_time","priority_score"],ascending=[True,False]).reset_index(drop=True)

def simulate(t,a,group_limit,name):
    cash=a.initial_capital; openp=[]; ledger=[]; eq=[]
    def close(now):
        nonlocal cash,openp
        due=[p for p in openp if p["exit_time"]<=now]; openp=[p for p in openp if p["exit_time"]>now]
        for p in sorted(due,key=lambda x:x["exit_time"]):
            pnl=p["margin"]*a.leverage*p["net_return"]; cash+=p["margin"]+pnl
            ledger.append({**p,"pnl":pnl,"status":"executed","reason":"","version":name})
            eq.append({"time":p["exit_time"],"equity":cash+sum(x["margin"] for x in openp),
                       "open_positions":len(openp),"version":name})
    for tm,b in t.groupby("entry_time",sort=True):
        close(tm)
        for r in b.sort_values(["priority_score","symbol"],ascending=[False,True]).itertuples():
            equity=cash+sum(x["margin"] for x in openp); locked=sum(x["margin"] for x in openp)
            reason=None
            if len(openp)>=a.max_positions: reason="max_positions"
            elif sum(x["symbol"]==r.symbol for x in openp)>=1: reason="same_symbol"
            elif group_limit is not None and sum(x["group"]==r.group for x in openp)>=group_limit: reason=f"group_{r.group}_full"
            elif locked+a.margin_per_trade>equity*a.max_margin_utilisation+1e-9: reason="margin_limit"
            elif cash<a.margin_per_trade: reason="cash"
            if reason:
                ledger.append({"trade_id":r.trade_id,"symbol":r.symbol,"group":r.group,
                               "entry_time":r.entry_time,"exit_time":r.exit_time,"margin":a.margin_per_trade,
                               "leverage":a.leverage,"net_return":r.net_return,"priority_score":r.priority_score,
                               "pnl":np.nan,"status":"rejected","reason":reason,"version":name})
                continue
            cash-=a.margin_per_trade
            openp.append({"trade_id":r.trade_id,"symbol":r.symbol,"group":r.group,
                          "entry_time":r.entry_time,"exit_time":r.exit_time,"margin":a.margin_per_trade,
                          "leverage":a.leverage,"net_return":r.net_return,"priority_score":r.priority_score})
            eq.append({"time":tm,"equity":cash+sum(x["margin"] for x in openp),
                       "open_positions":len(openp),"version":name})
    if openp: close(max(x["exit_time"] for x in openp))
    L=pd.DataFrame(ledger); E=pd.DataFrame(eq).sort_values("time")
    E["peak"]=E.equity.cummax(); E["drawdown"]=E.equity/E.peak-1
    X=L[L.status=="executed"]; win=X.loc[X.pnl>0,"pnl"].sum(); loss=-X.loc[X.pnl<0,"pnl"].sum()
    pf=math.inf if loss==0 and win>0 else (win/loss if loss>0 else math.nan)
    ret=cash/a.initial_capital-1; dd=E.drawdown.min(); cal=ret/abs(dd) if dd<0 else math.inf
    S={"version":name,"group_limit":"none" if group_limit is None else group_limit,
       "final_equity":cash,"total_return":ret,"max_drawdown":dd,"calmar_ratio":cal,
       "profit_factor":pf,"win_rate":(X.pnl>0).mean(),"executed_trades":len(X),
       "rejected_trades":(L.status=="rejected").sum(),"execution_rate":len(X)/len(t),
       "average_concurrent_positions":E.open_positions.mean(),
       "maximum_concurrent_positions":E.open_positions.max()}
    return L,E,S

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trades",type=Path,default=Path("/app/ab_results_all/universe_ab_trades.csv"))
    p.add_argument("--symbols",type=Path,default=Path("/app/ab_results_all/universe_ab_symbol_details.csv"))
    p.add_argument("--config",default="all_eligible")
    p.add_argument("--initial-capital",type=float,default=3000)
    p.add_argument("--margin-per-trade",type=float,default=100)
    p.add_argument("--leverage",type=float,default=10)
    p.add_argument("--max-positions",type=int,default=12)
    p.add_argument("--max-margin-utilisation",type=float,default=.8)
    p.add_argument("--output-dir",type=Path,default=Path("/app/portfolio_group_results"))
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    t=load(a.trades,a.symbols,a.config)
    versions=[("baseline",None),("group_3",3),("group_2",2),("group_1",1)]
    LS=[]; ES=[]; SS=[]
    for n,g in versions:
        L,E,S=simulate(t,a,g,n); LS.append(L); ES.append(E); SS.append(S)
    L=pd.concat(LS,ignore_index=True); E=pd.concat(ES,ignore_index=True); S=pd.DataFrame(SS)
    B=S[S.version=="baseline"].iloc[0]
    A=S[S.version!="baseline"].copy()
    A=A[A.total_return>=B.total_return*.8]
    if len(A):
        best=A.sort_values(["calmar_ratio","profit_factor","total_return"],ascending=False).iloc[0]
        decision="UPGRADE_CANDIDATE" if best.calmar_ratio>B.calmar_ratio else "KEEP_BASELINE"
        rec=best.version if decision=="UPGRADE_CANDIDATE" else "baseline"
    else:
        decision="KEEP_BASELINE"; rec="baseline"
    S.to_csv(a.output_dir/"portfolio_group_ab_summary.csv",index=False)
    L.to_csv(a.output_dir/"portfolio_group_ab_ledgers.csv",index=False)
    E.to_csv(a.output_dir/"portfolio_group_ab_equity_curves.csv",index=False)
    (a.output_dir/"portfolio_group_ab_recommendation.txt").write_text(
        f"Decision: {decision}\nRecommended version: {rec}\n",encoding="utf-8")
    print("\nTIDE PORTFOLIO GROUP-RISK A/B RESULT\n")
    print(S.to_string(index=False))
    print(f"\nDecision: {decision}\nRecommended version: {rec}")
if __name__=="__main__": main()
