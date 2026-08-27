#!/usr/bin/env python3
"""Build a monthly forward-test report from REAL Railway runtime logs."""
from __future__ import annotations
import argparse, os, re, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
RAILWAY_API="https://backboard.railway.com/graphql/v2"; BYBIT_KLINE="https://api.bybit.com/v5/market/kline"
def gql(token,query,variables=None):
 r=requests.post(RAILWAY_API,headers={"Project-Access-Token":token,"Content-Type":"application/json"},json={"query":query,"variables":variables or {}},timeout=60);r.raise_for_status();o=r.json()
 if o.get("errors"):raise RuntimeError(o["errors"])
 return o["data"]
def token_context(t):return gql(t,"query { projectToken { projectId environmentId } }")["projectToken"]
def project_services(t,pid):
 q="""query project($id: String!) { project(id: $id) { id name services { edges { node { id name } } } } }""";p=gql(t,q,{"id":pid})["project"];return p["name"],[x["node"] for x in p["services"]["edges"]]
def deployments(t,pid,sid,eid,limit=1000):
 q="""query deployments($input: DeploymentListInput!, $first: Int!) { deployments(input: $input, first: $first) { edges { node { id status createdAt } } } }""";d=gql(t,q,{"input":{"projectId":pid,"serviceId":sid,"environmentId":eid},"first":limit});return [x["node"] for x in d["deployments"]["edges"]]
def deployment_logs(t,did,limit=1000):
 # Railway's public API cookbook documents deploymentLogs with a small integer
 # limit (example: 100). 50,000 was rejected by the API as "Invalid input".
 q="""query deploymentLogs($deploymentId: String!, $limit: Int) { deploymentLogs(deploymentId: $deploymentId, limit: $limit) { timestamp message severity } }""";return gql(t,q,{"deploymentId":did,"limit":limit})["deploymentLogs"]
def month_bounds(m):
 y,n=map(int,m.split("-"));s=datetime(y,n,1,tzinfo=timezone.utc);e=datetime(y+(n==12),1 if n==12 else n+1,1,tzinfo=timezone.utc);return s,e
def dt(s):return datetime.fromisoformat(s.replace("Z","+00:00"))
ANCHOR=re.compile(r"(?P<ts>20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00)\s+(?P<symbol>[A-Z0-9]+USDT)\b")
FIELD={"confirmed":re.compile(r"\bconfirmed=(True|False)"),"regime":re.compile(r"\bregime=([a-zA-Z_]+)"),"raw_signal":re.compile(r"\braw_signal=(True|False)"),"score_confirmed":re.compile(r"\bscore_confirmed=(True|False)"),"raw_quality":re.compile(r"\braw_quality=([-+0-9.eE]+|nan)"),"confirmation_quality":re.compile(r"\bconfirmation_quality=([-+0-9.eE]+|nan)"),"combined_setup":re.compile(r"\bcombined_setup=([-+0-9.eE]+|nan)"),"confirmation_tests":re.compile(r"\bconfirmation_tests=(\d+)")}
GRADE_RE=re.compile(r"TIDE GRADE:\s*(A\+|A-|A|B\+|B|C)");SIDE_RE=re.compile(r"\b(?:side|direction)\s*[:=]\s*(LONG|SHORT|BUY|SELL)\b",re.I)
def val(n,x):
 m=FIELD[n].search(x)
 if not m:return None
 v=m.group(1)
 if v in("True","False"):return v=="True"
 if n=="confirmation_tests":return int(v)
 if n in("raw_quality","confirmation_quality","combined_setup"):return None if v=="nan" else float(v)
 return v
def infer_grade(e):
 if e.get("explicit_grade"):return e["explicit_grade"],False
 rq,cq,cs,ct=e.get("raw_quality"),e.get("confirmation_quality"),e.get("combined_setup"),e.get("confirmation_tests")
 if rq is None or cq is None or cs is None:return "CONFIRMED",True
 if rq>=75 and cq>=97 and cs>=82 and(ct or 0)>=3:return "A+",True
 if rq>=65 and cq>=95 and cs>=76 and(ct or 0)>=2:return "A",True
 if rq>=58 and cq>=90 and cs>=70 and(ct or 0)>=2:return "B+",True
 return "CONFIRMED",True
def parse_events(rows,start,end):
 rows=sorted(rows,key=lambda x:x.get("timestamp",""));text="\n".join(str(x.get("message","")) for x in rows);aa=list(ANCHOR.finditer(text));out=[]
 for i,a in enumerate(aa):
  t=dt(a.group("ts"))
  if not(start<=t<end):continue
  c=text[a.start():aa[i+1].start() if i+1<len(aa) else min(len(text),a.start()+1200)];e={"timestamp":t,"symbol":a.group("symbol")}
  for k in FIELD:e[k]=val(k,c)
  gm=GRADE_RE.search(c);e["explicit_grade"]=gm.group(1) if gm else None;sm=SIDE_RE.search(c);e["side"]=("LONG" if sm.group(1).upper() in("LONG","BUY") else "SHORT") if sm else("SHORT" if e.get("regime")=="downtrend" else("LONG" if e.get("regime")=="uptrend" else None))
  if e.get("confirmed") is True:e["grade"],e["grade_inferred"]=infer_grade(e);out.append(e)
 u={}
 for e in out:
  candle=e["timestamp"].replace(minute=e["timestamp"].minute//15*15,second=0,microsecond=0);u[(e["symbol"],candle)]=e
 return sorted(u.values(),key=lambda x:x["timestamp"])
def bybit_15m(sym,start,hours=8):
 s=int((start-timedelta(minutes=30)).timestamp()*1000);e=int((start+timedelta(hours=hours)).timestamp()*1000);r=requests.get(BYBIT_KLINE,params={"category":"linear","symbol":sym,"interval":"15","start":s,"end":e,"limit":1000},timeout=30);r.raise_for_status();j=r.json()
 if j.get("retCode")!=0:raise RuntimeError(j)
 d=pd.DataFrame(j["result"]["list"],columns=["ts","open","high","low","close","volume","turnover"])
 if d.empty:return d
 for c in["open","high","low","close"]:d[c]=pd.to_numeric(d[c])
 d["time"]=pd.to_datetime(pd.to_numeric(d["ts"]),unit="ms",utc=True);return d.sort_values("time").reset_index(drop=True)
def measure(e):
 d=bybit_15m(e["symbol"],e["timestamp"]);a=d[d.time>=pd.Timestamp(e["timestamp"])].copy()
 if a.empty:return e
 entry=float(a.iloc[0].open);sgn=-1 if e["side"]=="SHORT" else 1;e["entry_price"]=entry
 for h in(1,4,6):
  x=a[a.time<=pd.Timestamp(e["timestamp"]+timedelta(hours=h))]
  if x.empty:continue
  e[f"ret_{h}h"]=sgn*(float(x.iloc[-1].close)/entry-1)
  if sgn==1:e[f"mfe_{h}h"]=float(x.high.max()/entry-1);e[f"mae_{h}h"]=float(x.low.min()/entry-1)
  else:e[f"mfe_{h}h"]=float(1-x.low.min()/entry);e[f"mae_{h}h"]=float(1-x.high.max()/entry)
 return e
def pct(x):return"n/a" if x is None or pd.isna(x) else f"{100*x:+.2f}%"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--month",default=datetime.now(timezone.utc).strftime("%Y-%m"));ap.add_argument("--out",default="reports");ap.add_argument("--service-id",default=os.getenv("RAILWAY_SERVICE_ID"));a=ap.parse_args();token=os.getenv("RAILWAY_TOKEN")
 if not token:raise SystemExit("Missing RAILWAY_TOKEN")
 start,end=month_bounds(a.month);end=min(end,datetime.now(timezone.utc));ctx=token_context(token);pn,ss=project_services(token,ctx["projectId"]);print("Railway project:",pn,"projectId=",ctx["projectId"],"environmentId=",ctx["environmentId"]);print("Services visible to token:",[(s["name"],s["id"]) for s in ss])
 if a.service_id:
  targets=[s for s in ss if s["id"]==a.service_id]
  if not targets:raise RuntimeError(f"RAILWAY_SERVICE_ID {a.service_id} is not in this project. Visible services: {ss}")
 else:targets=[s for s in ss if"tide" in s["name"].lower()] or ss
 allrows=[];dc=0;ok=0;failed=0
 for svc in targets:
  deps=deployments(token,ctx["projectId"],svc["id"],ctx["environmentId"]);chosen=[d for d in deps if dt(d["createdAt"])<end];print(f"Service {svc['name']} ({svc['id']}): {len(deps)} deployments returned; inspecting {len(chosen)}");dc+=len(chosen)
  for n,d in enumerate(chosen,1):
   try:
    rows=deployment_logs(token,d["id"],1000);allrows.extend(rows);ok+=1;print(svc["name"],n,d["id"],d["status"],len(rows))
   except Exception as ex:failed+=1;print("WARN",svc["name"],d["id"],d["status"],ex)
 ev=parse_events(allrows,start,end);print("Log fetch success/fail:",ok,failed);print("Confirmed events:",len(ev));me=[]
 for i,e in enumerate(ev,1):
  try:me.append(measure(e));print(f"market {i}/{len(ev)} {e['symbol']}")
  except Exception as ex:print("MARKET WARN",e["symbol"],ex);me.append(e)
  time.sleep(.05)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(me)
 if not df.empty:
  df["timestamp"]=df["timestamp"].astype(str);df.to_csv(out/f"tide_live_signals_{a.month}.csv",index=False)
 lines=[f"# Tide live forward-test — {a.month}","","Source: actual Railway deployment logs; market outcomes: Bybit 15m candles.",f"Window (UTC): {start.isoformat()} to {end.isoformat()}",f"Services inspected: {', '.join(s['name'] for s in targets)}",f"Deployments inspected: {dc}",f"Deployment log fetches successful: {ok}",f"Deployment log fetches failed: {failed}",f"Log rows fetched: {len(allrows)}",f"Unique confirmed signals: {len(ev)}",""]
 if df.empty:lines+=["No confirmed=True signal was reconstructed from the retained Railway logs."]
 else:
  lines+=["## Headline",""]
  for h in(1,4,6):
   c=f"ret_{h}h";x=pd.to_numeric(df[c],errors="coerce").dropna() if c in df else pd.Series(dtype=float)
   if len(x):lines+=[f"- {h}h: n={len(x)}, mean={pct(x.mean())}, median={pct(x.median())}, win rate={(x>0).mean()*100:.1f}%"]
  lines+=["","## By grade",""]
  for grade,g in df.groupby("grade",dropna=False):
   x=pd.to_numeric(g["ret_4h"],errors="coerce").dropna() if"ret_4h" in g else pd.Series(dtype=float);lines+=[f"- {grade}: signals={len(g)}, 4h mean={pct(x.mean() if len(x) else None)}, 4h win={(x>0).mean()*100:.1f}%" if len(x) else f"- {grade}: signals={len(g)}, 4h unavailable"]
 (out/f"tide_monthly_summary_{a.month}.md").write_text("\n".join(lines),encoding="utf-8");print("\n".join(lines));print("\nSaved to",out)
if __name__=="__main__":main()
