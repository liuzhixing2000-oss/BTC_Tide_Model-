#!/usr/bin/env python3
"""Build a monthly forward-test report from REAL Railway runtime logs.

Key rules:
- Use only logs that actually ran on Railway (no historical rerun).
- A production candidate is a CLOSED 15m candle with score_confirmed=True.
- Tide V10.x is a long-rebound strategy, so direction defaults to LONG unless an explicit side/direction field says otherwise.
- Railway GraphQL returns timestamp outside message; parser uses row timestamp.
- Railway stdout can interleave multiple threads; messages are segmented around SYMBOLUSDT anchors.
- Raw Railway rows plus parser diagnostics are saved into reports/.
"""
from __future__ import annotations
import argparse, json, os, re, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
REPORT_VERSION="v4.1-counter-fix"
RAILWAY_API="https://backboard.railway.com/graphql/v2";BYBIT_KLINE="https://api.bybit.com/v5/market/kline"
def gql(t,q,v=None):
 r=requests.post(RAILWAY_API,headers={"Project-Access-Token":t,"Content-Type":"application/json"},json={"query":q,"variables":v or {}},timeout=60);r.raise_for_status();o=r.json()
 if o.get("errors"):raise RuntimeError(o["errors"])
 return o["data"]
def token_context(t):return gql(t,"query { projectToken { projectId environmentId } }")["projectToken"]
def project_services(t,p):
 q="query project($id: String!) { project(id: $id) { id name services { edges { node { id name } } } } }";x=gql(t,q,{"id":p})["project"];return x["name"],[e["node"] for e in x["services"]["edges"]]
def deployments(t,p,s,e,limit=1000):
 q="query deployments($input: DeploymentListInput!, $first: Int!) { deployments(input: $input, first: $first) { edges { node { id status createdAt } } } }";return [x["node"] for x in gql(t,q,{"input":{"projectId":p,"serviceId":s,"environmentId":e},"first":limit})["deployments"]["edges"]]
def deployment_logs(t,d,limit=1000):
 q="query deploymentLogs($deploymentId: String!, $limit: Int) { deploymentLogs(deploymentId: $deploymentId, limit: $limit) { timestamp message severity } }";return gql(t,q,{"deploymentId":d,"limit":limit})["deploymentLogs"]
def month_bounds(m):
 y,n=map(int,m.split("-"));return datetime(y,n,1,tzinfo=timezone.utc),datetime(y+(n==12),1 if n==12 else n+1,1,tzinfo=timezone.utc)
def dt(v):
 if isinstance(v,datetime):return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
 return datetime.fromisoformat(str(v).replace("Z","+00:00"))
ISO=r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)";ANCHOR=re.compile(rf"(?:(?P<ts>{ISO})\s+)?(?P<symbol>[A-Z0-9]{{1,32}}USDT)\b")
FIELD={"confirmed":re.compile(r"\bconfirmed\s*=\s*(True|False)"),"regime":re.compile(r"\bregime\s*=\s*([A-Za-z_]+)"),"raw_signal":re.compile(r"\braw_signal\s*=\s*(True|False)"),"score_confirmed":re.compile(r"\bscore_confirmed\s*=\s*(True|False)"),"raw_quality":re.compile(r"\braw_quality\s*=\s*([-+0-9.eE]+|nan)"),"confirmation_quality":re.compile(r"\bconfirmation_quality\s*=\s*([-+0-9.eE]+|nan)"),"combined_setup":re.compile(r"\bcombined_setup\s*=\s*([-+0-9.eE]+|nan)"),"confirmation_tests":re.compile(r"\bconfirmation_tests\s*=\s*(\d+)")}
GRADE_RE=re.compile(r"TIDE GRADE:\s*(A\+|A-|A|B\+|B|C)");SIDE_RE=re.compile(r"\b(?:side|direction)\s*[:=]\s*(LONG|SHORT|BUY|SELL)\b",re.I)
def field(n,t):
 m=FIELD[n].search(t)
 if not m:return None
 x=m.group(1)
 if x in("True","False"):return x=="True"
 if n=="confirmation_tests":return int(x)
 if n in("raw_quality","confirmation_quality","combined_setup"):return None if x.lower()=="nan" else float(x)
 return x
def segments(msg):
 text=str(msg or "");aa=list(ANCHOR.finditer(text));out=[]
 for i,a in enumerate(aa):
  c=text[a.start():aa[i+1].start() if i+1<len(aa) else len(text)]
  if any(k in c for k in("confirmed=","score_confirmed=","raw_signal=","TIDE GRADE:")):out.append((a,c))
 return out
def infer_grade(e):
 if e.get("explicit_grade"):return e["explicit_grade"],False
 rq,cq,cs,ct=e.get("raw_quality"),e.get("confirmation_quality"),e.get("combined_setup"),e.get("confirmation_tests")
 if rq is None or cq is None or cs is None:return "CONFIRMED",True
 if rq>=75 and cq>=97 and cs>=82 and (ct or 0)>=3:return "A+",True
 if rq>=65 and cq>=95 and cs>=76 and (ct or 0)>=2:return "A",True
 if rq>=58 and cq>=90 and cs>=70 and (ct or 0)>=2:return "B+",True
 return "CONFIRMED",True
def make_event(a,c,rowts):
 t=dt(a.group("ts")) if a.group("ts") else dt(rowts);e={"timestamp":t,"symbol":a.group("symbol")}
 for k in FIELD:e[k]=field(k,c)
 gm=GRADE_RE.search(c);e["explicit_grade"]=gm.group(1) if gm else None;sm=SIDE_RE.search(c);e["side"]=("LONG" if sm.group(1).upper() in("LONG","BUY") else "SHORT") if sm else "LONG";return e
def parse_events(rows,start,end):
 d=Counter();cand=[]
 for row in sorted(rows,key=lambda x:str(x.get("timestamp",""))):
  msg=str(row.get("message",""));d["rows"]+=1;d["confirmed_true_literals"]+=len(re.findall(r"(?<!score_)confirmed\s*=\s*True",msg));d["score_confirmed_true_literals"]+=len(re.findall(r"\bscore_confirmed\s*=\s*True",msg));d["raw_signal_true_literals"]+=len(re.findall(r"\braw_signal\s*=\s*True",msg));seg=segments(msg);d["segments"]+=len(seg)
  for a,c in seg:
   try:e=make_event(a,c,row.get("timestamp"))
   except Exception:d["bad_timestamp"]+=1;continue
   if not start<=e["timestamp"]<end:continue
   if e.get("confirmed") is True:d["closed_candle_segments"]+=1
   if e.get("score_confirmed") is True:d["signal_true_segments"]+=1
   if e.get("confirmed") is True and e.get("score_confirmed") is True:e["grade"],e["grade_inferred"]=infer_grade(e);cand.append(e);d["qualified_before_dedupe"]+=1
 uniq={}
 for e in cand:
  candle=e["timestamp"].replace(minute=e["timestamp"].minute//15*15,second=0,microsecond=0);key=(e["symbol"],candle);old=uniq.get(key)
  if not old or sum(v is not None for v in e.values())>sum(v is not None for v in old.values()):uniq[key]=e
 d["unique_events"]=len(uniq);return sorted(uniq.values(),key=lambda x:x["timestamp"]),d
def parser_self_test():
 r={"timestamp":"2026-08-10T12:15:01.000Z","message":"FILUSDT confirmed=True regime=downtrend raw_signal=True score_confirmed=True raw_quality=70 confirmation_quality=96 combined_setup=76 confirmation_tests=3"};ev,d=parse_events([r],datetime(2026,8,1,tzinfo=timezone.utc),datetime(2026,9,1,tzinfo=timezone.utc));assert len(ev)==1,(ev,d)
def bybit_15m(sym,start,hours=8):
 s=int((start-timedelta(minutes=30)).timestamp()*1000);e=int((start+timedelta(hours=hours)).timestamp()*1000);r=requests.get(BYBIT_KLINE,params={"category":"linear","symbol":sym,"interval":"15","start":s,"end":e,"limit":1000},timeout=30);r.raise_for_status();j=r.json()
 if j.get("retCode")!=0:raise RuntimeError(j)
 df=pd.DataFrame(j["result"]["list"],columns=["ts","open","high","low","close","volume","turnover"])
 if df.empty:return df
 for c in("open","high","low","close"):df[c]=pd.to_numeric(df[c])
 df["time"]=pd.to_datetime(pd.to_numeric(df["ts"]),unit="ms",utc=True);return df.sort_values("time").reset_index(drop=True)
def measure(e):
 df=bybit_15m(e["symbol"],e["timestamp"]);a=df[df.time>=pd.Timestamp(e["timestamp"])].copy()
 if a.empty:return e
 entry=float(a.iloc[0].open);sgn=-1 if e["side"]=="SHORT" else 1;e["entry_price"]=entry
 for h in(1,4,6):
  x=a[a.time<=pd.Timestamp(e["timestamp"]+timedelta(hours=h))]
  if x.empty:continue
  e[f"ret_{h}h"]=sgn*(float(x.iloc[-1].close)/entry-1)
  if sgn==1:e[f"mfe_{h}h"]=float(x.high.max()/entry-1);e[f"mae_{h}h"]=float(x.low.min()/entry-1)
  else:e[f"mfe_{h}h"]=float(1-x.low.min()/entry);e[f"mae_{h}h"]=float(1-x.high.max()/entry)
 return e
def pct(x):return "n/a" if x is None or pd.isna(x) else f"{100*x:+.2f}%"
def main():
 parser_self_test();print("Report parser:",REPORT_VERSION,"self-test=PASS");ap=argparse.ArgumentParser();ap.add_argument("--month",default=datetime.now(timezone.utc).strftime("%Y-%m"));ap.add_argument("--out",default="reports");ap.add_argument("--service-id",default=os.getenv("RAILWAY_SERVICE_ID"));a=ap.parse_args();token=os.getenv("RAILWAY_TOKEN")
 if not token:raise SystemExit("Missing RAILWAY_TOKEN")
 start,end=month_bounds(a.month);end=min(end,datetime.now(timezone.utc));ctx=token_context(token);pn,ss=project_services(token,ctx["projectId"]);print("Railway project:",pn);print("Services:",[(s["name"],s["id"]) for s in ss]);targets=[s for s in ss if s["id"]==a.service_id] if a.service_id else ([s for s in ss if "tide" in s["name"].lower()] or ss)
 if not targets:raise RuntimeError("No target Railway service")
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);allrows=[];dc=ok=fail=0
 for svc in targets:
  deps=deployments(token,ctx["projectId"],svc["id"],ctx["environmentId"]);chosen=[d for d in deps if dt(d["createdAt"])<end];dc+=len(chosen);print(f"Service {svc['name']}: deployments={len(chosen)}")
  for n,dpl in enumerate(chosen,1):
   try:
    rows=deployment_logs(token,dpl["id"],1000);ok+=1
    for r in rows:r["deployment_id"]=dpl["id"];r["deployment_status"]=dpl["status"];r["service_name"]=svc["name"]
    allrows.extend(rows);print(n,dpl["id"],dpl["status"],len(rows))
   except Exception as ex:fail+=1;print("WARN",dpl["id"],repr(ex))
 raw=out/f"railway_logs_{a.month}.jsonl"
 with raw.open("w",encoding="utf-8") as f:
  for r in allrows:f.write(json.dumps(r,ensure_ascii=False,default=str)+"\n")
 ev,diag=parse_events(allrows,start,end)
 # Counter only accepts numeric increments. Convert before adding string metadata.
 diag_dict=dict(diag);diag_dict.update({"report_version":REPORT_VERSION,"deployments":dc,"log_fetch_ok":ok,"log_fetch_failed":fail,"log_rows":len(allrows)})
 (out/f"parser_diagnostics_{a.month}.json").write_text(json.dumps(diag_dict,indent=2,default=str),encoding="utf-8");print("PARSER DIAGNOSTICS",json.dumps(diag_dict,sort_keys=True));print("Unique qualified events:",len(ev))
 measured=[]
 for i,e in enumerate(ev,1):
  try:measured.append(measure(e));print(f"market {i}/{len(ev)} {e['symbol']}")
  except Exception as ex:print("MARKET WARN",e["symbol"],repr(ex));measured.append(e)
  time.sleep(.05)
 df=pd.DataFrame(measured)
 if not df.empty:df["timestamp"]=df["timestamp"].astype(str);df.to_csv(out/f"tide_live_signals_{a.month}.csv",index=False)
 lines=[f"# Tide live forward-test — {a.month}","",f"Report parser: {REPORT_VERSION}","Source: actual Railway deployment logs; market outcomes: Bybit 15m candles.",f"Window (UTC): {start.isoformat()} to {end.isoformat()}",f"Deployments inspected: {dc}",f"Log rows fetched: {len(allrows)}",f"Closed-candle diagnostic segments: {diag.get('closed_candle_segments',0)}",f"Signal-true diagnostic segments: {diag.get('signal_true_segments',0)}",f"Unique production signals: {len(ev)}",""]
 if df.empty:lines.append("No production signal reconstructed. Inspect parser_diagnostics and railway_logs artifacts before interpreting this as a model result.")
 else:
  lines+=["## Forward outcome",""]
  for h in(1,4,6):
   c=f"ret_{h}h";x=pd.to_numeric(df[c],errors="coerce").dropna() if c in df else pd.Series(dtype=float)
   if len(x):lines.append(f"- {h}h: n={len(x)}, mean={pct(x.mean())}, median={pct(x.median())}, win rate={(x>0).mean()*100:.1f}%")
  lines+=["","## By reconstructed grade",""]
  for grade,g in df.groupby("grade",dropna=False):
   x=pd.to_numeric(g["ret_4h"],errors="coerce").dropna() if "ret_4h" in g else pd.Series(dtype=float);lines.append(f"- {grade}: signals={len(g)}, 4h mean={pct(x.mean() if len(x) else None)}, 4h win={(x>0).mean()*100:.1f}%" if len(x) else f"- {grade}: signals={len(g)}, 4h unavailable")
 (out/f"tide_monthly_summary_{a.month}.md").write_text("\n".join(lines),encoding="utf-8");print("\n".join(lines));print("Saved to",out)
if __name__=="__main__":main()
