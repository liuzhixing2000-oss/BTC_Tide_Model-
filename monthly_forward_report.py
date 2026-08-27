#!/usr/bin/env python3
"""Build a monthly forward-test report from REAL Railway runtime logs.

This does not re-backtest the model. It downloads the deployments/logs that
actually ran on Railway, reconstructs confirmed signals, then asks Bybit for
subsequent 15m candles and measures what happened after each live signal.
"""
from __future__ import annotations
import argparse, calendar, json, math, os, re, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

RAILWAY_API = "https://backboard.railway.com/graphql/v2"
BYBIT_KLINE = "https://api.bybit.com/v5/market/kline"
DEFAULT_SERVICE_ID = "192b72a4-32a4-4064-a279-c19a070c1896"


def gql(token, query, variables=None):
    r = requests.post(RAILWAY_API, headers={"Project-Access-Token": token, "Content-Type": "application/json"},
                      json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status(); obj = r.json()
    if obj.get("errors"): raise RuntimeError(obj["errors"])
    return obj["data"]


def token_context(token):
    q = "query { projectToken { projectId environmentId } }"
    return gql(token, q)["projectToken"]


def deployments(token, project_id, service_id, limit=1000):
    q = """query deployments($input: DeploymentListInput!, $first: Int!) {
      deployments(input: $input, first: $first) { edges { node { id status createdAt } } }
    }"""
    d = gql(token, q, {"input": {"projectId": project_id, "serviceId": service_id}, "first": limit})
    return [x["node"] for x in d["deployments"]["edges"]]


def deployment_logs(token, deployment_id, limit=50000):
    q = """query deploymentLogs($deploymentId: String!, $limit: Int) {
      deploymentLogs(deploymentId: $deploymentId, limit: $limit) { timestamp message severity }
    }"""
    return gql(token, q, {"deploymentId": deployment_id, "limit": limit})["deploymentLogs"]


def month_bounds(month):
    y, m = map(int, month.split("-")); start = datetime(y, m, 1, tzinfo=timezone.utc)
    end = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    return start, end


def dt(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))

# Railway output can interleave threads, so we do not assume one message == one event.
# Anchor each event on ISO timestamp + SYMBOLUSDT and read fields only until next anchor.
ANCHOR = re.compile(r"(?P<ts>20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00)\s+(?P<symbol>[A-Z0-9]+USDT)\b")
FIELD = {
    "confirmed": re.compile(r"\bconfirmed=(True|False)"),
    "regime": re.compile(r"\bregime=([a-zA-Z_]+)"),
    "raw_signal": re.compile(r"\braw_signal=(True|False)"),
    "score_confirmed": re.compile(r"\bscore_confirmed=(True|False)"),
    "raw_quality": re.compile(r"\braw_quality=([-+0-9.eE]+|nan)"),
    "confirmation_quality": re.compile(r"\bconfirmation_quality=([-+0-9.eE]+|nan)"),
    "combined_setup": re.compile(r"\bcombined_setup=([-+0-9.eE]+|nan)"),
    "confirmation_tests": re.compile(r"\bconfirmation_tests=(\d+)"),
}
GRADE_RE = re.compile(r"TIDE GRADE:\s*(A\+|A-|A|B\+|B|C)")
SIDE_RE = re.compile(r"\b(?:side|direction)\s*[:=]\s*(LONG|SHORT|BUY|SELL)\b", re.I)


def val(name, text):
    m = FIELD[name].search(text)
    if not m: return None
    x = m.group(1)
    if x in ("True", "False"): return x == "True"
    if name == "confirmation_tests": return int(x)
    if name in ("raw_quality", "confirmation_quality", "combined_setup"):
        return None if x == "nan" else float(x)
    return x


def infer_grade(e):
    # Prefer explicit grade if logged. Fallback is intentionally conservative and
    # labelled inferred; it mirrors the production hierarchy, not research truth.
    if e.get("explicit_grade"): return e["explicit_grade"], False
    rq, cq, cs, ct = e.get("raw_quality"), e.get("confirmation_quality"), e.get("combined_setup"), e.get("confirmation_tests")
    if rq is None or cq is None or cs is None: return "CONFIRMED", True
    if rq >= 75 and cq >= 97 and cs >= 82 and (ct or 0) >= 3: return "A+", True
    if rq >= 65 and cq >= 95 and cs >= 76 and (ct or 0) >= 2: return "A", True
    if rq >= 58 and cq >= 90 and cs >= 70 and (ct or 0) >= 2: return "B+", True
    return "CONFIRMED", True


def parse_events(rows, start, end):
    # Sort then concatenate: anchors let us recover many records even when messages are glued.
    rows = sorted(rows, key=lambda x: x.get("timestamp", ""))
    text = "\n".join(str(x.get("message", "")) for x in rows)
    anchors = list(ANCHOR.finditer(text)); out = []
    for i, a in enumerate(anchors):
        t = dt(a.group("ts"));
        if not (start <= t < end): continue
        chunk = text[a.start(): anchors[i+1].start() if i+1 < len(anchors) else min(len(text), a.start()+1200)]
        e = {"timestamp": t, "symbol": a.group("symbol")}
        for k in FIELD: e[k] = val(k, chunk)
        gm = GRADE_RE.search(chunk); e["explicit_grade"] = gm.group(1) if gm else None
        sm = SIDE_RE.search(chunk)
        if sm: e["side"] = "LONG" if sm.group(1).upper() in ("LONG", "BUY") else "SHORT"
        else: e["side"] = "SHORT" if e.get("regime") == "downtrend" else ("LONG" if e.get("regime") == "uptrend" else None)
        # A real candidate must have reached confirmed=True. raw/score flags are kept for diagnostics.
        if e.get("confirmed") is True:
            e["grade"], e["grade_inferred"] = infer_grade(e); out.append(e)
    # dedupe same symbol/candle, common across restart/replica overlap
    uniq = {}
    for e in out:
        candle = e["timestamp"].replace(minute=(e["timestamp"].minute//15)*15, second=0, microsecond=0)
        uniq[(e["symbol"], candle)] = e
    return sorted(uniq.values(), key=lambda x: x["timestamp"])


def bybit_15m(symbol, start, hours=8):
    s = int((start - timedelta(minutes=30)).timestamp()*1000); e = int((start + timedelta(hours=hours)).timestamp()*1000)
    r = requests.get(BYBIT_KLINE, params={"category":"linear","symbol":symbol,"interval":"15","start":s,"end":e,"limit":1000}, timeout=30)
    r.raise_for_status(); j=r.json()
    if j.get("retCode") != 0: raise RuntimeError(j)
    data=j["result"]["list"]
    df=pd.DataFrame(data, columns=["ts","open","high","low","close","volume","turnover"])
    if df.empty: return df
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c])
    df["time"]=pd.to_datetime(pd.to_numeric(df["ts"]),unit="ms",utc=True)
    return df.sort_values("time").reset_index(drop=True)


def measure(e):
    df=bybit_15m(e["symbol"], e["timestamp"])
    after=df[df.time >= pd.Timestamp(e["timestamp"])].copy()
    if after.empty: return e
    entry=float(after.iloc[0].open); sign=-1 if e["side"]=="SHORT" else 1
    e["entry_price"]=entry
    for h in (1,4,6):
        x=after[after.time <= pd.Timestamp(e["timestamp"]+timedelta(hours=h))]
        if x.empty: continue
        e[f"ret_{h}h"] = sign*(float(x.iloc[-1].close)/entry-1)
        if sign==1:
            e[f"mfe_{h}h"]=float(x.high.max()/entry-1); e[f"mae_{h}h"]=float(x.low.min()/entry-1)
        else:
            e[f"mfe_{h}h"]=float(1-x.low.min()/entry); e[f"mae_{h}h"]=float(1-x.high.max()/entry)
    return e


def pct(x): return "n/a" if x is None or pd.isna(x) else f"{100*x:+.2f}%"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--month", default=datetime.now(timezone.utc).strftime("%Y-%m")); ap.add_argument("--out",default="reports")
    ap.add_argument("--service-id",default=os.getenv("RAILWAY_SERVICE_ID",DEFAULT_SERVICE_ID)); a=ap.parse_args()
    token=os.getenv("RAILWAY_TOKEN")
    if not token: raise SystemExit("Missing RAILWAY_TOKEN (Railway Project Token).")
    start,end=month_bounds(a.month); now=datetime.now(timezone.utc); end=min(end,now)
    ctx=token_context(token); deps=deployments(token,ctx["projectId"],a.service_id)
    chosen=[d for d in deps if dt(d["createdAt"]) < end and d["status"] not in ("REMOVED",)]
    print(f"Fetching {len(chosen)} deployments for {a.month} ...")
    allrows=[]
    for n,d in enumerate(chosen,1):
        try:
            rows=deployment_logs(token,d["id"]); allrows.extend(rows); print(n,d["id"],len(rows))
        except Exception as ex: print("WARN",d["id"],ex)
    events=parse_events(allrows,start,end); print("Confirmed events:",len(events))
    measured=[]
    for i,e in enumerate(events,1):
        try: measured.append(measure(e)); print(f"market {i}/{len(events)} {e['symbol']}")
        except Exception as ex: print("MARKET WARN",e["symbol"],ex); measured.append(e)
        time.sleep(.05)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(measured)
    if not df.empty:
        for c in df.columns:
            if c=="timestamp": df[c]=df[c].astype(str)
        df.to_csv(out/f"tide_live_signals_{a.month}.csv",index=False)
    lines=[f"# Tide live forward-test — {a.month}","",f"Source: actual Railway deployment logs; market outcomes: Bybit 15m candles.",f"Window (UTC): {start.isoformat()} to {end.isoformat()}",f"Deployments inspected: {len(chosen)}",f"Unique confirmed signals: {len(events)}",""]
    if df.empty: lines += ["No confirmed=True signal was reconstructed from the retained Railway logs."]
    else:
        lines += ["## Headline",""]
        for h in (1,4,6):
            c=f"ret_{h}h"
            if c in df:
                x=pd.to_numeric(df[c],errors="coerce").dropna()
                if len(x): lines += [f"- {h}h: n={len(x)}, mean={pct(x.mean())}, median={pct(x.median())}, win rate={(x>0).mean()*100:.1f}%"]
        lines += ["", "## By grade",""]
        for grade,g in df.groupby("grade",dropna=False):
            x=pd.to_numeric(g.get("ret_4h"),errors="coerce").dropna()
            lines += [f"- {grade}: signals={len(g)}, 4h mean={pct(x.mean() if len(x) else None)}, 4h win={(x>0).mean()*100:.1f}%" if len(x) else f"- {grade}: signals={len(g)}, 4h unavailable"]
        lines += ["", "## Important interpretation", "", "This is a forward test of signals that actually appeared in retained Railway logs, not a historical model rerun. Duplicate symbol/candle events are removed. When an explicit LONG/SHORT field is absent, direction is inferred as LONG in uptrend and SHORT in downtrend; inspect the CSV `side` field before treating P/L as production-grade. Grades marked `grade_inferred=True` were reconstructed because the raw diagnostic line did not contain the Telegram grade label."]
    (out/f"tide_monthly_summary_{a.month}.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines)); print("\nSaved to",out)

if __name__=="__main__": main()
