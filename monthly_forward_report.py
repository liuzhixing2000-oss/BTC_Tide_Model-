#!/usr/bin/env python3
"""Build a monthly forward-test report from REAL Railway runtime logs.

Key rules:
- Use only logs that actually ran on Railway (no historical rerun).
- A production candidate is a CLOSED 15m candle with score_confirmed=True.
- Tide V10.x is a long-rebound strategy, so direction defaults to LONG unless an
  explicit side/direction field says otherwise.
- Railway stdout can interleave multiple threads. The parser therefore works
  row-by-row first and then falls back to glued-message segmentation.
- Raw Railway rows plus parser diagnostics are written into reports/ so future
  parser changes can be audited without repeatedly guessing the log format.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

RAILWAY_API = "https://backboard.railway.com/graphql/v2"
BYBIT_KLINE = "https://api.bybit.com/v5/market/kline"


def gql(token, query, variables=None):
    r = requests.post(
        RAILWAY_API,
        headers={"Project-Access-Token": token, "Content-Type": "application/json"},
        json={"query": query, "variables": variables or {}},
        timeout=60,
    )
    r.raise_for_status()
    obj = r.json()
    if obj.get("errors"):
        raise RuntimeError(obj["errors"])
    return obj["data"]


def token_context(token):
    return gql(token, "query { projectToken { projectId environmentId } }")["projectToken"]


def project_services(token, project_id):
    q = """query project($id: String!) {
      project(id: $id) { id name services { edges { node { id name } } } }
    }"""
    p = gql(token, q, {"id": project_id})["project"]
    return p["name"], [x["node"] for x in p["services"]["edges"]]


def deployments(token, project_id, service_id, environment_id, limit=1000):
    q = """query deployments($input: DeploymentListInput!, $first: Int!) {
      deployments(input: $input, first: $first) {
        edges { node { id status createdAt } }
      }
    }"""
    d = gql(
        token,
        q,
        {
            "input": {
                "projectId": project_id,
                "serviceId": service_id,
                "environmentId": environment_id,
            },
            "first": limit,
        },
    )
    return [x["node"] for x in d["deployments"]["edges"]]


def deployment_logs(token, deployment_id, limit=1000):
    q = """query deploymentLogs($deploymentId: String!, $limit: Int) {
      deploymentLogs(deploymentId: $deploymentId, limit: $limit) {
        timestamp message severity
      }
    }"""
    return gql(token, q, {"deploymentId": deployment_id, "limit": limit})["deploymentLogs"]


def month_bounds(month):
    y, m = map(int, month.split("-"))
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    end = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    return start, end


def dt(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


ISO = r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)"
# Real V10.x diagnostic rows look like:
# 2026-08-27T...+00:00 SYMBOLUSDT confirmed=False regime=... raw_signal=False
# score_confirmed=False raw_quality=... confirmation_quality=... combined_setup=...
ANCHOR = re.compile(rf"(?:(?P<ts>{ISO})\s+)?(?P<symbol>[A-Z0-9]{{1,32}}USDT)\b")

FIELD_PATTERNS = {
    "confirmed": re.compile(r"\bconfirmed\s*=\s*(True|False)"),
    "regime": re.compile(r"\bregime\s*=\s*([A-Za-z_]+)"),
    "raw_signal": re.compile(r"\braw_signal\s*=\s*(True|False)"),
    "score_confirmed": re.compile(r"\bscore_confirmed\s*=\s*(True|False)"),
    "raw_quality": re.compile(r"\braw_quality\s*=\s*([-+0-9.eE]+|nan)"),
    "confirmation_quality": re.compile(r"\bconfirmation_quality\s*=\s*([-+0-9.eE]+|nan)"),
    "combined_setup": re.compile(r"\bcombined_setup\s*=\s*([-+0-9.eE]+|nan)"),
    "confirmation_tests": re.compile(r"\bconfirmation_tests\s*=\s*(\d+)"),
}
GRADE_RE = re.compile(r"TIDE GRADE:\s*(A\+|A-|A|B\+|B|C)")
SIDE_RE = re.compile(r"\b(?:side|direction)\s*[:=]\s*(LONG|SHORT|BUY|SELL)\b", re.I)


def read_field(name, text):
    m = FIELD_PATTERNS[name].search(text)
    if not m:
        return None
    x = m.group(1)
    if x in ("True", "False"):
        return x == "True"
    if name == "confirmation_tests":
        return int(x)
    if name in ("raw_quality", "confirmation_quality", "combined_setup"):
        return None if x.lower() == "nan" else float(x)
    return x


def infer_grade(e):
    """Conservative reconstruction only; explicit Telegram grades are preferred."""
    if e.get("explicit_grade"):
        return e["explicit_grade"], False
    rq, cq, cs, ct = (
        e.get("raw_quality"),
        e.get("confirmation_quality"),
        e.get("combined_setup"),
        e.get("confirmation_tests"),
    )
    if rq is None or cq is None or cs is None:
        return "CONFIRMED", True
    # These buckets are diagnostic approximations, not a replacement for the
    # full V10.x grade function (which also uses context/strength/volume fields).
    if rq >= 75 and cq >= 97 and cs >= 82 and (ct or 0) >= 3:
        return "A+", True
    if rq >= 65 and cq >= 95 and cs >= 76 and (ct or 0) >= 2:
        return "A", True
    if rq >= 58 and cq >= 90 and cs >= 70 and (ct or 0) >= 2:
        return "B+", True
    return "CONFIRMED", True


def segment_message(message):
    """Split a possibly interleaved stdout message into symbol-centred chunks."""
    text = str(message or "")
    anchors = list(ANCHOR.finditer(text))
    if not anchors:
        return []
    out = []
    for i, a in enumerate(anchors):
        # Ignore words such as 'symbols' that are not followed by Tide fields.
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        chunk = text[a.start():end]
        if not any(k in chunk for k in ("confirmed=", "score_confirmed=", "raw_signal=", "TIDE GRADE:")):
            continue
        out.append((a, chunk))
    return out


def parse_events(rows, start, end):
    diagnostics = Counter()
    candidates = []

    # Primary parser: preserve Railway row timestamps so glued output does not
    # corrupt chronological ordering.
    ordered = sorted(rows, key=lambda x: str(x.get("timestamp", "")))
    for row in ordered:
        msg = str(row.get("message", ""))
        diagnostics["rows"] += 1
        diagnostics["literal_confirmed_true"] += msg.count("confirmed=True")
        diagnostics["literal_score_confirmed_true"] += msg.count("score_confirmed=True")
        diagnostics["literal_raw_signal_true"] += msg.count("raw_signal=True")
        diagnostics["literal_tide_grade"] += msg.count("TIDE GRADE:")
        diagnostics["literal_tide_signal"] += msg.count("Tide SIGNAL:")

        segments = segment_message(msg)
        diagnostics["segments"] += len(segments)
        for a, chunk in segments:
            internal_ts = a.group("ts")
            try:
                event_time = dt(internal_ts) if internal_ts else dt(row.get("timestamp"))
            except Exception:
                diagnostics["bad_timestamp"] += 1
                continue
            if not (start <= event_time < end):
                continue

            e = {"timestamp": event_time, "symbol": a.group("symbol")}
            for k in FIELD_PATTERNS:
                e[k] = read_field(k, chunk)
            gm = GRADE_RE.search(chunk)
            e["explicit_grade"] = gm.group(1) if gm else None
            sm = SIDE_RE.search(chunk)
            if sm:
                e["side"] = "LONG" if sm.group(1).upper() in ("LONG", "BUY") else "SHORT"
            else:
                # V10.x Tide is explicitly a long-rebound strategy. Regime=downtrend
                # is context for buying a reclaim; it does NOT mean SHORT.
                e["side"] = "LONG"

            # Correct production-event definition from crypto_tide_engine_v10_9:
            # confirmed = candle confirm flag; score_confirmed = latest['signal'].
            # We only forward-test when BOTH are true.
            if e.get("confirmed") is True and e.get("score_confirmed") is True:
                e["grade"], e["grade_inferred"] = infer_grade(e)
                candidates.append(e)
                diagnostics["qualified_events_before_dedupe"] += 1

    # Fallback: some stdout fragments lose the first row timestamp/symbol due to
    # interleaving. Concatenate neighbouring messages and re-run segmentation.
    # Only add events not already recovered above.
    glued = "\n".join(str(x.get("message", "")) for x in ordered)
    for a, chunk in segment_message(glued):
        internal_ts = a.group("ts")
        if not internal_ts:
            continue
        try:
            event_time = dt(internal_ts)
        except Exception:
            continue
        if not (start <= event_time < end):
            continue
        if read_field("confirmed", chunk) is not True or read_field("score_confirmed", chunk) is not True:
            continue
        e = {"timestamp": event_time, "symbol": a.group("symbol")}
        for k in FIELD_PATTERNS:
            e[k] = read_field(k, chunk)
        gm = GRADE_RE.search(chunk)
        e["explicit_grade"] = gm.group(1) if gm else None
        sm = SIDE_RE.search(chunk)
        e["side"] = "LONG" if not sm or sm.group(1).upper() in ("LONG", "BUY") else "SHORT"
        e["grade"], e["grade_inferred"] = infer_grade(e)
        candidates.append(e)
        diagnostics["fallback_events_before_dedupe"] += 1

    # Restart/replica overlaps can duplicate the same signal. One signal per
    # symbol per 15m candle is the correct forward-test unit.
    uniq = {}
    for e in candidates:
        candle = e["timestamp"].replace(
            minute=(e["timestamp"].minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        key = (e["symbol"], candle)
        old = uniq.get(key)
        # Prefer the richer event if duplicate rows differ in completeness.
        richness = sum(v is not None for v in e.values())
        old_richness = sum(v is not None for v in old.values()) if old else -1
        if richness > old_richness:
            uniq[key] = e
    diagnostics["unique_events"] = len(uniq)
    return sorted(uniq.values(), key=lambda x: x["timestamp"]), diagnostics


def bybit_15m(symbol, start, hours=8):
    s = int((start - timedelta(minutes=30)).timestamp() * 1000)
    e = int((start + timedelta(hours=hours)).timestamp() * 1000)
    r = requests.get(
        BYBIT_KLINE,
        params={
            "category": "linear",
            "symbol": symbol,
            "interval": "15",
            "start": s,
            "end": e,
            "limit": 1000,
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("retCode") != 0:
        raise RuntimeError(j)
    df = pd.DataFrame(j["result"]["list"], columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    if df.empty:
        return df
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    df["time"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def measure(e):
    df = bybit_15m(e["symbol"], e["timestamp"])
    after = df[df.time >= pd.Timestamp(e["timestamp"])].copy()
    if after.empty:
        return e
    entry = float(after.iloc[0].open)
    sign = -1 if e["side"] == "SHORT" else 1
    e["entry_price"] = entry
    for h in (1, 4, 6):
        x = after[after.time <= pd.Timestamp(e["timestamp"] + timedelta(hours=h))]
        if x.empty:
            continue
        e[f"ret_{h}h"] = sign * (float(x.iloc[-1].close) / entry - 1)
        if sign == 1:
            e[f"mfe_{h}h"] = float(x.high.max() / entry - 1)
            e[f"mae_{h}h"] = float(x.low.min() / entry - 1)
        else:
            e[f"mfe_{h}h"] = float(1 - x.low.min() / entry)
            e[f"mae_{h}h"] = float(1 - x.high.max() / entry)
    return e


def pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{100*x:+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=datetime.now(timezone.utc).strftime("%Y-%m"))
    ap.add_argument("--out", default="reports")
    ap.add_argument("--service-id", default=os.getenv("RAILWAY_SERVICE_ID"))
    args = ap.parse_args()

    token = os.getenv("RAILWAY_TOKEN")
    if not token:
        raise SystemExit("Missing RAILWAY_TOKEN")

    start, end = month_bounds(args.month)
    end = min(end, datetime.now(timezone.utc))
    ctx = token_context(token)
    project_name, services = project_services(token, ctx["projectId"])
    print("Railway project:", project_name)
    print("Services visible to token:", [(s["name"], s["id"]) for s in services])

    if args.service_id:
        targets = [s for s in services if s["id"] == args.service_id]
        if not targets:
            raise RuntimeError(f"RAILWAY_SERVICE_ID {args.service_id} not visible; services={services}")
    else:
        targets = [s for s in services if "tide" in s["name"].lower()] or services

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"railway_logs_{args.month}.jsonl"

    allrows = []
    dep_count = ok = failed = truncated = 0
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for svc in targets:
            deps = deployments(token, ctx["projectId"], svc["id"], ctx["environmentId"])
            chosen = [d for d in deps if dt(d["createdAt"]) < end]
            dep_count += len(chosen)
            print(f"Service {svc['name']} ({svc['id']}): {len(deps)} deployments returned; inspecting {len(chosen)}")
            for n, dep in enumerate(chosen, 1):
                try:
                    rows = deployment_logs(token, dep["id"], 1000)
                    ok += 1
                    if len(rows) >= 1000:
                        truncated += 1
                    for row in rows:
                        enriched = dict(row)
                        enriched["deployment_id"] = dep["id"]
                        enriched["deployment_status"] = dep["status"]
                        enriched["service_name"] = svc["name"]
                        raw_file.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
                    allrows.extend(rows)
                    print(svc["name"], n, dep["id"], dep["status"], len(rows))
                except Exception as ex:
                    failed += 1
                    print("WARN", svc["name"], dep["id"], dep["status"], ex)

    events, diag = parse_events(allrows, start, end)
    print("Log fetch success/fail:", ok, failed)
    print("Parser diagnostics:", dict(diag))
    print("Confirmed production events:", len(events))

    measured = []
    for i, event in enumerate(events, 1):
        try:
            measured.append(measure(event))
            print(f"market {i}/{len(events)} {event['symbol']}")
        except Exception as ex:
            print("MARKET WARN", event["symbol"], ex)
            measured.append(event)
        time.sleep(0.05)

    df = pd.DataFrame(measured)
    if not df.empty:
        df["timestamp"] = df["timestamp"].astype(str)
        df.to_csv(out / f"tide_live_signals_{args.month}.csv", index=False)

    diag_payload = {
        "month": args.month,
        "window_utc": [start.isoformat(), end.isoformat()],
        "services": [s["name"] for s in targets],
        "deployments_inspected": dep_count,
        "deployment_log_fetches_successful": ok,
        "deployment_log_fetches_failed": failed,
        "deployments_at_1000_row_cap": truncated,
        "log_rows_fetched": len(allrows),
        "parser": dict(diag),
    }
    (out / f"parser_diagnostics_{args.month}.json").write_text(
        json.dumps(diag_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Tide live forward-test — {args.month}",
        "",
        "Source: actual Railway deployment logs; market outcomes: Bybit 15m candles.",
        f"Window (UTC): {start.isoformat()} to {end.isoformat()}",
        f"Services inspected: {', '.join(s['name'] for s in targets)}",
        f"Deployments inspected: {dep_count}",
        f"Deployment log fetches successful: {ok}",
        f"Deployment log fetches failed: {failed}",
        f"Deployments at 1000-row cap: {truncated}",
        f"Log rows fetched: {len(allrows)}",
        f"Raw `confirmed=True` occurrences: {diag['literal_confirmed_true']}",
        f"Raw `score_confirmed=True` occurrences: {diag['literal_score_confirmed_true']}",
        f"Raw `raw_signal=True` occurrences: {diag['literal_raw_signal_true']}",
        f"Unique confirmed production signals: {len(events)}",
        "",
    ]

    if df.empty:
        lines += [
            "No closed-candle `score_confirmed=True` production signal was reconstructed.",
            "See the saved raw Railway JSONL and parser diagnostics in this artifact before changing the model.",
        ]
    else:
        lines += ["## Headline", ""]
        for h in (1, 4, 6):
            c = f"ret_{h}h"
            x = pd.to_numeric(df[c], errors="coerce").dropna() if c in df else pd.Series(dtype=float)
            if len(x):
                lines += [
                    f"- {h}h: n={len(x)}, mean={pct(x.mean())}, median={pct(x.median())}, win rate={(x > 0).mean() * 100:.1f}%"
                ]
        lines += ["", "## By reconstructed grade", ""]
        for grade, g in df.groupby("grade", dropna=False):
            x = pd.to_numeric(g["ret_4h"], errors="coerce").dropna() if "ret_4h" in g else pd.Series(dtype=float)
            if len(x):
                lines += [f"- {grade}: signals={len(g)}, 4h mean={pct(x.mean())}, 4h win={(x > 0).mean() * 100:.1f}%"]
            else:
                lines += [f"- {grade}: signals={len(g)}, 4h unavailable"]
        lines += [
            "",
            "## Interpretation note",
            "",
            "`confirmed` is the Bybit candle-close flag. `score_confirmed` is the model's final `latest['signal']` flag. A forward-test event requires both to be True. V10.x is a long-rebound strategy; downtrend regime is context, not a SHORT direction. Reconstructed grades are approximate unless an explicit Telegram grade string is present in the retained log chunk.",
        ]

    (out / f"tide_monthly_summary_{args.month}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Saved to", out)


if __name__ == "__main__":
    main()
