import json, os, sys, time, threading
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from pybit.unified_trading import HTTP, WebSocket

# Universe / screening
MIN_LISTING_DAYS = 90
MIN_TURNOVER_24H = 5_000_000
MAX_CANDIDATES = 60
TOP_N = 10
ALWAYS_INCLUDE = ["BTCUSDT", "ETHUSDT"]
BACKTEST_DAYS_15M = 60
BACKTEST_DAYS_1H = 90
MIN_TRADES = 10
MAX_DRAWDOWN = -0.15

# Tide model
LOOKBACK = 24
VOL_LOOKBACK = 24
VOL_MULT = 1.5
WICK_MIN = 0.35
HOLD_BARS = 24
COOLDOWN_BARS = 24
COST = 0.001
PRE_MINUTES = 3
RESCAN_HOURS = 24

RESULTS_CSV = Path("universe_backtest_results.csv")
SELECTED_JSON = Path("selected_symbols.json")
STATE_FILE = Path("alert_state.json")
EXCLUDED_BASES = {"USDC","USDE","DAI","FDUSD","TUSD","PYUSD","USDD","USD","USDT"}

http = HTTP(testnet=False)
market_data = {}
data_lock = threading.Lock()
state_lock = threading.Lock()


def log(*x):
    print(datetime.now(timezone.utc).isoformat(), *x, flush=True)


def send_tg(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log("Telegram variables missing")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text}, timeout=20)
        log("Telegram", r.status_code, r.text[:160])
    except Exception as e:
        log("Telegram error", repr(e))


def load_json(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


alert_state = load_json(STATE_FILE, {})


def get_universe():
    items, cursor = [], ""
    while True:
        kw = {"category":"linear", "status":"Trading", "limit":1000}
        if cursor: kw["cursor"] = cursor
        result = http.get_instruments_info(**kw)["result"]
        items += result.get("list", [])
        cursor = result.get("nextPageCursor", "")
        if not cursor: break
        time.sleep(0.15)

    now_ms = int(time.time()*1000)
    out = []
    for x in items:
        launch = int(x.get("launchTime") or 0)
        age = (now_ms-launch)/86_400_000 if launch else 0
        if (x.get("quoteCoin") == "USDT" and x.get("settleCoin") == "USDT"
            and x.get("contractType") == "LinearPerpetual"
            and x.get("baseCoin") not in EXCLUDED_BASES
            and age >= MIN_LISTING_DAYS):
            out.append({"symbol":x["symbol"], "age_days":age})
    return out


def get_turnovers():
    rows = http.get_tickers(category="linear")["result"].get("list", [])
    return {x["symbol"]: float(x.get("turnover24h") or 0) for x in rows}


def fetch_klines(symbol, interval, days):
    minutes = int(interval)
    need = int(days*24*60/minutes)
    rows, end_ms = [], None
    while len(rows) < need:
        limit = min(1000, need-len(rows))
        kw = {"category":"linear", "symbol":symbol, "interval":interval, "limit":limit}
        if end_ms is not None: kw["end"] = end_ms
        batch = http.get_kline(**kw)["result"].get("list", [])
        if not batch: break
        rows += batch
        end_ms = min(int(r[0]) for r in batch)-1
        if len(batch) < limit: break
        time.sleep(0.08)
    if not rows: raise RuntimeError(f"No klines {symbol} {interval}")
    df = pd.DataFrame([{
        "open_time":pd.to_datetime(int(r[0]),unit="ms",utc=True),
        "open":float(r[1]),"high":float(r[2]),"low":float(r[3]),
        "close":float(r[4]),"volume":float(r[5]),"turnover":float(r[6])
    } for r in rows])
    return df.sort_values("open_time").drop_duplicates("open_time").tail(need).reset_index(drop=True)


def model_frame(df15, df1h):
    df1h = df1h.copy(); df15 = df15.copy()
    df1h["ma50"] = df1h.close.rolling(50).mean()
    df1h["ma200"] = df1h.close.rolling(200).mean()
    df1h["regime"] = np.where(df1h.ma50 < df1h.ma200, "downtrend",
                        np.where(df1h.ma50 > df1h.ma200, "uptrend", "range"))
    df = pd.merge_asof(df15.sort_values("open_time"),
        df1h[["open_time","regime","ma50","ma200"]].sort_values("open_time"),
        on="open_time", direction="backward")
    df["rolling_low"] = df.low.rolling(LOOKBACK).min().shift(1)
    df["rolling_high"] = df.high.rolling(LOOKBACK).max().shift(1)
    df["avg_vol"] = df.volume.rolling(VOL_LOOKBACK).mean().shift(1)
    df["vol_multiple"] = np.where(df.avg_vol>0, df.volume/df.avg_vol, 0)
    df["wick"] = df[["open","close"]].min(axis=1)-df.low
    df["range"] = df.high-df.low
    df["wick_ratio"] = np.where(df.range>0, df.wick/df.range, 0)
    df["raw_signal"] = ((df.regime=="downtrend") & (df.low<df.rolling_low)
        & (df.close>df.rolling_low) & (df.wick_ratio>WICK_MIN)
        & (df.vol_multiple>VOL_MULT))
    accepted = np.zeros(len(df), dtype=bool); last = -10**9
    for i in np.flatnonzero(df.raw_signal.to_numpy()):
        if i-last >= COOLDOWN_BARS:
            accepted[i] = True; last = i
    df["signal"] = accepted
    return df


def backtest(symbol, turnover, age):
    df = model_frame(fetch_klines(symbol,"15",BACKTEST_DAYS_15M),
                     fetch_klines(symbol,"60",BACKTEST_DAYS_1H))
    entries = np.flatnonzero(df.signal.to_numpy())
    trades=[]
    for i in entries:
        j=i+HOLD_BARS
        if j>=len(df): continue
        trades.append((df.iloc[i].open_time, df.iloc[j].close/df.iloc[i].close-1-COST))
    rets=np.array([x[1] for x in trades],dtype=float)
    if len(rets):
        eq=np.cumprod(1+rets); peak=np.maximum.accumulate(eq); dd=eq/peak-1
        total=float(eq[-1]-1); avg=float(rets.mean()); med=float(np.median(rets));
        win=float((rets>0).mean()); maxdd=float(dd.min())
    else:
        total=avg=med=win=0.0; maxdd=0.0
    mid=df.open_time.iloc[len(df)//2]
    first=np.array([r for t,r in trades if t<mid]); second=np.array([r for t,r in trades if t>=mid])
    first_ret=float(np.prod(1+first)-1) if len(first) else 0.0
    second_ret=float(np.prod(1+second)-1) if len(second) else 0.0
    return {"symbol":symbol,"turnover24h":turnover,"age_days":age,"trades":len(rets),
            "total_return":total,"average_return":avg,"median_return":med,
            "win_rate":win,"max_drawdown":maxdd,
            "first_half_return":first_ret,"second_half_return":second_ret}


def scan_select():
    universe=pd.DataFrame(get_universe()); turns=get_turnovers()
    universe["turnover24h"]=universe.symbol.map(turns).fillna(0)
    candidates=universe[universe.turnover24h>=MIN_TURNOVER_24H].nlargest(MAX_CANDIDATES,"turnover24h")
    log("Backtesting",len(candidates),"of",len(universe),"age-qualified contracts")
    rows=[]
    for n,x in enumerate(candidates.itertuples(index=False),1):
        try:
            log(f"[{n}/{len(candidates)}]",x.symbol)
            rows.append(backtest(x.symbol,float(x.turnover24h),float(x.age_days)))
        except Exception as e:
            log("Backtest failed",x.symbol,repr(e))
        time.sleep(0.12)
    if not rows: raise RuntimeError("No completed backtests")
    r=pd.DataFrame(rows)
    r["eligible"]=(r.trades>=MIN_TRADES)&(r.total_return>0)&(r.median_return>0)&(r.max_drawdown>=MAX_DRAWDOWN)&(r.first_half_return>0)&(r.second_half_return>0)
    r["score"]=np.nan
    p=r[r.eligible]
    if len(p):
        r.loc[p.index,"score"]=(.30*p.total_return.rank(pct=True)+.20*p.median_return.rank(pct=True)+
            .15*p.win_rate.rank(pct=True)+.15*p.max_drawdown.rank(pct=True)+
            .15*np.minimum(p.first_half_return,p.second_half_return).rank(pct=True)+
            .05*p.turnover24h.rank(pct=True))
    r=r.sort_values(["eligible","score","total_return"],ascending=[False,False,False],na_position="last")
    r.to_csv(RESULTS_CSV,index=False)
    selected=r.loc[r.eligible,"symbol"].head(TOP_N).tolist()
    for s in reversed(ALWAYS_INCLUDE):
        if s in r.symbol.values and s not in selected: selected.insert(0,s)
    selected=list(dict.fromkeys(selected))[:TOP_N+len(ALWAYS_INCLUDE)]
    save_json(SELECTED_JSON,{"generated_at":datetime.now(timezone.utc).isoformat(),"symbols":selected})
    send_tg("✅ Tide universe scan complete\n\nSelected:\n"+"\n".join(f"{i+1}. {s}" for i,s in enumerate(selected))+"\n\nForward-test before increasing size.")
    return selected


def init_live(symbols):
    for n,s in enumerate(symbols,1):
        log("Loading live",n,"/",len(symbols),s)
        market_data[s]={"15":fetch_klines(s,"15",4),"60":fetch_klines(s,"60",14)}


def update_candle(symbol, interval, c):
    row={"open_time":pd.to_datetime(int(c["start"]),unit="ms",utc=True),
         "open":float(c["open"]),"high":float(c["high"]),"low":float(c["low"]),
         "close":float(c["close"]),"volume":float(c["volume"]),"turnover":float(c.get("turnover") or 0)}
    with data_lock:
        df=market_data[symbol][interval]
        df=df[df.open_time!=row["open_time"]]
        market_data[symbol][interval]=pd.concat([df,pd.DataFrame([row])],ignore_index=True).sort_values("open_time").drop_duplicates("open_time").tail(500).reset_index(drop=True)


def duplicate(symbol, kind, key):
    with state_lock:
        st=alert_state.setdefault(symbol,{})
        if st.get(kind)==key: return True
        st[kind]=key; save_json(STATE_FILE,alert_state); return False


def live_check(symbol, confirmed):
    with data_lock:
        df=model_frame(market_data[symbol]["15"].copy(),market_data[symbol]["60"].copy())
    x=df.iloc[-1]; close_time=x.open_time+pd.Timedelta(minutes=15)
    mins=(close_time-pd.Timestamp.now(tz="UTC")).total_seconds()/60
    if not bool(x.raw_signal): return
    kind="confirmed" if confirmed else "pre"
    if not confirmed and not (0<=mins<=PRE_MINUTES): return
    key=x.open_time.isoformat()
    if duplicate(symbol,kind,key): return
    title="🚨 CONFIRMED" if confirmed else "⚠️ PRE-SIGNAL"
    send_tg(f"{title}: {symbol}\n\nCandle close UTC: {close_time}\nPrice: {x.close:.8g}\nRegime: {x.regime}\nRolling low: {x.rolling_low:.8g}\nRolling high: {x.rolling_high:.8g}\nWick ratio: {x.wick_ratio:.4f}\nVolume multiple: {x.vol_multiple:.2f}\n\nResearch exit: fixed 6h. Do not chase late alerts.")


def callback_for(symbol, interval):
    def cb(msg):
        try:
            for c in msg.get("data",[]):
                confirmed=bool(c.get("confirm",False)); update_candle(symbol,interval,c)
                if interval=="15": live_check(symbol,confirmed)
        except Exception as e: log("Callback error",symbol,interval,repr(e))
    return cb


def monitor(symbols):
    init_live(symbols)
    ws=WebSocket(testnet=False,channel_type="linear")
    for s in symbols:
        ws.kline_stream(interval=15,symbol=s,callback=callback_for(s,"15"))
        ws.kline_stream(interval=60,symbol=s,callback=callback_for(s,"60"))
    send_tg("✅ Tide universe monitor started\n\n"+"\n".join(symbols))
    started=time.monotonic()
    while True:
        if (time.monotonic()-started)/3600>=RESCAN_HOURS:
            log("Daily rescan restart")
            os.execv(sys.executable,[sys.executable,*sys.argv])
        log("Heartbeat monitoring",len(symbols),"symbols")
        time.sleep(60)


def main():
    selected=scan_select()
    if not selected: raise RuntimeError("No symbols selected")
    monitor(selected)

if __name__=="__main__": main()
