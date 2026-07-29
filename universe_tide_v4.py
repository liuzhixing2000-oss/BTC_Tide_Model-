import json
import os
import sys
import time
import threading
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from pybit.unified_trading import HTTP, WebSocket

# ============================================================
# Tide Universe V4 — two-stage robust screening
# ============================================================

# ---------- Universe ----------
MIN_LISTING_DAYS = 120
MIN_TURNOVER_24H = 10_000_000
MAX_STAGE1_CANDIDATES = 60
STAGE2_CANDIDATES = 20
TOP_N_TO_MONITOR = 8
BENCHMARKS = ["BTCUSDT", "ETHUSDT"]
MONITOR_FAILED_BENCHMARKS = False

# ---------- Backtest windows ----------
STAGE1_DAYS_15M = 60
STAGE1_DAYS_1H = 90
STAGE2_DAYS_15M = 180
STAGE2_DAYS_1H = 210
VALIDATION_DAYS = 45

# ---------- Tide entry model ----------
LOOKBACK_BARS = 24
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35
HOLD_BARS = 24
COOLDOWN_BARS = 24
FEE_SLIPPAGE = 0.001

# ---------- Robust selection ----------
MIN_STAGE2_TRADES = 20
MIN_VALIDATION_TRADES = 5
MAX_ALLOWED_DRAWDOWN = -0.12

# ---------- Realtime ----------
PRE_ALERT_WINDOW_MINUTES = 3
RESCAN_HOURS = 168  # weekly; reduces Railway cost and selection churn

# ---------- Output ----------
STAGE1_CSV = Path("stage1_results.csv")
STAGE2_CSV = Path("stage2_full_results.csv")
SELECTED_JSON = Path("selected_symbols.json")
STATE_FILE = Path("alert_state.json")

EXCLUDED_BASES = {
    "USDC", "USDE", "DAI", "FDUSD", "TUSD", "PYUSD", "USDD",
    "USD", "USDT", "EUR",
}
EXCLUDED_SYMBOLS = {
    # Tokenised equities / TradFi contracts
    "AAPLUSDT", "TSLAUSDT", "NVDAUSDT", "AMZNUSDT", "METAUSDT",
    "GOOGLUSDT", "MSFTUSDT",
}

http = HTTP(testnet=False)
market_data: dict[str, dict[str, pd.DataFrame]] = {}
data_lock = threading.Lock()
state_lock = threading.Lock()


def log(*parts) -> None:
    print(datetime.now(timezone.utc).isoformat(), *parts, flush=True)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


alert_state = load_json(STATE_FILE, {})


# ============================================================
# Telegram
# ============================================================

def telegram_credentials() -> tuple[str, str]:
    return (
        os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )


def send_tg(text: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        log("Telegram variables missing")
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        log("Telegram", response.status_code, response.text[:180])
    except Exception as exc:
        log("Telegram error", repr(exc))


def send_tg_photo(image: BytesIO, caption: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        return
    image.seek(0)
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("tide_v4_ranking.png", image, "image/png")},
            timeout=60,
        )
        log("Telegram photo", response.status_code, response.text[:180])
    except Exception as exc:
        log("Telegram photo error", repr(exc))


def send_tg_document(path: Path, caption: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id or not path.exists():
        return
    try:
        with path.open("rb") as file:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (path.name, file, "text/csv")},
                timeout=90,
            )
        log("Telegram document", response.status_code, response.text[:180])
    except Exception as exc:
        log("Telegram document error", repr(exc))


# ============================================================
# Bybit REST
# ============================================================

def get_universe() -> pd.DataFrame:
    rows, cursor = [], ""
    while True:
        kwargs = {"category": "linear", "status": "Trading", "limit": 1000}
        if cursor:
            kwargs["cursor"] = cursor
        result = http.get_instruments_info(**kwargs)["result"]
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
        time.sleep(0.15)

    ticker_rows = http.get_tickers(category="linear")["result"].get("list", [])
    turnovers = {
        item["symbol"]: float(item.get("turnover24h") or 0)
        for item in ticker_rows
    }

    now_ms = int(time.time() * 1000)
    output = []
    for item in rows:
        symbol = item.get("symbol", "")
        base = item.get("baseCoin", "")
        launch = int(item.get("launchTime") or 0)
        age_days = (now_ms - launch) / 86_400_000 if launch else 0

        if item.get("quoteCoin") != "USDT":
            continue
        if item.get("settleCoin") != "USDT":
            continue
        if item.get("contractType") != "LinearPerpetual":
            continue
        if base in EXCLUDED_BASES or symbol in EXCLUDED_SYMBOLS:
            continue
        if age_days < MIN_LISTING_DAYS:
            continue
        turnover = turnovers.get(symbol, 0.0)
        if turnover < MIN_TURNOVER_24H:
            continue

        output.append({
            "symbol": symbol,
            "age_days": age_days,
            "turnover24h": turnover,
        })

    return pd.DataFrame(output).sort_values("turnover24h", ascending=False)


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    interval_minutes = int(interval)
    required = int(days * 24 * 60 / interval_minutes)
    rows: list[list[str]] = []
    end_ms = None

    while len(rows) < required:
        limit = min(1000, required - len(rows))
        kwargs = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if end_ms is not None:
            kwargs["end"] = end_ms

        batch = http.get_kline(**kwargs)["result"].get("list", [])
        if not batch:
            break
        rows.extend(batch)
        end_ms = min(int(row[0]) for row in batch) - 1
        if len(batch) < limit:
            break
        time.sleep(0.08)

    if not rows:
        raise RuntimeError(f"No kline data for {symbol} {interval}")

    frame = pd.DataFrame([{
        "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "turnover": float(row[6]),
    } for row in rows])

    return (
        frame.sort_values("open_time")
        .drop_duplicates("open_time")
        .tail(required)
        .reset_index(drop=True)
    )


# ============================================================
# Model / backtest
# ============================================================

def model_frame(df15: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    df15 = df15.copy()
    df1h = df1h.copy()

    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()
    df1h["regime"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(df1h["ma50"] > df1h["ma200"], "uptrend", "range"),
    )

    df = pd.merge_asof(
        df15.sort_values("open_time"),
        df1h[["open_time", "regime", "ma50", "ma200"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    df["rolling_low"] = df["low"].rolling(LOOKBACK_BARS).min().shift(1)
    df["rolling_high"] = df["high"].rolling(LOOKBACK_BARS).max().shift(1)
    df["avg_volume"] = df["volume"].rolling(VOLUME_LOOKBACK).mean().shift(1)
    df["volume_multiple"] = np.where(
        df["avg_volume"] > 0,
        df["volume"] / df["avg_volume"],
        0,
    )
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]
    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] / df["candle_range"],
        0,
    )

    df["raw_signal"] = (
        (df["regime"] == "downtrend")
        & (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & (df["volume_multiple"] > VOLUME_MULTIPLIER)
    )

    # Live-compatible cooldown. No future knowledge is used.
    accepted = np.zeros(len(df), dtype=bool)
    last_entry = -10**9
    for idx in np.flatnonzero(df["raw_signal"].to_numpy()):
        if idx - last_entry >= COOLDOWN_BARS:
            accepted[idx] = True
            last_entry = idx
    df["signal"] = accepted
    return df


def trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    for entry_idx in np.flatnonzero(df["signal"].to_numpy()):
        exit_idx = entry_idx + HOLD_BARS
        if exit_idx >= len(df):
            continue
        entry_price = float(df.iloc[entry_idx]["close"])
        exit_price = float(df.iloc[exit_idx]["close"])
        trades.append({
            "entry_time": df.iloc[entry_idx]["open_time"],
            "net_return": exit_price / entry_price - 1 - FEE_SLIPPAGE,
        })
    return pd.DataFrame(trades)


def metrics(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {
            "trades": 0,
            "total_return": 0.0,
            "average_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
        }
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    return {
        "trades": int(len(returns)),
        "total_return": float(equity[-1] - 1),
        "average_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.min(drawdown)),
    }


def backtest_symbol(symbol: str, turnover: float, age_days: float, days15: int, days1h: int) -> dict:
    df = model_frame(
        fetch_klines(symbol, "15", days15),
        fetch_klines(symbol, "60", days1h),
    )
    trades = trade_frame(df)
    all_returns = trades["net_return"].to_numpy(dtype=float) if not trades.empty else np.array([])
    result = metrics(all_returns)

    validation_start = df["open_time"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    if trades.empty:
        train_returns = np.array([])
        validation_returns = np.array([])
    else:
        train_returns = trades.loc[
            trades["entry_time"] < validation_start,
            "net_return",
        ].to_numpy(dtype=float)
        validation_returns = trades.loc[
            trades["entry_time"] >= validation_start,
            "net_return",
        ].to_numpy(dtype=float)

    train = metrics(train_returns)
    validation = metrics(validation_returns)

    return {
        "symbol": symbol,
        "turnover24h": turnover,
        "age_days": age_days,
        **result,
        "train_trades": train["trades"],
        "train_return": train["total_return"],
        "validation_trades": validation["trades"],
        "validation_return": validation["total_return"],
        "validation_median": validation["median_return"],
        "validation_win_rate": validation["win_rate"],
    }


def stage1_score(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["stage1_score"] = (
        0.45 * results["total_return"].rank(pct=True)
        + 0.20 * results["median_return"].fillna(-999).rank(pct=True)
        + 0.15 * results["win_rate"].fillna(0).rank(pct=True)
        + 0.10 * results["max_drawdown"].rank(pct=True)
        + 0.10 * results["turnover24h"].rank(pct=True)
    )
    return results.sort_values("stage1_score", ascending=False)


def robust_stage2_score(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    eligible = (
        (results["trades"] >= MIN_STAGE2_TRADES)
        & (results["total_return"] > 0)
        & (results["median_return"] > 0)
        & (results["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN)
        & (results["train_return"] > 0)
        & (results["validation_trades"] >= MIN_VALIDATION_TRADES)
        & (results["validation_return"] > 0)
        & (results["validation_median"] > 0)
    )
    results["eligible"] = eligible
    results["score"] = np.nan

    pool = results.loc[eligible].copy()
    if not pool.empty:
        score = (
            0.20 * pool["total_return"].rank(pct=True)
            + 0.30 * pool["validation_return"].rank(pct=True)
            + 0.15 * pool["validation_median"].rank(pct=True)
            + 0.10 * pool["win_rate"].rank(pct=True)
            + 0.10 * pool["max_drawdown"].rank(pct=True)
            + 0.10 * pool["trades"].clip(upper=60).rank(pct=True)
            + 0.05 * pool["turnover24h"].rank(pct=True)
        )
        results.loc[pool.index, "score"] = score

    return results.sort_values(
        ["eligible", "score", "validation_return", "total_return"],
        ascending=[False, False, False, False],
        na_position="last",
    )


# ============================================================
# Visual report
# ============================================================

def ranking_chart(results: pd.DataFrame, top_n: int = 12) -> BytesIO:
    ranked = results.head(top_n).copy().iloc[::-1]
    labels = ranked["symbol"].tolist()
    total_pct = (ranked["total_return"] * 100).tolist()
    validation_pct = (ranked["validation_return"] * 100).tolist()

    fig = Figure(figsize=(10, 7), dpi=150)
    ax = fig.subplots()
    y = np.arange(len(labels))
    height = 0.36
    ax.barh(y - height / 2, total_pct, height=height, label="180-day total")
    ax.barh(y + height / 2, validation_pct, height=height, label=f"Last {VALIDATION_DAYS} days")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Backtest return (%)")
    ax.set_title("Tide Universe V4 — full-model stage-2 ranking")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer


def send_report(results: pd.DataFrame, selected: list[str]) -> None:
    try:
        send_tg_photo(
            ranking_chart(results),
            "V4 stage-2 results: 180-day total return versus the final validation period.",
        )
    except Exception as exc:
        log("Chart failed", repr(exc))

    send_tg_document(STAGE2_CSV, "Full V4 stage-2 backtest results")

    lines = ["🏆 Tide Universe V4 selected symbols", ""]
    for row in results[results["symbol"].isin(selected)].itertuples(index=False):
        score = "n/a" if pd.isna(row.score) else f"{row.score:.3f}"
        lines.extend([
            row.symbol,
            f"180d return: {row.total_return:.1%}",
            f"Validation return: {row.validation_return:.1%}",
            f"Trades: {row.trades} ({row.validation_trades} validation)",
            f"Win rate: {row.win_rate:.1%}",
            f"Median trade: {row.median_return:.2%}",
            f"Max DD: {row.max_drawdown:.1%}",
            f"Score: {score}",
            "",
        ])
    text = "\n".join(lines)
    for start in range(0, len(text), 3500):
        send_tg(text[start:start + 3500])


# ============================================================
# Two-stage scan
# ============================================================

def scan_and_select() -> tuple[list[str], pd.DataFrame]:
    universe = get_universe().head(MAX_STAGE1_CANDIDATES)
    log("Stage 1 candidates:", len(universe))

    stage1_rows = []
    for idx, row in enumerate(universe.itertuples(index=False), start=1):
        try:
            log(f"Stage1 [{idx}/{len(universe)}] {row.symbol}")
            stage1_rows.append(backtest_symbol(
                row.symbol,
                float(row.turnover24h),
                float(row.age_days),
                STAGE1_DAYS_15M,
                STAGE1_DAYS_1H,
            ))
        except Exception as exc:
            log("Stage1 failed", row.symbol, repr(exc))

    stage1 = stage1_score(pd.DataFrame(stage1_rows))
    stage1.to_csv(STAGE1_CSV, index=False)

    stage2_symbols = stage1.head(STAGE2_CANDIDATES)["symbol"].tolist()
    for benchmark in BENCHMARKS:
        if benchmark in universe["symbol"].values and benchmark not in stage2_symbols:
            stage2_symbols.append(benchmark)

    lookup = universe.set_index("symbol")
    stage2_rows = []
    for idx, symbol in enumerate(stage2_symbols, start=1):
        try:
            log(f"Stage2 [{idx}/{len(stage2_symbols)}] {symbol}")
            stage2_rows.append(backtest_symbol(
                symbol,
                float(lookup.loc[symbol, "turnover24h"]),
                float(lookup.loc[symbol, "age_days"]),
                STAGE2_DAYS_15M,
                STAGE2_DAYS_1H,
            ))
        except Exception as exc:
            log("Stage2 failed", symbol, repr(exc))

    stage2 = robust_stage2_score(pd.DataFrame(stage2_rows))
    stage2.to_csv(STAGE2_CSV, index=False)

    selected = stage2.loc[stage2["eligible"], "symbol"].head(TOP_N_TO_MONITOR).tolist()
    if MONITOR_FAILED_BENCHMARKS:
        for benchmark in BENCHMARKS:
            if benchmark in stage2["symbol"].values and benchmark not in selected:
                selected.append(benchmark)

    save_json(SELECTED_JSON, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "stage2_symbols": stage2_symbols,
        "parameters": {
            "stage1_days": STAGE1_DAYS_15M,
            "stage2_days": STAGE2_DAYS_15M,
            "validation_days": VALIDATION_DAYS,
            "min_stage2_trades": MIN_STAGE2_TRADES,
        },
    })

    send_tg(
        "✅ Tide Universe V4 scan complete\n\n"
        f"Stage 1: {len(stage1)} coins / {STAGE1_DAYS_15M} days\n"
        f"Stage 2: {len(stage2)} coins / {STAGE2_DAYS_15M} days\n"
        f"Live selected: {len(selected)}\n\n"
        + ("\n".join(selected) if selected else "No coin passed the robust filter.")
    )
    send_report(stage2, selected)
    return selected, stage2


# ============================================================
# Realtime monitor
# ============================================================

def already_alerted(symbol: str, alert_type: str, signal_key: str) -> bool:
    with state_lock:
        symbol_state = alert_state.setdefault(symbol, {})
        if symbol_state.get(alert_type) == signal_key:
            return True
        symbol_state[alert_type] = signal_key
        save_json(STATE_FILE, alert_state)
        return False


def initialise_market_data(symbols: list[str]) -> None:
    for idx, symbol in enumerate(symbols, start=1):
        log(f"Loading live [{idx}/{len(symbols)}] {symbol}")
        market_data[symbol] = {
            "15": fetch_klines(symbol, "15", 4),
            "60": fetch_klines(symbol, "60", 14),
        }
        time.sleep(0.1)


def update_candle(symbol: str, interval: str, candle: dict) -> None:
    row = {
        "open_time": pd.to_datetime(int(candle["start"]), unit="ms", utc=True),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
        "turnover": float(candle.get("turnover") or 0),
    }
    with data_lock:
        df = market_data[symbol][interval]
        df = df[df["open_time"] != row["open_time"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        market_data[symbol][interval] = (
            df.sort_values("open_time")
            .drop_duplicates("open_time")
            .tail(500)
            .reset_index(drop=True)
        )


def send_live_alert(symbol: str, latest: pd.Series, alert_type: str, minutes_to_close: float) -> None:
    signal_key = latest["open_time"].isoformat()
    if already_alerted(symbol, alert_type, signal_key):
        return

    title = (
        f"⚠️ Tide PRE-SIGNAL: {symbol}"
        if alert_type == "pre"
        else f"🚨 Tide CONFIRMED SIGNAL: {symbol}"
    )
    note = (
        "The 15m candle is still open and the setup can disappear."
        if alert_type == "pre"
        else "The 15m candle closed and the setup remains valid."
    )
    candle_close = latest["open_time"] + pd.Timedelta(minutes=15)
    send_tg(f"""{title}

Candle close UTC: {candle_close}
Entry/current price: {latest['close']:.8g}
1h regime: {latest['regime']}
Rolling low: {latest['rolling_low']:.8g}
Rolling high: {latest['rolling_high']:.8g}
Signal low: {latest['low']:.8g}
Lower wick ratio: {latest['lower_wick_ratio']:.4f}
Volume multiple: {latest['volume_multiple']:.2f}
Minutes to close: {max(0, minutes_to_close):.1f}

Research exit: fixed 6 hours.
{note}
Do not chase delayed alerts.""")


def calculate_live_signal(symbol: str, confirmed: bool) -> None:
    with data_lock:
        df15 = market_data[symbol]["15"].copy()
        df1h = market_data[symbol]["60"].copy()
    df = model_frame(df15, df1h)
    latest = df.iloc[-1]
    candle_close = latest["open_time"] + pd.Timedelta(minutes=15)
    minutes_to_close = (candle_close - pd.Timestamp.now(tz="UTC")).total_seconds() / 60
    signal = bool(latest["raw_signal"])

    log(symbol, f"confirmed={confirmed}", f"regime={latest['regime']}", f"signal={signal}")
    if not signal:
        return
    if confirmed:
        send_live_alert(symbol, latest, "confirmed", 0.0)
    elif 0 <= minutes_to_close <= PRE_ALERT_WINDOW_MINUTES:
        send_live_alert(symbol, latest, "pre", minutes_to_close)


def make_callback(symbol: str, interval: str):
    def callback(message: dict) -> None:
        try:
            for candle in message.get("data", []):
                confirmed = bool(candle.get("confirm", False))
                update_candle(symbol, interval, candle)
                if interval == "15":
                    calculate_live_signal(symbol, confirmed)
                elif confirmed:
                    log("Closed 1h candle", symbol)
        except Exception as exc:
            log("Callback error", symbol, interval, repr(exc))
    return callback


def start_monitor(symbols: list[str]) -> None:
    if not symbols:
        send_tg("⚠️ V4 found no robust symbol. It will rescan in one week.")
        while True:
            time.sleep(3600)

    initialise_market_data(symbols)
    websocket = WebSocket(testnet=False, channel_type="linear")
    for symbol in symbols:
        websocket.kline_stream(
            interval=15,
            symbol=symbol,
            callback=make_callback(symbol, "15"),
        )
        websocket.kline_stream(
            interval=60,
            symbol=symbol,
            callback=make_callback(symbol, "60"),
        )

    send_tg("✅ Tide Universe V4 realtime monitor started\n\n" + "\n".join(symbols))
    started = time.monotonic()
    while True:
        if (time.monotonic() - started) / 3600 >= RESCAN_HOURS:
            log("Weekly rescan due; restarting process.")
            os.execv(sys.executable, [sys.executable, *sys.argv])
        log("Heartbeat monitoring", len(symbols), "symbols")
        time.sleep(60)


def main() -> None:
    selected, _ = scan_and_select()
    start_monitor(selected)


if __name__ == "__main__":
    main()
