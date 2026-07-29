import json
import os
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from pybit.unified_trading import HTTP, WebSocket


# ============================================================
# Configuration
# ============================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

LOOKBACK_BARS = 24
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35

# 仅在15m K线收盘前最后3分钟发送预警
PRE_ALERT_WINDOW_MINUTES = 3

STATE_FILE = "alert_state.json"


# ============================================================
# Telegram
# ============================================================

def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets are missing.", flush=True)
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": text,
            },
            timeout=20,
        )

        if response.status_code == 200:
            print("Telegram message sent.", flush=True)
        else:
            print(
                "Telegram send failed:",
                response.status_code,
                response.text,
                flush=True,
            )

    except Exception as error:
        print("Telegram request error:", error, flush=True)


# ============================================================
# Alert state
# ============================================================

state_lock = threading.Lock()


def load_alert_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_alert_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)
    except Exception as error:
        print("Could not save alert state:", error, flush=True)


alert_state = load_alert_state()


def already_alerted(symbol: str, alert_type: str, signal_key: str) -> bool:
    with state_lock:
        symbol_state = alert_state.setdefault(symbol, {})

        if symbol_state.get(alert_type) == signal_key:
            return True

        symbol_state[alert_type] = signal_key
        save_alert_state(alert_state)
        return False


# ============================================================
# Bybit historical data
# ============================================================

http = HTTP(testnet=False)


def get_historical_klines(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    response = http.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    rows = response["result"]["list"]

    if not rows:
        raise RuntimeError(
            f"No historical data returned for {symbol} {interval}"
        )

    parsed = []

    for row in rows:
        parsed.append(
            {
                "open_time": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    return (
        pd.DataFrame(parsed)
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )


# ============================================================
# Runtime storage
# ============================================================

data_lock = threading.Lock()

market_data = {
    symbol: {
        "15": get_historical_klines(symbol, "15", 300),
        "60": get_historical_klines(symbol, "60", 300),
    }
    for symbol in SYMBOLS
}


# ============================================================
# Data update
# ============================================================

def update_candle(
    symbol: str,
    interval: str,
    candle: dict,
) -> None:
    row = {
        "open_time": pd.to_datetime(
            int(candle["start"]),
            unit="ms",
            utc=True,
        ),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
    }

    with data_lock:
        df = market_data[symbol][interval]

        df = df[df["open_time"] != row["open_time"]]

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

        market_data[symbol][interval] = (
            df.sort_values("open_time")
            .drop_duplicates("open_time")
            .tail(400)
            .reset_index(drop=True)
        )


# ============================================================
# Model calculation
# ============================================================

def build_model_frame(symbol: str) -> pd.DataFrame:
    with data_lock:
        df15 = market_data[symbol]["15"].copy()
        df1h = market_data[symbol]["60"].copy()

    now = pd.Timestamp.now(tz="UTC")

    # 1h趋势只使用已经收盘的1h K线，避免趋势状态盘中重绘
    df1h["close_time"] = (
        df1h["open_time"] + pd.Timedelta(hours=1)
    )

    closed_1h = df1h[df1h["close_time"] <= now].copy()

    if len(df15) < 50 or len(closed_1h) < 210:
        raise RuntimeError(f"{symbol}: insufficient historical data")

    closed_1h["ma50"] = (
        closed_1h["close"].rolling(50).mean()
    )

    closed_1h["ma200"] = (
        closed_1h["close"].rolling(200).mean()
    )

    closed_1h["regime_1h"] = np.where(
        closed_1h["ma50"] < closed_1h["ma200"],
        "downtrend",
        np.where(
            closed_1h["ma50"] > closed_1h["ma200"],
            "uptrend",
            "range",
        ),
    )

    regime = closed_1h[
        ["open_time", "regime_1h", "ma50", "ma200"]
    ].sort_values("open_time")

    df = pd.merge_asof(
        df15.sort_values("open_time"),
        regime,
        on="open_time",
        direction="backward",
    )

    df["rolling_low"] = (
        df["low"]
        .rolling(LOOKBACK_BARS)
        .min()
        .shift(1)
    )

    df["rolling_high"] = (
        df["high"]
        .rolling(LOOKBACK_BARS)
        .max()
        .shift(1)
    )

    df["avg_volume"] = (
        df["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
        .shift(1)
    )

    df["volume_multiple"] = np.where(
        df["avg_volume"] > 0,
        df["volume"] / df["avg_volume"],
        0,
    )

    df["volume_spike"] = (
        df["volume_multiple"] > VOLUME_MULTIPLIER
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    )

    df["candle_range"] = df["high"] - df["low"]

    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] / df["candle_range"],
        0,
    )

    df["break_rolling_low"] = (
        df["low"] < df["rolling_low"]
    )

    df["reclaim_rolling_low"] = (
        df["close"] > df["rolling_low"]
    )

    df["long_signal"] = (
        (df["regime_1h"] == "downtrend")
        & df["break_rolling_low"]
        & df["reclaim_rolling_low"]
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & df["volume_spike"]
    )

    return df


def send_signal_alert(
    symbol: str,
    latest: pd.Series,
    alert_type: str,
    minutes_to_close: float,
) -> None:
    candle_open_time = latest["open_time"]
    candle_close_time = (
        candle_open_time + pd.Timedelta(minutes=15)
    )

    signal_key = candle_open_time.isoformat()

    if already_alerted(symbol, alert_type, signal_key):
        print(
            f"{symbol}: duplicate {alert_type}; skipped.",
            flush=True,
        )
        return

    if alert_type == "pre":
        title = f"⚠️ Tide Model PRE-SIGNAL: {symbol}"
        explanation = (
            "Conditions are currently satisfied, but the "
            "15m candle has not closed yet.\n"
            "This signal may disappear before candle close."
        )
    else:
        title = f"🚨 Tide Model CONFIRMED SIGNAL: {symbol}"
        explanation = (
            "The 15m candle has formally closed and the "
            "signal remains valid."
        )

    if symbol == "BTCUSDT":
        exit_plan = "BTC exit plan: fixed 6 hours."
    else:
        exit_plan = (
            "ETH exit plan: rolling-high target, "
            "otherwise maximum 12 hours."
        )

    message = f"""
{title}

Candle open UTC: {candle_open_time}
Candle close UTC: {candle_close_time}

Current / entry price: {latest["close"]:.2f}
1h regime: {latest["regime_1h"]}

Rolling low: {latest["rolling_low"]:.2f}
Rolling high: {latest["rolling_high"]:.2f}
Signal low: {latest["low"]:.2f}

Lower wick ratio: {latest["lower_wick_ratio"]:.4f}
Volume multiple: {latest["volume_multiple"]:.2f}

Minutes to candle close: {minutes_to_close:.1f}

{explanation}

{exit_plan}

Do not chase delayed signals.
"""

    send_telegram_message(message)


def calculate_signal(
    symbol: str,
    candle_confirmed: bool,
) -> None:
    df = build_model_frame(symbol)
    latest = df.iloc[-1]

    now = pd.Timestamp.now(tz="UTC")
    candle_close_time = (
        latest["open_time"] + pd.Timedelta(minutes=15)
    )

    minutes_to_close = (
        candle_close_time - now
    ).total_seconds() / 60

    conditions = {
        "downtrend": latest["regime_1h"] == "downtrend",
        "break_low": bool(latest["break_rolling_low"]),
        "reclaim": bool(latest["reclaim_rolling_low"]),
        "wick": (
            latest["lower_wick_ratio"]
            > LOWER_WICK_THRESHOLD
        ),
        "volume": bool(latest["volume_spike"]),
    }

    print(
        f'{symbol} | {latest["open_time"]} | '
        f'close={latest["close"]:.2f} | '
        f'confirmed={candle_confirmed} | '
        f'regime={latest["regime_1h"]} | '
        f'conditions={conditions} | '
        f'minutes_to_close={minutes_to_close:.2f}',
        flush=True,
    )

    if not bool(latest["long_signal"]):
        return

    # 正式收盘确认
    if candle_confirmed:
        send_signal_alert(
            symbol=symbol,
            latest=latest,
            alert_type="confirmed",
            minutes_to_close=0,
        )
        return

    # 盘中预警：只在收盘前最后3分钟发送
    if 0 <= minutes_to_close <= PRE_ALERT_WINDOW_MINUTES:
        send_signal_alert(
            symbol=symbol,
            latest=latest,
            alert_type="pre",
            minutes_to_close=minutes_to_close,
        )


# ============================================================
# WebSocket callback
# ============================================================

def make_callback(symbol: str, interval: str):
    def callback(message: dict) -> None:
        try:
            candles = message.get("data", [])

            for candle in candles:
                confirmed = bool(
                    candle.get("confirm", False)
                )

                # 无论是否收盘，都更新当前K线
                update_candle(
                    symbol=symbol,
                    interval=interval,
                    candle=candle,
                )

                # 1h只负责更新趋势数据
                if interval == "60":
                    if confirmed:
                        print(
                            f"Closed 1h candle: {symbol}",
                            flush=True,
                        )
                    continue

                # 15m每次推送都重新计算
                calculate_signal(
                    symbol=symbol,
                    candle_confirmed=confirmed,
                )

        except Exception as error:
            print(
                f"Callback error: {symbol} {interval}: {error}",
                flush=True,
            )

    return callback


# ============================================================
# Start monitor
# ============================================================

def main() -> None:
    print(
        "Starting Tide Model V2.1 real-time monitor...",
        flush=True,
    )

    print("Symbols:", SYMBOLS, flush=True)

    websocket = WebSocket(
        testnet=False,
        channel_type="linear",
    )

    for symbol in SYMBOLS:
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

    send_telegram_message(
        "✅ Tide Model V2.1 monitor started.\n\n"
        "BTCUSDT + ETHUSDT\n"
        "PRE_SIGNAL: last 3 minutes before 15m close\n"
        "CONFIRMED: after formal 15m close"
    )

    while True:
        print(
            "Heartbeat:",
            datetime.now(timezone.utc).isoformat(),
            flush=True,
        )
        time.sleep(60)


if __name__ == "__main__":
    main()
