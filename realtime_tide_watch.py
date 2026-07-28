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

LOOKBACK_BARS = 24          # 过去6小时，15m × 24
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35
CLUSTER_GAP_BARS = 72       # 18小时
FRESH_MINUTES = 30

STATE_FILE = "alert_state.json"


# ============================================================
# Telegram
# ============================================================

def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets are missing.")
        return

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=20,
    )

    if response.status_code == 200:
        print("Telegram message sent.")
    else:
        print("Telegram send failed:", response.text)


# ============================================================
# Alert state — same signal only sends once
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
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


alert_state = load_alert_state()


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
        raise RuntimeError(f"No historical data returned for {symbol} {interval}")

    parsed = []

    for row in rows:
        parsed.append(
            {
                "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    df = pd.DataFrame(parsed)
    df = df.sort_values("open_time").drop_duplicates("open_time")
    df = df.reset_index(drop=True)

    return df


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
# Model
# ============================================================

def calculate_signal(symbol: str) -> None:
    with data_lock:
        df15 = market_data[symbol]["15"].copy()
        df1h = market_data[symbol]["60"].copy()

    if len(df15) < 50 or len(df1h) < 210:
        print(f"{symbol}: insufficient data")
        return

    # 1h regime
    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()

    df1h["regime_1h"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(
            df1h["ma50"] > df1h["ma200"],
            "uptrend",
            "range",
        ),
    )

    regime = df1h[
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

    df["volume_spike"] = (
        df["volume"] > df["avg_volume"] * VOLUME_MULTIPLIER
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

    df["long_signal"] = (
        (df["regime_1h"] == "downtrend")
        & (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & (df["volume_spike"])
    )

    latest = df.iloc[-1]

    print(
        f'{symbol} | {latest["open_time"]} | '
        f'close={latest["close"]:.2f} | '
        f'regime={latest["regime_1h"]} | '
        f'signal={bool(latest["long_signal"])}'
    )

    if not bool(latest["long_signal"]):
        return

    signal_time = latest["open_time"]
    signal_key = signal_time.isoformat()

    now = pd.Timestamp.now(tz="UTC")
    minutes_since_signal = (
        now - signal_time
    ).total_seconds() / 60

    if minutes_since_signal > FRESH_MINUTES:
        print(
            f"{symbol}: signal is already late "
            f"({minutes_since_signal:.1f} minutes). No alert."
        )
        return

    with state_lock:
        if alert_state.get(symbol) == signal_key:
            print(f"{symbol}: duplicate signal, no alert.")
            return

        alert_state[symbol] = signal_key
        save_alert_state(alert_state)

    message = f"""
🚨 Tide Model FRESH SIGNAL: {symbol}

Signal time UTC: {signal_time}
Entry price: {latest["close"]:.2f}
1h regime: {latest["regime_1h"]}

Rolling low: {latest["rolling_low"]:.2f}
Rolling high: {latest["rolling_high"]:.2f}
Signal low: {latest["low"]:.2f}

Lower wick ratio: {latest["lower_wick_ratio"]:.4f}
Volume multiple: {latest["volume"] / latest["avg_volume"]:.2f}

Signal age: {minutes_since_signal:.1f} minutes

BTC exit plan:
Fixed 6 hours.

ETH exit plan:
Rolling high target or maximum 12 hours.

Fresh signal only. Do not chase late entries.
"""

    send_telegram_message(message)


# ============================================================
# Update closed candles
# ============================================================

def update_closed_candle(
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

        df = (
            df.sort_values("open_time")
            .drop_duplicates("open_time")
            .tail(400)
            .reset_index(drop=True)
        )

        market_data[symbol][interval] = df


# ============================================================
# WebSocket callbacks
# ============================================================

def make_callback(symbol: str, interval: str):
    def callback(message: dict) -> None:
        try:
            candles = message.get("data", [])

            for candle in candles:
                # Only use a formally closed candle
                if not candle.get("confirm", False):
                    continue

                update_closed_candle(symbol, interval, candle)

                print(
                    f"Closed candle received: "
                    f"{symbol} {interval}m "
                    f'{candle["start"]}'
                )

                # Recalculate after each closed 15m candle
                if interval == "15":
                    calculate_signal(symbol)

        except Exception as error:
            print(
                f"Callback error for {symbol} "
                f"{interval}: {error}"
            )

    return callback


# ============================================================
# Start websocket
# ============================================================

def main() -> None:
    print("Starting Tide Model real-time monitor...")
    print("Symbols:", SYMBOLS)

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
        "✅ Tide Model real-time monitor started."
    )

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
