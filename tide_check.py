import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import requests


SYMBOL = "BTC-USD"

LOOKBACK_BARS = 24          # 6h on 15m
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35
CLUSTER_GAP_BARS = 72       # 18h on 15m
HOLD_BARS = 24              # 6h on 15m

def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets not found. Skip sending message.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        },
        timeout=20,
    )

    if response.status_code != 200:
        print("Telegram send failed:", response.text)
    else:
        print("Telegram message sent.")


def download_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol} {interval} {period}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    df = df.reset_index()

    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "open_time"})

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    return df


def build_1h_regime(df1h: pd.DataFrame) -> pd.DataFrame:
    df1h = df1h.copy()
    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()

    df1h["regime_1h"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(df1h["ma50"] > df1h["ma200"], "uptrend", "range"),
    )

    return df1h[["open_time", "ma50", "ma200", "regime_1h"]]


def check_signal() -> None:
    df1h = download_data(SYMBOL, "1h", "730d")
    df15 = download_data(SYMBOL, "15m", "60d")

    regime = build_1h_regime(df1h)

    df = pd.merge_asof(
        df15.sort_values("open_time"),
        regime.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    df["rolling_low"] = df["low"].rolling(LOOKBACK_BARS).min().shift(1)
    df["rolling_high"] = df["high"].rolling(LOOKBACK_BARS).max().shift(1)

    df["avg_volume"] = df["volume"].rolling(VOLUME_LOOKBACK).mean().shift(1)
    df["volume_spike"] = df["volume"] > df["avg_volume"] * VOLUME_MULTIPLIER

    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]

    df["strong_bullish_sweep_15m"] = (
        (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & (df["volume_spike"])
    )

    df["long_signal"] = (
        (df["regime_1h"] == "downtrend")
        & (df["strong_bullish_sweep_15m"])
    )

    latest_row = df.iloc[-1]
    recent = df.tail(CLUSTER_GAP_BARS).copy()
    recent_signals = recent[recent["long_signal"]].copy()

    print("Checked at UTC:", datetime.now(timezone.utc))
    print("Latest candle UTC:", latest_row["open_time"])
    print("Latest close:", latest_row["close"])
    print("Current 1h regime:", latest_row["regime_1h"])
    print("Recent signal count in last 18h:", len(recent_signals))

    if len(recent_signals) == 0:
        print("Status: NO_SIGNAL")
        return

    last_signal = recent_signals.iloc[-1]
    signal_idx = last_signal.name
    latest_idx = df.index[-1]
    bars_since_signal = latest_idx - signal_idx
    hours_since_signal = bars_since_signal * 15 / 60

    print("")
    print("Latest signal:")
    print("Signal time UTC:", last_signal["open_time"])
    print("Signal close / model entry:", last_signal["close"])
    print("Rolling low:", last_signal["rolling_low"])
    print("Low:", last_signal["low"])
    print("Lower wick ratio:", last_signal["lower_wick_ratio"])
    print("Volume spike:", bool(last_signal["volume_spike"]))
    print("Bars since signal:", bars_since_signal)
    print("Hours since signal:", hours_since_signal)

    if bars_since_signal <= 1:
    status = "FRESH_SIGNAL"
    print("Status:", status)
    print("这是刚出现的信号，可以作为 forward test 候选入场。")

    message = f"""
🚨 BTC Tide Model Signal

Status: {status}

Signal time UTC: {last_signal["open_time"]}
Entry price: {last_signal["close"]:.2f}
Latest close: {latest_row["close"]:.2f}
1h regime: {latest_row["regime_1h"]}

Rolling low: {last_signal["rolling_low"]:.2f}
Low: {last_signal["low"]:.2f}
Lower wick ratio: {last_signal["lower_wick_ratio"]:.4f}
Volume spike: {bool(last_signal["volume_spike"])}

Hours since signal: {hours_since_signal:.2f}

Note: Fresh forward-test candidate. Not financial advice.
"""
    send_telegram_message(message)

elif bars_since_signal <= HOLD_BARS:
    status = "ACTIVE_BUT_LATE"
    print("Status:", status)
    print("信号仍在 6 小时模型窗口内，但不是刚出现。实际交易不建议追，只记录观察。")

    message = f"""
⚠️ BTC Tide Model Active Signal

Status: {status}

Signal time UTC: {last_signal["open_time"]}
Entry price: {last_signal["close"]:.2f}
Latest close: {latest_row["close"]:.2f}
1h regime: {latest_row["regime_1h"]}

Hours since signal: {hours_since_signal:.2f}

Note: Signal is still inside 6h model window, but not fresh. Do not chase; record only.
"""
    send_telegram_message(message)

else:
    print("Status: EXPIRED_SIGNAL")
    print("信号已经超过 6 小时模型窗口，不应追。只能复盘。")


if __name__ == "__main__":
    check_signal()
