#!/usr/bin/env python3
"""BTC 4H three-candle study v2: observe two completed candles, trade the third."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


API = "https://fapi.binance.com/fapi/v1/klines"


def download(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    rows = []
    while cursor < end_ms:
        query = urlencode({"symbol": symbol, "interval": "4h", "startTime": cursor,
                           "endTime": end_ms, "limit": 1500})
        with urlopen(f"{API}?{query}", timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        rows.extend(batch)
        new_cursor = int(batch[-1][0]) + 1
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.05)
    cols = ["time", "open", "high", "low", "close", "volume", "close_time", "qv",
            "trades", "tb", "tq", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df.time, unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df.set_index("time")[["open", "high", "low", "close", "volume"]]


def build(df: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    body = df.close - df.open
    abs_body = body.abs()
    candle_range = (df.high - df.low).replace(0, np.nan)
    upper = df.high - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df.low

    # At row t, candle 1 is t-1, candle 2 is t, and the traded candle is t+1.
    c1 = np.where(body.shift(1) >= 0, "G", "R")
    c2 = np.where(body >= 0, "G", "R")
    z["colour_pattern"] = pd.Series(c1, index=df.index) + pd.Series(c2, index=df.index)
    z["second_body_stronger"] = np.where(abs_body >= abs_body.shift(1), "STRONGER", "WEAKER")
    z["second_range_larger"] = np.where(candle_range >= candle_range.shift(1), "EXPAND", "CONTRACT")
    z["second_volume_higher"] = np.where(df.volume >= df.volume.shift(1), "VOL_UP", "VOL_DOWN")
    z["second_upper_wick"] = upper / candle_range
    z["second_lower_wick"] = lower / candle_range
    z["second_body_ratio"] = body / candle_range
    z["two_candle_return"] = df.close / df.open.shift(1) - 1

    # Coarse, predeclared buckets keep the study transparent and limit overfitting.
    z["wick_state"] = np.select(
        [z.second_upper_wick >= 0.45, z.second_lower_wick >= 0.45],
        ["LONG_UPPER", "LONG_LOWER"], default="NORMAL")
    z["detail_pattern"] = (z.colour_pattern + "|" + z.second_body_stronger + "|" +
                           z.second_range_larger + "|" + z.second_volume_higher + "|" + z.wick_state)

    z["entry_time"] = df.index.to_series().shift(-1)
    z["entry"] = df.open.shift(-1)
    z["exit"] = df.close.shift(-1)
    z["next_high"] = df.high.shift(-1)
    z["next_low"] = df.low.shift(-1)
    z["next_return"] = z.exit / z.entry - 1
    z["next_green"] = z.next_return > 0
    return z.dropna().copy()


def group_table(data: pd.DataFrame, key: str, cost: float) -> pd.DataFrame:
    rows = []
    for name, d in data.groupby(key):
        mean = d.next_return.mean()
        side = 1 if mean >= 0 else -1
        gross = side * d.next_return
        net = gross - cost
        rows.append({
            "pattern": name, "samples": len(d), "next_green_rate": d.next_green.mean(),
            "historically_better_side": "LONG" if side > 0 else "SHORT",
            "avg_directional_gross": gross.mean(), "avg_directional_net_in_sample": net.mean(),
        })
    return pd.DataFrame(rows).sort_values(["avg_directional_net_in_sample", "samples"], ascending=False)


def walk_forward(data: pd.DataFrame, key: str, cost: float, min_train: int,
                 retrain_every: int, min_group_samples: int, edge_buffer: float) -> pd.DataFrame:
    chunks = []
    for start in range(min_train, len(data), retrain_every):
        stop = min(start + retrain_every, len(data))
        train, test = data.iloc[:start], data.iloc[start:stop].copy()
        stats = train.groupby(key).next_return.agg(["mean", "count"])
        test = test.join(stats, on=key)
        test["signal"] = np.where(
            (test["count"] >= min_group_samples) & (test["mean"].abs() >= cost + edge_buffer),
            np.sign(test["mean"]), 0)
        chunks.append(test)
    return pd.concat(chunks) if chunks else data.iloc[0:0].copy()


def performance(oos: pd.DataFrame, cost: float, name: str) -> dict:
    d = oos[oos.signal != 0].copy()
    if d.empty:
        return {"strategy": name, "trades": 0}
    gross = d.signal * d.next_return
    net = gross - cost
    long_mae = d.next_low / d.entry - 1
    short_mae = 1 - d.next_high / d.entry
    mae = pd.Series(np.where(d.signal > 0, long_mae, short_mae), index=d.index)
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    wins, losses = net[net > 0], net[net < 0]
    return {
        "strategy": name, "trades": len(d), "longs": int((d.signal > 0).sum()),
        "shorts": int((d.signal < 0).sum()), "win_rate": (net > 0).mean(),
        "avg_gross": gross.mean(), "avg_net": net.mean(), "median_net": net.median(),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() else np.inf,
        "compounded_return": equity.iloc[-1] - 1, "max_drawdown": dd.min(),
        "avg_mae": mae.mean(), "worst_mae": mae.min(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--round-trip-bps", type=float, default=12.0)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--retrain-every", type=int, default=180)
    p.add_argument("--min-group-samples", type=int, default=100)
    p.add_argument("--edge-buffer-bps", type=float, default=0.0)
    p.add_argument("--output", default="btc_4h_three_candle_v2_results")
    a = p.parse_args()
    cost, buffer = a.round_trip_bps / 10000, a.edge_buffer_bps / 10000
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    data = build(download(a.symbol, a.start, a.end))

    colour = group_table(data, "colour_pattern", cost)
    detail = group_table(data, "detail_pattern", cost)
    reports, all_events = [], []
    for key, label in [("colour_pattern", "four_colour_patterns"), ("detail_pattern", "two_candle_detail")]:
        oos = walk_forward(data, key, cost, a.min_train, a.retrain_every,
                           a.min_group_samples, buffer)
        reports.append(performance(oos, cost, label))
        oos["model"] = label
        all_events.append(oos)

    summary = pd.DataFrame(reports)
    colour.to_csv(out / "four_pattern_full_sample.csv", index=False)
    detail.to_csv(out / "detail_pattern_full_sample.csv", index=False)
    summary.to_csv(out / "walk_forward_summary.csv", index=False)
    pd.concat(all_events).to_csv(out / "walk_forward_events.csv")
    print("\nFULL-SAMPLE DESCRIPTIVE FOUR-PATTERN TABLE (NOT A TRADING BACKTEST)")
    print(colour.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWALK-FORWARD OUT-OF-SAMPLE RESULTS")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()
