#!/usr/bin/env python3
"""BTC 4H next-candle direction study (v1).

Uses only data known when a new 4H candle opens.  Signals are generated from
the just-closed candle, entered at the next open, and exited at that close.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


API = "https://fapi.binance.com/fapi/v1/klines"
FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "green_ratio_10",
    "body_balance_10", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "range_atr", "volume_ratio", "volume_trend", "ema20_gap", "ema50_gap",
    "ema20_slope", "ema50_slope", "ema_alignment", "bb_position", "bb_width",
    "bb_width_change", "rsi14", "realized_vol_10",
]


def download_klines(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urlencode({"symbol": symbol, "interval": "4h", "startTime": cursor,
                           "endTime": end_ms, "limit": 1500})
        with urlopen(f"{API}?{query}", timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    if not rows:
        raise RuntimeError("No candles downloaded. Check symbol, dates, and network access.")
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df.set_index("open_time")[["open", "high", "low", "close", "volume"]]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    candle_range = (x.high - x.low).replace(0, np.nan)
    body = x.close - x.open
    abs_body = body.abs()
    ret = x.close.pct_change()
    ema20 = x.close.ewm(span=20, adjust=False).mean()
    ema50 = x.close.ewm(span=50, adjust=False).mean()
    mid = x.close.rolling(20).mean()
    std = x.close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    tr = pd.concat([(x.high - x.low), (x.high - x.close.shift()).abs(),
                    (x.low - x.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    for n in [1, 2, 3, 5, 10]:
        x[f"ret_{n}"] = x.close.pct_change(n)
    x["green_ratio_10"] = (body > 0).rolling(10).mean()
    x["body_balance_10"] = body.rolling(10).sum() / abs_body.rolling(10).sum().replace(0, np.nan)
    x["body_ratio"] = body / candle_range
    x["upper_wick_ratio"] = (x.high - x[["open", "close"]].max(axis=1)) / candle_range
    x["lower_wick_ratio"] = (x[["open", "close"]].min(axis=1) - x.low) / candle_range
    x["range_atr"] = candle_range / atr
    x["volume_ratio"] = x.volume / x.volume.rolling(20).mean()
    x["volume_trend"] = x.volume.rolling(5).mean() / x.volume.rolling(20).mean()
    x["ema20_gap"] = x.close / ema20 - 1
    x["ema50_gap"] = x.close / ema50 - 1
    x["ema20_slope"] = ema20.pct_change(3)
    x["ema50_slope"] = ema50.pct_change(3)
    x["ema_alignment"] = ema20 / ema50 - 1
    x["bb_position"] = (x.close - lower) / (upper - lower).replace(0, np.nan)
    x["bb_width"] = (upper - lower) / mid
    x["bb_width_change"] = x["bb_width"].pct_change(3)
    x["rsi14"] = rsi(x.close)
    x["realized_vol_10"] = ret.rolling(10).std()

    # Row t features predict candle t+1; these values are unknowable until t+1 closes.
    x["entry_time"] = x.index.to_series().shift(-1)
    x["entry"] = x.open.shift(-1)
    x["exit"] = x.close.shift(-1)
    x["next_high"] = x.high.shift(-1)
    x["next_low"] = x.low.shift(-1)
    x["next_return"] = x["exit"] / x["entry"] - 1
    x["target_up"] = (x["next_return"] > 0).astype(int)
    return x.dropna(subset=FEATURES + ["entry", "exit", "next_high", "next_low"]).copy()


def walk_forward_probabilities(data: pd.DataFrame, min_train: int, retrain_every: int) -> pd.Series:
    probs = pd.Series(np.nan, index=data.index, name="prob_up")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced")),
    ])
    for start in range(min_train, len(data), retrain_every):
        stop = min(start + retrain_every, len(data))
        train = data.iloc[:start]
        test = data.iloc[start:stop]
        model.fit(train[FEATURES], train["target_up"])
        probs.iloc[start:stop] = model.predict_proba(test[FEATURES])[:, 1]
    return probs


def metrics(frame: pd.DataFrame, signal: pd.Series, cost: float, name: str) -> dict:
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal, index=frame.index)
    selected = signal.ne(0)
    d = frame.loc[selected].copy()
    sig = signal.loc[selected].astype(int)
    if d.empty:
        return {"strategy": name, "trades": 0}
    gross = sig * d.next_return
    net = gross - cost
    # Intracandle adverse excursion from entry, expressed from the chosen side.
    long_mae = d.next_low / d.entry - 1
    short_mae = 1 - d.next_high / d.entry
    mae = pd.Series(np.where(sig > 0, long_mae, short_mae), index=d.index)
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    wins, losses = net[net > 0], net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "strategy": name, "trades": int(len(d)), "longs": int((sig > 0).sum()), "shorts": int((sig < 0).sum()),
        "win_rate": float((net > 0).mean()), "avg_gross": float(gross.mean()), "avg_net": float(net.mean()),
        "median_net": float(net.median()), "profit_factor": float(pf),
        "compounded_return": float(equity.iloc[-1] - 1), "max_drawdown": float(drawdown.min()),
        "avg_mae": float(mae.mean()), "worst_mae": float(mae.min()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--round-trip-bps", type=float, default=12.0,
                   help="Total fee + slippage for entry and exit, in basis points")
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--retrain-every", type=int, default=180)
    p.add_argument("--output", default="btc_4h_v1_results")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = download_klines(args.symbol, args.start, args.end)
    data = make_dataset(raw)
    data["prob_up"] = walk_forward_probabilities(data, args.min_train, args.retrain_every)
    oos = data.dropna(subset=["prob_up"]).copy()
    cost = args.round_trip_bps / 10_000

    reports = []
    reports.append(metrics(oos, pd.Series(1, index=oos.index), cost, "always_long"))
    reports.append(metrics(oos, np.where(oos.body_ratio > 0, 1, -1), cost, "previous_candle_continuation"))
    reports.append(metrics(oos, np.where(oos.body_ratio > 0, -1, 1), cost, "previous_candle_reversal"))
    for threshold in [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65]:
        signal = pd.Series(np.where(oos.prob_up >= threshold, 1,
                                    np.where(oos.prob_up <= 1 - threshold, -1, 0)), index=oos.index)
        reports.append(metrics(oos, signal, cost, f"logistic_p{threshold:.3f}"))

    summary = pd.DataFrame(reports)
    events = oos[["entry_time", "entry", "exit", "next_high", "next_low", "next_return", "prob_up"] + FEATURES]
    summary.to_csv(out / "summary.csv", index=False)
    events.to_csv(out / "oos_predictions.csv", index=True)
    config = vars(args) | {"feature_count": len(FEATURES), "features": FEATURES,
                           "oos_start": str(oos.entry_time.min()), "oos_end": str(oos.entry_time.max())}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBTC 4H NEXT-CANDLE V1 — OUT-OF-SAMPLE RESULTS")
    print(summary.to_string(index=False, float_format=lambda z: f"{z:.6f}"))
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()
