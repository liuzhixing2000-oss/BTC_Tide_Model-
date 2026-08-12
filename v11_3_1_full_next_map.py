#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11.3.1 FAST — FULL 15m NEXT SCORE MAP (5–95)
=============================================

Purpose
-------
Test whether 15m Next quality is NON-LINEAR:
is there a "sweet spot" (for example 50-65) rather than "higher is always better"?

Important
---------
This is a research script only. It does NOT change the live V10.8.2 engine.

It reuses the exact BASE_15M_NEXT score construction from V11.2 so the
new band results are directly comparable with the previous run.

Bands
-----
<50
50-55
55-60
60-65
65-70
70-75
75-80
80-85
85-90
90-95
>=95

Also tests broader candidate production zones:
50-60, 50-65, 55-65, 55-70, 60-70, >=50, >=55, >=60, >=70, >=80, >=90.

Outputs
-------
v11_3_1_full_next_map/
  trades.csv
  band_results.csv
  zone_results.csv
  half_stability.csv
  report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-file", type=Path, default=Path("crypto_tide_engine_v10_8_2_2.py"))
    p.add_argument("--bundle-dir", type=Path, default=Path("v10_bundle"))
    p.add_argument("--output-dir", type=Path, default=Path("v11_3_1_full_next_map"))
    p.add_argument("--max-symbols", type=int, default=60)
    p.add_argument("--full", action="store_true")
    p.add_argument("--days-15m", type=int, default=60)
    p.add_argument("--days-1h", type=int, default=90)
    p.add_argument("--validation-days", type=int, default=15)
    p.add_argument("--hold-hours", type=float, default=4.0)
    p.add_argument("--structure-buffer-atr", type=float, default=0.50)
    return p.parse_args()


def sf(v: Any, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def normalize_time(s):
    return pd.to_datetime(s, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def load_engine(path: Path, data_dir: Path):
    candidates = [
        path,
        Path("crypto_tide_engine_v10_8_2_2.py"),
        Path("crypto_tide_engine_v10_8_2_1.py"),
        Path("archive/crypto_tide_engine_v10_8_2_2.py"),
        Path("archive/crypto_tide_engine_v10_8_2_1.py"),
    ]
    selected = next((x for x in candidates if x.exists()), None)
    if selected is None:
        raise FileNotFoundError("No compatible Tide engine found.")
    print("Using Tide engine:", selected)

    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIDE_DATA_DIR"] = str(data_dir.resolve())

    spec = importlib.util.spec_from_file_location("v113_engine", selected)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_setup(row):
    return bool(row.get("signal", False))


def next_score(next_row, setup_row):
    # Exact same baseline score formula used in V11.2.
    o, h, l, c, v = [sf(next_row.get(k)) for k in ["open", "high", "low", "close", "volume"]]
    if not all(np.isfinite(z) for z in [o, h, l, c, v]) or h <= l:
        return np.nan

    rng = h - l
    body = np.clip((c - o) / rng, 0, 1)
    close_pos = np.clip((c - l) / rng, 0, 1)

    atr = sf(setup_row.get("atr14"))
    impulse = np.clip((c - o) / atr, 0, 1.5) / 1.5 if np.isfinite(atr) and atr > 0 else 0

    vr = sf(setup_row.get("volume"))
    vol = np.clip(v / vr, 0, 3) / 3 if np.isfinite(vr) and vr > 0 else 0

    return float(np.clip(
        100 * (0.35 * impulse + 0.25 * body + 0.25 * close_pos + 0.15 * vol),
        0, 100
    ))


def structural_stop(setup, entry, buffer_atr):
    atr = sf(setup.get("confirmation_signal_atr", setup.get("atr14")))
    low = sf(setup.get("confirmation_signal_low", setup.get("low")))
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(low):
        return np.nan, np.nan
    stop = low - buffer_atr * atr
    risk = (entry - stop) / entry if entry > 0 else np.nan
    return stop, risk


def simulate_15m_trade(engine, df15, entry_i, entry, stop, hold_hours):
    if not np.isfinite(stop) or stop >= entry:
        return None

    bars = max(1, int(round(hold_hours * 4)))
    end = min(entry_i + bars, len(df15) - 1)
    exit_i = end
    exit_price = float(df15.iloc[end]["close"])
    reason = "fixed_4h"

    # Entry occurs at close of entry_i. Start stop checking next candle.
    for j in range(entry_i + 1, end + 1):
        if float(df15.iloc[j]["low"]) <= stop:
            exit_i = j
            exit_price = stop
            reason = "initial_stop"
            break

    path = df15.iloc[entry_i:exit_i + 1]
    return {
        "net_return": exit_price / entry - 1 - float(engine.FEE_SLIPPAGE),
        "mfe_pct": float(path["high"].max()) / entry - 1,
        "mae_pct": float(path["low"].min()) / entry - 1,
        "exit_reason": reason,
        "exit_time": pd.Timestamp(df15.iloc[exit_i]["open_time"]) + pd.Timedelta(minutes=15),
    }


def pf(r):
    r = pd.Series(r).dropna().astype(float)
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    if gl == 0:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def mdd(r):
    r = pd.Series(r).dropna().astype(float)
    if r.empty:
        return np.nan
    eq = (1 + r.clip(lower=-0.999)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def longest_loss_streak(r):
    best = cur = 0
    for x in pd.Series(r).dropna():
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def metrics(g):
    if g.empty:
        return {
            "trades": 0, "expectancy": np.nan, "median": np.nan,
            "win_rate": np.nan, "profit_factor": np.nan,
            "total_compound_return": np.nan, "max_drawdown": np.nan,
            "avg_stop_risk": np.nan, "pct_stop_le2": np.nan,
            "avg_mfe": np.nan, "avg_mae": np.nan,
            "longest_loss_streak": np.nan,
        }

    g = g.sort_values("entry_time")
    r = g["net_return"].astype(float)
    risk = g["hard_stop_risk_pct"].astype(float)
    return {
        "trades": len(g),
        "expectancy": float(r.mean()),
        "median": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": pf(r),
        "total_compound_return": float((1 + r.clip(lower=-0.999)).prod() - 1),
        "max_drawdown": mdd(r),
        "avg_stop_risk": float(risk.mean()),
        "pct_stop_le2": float((risk <= 0.02).mean()),
        "avg_mfe": float(g["mfe_pct"].mean()),
        "avg_mae": float(g["mae_pct"].mean()),
        "longest_loss_streak": longest_loss_streak(r),
    }


BANDS = [
    ("LT5", None, 5),
    ("05_10", 5, 10),
    ("10_15", 10, 15),
    ("15_20", 15, 20),
    ("20_25", 20, 25),
    ("25_30", 25, 30),
    ("30_35", 30, 35),
    ("35_40", 35, 40),
    ("40_45", 40, 45),
    ("45_50", 45, 50),
    ("50_55", 50, 55),
    ("55_60", 55, 60),
    ("60_65", 60, 65),
    ("65_70", 65, 70),
    ("70_75", 70, 75),
    ("75_80", 75, 80),
    ("80_85", 80, 85),
    ("85_90", 85, 90),
    ("90_95", 90, 95),
    ("GE95", 95, None),
]

# Broad zones are deliberately overlapping. They help identify whether a wider
# positive-expectancy production range can increase alert frequency without
# simply accepting every weak signal.
ZONES = [
    ("05_15", 5, 15), ("10_20", 10, 20), ("15_25", 15, 25),
    ("20_30", 20, 30), ("25_35", 25, 35), ("30_40", 30, 40),
    ("35_45", 35, 45), ("40_50", 40, 50), ("45_55", 45, 55),
    ("50_60", 50, 60), ("55_65", 55, 65), ("60_70", 60, 70),
    ("65_75", 65, 75), ("70_80", 70, 80), ("75_85", 75, 85),
    ("80_90", 80, 90), ("85_95", 85, 95),

    ("20_40", 20, 40), ("25_45", 25, 45), ("30_50", 30, 50),
    ("35_55", 35, 55), ("40_60", 40, 60), ("45_65", 45, 65),
    ("50_70", 50, 70), ("55_70", 55, 70), ("55_75", 55, 75),
    ("60_75", 60, 75), ("60_80", 60, 80),

    ("GE20", 20, None), ("GE30", 30, None), ("GE40", 40, None),
    ("GE50", 50, None), ("GE55", 55, None), ("GE60", 60, None),
    ("GE70", 70, None), ("GE80", 80, None), ("GE90", 90, None),
]

def select_range(df, lo, hi):
    mask = pd.Series(True, index=df.index)
    if lo is not None:
        mask &= df["next_score"] >= lo
    if hi is not None:
        mask &= df["next_score"] < hi
    return df[mask]


def split_metrics(g, validation_start):
    train = g[g["setup_time"] < validation_start]
    val = g[g["setup_time"] >= validation_start]
    A, T, V = metrics(g), metrics(train), metrics(val)

    row = {}
    for prefix, d in [("", A), ("train_", T), ("validation_", V)]:
        row.update({prefix + k: v for k, v in d.items()})
    return row


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    engine = load_engine(a.engine_file, a.output_dir / "engine_data")

    universe = pd.read_csv(a.bundle_dir / "stage2_full_results.csv")
    if "eligible" in universe.columns:
        universe = universe[universe["eligible"].astype(str).str.lower().eq("true")]
    if "score" in universe.columns:
        universe = universe.sort_values("score", ascending=False)
    if not a.full:
        universe = universe.head(a.max_symbols)

    print("=" * 100)
    print("V11.3.1 FAST — FULL 15m NEXT SCORE MAP (5–95)")
    print("=" * 100)
    print("Symbols:", len(universe))

    rows = []

    for n, symbol in enumerate(universe["symbol"].astype(str), 1):
        print(f"[{n}/{len(universe)}] {symbol}", flush=True)
        try:
            df15 = engine.model_frame(
                engine.fetch_klines(symbol, "15", a.days_15m),
                engine.fetch_klines(symbol, "60", a.days_1h),
            )
            df15["open_time"] = normalize_time(df15["open_time"])
            df15 = df15.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        except Exception as exc:
            print("  FAILED:", type(exc).__name__, exc)
            continue

        setups = 0
        for i in range(len(df15) - 1):
            setup = df15.iloc[i]
            if not is_setup(setup):
                continue
            setups += 1

            nxt = df15.iloc[i + 1]
            score = next_score(nxt, setup)
            if not np.isfinite(score):
                continue

            # Enter at the close of the completed Next 15m candle.
            entry_i = i + 1
            entry_time = pd.Timestamp(nxt["open_time"]) + pd.Timedelta(minutes=15)
            entry = float(nxt["close"])
            stop, risk = structural_stop(setup, entry, a.structure_buffer_atr)
            sim = simulate_15m_trade(engine, df15, entry_i, entry, stop, a.hold_hours)
            if sim is None:
                continue

            rows.append({
                "symbol": symbol,
                "setup_time": pd.Timestamp(setup["open_time"]) + pd.Timedelta(minutes=15),
                "entry_time": entry_time,
                "next_score": score,
                "entry_price": entry,
                "stop_price": stop,
                "hard_stop_risk_pct": risk,
                **sim,
            })

        print("  raw 15m setups:", setups)

    if not rows:
        raise RuntimeError("No trades generated.")

    trades = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    trades.to_csv(a.output_dir / "trades.csv", index=False)

    validation_start = trades["setup_time"].max() - pd.Timedelta(days=a.validation_days)

    # Exact bands.
    band_rows = []
    for name, lo, hi in BANDS:
        g = select_range(trades, lo, hi)
        row = {"band": name, "lower": lo, "upper_exclusive": hi}
        row.update(split_metrics(g, validation_start))
        band_rows.append(row)

    bands = pd.DataFrame(band_rows)
    bands.to_csv(a.output_dir / "band_results.csv", index=False)

    # Wider candidate zones.
    zone_rows = []
    for name, lo, hi in ZONES:
        g = select_range(trades, lo, hi)
        row = {"zone": name, "lower": lo, "upper_exclusive": hi}
        row.update(split_metrics(g, validation_start))
        zone_rows.append(row)

    zones = pd.DataFrame(zone_rows)

    # Stability score favors validation expectancy/PF while requiring sample size.
    sample_factor = np.minimum(zones["validation_trades"].fillna(0) / 20.0, 1.0)
    exp_rank = zones["validation_expectancy"].rank(pct=True)
    pf_rank = zones["validation_profit_factor"].replace(np.inf, 20).rank(pct=True)
    risk_rank = (-zones["validation_avg_stop_risk"]).rank(pct=True)
    zones["research_score"] = sample_factor * (
        0.50 * exp_rank + 0.30 * pf_rank + 0.20 * risk_rank
    )
    zones = zones.sort_values(
        ["research_score", "validation_expectancy"],
        ascending=False
    )
    zones.to_csv(a.output_dir / "zone_results.csv", index=False)

    # Time stability: split pre-validation history into early/middle plus validation.
    tmin = trades["setup_time"].min()
    train_end = validation_start
    train_mid = tmin + (train_end - tmin) / 2

    periods = [
        ("EARLY", tmin, train_mid),
        ("MIDDLE", train_mid, train_end),
        ("VALIDATION", validation_start, trades["setup_time"].max() + pd.Timedelta(seconds=1)),
    ]

    stability_rows = []
    for zone, lo, hi in ZONES:
        zg = select_range(trades, lo, hi)
        for period, start, end in periods:
            g = zg[(zg["setup_time"] >= start) & (zg["setup_time"] < end)]
            stability_rows.append({
                "zone": zone,
                "period": period,
                **metrics(g),
            })

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(a.output_dir / "half_stability.csv", index=False)

    print("\nEXACT NEXT SCORE BANDS — VALIDATION\n")
    print(
        bands[
            [
                "band", "validation_trades", "validation_expectancy",
                "validation_win_rate", "validation_profit_factor",
                "validation_max_drawdown", "validation_avg_stop_risk",
                "validation_pct_stop_le2", "validation_avg_mfe",
                "validation_avg_mae",
            ]
        ].to_string(index=False)
    )

    print("\nTOP CANDIDATE ZONES\n")
    print(
        zones[
            [
                "zone", "validation_trades", "validation_expectancy",
                "validation_profit_factor", "validation_max_drawdown",
                "validation_avg_stop_risk", "validation_pct_stop_le2",
                "research_score",
            ]
        ].head(12).to_string(index=False)
    )

    report = [
        "V11.3.1 FAST — FULL 15m NEXT SCORE MAP (5–95)",
        "=" * 100,
        f"Symbols: {len(universe)}",
        f"Validation starts: {validation_start.isoformat()}",
        "",
        "EXACT BANDS — VALIDATION",
        "-" * 100,
        bands[
            [
                "band", "validation_trades", "validation_expectancy",
                "validation_win_rate", "validation_profit_factor",
                "validation_max_drawdown", "validation_avg_stop_risk",
                "validation_pct_stop_le2", "validation_avg_mfe",
                "validation_avg_mae", "validation_longest_loss_streak",
            ]
        ].to_string(index=False),
        "",
        "CANDIDATE ZONES",
        "-" * 100,
        zones.to_string(index=False),
        "",
        "TIME STABILITY",
        "-" * 100,
        stability.to_string(index=False),
    ]
    (a.output_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nOutput:", a.output_dir)


if __name__ == "__main__":
    main()
