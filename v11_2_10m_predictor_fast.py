#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11.2 FAST — 10m Predictor of 15m Next
======================================

Research question
-----------------
Can we keep the quality advantage of waiting for a strong 15m Next candle,
but enter 5 minutes earlier by using the first TWO closed 5m candles of that
15m Next period?

Timeline
--------
15m setup closes at T0.

At T0+5m:
- first 5m candle is known

At T0+10m:
- first two 5m candles are known
- last 5m candle of the 15m Next candle is NOT known yet

At T0+15m:
- full 15m Next candle is known

This test compares:

1) BASE_15M_NEXT
   Wait full 15m Next candle and enter at T0+15m if the final 15m Next score
   passes threshold.

2) PREDICT_10M
   At T0+10m, use only the first two 5m candles to estimate the probability
   that the eventual 15m Next candle will be strong.
   If the 10m predictor score passes threshold, enter 5 minutes early.

3) PREDICT_10M_STRICT
   Same, but requires both 5m candles to close bullish / constructive.

No look-ahead
-------------
The 10m predictor never uses the third 5m candle.

FAST defaults
-------------
- top 60 robust symbols
- 45d 5m
- 60d 15m
- 90d 1h
- validation = last 15d

Outputs
-------
v11_2_10m_predictor_fast/
- trades.csv
- predictor_threshold_results.csv
- final_next_threshold_stability.csv
- rescue_results.csv
- report.txt
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
    p.add_argument("--output-dir", type=Path, default=Path("v11_2_10m_predictor_fast"))
    p.add_argument("--max-symbols", type=int, default=60)
    p.add_argument("--full", action="store_true")
    p.add_argument("--days-5m", type=int, default=45)
    p.add_argument("--days-15m", type=int, default=60)
    p.add_argument("--days-1h", type=int, default=90)
    p.add_argument("--validation-days", type=int, default=15)
    p.add_argument("--hold-hours", type=float, default=4.0)
    p.add_argument("--structure-buffer-atr", type=float, default=0.50)
    p.add_argument(
        "--predictor-thresholds",
        type=str,
        default="50,55,60,65,70,75,80,85",
    )
    p.add_argument(
        "--final-next-thresholds",
        type=str,
        default="50,55,60,65,70,75,80,85,90",
    )
    return p.parse_args()


def sf(v: Any, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def normalize_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


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

    spec = importlib.util.spec_from_file_location("v112_engine", selected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import engine: {selected}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare_5m(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["open_time"] = normalize_time(x["open_time"])
    x = x.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    c = x["close"].astype(float)
    v = x["volume"].astype(float)

    rng = (h - l).replace(0, np.nan)
    x["body_frac"] = ((c - o) / rng).clip(-1, 1)
    x["close_pos"] = ((c - l) / rng).clip(0, 1)

    prev = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev).abs(), (l - prev).abs()],
        axis=1,
    ).max(axis=1)
    x["atr14"] = tr.rolling(14, min_periods=8).mean()

    x["ret_atr"] = (c - o) / x["atr14"].replace(0, np.nan)
    x["volume_mult"] = v / v.rolling(20, min_periods=10).median()
    x["prior_high3"] = h.shift(1).rolling(3, min_periods=2).max()
    x["micro_break"] = (c > x["prior_high3"]).astype(float)
    x["close_time"] = x["open_time"] + pd.Timedelta(minutes=5)

    return x


def is_setup(row: pd.Series) -> bool:
    return bool(row.get("signal", False))


def final_15m_next_score(next_row: pd.Series, setup_row: pd.Series) -> float:
    o = sf(next_row.get("open"))
    h = sf(next_row.get("high"))
    l = sf(next_row.get("low"))
    c = sf(next_row.get("close"))
    v = sf(next_row.get("volume"))
    if not all(np.isfinite(z) for z in [o, h, l, c, v]) or h <= l:
        return np.nan

    rng = h - l
    body = np.clip((c - o) / rng, 0, 1)
    close_pos = np.clip((c - l) / rng, 0, 1)

    atr = sf(setup_row.get("atr14"))
    impulse = np.clip((c - o) / atr, 0, 1.5) / 1.5 if np.isfinite(atr) and atr > 0 else 0

    vr = sf(setup_row.get("volume"))
    vol = np.clip(v / vr, 0, 3) / 3 if np.isfinite(vr) and vr > 0 else 0

    return float(np.clip(100 * (0.35 * impulse + 0.25 * body + 0.25 * close_pos + 0.15 * vol), 0, 100))


def predictor_10m_score(r1: pd.Series, r2: pd.Series) -> tuple[float, bool]:
    """
    Uses only first two 5m candles.

    We reward:
    - cumulative progress over 10m
    - both closes near highs
    - constructive bodies
    - volume support
    - at least one micro break
    """
    open1 = sf(r1.get("open"))
    close2 = sf(r2.get("close"))
    atr = np.nanmean([sf(r1.get("atr14")), sf(r2.get("atr14"))])

    progress = (
        np.clip((close2 - open1) / atr, 0, 2.0) / 2.0
        if np.isfinite(atr) and atr > 0 and np.isfinite(open1) and np.isfinite(close2)
        else 0.0
    )

    body = np.nanmean([
        np.clip(sf(r1.get("body_frac"), 0), 0, 1),
        np.clip(sf(r2.get("body_frac"), 0), 0, 1),
    ])

    close_pos = np.nanmean([
        np.clip(sf(r1.get("close_pos"), 0), 0, 1),
        np.clip(sf(r2.get("close_pos"), 0), 0, 1),
    ])

    volume = np.nanmean([
        np.clip(sf(r1.get("volume_mult"), 0), 0, 3) / 3,
        np.clip(sf(r2.get("volume_mult"), 0), 0, 3) / 3,
    ])

    micro = max(sf(r1.get("micro_break"), 0), sf(r2.get("micro_break"), 0))

    score = 100 * (
        0.35 * progress
        + 0.20 * body
        + 0.20 * close_pos
        + 0.15 * volume
        + 0.10 * micro
    )

    strict_ok = (
        sf(r1.get("close")) > sf(r1.get("open"))
        and sf(r2.get("close")) >= sf(r2.get("open"))
        and sf(r2.get("close")) >= sf(r1.get("close")) * 0.998
    )

    return float(np.clip(score, 0, 100)), bool(strict_ok)


def structural_stop(setup: pd.Series, entry: float, buffer_atr: float):
    atr = sf(setup.get("confirmation_signal_atr", setup.get("atr14")))
    low = sf(setup.get("confirmation_signal_low", setup.get("low")))
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(low):
        return np.nan, np.nan

    stop = low - buffer_atr * atr
    risk = (entry - stop) / entry if entry > 0 else np.nan
    return stop, risk


def simulate_trade(engine, df5, idx, entry, stop, hold_hours):
    if not np.isfinite(stop) or stop >= entry:
        return None

    end = min(idx + int(round(hold_hours * 12)), len(df5) - 1)
    exit_idx = end
    exit_price = float(df5.iloc[end]["close"])
    reason = "fixed_4h"

    for j in range(idx + 1, end + 1):
        if float(df5.iloc[j]["low"]) <= stop:
            exit_idx = j
            exit_price = stop
            reason = "initial_stop"
            break

    path = df5.iloc[idx : exit_idx + 1]

    return {
        "net_return": exit_price / entry - 1 - float(engine.FEE_SLIPPAGE),
        "mfe_pct": float(path["high"].max()) / entry - 1,
        "mae_pct": float(path["low"].min()) / entry - 1,
        "exit_reason": reason,
        "exit_time": pd.Timestamp(df5.iloc[exit_idx]["close_time"]),
    }


def profit_factor(r):
    r = pd.Series(r).dropna().astype(float)
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    if gl == 0:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def max_drawdown(r):
    r = pd.Series(r).dropna().astype(float)
    if r.empty:
        return np.nan
    eq = (1 + r.clip(lower=-0.999)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summarize(g):
    if g.empty:
        return dict(
            trades=0,
            expectancy=np.nan,
            win_rate=np.nan,
            profit_factor=np.nan,
            max_drawdown=np.nan,
            avg_risk=np.nan,
            pct_risk_le2=np.nan,
            avg_delay=np.nan,
        )
    r = g["net_return"].astype(float)
    risk = g["hard_stop_risk_pct"].astype(float)
    return dict(
        trades=len(g),
        expectancy=float(r.mean()),
        win_rate=float((r > 0).mean()),
        profit_factor=profit_factor(r),
        max_drawdown=max_drawdown(r),
        avg_risk=float(risk.mean()),
        pct_risk_le2=float((risk <= 0.02).mean()),
        avg_delay=float(g["delay_minutes"].mean()),
    )


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    predictor_thresholds = [float(x) for x in a.predictor_thresholds.split(",")]
    final_thresholds = [float(x) for x in a.final_next_thresholds.split(",")]

    engine = load_engine(a.engine_file, a.output_dir / "engine_data")

    universe = pd.read_csv(a.bundle_dir / "stage2_full_results.csv")
    if "eligible" in universe.columns:
        universe = universe[
            universe["eligible"].astype(str).str.lower().eq("true")
        ]
    if "score" in universe.columns:
        universe = universe.sort_values("score", ascending=False)

    if not a.full:
        universe = universe.head(a.max_symbols)

    print("=" * 100)
    print("V11.2 FAST — 10m PREDICTOR OF 15m NEXT")
    print("=" * 100)
    print("Symbols:", len(universe))

    rows = []

    for n, symbol in enumerate(universe["symbol"].astype(str), 1):
        print(f"[{n}/{len(universe)}] {symbol}", flush=True)

        try:
            df5 = prepare_5m(engine.fetch_klines(symbol, "5", a.days_5m))
            df15 = engine.model_frame(
                engine.fetch_klines(symbol, "15", a.days_15m),
                engine.fetch_klines(symbol, "60", a.days_1h),
            )
            df15["open_time"] = normalize_time(df15["open_time"])
            df15 = df15.sort_values("open_time").reset_index(drop=True)
        except Exception as exc:
            print("  FAILED:", type(exc).__name__, exc)
            continue

        for i in range(len(df15) - 1):
            setup = df15.iloc[i]
            if not is_setup(setup):
                continue

            t0 = pd.Timestamp(setup["open_time"]) + pd.Timedelta(minutes=15)

            ids = np.flatnonzero(
                (df5["close_time"] > t0).to_numpy()
            )
            if len(ids) < 3:
                continue

            i1, i2, i3 = map(int, ids[:3])
            r1, r2, r3 = df5.iloc[i1], df5.iloc[i2], df5.iloc[i3]

            pred_score, strict_ok = predictor_10m_score(r1, r2)

            final_row = df15.iloc[i + 1]
            final_score = final_15m_next_score(final_row, setup)

            final_close = pd.Timestamp(final_row["open_time"]) + pd.Timedelta(minutes=15)
            base_ids = np.flatnonzero((df5["close_time"] >= final_close).to_numpy())
            if not len(base_ids):
                continue

            ibase = int(base_ids[0])
            base_entry = float(df5.iloc[ibase]["close"])
            base_stop, base_risk = structural_stop(setup, base_entry, a.structure_buffer_atr)

            pred_entry = float(r2["close"])
            pred_stop, pred_risk = structural_stop(setup, pred_entry, a.structure_buffer_atr)

            common = {
                "symbol": symbol,
                "setup_time": t0,
                "predictor10_score": pred_score,
                "predictor10_strict_ok": strict_ok,
                "final15_next_score": final_score,
                "baseline15_entry": base_entry,
                "baseline15_risk": base_risk,
                "predict10_entry": pred_entry,
                "predict10_risk": pred_risk,
            }

            # Baseline final 15m Next threshold stability
            for th in final_thresholds:
                if np.isfinite(final_score) and final_score >= th:
                    res = simulate_trade(
                        engine, df5, ibase, base_entry, base_stop, a.hold_hours
                    )
                    if res:
                        rows.append({
                            **common,
                            "method": "BASE_15M_NEXT",
                            "threshold": th,
                            "entry_time": final_close,
                            "entry_price": base_entry,
                            "hard_stop_risk_pct": base_risk,
                            "delay_minutes": 15,
                            **res,
                        })

            # 10m predictor thresholds
            for th in predictor_thresholds:
                if pred_score >= th:
                    res = simulate_trade(
                        engine, df5, i2, pred_entry, pred_stop, a.hold_hours
                    )
                    if res:
                        rows.append({
                            **common,
                            "method": "PREDICT_10M",
                            "threshold": th,
                            "entry_time": pd.Timestamp(r2["close_time"]),
                            "entry_price": pred_entry,
                            "hard_stop_risk_pct": pred_risk,
                            "delay_minutes": 10,
                            **res,
                        })

                    if strict_ok:
                        res2 = simulate_trade(
                            engine, df5, i2, pred_entry, pred_stop, a.hold_hours
                        )
                        if res2:
                            rows.append({
                                **common,
                                "method": "PREDICT_10M_STRICT",
                                "threshold": th,
                                "entry_time": pd.Timestamp(r2["close_time"]),
                                "entry_price": pred_entry,
                                "hard_stop_risk_pct": pred_risk,
                                "delay_minutes": 10,
                                **res2,
                            })

    if not rows:
        raise RuntimeError("No trades generated.")

    trades = pd.DataFrame(rows)
    trades["setup_time"] = normalize_time(trades["setup_time"])
    trades["entry_time"] = normalize_time(trades["entry_time"])
    trades["exit_time"] = normalize_time(trades["exit_time"])

    trades.to_csv(a.output_dir / "trades.csv", index=False)

    validation_start = trades["setup_time"].max() - pd.Timedelta(days=a.validation_days)

    out_rows = []
    for (method, threshold), g in trades.groupby(["method", "threshold"]):
        allm = summarize(g)
        val = g[g["setup_time"] >= validation_start]
        vm = summarize(val)

        out_rows.append({
            "method": method,
            "threshold": threshold,
            **allm,
            "validation_trades": vm["trades"],
            "validation_expectancy": vm["expectancy"],
            "validation_win_rate": vm["win_rate"],
            "validation_profit_factor": vm["profit_factor"],
            "validation_max_drawdown": vm["max_drawdown"],
            "validation_avg_risk": vm["avg_risk"],
            "validation_pct_risk_le2": vm["pct_risk_le2"],
            "validation_avg_delay": vm["avg_delay"],
        })

    results = pd.DataFrame(out_rows)

    predictor_results = results[
        results["method"].isin(["PREDICT_10M", "PREDICT_10M_STRICT"])
    ].copy()

    predictor_results["research_score"] = (
        0.45 * predictor_results["validation_expectancy"].rank(pct=True)
        + 0.25 * predictor_results["validation_profit_factor"].replace(np.inf, 20).rank(pct=True)
        + 0.20 * predictor_results["validation_pct_risk_le2"].rank(pct=True)
        + 0.10 * (-predictor_results["validation_avg_risk"]).rank(pct=True)
    )

    predictor_results = predictor_results.sort_values(
        ["research_score", "validation_expectancy"],
        ascending=False,
    )
    predictor_results.to_csv(
        a.output_dir / "predictor_threshold_results.csv",
        index=False,
    )

    stability = results[
        results["method"].eq("BASE_15M_NEXT")
    ].sort_values("threshold")
    stability.to_csv(
        a.output_dir / "final_next_threshold_stability.csv",
        index=False,
    )

    # Rescue: baseline 15m risk >2%, predictor risk <=2%
    rescue_source = trades[
        trades["method"].isin(["PREDICT_10M", "PREDICT_10M_STRICT"])
        & (trades["baseline15_risk"] > 0.02)
    ].copy()

    rescue_rows = []
    for (method, threshold), g in rescue_source.groupby(["method", "threshold"]):
        rescue_rows.append({
            "method": method,
            "threshold": threshold,
            "trades": len(g),
            "avg_baseline_risk": float(g["baseline15_risk"].mean()),
            "avg_predictor_risk": float(g["hard_stop_risk_pct"].mean()),
            "rescued_to_le2": int((g["hard_stop_risk_pct"] <= 0.02).sum()),
            "rescue_rate": float((g["hard_stop_risk_pct"] <= 0.02).mean()),
            "expectancy": float(g["net_return"].mean()),
            "profit_factor": profit_factor(g["net_return"]),
        })

    rescue = pd.DataFrame(rescue_rows)
    rescue.to_csv(a.output_dir / "rescue_results.csv", index=False)

    report = [
        "V11.2 FAST — 10m PREDICTOR OF 15m NEXT",
        "=" * 100,
        f"Validation starts: {validation_start.isoformat()}",
        "",
        "TOP 10m PREDICTOR RESULTS",
        "-" * 100,
        predictor_results.head(20).to_string(index=False),
        "",
        "FINAL 15m NEXT THRESHOLD STABILITY",
        "-" * 100,
        stability.to_string(index=False),
        "",
        "HIGH-RISK RESCUE",
        "-" * 100,
        rescue.head(20).to_string(index=False) if not rescue.empty else "None",
    ]

    (a.output_dir / "report.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nTOP 10m PREDICTOR RESULTS\n")
    print(
        predictor_results[
            [
                "method",
                "threshold",
                "validation_trades",
                "validation_expectancy",
                "validation_profit_factor",
                "validation_avg_risk",
                "validation_pct_risk_le2",
                "research_score",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print("\n15m NEXT THRESHOLD STABILITY\n")
    print(
        stability[
            [
                "threshold",
                "validation_trades",
                "validation_expectancy",
                "validation_profit_factor",
                "validation_avg_risk",
            ]
        ].to_string(index=False)
    )

    print("\nOutput:", a.output_dir)


if __name__ == "__main__":
    main()
