#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11.1 TRUE 5m NEXT SCORE CHAMPIONSHIP — FAST
============================================

User hypothesis
---------------
1) 15m finds the rebound/setup.
2) The 15m rebound candle closes and becomes known.
3) Do NOT wait another full 15m candle.
4) Look only at the FIRST 5m candle after the 15m setup closes.
5) If that 5m "Next" candle is strong enough, enter immediately at its close.

This script tests exactly that idea.

No look-ahead
-------------
At the 5m entry time, the script uses ONLY:
- the already-closed 15m setup candle
- the already-closed first 5m candle after it
It never uses the later 15m candle to decide the 5m entry.

Baseline
--------
BASE_15M_NEXT:
- wait the full next 15m candle
- enter at that next 15m close only if its 15m Next quality >= threshold

5m methods
----------
FIRST_5M_NEXT:
- inspect only the first 5m candle after the 15m setup
- enter if 5m Next score >= threshold

FIRST_OF_3_5M:
- inspect up to the 3 constituent 5m candles of the next 15m period
- enter on the first one whose 5m Next score >= threshold
- this shows whether "one chance only" or "within 15m" works better

FAST defaults
-------------
- top 60 robust symbols
- 45d 5m
- 60d 15m
- 90d 1h
- validation = last 15d

Quick Railway run:
  python v11_1_true_5m_next_fast.py

Full universe:
  python v11_1_true_5m_next_fast.py --full
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


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine-file",
        type=Path,
        default=Path("crypto_tide_engine_v10_8_2_2.py"),
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("v10_bundle"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("v11_1_true_5m_next_fast"),
    )
    p.add_argument("--max-symbols", type=int, default=60)
    p.add_argument("--full", action="store_true")
    p.add_argument("--days-5m", type=int, default=45)
    p.add_argument("--days-15m", type=int, default=60)
    p.add_argument("--days-1h", type=int, default=90)
    p.add_argument("--validation-days", type=int, default=15)
    p.add_argument("--hold-hours", type=float, default=4.0)
    p.add_argument("--structure-buffer-atr", type=float, default=0.50)

    # 5m thresholds are deliberately broad because 5m score distribution
    # is not assumed to match the old 15m 95-point threshold.
    p.add_argument(
        "--thresholds",
        type=str,
        default="55,60,65,70,75,80,85,90",
    )
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def sf(v: Any, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def normalize_time(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series, utc=True, errors="coerce")
        .astype("datetime64[ns, UTC]")
    )


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
        raise FileNotFoundError(
            "No compatible Tide engine found. Tried: "
            + ", ".join(str(x) for x in candidates)
        )

    print("Using Tide engine:", selected)

    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIDE_DATA_DIR"] = str(data_dir.resolve())

    spec = importlib.util.spec_from_file_location("v111_true_engine", selected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import engine: {selected}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 5m scoring
# ============================================================

def prepare_5m(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["open_time"] = normalize_time(x["open_time"])
    x = (
        x.dropna(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    c = x["close"].astype(float)
    v = x["volume"].astype(float)

    rng = (h - l).replace(0, np.nan)
    body = (c - o) / rng
    close_pos = (c - l) / rng

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean()

    ret_atr = (c - o) / atr.replace(0, np.nan)
    volume_mult = v / v.rolling(20, min_periods=10).median()
    prior_high_3 = h.shift(1).rolling(3, min_periods=2).max()
    micro_break = (c > prior_high_3).astype(float)

    # 5m Next Score:
    # strong positive body + close near high + range expansion + volume + micro break.
    # It is intentionally a ranking score, and we test many thresholds.
    body_score = (body.clip(0, 1) * 100)
    close_score = (close_pos.clip(0, 1) * 100)
    impulse_score = (ret_atr.clip(0, 1.5) / 1.5 * 100)
    volume_score = (volume_mult.clip(0, 3) / 3 * 100)
    break_score = micro_break * 100

    x["next5_score"] = (
        0.30 * impulse_score
        + 0.25 * body_score
        + 0.20 * close_score
        + 0.15 * volume_score
        + 0.10 * break_score
    ).clip(0, 100)

    x["volume_mult_5m"] = volume_mult
    x["close_pos_5m"] = close_pos
    x["ret_atr_5m"] = ret_atr
    x["close_time"] = x["open_time"] + pd.Timedelta(minutes=5)

    return x


# ============================================================
# 15m setup and baseline next score
# ============================================================

def is_raw_15m_setup(row: pd.Series) -> bool:
    """
    Use the raw Tide signal itself as the 15m setup.
    This is earlier than waiting for the later scored/confirmed row.
    """
    return bool(row.get("signal", False))


def next15_score_from_candle(next_row: pd.Series, setup_row: pd.Series) -> float:
    """
    A simple forward 15m continuation score used ONLY for baseline comparison.
    It is computed from the fully closed next 15m candle.

    The 5m method never sees this value before entry.
    """
    o = sf(next_row.get("open"))
    h = sf(next_row.get("high"))
    l = sf(next_row.get("low"))
    c = sf(next_row.get("close"))
    v = sf(next_row.get("volume"))
    if not all(np.isfinite(z) for z in [o, h, l, c, v]) or h <= l:
        return np.nan

    rng = h - l
    body = max(0.0, min(1.0, (c - o) / rng))
    close_pos = max(0.0, min(1.0, (c - l) / rng))

    setup_atr = sf(
        setup_row.get(
            "atr14",
            setup_row.get("confirmation_signal_atr"),
        )
    )
    impulse = (
        max(0.0, min(1.5, (c - o) / setup_atr)) / 1.5
        if np.isfinite(setup_atr) and setup_atr > 0
        else 0.0
    )

    volume_ref = sf(setup_row.get("volume"))
    volume_score = (
        max(0.0, min(3.0, v / volume_ref)) / 3.0
        if np.isfinite(volume_ref) and volume_ref > 0
        else 0.0
    )

    return float(
        np.clip(
            100
            * (
                0.35 * impulse
                + 0.25 * body
                + 0.25 * close_pos
                + 0.15 * volume_score
            ),
            0,
            100,
        )
    )


# ============================================================
# Structural stop / exit
# ============================================================

def structural_stop(
    setup: pd.Series,
    entry_price: float,
    buffer_atr: float,
):
    atr = sf(
        setup.get(
            "confirmation_signal_atr",
            setup.get("atr14"),
        )
    )
    sig_low = sf(
        setup.get(
            "confirmation_signal_low",
            setup.get("low"),
        )
    )

    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(sig_low):
        return np.nan, np.nan

    stop = sig_low - buffer_atr * atr
    risk = (
        (entry_price - stop) / entry_price
        if entry_price > 0
        else np.nan
    )
    return stop, risk


def simulate_trade(
    engine,
    df5: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    hold_hours: float,
):
    if not np.isfinite(stop_price) or stop_price >= entry_price:
        return None

    end = min(
        entry_idx + int(round(hold_hours * 12)),
        len(df5) - 1,
    )

    exit_idx = end
    exit_price = float(df5.iloc[end]["close"])
    reason = "fixed_4h"

    for j in range(entry_idx + 1, end + 1):
        if float(df5.iloc[j]["low"]) <= stop_price:
            exit_idx = j
            exit_price = stop_price
            reason = "initial_stop"
            break

    path = df5.iloc[entry_idx : exit_idx + 1]

    return {
        "net_return": (
            exit_price / entry_price
            - 1
            - float(engine.FEE_SLIPPAGE)
        ),
        "mfe_pct": float(path["high"].max()) / entry_price - 1,
        "mae_pct": float(path["low"].min()) / entry_price - 1,
        "exit_reason": reason,
        "exit_time": pd.Timestamp(df5.iloc[exit_idx]["open_time"])
        + pd.Timedelta(minutes=5),
    }


# ============================================================
# Metrics
# ============================================================

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
        return {
            "trades": 0,
            "expectancy": np.nan,
            "win_rate": np.nan,
            "pf": np.nan,
            "mdd": np.nan,
            "avg_risk": np.nan,
            "pct_risk_le2": np.nan,
            "avg_mfe": np.nan,
            "avg_mae": np.nan,
            "avg_delay_min": np.nan,
        }

    r = g["net_return"].astype(float)
    risk = g["hard_stop_risk_pct"].astype(float)

    return {
        "trades": len(g),
        "expectancy": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "pf": profit_factor(r),
        "mdd": max_drawdown(r),
        "avg_risk": float(risk.mean()),
        "pct_risk_le2": float((risk <= 0.02).mean()),
        "avg_mfe": float(g["mfe_pct"].mean()),
        "avg_mae": float(g["mae_pct"].mean()),
        "avg_delay_min": float(g["delay_minutes"].mean()),
    }


# ============================================================
# Main
# ============================================================

def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [
        float(x.strip())
        for x in a.thresholds.split(",")
        if x.strip()
    ]

    engine = load_engine(
        a.engine_file,
        a.output_dir / "engine_research_data",
    )

    universe = pd.read_csv(
        a.bundle_dir / "stage2_full_results.csv"
    )

    if "eligible" in universe.columns:
        universe = universe[
            universe["eligible"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]

    if "score" in universe.columns:
        universe = universe.sort_values(
            "score",
            ascending=False,
        )

    if not a.full:
        universe = universe.head(
            a.max_symbols
        )

    print("=" * 100)
    print("V11.1 TRUE 5m NEXT SCORE CHAMPIONSHIP — FAST")
    print("=" * 100)
    print("Symbols:", len(universe))
    print("Thresholds:", thresholds)

    rows = []

    for n, symbol in enumerate(
        universe["symbol"].astype(str),
        1,
    ):
        print(
            f"[{n}/{len(universe)}] {symbol}",
            flush=True,
        )

        try:
            df5 = prepare_5m(
                engine.fetch_klines(
                    symbol,
                    "5",
                    a.days_5m,
                )
            )

            df15 = engine.model_frame(
                engine.fetch_klines(
                    symbol,
                    "15",
                    a.days_15m,
                ),
                engine.fetch_klines(
                    symbol,
                    "60",
                    a.days_1h,
                ),
            )
            df15["open_time"] = normalize_time(
                df15["open_time"]
            )
            df15 = (
                df15.sort_values("open_time")
                .reset_index(drop=True)
            )

        except Exception as exc:
            print(
                "  FAILED:",
                type(exc).__name__,
                exc,
            )
            continue

        close5 = df5["close_time"].to_numpy()
        setups = 0

        for i in range(len(df15) - 1):
            setup = df15.iloc[i]

            if not is_raw_15m_setup(setup):
                continue

            setups += 1

            setup_close_time = (
                pd.Timestamp(setup["open_time"])
                + pd.Timedelta(minutes=15)
            )

            # First 3 fully closed 5m bars AFTER the 15m setup closes.
            candidate_ids = np.flatnonzero(
                close5 > np.datetime64(
                    setup_close_time.to_datetime64()
                )
            )

            if len(candidate_ids) < 3:
                continue

            first3 = [
                int(x)
                for x in candidate_ids[:3]
            ]

            # Baseline next 15m candle.
            next15 = df15.iloc[i + 1]
            next15_close_time = (
                pd.Timestamp(next15["open_time"])
                + pd.Timedelta(minutes=15)
            )
            baseline_next15_score = next15_score_from_candle(
                next15,
                setup,
            )

            # Map baseline entry to the first 5m close at/after next15 close.
            base_ids = np.flatnonzero(
                close5
                >= np.datetime64(
                    next15_close_time.to_datetime64()
                )
            )
            if len(base_ids) == 0:
                continue

            base_idx = int(base_ids[0])
            base_entry = float(
                df5.iloc[base_idx]["close"]
            )
            base_stop, base_risk = structural_stop(
                setup,
                base_entry,
                a.structure_buffer_atr,
            )

            common = {
                "symbol": symbol,
                "setup_time": setup_close_time,
                "setup_low": sf(setup.get("low")),
                "setup_atr": sf(setup.get("atr14")),
                "baseline_next15_score": baseline_next15_score,
                "baseline_entry": base_entry,
                "baseline_risk": base_risk,
                "first5_score": float(
                    df5.iloc[first3[0]]["next5_score"]
                ),
                "max3_5m_score": float(
                    max(
                        df5.iloc[x]["next5_score"]
                        for x in first3
                    )
                ),
            }

            for threshold in thresholds:
                # --------------------------------------------------
                # BASE_15M_NEXT
                # --------------------------------------------------
                if (
                    np.isfinite(baseline_next15_score)
                    and baseline_next15_score >= threshold
                ):
                    result = simulate_trade(
                        engine,
                        df5,
                        base_idx,
                        base_entry,
                        base_stop,
                        a.hold_hours,
                    )
                    if result is not None:
                        rows.append(
                            {
                                **common,
                                "method": "BASE_15M_NEXT",
                                "threshold": threshold,
                                "entry_time": next15_close_time,
                                "entry_price": base_entry,
                                "hard_stop_risk_pct": base_risk,
                                "delay_minutes": 15,
                                "trigger_score": baseline_next15_score,
                                **result,
                            }
                        )

                # --------------------------------------------------
                # FIRST_5M_NEXT
                # --------------------------------------------------
                first_idx = first3[0]
                first_score = float(
                    df5.iloc[first_idx]["next5_score"]
                )

                if first_score >= threshold:
                    entry = float(
                        df5.iloc[first_idx]["close"]
                    )
                    stop, risk = structural_stop(
                        setup,
                        entry,
                        a.structure_buffer_atr,
                    )
                    result = simulate_trade(
                        engine,
                        df5,
                        first_idx,
                        entry,
                        stop,
                        a.hold_hours,
                    )
                    if result is not None:
                        rows.append(
                            {
                                **common,
                                "method": "FIRST_5M_NEXT",
                                "threshold": threshold,
                                "entry_time": df5.iloc[first_idx]["close_time"],
                                "entry_price": entry,
                                "hard_stop_risk_pct": risk,
                                "delay_minutes": 5,
                                "trigger_score": first_score,
                                **result,
                            }
                        )

                # --------------------------------------------------
                # FIRST_OF_3_5M
                # --------------------------------------------------
                hit = None
                for k, idx5 in enumerate(
                    first3,
                    start=1,
                ):
                    score5 = float(
                        df5.iloc[idx5]["next5_score"]
                    )
                    if score5 >= threshold:
                        hit = (k, idx5, score5)
                        break

                if hit is not None:
                    k, idx5, score5 = hit
                    entry = float(
                        df5.iloc[idx5]["close"]
                    )
                    stop, risk = structural_stop(
                        setup,
                        entry,
                        a.structure_buffer_atr,
                    )
                    result = simulate_trade(
                        engine,
                        df5,
                        idx5,
                        entry,
                        stop,
                        a.hold_hours,
                    )
                    if result is not None:
                        rows.append(
                            {
                                **common,
                                "method": "FIRST_OF_3_5M",
                                "threshold": threshold,
                                "entry_time": df5.iloc[idx5]["close_time"],
                                "entry_price": entry,
                                "hard_stop_risk_pct": risk,
                                "delay_minutes": 5 * k,
                                "trigger_score": score5,
                                **result,
                            }
                        )

        print(
            "  raw 15m setups:",
            setups,
        )

    if not rows:
        raise RuntimeError(
            "No qualifying trades generated."
        )

    trades = pd.DataFrame(rows)
    trades["setup_time"] = normalize_time(
        trades["setup_time"]
    )
    trades["entry_time"] = normalize_time(
        trades["entry_time"]
    )
    trades["exit_time"] = normalize_time(
        trades["exit_time"]
    )

    trades.to_csv(
        a.output_dir / "trades.csv",
        index=False,
    )

    validation_start = (
        trades["setup_time"].max()
        - pd.Timedelta(
            days=a.validation_days
        )
    )

    result_rows = []

    for (method, threshold), g in trades.groupby(
        ["method", "threshold"]
    ):
        train = g[
            g["setup_time"]
            < validation_start
        ]
        val = g[
            g["setup_time"]
            >= validation_start
        ]

        A = summarize(g)
        T = summarize(train)
        V = summarize(val)

        result_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "trades": A["trades"],
                "expectancy": A["expectancy"],
                "win_rate": A["win_rate"],
                "profit_factor": A["pf"],
                "max_drawdown": A["mdd"],
                "avg_risk": A["avg_risk"],
                "pct_risk_le2": A["pct_risk_le2"],
                "avg_delay_min": A["avg_delay_min"],
                "train_trades": T["trades"],
                "train_expectancy": T["expectancy"],
                "validation_trades": V["trades"],
                "validation_expectancy": V["expectancy"],
                "validation_win_rate": V["win_rate"],
                "validation_profit_factor": V["pf"],
                "validation_max_drawdown": V["mdd"],
                "validation_avg_risk": V["avg_risk"],
                "validation_pct_risk_le2": V["pct_risk_le2"],
                "validation_avg_delay_min": V["avg_delay_min"],
            }
        )

    results = pd.DataFrame(
        result_rows
    )

    results["research_score"] = (
        0.40
        * results[
            "validation_expectancy"
        ].rank(pct=True)
        + 0.20
        * results[
            "validation_profit_factor"
        ]
        .replace(np.inf, 20)
        .rank(pct=True)
        + 0.18
        * results[
            "validation_pct_risk_le2"
        ].rank(pct=True)
        + 0.12
        * (
            -results[
                "validation_avg_risk"
            ]
        ).rank(pct=True)
        + 0.10
        * (
            -results[
                "validation_avg_delay_min"
            ]
        ).rank(pct=True)
    )

    results = results.sort_values(
        [
            "research_score",
            "validation_expectancy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results.to_csv(
        a.output_dir
        / "threshold_results.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Direct high-risk rescue comparison
    # ------------------------------------------------------------
    high_risk = trades[
        trades["baseline_risk"] > 0.02
    ].copy()

    rescue_rows = []
    for (method, threshold), g in high_risk.groupby(
        ["method", "threshold"]
    ):
        rescue_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "trades": len(g),
                "avg_baseline_risk": g[
                    "baseline_risk"
                ].mean(),
                "avg_new_risk": g[
                    "hard_stop_risk_pct"
                ].mean(),
                "rescued_to_le2": int(
                    (
                        g["hard_stop_risk_pct"]
                        <= 0.02
                    ).sum()
                ),
                "rescue_rate": float(
                    (
                        g["hard_stop_risk_pct"]
                        <= 0.02
                    ).mean()
                ),
                "expectancy": g[
                    "net_return"
                ].mean(),
                "profit_factor": profit_factor(
                    g["net_return"]
                ),
                "avg_delay_min": g[
                    "delay_minutes"
                ].mean(),
            }
        )

    rescue = pd.DataFrame(
        rescue_rows
    ).sort_values(
        [
            "rescue_rate",
            "expectancy",
        ],
        ascending=False,
    )

    rescue.to_csv(
        a.output_dir
        / "high_risk_rescue.csv",
        index=False,
    )

    report = [
        "V11.1 TRUE 5m NEXT SCORE CHAMPIONSHIP — FAST",
        "=" * 100,
        f"Validation starts: {validation_start.isoformat()}",
        "",
        "This is the requested logic:",
        "15m raw setup closes -> inspect first 5m Next candle -> enter at 5m close if score passes.",
        "No future 15m information is used for the 5m decision.",
        "",
        "TOP RESULTS",
        "-" * 100,
        results.head(25).to_string(index=False),
        "",
        "HIGH BASELINE-RISK RESCUE",
        "-" * 100,
        rescue.head(25).to_string(index=False),
    ]

    (
        a.output_dir
        / "report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nTOP RESULTS\n")
    print(
        results[
            [
                "method",
                "threshold",
                "validation_trades",
                "validation_expectancy",
                "validation_profit_factor",
                "validation_avg_risk",
                "validation_pct_risk_le2",
                "validation_avg_delay_min",
                "research_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nOutput:", a.output_dir)


if __name__ == "__main__":
    main()
