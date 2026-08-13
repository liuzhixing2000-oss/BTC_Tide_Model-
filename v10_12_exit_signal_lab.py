#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10.12 EXIT SIGNAL LAB — 15m REVERSAL / STRUCTURE / MOMENTUM
============================================

Purpose
-------
Apply the CURRENT V10.9 Dynamic Risk grading/filter logic retrospectively
to the last 180 days, then simulate the portfolio in chronological order.

Important scope
---------------
This is a "current rules applied to history" test:
- current robust-eligible symbols from v10_bundle/stage2_full_results.csv
- current V10.9 signal scoring / A+, A, A-, B+ grading
- current Dynamic Risk sizing
- current structural stop + fixed-4h exit
- current portfolio limits (positions / margin / total open risk / daily loss)

It does NOT reconstruct which symbols would have been Stage2-eligible on each
historical day. That would require a much heavier walk-forward universe rebuild.
So this is intentionally a retrospective test of today's filters.

Policies reported
-----------------
CORE
    A-, A, A+ only. This matches current production-trade grades.

CORE_PLUS_BPLUS
    A-, A, A+, B+. This is a what-if where B+ is actually traded with the same
    dynamic-risk sizing. In current live code B+ is secondary/watch tracking,
    so treat this policy as an optional expansion test, not exact live behavior.

Railway
-------
python v10_9_current_filters_180d_backtest.py

Useful faster test:
python v10_9_current_filters_180d_backtest.py --max-symbols 60

Outputs
-------
v10_9_180d_backtest/
  candidate_signals.csv
  portfolio_trades.csv
  policy_summary.csv
  grade_summary.csv
  monthly_summary.csv
  report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine-file",
        type=Path,
        default=Path("crypto_tide_engine_v10_9_dynamic_risk.py"),
    )
    p.add_argument("--bundle-dir", type=Path, default=Path("v10_bundle"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("v10_12_exit_signal_lab"),
    )
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--days-1h", type=int, default=210)
    p.add_argument("--days-4h", type=int, default=240)
    p.add_argument(
        "--max-hold-hours",
        type=float,
        default=24.0,
        help="Safety cap for non-time-based exit tests.",
    )
    p.add_argument(
        "--required-score",
        type=float,
        default=float(os.getenv("BACKTEST_REQUIRED_SIGNAL_SCORE", "70")),
        help="Historical market-layer required score. Default 70.",
    )
    p.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="0 = all current robust-eligible symbols.",
    )
    p.add_argument(
        "--initial-capital",
        type=float,
        default=0.0,
        help="0 = engine PORTFOLIO_CAPITAL_USDT.",
    )
    return p.parse_args()


def load_engine(path: Path, output_dir: Path):
    candidates = [
        path,
        Path("crypto_tide_engine_v10_9_dynamic_risk.py"),
        Path("crypto_tide_engine_v10_8_2_2.py"),
    ]
    selected = next((x for x in candidates if x.exists()), None)
    if selected is None:
        raise FileNotFoundError(
            "Cannot find V10.9 engine. Upload "
            "crypto_tide_engine_v10_9_dynamic_risk.py to repo root."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep research cache separate from live position state, while still
    # allowing Bybit candle cache persistence if Railway Volume is mounted.
    research_data = output_dir / "engine_research_data"
    research_data.mkdir(parents=True, exist_ok=True)
    os.environ["TIDE_DATA_DIR"] = str(research_data.resolve())

    print("Using engine:", selected, flush=True)
    spec = importlib.util.spec_from_file_location("v109_backtest_engine", selected)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# Historical HTF context, same definition as live engine
# ---------------------------------------------------------------------

def prepare_htf(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    x = df.copy()
    x["open_time"] = pd.to_datetime(x["open_time"], utc=True)
    x["close_time"] = x["open_time"] + pd.Timedelta(hours=hours)
    c = x["close"].astype(float)
    x["ema50_hist"] = c.ewm(span=50, adjust=False).mean()
    x["ema200_hist"] = c.ewm(span=200, adjust=False).mean()
    x["ema50_slope3_hist"] = x["ema50_hist"].pct_change(3)

    up = (
        (c > x["ema50_hist"])
        & (x["ema50_hist"] > x["ema200_hist"])
        & (x["ema50_slope3_hist"] > 0)
    )
    down = (
        (c < x["ema50_hist"])
        & (x["ema50_hist"] < x["ema200_hist"])
        & (x["ema50_slope3_hist"] < 0)
    )
    x["trend_hist"] = np.select([up, down], ["UP", "DOWN"], default="MIXED")

    slope = x["ema50_slope3_hist"].fillna(0.0).astype(float)
    strength = (
        50
        + 20 * np.tanh(slope * 200)
        + 15 * np.tanh((c / x["ema50_hist"] - 1.0) * 50)
        + 15 * np.tanh(
            (x["ema50_hist"] / x["ema200_hist"] - 1.0) * 30
        )
    )
    x["strength_hist"] = np.clip(strength, 0, 100)

    return x[
        [
            "close_time",
            "trend_hist",
            "strength_hist",
            "ema50_hist",
            "ema200_hist",
        ]
    ].dropna(subset=["close_time"]).sort_values("close_time")


def htf_at(prepared: pd.DataFrame, signal_close: pd.Timestamp) -> dict:
    if prepared.empty:
        return {"trend": "MIXED", "strength": 50.0}

    times = prepared["close_time"].values
    # UTC-naive nanoseconds comparison via pandas searchsorted is robust.
    idx = prepared["close_time"].searchsorted(signal_close, side="right") - 1
    if idx < 204:
        return {"trend": "MIXED", "strength": 50.0}
    r = prepared.iloc[int(idx)]
    return {
        "trend": str(r["trend_hist"]),
        "strength": float(r["strength_hist"]),
    }


# ---------------------------------------------------------------------
# Exact production-like 4h trade path
# ---------------------------------------------------------------------

@dataclass
class TradePath:
    exit_time: pd.Timestamp
    exit_price: float
    net_return: float
    exit_reason: str
    bars_held: int
    mfe_pct: float
    mae_pct: float


def prepare_exit_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    c = x["close"].astype(float)
    v = x["volume"].astype(float)

    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()

    x["_body"] = body
    x["_upper_wick"] = h - np.maximum(o, c)
    x["_range"] = rng
    x["_close_pos"] = ((c - l) / rng).clip(0, 1)
    x["_ema20"] = c.ewm(span=20, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    x["_macd_hist"] = macd - macd_sig
    x["_vol_med20"] = v.rolling(20, min_periods=10).median()
    return x


def make_trade_path(engine, df, entry_idx, idx, exit_price, entry, highest, lowest, reason):
    return TradePath(
        exit_time=pd.Timestamp(df.iloc[idx]["open_time"]) + pd.Timedelta(minutes=15),
        exit_price=float(exit_price),
        net_return=float(exit_price) / entry - 1 - float(engine.FEE_SLIPPAGE),
        exit_reason=reason,
        bars_held=int(idx - entry_idx),
        mfe_pct=highest / entry - 1,
        mae_pct=lowest / entry - 1,
    )


def simulate_exit_method(engine, df: pd.DataFrame, entry_idx: int, risk: dict, method: str, max_hold_hours: float):
    """
    SAME entry and SAME V10.9 structural stop for every method.

    BASE_FIXED_4H
      Existing control: exit after 16 completed 15m bars.

    PINBAR_CONFIRM
      Hold until a bearish upper-wick rejection candle is confirmed by the next
      15m close breaking its low.

    BEAR_ENGULF_VOL
      Hold until bearish engulfing + volume >= 20-bar median.

    STRUCTURE_BREAK
      Hold until a 15m close breaks below the most recent confirmed 5-bar pivot low.

    MOMENTUM_FADE
      Once trade has reached at least +1% MFE, exit when close < EMA20 and
      MACD histogram is negative for 2 consecutive completed bars.

    HYBRID_2OF4
      Exit when any 2 of the 4 signals above are simultaneously true.

    HYBRID_STRICT
      Exit immediately on structure break; otherwise require 2 of
      pinbar-confirm / engulf-volume / momentum-fade.

    Non-baseline methods have no fixed 4h exit. max_hold_hours is only a
    research safety cap (default 24h).
    """
    x = prepare_exit_indicators(df)
    entry = float(x.iloc[entry_idx]["close"])
    atr = float(risk["atr"])
    hard = float(risk["hard_stop"])
    if not np.isfinite(hard) or hard >= entry or not np.isfinite(atr) or atr <= 0:
        return None

    current_stop = hard
    highest = entry
    lowest = entry
    max_bars = 16 if method == "BASE_FIXED_4H" else max(1, int(round(max_hold_hours * 4)))
    last = min(entry_idx + max_bars, len(x) - 1)

    last_confirmed_pivot = np.nan

    for idx in range(entry_idx + 1, last + 1):
        row = x.iloc[idx]
        prev = x.iloc[idx - 1]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        open_ = float(row["open"])

        highest = max(highest, high)
        lowest = min(lowest, low)
        mfe = highest / entry - 1

        # Existing structural stop first
        if low <= current_stop:
            return make_trade_path(
                engine, x, entry_idx, idx, current_stop, entry, highest, lowest,
                "dynamic_structure_stop" if current_stop > hard else "hard_structure_stop"
            )

        # Confirmed 5-bar pivot at i-2 (known only now)
        pivot = idx - 2
        if pivot >= entry_idx + 2 and pivot + 2 < len(x):
            plow = float(x.iloc[pivot]["low"])
            left_min = float(x.iloc[pivot - 2:pivot]["low"].min())
            right_min = float(x.iloc[pivot + 1:pivot + 3]["low"].min())
            if plow < left_min and plow <= right_min:
                last_confirmed_pivot = plow
                candidate = plow - 0.20 * atr
                if candidate > current_stop and candidate < close:
                    current_stop = candidate

        held = idx - entry_idx

        if method == "BASE_FIXED_4H":
            if held >= 16:
                return make_trade_path(
                    engine, x, entry_idx, idx, close, entry, highest, lowest, "fixed_4h"
                )
            continue

        # 1. Upper-wick rejection on prior bar + confirmation now
        prev_rng = float(prev["_range"]) if np.isfinite(prev["_range"]) else np.nan
        prev_body = float(prev["_body"]) if np.isfinite(prev["_body"]) else np.nan
        prev_upper = float(prev["_upper_wick"]) if np.isfinite(prev["_upper_wick"]) else np.nan
        prev_close_pos = float(prev["_close_pos"]) if np.isfinite(prev["_close_pos"]) else np.nan
        pinbar_prev = (
            np.isfinite(prev_rng) and prev_rng > 0
            and np.isfinite(prev_body) and np.isfinite(prev_upper)
            and prev_upper / prev_rng >= 0.45
            and prev_upper >= max(2.0 * prev_body, 0.10 * atr)
            and prev_close_pos <= 0.55
        )
        pinbar_confirm = pinbar_prev and close < float(prev["low"])

        # 2. Bearish engulfing + volume confirmation
        prev_open = float(prev["open"])
        prev_close = float(prev["close"])
        bearish_now = close < open_
        prev_bullish = prev_close > prev_open
        engulf = bearish_now and prev_bullish and open_ >= prev_close and close <= prev_open
        vol_med = float(row["_vol_med20"]) if np.isfinite(row["_vol_med20"]) else np.nan
        engulf_vol = engulf and np.isfinite(vol_med) and float(row["volume"]) >= vol_med

        # 3. Break of most recent confirmed pivot low
        structure_break = (
            np.isfinite(last_confirmed_pivot)
            and close < float(last_confirmed_pivot)
        )

        # 4. Momentum fade after trade had at least +1% MFE
        ema20 = float(row["_ema20"]) if np.isfinite(row["_ema20"]) else np.nan
        hist_now = float(row["_macd_hist"]) if np.isfinite(row["_macd_hist"]) else np.nan
        hist_prev = float(prev["_macd_hist"]) if np.isfinite(prev["_macd_hist"]) else np.nan
        momentum_fade = (
            mfe >= 0.01
            and np.isfinite(ema20) and close < ema20
            and np.isfinite(hist_now) and np.isfinite(hist_prev)
            and hist_now < 0 and hist_prev < 0
        )

        if method == "PINBAR_CONFIRM" and pinbar_confirm:
            return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, "pinbar_confirm")

        if method == "BEAR_ENGULF_VOL" and engulf_vol:
            return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, "bear_engulf_vol")

        if method == "STRUCTURE_BREAK" and structure_break:
            return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, "structure_break")

        if method == "MOMENTUM_FADE" and momentum_fade:
            return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, "momentum_fade")

        score = int(pinbar_confirm) + int(engulf_vol) + int(structure_break) + int(momentum_fade)

        if method == "HYBRID_2OF4" and score >= 2:
            return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, "hybrid_2of4")

        if method == "HYBRID_STRICT":
            other = int(pinbar_confirm) + int(engulf_vol) + int(momentum_fade)
            if structure_break or other >= 2:
                reason = "hybrid_structure_break" if structure_break else "hybrid_strict_2of3"
                return make_trade_path(engine, x, entry_idx, idx, close, entry, highest, lowest, reason)

    row = x.iloc[last]
    close = float(row["close"])
    return make_trade_path(
        engine, x, entry_idx, last, close, entry, highest, lowest, "research_time_cap"
    )


EXIT_METHODS = (
    "BASE_FIXED_4H",
    "PINBAR_CONFIRM",
    "BEAR_ENGULF_VOL",
    "STRUCTURE_BREAK",
    "MOMENTUM_FADE",
    "HYBRID_2OF4",
    "HYBRID_STRICT",
)


# ---------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------

def collect_symbol_candidates(
    engine,
    symbol: str,
    meta: dict,
    days: int,
    days1h: int,
    days4h: int,
    required_score: float,
    max_hold_hours: float,
) -> list[dict]:
    df15_raw = engine.fetch_klines(symbol, "15", days)
    df1h_raw = engine.fetch_klines(symbol, "60", days1h)
    df4h_raw = engine.fetch_klines(symbol, "240", days4h)

    df = engine.model_frame(df15_raw, df1h_raw)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    h1p = prepare_htf(df1h_raw, 1)
    h4p = prepare_htf(df4h_raw, 4)

    out = []

    # Current live code grades every completed scored setup, not just df["signal"].
    scored_indices = np.flatnonzero(
        pd.notna(df["combined_setup_score"]).to_numpy()
    )

    for entry_idx in scored_indices:
        latest = df.iloc[int(entry_idx)]
        signal_close = pd.Timestamp(latest["open_time"]) + pd.Timedelta(minutes=15)

        if entry_idx + 4 >= len(df):
            continue

        historical_score = meta.get("historical_score")
        score = float(engine.signal_score(latest, historical_score))

        assessment_meta = {
            **meta,
            "minimum_signal_score": float(required_score),
        }
        assessment = engine.production_entry_assessment(
            latest, score, assessment_meta
        )

        h1 = htf_at(h1p, signal_close)
        h4 = htf_at(h4p, signal_close)
        htf = {
            "h1_trend": h1["trend"],
            "h1_strength": h1["strength"],
            "h4_trend": h4["trend"],
            "h4_strength": h4["strength"],
        }
        grade = engine.tide_grade(latest, score, assessment, htf)

        if grade["grade"] not in {"A+", "A", "A-", "B+"}:
            continue

        risk = assessment["risk"]
        sizing = engine.dynamic_position_size(score, risk)
        for exit_method in EXIT_METHODS:
            path = simulate_exit_method(
                engine, df, int(entry_idx), risk, exit_method, max_hold_hours
            )
            if path is None:
                continue

            out.append(
              {
                "exit_method": exit_method,
                "symbol": symbol,
                "entry_time": signal_close,
                "exit_time": path.exit_time,
                "grade": grade["grade"],
                "grade_label": grade["label"],
                "entry_price": float(latest["close"]),
                "exit_price": path.exit_price,
                "net_return": path.net_return,
                "exit_reason": path.exit_reason,
                "bars_held": path.bars_held,
                "mfe_pct": path.mfe_pct,
                "mae_pct": path.mae_pct,
                "raw_quality": float(latest.get("raw_quality_score", np.nan)),
                "next_quality": float(assessment["next_quality"]),
                "combined_quality": float(assessment["combined_quality"]),
                "confirmation_tests": int(assessment["confirmation_tests"]),
                "signal_score": score,
                "volume_multiple": float(latest.get("volume_multiple", np.nan)),
                "hard_stop": float(risk["hard_stop"]),
                "hard_stop_risk_pct": float(risk["risk_pct"]),
                "h4_trend": h4["trend"],
                "h4_strength": h4["strength"],
                "h1_trend": h1["trend"],
                "h1_strength": h1["strength"],
                "planned_account_risk_pct": float(sizing["account_risk_pct"]),
                "effective_account_risk_pct": float(
                    sizing["effective_account_risk_pct"]
                ),
                "planned_loss_usdt": float(
                    sizing["effective_planned_loss_usdt"]
                ),
                "notional_usdt": float(sizing["suggested_notional_usdt"]),
                "leverage": float(sizing["suggested_leverage"]),
                "margin_usdt": float(sizing["suggested_margin_usdt"]),
                "strict_production": bool(assessment["production"]),
                "grade_reasons": " | ".join(grade.get("reasons", [])),
              }
            )

    return out


# ---------------------------------------------------------------------
# Chronological portfolio simulation
# ---------------------------------------------------------------------

def grade_rank(g: str) -> int:
    return {"B+": 1, "A-": 2, "A": 3, "A+": 4}.get(g, 0)


def portfolio_sim(
    engine,
    candidates: pd.DataFrame,
    allowed_grades: set[str],
    capital: float,
    policy_name: str,
):
    c = candidates[candidates["grade"].isin(allowed_grades)].copy()
    if c.empty:
        return pd.DataFrame(), {}

    c = c.sort_values(
        ["entry_time", "grade", "signal_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    active: list[dict] = []
    accepted = []
    daily_realised_loss: dict[str, float] = {}

    max_positions = int(engine.PORTFOLIO_MAX_POSITIONS)
    max_margin = capital * float(engine.PORTFOLIO_MAX_MARGIN_UTILISATION)
    max_open_risk = capital * float(engine.MAX_TOTAL_OPEN_RISK_PCT)
    max_daily_loss = capital * float(engine.MAX_DAILY_LOSS_PCT)
    sydney = ZoneInfo("Australia/Sydney")

    blocked = {
        "existing_symbol": 0,
        "position_count": 0,
        "margin": 0,
        "open_risk": 0,
        "daily_loss": 0,
    }

    for row in c.itertuples(index=False):
        t = pd.Timestamp(row.entry_time)

        # Realise positions closed by this entry time.
        still_active = []
        for p in active:
            if pd.Timestamp(p["exit_time"]) <= t:
                pnl = p["pnl_usdt"]
                if pnl < 0:
                    d = pd.Timestamp(p["exit_time"]).tz_convert(sydney).date().isoformat()
                    daily_realised_loss[d] = daily_realised_loss.get(d, 0.0) + (-pnl)
            else:
                still_active.append(p)
        active = still_active

        if any(p["symbol"] == row.symbol for p in active):
            blocked["existing_symbol"] += 1
            continue

        today = t.tz_convert(sydney).date().isoformat()
        if daily_realised_loss.get(today, 0.0) >= max_daily_loss - 1e-9:
            blocked["daily_loss"] += 1
            continue

        used_margin = sum(float(p["margin_usdt"]) for p in active)
        open_risk = sum(float(p["planned_loss_usdt"]) for p in active)

        if len(active) >= max_positions:
            blocked["position_count"] += 1
            continue
        if used_margin + float(row.margin_usdt) > max_margin + 1e-9:
            blocked["margin"] += 1
            continue
        if open_risk + float(row.planned_loss_usdt) > max_open_risk + 1e-9:
            blocked["open_risk"] += 1
            continue

        pnl = float(row.net_return) * float(row.notional_usdt)
        rec = row._asdict()
        rec["policy"] = policy_name
        rec["pnl_usdt"] = pnl
        accepted.append(rec)
        active.append(rec)

    trades = pd.DataFrame(accepted)
    if trades.empty:
        return trades, blocked

    trades = trades.sort_values("exit_time").reset_index(drop=True)
    trades["cumulative_pnl_usdt"] = trades["pnl_usdt"].cumsum()
    trades["equity_usdt"] = capital + trades["cumulative_pnl_usdt"]
    trades["equity_peak_usdt"] = trades["equity_usdt"].cummax()
    trades["drawdown_pct"] = (
        trades["equity_usdt"] / trades["equity_peak_usdt"] - 1
    )
    return trades, blocked


def summarize_policy(trades: pd.DataFrame, capital: float, blocked: dict, policy: str):
    if trades.empty:
        return {
            "policy": policy,
            "trades": 0,
            "pnl_usdt": 0.0,
            "return_on_reference_capital": 0.0,
            "expectancy_net_return": np.nan,
            "expectancy_pnl_usdt": np.nan,
            "win_rate": np.nan,
            "profit_factor_pnl": np.nan,
            "max_drawdown": np.nan,
            **{f"blocked_{k}": v for k, v in blocked.items()},
        }

    pnl = trades["pnl_usdt"].astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = np.inf if gl == 0 and gp > 0 else (gp / gl if gl > 0 else np.nan)

    return {
        "policy": policy,
        "trades": int(len(trades)),
        "pnl_usdt": float(pnl.sum()),
        "return_on_reference_capital": float(pnl.sum() / capital),
        "expectancy_net_return": float(trades["net_return"].mean()),
        "expectancy_pnl_usdt": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor_pnl": pf,
        "max_drawdown": float(trades["drawdown_pct"].min()),
        "ending_equity_usdt": float(capital + pnl.sum()),
        **{f"blocked_{k}": v for k, v in blocked.items()},
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    engine = load_engine(a.engine_file, a.output_dir)

    capital = (
        float(a.initial_capital)
        if a.initial_capital > 0
        else float(engine.PORTFOLIO_CAPITAL_USDT)
    )

    stage2_path = a.bundle_dir / "stage2_full_results.csv"
    stage2 = pd.read_csv(stage2_path)
    stage2["eligible"] = stage2["eligible"].astype(str).str.lower().eq("true")
    pool = stage2[stage2["eligible"]].copy()
    if "score" in pool.columns:
        pool = pool.sort_values("score", ascending=False)

    if a.max_symbols > 0:
        pool = pool.head(a.max_symbols)

    print("=" * 100, flush=True)
    print("V10.12 EXIT SIGNAL LAB — 15m REVERSAL / STRUCTURE / MOMENTUM", flush=True)
    print("=" * 100, flush=True)
    print("Eligible symbols:", len(pool), flush=True)
    print("History days:", a.days, flush=True)
    print("Market-layer required signal score:", a.required_score, flush=True)
    print("Reference capital:", capital, "USDT", flush=True)
    print(
        "Risk settings:",
        f"max_positions={engine.PORTFOLIO_MAX_POSITIONS}",
        f"max_margin={engine.PORTFOLIO_MAX_MARGIN_UTILISATION:.1%}",
        f"max_open_risk={engine.MAX_TOTAL_OPEN_RISK_PCT:.1%}",
        f"max_daily_loss={engine.MAX_DAILY_LOSS_PCT:.1%}",
        flush=True,
    )

    candidates = []

    for n, row in enumerate(pool.itertuples(index=False), 1):
        symbol = str(row.symbol)
        meta = {
            "historical_score": (
                None if pd.isna(getattr(row, "score", np.nan))
                else float(getattr(row, "score"))
            ),
            "minimum_signal_score": float(a.required_score),
            "eligibility_tier": getattr(row, "eligibility_tier", "unknown"),
        }
        print(f"[{n}/{len(pool)}] {symbol}", flush=True)
        try:
            found = collect_symbol_candidates(
                engine,
                symbol,
                meta,
                a.days,
                a.days_1h,
                a.days_4h,
                a.required_score,
                a.max_hold_hours,
            )
            candidates.extend(found)
            print("  actionable candidates:", len(found), flush=True)
        except Exception as exc:
            print("  FAILED:", type(exc).__name__, exc, flush=True)

    if not candidates:
        raise RuntimeError("No A-/A/A+/B+ candidates generated.")

    cand = pd.DataFrame(candidates)
    cand["entry_time"] = pd.to_datetime(cand["entry_time"], utc=True)
    cand["exit_time"] = pd.to_datetime(cand["exit_time"], utc=True)
    cand = cand.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    cand.to_csv(a.output_dir / "candidate_signals.csv", index=False)

    policies = [
        ("CORE", {"A-", "A", "A+"}),
        ("CORE_PLUS_BPLUS", {"B+", "A-", "A", "A+"}),
    ]

    all_trades = []
    summaries = []

    for exit_method in EXIT_METHODS:
        method_cand = cand[cand["exit_method"] == exit_method].copy()
        for name, grades in policies:
            policy_label = f"{exit_method}__{name}"
            trades, blocked = portfolio_sim(
                engine, method_cand, grades, capital, policy_label
            )
            if not trades.empty:
                all_trades.append(trades)
            summary = summarize_policy(trades, capital, blocked, policy_label)
            summary["exit_method"] = exit_method
            summary["grade_policy"] = name
            summaries.append(summary)

    portfolio_trades = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades else pd.DataFrame()
    )
    portfolio_trades.to_csv(
        a.output_dir / "portfolio_trades.csv", index=False
    )

    policy_summary = pd.DataFrame(summaries)
    policy_summary.to_csv(a.output_dir / "policy_summary.csv", index=False)

    # Candidate-level grade statistics before portfolio collision/capacity gates.
    grade_rows = []
    for (exit_method, grade), g in cand.groupby(["exit_method", "grade"]):
        r = g["net_return"].astype(float)
        pnl = r * g["notional_usdt"].astype(float)
        gains = float(pnl[pnl > 0].sum())
        losses = float(-pnl[pnl < 0].sum())
        grade_rows.append(
            {
                "exit_method": exit_method,
                "grade": grade,
                "signals": len(g),
                "avg_net_return": float(r.mean()),
                "win_rate": float((r > 0).mean()),
                "avg_stop_risk": float(g["hard_stop_risk_pct"].mean()),
                "pct_stop_risk_gt2": float(
                    (g["hard_stop_risk_pct"] > 0.02).mean()
                ),
                "avg_notional_usdt": float(g["notional_usdt"].mean()),
                "standalone_pnl_usdt_no_overlap_control": float(pnl.sum()),
                "profit_factor_pnl": (
                    np.inf if losses == 0 and gains > 0
                    else gains / losses if losses > 0 else np.nan
                ),
            }
        )
    grade_summary = pd.DataFrame(grade_rows)
    grade_summary["grade_rank"] = grade_summary["grade"].map(
        {"B+": 0, "A-": 1, "A": 2, "A+": 3}
    )
    grade_summary = grade_summary.sort_values(
        ["exit_method", "grade_rank"]
    ).drop(columns=["grade_rank"])
    grade_summary.to_csv(a.output_dir / "grade_summary.csv", index=False)

    exit_rows = []
    for method, g in cand.groupby("exit_method"):
        r = g["net_return"].astype(float)
        gp = float(r[r > 0].sum())
        gl = float(-r[r < 0].sum())
        exit_rows.append({
            "exit_method": method,
            "signals": int(len(g)),
            "avg_net_return": float(r.mean()),
            "median_net_return": float(r.median()),
            "win_rate": float((r > 0).mean()),
            "profit_factor": np.inf if gl == 0 and gp > 0 else (gp / gl if gl > 0 else np.nan),
            "avg_hours_held": float(g["bars_held"].mean() / 4.0),
            "avg_mfe_pct": float(g["mfe_pct"].mean()),
            "avg_mae_pct": float(g["mae_pct"].mean()),
            "avg_mfe_giveback_pct": float((g["mfe_pct"] - g["net_return"]).mean()),
            "time_cap_rate": float((g["exit_reason"] == "research_time_cap").mean()),
        })
    exit_comparison = pd.DataFrame(exit_rows).sort_values("avg_net_return", ascending=False)
    exit_comparison.to_csv(a.output_dir / "exit_method_comparison.csv", index=False)

    base = cand[cand["exit_method"] == "BASE_FIXED_4H"][
        ["symbol", "entry_time", "grade", "net_return", "bars_held"]
    ].rename(columns={"net_return": "fixed4h_return", "bars_held": "fixed4h_bars"})

    delta_rows = []
    for method in EXIT_METHODS:
        if method == "BASE_FIXED_4H":
            continue
        g = cand[cand["exit_method"] == method].merge(
            base, on=["symbol", "entry_time", "grade"], how="inner"
        )
        if g.empty:
            continue
        d = g["net_return"].astype(float) - g["fixed4h_return"].astype(float)
        delta_rows.append({
            "exit_method": method,
            "signals": int(len(g)),
            "avg_return": float(g["net_return"].mean()),
            "fixed4h_avg_return_same_signals": float(g["fixed4h_return"].mean()),
            "avg_delta_vs_fixed4h": float(d.mean()),
            "pct_better_than_fixed4h": float((d > 0).mean()),
            "pct_worse_than_fixed4h": float((d < 0).mean()),
            "rescued_gt_0_5pct": int((d > 0.005).sum()),
            "gave_back_gt_0_5pct": int((d < -0.005).sum()),
            "avg_hold_hours": float(g["bars_held"].mean() / 4.0),
        })
    vs_fixed4h = pd.DataFrame(delta_rows).sort_values("avg_delta_vs_fixed4h", ascending=False)
    vs_fixed4h.to_csv(a.output_dir / "vs_fixed4h.csv", index=False)

    monthly_rows = []
    if not portfolio_trades.empty:
        p = portfolio_trades.copy()
        p["month"] = p["exit_time"].dt.strftime("%Y-%m")
        for (policy, month), g in p.groupby(["policy", "month"]):
            monthly_rows.append(
                {
                    "policy": policy,
                    "month": month,
                    "trades": len(g),
                    "pnl_usdt": float(g["pnl_usdt"].sum()),
                    "win_rate": float((g["pnl_usdt"] > 0).mean()),
                    "avg_net_return": float(g["net_return"].mean()),
                }
            )
    pd.DataFrame(monthly_rows).to_csv(
        a.output_dir / "monthly_summary.csv", index=False
    )

    print("\nPOLICY SUMMARY\n", flush=True)
    print(policy_summary.to_string(index=False), flush=True)

    print("\nEXIT METHOD COMPARISON — SAME SIGNALS\n", flush=True)
    print(exit_comparison.to_string(index=False), flush=True)

    print("\nVS FIXED-4H — SAME SIGNALS\n", flush=True)
    print(vs_fixed4h.to_string(index=False), flush=True)

    print("\nGRADE SUMMARY — BEFORE PORTFOLIO GATES\n", flush=True)
    print(grade_summary.to_string(index=False), flush=True)

    report = [
        "V10.12 EXIT SIGNAL LAB — 15m REVERSAL / STRUCTURE / MOMENTUM",
        "=" * 100,
        f"Eligible symbols tested: {len(pool)}",
        f"History days: {a.days}",
        f"Required signal score used: {a.required_score:.1f}",
        f"Reference capital: {capital:.2f} USDT",
        "",
        "IMPORTANT:",
        "This applies today's filters to history using today's robust-eligible universe.",
        "It does not reconstruct historical daily Stage2 eligibility, so it is not a perfect",
        "walk-forward replica of what the live scanner would have known on each old date.",
        "",
        "CORE = A-/A/A+ production grades.",
        "CORE_PLUS_BPLUS = what-if: B+ is also actually traded with Dynamic Risk sizing.",
        "",
        "POLICY SUMMARY",
        "-" * 100,
        policy_summary.to_string(index=False),
        "",
        "EXIT METHOD COMPARISON — SAME SIGNALS",
        "-" * 100,
        exit_comparison.to_string(index=False),
        "",
        "VS FIXED-4H — SAME SIGNALS",
        "-" * 100,
        vs_fixed4h.to_string(index=False),
        "",
        "GRADE SUMMARY BEFORE PORTFOLIO GATES",
        "-" * 100,
        grade_summary.to_string(index=False),
    ]
    (a.output_dir / "report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nOutput:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
