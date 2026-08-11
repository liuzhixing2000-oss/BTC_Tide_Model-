#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Tide V10.7 Research Filter Lab
=====================================

Purpose
-------
Use the current Tide entry logic and historical candles to test, BEFORE deployment:

1. Signal-score thresholds
2. Volume thresholds
3. Confirmation thresholds
4. Raw / next-candle / combined-quality thresholds
5. Combined filters
6. Score-based dynamic position sizing
7. Per-score-band exit-method performance
8. Per-symbol performance
9. MFE / MAE by signal quality

This program is RESEARCH ONLY:
- It does not send Telegram messages.
- It does not open exchange orders.
- It does not modify the live bundle.
- It writes results to a separate output directory.

Recommended repository layout
-----------------------------
crypto_tide_engine_v10_6.py
research_filter_lab_v10_7.py
v10_bundle/
    stage2_full_results.csv
    exit_config.json
    entry_parameter_config.json
    online_learning.json
    market_regime.json
    manifest.json

Run
---
python research_filter_lab_v10_7.py \
  --engine-file crypto_tide_engine_v10_6.py \
  --bundle-dir v10_bundle \
  --output-dir research_v10_7 \
  --days-15m 180 \
  --days-1h 210

Important statistical note
--------------------------
The final 45 days are treated as validation. Rankings should be judged primarily
by validation expectancy, validation profit factor, and validation drawdown,
not by total in-sample return.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SCORE_THRESHOLDS = [0, 60, 65, 70, 75, 80, 85, 90]
VOLUME_THRESHOLDS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
CONFIRM_THRESHOLDS = [0, 1, 2, 3]
RAW_THRESHOLDS = [0, 60, 70, 80, 90]
NEXT_THRESHOLDS = [0, 60, 70, 80, 90]
COMBINED_THRESHOLDS = [0, 62, 70, 80, 90]

SCORE_BANDS = [
    ("60_70", 60, 70),
    ("70_80", 70, 80),
    ("80_90", 80, 90),
    ("90_plus", 90, float("inf")),
]

# Keep the grid meaningful rather than creating thousands of nearly identical
# combinations. Each tuple is:
# score_min, volume_min, confirmation_min, raw_min, next_min, combined_min
PREDEFINED_COMBINATIONS = [
    (60, 0.0, 0, 0, 0, 62),
    (70, 0.0, 0, 0, 0, 62),
    (75, 0.0, 0, 0, 0, 62),
    (80, 0.0, 0, 0, 0, 62),
    (85, 0.0, 0, 0, 0, 62),
    (90, 0.0, 0, 0, 0, 62),

    (70, 1.0, 2, 0, 0, 62),
    (75, 1.0, 2, 0, 0, 70),
    (80, 1.0, 2, 0, 0, 70),
    (80, 1.5, 2, 0, 0, 70),
    (80, 2.0, 2, 0, 0, 70),

    (85, 1.0, 2, 0, 0, 70),
    (85, 1.5, 2, 0, 0, 70),
    (85, 2.0, 2, 0, 0, 70),
    (85, 2.0, 3, 0, 0, 70),
    (85, 3.0, 3, 0, 0, 75),

    (90, 1.0, 2, 0, 0, 75),
    (90, 2.0, 2, 0, 0, 75),
    (90, 2.0, 3, 0, 0, 80),
    (90, 3.0, 3, 0, 0, 80),
    (90, 5.0, 3, 0, 0, 85),

    # Raw / next-candle experiments
    (75, 0.0, 2, 70, 0, 70),
    (75, 0.0, 2, 0, 70, 70),
    (80, 0.0, 3, 70, 80, 75),
    (85, 1.0, 3, 80, 80, 80),
    (90, 2.0, 3, 85, 85, 85),
]

POSITION_RULES = {
    "equal_1R": lambda score: 1.0,
    "current_tiers": lambda score: (
        0.25 if score < 65 else
        0.50 if score < 75 else
        0.75 if score < 85 else
        1.00
    ),
    "quality_tilt_moderate": lambda score: (
        0.0 if score < 70 else
        0.35 if score < 75 else
        0.60 if score < 80 else
        0.85 if score < 85 else
        1.10 if score < 90 else
        1.35
    ),
    "quality_tilt_strong": lambda score: (
        0.0 if score < 75 else
        0.40 if score < 80 else
        0.75 if score < 85 else
        1.10 if score < 90 else
        1.50
    ),
    "high_score_only": lambda score: (
        0.0 if score < 85 else
        1.0 if score < 90 else
        1.25
    ),
}


# ============================================================
# Utilities
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine-file",
        type=Path,
        default=Path("crypto_tide_engine_v10_6.py"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("v10_bundle"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_v10_7"),
    )
    parser.add_argument("--days-15m", type=int, default=180)
    parser.add_argument("--days-1h", type=int, default=210)
    parser.add_argument("--validation-days", type=int, default=45)
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated symbols. Empty = all eligible bundle symbols.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="0 = no cap. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--minimum-trades",
        type=int,
        default=8,
        help="Minimum total trades for a filter row to be considered.",
    )
    parser.add_argument(
        "--minimum-validation-trades",
        type=int,
        default=3,
    )
    return parser.parse_args()


def load_engine(engine_file: Path, research_data_dir: Path):
    if not engine_file.exists():
        raise FileNotFoundError(f"Engine file not found: {engine_file}")

    # Keep engine cache/state separate from production files.
    os.environ["TIDE_DATA_DIR"] = str(research_data_dir.resolve())

    spec = importlib.util.spec_from_file_location(
        "crypto_tide_research_engine",
        engine_file,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import engine: {engine_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_bundle_learning_state(bundle_dir: Path, research_data_dir: Path) -> None:
    research_data_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage2_full_results.csv",
        "exit_config.json",
        "entry_parameter_config.json",
        "online_learning.json",
        "market_regime.json",
    ]:
        source = bundle_dir / name
        destination = research_data_dir / name
        if source.exists():
            destination.write_bytes(source.read_bytes())


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def maximum_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def geometric_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    clipped = returns.clip(lower=-0.999)
    return float((1.0 + clipped).prod() - 1.0)


def annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    std = float(returns.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return np.nan
    # Trade-level Sharpe, annualized using 252 as a comparison convention.
    return float(returns.mean() / std * math.sqrt(252))


def summarize_returns(
    frame: pd.DataFrame,
    return_column: str = "net_return",
) -> dict[str, Any]:
    returns = frame[return_column].dropna().astype(float)
    if returns.empty:
        return {
            "trades": 0,
            "total_return": 0.0,
            "average_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "average_mfe": np.nan,
            "average_mae": np.nan,
            "median_post_exit_upside_6h": np.nan,
        }

    return {
        "trades": int(len(returns)),
        "total_return": geometric_return(returns),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": profit_factor(returns),
        "max_drawdown": maximum_drawdown(returns),
        "sharpe": annualized_sharpe(returns),
        "average_mfe": float(frame["mfe_pct"].mean()),
        "average_mae": float(frame["mae_pct"].mean()),
        "median_post_exit_upside_6h": float(
            frame["post_exit_upside_6h"].median()
        ),
    }


def train_validation_summary(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    return_column: str = "net_return",
) -> dict[str, Any]:
    train = frame[frame["entry_time"] < validation_start]
    validation = frame[frame["entry_time"] >= validation_start]

    train_metrics = summarize_returns(train, return_column)
    validation_metrics = summarize_returns(validation, return_column)

    return {
        **summarize_returns(frame, return_column),
        "train_trades": train_metrics["trades"],
        "train_return": train_metrics["total_return"],
        "train_expectancy": train_metrics["average_return"],
        "train_profit_factor": train_metrics["profit_factor"],
        "validation_trades": validation_metrics["trades"],
        "validation_return": validation_metrics["total_return"],
        "validation_expectancy": validation_metrics["average_return"],
        "validation_median": validation_metrics["median_return"],
        "validation_win_rate": validation_metrics["win_rate"],
        "validation_profit_factor": validation_metrics["profit_factor"],
        "validation_max_drawdown": validation_metrics["max_drawdown"],
        "validation_sharpe": validation_metrics["sharpe"],
        "validation_average_mfe": validation_metrics["average_mfe"],
        "validation_average_mae": validation_metrics["average_mae"],
    }


# ============================================================
# Exact historical signal-feature construction
# ============================================================

def historical_signal_score(
    engine,
    row: pd.Series,
    historical_score: float | None,
) -> float:
    """
    Reproduce engine.signal_score without relying on mutable live_metadata.
    """
    components = engine.signal_components(row, historical_score)

    raw_model_score = 100 * (
        engine.ACTIVE_SIGNAL_WEIGHTS["historical"]
        * components["historical"]
        + engine.ACTIVE_SIGNAL_WEIGHTS["wick"]
        * components["wick"]
        + engine.ACTIVE_SIGNAL_WEIGHTS["volume"]
        * components["volume"]
        + engine.ACTIVE_SIGNAL_WEIGHTS["close_position"]
        * components["close_position"]
    )

    confirmation_adjustment = 20.0 * (
        components["confirmation"] - 0.50
    )
    setup_score = safe_float(row.get("combined_setup_score"))
    setup_adjustment = (
        0.0
        if not np.isfinite(setup_score)
        else 0.10 * (setup_score - 62.0)
    )

    return round(float(np.clip(
        raw_model_score + confirmation_adjustment + setup_adjustment,
        0,
        100,
    )), 1)


def measure_trade_path(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    exit_price: float,
) -> dict[str, float]:
    path = df.iloc[entry_idx:exit_idx + 1]
    if path.empty:
        return {
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "post_exit_upside_6h": 0.0,
            "post_exit_downside_6h": 0.0,
        }

    highest = float(path["high"].max())
    lowest = float(path["low"].min())

    post_end = min(exit_idx + 24, len(df) - 1)
    if post_end > exit_idx:
        post = df.iloc[exit_idx + 1:post_end + 1]
        post_high = float(post["high"].max())
        post_low = float(post["low"].min())
    else:
        post_high = exit_price
        post_low = exit_price

    return {
        "mfe_pct": highest / entry_price - 1.0,
        "mae_pct": lowest / entry_price - 1.0,
        "post_exit_upside_6h": max(
            0.0,
            post_high / exit_price - 1.0,
        ),
        "post_exit_downside_6h": min(
            0.0,
            post_low / exit_price - 1.0,
        ),
    }


def build_trade_dataset_for_symbol(
    engine,
    symbol: str,
    historical_score: float | None,
    days_15m: int,
    days_1h: int,
) -> list[dict[str, Any]]:
    df15 = engine.fetch_klines(symbol, "15", days_15m)
    df1h = engine.fetch_klines(symbol, "60", days_1h)
    df = engine.model_frame(df15, df1h)

    signal_indices = np.flatnonzero(df["signal"].to_numpy())
    rows: list[dict[str, Any]] = []

    for entry_idx in signal_indices:
        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        score = historical_signal_score(
            engine,
            entry_row,
            historical_score,
        )

        risk = engine.entry_risk_levels(entry_row)
        structure_risk_pct = (
            max(0.0, entry_price - float(risk["structure_stop"]))
            / entry_price
        )
        catastrophe_risk_pct = (
            max(0.0, entry_price - float(risk["catastrophe_stop"]))
            / entry_price
        )

        base_features = {
            "symbol": symbol,
            "entry_idx": int(entry_idx),
            "entry_time": pd.Timestamp(entry_row["open_time"]),
            "entry_price": entry_price,
            "regime": str(entry_row.get("regime", "")),
            "signal_score": score,
            "historical_score": historical_score,
            "raw_quality_score": safe_float(
                entry_row.get("raw_quality_score")
            ),
            "next_candle_quality_score": safe_float(
                entry_row.get("confirmation_quality_score")
            ),
            "combined_setup_score": safe_float(
                entry_row.get("combined_setup_score")
            ),
            "confirmation_tests": int(
                safe_float(
                    entry_row.get("secondary_confirmation_tests"),
                    0,
                )
            ),
            "volume_multiple": safe_float(
                entry_row.get(
                    "confirmation_signal_volume_multiple",
                    entry_row.get("volume_multiple"),
                )
            ),
            "lower_wick_ratio": safe_float(
                entry_row.get(
                    "confirmation_signal_wick_ratio",
                    entry_row.get("lower_wick_ratio"),
                )
            ),
            "close_position": safe_float(
                entry_row.get(
                    "confirmation_signal_close_position",
                    np.nan,
                )
            ),
            "atr": float(risk["atr"]),
            "atr_pct": (
                float(risk["atr"]) / entry_price
                if entry_price > 0 else np.nan
            ),
            "hard_stop": float(risk["hard_stop"]),
            "hard_stop_risk_pct": float(risk["risk_pct"]),
            "structure_stop": float(risk["structure_stop"]),
            "structure_risk_pct": structure_risk_pct,
            "catastrophe_stop": float(risk["catastrophe_stop"]),
            "catastrophe_risk_pct": catastrophe_risk_pct,
        }

        for method in engine.EXIT_METHOD_NAMES:
            result = engine.run_exit_method(df, entry_idx, method)
            if result is None:
                continue

            net_return = (
                result.exit_price / entry_price
                - 1.0
                - engine.FEE_SLIPPAGE
            )
            path = measure_trade_path(
                df,
                entry_idx,
                result.exit_idx,
                entry_price,
                result.exit_price,
            )

            rows.append({
                **base_features,
                "exit_method": method,
                "exit_idx": int(result.exit_idx),
                "exit_time": pd.Timestamp(
                    df.iloc[result.exit_idx]["open_time"]
                ),
                "exit_price": float(result.exit_price),
                "exit_reason": str(result.reason),
                "hold_bars": int(result.exit_idx - entry_idx),
                "hold_hours": float(
                    (result.exit_idx - entry_idx) * 0.25
                ),
                "net_return": float(net_return),
                **path,
            })

    return rows


# ============================================================
# Filter experiments
# ============================================================

@dataclass(frozen=True)
class FilterSpec:
    name: str
    score_min: float = 0.0
    volume_min: float = 0.0
    confirmation_min: int = 0
    raw_min: float = 0.0
    next_min: float = 0.0
    combined_min: float = 0.0

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        return (
            (frame["signal_score"] >= self.score_min)
            & (frame["volume_multiple"] >= self.volume_min)
            & (frame["confirmation_tests"] >= self.confirmation_min)
            & (frame["raw_quality_score"] >= self.raw_min)
            & (
                frame["next_candle_quality_score"]
                >= self.next_min
            )
            & (
                frame["combined_setup_score"]
                >= self.combined_min
            )
        )


def single_dimension_filters() -> list[FilterSpec]:
    filters = [FilterSpec(name="baseline_all")]

    filters.extend(
        FilterSpec(
            name=f"score_ge_{threshold}",
            score_min=threshold,
        )
        for threshold in SCORE_THRESHOLDS[1:]
    )
    filters.extend(
        FilterSpec(
            name=f"volume_ge_{threshold:g}",
            volume_min=threshold,
        )
        for threshold in VOLUME_THRESHOLDS[1:]
    )
    filters.extend(
        FilterSpec(
            name=f"confirmation_ge_{threshold}",
            confirmation_min=threshold,
        )
        for threshold in CONFIRM_THRESHOLDS[1:]
    )
    filters.extend(
        FilterSpec(
            name=f"raw_ge_{threshold}",
            raw_min=threshold,
        )
        for threshold in RAW_THRESHOLDS[1:]
    )
    filters.extend(
        FilterSpec(
            name=f"next_ge_{threshold}",
            next_min=threshold,
        )
        for threshold in NEXT_THRESHOLDS[1:]
    )
    filters.extend(
        FilterSpec(
            name=f"combined_ge_{threshold}",
            combined_min=threshold,
        )
        for threshold in COMBINED_THRESHOLDS[1:]
    )

    return filters


def combination_filters() -> list[FilterSpec]:
    result = []
    seen = set()

    for values in PREDEFINED_COMBINATIONS:
        score, volume, confirm, raw, nxt, combined = values
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        name = (
            f"S{score:g}_V{volume:g}_C{confirm}_"
            f"R{raw:g}_N{nxt:g}_Q{combined:g}"
        )
        result.append(FilterSpec(
            name=name,
            score_min=score,
            volume_min=volume,
            confirmation_min=confirm,
            raw_min=raw,
            next_min=nxt,
            combined_min=combined,
        ))

    return result


def evaluate_filter_specs(
    trades: pd.DataFrame,
    specs: Iterable[FilterSpec],
    validation_start: pd.Timestamp,
    minimum_trades: int,
    minimum_validation_trades: int,
) -> pd.DataFrame:
    rows = []

    for method, method_frame in trades.groupby("exit_method"):
        for spec in specs:
            filtered = method_frame[spec.mask(method_frame)].copy()
            metrics = train_validation_summary(
                filtered,
                validation_start,
            )
            eligible = (
                metrics["trades"] >= minimum_trades
                and metrics["validation_trades"]
                >= minimum_validation_trades
            )

            rows.append({
                "exit_method": method,
                **asdict(spec),
                **metrics,
                "eligible": eligible,
            })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    eligible = result["eligible"].fillna(False)
    result["research_score"] = np.nan

    if eligible.any():
        pool = result[eligible].copy()
        pf_rank = (
            pool["validation_profit_factor"]
            .replace(np.inf, 999)
            .fillna(0)
            .rank(pct=True)
        )
        result.loc[pool.index, "research_score"] = (
            0.32
            * pool["validation_expectancy"].rank(pct=True)
            + 0.20
            * pool["validation_return"].rank(pct=True)
            + 0.12
            * pool["validation_median"].rank(pct=True)
            + 0.10
            * pool["validation_win_rate"].rank(pct=True)
            + 0.10
            * pool["validation_max_drawdown"].rank(pct=True)
            + 0.08
            * pf_rank
            + 0.05
            * pool["validation_sharpe"].fillna(-999).rank(pct=True)
            + 0.03
            * pool["validation_trades"].clip(upper=100).rank(pct=True)
        )

    return result.sort_values(
        [
            "eligible",
            "research_score",
            "validation_expectancy",
            "validation_profit_factor",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


# ============================================================
# Dynamic sizing experiments
# ============================================================

def evaluate_position_rules(
    trades: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []

    for method, method_frame in trades.groupby("exit_method"):
        ordered = method_frame.sort_values("entry_time").copy()

        for rule_name, rule in POSITION_RULES.items():
            weights = ordered["signal_score"].map(rule).astype(float)
            selected = ordered[weights > 0].copy()
            if selected.empty:
                continue

            selected_weights = weights.loc[selected.index]
            selected["weighted_return"] = (
                selected["net_return"] * selected_weights
            )
            metrics = train_validation_summary(
                selected,
                validation_start,
                return_column="weighted_return",
            )

            rows.append({
                "exit_method": method,
                "position_rule": rule_name,
                "average_R": float(selected_weights.mean()),
                "maximum_R": float(selected_weights.max()),
                **metrics,
            })

    return pd.DataFrame(rows).sort_values(
        [
            "validation_expectancy",
            "validation_profit_factor",
            "validation_max_drawdown",
        ],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


# ============================================================
# Score-band and coin analysis
# ============================================================

def score_band_exit_analysis(
    trades: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []

    for band_name, lower, upper in SCORE_BANDS:
        band = trades[
            (trades["signal_score"] >= lower)
            & (trades["signal_score"] < upper)
        ]

        for method, group in band.groupby("exit_method"):
            rows.append({
                "score_band": band_name,
                "exit_method": method,
                **train_validation_summary(
                    group,
                    validation_start,
                ),
            })

    return pd.DataFrame(rows).sort_values(
        [
            "score_band",
            "validation_expectancy",
            "validation_profit_factor",
        ],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def per_symbol_analysis(
    trades: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []

    for (symbol, method), group in trades.groupby(
        ["symbol", "exit_method"]
    ):
        rows.append({
            "symbol": symbol,
            "exit_method": method,
            **train_validation_summary(
                group,
                validation_start,
            ),
            "average_signal_score": float(
                group["signal_score"].mean()
            ),
            "average_volume_multiple": float(
                group["volume_multiple"].mean()
            ),
            "confirmation_3_share": float(
                (group["confirmation_tests"] == 3).mean()
            ),
        })

    return pd.DataFrame(rows).sort_values(
        [
            "validation_expectancy",
            "validation_profit_factor",
            "validation_trades",
        ],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def feature_bucket_analysis(
    trades: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Use one row per entry/exit method and create transparent, fixed buckets.
    """
    frame = trades.copy()

    frame["score_bucket"] = pd.cut(
        frame["signal_score"],
        bins=[-np.inf, 65, 70, 75, 80, 85, 90, np.inf],
        right=False,
    )
    frame["volume_bucket"] = pd.cut(
        frame["volume_multiple"],
        bins=[-np.inf, 0.5, 1, 1.5, 2, 3, 5, np.inf],
        right=False,
    )
    frame["stop_risk_bucket"] = pd.cut(
        frame["hard_stop_risk_pct"],
        bins=[-np.inf, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, np.inf],
        right=False,
    )

    rows = []
    for feature_name in [
        "score_bucket",
        "volume_bucket",
        "confirmation_tests",
        "stop_risk_bucket",
    ]:
        for (method, bucket), group in frame.groupby(
            ["exit_method", feature_name],
            observed=True,
        ):
            rows.append({
                "feature": feature_name,
                "bucket": str(bucket),
                "exit_method": method,
                **train_validation_summary(
                    group,
                    validation_start,
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# Human-readable report
# ============================================================

def format_pct(value: Any) -> str:
    value = safe_float(value)
    return "n/a" if not np.isfinite(value) else f"{value:.2%}"


def format_num(value: Any, digits: int = 3) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "n/a"
    if np.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def write_text_report(
    path: Path,
    trades: pd.DataFrame,
    filter_results: pd.DataFrame,
    sizing_results: pd.DataFrame,
    score_exit_results: pd.DataFrame,
    symbol_results: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> None:
    lines = [
        "CRYPTO TIDE V10.7 RESEARCH FILTER LAB",
        "=" * 72,
        f"Generated at UTC: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"Validation begins: {validation_start.isoformat()}",
        f"Symbols: {trades['symbol'].nunique()}",
        f"Signal events: {trades[['symbol','entry_time']].drop_duplicates().shape[0]}",
        f"Trade-method rows: {len(trades)}",
        "",
        "IMPORTANT",
        "-" * 72,
        "Rankings below are research results, not proof of future profitability.",
        "Give most weight to validation expectancy, validation PF, trade count,",
        "and drawdown. Very small samples must not be deployed.",
        "",
        "TOP FILTER + EXIT COMBINATIONS",
        "-" * 72,
    ]

    top_filters = filter_results[
        filter_results["eligible"]
    ].head(20)

    for rank, row in enumerate(top_filters.itertuples(index=False), start=1):
        lines.append(
            f"{rank:>2}. {row.name} | exit={row.exit_method} | "
            f"val trades={row.validation_trades} | "
            f"val exp={format_pct(row.validation_expectancy)} | "
            f"val PF={format_num(row.validation_profit_factor)} | "
            f"val return={format_pct(row.validation_return)} | "
            f"val DD={format_pct(row.validation_max_drawdown)}"
        )

    lines.extend([
        "",
        "POSITION-SIZING EXPERIMENTS",
        "-" * 72,
    ])

    for rank, row in enumerate(
        sizing_results.head(15).itertuples(index=False),
        start=1,
    ):
        lines.append(
            f"{rank:>2}. {row.position_rule} | exit={row.exit_method} | "
            f"val trades={row.validation_trades} | "
            f"val exp={format_pct(row.validation_expectancy)} | "
            f"val PF={format_num(row.validation_profit_factor)} | "
            f"val DD={format_pct(row.validation_max_drawdown)}"
        )

    lines.extend([
        "",
        "BEST EXIT BY SCORE BAND",
        "-" * 72,
    ])

    for band in [name for name, _, _ in SCORE_BANDS]:
        subset = score_exit_results[
            score_exit_results["score_band"] == band
        ]
        if subset.empty:
            continue
        best = subset.iloc[0]
        lines.append(
            f"{band}: {best['exit_method']} | "
            f"val trades={int(best['validation_trades'])} | "
            f"val exp={format_pct(best['validation_expectancy'])} | "
            f"val PF={format_num(best['validation_profit_factor'])}"
        )

    lines.extend([
        "",
        "TOP SYMBOL / EXIT PAIRS",
        "-" * 72,
    ])

    usable_symbols = symbol_results[
        symbol_results["validation_trades"] >= 3
    ].head(20)

    for rank, row in enumerate(
        usable_symbols.itertuples(index=False),
        start=1,
    ):
        lines.append(
            f"{rank:>2}. {row.symbol} | {row.exit_method} | "
            f"val trades={row.validation_trades} | "
            f"val exp={format_pct(row.validation_expectancy)} | "
            f"val PF={format_num(row.validation_profit_factor)}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    research_data_dir = args.output_dir / "engine_research_data"
    copy_bundle_learning_state(
        args.bundle_dir,
        research_data_dir,
    )
    engine = load_engine(
        args.engine_file,
        research_data_dir,
    )

    stage2_path = args.bundle_dir / "stage2_full_results.csv"
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Missing {stage2_path}. Run with the current v10_bundle."
        )

    stage2 = pd.read_csv(stage2_path)
    if "eligible" in stage2.columns:
        stage2["eligible"] = (
            stage2["eligible"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        eligible = stage2[stage2["eligible"]].copy()
    else:
        eligible = stage2.copy()

    if args.symbols.strip():
        requested = {
            symbol.strip().upper()
            for symbol in args.symbols.split(",")
            if symbol.strip()
        }
        eligible = eligible[
            eligible["symbol"].isin(requested)
        ].copy()

    if args.max_symbols > 0:
        sort_columns = [
            column for column in [
                "score",
                "validation_expectancy",
                "validation_return",
            ]
            if column in eligible.columns
        ]
        if sort_columns:
            eligible = eligible.sort_values(
                sort_columns,
                ascending=False,
                na_position="last",
            )
        eligible = eligible.head(args.max_symbols)

    if eligible.empty:
        raise RuntimeError("No symbols selected for research.")

    print("=" * 80)
    print("CRYPTO TIDE V10.7 RESEARCH FILTER LAB")
    print("=" * 80)
    print("Symbols:", len(eligible))
    print("Engine:", args.engine_file)
    print("Bundle:", args.bundle_dir)
    print("Output:", args.output_dir)

    all_rows: list[dict[str, Any]] = []

    for index, stage_row in enumerate(
        eligible.itertuples(index=False),
        start=1,
    ):
        symbol = str(stage_row.symbol)
        historical_score = safe_float(
            getattr(stage_row, "score", np.nan),
            default=np.nan,
        )
        if not np.isfinite(historical_score):
            historical_score = None

        print(
            f"[{index}/{len(eligible)}] {symbol} "
            f"historical_score={historical_score}"
        )

        try:
            rows = build_trade_dataset_for_symbol(
                engine,
                symbol,
                historical_score,
                args.days_15m,
                args.days_1h,
            )
            all_rows.extend(rows)
            print(
                f"    signal-method rows: {len(rows)}"
            )
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    if not all_rows:
        raise RuntimeError(
            "Research produced no trade rows. Check data access and bundle."
        )

    trades = pd.DataFrame(all_rows)
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        utc=True,
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        utc=True,
    )

    validation_start = (
        trades["entry_time"].max()
        - pd.Timedelta(days=args.validation_days)
    )

    # Save the master dataset first. This is the most valuable output.
    master_path = args.output_dir / "research_trade_features.csv"
    trades.to_csv(master_path, index=False)

    filter_specs = (
        single_dimension_filters()
        + combination_filters()
    )
    filter_results = evaluate_filter_specs(
        trades,
        filter_specs,
        validation_start,
        args.minimum_trades,
        args.minimum_validation_trades,
    )
    filter_results.to_csv(
        args.output_dir / "filter_exit_rankings.csv",
        index=False,
    )

    sizing_results = evaluate_position_rules(
        trades,
        validation_start,
    )
    sizing_results.to_csv(
        args.output_dir / "position_sizing_rankings.csv",
        index=False,
    )

    score_exit_results = score_band_exit_analysis(
        trades,
        validation_start,
    )
    score_exit_results.to_csv(
        args.output_dir / "score_band_exit_rankings.csv",
        index=False,
    )

    symbol_results = per_symbol_analysis(
        trades,
        validation_start,
    )
    symbol_results.to_csv(
        args.output_dir / "per_symbol_exit_rankings.csv",
        index=False,
    )

    bucket_results = feature_bucket_analysis(
        trades,
        validation_start,
    )
    bucket_results.to_csv(
        args.output_dir / "feature_bucket_analysis.csv",
        index=False,
    )

    write_text_report(
        args.output_dir / "research_report.txt",
        trades,
        filter_results,
        sizing_results,
        score_exit_results,
        symbol_results,
        validation_start,
    )

    print("\nResearch complete.")
    print("Master dataset:", master_path)
    print(
        "Top rankings:",
        args.output_dir / "filter_exit_rankings.csv",
    )
    print(
        "Position sizing:",
        args.output_dir / "position_sizing_rankings.csv",
    )
    print(
        "Score-band exits:",
        args.output_dir / "score_band_exit_rankings.csv",
    )
    print(
        "Human report:",
        args.output_dir / "research_report.txt",
    )

    top = filter_results[
        filter_results["eligible"]
    ].head(10)

    if not top.empty:
        print("\nTOP 10 VALIDATION-RANKED FILTERS")
        display_columns = [
            "name",
            "exit_method",
            "trades",
            "validation_trades",
            "validation_expectancy",
            "validation_profit_factor",
            "validation_return",
            "validation_max_drawdown",
            "research_score",
        ]
        print(top[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
