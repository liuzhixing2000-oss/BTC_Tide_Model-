#!/usr/bin/env python3
"""
Universe A/B Backtest for Tide Universe V7.1

Purpose
-------
Compare the CURRENT universe rules with wider/lower-liquidity alternatives
while keeping the Tide entry model, entry parameters, fees, validation rules,
and per-symbol exit laboratory unchanged.

Important limitation
--------------------
This is a "current-universe snapshot" comparison:
- It uses today's listed contracts and today's 24h turnover to define each universe.
- It then backtests those symbols over historical candles.
- It does NOT reconstruct the exact historical universe/turnover on every past day.

That makes it suitable for deciding whether widening today's candidate pool
improves historical Tide performance, but it is not a survivorship-bias-free
institutional walk-forward study.

Usage
-----
1. Put this file beside your current Tide V7.1 file, or pass --model-file.
2. Install the same packages used by Tide V7.1.
3. Run:

   python universe_ab_backtest.py

Optional:
   python universe_ab_backtest.py --configs current expanded_same_cap expanded_full
   python universe_ab_backtest.py --model-file "tide_universe_v7_1_stop_alerts(1).py"
   python universe_ab_backtest.py --capital-per-trade 0.125

Outputs
-------
ab_results/universe_ab_summary.csv
ab_results/universe_ab_symbol_details.csv
ab_results/universe_ab_trades.csv
ab_results/universe_ab_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Experiment configurations
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    min_turnover_24h: float
    stage1_candidates: int
    stage2_candidates: int
    monitor_count: int
    description: str


CONFIGS: dict[str, ExperimentConfig] = {
    "current": ExperimentConfig(
        name="current",
        min_turnover_24h=10_000_000,
        stage1_candidates=60,
        stage2_candidates=20,
        monitor_count=8,
        description="Current V7.1 universe: $10m turnover, Top60 → Top20 → monitor 8",
    ),
    "expanded_same_cap": ExperimentConfig(
        name="expanded_same_cap",
        min_turnover_24h=5_000_000,
        stage1_candidates=200,
        stage2_candidates=40,
        monitor_count=8,
        description="Wider universe only: $5m turnover, Top200 → Top40 → monitor 8",
    ),
    "expanded_full": ExperimentConfig(
        name="expanded_full",
        min_turnover_24h=5_000_000,
        stage1_candidates=200,
        stage2_candidates=40,
        monitor_count=15,
        description="Proposed wider version: $5m turnover, Top200 → Top40 → monitor 15",
    ),
    "low_liquidity": ExperimentConfig(
        name="low_liquidity",
        min_turnover_24h=3_000_000,
        stage1_candidates=200,
        stage2_candidates=40,
        monitor_count=15,
        description="Stress test: $3m turnover, Top200 → Top40 → monitor 15",
    ),
}


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(*parts: Any) -> None:
    print(utc_now(), *parts, flush=True)


def load_model_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    spec = importlib.util.spec_from_file_location("tide_v71_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detect_default_model_file() -> Path:
    candidates = [
        Path("tide_universe_v7_1_stop_alerts(1).py"),
        Path("tide_universe_v7_1_stop_alerts.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matching = sorted(Path(".").glob("tide_universe_v7_1_stop_alerts*.py"))
    if matching:
        return matching[0]

    raise FileNotFoundError(
        "Could not locate Tide V7.1. Use --model-file to specify it."
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def run_with_retries(func, *args, attempts: int = 3, **kwargs):
    """Retry transient API failures with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(1.5 * (2 ** (attempt - 1)))


# ============================================================
# Current-universe snapshot
# ============================================================

def get_universe_snapshot(model, absolute_min_turnover: float) -> pd.DataFrame:
    """
    Reimplements model.get_universe(), but accepts a lower turnover floor so
    all A/B configurations can be built from the same exchange snapshot.
    """
    rows: list[dict] = []
    cursor = ""

    while True:
        kwargs = {
            "category": "linear",
            "status": "Trading",
            "limit": 1000,
        }
        if cursor:
            kwargs["cursor"] = cursor

        result = model.http.get_instruments_info(**kwargs)["result"]
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
        time.sleep(0.15)

    ticker_rows = model.http.get_tickers(category="linear")["result"].get("list", [])
    turnovers = {
        item["symbol"]: float(item.get("turnover24h") or 0.0)
        for item in ticker_rows
    }

    now_ms = int(time.time() * 1000)
    output: list[dict] = []

    for item in rows:
        symbol = item.get("symbol", "")
        base = item.get("baseCoin", "")
        launch = int(item.get("launchTime") or 0)
        age_days = (now_ms - launch) / 86_400_000 if launch else 0.0

        if item.get("quoteCoin") != "USDT":
            continue
        if item.get("settleCoin") != "USDT":
            continue
        if item.get("contractType") != "LinearPerpetual":
            continue
        if base in model.EXCLUDED_BASES or symbol in model.EXCLUDED_SYMBOLS:
            continue
        if age_days < model.MIN_LISTING_DAYS:
            continue

        turnover = turnovers.get(symbol, 0.0)
        if turnover < absolute_min_turnover:
            continue

        output.append({
            "symbol": symbol,
            "age_days": age_days,
            "turnover24h": turnover,
        })

    frame = pd.DataFrame(output)
    if frame.empty:
        raise RuntimeError("No eligible symbols in the exchange snapshot.")

    return (
        frame.sort_values("turnover24h", ascending=False)
        .drop_duplicates("symbol")
        .reset_index(drop=True)
    )


# ============================================================
# Candle cache
# ============================================================

def cache_path(cache_dir: Path, symbol: str, interval: str, days: int) -> Path:
    return cache_dir / f"{symbol}_{interval}_{days}d.csv.gz"


def read_cached_klines(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, compression="gzip")
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        expected = {"open_time", "open", "high", "low", "close", "volume", "turnover"}
        if not expected.issubset(df.columns):
            return None
        return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    except Exception:
        return None


def fetch_klines_cached(
    model,
    cache_dir: Path,
    symbol: str,
    interval: str,
    days: int,
    refresh_cache: bool,
) -> pd.DataFrame:
    path = cache_path(cache_dir, symbol, interval, days)

    if not refresh_cache:
        cached = read_cached_klines(path)
        if cached is not None and not cached.empty:
            return cached

    df = model.fetch_klines(symbol, interval, days)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return df


# ============================================================
# Stage 1 / Stage 2 using unchanged Tide logic
# ============================================================

def stage1_one_symbol(
    model,
    row: pd.Series,
    cache_dir: Path,
    refresh_cache: bool,
    frozen_entry_params: dict,
) -> dict:
    symbol = str(row["symbol"])
    df15 = fetch_klines_cached(
        model, cache_dir, symbol, "15", model.STAGE1_DAYS_15M, refresh_cache
    )
    df1h = fetch_klines_cached(
        model, cache_dir, symbol, "60", model.STAGE1_DAYS_1H, refresh_cache
    )
    df = model.model_frame(df15, df1h, frozen_entry_params)
    result = model.evaluate_method(symbol, df, model.DEFAULT_EXIT_METHOD)

    return {
        "symbol": symbol,
        "turnover24h": float(row["turnover24h"]),
        "age_days": float(row["age_days"]),
        "trades": result["trades"],
        "total_return": result["total_return"],
        "average_return": result["average_return"],
        "median_return": result["median_return"],
        "win_rate": result["win_rate"],
        "max_drawdown": result["max_drawdown"],
        "train_return": result["train_return"],
        "validation_trades": result["validation_trades"],
        "validation_return": result["validation_return"],
        "validation_median": result["validation_median"],
        "validation_win_rate": result["validation_win_rate"],
    }


def stage2_one_symbol(
    model,
    symbol: str,
    lookup: pd.DataFrame,
    cache_dir: Path,
    refresh_cache: bool,
    frozen_entry_params: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df15 = fetch_klines_cached(
        model, cache_dir, symbol, "15", model.STAGE2_DAYS_15M, refresh_cache
    )
    df1h = fetch_klines_cached(
        model, cache_dir, symbol, "60", model.STAGE2_DAYS_1H, refresh_cache
    )
    df = model.model_frame(df15, df1h, frozen_entry_params)

    chosen, ranked_methods = model.choose_best_exit(symbol, df)
    chosen_method = str(chosen["exit_method"])
    trades = model.trade_frame(df, chosen_method).copy()

    if not trades.empty:
        trades.insert(0, "symbol", symbol)
        trades["exit_method"] = chosen_method

    row = {
        "symbol": symbol,
        "turnover24h": float(lookup.loc[symbol, "turnover24h"]),
        "age_days": float(lookup.loc[symbol, "age_days"]),
        "exit_method": chosen_method,
        "exit_score": chosen.get("exit_score"),
        "exit_selection_note": chosen.get("selection_note"),
        "trades": int(chosen["trades"]),
        "total_return": float(chosen["total_return"]),
        "average_return": float(chosen["average_return"]),
        "median_return": float(chosen["median_return"]),
        "win_rate": float(chosen["win_rate"]),
        "max_drawdown": float(chosen["max_drawdown"]),
        "train_return": float(chosen["train_return"]),
        "validation_trades": int(chosen["validation_trades"]),
        "validation_return": float(chosen["validation_return"]),
        "validation_expectancy": (
            np.nan
            if pd.isna(chosen.get("validation_expectancy"))
            else float(chosen["validation_expectancy"])
        ),
        "validation_median": float(chosen["validation_median"]),
        "validation_win_rate": float(chosen["validation_win_rate"]),
        "validation_profit_factor": (
            np.nan
            if pd.isna(chosen.get("validation_profit_factor"))
            else float(chosen["validation_profit_factor"])
        ),
        "validation_sell_early_median_6h": (
            np.nan
            if pd.isna(chosen.get("validation_sell_early_median_6h"))
            else float(chosen["validation_sell_early_median_6h"])
        ),
    }
    return row, ranked_methods, trades


# ============================================================
# Portfolio-level comparison
# ============================================================

def profit_factor(returns: np.ndarray) -> float:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses == 0:
        return math.inf if gains > 0 else math.nan
    return gains / losses


def portfolio_metrics(
    trades: pd.DataFrame,
    capital_per_trade: float,
) -> dict:
    """
    Capital-normalised approximation:
    each closed trade contributes capital_per_trade × net_return to equity.
    The same fraction is used for every configuration, so signal-count effects
    are comparable. Overlapping margin constraints are not reconstructed.
    """
    if trades.empty:
        return {
            "trade_count": 0,
            "winning_trades": 0,
            "win_rate": np.nan,
            "average_trade_return": np.nan,
            "median_trade_return": np.nan,
            "profit_factor": np.nan,
            "capital_normalised_return": 0.0,
            "max_drawdown": 0.0,
            "annualised_return_approx": np.nan,
            "sharpe_approx": np.nan,
            "first_entry": None,
            "last_exit": None,
        }

    ordered = trades.copy()
    ordered["entry_time"] = pd.to_datetime(ordered["entry_time"], utc=True)
    ordered["exit_time"] = pd.to_datetime(ordered["exit_time"], utc=True)
    ordered = ordered.sort_values(["exit_time", "entry_time", "symbol"])

    returns = ordered["net_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + capital_per_trade * returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0

    first_entry = ordered["entry_time"].min()
    last_exit = ordered["exit_time"].max()
    elapsed_days = max((last_exit - first_entry).total_seconds() / 86400.0, 1.0)
    final_return = float(equity[-1] - 1.0)
    annualised = (
        float((1.0 + final_return) ** (365.0 / elapsed_days) - 1.0)
        if final_return > -1.0
        else -1.0
    )

    scaled_returns = capital_per_trade * returns
    sharpe = (
        float(
            np.mean(scaled_returns)
            / np.std(scaled_returns, ddof=1)
            * np.sqrt(max(len(scaled_returns), 1))
        )
        if len(scaled_returns) >= 2 and np.std(scaled_returns, ddof=1) > 0
        else np.nan
    )

    return {
        "trade_count": int(len(returns)),
        "winning_trades": int((returns > 0).sum()),
        "win_rate": float((returns > 0).mean()),
        "average_trade_return": float(np.mean(returns)),
        "median_trade_return": float(np.median(returns)),
        "profit_factor": profit_factor(returns),
        "capital_normalised_return": final_return,
        "max_drawdown": float(np.min(drawdowns)),
        "annualised_return_approx": annualised,
        "sharpe_approx": sharpe,
        "first_entry": first_entry,
        "last_exit": last_exit,
    }


def validation_only(trades: pd.DataFrame, model) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    output = trades.copy()
    output["entry_time"] = pd.to_datetime(output["entry_time"], utc=True)
    latest = output["entry_time"].max()
    cutoff = latest - pd.Timedelta(days=model.VALIDATION_DAYS)
    return output[output["entry_time"] >= cutoff].copy()


# ============================================================
# One configuration
# ============================================================

def run_configuration(
    model,
    config: ExperimentConfig,
    universe_snapshot: pd.DataFrame,
    cache_dir: Path,
    refresh_cache: bool,
    frozen_entry_params: dict,
    capital_per_trade: float,
    max_workers: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    log("=" * 72)
    log("START", config.name, "-", config.description)

    universe = (
        universe_snapshot[
            universe_snapshot["turnover24h"] >= config.min_turnover_24h
        ]
        .head(config.stage1_candidates)
        .copy()
    )
    if universe.empty:
        raise RuntimeError(f"{config.name}: universe is empty")

    stage1_rows: list[dict] = []
    jobs = [row.copy() for _, row in universe.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_with_retries,
                stage1_one_symbol,
                model,
                row,
                cache_dir,
                refresh_cache,
                frozen_entry_params,
            ): str(row["symbol"])
            for row in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                stage1_rows.append(future.result())
                log(f"{config.name} Stage1 [{completed}/{len(futures)}]", symbol, "done")
            except Exception as exc:
                log(config.name, "Stage1 failed", symbol, repr(exc))

    if not stage1_rows:
        raise RuntimeError(f"{config.name}: Stage1 produced no results")

    stage1 = model.stage1_score(pd.DataFrame(stage1_rows))
    stage2_symbols = stage1.head(config.stage2_candidates)["symbol"].tolist()

    # Preserve current model behaviour: append BTC/ETH benchmarks to Stage 2.
    eligible_snapshot_symbols = set(universe_snapshot["symbol"])
    for benchmark in model.BENCHMARKS:
        if benchmark in eligible_snapshot_symbols and benchmark not in stage2_symbols:
            stage2_symbols.append(benchmark)

    lookup = universe_snapshot.set_index("symbol")
    stage2_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_with_retries,
                stage2_one_symbol,
                model,
                symbol,
                lookup,
                cache_dir,
                refresh_cache,
                frozen_entry_params,
            ): symbol
            for symbol in stage2_symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                row, _, trades = future.result()
                stage2_rows.append(row)
                if not trades.empty:
                    trade_frames.append(trades)
                log(f"{config.name} Stage2 [{completed}/{len(futures)}]", symbol, "done")
            except Exception as exc:
                log(config.name, "Stage2 failed", symbol, repr(exc))

    if not stage2_rows:
        raise RuntimeError(f"{config.name}: Stage2 produced no results")

    stage2 = model.robust_stage2_score(pd.DataFrame(stage2_rows))
    selected = (
        stage2.loc[stage2["eligible"]]
        .head(config.monitor_count)
        .copy()
    )
    selected_symbols = selected["symbol"].tolist()

    all_trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    )
    if not all_trades.empty:
        selected_trades = all_trades[
            all_trades["symbol"].isin(selected_symbols)
        ].copy()
    else:
        selected_trades = all_trades.copy()

    full_metrics = portfolio_metrics(selected_trades, capital_per_trade)
    validation_trades = validation_only(selected_trades, model)
    validation_metrics = portfolio_metrics(validation_trades, capital_per_trade)

    summary = {
        "config": config.name,
        "description": config.description,
        "min_turnover_24h": config.min_turnover_24h,
        "stage1_requested": config.stage1_candidates,
        "stage1_actual": int(len(universe)),
        "stage2_requested": config.stage2_candidates,
        "stage2_actual": int(len(stage2)),
        "monitor_requested": config.monitor_count,
        "selected_count": int(len(selected)),
        "selected_symbols": ",".join(selected_symbols),
        "strict_selected": int(
            (selected.get("eligibility_tier", pd.Series(dtype=str)) == "strict").sum()
        ),
        "relaxed_selected": int(
            (selected.get("eligibility_tier", pd.Series(dtype=str)) == "relaxed").sum()
        ),
        **{f"full_{key}": value for key, value in full_metrics.items()},
        **{
            f"validation_{key}": value
            for key, value in validation_metrics.items()
        },
    }

    selected = selected.copy()
    selected.insert(0, "config", config.name)

    selected_trades = selected_trades.copy()
    if not selected_trades.empty:
        selected_trades.insert(0, "config", config.name)

    log(
        "DONE",
        config.name,
        f"selected={len(selected_symbols)}",
        f"trades={full_metrics['trade_count']}",
        f"return={full_metrics['capital_normalised_return']:.2%}",
        f"maxDD={full_metrics['max_drawdown']:.2%}",
        f"PF={full_metrics['profit_factor']:.3f}"
        if np.isfinite(full_metrics["profit_factor"])
        else f"PF={full_metrics['profit_factor']}",
    )
    return summary, selected, selected_trades


def build_recommendation(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty or "current" not in set(summary_df["config"]):
        return {
            "decision": "INCONCLUSIVE",
            "recommended_config": None,
            "reason": "The current baseline was not included.",
        }

    baseline = summary_df.loc[summary_df["config"] == "current"].iloc[0]
    candidates = summary_df[summary_df["config"] != "current"].copy()
    passed = []

    for _, row in candidates.iterrows():
        base_return = float(baseline.get("validation_capital_normalised_return", 0.0))
        base_dd = float(baseline.get("validation_max_drawdown", 0.0))
        base_pf = float(baseline.get("validation_profit_factor", float("nan")))

        candidate_return = float(row.get("validation_capital_normalised_return", 0.0))
        candidate_dd = float(row.get("validation_max_drawdown", 0.0))
        candidate_pf = float(row.get("validation_profit_factor", float("nan")))

        return_gain = candidate_return - base_return
        extra_drawdown = abs(min(candidate_dd, 0.0)) - abs(min(base_dd, 0.0))
        pf_ok = (
            True
            if not np.isfinite(base_pf)
            else np.isfinite(candidate_pf) and candidate_pf >= max(1.0, base_pf * 0.95)
        )

        if return_gain > 0 and extra_drawdown <= 0.03 and pf_ok:
            passed.append((return_gain - max(extra_drawdown, 0.0), row["config"]))

    if not passed:
        return {
            "decision": "KEEP_CURRENT",
            "recommended_config": "current",
            "reason": (
                "No wider version improved validation return while preserving "
                "profit factor and limiting additional drawdown to 3 percentage points."
            ),
        }

    passed.sort(reverse=True)
    return {
        "decision": "UPGRADE_CANDIDATE",
        "recommended_config": passed[0][1],
        "reason": (
            "This version improved validation return while preserving profit "
            "factor and keeping additional drawdown within the set limit."
        ),
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B-test Tide V7.1 universe configurations."
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Path to tide_universe_v7_1_stop_alerts*.py",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help="Configurations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ab_results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ab_cache"),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore local candle cache and download again.",
    )
    parser.add_argument(
        "--capital-per-trade",
        type=float,
        default=0.125,
        help=(
            "Fixed fraction of capital allocated per trade for the comparable "
            "portfolio-return approximation. Default 0.125 = 12.5%%."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers. Use 2-4 to reduce API rate-limit risk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.capital_per_trade <= 1:
        raise ValueError("--capital-per-trade must be in (0, 1].")
    if not 1 <= args.max_workers <= 8:
        raise ValueError("--max-workers must be between 1 and 8.")

    model_file = args.model_file or detect_default_model_file()
    model = load_model_module(model_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the current entry parameters across every A/B configuration.
    # This isolates the effect of changing Universe/Stage sizes.
    frozen_entry_params = model.load_entry_params()
    log("Model file:", model_file)
    log("Frozen entry parameters:", model.parameter_key(frozen_entry_params))

    absolute_floor = min(CONFIGS[name].min_turnover_24h for name in args.configs)
    universe_snapshot = get_universe_snapshot(model, absolute_floor)
    universe_snapshot.to_csv(
        args.output_dir / "universe_snapshot.csv",
        index=False,
    )
    log(
        "Universe snapshot:",
        len(universe_snapshot),
        "symbols above",
        f"${absolute_floor:,.0f}",
    )

    summaries: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for name in args.configs:
        summary, selected, trades = run_configuration(
            model=model,
            config=CONFIGS[name],
            universe_snapshot=universe_snapshot,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
            frozen_entry_params=frozen_entry_params,
            capital_per_trade=args.capital_per_trade,
            max_workers=args.max_workers,
        )
        summaries.append(summary)
        selected_frames.append(selected)
        if not trades.empty:
            trade_frames.append(trades)

        # Save partial results after every completed configuration.
        pd.DataFrame(summaries).to_csv(
            args.output_dir / "universe_ab_summary.csv",
            index=False,
        )
        pd.concat(selected_frames, ignore_index=True).to_csv(
            args.output_dir / "universe_ab_symbol_details.csv",
            index=False,
        )
        if trade_frames:
            pd.concat(trade_frames, ignore_index=True).to_csv(
                args.output_dir / "universe_ab_trades.csv",
                index=False,
            )

    summary_df = pd.DataFrame(summaries)

    # Add direct deltas versus current when current was run.
    if "current" in summary_df["config"].values:
        baseline = summary_df.loc[
            summary_df["config"] == "current"
        ].iloc[0]
        for metric in [
            "full_capital_normalised_return",
            "full_max_drawdown",
            "full_profit_factor",
            "full_average_trade_return",
            "full_trade_count",
            "validation_capital_normalised_return",
            "validation_max_drawdown",
            "validation_profit_factor",
            "validation_average_trade_return",
            "validation_trade_count",
        ]:
            if metric in summary_df.columns:
                summary_df[f"delta_vs_current__{metric}"] = (
                    summary_df[metric] - baseline[metric]
                )

    summary_df.to_csv(
        args.output_dir / "universe_ab_summary.csv",
        index=False,
    )

    recommendation = build_recommendation(summary_df)
    (args.output_dir / "universe_ab_recommendation.json").write_text(
        json.dumps(json_safe(recommendation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "universe_ab_recommendation.txt").write_text(
        f"Decision: {recommendation.get('decision')}\n"
        f"Recommended config: {recommendation.get('recommended_config')}\n"
        f"Reason: {recommendation.get('reason')}\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": utc_now(),
        "model_file": str(model_file.resolve()),
        "model_default_exit_method": model.DEFAULT_EXIT_METHOD,
        "frozen_entry_parameters": frozen_entry_params,
        "frozen_entry_parameter_key": model.parameter_key(frozen_entry_params),
        "validation_days": model.VALIDATION_DAYS,
        "fee_slippage": model.FEE_SLIPPAGE,
        "capital_per_trade": args.capital_per_trade,
        "configurations": [asdict(CONFIGS[name]) for name in args.configs],
        "limitations": [
            "Uses today's contract list and today's 24h turnover to define universes.",
            "Does not reconstruct historical daily membership or historical turnover.",
            "Capital-normalised return applies a fixed capital fraction per closed trade.",
            "Overlapping exchange margin constraints are not reconstructed.",
            "Entry parameters are frozen across configurations to isolate universe changes.",
            "Per-symbol exit strategy is selected by the unchanged V7.1 exit laboratory.",
        ],
    }
    (args.output_dir / "universe_ab_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_columns = [
        "config",
        "selected_count",
        "full_trade_count",
        "full_capital_normalised_return",
        "full_max_drawdown",
        "full_profit_factor",
        "full_win_rate",
        "full_average_trade_return",
        "validation_trade_count",
        "validation_capital_normalised_return",
        "validation_max_drawdown",
        "validation_profit_factor",
        "validation_win_rate",
    ]
    available = [column for column in display_columns if column in summary_df.columns]

    print("\n" + "=" * 110)
    print("TIDE UNIVERSE A/B RESULT")
    print("=" * 110)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(summary_df[available].to_string(index=False))

    print("\nSaved to:", args.output_dir.resolve())
    print(
        "\nDecision rule suggestion:\n"
        "Upgrade only if the wider version improves validation return and/or "
        "profit factor without a disproportionate increase in max drawdown. "
        "Do not select a version solely because it creates more trades."
    )


if __name__ == "__main__":
    main()
