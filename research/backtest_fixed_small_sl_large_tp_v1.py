#!/usr/bin/env python3
"""Fair exit-only comparison for Crypto Tide Engine V10.6.

The script imports the user's V10.6 engine, uses its exact model_frame() entry
signals, and changes only the exit logic.  It compares the seven production
V10.6 exits with fixed ATR-risk targets and optional 1R break-even protection.

Examples
--------
python backtest_fixed_small_sl_large_tp_v1.py --symbols BTCUSDT ETHUSDT SOLUSDT
python backtest_fixed_small_sl_large_tp_v1.py --symbols all --max-symbols 60
python backtest_fixed_small_sl_large_tp_v1.py --engine crypto_tide_engine_v10_6.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FIXED_SPECS = [
    ("fixed_sl0.50atr_tp2r", 0.50, 2.0, False),
    ("fixed_sl0.50atr_tp3r", 0.50, 3.0, False),
    ("fixed_sl0.50atr_tp4r", 0.50, 4.0, False),
    ("fixed_sl0.75atr_tp2r", 0.75, 2.0, False),
    ("fixed_sl0.75atr_tp3r", 0.75, 3.0, False),
    ("fixed_sl1.00atr_tp3r", 1.00, 3.0, False),
    ("be1r_sl0.50atr_tp2r", 0.50, 2.0, True),
    ("be1r_sl0.50atr_tp3r", 0.50, 3.0, True),
    ("be1r_sl0.50atr_tp4r", 0.50, 4.0, True),
    ("be1r_sl0.75atr_tp2r", 0.75, 2.0, True),
    ("be1r_sl0.75atr_tp3r", 0.75, 3.0, True),
    ("be1r_sl1.00atr_tp3r", 1.00, 3.0, True),
]


@dataclass(frozen=True)
class FixedExit:
    exit_idx: int
    exit_price: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V10.6 exits with fixed small-SL/large-TP exits."
    )
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Bybit USDT perpetual symbols, or the single word 'all'.",
    )
    parser.add_argument("--max-symbols", type=int, default=60)
    parser.add_argument("--days15", type=int, default=180)
    parser.add_argument("--days1h", type=int, default=210)
    parser.add_argument("--max-hold-hours", type=float, default=24.0)
    parser.add_argument("--validation-days", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, default=Path("fixed_rr_results"))
    parser.add_argument("--pause", type=float, default=0.15)
    return parser.parse_args()


def locate_engine(requested: Path | None) -> Path:
    candidates = [
        requested,
        Path("crypto_tide_engine_v10_6.py"),
        Path("Pasted text(7).txt"),
        Path("upload/Pasted text(7).txt"),
        Path("Pasted text(6).txt"),
        Path("upload/Pasted text(6).txt"),
    ]
    for path in candidates:
        if path and path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "V10.6 engine not found. Put it beside this script or pass --engine PATH."
    )


def import_engine(path: Path):
    spec = importlib.util.spec_from_file_location("tide_v10_6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ["fetch_klines", "model_frame", "trade_frame", "metrics"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Engine is missing: {', '.join(missing)}")
    return module


def fixed_exit(
    df: pd.DataFrame,
    entry_idx: int,
    stop_atr: float,
    target_r: float,
    break_even_at_1r: bool,
    max_bars: int,
) -> FixedExit | None:
    row = df.iloc[entry_idx]
    entry = float(row["close"])
    atr = float(row.get("confirmation_signal_atr", row.get("atr14", np.nan)))
    if not np.isfinite(atr) or atr <= 0:
        return None

    risk = stop_atr * atr
    initial_stop = entry - risk
    stop = initial_stop
    target = entry + target_r * risk
    last = min(entry_idx + max_bars, len(df) - 1)

    for idx in range(entry_idx + 1, last + 1):
        candle = df.iloc[idx]
        low, high = float(candle["low"]), float(candle["high"])

        # Conservative OHLC rule: if stop and target are both touched in one
        # 15m candle, count the stop first because intrabar order is unknown.
        if low <= stop:
            reason = "break_even" if stop >= entry else "fixed_stop"
            return FixedExit(idx, stop, reason)
        if high >= target:
            return FixedExit(idx, target, "fixed_target")

        # BE becomes active only after a fully processed candle reaches 1R.
        # This avoids assuming whether +1R or the later pullback happened first.
        if break_even_at_1r and stop < entry and high >= entry + risk:
            stop = entry

    return FixedExit(last, float(df.iloc[last]["close"]), "maximum_hold_timeout")


def fixed_trade_frame(
    model,
    df: pd.DataFrame,
    name: str,
    stop_atr: float,
    target_r: float,
    break_even_at_1r: bool,
    max_bars: int,
) -> pd.DataFrame:
    rows = []
    fee = float(model.FEE_SLIPPAGE)
    for entry_idx in np.flatnonzero(df["signal"].to_numpy()):
        result = fixed_exit(
            df, entry_idx, stop_atr, target_r, break_even_at_1r, max_bars
        )
        if result is None:
            continue
        entry_row = df.iloc[entry_idx]
        entry = float(entry_row["close"])
        gross = result.exit_price / entry - 1.0
        post_end = min(result.exit_idx + 24, len(df) - 1)
        post_high = (
            float(df.iloc[result.exit_idx + 1 : post_end + 1]["high"].max())
            if post_end > result.exit_idx else result.exit_price
        )
        rows.append({
            "strategy": name,
            "entry_time": entry_row["open_time"],
            "exit_time": df.iloc[result.exit_idx]["open_time"],
            "entry_price": entry,
            "exit_price": result.exit_price,
            "gross_return": gross,
            "net_return": gross - fee,
            "exit_reason": result.reason,
            "hold_bars": result.exit_idx - entry_idx,
            "stop_atr": stop_atr,
            "target_r": target_r,
            "break_even_at_1r": break_even_at_1r,
            "post_exit_upside_6h": max(0.0, post_high / result.exit_price - 1),
        })
    return pd.DataFrame(rows)


def max_consecutive_losses(returns: np.ndarray) -> int:
    worst = current = 0
    for value in returns:
        current = current + 1 if value < 0 else 0
        worst = max(worst, current)
    return worst


def summarise(model, trades: pd.DataFrame, validation_start: pd.Timestamp) -> dict:
    returns = trades["net_return"].to_numpy(float) if not trades.empty else np.array([])
    base = model.metrics(returns)
    validation = trades[trades["entry_time"] >= validation_start] if not trades.empty else trades
    vr = validation["net_return"].to_numpy(float) if not validation.empty else np.array([])
    vm = model.metrics(vr)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    counts = trades["exit_reason"].value_counts() if not trades.empty else pd.Series(dtype=int)
    return {
        **base,
        "average_win": float(wins.mean()) if len(wins) else np.nan,
        "average_loss": float(losses.mean()) if len(losses) else np.nan,
        "realised_win_loss_ratio": (
            float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan
        ),
        "max_consecutive_losses": max_consecutive_losses(returns),
        "target_rate": float(counts.get("fixed_target", 0) / len(trades)) if len(trades) else np.nan,
        "stop_rate": float(counts.get("fixed_stop", 0) / len(trades)) if len(trades) else np.nan,
        "break_even_rate": float(counts.get("break_even", 0) / len(trades)) if len(trades) else np.nan,
        "timeout_rate": float(counts.get("maximum_hold_timeout", 0) / len(trades)) if len(trades) else np.nan,
        "median_hold_hours": float(trades["hold_bars"].median() / 4) if len(trades) else np.nan,
        "sell_early_median_6h": float(trades["post_exit_upside_6h"].median()) if len(trades) else np.nan,
        "validation_trades": vm["trades"],
        "validation_return": vm["total_return"],
        "validation_expectancy": vm["average_return"],
        "validation_win_rate": vm["win_rate"],
        "validation_max_drawdown": vm["max_drawdown"],
        "validation_profit_factor": vm["profit_factor"],
    }


def dynamic_trades(model, df: pd.DataFrame, method: str) -> pd.DataFrame:
    trades = model.trade_frame(df, method).copy()
    if trades.empty:
        return trades
    trades["strategy"] = "v10_6_" + method
    trades["gross_return"] = trades["net_return"] + float(model.FEE_SLIPPAGE)
    return trades


def resolve_symbols(model, args: argparse.Namespace) -> list[str]:
    symbols = [s.upper() for s in args.symbols]
    if symbols == ["ALL"]:
        universe = model.get_universe()
        if args.max_symbols > 0:
            universe = universe.head(args.max_symbols)
        return universe["symbol"].tolist()
    return list(dict.fromkeys(symbols))


def main() -> None:
    args = parse_args()
    engine_path = locate_engine(args.engine)
    model = import_engine(engine_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols(model, args)
    max_bars = max(1, int(round(args.max_hold_hours * 4)))
    all_trades, summaries, failures = [], [], []

    print(f"Engine: {engine_path}")
    print(f"Symbols: {len(symbols)} | 15m={args.days15}d | 1h={args.days1h}d")
    for number, symbol in enumerate(symbols, 1):
        print(f"[{number}/{len(symbols)}] {symbol}", flush=True)
        try:
            df15 = model.fetch_klines(symbol, "15", args.days15)
            df1h = model.fetch_klines(symbol, "60", args.days1h)
            df = model.model_frame(df15, df1h)
            validation_start = df["open_time"].max() - pd.Timedelta(days=args.validation_days)

            strategy_frames = []
            for method in model.EXIT_METHOD_NAMES:
                strategy_frames.append(dynamic_trades(model, df, method))
            for spec in FIXED_SPECS:
                strategy_frames.append(
                    fixed_trade_frame(model, df, *spec, max_bars=max_bars)
                )

            for trades in strategy_frames:
                if trades.empty:
                    continue
                strategy = str(trades["strategy"].iloc[0])
                trades.insert(0, "symbol", symbol)
                summaries.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    **summarise(model, trades, validation_start),
                })
                all_trades.append(trades)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": repr(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr)
        time.sleep(max(0.0, args.pause))

    detail = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    per_symbol = pd.DataFrame(summaries)
    if per_symbol.empty:
        raise RuntimeError(f"No results generated. Failures: {failures}")

    numeric = [c for c in per_symbol.columns if c not in {"symbol", "strategy"}]
    overall_rows = []
    for strategy, group in detail.groupby("strategy"):
        # Portfolio comparison uses chronological equal-weight trade sequence.
        group = group.sort_values("entry_time")
        validation_start = group["entry_time"].max() - pd.Timedelta(days=args.validation_days)
        overall_rows.append({
            "strategy": strategy,
            "symbols_with_trades": int(group["symbol"].nunique()),
            **summarise(model, group, validation_start),
        })
    overall = pd.DataFrame(overall_rows).sort_values(
        ["validation_expectancy", "validation_return"], ascending=False
    )

    # Best strategy per symbol is selected only by validation expectancy and
    # requires at least five validation trades, preventing one-trade winners.
    eligible = per_symbol[per_symbol["validation_trades"] >= 5].copy()
    best = (
        eligible.sort_values(["symbol", "validation_expectancy", "validation_profit_factor"],
                             ascending=[True, False, False])
        .groupby("symbol", as_index=False).head(1)
    )

    detail.to_csv(args.output_dir / "trade_by_trade.csv", index=False)
    per_symbol.to_csv(args.output_dir / "results_by_symbol.csv", index=False)
    overall.to_csv(args.output_dir / "overall_comparison.csv", index=False)
    best.to_csv(args.output_dir / "best_strategy_by_symbol.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    metadata = {
        "engine": str(engine_path),
        "symbols_requested": symbols,
        "symbols_completed": sorted(per_symbol["symbol"].unique().tolist()),
        "days15": args.days15,
        "days1h": args.days1h,
        "validation_days": args.validation_days,
        "max_hold_hours": args.max_hold_hours,
        "fee_slippage_per_round_trip": float(model.FEE_SLIPPAGE),
        "same_candle_rule": "stop_first_conservative",
        "break_even_rule": "activate_after_candle_reaches_1R; exit at entry minus fees",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop strategies (validation expectancy):")
    cols = ["strategy", "trades", "validation_trades", "validation_expectancy",
            "validation_return", "validation_win_rate", "max_drawdown", "profit_factor"]
    print(overall[cols].head(15).to_string(index=False))
    print(f"\nSaved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
