import json
import os
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from pybit.unified_trading import HTTP, WebSocket


# ============================================================
# Tide Universe V4.4
# Automatic universe selection + per-symbol exit optimisation
# + realtime entry and exit alerts
# ============================================================

# ---------- Universe ----------
MIN_LISTING_DAYS = 120
MIN_TURNOVER_24H = 10_000_000
MAX_STAGE1_CANDIDATES = 60
STAGE2_CANDIDATES = 20
TOP_N_TO_MONITOR = 8
BENCHMARKS = ["BTCUSDT", "ETHUSDT"]
MONITOR_FAILED_BENCHMARKS = False

# ---------- Backtest windows ----------
STAGE1_DAYS_15M = 60
STAGE1_DAYS_1H = 90
STAGE2_DAYS_15M = 180
STAGE2_DAYS_1H = 210
VALIDATION_DAYS = 45

# ---------- Tide entry model ----------
LOOKBACK_BARS = 24
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35
COOLDOWN_BARS = 24
FEE_SLIPPAGE = 0.001

# ---------- Robust selection ----------
MIN_STAGE2_TRADES = 20
MIN_VALIDATION_TRADES = 5
MAX_ALLOWED_DRAWDOWN = -0.12

# ---------- Exit optimisation ----------
EXIT_MAX_BARS = 48
DEFAULT_EXIT_METHOD = "fixed_6h"
EXIT_METHOD_NAMES = [
    "fixed_3h",
    "fixed_6h",
    "fixed_9h",
    "fixed_12h",
    "rolling_high_12h_cap",
    "atr_trail_1.0",
    "atr_trail_1.5",
    "atr_trail_2.0",
    "mfe_giveback_30pct",
    "mfe_giveback_40pct",
    "mfe_giveback_50pct",
    "hybrid_target_atr1.5",
]

# ---------- Realtime ----------
PRE_ALERT_WINDOW_MINUTES = 3
FULL_RESCAN_HOURS = 168
WATCHLIST_REFRESH_HOURS = 6
BASE_MARGIN_USDT = float(os.getenv("BASE_MARGIN_USDT", "200"))

# ---------- Output ----------
STAGE1_CSV = Path("stage1_results.csv")
STAGE2_CSV = Path("stage2_full_results.csv")
EXIT_RESULTS_CSV = Path("exit_method_results.csv")
EXIT_CONFIG_JSON = Path("exit_config.json")
SELECTED_JSON = Path("selected_symbols.json")
STATE_FILE = Path("alert_state.json")
ACTIVE_POSITIONS_FILE = Path("active_positions.json")

EXCLUDED_BASES = {
    "USDC", "USDE", "DAI", "FDUSD", "TUSD", "PYUSD", "USDD",
    "USD", "USDT", "EUR",
}
EXCLUDED_SYMBOLS = {
    "AAPLUSDT", "TSLAUSDT", "NVDAUSDT", "AMZNUSDT", "METAUSDT",
    "GOOGLUSDT", "MSFTUSDT",
}

http = HTTP(testnet=False)
market_data: dict[str, dict[str, pd.DataFrame]] = {}
data_lock = threading.Lock()
state_lock = threading.Lock()
position_lock = threading.Lock()
live_metadata: dict[str, dict] = {}


def log(*parts) -> None:
    print(datetime.now(timezone.utc).isoformat(), *parts, flush=True)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


alert_state = load_json(STATE_FILE, {})
active_positions = load_json(ACTIVE_POSITIONS_FILE, {})


# ============================================================
# Telegram
# ============================================================

def telegram_credentials() -> tuple[str, str]:
    return (
        os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )


def send_tg(text: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        log("Telegram variables missing")
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        log("Telegram", response.status_code, response.text[:180])
    except Exception as exc:
        log("Telegram error", repr(exc))


def send_tg_photo(image: BytesIO, caption: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        return
    image.seek(0)
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("tide_v44_ranking.png", image, "image/png")},
            timeout=60,
        )
        log("Telegram photo", response.status_code)
    except Exception as exc:
        log("Telegram photo error", repr(exc))


def send_tg_document(path: Path, caption: str) -> None:
    token, chat_id = telegram_credentials()
    if not token or not chat_id or not path.exists():
        return
    try:
        with path.open("rb") as file:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (path.name, file, "text/csv")},
                timeout=90,
            )
        log("Telegram document", response.status_code)
    except Exception as exc:
        log("Telegram document error", repr(exc))


# ============================================================
# Bybit data
# ============================================================

def get_universe() -> pd.DataFrame:
    rows, cursor = [], ""
    while True:
        kwargs = {"category": "linear", "status": "Trading", "limit": 1000}
        if cursor:
            kwargs["cursor"] = cursor
        result = http.get_instruments_info(**kwargs)["result"]
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
        time.sleep(0.15)

    ticker_rows = http.get_tickers(category="linear")["result"].get("list", [])
    turnovers = {
        item["symbol"]: float(item.get("turnover24h") or 0)
        for item in ticker_rows
    }

    now_ms = int(time.time() * 1000)
    output = []
    for item in rows:
        symbol = item.get("symbol", "")
        base = item.get("baseCoin", "")
        launch = int(item.get("launchTime") or 0)
        age_days = (now_ms - launch) / 86_400_000 if launch else 0

        if item.get("quoteCoin") != "USDT":
            continue
        if item.get("settleCoin") != "USDT":
            continue
        if item.get("contractType") != "LinearPerpetual":
            continue
        if base in EXCLUDED_BASES or symbol in EXCLUDED_SYMBOLS:
            continue
        if age_days < MIN_LISTING_DAYS:
            continue

        turnover = turnovers.get(symbol, 0.0)
        if turnover < MIN_TURNOVER_24H:
            continue

        output.append({
            "symbol": symbol,
            "age_days": age_days,
            "turnover24h": turnover,
        })

    frame = pd.DataFrame(output)
    if frame.empty:
        raise RuntimeError("No eligible symbols.")
    return frame.sort_values("turnover24h", ascending=False)


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    interval_minutes = int(interval)
    required = int(days * 24 * 60 / interval_minutes)
    rows = []
    end_ms = None

    while len(rows) < required:
        limit = min(1000, required - len(rows))
        kwargs = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if end_ms is not None:
            kwargs["end"] = end_ms

        batch = http.get_kline(**kwargs)["result"].get("list", [])
        if not batch:
            break
        rows.extend(batch)
        end_ms = min(int(row[0]) for row in batch) - 1
        if len(batch) < limit:
            break
        time.sleep(0.08)

    if not rows:
        raise RuntimeError(f"No kline data for {symbol} {interval}")

    frame = pd.DataFrame([{
        "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "turnover": float(row[6]),
    } for row in rows])

    return (
        frame.sort_values("open_time")
        .drop_duplicates("open_time")
        .tail(required)
        .reset_index(drop=True)
    )


# ============================================================
# Entry model
# ============================================================

def model_frame(df15: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    df15 = df15.copy()
    df1h = df1h.copy()

    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()
    df1h["regime"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(df1h["ma50"] > df1h["ma200"], "uptrend", "range"),
    )

    df = pd.merge_asof(
        df15.sort_values("open_time"),
        df1h[["open_time", "regime", "ma50", "ma200"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    df["rolling_low"] = df["low"].rolling(LOOKBACK_BARS).min().shift(1)
    df["rolling_high"] = df["high"].rolling(LOOKBACK_BARS).max().shift(1)
    df["avg_volume"] = df["volume"].rolling(VOLUME_LOOKBACK).mean().shift(1)
    df["volume_multiple"] = np.where(
        df["avg_volume"] > 0,
        df["volume"] / df["avg_volume"],
        0.0,
    )
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]
    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] / df["candle_range"],
        0.0,
    )

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["raw_signal"] = (
        (df["regime"] == "downtrend")
        & (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & (df["volume_multiple"] > VOLUME_MULTIPLIER)
    )

    accepted = np.zeros(len(df), dtype=bool)
    last_entry = -10**9
    for idx in np.flatnonzero(df["raw_signal"].to_numpy()):
        if idx - last_entry >= COOLDOWN_BARS:
            accepted[idx] = True
            last_entry = idx
    df["signal"] = accepted
    return df


# ============================================================
# Exit backtest engines
# ============================================================

@dataclass(frozen=True)
class ExitResult:
    exit_idx: int
    exit_price: float
    reason: str


def fixed_exit(df: pd.DataFrame, entry_idx: int, bars: int) -> ExitResult | None:
    idx = entry_idx + bars
    if idx >= len(df):
        return None
    return ExitResult(idx, float(df.iloc[idx]["close"]), f"fixed_{bars}")


def rolling_high_exit(df: pd.DataFrame, entry_idx: int) -> ExitResult | None:
    entry = df.iloc[entry_idx]
    target = float(entry["rolling_high"])
    entry_price = float(entry["close"])
    last = min(entry_idx + EXIT_MAX_BARS, len(df) - 1)
    if last <= entry_idx:
        return None
    if np.isfinite(target) and target > entry_price:
        for idx in range(entry_idx + 1, last + 1):
            if float(df.iloc[idx]["high"]) >= target:
                return ExitResult(idx, target, "rolling_high_target")
    return ExitResult(last, float(df.iloc[last]["close"]), "rolling_high_timeout")


def atr_exit(df: pd.DataFrame, entry_idx: int, multiple: float) -> ExitResult | None:
    entry = df.iloc[entry_idx]
    entry_price = float(entry["close"])
    atr = float(entry["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    last = min(entry_idx + EXIT_MAX_BARS, len(df) - 1)
    highest = entry_price
    stop = entry_price - multiple * atr

    for idx in range(entry_idx + 1, last + 1):
        row = df.iloc[idx]
        if float(row["low"]) <= stop:
            return ExitResult(idx, stop, f"atr_{multiple:.1f}_stop")
        highest = max(highest, float(row["high"]))
        stop = max(stop, highest - multiple * atr)

    return ExitResult(last, float(df.iloc[last]["close"]), f"atr_{multiple:.1f}_timeout")


def mfe_exit(df: pd.DataFrame, entry_idx: int, giveback: float) -> ExitResult | None:
    entry = df.iloc[entry_idx]
    entry_price = float(entry["close"])
    atr = float(entry["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    last = min(entry_idx + EXIT_MAX_BARS, len(df) - 1)
    highest = entry_price
    activated = False
    stop = -np.inf

    for idx in range(entry_idx + 1, last + 1):
        row = df.iloc[idx]
        if activated and float(row["low"]) <= stop:
            return ExitResult(idx, stop, f"mfe_{giveback:.0%}")
        highest = max(highest, float(row["high"]))
        favourable = highest - entry_price
        if favourable >= atr:
            activated = True
            stop = max(stop, highest - giveback * favourable)

    return ExitResult(last, float(df.iloc[last]["close"]), f"mfe_{giveback:.0%}_timeout")


def hybrid_exit(df: pd.DataFrame, entry_idx: int) -> ExitResult | None:
    entry = df.iloc[entry_idx]
    entry_price = float(entry["close"])
    target = float(entry["rolling_high"])
    atr = float(entry["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    last = min(entry_idx + EXIT_MAX_BARS, len(df) - 1)
    highest = entry_price
    stop = entry_price - 1.5 * atr
    target_hit = False

    for idx in range(entry_idx + 1, last + 1):
        row = df.iloc[idx]
        if float(row["low"]) <= stop:
            price = 0.5 * target + 0.5 * stop if target_hit else stop
            return ExitResult(idx, price, "hybrid_trail")
        if (
            not target_hit and np.isfinite(target)
            and target > entry_price and float(row["high"]) >= target
        ):
            target_hit = True
        highest = max(highest, float(row["high"]))
        stop = max(stop, highest - 1.5 * atr)

    close = float(df.iloc[last]["close"])
    price = 0.5 * target + 0.5 * close if target_hit else close
    return ExitResult(last, price, "hybrid_timeout")


def run_exit_method(df: pd.DataFrame, entry_idx: int, method: str) -> ExitResult | None:
    if method == "fixed_3h":
        return fixed_exit(df, entry_idx, 12)
    if method == "fixed_6h":
        return fixed_exit(df, entry_idx, 24)
    if method == "fixed_9h":
        return fixed_exit(df, entry_idx, 36)
    if method == "fixed_12h":
        return fixed_exit(df, entry_idx, 48)
    if method == "rolling_high_12h_cap":
        return rolling_high_exit(df, entry_idx)
    if method == "atr_trail_1.0":
        return atr_exit(df, entry_idx, 1.0)
    if method == "atr_trail_1.5":
        return atr_exit(df, entry_idx, 1.5)
    if method == "atr_trail_2.0":
        return atr_exit(df, entry_idx, 2.0)
    if method == "mfe_giveback_30pct":
        return mfe_exit(df, entry_idx, 0.30)
    if method == "mfe_giveback_40pct":
        return mfe_exit(df, entry_idx, 0.40)
    if method == "mfe_giveback_50pct":
        return mfe_exit(df, entry_idx, 0.50)
    if method == "hybrid_target_atr1.5":
        return hybrid_exit(df, entry_idx)
    raise ValueError(f"Unknown exit method: {method}")


def trade_frame(df: pd.DataFrame, method: str = DEFAULT_EXIT_METHOD) -> pd.DataFrame:
    trades = []
    for entry_idx in np.flatnonzero(df["signal"].to_numpy()):
        result = run_exit_method(df, entry_idx, method)
        if result is None:
            continue
        entry_price = float(df.iloc[entry_idx]["close"])
        net_return = result.exit_price / entry_price - 1 - FEE_SLIPPAGE
        trades.append({
            "entry_time": df.iloc[entry_idx]["open_time"],
            "exit_time": df.iloc[result.exit_idx]["open_time"],
            "entry_price": entry_price,
            "exit_price": result.exit_price,
            "net_return": net_return,
            "exit_reason": result.reason,
            "hold_bars": result.exit_idx - entry_idx,
        })
    return pd.DataFrame(trades)


def metrics(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {
            "trades": 0,
            "total_return": 0.0,
            "average_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "profit_factor": np.nan,
        }

    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    pf = np.inf if losses == 0 and gains > 0 else (gains / losses if losses > 0 else np.nan)

    return {
        "trades": int(len(returns)),
        "total_return": float(equity[-1] - 1),
        "average_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.min(drawdown)),
        "profit_factor": pf,
    }


def evaluate_method(symbol: str, df: pd.DataFrame, method: str) -> dict:
    trades = trade_frame(df, method)
    all_returns = trades["net_return"].to_numpy(dtype=float) if not trades.empty else np.array([])
    result = metrics(all_returns)
    validation_start = df["open_time"].max() - pd.Timedelta(days=VALIDATION_DAYS)

    if trades.empty:
        train_returns = np.array([])
        validation_returns = np.array([])
    else:
        train_returns = trades.loc[
            trades["entry_time"] < validation_start, "net_return"
        ].to_numpy(dtype=float)
        validation_returns = trades.loc[
            trades["entry_time"] >= validation_start, "net_return"
        ].to_numpy(dtype=float)

    train = metrics(train_returns)
    validation = metrics(validation_returns)

    return {
        "symbol": symbol,
        "exit_method": method,
        **result,
        "train_return": train["total_return"],
        "validation_trades": validation["trades"],
        "validation_return": validation["total_return"],
        "validation_expectancy": validation["average_return"],
        "validation_median": validation["median_return"],
        "validation_win_rate": validation["win_rate"],
        "validation_max_drawdown": validation["max_drawdown"],
        "validation_profit_factor": validation["profit_factor"],
    }


def exit_method_score(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["exit_eligible"] = (
        (rows["trades"] >= MIN_STAGE2_TRADES)
        & (rows["validation_trades"] >= MIN_VALIDATION_TRADES)
        & (rows["train_return"] > 0)
        & (rows["validation_return"] > 0)
        & (rows["validation_expectancy"] > 0)
        & (rows["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN)
    )
    rows["exit_score"] = np.nan
    pool = rows[rows["exit_eligible"]].copy()
    if not pool.empty:
        score = (
            0.35 * pool["validation_expectancy"].rank(pct=True)
            + 0.20 * pool["validation_return"].rank(pct=True)
            + 0.15 * pool["validation_median"].rank(pct=True)
            + 0.10 * pool["validation_win_rate"].rank(pct=True)
            + 0.10 * pool["validation_max_drawdown"].rank(pct=True)
            + 0.05 * pool["average_return"].rank(pct=True)
            + 0.05 * pool["profit_factor"].replace(np.inf, 999).rank(pct=True)
        )
        rows.loc[pool.index, "exit_score"] = score
    return rows.sort_values(
        ["exit_eligible", "exit_score", "validation_expectancy"],
        ascending=[False, False, False],
        na_position="last",
    )


def choose_best_exit(symbol: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = pd.DataFrame([
        evaluate_method(symbol, df, method)
        for method in EXIT_METHOD_NAMES
    ])
    ranked = exit_method_score(rows)

    eligible = ranked[ranked["exit_eligible"]]
    if eligible.empty:
        fallback = ranked[ranked["exit_method"] == DEFAULT_EXIT_METHOD].iloc[0]
        chosen = fallback.to_dict()
        chosen["selection_note"] = "fallback_no_method_passed_robust_filter"
    else:
        chosen = eligible.iloc[0].to_dict()
        chosen["selection_note"] = "best_robust_validation_score"

    return chosen, ranked


# ============================================================
# Stage 1 and stage 2
# ============================================================

def backtest_symbol_fixed(
    symbol: str,
    turnover: float,
    age_days: float,
    days15: int,
    days1h: int,
) -> dict:
    df = model_frame(
        fetch_klines(symbol, "15", days15),
        fetch_klines(symbol, "60", days1h),
    )
    result = evaluate_method(symbol, df, DEFAULT_EXIT_METHOD)
    return {
        "symbol": symbol,
        "turnover24h": turnover,
        "age_days": age_days,
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


def stage1_score(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["stage1_score"] = (
        0.45 * results["total_return"].rank(pct=True)
        + 0.20 * results["median_return"].fillna(-999).rank(pct=True)
        + 0.15 * results["win_rate"].fillna(0).rank(pct=True)
        + 0.10 * results["max_drawdown"].rank(pct=True)
        + 0.10 * results["turnover24h"].rank(pct=True)
    )
    return results.sort_values("stage1_score", ascending=False)


def robust_stage2_score(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["eligible"] = (
        (results["trades"] >= MIN_STAGE2_TRADES)
        & (results["total_return"] > 0)
        & (results["median_return"] > 0)
        & (results["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN)
        & (results["train_return"] > 0)
        & (results["validation_trades"] >= MIN_VALIDATION_TRADES)
        & (results["validation_return"] > 0)
        & (results["validation_median"] > 0)
    )
    results["score"] = np.nan
    pool = results[results["eligible"]].copy()
    if not pool.empty:
        score = (
            0.20 * pool["total_return"].rank(pct=True)
            + 0.30 * pool["validation_return"].rank(pct=True)
            + 0.15 * pool["validation_median"].rank(pct=True)
            + 0.10 * pool["win_rate"].rank(pct=True)
            + 0.10 * pool["max_drawdown"].rank(pct=True)
            + 0.10 * pool["trades"].clip(upper=60).rank(pct=True)
            + 0.05 * pool["turnover24h"].rank(pct=True)
        )
        results.loc[pool.index, "score"] = score

    return results.sort_values(
        ["eligible", "score", "validation_return", "total_return"],
        ascending=[False, False, False, False],
        na_position="last",
    )


def scan_and_select() -> tuple[list[str], pd.DataFrame]:
    universe = get_universe().head(MAX_STAGE1_CANDIDATES)
    log("Stage 1 candidates:", len(universe))

    stage1_rows = []
    for idx, row in enumerate(universe.itertuples(index=False), start=1):
        try:
            log(f"Stage1 [{idx}/{len(universe)}] {row.symbol}")
            stage1_rows.append(backtest_symbol_fixed(
                row.symbol,
                float(row.turnover24h),
                float(row.age_days),
                STAGE1_DAYS_15M,
                STAGE1_DAYS_1H,
            ))
        except Exception as exc:
            log("Stage1 failed", row.symbol, repr(exc))

    if not stage1_rows:
        raise RuntimeError("Stage1 produced no results.")

    stage1 = stage1_score(pd.DataFrame(stage1_rows))
    stage1.to_csv(STAGE1_CSV, index=False)

    stage2_symbols = stage1.head(STAGE2_CANDIDATES)["symbol"].tolist()
    for benchmark in BENCHMARKS:
        if benchmark in universe["symbol"].values and benchmark not in stage2_symbols:
            stage2_symbols.append(benchmark)

    lookup = universe.set_index("symbol")
    stage2_rows = []
    all_exit_rows = []
    exit_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_days": VALIDATION_DAYS,
        "default_method": DEFAULT_EXIT_METHOD,
        "symbols": {},
    }

    for idx, symbol in enumerate(stage2_symbols, start=1):
        try:
            log(f"Stage2 + ExitLab [{idx}/{len(stage2_symbols)}] {symbol}")
            df = model_frame(
                fetch_klines(symbol, "15", STAGE2_DAYS_15M),
                fetch_klines(symbol, "60", STAGE2_DAYS_1H),
            )
            chosen, ranked_methods = choose_best_exit(symbol, df)
            ranked_methods["turnover24h"] = float(lookup.loc[symbol, "turnover24h"])
            ranked_methods["age_days"] = float(lookup.loc[symbol, "age_days"])
            all_exit_rows.append(ranked_methods)

            stage2_rows.append({
                "symbol": symbol,
                "turnover24h": float(lookup.loc[symbol, "turnover24h"]),
                "age_days": float(lookup.loc[symbol, "age_days"]),
                "exit_method": chosen["exit_method"],
                "exit_score": chosen.get("exit_score"),
                "exit_selection_note": chosen["selection_note"],
                "trades": int(chosen["trades"]),
                "total_return": float(chosen["total_return"]),
                "average_return": float(chosen["average_return"]),
                "median_return": float(chosen["median_return"]),
                "win_rate": float(chosen["win_rate"]),
                "max_drawdown": float(chosen["max_drawdown"]),
                "train_return": float(chosen["train_return"]),
                "validation_trades": int(chosen["validation_trades"]),
                "validation_return": float(chosen["validation_return"]),
                "validation_median": float(chosen["validation_median"]),
                "validation_win_rate": float(chosen["validation_win_rate"]),
            })

            exit_config["symbols"][symbol] = {
                "method": chosen["exit_method"],
                "exit_score": None if pd.isna(chosen.get("exit_score")) else float(chosen["exit_score"]),
                "validation_expectancy": None if pd.isna(chosen["validation_expectancy"]) else float(chosen["validation_expectancy"]),
                "validation_return": float(chosen["validation_return"]),
                "validation_trades": int(chosen["validation_trades"]),
                "selection_note": chosen["selection_note"],
            }
        except Exception as exc:
            log("Stage2 failed", symbol, repr(exc))

    if not stage2_rows:
        raise RuntimeError("Stage2 produced no results.")

    stage2 = robust_stage2_score(pd.DataFrame(stage2_rows))
    stage2.to_csv(STAGE2_CSV, index=False)
    pd.concat(all_exit_rows, ignore_index=True).to_csv(EXIT_RESULTS_CSV, index=False)
    save_json(EXIT_CONFIG_JSON, exit_config)

    selected = stage2.loc[stage2["eligible"], "symbol"].head(TOP_N_TO_MONITOR).tolist()
    if MONITOR_FAILED_BENCHMARKS:
        for benchmark in BENCHMARKS:
            if benchmark in stage2["symbol"].values and benchmark not in selected:
                selected.append(benchmark)

    save_json(SELECTED_JSON, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "stage2_symbols": stage2_symbols,
        "selection_rule": "best per-symbol exit method + robust stage2 ranking",
    })

    lines = [
        "✅ Tide Universe V4.4 weekly learning complete",
        "",
        f"Stage 1 coins: {len(stage1)}",
        f"Stage 2 coins: {len(stage2)}",
        f"Selected for live regime check: {len(selected)}",
        "",
    ]
    for row in stage2[stage2["symbol"].isin(selected)].itertuples(index=False):
        lines.append(
            f"{row.symbol}: {row.exit_method} | "
            f"validation {row.validation_return:.1%} | "
            f"score {row.score:.3f}"
        )
    send_tg("\n".join(lines))
    send_tg_document(STAGE2_CSV, "V4.4 symbol ranking")
    send_tg_document(EXIT_RESULTS_CSV, "V4.4 all exit-method results")
    return selected, stage2


def cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    return time.time() - path.stat().st_mtime <= max_age_hours * 3600


def load_or_run_stage2() -> pd.DataFrame:
    if (
        cache_is_fresh(STAGE2_CSV, FULL_RESCAN_HOURS)
        and EXIT_CONFIG_JSON.exists()
    ):
        try:
            cached = pd.read_csv(STAGE2_CSV)
            if not cached.empty and "eligible" in cached.columns:
                cached["eligible"] = cached["eligible"].astype(str).str.lower().eq("true")
                log("Using cached V4.4 weekly results")
                return cached
        except Exception as exc:
            log("Cache load failed", repr(exc))
    _, stage2 = scan_and_select()
    return stage2


# ============================================================
# Live selection and signal scoring
# ============================================================

def current_1h_regime(symbol: str) -> str:
    df1h = fetch_klines(symbol, "60", 14).copy()
    now = pd.Timestamp.now(tz="UTC")
    df1h["close_time"] = df1h["open_time"] + pd.Timedelta(hours=1)
    closed = df1h[df1h["close_time"] <= now].copy()
    if len(closed) < 200:
        return "unknown"
    ma50 = float(closed["close"].tail(50).mean())
    ma200 = float(closed["close"].tail(200).mean())
    if ma50 < ma200:
        return "downtrend"
    if ma50 > ma200:
        return "uptrend"
    return "range"


def load_exit_config() -> dict:
    return load_json(EXIT_CONFIG_JSON, {
        "default_method": DEFAULT_EXIT_METHOD,
        "symbols": {},
    })


def symbol_exit_method(symbol: str) -> str:
    config = load_exit_config()
    return config.get("symbols", {}).get(symbol, {}).get(
        "method", config.get("default_method", DEFAULT_EXIT_METHOD)
    )


def select_current_watchlist(results: pd.DataFrame) -> list[str]:
    global live_metadata
    live_metadata = {}

    pool = results[results["eligible"]].sort_values(
        ["score", "validation_return", "total_return"],
        ascending=[False, False, False],
    )

    selected = []
    status_lines = []
    for row in pool.itertuples(index=False):
        if len(selected) >= TOP_N_TO_MONITOR:
            break
        try:
            regime = current_1h_regime(row.symbol)
        except Exception as exc:
            log("Regime failed", row.symbol, repr(exc))
            regime = "unknown"

        status_lines.append(f"{row.symbol}: {regime}")
        if regime != "downtrend":
            continue

        selected.append(row.symbol)
        live_metadata[row.symbol] = {
            "historical_score": None if pd.isna(row.score) else float(row.score),
            "validation_return": float(row.validation_return),
            "validation_trades": int(row.validation_trades),
            "exit_method": row.exit_method,
            "live_regime": regime,
        }

    send_tg(
        "🔄 Tide V4.4 watchlist refreshed\n\n"
        f"Current downtrend symbols: {len(selected)}\n"
        + (
            "\n".join(
                f"{symbol} → {symbol_exit_method(symbol)}"
                for symbol in selected
            )
            if selected else
            "No eligible coin is currently in a 1h downtrend."
        )
    )
    log("Regimes:", " | ".join(status_lines))
    return selected


def signal_score(latest: pd.Series, historical_score: float | None) -> float:
    hist = 0.50 if historical_score is None or np.isnan(historical_score) else historical_score
    wick_norm = np.clip(
        (float(latest["lower_wick_ratio"]) - LOWER_WICK_THRESHOLD)
        / (0.95 - LOWER_WICK_THRESHOLD),
        0, 1,
    )
    volume_norm = np.clip(
        (float(latest["volume_multiple"]) - VOLUME_MULTIPLIER)
        / (4.0 - VOLUME_MULTIPLIER),
        0, 1,
    )
    candle_range = max(float(latest["high"] - latest["low"]), 1e-12)
    close_position = np.clip(
        float(latest["close"] - latest["low"]) / candle_range,
        0, 1,
    )
    return round(float(
        40 * np.clip(hist, 0, 1)
        + 25 * wick_norm
        + 20 * volume_norm
        + 15 * close_position
    ), 1)


def risk_tier(score: float) -> tuple[str, float]:
    if score >= 85:
        return "1.00R", 1.00
    if score >= 75:
        return "0.75R", 0.75
    if score >= 65:
        return "0.50R", 0.50
    return "0.25R", 0.25


# ============================================================
# Realtime positions and exits
# ============================================================

def already_alerted(symbol: str, alert_type: str, signal_key: str) -> bool:
    with state_lock:
        symbol_state = alert_state.setdefault(symbol, {})
        if symbol_state.get(alert_type) == signal_key:
            return True
        symbol_state[alert_type] = signal_key
        save_json(STATE_FILE, alert_state)
        return False


def open_research_position(symbol: str, latest: pd.Series) -> None:
    with position_lock:
        existing = active_positions.get(symbol)
        if existing and existing.get("status") == "active":
            return

        method = symbol_exit_method(symbol)
        entry_price = float(latest["close"])
        atr = float(latest["atr14"])
        target = float(latest["rolling_high"])

        active_positions[symbol] = {
            "status": "active",
            "method": method,
            "entry_time": latest["open_time"].isoformat(),
            "entry_price": entry_price,
            "entry_atr": atr,
            "rolling_high_target": target,
            "highest_price": entry_price,
            "bars_held": 0,
            "target_hit": False,
            "current_stop": None,
        }
        save_json(ACTIVE_POSITIONS_FILE, active_positions)


def realtime_exit_update(symbol: str, latest: pd.Series) -> None:
    with position_lock:
        position = active_positions.get(symbol)
        if not position or position.get("status") != "active":
            return

        method = position["method"]
        entry = float(position["entry_price"])
        atr = float(position["entry_atr"])
        target = float(position["rolling_high_target"])
        high = float(latest["high"])
        low = float(latest["low"])
        close = float(latest["close"])
        bars = int(position.get("bars_held", 0)) + 1
        highest = max(float(position.get("highest_price", entry)), high)
        target_hit = bool(position.get("target_hit", False))
        exit_price = None
        reason = None

        fixed_map = {
            "fixed_3h": 12,
            "fixed_6h": 24,
            "fixed_9h": 36,
            "fixed_12h": 48,
        }

        if method in fixed_map:
            if bars >= fixed_map[method]:
                exit_price, reason = close, method

        elif method == "rolling_high_12h_cap":
            if np.isfinite(target) and target > entry and high >= target:
                exit_price, reason = target, "rolling_high_target"
            elif bars >= EXIT_MAX_BARS:
                exit_price, reason = close, "rolling_high_timeout"

        elif method.startswith("atr_trail_"):
            multiple = float(method.rsplit("_", 1)[1])
            old_stop = position.get("current_stop")
            if old_stop is None:
                old_stop = entry - multiple * atr
            if low <= float(old_stop):
                exit_price, reason = float(old_stop), f"atr_{multiple:.1f}_stop"
            else:
                position["current_stop"] = max(
                    float(old_stop), highest - multiple * atr
                )
                if bars >= EXIT_MAX_BARS:
                    exit_price, reason = close, f"atr_{multiple:.1f}_timeout"

        elif method.startswith("mfe_giveback_"):
            giveback = float(method.split("_")[-1].replace("pct", "")) / 100
            favourable = highest - entry
            old_stop = position.get("current_stop")
            if old_stop is not None and low <= float(old_stop):
                exit_price, reason = float(old_stop), f"mfe_giveback_{giveback:.0%}"
            else:
                if favourable >= atr:
                    new_stop = highest - giveback * favourable
                    position["current_stop"] = (
                        new_stop if old_stop is None
                        else max(float(old_stop), new_stop)
                    )
                if bars >= EXIT_MAX_BARS:
                    exit_price, reason = close, "mfe_timeout"

        elif method == "hybrid_target_atr1.5":
            old_stop = position.get("current_stop")
            if old_stop is None:
                old_stop = entry - 1.5 * atr

            if low <= float(old_stop):
                exit_price = (
                    0.5 * target + 0.5 * float(old_stop)
                    if target_hit else float(old_stop)
                )
                reason = "hybrid_target_plus_trail" if target_hit else "hybrid_trail"
            else:
                if (
                    not target_hit and np.isfinite(target)
                    and target > entry and high >= target
                ):
                    target_hit = True
                position["current_stop"] = max(
                    float(old_stop), highest - 1.5 * atr
                )
                if bars >= EXIT_MAX_BARS:
                    exit_price = (
                        0.5 * target + 0.5 * close
                        if target_hit else close
                    )
                    reason = "hybrid_timeout"

        position["bars_held"] = bars
        position["highest_price"] = highest
        position["target_hit"] = target_hit
        active_positions[symbol] = position

        if exit_price is not None:
            net_return = exit_price / entry - 1 - FEE_SLIPPAGE
            position["status"] = "closed"
            position["exit_time"] = latest["open_time"].isoformat()
            position["exit_price"] = exit_price
            position["exit_reason"] = reason
            position["net_return"] = net_return
            active_positions[symbol] = position
            save_json(ACTIVE_POSITIONS_FILE, active_positions)

            send_tg(f"""🏁 Tide EXIT ALERT: {symbol}

Exit method: {method}
Exit reason: {reason}
Entry price: {entry:.8g}
Research exit price: {exit_price:.8g}
Estimated net return: {net_return:.2%}
Bars held: {bars}
Hours held: {bars * 0.25:.2f}

This is a research alert, not an exchange order.""")
            return

        save_json(ACTIVE_POSITIONS_FILE, active_positions)


# ============================================================
# Realtime monitor
# ============================================================

def initialise_market_data(symbols: list[str]) -> None:
    for idx, symbol in enumerate(symbols, start=1):
        log(f"Loading live [{idx}/{len(symbols)}] {symbol}")
        market_data[symbol] = {
            "15": fetch_klines(symbol, "15", 4),
            "60": fetch_klines(symbol, "60", 14),
        }
        time.sleep(0.1)


def update_candle(symbol: str, interval: str, candle: dict) -> None:
    row = {
        "open_time": pd.to_datetime(int(candle["start"]), unit="ms", utc=True),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
        "turnover": float(candle.get("turnover") or 0),
    }
    with data_lock:
        df = market_data[symbol][interval]
        df = df[df["open_time"] != row["open_time"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        market_data[symbol][interval] = (
            df.sort_values("open_time")
            .drop_duplicates("open_time")
            .tail(500)
            .reset_index(drop=True)
        )


def send_live_alert(
    symbol: str,
    latest: pd.Series,
    alert_type: str,
    minutes_to_close: float,
) -> None:
    signal_key = latest["open_time"].isoformat()
    if already_alerted(symbol, alert_type, signal_key):
        return

    meta = live_metadata.get(symbol, {})
    hist_score = meta.get("historical_score")
    score = signal_score(latest, hist_score)
    tier_label, tier_multiplier = risk_tier(score)
    suggested_margin = BASE_MARGIN_USDT * tier_multiplier
    exit_method = symbol_exit_method(symbol)

    title = (
        f"⚠️ Tide PRE-SIGNAL: {symbol}"
        if alert_type == "pre"
        else f"🚨 Tide CONFIRMED SIGNAL: {symbol}"
    )
    note = (
        "The 15m candle is still open and the setup can disappear."
        if alert_type == "pre"
        else "The 15m candle closed and the research position tracker has started."
    )

    send_tg(f"""{title}

Candle close UTC: {latest['open_time'] + pd.Timedelta(minutes=15)}
Entry/current price: {latest['close']:.8g}
1h regime: {latest['regime']}
Rolling low: {latest['rolling_low']:.8g}
Rolling high: {latest['rolling_high']:.8g}
Signal low: {latest['low']:.8g}
ATR14: {latest['atr14']:.8g}
Lower wick ratio: {latest['lower_wick_ratio']:.4f}
Volume multiple: {latest['volume_multiple']:.2f}

Signal score: {score:.1f}/100
Research position tier: {tier_label}
Example margin: {suggested_margin:.0f} USDT

Learned exit method: {exit_method}
{note}
Do not chase delayed alerts.""")

    if alert_type == "confirmed":
        open_research_position(symbol, latest)


def calculate_live_signal(symbol: str, confirmed: bool) -> None:
    with data_lock:
        df15 = market_data[symbol]["15"].copy()
        df1h = market_data[symbol]["60"].copy()

    df = model_frame(df15, df1h)
    latest = df.iloc[-1]
    candle_close = latest["open_time"] + pd.Timedelta(minutes=15)
    minutes_to_close = (
        candle_close - pd.Timestamp.now(tz="UTC")
    ).total_seconds() / 60
    signal = bool(latest["raw_signal"])

    if confirmed:
        realtime_exit_update(symbol, latest)

    log(
        symbol,
        f"confirmed={confirmed}",
        f"regime={latest['regime']}",
        f"signal={signal}",
    )
    if not signal:
        return
    if confirmed:
        send_live_alert(symbol, latest, "confirmed", 0.0)
    elif 0 <= minutes_to_close <= PRE_ALERT_WINDOW_MINUTES:
        send_live_alert(symbol, latest, "pre", minutes_to_close)


def make_callback(symbol: str, interval: str):
    def callback(message: dict) -> None:
        try:
            for candle in message.get("data", []):
                confirmed = bool(candle.get("confirm", False))
                update_candle(symbol, interval, candle)
                if interval == "15":
                    calculate_live_signal(symbol, confirmed)
                elif confirmed:
                    log("Closed 1h candle", symbol)
        except Exception as exc:
            log("Callback error", symbol, interval, repr(exc))
    return callback


def start_monitor(symbols: list[str]) -> None:
    if not symbols:
        send_tg(
            "⚠️ V4.4 found no eligible coin currently in a 1h downtrend.\n"
            f"It will refresh again in {WATCHLIST_REFRESH_HOURS} hours."
        )
        started = time.monotonic()
        while True:
            if (time.monotonic() - started) / 3600 >= WATCHLIST_REFRESH_HOURS:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            time.sleep(60)

    initialise_market_data(symbols)
    websocket = WebSocket(testnet=False, channel_type="linear")

    for symbol in symbols:
        websocket.kline_stream(
            interval=15,
            symbol=symbol,
            callback=make_callback(symbol, "15"),
        )
        websocket.kline_stream(
            interval=60,
            symbol=symbol,
            callback=make_callback(symbol, "60"),
        )

    send_tg(
        "✅ Tide Universe V4.4 realtime monitor started\n\n"
        + "\n".join(
            f"{symbol} → {symbol_exit_method(symbol)}"
            for symbol in symbols
        )
        + f"\n\nWatchlist refresh: every {WATCHLIST_REFRESH_HOURS} hours"
        + f"\nFull learning refresh: every {FULL_RESCAN_HOURS} hours"
    )

    started = time.monotonic()
    while True:
        if (time.monotonic() - started) / 3600 >= WATCHLIST_REFRESH_HOURS:
            log("Watchlist refresh due; restarting.")
            os.execv(sys.executable, [sys.executable, *sys.argv])
        log("Heartbeat monitoring", len(symbols), "symbols")
        time.sleep(60)


def main() -> None:
    stage2 = load_or_run_stage2()
    selected = select_current_watchlist(stage2)
    start_monitor(selected)


if __name__ == "__main__":
    main()
