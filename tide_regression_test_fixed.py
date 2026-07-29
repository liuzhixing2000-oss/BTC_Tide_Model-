
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP


# ============================================================
# Regression test: BTC / ETH
# Compares:
# A) raw_all_signals
# B) cooldown_first_6h
# C) cluster_last_18h (historical-only, uses future information)
# ============================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

DAYS_15M = 90
DAYS_1H = 120

LOOKBACK_BARS = 24          # 6h on 15m
VOLUME_LOOKBACK = 24
VOLUME_MULTIPLIER = 1.5
LOWER_WICK_THRESHOLD = 0.35

HOLD_BARS = 24              # fixed 6h
COOLDOWN_BARS = 24          # 6h
CLUSTER_GAP_BARS = 72       # 18h
FEE_SLIPPAGE = 0.001        # 0.1%

OUTPUT_DIR = Path("regression_output")
OUTPUT_DIR.mkdir(exist_ok=True)

http = HTTP(testnet=False)


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

        response = http.get_kline(**kwargs)
        batch = response["result"].get("list", [])

        if not batch:
            break

        rows.extend(batch)
        oldest = min(int(row[0]) for row in batch)
        end_ms = oldest - 1

        if len(batch) < limit:
            break

        time.sleep(0.08)

    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    frame = pd.DataFrame(
        [
            {
                "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
        ]
    )

    return (
        frame.sort_values("open_time")
        .drop_duplicates("open_time")
        .tail(required)
        .reset_index(drop=True)
    )


def build_model_frame(df15: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    df15 = df15.copy()
    df1h = df1h.copy()

    # 1h signal becomes available only AFTER the 1h candle closes.
    df1h["close_time_1h"] = df1h["open_time"] + pd.Timedelta(hours=1)

    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()

    df1h["regime_1h"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(
            df1h["ma50"] > df1h["ma200"],
            "uptrend",
            "range",
        ),
    )

    regime = df1h[
        ["close_time_1h", "regime_1h", "ma50", "ma200"]
    ].dropna().sort_values("close_time_1h")

    df = pd.merge_asof(
        df15.sort_values("open_time"),
        regime,
        left_on="open_time",
        right_on="close_time_1h",
        direction="backward",
    )

    df["rolling_low"] = (
        df["low"]
        .rolling(LOOKBACK_BARS)
        .min()
        .shift(1)
    )

    df["avg_volume"] = (
        df["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
        .shift(1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    )

    df["candle_range"] = df["high"] - df["low"]

    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] / df["candle_range"],
        0,
    )

    df["volume_multiple"] = np.where(
        df["avg_volume"] > 0,
        df["volume"] / df["avg_volume"],
        0,
    )

    df["raw_signal"] = (
        (df["regime_1h"] == "downtrend")
        & (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > LOWER_WICK_THRESHOLD)
        & (df["volume_multiple"] > VOLUME_MULTIPLIER)
    )

    return df


def select_raw_all(df: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(df["raw_signal"].to_numpy())


def select_cooldown_first(df: pd.DataFrame) -> np.ndarray:
    selected = []
    last_selected = -10**9

    for idx in np.flatnonzero(df["raw_signal"].to_numpy()):
        if idx - last_selected >= COOLDOWN_BARS:
            selected.append(idx)
            last_selected = idx

    return np.array(selected, dtype=int)


def select_cluster_last(df: pd.DataFrame) -> np.ndarray:
    """
    Historical-only comparison.
    Signals separated by <=18h belong to one cluster;
    only the LAST signal in each cluster is selected.
    This uses future information and is not directly live-tradable.
    """
    raw_indices = np.flatnonzero(df["raw_signal"].to_numpy())

    if len(raw_indices) == 0:
        return np.array([], dtype=int)

    selected = []
    cluster_last = raw_indices[0]

    for idx in raw_indices[1:]:
        if idx - cluster_last <= CLUSTER_GAP_BARS:
            cluster_last = idx
        else:
            selected.append(cluster_last)
            cluster_last = idx

    selected.append(cluster_last)
    return np.array(selected, dtype=int)


def make_trades(
    df: pd.DataFrame,
    entry_indices: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    rows = []

    for entry_idx in entry_indices:
        exit_idx = entry_idx + HOLD_BARS

        if exit_idx >= len(df):
            continue

        entry = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        gross_return = exit_row["close"] / entry["close"] - 1
        net_return = gross_return - FEE_SLIPPAGE

        rows.append(
            {
                "mode": mode,
                "entry_idx": int(entry_idx),
                "entry_time": entry["open_time"],
                "exit_time": exit_row["open_time"],
                "entry_price": float(entry["close"]),
                "exit_price": float(exit_row["close"]),
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "regime_1h": entry["regime_1h"],
                "rolling_low": float(entry["rolling_low"]),
                "signal_low": float(entry["low"]),
                "lower_wick_ratio": float(entry["lower_wick_ratio"]),
                "volume_multiple": float(entry["volume_multiple"]),
            }
        )

    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "total_return": 0.0,
            "average_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
        }

    returns = trades["net_return"].to_numpy(dtype=float)
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1

    return {
        "trades": int(len(trades)),
        "total_return": float(equity[-1] - 1),
        "average_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.min(drawdown)),
    }


def run_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nDownloading {symbol} ...")

    df15 = fetch_klines(symbol, "15", DAYS_15M)
    df1h = fetch_klines(symbol, "60", DAYS_1H)

    print(
        f"{symbol} 15m range:",
        df15["open_time"].min(),
        "→",
        df15["open_time"].max(),
        f"({len(df15)} bars)",
    )

    print(
        f"{symbol} 1h range:",
        df1h["open_time"].min(),
        "→",
        df1h["open_time"].max(),
        f"({len(df1h)} bars)",
    )

    df = build_model_frame(df15, df1h)

    modes = {
        "raw_all_signals": select_raw_all(df),
        "cooldown_first_6h": select_cooldown_first(df),
        "cluster_last_18h": select_cluster_last(df),
    }

    print("\n------------------------------")
    print(f"{symbol} SIGNAL COUNTS")
    print("------------------------------")
    print(f"Raw signals: {len(modes['raw_all_signals'])}")
    print(f"Cooldown first 6h: {len(modes['cooldown_first_6h'])}")
    print(f"Cluster last 18h: {len(modes['cluster_last_18h'])}")

    summaries = []
    all_trades = []

    for mode, indices in modes.items():
        trades = make_trades(df, indices, mode)
        summary = summarise(trades)
        summary["symbol"] = symbol
        summary["mode"] = mode

        summaries.append(summary)
        all_trades.append(trades)

        print(
            f"{symbol} | {mode} | "
            f"trades={summary['trades']} | "
            f"return={summary['total_return']:.2%} | "
            f"win={summary['win_rate']:.1%} | "
            f"median={summary['median_return']:.3%} | "
            f"DD={summary['max_drawdown']:.2%}"
        )

    summary_df = pd.DataFrame(summaries)

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame()
    )

    summary_df.to_csv(
        OUTPUT_DIR / f"{symbol}_summary.csv",
        index=False,
    )

    trades_df.to_csv(
        OUTPUT_DIR / f"{symbol}_trades.csv",
        index=False,
    )

    return summary_df, trades_df


all_summaries = []
all_trade_frames = []

for symbol in SYMBOLS:
    summary_df, trades_df = run_symbol(symbol)
    all_summaries.append(summary_df)
    all_trade_frames.append(trades_df)

combined_summary = pd.concat(all_summaries, ignore_index=True)
combined_trades = pd.concat(all_trade_frames, ignore_index=True)

combined_summary.to_csv(
    OUTPUT_DIR / "combined_summary.csv",
    index=False,
)

combined_trades.to_csv(
    OUTPUT_DIR / "combined_trades.csv",
    index=False,
)

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

display_columns = [
    "symbol",
    "mode",
    "trades",
    "total_return",
    "average_return",
    "median_return",
    "win_rate",
    "max_drawdown",
]

print(combined_summary[display_columns].to_string(index=False))

print("\nFiles saved in:", OUTPUT_DIR.resolve())
