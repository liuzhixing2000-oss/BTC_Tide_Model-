import json
import os
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from pybit.unified_trading import HTTP, WebSocket


# ============================================================
# Tide Universe V7 Adaptive
# Automatic universe selection + daily per-symbol stop/exit optimisation
# + realtime entry, logic-failure stop, and smart-exit alerts
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
# Two-level eligibility:
# strict = preferred production candidates
# relaxed = additional candidates allowed with a score penalty
MIN_STAGE2_TRADES = 20
MIN_VALIDATION_TRADES = 5
MAX_ALLOWED_DRAWDOWN = -0.12

RELAXED_MIN_STAGE2_TRADES = 15
RELAXED_MIN_VALIDATION_TRADES = 4
RELAXED_MAX_ALLOWED_DRAWDOWN = -0.15
RELAXED_SCORE_PENALTY = 0.90

# Daily exit-method stability. A method is not changed for a tiny statistical edge.
EXIT_SWITCH_MIN_SCORE_GAIN = 0.08
EXIT_SWITCH_MIN_EXPECTANCY_GAIN = 0.0005

# ---------- Conservative entry-parameter optimisation ----------
# Only a small, pre-defined candidate library is tested. This avoids a huge
# brute-force grid and reduces daily overfitting.
ENTRY_PARAMETER_CANDIDATES = [
    {"lookback_bars": 20, "volume_lookback": 24, "volume_multiplier": 1.4,
     "lower_wick_threshold": 0.30, "cooldown_bars": 24},
    {"lookback_bars": 24, "volume_lookback": 24, "volume_multiplier": 1.5,
     "lower_wick_threshold": 0.35, "cooldown_bars": 24},
    {"lookback_bars": 28, "volume_lookback": 24, "volume_multiplier": 1.5,
     "lower_wick_threshold": 0.35, "cooldown_bars": 24},
    {"lookback_bars": 24, "volume_lookback": 32, "volume_multiplier": 1.6,
     "lower_wick_threshold": 0.35, "cooldown_bars": 24},
    {"lookback_bars": 32, "volume_lookback": 32, "volume_multiplier": 1.6,
     "lower_wick_threshold": 0.40, "cooldown_bars": 24},
    {"lookback_bars": 24, "volume_lookback": 24, "volume_multiplier": 1.7,
     "lower_wick_threshold": 0.40, "cooldown_bars": 32},
    {"lookback_bars": 20, "volume_lookback": 20, "volume_multiplier": 1.6,
     "lower_wick_threshold": 0.40, "cooldown_bars": 32},
    {"lookback_bars": 28, "volume_lookback": 28, "volume_multiplier": 1.4,
     "lower_wick_threshold": 0.35, "cooldown_bars": 32},
]
ENTRY_PARAM_MIN_TOTAL_TRADES = 60
ENTRY_PARAM_MIN_VALIDATION_TRADES = 18
ENTRY_PARAM_SWITCH_MIN_SCORE_GAIN = 0.07
ENTRY_PARAM_SWITCH_MIN_EXPECTANCY_GAIN = 0.00035

# ---------- Bounded online learning ----------
# The live learner changes only the four signal-score weights. It never changes
# the core entry rules, and it requires a minimum live sample before adapting.
DEFAULT_SIGNAL_WEIGHTS = {
    "historical": 0.40,
    "wick": 0.25,
    "volume": 0.20,
    "close_position": 0.15,
}
ONLINE_LEARNING_MIN_TRADES = 20
ONLINE_LEARNING_MAX_TRADES = 200
ONLINE_LEARNING_MAX_BLEND = 0.35

# Market-environment controls for this long-rebound strategy.
MARKET_REGIME_SAMPLE_SIZE = 12
SUPPORTIVE_MONITOR_CAP = 8
CAUTION_MONITOR_CAP = 5
HOSTILE_MONITOR_CAP = 2
SUPPORTIVE_MIN_SIGNAL_SCORE = 60.0
CAUTION_MIN_SIGNAL_SCORE = 70.0
HOSTILE_MIN_SIGNAL_SCORE = 82.0

# ---------- Daily risk/exit optimisation ----------
EXIT_MAX_BARS = 48                  # 12 hours on 15m candles
DEFAULT_EXIT_METHOD = "baseline_fixed6_no_stop"

# Every Stage-2 symbol is tested against all strategies below each morning.
# The chosen strategy is frozen into each new position at entry time.
EXIT_METHOD_NAMES = [
    "baseline_fixed6_no_stop",
    "structure_fixed6",
    "structure_mfe30_act075",
    "structure_mfe40_act100",
    "structure_mfe50_act125",
    "structure_time_staged_mfe",
    "structure_time_smart_reversal",
    "structure_atr_trail_1.5",
    "structure_rolling_high",
]

# Shared stop parameters. These are deliberately simple enough to avoid
# an excessive parameter search and daily overfitting.
STRUCTURE_BUFFER_ATR = 0.20
CATASTROPHE_STOP_ATR = 2.00
TIME_STOP_BARS = 16                 # 4 hours
TIME_STOP_MIN_MFE_ATR = 0.30

# Smart-reversal parameters
REVERSAL_VOLUME_MULTIPLE = 1.80
REVERSAL_BODY_RATIO = 0.60
REVERSAL_SWING_LOOKBACK = 3

# Do not issue a logic warning on the first small intrabar dip.
LOGIC_WARNING_CLOSES_BELOW_STRUCTURE = 1
LOGIC_EXIT_CLOSES_BELOW_STRUCTURE = 2

# ---------- Realtime ----------
PRE_ALERT_WINDOW_MINUTES = 3
# Full Stage1 + Stage2 + exit optimisation runs once each Sydney morning.
DAILY_RESCAN_LOCAL_HOUR = 8
LOCAL_TIMEZONE = ZoneInfo("Australia/Sydney")
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
MARKET_REGIME_JSON = Path("market_regime.json")
ENTRY_PARAM_CONFIG_JSON = Path("entry_parameter_config.json")
ENTRY_PARAM_RESULTS_CSV = Path("entry_parameter_results.csv")
LIVE_TRADE_JOURNAL_CSV = Path("live_trade_journal.csv")
ONLINE_LEARNING_JSON = Path("online_learning.json")

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


def default_entry_params() -> dict:
    return {
        "lookback_bars": LOOKBACK_BARS,
        "volume_lookback": VOLUME_LOOKBACK,
        "volume_multiplier": VOLUME_MULTIPLIER,
        "lower_wick_threshold": LOWER_WICK_THRESHOLD,
        "cooldown_bars": COOLDOWN_BARS,
    }


def load_entry_params() -> dict:
    config = load_json(ENTRY_PARAM_CONFIG_JSON, {})
    params = config.get("chosen_parameters", default_entry_params())
    merged = default_entry_params()
    merged.update(params)
    return merged


def load_signal_weights() -> dict:
    state = load_json(ONLINE_LEARNING_JSON, {})
    weights = state.get("weights", DEFAULT_SIGNAL_WEIGHTS)
    merged = DEFAULT_SIGNAL_WEIGHTS.copy()
    merged.update(weights)
    total = sum(max(0.0, float(value)) for value in merged.values())
    if total <= 0:
        return DEFAULT_SIGNAL_WEIGHTS.copy()
    return {
        key: max(0.0, float(value)) / total
        for key, value in merged.items()
    }


ACTIVE_ENTRY_PARAMS = load_entry_params()
ACTIVE_SIGNAL_WEIGHTS = load_signal_weights()


def parameter_key(params: dict) -> str:
    return (
        f"lb{int(params['lookback_bars'])}_"
        f"vl{int(params['volume_lookback'])}_"
        f"vm{float(params['volume_multiplier']):.2f}_"
        f"wick{float(params['lower_wick_threshold']):.2f}_"
        f"cd{int(params['cooldown_bars'])}"
    )


def append_live_trade(row: dict) -> None:
    frame = pd.DataFrame([row])
    header = not LIVE_TRADE_JOURNAL_CSV.exists()
    frame.to_csv(
        LIVE_TRADE_JOURNAL_CSV,
        mode="a",
        header=header,
        index=False,
    )


def update_online_signal_weights() -> dict:
    global ACTIVE_SIGNAL_WEIGHTS

    if not LIVE_TRADE_JOURNAL_CSV.exists():
        ACTIVE_SIGNAL_WEIGHTS = DEFAULT_SIGNAL_WEIGHTS.copy()
        save_json(ONLINE_LEARNING_JSON, {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": 0,
            "weights": ACTIVE_SIGNAL_WEIGHTS,
            "note": "waiting_for_live_trade_history",
        })
        return ACTIVE_SIGNAL_WEIGHTS

    try:
        journal = pd.read_csv(LIVE_TRADE_JOURNAL_CSV).tail(
            ONLINE_LEARNING_MAX_TRADES
        )
    except Exception as exc:
        log("Online journal load failed", repr(exc))
        return ACTIVE_SIGNAL_WEIGHTS

    feature_columns = {
        "historical": "historical_component",
        "wick": "wick_component",
        "volume": "volume_component",
        "close_position": "close_position_component",
    }
    required = [*feature_columns.values(), "net_return"]
    usable = journal.dropna(subset=required).copy()

    if len(usable) < ONLINE_LEARNING_MIN_TRADES:
        ACTIVE_SIGNAL_WEIGHTS = DEFAULT_SIGNAL_WEIGHTS.copy()
        save_json(ONLINE_LEARNING_JSON, {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": int(len(usable)),
            "weights": ACTIVE_SIGNAL_WEIGHTS,
            "note": "minimum_live_sample_not_reached",
        })
        return ACTIVE_SIGNAL_WEIGHTS

    correlations = {}
    adjusted = {}
    for key, column in feature_columns.items():
        ranked_feature = usable[column].rank(method="average")
        ranked_return = usable["net_return"].rank(method="average")
        corr = ranked_feature.corr(ranked_return)
        corr = 0.0 if pd.isna(corr) else float(np.clip(corr, -0.30, 0.30))
        correlations[key] = corr
        adjusted[key] = DEFAULT_SIGNAL_WEIGHTS[key] * float(np.exp(1.5 * corr))

    adjusted_total = sum(adjusted.values())
    learned = {
        key: value / adjusted_total
        for key, value in adjusted.items()
    }
    progress = min(
        1.0,
        (len(usable) - ONLINE_LEARNING_MIN_TRADES) / 100.0,
    )
    blend = ONLINE_LEARNING_MAX_BLEND * progress
    blended = {
        key: (
            (1 - blend) * DEFAULT_SIGNAL_WEIGHTS[key]
            + blend * learned[key]
        )
        for key in DEFAULT_SIGNAL_WEIGHTS
    }

    # Hard bounds stop a small live sample from dominating the score.
    bounded = {
        key: float(np.clip(value, 0.10, 0.55))
        for key, value in blended.items()
    }
    total = sum(bounded.values())
    ACTIVE_SIGNAL_WEIGHTS = {
        key: value / total
        for key, value in bounded.items()
    }
    save_json(ONLINE_LEARNING_JSON, {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(len(usable)),
        "blend": blend,
        "rank_correlations": correlations,
        "weights": ACTIVE_SIGNAL_WEIGHTS,
        "note": "bounded_rank_correlation_update",
    })
    return ACTIVE_SIGNAL_WEIGHTS


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
            files={"photo": ("tide_v45_ranking.png", image, "image/png")},
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

def model_frame(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    entry_params: dict | None = None,
) -> pd.DataFrame:
    params = ACTIVE_ENTRY_PARAMS if entry_params is None else entry_params
    lookback_bars = int(params["lookback_bars"])
    volume_lookback = int(params["volume_lookback"])
    volume_multiplier = float(params["volume_multiplier"])
    lower_wick_threshold = float(params["lower_wick_threshold"])
    cooldown_bars = int(params["cooldown_bars"])

    df15 = df15.copy()
    df1h = df1h.copy()

    # A 1h candle may only influence 15m bars after that 1h candle closes.
    # This removes the look-ahead bias present when both frames are merged
    # directly on their opening timestamps.
    df15["open_time"] = pd.to_datetime(
        df15["open_time"], utc=True
    ).astype("datetime64[ns, UTC]")
    df1h["open_time"] = pd.to_datetime(
        df1h["open_time"], utc=True
    ).astype("datetime64[ns, UTC]")
    df1h["close_time_1h"] = (
        df1h["open_time"] + pd.Timedelta(hours=1)
    ).astype("datetime64[ns, UTC]")

    df1h["ma50"] = df1h["close"].rolling(50).mean()
    df1h["ma200"] = df1h["close"].rolling(200).mean()
    df1h["regime"] = np.where(
        df1h["ma50"] < df1h["ma200"],
        "downtrend",
        np.where(df1h["ma50"] > df1h["ma200"], "uptrend", "range"),
    )

    regime = (
        df1h[["close_time_1h", "regime", "ma50", "ma200"]]
        .dropna()
        .sort_values("close_time_1h")
    )
    df = pd.merge_asof(
        df15.sort_values("open_time"),
        regime,
        left_on="open_time",
        right_on="close_time_1h",
        direction="backward",
    )

    df["rolling_low"] = df["low"].rolling(lookback_bars).min().shift(1)
    df["rolling_high"] = df["high"].rolling(lookback_bars).max().shift(1)
    df["avg_volume"] = df["volume"].rolling(volume_lookback).mean().shift(1)
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
    df["body_ratio"] = np.where(
        df["candle_range"] > 0,
        (df["close"] - df["open"]).abs() / df["candle_range"],
        0.0,
    )

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # RSI is recorded for diagnostics. It is not used alone as an exit trigger.
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    df["raw_signal"] = (
        (df["regime"] == "downtrend")
        & (df["low"] < df["rolling_low"])
        & (df["close"] > df["rolling_low"])
        & (df["lower_wick_ratio"] > lower_wick_threshold)
        & (df["volume_multiple"] > volume_multiplier)
    )

    accepted = np.zeros(len(df), dtype=bool)
    last_entry = -10**9
    for idx in np.flatnonzero(df["raw_signal"].to_numpy()):
        if idx - last_entry >= cooldown_bars:
            accepted[idx] = True
            last_entry = idx
    df["signal"] = accepted
    return df


# ============================================================
# Daily stop/exit backtest engine
# ============================================================

@dataclass(frozen=True)
class ExitResult:
    exit_idx: int
    exit_price: float
    reason: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    stop_mode: str                 # none | structure | structure_time
    exit_mode: str                 # fixed6 | delayed_mfe | staged_mfe |
                                    # smart_reversal | atr_trail |
                                    # rolling_high
    mfe_activation_atr: float = 1.0
    mfe_giveback: float = 0.40
    atr_trail_multiple: float = 1.5


STRATEGY_SPECS = {
    "baseline_fixed6_no_stop": StrategySpec(
        "baseline_fixed6_no_stop", "none", "fixed6"
    ),
    "structure_fixed6": StrategySpec(
        "structure_fixed6", "structure", "fixed6"
    ),
    "structure_mfe30_act075": StrategySpec(
        "structure_mfe30_act075", "structure", "delayed_mfe",
        mfe_activation_atr=0.75, mfe_giveback=0.30,
    ),
    "structure_mfe40_act100": StrategySpec(
        "structure_mfe40_act100", "structure", "delayed_mfe",
        mfe_activation_atr=1.00, mfe_giveback=0.40,
    ),
    "structure_mfe50_act125": StrategySpec(
        "structure_mfe50_act125", "structure", "delayed_mfe",
        mfe_activation_atr=1.25, mfe_giveback=0.50,
    ),
    "structure_time_staged_mfe": StrategySpec(
        "structure_time_staged_mfe", "structure_time", "staged_mfe",
        mfe_activation_atr=1.00,
    ),
    "structure_time_smart_reversal": StrategySpec(
        "structure_time_smart_reversal", "structure_time", "smart_reversal",
        mfe_activation_atr=1.00,
    ),
    "structure_atr_trail_1.5": StrategySpec(
        "structure_atr_trail_1.5", "structure", "atr_trail",
        atr_trail_multiple=1.5,
    ),
    "structure_rolling_high": StrategySpec(
        "structure_rolling_high", "structure", "rolling_high"
    ),
}


def staged_giveback(mfe_atr: float) -> float:
    """Give normal rebounds room, protect mature moves, and avoid choking trends."""
    if mfe_atr < 2.0:
        return 0.50
    if mfe_atr < 4.0:
        return 0.40
    return 0.45


def strategy_spec(method: str) -> StrategySpec:
    try:
        return STRATEGY_SPECS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {method}") from exc


def run_exit_method(
    df: pd.DataFrame,
    entry_idx: int,
    method: str,
) -> ExitResult | None:
    spec = strategy_spec(method)
    entry_row = df.iloc[entry_idx]
    entry = float(entry_row["close"])
    atr = float(entry_row["atr14"])
    signal_low = float(entry_row["low"])
    rolling_high_target = float(entry_row["rolling_high"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    last = min(entry_idx + EXIT_MAX_BARS, len(df) - 1)
    if last <= entry_idx:
        return None

    # The tighter of structure and catastrophe protection is used.
    structure_stop = signal_low - STRUCTURE_BUFFER_ATR * atr
    catastrophe_stop = entry - CATASTROPHE_STOP_ATR * atr
    hard_stop = max(structure_stop, catastrophe_stop)

    highest = entry
    current_profit_stop = -np.inf
    closes_below_structure = 0
    activated = False

    for idx in range(entry_idx + 1, last + 1):
        row = df.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        open_price = float(row["open"])

        # Conservative intrabar ordering: an existing stop is checked before
        # the same candle is allowed to create a new high and raise that stop.
        if spec.stop_mode != "none" and low <= hard_stop:
            return ExitResult(idx, hard_stop, "hard_structure_or_catastrophe_stop")

        if np.isfinite(current_profit_stop) and low <= current_profit_stop:
            return ExitResult(idx, current_profit_stop, "profit_protection_stop")

        structure_level = signal_low
        if close < structure_level:
            closes_below_structure += 1
        else:
            closes_below_structure = 0

        bearish = close < open_price
        volume_reversal = (
            bearish
            and float(row["volume_multiple"]) >= REVERSAL_VOLUME_MULTIPLE
            and float(row["body_ratio"]) >= REVERSAL_BODY_RATIO
        )
        previous_swing_low = (
            float(df.iloc[max(entry_idx, idx - REVERSAL_SWING_LOOKBACK):idx]["low"].min())
            if idx > entry_idx + 1 else structure_level
        )
        broke_swing_low = close < previous_swing_low

        # Confirmed logic failure: two closed 15m candles below the signal low,
        # or a strong high-volume bearish close below the reclaimed level.
        if spec.stop_mode != "none":
            if closes_below_structure >= LOGIC_EXIT_CLOSES_BELOW_STRUCTURE:
                return ExitResult(idx, close, "logic_failure_two_closes_below_signal_low")
            if close < structure_level and volume_reversal:
                return ExitResult(idx, close, "logic_failure_volume_breakdown")

        highest = max(highest, high)
        favourable = highest - entry
        mfe_atr = favourable / atr

        if (
            spec.stop_mode == "structure_time"
            and idx - entry_idx >= TIME_STOP_BARS
            and mfe_atr < TIME_STOP_MIN_MFE_ATR
        ):
            return ExitResult(idx, close, "time_invalidation_low_mfe")

        if spec.exit_mode == "fixed6":
            if idx - entry_idx >= 24:
                return ExitResult(idx, close, "fixed_6h")

        elif spec.exit_mode == "delayed_mfe":
            if mfe_atr >= spec.mfe_activation_atr:
                activated = True
            if activated:
                candidate = highest - spec.mfe_giveback * favourable
                current_profit_stop = max(current_profit_stop, candidate)

        elif spec.exit_mode == "staged_mfe":
            if mfe_atr >= spec.mfe_activation_atr:
                activated = True
            if activated:
                giveback = staged_giveback(mfe_atr)
                candidate = highest - giveback * favourable
                current_profit_stop = max(current_profit_stop, candidate)

        elif spec.exit_mode == "smart_reversal":
            # A smart exit requires an established profit first. A single
            # volume candle is not enough; it must also damage short structure.
            if mfe_atr >= spec.mfe_activation_atr:
                activated = True
            if activated:
                giveback = staged_giveback(mfe_atr)
                candidate = highest - giveback * favourable
                current_profit_stop = max(current_profit_stop, candidate)

                if volume_reversal and broke_swing_low:
                    return ExitResult(idx, close, "smart_volume_structure_reversal")

        elif spec.exit_mode == "atr_trail":
            if favourable >= atr:
                activated = True
            if activated:
                candidate = highest - spec.atr_trail_multiple * atr
                current_profit_stop = max(current_profit_stop, candidate)

        elif spec.exit_mode == "rolling_high":
            if (
                np.isfinite(rolling_high_target)
                and rolling_high_target > entry
                and high >= rolling_high_target
            ):
                return ExitResult(idx, rolling_high_target, "rolling_high_target")

        else:
            raise ValueError(f"Unknown exit mode: {spec.exit_mode}")

    return ExitResult(last, float(df.iloc[last]["close"]), "maximum_12h_timeout")


def trade_frame(df: pd.DataFrame, method: str = DEFAULT_EXIT_METHOD) -> pd.DataFrame:
    trades = []
    for entry_idx in np.flatnonzero(df["signal"].to_numpy()):
        result = run_exit_method(df, entry_idx, method)
        if result is None:
            continue

        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        net_return = result.exit_price / entry_price - 1 - FEE_SLIPPAGE

        # Diagnostics for whether an exit sold too early.
        post_end = min(result.exit_idx + 24, len(df) - 1)
        post_high = float(
            df.iloc[result.exit_idx + 1:post_end + 1]["high"].max()
        ) if post_end > result.exit_idx else result.exit_price
        post_exit_upside = max(0.0, post_high / result.exit_price - 1)

        trades.append({
            "entry_time": entry_row["open_time"],
            "exit_time": df.iloc[result.exit_idx]["open_time"],
            "entry_price": entry_price,
            "exit_price": result.exit_price,
            "net_return": net_return,
            "exit_reason": result.reason,
            "hold_bars": result.exit_idx - entry_idx,
            "signal_low": float(entry_row["low"]),
            "entry_atr": float(entry_row["atr14"]),
            "post_exit_upside_6h": post_exit_upside,
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
    pf = np.inf if losses == 0 and gains > 0 else (
        gains / losses if losses > 0 else np.nan
    )

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
    all_returns = (
        trades["net_return"].to_numpy(dtype=float)
        if not trades.empty else np.array([])
    )
    result = metrics(all_returns)
    validation_start = df["open_time"].max() - pd.Timedelta(days=VALIDATION_DAYS)

    if trades.empty:
        train_returns = np.array([])
        validation_returns = np.array([])
        sell_early = np.nan
        stop_rate = np.nan
    else:
        train_returns = trades.loc[
            trades["entry_time"] < validation_start, "net_return"
        ].to_numpy(dtype=float)
        validation_mask = trades["entry_time"] >= validation_start
        validation_returns = trades.loc[
            validation_mask, "net_return"
        ].to_numpy(dtype=float)
        validation_trades = trades.loc[validation_mask]
        sell_early = (
            float(validation_trades["post_exit_upside_6h"].median())
            if not validation_trades.empty else np.nan
        )
        stop_rate = (
            float(validation_trades["exit_reason"].str.contains(
                "stop|logic_failure|time_invalidation", regex=True
            ).mean())
            if not validation_trades.empty else np.nan
        )

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
        "validation_sell_early_median_6h": sell_early,
        "validation_stop_rate": stop_rate,
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
        pf_rank = pool["validation_profit_factor"].replace(
            np.inf, 999
        ).fillna(0).rank(pct=True)
        sell_early_rank = (
            -pool["validation_sell_early_median_6h"].fillna(1.0)
        ).rank(pct=True)

        score = (
            0.30 * pool["validation_expectancy"].rank(pct=True)
            + 0.18 * pool["validation_return"].rank(pct=True)
            + 0.12 * pool["validation_median"].rank(pct=True)
            + 0.10 * pool["validation_win_rate"].rank(pct=True)
            + 0.12 * pool["validation_max_drawdown"].rank(pct=True)
            + 0.08 * pf_rank
            + 0.05 * pool["average_return"].rank(pct=True)
            + 0.05 * sell_early_rank
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
        fallback = ranked[
            ranked["exit_method"] == DEFAULT_EXIT_METHOD
        ].iloc[0]
        chosen = fallback.to_dict()
        chosen["selection_note"] = "fallback_no_strategy_passed_robust_filter"
    else:
        chosen = eligible.iloc[0].to_dict()
        chosen["selection_note"] = "best_daily_stop_exit_validation_score"

    return chosen, ranked


# ============================================================
# Conservative daily entry-parameter optimiser
# ============================================================

def evaluate_entry_parameter_set(
    cached_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    params: dict,
) -> dict:
    symbol_rows = []
    for symbol, (df15, df1h) in cached_data.items():
        try:
            df = model_frame(df15, df1h, params)
            result = evaluate_method(symbol, df, DEFAULT_EXIT_METHOD)
            symbol_rows.append(result)
        except Exception as exc:
            log("Entry parameter evaluation failed", symbol, repr(exc))

    if not symbol_rows:
        return {
            "parameter_key": parameter_key(params),
            **params,
            "symbols_tested": 0,
            "total_trades": 0,
            "validation_trades": 0,
            "mean_validation_expectancy": np.nan,
            "median_validation_return": np.nan,
            "positive_validation_share": 0.0,
            "positive_train_share": 0.0,
            "worst_drawdown": -1.0,
            "eligible": False,
        }

    frame = pd.DataFrame(symbol_rows)
    validation_expectancy = frame["validation_expectancy"].replace(
        [np.inf, -np.inf], np.nan
    )
    return {
        "parameter_key": parameter_key(params),
        **params,
        "symbols_tested": int(len(frame)),
        "total_trades": int(frame["trades"].sum()),
        "validation_trades": int(frame["validation_trades"].sum()),
        "mean_validation_expectancy": float(validation_expectancy.mean()),
        "median_validation_return": float(frame["validation_return"].median()),
        "positive_validation_share": float(
            (frame["validation_return"] > 0).mean()
        ),
        "positive_train_share": float((frame["train_return"] > 0).mean()),
        "worst_drawdown": float(frame["max_drawdown"].min()),
    }


def score_entry_parameters(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["eligible"] = (
        (rows["total_trades"] >= ENTRY_PARAM_MIN_TOTAL_TRADES)
        & (
            rows["validation_trades"]
            >= ENTRY_PARAM_MIN_VALIDATION_TRADES
        )
        & (rows["mean_validation_expectancy"] > 0)
        & (rows["median_validation_return"] > 0)
        & (rows["positive_validation_share"] >= 0.50)
        & (rows["positive_train_share"] >= 0.50)
        & (rows["worst_drawdown"] >= RELAXED_MAX_ALLOWED_DRAWDOWN)
    )
    rows["parameter_score"] = np.nan
    pool = rows[rows["eligible"]].copy()

    if not pool.empty:
        score = (
            0.30
            * pool["mean_validation_expectancy"].rank(pct=True)
            + 0.22
            * pool["median_validation_return"].rank(pct=True)
            + 0.18
            * pool["positive_validation_share"].rank(pct=True)
            + 0.10
            * pool["positive_train_share"].rank(pct=True)
            + 0.12
            * pool["worst_drawdown"].rank(pct=True)
            + 0.08
            * pool["validation_trades"].clip(upper=200).rank(pct=True)
        )
        rows.loc[pool.index, "parameter_score"] = score

    return rows.sort_values(
        [
            "eligible",
            "parameter_score",
            "mean_validation_expectancy",
            "median_validation_return",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    )


def choose_daily_entry_parameters(
    cached_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[dict, pd.DataFrame, str]:
    global ACTIVE_ENTRY_PARAMS

    candidates = []
    seen = set()
    for params in [load_entry_params(), *ENTRY_PARAMETER_CANDIDATES]:
        key = parameter_key(params)
        if key not in seen:
            candidates.append(params)
            seen.add(key)

    rows = pd.DataFrame([
        evaluate_entry_parameter_set(cached_data, params)
        for params in candidates
    ])
    ranked = score_entry_parameters(rows)
    previous = load_entry_params()
    previous_key = parameter_key(previous)

    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        chosen = previous
        note = "kept_previous_no_parameter_set_passed"
    else:
        best = eligible.iloc[0]
        chosen = {
            "lookback_bars": int(best["lookback_bars"]),
            "volume_lookback": int(best["volume_lookback"]),
            "volume_multiplier": float(best["volume_multiplier"]),
            "lower_wick_threshold": float(best["lower_wick_threshold"]),
            "cooldown_bars": int(best["cooldown_bars"]),
        }
        note = "best_bounded_daily_parameter_score"

        if parameter_key(chosen) != previous_key:
            old_rows = ranked[ranked["parameter_key"] == previous_key]
            if not old_rows.empty:
                old = old_rows.iloc[0]
                score_gain = (
                    float(best["parameter_score"])
                    - float(old["parameter_score"])
                    if pd.notna(best["parameter_score"])
                    and pd.notna(old["parameter_score"])
                    else -np.inf
                )
                expectancy_gain = (
                    float(best["mean_validation_expectancy"])
                    - float(old["mean_validation_expectancy"])
                    if pd.notna(best["mean_validation_expectancy"])
                    and pd.notna(old["mean_validation_expectancy"])
                    else -np.inf
                )
                if (
                    bool(old.get("eligible", False))
                    and (
                        score_gain < ENTRY_PARAM_SWITCH_MIN_SCORE_GAIN
                        or expectancy_gain
                        < ENTRY_PARAM_SWITCH_MIN_EXPECTANCY_GAIN
                    )
                ):
                    chosen = previous
                    note = "kept_previous_entry_params_stability_filter"

    ACTIVE_ENTRY_PARAMS = chosen.copy()
    save_json(ENTRY_PARAM_CONFIG_JSON, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_parameters": chosen,
        "selection_note": note,
        "previous_parameter_key": previous_key,
        "chosen_parameter_key": parameter_key(chosen),
    })
    ranked.to_csv(ENTRY_PARAM_RESULTS_CSV, index=False)
    return chosen, ranked, note


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

    strict = (
        (results["trades"] >= MIN_STAGE2_TRADES)
        & (results["total_return"] > 0)
        & (results["median_return"] > 0)
        & (results["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN)
        & (results["train_return"] > 0)
        & (results["validation_trades"] >= MIN_VALIDATION_TRADES)
        & (results["validation_return"] > 0)
        & (results["validation_median"] > 0)
        & (results["validation_expectancy"] > 0)
    )

    relaxed = (
        ~strict
        & (results["trades"] >= RELAXED_MIN_STAGE2_TRADES)
        & (results["total_return"] > 0)
        & (results["average_return"] > 0)
        & (results["max_drawdown"] >= RELAXED_MAX_ALLOWED_DRAWDOWN)
        & (results["train_return"] > 0)
        & (results["validation_trades"] >= RELAXED_MIN_VALIDATION_TRADES)
        & (results["validation_expectancy"] > 0)
        & (
            (results["validation_return"] > 0)
            | (results["validation_median"] > 0)
        )
    )

    results["eligibility_tier"] = np.select(
        [strict, relaxed],
        ["strict", "relaxed"],
        default="rejected",
    )
    results["eligible"] = results["eligibility_tier"] != "rejected"
    results["score"] = np.nan

    pool = results[results["eligible"]].copy()
    if not pool.empty:
        base_score = (
            0.16 * pool["total_return"].rank(pct=True)
            + 0.24 * pool["validation_return"].rank(pct=True)
            + 0.20 * pool["validation_expectancy"].rank(pct=True)
            + 0.10 * pool["validation_median"].rank(pct=True)
            + 0.08 * pool["win_rate"].rank(pct=True)
            + 0.12 * pool["max_drawdown"].rank(pct=True)
            + 0.05 * pool["trades"].clip(upper=60).rank(pct=True)
            + 0.05 * pool["turnover24h"].rank(pct=True)
        )
        penalties = np.where(
            pool["eligibility_tier"].eq("relaxed"),
            RELAXED_SCORE_PENALTY,
            1.0,
        )
        results.loc[pool.index, "score"] = base_score * penalties

    return results.sort_values(
        [
            "eligible",
            "eligibility_tier",
            "score",
            "validation_expectancy",
            "validation_return",
        ],
        ascending=[False, True, False, False, False],
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

    # Download Stage-2 data once, then reuse it for both the bounded entry
    # optimiser and the per-symbol exit laboratory.
    stage2_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for idx, symbol in enumerate(stage2_symbols, start=1):
        try:
            log(f"Stage2 data [{idx}/{len(stage2_symbols)}] {symbol}")
            stage2_cache[symbol] = (
                fetch_klines(symbol, "15", STAGE2_DAYS_15M),
                fetch_klines(symbol, "60", STAGE2_DAYS_1H),
            )
        except Exception as exc:
            log("Stage2 data failed", symbol, repr(exc))

    if not stage2_cache:
        raise RuntimeError("Stage2 data download produced no usable symbols.")

    chosen_entry_params, entry_parameter_results, entry_parameter_note = (
        choose_daily_entry_parameters(stage2_cache)
    )
    log(
        "Chosen daily entry parameters:",
        parameter_key(chosen_entry_params),
        entry_parameter_note,
    )

    stage2_rows = []
    all_exit_rows = []
    previous_exit_config = load_json(EXIT_CONFIG_JSON, {"symbols": {}})
    exit_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_days": VALIDATION_DAYS,
        "default_method": DEFAULT_EXIT_METHOD,
        "entry_parameters": chosen_entry_params,
        "entry_parameter_selection_note": entry_parameter_note,
        "symbols": {},
    }

    for idx, symbol in enumerate(stage2_symbols, start=1):
        if symbol not in stage2_cache:
            continue
        try:
            log(f"Stage2 + ExitLab [{idx}/{len(stage2_symbols)}] {symbol}")
            raw_df15, raw_df1h = stage2_cache[symbol]
            df = model_frame(
                raw_df15,
                raw_df1h,
                chosen_entry_params,
            )
            chosen, ranked_methods = choose_best_exit(symbol, df)

            # Daily rescans may produce tiny ranking changes. Keep the previous
            # method unless the new method is meaningfully better and the old
            # method still has a valid comparison row.
            previous_method = (
                previous_exit_config.get("symbols", {})
                .get(symbol, {})
                .get("method")
            )
            if previous_method and previous_method != chosen["exit_method"]:
                old_rows = ranked_methods[
                    ranked_methods["exit_method"] == previous_method
                ]
                if not old_rows.empty:
                    old = old_rows.iloc[0]
                    new_score = chosen.get("exit_score")
                    old_score = old.get("exit_score")
                    new_exp = chosen.get("validation_expectancy")
                    old_exp = old.get("validation_expectancy")

                    score_gain = (
                        float(new_score) - float(old_score)
                        if pd.notna(new_score) and pd.notna(old_score)
                        else -np.inf
                    )
                    expectancy_gain = (
                        float(new_exp) - float(old_exp)
                        if pd.notna(new_exp) and pd.notna(old_exp)
                        else -np.inf
                    )

                    if (
                        bool(old.get("exit_eligible", False))
                        and (
                            score_gain < EXIT_SWITCH_MIN_SCORE_GAIN
                            or expectancy_gain < EXIT_SWITCH_MIN_EXPECTANCY_GAIN
                        )
                    ):
                        chosen = old.to_dict()
                        chosen["selection_note"] = (
                            "kept_previous_method_daily_stability_filter"
                        )

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
                "validation_expectancy": (
                    np.nan
                    if pd.isna(chosen.get("validation_expectancy"))
                    else float(chosen["validation_expectancy"])
                ),
                "validation_median": float(chosen["validation_median"]),
                "validation_win_rate": float(chosen["validation_win_rate"]),
                "validation_profit_factor": (
                    np.nan if pd.isna(chosen.get("validation_profit_factor"))
                    else float(chosen["validation_profit_factor"])
                ),
                "validation_sell_early_median_6h": (
                    np.nan if pd.isna(chosen.get("validation_sell_early_median_6h"))
                    else float(chosen["validation_sell_early_median_6h"])
                ),
            })

            exit_config["symbols"][symbol] = {
                "method": chosen["exit_method"],
                "exit_score": None if pd.isna(chosen.get("exit_score")) else float(chosen["exit_score"]),
                "validation_expectancy": None if pd.isna(chosen["validation_expectancy"]) else float(chosen["validation_expectancy"]),
                "validation_return": float(chosen["validation_return"]),
                "validation_trades": int(chosen["validation_trades"]),
                "validation_profit_factor": (
                    None if pd.isna(chosen.get("validation_profit_factor"))
                    else float(chosen["validation_profit_factor"])
                ),
                "validation_sell_early_median_6h": (
                    None if pd.isna(chosen.get("validation_sell_early_median_6h"))
                    else float(chosen["validation_sell_early_median_6h"])
                ),
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

    learned_weights = update_online_signal_weights()
    online_state = load_json(ONLINE_LEARNING_JSON, {})

    lines = [
        "✅ Tide Universe V7 Adaptive daily learning complete",
        "",
        f"Stage 1 coins: {len(stage1)}",
        f"Stage 2 coins: {len(stage2)}",
        f"Strict eligible: {(stage2['eligibility_tier'] == 'strict').sum()}",
        f"Relaxed eligible: {(stage2['eligibility_tier'] == 'relaxed').sum()}",
        f"Selected for live regime check: {len(selected)}",
        "",
        f"Entry parameters: {parameter_key(chosen_entry_params)}",
        f"Entry parameter note: {entry_parameter_note}",
        f"Online-learning live samples: {online_state.get('sample_count', 0)}",
        (
            "Signal weights: "
            + ", ".join(
                f"{key}={value:.1%}"
                for key, value in learned_weights.items()
            )
        ),
        "",
    ]
    for row in stage2[stage2["symbol"].isin(selected)].itertuples(index=False):
        lines.append(
            f"{row.symbol}: {row.exit_method} | "
            f"validation {row.validation_return:.1%} | "
            f"score {row.score:.3f}"
        )
    send_tg("\n".join(lines))
    send_tg_document(STAGE2_CSV, "V7 symbol ranking")
    send_tg_document(EXIT_RESULTS_CSV, "V7 all exit-method results")
    send_tg_document(
        ENTRY_PARAM_RESULTS_CSV,
        "V7 bounded entry-parameter comparison",
    )
    return selected, stage2


def latest_required_daily_scan_time() -> datetime:
    """Return the most recent scheduled 08:00 Sydney scan time."""
    now_local = datetime.now(LOCAL_TIMEZONE)
    scheduled_today = now_local.replace(
        hour=DAILY_RESCAN_LOCAL_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    if now_local < scheduled_today:
        return scheduled_today - timedelta(days=1)
    return scheduled_today


def daily_full_rescan_due() -> bool:
    if not STAGE2_CSV.exists() or not EXIT_CONFIG_JSON.exists():
        return True
    file_time = datetime.fromtimestamp(
        STAGE2_CSV.stat().st_mtime,
        tz=timezone.utc,
    ).astimezone(LOCAL_TIMEZONE)
    return file_time < latest_required_daily_scan_time()


def seconds_until_next_daily_scan() -> float:
    now_local = datetime.now(LOCAL_TIMEZONE)
    next_scan = now_local.replace(
        hour=DAILY_RESCAN_LOCAL_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    if next_scan <= now_local:
        next_scan += timedelta(days=1)
    return max(0.0, (next_scan - now_local).total_seconds())


def load_or_run_stage2() -> pd.DataFrame:
    if not daily_full_rescan_due():
        try:
            cached = pd.read_csv(STAGE2_CSV)
            if not cached.empty and "eligible" in cached.columns:
                cached["eligible"] = cached["eligible"].astype(str).str.lower().eq("true")
                log("Using today's V6 daily learning results")
                return cached
        except Exception as exc:
            log("Cache load failed", repr(exc))

    log(
        "Daily full rescan is due:",
        f"{DAILY_RESCAN_LOCAL_HOUR:02d}:00 Australia/Sydney",
    )
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



def market_environment(results: pd.DataFrame) -> dict:
    """
    Classify whether the market is suitable for a long liquidity-sweep rebound.
    This is deliberately conservative: an orderly weak market can be useful,
    while a fast market-wide liquidation is treated as hostile.
    """
    btc = fetch_klines("BTCUSDT", "60", 21).copy()
    now = pd.Timestamp.now(tz="UTC")
    btc["close_time"] = btc["open_time"] + pd.Timedelta(hours=1)
    btc = btc[btc["close_time"] <= now].copy()

    if len(btc) < 200:
        result = {
            "mode": "caution",
            "reason": "insufficient_btc_history",
            "monitor_cap": CAUTION_MONITOR_CAP,
            "minimum_signal_score": CAUTION_MIN_SIGNAL_SCORE,
        }
        save_json(MARKET_REGIME_JSON, result)
        return result

    btc["ma50"] = btc["close"].rolling(50).mean()
    btc["ma200"] = btc["close"].rolling(200).mean()
    previous_close = btc["close"].shift(1)
    btc["tr"] = pd.concat([
        btc["high"] - btc["low"],
        (btc["high"] - previous_close).abs(),
        (btc["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    btc["atr14_pct"] = btc["tr"].rolling(14).mean() / btc["close"]

    latest = btc.iloc[-1]
    close = float(latest["close"])
    ma50 = float(latest["ma50"])
    ma200 = float(latest["ma200"])
    return_24h = close / float(btc.iloc[-25]["close"]) - 1
    distance_ma200 = close / ma200 - 1
    current_atr_pct = float(latest["atr14_pct"])
    atr_history = btc["atr14_pct"].dropna().tail(24 * 14)
    atr_percentile = (
        float((atr_history <= current_atr_pct).mean())
        if not atr_history.empty else 0.5
    )

    # Breadth uses the best-ranked Stage2 names, not the whole exchange.
    breadth_symbols = (
        results.sort_values(
            ["eligible", "score"],
            ascending=[False, False],
            na_position="last",
        )["symbol"]
        .head(MARKET_REGIME_SAMPLE_SIZE)
        .tolist()
    )
    regimes = []
    for symbol in breadth_symbols:
        try:
            regimes.append(current_1h_regime(symbol))
        except Exception as exc:
            log("Breadth regime failed", symbol, repr(exc))

    downtrend_share = (
        regimes.count("downtrend") / len(regimes)
        if regimes else 0.5
    )

    crash_condition = (
        return_24h <= -0.07
        or (
            return_24h <= -0.045
            and atr_percentile >= 0.85
        )
        or (
            distance_ma200 <= -0.10
            and atr_percentile >= 0.90
        )
    )

    supportive_condition = (
        not crash_condition
        and return_24h > -0.04
        and atr_percentile < 0.88
        and 0.20 <= downtrend_share <= 0.85
    )

    if crash_condition:
        mode = "hostile"
        monitor_cap = HOSTILE_MONITOR_CAP
        minimum_signal_score = HOSTILE_MIN_SIGNAL_SCORE
        reason = "fast_btc_decline_or_extreme_volatility"
    elif supportive_condition:
        mode = "supportive"
        monitor_cap = SUPPORTIVE_MONITOR_CAP
        minimum_signal_score = SUPPORTIVE_MIN_SIGNAL_SCORE
        reason = "orderly_weakness_with_usable_breadth"
    else:
        mode = "caution"
        monitor_cap = CAUTION_MONITOR_CAP
        minimum_signal_score = CAUTION_MIN_SIGNAL_SCORE
        reason = "mixed_or_uncertain_market"

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "reason": reason,
        "monitor_cap": monitor_cap,
        "minimum_signal_score": minimum_signal_score,
        "btc_return_24h": return_24h,
        "btc_distance_ma200": distance_ma200,
        "btc_atr_percentile": atr_percentile,
        "stage2_downtrend_share": downtrend_share,
        "sampled_symbols": len(regimes),
    }
    save_json(MARKET_REGIME_JSON, result)
    return result


def load_market_environment() -> dict:
    return load_json(MARKET_REGIME_JSON, {
        "mode": "caution",
        "reason": "market_regime_not_ready",
        "monitor_cap": CAUTION_MONITOR_CAP,
        "minimum_signal_score": CAUTION_MIN_SIGNAL_SCORE,
    })

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

    environment = market_environment(results)
    monitor_cap = int(environment["monitor_cap"])

    pool = results[results["eligible"]].sort_values(
        [
            "eligibility_tier",
            "score",
            "validation_expectancy",
            "validation_return",
        ],
        ascending=[True, False, False, False],
    )

    selected = []
    status_lines = []
    for row in pool.itertuples(index=False):
        if len(selected) >= monitor_cap:
            break
        try:
            regime = current_1h_regime(row.symbol)
        except Exception as exc:
            log("Regime failed", row.symbol, repr(exc))
            regime = "unknown"

        status_lines.append(
            f"{row.symbol}: {regime}/{row.eligibility_tier}"
        )
        if regime != "downtrend":
            continue

        selected.append(row.symbol)
        live_metadata[row.symbol] = {
            "historical_score": None if pd.isna(row.score) else float(row.score),
            "validation_return": float(row.validation_return),
            "validation_expectancy": (
                None
                if pd.isna(row.validation_expectancy)
                else float(row.validation_expectancy)
            ),
            "validation_trades": int(row.validation_trades),
            "eligibility_tier": row.eligibility_tier,
            "exit_method": row.exit_method,
            "live_regime": regime,
            "market_mode": environment["mode"],
            "minimum_signal_score": environment["minimum_signal_score"],
        }

    send_tg(
        "🔄 Tide V6 watchlist refreshed\n\n"
        f"Market mode: {environment['mode'].upper()}\n"
        f"Reason: {environment['reason']}\n"
        f"BTC 24h: {environment.get('btc_return_24h', 0):.2%}\n"
        f"BTC ATR percentile: {environment.get('btc_atr_percentile', 0):.0%}\n"
        f"Stage2 downtrend breadth: "
        f"{environment.get('stage2_downtrend_share', 0):.0%}\n"
        f"Required signal score: "
        f"{environment['minimum_signal_score']:.0f}\n"
        f"Monitor cap: {monitor_cap}\n\n"
        f"Current downtrend symbols: {len(selected)}\n"
        + (
            "\n".join(
                f"{symbol} → {symbol_exit_method(symbol)} "
                f"[{live_metadata[symbol]['eligibility_tier']}]"
                for symbol in selected
            )
            if selected else
            "No eligible coin is currently in a 1h downtrend."
        )
    )
    log("Regimes:", " | ".join(status_lines))
    return selected


def signal_components(
    latest: pd.Series,
    historical_score: float | None,
) -> dict:
    params = ACTIVE_ENTRY_PARAMS
    hist = (
        0.50
        if historical_score is None or np.isnan(historical_score)
        else historical_score
    )
    wick_threshold = float(params["lower_wick_threshold"])
    volume_threshold = float(params["volume_multiplier"])
    wick_norm = float(np.clip(
        (float(latest["lower_wick_ratio"]) - wick_threshold)
        / max(0.95 - wick_threshold, 1e-9),
        0, 1,
    ))
    volume_norm = float(np.clip(
        (float(latest["volume_multiple"]) - volume_threshold)
        / max(4.0 - volume_threshold, 1e-9),
        0, 1,
    ))
    candle_range = max(float(latest["high"] - latest["low"]), 1e-12)
    close_position = float(np.clip(
        float(latest["close"] - latest["low"]) / candle_range,
        0, 1,
    ))
    return {
        "historical": float(np.clip(hist, 0, 1)),
        "wick": wick_norm,
        "volume": volume_norm,
        "close_position": close_position,
    }


def signal_score(latest: pd.Series, historical_score: float | None) -> float:
    components = signal_components(latest, historical_score)
    return round(float(
        100
        * sum(
            ACTIVE_SIGNAL_WEIGHTS[key] * components[key]
            for key in DEFAULT_SIGNAL_WEIGHTS
        )
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
# Realtime positions, logic stops, and smart exits
# ============================================================

def already_alerted(symbol: str, alert_type: str, signal_key: str) -> bool:
    with state_lock:
        symbol_state = alert_state.setdefault(symbol, {})
        if symbol_state.get(alert_type) == signal_key:
            return True
        symbol_state[alert_type] = signal_key
        save_json(STATE_FILE, alert_state)
        return False


def entry_risk_levels(latest: pd.Series) -> dict:
    """Calculate the same stop levels used by the live position tracker."""
    entry_price = float(latest["close"])
    atr = float(latest["atr14"])
    signal_low = float(latest["low"])

    structure_stop = signal_low - STRUCTURE_BUFFER_ATR * atr
    catastrophe_stop = entry_price - CATASTROPHE_STOP_ATR * atr
    hard_stop = max(structure_stop, catastrophe_stop)

    risk_distance = max(0.0, entry_price - hard_stop)
    risk_pct = risk_distance / entry_price if entry_price > 0 else 0.0

    return {
        "entry_price": entry_price,
        "atr": atr,
        "signal_low": signal_low,
        "structure_stop": structure_stop,
        "catastrophe_stop": catastrophe_stop,
        "hard_stop": hard_stop,
        "risk_distance": risk_distance,
        "risk_pct": risk_pct,
    }


def open_research_position(
    symbol: str,
    latest: pd.Series,
    score: float,
    components: dict,
    meta: dict,
) -> None:
    with position_lock:
        existing = active_positions.get(symbol)
        if existing and existing.get("status") == "active":
            return

        method = symbol_exit_method(symbol)
        spec = strategy_spec(method)
        risk = entry_risk_levels(latest)
        entry_price = risk["entry_price"]
        atr = risk["atr"]
        signal_low = risk["signal_low"]
        hard_stop = risk["hard_stop"]
        target = float(latest["rolling_high"])

        active_positions[symbol] = {
            "status": "active",
            "method": method,
            "stop_mode": spec.stop_mode,
            "exit_mode": spec.exit_mode,
            "entry_time": latest["open_time"].isoformat(),
            "entry_price": entry_price,
            "entry_atr": atr,
            "signal_low": signal_low,
            "hard_stop": hard_stop,
            "rolling_high_target": target,
            "highest_price": entry_price,
            "bars_held": 0,
            "target_hit": False,
            "current_stop": None,
            "closes_below_structure": 0,
            "logic_warning_sent": False,
            "entry_signal_score": score,
            "historical_component": components["historical"],
            "wick_component": components["wick"],
            "volume_component": components["volume"],
            "close_position_component": components["close_position"],
            "market_mode": meta.get("market_mode", "caution"),
            "eligibility_tier": meta.get("eligibility_tier", "unknown"),
            "entry_parameter_key": parameter_key(ACTIVE_ENTRY_PARAMS),
            "signal_weights": ACTIVE_SIGNAL_WEIGHTS.copy(),
        }
        save_json(ACTIVE_POSITIONS_FILE, active_positions)


def close_research_position(
    symbol: str,
    position: dict,
    latest: pd.Series,
    exit_price: float,
    reason: str,
) -> None:
    entry = float(position["entry_price"])
    bars = int(position["bars_held"])
    net_return = exit_price / entry - 1 - FEE_SLIPPAGE

    position["status"] = "closed"
    position["exit_time"] = latest["open_time"].isoformat()
    position["exit_price"] = exit_price
    position["exit_reason"] = reason
    position["net_return"] = net_return
    active_positions[symbol] = position
    save_json(ACTIVE_POSITIONS_FILE, active_positions)

    append_live_trade({
        "symbol": symbol,
        "entry_time": position.get("entry_time"),
        "exit_time": position.get("exit_time"),
        "entry_price": entry,
        "exit_price": exit_price,
        "net_return": net_return,
        "exit_reason": reason,
        "bars_held": bars,
        "method": position.get("method"),
        "entry_signal_score": position.get("entry_signal_score"),
        "historical_component": position.get("historical_component"),
        "wick_component": position.get("wick_component"),
        "volume_component": position.get("volume_component"),
        "close_position_component": position.get(
            "close_position_component"
        ),
        "market_mode": position.get("market_mode"),
        "eligibility_tier": position.get("eligibility_tier"),
        "entry_parameter_key": position.get("entry_parameter_key"),
    })

    send_tg(f"""🏁 Tide V7 EXIT ALERT: {symbol}

Daily learned strategy: {position['method']}
Exit reason: {reason}
Entry price: {entry:.8g}
Research exit price: {exit_price:.8g}
Estimated net return: {net_return:.2%}
Signal low: {float(position['signal_low']):.8g}
Hard stop: {float(position['hard_stop']):.8g}
Highest tracked price: {float(position['highest_price']):.8g}
Bars held: {bars}
Hours held: {bars * 0.25:.2f}

This is a research alert, not an exchange order.""")


def realtime_exit_update(symbol: str, latest: pd.Series) -> None:
    with position_lock:
        position = active_positions.get(symbol)
        if not position or position.get("status") != "active":
            return

        method = position["method"]
        spec = strategy_spec(method)
        entry = float(position["entry_price"])
        atr = float(position["entry_atr"])
        signal_low = float(position["signal_low"])
        hard_stop = float(position["hard_stop"])
        target = float(position["rolling_high_target"])

        high = float(latest["high"])
        low = float(latest["low"])
        close = float(latest["close"])
        open_price = float(latest["open"])
        bars = int(position.get("bars_held", 0)) + 1
        previous_highest = float(position.get("highest_price", entry))
        highest = max(previous_highest, high)
        current_stop = position.get("current_stop")
        closes_below = int(position.get("closes_below_structure", 0))

        # Existing stops are checked before the current candle raises them.
        if spec.stop_mode != "none" and low <= hard_stop:
            position["bars_held"] = bars
            position["highest_price"] = highest
            close_research_position(
                symbol, position, latest, hard_stop,
                "hard_structure_or_catastrophe_stop",
            )
            return

        if current_stop is not None and low <= float(current_stop):
            position["bars_held"] = bars
            position["highest_price"] = highest
            close_research_position(
                symbol, position, latest, float(current_stop),
                "profit_protection_stop",
            )
            return

        closes_below = closes_below + 1 if close < signal_low else 0
        bearish = close < open_price
        volume_reversal = (
            bearish
            and float(latest.get("volume_multiple", 0.0))
            >= REVERSAL_VOLUME_MULTIPLE
            and float(latest.get("body_ratio", 0.0))
            >= REVERSAL_BODY_RATIO
        )

        # One confirmed close below the reclaimed level produces a warning.
        if (
            spec.stop_mode != "none"
            and closes_below >= LOGIC_WARNING_CLOSES_BELOW_STRUCTURE
            and not bool(position.get("logic_warning_sent", False))
        ):
            send_tg(f"""⚠️ Tide V7 LOGIC WARNING: {symbol}

Strategy: {method}
Current close: {close:.8g}
Signal low: {signal_low:.8g}
Hard stop: {hard_stop:.8g}
Closes below signal low: {closes_below}

The liquidity-reclaim setup is weakening. A second close below
the signal low, or a high-volume bearish breakdown, confirms exit.""")
            position["logic_warning_sent"] = True

        if spec.stop_mode != "none":
            if closes_below >= LOGIC_EXIT_CLOSES_BELOW_STRUCTURE:
                position["bars_held"] = bars
                position["highest_price"] = highest
                position["closes_below_structure"] = closes_below
                close_research_position(
                    symbol, position, latest, close,
                    "logic_failure_two_closes_below_signal_low",
                )
                return
            if close < signal_low and volume_reversal:
                position["bars_held"] = bars
                position["highest_price"] = highest
                position["closes_below_structure"] = closes_below
                close_research_position(
                    symbol, position, latest, close,
                    "logic_failure_volume_breakdown",
                )
                return

        favourable = highest - entry
        mfe_atr = favourable / atr if atr > 0 else 0.0

        if (
            spec.stop_mode == "structure_time"
            and bars >= TIME_STOP_BARS
            and mfe_atr < TIME_STOP_MIN_MFE_ATR
        ):
            position["bars_held"] = bars
            position["highest_price"] = highest
            close_research_position(
                symbol, position, latest, close,
                "time_invalidation_low_mfe",
            )
            return

        exit_price = None
        reason = None

        if spec.exit_mode == "fixed6":
            if bars >= 24:
                exit_price, reason = close, "fixed_6h"

        elif spec.exit_mode == "delayed_mfe":
            if mfe_atr >= spec.mfe_activation_atr:
                candidate = highest - spec.mfe_giveback * favourable
                current_stop = (
                    candidate if current_stop is None
                    else max(float(current_stop), candidate)
                )

        elif spec.exit_mode == "staged_mfe":
            if mfe_atr >= spec.mfe_activation_atr:
                candidate = highest - staged_giveback(mfe_atr) * favourable
                current_stop = (
                    candidate if current_stop is None
                    else max(float(current_stop), candidate)
                )

        elif spec.exit_mode == "smart_reversal":
            if mfe_atr >= spec.mfe_activation_atr:
                candidate = highest - staged_giveback(mfe_atr) * favourable
                current_stop = (
                    candidate if current_stop is None
                    else max(float(current_stop), candidate)
                )

                # Realtime approximation: a high-volume bearish candle must
                # also close beneath the prior three completed 15m lows.
                with data_lock:
                    live_df = market_data.get(symbol, {}).get("15", pd.DataFrame()).copy()
                if len(live_df) >= REVERSAL_SWING_LOOKBACK + 1:
                    prior = live_df.iloc[-(REVERSAL_SWING_LOOKBACK + 1):-1]
                    prior_swing_low = float(prior["low"].min())
                    if volume_reversal and close < prior_swing_low:
                        exit_price, reason = close, "smart_volume_structure_reversal"

        elif spec.exit_mode == "atr_trail":
            if favourable >= atr:
                candidate = highest - spec.atr_trail_multiple * atr
                current_stop = (
                    candidate if current_stop is None
                    else max(float(current_stop), candidate)
                )

        elif spec.exit_mode == "rolling_high":
            if np.isfinite(target) and target > entry and high >= target:
                exit_price, reason = target, "rolling_high_target"

        if exit_price is None and bars >= EXIT_MAX_BARS:
            exit_price, reason = close, "maximum_12h_timeout"

        position["bars_held"] = bars
        position["highest_price"] = highest
        position["current_stop"] = current_stop
        position["closes_below_structure"] = closes_below
        active_positions[symbol] = position

        if exit_price is not None:
            close_research_position(
                symbol, position, latest, float(exit_price), str(reason)
            )
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
    components = signal_components(latest, hist_score)
    score = signal_score(latest, hist_score)
    tier_label, tier_multiplier = risk_tier(score)
    suggested_margin = BASE_MARGIN_USDT * tier_multiplier
    exit_method = symbol_exit_method(symbol)
    exit_spec = strategy_spec(exit_method)
    risk = entry_risk_levels(latest)

    stop_status = (
        "ACTIVE"
        if exit_spec.stop_mode != "none"
        else "REFERENCE ONLY — selected strategy has no automatic stop"
    )
    stop_note = (
        "This stop is now tracked by the research position monitor."
        if alert_type == "confirmed" and exit_spec.stop_mode != "none"
        else (
            "This is provisional until the 15m candle closes."
            if alert_type == "pre"
            else "The selected baseline strategy does not automatically execute this stop."
        )
    )

    minimum_score = float(
        meta.get("minimum_signal_score", CAUTION_MIN_SIGNAL_SCORE)
    )
    if score < minimum_score:
        log(
            "Signal rejected by market layer",
            symbol,
            f"score={score}",
            f"required={minimum_score}",
            f"mode={meta.get('market_mode', 'caution')}",
        )
        return

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
Signal low: {risk['signal_low']:.8g}
ATR14: {risk['atr']:.8g}

STRUCTURE STOP: {risk['structure_stop']:.8g}
CATASTROPHE STOP: {risk['catastrophe_stop']:.8g}
RESEARCH HARD STOP: {risk['hard_stop']:.8g}
Price risk to hard stop: {risk['risk_distance']:.8g} ({risk['risk_pct']:.2%})
Stop status: {stop_status}
{stop_note}

Lower wick ratio: {latest['lower_wick_ratio']:.4f}
Volume multiple: {latest['volume_multiple']:.2f}

Signal score: {score:.1f}/100
Market mode: {meta.get('market_mode', 'caution').upper()}
Required score: {minimum_score:.0f}
Eligibility tier: {meta.get('eligibility_tier', 'unknown')}
Research position tier: {tier_label}
Example margin: {suggested_margin:.0f} USDT

Daily learned stop/exit strategy: {exit_method}
{note}
Do not chase delayed alerts.""")

    if alert_type == "confirmed":
        open_research_position(
            symbol,
            latest,
            score,
            components,
            meta,
        )


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
            "⚠️ V7 found no eligible coin currently in a 1h downtrend.\n"
            f"It will refresh again in {WATCHLIST_REFRESH_HOURS} hours."
        )
        started = time.monotonic()
        next_daily_scan = time.monotonic() + seconds_until_next_daily_scan()
        while True:
            if time.monotonic() >= next_daily_scan:
                log("Sydney morning full rescan due; restarting.")
                os.execv(sys.executable, [sys.executable, *sys.argv])
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
        "✅ Tide Universe V7 Adaptive realtime monitor started\n\n"
        + "\n".join(
            f"{symbol} → {symbol_exit_method(symbol)}"
            for symbol in symbols
        )
        + f"\n\nWatchlist refresh: every {WATCHLIST_REFRESH_HOURS} hours"
        + f"\nFull learning refresh: daily at {DAILY_RESCAN_LOCAL_HOUR:02d}:00 Sydney time"
    )

    started = time.monotonic()
    next_daily_scan = time.monotonic() + seconds_until_next_daily_scan()
    while True:
        if time.monotonic() >= next_daily_scan:
            log("Sydney morning full rescan due; restarting.")
            os.execv(sys.executable, [sys.executable, *sys.argv])
        if (time.monotonic() - started) / 3600 >= WATCHLIST_REFRESH_HOURS:
            log("Watchlist refresh due; restarting.")
            os.execv(sys.executable, [sys.executable, *sys.argv])
        log("Heartbeat monitoring", len(symbols), "symbols")
        time.sleep(60)


def main() -> None:
    update_online_signal_weights()
    stage2 = load_or_run_stage2()
    selected = select_current_watchlist(stage2)
    start_monitor(selected)


if __name__ == "__main__":
    main()