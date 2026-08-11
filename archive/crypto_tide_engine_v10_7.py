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
# Crypto Tide Engine V10.7 Confirmed-Only Production Runtime
# All-eligible Bybit universe + daily per-symbol stop/exit optimisation
# + mandatory adaptive stops + next-candle confirmation + portfolio admission control
# ============================================================

# ---------- Universe ----------
MIN_LISTING_DAYS = 120
MIN_TURNOVER_24H = 0
MAX_STAGE1_CANDIDATES = 9999
STAGE2_CANDIDATES = 9999
TOP_N_TO_MONITOR = 9999
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

# ---------- V10.6 scored confirmation ----------
# Confirmation is no longer a binary 2-of-3 gate. The raw sweep/reclaim candle
# contributes most of the setup quality; the next completed 15m candle adds or
# subtracts confidence. A very strong raw signal can therefore remain tradable
# even when the next candle is not a textbook confirmation.
SECONDARY_CONFIRMATION_ENABLED = True
CONFIRM_INVALIDATION_BUFFER_ATR = 0.20

RAW_QUALITY_WEIGHT = float(os.getenv("RAW_QUALITY_WEIGHT", "0.75"))
CONFIRMATION_QUALITY_WEIGHT = float(
    os.getenv("CONFIRMATION_QUALITY_WEIGHT", "0.25")
)
MIN_COMBINED_SETUP_SCORE = float(
    os.getenv("MIN_COMBINED_SETUP_SCORE", "62")
)
MIN_RAW_QUALITY_SCORE = float(
    os.getenv("MIN_RAW_QUALITY_SCORE", "58")
)

# ---------- V10.7 confirmed-only production filters ----------
# Only fully qualified entries are sent to Telegram. Rejected setups remain
# available in the local audit/outcome files but generate no alert.
PRODUCTION_MIN_NEXT_QUALITY = float(
    os.getenv("PRODUCTION_MIN_NEXT_QUALITY", "95")
)
PRODUCTION_MIN_COMBINED_QUALITY = float(
    os.getenv("PRODUCTION_MIN_COMBINED_QUALITY", "70")
)
PRODUCTION_MIN_SIGNAL_SCORE = float(
    os.getenv("PRODUCTION_MIN_SIGNAL_SCORE", "70")
)
PRODUCTION_MIN_CONFIRMATION_TESTS = int(
    os.getenv("PRODUCTION_MIN_CONFIRMATION_TESTS", "2")
)
PRODUCTION_MAX_HARD_STOP_RISK_PCT = float(
    os.getenv("PRODUCTION_MAX_HARD_STOP_RISK_PCT", "0.02")
)
PRODUCTION_STRUCTURE_BUFFER_ATR = float(
    os.getenv("PRODUCTION_STRUCTURE_BUFFER_ATR", "0.50")
)
PRODUCTION_FIXED_HOLD_BARS = int(
    os.getenv("PRODUCTION_FIXED_HOLD_BARS", "14")
)
SEND_RESEARCH_ALERTS = False
SEND_PRE_ALERTS = False
PRODUCTION_EXIT_METHOD = os.getenv(
    "PRODUCTION_EXIT_METHOD", "structure_fixed3_5h"
)
ALLOW_DAILY_EXIT_SWITCH = False

# ---------- V9 adaptive ATR/structure stop ----------
# Every production exit method now has an initial stop. The ATR multipliers
# adapt modestly to volatility, while the reclaimed signal low remains the
# main structural reference.
ADAPTIVE_STOP_MIN_STRUCTURE_BUFFER_ATR = 0.15
ADAPTIVE_STOP_MAX_STRUCTURE_BUFFER_ATR = 0.40
ADAPTIVE_STOP_MIN_CATASTROPHE_ATR = 1.50
ADAPTIVE_STOP_MAX_CATASTROPHE_ATR = 2.25
ADAPTIVE_STOP_REFERENCE_ATR_PCT = 0.04

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
SUPPORTIVE_MONITOR_CAP = 9999
CAUTION_MONITOR_CAP = 9999
HOSTILE_MONITOR_CAP = 9999
SUPPORTIVE_MIN_SIGNAL_SCORE = 60.0
CAUTION_MIN_SIGNAL_SCORE = 70.0
HOSTILE_MIN_SIGNAL_SCORE = 82.0

# ---------- Daily risk/exit optimisation ----------
EXIT_MAX_BARS = 96                  # 24 hours on 15m candles
DEFAULT_EXIT_METHOD = "structure_fixed3_5h"

# Every Stage-2 symbol is tested against all strategies below each morning.
# The chosen strategy is frozen into each new position at entry time.
EXIT_METHOD_NAMES = [
    "structure_fixed6",
    "structure_fixed12",
    "dynamic_structure",
    "trend_health",
    "hybrid_structure_trend",
    "structure_atr_trail_2.5",
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

# ---------- V10.3 production notification/risk defaults ----------
# PRE alerts are intentionally disabled. Telegram receives only confirmed
# entries, exits/stops, startup status, and system errors.
SEND_PRE_ALERTS = os.getenv("SEND_PRE_ALERTS", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# This program generates research alerts; it does not place exchange orders.
# Leverage is therefore a recommended/displayed setting only.
DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE", "50"))
BASE_MARGIN_USDT = float(os.getenv("BASE_MARGIN_USDT", "150"))
PORTFOLIO_CAPITAL_USDT = float(os.getenv("PORTFOLIO_CAPITAL_USDT", "3000"))
PORTFOLIO_MAX_POSITIONS = int(os.getenv("PORTFOLIO_MAX_POSITIONS", "12"))
PORTFOLIO_MAX_MARGIN_UTILISATION = float(
    os.getenv("PORTFOLIO_MAX_MARGIN_UTILISATION", "0.80")
)

# V10.6 sizes each position from the model stop distance rather than using a
# fixed notional. Leverage controls margin efficiency, not planned account loss.
BASE_ACCOUNT_RISK_PCT = float(os.getenv("BASE_ACCOUNT_RISK_PCT", "0.01"))
MAX_SIGNAL_RISK_PCT = float(os.getenv("MAX_SIGNAL_RISK_PCT", "0.0125"))
MAX_TOTAL_OPEN_RISK_PCT = float(
    os.getenv("MAX_TOTAL_OPEN_RISK_PCT", "0.05")
)
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
MIN_POSITION_NOTIONAL_USDT = float(
    os.getenv("MIN_POSITION_NOTIONAL_USDT", "250")
)
MAX_POSITION_NOTIONAL_USDT = float(
    os.getenv("MAX_POSITION_NOTIONAL_USDT", "7500")
)

# Approximate liquidation-safety inputs. Actual exchange liquidation prices
# depend on maintenance tiers, fees, funding, and cross/isolated margin.
LIQUIDATION_MAINTENANCE_BUFFER_PCT = float(
    os.getenv("LIQUIDATION_MAINTENANCE_BUFFER_PCT", "0.005")
)
LIQUIDATION_EXTRA_SAFETY_PCT = float(
    os.getenv("LIQUIDATION_EXTRA_SAFETY_PCT", "0.005")
)
WEBSOCKET_SYMBOLS_PER_CONNECTION = int(
    os.getenv("WEBSOCKET_SYMBOLS_PER_CONNECTION", "35")
)

# ---------- Persistent storage / incremental cache ----------
# Mount a Railway Volume at /data. Learned state and candle caches survive
# redeploys, so changing a variable no longer triggers a six-hour full rescan.
DATA_DIR = Path(os.getenv("TIDE_DATA_DIR", "/data/tide_v9_2"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
KLINE_CACHE_DIR = DATA_DIR / "kline_cache"
KLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_SCAN_MARKER_JSON = DATA_DIR / "daily_scan_marker.json"
INCREMENTAL_REFRESH_DAYS = int(os.getenv("INCREMENTAL_REFRESH_DAYS", "3"))

# ---------- Output ----------
STAGE1_CSV = DATA_DIR / "stage1_results.csv"
STAGE2_CSV = DATA_DIR / "stage2_full_results.csv"
EXIT_RESULTS_CSV = DATA_DIR / "exit_method_results.csv"
EXIT_CONFIG_JSON = DATA_DIR / "exit_config.json"
SELECTED_JSON = DATA_DIR / "selected_symbols.json"
STATE_FILE = DATA_DIR / "alert_state.json"
ACTIVE_POSITIONS_FILE = DATA_DIR / "active_positions.json"
MARKET_REGIME_JSON = DATA_DIR / "market_regime.json"
ENTRY_PARAM_CONFIG_JSON = DATA_DIR / "entry_parameter_config.json"
ENTRY_PARAM_RESULTS_CSV = DATA_DIR / "entry_parameter_results.csv"
LIVE_TRADE_JOURNAL_CSV = DATA_DIR / "live_trade_journal.csv"
ONLINE_LEARNING_JSON = DATA_DIR / "online_learning.json"
SETUP_AUDIT_CSV = DATA_DIR / "setup_audit.csv"
PENDING_SETUP_OUTCOMES_JSON = DATA_DIR / "pending_setup_outcomes.json"

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


alert_state = load_json(STATE_FILE, {})
active_positions = load_json(ACTIVE_POSITIONS_FILE, {})
pending_setup_outcomes = load_json(PENDING_SETUP_OUTCOMES_JSON, {})


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


def kline_cache_path(symbol: str, interval: str) -> Path:
    safe_symbol = "".join(ch for ch in symbol if ch.isalnum() or ch in "-_")
    return KLINE_CACHE_DIR / f"{safe_symbol}_{interval}.csv.gz"


def parse_kline_rows(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "turnover": float(row[6]),
    } for row in rows])


def fetch_kline_window(
    symbol: str,
    interval: str,
    required: int,
    start_ms: int | None = None,
) -> pd.DataFrame:
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
        if start_ms is not None:
            kwargs["start"] = start_ms
        if end_ms is not None:
            kwargs["end"] = end_ms
        batch = http.get_kline(**kwargs)["result"].get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(row[0]) for row in batch)
        end_ms = oldest - 1
        if start_ms is not None and oldest <= start_ms:
            break
        if len(batch) < limit:
            break
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    return parse_kline_rows(rows)


def read_kline_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, compression="gzip")
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        return frame.sort_values("open_time").drop_duplicates("open_time")
    except Exception as exc:
        log("Kline cache read failed", path.name, repr(exc))
        return pd.DataFrame()


def write_kline_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Use persistent history and refresh only the recent overlap window."""
    interval_minutes = int(interval)
    required = int(days * 24 * 60 / interval_minutes)
    path = kline_cache_path(symbol, interval)
    cached = read_kline_cache(path)

    if cached.empty or len(cached) < max(200, int(required * 0.90)):
        fresh = fetch_kline_window(symbol, interval, required)
        if fresh.empty:
            raise RuntimeError(f"No kline data for {symbol} {interval}")
        combined = fresh
        mode = "full"
    else:
        overlap_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=INCREMENTAL_REFRESH_DAYS
        )
        incremental_required = max(
            100,
            int(INCREMENTAL_REFRESH_DAYS * 24 * 60 / interval_minutes) + 20,
        )
        recent = fetch_kline_window(
            symbol,
            interval,
            incremental_required,
            int(overlap_start.timestamp() * 1000),
        )
        combined = pd.concat([cached, recent], ignore_index=True)
        mode = "incremental"

    combined = (
        combined.sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .tail(required)
        .reset_index(drop=True)
    )
    write_kline_cache(path, combined)
    log("Kline cache", mode, symbol, interval, f"bars={len(combined)}")
    return combined


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
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema30"] = df["close"].ewm(span=30, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff(3)

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

    # V10.4: the following completed candle is a scored quality input rather
    # than a hard 2-of-3 gate.
    previous_raw = df["raw_signal"].shift(1).fillna(False).astype(bool)
    previous_signal_close = df["close"].shift(1)
    previous_signal_low = df["low"].shift(1)
    previous_signal_high = df["high"].shift(1)
    previous_signal_open = df["open"].shift(1)
    previous_signal_atr = df["atr14"].shift(1)
    previous_rolling_high = df["rolling_high"].shift(1)
    previous_wick_ratio = df["lower_wick_ratio"].shift(1)
    previous_volume_multiple = df["volume_multiple"].shift(1)

    previous_range = (previous_signal_high - previous_signal_low).clip(
        lower=1e-12
    )
    previous_close_position = (
        (previous_signal_close - previous_signal_low) / previous_range
    ).clip(0, 1)

    wick_quality = (
        (previous_wick_ratio - lower_wick_threshold)
        / max(0.95 - lower_wick_threshold, 1e-9)
    ).clip(0, 1)
    volume_quality = (
        (previous_volume_multiple - volume_multiplier)
        / max(4.0 - volume_multiplier, 1e-9)
    ).clip(0, 1)

    # Raw candle quality is intentionally dominant.
    raw_quality_score = 100 * (
        0.45 * wick_quality
        + 0.35 * volume_quality
        + 0.20 * previous_close_position
    )

    confirm_bullish = df["close"] > df["open"]
    confirm_above_signal_close = df["close"] > previous_signal_close
    confirm_above_ema20 = df["close"] > df["ema20"]
    confirmation_tests = (
        confirm_bullish.astype(int)
        + confirm_above_signal_close.astype(int)
        + confirm_above_ema20.astype(int)
    )

    signal_return = (
        (df["close"] / previous_signal_close) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return_bonus = (signal_return / 0.02).clip(-1, 1)

    confirmation_quality_score = (
        30.0 * confirm_bullish.astype(float)
        + 30.0 * confirm_above_signal_close.astype(float)
        + 25.0 * confirm_above_ema20.astype(float)
        + 15.0 * ((return_bonus + 1.0) / 2.0)
    ).clip(0, 100)

    combined_setup_score = (
        RAW_QUALITY_WEIGHT * raw_quality_score
        + CONFIRMATION_QUALITY_WEIGHT * confirmation_quality_score
    )

    confirmation_not_invalidated = (
        df["low"]
        > previous_signal_low
        - CONFIRM_INVALIDATION_BUFFER_ATR * previous_signal_atr
    )

    df["secondary_confirmation_tests"] = confirmation_tests
    df["raw_quality_score"] = np.where(
        previous_raw, raw_quality_score, np.nan
    )
    df["confirmation_quality_score"] = np.where(
        previous_raw, confirmation_quality_score, np.nan
    )
    df["combined_setup_score"] = np.where(
        previous_raw, combined_setup_score, np.nan
    )
    df["secondary_confirmation_pass"] = (
        previous_raw
        & confirmation_not_invalidated
        & (raw_quality_score >= MIN_RAW_QUALITY_SCORE)
        & (combined_setup_score >= MIN_COMBINED_SETUP_SCORE)
    )

    # Preserve original Tide-candle diagnostics for scoring, alerts, and stops.
    df["confirmation_signal_low"] = np.where(
        previous_raw, previous_signal_low, np.nan
    )
    df["confirmation_signal_atr"] = np.where(
        previous_raw, previous_signal_atr, np.nan
    )
    df["confirmation_signal_close"] = np.where(
        previous_raw, previous_signal_close, np.nan
    )
    df["confirmation_signal_open"] = np.where(
        previous_raw, previous_signal_open, np.nan
    )
    df["confirmation_signal_high"] = np.where(
        previous_raw, previous_signal_high, np.nan
    )
    df["confirmation_signal_wick_ratio"] = np.where(
        previous_raw, previous_wick_ratio, np.nan
    )
    df["confirmation_signal_volume_multiple"] = np.where(
        previous_raw, previous_volume_multiple, np.nan
    )
    df["confirmation_signal_close_position"] = np.where(
        previous_raw, previous_close_position, np.nan
    )
    df["confirmation_rolling_high"] = np.where(
        previous_raw, previous_rolling_high, np.nan
    )

    candidate_signal = (
        df["secondary_confirmation_pass"]
        if SECONDARY_CONFIRMATION_ENABLED
        else df["raw_signal"]
    )

    accepted = np.zeros(len(df), dtype=bool)
    last_entry = -10**9
    for idx in np.flatnonzero(candidate_signal.to_numpy()):
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
    stop_mode: str
    exit_mode: str
    atr_trail_multiple: float = 2.5


STRATEGY_SPECS = {
    "structure_fixed3_5h": StrategySpec(
        "structure_fixed3_5h", "structure", "fixed3_5h"
    ),
    "structure_fixed6": StrategySpec("structure_fixed6", "structure", "fixed6"),
    "structure_fixed12": StrategySpec("structure_fixed12", "structure", "fixed12"),
    "dynamic_structure": StrategySpec("dynamic_structure", "structure", "dynamic_structure"),
    "trend_health": StrategySpec("trend_health", "structure", "trend_health"),
    "hybrid_structure_trend": StrategySpec("hybrid_structure_trend", "structure", "hybrid_structure_trend"),
    "structure_atr_trail_2.5": StrategySpec("structure_atr_trail_2.5", "structure", "atr_trail", 2.5),
    "structure_rolling_high": StrategySpec("structure_rolling_high", "structure", "rolling_high"),
}


def strategy_spec(method: str) -> StrategySpec:
    # Old V10.5 MFE methods are deliberately upgraded so an old bundle cannot
    # reactivate the premature profit-protection exit.
    legacy = {
        "baseline_fixed6_no_stop": "structure_fixed6",
        "structure_mfe30_act075": "hybrid_structure_trend",
        "structure_mfe40_act100": "hybrid_structure_trend",
        "structure_mfe50_act125": "hybrid_structure_trend",
        "structure_time_staged_mfe": "hybrid_structure_trend",
        "structure_time_smart_reversal": "hybrid_structure_trend",
        "structure_atr_trail_1.5": "structure_atr_trail_2.5",
    }
    method = legacy.get(method, method)
    return STRATEGY_SPECS.get(method, STRATEGY_SPECS[DEFAULT_EXIT_METHOD])


def confirmed_swing_low(df: pd.DataFrame, idx: int, entry_idx: int) -> float | None:
    pivot = idx - 2
    if pivot < entry_idx + 2 or pivot + 2 >= len(df):
        return None
    low = float(df.iloc[pivot]["low"])
    left = df.iloc[pivot-2:pivot]["low"]
    right = df.iloc[pivot+1:pivot+3]["low"]
    if low < float(left.min()) and low <= float(right.min()):
        return low
    return None


def trend_health_score(df: pd.DataFrame, idx: int, structure_level: float) -> float:
    row = df.iloc[idx]
    score = 0.0
    score += 25.0 if float(row["close"]) > float(row["ema20"]) else 0.0
    score += 15.0 if float(row.get("ema20_slope", 0.0)) >= 0 else 0.0
    score += 25.0 if float(row["close"]) > structure_level else 0.0
    score += 15.0 if float(row["close"]) >= float(row["open"]) else 0.0
    score += 10.0 if float(row.get("rsi14", 50.0)) >= 45 else 0.0
    bearish_volume = float(row.get("volume_multiple", 0.0)) >= 1.5 and float(row["close"]) < float(row["open"])
    score += 10.0 if not bearish_volume else 0.0
    return score


def run_exit_method(df: pd.DataFrame, entry_idx: int, method: str) -> ExitResult | None:
    spec = strategy_spec(method)
    entry_row = df.iloc[entry_idx]
    entry = float(entry_row["close"])
    atr = float(entry_row.get("confirmation_signal_atr", entry_row["atr14"]))
    signal_low = float(entry_row.get("confirmation_signal_low", entry_row["low"]))
    target = float(entry_row.get("confirmation_rolling_high", entry_row["rolling_high"]))
    if not np.isfinite(atr) or atr <= 0: return None
    if not np.isfinite(signal_low): signal_low = float(entry_row["low"])
    last = min(entry_idx + EXIT_MAX_BARS, len(df)-1)
    sb, cm = adaptive_stop_multipliers(entry, atr)
    hard_stop = max(signal_low - sb*atr, entry - cm*atr)
    structure_stop = hard_stop
    highest = entry
    ema_below = 0
    weak_health = 0
    for idx in range(entry_idx+1, last+1):
        row=df.iloc[idx]; high=float(row["high"]); low=float(row["low"]); close=float(row["close"])
        if low <= structure_stop:
            reason = "dynamic_structure_stop" if structure_stop > hard_stop else "hard_structure_or_catastrophe_stop"
            return ExitResult(idx, structure_stop, reason)
        swing=confirmed_swing_low(df, idx, entry_idx)
        if swing is not None:
            candidate=swing - 0.20*atr
            if candidate > structure_stop and candidate < close:
                structure_stop=candidate
        highest=max(highest, high)
        ema_below = ema_below+1 if close < float(row["ema20"]) else 0
        health=trend_health_score(df, idx, structure_stop)
        weak_health = weak_health+1 if health < 40 else 0
        if spec.exit_mode == "fixed3_5h" and idx-entry_idx >= PRODUCTION_FIXED_HOLD_BARS:
            return ExitResult(idx, close, "fixed_3_5h")
        if spec.exit_mode == "fixed6" and idx-entry_idx >= 24: return ExitResult(idx, close, "fixed_6h")
        if spec.exit_mode == "fixed12" and idx-entry_idx >= 48: return ExitResult(idx, close, "fixed_12h")
        if spec.exit_mode == "trend_health" and ema_below >= 2 and weak_health >= 2: return ExitResult(idx, close, "trend_health_breakdown")
        if spec.exit_mode == "hybrid_structure_trend" and weak_health >= 2 and (ema_below >= 2 or close < structure_stop + 0.25*atr): return ExitResult(idx, close, "hybrid_trend_structure_breakdown")
        if spec.exit_mode == "atr_trail" and highest-entry >= 1.5*atr:
            structure_stop=max(structure_stop, highest-spec.atr_trail_multiple*atr)
        if spec.exit_mode == "rolling_high" and np.isfinite(target) and target > entry and high >= target:
            return ExitResult(idx, target, "rolling_high_target")
    return ExitResult(last, float(df.iloc[last]["close"]), "maximum_24h_timeout")


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

    # V9 monitors every robust-eligible symbol. Entry rules, confirmation,
    # market score threshold, and portfolio capacity still control actual alerts.
    selected = stage2.loc[stage2["eligible"], "symbol"].tolist()
    if MONITOR_FAILED_BENCHMARKS:
        for benchmark in BENCHMARKS:
            if benchmark in stage2["symbol"].values and benchmark not in selected:
                selected.append(benchmark)

    save_json(SELECTED_JSON, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "stage2_symbols": stage2_symbols,
        "selection_rule": "all robust-eligible symbols + per-symbol learned exit method",
    })

    learned_weights = update_online_signal_weights()
    online_state = load_json(ONLINE_LEARNING_JSON, {})

    lines = [
        "✅ Crypto Tide Engine V10.6 daily all-eligible learning complete",
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
    send_tg_document(STAGE2_CSV, "V9.2 all-eligible symbol ranking")
    send_tg_document(EXIT_RESULTS_CSV, "V9 all exit-method results")
    send_tg_document(
        ENTRY_PARAM_RESULTS_CSV,
        "V9.2 bounded entry-parameter comparison",
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
    marker = load_json(DAILY_SCAN_MARKER_JSON, {})
    completed_at = marker.get("completed_at_utc")
    if completed_at:
        try:
            completed = datetime.fromisoformat(completed_at).astimezone(
                LOCAL_TIMEZONE
            )
            return completed < latest_required_daily_scan_time()
        except Exception:
            pass
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
                cached["eligible"] = (
                    cached["eligible"].astype(str).str.lower().eq("true")
                )
                log(
                    "Fast restore: using today's persistent V9.2 learning results",
                    f"rows={len(cached)}",
                    f"data_dir={DATA_DIR}",
                )
                return cached
        except Exception as exc:
            log("Persistent cache load failed", repr(exc))

    log(
        "Daily incremental rescan is due:",
        f"{DAILY_RESCAN_LOCAL_HOUR:02d}:00 Australia/Sydney",
        f"refresh_days={INCREMENTAL_REFRESH_DAYS}",
    )
    _, stage2 = scan_and_select()
    save_json(DAILY_SCAN_MARKER_JSON, {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sydney_date": datetime.now(LOCAL_TIMEZONE).date().isoformat(),
        "stage2_rows": int(len(stage2)),
        "version": "9.2",
    })
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
    """Return the live production exit.

    V10.6.1 keeps the daily Exit Lab as research, but freezes live positions
    to the validated production exit unless ALLOW_DAILY_EXIT_SWITCH is enabled.
    """
    if not ALLOW_DAILY_EXIT_SWITCH:
        return PRODUCTION_EXIT_METHOD

    config = load_exit_config()
    return config.get("symbols", {}).get(symbol, {}).get(
        "method", config.get("default_method", DEFAULT_EXIT_METHOD)
    )


def select_current_watchlist(results: pd.DataFrame) -> list[str]:
    """V9: monitor every robust-eligible symbol; signal logic checks regime."""
    global live_metadata
    live_metadata = {}

    environment = market_environment(results)
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
    for row in pool.itertuples(index=False):
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
            "live_regime": "checked_at_signal_time",
            "market_mode": environment["mode"],
            "minimum_signal_score": environment["minimum_signal_score"],
        }

    send_tg(
        "🔄 Crypto Tide V10.7 all-eligible watchlist refreshed\n\n"
        f"Market mode: {environment['mode'].upper()}\n"
        f"Reason: {environment['reason']}\n"
        f"Required signal score: {environment['minimum_signal_score']:.0f}\n"
        f"Robust-eligible symbols monitored: {len(selected)}\n"
        f"Portfolio capacity: {PORTFOLIO_MAX_POSITIONS} positions, "
        f"{PORTFOLIO_MAX_MARGIN_UTILISATION:.0%} margin utilisation\n"
        f"Reference capital: {PORTFOLIO_CAPITAL_USDT:.0f} USDT"
    )
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

    source_wick = latest.get(
        "confirmation_signal_wick_ratio",
        latest["lower_wick_ratio"],
    )
    source_volume = latest.get(
        "confirmation_signal_volume_multiple",
        latest["volume_multiple"],
    )
    source_close_position = latest.get(
        "confirmation_signal_close_position",
        np.nan,
    )

    if pd.isna(source_wick):
        source_wick = latest["lower_wick_ratio"]
    if pd.isna(source_volume):
        source_volume = latest["volume_multiple"]

    if pd.isna(source_close_position):
        candle_range = max(float(latest["high"] - latest["low"]), 1e-12)
        source_close_position = float(
            np.clip(
                float(latest["close"] - latest["low"]) / candle_range,
                0,
                1,
            )
        )

    wick_norm = float(np.clip(
        (float(source_wick) - wick_threshold)
        / max(0.95 - wick_threshold, 1e-9),
        0,
        1,
    ))
    volume_norm = float(np.clip(
        (float(source_volume) - volume_threshold)
        / max(4.0 - volume_threshold, 1e-9),
        0,
        1,
    ))
    close_position = float(np.clip(source_close_position, 0, 1))

    confirmation_quality = latest.get("confirmation_quality_score", np.nan)
    confirmation_norm = (
        0.50
        if pd.isna(confirmation_quality)
        else float(np.clip(float(confirmation_quality) / 100.0, 0, 1))
    )

    return {
        "historical": float(np.clip(hist, 0, 1)),
        "wick": wick_norm,
        "volume": volume_norm,
        "close_position": close_position,
        "confirmation": confirmation_norm,
    }


def signal_score(latest: pd.Series, historical_score: float | None) -> float:
    components = signal_components(latest, historical_score)

    raw_model_score = 100 * (
        ACTIVE_SIGNAL_WEIGHTS["historical"] * components["historical"]
        + ACTIVE_SIGNAL_WEIGHTS["wick"] * components["wick"]
        + ACTIVE_SIGNAL_WEIGHTS["volume"] * components["volume"]
        + ACTIVE_SIGNAL_WEIGHTS["close_position"] * components["close_position"]
    )

    # Confirmation modifies the raw/historical model score; it is no longer an
    # all-or-nothing gate.
    confirmation_adjustment = 20.0 * (
        components["confirmation"] - 0.50
    )
    setup_score = latest.get("combined_setup_score", np.nan)
    setup_adjustment = (
        0.0
        if pd.isna(setup_score)
        else 0.10 * (float(setup_score) - 62.0)
    )

    return round(float(np.clip(
        raw_model_score + confirmation_adjustment + setup_adjustment,
        0,
        100,
    )), 1)



def account_risk_pct_for_score(score: float) -> float:
    """Map signal quality to planned account risk."""
    if score >= 90:
        return min(MAX_SIGNAL_RISK_PCT, 0.0125)
    if score >= 80:
        return min(MAX_SIGNAL_RISK_PCT, 0.0100)
    if score >= 70:
        return min(MAX_SIGNAL_RISK_PCT, 0.0075)
    return min(MAX_SIGNAL_RISK_PCT, 0.0050)


def approximate_safe_leverage(stop_risk_pct: float) -> float:
    """Conservative approximation; exchange liquidation is venue-specific."""
    required_distance = (
        max(stop_risk_pct, 0.0)
        + LIQUIDATION_MAINTENANCE_BUFFER_PCT
        + LIQUIDATION_EXTRA_SAFETY_PCT
    )
    if required_distance <= 0:
        return DEFAULT_LEVERAGE
    return max(1.0, min(DEFAULT_LEVERAGE, 1.0 / required_distance))


def dynamic_position_size(score: float, risk: dict) -> dict:
    stop_risk_pct = max(float(risk.get("risk_pct", 0.0)), 1e-6)
    account_risk_pct = account_risk_pct_for_score(score)
    planned_loss = PORTFOLIO_CAPITAL_USDT * account_risk_pct

    uncapped_notional = planned_loss / stop_risk_pct
    notional = float(np.clip(
        uncapped_notional,
        MIN_POSITION_NOTIONAL_USDT,
        MAX_POSITION_NOTIONAL_USDT,
    ))

    safe_leverage = approximate_safe_leverage(stop_risk_pct)
    recommended_leverage = max(1.0, min(DEFAULT_LEVERAGE, safe_leverage))
    margin = notional / recommended_leverage

    # If the notional cap/floor changed sizing, report the resulting real risk.
    effective_planned_loss = notional * stop_risk_pct
    effective_account_risk_pct = (
        effective_planned_loss / PORTFOLIO_CAPITAL_USDT
        if PORTFOLIO_CAPITAL_USDT > 0 else 0.0
    )

    return {
        "account_risk_pct": account_risk_pct,
        "planned_loss_usdt": planned_loss,
        "effective_planned_loss_usdt": effective_planned_loss,
        "effective_account_risk_pct": effective_account_risk_pct,
        "stop_risk_pct": stop_risk_pct,
        "uncapped_notional_usdt": uncapped_notional,
        "suggested_notional_usdt": notional,
        "suggested_leverage": recommended_leverage,
        "suggested_margin_usdt": margin,
        "leverage_capped_for_stop_safety": recommended_leverage < DEFAULT_LEVERAGE,
    }


def active_open_risk_status() -> dict:
    active = [
        position
        for position in active_positions.values()
        if position.get("status") == "active"
    ]
    open_risk = sum(
        float(position.get("effective_planned_loss_usdt", 0.0))
        for position in active
    )
    max_open_risk = PORTFOLIO_CAPITAL_USDT * MAX_TOTAL_OPEN_RISK_PCT
    return {
        "active": active,
        "open_risk_usdt": open_risk,
        "max_open_risk_usdt": max_open_risk,
    }


def realised_loss_today_usdt() -> float:
    if not LIVE_TRADE_JOURNAL_CSV.exists():
        return 0.0
    try:
        journal = pd.read_csv(LIVE_TRADE_JOURNAL_CSV)
        if journal.empty or "exit_time" not in journal.columns:
            return 0.0
        exit_times = pd.to_datetime(journal["exit_time"], utc=True, errors="coerce")
        today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
        local_dates = exit_times.dt.tz_convert(LOCAL_TIMEZONE).dt.date
        today_rows = journal[local_dates == today]
        if today_rows.empty:
            return 0.0
        losses = []
        for row in today_rows.to_dict(orient="records"):
            net_return = float(row.get("net_return", 0.0))
            notional = float(row.get("suggested_notional", 0.0))
            pnl = net_return * notional
            if pnl < 0:
                losses.append(-pnl)
        return float(sum(losses))
    except Exception as exc:
        log("Daily loss calculation failed", repr(exc))
        return 0.0


def append_setup_audit(row: dict) -> None:
    frame = pd.DataFrame([row])
    SETUP_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        SETUP_AUDIT_CSV,
        mode="a",
        header=not SETUP_AUDIT_CSV.exists(),
        index=False,
    )


def register_pending_setup_outcomes(
    symbol: str,
    latest: pd.Series,
    decision: str,
    score: float | None,
    reason: str,
) -> None:
    setup_key = (
        f"{symbol}:{latest['open_time'].isoformat()}:"
        f"{latest.get('confirmation_signal_close', latest['close'])}"
    )
    if setup_key in pending_setup_outcomes:
        return
    reference_price = float(latest.get(
        "confirmation_signal_close",
        latest["close"],
    ))
    pending_setup_outcomes[setup_key] = {
        "symbol": symbol,
        "decision_time": latest["open_time"].isoformat(),
        "reference_price": reference_price,
        "decision": decision,
        "reason": reason,
        "signal_score": score,
        "raw_quality_score": latest.get("raw_quality_score"),
        "confirmation_quality_score": latest.get(
            "confirmation_quality_score"
        ),
        "combined_setup_score": latest.get("combined_setup_score"),
        "market_required_score": live_metadata.get(
            symbol, {}
        ).get("minimum_signal_score"),
        "outcomes_recorded": [],
    }
    save_json(PENDING_SETUP_OUTCOMES_JSON, pending_setup_outcomes)


def update_pending_setup_outcomes(symbol: str, latest: pd.Series) -> None:
    changed = False
    current_time = pd.Timestamp(latest["open_time"])
    current_price = float(latest["close"])
    for key, setup in list(pending_setup_outcomes.items()):
        if setup.get("symbol") != symbol:
            continue
        start = pd.Timestamp(setup["decision_time"])
        elapsed_hours = (current_time - start).total_seconds() / 3600
        recorded = set(setup.get("outcomes_recorded", []))
        for horizon in (1, 3, 6):
            if elapsed_hours >= horizon and horizon not in recorded:
                reference = float(setup["reference_price"])
                outcome_return = (
                    current_price / reference - 1.0
                    if reference > 0 else np.nan
                )
                append_setup_audit({
                    "event_type": f"outcome_{horizon}h",
                    "symbol": symbol,
                    "setup_time": setup["decision_time"],
                    "recorded_time": latest["open_time"].isoformat(),
                    "decision": setup.get("decision"),
                    "reason": setup.get("reason"),
                    "signal_score": setup.get("signal_score"),
                    "raw_quality_score": setup.get("raw_quality_score"),
                    "confirmation_quality_score": setup.get(
                        "confirmation_quality_score"
                    ),
                    "combined_setup_score": setup.get(
                        "combined_setup_score"
                    ),
                    "market_required_score": setup.get(
                        "market_required_score"
                    ),
                    "reference_price": reference,
                    "outcome_price": current_price,
                    "outcome_return": outcome_return,
                })
                recorded.add(horizon)
                changed = True
        setup["outcomes_recorded"] = sorted(recorded)
        if all(h in recorded for h in (1, 3, 6)):
            pending_setup_outcomes.pop(key, None)
        else:
            pending_setup_outcomes[key] = setup
    if changed:
        save_json(PENDING_SETUP_OUTCOMES_JSON, pending_setup_outcomes)


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


def adaptive_stop_multipliers(entry_price: float, atr: float) -> tuple[float, float]:
    """Return modestly adaptive structure and catastrophe ATR multipliers."""
    atr_pct = atr / entry_price if entry_price > 0 else 0.0
    volatility_scale = float(np.clip(
        atr_pct / ADAPTIVE_STOP_REFERENCE_ATR_PCT,
        0.0,
        1.0,
    ))
    structure_buffer = (
        ADAPTIVE_STOP_MIN_STRUCTURE_BUFFER_ATR
        + volatility_scale
        * (
            ADAPTIVE_STOP_MAX_STRUCTURE_BUFFER_ATR
            - ADAPTIVE_STOP_MIN_STRUCTURE_BUFFER_ATR
        )
    )
    catastrophe_multiple = (
        ADAPTIVE_STOP_MIN_CATASTROPHE_ATR
        + volatility_scale
        * (
            ADAPTIVE_STOP_MAX_CATASTROPHE_ATR
            - ADAPTIVE_STOP_MIN_CATASTROPHE_ATR
        )
    )
    return float(structure_buffer), float(catastrophe_multiple)


def entry_risk_levels(latest: pd.Series) -> dict:
    """Calculate V9 adaptive stop levels used by backtest and live tracking."""
    entry_price = float(latest["close"])
    atr = float(latest.get("confirmation_signal_atr", latest["atr14"]))
    if not np.isfinite(atr) or atr <= 0:
        atr = float(latest["atr14"])
    signal_low = float(latest.get("confirmation_signal_low", latest["low"]))
    if not np.isfinite(signal_low):
        signal_low = float(latest["low"])

    structure_buffer_atr = PRODUCTION_STRUCTURE_BUFFER_ATR
    _, catastrophe_multiple_atr = adaptive_stop_multipliers(
        entry_price, atr
    )
    structure_stop = signal_low - structure_buffer_atr * atr
    catastrophe_stop = entry_price - catastrophe_multiple_atr * atr

    # V10.7 production stop: use the tested structural stop directly.
    # The catastrophe level is retained for display/research only.
    hard_stop = structure_stop

    risk_distance = max(0.0, entry_price - hard_stop)
    risk_pct = risk_distance / entry_price if entry_price > 0 else 0.0

    return {
        "entry_price": entry_price,
        "atr": atr,
        "signal_low": signal_low,
        "structure_stop": structure_stop,
        "catastrophe_stop": catastrophe_stop,
        "hard_stop": hard_stop,
        "structure_buffer_atr": structure_buffer_atr,
        "catastrophe_multiple_atr": catastrophe_multiple_atr,
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
            "lowest_price": entry_price,
            "maximum_favourable_excursion_pct": 0.0,
            "maximum_adverse_excursion_pct": 0.0,
            "bars_held": 0,
            "target_hit": False,
            "current_stop": None,
            "closes_below_structure": 0,
            "logic_warning_sent": False,
            "entry_signal_score": score,
            "account_risk_pct": dynamic_position_size(
                score, risk
            )["account_risk_pct"],
            "planned_loss_usdt": dynamic_position_size(
                score, risk
            )["planned_loss_usdt"],
            "effective_planned_loss_usdt": dynamic_position_size(
                score, risk
            )["effective_planned_loss_usdt"],
            "effective_account_risk_pct": dynamic_position_size(
                score, risk
            )["effective_account_risk_pct"],
            "suggested_margin": dynamic_position_size(
                score, risk
            )["suggested_margin_usdt"],
            "suggested_leverage": dynamic_position_size(
                score, risk
            )["suggested_leverage"],
            "suggested_notional": dynamic_position_size(
                score, risk
            )["suggested_notional_usdt"],
            "leverage_capped_for_stop_safety": dynamic_position_size(
                score, risk
            )["leverage_capped_for_stop_safety"],
            "portfolio_reference_capital": PORTFOLIO_CAPITAL_USDT,
            "secondary_confirmation_tests": int(
                latest.get("secondary_confirmation_tests", 0)
            ),
            "raw_quality_score": float(
                latest.get("raw_quality_score", np.nan)
            ),
            "confirmation_quality_score": float(
                latest.get("confirmation_quality_score", np.nan)
            ),
            "combined_setup_score": float(
                latest.get("combined_setup_score", np.nan)
            ),
            "confirmation_signal_close": float(
                latest.get("confirmation_signal_close", entry_price)
            ),
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
        "maximum_favourable_excursion_pct": position.get(
            "maximum_favourable_excursion_pct"
        ),
        "maximum_adverse_excursion_pct": position.get(
            "maximum_adverse_excursion_pct"
        ),
        "method": position.get("method"),
        "entry_signal_score": position.get("entry_signal_score"),
        "suggested_margin": position.get("suggested_margin"),
        "suggested_leverage": position.get("suggested_leverage"),
        "suggested_notional": position.get("suggested_notional"),
        "account_risk_pct": position.get("account_risk_pct"),
        "planned_loss_usdt": position.get("planned_loss_usdt"),
        "effective_planned_loss_usdt": position.get(
            "effective_planned_loss_usdt"
        ),
        "effective_account_risk_pct": position.get(
            "effective_account_risk_pct"
        ),
        "leverage_capped_for_stop_safety": position.get(
            "leverage_capped_for_stop_safety"
        ),
        "secondary_confirmation_tests": position.get(
            "secondary_confirmation_tests"
        ),
        "raw_quality_score": position.get("raw_quality_score"),
        "confirmation_quality_score": position.get(
            "confirmation_quality_score"
        ),
        "combined_setup_score": position.get("combined_setup_score"),
        "confirmation_signal_close": position.get("confirmation_signal_close"),
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

    send_tg(f"""🏁 Crypto Tide V10.7 EXIT: {symbol}

Production strategy: {position['method']}
Exit reason: {reason}
Entry price: {entry:.8g}
Research exit price: {exit_price:.8g}
Estimated net return: {net_return:.2%}
Signal low: {float(position['signal_low']):.8g}
Hard stop: {float(position['hard_stop']):.8g}
Highest tracked price: {float(position['highest_price']):.8g}
Bars held: {bars}
Hours held: {bars * 0.25:.2f}

This is a model exit alert, not an exchange order.""")


def realtime_exit_update(symbol: str, latest: pd.Series) -> None:
    with position_lock:
        position=active_positions.get(symbol)
        if not position or position.get("status") != "active": return
        spec=strategy_spec(position.get("method", DEFAULT_EXIT_METHOD))
        entry=float(position["entry_price"]); atr=float(position["entry_atr"])
        hard=float(position["hard_stop"]); current=float(position.get("current_stop") or hard)
        high=float(latest["high"]); low=float(latest["low"]); close=float(latest["close"])
        bars=int(position.get("bars_held",0))+1
        highest=max(float(position.get("highest_price",entry)), high)
        lowest=min(float(position.get("lowest_price",entry)), low)
        if low <= current:
            position.update({"bars_held":bars,"highest_price":highest,"lowest_price":lowest})
            close_research_position(symbol, position, latest, current, "dynamic_structure_stop" if current>hard else "hard_structure_or_catastrophe_stop")
            return
        with data_lock:
            live_df=market_data.get(symbol,{}).get("15",pd.DataFrame()).copy()
        if len(live_df) >= 5:
            pivot=len(live_df)-3
            plow=float(live_df.iloc[pivot]["low"])
            if plow < float(live_df.iloc[pivot-2:pivot]["low"].min()) and plow <= float(live_df.iloc[pivot+1:pivot+3]["low"].min()):
                candidate=plow-0.20*atr
                if candidate>current and candidate<close: current=candidate
        ema20=float(latest.get("ema20", close)); slope=float(latest.get("ema20_slope",0.0))
        ema_below=int(position.get("ema_below_count",0))+1 if close<ema20 else 0
        health=0.0
        health += 25 if close>ema20 else 0
        health += 15 if slope>=0 else 0
        health += 25 if close>current else 0
        health += 15 if close>=float(latest["open"]) else 0
        health += 10 if float(latest.get("rsi14",50))>=45 else 0
        health += 10 if not (float(latest.get("volume_multiple",0))>=1.5 and close<float(latest["open"])) else 0
        weak=int(position.get("weak_health_count",0))+1 if health<40 else 0
        exit_price=reason=None
        if spec.exit_mode=="fixed3_5h" and bars>=PRODUCTION_FIXED_HOLD_BARS:
            exit_price,reason=close,"fixed_3_5h"
        elif spec.exit_mode=="fixed6" and bars>=24: exit_price,reason=close,"fixed_6h"
        elif spec.exit_mode=="fixed12" and bars>=48: exit_price,reason=close,"fixed_12h"
        elif spec.exit_mode=="trend_health" and ema_below>=2 and weak>=2: exit_price,reason=close,"trend_health_breakdown"
        elif spec.exit_mode=="hybrid_structure_trend" and weak>=2 and (ema_below>=2 or close<current+0.25*atr): exit_price,reason=close,"hybrid_trend_structure_breakdown"
        elif spec.exit_mode=="atr_trail" and highest-entry>=1.5*atr: current=max(current,highest-spec.atr_trail_multiple*atr)
        elif spec.exit_mode=="rolling_high":
            target=float(position.get("rolling_high_target",np.nan))
            if np.isfinite(target) and target>entry and high>=target: exit_price,reason=target,"rolling_high_target"
        if exit_price is None and bars>=EXIT_MAX_BARS: exit_price,reason=close,"maximum_24h_timeout"
        position.update({"bars_held":bars,"highest_price":highest,"lowest_price":lowest,"maximum_favourable_excursion_pct":highest/entry-1,"maximum_adverse_excursion_pct":lowest/entry-1,"current_stop":current,"ema_below_count":ema_below,"weak_health_count":weak,"trend_health_score":health})
        active_positions[symbol]=position
        if exit_price is not None:
            close_research_position(symbol, position, latest, float(exit_price), str(reason)); return
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


def portfolio_capacity_status(sizing: dict) -> dict:
    """Check position count, margin, total open risk, and daily loss limits."""
    with position_lock:
        risk_status = active_open_risk_status()
        active = risk_status["active"]
        used_margin = sum(
            float(position.get("suggested_margin", 0.0))
            for position in active
        )
        max_margin = PORTFOLIO_CAPITAL_USDT * PORTFOLIO_MAX_MARGIN_UTILISATION
        proposed_margin = float(sizing["suggested_margin_usdt"])
        proposed_risk = float(sizing["effective_planned_loss_usdt"])
        daily_loss = realised_loss_today_usdt()
        max_daily_loss = PORTFOLIO_CAPITAL_USDT * MAX_DAILY_LOSS_PCT

        reasons = []
        if len(active) >= PORTFOLIO_MAX_POSITIONS:
            reasons.append("maximum_position_count")
        if used_margin + proposed_margin > max_margin + 1e-9:
            reasons.append("maximum_margin_utilisation")
        if (
            risk_status["open_risk_usdt"] + proposed_risk
            > risk_status["max_open_risk_usdt"] + 1e-9
        ):
            reasons.append("maximum_total_open_risk")
        if daily_loss >= max_daily_loss - 1e-9:
            reasons.append("maximum_daily_loss")

        return {
            "allowed": not reasons,
            "active_count": len(active),
            "used_margin": used_margin,
            "max_margin": max_margin,
            "open_risk_usdt": risk_status["open_risk_usdt"],
            "max_open_risk_usdt": risk_status["max_open_risk_usdt"],
            "daily_loss_usdt": daily_loss,
            "max_daily_loss_usdt": max_daily_loss,
            "reasons": reasons,
        }




def production_entry_assessment(
    latest: pd.Series,
    score: float,
    meta: dict,
) -> dict:
    """Classify a completed scored setup as PRODUCTION or RESEARCH."""
    risk = entry_risk_levels(latest)
    next_quality = float(latest.get("confirmation_quality_score", 0.0))
    combined_quality = float(latest.get("combined_setup_score", 0.0))
    market_required_score = float(
        meta.get("minimum_signal_score", CAUTION_MIN_SIGNAL_SCORE)
    )
    required_score = max(
        PRODUCTION_MIN_SIGNAL_SCORE,
        market_required_score,
    )
    confirmation_tests = int(
        latest.get("secondary_confirmation_tests", 0)
    )

    failures = []
    if next_quality < PRODUCTION_MIN_NEXT_QUALITY:
        failures.append(
            f"Next quality {next_quality:.1f} < "
            f"{PRODUCTION_MIN_NEXT_QUALITY:.1f}"
        )
    if combined_quality < PRODUCTION_MIN_COMBINED_QUALITY:
        failures.append(
            f"Combined quality {combined_quality:.1f} < "
            f"{PRODUCTION_MIN_COMBINED_QUALITY:.1f}"
        )
    if confirmation_tests < PRODUCTION_MIN_CONFIRMATION_TESTS:
        failures.append(
            f"Confirmation tests {confirmation_tests}/3 < "
            f"{PRODUCTION_MIN_CONFIRMATION_TESTS}/3"
        )
    if risk["risk_pct"] > PRODUCTION_MAX_HARD_STOP_RISK_PCT:
        failures.append(
            f"Hard-stop risk {risk['risk_pct']:.2%} > "
            f"{PRODUCTION_MAX_HARD_STOP_RISK_PCT:.2%}"
        )
    if score < required_score:
        failures.append(
            f"Signal score {score:.1f} < market requirement "
            f"{required_score:.1f}"
        )

    return {
        "production": not failures,
        "failures": failures,
        "risk": risk,
        "next_quality": next_quality,
        "combined_quality": combined_quality,
        "required_score": required_score,
        "confirmation_tests": confirmation_tests,
    }


def send_research_alert(
    symbol: str,
    latest: pd.Series,
    score: float,
    meta: dict,
    assessment: dict,
) -> None:
    """Send a marked observation alert without opening a research position."""
    if not SEND_RESEARCH_ALERTS:
        return

    signal_key = latest["open_time"].isoformat()
    if already_alerted(symbol, "research", signal_key):
        return

    risk = assessment["risk"]
    failures = assessment["failures"]
    failure_lines = "\n".join(
        f"❌ {reason}" for reason in failures
    ) or "❌ Did not pass the production entry gate."

    send_tg(f"""🟡 Tide RESEARCH SIGNAL: {symbol}

STATUS: RECORD ONLY — NOT A PRODUCTION ENTRY

Candle close UTC: {latest['open_time'] + pd.Timedelta(minutes=15)}
Reference/current price: {latest['close']:.8g}
1h regime: {latest['regime']}

Production filter result:
{failure_lines}

Filter details:
Next-candle quality: {assessment['next_quality']:.1f}/100
Required Next quality: {PRODUCTION_MIN_NEXT_QUALITY:.1f}

Combined setup quality: {assessment['combined_quality']:.1f}/100
Required Combined quality: {PRODUCTION_MIN_COMBINED_QUALITY:.1f}

Hard-stop risk: {risk['risk_pct']:.2%}
Maximum production hard-stop risk: {PRODUCTION_MAX_HARD_STOP_RISK_PCT:.2%}

Signal score: {score:.1f}/100
Market-required score: {assessment['required_score']:.1f}
Confirmation tests: {int(latest.get('secondary_confirmation_tests', 0))}/3
Raw Tide quality: {float(latest.get('raw_quality_score', 0)):.1f}/100
Volume multiple: {float(latest.get('volume_multiple', 0)):.2f}
Lower wick ratio: {float(latest.get('lower_wick_ratio', 0)):.4f}

Reference structure stop: {risk['structure_stop']:.8g}
Reference catastrophe stop: {risk['catastrophe_stop']:.8g}
Reference hard stop: {risk['hard_stop']:.8g}

Research tracking:
The setup is recorded in the audit/outcome dataset, but no live research
position is opened and it does not consume portfolio risk capacity.

Label: RESEARCH_REJECTED
Do not treat this alert as an entry instruction.""")

def send_live_alert(
    symbol: str,
    latest: pd.Series,
    alert_type: str,
    minutes_to_close: float,
) -> None:
    if alert_type == "pre" and not SEND_PRE_ALERTS:
        log(symbol, "V10.3 pre-alert suppressed")
        return

    signal_key = latest["open_time"].isoformat()
    if already_alerted(symbol, alert_type, signal_key):
        return

    meta = live_metadata.get(symbol, {})
    hist_score = meta.get("historical_score")
    components = signal_components(latest, hist_score)
    score = signal_score(latest, hist_score)
    tier_label, tier_multiplier = risk_tier(score)
    exit_method = symbol_exit_method(symbol)
    exit_spec = strategy_spec(exit_method)
    risk = entry_risk_levels(latest)
    sizing = dynamic_position_size(score, risk)
    suggested_margin = sizing["suggested_margin_usdt"]
    suggested_notional = sizing["suggested_notional_usdt"]
    suggested_leverage = sizing["suggested_leverage"]

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

    capacity = portfolio_capacity_status(sizing)
    if alert_type == "confirmed" and not capacity["allowed"]:
        log(
            "Confirmed signal rejected by V9 portfolio layer",
            symbol,
            f"reasons={','.join(capacity['reasons'])}",
            f"active={capacity['active_count']}",
            f"used_margin={capacity['used_margin']:.2f}",
            f"proposed_margin={suggested_margin:.2f}",
            f"proposed_risk={sizing['effective_planned_loss_usdt']:.2f}",
            f"max_margin={capacity['max_margin']:.2f}",
            f"open_risk={capacity['open_risk_usdt']:.2f}",
            f"max_open_risk={capacity['max_open_risk_usdt']:.2f}",
        )
        return

    title = (
        f"⚠️ Tide PRE-SIGNAL: {symbol}"
        if alert_type == "pre"
        else f"🟢 Tide PRODUCTION ENTRY: {symbol}"
    )
    note = (
        "The 15m candle is still open and the setup can disappear."
        if alert_type == "pre"
        else (
            "The raw Tide candle and next-candle quality produced an actionable "
            "combined score. Confirmation is a scoring input, not a hard gate."
        )
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
Adaptive structure buffer: {risk['structure_buffer_atr']:.2f} ATR
Adaptive catastrophe distance: {risk['catastrophe_multiple_atr']:.2f} ATR
Raw Tide quality: {float(latest.get('raw_quality_score', 0)):.1f}/100
Next-candle quality: {float(latest.get('confirmation_quality_score', 0)):.1f}/100
Combined setup quality: {float(latest.get('combined_setup_score', 0)):.1f}/100
Confirmation tests: {int(latest.get('secondary_confirmation_tests', 0))}/3
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

Planned account risk: {sizing['effective_account_risk_pct']:.2%}
Maximum planned loss: {sizing['effective_planned_loss_usdt']:.2f} USDT
Suggested notional: {suggested_notional:.0f} USDT
Suggested leverage: {suggested_leverage:.1f}x
Required margin: {suggested_margin:.2f} USDT
50x reduced for stop safety: {'YES' if sizing['leverage_capped_for_stop_safety'] else 'NO'}

Portfolio active positions: {capacity['active_count']}/{PORTFOLIO_MAX_POSITIONS}
Portfolio margin used: {capacity['used_margin']:.0f}/{capacity['max_margin']:.0f} USDT
Portfolio open risk: {capacity['open_risk_usdt']:.0f}/{capacity['max_open_risk_usdt']:.0f} USDT
Today's realised losses: {capacity['daily_loss_usdt']:.0f}/{capacity['max_daily_loss_usdt']:.0f} USDT

Production stop/exit strategy: {exit_method}\nDaily Exit Lab: {'LIVE SWITCHING' if ALLOW_DAILY_EXIT_SWITCH else 'RESEARCH ONLY'}
{note}

Important: leverage is a display recommendation only. Confirm exchange
liquidation distance, fees, funding, and stop placement before entering.
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
    raw_signal = bool(latest["raw_signal"])
    confirmed_signal = bool(latest["signal"])

    if confirmed:
        realtime_exit_update(symbol, latest)
        update_pending_setup_outcomes(symbol, latest)

    log(
        symbol,
        f"confirmed={confirmed}",
        f"regime={latest['regime']}",
        f"raw_signal={raw_signal}",
        f"score_confirmed={confirmed_signal}",
        f"raw_quality={latest.get('raw_quality_score', np.nan)}",
        f"confirmation_quality={latest.get('confirmation_quality_score', np.nan)}",
        f"combined_setup={latest.get('combined_setup_score', np.nan)}",
        f"confirmation_tests={int(latest.get('secondary_confirmation_tests', 0))}",
    )

    if confirmed:
        has_scored_setup = pd.notna(latest.get("combined_setup_score", np.nan))
        meta = live_metadata.get(symbol, {})
        hist_score = meta.get("historical_score")
        score = signal_score(latest, hist_score) if has_scored_setup else None

        if has_scored_setup:
            assessment = production_entry_assessment(
                latest, float(score), meta
            )

            if assessment["production"]:
                send_live_alert(symbol, latest, "confirmed", 0.0)
                decision = "production_entry"
                reason = "v10_6_1_production_filters_passed"
            else:
                decision = "research_rejected_silent"
                reason = " | ".join(assessment["failures"])
                log(
                    symbol,
                    "V10.7 setup silently rejected",
                    reason,
                )

            register_pending_setup_outcomes(
                symbol, latest, decision, score, reason
            )
            append_setup_audit({
                "event_type": "decision",
                "symbol": symbol,
                "setup_time": latest["open_time"].isoformat(),
                "decision": decision,
                "reason": reason,
                "signal_score": score,
                "raw_quality_score": latest.get("raw_quality_score"),
                "confirmation_quality_score": latest.get(
                    "confirmation_quality_score"
                ),
                "combined_setup_score": latest.get("combined_setup_score"),
                "confirmation_tests": latest.get(
                    "secondary_confirmation_tests"
                ),
                "hard_stop_risk_pct": assessment["risk"]["risk_pct"],
                "market_required_score": meta.get("minimum_signal_score"),
                "reference_price": latest.get(
                    "confirmation_signal_close", latest["close"]
                ),
            })
        elif raw_signal:
            log(symbol, "raw Tide candle closed; waiting for next-candle scoring")
        return

    # V10.7 sends no pre-signals or observation alerts.
    return


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


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def start_monitor(symbols: list[str]) -> None:
    if not symbols:
        send_tg(
            "⚠️ Crypto Tide V10.7 found no robust-eligible symbols.\n"
            f"It will refresh again in {WATCHLIST_REFRESH_HOURS} hours."
        )
        started = time.monotonic()
        next_daily_scan = time.monotonic() + seconds_until_next_daily_scan()
        while True:
            if time.monotonic() >= next_daily_scan:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            if (time.monotonic() - started) / 3600 >= WATCHLIST_REFRESH_HOURS:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            time.sleep(60)

    initialise_market_data(symbols)

    # A single websocket carrying hundreds of symbols is fragile. V9 divides
    # the all-eligible universe across several connections.
    websocket_connections = []
    symbol_batches = chunked(symbols, WEBSOCKET_SYMBOLS_PER_CONNECTION)
    for batch_index, batch in enumerate(symbol_batches, start=1):
        log(
            f"Starting websocket batch {batch_index}/{len(symbol_batches)}",
            f"symbols={len(batch)}",
        )
        websocket = WebSocket(testnet=False, channel_type="linear")
        websocket_connections.append(websocket)
        for symbol in batch:
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
        time.sleep(0.5)

    send_tg(
        "✅ Crypto Tide Engine V10.7 confirmed-only monitor started\n\n"
        f"All robust-eligible symbols: {len(symbols)}\n"
        f"Websocket connections: {len(websocket_connections)}\n"
        f"Max research positions: {PORTFOLIO_MAX_POSITIONS}\n"
        f"Reference capital: {PORTFOLIO_CAPITAL_USDT:.0f} USDT\n"
        f"Maximum margin utilisation: {PORTFOLIO_MAX_MARGIN_UTILISATION:.0%}\n"
        f"Base margin: {BASE_MARGIN_USDT:.0f} USDT\n"
        f"Maximum leverage: {DEFAULT_LEVERAGE:.0f}x\n"
        f"Base account risk: {BASE_ACCOUNT_RISK_PCT:.2%}\n"
        f"Maximum signal risk: {MAX_SIGNAL_RISK_PCT:.2%}\n"
        f"Maximum total open risk: {MAX_TOTAL_OPEN_RISK_PCT:.2%}\n"
        f"Maximum daily loss: {MAX_DAILY_LOSS_PCT:.2%}\n"
        f"Pre-alert Telegram: {'ON' if SEND_PRE_ALERTS else 'OFF'}\n"
        f"Watchlist refresh: every {WATCHLIST_REFRESH_HOURS} hours\n"
        f"Full learning refresh: daily at "
        f"{DAILY_RESCAN_LOCAL_HOUR:02d}:00 Sydney time"
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
        log(
            "Heartbeat monitoring",
            len(symbols),
            "symbols across",
            len(websocket_connections),
            "connections",
        )
        time.sleep(60)


def main() -> None:
    token, chat_id = telegram_credentials()
    log(
        "V9.2 startup",
        f"persistent_data_dir={DATA_DIR}",
        f"telegram_token_set={bool(token)}",
        f"telegram_chat_id_set={bool(chat_id)}",
    )
    update_online_signal_weights()
    stage2 = load_or_run_stage2()
    selected = select_current_watchlist(stage2)
    start_monitor(selected)


if __name__ == "__main__":
    main()
