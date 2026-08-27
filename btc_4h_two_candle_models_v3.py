#!/usr/bin/env python3
"""Compare several two-candle models for predicting the third BTC 4H candle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from btc_4h_three_candle_v2 import download


FEATURES = [
    "c1_return", "c1_body_ratio", "c1_upper_wick", "c1_lower_wick", "c1_log_volume",
    "c2_return", "c2_body_ratio", "c2_upper_wick", "c2_lower_wick", "c2_log_volume",
    "return_change", "body_strength_ratio", "range_ratio", "volume_ratio",
    "same_colour", "two_candle_return", "c2_engulfs_c1",
]


def numeric_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    rng = (df.high - df.low).replace(0, np.nan)
    body = df.close - df.open
    abs_body = body.abs()
    upper = df.high - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df.low
    ret = df.close / df.open - 1

    # c1 = older observed candle (t-1); c2 = most recent completed candle (t).
    for prefix, lag in [("c1", 1), ("c2", 0)]:
        out[f"{prefix}_return"] = ret.shift(lag)
        out[f"{prefix}_body_ratio"] = (body / rng).shift(lag)
        out[f"{prefix}_upper_wick"] = (upper / rng).shift(lag)
        out[f"{prefix}_lower_wick"] = (lower / rng).shift(lag)
        out[f"{prefix}_log_volume"] = np.log1p(df.volume).shift(lag)
    out["return_change"] = ret - ret.shift(1)
    out["body_strength_ratio"] = abs_body / abs_body.shift(1).replace(0, np.nan)
    out["range_ratio"] = rng / rng.shift(1)
    out["volume_ratio"] = df.volume / df.volume.shift(1).replace(0, np.nan)
    out["same_colour"] = ((body >= 0) == (body.shift(1) >= 0)).astype(int)
    out["two_candle_return"] = df.close / df.open.shift(1) - 1
    c1_hi = df[["open", "close"]].max(axis=1).shift(1)
    c1_lo = df[["open", "close"]].min(axis=1).shift(1)
    c2_hi = df[["open", "close"]].max(axis=1)
    c2_lo = df[["open", "close"]].min(axis=1)
    out["c2_engulfs_c1"] = ((c2_hi >= c1_hi) & (c2_lo <= c1_lo)).astype(int)

    out["entry_time"] = df.index.to_series().shift(-1)
    out["entry"] = df.open.shift(-1)
    out["exit"] = df.close.shift(-1)
    out["next_high"] = df.high.shift(-1)
    out["next_low"] = df.low.shift(-1)
    out["next_return"] = out.exit / out.entry - 1
    out["target"] = (out.next_return > 0).astype(int)
    # Winsorisation levels are fixed, not fitted with future data.
    for c in ["body_strength_ratio", "range_ratio", "volume_ratio"]:
        out[c] = out[c].clip(0, 10)
    return out.replace([np.inf, -np.inf], np.nan).dropna().copy()


def models(seed: int) -> dict:
    return {
        "logistic": Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=30, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, min_samples_leaf=30, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=2.0, random_state=seed),
    }


def walk_forward(data: pd.DataFrame, min_train: int, retrain_every: int, seed: int) -> pd.DataFrame:
    chunks = []
    for start in range(min_train, len(data), retrain_every):
        stop = min(start + retrain_every, len(data))
        train, test = data.iloc[:start], data.iloc[start:stop]
        for name, model in models(seed).items():
            model.fit(train[FEATURES], train.target)
            d = test[["entry_time", "entry", "exit", "next_high", "next_low", "next_return"]].copy()
            d["model"] = name
            d["prob_up"] = model.predict_proba(test[FEATURES])[:, 1]
            chunks.append(d)
    return pd.concat(chunks).sort_index()


def score(events: pd.DataFrame, model: str, threshold: float, cost: float) -> dict:
    d = events[events.model == model].copy()
    d["signal"] = np.where(d.prob_up >= threshold, 1,
                           np.where(d.prob_up <= 1 - threshold, -1, 0))
    d = d[d.signal != 0]
    if d.empty:
        return {"model": model, "threshold": threshold, "trades": 0}
    gross = d.signal * d.next_return
    net = gross - cost
    eq = (1 + net).cumprod(); dd = eq / eq.cummax() - 1
    wins, losses = net[net > 0], net[net < 0]
    return {
        "model": model, "threshold": threshold, "trades": len(d),
        "longs": int((d.signal > 0).sum()), "shorts": int((d.signal < 0).sum()),
        "direction_accuracy": (gross > 0).mean(), "net_win_rate": (net > 0).mean(),
        "avg_gross": gross.mean(), "avg_net": net.mean(),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() else np.inf,
        "compounded_return": eq.iloc[-1] - 1, "max_drawdown": dd.min(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--round-trip-bps", type=float, default=12.0)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--retrain-every", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="btc_4h_two_candle_models_v3_results")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    data = numeric_dataset(download(a.symbol, a.start, a.end))
    events = walk_forward(data, a.min_train, a.retrain_every, a.seed)
    rows = []
    for model in models(a.seed):
        for threshold in [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65,
                          0.675, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.75]:
            rows.append(score(events, model, threshold, a.round_trip_bps / 10000))
    summary = pd.DataFrame(rows).sort_values(["direction_accuracy", "trades"], ascending=False)
    eligible = summary[summary.trades >= 100].copy()
    summary.to_csv(out / "all_model_thresholds.csv", index=False)
    events.to_csv(out / "walk_forward_probabilities.csv")
    print("\nTOP DIRECTION ACCURACY (MINIMUM 100 TRADES)")
    print(eligible.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTOP NET EXPECTANCY (MINIMUM 100 TRADES)")
    print(eligible.sort_values("avg_net", ascending=False).head(20).to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()
