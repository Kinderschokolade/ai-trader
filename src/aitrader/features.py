"""Point-in-time features of a signal: company size and price trend before the filing.

Everything uses only data up to and including the filing date (`known_date`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.fundamentals import shares_asof


def bucketize(values: pd.Series, edges, labels) -> pd.Series:
    edges = [float(e) for e in edges]
    if len(labels) != len(edges) + 1:
        raise ValueError(f"Need {len(edges) + 1} labels for {len(edges)} edges, got {labels}")
    idx = np.digitize(values.fillna(np.nan).values, edges)
    out = pd.Series([labels[i] for i in idx], index=values.index)
    out[values.isna()] = "unknown"
    return out


def add_features(signals: pd.DataFrame, store, con, strategy, benchmark: str) -> pd.DataFrame:
    df = signals.copy()
    if df.empty:
        for c in ("pre_return", "size_value", "size_bucket", "trend_bucket"):
            df[c] = pd.Series(dtype=object)
        return df
    look = int(strategy.get("features.trend_lookback_days", 63))
    relative = bool(strategy.get("features.trend_relative_to_benchmark", True))
    size_metric = strategy.get("features.size_metric", "market_cap")
    store.preload(list(df["price_ticker"].dropna().unique()) + [benchmark])
    bench = store.get(benchmark)

    pre, close_px, dollar_vol = [], [], []
    for t, d in zip(df["price_ticker"], df["known_date"]):
        s = store.get(t)
        i = store.index_on_or_before(d)
        if s is None or i is None or i < s.first_idx or i - look < s.first_idx or i > s.last_idx:
            pre.append(np.nan)
            close_px.append(s.close[i] if s is not None and i is not None and s.first_idx <= i <= s.last_idx else np.nan)
            dollar_vol.append(np.nan)
            continue
        r = s.adj_close[i] / s.adj_close[i - look] - 1
        if relative and bench is not None and i - look >= bench.first_idx:
            r -= bench.adj_close[i] / bench.adj_close[i - look] - 1
        pre.append(r)
        close_px.append(s.close[i])
        lo = max(s.first_idx, i - look + 1)
        dollar_vol.append(float(np.nanmean(s.close[lo:i + 1] * s.volume[lo:i + 1])))
    df["pre_return"] = pre
    df["close_at_signal"] = close_px

    if size_metric == "market_cap":
        shares = shares_asof(con, df["issuer_cik"], df["known_date"])
        df["size_value"] = shares.values * df["close_at_signal"].values
    elif size_metric == "dollar_volume":
        df["size_value"] = dollar_vol
    else:
        raise ValueError(f"Unknown features.size_metric {size_metric!r}")

    df["size_bucket"] = bucketize(df["size_value"], strategy.get("features.size_edges"), strategy.get("features.size_labels"))
    df["trend_bucket"] = bucketize(df["pre_return"], strategy.get("features.trend_edges"), strategy.get("features.trend_labels"))
    return df


def apply_feature_filters(df: pd.DataFrame, strategy) -> pd.DataFrame:
    size = strategy.get("filters.size_buckets") or []
    trend = strategy.get("filters.trend_buckets") or []
    if size:
        df = df[df["size_bucket"].isin(size)]
    if trend:
        df = df[df["trend_bucket"].isin(trend)]
    return df
