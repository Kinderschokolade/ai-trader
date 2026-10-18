"""Event study: how does a stock perform, relative to the benchmark, in the days after a signal?

For every signal we buy at the entry day (same rule as the backtest) and compute the
buy-and-hold abnormal return
    BHAR(h) = P(t+h)/P(t) - B(t+h)/B(t)
for h = 0 ... horizon trading days. If the stock stops trading, its value is frozen at the
last price minus the delisting haircut while the benchmark keeps moving.
Curves are averaged per bucket (company size x trend before the filing), with the spread
(25th-75th percentile) and the standard error of the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_paths(signals: pd.DataFrame, store, settings, delist_classes: dict, horizon: int):
    """Returns (matrix [n_events, horizon+1] of BHAR, DataFrame of events actually used)."""
    lag = int(settings.get("execution.entry_lag_days", 1))
    use_open = settings.get("execution.execution_price", "open") == "open"
    haircut = {k: float(v) for k, v in (settings.get("delisting.haircut", {}) or {}).items()}
    bench = store.get(settings.get("portfolio.benchmark", "SPY"))
    n_cal = len(store.calendar)
    store.preload(signals["price_ticker"].dropna().unique())

    paths, used, reasons = [], [], []
    for row in signals.itertuples(index=False):
        ser = store.get(row.price_ticker)
        i = store.index_after(row.known_date, lag)
        if ser is None or i is None or i < ser.first_idx or i > ser.last_idx:
            reasons.append("no_price")
            continue
        p0 = ser.adj_open[i] if use_open else ser.adj_close[i]
        b0 = bench.adj_open[i] if use_open else bench.adj_close[i]
        end = min(i + horizon, n_cal - 1)
        h = np.arange(0, end - i + 1)
        stock = ser.adj_close[i:end + 1].astype(float).copy()
        if ser.ended and ser.last_idx < end:
            cls = delist_classes.get(str(row.issuer_cik), "unchecked")
            cut = haircut.get(cls, haircut.get("unknown", 0.5))
            k = ser.last_idx - i
            stock[k + 1:] = ser.adj_close[ser.last_idx] * (1 - cut)
        elif ser.last_idx < end:  # data simply ends (end of download) -> censored
            stock[ser.last_idx - i + 1:] = np.nan
        path = np.full(horizon + 1, np.nan)
        path[h] = stock / p0 - bench.adj_close[i:end + 1] / b0
        path[0] = ser.adj_close[i] / p0 - bench.adj_close[i] / b0
        paths.append(path)
        used.append(row)
        reasons.append("ok")
    mat = np.vstack(paths) if paths else np.empty((0, horizon + 1))
    used_df = pd.DataFrame(used, columns=signals.columns) if used else signals.iloc[0:0].copy()
    return mat, used_df, pd.Series(reasons).value_counts().to_dict()


def _curve(mat: np.ndarray) -> pd.DataFrame:
    with np.errstate(all="ignore"):
        n = np.sum(~np.isnan(mat), axis=0)
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0, ddof=1)
        return pd.DataFrame({
            "day": np.arange(mat.shape[1]),
            "n": n,
            "mean": mean,
            "median": np.nanmedian(mat, axis=0),
            "p25": np.nanpercentile(mat, 25, axis=0),
            "p75": np.nanpercentile(mat, 75, axis=0),
            "sem": sd / np.sqrt(np.maximum(n, 1)),
        })


def summarize(curves: pd.DataFrame, min_events: int) -> pd.DataFrame:
    rows = []
    for (dim, bucket), c in curves.groupby(["dimension", "bucket"], sort=False):
        c = c[c["n"] > 0]
        if c.empty:
            continue
        smooth = c["mean"].rolling(21, center=True, min_periods=1).mean()
        k = int(np.nanargmax(smooth.values)) if smooth.notna().any() else 0
        peak = c.iloc[k]
        row = {
            "dimension": dim, "bucket": bucket, "n_events": int(c["n"].iloc[0]),
            "peak_day": int(peak["day"]), "mean_at_peak": float(peak["mean"]),
            "t_stat_at_peak": float(peak["mean"] / peak["sem"]) if peak["sem"] > 0 else np.nan,
            "enough_events": bool(c["n"].iloc[0] >= min_events),
        }
        for h in (21, 63, 126, 252, 504):
            hit = c[c["day"] == h]
            row[f"mean_d{h}"] = float(hit["mean"].iloc[0]) if len(hit) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(signals: pd.DataFrame, store, settings, strategy, delist_classes: dict):
    horizon = int(strategy.get("event_study.horizon_days", 504))
    mat, used, reasons = event_paths(signals, store, settings, delist_classes, horizon)
    frames = []
    c = _curve(mat)
    c.insert(0, "bucket", "all")
    c.insert(0, "dimension", "all")
    frames.append(c)
    dims = strategy.get("event_study.bucket_by", ["size_bucket", "trend_bucket"]) or []
    for dim in dims:
        for bucket in _ordered(used[dim], strategy, dim):
            m = mat[(used[dim] == bucket).values]
            if len(m) == 0:
                continue
            c = _curve(m)
            c.insert(0, "bucket", bucket)
            c.insert(0, "dimension", dim)
            frames.append(c)
    if len(dims) >= 2:  # combinations of the first two dimensions
        a, b = dims[0], dims[1]
        for ba in _ordered(used[a], strategy, a):
            for bb in _ordered(used[b], strategy, b):
                m = mat[((used[a] == ba) & (used[b] == bb)).values]
                if len(m) == 0:
                    continue
                c = _curve(m)
                c.insert(0, "bucket", f"{ba} | {bb}")
                c.insert(0, "dimension", f"{a} x {b}")
                frames.append(c)
    curves = pd.concat(frames, ignore_index=True)
    summary = summarize(curves, int(strategy.get("event_study.min_bucket_events", 30)))
    per_event = used.copy()
    for h in (21, 63, 126, 252):
        if h < mat.shape[1]:
            per_event[f"bhar_d{h}"] = mat[:, h]
    return curves, summary, per_event, reasons


def _ordered(values: pd.Series, strategy, dim: str) -> list[str]:
    labels = {
        "size_bucket": strategy.get("features.size_labels", []),
        "trend_bucket": strategy.get("features.trend_labels", []),
    }.get(dim, [])
    present = list(dict.fromkeys(values.dropna()))
    ordered = [l for l in labels if l in present] + [p for p in present if p not in labels]
    return ordered
