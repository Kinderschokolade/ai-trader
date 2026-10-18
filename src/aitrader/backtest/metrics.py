"""Performance metrics from an equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def _cagr(values: np.ndarray) -> float:
    if len(values) < 2 or values[0] <= 0:
        return np.nan
    years = (len(values) - 1) / DAYS
    return float((values[-1] / values[0]) ** (1 / years) - 1) if values[-1] > 0 else -1.0


def _max_dd(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1))


def compute(result) -> dict:
    eq = result.equity
    v = eq["equity"].to_numpy(float)
    b = eq["benchmark"].to_numpy(float)
    r = np.diff(v) / v[:-1]
    rb = np.diff(b) / b[:-1]
    vol = float(np.std(r, ddof=1) * np.sqrt(DAYS)) if len(r) > 2 else np.nan
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(DAYS)) if len(r) > 2 and np.std(r) > 0 else np.nan
    cagr, bcagr = _cagr(v), _cagr(b)
    mdd = _max_dd(v)
    beta = float(np.cov(r, rb)[0, 1] / np.var(rb, ddof=1)) if len(r) > 2 and np.var(rb) > 0 else np.nan
    t = result.trades
    closed = t[t["reason"] != "open"] if not t.empty else t
    out = {
        "strategy": result.label,
        "start": str(eq["date"].iloc[0].date()),
        "end": str(eq["date"].iloc[-1].date()),
        "final_equity": float(v[-1]),
        "total_return": float(v[-1] / v[0] - 1),
        "cagr": cagr,
        "benchmark_cagr": bcagr,
        "excess_cagr": cagr - bcagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "beta": beta,
        "avg_invested_pct": float((eq["invested"] / eq["equity"]).mean()),
        "avg_positions": float(eq["n_positions"].mean()),
        "n_trades": int(len(t)),
        "hit_rate": float((closed["return"] > 0).mean()) if len(closed) else np.nan,
        "avg_trade_return": float(closed["return"].mean()) if len(closed) else np.nan,
        "median_trade_return": float(closed["return"].median()) if len(closed) else np.nan,
        "avg_holding_days": float(closed["holding_days"].mean()) if len(closed) else np.nan,
    }
    out.update(result.stats)
    return out
