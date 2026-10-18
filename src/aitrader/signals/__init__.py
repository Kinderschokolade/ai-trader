"""Signal builders. Every builder takes (con, strategy) and returns a DataFrame with at least:

    signal_id, strategy, side ('buy' | 'sell'), issuer_cik, issuer_name, ticker,
    known_date   (the day the information became public - we can only act after it),
    total_value  (USD, used for ranking)

The same builders run in backtests (whole history) and live (only recent filings)."""
from __future__ import annotations

import pandas as pd

from .insider import build_cluster_signals, build_large_signals

BUILDERS = {
    "insider_cluster": build_cluster_signals,
    "insider_large": build_large_signals,
}


def build_signals(con, strategy) -> pd.DataFrame:
    if strategy.status != "active":
        raise NotImplementedError(
            f"Strategy {strategy.name!r} has status {strategy.status!r} (planned for phase 2)."
        )
    try:
        builder = BUILDERS[strategy.type]
    except KeyError:
        raise NotImplementedError(f"No signal builder for strategy type {strategy.type!r}") from None
    df = builder(con, strategy)
    df["strategy"] = strategy.name
    return attach_price_tickers(con, df)


def attach_price_tickers(con, df: pd.DataFrame) -> pd.DataFrame:
    """Add `price_ticker`: the symbol under which prices are stored (see data.prices)."""
    from ..data.prices import yahoo_symbol

    if df.empty:
        df["price_ticker"] = pd.Series(dtype=str)
        return df
    tm = con.execute("SELECT issuer_cik, signal_ticker AS ticker, price_ticker FROM ticker_map").df()
    out = df.merge(tm, on=["issuer_cik", "ticker"], how="left")
    out["price_ticker"] = out["price_ticker"].fillna(out["ticker"].map(lambda t: yahoo_symbol(t) if isinstance(t, str) else None))
    return out
