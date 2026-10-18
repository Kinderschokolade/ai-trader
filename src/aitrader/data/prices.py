"""Daily prices (Yahoo Finance via yfinance) and the in-memory price store used by backtests.

Yahoo usually has no data for delisted companies and only knows a company's current
ticker. So for every (issuer CIK, ticker at signal time) we try:
  1. the ticker from the filing (Yahoo spelling, e.g. BRK.B -> BRK-B)
  2. the company's current ticker according to the SEC (company_tickers.json)
and record what worked in `ticker_map`. Signals that still have no prices are reported
by the backtest ("skipped: no_price") and checked by the delisting module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..db import insert_df

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
BAD_TICKERS = {"", "NONE", "N/A", "NA", "NULL", "-", "--", "NOTICKER"}


def clean_ticker(raw) -> str | None:
    """Filing tickers are free text: 'abc', 'ABC, ABCD', 'NYSE: ABC', 'BRK.B'."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    t = str(raw).upper().strip()
    t = re.sub(r"^(NYSE|NASDAQ|AMEX|OTC|NYSEAMERICAN|NYSE AMERICAN|NYSEARCA)\s*[:\-]?\s*", "", t)
    t = re.split(r"[\s,;/]+", t)[0] if t else t
    t = t.strip(".-")
    if t in BAD_TICKERS or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,9}", t):
        return None
    return t


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


class YahooSource:
    name = "yahoo"

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        frames = []
        for i in range(0, len(tickers), 50):
            chunk = tickers[i:i + 50]
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=False,
                                  group_by="ticker", threads=True, progress=False)
            except Exception as exc:  # network hiccups: try one by one
                print(f"  yahoo batch failed ({exc}); retrying tickers one by one")
                raw = None
            frames.extend(_split_download(raw, chunk) if raw is not None else [])
            if raw is None:
                for t in chunk:
                    try:
                        one = yf.download(t, start=start, end=end, auto_adjust=False, actions=False,
                                          group_by="ticker", progress=False)
                        frames.extend(_split_download(one, [t]))
                    except Exception:
                        pass
        if not frames:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])
        return pd.concat(frames, ignore_index=True)


def _split_download(raw: pd.DataFrame, tickers: list[str]) -> list[pd.DataFrame]:
    """yf.download returns MultiIndex columns (ticker, field) or (field, ticker) depending on version."""
    out = []
    if raw is None or raw.empty:
        return out
    fields = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    cols = raw.columns
    for t in tickers:
        if isinstance(cols, pd.MultiIndex):
            if t in cols.get_level_values(0):
                sub = raw[t]
            elif t in cols.get_level_values(1):
                sub = raw.xs(t, axis=1, level=1)
            else:
                continue
        else:
            if len(tickers) != 1:
                continue
            sub = raw
        sub = sub.rename(columns=fields)
        if "close" not in sub.columns:
            continue
        if "adj_close" not in sub.columns:
            sub["adj_close"] = sub["close"]
        sub = sub[[c for c in fields.values() if c in sub.columns]].dropna(subset=["close"])
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.date
        sub.insert(0, "ticker", t)
        out.append(sub)
    return out


def get_source(name: str):
    if name == "yahoo":
        return YahooSource()
    raise ValueError(f"Unknown price source {name!r}")


def sec_current_tickers(client) -> dict[str, str]:
    """issuer CIK (no leading zeros) -> current ticker according to the SEC."""
    data = client.get(SEC_TICKERS_URL).json()
    out: dict[str, str] = {}
    for row in data.values():
        out.setdefault(str(row["cik_str"]), str(row["ticker"]).upper())
    return out


def _store(con, df: pd.DataFrame, source: str, requested: list[str]) -> set[str]:
    got = set(df["ticker"].unique()) if not df.empty else set()
    if not df.empty:
        insert_df(con, "prices", df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]], replace=True)
    now = datetime.now()
    status = []
    for t in requested:
        if t in got:
            sub = df[df["ticker"] == t]
            status.append((t, source, min(sub["date"]), max(sub["date"]), len(sub), now, "ok"))
        else:
            status.append((t, source, None, None, 0, now, "no_data"))
    insert_df(con, "price_status", pd.DataFrame(status, columns=["ticker", "source", "first_date", "last_date", "n_rows", "fetched_at", "status"]), replace=True)
    return got


def ensure_prices(con, settings, pairs: pd.DataFrame, client=None, refresh: bool = False, log=print) -> None:
    """Download prices for (issuer_cik, ticker) pairs plus the benchmark.

    pairs: DataFrame with columns issuer_cik, ticker (ticker at signal time).
    """
    source = get_source(settings.get("data.price_source", "yahoo"))
    start = settings.get("data.start")
    end = (datetime.fromisoformat(settings.data_end) + timedelta(days=1)).date().isoformat()

    known = {r[0]: r[1] for r in con.execute("SELECT ticker, status FROM price_status").fetchall()}
    bench = settings.get("portfolio.benchmark", "SPY")
    if refresh or bench not in known:
        _store(con, source.fetch([bench], start, end), source.name, [bench])

    pairs = pairs.dropna(subset=["ticker"]).drop_duplicates(["issuer_cik", "ticker"])
    mapped = {(r[0], r[1]) for r in con.execute("SELECT issuer_cik, signal_ticker FROM ticker_map").fetchall()}
    todo = pairs[[(c, t) not in mapped for c, t in zip(pairs["issuer_cik"], pairs["ticker"])]] if not refresh else pairs

    # Pass 1: the ticker as written in the filing.
    first = sorted({yahoo_symbol(t) for t in todo["ticker"]})
    need = [s for s in first if refresh or s not in known]
    log(f"prices: {len(todo):,} company/ticker pairs, {len(need):,} symbols to download")
    for i in range(0, len(need), 500):
        part = need[i:i + 500]
        got = _store(con, source.fetch(part, start, end), source.name, part)
        known.update({s: ("ok" if s in got else "no_data") for s in part})
        log(f"  {min(i + 500, len(need)):,}/{len(need):,} symbols, {len(got)} with data")

    # Pass 2: current SEC ticker for companies whose filing ticker had no data.
    mapping_rows = []
    missing = []
    for cik, t in zip(todo["issuer_cik"], todo["ticker"]):
        sym = yahoo_symbol(t)
        if known.get(sym) == "ok":
            mapping_rows.append((cik, t, sym))
        else:
            missing.append((cik, t))
    if missing and client is not None:
        current = sec_current_tickers(client)
        alt = {}
        for cik, t in missing:
            cur = current.get(str(cik))
            if cur and yahoo_symbol(cur) != yahoo_symbol(t):
                alt[(cik, t)] = yahoo_symbol(cur)
        need2 = sorted({s for s in alt.values() if refresh or s not in known})
        if need2:
            log(f"prices: retrying {len(need2):,} companies under their current SEC ticker")
            for i in range(0, len(need2), 500):
                part = need2[i:i + 500]
                got = _store(con, source.fetch(part, start, end), source.name, part)
                known.update({s: ("ok" if s in got else "no_data") for s in part})
        for (cik, t), sym in alt.items():
            if known.get(sym) == "ok":
                mapping_rows.append((cik, t, sym))
    if mapping_rows:
        insert_df(con, "ticker_map", pd.DataFrame(mapping_rows, columns=["issuer_cik", "signal_ticker", "price_ticker"]).drop_duplicates(["issuer_cik", "signal_ticker"]), replace=True)
    n_ok = len(mapping_rows)
    log(f"prices: {n_ok:,}/{len(todo):,} pairs have prices")


# ---------------------------------------------------------------------------------------
# In-memory access for backtests and event studies
# ---------------------------------------------------------------------------------------

@dataclass
class Series:
    adj_close: np.ndarray   # aligned to the calendar, forward-filled inside the data range
    adj_open: np.ndarray
    close: np.ndarray       # unadjusted close (for market cap)
    volume: np.ndarray
    has_data: np.ndarray    # True on days with an actual price row
    first_idx: int
    last_idx: int
    ended: bool             # stops trading before the end of the data (delisted / ticker gap)


class PriceStore:
    def __init__(self, con, calendar: pd.DatetimeIndex, stale_days: int = 10):
        self.con = con
        self.calendar = calendar
        self.stale_days = stale_days
        self._cache: dict[str, Series | None] = {}

    @classmethod
    def from_benchmark(cls, con, benchmark: str, start: str, end: str, stale_days: int = 10) -> "PriceStore":
        rows = con.execute("SELECT date FROM prices WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
                           [benchmark, start, end]).fetchall()
        if not rows:
            raise RuntimeError(f"No prices for benchmark {benchmark}. Run `aitrader download prices <strategy>` first.")
        cal = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        return cls(con, cal, stale_days)

    def preload(self, tickers) -> None:
        tickers = [t for t in set(tickers) if t and t not in self._cache]
        if not tickers:
            return
        start, end = self.calendar[0].date(), self.calendar[-1].date()
        df = self.con.execute(
            "SELECT ticker, date, open, close, adj_close, volume FROM prices "
            "WHERE ticker IN (SELECT unnest(?)) AND date BETWEEN ? AND ? ORDER BY ticker, date",
            [tickers, start, end]).df()
        groups = dict(tuple(df.groupby("ticker"))) if not df.empty else {}
        for t in tickers:
            self._cache[t] = self._build(groups.get(t))

    def _build(self, df: pd.DataFrame | None) -> Series | None:
        if df is None or df.empty:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").set_index("date").sort_index()
        df = df[df["adj_close"] > 0]
        if df.empty:
            return None
        factor = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        df["adj_open"] = (df["open"].fillna(df["close"]) * factor)
        has = pd.Series(False, index=self.calendar)
        has.loc[has.index.intersection(df.index)] = True
        aligned = df.reindex(self.calendar)
        idx_valid = np.flatnonzero(has.values)
        if len(idx_valid) == 0:
            return None
        first, last = int(idx_valid[0]), int(idx_valid[-1])
        adj_close = np.array(aligned["adj_close"].ffill(), dtype=float)
        adj_close[:first] = np.nan
        adj_close[last + 1:] = np.nan
        adj_open = np.array(aligned["adj_open"], dtype=float)
        adj_open = np.where(np.isnan(adj_open), adj_close, adj_open)
        close = np.array(aligned["close"].ffill(), dtype=float)
        close[:first] = np.nan
        close[last + 1:] = np.nan
        ended = last < len(self.calendar) - 1 - self.stale_days
        return Series(adj_close, adj_open, close, np.array(aligned["volume"].fillna(0), dtype=float),
                      np.array(has.values, dtype=bool), first, last, ended)

    def get(self, ticker: str | None) -> Series | None:
        if not ticker:
            return None
        if ticker not in self._cache:
            self.preload([ticker])
        return self._cache.get(ticker)

    def index_after(self, d, lag: int) -> int | None:
        """Calendar index of the trading day `lag` days after date d (lag=1: next trading day;
        lag=0: d itself if it is a trading day, else the next one)."""
        ts = pd.Timestamp(d)
        i = int(self.calendar.searchsorted(ts, side="right" if lag >= 1 else "left"))
        i += max(lag - 1, 0)
        return i if i < len(self.calendar) else None

    def index_on_or_before(self, d) -> int | None:
        i = int(self.calendar.searchsorted(pd.Timestamp(d), side="right")) - 1
        return i if i >= 0 else None
