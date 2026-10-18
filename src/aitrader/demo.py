"""Synthetic database for trying every command without downloading anything.

The data is fake but has the same tables as the real download. A known effect is planted:
after a cluster buy, micro and small caps drift upwards for ~6 months (strongest after a
price fall); large caps get nothing. A good pipeline should find exactly that.
Some companies go bankrupt or get acquired, and a few never have prices.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .db import connect, insert_df

N_COMPANIES = 400
DRIFT = {"micro": 0.18, "small": 0.10, "mid": 0.04, "large": 0.0}   # planted 6-month abnormal drift
VOL = {"micro": 0.55, "small": 0.40, "mid": 0.30, "large": 0.22}


def _size(mcap: float) -> str:
    return "micro" if mcap < 3e8 else "small" if mcap < 2e9 else "mid" if mcap < 1e10 else "large"


def build_demo_db(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    path = Path(path)
    if path.exists():
        path.unlink()
    con = connect(path)
    cal = pd.bdate_range("2006-01-02", "2025-12-31")
    n = len(cal)

    # benchmark
    spy_r = rng.normal(0.08 / 252, 0.18 / np.sqrt(252), n)
    spy = 100 * np.exp(np.cumsum(spy_r))
    price_frames = [_price_frame("SPY", cal, spy)]

    companies = []
    filings, owners, trans = [], [], []
    acc = 0
    for k in range(N_COMPANIES):
        cik, ticker = str(100000 + k), f"T{k:03d}"
        mcap0 = float(np.exp(rng.uniform(np.log(5e7), np.log(8e10))))
        size = _size(mcap0)
        p0 = rng.uniform(5, 80)
        shares = mcap0 / p0
        # idiosyncratic log return with -sigma^2/2 drift, so its arithmetic expectation is zero
        sig = VOL[size] / np.sqrt(252)
        r = spy_r * rng.uniform(0.7, 1.4) + rng.normal(-sig ** 2 / 2, sig, n)
        end_idx = n - 1
        fate = "alive"
        roll = rng.random()
        if roll < 0.06:
            fate, end_idx = "bankruptcy", int(rng.integers(n // 4, n - 300))
            r[end_idx - 120:end_idx] -= 0.006  # slide before the end
        elif roll < 0.12:
            fate, end_idx = "acquired", int(rng.integers(n // 4, n - 300))
            r[end_idx - 1] += 0.25  # takeover premium
        # insider events: clusters (2-3 insiders) and lone buys
        events = []
        for _ in range(rng.poisson(5)):
            events.append(("cluster", int(rng.integers(300, end_idx - 5)), int(rng.integers(2, 4))))
        for _ in range(rng.poisson(8)):
            events.append(("single", int(rng.integers(300, end_idx - 5)), 1))
        for kind, i, n_ins in events:
            if kind == "cluster":
                drift = DRIFT[size]
                pre = np.sum(r[i - 63:i] - spy_r[i - 63:i])
                if pre < -0.15:
                    drift *= 1.5
                ramp_end = min(i + 1 + 126, n)
                r[i + 1:ramp_end] += drift / 126
        prices = p0 * np.exp(np.cumsum(r))
        if fate != "no_prices" and rng.random() > 0.025:
            price_frames.append(_price_frame(ticker, cal[:end_idx + 1], prices[:end_idx + 1]))
            has_prices = True
        else:
            has_prices = False
        companies.append((cik, ticker, size, fate, end_idx, has_prices, shares))
        # Form 4 rows
        for kind, i, n_ins in events:
            for j in range(n_ins):
                acc += 1
                fd = cal[min(i + (j * 3 if kind == "cluster" else 0), end_idx)]
                px = float(prices[min(i, end_idx)])
                value = float(rng.uniform(20_000, 400_000) if kind == "cluster" else rng.uniform(5_000, 600_000))
                a = f"0000{acc:08d}-{fd:%y}-{acc:06d}"
                filings.append((a, fd.date(), fd.date(), "4", cik, f"Company {k:03d} Inc", ticker, "0", f"{fd.year}q{(fd.month - 1) // 3 + 1}"))
                owner = f"{cik}{(j + i) % 7}" if kind == "cluster" else f"{cik}9"
                owners.append((a, owner, f"Insider {owner}", "Director" if j % 2 == 0 else "Officer", "CEO" if j == 1 else ""))
                trans.append((a, str(acc), "Common Stock", (fd - pd.Timedelta(days=2)).date(), "P", "A",
                              value / px, px, value / px * 10, "D"))

    prices_df = pd.concat(price_frames, ignore_index=True)
    insert_df(con, "prices", prices_df)
    status = prices_df.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    status.columns = ["ticker", "first_date", "last_date", "n_rows"]
    status["source"], status["fetched_at"], status["status"] = "demo", datetime.now(), "ok"
    insert_df(con, "price_status", status[["ticker", "source", "first_date", "last_date", "n_rows", "fetched_at", "status"]])

    cols_f = ["accession", "filing_date", "period_of_report", "document_type", "issuer_cik", "issuer_name", "ticker", "aff10b5one", "quarter"]
    insert_df(con, "form4_filings", pd.DataFrame(filings, columns=cols_f))
    insert_df(con, "form4_owners", pd.DataFrame(owners, columns=["accession", "owner_cik", "owner_name", "relationship", "title"]))
    insert_df(con, "form4_trans", pd.DataFrame(trans, columns=["accession", "trans_sk", "security_title", "trans_date", "trans_code", "acq_disp", "shares", "price", "shares_after", "direct_indirect"]))
    quarters = sorted({f[-1] for f in filings})
    insert_df(con, "loaded_quarters", pd.DataFrame({"source": "form4", "quarter": quarters, "loaded_at": datetime.now(), "n_filings": 0}))

    tm, so, dl = [], [], []
    for cik, ticker, size, fate, end_idx, has_prices, shares in companies:
        if has_prices:
            tm.append((cik, ticker, ticker))
        for y in range(2005, 2026):
            so.append((cik, pd.Timestamp(f"{y}-12-31").date(), pd.Timestamp(f"{y + 1}-02-15").date(), shares, "EntityCommonStockSharesOutstanding"))
        if fate in ("bankruptcy", "acquired"):
            dl.append((cik, cal[end_idx].date(), fate, "8-K" if fate == "bankruptcy" else "DEFM14A", cal[end_idx].date(), datetime.now()))
        elif not has_prices:
            dl.append((cik, None, "unknown", None, None, datetime.now()))
    insert_df(con, "ticker_map", pd.DataFrame(tm, columns=["issuer_cik", "signal_ticker", "price_ticker"]))
    insert_df(con, "shares_outstanding", pd.DataFrame(so, columns=["cik", "end_date", "filed_date", "shares", "concept"]))
    insert_df(con, "delistings", pd.DataFrame(dl, columns=["cik", "last_price_date", "classification", "evidence_form", "evidence_date", "checked_at"]))
    con.close()


def _price_frame(ticker: str, dates, close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, float)
    openp = np.r_[close[0], close[:-1]] * (1 + np.random.default_rng(len(close)).normal(0, 0.003, len(close)))
    return pd.DataFrame({
        "ticker": ticker, "date": pd.DatetimeIndex(dates).date, "open": openp,
        "high": np.maximum(openp, close) * 1.005, "low": np.minimum(openp, close) * 0.995,
        "close": close, "adj_close": close, "volume": 1e5,
    })
