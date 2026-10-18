"""Insider buying signals from Form 4 filings.

One "insider" is one Form 4 filing owner. Joint filings (e.g. a fund and its manager file
together) count as ONE insider, identified by the smallest owner CIK in the filing, so they
cannot fake a cluster on their own.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data.prices import clean_ticker

TRUTHY = {"1", "TRUE", "Y", "YES"}


def _roles(rel) -> set[str]:
    if not isinstance(rel, str):
        return set()
    return {re.sub(r"[^a-z0-9]", "", t.lower()) for t in rel.split(",") if t.strip()}


def load_purchases(con, strategy) -> pd.DataFrame:
    """One row per Form 4 filing that contains open-market purchases (code P), after filters."""
    f = strategy.get("filters", {}) or {}
    doc_types = f.get("document_types", ["4"])
    df = con.execute("""
        WITH buys AS (
            SELECT accession,
                   sum(shares * price) AS value,
                   sum(shares) AS shares,
                   min(trans_date) AS first_trade,
                   max(trans_date) AS last_trade
            FROM form4_trans
            WHERE trans_code = 'P' AND acq_disp = 'A' AND price > 0 AND shares > 0
            GROUP BY accession
        ), own AS (
            SELECT accession,
                   min(owner_cik) AS insider_id,
                   string_agg(DISTINCT owner_name, '; ') AS owner_names,
                   string_agg(DISTINCT relationship, ',') AS relationship,
                   string_agg(DISTINCT title, '; ') AS titles
            FROM form4_owners GROUP BY accession
        )
        SELECT f.accession, f.filing_date, f.issuer_cik, f.issuer_name, f.ticker AS raw_ticker,
               f.document_type, f.aff10b5one,
               b.value, b.shares, b.value / b.shares AS avg_price, b.first_trade, b.last_trade,
               o.insider_id, o.owner_names, o.relationship, o.titles
        FROM form4_filings f
        JOIN buys b USING (accession)
        LEFT JOIN own o USING (accession)
        WHERE list_contains(?, f.document_type)
    """, [list(doc_types)]).df()
    if df.empty:
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["last_trade"] = pd.to_datetime(df["last_trade"])
    df["ticker"] = df["raw_ticker"].map(clean_ticker)
    df = df[df["ticker"].notna() & df["insider_id"].notna()]

    df = df[df["avg_price"] >= float(f.get("min_price", 0))]
    df = df[df["value"] >= float(f.get("min_filing_value_usd", 0))]
    delay = (df["filing_date"] - df["last_trade"]).dt.days
    df = df[delay.isna() | (delay <= int(f.get("max_report_delay_days", 10_000)))]
    if f.get("exclude_10b5_1", False):
        df = df[~df["aff10b5one"].fillna("").astype(str).str.upper().isin(TRUTHY)]
    allowed = {r.lower() for r in f.get("roles", [])}
    if allowed:
        df = df[df["relationship"].map(lambda r: bool(_roles(r) & allowed))]
    return df.reset_index(drop=True)


def _join_unique(values, limit: int = 6) -> str:
    seen = list(dict.fromkeys(v for v in values if isinstance(v, str) and v))
    extra = len(seen) - limit
    return "; ".join(seen[:limit]) + (f" (+{extra} more)" if extra > 0 else "")


def _role_text(rels) -> str:
    out: set[str] = set()
    for r in rels:
        out |= _roles(r)
    return ",".join(sorted(out))


def build_cluster_signals(con, strategy) -> pd.DataFrame:
    p = load_purchases(con, strategy)
    window = pd.Timedelta(days=int(strategy.get("signal.window_days", 14)))
    min_insiders = int(strategy.get("signal.min_insiders", 2))
    cooldown = pd.Timedelta(days=int(strategy.get("signal.cooldown_days", 90)))
    rows = []
    for cik, g in p.sort_values("filing_date").groupby("issuer_cik", sort=False):
        if g["insider_id"].nunique() < min_insiders:
            continue
        dates = g["filing_date"].values
        last_signal = None
        for d in np.unique(dates):
            d = pd.Timestamp(d)
            if last_signal is not None and d - last_signal <= cooldown:
                continue
            w = g[(g["filing_date"] > d - window) & (g["filing_date"] <= d)]
            n = w["insider_id"].nunique()
            if n >= min_insiders:
                last_signal = d
                rows.append({
                    "signal_id": f"{strategy.name}:{cik}:{d.date()}",
                    "side": "buy",
                    "issuer_cik": cik,
                    "issuer_name": w["issuer_name"].iloc[-1],
                    "ticker": w["ticker"].iloc[-1],
                    "known_date": d,
                    "n_insiders": n,
                    "total_value": float(w["value"].sum()),
                    "avg_price": float((w["value"].sum()) / w["shares"].sum()),
                    "insiders": _join_unique(w["owner_names"]),
                    "roles": _role_text(w["relationship"]),
                    "first_trade": w["last_trade"].min(),
                })
    return pd.DataFrame(rows, columns=_COLUMNS)


def build_large_signals(con, strategy) -> pd.DataFrame:
    p = load_purchases(con, strategy)
    min_value = float(strategy.get("signal.min_value_usd", 100_000))
    cooldown = pd.Timedelta(days=int(strategy.get("signal.cooldown_days", 90)))
    big = p[p["value"] >= min_value].sort_values("filing_date")
    rows = []
    for cik, g in big.groupby("issuer_cik", sort=False):
        last_signal = None
        for d, w in g.groupby("filing_date", sort=True):
            if last_signal is not None and d - last_signal <= cooldown:
                continue
            last_signal = d
            rows.append({
                "signal_id": f"{strategy.name}:{cik}:{d.date()}",
                "side": "buy",
                "issuer_cik": cik,
                "issuer_name": w["issuer_name"].iloc[-1],
                "ticker": w["ticker"].iloc[-1],
                "known_date": d,
                "n_insiders": w["insider_id"].nunique(),
                "total_value": float(w["value"].sum()),
                "avg_price": float(w["value"].sum() / w["shares"].sum()),
                "insiders": _join_unique(w["owner_names"]),
                "roles": _role_text(w["relationship"]),
                "first_trade": w["last_trade"].min(),
            })
    return pd.DataFrame(rows, columns=_COLUMNS)


_COLUMNS = ["signal_id", "side", "issuer_cik", "issuer_name", "ticker", "known_date", "n_insiders",
            "total_value", "avg_price", "insiders", "roles", "first_trade"]
