"""Shares outstanding from SEC XBRL company facts -> point-in-time market cap.

https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
Each value has the date the count refers to (`end`) and the date it was filed (`filed`).
For a signal on day d we only use values filed on or before d (no look-ahead).
XBRL starts around 2009-2011, so earlier signals fall into the "unknown" size bucket
unless `size_metric = "dollar_volume"` is used.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..db import insert_df

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"
CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]


def parse_facts(cik: str, data: dict) -> pd.DataFrame:
    rows = []
    facts = data.get("facts", {})
    for ns, concept in CONCEPTS:
        units = facts.get(ns, {}).get(concept, {}).get("units", {})
        for item in units.get("shares", []):
            if item.get("val") and item.get("end") and item.get("filed"):
                rows.append((cik, item["end"], item["filed"], float(item["val"]), concept))
    df = pd.DataFrame(rows, columns=["cik", "end_date", "filed_date", "shares", "concept"])
    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
        df["filed_date"] = pd.to_datetime(df["filed_date"]).dt.date
        df = df.drop_duplicates(["end_date", "filed_date", "concept"])
    return df


def download(con, client, ciks, refresh: bool = False, log=print) -> None:
    ciks = sorted({str(c).lstrip("0") for c in ciks if c})
    done = {r[0] for r in con.execute("SELECT cik FROM fundamentals_status").fetchall()}
    todo = [c for c in ciks if refresh or c not in done]
    log(f"fundamentals: {len(todo):,} companies to fetch ({len(ciks) - len(todo):,} cached)")
    for n, cik in enumerate(todo, 1):
        resp = client.get(FACTS_URL.format(cik=cik))
        if resp.status_code == 404:
            status, df = "no_data", None
        else:
            df = parse_facts(cik, resp.json())
            status = "ok" if not df.empty else "no_shares"
        con.execute("DELETE FROM shares_outstanding WHERE cik = ?", [cik])
        if df is not None and not df.empty:
            insert_df(con, "shares_outstanding", df)
        con.execute("INSERT OR REPLACE INTO fundamentals_status VALUES (?, ?, ?)", [cik, datetime.now(), status])
        if n % 250 == 0:
            log(f"  {n:,}/{len(todo):,}")


def shares_asof(con, ciks, dates) -> pd.Series:
    """Latest shares outstanding filed on or before each (cik, date). Returns aligned Series."""
    q = pd.DataFrame({"cik": [str(c) for c in ciks], "d": pd.to_datetime(list(dates)).date})
    q["_i"] = range(len(q))
    con.register("_q", q)
    try:
        res = con.execute("""
            SELECT q._i, arg_max(s.shares, s.filed_date) AS shares
            FROM _q q JOIN shares_outstanding s ON s.cik = q.cik AND s.filed_date <= q.d
                 AND s.end_date >= q.d - INTERVAL 18 MONTH
            GROUP BY q._i
        """).df()
    finally:
        con.unregister("_q")
    out = pd.Series(float("nan"), index=range(len(q)))
    if not res.empty:
        out.loc[res["_i"].values] = res["shares"].values
    return out
