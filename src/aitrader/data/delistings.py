"""Why did a stock stop trading? Classify from the company's own SEC filings.

For every issuer whose price series ends early (or never had prices) we read the EDGAR
submission history (https://data.sec.gov/submissions/CIK##########.json) and look at
filings around the last trading day:

  bankruptcy         8-K item 1.03 (bankruptcy or receivership)
  acquired           merger proxy (DEFM14A/C), tender offer (SC TO-T, SC 14D9),
                     or 8-K item 2.01 together with a delisting/deregistration form
  delisted_distress  8-K item 3.01 (failure to meet listing rules) without merger evidence
  deregistered       Form 25 / Form 15 without any of the above
  still_filing       10-K/10-Q filed well after the last price: probably a ticker change
  unknown            nothing conclusive

The backtest turns the classification into a haircut on the last price (config
[delisting.haircut]).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
FILES_URL = "https://data.sec.gov/submissions/{name}"

MERGER_FORMS = {"DEFM14A", "DEFM14C", "SC TO-T", "SC 14D9", "SC TO-T/A", "SC 14D9/A", "PREM14A"}
DEREG_FORMS = {"25", "25-NSE", "15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G"}
REPORT_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}


def _filings_frame(block: dict) -> pd.DataFrame:
    n = len(block.get("form", []))
    return pd.DataFrame({
        "form": block.get("form", []),
        "date": pd.to_datetime(block.get("filingDate", [])),
        "items": block.get("items", [""] * n) or [""] * n,
    })


def fetch_filings(client, cik: str, since: date) -> pd.DataFrame:
    resp = client.get(SUBMISSIONS_URL.format(cik=cik))
    if resp.status_code == 404:
        return pd.DataFrame(columns=["form", "date", "items"])
    data = resp.json()
    frames = [_filings_frame(data.get("filings", {}).get("recent", {}))]
    # Older filings live in extra pages; fetch only those that overlap the period we need.
    for f in data.get("filings", {}).get("files", []):
        if f.get("filingTo") and pd.Timestamp(f["filingTo"]).date() >= since:
            r = client.get(FILES_URL.format(name=f["name"]))
            if r.status_code == 200:
                frames.append(_filings_frame(r.json()))
    return pd.concat(frames, ignore_index=True)


def classify(filings: pd.DataFrame, last_date: date | None) -> tuple[str, str | None, date | None]:
    if filings.empty:
        return "unknown", None, None
    if last_date is None:  # never had prices: look at the whole history after the signals
        lo, hi = pd.Timestamp("1990-01-01"), pd.Timestamp("2100-01-01")
        ref = None
    else:
        ref = pd.Timestamp(last_date)
        lo, hi = ref - timedelta(days=180), ref + timedelta(days=365)
    win = filings[(filings["date"] >= lo) & (filings["date"] <= hi)]
    items = win["items"].fillna("").astype(str)
    is8k = win["form"].str.startswith("8-K")

    def first(mask):
        hit = win[mask].sort_values("date")
        if hit.empty:
            return None
        return hit.iloc[0]["form"], hit.iloc[0]["date"].date()

    if (hit := first(is8k & items.str.contains("1.03", regex=False))):
        return "bankruptcy", *hit
    dereg = win["form"].isin(DEREG_FORMS)
    if (hit := first(win["form"].isin(MERGER_FORMS))) or \
       (dereg.any() and (hit := first(is8k & items.str.contains("2.01", regex=False)))):
        return "acquired", *hit
    if (hit := first(is8k & items.str.contains("3.01", regex=False))):
        return "delisted_distress", *hit
    if (hit := first(dereg)):
        return "deregistered", *hit
    if ref is not None:
        later = filings[filings["form"].isin(REPORT_FORMS) & (filings["date"] > ref + timedelta(days=120))]
        if not later.empty:
            return "still_filing", later.iloc[0]["form"], later["date"].min().date()
    return "unknown", None, None


def candidates(con, settings, pairs: pd.DataFrame) -> pd.DataFrame:
    """Issuers from `pairs` (issuer_cik, ticker) whose prices end early or are missing."""
    stale = int(settings.get("delisting.stale_days", 10))
    bench = settings.get("portfolio.benchmark", "SPY")
    con.register("_pairs", pairs[["issuer_cik", "ticker"]].drop_duplicates())
    try:
        df = con.execute(f"""
            WITH b AS (SELECT max(date) AS bmax FROM prices WHERE ticker = ?),
            m AS (
                SELECT p.issuer_cik,
                       max(ps.last_date) AS last_date
                FROM _pairs p
                LEFT JOIN ticker_map tm ON tm.issuer_cik = p.issuer_cik AND tm.signal_ticker = p.ticker
                LEFT JOIN price_status ps ON ps.ticker = tm.price_ticker AND ps.status = 'ok'
                GROUP BY p.issuer_cik
            )
            SELECT m.issuer_cik AS cik, m.last_date
            FROM m, b
            WHERE m.last_date IS NULL OR m.last_date < b.bmax - INTERVAL {stale * 7 // 5 + 2} DAY
        """, [bench]).df()
    finally:
        con.unregister("_pairs")
    return df


def download(con, client, settings, pairs: pd.DataFrame, refresh: bool = False, log=print) -> None:
    cand = candidates(con, settings, pairs)
    done = {r[0] for r in con.execute("SELECT cik FROM delistings").fetchall()}
    todo = cand if refresh else cand[~cand["cik"].isin(done)]
    log(f"delistings: {len(cand):,} companies stop trading early or have no prices; checking {len(todo):,}")
    rows = []
    for n, (cik, last) in enumerate(zip(todo["cik"], todo["last_date"]), 1):
        last_d = pd.Timestamp(last).date() if pd.notna(last) else None
        since = (last_d or date(2006, 1, 1)) - timedelta(days=200)
        cls, form, ev_date = classify(fetch_filings(client, cik, since), last_d)
        rows.append((cik, last_d, cls, form, ev_date, datetime.now()))
        if n % 100 == 0:
            log(f"  {n:,}/{len(todo):,}")
            _flush(con, rows)
            rows = []
    _flush(con, rows)
    summary = con.execute("SELECT classification, count(*) n FROM delistings GROUP BY 1 ORDER BY 2 DESC").df()
    log("delistings summary:\n" + summary.to_string(index=False))


def _flush(con, rows) -> None:
    if not rows:
        return
    from ..db import insert_df
    df = pd.DataFrame(rows, columns=["cik", "last_price_date", "classification", "evidence_form", "evidence_date", "checked_at"])
    insert_df(con, "delistings", df, replace=True)
