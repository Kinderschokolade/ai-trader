"""SEC Form 4 insider transactions from the quarterly "Insider Transactions Data Sets".

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
One zip per quarter (from 2006 Q1) with tab-separated tables. We keep the filing header,
the reporting owners and the non-derivative transactions with codes P (open-market
purchase) and S (open-market sale).

Note for live use: the data sets are published after each quarter ends. For daily signals
the live module (phase 2) parses new Form 4 XML filings from the EDGAR daily index into the
same tables.
"""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..db import insert_df

DATASET_URL = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
)
KEEP_CODES = ("P", "S")


def quarter_of(d: date) -> str:
    return f"{d.year}q{(d.month - 1) // 3 + 1}"


def quarter_range(start: str, end: str) -> list[str]:
    """'2006-01-01', '2007-05-01' -> ['2006q1', ..., '2007q2']"""
    s = datetime.fromisoformat(start).date()
    e = datetime.fromisoformat(end).date()
    out, y, q = [], s.year, (s.month - 1) // 3 + 1
    while (y, q) <= (e.year, (e.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def parse_dates(s: pd.Series) -> pd.Series:
    """Data set dates look like '31-MAR-2023'; fall back to pandas' parser otherwise."""
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    if out.notna().sum() < 0.5 * s.notna().sum():
        out = pd.to_datetime(s, errors="coerce")
    return out.dt.date


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _read_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    members = [m for m in zf.namelist() if m.upper().endswith(name.upper())]
    if not members:
        raise KeyError(f"{name} not found in zip (members: {zf.namelist()})")
    with zf.open(members[0]) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        df = pd.read_csv(text, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                         na_values=[""], low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def parse_dataset(zip_bytes: bytes, quarter: str) -> dict[str, pd.DataFrame]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    sub = _read_member(zf, "SUBMISSION.tsv")
    own = _read_member(zf, "REPORTINGOWNER.tsv")
    trn = _read_member(zf, "NONDERIV_TRANS.tsv")

    filings = pd.DataFrame({
        "accession": sub["ACCESSION_NUMBER"],
        "filing_date": parse_dates(_col(sub, "FILING_DATE")),
        "period_of_report": parse_dates(_col(sub, "PERIOD_OF_REPORT")),
        "document_type": _col(sub, "DOCUMENT_TYPE"),
        "issuer_cik": _col(sub, "ISSUERCIK").str.lstrip("0"),
        "issuer_name": _col(sub, "ISSUERNAME"),
        "ticker": _col(sub, "ISSUERTRADINGSYMBOL").str.strip().str.upper(),
        "aff10b5one": _col(sub, "AFF10B5ONE"),
        "quarter": quarter,
    })
    filings = filings[filings["document_type"].isin(["4", "4/A"])]
    filings = filings.drop_duplicates("accession")
    keep = set(filings["accession"])

    owners = pd.DataFrame({
        "accession": own["ACCESSION_NUMBER"],
        "owner_cik": _col(own, "RPTOWNERCIK").str.lstrip("0"),
        "owner_name": _col(own, "RPTOWNERNAME"),
        "relationship": _col(own, "RPTOWNER_RELATIONSHIP"),
        "title": _col(own, "RPTOWNER_TITLE"),
    })
    owners = owners[owners["accession"].isin(keep)]

    trans = pd.DataFrame({
        "accession": trn["ACCESSION_NUMBER"],
        "trans_sk": _col(trn, "NONDERIV_TRANS_SK"),
        "security_title": _col(trn, "SECURITY_TITLE"),
        "trans_date": parse_dates(_col(trn, "TRANS_DATE")),
        "trans_code": _col(trn, "TRANS_CODE").str.strip().str.upper(),
        "acq_disp": _col(trn, "TRANS_ACQUIRED_DISP_CD").str.strip().str.upper(),
        "shares": _num(_col(trn, "TRANS_SHARES")),
        "price": _num(_col(trn, "TRANS_PRICEPERSHARE")),
        "shares_after": _num(_col(trn, "SHRS_OWND_FOLWNG_TRANS")),
        "direct_indirect": _col(trn, "DIRECT_INDIRECT_OWNERSHIP"),
    })
    trans = trans[trans["accession"].isin(keep) & trans["trans_code"].isin(KEEP_CODES)]
    # Only keep filings that actually contain a P/S transaction.
    used = set(trans["accession"])
    filings = filings[filings["accession"].isin(used)]
    owners = owners[owners["accession"].isin(used)]
    return {"filings": filings, "owners": owners, "trans": trans}


def store_quarter(con, quarter: str, tables: dict[str, pd.DataFrame]) -> int:
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM form4_owners WHERE accession IN (SELECT accession FROM form4_filings WHERE quarter = ?)", [quarter])
        con.execute("DELETE FROM form4_trans WHERE accession IN (SELECT accession FROM form4_filings WHERE quarter = ?)", [quarter])
        con.execute("DELETE FROM form4_filings WHERE quarter = ?", [quarter])
        # The same accession can (rarely) appear in two quarterly files; the later load wins.
        con.register("_new_acc", tables["filings"][["accession"]])
        for t in ("form4_owners", "form4_trans", "form4_filings"):
            con.execute(f"DELETE FROM {t} WHERE accession IN (SELECT accession FROM _new_acc)")
        con.unregister("_new_acc")
        insert_df(con, "form4_filings", tables["filings"], replace=True)
        insert_df(con, "form4_owners", tables["owners"])
        insert_df(con, "form4_trans", tables["trans"])
        con.execute("INSERT OR REPLACE INTO loaded_quarters VALUES ('form4', ?, now(), ?)",
                    [quarter, len(tables["filings"])])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(tables["filings"])


def download(con, client, settings, start: str | None = None, end: str | None = None,
             force: bool = False, log=print) -> None:
    start = start or settings.get("data.start")
    end = end or settings.data_end
    cache = settings.path("raw_cache") / "form4"
    cache.mkdir(parents=True, exist_ok=True)
    keep_raw = bool(settings.get("paths.keep_raw", False))
    loaded = {r[0] for r in con.execute("SELECT quarter FROM loaded_quarters WHERE source='form4'").fetchall()}
    current = quarter_of(date.today())
    for q in quarter_range(start, end):
        if q in loaded and not force and q != current:
            continue
        zpath = cache / f"{q}_form345.zip"
        if zpath.exists() and not force:
            data = zpath.read_bytes()
        else:
            resp = client.get(DATASET_URL.format(q=q))
            if resp.status_code == 404:
                log(f"form4 {q}: not published yet")
                continue
            data = resp.content
            if keep_raw:
                zpath.write_bytes(data)
        tables = parse_dataset(data, q)
        n = store_quarter(con, q, tables)
        n_buy = int((tables["trans"]["trans_code"] == "P").sum())
        log(f"form4 {q}: {n:,} filings with P/S trades ({n_buy:,} purchase lines)")
        if zpath.exists() and not keep_raw:
            zpath.unlink()
