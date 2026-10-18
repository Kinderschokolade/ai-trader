"""Local DuckDB database. One file holds all downloaded data and caches."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS form4_filings (
    accession VARCHAR PRIMARY KEY,
    filing_date DATE,
    period_of_report DATE,
    document_type VARCHAR,
    issuer_cik VARCHAR,
    issuer_name VARCHAR,
    ticker VARCHAR,
    aff10b5one VARCHAR,
    quarter VARCHAR
);
CREATE TABLE IF NOT EXISTS form4_owners (
    accession VARCHAR,
    owner_cik VARCHAR,
    owner_name VARCHAR,
    relationship VARCHAR,
    title VARCHAR
);
CREATE TABLE IF NOT EXISTS form4_trans (
    accession VARCHAR,
    trans_sk VARCHAR,
    security_title VARCHAR,
    trans_date DATE,
    trans_code VARCHAR,
    acq_disp VARCHAR,
    shares DOUBLE,
    price DOUBLE,
    shares_after DOUBLE,
    direct_indirect VARCHAR
);
CREATE TABLE IF NOT EXISTS loaded_quarters (
    source VARCHAR,
    quarter VARCHAR,
    loaded_at TIMESTAMP,
    n_filings INTEGER,
    PRIMARY KEY (source, quarter)
);
CREATE TABLE IF NOT EXISTS prices (
    ticker VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS price_status (
    ticker VARCHAR PRIMARY KEY,
    source VARCHAR,
    first_date DATE,
    last_date DATE,
    n_rows INTEGER,
    fetched_at TIMESTAMP,
    status VARCHAR
);
CREATE TABLE IF NOT EXISTS ticker_map (
    issuer_cik VARCHAR,
    signal_ticker VARCHAR,
    price_ticker VARCHAR,
    PRIMARY KEY (issuer_cik, signal_ticker)
);
CREATE TABLE IF NOT EXISTS shares_outstanding (
    cik VARCHAR,
    end_date DATE,
    filed_date DATE,
    shares DOUBLE,
    concept VARCHAR
);
CREATE TABLE IF NOT EXISTS fundamentals_status (
    cik VARCHAR PRIMARY KEY,
    fetched_at TIMESTAMP,
    status VARCHAR
);
CREATE TABLE IF NOT EXISTS delistings (
    cik VARCHAR PRIMARY KEY,
    last_price_date DATE,
    classification VARCHAR,
    evidence_form VARCHAR,
    evidence_date DATE,
    checked_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key VARCHAR PRIMARY KEY,
    model VARCHAR,
    response VARCHAR,
    created_at TIMESTAMP
);
"""


def connect(path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if read_only and not path.exists():
        raise FileNotFoundError(f"Database {path} does not exist yet. Run `aitrader download ...` first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def insert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, replace: bool = False) -> int:
    """Insert a DataFrame whose columns match (a subset of) the table's columns."""
    if df is None or df.empty:
        return 0
    cols = ", ".join(df.columns)
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    con.register("_insert_df", df)
    try:
        con.execute(f"{verb} INTO {table} ({cols}) SELECT {cols} FROM _insert_df")
    finally:
        con.unregister("_insert_df")
    return len(df)


def table_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()]
    rows = [(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in sorted(tables)]
    return pd.DataFrame(rows, columns=["table", "rows"])
