"""Unit tests on small hand-made data (no network)."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from aitrader.backtest.engine import Engine
from aitrader.backtest.metrics import compute
from aitrader.config import Settings, Strategy, parse_override
from aitrader.data import delistings, form4
from aitrader.data.prices import PriceStore, clean_ticker
from aitrader.db import connect, insert_df
from aitrader.signals.insider import build_cluster_signals, build_large_signals

GLOBAL = {
    "portfolio": {"benchmark": "SPY", "initial_capital": 1000},
    "execution": {"execution_price": "close", "entry_lag_days": 1, "fractional_shares": True,
                  "slippage_bps": 0, "commission_per_share": 0, "commission_min": 0,
                  "commission_max_pct": 1, "min_trade_value": 0},
    "delisting": {"stale_days": 2, "haircut": {"bankruptcy": 1.0, "acquired": 0.0, "unknown": 0.5, "unchecked": 0.5}},
}


def strat(**over):
    raw = {"strategy": {"name": "t", "type": "insider_cluster", "status": "active", "allocation": 1.0},
           "sizing": {"mode": "percent_of_equity", "value": 0.5, "max_positions": 0, "rank_by": "total_value"},
           "exit": {"holding_days": 3, "forced_exit": "oldest_first"},
           "signal": {"window_days": 14, "min_insiders": 2, "cooldown_days": 30},
           "filters": {"document_types": ["4"], "roles": ["director", "officer"], "min_price": 1}}
    s = Strategy("t", raw)
    return s.with_overrides(over, variant="base") if over else s


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "t.duckdb")
    cal = pd.bdate_range("2020-01-01", periods=20)
    rows = []
    for i, d in enumerate(cal):
        rows.append(("SPY", d.date(), 100, 100, 100, 100, 100, 1))
        rows.append(("AAA", d.date(), 10 + i, 10 + i, 10 + i, 10 + i, 10 + i, 1))      # rises 1/day
        rows.append(("BBB", d.date(), 20, 20, 20, 20, 20, 1))                           # flat
        if i <= 8:
            rows.append(("CCC", d.date(), 50, 50, 50, 50, 50, 1))                       # stops on day 8
    insert_df(c, "prices", pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]))
    return c


def settings():
    return Settings(home=None, raw=GLOBAL)


def store(con):
    return PriceStore.from_benchmark(con, "SPY", "2020-01-01", "2020-12-31", stale_days=2)


def sig(rows):
    return pd.DataFrame(rows, columns=["signal_id", "side", "issuer_cik", "price_ticker", "known_date", "total_value"])


def test_engine_accounting_and_holding_period(con):
    st = store(con)
    cal = st.calendar
    s = sig([("a", "buy", "1", "AAA", cal[0], 1.0)])
    res = Engine(settings(), strat(), st).run(s, cal[0], cal[-1], 1000)
    t = res.trades
    assert len(t) == 1 and t.iloc[0]["reason"] == "holding_period"
    # entry at close of day 1 (price 11), exit 3 days later (price 14), 500 invested
    assert t.iloc[0]["entry_price"] == pytest.approx(11)
    assert t.iloc[0]["exit_price"] == pytest.approx(14)
    assert res.equity["equity"].iloc[-1] == pytest.approx(1000 + 500 * (14 / 11 - 1))


def test_forced_exit_sells_oldest(con):
    st = store(con)
    cal = st.calendar
    s = sig([("a", "buy", "1", "AAA", cal[0], 1.0), ("b", "buy", "2", "BBB", cal[1], 1.0),
             ("c", "buy", "3", "CCC", cal[1], 0.5)])
    res = Engine(settings(), strat(**{"sizing.value": 0.6, "exit.holding_days": 0}), st).run(s, cal[0], cal[-1], 1000)
    t = res.trades.set_index("signal_id")
    assert t.loc["a", "reason"] == "forced_oldest"      # AAA sold to make room for BBB
    assert t.loc["b", "reason"] == "open"                # bought today: never force-sold the same day
    assert "c" not in t.index and res.stats["skipped_no_cash"] == 1  # no partial positions


def test_delisting_haircut(con):
    st = store(con)
    cal = st.calendar
    s = sig([("c", "buy", "9", "CCC", cal[0], 1.0)])
    eng = Engine(settings(), strat(**{"exit.holding_days": 0}), st, delist_classes={"9": "bankruptcy"})
    res = eng.run(s, cal[0], cal[-1], 1000)
    t = res.trades.iloc[0]
    assert t["reason"] == "delisted:bankruptcy" and t["proceeds"] == pytest.approx(0)
    eng = Engine(settings(), strat(**{"exit.holding_days": 0}), st, delist_classes={"9": "acquired"})
    t = eng.run(s, cal[0], cal[-1], 1000).trades.iloc[0]
    assert t["reason"] == "delisted:acquired" and t["return"] == pytest.approx(0)


def test_no_price_is_skipped_and_counted(con):
    st = store(con)
    cal = st.calendar
    s = sig([("x", "buy", "5", "ZZZ", cal[0], 1.0)])
    res = Engine(settings(), strat(), st).run(s, cal[0], cal[-1], 1000)
    assert res.stats["skipped_no_price"] == 1
    m = compute(res)
    assert m["total_return"] == pytest.approx(0)


def _form4_rows(c, items):
    f, o, t = [], [], []
    for k, (cik, day, owner, value, role) in enumerate(items):
        a = f"acc{k}"
        f.append((a, pd.Timestamp(day).date(), "4", cik, f"Co{cik}", f"T{cik}", "0", "2020q1"))
        o.append((a, owner, f"Name{owner}", role, ""))
        t.append((a, str(k), "Common", pd.Timestamp(day).date(), "P", "A", value / 10, 10.0))
    insert_df(c, "form4_filings", pd.DataFrame(f, columns=["accession", "filing_date", "document_type", "issuer_cik", "issuer_name", "ticker", "aff10b5one", "quarter"]))
    insert_df(c, "form4_owners", pd.DataFrame(o, columns=["accession", "owner_cik", "owner_name", "relationship", "title"]))
    insert_df(c, "form4_trans", pd.DataFrame(t, columns=["accession", "trans_sk", "security_title", "trans_date", "trans_code", "acq_disp", "shares", "price"]))


def test_cluster_and_large_signals(tmp_path):
    c = connect(tmp_path / "s.duckdb")
    _form4_rows(c, [
        ("1", "2020-01-01", "a", 50_000, "Director"),
        ("1", "2020-01-10", "b", 50_000, "Officer"),      # 2nd insider within 14 days -> signal
        ("1", "2020-01-12", "c", 50_000, "Officer"),      # inside cooldown -> no new signal
        ("2", "2020-01-01", "a", 50_000, "Director"),
        ("2", "2020-02-01", "b", 50_000, "Director"),     # too far apart -> no signal
        ("3", "2020-01-01", "x", 500_000, "TenPercentOwner"),  # role filtered out
        ("3", "2020-01-02", "y", 500_000, "TenPercentOwner"),
        ("4", "2020-03-01", "z", 300_000, "Director,Officer"),  # large single buy
    ])
    cl = build_cluster_signals(c, strat())
    assert list(cl["issuer_cik"]) == ["1"]
    assert cl.iloc[0]["known_date"] == pd.Timestamp("2020-01-10") and cl.iloc[0]["n_insiders"] == 2
    lg = build_large_signals(c, strat(**{"strategy.type": "insider_large", "signal.min_value_usd": 100_000}))
    assert list(lg["issuer_cik"]) == ["4"]


def test_form4_parser_roundtrip(tmp_path):
    sub = "ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\n" \
          "0001-20-1\t15-JAN-2020\t13-JAN-2020\t4\t0000123\tACME\tacme\n" \
          "0001-20-2\t15-JAN-2020\t13-JAN-2020\t3\t0000123\tACME\tACME\n"
    own = "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n0001-20-1\t0000999\tDoe\tDirector\t\n"
    trn = "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_ACQUIRED_DISP_CD\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tSHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n" \
          "0001-20-1\t1\tCommon\t13-JAN-2020\tP\tA\t100\t9.5\t1100\tD\n0001-20-1\t2\tCommon\t13-JAN-2020\tM\tA\t5\t0\t1105\tD\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("REPORTINGOWNER.tsv", own)
        z.writestr("NONDERIV_TRANS.tsv", trn)
    tables = form4.parse_dataset(buf.getvalue(), "2020q1")
    assert list(tables["filings"]["accession"]) == ["0001-20-1"]
    assert tables["filings"].iloc[0]["issuer_cik"] == "123" and tables["filings"].iloc[0]["ticker"] == "ACME"
    assert list(tables["trans"]["trans_code"]) == ["P"]
    c = connect(tmp_path / "f.duckdb")
    form4.store_quarter(c, "2020q1", tables)
    form4.store_quarter(c, "2020q1", tables)  # idempotent
    assert c.execute("SELECT count(*) FROM form4_trans").fetchone()[0] == 1


def test_delisting_classification():
    f = pd.DataFrame({"form": ["10-Q", "8-K", "25-NSE"], "date": pd.to_datetime(["2019-11-01", "2020-01-05", "2020-01-20"]),
                      "items": ["", "1.03,9.01", ""]})
    assert delistings.classify(f, pd.Timestamp("2020-01-10").date())[0] == "bankruptcy"
    f2 = pd.DataFrame({"form": ["DEFM14A", "15-12B"], "date": pd.to_datetime(["2019-12-01", "2020-02-01"]), "items": ["", ""]})
    assert delistings.classify(f2, pd.Timestamp("2020-01-10").date())[0] == "acquired"
    f3 = pd.DataFrame({"form": ["10-Q", "10-K"], "date": pd.to_datetime(["2020-05-01", "2021-03-01"]), "items": ["", ""]})
    assert delistings.classify(f3, pd.Timestamp("2020-01-10").date())[0] == "still_filing"


def test_helpers():
    assert clean_ticker("nyse: brk.b") == "BRK.B"
    assert clean_ticker("ABC, ABCD") == "ABC"
    assert clean_ticker("none") is None
    assert parse_override("exit.holding_days=63") == ("exit.holding_days", 63)
    assert parse_override("filters.size_buckets=['micro','small']") == ("filters.size_buckets", ["micro", "small"])
    v = strat(**{"sweep": None}).raw
    s = Strategy("t", {**strat().raw, "sweep": {"exit.holding_days": [1, 2], "sizing.value": [0.1, 0.2]}})
    assert len(s.variants()) == 4
    assert form4.quarter_range("2006-02-01", "2006-12-31") == ["2006q1", "2006q2", "2006q3", "2006q4"]


def test_ensure_prices_remaps_to_current_sec_ticker(tmp_path, monkeypatch):
    from aitrader.data import prices as P

    class FakeSource:
        name = "fake"

        def fetch(self, tickers, start, end):
            rows = [(t, pd.Timestamp("2020-01-02").date(), 1, 1, 1, 1, 1, 1) for t in tickers if t in {"SPY", "AAA", "NEWT"}]
            return pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

    class FakeResp:
        def json(self):
            return {"0": {"cik_str": 2, "ticker": "NEWT", "title": "x"}}

    class FakeClient:
        def get(self, url):
            return FakeResp()

    monkeypatch.setattr(P, "get_source", lambda name: FakeSource())
    c = connect(tmp_path / "p.duckdb")
    s = Settings(home=tmp_path, raw={"data": {"start": "2020-01-01", "end": "2020-02-01"}, "portfolio": {"benchmark": "SPY"}})
    pairs = pd.DataFrame({"issuer_cik": ["1", "2", "3"], "ticker": ["AAA", "OLDT", "GONE"]})
    P.ensure_prices(c, s, pairs, client=FakeClient(), log=lambda *a: None)
    tm = dict(c.execute("SELECT signal_ticker, price_ticker FROM ticker_map").fetchall())
    assert tm == {"AAA": "AAA", "OLDT": "NEWT"}
    st = dict(c.execute("SELECT ticker, status FROM price_status").fetchall())
    assert st["GONE"] == "no_data" and st["SPY"] == "ok"
