"""Daily portfolio simulation for one strategy pot.

Rules (all configurable in the strategy TOML):
  * Every new lead gets the same size: a percentage of the pot's current value
    ([sizing] mode = "percent_of_equity") or a fixed USD amount.
  * A position is sold after [exit] holding_days trading days.
  * If there is not enough cash for a new lead, the OLDEST positions are sold early
    ([exit] forced_exit = "oldest_first") until the lead can be bought.
  * If a held stock stops trading, the position is closed at its last price minus a
    haircut that depends on why it stopped (bankruptcy, acquisition, ...).
  * Trades happen at the open (or close) of the entry day, with slippage and IBKR-like
    commissions. Order within a day: delistings, time exits, source sells, then buys.

The same engine will later drive the paper account: the live module produces the same
signal rows and replaces simulated fills by IBKR orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Position:
    signal_id: str
    ticker: str
    issuer_cik: str
    entry_idx: int
    entry_price: float
    shares: float
    cost: float


@dataclass
class BacktestResult:
    label: str
    equity: pd.DataFrame       # date, cash, invested, equity, n_positions, benchmark
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


class Engine:
    def __init__(self, settings, strategy, store, delist_classes: dict[str, str] | None = None):
        g, s = settings, strategy
        self.store = store
        self.exec_price = g.get("execution.execution_price", "open")
        self.lag = int(g.get("execution.entry_lag_days", 1))
        self.fractional = bool(g.get("execution.fractional_shares", True))
        self.slip = float(g.get("execution.slippage_bps", 0)) / 1e4
        self.c_share = float(g.get("execution.commission_per_share", 0))
        self.c_min = float(g.get("execution.commission_min", 0))
        self.c_max = float(g.get("execution.commission_max_pct", 1))
        self.min_trade = float(g.get("execution.min_trade_value", 0))
        self.haircut = {k: float(v) for k, v in (g.get("delisting.haircut", {}) or {}).items()}
        self.benchmark = g.get("portfolio.benchmark", "SPY")
        self.size_mode = s.get("sizing.mode", "percent_of_equity")
        self.size_value = float(s.get("sizing.value", 0.04))
        self.max_pos = int(s.get("sizing.max_positions", 0) or 0)
        self.allow_add = bool(s.get("sizing.allow_add_to_existing", False))
        self.allow_partial = bool(s.get("sizing.allow_partial", False))
        self.rank_by = s.get("sizing.rank_by", "total_value")
        self.hold = int(s.get("exit.holding_days", 0) or 0)
        self.forced = s.get("exit.forced_exit", "oldest_first")
        self.min_hold_forced = int(s.get("exit.min_holding_days_before_forced_exit", 0) or 0)
        self.source_sell = bool(s.get("exit.sell_on_source_sell", False))
        self.delist = delist_classes or {}
        self.label = strategy.label

    # -- prices ---------------------------------------------------------------------
    def _px(self, ser, i: int) -> float:
        j = min(i, ser.last_idx)
        if self.exec_price == "open" and j == i:
            return float(ser.adj_open[j])
        return float(ser.adj_close[j])

    @staticmethod
    def _mark(ser, i: int) -> float:
        return float(ser.adj_close[min(i, ser.last_idx)])

    def _commission(self, shares: float, value: float) -> float:
        if value <= 0:
            return 0.0
        return float(min(max(self.c_share * shares, self.c_min), self.c_max * value))

    # -- main loop ------------------------------------------------------------------
    def run(self, signals: pd.DataFrame, start, end, capital: float) -> BacktestResult:
        cal = self.store.calendar
        i0 = int(cal.searchsorted(pd.Timestamp(start), side="left"))
        i1 = int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1
        if i1 <= i0:
            raise ValueError(f"No trading days between {start} and {end}")

        sig = signals.copy()
        sig["entry_idx"] = [self.store.index_after(d, self.lag) for d in sig["known_date"]]
        sig = sig[sig["entry_idx"].notna()]
        sig["entry_idx"] = sig["entry_idx"].astype(int)
        sig = sig[(sig["entry_idx"] >= i0) & (sig["entry_idx"] <= i1)]
        self.store.preload(sig["price_ticker"].dropna().unique())
        rank = self.rank_by if self.rank_by in sig.columns else None
        buys = {i: g.sort_values(rank, ascending=False) if rank else g
                for i, g in sig[sig["side"] == "buy"].groupby("entry_idx")}
        sells = {i: set(g["price_ticker"]) for i, g in sig[sig["side"] == "sell"].groupby("entry_idx")}

        cash = float(capital)
        positions: list[Position] = []
        trades: list[dict] = []
        skipped: dict[str, int] = {}
        eq_rows = []

        def close(pos: Position, i: int, price: float, reason: str, charge: bool = True):
            nonlocal cash
            px = price * (1 - self.slip) if charge else price
            value = pos.shares * px
            comm = self._commission(pos.shares, value) if charge else 0.0
            cash += value - comm
            positions.remove(pos)
            trades.append({
                "signal_id": pos.signal_id, "ticker": pos.ticker, "issuer_cik": pos.issuer_cik,
                "entry_date": cal[pos.entry_idx].date(), "entry_price": pos.entry_price,
                "exit_date": cal[i].date(), "exit_price": px, "shares": pos.shares,
                "cost": pos.cost, "proceeds": value - comm, "pnl": value - comm - pos.cost,
                "return": (value - comm) / pos.cost - 1 if pos.cost else np.nan,
                "holding_days": i - pos.entry_idx, "reason": reason,
            })

        for i in range(i0, i1 + 1):
            # 1) held stocks that stopped trading
            for pos in list(positions):
                ser = self.store.get(pos.ticker)
                if ser.ended and i > ser.last_idx:
                    cls = self.delist.get(str(pos.issuer_cik), "unchecked")
                    cut = self.haircut.get(cls, self.haircut.get("unknown", 0.5))
                    px = float(ser.adj_close[ser.last_idx]) * (1 - cut)
                    close(pos, i, px, f"delisted:{cls}", charge=px > 0)
            # 2) holding period over
            if self.hold > 0:
                for pos in list(positions):
                    if i - pos.entry_idx >= self.hold:
                        close(pos, i, self._px(self.store.get(pos.ticker), i), "holding_period")
            # 3) the copied person sold
            if self.source_sell and i in sells:
                for pos in list(positions):
                    if pos.ticker in sells[i]:
                        close(pos, i, self._px(self.store.get(pos.ticker), i), "source_sell")
            # 4) new leads
            for row in (buys.get(i).itertuples(index=False) if i in buys else []):
                ser = self.store.get(row.price_ticker)
                if ser is None or i < ser.first_idx or i > ser.last_idx:
                    skipped["no_price"] = skipped.get("no_price", 0) + 1
                    continue
                if not self.allow_add and any(p.ticker == row.price_ticker for p in positions):
                    skipped["already_held"] = skipped.get("already_held", 0) + 1
                    continue
                equity = cash + sum(p.shares * self._px(self.store.get(p.ticker), i) for p in positions)
                target = self.size_value * equity if self.size_mode == "percent_of_equity" else self.size_value
                if self.max_pos and len(positions) >= self.max_pos:
                    oldest = self._oldest(positions, i)
                    if oldest is None:
                        skipped["max_positions"] = skipped.get("max_positions", 0) + 1
                        continue
                    close(oldest, i, self._px(self.store.get(oldest.ticker), i), "forced_oldest")
                while cash < target and self.forced == "oldest_first":
                    oldest = self._oldest(positions, i)
                    if oldest is None:
                        break
                    close(oldest, i, self._px(self.store.get(oldest.ticker), i), "forced_oldest")
                amount = min(target, cash)
                if amount < max(self.min_trade, 1e-9) or (not self.allow_partial and amount < 0.98 * target):
                    skipped["no_cash"] = skipped.get("no_cash", 0) + 1
                    continue
                px = self._px(ser, i) * (1 + self.slip)
                comm = self._commission(amount / px, amount)
                shares = (amount - comm) / px
                if not self.fractional:
                    shares = np.floor(shares)
                    if shares <= 0:
                        skipped["no_cash"] = skipped.get("no_cash", 0) + 1
                        continue
                cost = shares * px + comm
                cash -= cost
                positions.append(Position(row.signal_id, row.price_ticker, str(row.issuer_cik), i, px, shares, cost))
            # 5) end of day
            invested = sum(p.shares * self._mark(self.store.get(p.ticker), i) for p in positions)
            eq_rows.append((cal[i], cash, invested, cash + invested, len(positions)))

        for pos in positions:  # still open at the end
            ser = self.store.get(pos.ticker)
            px = self._mark(ser, i1)
            trades.append({
                "signal_id": pos.signal_id, "ticker": pos.ticker, "issuer_cik": pos.issuer_cik,
                "entry_date": cal[pos.entry_idx].date(), "entry_price": pos.entry_price,
                "exit_date": None, "exit_price": px, "shares": pos.shares, "cost": pos.cost,
                "proceeds": pos.shares * px, "pnl": pos.shares * px - pos.cost,
                "return": pos.shares * px / pos.cost - 1, "holding_days": i1 - pos.entry_idx, "reason": "open",
            })

        equity = pd.DataFrame(eq_rows, columns=["date", "cash", "invested", "equity", "n_positions"])
        bench = self.store.get(self.benchmark)
        b = bench.adj_close[i0:i1 + 1]
        equity["benchmark"] = b / b[0] * capital
        trades_df = pd.DataFrame(trades)
        stats = {
            "signals_in_period": int((sig["side"] == "buy").sum()),
            "executed": int(len(trades_df)),
            **{f"skipped_{k}": v for k, v in sorted(skipped.items())},
        }
        if not trades_df.empty:
            for reason, n in trades_df["reason"].value_counts().items():
                stats[f"exit_{reason}"] = int(n)
        return BacktestResult(self.label, equity, trades_df, stats)

    def _oldest(self, positions: list[Position], i: int) -> Position | None:
        # never sell something bought the same day to make room
        eligible = [p for p in positions if i - p.entry_idx >= max(1, self.min_hold_forced)]
        return min(eligible, key=lambda p: p.entry_idx) if eligible else None
