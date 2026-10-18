"""Glue: build signals, features and run backtests / event studies / sweeps / walk-forward."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import llm
from .backtest import event_study, metrics
from .backtest.engine import Engine
from .data.prices import PriceStore
from .signals import build_signals


class Context:
    def __init__(self, settings, con):
        self.settings = settings
        self.con = con
        self.benchmark = settings.get("portfolio.benchmark", "SPY")
        self.store = PriceStore.from_benchmark(
            con, self.benchmark, settings.get("data.start"), settings.data_end,
            int(settings.get("delisting.stale_days", 10)))
        self.delist = {r[0]: r[1] for r in con.execute("SELECT cik, classification FROM delistings").fetchall()}
        self._signal_cache: dict[str, pd.DataFrame] = {}

    def signals(self, strategy, log=print) -> pd.DataFrame:
        key = json.dumps({k: strategy.get(k) for k in ("strategy.type", "signal", "filters", "features", "llm")},
                         sort_keys=True, default=str)
        if key not in self._signal_cache:
            sig = build_signals(self.con, strategy)
            sig = feat.add_features(sig, self.store, self.con, strategy, self.benchmark)
            sig = feat.apply_feature_filters(sig, strategy)
            sig = llm.apply_filter(sig, strategy, self.con, mode="backtest", log=log)
            self._signal_cache[key] = sig.reset_index(drop=True)
        return self._signal_cache[key]

    def capital(self, strategy) -> float:
        return float(self.settings.get("portfolio.initial_capital", 1_000_000)) * float(strategy.get("strategy.allocation", 1.0))

    def backtest(self, strategy, start, end, log=print):
        sig = self.signals(strategy, log)
        eng = Engine(self.settings, strategy, self.store, self.delist)
        res = eng.run(sig, start, end, self.capital(strategy))
        return res, metrics.compute(res)

    def event_study(self, strategy, start, end, log=print):
        sig = self.signals(strategy, log)
        sig = sig[(sig["known_date"] >= pd.Timestamp(start)) & (sig["known_date"] <= pd.Timestamp(end))]
        return event_study.run(sig, self.store, self.settings, strategy, self.delist)


# --- validation windows ---------------------------------------------------------------

def windows(settings) -> list[dict]:
    start = pd.Timestamp(settings.get("validation.start"))
    end = pd.Timestamp(settings.get("validation.end"))
    tr, te, st = (int(settings.get(f"validation.{k}")) for k in ("train_years", "test_years", "step_years"))
    out, k = [], 0
    while True:
        a = start + pd.DateOffset(years=k * st)
        b = a + pd.DateOffset(years=tr) - pd.Timedelta(days=1)
        c = b + pd.Timedelta(days=1)
        d = c + pd.DateOffset(years=te) - pd.Timedelta(days=1)
        if d > end:
            break
        out.append({"window": len(out) + 1, "train_start": a.date(), "train_end": b.date(),
                    "test_start": c.date(), "test_end": d.date()})
        k += 1
    if not out:
        raise ValueError("No walk-forward window fits into [validation] start..end; shorten train/test years.")
    return out


def default_event_period(settings, strategy) -> tuple[str, str]:
    s = strategy.get("event_study.start") or ""
    e = strategy.get("event_study.end") or ""
    if s and e:
        return s, e
    w = windows(settings)[0]
    return s or str(w["train_start"]), e or str(w["train_end"])


def results_dir(settings, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = settings.path("results") / f"{stamp}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def score(m: dict, metric: str) -> float:
    v = m.get(metric, np.nan)
    return -np.inf if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def walkforward(ctx: Context, strategy, log=print):
    variants = strategy.variants()
    metric = ctx.settings.get("validation.select_metric", "sharpe")
    rows, chosen_results, all_test = [], [], []
    for w in windows(ctx.settings):
        train = []
        for v in variants:
            _, m = ctx.backtest(v, w["train_start"], w["train_end"], log)
            train.append((v, m))
        best_v, best_m = max(train, key=lambda vm: score(vm[1], metric))
        test_metrics = []
        for v in variants:
            res, m = ctx.backtest(v, w["test_start"], w["test_end"], log)
            test_metrics.append((v, m))
            all_test.append({"window": w["window"], "variant": v.variant, **{k: m.get(k) for k in ("cagr", "sharpe", "max_drawdown", "excess_cagr")}})
            if v.variant == best_v.variant:
                res.label = f"window {w['window']}: {best_v.variant}"
                chosen_results.append((w, res))
                chosen_m = m
        others = [score(m, metric) for _, m in test_metrics]
        rows.append({
            **w, "chosen_variant": best_v.variant, f"train_{metric}": best_m.get(metric),
            f"test_{metric}": chosen_m.get(metric), f"test_{metric}_median_all_variants": float(np.median(others)),
            "test_rank_of_chosen": int(1 + sum(o > score(chosen_m, metric) for o in others)),
            "n_variants": len(variants), "test_cagr": chosen_m.get("cagr"),
            "test_benchmark_cagr": chosen_m.get("benchmark_cagr"), "test_max_drawdown": chosen_m.get("max_drawdown"),
        })
        log(f"window {w['window']}: train {w['train_start']}..{w['train_end']} -> chose {best_v.variant}; "
            f"test {metric} {chosen_m.get(metric):.2f} (median of all variants {np.median(others):.2f})")
    return pd.DataFrame(rows), chosen_results, pd.DataFrame(all_test)
