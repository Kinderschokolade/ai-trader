"""Command line interface: `aitrader <command> ...` (run `aitrader -h`)."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import reporting as rep
from .config import list_strategies, load_settings, load_strategy
from .db import connect, table_counts


def _strategy(settings, name, overrides):
    return load_strategy(settings, name, overrides)


def _period(settings, args):
    return (args.start or settings.get("validation.start"), args.end or settings.get("validation.end"))


# --- download ------------------------------------------------------------------------

def cmd_download(args, settings):
    from .data import delistings, form4, fundamentals, prices
    from .data.sec_http import client_from_settings
    from .signals import build_signals

    con = connect(settings.db_path)
    client = client_from_settings(settings)
    what = args.what
    if what in ("form4", "all"):
        form4.download(con, client, settings, args.start, args.end, force=args.force)
    if what in ("prices", "fundamentals", "delistings", "all"):
        if not args.strategy:
            sys.exit(f"`download {what}` needs a strategy name, e.g. `aitrader download {what} insider_cluster`")
        strat = _strategy(settings, args.strategy, args.set)
        sig = build_signals(con, strat)
        print(f"{strat.name}: {len(sig):,} signals need data")
        pairs = sig[["issuer_cik", "ticker"]]
        if what in ("prices", "all"):
            prices.ensure_prices(con, settings, pairs, client=client, refresh=args.refresh)
        if what in ("fundamentals", "all") and strat.get("features.size_metric", "market_cap") == "market_cap":
            fundamentals.download(con, client, sig["issuer_cik"].unique(), refresh=args.refresh)
        if what in ("delistings", "all"):
            delistings.download(con, client, settings, pairs, refresh=args.refresh)
    print(table_counts(con).to_string(index=False))


def cmd_status(args, settings):
    con = connect(settings.db_path, read_only=True)
    print(f"database: {settings.db_path}")
    print(table_counts(con).to_string(index=False))
    q = con.execute("SELECT min(quarter), max(quarter), count(*) FROM loaded_quarters WHERE source='form4'").fetchone()
    print(f"form4 quarters loaded: {q[2]} ({q[0]} .. {q[1]})")
    print("\nstrategies:")
    for s in list_strategies(settings):
        print(f"  {s.name:18s} {s.status:7s} allocation={s.get('strategy.allocation')}  {s.path.name}")


# --- analysis --------------------------------------------------------------------------

def _ctx(settings):
    from .runner import Context
    con = connect(settings.db_path)
    return Context(settings, con)


def cmd_signals(args, settings):
    from .runner import results_dir
    ctx = _ctx(settings)
    strat = _strategy(settings, args.strategy, args.set)
    sig = ctx.signals(strat)
    out = results_dir(settings, f"signals_{strat.name}")
    rep.write_csv(sig, out / "signals.csv")
    by_year = sig.groupby(sig["known_date"].dt.year).size().rename("signals")
    print(by_year.to_string())
    print("\nsize x trend buckets:")
    print(pd.crosstab(sig["size_bucket"], sig["trend_bucket"], margins=True).to_string())
    missing = sig["price_ticker"].isna() | ~sig["price_ticker"].isin(
        [r[0] for r in ctx.con.execute("SELECT ticker FROM price_status WHERE status='ok'").fetchall()])
    print(f"\n{missing.sum():,}/{len(sig):,} signals without price data")
    print(f"\nwritten: {out}")


def cmd_event_study(args, settings):
    from .runner import default_event_period, results_dir
    ctx = _ctx(settings)
    strat = _strategy(settings, args.strategy, args.set)
    start, end = (args.start, args.end) if args.start and args.end else default_event_period(settings, strat)
    curves, summary, per_event, reasons = ctx.event_study(strat, start, end)
    out = results_dir(settings, f"event-study_{strat.name}")
    rep.write_csv(curves, out / "curves.csv")
    rep.write_csv(summary, out / "summary.csv")
    rep.write_csv(per_event, out / "events.csv")
    min_n = int(strat.get("event_study.min_bucket_events", 30))
    title = f"{strat.label}: abnormal return after signal ({start} .. {end})"
    rep.plot_event_curves(curves, summary, out / "01_all.png", title, "all", min_n)
    dims = strat.get("event_study.bucket_by", []) or []
    for k, d in enumerate(dims, 2):
        rep.plot_event_curves(curves, summary, out / f"0{k}_{d}.png", f"{strat.label}: by {d.replace('_', ' ')}", d, min_n)
    if len(dims) >= 2:
        a, b = dims[0], dims[1]
        la = [x for x in strat.get("features.size_labels", []) if x in set(per_event[a])] if a == "size_bucket" else sorted(set(per_event[a]))
        lb = [x for x in strat.get("features.trend_labels", []) if x in set(per_event[b])] if b == "trend_bucket" else sorted(set(per_event[b]))
        if "unknown" in set(per_event[a]):
            la.append("unknown")
        rep.plot_event_grid(curves, out / "09_grid.png", f"{strat.label}: {a} x {b}", la, lb, f"{a} x {b}", min_n)
    print(f"events used: {reasons}")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\nwritten: {out}")


def cmd_backtest(args, settings):
    from .runner import results_dir
    ctx = _ctx(settings)
    strat = _strategy(settings, args.strategy, args.set)
    start, end = _period(settings, args)
    res, m = ctx.backtest(strat, start, end)
    out = results_dir(settings, f"backtest_{strat.name}")
    rep.write_csv(res.equity, out / "equity.csv")
    rep.write_csv(res.trades, out / "trades.csv")
    rep.write_csv(pd.DataFrame([m]), out / "metrics.csv")
    rep.plot_equity([res], out / "equity.png", f"{strat.label} ({start} .. {end})")
    _print_metrics(pd.DataFrame([m]))
    print(f"\nwritten: {out}")


def cmd_compare(args, settings):
    from .runner import results_dir
    ctx = _ctx(settings)
    start, end = _period(settings, args)
    results, rows = [], []
    for name in args.strategies:
        strat = _strategy(settings, name, args.set)
        res, m = ctx.backtest(strat, start, end)
        results.append(res)
        rows.append(m)
    table = pd.DataFrame(rows)
    out = results_dir(settings, "compare_" + "_".join(args.strategies))
    rep.write_csv(table, out / "metrics.csv")
    for r in results:
        rep.write_csv(r.equity, out / f"equity_{r.label}.csv")
    rep.plot_equity(results, out / "equity.png", f"Comparison ({start} .. {end})")
    _print_metrics(table)
    print(f"\nwritten: {out}")


def cmd_sweep(args, settings):
    from .runner import results_dir, score
    ctx = _ctx(settings)
    strat = _strategy(settings, args.strategy, args.set)
    start, end = _period(settings, args)
    metric = args.metric or settings.get("validation.select_metric", "sharpe")
    results, rows = [], []
    variants = strat.variants()
    for n, v in enumerate(variants, 1):
        res, m = ctx.backtest(v, start, end)
        results.append(res)
        rows.append({"variant": v.variant, **m})
        print(f"  [{n}/{len(variants)}] {v.variant}: {metric}={m.get(metric):.3f} cagr={m['cagr']:.1%}")
    table = pd.DataFrame(rows).sort_values(metric, ascending=False, key=lambda s: s.fillna(-1e9))
    out = results_dir(settings, f"sweep_{strat.name}")
    rep.write_csv(table, out / "metrics.csv")
    ranking = list(table["strategy"])
    rep.plot_equity(results, out / "equity.png", f"{strat.name}: parameter sweep ({start} .. {end}), sorted by {metric}", ranking)
    _print_metrics(table)
    print("\nNote: a sweep over the whole period is in-sample. Use `walkforward` to judge parameters out of sample.")
    print(f"written: {out}")


def cmd_walkforward(args, settings):
    from .runner import results_dir, walkforward
    ctx = _ctx(settings)
    strat = _strategy(settings, args.strategy, args.set)
    metric = settings.get("validation.select_metric", "sharpe")
    table, chosen, all_test = walkforward(ctx, strat)
    out = results_dir(settings, f"walkforward_{strat.name}")
    rep.write_csv(table, out / "windows.csv")
    rep.write_csv(all_test, out / "all_variants_test.csv")
    rep.plot_walkforward(table, all_test, metric, out / "test_scores.png",
                         f"{strat.name}: walk-forward, test {metric} of chosen variant vs all variants")
    rep.plot_equity([r for _, r in chosen], out / "test_equity.png", f"{strat.name}: test windows (chosen variants)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print(f"\nwritten: {out}")


def cmd_demo(args, settings):
    from .demo import build_demo_db
    path = settings.db_path
    build_demo_db(path)
    print(f"Synthetic demo database written to {path}")
    print("Try: aitrader --db", path, "event-study insider_cluster")


def _print_metrics(table: pd.DataFrame):
    cols = [c for c in ["strategy", "cagr", "benchmark_cagr", "excess_cagr", "sharpe", "max_drawdown", "volatility",
                        "n_trades", "hit_rate", "avg_trade_return", "avg_holding_days", "avg_positions", "avg_invested_pct",
                        "signals_in_period", "skipped_no_price", "exit_forced_oldest"] if c in table.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table[cols].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aitrader", description="Copy-trading research and paper trading")
    p.add_argument("--db", help="database file (default: [paths] db in config/global.toml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, strategy=True, period=True):
        if strategy:
            sp.add_argument("strategy", help="strategy name (file in config/strategies) or path to a .toml")
        if period:
            sp.add_argument("--start")
            sp.add_argument("--end")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a strategy parameter, e.g. --set exit.holding_days=63 (repeatable)")

    d = sub.add_parser("download", help="download data into the local database")
    d.add_argument("what", choices=["form4", "prices", "fundamentals", "delistings", "all"])
    d.add_argument("strategy", nargs="?")
    d.add_argument("--start")
    d.add_argument("--end")
    d.add_argument("--force", action="store_true", help="re-download already loaded Form 4 quarters")
    d.add_argument("--refresh", action="store_true", help="re-download prices / fundamentals / delistings")
    d.add_argument("--set", action="append", default=[])
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("status", help="show what is in the database")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("signals", help="build signals and write them to CSV")
    common(s, period=False)
    s.set_defaults(func=cmd_signals)

    s = sub.add_parser("event-study", help="abnormal return curves after signals, by bucket")
    common(s)
    s.set_defaults(func=cmd_event_study)

    s = sub.add_parser("backtest", help="simulate one strategy")
    common(s)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("compare", help="simulate several strategies over the same period")
    s.add_argument("strategies", nargs="+")
    common(s, strategy=False)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("sweep", help="simulate every combination of the strategy's [sweep] table")
    common(s)
    s.add_argument("--metric")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("walkforward", help="choose parameters on train windows, judge on test windows")
    common(s, period=False)
    s.set_defaults(func=cmd_walkforward)

    s = sub.add_parser("demo", help="write a synthetic database to try every command offline")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(db=args.db)
        args.func(args, settings)
    except (ValueError, NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
