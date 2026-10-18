# ai-trader

Research and (later) paper trading for copy-trading strategies:

| Strategy | Config | Status |
|---|---|---|
| Insider cluster buys (≥2 insiders within 14 days) | `config/strategies/insider_cluster.toml` | **active**: download, event study, backtest, sweep, walk-forward |
| Large insider buys (≥ $100k) | `config/strategies/insider_large.toml` | **active** |
| Berkshire Hathaway 13F | `config/strategies/buffett.toml` | draft (phase 2) |
| 8 superinvestors 13F | `config/strategies/superinvestors.toml` | draft (phase 2) |
| Trump + executive branch (OGE 278-T) | `config/strategies/executive_branch.toml` | draft (phase 2, paper only) |
| Polymarket insider-bet logger (no trading) | `config/loggers/polymarket.toml` | draft (phase 2) |

Each strategy gets 1/5 of the paper account (`allocation = 0.2`).

All data goes into one local DuckDB file (`data/aitrader.duckdb`). Every strategy is defined
by its own TOML file: parameters, filters, the parameter grid for sweeps and optional LLM prompts.

---

## 1. Setup on the Hetzner VM (Docker)

1. Copy the repo to the VM. From your Mac:
   ```bash
   rsync -av --exclude .venv --exclude data --exclude results ~/code/ai-trader/ root@<VM-IP>:~/ai-trader/
   ```
   (Or push it to a private GitHub repo and `git clone` it on the VM.)
2. Log in to the VM and go into the folder:
   ```bash
   ssh root@<VM-IP>
   cd ~/ai-trader
   ```
3. Open `config/global.toml` and set the SEC contact, otherwise the SEC blocks all downloads:
   ```toml
   [sec]
   user_agent = "Marius ai-trader research your@email.org"
   ```
4. Build the image (once, and again after code changes):
   ```bash
   docker compose build
   ```
5. Every command below runs as
   ```bash
   docker compose run --rm aitrader <command>
   ```
   Tip: `alias at='docker compose run --rm aitrader'` and then just `at status`.

### Without Docker (Mac or VM)

```bash
cd ~/code/ai-trader
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
aitrader --help
```

---

## 2. Try everything offline first (synthetic data)

```bash
aitrader --db data/demo.duckdb demo
aitrader --db data/demo.duckdb event-study insider_cluster
aitrader --db data/demo.duckdb walkforward insider_cluster
```

The demo data is fake. It has a planted effect: after a cluster buy, micro caps drift up
about 18 % over 6 months, small caps 10 %, mid caps 4 %, large caps 0. The event study
should find roughly these numbers. Use it to check that the pipeline works.

---

## 3. Download the real data (in this order)

| Step | Command | What it does | Rough duration |
|---|---|---|---|
| 1 | `download form4` | SEC insider data sets, 2006 Q1 until today | 15–30 min |
| 2 | `download prices insider_cluster` | Yahoo prices for every company with a signal, plus SPY | 20–60 min |
| 3 | `download fundamentals insider_cluster` | shares outstanding (for market cap) from SEC XBRL | 10–20 min |
| 4 | `download delistings insider_cluster` | why stocks stopped trading (bankruptcy, merger, ...) | 5–15 min |
| — | `download all insider_cluster` | steps 1–4 in one go | |
| — | `status` | what is in the database | |

Repeat steps 2–4 for `insider_large` (already downloaded pieces are skipped).
All steps can be interrupted and restarted.

---

## 4. Research workflow (one strategy at a time)

```bash
aitrader signals insider_cluster        # signals per year and per bucket -> signals.csv
aitrader event-study insider_cluster    # return curves after the signal, by size x trend
aitrader backtest insider_cluster       # one simulation with the TOML parameters
aitrader sweep insider_cluster          # every combination of the [sweep] table
aitrader walkforward insider_cluster    # choose on train windows, judge on test windows
aitrader compare insider_cluster insider_large
```

Each command writes a new folder `results/<time>_<command>_<strategy>/` with PNG plots and CSVs.

**Change parameters** in the TOML file, or for a single run:
```bash
aitrader backtest insider_cluster --set exit.holding_days=63 --set "filters.size_buckets=['micro','small']"
```

**Period:** backtests default to `[validation] start..end` in `global.toml`; override with
`--start 2015-01-01 --end 2020-12-31`. The event study defaults to the first train window, so
the parameters you read off the curves are not tuned on test data.

### Walk-forward validation

`[validation]` in `global.toml`: train 8 years, test 5 years, step 2 years, from 2007 to 2025:

| Window | Train | Test |
|---|---|---|
| 1 | 2007–2014 | 2015–2019 |
| 2 | 2009–2016 | 2017–2021 |
| 3 | 2011–2018 | 2019–2023 |
| 4 | 2013–2020 | 2021–2025 |

On every train window the best variant of the `[sweep]` grid is chosen (by `select_metric`),
then judged on the following test window. `test_scores.png` shows the chosen variant against
all variants: if the chosen one is not consistently near the top, the "best" parameters were luck.

---

## 5. How the simulation works

* **Timing:** we act on the SEC *filing* date, never the trade date. Entry is the next trading
  day (`entry_lag_days = 1`) at the open (`execution_price`).
* **Sizing:** every new lead gets the same share of the strategy's current value
  (`[sizing] value = 0.04` → 4 %), or a fixed USD amount (`mode = "fixed_amount"`).
* **Exit:** after `holding_days` trading days. If cash is short for a new lead, the **oldest**
  position is sold early. A position bought the same day is never sold to make room; if there is
  still not enough cash, the lead is skipped (`allow_partial = false`).
* **Costs:** IBKR fixed commissions and `slippage_bps` on every trade.
* **Delisted stocks:** if a held stock stops trading, it is sold at the last price minus a haircut
  that depends on the reason found in the company's SEC filings (`[delisting.haircut]`).
* **Missing prices:** Yahoo often has no data for delisted companies. Such signals are skipped and
  counted (`skipped_no_price` in `metrics.csv`). `download delistings` also classifies these
  companies, so you can see how many of the skipped ones went bankrupt.
* **Features:** `size_bucket` = market cap at the filing date (SEC shares outstanding × price; XBRL
  data starts ~2009, earlier signals are "unknown" — switch to `size_metric = "dollar_volume"` to
  cover them). `trend_bucket` = return over the 63 trading days before the filing, relative to SPY.

### Important metrics in `metrics.csv`

`cagr`, `benchmark_cagr`, `excess_cagr`, `sharpe`, `max_drawdown`, `hit_rate`,
`avg_holding_days` (much lower than `holding_days` means forced exits dominate: too many
signals for the position size), `exit_forced_oldest`, `skipped_no_price`.

---

## 6. LLM prompts

Every strategy TOML has an `[llm]` section with a system and a user prompt. When enabled, each
signal is sent to the model and only "buy" answers are traded. Answers are cached in the database.
Needs `ANTHROPIC_API_KEY` in the environment.

In backtests the model already knows how most historical stocks did. Therefore the filter refuses
to run on signals before `knowledge_cutoff` unless you set `allow_lookahead = true`. It is meant
for live/paper trading.

---

## 7. Project layout

```
config/global.toml            shared settings (paths, SEC, costs, validation windows)
config/strategies/*.toml      one file per strategy
src/aitrader/data/            downloaders: form4, prices, fundamentals, delistings
src/aitrader/signals/         signal builders (same code for backtest and live)
src/aitrader/features.py      size and trend buckets
src/aitrader/backtest/        engine, metrics, event study
src/aitrader/runner.py        sweeps and walk-forward
src/aitrader/reporting.py     plots and CSVs
src/aitrader/cli.py           the `aitrader` command
tests/                        unit tests (`pytest`)
```

## 8. Phase 2 (planned)

* 13F strategies (Buffett, 8 investors) from SEC EDGAR.
* Executive branch trades (Disclosed Capitol API) — paper trading only, history too short.
* Polymarket logger.
* Live mode: daily Form 4 filings from the EDGAR daily index (the quarterly data sets lag by a
  quarter), IB Gateway container, noon scheduler, orders on the IBKR paper account.
