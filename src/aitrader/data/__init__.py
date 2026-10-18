"""Data downloaders. Each module writes into the local DuckDB database and can be re-run
incrementally (already loaded pieces are skipped), so the same code serves the initial
history download and later daily updates."""
