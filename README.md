# Daily S&P 500 screen

A GitHub Actions job runs `engine.py` every weekday morning (5:15 AM Mountain), pulls
prices and fundamentals for the S&P 500 from Yahoo Finance, ranks every name, and commits:

- `data/report.json` – today's 8 long-term picks + 2 short-term trade setups, market regime, methodology
- `data/universe.json` – every ticker's metrics and signals (used to evaluate held positions)
- `data/history/YYYY-MM-DD.json` – archive of each day's picks
- `data/fundamentals_cache.json` – last-good fundamentals (resilience against Yahoo hiccups)

A daily Claude scheduled task reads these files and updates the Stock Desk dashboard.

To include a stock you own that is not in the S&P 500, add its ticker to `extra_tickers.txt`.

Rules-based research tool. Not investment advice.
