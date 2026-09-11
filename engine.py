"""
Daily S&P 500 screening engine
==============================

Runs once per weekday (GitHub Actions) and writes:

    data/report.json              - today's picks, market regime, holdings-ready signals
    data/universe.json            - every ticker's metrics (used for holdings evaluation)
    data/history/YYYY-MM-DD.json  - picks archive
    data/fundamentals_cache.json  - last-good fundamentals per ticker (resilience)

Design
------
* LONG-TERM composite score (0-100) = weighted average of five factor scores, each the
  mean percentile rank (across the universe) of its underlying metrics:
      quality  25%  ROE, operating margin, FCF yield, debt/equity (inverse)
      value    20%  forward P/E vs sector (inverse), trailing PEG (inverse), EV/EBITDA (inverse)
      growth   20%  revenue growth, earnings growth, implied forward EPS growth
      momentum 20%  6-mo return, 12-1 month return, price vs SMA200, SMA50>SMA200
      analyst  15%  mean recommendation (inverse), target upside, analyst count
  Top 8 by score, max 2 per sector, with data-quality and falling-knife guards.

* SHORT-TERM (2 picks) come from two mechanical setups scanned over the same universe:
      "pullback"  uptrend (SMA50>SMA200, price>SMA200), RSI14 in 30-48, price 2-8% under SMA20
      "breakout"  close above the prior 20-day high on >=1.5x average volume, RSI14 < 75
  Names with earnings inside 10 calendar days are excluded. Each pick carries an
  entry, a stop, and a target (2R by default) so the trade plan is explicit.

* Every ticker also gets a market-data-only `lt_signal` (add-zone / hold / trim-zone /
  sell-zone). The daily Claude task combines that with your cost basis to produce the
  final Hold / Add / Trim / Sell call on positions you actually own.

No argparse: settings are module constants or environment variables.
This is a rules-based screen for research, not investment advice.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("STOCKS_DATA_DIR", "data"))
EXTRA_TICKERS_FILE = Path(os.environ.get("STOCKS_EXTRA_TICKERS", "extra_tickers.txt"))
PRICE_PERIOD = "2y"
FUNDAMENTALS_WORKERS = int(os.environ.get("STOCKS_WORKERS", "6"))
FUNDAMENTALS_MAX_AGE_DAYS = 10          # use cached fundamentals up to this old if today's fetch fails
MIN_ANALYSTS_FOR_PICK = 5
LT_TOP_N = 8
ST_TOP_N = 2
MAX_PER_SECTOR = 2
EARNINGS_EXCLUSION_DAYS_ST = 10
FACTOR_WEIGHTS = {"quality": 0.25, "value": 0.20, "growth": 0.20, "momentum": 0.20, "analyst": 0.15}
MIN_FACTORS_PRESENT = 4

BENCHMARKS = ["SPY", "^VIX"]

INFO_KEYS = [
    "shortName", "sector", "industry", "marketCap", "currentPrice",
    "trailingPE", "forwardPE", "trailingPegRatio", "pegRatio", "enterpriseToEbitda",
    "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity", "freeCashflow",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth", "forwardEps", "trailingEps",
    "recommendationMean", "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions",
    "dividendYield", "beta", "earningsTimestampStart", "earningsTimestamp",
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
def _normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def get_universe() -> pd.DataFrame:
    """Return DataFrame[ticker, name, sector] for the S&P 500 plus any extra tickers."""
    import requests

    df = None
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (stocks-screener)"}, timeout=30,
        ).text
        from io import StringIO
        tables = pd.read_html(StringIO(html))
        t = tables[0]
        df = pd.DataFrame({
            "ticker": t["Symbol"].map(_normalize_ticker),
            "name": t["Security"],
            "sector": t["GICS Sector"],
        })
        log(f"Universe from Wikipedia: {len(df)} names")
    except Exception as e:  # noqa: BLE001
        log(f"Wikipedia constituents failed ({e!r}); falling back to GitHub dataset")
        csv = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        )
        df = pd.DataFrame({
            "ticker": csv["Symbol"].map(_normalize_ticker),
            "name": csv["Security"] if "Security" in csv else csv.iloc[:, 1],
            "sector": csv["GICS Sector"] if "GICS Sector" in csv else csv.iloc[:, 2],
        })
        log(f"Universe from GitHub dataset: {len(df)} names")

    df["in_sp500"] = True
    if EXTRA_TICKERS_FILE.exists():
        extras = [
            _normalize_ticker(x) for x in EXTRA_TICKERS_FILE.read_text().split()
            if x.strip() and not x.startswith("#")
        ]
        extras = [x for x in extras if x not in set(df["ticker"])]
        if extras:
            df = pd.concat([df, pd.DataFrame({"ticker": extras, "name": extras, "sector": None, "in_sp500": False})])
            log(f"Added {len(extras)} extra tickers: {extras}")
    return df.drop_duplicates("ticker").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Data fetch (yfinance)
# --------------------------------------------------------------------------------------
def fetch_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} (adjusted)."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    batch = 100
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=PRICE_PERIOD, interval="1d", auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001
                log(f"download attempt {attempt + 1} failed for batch {i}: {e!r}")
                time.sleep(5 * (attempt + 1))
                raw = None
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for t in chunk:
                if t in raw.columns.get_level_values(0):
                    d = raw[t].dropna(how="all")
                    if len(d) > 60:
                        out[t] = d
        else:  # single ticker chunk
            d = raw.dropna(how="all")
            if len(d) > 60:
                out[chunk[0]] = d
        log(f"prices: {min(i + batch, len(tickers))}/{len(tickers)} requested, {len(out)} received")
    return out


def _fetch_one_info(ticker: str) -> dict:
    import yfinance as yf

    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            if not info or ("regularMarketPrice" not in info and "currentPrice" not in info):
                raise ValueError("empty info")
            row = {k: info.get(k) for k in INFO_KEYS}
            row["fetched"] = date.today().isoformat()
            return row
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"_error": repr(e)[:200]}
            time.sleep(2 + 3 * attempt)
    return {"_error": "unknown"}


def fetch_fundamentals(tickers: list[str], cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:  # noqa: BLE001
            cache = {}
    fresh: dict[str, dict] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=FUNDAMENTALS_WORKERS) as ex:
        futs = {ex.submit(_fetch_one_info, t): t for t in tickers}
        done = 0
        for f in as_completed(futs):
            t = futs[f]
            row = f.result()
            done += 1
            if "_error" in row:
                failed.append(t)
            else:
                fresh[t] = row
            if done % 50 == 0:
                log(f"fundamentals: {done}/{len(tickers)} ({len(failed)} failed so far)")
    log(f"fundamentals: {len(fresh)} fresh, {len(failed)} failed")

    # merge with cache: fresh wins; failed fall back to cache if recent enough
    merged = dict(cache)
    merged.update(fresh)
    today = date.today()
    result: dict[str, dict] = {}
    for t in tickers:
        row = merged.get(t)
        if not row:
            continue
        try:
            age = (today - date.fromisoformat(row.get("fetched", "2000-01-01"))).days
        except Exception:  # noqa: BLE001
            age = 999
        if age <= FUNDAMENTALS_MAX_AGE_DAYS:
            r = dict(row)
            r["fundamentals_age_days"] = age
            result[t] = r
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(merged, indent=0, sort_keys=True))
    return result


# --------------------------------------------------------------------------------------
# Technicals
# --------------------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    v = rsi.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _ret(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def compute_technicals(px: pd.DataFrame) -> dict:
    c = px["Close"].astype(float)
    h = px["High"].astype(float)
    l = px["Low"].astype(float)
    v = px["Volume"].astype(float)
    last = float(c.iloc[-1])
    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan")
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    hi52 = float(c.iloc[-252:].max())
    lo52 = float(c.iloc[-252:].min())
    vol20 = float(v.iloc[-21:-1].mean()) if len(v) > 21 else float("nan")
    prior_20d_high = float(h.iloc[-21:-1].max()) if len(h) > 21 else float("nan")
    low_10d = float(l.iloc[-10:].min())
    daily = c.pct_change().dropna()
    vol_ann = float(daily.iloc[-60:].std() * math.sqrt(252)) if len(daily) >= 60 else float("nan")
    return {
        "price": round(last, 2),
        "as_of": str(pd.Timestamp(c.index[-1]).date()),
        "chg_1d": _ret(c, 1),
        "ret_1m": _ret(c, 21),
        "ret_3m": _ret(c, 63),
        "ret_6m": _ret(c, 126),
        "ret_12m": _ret(c, 252),
        "ret_12_1": (float(c.iloc[-22] / c.iloc[-253] - 1) if len(c) > 253 else float("nan")),
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "pct_vs_sma20": last / sma20 - 1 if sma20 else float("nan"),
        "pct_vs_sma50": last / sma50 - 1 if sma50 else float("nan"),
        "pct_vs_sma200": last / sma200 - 1 if sma200 and not math.isnan(sma200) else float("nan"),
        "golden": bool(sma50 > sma200) if not math.isnan(sma200) else False,
        "rsi14": _rsi(c),
        "atr14": atr14,
        "atr_pct": atr14 / last if last else float("nan"),
        "hi52": hi52, "lo52": lo52,
        "pct_from_hi52": last / hi52 - 1,
        "vol20": vol20,
        "vol_ratio": float(v.iloc[-1] / vol20) if vol20 and not math.isnan(vol20) and vol20 > 0 else float("nan"),
        "prior_20d_high": prior_20d_high,
        "low_10d": low_10d,
        "vol_ann": vol_ann,
        "avg_dollar_vol": float((c * v).iloc[-21:].mean()),
    }


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _pct_rank(s: pd.Series, higher_is_better: bool = True, lo_q=0.02, hi_q=0.98) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 10:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    s = s.clip(lo, hi)
    r = s.rank(pct=True)
    return (r if higher_is_better else 1 - r) * 100


def _sector_relative(s: pd.Series, sectors: pd.Series, higher_is_better: bool) -> pd.Series:
    out = pd.Series(np.nan, index=s.index)
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub = pd.to_numeric(s.loc[idx], errors="coerce")
        if sub.notna().sum() >= 5:
            out.loc[idx] = _pct_rank(sub, higher_is_better, 0.05, 0.95)
    # fall back to universe rank where sector too small / unknown
    uni = _pct_rank(s, higher_is_better)
    return out.fillna(uni)


def build_frame(universe: pd.DataFrame, tech: dict[str, dict], fund: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, u in universe.iterrows():
        t = u["ticker"]
        if t not in tech:
            continue
        r = {"ticker": t, "name": u["name"], "sector": u["sector"], "in_sp500": bool(u["in_sp500"])}
        r.update(tech[t])
        f = fund.get(t, {})
        r["has_fundamentals"] = bool(f)
        r["fundamentals_age_days"] = f.get("fundamentals_age_days")
        if f.get("shortName"):
            r["name"] = f["shortName"] if pd.isna(r["name"]) or r["name"] == t else r["name"]
        if not r["sector"] and f.get("sector"):
            r["sector"] = f["sector"]
        r["industry"] = f.get("industry")
        r["market_cap"] = f.get("marketCap")
        r["fpe"] = f.get("forwardPE")
        r["tpe"] = f.get("trailingPE")
        r["peg"] = f.get("trailingPegRatio") or f.get("pegRatio")
        r["ev_ebitda"] = f.get("enterpriseToEbitda")
        r["roe"] = f.get("returnOnEquity")
        r["op_margin"] = f.get("operatingMargins")
        r["net_margin"] = f.get("profitMargins")
        r["debt_equity"] = f.get("debtToEquity")
        fcf, mc = f.get("freeCashflow"), f.get("marketCap")
        r["fcf_yield"] = (fcf / mc) if fcf and mc else None
        r["rev_growth"] = f.get("revenueGrowth")
        r["eps_growth"] = f.get("earningsGrowth") if f.get("earningsGrowth") is not None else f.get("earningsQuarterlyGrowth")
        fe, te = f.get("forwardEps"), f.get("trailingEps")
        r["fwd_eps_growth"] = (fe / te - 1) if fe and te and te > 0 else None
        r["rec_mean"] = f.get("recommendationMean")
        r["rec_key"] = f.get("recommendationKey")
        r["n_analysts"] = f.get("numberOfAnalystOpinions")
        tgt = f.get("targetMeanPrice")
        r["target_upside"] = (tgt / r["price"] - 1) if tgt and r["price"] else None
        r["div_yield"] = f.get("dividendYield")
        r["beta"] = f.get("beta")
        ets = f.get("earningsTimestampStart") or f.get("earningsTimestamp")
        if ets:
            try:
                ed = datetime.fromtimestamp(int(ets), tz=timezone.utc).date()
                r["next_earnings"] = ed.isoformat()
                r["earnings_in_days"] = (ed - date.today()).days
            except Exception:  # noqa: BLE001
                r["next_earnings"], r["earnings_in_days"] = None, None
        else:
            r["next_earnings"], r["earnings_in_days"] = None, None
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ticker")
    # sanitize fundamentals that are clearly junk
    for col in ["fpe", "tpe", "peg", "ev_ebitda"]:
        df.loc[(df[col] <= 0) | (df[col] > 500), col] = np.nan
    df.loc[df["debt_equity"] < 0, "debt_equity"] = np.nan
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    sec = df["sector"].fillna("Unknown")
    q = pd.concat([
        _pct_rank(df["roe"]), _pct_rank(df["op_margin"]), _pct_rank(df["fcf_yield"]),
        _pct_rank(df["debt_equity"], higher_is_better=False),
    ], axis=1).mean(axis=1, skipna=True)
    v = pd.concat([
        _sector_relative(df["fpe"], sec, higher_is_better=False),
        _pct_rank(df["peg"], higher_is_better=False),
        _sector_relative(df["ev_ebitda"], sec, higher_is_better=False),
    ], axis=1).mean(axis=1, skipna=True)
    g = pd.concat([
        _pct_rank(df["rev_growth"]), _pct_rank(df["eps_growth"]), _pct_rank(df["fwd_eps_growth"]),
    ], axis=1).mean(axis=1, skipna=True)
    m = pd.concat([
        _pct_rank(df["ret_6m"]), _pct_rank(df["ret_12_1"]), _pct_rank(df["pct_vs_sma200"]),
        df["golden"].map({True: 100.0, False: 0.0}),
    ], axis=1).mean(axis=1, skipna=True)
    a = pd.concat([
        _pct_rank(df["rec_mean"], higher_is_better=False), _pct_rank(df["target_upside"]),
        _pct_rank(df["n_analysts"]),
    ], axis=1).mean(axis=1, skipna=True)
    factors = pd.DataFrame({"quality": q, "value": v, "growth": g, "momentum": m, "analyst": a})
    # value/quality/growth/analyst require fundamentals; where the whole factor is NaN it is skipped
    w = pd.Series(FACTOR_WEIGHTS)
    present = factors.notna()
    weighted = (factors.fillna(0) * w).sum(axis=1) / (present * w).sum(axis=1).replace(0, np.nan)
    df = df.copy()
    for c in factors.columns:
        df[f"f_{c}"] = factors[c].round(1)
    df["factors_present"] = present.sum(axis=1)
    df["score"] = weighted.round(1)
    df["rank"] = df["score"].rank(ascending=False, method="min")
    df["score_pct"] = df["score"].rank(pct=True) * 100
    return df


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------
def lt_signal(r: pd.Series) -> str:
    """Market-data-only zone for a long-term holding (cost basis applied downstream)."""
    above200 = r.get("pct_vs_sma200", np.nan)
    if pd.notna(above200):
        if above200 < 0 and not r.get("golden", False) and r.get("score_pct", 50) < 30:
            return "sell-zone"
        if above200 > 0.25 or r.get("rsi14", 50) > 80:
            return "trim-zone"
        if above200 > 0 and r.get("score_pct", 0) >= 85 and 35 <= r.get("rsi14", 50) <= 55:
            return "add-zone"
    return "hold"


def trend_label(r: pd.Series) -> str:
    a200, a50 = r.get("pct_vs_sma200", np.nan), r.get("pct_vs_sma50", np.nan)
    if pd.isna(a200):
        return "unknown"
    if a200 > 0 and r.get("golden", False):
        return "uptrend" if a50 >= 0 else "uptrend-pullback"
    if a200 < 0 and not r.get("golden", False):
        return "downtrend"
    return "transition"


def pick_long_term(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cand = df[
        (df["in_sp500"])
        & (df["factors_present"] >= MIN_FACTORS_PRESENT)
        & (df["n_analysts"].fillna(0) >= MIN_ANALYSTS_FOR_PICK)
        & ~((df["pct_from_hi52"] < -0.35) & (df["pct_vs_sma200"] < 0))   # falling-knife guard
    ].sort_values("score", ascending=False)
    picks, per_sector = [], {}
    for t, r in cand.iterrows():
        if t in exclude:
            continue
        s = r["sector"] or "Unknown"
        if per_sector.get(s, 0) >= MAX_PER_SECTOR:
            continue
        picks.append(t)
        per_sector[s] = per_sector.get(s, 0) + 1
        if len(picks) == LT_TOP_N:
            break
    return picks


def scan_short_term(df: pd.DataFrame) -> list[dict]:
    setups = []
    liquid = df[(df["avg_dollar_vol"] > 50e6) & (df["in_sp500"])]
    for t, r in liquid.iterrows():
        eid = r.get("earnings_in_days")
        if eid is not None and pd.notna(eid) and 0 <= eid <= EARNINGS_EXCLUSION_DAYS_ST:
            continue
        price, atr = r["price"], r["atr14"]
        if pd.isna(atr) or atr <= 0:
            continue
        # Setup A: pullback in an uptrend
        if (r["golden"] and r["pct_vs_sma200"] > 0 and 30 <= r["rsi14"] <= 48
                and -0.08 <= r["pct_vs_sma20"] <= -0.02 and r["ret_3m"] > 0):
            stop = round(min(r["low_10d"], price - 1.5 * atr), 2)
            risk = price - stop
            if risk <= 0 or risk / price > 0.10:
                continue
            target = round(max(r["prior_20d_high"], price + 2 * risk), 2)
            strength = (48 - r["rsi14"]) / 18 * 30 + min(r["ret_3m"], 0.3) / 0.3 * 40 + r.get("score_pct", 50) / 100 * 30
            setups.append({
                "ticker": t, "setup": "pullback", "strength": round(float(strength), 1),
                "entry": price, "stop": stop, "target": target,
                "risk_pct": round(risk / price, 4), "reward_risk": round((target - price) / risk, 2),
                "why": (f"Uptrend (price {r['pct_vs_sma200']:+.1%} vs 200-day, 50>200), RSI {r['rsi14']:.0f} "
                        f"after a {r['pct_vs_sma20']:+.1%} dip below the 20-day; 3-mo return {r['ret_3m']:+.1%}."),
            })
        # Setup B: breakout on volume
        elif (pd.notna(r["prior_20d_high"]) and price > r["prior_20d_high"]
              and r["vol_ratio"] >= 1.5 and r["rsi14"] < 75 and r["pct_vs_sma200"] > 0):
            stop = round(max(price - 2 * atr, r["prior_20d_high"] * 0.97), 2)
            risk = price - stop
            if risk <= 0 or risk / price > 0.10:
                continue
            target = round(price + 2 * risk, 2)
            strength = min(r["vol_ratio"], 3) / 3 * 40 + (75 - r["rsi14"]) / 30 * 20 + min(r["ret_1m"], 0.2) / 0.2 * 20 + r.get("score_pct", 50) / 100 * 20
            setups.append({
                "ticker": t, "setup": "breakout", "strength": round(float(strength), 1),
                "entry": price, "stop": stop, "target": target,
                "risk_pct": round(risk / price, 4), "reward_risk": round((target - price) / risk, 2),
                "why": (f"Closed above the prior 20-day high ({r['prior_20d_high']:.2f}) on {r['vol_ratio']:.1f}x "
                        f"average volume; RSI {r['rsi14']:.0f} leaves room before overbought."),
            })
    setups.sort(key=lambda s: s["strength"], reverse=True)
    # diversify: one per setup type if both exist, else best two
    chosen, seen_setup = [], set()
    for s in setups:
        if s["setup"] in seen_setup and len(setups) > 2:
            continue
        chosen.append(s)
        seen_setup.add(s["setup"])
        if len(chosen) == ST_TOP_N:
            break
    if len(chosen) < ST_TOP_N:
        for s in setups:
            if s not in chosen:
                chosen.append(s)
            if len(chosen) == ST_TOP_N:
                break
    return chosen


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:  # noqa: BLE001
        pass
    return v


def row_dict(r: pd.Series, keys: list[str]) -> dict:
    return {k: _clean(r.get(k)) for k in keys}


UNIVERSE_KEYS = [
    "name", "sector", "industry", "in_sp500", "price", "as_of", "chg_1d", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "sma20", "sma50", "sma200", "pct_vs_sma50", "pct_vs_sma200", "golden", "rsi14", "atr_pct", "hi52", "lo52",
    "pct_from_hi52", "vol_ratio", "vol_ann", "market_cap", "fpe", "tpe", "peg", "ev_ebitda", "roe", "op_margin",
    "net_margin", "debt_equity", "fcf_yield", "rev_growth", "eps_growth", "fwd_eps_growth", "rec_mean", "rec_key",
    "n_analysts", "target_upside", "div_yield", "beta", "next_earnings", "earnings_in_days", "has_fundamentals",
    "fundamentals_age_days", "f_quality", "f_value", "f_growth", "f_momentum", "f_analyst", "factors_present",
    "score", "rank", "score_pct", "lt_signal", "trend",
]


def _pct(x, digits=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}%}"


def lt_reasons(r: pd.Series, sector_med_fpe: float | None) -> list[str]:
    out = []
    if pd.notna(r.get("roe")):
        out.append(f"Quality: ROE {r['roe']:.0%}, operating margin {r['op_margin']:.0%}" +
                   (f", FCF yield {r['fcf_yield']:.1%}" if pd.notna(r.get("fcf_yield")) else "") + ".")
    if pd.notna(r.get("fpe")):
        s = f"Value: forward P/E {r['fpe']:.1f}"
        if sector_med_fpe:
            s += f" vs {r['sector']} median {sector_med_fpe:.1f}"
        if pd.notna(r.get("peg")):
            s += f", PEG {r['peg']:.2f}"
        out.append(s + ".")
    if pd.notna(r.get("rev_growth")) or pd.notna(r.get("eps_growth")):
        out.append(f"Growth: revenue {_pct(r.get('rev_growth'), 0)} y/y, earnings {_pct(r.get('eps_growth'), 0)} y/y" +
                   (f", forward EPS implies {_pct(r.get('fwd_eps_growth'), 0)}" if pd.notna(r.get("fwd_eps_growth")) else "") + ".")
    out.append(f"Momentum: 6-mo {_pct(r['ret_6m'])}, 12-mo {_pct(r['ret_12m'])}, price {_pct(r['pct_vs_sma200'])} vs 200-day, "
               f"{'50-day above 200-day' if r['golden'] else '50-day below 200-day'}, {_pct(r['pct_from_hi52'])} from 52-wk high.")
    if pd.notna(r.get("rec_mean")):
        out.append(f"Street: {int(r['n_analysts'])} analysts, mean rating {r['rec_mean']:.2f} ({r.get('rec_key') or 'n/a'}), "
                   f"mean target implies {_pct(r.get('target_upside'))}.")
    if r.get("earnings_in_days") is not None and pd.notna(r.get("earnings_in_days")) and 0 <= r["earnings_in_days"] <= 14:
        out.append(f"Heads-up: earnings expected {r['next_earnings']} ({int(r['earnings_in_days'])} days).")
    return out


def market_regime(bench: dict[str, pd.DataFrame], df: pd.DataFrame) -> dict:
    out = {}
    spy = bench.get("SPY")
    if spy is not None and len(spy) > 200:
        t = compute_technicals(spy)
        out.update({
            "spy_price": t["price"], "spy_chg_1d": _clean(t["chg_1d"]), "spy_ret_1m": _clean(t["ret_1m"]),
            "spy_ret_3m": _clean(t["ret_3m"]), "spy_ret_12m": _clean(t["ret_12m"]),
            "spy_vs_sma50": _clean(t["pct_vs_sma50"]), "spy_vs_sma200": _clean(t["pct_vs_sma200"]),
            "spy_pct_from_hi52": _clean(t["pct_from_hi52"]), "spy_rsi14": _clean(t["rsi14"]),
        })
    vix = bench.get("^VIX")
    if vix is not None and len(vix) > 5:
        out["vix"] = round(float(vix["Close"].iloc[-1]), 2)
    sp = df[df["in_sp500"]]
    out["breadth_above_sma50"] = _clean((sp["pct_vs_sma50"] > 0).mean())
    out["breadth_above_sma200"] = _clean((sp["pct_vs_sma200"] > 0).mean())
    out["pct_golden"] = _clean(sp["golden"].mean())
    out["median_chg_1d"] = _clean(sp["chg_1d"].median())
    # simple regime label
    a200, b200 = out.get("spy_vs_sma200"), out.get("breadth_above_sma200")
    if a200 is None:
        out["regime"] = "unknown"
    elif a200 > 0 and (b200 or 0) > 0.55:
        out["regime"] = "bull"
    elif a200 < 0 and (b200 or 1) < 0.45:
        out["regime"] = "bear"
    else:
        out["regime"] = "mixed"
    sector_ret = sp.groupby("sector")["ret_1m"].median().sort_values(ascending=False)
    out["sector_1m_median_returns"] = {k: _clean(v) for k, v in sector_ret.items()}
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "history").mkdir(exist_ok=True)

    universe = get_universe()
    tickers = universe["ticker"].tolist()

    log("Fetching prices…")
    prices = fetch_prices(tickers)
    bench = fetch_prices(BENCHMARKS)
    log(f"Prices for {len(prices)} tickers; benchmarks: {list(bench)}")

    log("Fetching fundamentals…")
    fund = fetch_fundamentals(tickers, DATA_DIR / "fundamentals_cache.json")

    log("Computing technicals…")
    tech = {}
    for t, px in prices.items():
        try:
            tech[t] = compute_technicals(px)
        except Exception as e:  # noqa: BLE001
            log(f"technicals failed for {t}: {e!r}")

    df = build_frame(universe, tech, fund)
    df = score(df)
    df["lt_signal"] = df.apply(lt_signal, axis=1)
    df["trend"] = df.apply(trend_label, axis=1)

    st = scan_short_term(df)
    st_tickers = {s["ticker"] for s in st}
    lt = pick_long_term(df, exclude=st_tickers)

    sector_med_fpe = df[df["in_sp500"]].groupby("sector")["fpe"].median().to_dict()

    def pick_block(t: str, kind: str, extra: dict | None = None) -> dict:
        r = df.loc[t]
        d = {"ticker": t, "kind": kind}
        d.update(row_dict(r, ["name", "sector", "industry", "price", "chg_1d", "score", "rank", "score_pct",
                              "f_quality", "f_value", "f_growth", "f_momentum", "f_analyst",
                              "fpe", "peg", "roe", "op_margin", "fcf_yield", "rev_growth", "eps_growth",
                              "ret_1m", "ret_3m", "ret_6m", "ret_12m", "pct_vs_sma50", "pct_vs_sma200", "golden",
                              "rsi14", "pct_from_hi52", "hi52", "market_cap", "rec_mean", "rec_key", "n_analysts",
                              "target_upside", "div_yield", "beta", "vol_ann", "next_earnings", "earnings_in_days",
                              "trend", "lt_signal"]))
        d["reasons"] = lt_reasons(r, sector_med_fpe.get(r["sector"]))
        if extra:
            d.update(extra)
        return d

    picks = [pick_block(t, "long-term") for t in lt]
    for s in st:
        extra = {k: s[k] for k in ["setup", "strength", "entry", "stop", "target", "risk_pct", "reward_risk"]}
        extra["trade_plan"] = s["why"]
        picks.append(pick_block(s["ticker"], "short-term", extra))

    regime = market_regime(bench, df)

    as_of = df["as_of"].mode().iloc[0] if "as_of" in df and not df["as_of"].isna().all() else date.today().isoformat()
    report = {
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": int(df["in_sp500"].sum()),
        "with_fundamentals": int(df["has_fundamentals"].sum()),
        "runtime_seconds": round(time.time() - t0, 1),
        "methodology": {
            "long_term": "Composite of quality 25%, value 20%, growth 20%, momentum 20%, analyst 15% "
                         "(percentile ranks across S&P 500; value ranked within sector). Top 8, max 2 per sector, "
                         ">=5 analysts, falling-knife guard.",
            "short_term": "Pullback-in-uptrend or volume breakout setups; no earnings within 10 days; "
                          "stop = max(1.5-2x ATR, 10-day low); target = 2R.",
            "weights": FACTOR_WEIGHTS,
        },
        "market": regime,
        "picks": picks,
        "top25_by_score": [
            row_dict(df.loc[t], ["name", "sector", "price", "score", "rank", "trend", "lt_signal"]) | {"ticker": t}
            for t in df[df["in_sp500"]].sort_values("score", ascending=False).head(25).index
        ],
        "sector_median_fpe": {k: _clean(v) for k, v in sector_med_fpe.items()},
    }

    universe_out = {t: row_dict(r, UNIVERSE_KEYS) for t, r in df.iterrows()}

    (DATA_DIR / "report.json").write_text(json.dumps(report, indent=1))
    (DATA_DIR / "universe.json").write_text(json.dumps({"as_of": as_of, "tickers": universe_out}, separators=(",", ":")))
    (DATA_DIR / "history" / f"{as_of}.json").write_text(json.dumps(
        {"as_of": as_of, "market": {k: regime.get(k) for k in ["regime", "spy_price", "spy_chg_1d", "vix"]},
         "picks": [{k: p.get(k) for k in ["ticker", "kind", "price", "score", "setup", "entry", "stop", "target"]} for p in picks]},
        indent=1))
    log(f"Done in {report['runtime_seconds']}s. Picks: {[p['ticker'] for p in picks]}")
    return report


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
