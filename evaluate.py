"""
Holdings evaluator (stdlib only, so it runs anywhere — including the Claude task sandbox).

Inputs
------
universe.json  - produced by engine.py (data/universe.json)
holdings.json  - list of positions from the dashboard database:
                 [{"id": "...", "ticker": "AAPL", "shares": 10, "avg_cost": 180.5,
                   "opened": "2026-06-01", "type": "long-term"|"short-term",
                   "stop": null, "target": null, "notes": ""}]

Output
------
{"holdings_eval": {<id>: {...}}, "quotes": {<TICKER>: {...}}, "totals": {...}}

Rules (deliberately simple and auditable)
-----------------------------------------
Long-term positions
  SELL  loss >= 20% from cost AND downtrend; or sell-zone signal while under water;
        or downtrend with composite score in bottom quartile
  TRIM  trim-zone signal (>25% above 200-day or RSI > 80); or gain >= 50% with score below median
  ADD   add-zone signal (top-15% score, uptrend, RSI 35-55)
  HOLD  otherwise
Short-term positions
  SELL  price <= stop (default stop = cost - 7%); price >= target (default = cost + 14%);
        or held longer than 28 calendar days (time stop)
  TRIM  >= 60% of the way to target and RSI > 70
  HOLD  otherwise, with a suggested trailing stop
Flags: earnings within 7 days, stale fundamentals, ticker not in universe.

Usage (paste-and-run, no argparse):
    import evaluate; out = evaluate.evaluate("data/universe.json", "holdings.json")
or  python evaluate.py   (reads the two paths from STOCKS_UNIVERSE / STOCKS_HOLDINGS env vars)
"""
from __future__ import annotations

import json
import os
from datetime import date

QUOTE_KEYS = ["name", "sector", "price", "as_of", "chg_1d", "ret_1m", "ret_3m", "trend", "lt_signal", "score",
              "rank", "score_pct", "rsi14", "pct_vs_sma50", "pct_vs_sma200", "pct_from_hi52", "atr_pct",
              "next_earnings", "earnings_in_days", "fundamentals_age_days", "f_quality", "f_value", "f_growth",
              "f_momentum", "f_analyst"]


def _f(x, default=None):
    return default if x is None else x


def _pct(x):
    return "n/a" if x is None else f"{x:+.1%}"


def evaluate_one(h: dict, u: dict | None, today: date) -> dict:
    tkr = h["ticker"].upper()
    shares = float(_f(h.get("shares"), 0) or 0)
    cost = float(_f(h.get("avg_cost"), 0) or 0)
    kind = (h.get("type") or "long-term").lower()
    out = {"ticker": tkr, "type": kind, "shares": shares, "avg_cost": cost, "action": "hold",
           "confidence": "low", "reasons": [], "flags": []}
    if not u or u.get("price") is None:
        out["action"] = "no-data"
        out["flags"].append("No market data — if this ticker is not in the S&P 500, add it to extra_tickers.txt in the GitHub repo.")
        return out

    price = float(u["price"])
    out["price"] = price
    out["as_of"] = u.get("as_of")
    out["value"] = round(price * shares, 2)
    if cost > 0:
        out["pl_pct"] = round(price / cost - 1, 4)
        out["pl_usd"] = round((price - cost) * shares, 2)
    pl = out.get("pl_pct")
    trend = u.get("trend") or "unknown"
    sig = u.get("lt_signal") or "hold"
    spct = _f(u.get("score_pct"), 50)
    rsi = _f(u.get("rsi14"), 50)
    eid = u.get("earnings_in_days")
    fired = 0

    if kind == "short-term":
        stop = h.get("stop") or (cost * 0.93 if cost else None)
        target = h.get("target") or (cost * 1.14 if cost else None)
        held_days = None
        if h.get("opened"):
            try:
                held_days = (today - date.fromisoformat(str(h["opened"])[:10])).days
            except ValueError:
                held_days = None
        out.update({"stop": round(stop, 2) if stop else None, "target": round(target, 2) if target else None,
                    "held_days": held_days})
        if stop and price <= stop:
            out["action"] = "sell"; fired += 2
            out["reasons"].append(f"Stop hit: price {price:.2f} is at/below your stop {stop:.2f}.")
        elif target and price >= target:
            out["action"] = "sell"; fired += 2
            out["reasons"].append(f"Target reached: price {price:.2f} vs target {target:.2f} ({_pct(pl)}). Take the trade off.")
        elif held_days is not None and held_days > 28:
            out["action"] = "sell"; fired += 1
            out["reasons"].append(f"Time stop: short-term trade open {held_days} days (limit 28). Exit or re-underwrite it as long-term.")
        elif target and cost and price >= cost + 0.6 * (target - cost) and rsi > 70:
            out["action"] = "trim"; fired += 1
            out["reasons"].append(f"60%+ of the way to target with RSI {rsi:.0f}: sell half, move the stop to breakeven.")
        else:
            atr_pct = _f(u.get("atr_pct"), 0.02)
            trail = max(stop or 0, price * (1 - 2 * atr_pct))
            out["suggested_stop"] = round(trail, 2)
            out["reasons"].append(f"Trade intact ({_pct(pl)}). Suggested trailing stop {trail:.2f} (2x ATR).")
    else:
        if pl is not None and pl <= -0.20 and trend == "downtrend":
            out["action"] = "sell"; fired += 2
            out["reasons"].append(f"Down {_pct(pl)} from cost in a confirmed downtrend (price below 200-day, 50-day below 200-day).")
        elif sig == "sell-zone" and (pl is None or pl < 0):
            out["action"] = "sell"; fired += 2
            out["reasons"].append(f"Sell-zone: downtrend and composite score in the bottom 30% (score pct {spct:.0f}).")
        elif trend == "downtrend" and spct < 25:
            out["action"] = "sell"; fired += 1
            out["reasons"].append(f"Downtrend with a bottom-quartile composite score ({spct:.0f}th pct). Thesis weakening.")
        elif sig == "trim-zone":
            out["action"] = "trim"; fired += 1
            a200 = u.get("pct_vs_sma200")
            out["reasons"].append(f"Overextended: {_pct(a200)} above the 200-day, RSI {rsi:.0f} ({_pct(pl)} vs cost). Trim 20-30% and let the rest run.")
        elif pl is not None and pl >= 0.50 and spct < 50:
            out["action"] = "trim"; fired += 1
            out["reasons"].append(f"Up {_pct(pl)} but composite score has slipped below median ({spct:.0f}th pct). Take some profit.")
        elif sig == "add-zone":
            out["action"] = "add"; fired += 1
            out["reasons"].append(f"Add-zone: top-15% composite score ({spct:.0f}th pct), uptrend, RSI {rsi:.0f} — a pullback in a leader.")
        else:
            out["reasons"].append(f"Hold: {trend.replace('-', ' ')}, composite score {spct:.0f}th pct, {_pct(pl)} vs cost.")

    if eid is not None and 0 <= eid <= 7:
        out["flags"].append(f"Earnings {u.get('next_earnings')} ({eid} days) — expect a gap; size accordingly.")
    if (u.get("fundamentals_age_days") or 0) > 3:
        out["flags"].append(f"Fundamentals are {u['fundamentals_age_days']} days old (Yahoo fetch failed recently).")
    if u.get("pct_from_hi52") is not None and u["pct_from_hi52"] <= -0.25:
        out["flags"].append(f"{_pct(u['pct_from_hi52'])} from its 52-week high.")
    out["confidence"] = "high" if fired >= 2 else "medium" if fired == 1 else "low"
    return out


def evaluate(universe_path: str, holdings_path: str, today: date | None = None) -> dict:
    today = today or date.today()
    uni = json.load(open(universe_path))
    tickers = uni.get("tickers", uni)
    holdings = json.load(open(holdings_path))
    if isinstance(holdings, dict):
        holdings = [dict(v, id=k) for k, v in holdings.items()]
    evals, quotes = {}, {}
    tot_value = tot_cost = 0.0
    for h in holdings:
        tkr = str(h.get("ticker", "")).upper().replace(".", "-")
        u = tickers.get(tkr)
        e = evaluate_one(dict(h, ticker=tkr), u, today)
        evals[h.get("id") or tkr] = e
        if u:
            quotes[tkr] = {k: u.get(k) for k in QUOTE_KEYS}
            tot_value += e.get("value") or 0
            tot_cost += (e["avg_cost"] or 0) * (e["shares"] or 0)
    actions = {}
    for e in evals.values():
        actions[e["action"]] = actions.get(e["action"], 0) + 1
    totals = {"positions": len(evals), "market_value": round(tot_value, 2), "cost_basis": round(tot_cost, 2),
              "pl_usd": round(tot_value - tot_cost, 2),
              "pl_pct": round(tot_value / tot_cost - 1, 4) if tot_cost else None, "actions": actions,
              "as_of": uni.get("as_of")}
    return {"holdings_eval": evals, "quotes": quotes, "totals": totals}


if __name__ == "__main__":
    res = evaluate(os.environ.get("STOCKS_UNIVERSE", "data/universe.json"),
                   os.environ.get("STOCKS_HOLDINGS", "holdings.json"))
    print(json.dumps(res, indent=1))
