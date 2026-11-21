#!/usr/bin/env python3
"""
Stock Scanner — daily swing & intraday scanner for Indian stocks (NSE).

What it does every trading day (run automatically by GitHub Actions):
  1. Loads the Nifty 100 (large cap), Midcap 150 and Smallcap 250 lists.
  2. Downloads ~3 years of daily prices for every stock.
  3. Looks for 3 swing setups (Breakout, Pullback, Momentum) and 1 intraday setup.
  4. BACKTESTS each setup on the past data to measure how often it actually
     hit its target — overall and for each individual stock.
  5. Reads recent news headlines for the best candidates and scores them.
  6. Tracks every pick it has ever made and records whether it worked.
  7. Writes everything to docs/data.json, which the phone app displays.

Personal, educational use only. Not investment advice.
"""
import io
import json
import os
import re
import sys
import time
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DIR = os.path.join(ROOT, "docs")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# ───────────────────────── Settings ─────────────────────────
CFG = {
    "min_price": 50,            # ignore stocks below ₹50
    "min_value_cr": 5,          # min average daily traded value (₹ crore) — liquidity
    "hold_days": 3,             # max holding period for swing trades (trading days)
    "top_swing": 15,
    "top_intraday": 10,
    "news_candidates": 40,      # how many top candidates get a news check
    "prior_weight": 6,          # how much a stock's own history is blended with the setup's overall history
    "target_options": [0.75, 1.0, 1.25, 1.5, 2.0],   # × ATR
    "stop_options": [1.0, 1.5],                      # × ATR
}

INDEX_FILES = {
    "Large cap": "ind_nifty100list.csv",
    "Mid cap": "ind_niftymidcap150list.csv",
    "Small cap": "ind_niftysmallcap250list.csv",
}
INDEX_BASES = [
    "https://archives.nseindia.com/content/indices/",
    "https://www.niftyindices.com/IndexConstituent/",
]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

SETUP_INFO = {
    "Breakout": "Price broke above its 20-day high on strong volume while in an uptrend.",
    "Pullback": "A stock in a strong uptrend dipped to its 20-day average and bounced back up.",
    "Momentum": "Stock is beating Nifty, closed near the day's high, and volume is rising.",
}
INTRADAY_INFO = ("Quiet, tight day in an uptrend. If price crosses today's high tomorrow, "
                 "buyers are taking control — a quick intraday move often follows.")


def log(*a):
    print(*a, flush=True)


# ───────────────────────── Helpers ─────────────────────────
def r2(x, n=2):
    try:
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else round(x, n)
    except Exception:
        return None


def clean(o):
    """Make data JSON-safe (no NaN, no numpy types)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return r2(o, 4)
    if o is None or (not isinstance(o, str) and pd.isna(o)):
        return None
    return o if isinstance(o, str) else str(o)


def clean_name(name):
    name = re.sub(r"\b(Limited|Ltd\.?|Ltd)\b", "", str(name), flags=re.I)
    return re.sub(r"\s+", " ", name).strip(" .,")


# ───────────────────────── 1. Stock list ─────────────────────────
def load_universe():
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "universe.csv")
    frames = []
    for cap, fname in INDEX_FILES.items():
        df = None
        for base in INDEX_BASES:
            try:
                r = requests.get(base + fname, headers=UA, timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                break
            except Exception as e:
                log("  list download failed:", base + fname, e)
        if df is None:
            continue
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Company Name": "name", "Industry": "sector", "Symbol": "symbol"})
        df = df[["symbol", "name", "sector"]].copy()
        df["cap"] = cap
        frames.append(df)

    if len(frames) == len(INDEX_FILES):
        u = pd.concat(frames).drop_duplicates("symbol")
        u.to_csv(cache, index=False)
        return u
    if os.path.exists(cache):
        log("  using saved stock list")
        return pd.read_csv(cache)
    if frames:
        return pd.concat(frames).drop_duplicates("symbol")
    sys.exit("Could not download the stock list and no saved copy exists.")


# ───────────────────────── 2. Prices ─────────────────────────
def _fix_index(df):
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


def download_prices(tickers, period="3y"):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
                break
            except Exception as e:
                log("  download error, retrying:", e)
                time.sleep(5 * (attempt + 1))
        if data is None or data.empty:
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
                if len(df) > 260:
                    out[t] = _fix_index(df.copy())
            except Exception:
                pass
        log(f"  prices: {min(i + 80, len(tickers))}/{len(tickers)} requested, {len(out)} ok")
        time.sleep(1)
    return out


# ───────────────────────── 3. Indicators & setups ─────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df, nifty_close):
    c = df.Close
    df["ema20"], df["ema50"], df["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["rsi"] = rsi(c)
    df["atr"] = atr(df)
    df["vol20"] = df.Volume.rolling(20).mean()
    df["vr"] = df.Volume / df.vol20
    df["hi20"] = df.High.rolling(20).max().shift(1)
    df["rng"] = df.High - df.Low
    df["nr7"] = df.rng <= df.rng.rolling(7).min()
    df["inside"] = (df.High <= df.High.shift()) & (df.Low >= df.Low.shift())
    n = nifty_close.reindex(df.index).ffill()
    df["rs20"] = c.pct_change(20) - n.pct_change(20)
    df["ret5"] = c.pct_change(5)
    df["ret20"] = c.pct_change(20)
    df["clv"] = (c - df.Low) / df.rng.replace(0, np.nan)
    df["value_cr"] = (c * df.Volume).rolling(20).mean() / 1e7
    return df


def swing_signals(df):
    c = df.Close
    up = (c > df.ema50) & (df.ema50 > df.ema200)
    liquid = df.value_cr >= CFG["min_value_cr"]
    breakout = up & (c > df.hi20) & (df.vr > 1.5) & df.rsi.between(55, 75) & (df.clv > 0.6)
    pullback = (up & (df.ema20 > df.ema50) & (df.Low <= df.ema20 * 1.01) & (c > df.ema20)
                & (c > df.Open) & df.rsi.between(42, 62))
    momentum = (up & (c > df.ema20) & (df.rs20 > 0.03) & df.rsi.between(58, 72) & (df.clv > 0.75)
                & (df.vr > 1.1) & (df.ret5 > 0) & ~breakout)
    return {
        "Breakout": (breakout & liquid).fillna(False).astype(bool),
        "Pullback": (pullback & liquid).fillna(False).astype(bool),
        "Momentum": (momentum & liquid).fillna(False).astype(bool),
    }


def intraday_signal(df):
    c = df.Close
    sig = ((df.nr7 | df.inside) & (c > df.ema20) & (df.ema20 > df.ema50) & (df.clv > 0.55)
           & (df.value_cr >= CFG["min_value_cr"] * 2) & df.rsi.between(50, 70))
    return sig.fillna(False).astype(bool)


# ───────────────────────── 4. Backtest ─────────────────────────
def collect_events(df, sig, hold):
    """For each past signal: entry at next day's open, then the next `hold` bars."""
    pos = np.flatnonzero(sig.values[:-hold - 1]) if len(sig) > hold + 1 else np.array([], int)
    pos = pos[pos > 200]                       # need warm-up for EMA200
    if len(pos) == 0:
        return None
    O, H, L, C = (df[k].values for k in ("Open", "High", "Low", "Close"))
    a = df.atr.values
    idx = pos[:, None] + 1 + np.arange(hold)[None, :]
    return {"entry": O[pos + 1], "ref": C[pos], "atr": a[pos], "O": O[idx], "H": H[idx], "L": L[idx], "C": C[idx]}


def simulate(ev, t_mult, s_mult):
    """Same rules as the app: levels are set from the signal-day close; next day we buy at the open only
    if it is not above the 'buy up to' price (no chasing gaps). Stop is checked before target (conservative).
    Returns (taken, win, return, days) arrays."""
    ref, a = ev["ref"], ev["atr"]
    tgt, sl = ref + t_mult * a, ref - s_mult * a
    taken = (ev["entry"] <= ref + 0.25 * a) & (ev["entry"] > sl)
    ev = {k: v[taken] for k, v in ev.items()}
    entry, a, tgt, sl = ev["entry"], ev["atr"], tgt[taken], sl[taken]
    n, h = ev["H"].shape
    res, win = np.full(n, np.nan), np.zeros(n, bool)
    days, done = np.full(n, h, float), np.zeros(n, bool)
    for d in range(h):
        O, H, L = ev["O"][:, d], ev["H"][:, d], ev["L"][:, d]
        gap_sl = ~done & (O <= sl) & (d > 0)
        hit_sl = ~done & ~gap_sl & (L <= sl)
        gap_t = ~done & ~gap_sl & ~hit_sl & (O >= tgt) & (d > 0)
        hit_t = ~done & ~gap_sl & ~hit_sl & ~gap_t & (H >= tgt)
        res[gap_sl] = O[gap_sl] / entry[gap_sl] - 1
        res[hit_sl] = sl[hit_sl] / entry[hit_sl] - 1
        res[gap_t] = O[gap_t] / entry[gap_t] - 1
        res[hit_t] = tgt[hit_t] / entry[hit_t] - 1
        w = gap_t | hit_t
        win |= w
        newly = gap_sl | hit_sl | w
        days[newly] = d + 1
        done |= newly
    rest = ~done
    res[rest] = ev["C"][rest, -1] / entry[rest] - 1
    return taken, win, res, days


def simulate_intraday(ev, trig_pct, t_mult, s_mult):
    """Next day: buy if price crosses trigger; win only if target reached AND stop never touched (conservative)."""
    trig = ev["prev_high"] * (1 + trig_pct)
    a = ev["atr"]
    tgt, sl = trig + t_mult * a, trig - s_mult * a
    O, H, L, C = ev["O"][:, 0], ev["H"][:, 0], ev["L"][:, 0], ev["C"][:, 0]
    fill = np.maximum(trig, O)
    entered = H >= trig
    win = entered & (H >= tgt) & (L > sl)
    loss = entered & (L <= sl)
    res = np.where(win, tgt / fill - 1, np.where(loss, sl / fill - 1, C / fill - 1))
    return entered, win, res


def merge_events(events):
    keys = events[0].keys()
    return {k: np.concatenate([e[k] for e in events]) for k in keys}


# ───────────────────────── 5. News ─────────────────────────
POS_WORDS = ["surge", "surges", "jump", "jumps", "rally", "rallies", "gain", "gains", "rise", "rises", "soar",
             "soars", "climb", "climbs", "record", "high", "beat", "beats", "upgrade", "upgrades", "buy",
             "order", "orders", "wins", "bags", "secures", "approval", "approves", "expansion", "strong",
             "outperform", "dividend", "bonus", "acquire", "acquisition", "partnership", "growth", "profit up",
             "net profit rises", "target price raised", "bullish", "breakout", "contract", "launch", "multibagger"]
NEG_WORDS = ["fall", "falls", "drop", "drops", "slump", "slumps", "plunge", "plunges", "decline", "declines",
             "loss", "losses", "downgrade", "downgrades", "sell", "fraud", "probe", "raid", "penalty", "fine",
             "weak", "miss", "misses", "cut", "cuts", "resign", "resigns", "default", "pledge", "ban", "crash",
             "tumble", "tumbles", "slide", "slides", "lawsuit", "sebi order", "investigation", "bearish",
             "lower", "slashes", "warning", "shutdown", "strike", "rejects", "rejected"]


def score_title(t):
    t = " " + re.sub(r"[^a-z0-9 ]", " ", t.lower()) + " "
    p = sum(1 for w in POS_WORDS if f" {w} " in t)
    n = sum(1 for w in NEG_WORDS if f" {w} " in t)
    return (p > n) - (n > p)


def fetch_news(company):
    q = requests.utils.quote(f'"{clean_name(company)}" stock when:7d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        root = ET.fromstring(r.content)
    except Exception as e:
        log("  news failed:", company, e)
        return {"score": 0, "label": "No news", "items": []}
    items = []
    for it in list(root.iter("item"))[:8]:
        title = it.findtext("title") or ""
        src = it.findtext("source") or ""
        if src and title.endswith(" - " + src):
            title = title[: -len(" - " + src)]
        items.append({"title": title, "link": it.findtext("link") or "", "source": src,
                      "date": it.findtext("pubDate") or "", "s": score_title(title)})
    if not items:
        return {"score": 0, "label": "No news", "items": []}
    sc = float(np.mean([i["s"] for i in items]))
    label = "Positive" if sc >= 0.25 else "Negative" if sc <= -0.25 else "Neutral"
    return {"score": round(sc, 2), "label": label, "items": items[:3]}


# ───────────────────────── 6. Results tracking ─────────────────────────
PICK_COLS = ["date", "type", "symbol", "name", "setup", "entry", "buy_upto", "target", "sl", "hold_days",
             "status", "exit_date", "ret_pct"]


def update_results(prices, today_str):
    path = os.path.join(DATA_DIR, "picks.csv")
    picks = pd.read_csv(path, dtype={"date": str, "exit_date": str}) if os.path.exists(path) else pd.DataFrame(columns=PICK_COLS)
    for i, p in picks[picks.status == "Open"].iterrows():
        df = prices.get(p.symbol + ".NS")
        if df is None:
            continue
        after = df[df.index > pd.Timestamp(p.date)]
        if len(after) == 0:
            continue
        if p.type == "intraday":
            b = after.iloc[0]
            fill = max(p.entry, b.Open)
            if b.High < p.entry:
                st, ret = "Not triggered", np.nan
            elif b.Low <= p.sl:
                st, ret = "Stop loss", p.sl / fill - 1
            elif b.High >= p.target:
                st, ret = "Target hit", p.target / fill - 1
            else:
                st, ret = "Closed at 3:30", b.Close / fill - 1
            picks.loc[i, ["status", "exit_date", "ret_pct"]] = [st, after.index[0].strftime("%Y-%m-%d"), r2(ret * 100)]
            continue
        entry = after.Open.iloc[0]
        buy_upto = p.buy_upto if "buy_upto" in p and not pd.isna(p.buy_upto) else p.entry * 1.01
        if entry > buy_upto or entry <= p.sl:
            why = "Skipped (gap up)" if entry > buy_upto else "Skipped (gap down)"
            picks.loc[i, ["status", "exit_date", "ret_pct"]] = [why, after.index[0].strftime("%Y-%m-%d"), None]
            continue
        st = None
        for d in range(min(len(after), int(p.hold_days))):
            b = after.iloc[d]
            if b.Low <= p.sl:
                st, ret, ex = "Stop loss", min(b.Open, p.sl) / entry - 1 if d else p.sl / entry - 1, after.index[d]
                break
            if b.High >= p.target:
                st, ret, ex = "Target hit", max(b.Open, p.target) / entry - 1 if d else p.target / entry - 1, after.index[d]
                break
        if st is None and len(after) >= int(p.hold_days):
            b = after.iloc[int(p.hold_days) - 1]
            st, ret, ex = "Time exit", b.Close / entry - 1, after.index[int(p.hold_days) - 1]
        if st:
            picks.loc[i, ["status", "exit_date", "ret_pct"]] = [st, ex.strftime("%Y-%m-%d"), r2(ret * 100)]
    return picks, path


def results_summary(picks):
    out = {}
    for typ in ("swing", "intraday"):
        p = picks[picks.type == typ]
        closed = p[p.status.isin(["Target hit", "Stop loss", "Time exit", "Closed at 3:30"])]
        n = len(closed)
        wins = int((closed.status == "Target hit").sum())
        profitable = int((closed.ret_pct.astype(float) > 0).sum())
        out[typ] = {
            "closed": n, "target_hits": wins, "profitable": profitable,
            "target_rate": r2(wins / n * 100, 1) if n else None,
            "profit_rate": r2(profitable / n * 100, 1) if n else None,
            "avg_ret": r2(closed.ret_pct.astype(float).mean()) if n else None,
            "open": int((p.status == "Open").sum()),
        }
    recent = picks.sort_values("date", ascending=False).head(60)
    out["recent"] = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in recent.to_dict("records")]
    return out


# ───────────────────────── Main ─────────────────────────
def main(universe=None, prices=None):
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    hold = CFG["hold_days"]

    log("1) Loading stock list…")
    if universe is None:
        universe = load_universe()
    universe = universe.dropna(subset=["symbol"])
    meta = {r.symbol: r for r in universe.itertuples()}
    log(f"   {len(universe)} stocks")

    log("2) Downloading prices…")
    if prices is None:
        prices = download_prices([s + ".NS" for s in universe.symbol] + ["^NSEI"])
        try:
            import yfinance as yf
            vix = yf.download("^INDIAVIX", period="1mo", progress=False, auto_adjust=True)
            vix_last = float(vix["Close"].dropna().values.ravel()[-1])
        except Exception:
            vix_last = None
    else:
        vix_last = None
    nifty = prices.pop("^NSEI", None)
    if nifty is None:
        sys.exit("Could not download Nifty 50 data.")
    nifty_c = nifty.Close
    last_date = nifty.index[-1]
    today_str = last_date.strftime("%Y-%m-%d")

    log("3) Indicators, setups & backtest…")
    swing_events = {k: [] for k in SETUP_INFO}
    swing_owner = {k: [] for k in SETUP_INFO}
    intra_events, intra_owner = [], []
    today_swing, today_intra = [], []
    sector_rows = []

    for t, df in prices.items():
        sym = t[:-3]
        m = meta.get(sym)
        if m is None:
            continue
        df = add_indicators(df, nifty_c)
        prices[t] = df
        last = df.iloc[-1]
        if df.index[-1] < last_date - pd.Timedelta(days=5):
            continue                                   # stale / suspended
        sector_rows.append({"sector": m.sector, "ret20": last.ret20, "ret5": last.ret5, "rs20": last.rs20})

        for name, sig in swing_signals(df).items():
            ev = collect_events(df, sig, hold)
            if ev:
                swing_events[name].append(ev)
                swing_owner[name].append(np.full(len(ev["entry"]), sym, dtype=object))
            if bool(sig.iloc[-1]) and last.Close >= CFG["min_price"]:
                today_swing.append((sym, name))

        isig = intraday_signal(df)
        ev = collect_events(df, isig, 1)
        if ev:
            pos = np.flatnonzero(isig.values[:-2]); pos = pos[pos > 200]
            ev["prev_high"] = df.High.values[pos]
            intra_events.append(ev)
            intra_owner.append(np.full(len(pos), sym, dtype=object))
        if bool(isig.iloc[-1]) and last.Close >= CFG["min_price"]:
            today_intra.append(sym)

    # Sector strength
    sec = pd.DataFrame(sector_rows)
    sector_list = []
    if len(sec):
        g = sec.groupby("sector").agg(ret20=("ret20", "median"), ret5=("ret5", "median"),
                                      breadth=("rs20", lambda s: float((s > 0).mean())), n=("ret20", "size"))
        g = g[g.n >= 3].sort_values("ret20", ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)
        sector_rank = g["rank"].to_dict()
        n_sectors = len(g)
        sector_list = [{"sector": s, "ret20": r2(r.ret20 * 100, 1), "ret5": r2(r.ret5 * 100, 1),
                        "breadth": r2(r.breadth * 100, 0), "stocks": int(r.n), "rank": int(r["rank"])}
                       for s, r in g.iterrows()]
    else:
        sector_rank, n_sectors = {}, 0

    # Choose target/stop per setup from the backtest
    setups_out, plan = {}, {}
    for name in SETUP_INFO:
        if not swing_events[name]:
            continue
        ev = merge_events(swing_events[name])
        owners = np.concatenate(swing_owner[name])
        best, best_any = None, None
        for tm in CFG["target_options"]:
            for sm in CFG["stop_options"]:
                _, win, res, days = simulate(ev, tm, sm)
                wr, exp = win.mean(), np.nanmean(res)
                cand = (exp, wr, tm, sm)
                if best_any is None or exp > best_any[0]:
                    best_any = cand
                # Priority: highest win rate among combos that still make money on average
                if exp > 0 and (best is None or (wr, exp) > (best[1], best[0])):
                    best = cand
        exp, wr, tm, sm = best or best_any
        taken, win, res, days = simulate(ev, tm, sm)
        per = pd.DataFrame({"sym": owners[taken], "win": win, "res": res, "days": days})
        stock_stats = per.groupby("sym").agg(n=("win", "size"), w=("win", "sum"), avg=("res", "mean"))
        plan[name] = {"tm": tm, "sm": sm, "global": float(wr), "stats": stock_stats,
                      "days_win": float(np.mean(days[win])) if win.any() else None}
        setups_out[name] = {"desc": SETUP_INFO[name], "target_atr": tm, "stop_atr": sm, "hold_days": hold,
                            "trades": int(len(win)), "win_rate": r2(wr * 100, 1), "avg_ret": r2(np.nanmean(res) * 100),
                            "profit_rate": r2(np.mean(res > 0) * 100, 1),
                            "avg_days_to_target": r2(plan[name]["days_win"], 1)}
        log(f"   {name}: {len(win)} past trades, target {tm}×ATR stop {sm}×ATR → win {wr:.1%}, avg {np.nanmean(res):.2%}")

    # Intraday plan
    intra_plan = None
    if intra_events:
        ev = merge_events(intra_events)
        owners = np.concatenate(intra_owner)
        best = None
        for tm in [0.3, 0.4, 0.5, 0.75]:
            entered, win, res = simulate_intraday(ev, 0.001, tm, 0.4)
            wr = win[entered].mean() if entered.any() else 0
            exp = np.nanmean(res[entered]) if entered.any() else -1
            if best is None or (wr >= 0.55 and exp > best[1]) or (best[0] < 0.55 and wr > best[0]):
                best = (wr, exp, tm)
        wr, exp, tm = best
        entered, win, res = simulate_intraday(ev, 0.001, tm, 0.4)
        per = pd.DataFrame({"sym": owners[entered], "win": win[entered]})
        intra_plan = {"tm": tm, "sm": 0.4, "global": float(wr),
                      "stats": per.groupby("sym").agg(n=("win", "size"), w=("win", "sum"))}
        setups_out["Intraday"] = {"desc": INTRADAY_INFO, "target_atr": tm, "stop_atr": 0.4, "hold_days": 0,
                                  "trades": int(entered.sum()), "win_rate": r2(wr * 100, 1),
                                  "avg_ret": r2(exp * 100)}
        log(f"   Intraday: {entered.sum()} past trades → win {wr:.1%}")

    K = CFG["prior_weight"]

    # Build today's swing candidates
    cands = []
    for sym, name in today_swing:
        if name not in plan:
            continue
        df, m, pl = prices[sym + ".NS"], meta[sym], plan[name]
        last = df.iloc[-1]
        st = pl["stats"].loc[sym] if sym in pl["stats"].index else None
        n = int(st.n) if st is not None else 0
        w = int(st.w) if st is not None else 0
        adj = (w + K * pl["global"]) / (n + K)
        c, a = float(last.Close), float(last.atr)
        tgt, sl = c + pl["tm"] * a, c - pl["sm"] * a
        srank = sector_rank.get(m.sector)
        why = [SETUP_INFO[name]]
        if last.vr >= 1.3:
            why.append(f"Volume is {last.vr:.1f}× its normal level.")
        if last.rs20 > 0:
            why.append(f"Beat Nifty by {last.rs20 * 100:.1f}% over the last month.")
        if srank and srank <= 5:
            why.append(f"Its sector ({m.sector}) is among the strongest right now (#{srank} of {n_sectors}).")
        cands.append({
            "symbol": sym, "name": m.name, "sector": m.sector, "cap": m.cap, "setup": name,
            "close": r2(c), "buy_upto": r2(c + 0.25 * a), "target": r2(tgt), "sl": r2(sl),
            "profit_pct": r2((tgt / c - 1) * 100), "risk_pct": r2((1 - sl / c) * 100),
            "rr": r2((tgt - c) / (c - sl)), "hold_days": hold,
            "usual_days": r2(pl["days_win"], 1),
            "stock_trades": n, "stock_wins": w, "stock_win": r2(w / n * 100, 0) if n else None,
            "setup_win": r2(pl["global"] * 100, 1), "win_score": r2(adj * 100, 1),
            "rsi": r2(last.rsi, 0), "vol_x": r2(last.vr, 1), "rs20": r2(last.rs20 * 100, 1),
            "sector_rank": srank, "why": why,
            "_sort": adj + (0.02 if srank and srank <= 5 else 0) + 0.005 * min(last.vr, 3),
        })
    # one entry per stock (best setup)
    best_by_sym = {}
    for cnd in cands:
        if cnd["symbol"] not in best_by_sym or cnd["_sort"] > best_by_sym[cnd["symbol"]]["_sort"]:
            best_by_sym[cnd["symbol"]] = cnd
    cands = sorted(best_by_sym.values(), key=lambda x: -x["_sort"])

    log(f"4) News check for top {min(len(cands), CFG['news_candidates'])} swing candidates…")
    checked = []
    for cnd in cands[:CFG["news_candidates"]]:
        nw = fetch_news(cnd["name"])
        cnd["news"] = nw
        cnd["_sort"] += 0.02 * nw["score"]
        if nw["label"] == "Positive":
            cnd["why"].append("Recent news looks positive.")
        checked.append(cnd)
        time.sleep(0.6)
    swing = [c for c in sorted(checked, key=lambda x: -x["_sort"]) if c["news"]["label"] != "Negative"]
    swing = swing[:CFG["top_swing"]]
    for i, c in enumerate(swing, 1):
        c["rank"] = i
        c.pop("_sort", None)

    # Intraday candidates
    intraday = []
    if intra_plan:
        for sym in today_intra:
            df, m = prices[sym + ".NS"], meta[sym]
            last = df.iloc[-1]
            st = intra_plan["stats"].loc[sym] if sym in intra_plan["stats"].index else None
            n = int(st.n) if st is not None else 0
            w = int(st.w) if st is not None else 0
            adj = (w + K * intra_plan["global"]) / (n + K)
            a = float(last.atr)
            trig = float(last.High) * 1.001
            tgt, sl = trig + intra_plan["tm"] * a, trig - intra_plan["sm"] * a
            intraday.append({
                "symbol": sym, "name": m.name, "sector": m.sector, "cap": m.cap,
                "close": r2(last.Close), "trigger": r2(trig), "target": r2(tgt), "sl": r2(sl),
                "profit_pct": r2((tgt / trig - 1) * 100), "risk_pct": r2((1 - sl / trig) * 100),
                "stock_trades": n, "stock_win": r2(w / n * 100, 0) if n else None,
                "setup_win": r2(intra_plan["global"] * 100, 1), "win_score": r2(adj * 100, 1),
                "why": [INTRADAY_INFO], "_sort": adj,
            })
        intraday = sorted(intraday, key=lambda x: -x["_sort"])[:CFG["top_intraday"]]
        for i, c in enumerate(intraday, 1):
            c["rank"] = i
            c.pop("_sort", None)

    log("5) Updating results of past picks…")
    picks, ppath = update_results(prices, today_str)
    picks = picks[~((picks.date == today_str) & (picks.status == "Open"))]
    new_rows = [{"date": today_str, "type": "swing", "symbol": c["symbol"], "name": c["name"], "setup": c["setup"],
                 "entry": c["close"], "buy_upto": c["buy_upto"], "target": c["target"], "sl": c["sl"], "hold_days": hold,
                 "status": "Open", "exit_date": None, "ret_pct": None} for c in swing]
    new_rows += [{"date": today_str, "type": "intraday", "symbol": c["symbol"], "name": c["name"], "setup": "Intraday",
                  "entry": c["trigger"], "buy_upto": None, "target": c["target"], "sl": c["sl"], "hold_days": 1,
                  "status": "Open", "exit_date": None, "ret_pct": None} for c in intraday]
    if new_rows:
        picks = pd.concat([picks, pd.DataFrame(new_rows)], ignore_index=True)
    picks.to_csv(ppath, index=False)
    results = results_summary(picks)

    # Market mood
    nl = nifty.copy()
    nl["ema20"], nl["ema50"] = ema(nl.Close, 20), ema(nl.Close, 50)
    nlast = nl.iloc[-1]
    if nlast.Close > nlast.ema20 > nlast.ema50:
        mood, mood_txt = "Strong", "Nifty is in an uptrend. Good conditions for swing buying."
    elif nlast.Close > nlast.ema50:
        mood, mood_txt = "Okay", "Nifty is mixed. Trade smaller and be quick to book profit."
    else:
        mood, mood_txt = "Weak", "Nifty is below its 50-day average. Most buy setups fail more often now — trade less."
    if vix_last and vix_last > 18:
        mood_txt += f" India VIX is high ({vix_last:.1f}), so expect bigger swings."

    out = {
        "generated_at": dt.datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
        "data_date": last_date.strftime("%d %b %Y"),
        "universe": len(prices), "settings": {k: CFG[k] for k in ("hold_days", "min_value_cr", "min_price")},
        "market": {"nifty": r2(nlast.Close), "nifty_chg": r2((nl.Close.iloc[-1] / nl.Close.iloc[-2] - 1) * 100),
                   "vix": r2(vix_last), "mood": mood, "mood_text": mood_txt},
        "setups": setups_out, "swing": swing, "intraday": intraday,
        "swing_candidates": len(cands), "sectors": sector_list, "results": results,
        "runtime_sec": int(time.time() - t0),
    }
    with open(os.path.join(DOCS_DIR, "data.json"), "w") as f:
        json.dump(clean(out), f, ensure_ascii=False, allow_nan=False)
    log(f"Done in {out['runtime_sec']}s — {len(swing)} swing, {len(intraday)} intraday picks.")
    return out


if __name__ == "__main__":
    main()
