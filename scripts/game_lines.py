"""Game lines: the bookmakers' own game total, spread and moneyline for each upcoming game.

Why: the market's total and spread are a much better guide to pace and margin than our team model alone
(it knows lineups and rest). game_model.py blends them in.

Source: PropLine, same key as lines.py (PROPLINE_API_KEY). One request per game. If the free plan does not carry
these markets, this script just says so in data/game_lines_health.json and the game model keeps working on its own.
It NEVER fails the daily run.

You can also type lines in by hand: create data/game_lines_manual.csv with the header
    date,home,away,total,home_spread
and one row per game, e.g.
    31/10/2026,WSH,CHA,216,-2.5
(home_spread is the home team's spread: -13.5 means the home team is favoured by 13.5). Manual rows win.

Run:  python scripts/game_lines.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lines as L   # reuses the PropLine call, team names and date helpers from lines.py

DATA = Path("data")
OUT = DATA / "game_lines_latest.json"
HEALTH = DATA / "game_lines_health.json"
GMARKETS = "h2h,spreads,totals"
TOTAL_RANGE = (195.0, 260.0)   # an NBA game total outside this is a misread (alternate or half-game line), so it is ignored


def dec(price):
    return 1 + price / 100 if price > 0 else 1 + 100 / -price


def fair_over(po, pu):
    a, b = 1 / dec(po), 1 / dec(pu)
    return a / (a + b)


def is_main(o, m):
    """Same rule as lines.py: a line counts as the main line unless the feed marks it as something else."""
    return str(o.get("line_type") or m.get("line_type") or "main").lower() == "main"


def sample(odds, n=14):
    """A few raw totals rows, saved in the health file so a wrong reading can be diagnosed."""
    rows = []
    for bk in odds.get("bookmakers") or []:
        for m in bk.get("markets") or []:
            if m.get("key") == "totals":
                for o in (m.get("outcomes") or [])[:n]:
                    rows.append([bk.get("key"), m.get("line_type"), o.get("line_type"), o.get("name"), o.get("point"), o.get("price")])
    return rows[:n]


def parse_game(odds, home_name, away_name):
    """-> dict(total, home_spread, home_ml, away_ml) averaged over books, or {} when nothing readable."""
    tots, sprs, mls_h, mls_a = [], [], [], []
    for bk in odds.get("bookmakers") or []:
        for m in bk.get("markets") or []:
            key = m.get("key")
            outs = m.get("outcomes") or []
            if key == "totals":
                by = {}
                for o in outs:
                    if not is_main(o, m):
                        continue
                    try:
                        pt, pr = float(o["point"]), int(round(float(o["price"])))
                    except Exception:
                        continue
                    side = str(o.get("name", "")).strip().lower()
                    if side in ("over", "under"):
                        by.setdefault(pt, {})[side] = pr
                best = None
                for pt, d in by.items():
                    if "over" in d and "under" in d:
                        gap = abs(fair_over(d["over"], d["under"]) - 0.5)
                        if best is None or gap < best[0]:
                            best = (gap, pt)
                if best and TOTAL_RANGE[0] <= best[1] <= TOTAL_RANGE[1]:
                    tots.append(best[1])
            elif key == "spreads":
                by = {}
                for o in outs:
                    if not is_main(o, m):
                        continue
                    try:
                        pt, pr = float(o["point"]), int(round(float(o["price"])))
                    except Exception:
                        continue
                    if o.get("name") == home_name:
                        by.setdefault(abs(pt), {})["h"] = (pt, pr)
                    elif o.get("name") == away_name:
                        by.setdefault(abs(pt), {})["a"] = (pt, pr)
                best = None
                for k, d in by.items():
                    if "h" in d and "a" in d:
                        gap = abs(dec(d["h"][1]) - dec(d["a"][1]))
                        if best is None or gap < best[0]:
                            best = (gap, d["h"][0])
                if best:
                    sprs.append(best[1])
            elif key == "h2h":
                for o in outs:
                    try:
                        pr = int(round(float(o["price"])))
                    except Exception:
                        continue
                    (mls_h if o.get("name") == home_name else mls_a if o.get("name") == away_name else []).append(pr)
    out = {}
    if tots:
        out["total"] = round(sum(tots) / len(tots) * 2) / 2
    if sprs:
        out["home_spread"] = round(sum(sprs) / len(sprs) * 2) / 2
    if mls_h and mls_a:
        out["home_ml"], out["away_ml"] = mls_h[0], mls_a[0]
    return out


def run(key, now=None, api_fn=None):
    api_fn = api_fn or L.api
    now = now or datetime.now(timezone.utc)
    warnings, games = [], {}
    run.sample = []
    events = api_fn("/sports/" + L.SPORT + "/events", None, key)
    horizon = now + timedelta(hours=L.LOOKAHEAD_HOURS)
    n_req = 1
    for e in events if isinstance(events, list) else []:
        try:
            t = datetime.fromisoformat(str(e["commence_time"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if not (now - timedelta(minutes=1) <= t <= horizon):
            continue
        home, away = L.TEAM_CODES.get(e.get("home_team")), L.TEAM_CODES.get(e.get("away_team"))
        if not home or not away:
            continue
        odds = api_fn(f"/sports/{L.SPORT}/events/{e['id']}/odds", {"markets": GMARKETS, "bookmakers": ",".join(L.BOOKS)}, key)
        n_req += 1
        if isinstance(odds, dict) and len(run.sample) < 2:
            run.sample.append({"game": f"{away} at {home}", "totals_rows [book, market line_type, outcome line_type, name, point, price]": sample(odds)})
        g = parse_game(odds if isinstance(odds, dict) else {}, e.get("home_team"), e.get("away_team"))
        if not g:
            warnings.append(f"{away} at {home}: no game lines (spread/total/moneyline) in the answer ({L.describe(odds) if isinstance(odds, dict) else 'bad answer'})")
            continue
        g["date"] = L.et_date(e["commence_time"]).strftime("%d/%m/%Y")
        games[f"{home}|{away}|{g['date']}"] = {"home": home, "away": away, **g}
    return games, warnings, n_req


def main():
    now = datetime.now(timezone.utc)
    DATA.mkdir(exist_ok=True)
    key = os.environ.get("PROPLINE_API_KEY", "").strip()
    games, warnings, n_req = {}, [], 0
    try:
        if not key:
            warnings.append("PROPLINE_API_KEY is not set")
        else:
            games, warnings, n_req = run(key, now)
    except Exception as e:   # never fail the daily run
        warnings.append(f"could not fetch game lines: {e}")
    OUT.write_text(json.dumps({"fetched": L.stamp(now), "epoch": int(now.timestamp()), "games": games}, separators=(",", ":")), encoding="utf-8")
    HEALTH.write_text(json.dumps({"run": L.stamp(now), "games_with_lines": len(games), "warnings": warnings, "requests": n_req,
                                  "raw_totals_sample": getattr(run, "sample", [])}, indent=2), encoding="utf-8")
    print(f"Game lines for {len(games)} games ({n_req} requests)")
    for w in warnings[:12]:
        print(f"  NOTE: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
