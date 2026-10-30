"""Sportsbook lines: points and PRA (points + rebounds + assists) lines for every player in the next games.

Source: PropLine (https://prop-line.com), free plan, 1,000 requests a day. Needs a free key in the
environment variable PROPLINE_API_KEY (on GitHub: a repository secret). Books used: DraftKings and FanDuel
(lines are nearly identical across popular books, and both post alternate lines).

What it does
  1. lists the upcoming NBA games,
  2. for each game asks for the points and PRA lines from those books (one request per game),
  3. matches the players to the players in data/projections_latest.json by game and name,
  4. writes data/lines_latest.json (compact) and data/lines_health.json.

The page does the maths (our chance of Over / Under at the book's line, and the expected value of the price).

Run:  python scripts/lines.py
Exit code is 1 when anything is wrong, so the workflow shows red.
"""
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")
PROJECTIONS = DATA / "projections_latest.json"
OUT = DATA / "lines_latest.json"
HISTORY = DATA / "lines_history.csv"
HEALTH = DATA / "lines_health.json"

BASE = "https://api.prop-line.com/v1"
SPORT = "basketball_nba"
MARKETS = {"player_points": "pts", "player_points_rebounds_assists": "pra"}
BOOKS = ["draftkings", "fanduel"]
MAX_EVENTS = 20
LOOKAHEAD_HOURS = 60
MIN_MATCH_SHARE = 0.5          # warn when fewer than this share of book players are matched to ours
GMT8 = timezone(timedelta(hours=8))

TEAM_CODES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO", "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH",
}


class LinesError(Exception):
    pass


def log(msg=""):
    print(msg, flush=True)


def stamp(now):
    d = now.astimezone(GMT8)
    return d.strftime("%d/%m/%Y %H:%M") + " GMT+8"


def norm(name):
    """Name used to match players between the book and ESPN: no accents, punctuation, suffixes or team tags."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"})


def et_date(iso_utc):
    from zoneinfo import ZoneInfo
    dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo("America/New_York")).date()


# ---------- the web call ----------

QUOTA = {"remaining": None, "limit": None}


def api(path, params=None, key=None):
    url = BASE + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    req = urllib.request.Request(url, headers={"X-API-Key": key, "User-Agent": "nba-props/1.0", "Accept": "application/json"})
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                for name, field in (("X-Daily-Remaining", "remaining"), ("X-Daily-Limit", "limit")):
                    v = r.headers.get(name)
                    if v is not None and v.isdigit():
                        QUOTA[field] = int(v)
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 401:
                raise LinesError("the PropLine key was rejected (401). Check the PROPLINE_API_KEY secret")
            if e.code == 429:
                raise LinesError(f"PropLine daily request limit reached (429): {body}")
            if e.code >= 500 and attempt == 0:
                last = f"{e.code} {body}"
                time.sleep(3)
                continue
            raise LinesError(f"PropLine returned {e.code} for {path}: {body}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = str(e)
            if attempt == 0:
                time.sleep(3)
                continue
            raise LinesError(f"could not reach PropLine ({last})")
    raise LinesError(f"PropLine request failed: {last}")


# ---------- reading the answer ----------

def parse_odds(odds):
    """-> {(player_norm, 'pts'|'pra', book, point): {'o': price, 'u': price, 'type': 'main'|'alt', 'name': display}}"""
    found = {}
    for bk in odds.get("bookmakers") or []:
        book = bk.get("key")
        for m in bk.get("markets") or []:
            mk = MARKETS.get(m.get("key"))
            if not mk:
                continue
            mtype = m.get("line_type") or "main"
            for o in m.get("outcomes") or []:
                player = o.get("description") or m.get("description")
                side = str(o.get("name") or "").strip().lower()
                point, price = o.get("point"), o.get("price")
                if not player or side not in ("over", "under") or point is None or price is None:
                    continue
                try:
                    point, price = float(point), int(round(float(price)))
                except (TypeError, ValueError):
                    continue
                typ = "main" if (o.get("line_type") or mtype) == "main" else "alt"
                rec = found.setdefault((norm(player), mk, book, point), {"o": None, "u": None, "type": typ, "name": re.sub(r"\s*\([^)]*\)\s*$", "", str(player))})
                rec["o" if side == "over" else "u"] = price
                if typ == "main":
                    rec["type"] = "main"
    return found


def describe(odds):
    """Short description of the structure, used when nothing could be read (so the shape can be fixed)."""
    try:
        bk = (odds.get("bookmakers") or [{}])[0]
        m = (bk.get("markets") or [{}])[0]
        o = (m.get("outcomes") or [{}])[0]
        return (f"top keys {sorted(odds.keys())[:12]}; book keys {sorted(bk.keys())[:12]}; "
                f"market keys {sorted(m.keys())[:12]}; outcome keys {sorted(o.keys())[:14]}")
    except Exception as e:
        return f"unreadable structure ({e})"


# ---------- history, for grading our numbers against real results ----------

HISTORY_COLS = ["fetched_utc", "game_date", "team", "opp", "player", "market", "book", "line", "over_price", "under_price",
                "our_p_over", "our_p_under", "our_mean", "our_median", "p_play"]


def _cdf(c, k):
    return 0.0 if k < 0 else (1.0 if k >= len(c) else c[k])


def append_history(now, players, proj_by_key):
    """One row per book main line, with what our model said at that moment. Machine file: ISO dates."""
    new = not HISTORY.exists()
    n = 0
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_COLS)
        for p in players:
            pr = proj_by_key.get((p["d"], p["t"], p["n"]))
            if not pr:
                continue
            iso = datetime.strptime(p["d"], "%d/%m/%Y").strftime("%Y-%m-%d")
            for mk in ("pts", "pra"):
                c = pr[mk]["cdf"]
                for book, point, over, under in p[mk]["m"]:
                    fl = math.floor(point)
                    p_under = _cdf(c, fl) if point != fl else _cdf(c, fl - 1)
                    w.writerow([now.strftime("%Y-%m-%dT%H:%M:%SZ"), iso, p["t"], pr["opp"], p["n"], mk, book, point, over, under,
                                round(1.0 - _cdf(c, fl), 4), round(p_under, 4), pr[mk]["mean"], pr[mk]["p50"], pr["p_play"]])
                    n += 1
    return n


# ---------- main ----------

def run(key, now=None, api_fn=None):
    api_fn = api_fn or api
    now = now or datetime.now(timezone.utc)
    errors, warnings = [], []
    if not PROJECTIONS.exists():
        raise LinesError("data/projections_latest.json is missing: run the simulation first")
    proj = json.loads(PROJECTIONS.read_text(encoding="utf-8"))
    by_game = {}
    for p in proj["players"]:
        by_game.setdefault((p["team"], p["opp"], p["home"]), []).append(p)
    ours_by_game = {}
    for p in proj["players"]:
        ours_by_game.setdefault((p["game_date"], p["team"] if p["home"] else p["opp"], p["opp"] if p["home"] else p["team"]), {})[norm(p["name"])] = p

    events = api_fn("/sports/" + SPORT + "/events", None, key)
    if not isinstance(events, list):
        raise LinesError(f"unexpected events answer: {str(events)[:150]}")
    horizon = now + timedelta(hours=LOOKAHEAD_HOURS)
    upcoming = []
    for e in events:
        try:
            t = datetime.fromisoformat(str(e["commence_time"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if now - timedelta(minutes=1) <= t <= horizon:
            upcoming.append((t, e))
    upcoming.sort(key=lambda x: x[0])
    upcoming = upcoming[:MAX_EVENTS]
    log(f"PropLine lists {len(events)} NBA events, {len(upcoming)} start in the next {LOOKAHEAD_HOURS} hours")

    out_players, unmatched, book_players, matched_players, games_matched, requests = {}, [], 0, 0, 0, 1
    for t, e in upcoming:
        home, away = TEAM_CODES.get(e.get("home_team")), TEAM_CODES.get(e.get("away_team"))
        if not home or not away:
            warnings.append(f"unknown team name in event {e.get('id')}: {e.get('home_team')} / {e.get('away_team')}")
            continue
        d = et_date(e["commence_time"])
        ours = None
        for off in (0, -1, 1):
            gd = (d + timedelta(days=off)).strftime("%d/%m/%Y")
            if (gd, home, away) in ours_by_game:
                ours, game_date = ours_by_game[(gd, home, away)], gd
                break
        if ours is None:
            warnings.append(f"no projections for {away} at {home} on {d.strftime('%d/%m/%Y')} (lines skipped)")
            continue
        odds = api_fn(f"/sports/{SPORT}/events/{e['id']}/odds", {"markets": ",".join(MARKETS), "bookmakers": ",".join(BOOKS)}, key)
        requests += 1
        found = parse_odds(odds if isinstance(odds, dict) else {})
        if not found:
            warnings.append(f"{away} at {home}: no player lines posted yet ({describe(odds) if isinstance(odds, dict) else 'bad answer'})")
            continue
        games_matched += 1
        names_seen = set()
        for (pn, mk, book, point), rec in found.items():
            names_seen.add(pn)
            p = ours.get(pn)
            if p is None:
                continue
            ent = out_players.setdefault((game_date, p["team"], p["name"]), {"n": p["name"], "t": p["team"], "d": game_date, "pts": {"m": [], "a": []}, "pra": {"m": [], "a": []}})
            ent[mk]["m" if rec["type"] == "main" else "a"].append([book, point, rec["o"], rec["u"]])
        book_players += len(names_seen)
        matched_players += len([n for n in names_seen if n in ours])
        for n in names_seen:
            if n not in ours:
                unmatched.append(n)
    if upcoming and games_matched == 0:
        errors.append("no game had readable player lines (see warnings)")
    share = matched_players / book_players if book_players else 1.0
    if book_players and share < MIN_MATCH_SHARE:
        warnings.append(f"only {matched_players} of {book_players} book players matched our players ({share:.0%})")
    players = list(out_players.values())
    for p in players:
        for mk in ("pts", "pra"):
            p[mk]["m"].sort(key=lambda r: (r[0], r[1]))
            p[mk]["a"].sort(key=lambda r: (r[0], r[1]))
    DATA.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "fetched": stamp(now), "epoch": int(now.timestamp()), "books": BOOKS, "source": "PropLine",
        "games": games_matched, "players": players}, separators=(",", ":")), encoding="utf-8")
    health = {"run": stamp(now), "ok": not errors, "errors": errors, "warnings": warnings, "warning_count": len(warnings),
              "events_listed": len(events), "events_upcoming": len(upcoming), "games_with_lines": games_matched,
              "players_with_lines": len(players), "book_players_unmatched": len(set(unmatched)), "unmatched_examples": sorted(set(unmatched))[:15],
              "requests_this_run": requests, "requests_left_today": QUOTA["remaining"], "daily_limit": QUOTA["limit"]}
    proj_by_key = {(p["game_date"], p["team"], p["name"]): p for p in proj["players"]}
    health["history_rows_added"] = append_history(now, players, proj_by_key)
    HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    log(f"Lines for {len(players)} players in {games_matched} games ({requests} requests; {QUOTA['remaining']} left today). Wrote {OUT}")
    for w in warnings:
        log(f"  WARNING: {w}")
    for er in errors:
        log(f"  ERROR: {er}")
    return health


def main():
    key = os.environ.get("PROPLINE_API_KEY", "").strip()
    try:
        if not key:
            raise LinesError("PROPLINE_API_KEY is not set. Add it as a repository secret (Settings > Secrets and variables > Actions)")
        health = run(key)
    except LinesError as e:
        DATA.mkdir(exist_ok=True)
        HEALTH.write_text(json.dumps({"run": stamp(datetime.now(timezone.utc)), "ok": False, "errors": [str(e)], "warnings": [], "warning_count": 0}, indent=2), encoding="utf-8")
        log(f"ERROR: {e}")
        return 1
    return 0 if health["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
