"""Shot zones: for every finished game, how many shots each player took and made from each court zone.

Reads ESPN play-by-play (same service as the box scores) and writes data/shot_zones.csv with one row per
player, game and zone:   event_id,athlete_id,team_id,zone,fga,fgm
Zones: rim (0-4 ft), short (4-10 ft), mid (10 ft out to the three-point line), corner3, atb3 (above-the-break three).
Free throws are left out. Shots without a court position are placed from the text ("22-foot three point jumper",
"driving layup"); shots that cannot be placed at all are counted as zone "unk".

It is safe to run again and again: games already in the file are skipped, so a run that stops half way continues
next time. A game ESPN has no play-by-play for (404) is stored once as zone "none" so it is not asked for again.

Run:  python scripts/shot_zones.py                 (all missing games, newest first)
      python scripts/shot_zones.py --max-minutes 100   (stop after about 100 minutes; run again to continue)
"""
import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
OUT = DATA / "shot_zones.csv"
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-props zones)", "Accept": "application/json"}
SENTINEL = 1e6
PAUSE = 0.4
FIELDS = ["event_id", "athlete_id", "team_id", "zone", "fga", "fgm"]
RIM_TYPES = ("layup", "dunk", "tip shot", "finger roll", "alley oop", "hook shot")   # near the basket when no position is given


def log(m=""):
    print(m, flush=True)


def fetch_json(url, retries=4):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:
            last = e
        time.sleep(2 * (a + 1))
    raise last


def good_xy(c):
    if not isinstance(c, dict):
        return None
    x, y = c.get("x"), c.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if abs(x) > SENTINEL or abs(y) > SENTINEL:
        return None
    return float(x), float(y)


def shooter_id(play):
    for p in play.get("participants") or []:
        a = p.get("athlete") or {}
        if a.get("id"):
            return str(a["id"])
    return None


def zone_of(play):
    """Zone name for one shooting play, or None for free throws. The basket sits at x=25, y=0 in ESPN's coordinates."""
    text = str(play.get("text", "")).lower()
    ptype = str((play.get("type") or {}).get("text", "")).lower()
    if "free throw" in text or "free throw" in ptype:
        return None
    three = "three point" in text or play.get("scoreValue") == 3
    xy = good_xy(play.get("coordinate"))
    if xy:
        x, y = xy
        d = math.hypot(x - 25.0, y)
        if three:
            return "corner3" if y <= 9.0 else "atb3"
        return "rim" if d <= 4.0 else "short" if d <= 10.0 else "mid"
    m = re.search(r"(\d+)-foot", text)
    if three:
        if m:
            return "corner3" if int(m.group(1)) <= 22 else "atb3"
        return "atb3"
    if m:
        d = int(m.group(1))
        return "rim" if d <= 4 else "short" if d <= 10 else "mid"
    if any(k in text or k in ptype for k in RIM_TYPES):
        return "rim"
    return "unk"


def parse_game(data):
    """-> list of (athlete_id, team_id, zone, fga, fgm)."""
    agg = defaultdict(lambda: [0, 0])
    for p in data.get("plays") or []:
        if not p.get("shootingPlay"):
            continue
        z = zone_of(p)
        sid = shooter_id(p)
        if z is None or not sid:
            continue
        tid = str((p.get("team") or {}).get("id") or "")
        a = agg[(sid, tid, z)]
        a[0] += 1
        a[1] += 1 if p.get("scoringPlay") else 0
    return [(s, t, z, a, m) for (s, t, z), (a, m) in agg.items()]


def games_wanted():
    """(event_id, date) for every game in player_games.csv, newest first."""
    seen = {}
    with open(DATA / "player_games.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen[r["event_id"]] = r["game_date_et"]
    return sorted(seen.items(), key=lambda kv: kv[1], reverse=True)


def done_events():
    if not OUT.exists():
        return set()
    with open(OUT, newline="", encoding="utf-8") as f:
        return {r["event_id"] for r in csv.DictReader(f)}


def main(fetch=fetch_json):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=100.0)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    new_file = not OUT.exists()
    done = done_events()
    todo = [(e, d) for e, d in games_wanted() if e not in done]
    log(f"{len(done)} games already stored, {len(todo)} to fetch")
    t0, n_ok, n_err = time.time(), 0, 0
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for eid, day in todo:
            if time.time() - t0 > args.max_minutes * 60:
                log("time limit reached; run again to continue")
                break
            try:
                rows = parse_game(fetch(f"{BASE}/summary?event={eid}"))
                if not rows:
                    rows = [("0", "0", "none", 0, 0)]
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    rows = [("0", "0", "none", 0, 0)]
                else:
                    n_err += 1
                    log(f"  {eid} ({day}): {e}")
                    continue
            except Exception as e:
                n_err += 1
                log(f"  {eid} ({day}): {e}")
                continue
            for s, t, z, a, m in rows:
                w.writerow([eid, s, t, z, a, m])
            n_ok += 1
            if n_ok % 100 == 0:
                f.flush()
                log(f"  {n_ok} games done ({time.time() - t0:.0f}s)")
            time.sleep(PAUSE)
    log(f"Done: {n_ok} games added, {n_err} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
