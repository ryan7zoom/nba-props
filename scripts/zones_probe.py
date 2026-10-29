"""Shot-zones probe: does ESPN's play-by-play really carry shot positions and shooter ids?

Fetches three finished games (the latest one, one from the middle of 2025-26, one from 2024-25) from the
same ESPN summary service the box-score script uses, and reports what the plays contain:
how many shots have a court position, how many name the shooter, the range of the positions,
what the shot descriptions look like, and a few raw examples.

Writes data/zones_probe.json. Exit code 1 if the data is too thin to build shot zones from.
Run:  python scripts/zones_probe.py
"""
import csv
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")
OUT = DATA / "zones_probe.json"
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-props probe)", "Accept": "application/json"}
GMT8 = timezone(timedelta(hours=8))
SENTINEL = 1e6          # ESPN uses huge negative numbers for "no position"
MIN_SHARE = 0.70        # share of shots that must have a position and a shooter


def log(msg=""):
    print(msg, flush=True)


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def pick_events():
    """(event_id, date) for the latest game, one near 15/01/2026 and one from 2024-25."""
    seen = {}
    with open(DATA / "player_games.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen[r["event_id"]] = r["game_date_et"]
    items = sorted(seen.items(), key=lambda kv: kv[1])
    if not items:
        raise SystemExit("data/player_games.csv has no games")

    def nearest(day):
        return min(items, key=lambda kv: abs((datetime.fromisoformat(kv[1]) - datetime.fromisoformat(day)).days))

    picks = [items[-1], nearest("2026-01-15"), nearest("2024-12-15")]
    out, ids = [], set()
    for eid, d in picks:
        if eid not in ids:
            ids.add(eid)
            out.append((eid, d))
    return out


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


def analyse(data):
    plays = data.get("plays") or []
    shots = [p for p in plays if p.get("shootingPlay")]
    with_xy = [p for p in shots if good_xy(p.get("coordinate"))]
    with_id = [p for p in shots if shooter_id(p)]
    both = [p for p in with_xy if shooter_id(p)]
    xs = [good_xy(p["coordinate"])[0] for p in with_xy]
    ys = [good_xy(p["coordinate"])[1] for p in with_xy]
    types = Counter(((p.get("type") or {}).get("text") or "?") for p in shots)
    feet = sum(1 for p in shots if "-foot" in str(p.get("text", "")).lower())
    free_throws = sum(1 for p in shots if "free throw" in str(p.get("text", "")).lower())
    samples = []
    for p in with_xy[:3] + [p for p in shots if not good_xy(p.get("coordinate"))][:1]:
        samples.append({"text": p.get("text"), "type": (p.get("type") or {}).get("text"), "scoreValue": p.get("scoreValue"),
                        "scoringPlay": p.get("scoringPlay"), "coordinate": p.get("coordinate"),
                        "team_id": (p.get("team") or {}).get("id"), "shooter_id": shooter_id(p),
                        "participants_n": len(p.get("participants") or [])})
    return {
        "top_level_keys": sorted(data.keys())[:25], "plays": len(plays), "shooting_plays": len(shots),
        "shots_with_position": len(with_xy), "shots_with_shooter_id": len(with_id), "shots_with_both": len(both),
        "share_with_both": round(len(both) / len(shots), 3) if shots else 0.0,
        "free_throws_among_shots": free_throws, "descriptions_with_feet": feet,
        "x_range": [min(xs), max(xs)] if xs else None, "y_range": [min(ys), max(ys)] if ys else None,
        "shot_types_top": types.most_common(20), "samples": samples,
    }


def main():
    DATA.mkdir(exist_ok=True)
    report = {"run": datetime.now(timezone.utc).astimezone(GMT8).strftime("%d/%m/%Y %H:%M") + " GMT+8", "games": [], "errors": []}
    for eid, d in pick_events():
        try:
            a = analyse(fetch(f"{BASE}/summary?event={eid}"))
        except Exception as e:
            report["errors"].append(f"event {eid}: {e}")
            log(f"event {eid}: FAILED {e}")
            continue
        a["event_id"], a["game_date"] = eid, datetime.fromisoformat(d).strftime("%d/%m/%Y")
        report["games"].append(a)
        log(f"{a['game_date']} event {eid}: {a['plays']} plays, {a['shooting_plays']} shots, {a['shots_with_position']} with a position, "
            f"{a['shots_with_shooter_id']} with a shooter, both {a['share_with_both']:.0%}; x {a['x_range']} y {a['y_range']}")
        log(f"   shot types: {a['shot_types_top'][:8]}")
        for s in a["samples"][:2]:
            log(f"   example: {s}")
        time.sleep(0.5)
    shots = sum(g["shooting_plays"] for g in report["games"])
    both = sum(g["shots_with_both"] for g in report["games"])
    share = both / shots if shots else 0.0
    report["overall_share_with_position_and_shooter"] = round(share, 3)
    report["usable"] = bool(shots and share >= MIN_SHARE and not report["errors"])
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nOverall: {both} of {shots} shots have a position and a shooter ({share:.0%}). "
        f"{'USABLE for shot zones' if report['usable'] else 'NOT usable as is: see the file for why'}. Wrote {OUT}")
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    sys.exit(main())
