"""Recent form: each projected player's last five games (opponent, points, rebounds, assists, minutes).

Reads data/player_games.csv (box scores) and data/projections_latest.json (who is playing next),
writes data/form_latest.json for the page. Only games the player actually played are listed.

Run:  python scripts/form.py
Exit code 1 if the projections or the box scores are missing.
"""
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")
PLAYER_GAMES = DATA / "player_games.csv"
PROJECTIONS = DATA / "projections_latest.json"
OUT = DATA / "form_latest.json"
GMT8 = timezone(timedelta(hours=8))
VALID_TYPES = {"2", "3", "5"}          # regular season, playoffs, play-in
LAST_N = 5


def log(msg=""):
    print(msg, flush=True)


def dmy(iso):
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y")


def run():
    if not PROJECTIONS.exists():
        raise SystemExit("data/projections_latest.json is missing: run the simulation first")
    if not PLAYER_GAMES.exists():
        raise SystemExit("data/player_games.csv is missing")
    want = {p["id"] for p in json.loads(PROJECTIONS.read_text(encoding="utf-8"))["players"]}
    games = {}
    last_date = ""
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["athlete_id"] not in want or r["season_type"] not in VALID_TYPES or r["dnp"] == "1":
                continue
            try:
                m = float(r["min"])
            except ValueError:
                continue
            if m <= 0:
                continue
            last_date = max(last_date, r["game_date_et"])
            games.setdefault(r["athlete_id"], []).append(
                (r["game_date_et"], r["event_id"], r["opp"], int(r["home"] or 0), int(r["pts"] or 0), int(r["reb"] or 0), int(r["ast"] or 0), round(m)))
    out = {}
    for aid, rows in games.items():
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out[aid] = [[dmy(d), opp, home, pts, reb, ast, mn] for d, _e, opp, home, pts, reb, ast, mn in rows[:LAST_N]]
    missing = len(want - set(out))
    now = datetime.now(timezone.utc)
    DATA.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": now.astimezone(GMT8).strftime("%d/%m/%Y %H:%M") + " GMT+8", "epoch": int(now.timestamp()),
        "data_through": dmy(last_date) if last_date else "", "games_per_player": LAST_N,
        "players_without_games": missing, "players": out}, separators=(",", ":")), encoding="utf-8")
    log(f"Form for {len(out)} players ({missing} projected players have no games on record). Wrote {OUT}")
    return 0


def main():
    try:
        return run()
    except SystemExit as e:
        log(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
