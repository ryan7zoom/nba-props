"""Per-player context for the builder: is he a starter, and how did he do the last times against tonight's opponent.
Writes data/player_context_latest.json (key = athlete id | event id). Needs data/player_games.csv and data/projections_latest.json.
Starter = started at least 3 of his last 5 games. h = his last 3 games against tonight's opponent (any season):
[date, home(1/0), minutes, points, rebounds, assists, field goals made, field goals attempted]."""
import csv, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")


def main():
    games = defaultdict(list)
    with open(DATA / "player_games.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["season_type"] not in ("2", "3", "5") or r["dnp"] == "1" or float(r["min"] or 0) <= 0:
                continue
            games[r["athlete_id"]].append(r)
    for v in games.values():
        v.sort(key=lambda r: (r["game_date_et"], r["event_id"]))
    out = {}
    for p in json.load(open(DATA / "projections_latest.json", encoding="utf-8"))["players"]:
        gs = games.get(str(p["id"]), [])
        last5 = gs[-5:]
        starts = sum(r["starter"] == "1" for r in last5)
        h2h = [r for r in gs if r["opp"] == p["opp"]][-3:]
        out[str(p["id"]) + "|" + str(p["event_id"])] = {
            "st": len(last5) >= 3 and starts >= 3, "s": starts, "n": len(last5),
            "o": p["opp"],
            "h": [[datetime.strptime(r["game_date_et"], "%Y-%m-%d").strftime("%d/%m/%Y"), int(r["home"]), round(float(r["min"])), int(float(r["pts"] or 0)),
                   int(float(r["reb"] or 0)), int(float(r["ast"] or 0)), int(float(r["fgm"] or 0)), int(float(r["fga"] or 0))] for r in reversed(h2h)]}
    (DATA / "player_context_latest.json").write_text(json.dumps({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "players": out}, separators=(",", ":")), encoding="utf-8")
    print(f"Player context for {len(out)} players, {sum(v['st'] for v in out.values())} starters")


if __name__ == "__main__":
    main()
