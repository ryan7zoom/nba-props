"""Absence adjustment: scale each player's points and PRA projection by how much production his team is missing.

Why: tested walk-forward on 2025-26 (tuned before 15/01/2026, checked after, 15,243 player-games), the simulation
projected points about 5% too high when a team had all its regulars, and too LOW when regulars were out
(players on teams missing 30+ points of production scored about 13% more than projected).
The adjustment  mult = 1 - 0.05 + 0.06 * (missing points / 10)  improved the check period by 2.1% (z = 11.6).

"Missing points" = for every player who averaged 18+ minutes over the team's last 10 games, his average points
times the chance he does NOT play in this game (a player who is not in the projection at all counts as out).

Runs right after simulate_props.py and rewrites data/projections_latest.json. Safe to run twice (it marks the file).
Run:  python scripts/absence_adjust.py
Needs: nothing beyond the Python standard library.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
PROJ = DATA / "projections_latest.json"
GAMES = DATA / "player_games.csv"
A, K, CAP = -0.05, 0.06, 35.0          # tuned on 2025-26 before 15/01/2026
MIN_REG = 18.0                         # a "regular" averaged this many minutes over the team's last 10 games
LAST_N = 10
CLIP = (0.85, 1.25)


def baselines():
    """{team: {athlete_id: (avg_min, avg_pts)}} over each team's last 10 games (a missed game counts as 0)."""
    by_team = defaultdict(dict)                     # team -> event_id -> date
    rows = defaultdict(list)                        # (team, event_id) -> [(aid, min, pts)]
    with open(GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["season_type"] not in ("2", "3", "5") or r["dnp"] == "1":
                continue
            m = float(r["min"] or 0)
            if m <= 0:
                continue
            by_team[r["team"]][r["event_id"]] = r["game_date_et"]
            rows[(r["team"], r["event_id"])].append((r["athlete_id"], m, float(r["pts"] or 0)))
    out = {}
    for t, evs in by_team.items():
        last = sorted(evs, key=lambda e: (evs[e], e))[-LAST_N:]
        agg = defaultdict(lambda: [0.0, 0.0])
        for e in last:
            for aid, m, p in rows[(t, e)]:
                agg[aid][0] += m
                agg[aid][1] += p
        n = float(len(last))
        out[t] = {a: (v[0] / n, v[1] / n) for a, v in agg.items() if v[0] / n >= MIN_REG}
    return out


def rescale(cdf, m):
    """New cdf after multiplying every outcome by m (mass moved between neighbouring whole numbers, so the mean is exactly m times)."""
    pmf = [cdf[0]] + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))]
    size = int(math.ceil((len(pmf) - 1) * m)) + 2
    new = [0.0] * size
    for x, p in enumerate(pmf):
        if p <= 0:
            continue
        pos = x * m
        lo = int(math.floor(pos))
        fr = pos - lo
        new[lo] += p * (1 - fr)
        new[lo + 1] += p * fr
    out, run = [], 0.0
    for v in new:
        run += v
        out.append(round(min(run, 1.0), 4))
        if run >= 0.9999:
            break
    out[-1] = 1.0 if out[-1] >= 0.9995 else out[-1]
    return out


def summarise(cdf, d):
    """Update mean and quantiles of a stat block from its cdf."""
    mean = 0.0
    prev = 0.0
    for k, c in enumerate(cdf):
        mean += k * (c - prev)
        prev = c
    d["mean"] = round(mean, 2)
    for key, q in (("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90)):
        if key in d:
            d[key] = next((k for k, c in enumerate(cdf) if c >= q), len(cdf) - 1)
    d["cdf"] = cdf


def main():
    if not PROJ.exists():
        print("data/projections_latest.json is missing: nothing to adjust")
        return 0
    js = json.loads(PROJ.read_text(encoding="utf-8"))
    if js.get("absence_adjusted"):
        print("already adjusted: nothing to do")
        return 0
    base = baselines()
    present = defaultdict(dict)                     # (event_id, team) -> {aid: p_play}
    for p in js["players"]:
        present[(p["event_id"], p["team"])][str(p["id"])] = float(p.get("p_play", 1.0))
    miss = {}
    for (ev, team), have in present.items():
        miss[(ev, team)] = sum(pts * (1.0 - have.get(aid, 0.0)) for aid, (mn, pts) in base.get(team, {}).items())
    n = 0
    for p in js["players"]:
        ms = miss[(p["event_id"], p["team"])]
        m = min(max(1.0 + A + K * min(ms, CAP) / 10.0, CLIP[0]), CLIP[1])
        pts, pra = p["pts"], p["pra"]
        m_pra = 1.0 + (m - 1.0) * (pts["mean"] / pra["mean"] if pra["mean"] else 0.0)
        summarise(rescale(pts["cdf"], m), pts)
        summarise(rescale(pra["cdf"], m_pra), pra)
        p["absence"] = {"missing_pts": round(ms, 1), "mult": round(m, 3)}
        n += 1
    js["absence_adjusted"] = {"a": A, "k": K, "cap": CAP, "players": n,
                              "note": "points and PRA scaled by team missing production; see scripts/absence_adjust.py"}
    PROJ.write_text(json.dumps(js, separators=(",", ":")), encoding="utf-8")
    ms_all = sorted(v for v in miss.values())
    print(f"absence adjustment applied to {n} players; team missing production median {ms_all[len(ms_all)//2]:.1f} points, max {ms_all[-1]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
