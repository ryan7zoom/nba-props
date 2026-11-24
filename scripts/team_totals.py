"""Team totals: a second opinion and a line ladder.

The team model (game_model.py) builds a game total from team ratings, then blends it 35/65 with the market and
turns it into a normal distribution. Two things it cannot do:

  1. It does not know about tonight's injuries. A rating is an average over the season; it does not react when a
     team's leading scorer is ruled out an hour before tip. This script adds a bottom-up total -- the sum of the
     player projections, each weighted by his chance of appearing -- so a real injury shows up as a disagreement
     between the two numbers instead of being invisible.

  2. It has no record of what the market said. Nothing in the repo has ever stored a game line, so the 35% weight
     and the spread of the distribution have never been checked against a closing number. This script appends every
     game line it sees to data/game_lines_history.csv, and grades what is there.

The ladder is the same idea as the player safety ladder: for a set of lines around our number, the chance of the
Over and the Under, the fair price, and the smallest price worth taking. Use it against your bookie's alternate
total lines.

Reads  data/projections_latest.json, data/games_latest.json, data/game_lines_latest.json,
       data/game_lines_history.csv, data/player_games.csv
Writes data/team_totals_latest.json, data/game_lines_history.csv (append), data/team_totals_grade.json

Run:   python scripts/team_totals.py
       python scripts/team_totals.py --grade
"""
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path("data")
PROJ = DATA / "projections_latest.json"
GAMES = DATA / "games_latest.json"
GLINES = DATA / "game_lines_latest.json"
GHIST = DATA / "game_lines_history.csv"
PGGAMES = DATA / "player_games.csv"
OUT = DATA / "team_totals_latest.json"
GRADE = DATA / "team_totals_grade.json"

GMT8 = timezone(timedelta(hours=8))
W_OURS = 0.35          # share of our own total when a market total exists (same as game_model.py)
EDGE = 0.02            # edge required before a rung is "worth taking"
RUNG_STEP = 1.0        # your bookie's alternate totals go up in 1s
RUNG_SPAN = 20         # and reach this far either side of his main line
LEAGUE_TOTAL = 229.0
HIST_COLS = ["fetched_utc", "game_date", "home", "away", "line_type", "total", "home_spread",
             "our_total_topdown", "our_total_bottomup", "our_margin", "our_home_win"]


def log(msg=""):
    print(msg, flush=True)


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def min_price(p, edge=EDGE):
    if not p or p <= 0:
        return None
    return round((1.0 + edge) / p, 3)


def american(d):
    if not d or d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2 else -int(round(100 / (d - 1)))


def bottom_up(proj):
    """{(event_id, team): points per 240 team minutes} from the player projections.

    Each player contributes p_play * expected points and p_play * expected minutes. Dividing one by the other
    gives points per team-minute, and multiplying by 240 gives the team total that the roster is actually worth.
    Doing it as a ratio rather than a raw sum matters: it is immune to the appearance-probability calibration,
    and it cannot invent points for a roster that adds up to more or fewer than 240 minutes.
    """
    pts, mins, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for pl in proj.get("players", []):
        k = (pl.get("event_id"), pl.get("team"))
        pp = pl.get("p_appear", pl.get("p_play")) or 0.0
        block = pl.get("pts") or {}
        pts[k] += pp * float(block.get("mean") or 0.0)
        mins[k] += pp * float((pl.get("minutes") or {}).get("mean") or 0.0)
        n[k] += 1
    per240 = {k: (240.0 * pts[k] / mins[k]) if mins[k] > 0 else 0.0 for k in pts}
    return per240, mins, n


def load_game_lines():
    out = {}
    if not GLINES.exists():
        return out
    try:
        j = json.loads(GLINES.read_text(encoding="utf-8"))
    except Exception:
        return out
    for g in (j.get("games") or {}).values():
        out[(g.get("home"), g.get("away"), g.get("date"))] = g
    return out


def ladder(mu, sigma, main, step=RUNG_STEP, span=RUNG_SPAN):
    rows = []
    if sigma <= 0:
        return rows
    centre = main if main is not None else mu
    d = -span
    while d <= span + 1e-9:
        line = round(centre + d, 1)
        if line <= 0:
            d += step
            continue
        p_over = 1.0 - phi((line - mu) / sigma)
        p_under = 1.0 - p_over
        row = {"line": line, "p_over": round(p_over, 4), "p_under": round(p_under, 4)}
        if p_over > 0.5:
            row.update({"fair": round(1 / p_over, 3), "min": min_price(p_over), "side": "over"})
        else:
            row.update({"fair": round(1 / p_under, 3), "min": min_price(p_under), "side": "under"})
        rows.append(row)
        d += step
    return rows


def run(grade_only=False):
    now = datetime.now(timezone.utc)
    if grade_only:
        return grade(now)
    proj = json.loads(PROJ.read_text(encoding="utf-8")) if PROJ.exists() else {"players": []}
    games = json.loads(GAMES.read_text(encoding="utf-8")) if GAMES.exists() else {"games": []}
    gl = load_game_lines()
    bu_pts, bu_min, bu_n = bottom_up(proj)

    sm = float(games.get("sigma_margin") or 11.9)
    st = float(games.get("sigma_total") or 17.5)
    sigma_team = math.sqrt((sm ** 2 + st ** 2) / 4.0)
    rows, hist = [], []
    for g in games.get("games", []):
        date = g.get("game_date")
        H, A = g.get("home"), g.get("away")
        p = g.get("proj") or {}
        total = float(p.get("total") or LEAGUE_TOTAL)
        margin = float(p.get("margin_home") or 0.0)
        market = gl.get((H, A, date)) or {}
        m_total = market.get("total")
        m_spread = market.get("home_spread")
        bu_h = bu_pts.get((g.get("event_id"), H)) or 0.0
        bu_a = bu_pts.get((g.get("event_id"), A)) or 0.0
        bu = bu_h + bu_a
        bu_players = (bu_n.get((g.get("event_id"), H)) or 0) + (bu_n.get((g.get("event_id"), A)) or 0)
        min_h = bu_min.get((g.get("event_id"), H)) or 0.0
        min_a = bu_min.get((g.get("event_id"), A)) or 0.0
        minutes_total = max(min_h, min_a)
        # the market's total is the sharpest single number; ours only moves it a third of the way
        blended = W_OURS * total + (1 - W_OURS) * float(m_total) if m_total else total
        between = round(bu - total, 1) if bu_players >= 10 else None
        rec = {
            "event_id": g.get("event_id"), "game_date": date, "home": H, "away": A,
            "tipoff_utc": g.get("tipoff_utc"),
            "our_total": round(total, 1), "our_margin": round(margin, 1),
            "our_home_pts": round(total / 2 + margin / 2, 1), "our_away_pts": round(total / 2 - margin / 2, 1),
            "market_total": m_total, "market_home_spread": m_spread,
            "blended_total": round(blended, 1),
            "bottom_up_total": round(bu, 1) if bu_players >= 10 else None,
            "bottom_up_vs_ours": between,
            "expected_minutes_per_team": round(minutes_total, 1) if bu_players >= 10 else None,
            "sigma_total": round(st, 2), "sigma_margin": round(sm, 2),
            "home_win": p.get("home_win"),
            "ladder": ladder(blended, st, m_total),
            "team_ladder_home": ladder(total / 2 + margin / 2, sigma_team, None),
            "team_ladder_away": ladder(total / 2 - margin / 2, sigma_team, None),
        }
        if between is not None and abs(between) >= 8.0:
            rec["flag"] = ("the player projections are " + f"{between:+.0f} points from the team model: "
                           "check tonight's injury report against the ratings")
        if minutes_total:
            rec["minutes_verdict"] = ("coherent" if 225 <= minutes_total <= 255 else
                                      f"roster too optimistic: {minutes_total:.0f} minutes for 240 available"
                                      if minutes_total > 255 else
                                      f"roster too thin: {minutes_total:.0f} minutes for 240 available")
        rows.append(rec)
        if m_total:
            hist.append({"fetched_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "game_date": date, "home": H,
                         "away": A, "line_type": "main", "total": m_total, "home_spread": m_spread,
                         "our_total_topdown": round(total, 1), "our_total_bottomup": round(bu, 1),
                         "our_margin": round(margin, 1), "our_home_win": p.get("home_win")})

    OUT.write_text(json.dumps({
        "generated": now.astimezone(GMT8).strftime("%d/%m/%Y %H:%M GMT+8"),
        "generated_epoch": int(now.timestamp()),
        "weight_on_our_total": W_OURS, "sigma_total": round(st, 2), "sigma_margin": round(sm, 2),
        "sigma_team_total": round(sigma_team, 2),
        "note": ("our_total is the ratings model; blended_total is what the ladder is drawn on "
                 "(35% ours, 65% the market when a market total exists). bottom_up_total is the sum of tonight's "
                 "player projections, so it reacts to injuries the ratings cannot see; a gap of 8+ points is "
                 "flagged. Team ladders are that team's own points, not the game total."),
        "games": rows}, separators=(",", ":")), encoding="utf-8")

    if hist:
        new = not GHIST.exists()
        with open(GHIST, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HIST_COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(hist)

    log(f"Wrote {OUT} for {len(rows)} games.")
    for r in rows:
        mt = f"{r['market_total']:g}" if r["market_total"] else "no market"
        bu = f"{r['bottom_up_total']:g}" if r["bottom_up_total"] else "n/a"
        log(f"  {r['away']} at {r['home']:<4} ours {r['our_total']:>6.1f}  market {mt:>9}  "
            f"blended {r['blended_total']:>6.1f}  bottom-up {bu:>6}  "
            f"min/team {r.get('expected_minutes_per_team') or 0:.0f}"
            + (f"  [{r['flag']}]" if r.get("flag") else ""))
    log(f"  appended {len(hist)} market lines to {GHIST}")
    return 0


def grade(now=None):
    """How close is our total to the closing market total, and is the spread of the distribution right?"""
    now = now or datetime.now(timezone.utc)
    if not GHIST.exists():
        log(f"No {GHIST} yet: team_totals.py appends one row per game line per run. Nothing to grade.")
        return 0
    games = {}
    if PGGAMES.exists():
        agg = defaultdict(float)
        with open(PGGAMES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["season_type"] not in ("2", "3", "5") or r["dnp"] == "1":
                    continue
                try:
                    if float(r["min"]) <= 0:
                        continue
                    agg[(r["game_date_et"], r["team"])] += float(r["pts"] or 0)
                except ValueError:
                    continue
        games = agg
    latest = {}
    with open(GHIST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["game_date"], r["home"], r["away"])
            if k not in latest or r["fetched_utc"] >= latest[k]["fetched_utc"]:
                latest[k] = r
    rows, skipped = [], 0
    for (date, home, away), r in latest.items():
        try:
            actual = games[(date, home)] + games[(date, away)]
            mt = float(r["total"])
        except (KeyError, ValueError):
            skipped += 1
            continue
        if actual <= 0:
            skipped += 1
            continue
        rows.append({"date": date, "home": home, "away": away, "market_total": mt, "actual": actual,
                     "our_total": float(r["our_total_topdown"]),
                     "bottom_up": float(r["our_total_bottomup"] or 0)})
    if not rows:
        log(f"no graded game lines yet ({skipped} saved lines are waiting for a final score)")
        GRADE.write_text(json.dumps({"games": 0, "note": "no finished games with a saved line yet"}, indent=2),
                         encoding="utf-8")
        return 0
    def mae(key):
        return round(sum(abs(r[key] - r["actual"]) for r in rows) / len(rows), 2)
    out = {"graded_at": now.astimezone(GMT8).strftime("%d/%m/%Y %H:%M GMT+8"), "games": len(rows),
           "mae_market_total": mae("market_total"), "mae_our_total": mae("our_total"),
           "bias_market_total": round(sum(r["market_total"] - r["actual"] for r in rows) / len(rows), 2),
           "bias_our_total": round(sum(r["our_total"] - r["actual"] for r in rows) / len(rows), 2),
           "note": ("mean absolute error of a game total against the final score. The market column is what the "
                    "bookmaker had; if it is not clearly better than ours, our team number is worth something. "
                    "Judge on 200+ games, not 20.")}
    if rows and any(r["bottom_up"] for r in rows):
        sub = [r for r in rows if r["bottom_up"]]
        if sub:
            out["mae_bottom_up_total"] = round(sum(abs(r["bottom_up"] - r["actual"]) for r in sub) / len(sub), 2)
            out["bottom_up_games"] = len(sub)
    GRADE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return run(grade_only="--grade" in argv)


if __name__ == "__main__":
    sys.exit(main())
