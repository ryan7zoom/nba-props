"""Learn the two numbers the model never learned: how good our probability is, and who actually suits up.

The pipeline produces a probability for every line. Until now that probability was used at face value, with a
fixed 35% weight against the bookmaker's price (shortlist.py MW, index.html MW, game_model.py W_OURS). Two things
were never measured:

  1. Is our probability calibrated? On real book lines it is not: at the main line our model says 36% over when
     46% actually happens, and 63% when 51% happens. The backtest cannot see this because it invents lines from
     our own median. This script measures the correction on real lines and writes it to data/calibration.json.

  2. Does a player who is not on the injury report actually play? The minutes model assumed 100%. Measured on
     three seasons of box scores: 88% for a 20+ minute player, 81% for 14-20, 64% for 8-14, 42% below that.
     Written to data/appear_rates.json.

Modes
  python scripts/calibrate.py            learn from everything on disk (run after grade_lines.py)
  python scripts/calibrate.py --report   print the current calibration without refitting

Reads  data/lines_history.csv, data/player_games.csv, data/injuries_official_history.csv
Writes data/calibration.json, data/appear_rates.json, data/calibration_report.json
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import grade_lines as gl

DATA = Path("data")
CAL = DATA / "calibration.json"
APPEAR = DATA / "appear_rates.json"
REPORT = DATA / "calibration_report.json"

GMT8 = timezone(timedelta(hours=8))
MIN_ROWS = 500            # below this, do not fit anything: fall back to a plain market blend
MIN_BIN = 250             # a probability bin needs this many lines before it is used as a knot
EDGES = (0.0, 0.20, 0.30, 0.40, 0.475, 0.525, 0.60, 0.70, 0.80, 1.0)
W_GRID = [round(i / 40, 4) for i in range(0, 41)]
W_FLOOR, W_CEIL = 0.10, 0.60     # a weight outside this is degenerate (ignore the market, or ignore the model)
SIMPLER_WINS_UNLESS = 0.0015     # a richer correction must cut out-of-sample log-loss by this much
MARKETS = ("pts", "reb", "ast", "pr", "pa", "ar", "pra")


def log(msg=""):
    print(msg, flush=True)


def sig(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def logit(p, e=1e-4):
    p = min(max(p, e), 1 - e)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- monotone recalibration map

def _pava(points):
    """Pool adjacent violators: make y non-decreasing in x, weighted by n. points = [[x, y, n], ...]."""
    blocks = [[p[0], p[1], p[2]] for p in points]     # x, weighted y, weight
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] <= blocks[i + 1][1]:
            i += 1
            continue
        a, b = blocks[i], blocks[i + 1]
        w = a[2] + b[2]
        merged = [a[0], (a[1] * a[2] + b[1] * b[2]) / w, w]
        blocks[i:i + 2] = [merged]
        if i:
            i -= 1
    return blocks


def fit_map(rows):
    """Monotone map from our probability to the observed frequency, on binned data.

    Only bins with enough lines become knots. Whatever is outside them is linear to (0,0) and (1,1), which keeps
    the map monotone and leaves the very safe lines (0.9+) roughly where the simulation put them -- they cannot be
    validated from main-line data, and inventing a correction there would break the safety ladder.
    """
    pts = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = [r for r in rows if lo <= r["ours_o"] < hi]
        if len(sel) < MIN_BIN:
            continue
        x = sum(r["ours_o"] for r in sel) / len(sel)
        y = sum(1.0 if r["over_won"] else 0.0 for r in sel) / len(sel)
        pts.append([x, y, float(len(sel)), len(sel)])
    if len(pts) < 2:
        return None, []
    blocks = _pava([[p[0], p[1], p[2]] for p in pts])
    knots = [[0.0, 0.0]] + [[round(b[0], 4), round(min(max(b[1], 0.0), 1.0), 4)] for b in blocks] + [[1.0, 1.0]]
    # drop knots whose x is not strictly increasing (can happen after pooling)
    clean = [knots[0]]
    for k in knots[1:]:
        if k[0] > clean[-1][0] + 1e-9:
            clean.append(k)
    detail = [{"our_probability": round(p[0], 4), "actual_frequency": round(p[1], 4), "rows": p[3]} for p in pts]
    return clean, detail


def apply_map(p, knots):
    """Piecewise-linear monotone map. Identity when there are no knots."""
    if not knots:
        return p
    p = min(max(p, 0.0), 1.0)
    if p <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots[:-1], knots[1:]):
        if x0 <= p <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return knots[-1][1]


def best_weight(rows, knots):
    """Weight on our (map-corrected) probability vs the market's, by log-loss on the same rows."""
    best, best_ll = 0.0, None
    for w in W_GRID:
        ll = 0.0
        for r in rows:
            p = apply_map(r["ours_o"], knots)
            pb = w * p + (1 - w) * r["fair_o"]
            y = 1.0 if r["over_won"] else 0.0
            pb = min(max(pb, 1e-6), 1 - 1e-6)
            ll -= math.log(pb if y else 1 - pb)
        ll /= len(rows)
        if best_ll is None or ll < best_ll - 1e-12:
            best, best_ll = w, ll
    return max(W_FLOOR, min(W_CEIL, best)), best_ll


def score_variant(rows, knots, w):
    ll = br = 0.0
    for r in rows:
        p = apply_map(r["ours_o"], knots)
        pb = min(max(w * p + (1 - w) * r["fair_o"], 1e-6), 1 - 1e-6)
        y = 1.0 if r["over_won"] else 0.0
        ll -= math.log(pb if y else 1 - pb)
        br += (pb - y) ** 2
    return {"logloss": round(ll / len(rows), 5), "brier": round(br / len(rows), 5), "rows": len(rows)}


# ---------------------------------------------------------------- who actually plays

APPEAR_BUCKETS = ((20.0, "20+ minutes"), (14.0, "14-20 minutes"), (8.0, "8-14 minutes"), (0.0, "under 8 minutes"))
DEFAULT_APPEAR = 0.85


def appearance_rates():
    """P(player appears in the box score | he is NOT on the official injury report), by recent minutes.

    This is the number the minutes model was assuming was 100%. It is measured on rotation candidates only:
    players who appeared for that team in at least 2 of its previous 10 games and whose last game was for it.
    Names are matched against the official report with the same normaliser the report parser uses (0.01% error
    on this data).
    """
    rows = []
    with open(DATA / "player_games.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["season_type"] not in ("2", "3", "5") or r["dnp"] == "1":
                continue
            try:
                if float(r["min"]) <= 0:
                    continue
            except ValueError:
                continue
            rows.append((r["game_date_et"], r["team"], r["player"], float(r["min"])))
    rows.sort()
    by_slot = defaultdict(dict)
    for d, t, p, m in rows:
        by_slot[(d, t)][p] = m
    team_days = defaultdict(list)
    for d, t, p, m in rows:
        if d not in team_days[t]:
            team_days[t].append(d)
    for t in team_days:
        team_days[t].sort()
    listed = defaultdict(set)
    hist_path = DATA / "injuries_official_history.csv"
    if hist_path.exists():
        with open(hist_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                listed[(r["game_date_et"], r["team"])].add(r["player_norm"])
    last_team = {}
    counts = {label: [0, 0] for _, label in APPEAR_BUCKETS}
    team_games = 0
    for team, days in team_days.items():
        hist = []
        for d in days:
            games = defaultdict(int)
            mins = defaultdict(float)
            for g in hist[-10:]:
                for a, m in g.items():
                    games[a] += 1
                    mins[a] += m
            cand = {a: mins[a] / games[a] for a in games if games[a] >= 2 and last_team.get(a) == team}
            lie = listed.get((d, team)) or set()
            if lie and cand:
                team_games += 1
                for a, mpg in cand.items():
                    if gl.norm(a) in lie:
                        continue
                    label = next(lab for floor, lab in APPEAR_BUCKETS if mpg >= floor)
                    counts[label][0] += 1
                    counts[label][1] += 1 if a in by_slot[(d, team)] else 0
            for a, m in by_slot[(d, team)].items():
                last_team[a] = team
            hist.append(dict(by_slot[(d, team)]))
    buckets = []
    for floor, label in APPEAR_BUCKETS:
        n, played = counts[label]
        rate = round(played / n, 3) if n >= 200 else DEFAULT_APPEAR
        buckets.append({"min_recent_minutes": floor, "label": label, "appearance_rate": rate,
                        "rows": n, "measured": n >= 200})
    return {"buckets": buckets, "team_games_measured": team_games,
            "note": "Chance a player who is NOT on the official injury report still appears. Used by "
                    "minutes_model.py when no status is on file; before this, 1.0 was assumed."}


def appear_prior(mpg, appear=None):
    """Look up the appearance prior for a player with the given recent minutes."""
    appear = appear or DEFAULT_APPEAR_SETTINGS
    for b in appear.get("buckets", []):
        if mpg >= b["min_recent_minutes"]:
            return b["appearance_rate"]
    return DEFAULT_APPEAR


DEFAULT_APPEAR_SETTINGS = {"buckets": [{"min_recent_minutes": 0.0, "appearance_rate": DEFAULT_APPEAR}]}


def load_appear():
    try:
        return json.loads(APPEAR.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_APPEAR_SETTINGS


# ---------------------------------------------------------------- load the fitted settings

def load_calibration():
    try:
        return json.loads(CAL.read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "blend_only", "global": {"knots": [], "w": 0.35}, "by_market": {},
                "note": "no data/calibration.json yet: plain 35% blend toward the market, as before"}


def apply_calibration(p, market, cal, market_fair=None):
    """Turn one of our probabilities into the number used for decisions.

    market_fair: the bookmaker's de-vigged chance at the SAME line, when he posts that line. Without a price there
    is nothing to blend with, so the correction stands on its own (that is the case for an alt line your bookie
    has not published yet).
    """
    spec = (cal.get("by_market") or {}).get(market) or cal.get("global") or {}
    knots = spec.get("knots") or []
    w = spec.get("w", 0.35)
    p2 = apply_map(p, knots)
    if market_fair is None:
        return min(max(p2, 0.005), 0.995)
    return min(max(w * p2 + (1 - w) * market_fair, 0.005), 0.995)


# ---------------------------------------------------------------- fit and write

def run(report_only=False):
    now = datetime.now(timezone.utc).astimezone(GMT8)
    if report_only:
        cal = load_calibration()
        log(json.dumps(cal, indent=2))
        return 0
    rows, skipped = gl.graded_rows()
    rep = {"fitted_at": now.strftime("%d/%m/%Y %H:%M") + " GMT+8", "rows": len(rows)}
    rep.update(skipped)

    appear = appearance_rates()
    APPEAR.write_text(json.dumps(dict(appear, fitted_utc=now.isoformat(timespec="seconds")), indent=2),
                      encoding="utf-8")
    log("Appearance priors (player NOT on the injury report):")
    for b in appear["buckets"]:
        log(f"  {b['label']:<16} {b['appearance_rate']:.1%}  ({b['rows']} team-game slots)")

    if len(rows) < MIN_ROWS:
        cal = {"mode": "blend_only", "fitted_at": now.strftime("%d/%m/%Y %H:%M") + " GMT+8", "rows": len(rows),
               "global": {"knots": [], "w": 0.35}, "by_market": {},
               "note": f"only {len(rows)} graded player-games (need {MIN_ROWS}): keeping the plain 35% market blend"}
        CAL.write_text(json.dumps(cal, indent=2), encoding="utf-8")
        REPORT.write_text(json.dumps(dict(rep, calibration=cal), indent=2), encoding="utf-8")
        log(f"Not enough graded rows yet: wrote the fallback to {CAL}")
        return 0

    # --- hold out the most recent third by date, fit on the rest, and let the data choose how fancy to be
    days = sorted({r["game_date"] for r in rows})
    cut = days[int(len(days) * 0.65)]
    tr = [r for r in rows if r["game_date"] < cut]
    te = [r for r in rows if r["game_date"] >= cut]
    knots_tr, detail = (None, [])
    if len(tr) >= MIN_ROWS:
        knots_tr, detail = fit_map(tr)
    w_tr, _ = best_weight(tr, knots_tr) if tr else (0.35, None)
    variants = {
        "raw model": (None, 1.0),
        "market blend only": (None, w_tr),
        "recalibrated": (knots_tr, 1.0),
        "recalibrated + market blend": (knots_tr, w_tr),
    }
    scored = {k: score_variant(te, kn, w) for k, (kn, w) in variants.items()}
    # parsimony: simplest first, richer versions must clearly beat it out of sample
    order = ["market blend only", "raw model", "recalibrated", "recalibrated + market blend"]
    best_name = order[0]
    for name in order[1:]:
        if scored[name]["logloss"] < scored[best_name]["logloss"] * (1 - SIMPLER_WINS_UNLESS):
            best_name = name
    rep["holdout"] = {"split_date": cut, "train_rows": len(tr), "test_rows": len(te),
                      "scores": scored, "chosen": best_name,
                      "note": "log-loss on the held-out days; a richer correction must win by "
                              f"{SIMPLER_WINS_UNLESS:.2%} to be used"}

    # --- final fit on everything, keeping the structure that won on the holdout
    use_map = "recalibrated" in best_name
    knots = None
    detail = []
    if use_map:
        knots, detail = fit_map(rows)
    w, _ = best_weight(rows, knots)
    if best_name == "raw model":
        w = 1.0
    cal = {
        "mode": best_name, "fitted_at": now.strftime("%d/%m/%Y %H:%M") + " GMT+8", "rows": len(rows),
        "game_days": len(days), "global": {"knots": knots or [], "w": round(w, 3)},
        "by_market": {},
        "reliability": detail,
        "note": ("p_used = w * map(p_model) + (1-w) * market_fair, where map() is a monotone correction learned "
                 "on real closing lines and market_fair is the bookmaker's de-vigged price at the SAME line. "
                 "Where the book posts no price for a line, only map(p_model) is used."),
    }
    # Per-market SHAPE (the reliability curve differs: PRA's main lines sit further from our median), but one
    # blend weight for every market. A per-market weight is a 2,000-row fit whose standard error is larger than
    # the differences between markets, and letting it float produced implausible weights (0.25 for points, 0.60
    # for PRA off the same data). The weight is the one number that must not be fitted per market.
    for m in MARKETS:
        sub = [r for r in rows if r["market"] == m]
        if len(sub) < MIN_ROWS:
            continue
        km, det = fit_map(sub) if use_map else (None, [])
        cal["by_market"][m] = {"knots": km or [], "w": round(w, 3), "rows": len(sub), "reliability": det}
    CAL.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(dict(rep, calibration={k: cal[k] for k in ("mode", "rows", "global", "by_market")},
                                      reliability=detail), indent=2), encoding="utf-8")

    log("\nHeld-out comparison (lower log-loss is better):")
    for k, v in sorted(scored.items(), key=lambda kv: kv[1]["logloss"]):
        log(f"  {k:<28} logloss {v['logloss']:.4f}  brier {v['brier']:.4f}"
            + ("   <- used" if k == best_name else ""))
    log(f"\nCalibration map (our probability -> what actually happened), {len(detail)} bins:")
    for d in detail:
        log(f"  {d['our_probability']:.3f} -> {d['actual_frequency']:.3f}  ({d['rows']} lines)")
    log(f"Weight on our number after that: {w:.2f} (rest on the market's price at the same line)")
    log(f"Wrote {CAL}, {APPEAR}, {REPORT}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="print the current calibration, do not refit")
    return run(report_only=p.parse_args(argv).report)


if __name__ == "__main__":
    sys.exit(main())
