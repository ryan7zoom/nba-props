"""Grade our player numbers against real results and against the sportsbook lines.

Reads data/lines_history.csv (written by lines.py: every book line we saw, with what our model said at that
moment) and data/player_games.csv (box scores). For every line whose game has been played it asks:
  - how often did the Over / Under actually win?
  - who predicted better: our model, the market's own fair odds, or a blend of the two?
  - what would betting the sides our "edge" picked have returned?

This is how we find out how much weight our numbers deserve against the market. Early on the sample is small;
the file says so. Writes data/lines_grade.json. Run:  python scripts/grade_lines.py
"""
import csv
import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")
HISTORY = DATA / "lines_history.csv"
PLAYER_GAMES = DATA / "player_games.csv"
OUT = DATA / "lines_grade.json"
GMT8 = timezone(timedelta(hours=8))
W_GRID = [i / 20 for i in range(21)]
RAW_THRESHOLDS = [0.0, 0.05, 0.10, 0.20]
BLEND_THRESHOLD = 0.03
MIN_N = 300            # below this the answers are noise


def log(msg=""):
    print(msg, flush=True)


def norm(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"})


def dec(a):
    return 1 + a / 100.0 if a > 0 else 1 + 100.0 / -a


def load_results():
    out = {}
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dnp"] == "1":
                continue
            try:
                if float(r["min"]) <= 0:
                    continue
                pts, reb, ast = int(r["pts"] or 0), int(r["reb"] or 0), int(r["ast"] or 0)
            except ValueError:
                continue
            out[(r["game_date_et"], norm(r["player"]), r["team"])] = {"pts": pts, "pra": pts + reb + ast}
    return out


def load_rows():
    """Latest snapshot per (game, player, market, book, line): the one closest to tip-off."""
    latest = {}
    with open(HISTORY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["game_date"], r["team"], r["player"], r["market"], r["book"], r["line"])
            if k not in latest or r["fetched_utc"] >= latest[k]["fetched_utc"]:
                latest[k] = r
    return list(latest.values())


def grade(rows, results):
    graded = []
    for r in rows:
        res = results.get((r["game_date"], norm(r["player"]), r["team"]))
        if not res:
            continue
        try:
            line = float(r["line"])
            o, u = int(float(r["over_price"])), int(float(r["under_price"]))
            po, pu = float(r["our_p_over"]), float(r["our_p_under"])
        except (ValueError, TypeError):
            continue
        actual = res[r["market"]]
        if actual == line:
            continue
        io, iu = 1 / dec(o), 1 / dec(u)
        fair_o = io / (io + iu)
        graded.append({"market": r["market"], "line": line, "o": o, "u": u, "ours_o": po, "ours_u": pu, "fair_o": fair_o,
                       "over_won": actual > line, "game_date": r["game_date"], "player": r["player"]})
    return graded


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)


def logloss(ps, ys):
    e = 1e-6
    return -sum(math.log(max(e, p) if y else max(e, 1 - p)) for p, y in zip(ps, ys)) / len(ys)


def bets(graded, picker):
    """picker(row) -> ('o'|'u'|None). Returns count, win rate, return per unit staked."""
    n = w = 0
    profit = 0.0
    for g in graded:
        side = picker(g)
        if side is None:
            continue
        n += 1
        won = g["over_won"] if side == "o" else not g["over_won"]
        price = g["o"] if side == "o" else g["u"]
        w += won
        profit += (dec(price) - 1) if won else -1.0
    return {"bets": n, "win_rate": round(w / n, 3) if n else None, "return_per_unit": round(profit / n, 3) if n else None}


def run():
    if not HISTORY.exists():
        log("No data/lines_history.csv yet: lines.py writes it each time it runs. Nothing to grade.")
        return 0
    if not PLAYER_GAMES.exists():
        log("data/player_games.csv is missing")
        return 1
    rows = load_rows()
    graded = grade(rows, load_results())
    now = datetime.now(timezone.utc).astimezone(GMT8)
    rep = {"graded_at": now.strftime("%d/%m/%Y %H:%M") + " GMT+8", "lines_seen": len(rows), "lines_graded": len(graded),
           "game_days": len({g["game_date"] for g in graded}), "enough_data": len(graded) >= MIN_N}
    if not graded:
        rep["note"] = "no finished games with saved lines yet"
        OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log("Nothing graded yet: no finished games with saved lines.")
        return 0
    ys = [1.0 if g["over_won"] else 0.0 for g in graded]
    ours, fair = [g["ours_o"] for g in graded], [g["fair_o"] for g in graded]
    best_w, best_ll = 0.0, None
    for w in W_GRID:
        ll = logloss([w * a + (1 - w) * b for a, b in zip(ours, fair)], ys)
        if best_ll is None or ll < best_ll:
            best_w, best_ll = w, ll
    rep["over_win_rate"] = round(sum(ys) / len(ys), 3)
    rep["accuracy"] = {"brier_ours": round(brier(ours, ys), 4), "brier_market": round(brier(fair, ys), 4),
                       "brier_best_blend": round(brier([best_w * a + (1 - best_w) * b for a, b in zip(ours, fair)], ys), 4),
                       "logloss_ours": round(logloss(ours, ys), 4), "logloss_market": round(logloss(fair, ys), 4),
                       "logloss_best_blend": round(best_ll, 4), "best_weight_on_our_model": best_w,
                       "note": "lower is better. weight 0 = trust the market only, 1 = trust our model only"}

    def raw_ev(g):
        eo = g["ours_o"] * (dec(g["o"]) - 1) - g["ours_u"]
        eu = g["ours_u"] * (dec(g["u"]) - 1) - g["ours_o"]
        return ("o", eo) if eo >= eu else ("u", eu)

    def blend_ev(g, w):
        pb = w * g["ours_o"] + (1 - w) * g["fair_o"]
        eo = pb * (dec(g["o"]) - 1) - (1 - pb)
        eu = (1 - pb) * (dec(g["u"]) - 1) - pb
        return ("o", eo) if eo >= eu else ("u", eu)

    rep["bets_by_raw_edge"] = {}
    for th in RAW_THRESHOLDS:
        rep["bets_by_raw_edge"][f"raw edge above {int(th * 100)}%"] = {
            "all": bets(graded, lambda g, th=th: raw_ev(g)[0] if raw_ev(g)[1] > th else None),
            "overs only": bets(graded, lambda g, th=th: "o" if raw_ev(g)[0] == "o" and raw_ev(g)[1] > th else None),
            "unders only": bets(graded, lambda g, th=th: "u" if raw_ev(g)[0] == "u" and raw_ev(g)[1] > th else None)}
    rep["bets_by_blended_edge"] = {
        f"blended edge above {int(BLEND_THRESHOLD * 100)}% (weight {best_w})": bets(graded, lambda g: blend_ev(g, best_w)[0] if blend_ev(g, best_w)[1] > BLEND_THRESHOLD else None)}
    rep["every_main_line_over"] = bets(graded, lambda g: "o")
    rep["every_main_line_under"] = bets(graded, lambda g: "u")
    if not rep["enough_data"]:
        rep["warning"] = f"only {len(graded)} graded lines; wait for {MIN_N}+ before trusting any of this"
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    a = rep["accuracy"]
    log(f"Graded {len(graded)} lines from {rep['game_days']} game days. Over won {rep['over_win_rate']:.1%}.")
    log(f"  Brier (lower is better): ours {a['brier_ours']}, market {a['brier_market']}, best blend {a['brier_best_blend']} (weight on our model {best_w})")
    for k, v in rep["bets_by_raw_edge"].items():
        log(f"  {k}: unders {v['unders only']} | overs {v['overs only']}")
    log(f"  every main-line under: {rep['every_main_line_under']}")
    if not rep["enough_data"]:
        log(f"  WARNING: {rep['warning']}")
    log(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
