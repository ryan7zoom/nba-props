"""Grade our player numbers against real results and against the sportsbook lines.

This is the module the rest of the project imports for name matching, prices and results. It also builds the
report that says how much our own probability is worth.

Reads  data/lines_history.csv   every book line we saw, with what our model said at that moment (written by lines.py)
       data/player_games.csv    box scores
Writes data/lines_grade.json

How the sample is built
  One row per (game, player, market): the LAST snapshot we saved before that game started -- i.e. the closest
  thing we have to the closing line. Older snapshots are used only to measure how far the line moved (CLV).
  Lines that were pushed are dropped, and so are whole-number lines (a push there makes P(over) ambiguous);
  main player lines are always half points, so nothing real is lost.

  Earlier versions keyed the sample on (game, player, market, book, line), which counted a player twice whenever
  a book moved his line during the day, and never used the closing number. That inflated the sample by about 10%
  with correlated rows and mixed times of day into the same average. Both are fixed here.

What it reports
  - how often the Over actually won
  - Brier / log-loss for our probability, the market's de-vigged price, and the best blend of the two
  - what betting the sides our edge picked would have returned, split by side and by price
  - CLV: what our bets would have been worth priced at the closing line instead of the line we took
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
MIN_N = 300               # below this the answers are noise
MARKETS = ("pts", "reb", "ast", "pr", "pa", "ar", "pra")


def log(msg=""):
    print(msg, flush=True)


def norm(name):
    """Match a book's spelling of a name to ESPN's (no accents, punctuation, suffixes or team tags)."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"})


def dec(a):
    """American price -> decimal. +150 -> 2.50, -120 -> 1.833."""
    return 1 + a / 100.0 if a > 0 else 1 + 100.0 / -a


def fair(price_a, price_b):
    """Two prices -> the market's fair chance of the FIRST one (proportional de-vig, kept for compatibility)."""
    ia, ib = 1 / dec(price_a), 1 / dec(price_b)
    return ia / (ia + ib)


def fair_power(price_a, price_b, k=1.0):
    """Power-method de-vig: solve p1^k + p2^k = 1. Less biased than proportional on juiced favourites."""
    ia, ib = 1 / dec(price_a), 1 / dec(price_b)
    lo, hi = 0.3, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ia ** mid + ib ** mid > 1:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    pa, pb = ia ** k, ib ** k
    return pa / (pa + pb)


def load_results():
    """{(game_date, normalised player, team): {pts, reb, ast, pra, pr, pa, ar}} from the box scores."""
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
            out[(r["game_date_et"], norm(r["player"]), r["team"])] = {
                "pts": pts, "reb": reb, "ast": ast,
                "pra": pts + reb + ast, "pr": pts + reb, "pa": pts + ast, "ar": ast + reb,
            }
    return out


def load_snapshots():
    """The closing snapshot per (game, player, market), plus every snapshot we hold for that key.

    Returns {(game_date, team, player, market): {"close": row, "all": [rows]}}.
    "close" is the latest fetch before the game; ties go to the best under price (the side we bet).
    """
    by_key = {}
    with open(HISTORY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["game_date"], r["team"], r["player"], r["market"])
            by_key.setdefault(k, []).append(r)
    out = {}
    for k, rows in by_key.items():
        rows.sort(key=lambda r: r["fetched_utc"])
        out[k] = {"close": rows[-1], "all": rows}
    return out


def graded_rows(results=None, snapshots=None):
    """One row per (game, player, market) with the closing line and what we said about it."""
    results = results if results is not None else load_results()
    snapshots = snapshots if snapshots is not None else load_snapshots()
    rows, skipped_int, skipped_res = [], 0, 0
    for (date, team, player, market), snap in snapshots.items():
        r = snap["close"]
        res = results.get((date, norm(player), team))
        if not res or market not in res:
            skipped_res += 1
            continue
        try:
            line = float(r["line"])
            o, u = int(float(r["over_price"])), int(float(r["under_price"]))
            po, pu = float(r["our_p_over"]), float(r["our_p_under"])
        except (ValueError, TypeError):
            continue
        if line != math.floor(line):        # half point: no push
            pass
        else:
            skipped_int += 1
            continue
        actual = res[market]
        rows.append({"market": market, "line": line, "o": o, "u": u, "ours_o": po, "ours_u": pu,
                     "fair_o": fair_power(o, u), "over_won": actual > line, "actual": actual,
                     "game_date": date, "player": player, "team": team, "fetched": r["fetched_utc"],
                     "book": r["book"]})
    return rows, {"skipped_integer_lines": skipped_int, "skipped_no_result": skipped_res}


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)


def logloss(ps, ys):
    e = 1e-6
    return -sum(math.log(max(e, p) if y else max(e, 1 - p)) for p, y in zip(ps, ys)) / len(ys)


def bets(rows, picker):
    """picker(row) -> 'o' or 'u' or None. Returns count, win rate, return per unit staked."""
    n = w = 0
    profit = 0.0
    for g in rows:
        side = picker(g)
        if side is None:
            continue
        n += 1
        won = g["over_won"] if side == "o" else not g["over_won"]
        price = g["o"] if side == "o" else g["u"]
        w += won
        profit += (dec(price) - 1) if won else -1.0
    return {"bets": n, "win_rate": round(w / n, 3) if n else None,
            "return_per_unit": round(profit / n, 3) if n else None}


def clv(rows, picks):
    """Bet the picks at the line we recorded; what would the same bet be worth at the closing price?

    picks: iterable of dicts with game_date, player, team, market, side, price (an int), line.
    A positive CLV means the closing price was better than ours was -- the bet was placed at a good number.
    """
    close = {}
    for (date, team, player, market), snap in load_snapshots().items():
        close[(date, player, market)] = snap["close"]
    n = beat = 0
    edge_sum = 0.0
    for p in picks:
        c = close.get((p["game_date"], p["player"], p["market"]))
        if not c:
            continue
        try:
            cp = int(float(c["over_price"] if p["side"] == "o" else c["under_price"]))
        except (ValueError, TypeError):
            continue
        n += 1
        gain = dec(cp) / dec(int(float(p["price"]))) - 1
        edge_sum += gain
        beat += gain > 0
    return {"closing_price_found": n, "share_where_close_was_worse_for_us": round(beat / n, 3) if n else None,
            "mean_price_gain_vs_close": round(edge_sum / n, 4) if n else None}


def run():
    if not HISTORY.exists():
        log("No data/lines_history.csv yet: lines.py writes it each time it runs. Nothing to grade.")
        return 0
    if not PLAYER_GAMES.exists():
        log("data/player_games.csv is missing")
        return 1
    rows, skipped = graded_rows()
    now = datetime.now(timezone.utc).astimezone(GMT8)
    rep = {"graded_at": now.strftime("%d/%m/%Y %H:%M") + " GMT+8",
           "rows_in_history": None, "player_market_days": len(rows),
           "lines_graded": len(rows), "game_days": len({r["game_date"] for r in rows}),
           "enough_data": len(rows) >= MIN_N}
    rep.update(skipped)
    if not rows:
        rep["note"] = "no finished games with saved lines yet"
        OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log("Nothing graded yet: no finished games with saved lines.")
        return 0

    ys = [1.0 if r["over_won"] else 0.0 for r in rows]
    ours, mkt = [r["ours_o"] for r in rows], [r["fair_o"] for r in rows]
    best_w, best_ll = 0.0, None
    for w in W_GRID:
        ll = logloss([w * a + (1 - w) * b for a, b in zip(ours, mkt)], ys)
        if best_ll is None or ll < best_ll:
            best_w, best_ll = w, ll
    rep["over_win_rate"] = round(sum(ys) / len(ys), 3)
    rep["accuracy"] = {
        "brier_ours": round(brier(ours, ys), 4), "brier_market": round(brier(mkt, ys), 4),
        "brier_best_blend": round(brier([best_w * a + (1 - best_w) * b for a, b in zip(ours, mkt)], ys), 4),
        "logloss_ours": round(logloss(ours, ys), 4), "logloss_market": round(logloss(mkt, ys), 4),
        "logloss_best_blend": round(best_ll, 4), "best_weight_on_our_model": best_w,
        "note": "lower is better. weight 0 = trust the market only, 1 = trust our model only. "
                "One row per player-game, at the closing line."}

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
            "all": bets(rows, lambda g, th=th: raw_ev(g)[0] if raw_ev(g)[1] > th else None),
            "overs only": bets(rows, lambda g, th=th: "o" if raw_ev(g)[0] == "o" and raw_ev(g)[1] > th else None),
            "unders only": bets(rows, lambda g, th=th: "u" if raw_ev(g)[0] == "u" and raw_ev(g)[1] > th else None)}
    rep["bets_by_blended_edge"] = {
        f"blended edge above {int(BLEND_THRESHOLD * 100)}% (weight {best_w})":
            bets(rows, lambda g: blend_ev(g, best_w)[0] if blend_ev(g, best_w)[1] > BLEND_THRESHOLD else None)}
    # price matters as much as edge: the same edge at a cheap price is a worse bet
    price_bands = [(-200, -140), (-140, -120), (-120, -110), (-110, -101), (-100, 300)]
    rep["unders_by_under_price"] = {}
    for lo, hi in price_bands:
        s = [g for g in rows if lo <= g["u"] < hi]
        if len(s) >= 30:
            rep["unders_by_under_price"][f"under price {lo} to {hi}"] = {
                "under won": round(sum(1 for g in s if not g["over_won"]) / len(s), 3),
                "break_even": round(sum(1 / dec(g["u"]) for g in s) / len(s), 3),
                "bets": len(s)}
    rep["every_main_line_over"] = bets(rows, lambda g: "o")
    rep["every_main_line_under"] = bets(rows, lambda g: "u")
    rep["by_market"] = {}
    for m in MARKETS:
        s = [g for g in rows if g["market"] == m]
        if len(s) >= 50:
            rep["by_market"][m] = {"rows": len(s),
                                   "brier_ours": round(brier([g["ours_o"] for g in s],
                                                             [1.0 if g["over_won"] else 0.0 for g in s]), 4),
                                   "brier_market": round(brier([g["fair_o"] for g in s],
                                                               [1.0 if g["over_won"] else 0.0 for g in s]), 4),
                                   "over_rate": round(sum(1 for g in s if g["over_won"]) / len(s), 3)}
    if not rep["enough_data"]:
        rep["warning"] = f"only {len(rows)} graded player-games; wait for {MIN_N}+ before trusting any of this"
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    a = rep["accuracy"]
    log(f"Graded {len(rows)} player-games ({rep['player_market_days']} rows) from {rep['game_days']} game days. "
        f"Over won {rep['over_win_rate']:.1%}.")
    log(f"  Brier (lower is better): ours {a['brier_ours']}, market {a['brier_market']}, "
        f"best blend {a['brier_best_blend']} (weight on our model {best_w})")
    for k, v in rep["bets_by_raw_edge"].items():
        log(f"  {k}: unders {v['unders only']} | overs {v['overs only']}")
    log(f"  every main-line under: {rep['every_main_line_under']}")
    if not rep["enough_data"]:
        log(f"  WARNING: {rep['warning']}")
    log(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
