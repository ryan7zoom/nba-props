"""Safety ladder: for every player and market, the lines that are safe at a chosen confidence, with the price
each one needs to be worth betting.

This is the answer to "what should I take at my bookie's alternate lines?" without needing the feed to publish
alternates. For each market it walks the simulated chance table and finds, for a series of confidence levels
(70% up to 99%), the TIGHTEST line that still clears that confidence:

    Under, 90% safe  =  the lowest line where P(under) is still >= 90%
    Over,  90% safe  =  the highest line where P(over)  is still >= 90%

Tightest = best price available at that safety, which is the whole point of shopping alternate lines. Every rung
carries the fair decimal price, the minimum price worth taking (after the edge you require), and -- when the feed
does publish that exact line -- the bookmaker's own price and the edge left on it.

Probabilities are the simulated chance tables (a real distribution of 20,000 draws per player-game), corrected by
data/calibration.json where that correction has been measured on real closing lines. A rung your bookie has not
priced cannot be checked against the market, so it shows the model's number alone and says so.

Reads  data/projections_latest.json, data/calibration.json, data/lines_latest.json (optional)
Writes data/alt_lines_latest.json, data/alt_lines.csv

Run:   python scripts/alt_lines.py
       python scripts/alt_lines.py --levels 0.85 0.90 0.95 --both-sides
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as cal_mod                       # noqa: E402
from grade_lines import dec, norm, fair_power                  # noqa: E402

DATA = Path("data")
PROJ = DATA / "projections_latest.json"
LINES = DATA / "lines_latest.json"
OUT_JSON = DATA / "alt_lines_latest.json"
OUT_CSV = DATA / "alt_lines.csv"

GMT8 = timezone(timedelta(hours=8))
MARKETS = ("pts", "pra", "reb", "ast", "pr", "pa", "ar")
LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "threes",
         "pr": "P+R", "pa": "P+A", "ar": "A+R", "pra": "PRA"}
LEVELS = (0.75, 0.85, 0.90, 0.95, 0.98)   # confidence levels shown per side
MIN_LEVEL = 0.70
MAX_RUNGS = 6                # per side, per market
MAX_BOOK = 3                 # book lines merged in per side
EDGE = 0.02                  # edge you want before a price is "worth taking"
MIN_PLAY = 0.75              # chance the player appears (news + measured appearance rate)
MIN_MINUTES = 8.0
MIN_MARKET_MEAN = 2.0        # a market whose average is under this is not worth laddering


def p_over_at(cdf, line):
    """P(stat > line) given cdf[k] = P(stat <= k)."""
    j = int(math.floor(line))
    if j < 0:
        return 1.0
    if j >= len(cdf):
        return 0.0
    return 1.0 - cdf[j]


def p_under_at(cdf, line):
    """P(stat < line). Half-point lines never push."""
    j = int(math.floor(line))
    if j < 0:
        return 0.0
    if j >= len(cdf):
        return 1.0
    return cdf[j]


def tightest_under(cdf, level):
    """Lowest line whose under chance still clears `level` -> (line, P(under)) or None."""
    for k, p in enumerate(cdf):
        if p >= level:
            return k + 0.5, float(p)
    return None


def tightest_over(cdf, level):
    """Highest line whose over chance still clears `level` -> (line, P(over)) or None."""
    best = None
    for j in range(len(cdf)):
        p = 1.0 - cdf[j]
        if p >= level:
            best = (j + 0.5, float(p))
        else:
            break
    return best


def min_price(p, edge=EDGE):
    """Smallest decimal price at which P(p) with edge still returns more than 1."""
    if p <= 0:
        return None
    return round((1.0 + edge) / p, 3)


def american(d):
    if not d or d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2 else -int(round(100 / (d - 1)))


def load_lines():
    """{(date, team, normalised name): {market: {line: {'o': price, 'u': price, 'book': key}}}} best price per line."""
    out = {}
    if not LINES.exists():
        return out
    try:
        j = json.loads(LINES.read_text(encoding="utf-8"))
    except Exception:
        return out
    for pl in j.get("players", []):
        key = (pl.get("d"), pl.get("t"), norm(pl.get("n", "")))
        rec = out.setdefault(key, {})
        for market, entry in pl.items():
            if market in ("n", "t", "d") or not isinstance(entry, dict):
                continue
            slot = rec.setdefault(market, {})
            for row in (entry.get("m") or []) + (entry.get("a") or []):
                try:
                    book, point, o, u = row[0], float(row[1]), row[2], row[3]
                except (ValueError, TypeError, IndexError):
                    continue
                cur = slot.setdefault(point, {})
                for side, price in (("o", o), ("u", u)):
                    if price is None:
                        continue
                    if side not in cur or dec(price) > dec(cur[side]):
                        cur[side] = price
                        cur[side + "_book"] = book
    return out



def _rung(cdf, market, line, side, level, slot, cal, edge):
    """One ladder entry: the model's read at `line`, the fair price, the price needed, and the book's price if any."""
    p_model = p_under_at(cdf, line) if side == "under" else p_over_at(cdf, line)
    price = slot.get("u" if side == "under" else "o")
    other = slot.get("o" if side == "under" else "u")
    book = slot.get("u_book" if side == "under" else "o_book")
    fair_side = None
    if price is not None and other is not None:
        fair_over = fair_power(other, price)
        fair_side = (1.0 - fair_over) if side == "under" else fair_over
    p_used = cal_mod.apply_calibration(min(max(p_model, 0.0005), 0.9995), market, cal, fair_side)
    r = {"level": round(level, 4), "line": line, "p_model": round(p_model, 4), "p": round(p_used, 4),
         "fair": round(1.0 / p_used, 3), "min": min_price(p_used, edge)}
    if price is not None:
        p_dec = dec(price)
        r.update({"book": book, "price": price, "dec": round(p_dec, 3),
                  "edge": round(p_used * p_dec - 1, 4), "worth": p_used * p_dec - 1 >= edge})
    return r


def build(levels, both_sides=True, edge=EDGE, max_rungs=MAX_RUNGS, max_book=MAX_BOOK):
    if not PROJ.exists():
        raise SystemExit("data/projections_latest.json is missing: run simulate_props.py --mode project first")
    proj = json.loads(PROJ.read_text(encoding="utf-8"))
    cal = cal_mod.load_calibration()
    lines = load_lines()
    rows, players = [], []
    for pl in proj.get("players", []):
        if pl.get("p_play", 0) < MIN_PLAY or pl.get("minutes", {}).get("mean", 0) < MIN_MINUTES:
            continue
        key = (pl.get("game_date"), pl.get("team"), norm(pl.get("name", "")))
        posted = lines.get(key, {})
        rec = {"id": pl.get("id"), "name": pl.get("name"), "team": pl.get("team"), "opp": pl.get("opp"),
               "game_date": pl.get("game_date"), "event_id": pl.get("event_id"),
               "p_play": pl.get("p_play"), "minutes": pl.get("minutes", {}).get("mean"),
               "flags": pl.get("flags"), "markets": {}}
        for market in MARKETS:
            block = pl.get(market)
            if not block or "cdf" not in block:
                continue
            if float(block.get("mean") or 0) < MIN_MARKET_MEAN:
                continue
            cdf = block["cdf"]
            slots = posted.get(market, {})
            sides = {}
            for side in (("under", "over") if both_sides else ("under",)):
                rungs = {}
                # 1. the shopping guide: the tightest line at each confidence level
                for level in levels:
                    hit = tightest_under(cdf, level) if side == "under" else tightest_over(cdf, level)
                    if hit:
                        line, p_model = hit
                        rungs[line] = _rung(cdf, market, line, side, level, slots.get(line, {}), cal, edge)
                        rungs[line]["src"] = "model"
                # 2. every line the feed actually publishes (main and alternate), so the book's own numbers show
                book_lines = sorted(slots, key=lambda L: abs(L - float(block.get("p50") or 0)))[:max_book]
                for line in book_lines:
                    if line in rungs:
                        rungs[line]["src"] = "both"
                        continue
                    p_model = p_under_at(cdf, line) if side == "under" else p_over_at(cdf, line)
                    rungs[line] = _rung(cdf, market, line, side, p_model, slots[line], cal, edge)
                    rungs[line]["src"] = "book"
                if rungs:
                    sides[side] = sorted(rungs.values(), key=lambda r: r["p"], reverse=True)
            if not sides:
                continue
            rec["markets"][market] = {"mean": block.get("mean"), "p50": block.get("p50"),
                                      "priced": bool(slots), "sides": sides}
            for side, rungs in sides.items():
                for r in rungs:
                    rows.append({"game_date": rec["game_date"], "player": rec["name"], "team": rec["team"],
                                 "opp": rec["opp"], "market": market, "market_label": LABEL.get(market, market),
                                 "side": side, "level": r["level"], "line": r["line"], "our_p": r["p"], "model_p": r["p_model"],
                                 "fair_price": r["fair"], "min_price": r["min"], "source": r["src"],
                                 "book": r.get("book", ""), "book_price": r.get("price", ""),
                                 "book_dec": r.get("dec", ""), "edge": r.get("edge", ""),
                                 "worth_taking": r.get("worth", ""), "mean": rec["markets"][market]["mean"],
                                 "p_play": rec["p_play"]})
        if rec["markets"]:
            players.append(rec)
    return players, rows

def run(levels, both_sides=True, edge=EDGE, max_rungs=MAX_RUNGS):
    now = datetime.now(timezone.utc).astimezone(GMT8)
    players, rows = build(levels, both_sides, edge, max_rungs)
    OUT_JSON.write_text(json.dumps({
        "generated": now.strftime("%d/%m/%Y %H:%M GMT+8"), "generated_epoch": int(now.timestamp()),
        "levels": list(levels), "edge_required": edge,
        "note": ("tight_line = the lowest Under line (highest Over line) that still clears the confidence level, "
                 "so it is the best price available at that safety. p_used is corrected by calibration.json. "
                 "min_price is the smallest decimal price worth taking at the required edge. "
                 "A rung with no book price is your model's read only -- nothing on the market to check it against."),
        "players": players}, separators=(",", ":")), encoding="utf-8")
    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(players)} players, {len(rows)} ladder rungs -> {OUT_JSON}, {OUT_CSV}")
    good = [r for r in rows if r["worth_taking"] is True]
    print(f"  rungs your bookie has priced AND that clear the {edge:.0%} edge: {len(good)}")
    for r in sorted(good, key=lambda r: -r["our_p"])[:8]:
        print(f"   {r['player']:<24} {r['market_label']:<9} {r['side']:<5} {r['line']:<6} "
              f"ours {r['our_p']:.0%}  {r['book']} {r['book_price']:+d}  edge {r['edge']:+.1%}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--levels", nargs="*", type=float, default=list(LEVELS))
    p.add_argument("--unders-only", action="store_true")
    p.add_argument("--edge", type=float, default=EDGE)
    p.add_argument("--max-rungs", type=int, default=MAX_RUNGS)
    a = p.parse_args(argv)
    return run(sorted(a.levels), not a.unders_only, a.edge, a.max_rungs)


if __name__ == "__main__":
    sys.exit(main())
