"""Daily Under shortlist: main-line player-prop Unders only, single bets, the best few per day.

Modes
  python scripts/shortlist.py            build today's shortlist and update the pick log
  python scripts/shortlist.py --grade    grade the pick log against box scores, compare with "bet every Under"

Reads    data/projections_latest.json, data/lines_latest.json, data/schedule.csv
Writes   data/shortlist_latest.json   what to show today
         data/picks_log.csv           every pick, frozen at tip-off (this is the record the test is judged on)
         data/shortlist_grade.json    result so far (--grade)

How a pick is chosen
  1. The Under line is the player's main line. If DraftKings and FanDuel differ, the LOWER line is used (the hardest
     one for an Under), so a pick still holds if your bookie sits on either line.
  2. Our chance of the Under comes from the simulation (cdf). It is blended with the book's own no-vig chance
     (MW share ours), the same provisional weight the rest of the project uses.
  3. A pick needs a blended chance of at least MIN_BLEND, a positive blended edge at the book's price, and our number
     must not sit more than MAX_GAP from the market's (a huge gap usually means news we do not have).
  4. Two shortlists: Points and PRA (PRA only for players the books post it for). Per list: at most PER_GAME picks
     per game, top TOP_N per day. A player can be on both lists.
  5. A pick is frozen into the log when its game tips off. Until then it can change as lines move.
"""
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_lines import norm, load_results  # noqa: E402

DATA = Path("data")
PROJ = DATA / "projections_latest.json"
LINES = DATA / "lines_latest.json"
SCHED = DATA / "schedule.csv"
HIST = DATA / "lines_history.csv"
OUT = DATA / "shortlist_latest.json"
LOG = DATA / "picks_log.csv"
GRADE = DATA / "shortlist_grade.json"

TOP_N = 4              # shortlist size per day, for EACH of the points list and the PRA list
PER_GAME = 2           # at most this many picks from one game (same-game results move together)
MW = 0.35              # weight on our model vs the market (provisional, as in index.html)
MIN_BLEND = 0.56       # blended chance of the Under must be at least this
MIN_EDGE = 0.02        # blended chance times the price must beat 1 by at least this
MAX_GAP = 0.20         # skip when our chance and the market's differ by more than this
MIN_DEC = 1.75         # skip prices worse than about -133
MIN_PLAY = 0.85        # chance the player plays
MIN_MINUTES = 18.0     # projected minutes
STATS = ("pts", "pra")
LOG_FIELDS = ["game_date", "player", "team", "opp", "market", "line", "book", "price", "our_p", "market_p",
              "blend_p", "edge", "tipoff_utc", "frozen", "logged_utc"]


def dec(a):
    return 1 + a / 100.0 if a > 0 else 1 + 100.0 / -a


def iso(d):
    """dd/mm/yyyy -> yyyy-mm-dd"""
    return datetime.strptime(d, "%d/%m/%Y").strftime("%Y-%m-%d")


def p_under(cdf, line):
    """Chance the Under wins, given it did not push. Half lines never push."""
    k = int(math.floor(line))
    cum = lambda i: 0.0 if i < 0 else (1.0 if i >= len(cdf) else cdf[i])
    if line != k:                       # x.5
        return cum(k)
    push = cum(k) - cum(k - 1)
    return cum(k - 1) / (1 - push) if push < 1 else 0.5


def load_tipoffs():
    out = {}
    try:
        with open(SCHED, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(r["game_date_et"], r["home"])] = out[(r["game_date_et"], r["away"])] = r["tipoff_utc"]
    except Exception:
        pass
    return out


def parse_tip(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def candidates():
    proj = json.load(open(PROJ, encoding="utf-8"))
    lines = json.load(open(LINES, encoding="utf-8"))
    tips = load_tipoffs()
    by = {}
    for p in lines["players"]:
        by[(p["d"], p["t"], norm(p["n"]))] = p
    cands = []
    for pl in proj["players"]:
        fl = pl.get("flags") or {}
        if fl.get("low_history") or fl.get("league_average_rates_used"):
            continue
        if pl.get("p_play", 0) < MIN_PLAY or pl["minutes"]["mean"] < MIN_MINUTES:
            continue
        lp = by.get((pl["game_date"], pl["team"], norm(pl["name"])))
        if not lp:
            continue
        for stat in STATS:
            e = lp.get(stat)
            if not e or not e.get("m") or stat not in pl:
                continue
            low = min(x[1] for x in e["m"])
            row = max((x for x in e["m"] if x[1] == low), key=lambda x: dec(x[3]))
            book, line, o_px, u_px = row
            io, iu = 1 / dec(o_px), 1 / dec(u_px)
            market_p = iu / (io + iu)
            ours = p_under(pl[stat]["cdf"], line)
            blend = MW * ours + (1 - MW) * market_p
            d = dec(u_px)
            edge = blend * d - 1
            if blend < MIN_BLEND or edge < MIN_EDGE or abs(ours - market_p) > MAX_GAP or d < MIN_DEC:
                continue
            d_iso = iso(pl["game_date"])
            cands.append({
                "game_date": d_iso, "player": pl["name"], "team": pl["team"], "opp": pl["opp"], "market": stat,
                "line": line, "book": book, "price": u_px, "our_p": round(ours, 3), "market_p": round(market_p, 3),
                "blend_p": round(blend, 3), "edge": round(edge, 3), "tipoff_utc": tips.get((d_iso, pl["team"]), ""),
                "all_lines": [[x[0], x[1], x[3]] for x in e["m"]], "mean": pl[stat]["mean"],
                "p_play": pl["p_play"], "minutes": pl["minutes"]["mean"], "event_id": pl["event_id"]})
    return cands


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick_day(cands, frozen):
    """Fill one day's TOP_N slots for ONE market: frozen picks keep theirs, the best live candidates take the rest."""
    taken_players = {r["player"] for r in frozen}
    chosen = []
    for c in sorted(cands, key=lambda c: -c["edge"]):
        if len(frozen) + len(chosen) >= TOP_N:
            break
        gk = tuple(sorted((c["team"], c["opp"])))
        n_game = sum(1 for r in frozen + chosen if tuple(sorted((r["team"], r["opp"]))) == gk)
        if c["player"] in taken_players or n_game >= PER_GAME:
            continue
        taken_players.add(c["player"])
        chosen.append(c)
    return chosen


def build():
    now = datetime.now(timezone.utc)
    cands = candidates()
    log = read_log()
    started = lambda tip: (parse_tip(tip) or now) <= now
    keep = [r for r in log if r["frozen"] == "1"]
    live_old = {(r["game_date"], r["player"], r["market"]): r for r in log if r["frozen"] != "1"}
    # a live pick whose game has now tipped off freezes with the numbers it last had
    for k, r in list(live_old.items()):
        if started(r["tipoff_utc"]):
            r["frozen"] = "1"
            keep.append(r)
            del live_old[k]
    dates = sorted({c["game_date"] for c in cands} | {r["game_date"] for r in live_old.values()})
    new_live, shown = [], []
    for d in dates:
        day = {"game_date": d, "markets": {}}
        for m in STATS:
            day_c = [c for c in cands if c["game_date"] == d and c["market"] == m and not started(c["tipoff_utc"])]
            day_frozen = [r for r in keep if r["game_date"] == d and r["market"] == m]
            chosen = pick_day(day_c, day_frozen)
            for c in chosen:
                c["frozen"] = "0"
                c["logged_utc"] = now.strftime("%Y-%m-%dT%H:%MZ")
                new_live.append(c)
            day["markets"][m] = chosen
        shown.append(day)
    write_log(sorted(keep + new_live, key=lambda r: (r["game_date"], r["player"])))
    out = {"generated_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "top_n": TOP_N, "days": shown,
           "rules": {"min_blend": MIN_BLEND, "min_edge": MIN_EDGE, "max_gap": MAX_GAP, "model_weight": MW,
                     "line_used": "lowest main line across DK/FD"}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for d in shown:
        for m, picks in d["markets"].items():
            print(d["game_date"], m, [f"{p['player']} U{p['line']:g} {p['price']} blend {p['blend_p']:.0%}" for p in picks] or "no picks")
    print(f"{len(cands)} qualifying candidates; log has {len(keep)} frozen picks")
    return 0


def wilson(w, n, z=1.96):
    if not n:
        return None, None
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


def grade():
    res = load_results()
    wins = losses = pushes = 0
    profit, price_sum, n = 0.0, 0.0, 0
    by_day = {}
    for r in read_log():
        if r["frozen"] != "1":
            continue
        x = res.get((r["game_date"], norm(r["player"]), r["team"]))
        if not x:
            continue
        actual, line, price = x[r["market"]], float(r["line"]), float(r["price"])
        d = dec(price)
        if actual == line:
            pushes += 1
            continue
        won = actual < line
        wins += won
        losses += not won
        profit += (d - 1) if won else -1.0
        price_sum += d
        n += 1
        day = by_day.setdefault(r["game_date"], [0, 0])
        day[0 if won else 1] += 1
    avg_dec = price_sum / n if n else None
    lo, hi = wilson(wins, n)
    baseline = every_under_baseline()
    rep = {"graded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "bets": n, "wins": wins,
           "losses": losses, "pushes": pushes, "win_rate": round(wins / n, 3) if n else None,
           "win_rate_95ci": [lo, hi], "avg_decimal_price": round(avg_dec, 3) if avg_dec else None,
           "break_even": round(1 / avg_dec, 3) if avg_dec else None,
           "units_won": round(profit, 2), "return_per_bet": round(profit / n, 3) if n else None,
           "by_day": {k: {"w": v[0], "l": v[1]} for k, v in sorted(by_day.items())},
           "every_main_line_under_baseline": baseline,
           "bets_needed_note": "A 95% interval is about +/-6 points at 60 bets and +/-3 points at 250. Judge on 250+ bets, not 60.",
           "verdict": verdict(n, wins, avg_dec, lo, hi)}
    GRADE.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("bets", "win_rate", "break_even", "units_won", "win_rate_95ci", "verdict")}))
    return 0


def verdict(n, w, avg_dec, lo, hi):
    if n < 60 or not avg_dec:
        return "too few bets to say anything"
    be = 1 / avg_dec
    if hi < be:
        return "below break-even with confidence: stop"
    if lo > be:
        return "above break-even with confidence: keep going"
    return "undecided: the interval still covers break-even, keep logging"


def every_under_baseline():
    """Win rate of betting every main-line Under (one per player, market, day) over the same period as the log."""
    try:
        res = load_results()
        seen, n, w = set(), 0, 0
        with open(HIST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    line = float(r["line"])
                    x = res.get((r["game_date"], norm(r["player"]), r["team"]))
                except ValueError:
                    continue
                k = (r["game_date"], r["player"], r["market"])
                if not x or k in seen or x[r["market"]] == line:
                    continue
                seen.add(k)
                n += 1
                w += x[r["market"]] < line
        return {"bets": n, "win_rate": round(w / n, 3) if n else None}
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(grade() if "--grade" in sys.argv else build())
