"""The daily card: the plays with the best price for their chance, plus the safest line available at each level.

Two lists, both built from the same calibrated probabilities:

  VALUE  one line per player-market with the best expected value at the price on offer. This is the old
         "shortlist": only now the probability is corrected by data/calibration.json (learned on real closing
         lines) instead of being taken at face value with a fixed 35% market blend, and it can use any market the
         feed prices (points, PRA, rebounds, assists, and the three combos), not just points and PRA.

  SAFE   the tightest line your bookie prices at a chosen confidence (default 88%), i.e. the best-paying bet that
         still clears that chance. One per player-market. This is the list to read when you are shopping
         alternate lines by hand: it tells you the number to look for and the price it needs to be worth taking.

Every row shows the model's chance, the bookmaker's de-vigged chance, the corrected chance actually used, the
price, and the edge left after the vig. You are the judge; the numbers are the evidence.

Modes
  python scripts/shortlist.py              rebuild from the data on disk (picks stay open for change)
  python scripts/shortlist.py --lock       same, and freeze the next slate into the log
  python scripts/shortlist.py --grade      grade the frozen log against box scores
  python scripts/shortlist.py --unders-only   reproduce the old under-only behaviour

Reads  data/projections_latest.json, data/lines_latest.json, data/calibration.json, data/schedule.csv
Writes data/shortlist_latest.json, data/picks_log.csv, data/shortlist_grade.json
"""
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as cal_mod                                   # noqa: E402
from grade_lines import dec, load_results, norm              # noqa: E402

DATA = Path("data")
PROJ = DATA / "projections_latest.json"
LINES = DATA / "lines_latest.json"
SCHED = DATA / "schedule.csv"
HIST = DATA / "lines_history.csv"
OUT = DATA / "shortlist_latest.json"
LOG = DATA / "picks_log.csv"
GRADE = DATA / "shortlist_grade.json"

TOP_N = 5                 # per list, per day
PER_GAME = 2              # at most this many plays from one game (same-game results move together)
PER_PLAYER = 1            # at most this many plays per player across all markets
MARKETS = ("pts", "pra", "reb", "ast", "pr", "pa", "ar")
MARKET_LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "pr": "P+R", "pa": "P+A", "ar": "A+R",
                "pra": "PRA", "tpm": "threes"}
MIN_EV = 0.03             # expected value required at the price on offer
MIN_DEC = 1.75            # skip prices worse than about -133 (measured: unders priced -140..-120 lose money)
MIN_P = 0.55              # a side must be at least this likely on the corrected number
MAX_GAP = 0.35            # raw model vs market this far apart usually means news we do not have: skip
MIN_PLAY = 0.70           # chance the player appears (news + measured appearance rate)
MIN_MINUTES = 14.0
SAFE_LEVEL = 0.88         # default confidence for the safe list
SAFE_MIN_DEC = 1.15
UNDERS_ONLY = False

LOG_FIELDS = ["game_date", "player", "team", "opp", "market", "line", "book", "price", "our_p", "market_p",
              "blend_p", "edge", "tipoff_utc", "frozen", "logged_utc", "all_lines", "mean", "p_play",
              "side", "list", "level", "source", "calibration", "p_appear"]


def iso(d):
    """dd/mm/yyyy -> yyyy-mm-dd"""
    return datetime.strptime(d, "%d/%m/%Y").strftime("%Y-%m-%d")


def p_side(cdf, line, side):
    """Chance of winning a side at a line, given cdf[k] = P(stat <= k). Integer lines are push-adjusted."""
    k = int(math.floor(line))
    cum = lambda i: 0.0 if i < 0 else (1.0 if i >= len(cdf) else cdf[i])
    if line != k:                                   # x.5: no push
        return cum(k) if side == "u" else 1.0 - cum(k)
    push = cum(k) - cum(k - 1)
    if push >= 1.0:
        return 0.5
    if side == "u":
        return cum(k - 1) / (1.0 - push)
    return (1.0 - cum(k)) / (1.0 - push)


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


def posted_lines(lines_json):
    """{(date, team, norm name): {market: {line: {side: (price, book), 'fair_o': p}}}} best price per line."""
    out = {}
    for p in lines_json.get("players", []):
        key = (p.get("d"), p.get("t"), norm(p.get("n", "")))
        rec = out.setdefault(key, {})
        for market, entry in p.items():
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
                    if side not in cur or dec(price) > dec(cur[side][0]):
                        cur[side] = (price, book)
    return out


def candidates(proj, lines_json, cal, unders_only=False, markets=MARKETS):
    """Every playable side at every line the book actually prices, with the corrected chance and the edge."""
    tips = load_tipoffs()
    posted = posted_lines(lines_json)
    out = []
    for pl in proj.get("players", []):
        if pl.get("p_play", 0) < MIN_PLAY or pl.get("minutes", {}).get("mean", 0) < MIN_MINUTES:
            continue
        if (pl.get("flags") or {}).get("league_average_rates_used"):
            continue
        key = (pl.get("game_date"), pl.get("team"), norm(pl.get("name", "")))
        plines = posted.get(key)
        if not plines:
            continue
        for market in markets:
            block = pl.get(market)
            if not block or "cdf" not in block:
                continue
            cdf = block["cdf"]
            for line, sides in (plines.get(market) or {}).items():
                if "o" not in sides or "u" not in sides:
                    continue
                o_px, o_book = sides["o"]
                u_px, u_book = sides["u"]
                # the market's own view at this line, so the blend knows what it is arguing with
                io, iu = 1.0 / dec(o_px), 1.0 / dec(u_px)
                for side, price, book, mkt_p in (("u", u_px, u_book, iu / (io + iu)),
                                                 ("o", o_px, o_book, io / (io + iu))):
                    if unders_only and side != "u":
                        continue
                    raw = p_side(cdf, line, side)
                    if raw < 0.02 or raw > 0.98:
                        continue
                    p = cal_mod.apply_calibration(raw, market, cal, mkt_p)
                    d = dec(price)
                    ev = p * d - 1.0
                    out.append({
                        "game_date": iso(pl["game_date"]), "team": pl["team"], "opp": pl["opp"],
                        "player": pl["name"], "athlete_id": pl["id"], "market": market, "line": line,
                        "side": side, "book": book, "price": price, "our_p": round(p, 4),
                        "raw_p": round(raw, 4), "market_p": round(mkt_p, 4), "dec": round(d, 3),
                        "edge": round(ev, 4), "mean": block.get("mean"), "p_play": pl.get("p_play"),
                        "p_appear": pl.get("p_appear", pl.get("p_play")),
                        "tipoff_utc": tips.get((iso(pl["game_date"]), pl["team"]), ""),
                        "event_id": pl.get("event_id"),
                        "blend_p": round(p, 4),
                        "all_lines": [[ln, sd["u"][0], sd["o"][0]]
                                      for ln, sd in sorted((plines.get(market) or {}).items())
                                      if "u" in sd and "o" in sd],
                    })
    return out


def build_value(cands):
    """One play per player-market: the line with the best expected value that also clears the safety filters."""
    best = {}
    for c in cands:
        if c["edge"] < MIN_EV or c["dec"] < MIN_DEC or c["our_p"] < MIN_P:
            continue
        if abs(c["raw_p"] - c["market_p"]) > MAX_GAP:
            continue
        k = (c["player"], c["market"])
        if k not in best or c["edge"] > best[k]["edge"]:
            best[k] = c
    plays = []
    for c in best.values():
        c = dict(c, list="value")
        plays.append(c)
    return plays


LADDER = DATA / "alt_lines_latest.json"


def load_ladder():
    """The safety ladder written by alt_lines.py, if it is there. One flat row per rung."""
    if not LADDER.exists():
        return []
    try:
        j = json.loads(LADDER.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for pl in j.get("players", []):
        for market, m in (pl.get("markets") or {}).items():
            for side, rungs in (m.get("sides") or {}).items():
                for r in rungs:
                    rows.append({"player": pl["name"], "team": pl["team"], "opp": pl["opp"],
                                 "game_date": iso(pl["game_date"]), "market": market, "side": side,
                                 "line": r["line"], "our_p": r["p"], "model_p": r["p_model"],
                                 "fair": r["fair"], "min_price": r["min"], "level": r["level"],
                                 "book": r.get("book"), "price": r.get("price"), "dec": r.get("dec"),
                                 "edge": r.get("edge"), "worth": r.get("worth", False),
                                 "mean": m.get("mean"), "p_play": pl.get("p_play"),
                                 "tipoff_utc": None})
    return rows


def build_safe(ladder=None, level=SAFE_LEVEL):
    """Priced rungs that are both safe and worth taking: chance at/above the level and a positive edge.

    The ladder in data/alt_lines_latest.json carries every confidence level; this keeps only what the bookmaker
    actually prices today, so a "safe" play is always one you can place. When the feed posts no alternate lines
    this list is empty, and the ladder table (alt_lines_latest.json) is what you shop from by hand.
    """
    ladder = ladder if ladder is not None else load_ladder()
    best = {}
    for r in ladder:
        if r.get("price") is None or r.get("edge") is None:
            continue
        if r["our_p"] < level - 0.02 or r["edge"] < 0:
            continue
        k = (r["player"], r["market"])
        cur = best.get(k)
        if cur is None or (r["our_p"], r["edge"]) > (cur["our_p"], cur["edge"]):
            best[k] = r
    out = [dict(r, list="safe") for r in best.values()]
    out.sort(key=lambda r: -r["edge"])
    return out


def pick_day(cands, frozen, per_game=PER_GAME, per_player=PER_PLAYER, top_n=TOP_N):
    """Fill one day's slots: frozen picks keep theirs, the best live candidates take the rest."""
    taken_players = {r["player"] for r in frozen}
    chosen = []
    def keyf(c):
        e = c.get("edge")
        return -(e if e is not None else -0.05)
    for c in sorted(cands, key=keyf):
        if len(frozen) + len(chosen) >= top_n:
            break
        gk = tuple(sorted((c["team"], c["opp"])))
        n_game = sum(1 for r in frozen + chosen if tuple(sorted((r["team"], r["opp"]))) == gk)
        if c["player"] in taken_players or n_game >= per_game:
            continue
        taken_players.add(c["player"])
        chosen.append(c)
    return chosen


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    with open(LOG, "w", newline="", encoding="utf-8") as w:
        wr = csv.DictWriter(w, fieldnames=LOG_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            r = dict(r)
            if isinstance(r.get("all_lines"), list):
                r["all_lines"] = json.dumps(r["all_lines"])
            wr.writerow(r)


def to_pick(r):
    p = dict(r)
    for k in ("line", "our_p", "market_p", "blend_p", "edge", "mean", "p_play", "p_appear", "level"):
        try:
            p[k] = float(r[k])
        except (KeyError, ValueError, TypeError):
            p[k] = None
    for k in ("price",):
        try:
            p[k] = int(float(r[k]))
        except (KeyError, ValueError, TypeError):
            pass
    try:
        raw = r.get("all_lines")
        p["all_lines"] = raw if isinstance(raw, list) else json.loads(raw or "[]")
    except ValueError:
        p["all_lines"] = []
    p["locked"] = r.get("frozen") == "1"
    p["side"] = r.get("side") or "u"
    p["market"] = r.get("market") or "pts"
    p["market_label"] = MARKET_LABEL.get(p["market"], p["market"])
    return p


def build(lock=False, unders_only=UNDERS_ONLY, safe_level=SAFE_LEVEL):
    now = datetime.now(timezone.utc)
    if not PROJ.exists():
        raise SystemExit(f"{PROJ} is missing: run simulate_props.py --mode project first")
    proj = json.loads(PROJ.read_text(encoding="utf-8"))
    if not LINES.exists():
        raise SystemExit(f"{LINES} is missing: run lines.py first")
    lines = json.loads(LINES.read_text(encoding="utf-8"))
    cal = cal_mod.load_calibration()
    cands = candidates(proj, lines, cal, unders_only=unders_only)
    value = build_value(cands)
    safe = build_safe(level=safe_level)
    tips_all = load_tipoffs()
    for c in safe:
        c["tipoff_utc"] = tips_all.get((c["game_date"], c["team"]), "")
    log = read_log()

    def started(tip, date):
        t = parse_tip(tip or "")
        if t:
            return t <= now
        return date < (now - timedelta(days=1)).strftime("%Y-%m-%d")

    keep = [r for r in log if r["frozen"] == "1"]
    live_old = {(r["game_date"], r["player"], r["market"], r.get("list") or "value"): r
                for r in log if r["frozen"] != "1"}
    for k, r in list(live_old.items()):
        if started(r.get("tipoff_utc"), r["game_date"]):
            r["frozen"] = "1"
            keep.append(r)
            del live_old[k]

    dates = sorted({c["game_date"] for c in value + safe} | {r["game_date"] for r in live_old.values()})
    new_live, shown = [], []
    open_dates = sorted({c["game_date"] for c in value + safe if not started(c["tipoff_utc"], c["game_date"])})
    lock_date = open_dates[0] if open_dates else None
    for d in dates:
        day = {"game_date": d, "value": [], "safe": []}
        for name, pool in (("value", value), ("safe", safe)):
            day_c = [c for c in pool if c["game_date"] == d and not started(c["tipoff_utc"], c["game_date"])]
            day_frozen = [r for r in keep if r["game_date"] == d and (r.get("list") or "value") == name]
            chosen = pick_day(day_c, day_frozen, top_n=TOP_N if name == "value" else TOP_N + 3)
            for c in chosen:
                c["frozen"] = "1" if (lock and d == lock_date) else "0"
                c["locked"] = c["frozen"] == "1"
                c["logged_utc"] = now.strftime("%Y-%m-%dT%H:%MZ")
                c["calibration"] = cal.get("mode", "")
                new_live.append(c)
            day[name] = [to_pick(r) for r in day_frozen] + [to_pick(c) for c in chosen]
        shown.append(day)
    write_log(sorted(keep + new_live, key=lambda r: (r["game_date"], r.get("list") or "", r["player"])))
    out = {"generated_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "generated": now.astimezone(
        timezone(timedelta(hours=8))).strftime("%d/%m/%Y %H:%M GMT+8"),
        "top_n": TOP_N, "locked_by_manual_run": bool(lock), "days": shown,
        "calibration": {k: cal.get(k) for k in ("mode", "rows", "fitted_at")},
        "rules": {"min_edge": MIN_EV, "min_price_dec": MIN_DEC, "min_chance": MIN_P, "max_gap": MAX_GAP,
                  "min_play": MIN_PLAY, "min_minutes": MIN_MINUTES, "safe_level": safe_level,
                  "unders_only": unders_only, "markets": list(MARKETS),
                  "line_used": "every line the feed prices, main and alternate; best price per line"}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for d in shown:
        for name in ("value", "safe"):
            picks = d[name]
            def fmt(p):
                px = f"{p['price']:+d}" if p.get("price") is not None else f"needs {p.get('min_price')}"
                ev = f" ev{p['edge']:+.1%}" if p.get("edge") is not None else " (no line posted)"
                return (f"{p['player'][:18]} {p['side'].upper()}{p['line']:g} "
                        f"{MARKET_LABEL.get(p['market'], p['market'])} {px} {p['our_p']:.0%}{ev}")
            print(d["game_date"], name, [fmt(p) for p in picks] or "none")
    print(f"{len(cands)} priced sides checked; {len(value)} value plays, {len(safe)} safe lines; "
          f"{len(keep) + sum(1 for c in new_live if c['frozen'] == '1')} locked in the log"
          + (f"; locked the {lock_date} slate" if lock and lock_date else ""))
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
    by_day, by_list, by_side = {}, {}, {}
    for r in read_log():
        if r["frozen"] != "1":
            continue
        x = res.get((r["game_date"], norm(r["player"]), r["team"]))
        if not x:
            continue
        market = r.get("market") or "pts"
        side = r.get("side") or "u"
        if market not in x:
            continue
        if not r.get("price"):
            continue
        actual, line, price = x[market], float(r["line"]), float(r["price"])
        d = dec(price)
        if actual == line:
            pushes += 1
            continue
        won = actual < line if side == "u" else actual > line
        wins += won
        losses += not won
        profit += (d - 1) if won else -1.0
        price_sum += d
        n += 1
        for bucket, key in ((by_day, r["game_date"]), (by_list, r.get("list") or "value"), (by_side, side)):
            b = bucket.setdefault(key, [0, 0, 0.0])
            b[0 if won else 1] += 1
            b[2] += (d - 1) if won else -1.0
    avg_dec = price_sum / n if n else None
    lo, hi = wilson(wins, n)
    baseline = every_under_baseline()
    rep = {"graded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "bets": n, "wins": wins,
           "losses": losses, "pushes": pushes, "win_rate": round(wins / n, 3) if n else None,
           "win_rate_95ci": [lo, hi], "avg_decimal_price": round(avg_dec, 3) if avg_dec else None,
           "break_even": round(1 / avg_dec, 3) if avg_dec else None,
           "units_won": round(profit, 2), "return_per_bet": round(profit / n, 3) if n else None,
           "by_day": {k: {"w": v[0], "l": v[1]} for k, v in sorted(by_day.items())},
           "by_list": {k: {"w": v[0], "l": v[1], "units": round(v[2], 2)} for k, v in sorted(by_list.items())},
           "by_side": {k: {"w": v[0], "l": v[1], "units": round(v[2], 2)} for k, v in sorted(by_side.items())},
           "every_main_line_under_baseline": baseline,
           "bets_needed_note": "A 95% interval is about +/-6 points at 60 bets and +/-3 points at 250. "
                               "Judge on 250+ bets, not 60.",
           "verdict": verdict(n, wins, avg_dec, lo, hi)}
    GRADE.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("bets", "win_rate", "break_even", "units_won", "win_rate_95ci",
                                          "verdict", "by_list", "by_side")}, indent=1))
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
    """Win rate of betting every main-line Under over the same period, as a sanity check on the sample."""
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
                if not x or k in seen or r["market"] not in x or x[r["market"]] == line:
                    continue
                seen.add(k)
                n += 1
                w += x[r["market"]] < line
        return {"bets": n, "win_rate": round(w / n, 3) if n else None}
    except Exception:
        return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--grade" in argv:
        return grade()
    global MIN_EV, MIN_DEC, MIN_P, SAFE_LEVEL
    for flag, name in (("--safe-level", "SAFE_LEVEL"), ("--min-ev", "MIN_EV"), ("--min-dec", "MIN_DEC"),
                       ("--min-chance", "MIN_P")):
        if flag in argv:
            globals()[name] = float(argv[argv.index(flag) + 1])
    return build(lock="--lock" in argv, unders_only="--unders-only" in argv, safe_level=SAFE_LEVEL)


if __name__ == "__main__":
    sys.exit(main())
