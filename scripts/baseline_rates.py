"""Step 2: per-minute baseline rates (points, rebounds, assists) for every player.

For each player we blend per-minute rates from the current season and the last
two seasons (older = less weight), plus a bit of extra weight on the last 10
games, then shrink toward the league (or position) average when the player has
few minutes. Which prior to use is chosen by the backtest, simplest first.

Modes
  backtest  walk-forward test on 2025-26 (each game predicted from earlier data
            only), tunes the settings on the early part of that season, checks
            them on the rest, and compares against simple baselines.
            Writes data/rates_params.json and data/rates_backtest.json.
  estimate  rates as of today for the current season. Writes data/player_rates.csv.
  both      backtest, then estimate.

Standard library only. Reads data/player_games.csv (made by update_data.py).
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")
PLAYER_GAMES = DATA / "player_games.csv"
RATES = DATA / "player_rates.csv"
PARAMS = DATA / "rates_params.json"
BACKTEST = DATA / "rates_backtest.json"

NBA = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL",
    "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH",
}
VALID_TYPES = {"2", "3", "5"}  # regular season, playoffs, play-in
STATS = ("pts", "reb", "ast")
POS_MAP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
MIN_EVAL = 10.0          # games below this many minutes are not scored in the backtest
OLDER_RATIO = 0.5        # two seasons ago counts half as much as last season (the 2:1 plan)
SPLIT_DATE = "2026-01-15"  # tune before this date, validate on/after it
TEST_SEASON = 2026       # season_year 2026 = the 2025-26 season

K_GRID = [0, 25, 50, 100, 200, 300, 500, 800, 1200]
A_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
C_GRID = [0.0, 0.5, 1.0, 2.0]
VARIANTS = ["league", "pos", "pos_tier"]

DEFAULT_PARAMS = {
    "b": OLDER_RATIO,
    # Settings tuned on the 2025-26 season (run the backtest workflow to re-tune).
    "pts": {"variant": "league", "K": 50, "a": 0.25, "c": 1.0, "phi": 2.326},
    "reb": {"variant": "pos", "K": 50, "a": 0.25, "c": 0.5, "phi": 1.088},
    "ast": {"variant": "league", "K": 25, "a": 0.25, "c": 1.0, "phi": 1.122},
}


def log(msg):
    print(msg, flush=True)


# ---------- loading ----------

def tier_of(mpg):
    if mpg >= 26:
        return "core"
    if mpg >= 16:
        return "role"
    return "bench"


def load_rows():
    rows = []
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["team"] not in NBA or r["opp"] not in NBA:
                continue
            if r["season_type"] not in VALID_TYPES or r["dnp"] == "1":
                continue
            m = float(r["min"])
            if m <= 0:
                continue
            rows.append({
                "date": r["game_date_et"], "event": r["event_id"],
                "sy": int(r["season_year"]), "aid": r["athlete_id"],
                "name": r["player"], "team": r["team"],
                "pos": POS_MAP.get(r["position"], "UNK"), "min": m,
                "y": (int(r["pts"]), int(r["reb"]), int(r["ast"])),
            })
    rows.sort(key=lambda r: (r["date"], r["event"]))
    if not rows:
        raise SystemExit("no usable rows in data/player_games.csv")
    return rows


# ---------- history for one target season ----------

class History:
    """Everything known before the target season starts, plus helpers."""

    def __init__(self, rows, target):
        self.target = target
        self.prev = {}  # season offset (1, 2) -> {aid: [M, G, pts, reb, ast]}
        for off in (1, 2):
            tot = defaultdict(lambda: [0.0, 0, 0, 0, 0])
            for r in rows:
                if r["sy"] == target - off:
                    t = tot[r["aid"]]
                    t[0] += r["min"]
                    t[1] += 1
                    for i in range(3):
                        t[2 + i] += r["y"][i]
            self.prev[off] = dict(tot)
        self.pos_counter = defaultdict(Counter)
        self.last10 = defaultdict(lambda: deque(maxlen=10))
        for r in rows:
            if r["sy"] < target:
                self.pos_counter[r["aid"]][r["pos"]] += 1
                self.last10[r["aid"]].append((r["min"], r["y"]))
        self.groups = self._group_rates(rows)

    def _group_rates(self, rows):
        """League / position / position+tier per-minute rates from earlier seasons."""
        per_player = defaultdict(lambda: [0.0, 0])
        for r in rows:
            if target_off(r["sy"], self.target) in (1, 2):
                p = per_player[(r["sy"], r["aid"])]
                p[0] += r["min"]
                p[1] += 1
        pos_of = {}
        for aid, c in self.pos_counter.items():
            pos_of[aid] = c.most_common(1)[0][0]
        league = [0.0, 0.0, 0.0, 0.0]
        pos = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        post = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for r in rows:
            if target_off(r["sy"], self.target) not in (1, 2):
                continue
            pm, pg = per_player[(r["sy"], r["aid"])]
            tier = tier_of(pm / pg)
            p = pos_of.get(r["aid"], "UNK")
            for g in (league, pos[p], post[(p, tier)]):
                g[0] += r["min"]
                for i in range(3):
                    g[1 + i] += r["y"][i]
        return {"league": league, "pos": dict(pos), "pos_tier": dict(post)}

    def mu(self, pos, tier):
        """Prior mean rates (per minute) for each variant: tuple of 3 per variant."""
        def rate(g):
            return tuple(g[1 + i] / g[0] for i in range(3))
        league = rate(self.groups["league"])
        g = self.groups["pos"].get(pos)
        pos_mu = rate(g) if g and g[0] >= 500 else league
        gt = self.groups["pos_tier"].get((pos, tier)) if tier else None
        post_mu = rate(gt) if gt and gt[0] >= 500 else pos_mu
        return {"league": league, "pos": pos_mu, "pos_tier": post_mu}


def target_off(sy, target):
    return target - sy


# ---------- features (one entry per scored player-game) ----------

def snapshot(h, cur, aid, pos):
    """State of knowledge about one player at this moment."""
    c = cur.get(aid, [0.0, 0, 0, 0, 0])
    p1 = h.prev[1].get(aid, [0.0, 0, 0, 0, 0])
    p2 = h.prev[2].get(aid, [0.0, 0, 0, 0, 0])
    l10 = h.last10.get(aid, ())
    l10m = sum(m for m, _ in l10)
    l10s = [sum(y[i] for _, y in l10) for i in range(3)]
    games = c[1] + p1[1] + p2[1]
    mpg = (c[0] + p1[0] + p2[0]) / games if games else 0.0
    tier = tier_of(mpg) if games else None
    return {
        "cur_M": c[0], "cur": c[2:5], "p1_M": p1[0], "p1": p1[2:5],
        "p2_M": p2[0], "p2": p2[2:5], "l10_M": l10m, "l10": l10s,
        "mu": h.mu(pos, tier), "tier": tier, "games": games,
    }


def best_pos(h, aid, cur_pos):
    c = h.pos_counter.get(aid)
    if cur_pos.get(aid):
        merged = Counter(c) if c else Counter()
        merged.update(cur_pos[aid])
        return merged.most_common(1)[0][0]
    return c.most_common(1)[0][0] if c else "UNK"


def build_features(rows, target):
    h = History(rows, target)
    season_rows = [r for r in rows if r["sy"] == target]
    if not season_rows:
        raise SystemExit(f"no rows for season_year {target}")
    start = season_rows[0]["date"]
    start_ord = datetime.fromisoformat(start).toordinal()
    cur = {}
    cur_pos = defaultdict(Counter)
    feats = []
    i = 0
    while i < len(season_rows):
        j = i
        while j < len(season_rows) and season_rows[j]["date"] == season_rows[i]["date"]:
            j += 1
        day = season_rows[i:j]
        day_no = datetime.fromisoformat(day[0]["date"]).toordinal() - start_ord
        for r in day:  # predict first, using only earlier days
            if r["min"] >= MIN_EVAL:
                pos = best_pos(h, r["aid"], cur_pos)
                f = snapshot(h, cur, r["aid"], pos)
                f.update(date=r["date"], day=day_no, m=r["min"], y=r["y"], aid=r["aid"])
                feats.append(f)
        for r in day:  # then learn from the day
            c = cur.setdefault(r["aid"], [0.0, 0, 0, 0, 0])
            c[0] += r["min"]
            c[1] += 1
            for k in range(3):
                c[2 + k] += r["y"][k]
            cur_pos[r["aid"]][r["pos"]] += 1
            h.last10[r["aid"]].append((r["min"], r["y"]))
        i = j
    return feats


# ---------- predictors ----------

def blended_rate(f, s, variant, K, a, b, c):
    sw = f["cur"][s] + a * (f["p1"][s] + b * f["p2"][s]) + c * f["l10"][s]
    mw = f["cur_M"] + a * (f["p1_M"] + b * f["p2_M"]) + c * f["l10_M"]
    mu = f["mu"][variant][s]
    den = K + mw
    return (K * mu + sw) / den if den > 0 else mu


def baseline_rate(f, s, name):
    mu = f["mu"]["pos_tier"][s]
    if name == "league_pos_mean":
        return f["mu"]["pos"][s]
    if name == "last_season":
        return f["p1"][s] / f["p1_M"] if f["p1_M"] > 0 else mu
    if name == "season_to_date":
        return f["cur"][s] / f["cur_M"] if f["cur_M"] > 0 else mu
    if name == "last_10_games":
        return f["l10"][s] / f["l10_M"] if f["l10_M"] > 0 else mu
    raise ValueError(name)


BASELINES = ["league_pos_mean", "last_season", "season_to_date", "last_10_games"]


def sse(feats, s, rate_fn):
    tot = 0.0
    for f in feats:
        d = f["y"][s] - rate_fn(f) * f["m"]
        tot += d * d
    return tot


SIMPLER_WINS_UNLESS = 0.0015  # a fancier prior must cut squared error by 0.15% to be used


def tune_variant(train, s, b, variant):
    """Coordinate search for one prior variant: (K, a) first, then recent-form weight c."""
    best = {"variant": variant, "K": 200, "a": 1.0, "c": 0.0}
    best_sse = float("inf")
    for K in K_GRID:
        for a in A_GRID:
            v = sse(train, s, lambda f: blended_rate(f, s, variant, K, a, b, 0.0))
            if v < best_sse:
                best_sse, best = v, {"variant": variant, "K": K, "a": a, "c": 0.0}
    for c in C_GRID:
        v = sse(train, s, lambda f: blended_rate(f, s, variant, best["K"], best["a"], b, c))
        if v < best_sse - 1e-9:
            best_sse, best["c"] = v, c
    return best, best_sse


def tune_stat(train, s, b):
    """Tune every prior variant, then keep the simplest one unless a fancier one clearly wins."""
    chosen, chosen_sse = None, None
    for variant in VARIANTS:  # ordered simplest first
        cand, cand_sse = tune_variant(train, s, b, variant)
        if chosen is None or cand_sse < chosen_sse * (1 - SIMPLER_WINS_UNLESS):
            chosen, chosen_sse = cand, cand_sse
    return chosen


def model_rate_fn(p, s, b):
    return lambda f: blended_rate(f, s, p[STATS[s]]["variant"], p[STATS[s]]["K"],
                                  p[STATS[s]]["a"], b, p[STATS[s]]["c"])


def evaluate(feats, params, b):
    """RMSE per method for points, rebounds, assists and PRA."""
    methods = {"model": [model_rate_fn(params, s, b) for s in range(3)]}
    for name in BASELINES:
        methods[name] = [(lambda f, s=s, name=name: baseline_rate(f, s, name)) for s in range(3)]
    out = {}
    for name, fns in methods.items():
        se = [0.0, 0.0, 0.0, 0.0]
        for f in feats:
            err_sum = 0.0
            for s in range(3):
                d = f["y"][s] - fns[s](f) * f["m"]
                se[s] += d * d
                err_sum += d
            se[3] += err_sum * err_sum
        n = max(len(feats), 1)
        out[name] = {k: round(math.sqrt(v / n), 4) for k, v in zip(("pts", "reb", "ast", "pra"), se)}
    return out


def paired_z(feats, fn_a, fn_b, s):
    """Mean squared-error difference (a - b) and its z-score; negative = a better."""
    diffs = []
    for f in feats:
        ea = f["y"][s] - fn_a(f) * f["m"]
        eb = f["y"][s] - fn_b(f) * f["m"]
        diffs.append(ea * ea - eb * eb)
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n) if var > 0 else float("inf")
    return mean, mean / se


def fit_phi(feats, params, s, b):
    """Overdispersion: Var(count) ~ phi * predicted count (quasi-Poisson)."""
    fn = model_rate_fn(params, s, b)
    num = den = 0.0
    for f in feats:
        pred = fn(f) * f["m"]
        num += (f["y"][s] - pred) ** 2
        den += pred
    return num / den if den else 1.0


# ---------- backtest ----------

def run_backtest(rows):
    b = OLDER_RATIO
    feats = build_features(rows, TEST_SEASON)
    train = [f for f in feats if f["date"] < SPLIT_DATE]
    valid = [f for f in feats if f["date"] >= SPLIT_DATE]
    log(f"Backtest season_year {TEST_SEASON}: {len(feats)} scored player-games "
        f"(tune on {len(train)} before {SPLIT_DATE}, check on {len(valid)})")

    params = {"b": b}
    for s, name in enumerate(STATS):
        params[name] = tune_stat(train, s, b)
        log(f"  tuned {name}: {params[name]}")
    for s, name in enumerate(STATS):
        params[name]["phi"] = round(fit_phi(train, params, s, b), 3)

    buckets = {
        "first 30 days": [f for f in valid + train if f["day"] < 30],
        "days 30-90": [f for f in valid + train if 30 <= f["day"] < 90],
        "after day 90": [f for f in valid + train if f["day"] >= 90],
    }
    report = {
        "season_year": TEST_SEASON, "min_minutes_scored": MIN_EVAL,
        "tuned_on_rows": len(train), "validated_on_rows": len(valid),
        "split_date": SPLIT_DATE,
        "train": evaluate(train, params, b), "validation": evaluate(valid, params, b),
        "by_period_all_rows": {k: {"rows": len(v), **evaluate(v, params, b)} for k, v in buckets.items()},
    }
    best_base = {}
    for s, name in enumerate(STATS):
        base = min(BASELINES, key=lambda n: report["validation"][n][name])
        best_base[name] = base
        mean, z = paired_z(valid, model_rate_fn(params, s, b),
                           lambda f, s=s, base=base: baseline_rate(f, s, base), s)
        report.setdefault("validation_vs_best_baseline", {})[name] = {
            "best_baseline": base, "mse_diff_per_row": round(mean, 4), "z": round(z, 2)}
    report["params"] = params

    def with_changes(**over):
        q = json.loads(json.dumps(params))
        for name in STATS:
            q[name].update(over)
        return q

    ablations = {
        "full model": params,
        "no recent-form term (c=0)": with_changes(c=0.0),
        "no shrinkage (K=0)": with_changes(K=0),
        "league-wide prior, not position+tier": with_changes(variant="league"),
        "ignore last season (a=0)": with_changes(a=0.0),
    }
    report["ablation_validation"] = {k: evaluate(valid, v, b)["model"] for k, v in ablations.items()}
    early = [f for f in feats if f["day"] < 30]
    report["ablation_first_30_days_in_sample"] = {
        k: evaluate(early, v, b)["model"] for k, v in ablations.items()}

    log("\nValidation RMSE (lower is better), predicted count given actual minutes:")
    log(f"  {'method':<18}{'pts':>8}{'reb':>8}{'ast':>8}{'PRA':>8}")
    for name in ["model"] + BASELINES:
        r = report["validation"][name]
        log(f"  {name:<18}{r['pts']:>8}{r['reb']:>8}{r['ast']:>8}{r['pra']:>8}")
    log("\nModel vs best baseline on validation (z < -2 means clearly better):")
    for name, r in report["validation_vs_best_baseline"].items():
        log(f"  {name}: vs {r['best_baseline']}, z = {r['z']}")
    log("\nAblation on validation (what each piece is worth):")
    log(f"  {'version':<40}{'pts':>8}{'reb':>8}{'ast':>8}{'PRA':>8}")
    for k, r in report["ablation_validation"].items():
        log(f"  {k:<40}{r['pts']:>8}{r['reb']:>8}{r['ast']:>8}{r['pra']:>8}")
    log("\nSame ablation on the first 30 days of the season (the regime we are in now;"
        " settings were tuned on data that includes these days):")
    log(f"  {'version':<40}{'pts':>8}{'reb':>8}{'ast':>8}{'PRA':>8}")
    for k, r in report["ablation_first_30_days_in_sample"].items():
        log(f"  {k:<40}{r['pts']:>8}{r['reb']:>8}{r['ast']:>8}{r['pra']:>8}")
    log("\nBy period of season (all rows), RMSE pts:")
    for k, v in report["by_period_all_rows"].items():
        log(f"  {k:<14} rows={v['rows']:<6} " + "  ".join(
            f"{m}={v[m]['pts']}" for m in ["model"] + BASELINES))

    DATA.mkdir(exist_ok=True)
    PARAMS.write_text(json.dumps({**params, "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "tuned_on": f"season_year {TEST_SEASON} before {SPLIT_DATE}"}, indent=2),
                      encoding="utf-8")
    BACKTEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return params


# ---------- estimate (current rates) ----------

def load_params():
    if PARAMS.exists():
        p = json.loads(PARAMS.read_text(encoding="utf-8"))
        for k in ("b", "pts", "reb", "ast"):
            if k not in p:
                raise SystemExit(f"{PARAMS} is missing '{k}'")
        return p, False
    log("WARNING: data/rates_params.json not found, using default settings")
    return DEFAULT_PARAMS, True


def run_estimate(rows, params):
    target = max(r["sy"] for r in rows)
    h = History(rows, target)
    cur = {}
    cur_pos = defaultdict(Counter)
    last_team = {}
    last_date = {}
    for r in rows:
        if r["sy"] == target:
            c = cur.setdefault(r["aid"], [0.0, 0, 0, 0, 0])
            c[0] += r["min"]
            c[1] += 1
            for k in range(3):
                c[2 + k] += r["y"][k]
            cur_pos[r["aid"]][r["pos"]] += 1
            h.last10[r["aid"]].append((r["min"], r["y"]))
        if r["sy"] >= target - 1:
            last_team[r["aid"]] = r["team"]
            last_date[r["aid"]] = r["date"]
    names = {r["aid"]: r["name"] for r in rows}
    b = params["b"]
    as_of = max(r["date"] for r in rows)
    out = []
    for aid in sorted(set(cur) | set(h.prev[1])):
        pos = best_pos(h, aid, cur_pos)
        f = snapshot(h, cur, aid, pos)
        row = {
            "athlete_id": aid, "player": names.get(aid, ""), "team": last_team.get(aid, ""),
            "position": pos, "tier": f["tier"] or "",
            "games_this_season": cur.get(aid, [0, 0])[1],
            "min_this_season": round(f["cur_M"], 1),
            "min_last_season": round(f["p1_M"], 1), "min_two_seasons_ago": round(f["p2_M"], 1),
            "mpg_last_10": round(f["l10_M"] / max(len(h.last10.get(aid, ())), 1), 1),
            "new_player": int(f["p1_M"] + f["p2_M"] == 0),
            "last_game_date": last_date.get(aid, ""),
        }
        for s, name in enumerate(STATS):
            p = params[name]
            rate = blended_rate(f, s, p["variant"], p["K"], p["a"], b, p["c"])
            mw = f["cur_M"] + p["a"] * (f["p1_M"] + b * f["p2_M"]) + p["c"] * f["l10_M"]
            sd = math.sqrt(p["phi"] * rate / (p["K"] + mw)) if p["K"] + mw > 0 else float("nan")
            row[f"{name}_per_min"] = round(rate, 4)
            row[f"{name}_per_min_sd"] = round(sd, 4)
            row[f"{name}_per36"] = round(rate * 36, 2)
        mw_pts = f["cur_M"] + params["pts"]["a"] * (f["p1_M"] + b * f["p2_M"])
        row["pts_reliability"] = round(mw_pts / (params["pts"]["K"] + mw_pts), 2) if mw_pts else 0.0
        out.append(row)
    out.sort(key=lambda r: -(r["min_this_season"] + r["min_last_season"]))
    if not out:
        raise SystemExit("no players to estimate")
    fields = list(out[0].keys())
    DATA.mkdir(exist_ok=True)
    with open(RATES, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    log(f"\nEstimate as of {as_of}, season_year {target}: {len(out)} players written to {RATES}")
    log(f"  with this-season minutes: {sum(1 for r in out if r['games_this_season'])}, "
        f"new players (no earlier seasons): {sum(r['new_player'] for r in out)}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "estimate", "both"], required=True)
    args = p.parse_args(argv)
    rows = load_rows()
    log(f"Loaded {len(rows)} usable player-games")
    if args.mode in ("backtest", "both"):
        run_backtest(rows)
    if args.mode in ("estimate", "both"):
        params, used_default = load_params()
        run_estimate(rows, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
